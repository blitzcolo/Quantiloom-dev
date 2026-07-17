// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes with SolarSpectralLUT integration
// Thermal IR sky uses the NN atmosphere downwelling spectrum (L_down)
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "atmosphere_nn.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;

// ============================================================================
// Solar Spectral LUT Buffer (Binding 15)
// ============================================================================
// ASTM G-173 solar irradiance curves for spectral sky background
// When numSamples > 0, use true spectral sky radiance
// Otherwise, fall back to LightingParams scalar/RGB values
// ============================================================================

[[vk::binding(15, 0)]] StructuredBuffer<SolarSpectralLUT> solarSpectralLUT;

// ============================================================================
// NN Atmosphere LUT (Bindings 17 + 20)
// ============================================================================
// Baked MODTRAN-surrogate spectral LUT. The miss shader uses the thermal
// downwelling spectrum (L_down) as the sky background for MWIR/LWIR; when
// a sky-dome network is trained (hasSky), it will switch to SampleAtmosSky.
// ============================================================================

[[vk::binding(17, 0)]] StructuredBuffer<AtmosNNHeader> atmosNNHeader;
[[vk::binding(20, 0)]] StructuredBuffer<float> atmosNNData;

// ============================================================================
// CIE 1931 Color Matching Functions LUT (Binding 19)
// ============================================================================
// High-precision CIE XYZ CMFs for VIS_FUSED mode spectral integration
// 401 samples (380-780nm @ 1nm resolution): float3(x_bar, y_bar, z_bar)
// Provides <0.1% error vs analytical approximation's 10-20% at edges
// ============================================================================

[[vk::binding(19, 0)]] StructuredBuffer<float3> cieCMF_LUT;

// ============================================================================
// Push Constants
// ============================================================================
// Camera data for accessing spectral mode and wavelength
// ============================================================================

[[vk::push_constant]] CameraData camera;

// ============================================================================
// Miss Entry Point
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    // Fetch lighting parameters and NN atmosphere header
    LightingParams lut = lightingParams[0];
    AtmosNNHeader atmos = atmosNNHeader[0];

    // Check if spectral solar LUT is available
    bool hasSpectralSolarLUT = (solarSpectralLUT[0].skyIrradiance.numSamples > 0);

    // ========================================================================
    // Sky Radiance
    // ========================================================================
    // Reflective bands (RGB / VIS / NIR / SWIR / reflective SINGLE) keep the
    // SolarSpectralLUT-based sky. Thermal bands use the NN downwelling
    // spectrum when baked (hasLdown), replacing the constant-temperature
    // Planck sky. A trained sky-dome network (hasSky) will plug in here.
    // ========================================================================

    // Choose sky radiance based on spectral mode
    if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
        // RGB mode: Direct RGB sky color (no spectral integration)
        payload.radiance = lut.skyRadiance_rgb;

    } else if (camera.spectral_mode == SPECTRAL_MODE_VIS_FUSED) {
        // ================================================================
        // VIS_FUSED mode: 32-wavelength spectral integration
        // ================================================================
        // Consistent with closesthit.rchit VIS_FUSED implementation:
        //   1. Sample 32 wavelengths uniformly across visible spectrum (380-780nm)
        //   2. Query spectral sky radiance at each wavelength
        //   3. Integrate via CIE XYZ color matching functions
        //   4. Convert XYZ → Linear RGB (sRGB D65)
        //   5. Apply chromaticity correction
        //
        // This ensures sky background color matches object reflections,
        // eliminating the "color discontinuity" issue.
        // ================================================================

        // NOTE: 400-780 nm (narrowed from 380 to match NN atmosphere coverage)
        const uint   NUM_WAVELENGTH_SAMPLES = 32;
        const float  LAMBDA_MIN_VIS = 400.0;
        const float  LAMBDA_MAX_VIS = 780.0;
        const float  LAMBDA_STEP = (LAMBDA_MAX_VIS - LAMBDA_MIN_VIS) / float(NUM_WAVELENGTH_SAMPLES - 1);

        float3 XYZ_accum = float3(0.0, 0.0, 0.0);

        for (uint i = 0; i < NUM_WAVELENGTH_SAMPLES; ++i) {
            float lambda = LAMBDA_MIN_VIS + float(i) * LAMBDA_STEP;

            // Query sky radiance at this wavelength
            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // Fallback: RGB → Illuminant spectrum (consistent with closesthit)
                sky_radiance_lambda = ConvertLinearRGBToIlluminantSpectrum(lut.skyRadiance_rgb, lambda);
            }

            // Weight by CIE color matching functions (use LUT for high precision)
            float3 xyz_cmf = SampleCIE_XYZ_LUT(cieCMF_LUT, lambda);
            float x_bar = xyz_cmf.x;
            float y_bar = xyz_cmf.y;
            float z_bar = xyz_cmf.z;

            // Riemann sum: XYZ += L(λ) × CMF(λ) × Δλ
            XYZ_accum.x += sky_radiance_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += sky_radiance_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += sky_radiance_lambda * z_bar * LAMBDA_STEP;
        }

        // Normalize XYZ for RGB input compatibility (see closesthit.rchit for details)
        XYZ_accum /= CIE_Y_INTEGRAL;

        // XYZ → Linear RGB (sRGB D65)
        payload.radiance = ConvertXYZToLinearRGB(XYZ_accum);

        // Apply chromaticity correction (consistent with closesthit)
        payload.radiance.r *= lut.chromaR_correction;
        payload.radiance.b *= lut.chromaB_correction;

        // Validation
        if (!isfinite(payload.radiance.r) || !isfinite(payload.radiance.g) || !isfinite(payload.radiance.b)) {
            payload.radiance = float3(0.0, 0.0, 0.0);
        }
        payload.radiance = clamp(payload.radiance, 0.0, 1000.0);

    } else if (camera.spectral_mode == SPECTRAL_MODE_SINGLE) {
        // Single wavelength mode: Query spectral sky at current wavelength
        float radiance_spectral;

        if (atmos.enabled != 0 && atmos.thermalBand != 0 && atmos.hasLdown != 0) {
            // Thermal single wavelength: NN downwelling spectrum (one sample)
            radiance_spectral = SampleAtmosLdown(atmos, atmosNNData, 0);
        } else if (hasSpectralSolarLUT) {
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], camera.wavelength_nm);
            radiance_spectral = sky_irr / PI;
        } else {
            radiance_spectral = lut.skyRadiance_spectral;
        }

        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    } else if (camera.spectral_mode == SPECTRAL_MODE_SWIR_FUSED) {
        // ================================================================
        // SWIR_FUSED mode: Sky radiance integration (1000-2500nm)
        // ================================================================
        // Must match closesthit.rchit calculation for unit consistency.
        // SWIR is reflection-dominated; use solar/sky spectral radiance.
        // ================================================================

        // NOTE: 1400-2400 nm (narrowed to match NN atmosphere coverage)
        const float SWIR_LAMBDA_MIN = 1400.0;
        const float SWIR_LAMBDA_MAX = 2400.0;
        const uint  NUM_SWIR_SAMPLES = 16;
        const float lambda_step = (SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN) / float(NUM_SWIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Fallback: RGB luminance / bandwidth (same as closesthit)
        float sky_luminance = 0.2126 * lut.skyRadiance_rgb.r +
                              0.7152 * lut.skyRadiance_rgb.g +
                              0.0722 * lut.skyRadiance_rgb.b;
        float swir_bandwidth = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;
        float sky_power_rgb = sky_luminance / swir_bandwidth;

        for (uint i = 0; i < NUM_SWIR_SAMPLES; ++i) {
            float lambda = SWIR_LAMBDA_MIN + float(i) * lambda_step;

            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                sky_radiance_lambda = sky_power_rgb;
            }

            radiance_accum += sky_radiance_lambda * lambda_step;
        }

        float band_width = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (camera.spectral_mode == SPECTRAL_MODE_NIR_FUSED) {
        // ================================================================
        // NIR_FUSED mode: Sky radiance integration (780-1400nm)
        // ================================================================
        // Near-IR is purely reflection-dominated (no thermal emission).
        // Uses same approach as SWIR with different wavelength range.
        // ================================================================

        // NOTE: 930-1200 nm (narrowed to match NN atmosphere coverage)
        const float NIR_LAMBDA_MIN = 930.0;
        const float NIR_LAMBDA_MAX = 1200.0;
        const uint  NUM_NIR_SAMPLES = 16;
        const float lambda_step = (NIR_LAMBDA_MAX - NIR_LAMBDA_MIN) / float(NUM_NIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Fallback: RGB luminance / bandwidth
        float sky_luminance = 0.2126 * lut.skyRadiance_rgb.r +
                              0.7152 * lut.skyRadiance_rgb.g +
                              0.0722 * lut.skyRadiance_rgb.b;
        float nir_bandwidth = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;
        float sky_power_rgb = sky_luminance / nir_bandwidth;

        for (uint i = 0; i < NUM_NIR_SAMPLES; ++i) {
            float lambda = NIR_LAMBDA_MIN + float(i) * lambda_step;

            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                sky_radiance_lambda = sky_power_rgb;
            }

            radiance_accum += sky_radiance_lambda * lambda_step;
        }

        float band_width = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED ||
               camera.spectral_mode == SPECTRAL_MODE_LWIR_FUSED) {
        // ================================================================
        // MWIR/LWIR_FUSED mode: Atmospheric thermal background
        // ================================================================
        // Thermal IR bands: sky is a blackbody emitter at atmosphere temperature.
        // MWIR: 3000-5000nm, LWIR: 8000-12000nm
        // Uses Planck radiation at atmosphere temperature for sky radiance.
        // ================================================================

        float lambda_min, lambda_max;
        if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED) {
            lambda_min = 3000.0;
            lambda_max = 5000.0;
        } else {
            lambda_min = 8000.0;
            lambda_max = 12000.0;
        }

        const uint NUM_IR_SAMPLES = 16;
        const float lambda_step = (lambda_max - lambda_min) / float(NUM_IR_SAMPLES - 1);

        float radiance_accum = 0.0;
        float T_atmosphere = lut.atmosphereTemperature_K;

        // Fallback if atmosphere temperature not set
        if (T_atmosphere <= 0.0) {
            T_atmosphere = 250.0;  // Typical cold sky effective temperature
        }

        bool useNNLdown = (atmos.enabled != 0 && atmos.hasLdown != 0);
        for (uint i = 0; i < NUM_IR_SAMPLES; ++i) {
            float lambda = lambda_min + float(i) * lambda_step;

            // Sky thermal radiation: NN downwelling spectrum when baked
            // (hasSky = 1 will switch this to a zenith-dependent sky network),
            // otherwise Planck blackbody at atmosphere temperature
            float L_sky = useNNLdown ? SampleAtmosLdown(atmos, atmosNNData, i)
                                     : IRPlanckRadiance(T_atmosphere, lambda);
            radiance_accum += L_sky * lambda_step;
        }

        float band_width = lambda_max - lambda_min;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else {
        // ================================================================
        // Fallback: Unknown mode (MULTISPECTRAL TBD, etc.)
        // ================================================================
        float radiance_spectral = lut.skyRadiance_spectral;
        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }
}
