// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes with SolarSpectralLUT integration
// Now with Delta-Tracking volumetric atmospheric scattering
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "atmospheric.hlsli"
#include "SpectralConversion.hlsli"

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
// Atmospheric Parameters Buffer (Binding 17)
// ============================================================================
// Physical parameters for Delta-Tracking volumetric atmospheric rendering
// When beta_rayleigh_550nm.x > 0, Delta-Tracking is enabled
// Otherwise, fall back to simple sky radiance
// ============================================================================

[[vk::binding(17, 0)]] StructuredBuffer<AtmosphericParams> atmosphericParams;

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
    // Fetch atmospheric and lighting parameters
    LightingParams lut = lightingParams[0];
    AtmosphericParams atmo = atmosphericParams[0];

    // Check if Delta-Tracking is enabled (beta_rayleigh > 0 indicates enabled)
    bool atmosphereEnabled = (atmo.beta_rayleigh_550nm.x > 1e-9);

    // Check if spectral solar LUT is available
    bool hasSpectralSolarLUT = (solarSpectralLUT[0].skyIrradiance.numSamples > 0);

    // Get ray information
    float3 ray_origin = WorldRayOrigin();
    float3 ray_dir = WorldRayDirection();

    // ========================================================================
    // Atmospheric Scattering (Delta-Tracking)
    // ========================================================================
    if (atmosphereEnabled) {
        // Planet center: assume camera at surface, planet center below
        // TODO: Make this configurable via uniform buffer
        float3 planet_center = float3(0.0, -atmo.planet_radius, 0.0);

        // Compute ray-atmosphere intersection
        float t_atmo_near, t_atmo_far;
        bool hits_atmosphere = RaySphereIntersection(
            ray_origin, ray_dir, planet_center,
            atmo.planet_radius + atmo.atmosphere_height,
            t_atmo_near, t_atmo_far);

        if (hits_atmosphere && t_atmo_far > 0.0) {
            // Ray enters atmosphere
            float t_min = max(t_atmo_near, 0.0);
            float t_max = t_atmo_far;

            // Initialize random state using ray coordinates
            // Use a simple hash of ray origin + direction for seed
            uint seed = uint(dot(ray_origin, float3(127.1, 311.7, 74.7))) +
                        uint(dot(ray_dir, float3(269.5, 183.3, 246.1)) * 1000.0);
            uint random_state = seed ^ 0xDEADBEEFu;

            // Perform Delta-Tracking
            float t_scatter;
            float transmittance;
            bool scattered = DeltaTracking(
                ray_origin, ray_dir,
                t_min, t_max,
                camera.wavelength_nm,
                atmo,
                planet_center,
                random_state,
                t_scatter,
                transmittance);

            if (scattered) {
                // Scattering event occurred
                float3 scatter_pos = ray_origin + ray_dir * t_scatter;
                float3 sun_dir = normalize(lut.sunDirection);

                // Get sun radiance at current wavelength
                float sun_radiance;
                if (hasSpectralSolarLUT) {
                    float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], camera.wavelength_nm);
                    sun_radiance = SunIrradianceToRadiance(sun_irr);
                } else {
                    sun_radiance = lut.sunRadiance_spectral;
                }

                // Compute single scattering
                float3 scattered_radiance = SingleScattering(
                    scatter_pos, ray_dir, sun_dir,
                    camera.wavelength_nm,
                    atmo, planet_center,
                    sun_radiance);

                payload.radiance = scattered_radiance;
                return;
            }

            // No scattering: ray escaped atmosphere without collision
            // Fall through to simple sky radiance (attenuated by transmittance if needed)
        }
    }

    // ========================================================================
    // Fallback: Simple Sky Radiance (no atmosphere or ray missed atmosphere)
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

        const uint   NUM_WAVELENGTH_SAMPLES = 32;
        const float  LAMBDA_MIN_VIS = 380.0;
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

            // Weight by CIE color matching functions
            float x_bar = CIE_X(lambda);
            float y_bar = CIE_Y(lambda);
            float z_bar = CIE_Z(lambda);

            // Riemann sum: XYZ += L(λ) × CMF(λ) × Δλ
            XYZ_accum.x += sky_radiance_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += sky_radiance_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += sky_radiance_lambda * z_bar * LAMBDA_STEP;
        }

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

        if (hasSpectralSolarLUT) {
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], camera.wavelength_nm);
            radiance_spectral = sky_irr / PI;
        } else {
            radiance_spectral = lut.skyRadiance_spectral;
        }

        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    } else {
        // Other spectral modes (MWIR, LWIR, etc.): Use scalar fallback
        float radiance_spectral = lut.skyRadiance_spectral;
        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }
}
