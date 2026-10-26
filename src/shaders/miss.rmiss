// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes with SolarSpectralLUT integration
// Thermal IR sky uses the NN atmosphere downwelling spectrum (L_down)
// ============================================================================

#include "common.hlsli"
#include "band_routing.hlsli"
#include "camera_response.hlsli"
#include "pbr.hlsli"
#include "atmosphere_nn.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"
#include "fusion_records.hlsli"

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
[[vk::binding(29, 0)]] ByteAddressBuffer cameraResponseTable;

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

[[vk::binding(19, 0)]] StructuredBuffer<float4> cieCMF_LUT;

// ============================================================================
// RGB -> Spectrum Coefficients (Binding 25)
// ============================================================================
// Jakob & Hanika sigmoid coefficients: xyz = (c0, c1, c2) of the quadratic in
// t = (lambda - 380) / 400, w unused. Three sub-tables of res^3 nodes indexed
// by which RGB channel is largest; core/RgbToSpectrum.hpp has the layout and
// the reason the z axis is warped.
// ============================================================================

[[vk::binding(25, 0)]] StructuredBuffer<float4> rgbToSpectrumTable;

// The push-constant block lives in common.hlsli, shared with raygen and the
// closest-hit shader. This one reads the camera wavelength out of it.

// ============================================================================
// Miss Entry Point
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    payload.fusionTerminalDepth = payload.depth;
    if (payload.fusionMediumCount != 0)
        payload.fusionFlags |= kFusionFlagAmbiguousMedium;
    // Fetch lighting parameters and NN atmosphere header
    LightingParams lut = lightingParams[0];
    AtmosNNHeader atmos = atmosNNHeader[0];

    // NN atmosphere composition for a primary miss. closesthit.rchit applies
    // L_pixel = tau * L_surface + L_path to every depth-0 hit, per wavelength
    // (or per RGB channel); a primary miss has a surface radiance too -- the
    // sky at the end of the ray -- and leaving it uncomposited opened a step
    // between sky and ground pixels at the horizon, exactly the
    // discontinuity the contract forbids. RayTCurrent() in a miss shader is
    // the ray's TMax (the camera far plane for a primary ray), which is the
    // range the LUT wants. Secondary misses return raw surface radiance for
    // the parent vertex's BRDF integration and must not composite, exactly
    // as in closesthit.
    //
    // Not every primary miss, though: see AtmosMissComposites. A ray looking
    // up from an airborne camera has no table to read, and AtmosCoordA keeps a
    // ground camera's upward line of sight inside the haze layer it crosses
    // instead of reading every sky pixel as ten kilometres of horizontal haze.
    const bool atmosComposite = (atmos.enabled != 0) && (payload.depth == 0) &&
                                AtmosMissComposites(atmos, WorldRayDirection());
    float atmosA = 0.0;
    float atmosAz = 0.0;
    if (atmosComposite) {
        atmosA = AtmosCoordA(atmos, WorldRayDirection(), RayTCurrent());
        atmosAz = AtmosRelAz(atmos, WorldRayDirection());
    }

    // Check if spectral solar LUT is available
    bool hasSpectralSolarLUT = HasSolarSpectralLUT(solarSpectralLUT);

    // ========================================================================
    // Sky Radiance
    // ========================================================================
    // Reflective bands (RGB / VIS / NIR / SWIR / reflective SINGLE) keep the
    // SolarSpectralLUT-based sky. Thermal bands use the NN downwelling
    // spectrum when baked (hasLdown), replacing the constant-temperature
    // Planck sky. A trained sky-dome network (hasSky) will plug in here.
    // ========================================================================

    // Choose sky radiance based on spectral mode
    if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_RGB) {
        // RGB mode: Direct RGB sky color (no spectral integration)
        payload.radiance = float4(lut.skyRadiance_rgb, 0.0);
        // Distributed over direction like the spectral bands' sky, each
        // channel at the wavelength the visible network uses for it.
        if (lut.skyDistribution == SKY_DISTRIBUTION_CLEAR) {
            const float3 skySun = SkySunDirection(lut.sunDirection);
            const float skyAnchor = SkyDomeAnchor(lightingParams, solarSpectralLUT, skySun);
            const float3 skyDir = normalize(WorldRayDirection());
            payload.radiance.r *= SkyDomeFactor(lightingParams, solarSpectralLUT, 650.0, skyDir, skySun, skyAnchor);
            payload.radiance.g *= SkyDomeFactor(lightingParams, solarSpectralLUT, 550.0, skyDir, skySun, skyAnchor);
            payload.radiance.b *= SkyDomeFactor(lightingParams, solarSpectralLUT, 450.0, skyDir, skySun, skyAnchor);
        }
        // Composition contract as in closesthit: the visible network's three
        // lambda slots carry R/G/B (650/550/450 nm).
        if (atmosComposite) {
            [unroll]
            for (uint ch = 0; ch < 3; ++ch) {
                const float tau_ch = SampleAtmosTau(atmos, atmosNNData, ch, atmosA);
                const float lpath_ch = SampleAtmosLpath(atmos, atmosNNData, ch, atmosA, atmosAz);
                payload.radiance[ch] = tau_ch * payload.radiance[ch] + lpath_ch;
            }
        }
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

    } else if (IsVisMode(SPEC_SPECTRAL_MODE)) {
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
        const float  LAMBDA_MIN_VIS = SPECTRAL_VIS_LAMBDA_MIN;
        const float  LAMBDA_MAX_VIS = SPECTRAL_VIS_LAMBDA_MAX;
        const float  LAMBDA_STEP = (LAMBDA_MAX_VIS - LAMBDA_MIN_VIS) / float(NUM_WAVELENGTH_SAMPLES - 1);

        float3 XYZ_accum = float3(0.0, 0.0, 0.0);

        // A ray past a dispersive refraction carries one wavelength; it must
        // bring back that wavelength's sky radiance and no more, or the caller
        // weights the whole band by a single cmf(λ_h) and the sky arrives
        // hundreds of times too bright. A quartet brings back four, unweighed,
        // for the same reason and by the same contract.
        //
        // Nothing is drawn here. A primary ray that reaches the sky is answered
        // by the deterministic grid whichever mode is running: the sky is
        // analytic, and four samples of an analytic function are worse than
        // thirty-two evaluations of it for the same cost in rays, which is
        // none.
        const bool heroMode = (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_VIS_HERO);
        const bool carriesQuartet = heroMode && (payload.heroLambda > 0.0);
        const bool heroRay = heroMode ? (payload.heroLambda < 0.0)
                                      : (payload.heroLambda > 0.0);
        const float heroLambdaAbs = abs(payload.heroLambda);
        const float4 quartet = carriesQuartet ? QuartetLambdas(payload.heroLambda)
                                              : float4(0.0, 0.0, 0.0, 0.0);
        const uint sampleCount = heroRay ? 1u
                               : (carriesQuartet ? VIS_HERO_QUARTET
                                                 : NUM_WAVELENGTH_SAMPLES);
        float heroRadiance = 0.0;
        float4 quadRadiance = float4(0.0, 0.0, 0.0, 0.0);

        // Which way the sky's light comes from (lighting.sky_distribution).
        // The sun and the anchor are per ray; the factor is per wavelength and
        // exactly 1 for a uniform sky.
        const float3 skyDir = normalize(WorldRayDirection());
        const float3 skySun = SkySunDirection(lut.sunDirection);
        const float skyAnchor = SkyDomeAnchor(lightingParams, solarSpectralLUT, skySun);

        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? heroLambdaAbs
                         : (carriesQuartet ? quartet[min(i, VIS_HERO_QUARTET - 1u)]
                                           : (LAMBDA_MIN_VIS + float(i) * LAMBDA_STEP));

            // Matching functions and D65 in one fetch; both are needed below.
            float4 cieSample = SampleCIE_LUT(cieCMF_LUT, lambda);

            // Query sky radiance at this wavelength
            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // No named illuminant, no sky light -- as in every other band
                // here, and as closesthit's analytic base (sky4) already has
                // it. This used to expand sky_radiance_rgb through D65, and
                // since the bounce residual is L_in - L_base with a base of 0,
                // that invented sky reached every surface as indirect light: a
                // scene with its sun and sky switched off still came out lit.
                sky_radiance_lambda = 0.0;
            }
            sky_radiance_lambda *= SkyDomeFactor(lightingParams, solarSpectralLUT, lambda,
                                                 skyDir, skySun, skyAnchor);

            // A primary miss composes like a primary hit: the sky at the end
            // of the ray is L_surface in the contract. A hero or quartet ray
            // takes the nearest baked sample, as closesthit does for the same
            // ray.
            if (atmosComposite) {
                const uint atmosIdx = (heroRay || carriesQuartet)
                    ? (uint)clamp(round((lambda - LAMBDA_MIN_VIS) / LAMBDA_STEP),
                                  0.0, float(NUM_WAVELENGTH_SAMPLES - 1))
                    : i;
                sky_radiance_lambda =
                    SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA) * sky_radiance_lambda +
                    SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
            }

            // Weight by CIE color matching functions (use LUT for high precision)
            float3 xyz_cmf = cieSample.xyz;
            float x_bar = xyz_cmf.x;
            float y_bar = xyz_cmf.y;
            float z_bar = xyz_cmf.z;

            if (heroRay) {
                heroRadiance = sky_radiance_lambda;   // scalar, caller weights it
                continue;
            }
            if (carriesQuartet) {
                // Four scalars, unweighed: the vertex that drew them weights
                // them, and it is the only one that knows the density.
                if      (i == 0u) quadRadiance.x = sky_radiance_lambda;
                else if (i == 1u) quadRadiance.y = sky_radiance_lambda;
                else if (i == 2u) quadRadiance.z = sky_radiance_lambda;
                else              quadRadiance.w = sky_radiance_lambda;
                continue;
            }

            // Riemann sum: XYZ += L(λ) × CMF(λ) × Δλ
            XYZ_accum.x += sky_radiance_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += sky_radiance_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += sky_radiance_lambda * z_bar * LAMBDA_STEP;
        }

        // Normalize XYZ for RGB input compatibility (see closesthit.rchit for details)
        XYZ_accum /= CIE_Y_INTEGRAL;

        // XYZ → Linear RGB (sRGB D65)
        payload.radiance = float4(ConvertXYZToLinearRGB(XYZ_accum), 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

        // Apply chromaticity correction (consistent with closesthit)
        payload.radiance.r *= lut.chromaR_correction;
        payload.radiance.b *= lut.chromaB_correction;

        // After the correction, for the reason given in closesthit.rchit: it
        // scales R and B against G and would turn one scalar into three.
        if (heroRay) {
            payload.radiance = float4(heroRadiance, heroRadiance, heroRadiance, 0.0);
            payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);
        }
        if (carriesQuartet) {
            payload.radiance = quadRadiance;
            payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);
        }

        // Validation
        if (any(!isfinite(payload.radiance))) {
            FusionFlagNonFinite();
            payload.radiance = float4(0.0, 0.0, 0.0, 0.0);
            payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);
        }
        if (fusionRecords.Load(kFusionQuantitativeWord * 4) == 0)
            payload.radiance = clamp(payload.radiance, 0.0, 1000.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

    } else if (IsSingleWavelengthMode() &&
               RoutesToVisibleSingle(SingleModeWavelength(payload.heroLambda))) {
        // Single wavelength mode: Query spectral sky at current wavelength
        float radiance_spectral;

        const float lambda = SingleModeWavelength(payload.heroLambda);
        if (atmos.enabled != 0 && atmos.thermalBand != 0 && atmos.hasLdown != 0) {
            // Thermal single wavelength: NN downwelling spectrum (one sample)
            const uint idx = SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT
                ? CameraAtmosLambdaIndex(cameraResponseTable, lambda, atmos.numLambda) : 0u;
            radiance_spectral = SampleAtmosLdown(atmos, atmosNNData, idx);
        } else if (hasSpectralSolarLUT) {
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
            radiance_spectral = sky_irr / PI;
            // The same distribution over direction as the visible band's sky.
            if (lut.skyDistribution == SKY_DISTRIBUTION_CLEAR) {
                const float3 skySun = SkySunDirection(lut.sunDirection);
                radiance_spectral *= SkyDomeFactor(
                    lightingParams, solarSpectralLUT, lambda, normalize(WorldRayDirection()),
                    skySun, SkyDomeAnchor(lightingParams, solarSpectralLUT, skySun));
            }
        } else {
            // The legacy scalar is an RGB average in arbitrary units. A
            // spectral measurement with no named illuminant has no sky light.
            radiance_spectral = 0.0;
        }

        if (atmosComposite) {
            // As in closesthit's SINGLE branch: one lambda, index 0 unless the
            // camera-response grid says otherwise.
            const uint atmosIdx = SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT
                ? CameraAtmosLambdaIndex(cameraResponseTable, lambda, atmos.numLambda) : 0u;
            radiance_spectral = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA) * radiance_spectral +
                                SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
        }

        payload.radiance = float4(radiance_spectral, radiance_spectral, radiance_spectral, 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);
    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_SWIR_FUSED ||
               (IsSingleWavelengthMode() &&
                RoutesToSwirBranch(SingleModeWavelength(payload.heroLambda)))) {
        // ================================================================
        // SWIR_FUSED mode: Sky radiance integration (1000-2500nm)
        // ================================================================
        // Must match closesthit.rchit calculation for unit consistency.
        // SWIR is reflection-dominated; use solar/sky spectral radiance.
        // ================================================================

        // NOTE: 1400-2400 nm (narrowed to match NN atmosphere coverage)
        const float SWIR_LAMBDA_MIN = SPECTRAL_SWIR_LAMBDA_MIN;
        const float SWIR_LAMBDA_MAX = SPECTRAL_SWIR_LAMBDA_MAX;
        const uint  NUM_SWIR_SAMPLES = 16;
        const float lambda_step = (SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN) / float(NUM_SWIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // No spectral solar LUT, no sky: closesthit's SWIR/NIR branches answer
        // 0 in this case ("no curve, no sun"), and the miss shader agrees.
        // The RGB triple is an average in arbitrary units and would invent
        // light a measurement does not have.

        // An environment bounce carries one wavelength and must get that
        // wavelength's sky back; the band average would be weighed by a single
        // reflectance sample and lose the correlation the bounce exists for.
        const bool cameraRay = SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT;
        const bool singleRay = IsSingleWavelengthMode();
        const float carriedLambda = SingleModeWavelength(payload.heroLambda);
        const bool heroRay = singleRay || (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_SWIR_SAMPLES;
        float heroRadiance = 0.0;

        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? carriedLambda
                                   : SWIR_LAMBDA_MIN + float(i) * lambda_step;

            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                sky_radiance_lambda = 0.0;
            }

            // A primary miss composes like a primary hit; the index scheme is
            // closesthit's SWIR branch: the fused sweep walks the baked grid,
            // a carried wavelength takes the nearest sample (or the camera
            // grid), a single-wavelength render reads the one-sample LUT.
            if (atmosComposite) {
                const uint atmosIdx = heroRay
                    ? (cameraRay
                        ? CameraAtmosLambdaIndex(cameraResponseTable, lambda, atmos.numLambda)
                        : (singleRay
                            ? 0u
                            : (uint)clamp(round((lambda - SWIR_LAMBDA_MIN) / lambda_step),
                                          0.0, float(NUM_SWIR_SAMPLES - 1))))
                    : i;
                sky_radiance_lambda =
                    SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA) * sky_radiance_lambda +
                    SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
            }

            if (heroRay) {
                // Scalar spectral radiance, by the contract on
                // Payload::heroLambda.
                heroRadiance = sky_radiance_lambda;
                continue;
            }

            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_SWIR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += sky_radiance_lambda * lambda_step * trapezoidW;
        }

        float band_width = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // Validation
        if (!isfinite(radiance_avg)) {
                FusionFlagNonFinite();
            radiance_avg = 0.0;
        }
        if (fusionRecords.Load(kFusionQuantitativeWord * 4) == 0)
            radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float4(radiance_avg, radiance_avg, radiance_avg, 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_NIR_FUSED) {
        // ================================================================
        // NIR_FUSED mode: Sky radiance integration (780-1400nm)
        // ================================================================
        // Near-IR is purely reflection-dominated (no thermal emission).
        // Uses same approach as SWIR with different wavelength range.
        // ================================================================

        // NOTE: 930-1200 nm (narrowed to match NN atmosphere coverage)
        const float NIR_LAMBDA_MIN = SPECTRAL_NIR_LAMBDA_MIN;
        const float NIR_LAMBDA_MAX = SPECTRAL_NIR_LAMBDA_MAX;
        const uint  NUM_NIR_SAMPLES = 16;
        const float lambda_step = (NIR_LAMBDA_MAX - NIR_LAMBDA_MIN) / float(NUM_NIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // No spectral solar LUT, no sky -- as in the SWIR branch above and in
        // closesthit.

        // Hero rays as in the SWIR branch above.
        const bool heroRay = (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_NIR_SAMPLES;
        float heroRadiance = 0.0;

        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? payload.heroLambda
                                   : NIR_LAMBDA_MIN + float(i) * lambda_step;

            float sky_radiance_lambda;
            if (hasSpectralSolarLUT) {
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT, lambda);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                sky_radiance_lambda = 0.0;
            }

            // A primary miss composes like a primary hit; the index scheme is
            // closesthit's NIR branch: the sweep walks the baked grid, a
            // carried wavelength takes the nearest sample.
            if (atmosComposite) {
                const uint atmosIdx = heroRay
                    ? (uint)clamp(round((lambda - NIR_LAMBDA_MIN) / lambda_step),
                                  0.0, float(NUM_NIR_SAMPLES - 1))
                    : i;
                sky_radiance_lambda =
                    SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA) * sky_radiance_lambda +
                    SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
            }

            if (heroRay) {
                heroRadiance = sky_radiance_lambda;   // scalar, caller weights it
                continue;
            }

            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_NIR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += sky_radiance_lambda * lambda_step * trapezoidW;
        }

        float band_width = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // Validation
        if (!isfinite(radiance_avg)) {
                FusionFlagNonFinite();
            radiance_avg = 0.0;
        }
        if (fusionRecords.Load(kFusionQuantitativeWord * 4) == 0)
            radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float4(radiance_avg, radiance_avg, radiance_avg, 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

    } else if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_MWIR_FUSED ||
               SPEC_SPECTRAL_MODE == SPECTRAL_MODE_LWIR_FUSED ||
               (IsSingleWavelengthMode() &&
                RoutesToThermalBranch(SingleModeWavelength(payload.heroLambda)))) {
        // ================================================================
        // MWIR/LWIR_FUSED mode: Atmospheric thermal background
        // ================================================================
        // Thermal IR bands: sky is a blackbody emitter at atmosphere temperature.
        // MWIR: 3000-5000nm, LWIR: 8000-12000nm
        // Uses Planck radiation at atmosphere temperature for sky radiance.
        // ================================================================

        float lambda_min, lambda_max;
        const bool cameraRay = SPEC_SPECTRAL_MODE == SPECTRAL_MODE_CAMERA_MEASUREMENT;
        const bool singleRay = IsSingleWavelengthMode();
        const float carriedLambda = SingleModeWavelength(payload.heroLambda);
        if (SPEC_SPECTRAL_MODE == SPECTRAL_MODE_MWIR_FUSED ||
            (singleRay && carriedLambda < SPECTRAL_LWIR_LAMBDA_MIN)) {
            // A single ray sweeps one sample at its own wavelength below;
            // the band bounds are what a fused sweep uses.
            lambda_min = SPECTRAL_MWIR_LAMBDA_MIN;
            lambda_max = SPECTRAL_MWIR_LAMBDA_MAX;
        } else {
            lambda_min = SPECTRAL_LWIR_LAMBDA_MIN;
            lambda_max = SPECTRAL_LWIR_LAMBDA_MAX;
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
        float cosZenith = WorldRayDirection().y;  // Y-up: +1 = zenith, 0 = horizon

        // A ray spawned by an environment bounce carries one wavelength and
        // must bring back that wavelength's sky radiance and no more. Returning
        // the band average to a surface that weighs it by a single reflectance
        // sample is what erased the spectral correlation these bands are for --
        // a sloped emissivity curve reflecting a cold sky would read as if the
        // sky were grey.
        const bool heroRay = singleRay || (payload.heroLambda > 0.0);
        const uint sampleCount = heroRay ? 1u : NUM_IR_SAMPLES;
        float heroRadiance = 0.0;

        for (uint i = 0; i < sampleCount; ++i) {
            float lambda = heroRay ? carriedLambda
                                   : lambda_min + float(i) * lambda_step;

            // The LUT is baked on the loop's sample points, so a sampled
            // wavelength takes the nearest index -- matching what the
            // closest-hit shader does for the same ray.
            uint atmosIdx = cameraRay ? CameraAtmosLambdaIndex(cameraResponseTable, lambda, atmos.numLambda)
                : singleRay ? 0u : heroRay
                ? (uint)clamp(round((lambda - lambda_min) / lambda_step),
                              0.0, float(NUM_IR_SAMPLES - 1))
                : i;

            float L_sky;
            if (useNNLdown) {
                L_sky = AtmosSkyRadianceIR(atmos, atmosNNData, atmosIdx, lambda,
                                           cosZenith, T_atmosphere);
            } else if (lut.skyEmissivityClear > 0.0) {
                // Analytic clear sky: colder overhead, warming toward the
                // horizon. T_atmosphere is the air temperature under this
                // model, not an effective sky temperature.
                L_sky = IRClearSkyRadiance(lut.skyEmissivityClear, cosZenith,
                                           T_atmosphere, lambda);
            } else {
                L_sky = IRPlanckRadiance(T_atmosphere, lambda);
            }

            // A primary miss composes like a primary hit: the sky at the end
            // of the ray goes through the view path like any other surface
            // radiance. atmosIdx was chosen above by the same rule closesthit
            // uses (sweep index, or the nearest sample for a carried lambda).
            //
            // Except above the horizon when the sky IS the network's: its
            // downwelling is already the radiance of the whole column along
            // that line of sight, near-surface layer included, and adding the
            // layer's path radiance again counted it twice, and made the sky
            // warmer than the network says it is. Below the horizon the ray
            // stands for distant ground and composes as before; toward the
            // horizon both sides approach the air's own blackbody, so the two
            // meet there.
            if (atmosComposite && !(useNNLdown && cosZenith > 0.0)) {
                L_sky = SampleAtmosTau(atmos, atmosNNData, atmosIdx, atmosA) * L_sky +
                        SampleAtmosLpath(atmos, atmosNNData, atmosIdx, atmosA, atmosAz);
            }

            if (heroRay) {
                // Scalar spectral radiance, by the contract on
                // Payload::heroLambda: no trapezoid weight, no band average.
                heroRadiance = L_sky;
                continue;
            }

            // Trapezoid rule. N samples span N-1 intervals, so the two
            // endpoints carry half weight. Summing N full-width rectangles
            // and then dividing by (N-1)*step made every fused band read
            // N/(N-1) high -- 6.7% for the 16-sample bands. Checked against
            // an isothermal cavity, which must render exactly its own
            // blackbody and read 1.066667x it.
            float trapezoidW = (i == 0 || i == NUM_IR_SAMPLES - 1) ? 0.5 : 1.0;
            radiance_accum += L_sky * lambda_step * trapezoidW;
        }

        float band_width = lambda_max - lambda_min;
        float radiance_avg = heroRay ? heroRadiance : (radiance_accum / band_width);

        // Validation
        if (!isfinite(radiance_avg)) {
                FusionFlagNonFinite();
            radiance_avg = 0.0;
        }
        if (fusionRecords.Load(kFusionQuantitativeWord * 4) == 0)
            radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        payload.radiance = float4(radiance_avg, radiance_avg, radiance_avg, 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);

    } else {
        // ================================================================
        // Fallback: Unknown mode (MULTISPECTRAL TBD, etc.)
        // ================================================================
        float radiance_spectral = lut.skyRadiance_spectral;
        payload.radiance = float4(radiance_spectral, radiance_spectral, radiance_spectral, 0.0);
        payload.fusionContributions = FusionClassify(payload.radiance.x, payload.fusionRoute);
    }
    FusionRecordSky(payload);
}
