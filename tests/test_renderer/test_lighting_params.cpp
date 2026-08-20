// ============================================================================
// Quantiloom - Unit Tests for renderer/LightingParams.hpp
// ============================================================================
// Tests cover:
// - Struct size and alignment (CPU/GPU consistency)
// - Field offset validation (must match shader common.hlsli)
// - Default value factory function
// - Atmosphere temperature validation
// - Physical plausibility of default values
// ============================================================================

#include <gtest/gtest.h>
#include "renderer/LightingParams.hpp"
#include <cstddef>
#include <cmath>

using namespace quantiloom;

// ============================================================================
// Struct Layout Tests (Critical for CPU/GPU Consistency)
// ============================================================================

TEST(LightingParamsTest, StructSize) {
    // LightingParams must be exactly 80 bytes to match GPU layout
    EXPECT_EQ(sizeof(LightingParams), 80u);
}

TEST(LightingParamsTest, FieldOffsets) {
    // These offsets MUST match common.hlsli layout
    // If any fails, CPU/GPU data will be misaligned causing rendering bugs
    EXPECT_EQ(offsetof(LightingParams, sunDirection), 0u);
    EXPECT_EQ(offsetof(LightingParams, sunRadiance_spectral), 12u);
    EXPECT_EQ(offsetof(LightingParams, sunRadiance_rgb), 16u);
    EXPECT_EQ(offsetof(LightingParams, skyRadiance_spectral), 28u);
    EXPECT_EQ(offsetof(LightingParams, skyRadiance_rgb), 32u);
    EXPECT_EQ(offsetof(LightingParams, skyEmissivityClear), 44u);
    EXPECT_EQ(offsetof(LightingParams, worldUnitsToMeters), 48u);
    EXPECT_EQ(offsetof(LightingParams, atmosphereTemperature_K), 52u);
    EXPECT_EQ(offsetof(LightingParams, chromaR_correction), 56u);
    EXPECT_EQ(offsetof(LightingParams, chromaB_correction), 60u);
    EXPECT_EQ(offsetof(LightingParams, enableShadowRays), 64u);
    EXPECT_EQ(offsetof(LightingParams, enableEnvironmentMap), 68u);
    // The last two padding floats. Nothing is left: another field here changes
    // sizeof, and with it the SDK/Studio pairing.
    EXPECT_EQ(offsetof(LightingParams, emissiveTriangleCount), 72u);
    EXPECT_EQ(offsetof(LightingParams, emissiveTotalPower), 76u);
}

TEST(LightingParamsTest, Alignment16Byte) {
    // Struct should be 16-byte aligned for GPU compatibility
    EXPECT_EQ(sizeof(LightingParams) % 16, 0u);
}

// ============================================================================
// Default Values Tests
// ============================================================================

TEST(LightingParamsTest, CreateDefaultLightingParams) {
    LightingParams params = CreateDefaultLightingParams();

    // Sun direction should be normalized
    float sunDirLen = glm::length(params.sunDirection);
    EXPECT_NEAR(sunDirLen, 1.0f, 1e-5f);

    // Default values should be positive and reasonable
    EXPECT_GT(params.sunRadiance_spectral, 0.0f);
    EXPECT_GT(params.skyRadiance_spectral, 0.0f);
    EXPECT_GT(params.sunRadiance_rgb.r, 0.0f);
    EXPECT_GT(params.skyRadiance_rgb.r, 0.0f);

    // The clear-sky emissivity is a fraction, and zero by default: the
    // isotropic blackbody sky is what a host gets before a config says
    // otherwise, which is what every scene rendered before the analytic model
    // existed.
    EXPECT_FLOAT_EQ(params.skyEmissivityClear, 0.0f);

    // World units should be positive
    EXPECT_GT(params.worldUnitsToMeters, 0.0f);

    // Atmosphere temperature should be in valid range
    EXPECT_TRUE(IsAtmosphereTemperatureValid(params.atmosphereTemperature_K));

    // Off, because a context built from these defaults has no environment map in
    // it yet -- the flag means "a real map is bound and may be sampled", not
    // "this scene would like image-based lighting". It defaulted on, so a fresh
    // context lit every scene with the placeholder cubemap.
    EXPECT_EQ(params.enableEnvironmentMap, 0u);
}

TEST(LightingParamsTest, DefaultAtmosphereTemperature) {
    LightingParams params = CreateDefaultLightingParams();

    // Default should be 260K (clear mid-latitude sky)
    EXPECT_FLOAT_EQ(params.atmosphereTemperature_K, 260.0f);
}

// ============================================================================
// Atmosphere Temperature Validation Tests
// ============================================================================

TEST(LightingParamsTest, AtmosphereTemperatureValidRange) {
    // Typical Earth atmosphere temperatures
    EXPECT_TRUE(IsAtmosphereTemperatureValid(200.0f));  // Very cold (stratosphere)
    EXPECT_TRUE(IsAtmosphereTemperatureValid(260.0f));  // Clear sky (default)
    EXPECT_TRUE(IsAtmosphereTemperatureValid(290.0f));  // Humid tropical
    EXPECT_TRUE(IsAtmosphereTemperatureValid(320.0f));  // Hot desert
}

TEST(LightingParamsTest, AtmosphereTemperatureInvalidRange) {
    // Unrealistic temperatures should fail validation
    EXPECT_FALSE(IsAtmosphereTemperatureValid(0.0f));     // Absolute zero
    EXPECT_FALSE(IsAtmosphereTemperatureValid(100.0f));   // Too cold for Earth
    EXPECT_FALSE(IsAtmosphereTemperatureValid(400.0f));   // Too hot for Earth
    EXPECT_FALSE(IsAtmosphereTemperatureValid(-50.0f));   // Negative
}

TEST(LightingParamsTest, AtmosphereTemperatureBoundary) {
    // Test boundary values
    EXPECT_TRUE(IsAtmosphereTemperatureValid(LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MIN));
    EXPECT_TRUE(IsAtmosphereTemperatureValid(LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MAX));

    EXPECT_FALSE(IsAtmosphereTemperatureValid(LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MIN - 0.1f));
    EXPECT_FALSE(IsAtmosphereTemperatureValid(LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MAX + 0.1f));
}

// ============================================================================
// Physical Plausibility Tests
// ============================================================================

TEST(LightingParamsTest, SkyRadianceLessThanSun) {
    LightingParams params = CreateDefaultLightingParams();

    // Sky radiance should be less than sun radiance (diffuse << direct)
    EXPECT_LT(params.skyRadiance_spectral, params.sunRadiance_spectral);
    EXPECT_LT(glm::length(params.skyRadiance_rgb), glm::length(params.sunRadiance_rgb));
}

TEST(LightingParamsTest, ClearSkyEmissivityIsOffByDefault) {
    LightingParams params = CreateDefaultLightingParams();

    // Zero selects the isotropic blackbody sky, which is what the thermal
    // bands had before the flat-slab model and what a host gets until a
    // config asks for the analytic one. The slot previously held a scalar
    // atmospheric transmittance, deprecated when the NN atmosphere took over
    // the view path.
    EXPECT_FLOAT_EQ(params.skyEmissivityClear, 0.0f);
}

TEST(LightingParamsTest, SunDirectionPointingUp) {
    LightingParams params = CreateDefaultLightingParams();

    // Sun direction Y component should be positive (sun above horizon)
    EXPECT_GT(params.sunDirection.y, 0.0f);
}

// ============================================================================
// Zero-Initialization Safety Tests
// ============================================================================

TEST(LightingParamsTest, ZeroInitializationSafe) {
    // Zero-initialized struct should have sane values
    LightingParams params{};

    // All values should be zero (safe default)
    EXPECT_FLOAT_EQ(params.sunRadiance_spectral, 0.0f);
    EXPECT_FLOAT_EQ(params.skyRadiance_spectral, 0.0f);
    EXPECT_FLOAT_EQ(params.skyEmissivityClear, 0.0f);
    EXPECT_FLOAT_EQ(params.worldUnitsToMeters, 0.0f);
    EXPECT_FLOAT_EQ(params.atmosphereTemperature_K, 0.0f);
}

// ============================================================================
// IR Downwelling Radiation Physical Tests
// ============================================================================

TEST(LightingParamsTest, AtmosphereTemperaturePhysicalMeaning) {
    // Test that atmosphere temperature is in physically meaningful range
    // for computing IR downwelling radiation

    // Clear sky: 240-270K (effective radiating temperature)
    EXPECT_TRUE(IsAtmosphereTemperatureValid(250.0f));

    // Cloudy sky: 270-300K (clouds are warmer emitters)
    EXPECT_TRUE(IsAtmosphereTemperatureValid(285.0f));

    // These are the values typically used in IR remote sensing
    // See: Idso & Jackson (1969) "Thermal radiation from the atmosphere"
}

TEST(LightingParamsTest, AtmosphereTemperatureAffectsIRDownwelling) {
    // Higher atmosphere temperature = more IR downwelling radiation
    // This is a conceptual test - actual Planck calculation is in shader

    f32 coldSky = 240.0f;   // Clear, dry, cold
    f32 warmSky = 290.0f;   // Humid, cloudy, warm

    // Both should be valid
    EXPECT_TRUE(IsAtmosphereTemperatureValid(coldSky));
    EXPECT_TRUE(IsAtmosphereTemperatureValid(warmSky));

    // Warm sky emits more (Stefan-Boltzmann: L ∝ T^4)
    // This is tested conceptually; actual calculation in blackbody.hlsli
    EXPECT_GT(warmSky, coldSky);
}

// ============================================================================
// Config Integration Tests (Conceptual)
// ============================================================================

TEST(LightingParamsTest, DefaultConstantsConsistency) {
    // Verify default constants are consistent with CreateDefaultLightingParams
    LightingParams params = CreateDefaultLightingParams();

    EXPECT_FLOAT_EQ(params.sunRadiance_spectral, LightingDefaults::SUN_RADIANCE_SPECTRAL);
    EXPECT_FLOAT_EQ(params.skyRadiance_spectral, LightingDefaults::SKY_RADIANCE_SPECTRAL);
    EXPECT_FLOAT_EQ(params.skyEmissivityClear, LightingDefaults::SKY_EMISSIVITY_CLEAR);
    EXPECT_FLOAT_EQ(params.worldUnitsToMeters, LightingDefaults::WORLD_UNITS_TO_METERS);
    EXPECT_FLOAT_EQ(params.atmosphereTemperature_K, LightingDefaults::ATMOSPHERE_TEMPERATURE_K);
    EXPECT_FLOAT_EQ(params.chromaR_correction, LightingDefaults::CHROMA_R_CORRECTION);
    EXPECT_FLOAT_EQ(params.chromaB_correction, LightingDefaults::CHROMA_B_CORRECTION);
}

TEST(LightingParamsTest, ChromaCorrectionDefaultsToIdentity) {
    LightingParams params = CreateDefaultLightingParams();

    // These were 0.7872 and 1.0437 -- the G/R and G/B of the equal-energy
    // illuminant in sRGB -- because an RGB light source was upsampled to a flat
    // spectrum, which is E rather than D65, so a nominally white sky rendered
    // warm. The scale corrected that on the way out, and applied to every
    // VIS_FUSED render including the ones lit by a measured solar spectrum,
    // which it pushed 16.6% short in red.
    //
    // The illuminant is now fitted against D65, so (1,1,1) integrates back to
    // sRGB white unaided. A default other than identity would mean the renderer
    // still applies a white balance nobody asked for.
    EXPECT_FLOAT_EQ(params.chromaR_correction, 1.0f);
    EXPECT_FLOAT_EQ(params.chromaB_correction, 1.0f);
}
