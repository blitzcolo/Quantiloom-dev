/**
 * @file LightingParams.hpp
 * @brief Runtime lighting parameters for GPU shaders (sun/sky radiance, atmosphere config)
 *
 * Provides LightingParams struct for runtime lighting configuration:
 * - Sun direction and radiance (direct illumination)
 * - Sky radiance (ambient/diffuse illumination)
 * - Clear-sky emissivity and atmosphere temperature
 * - World unit scaling for physically-correct Beer-Lambert attenuation
 *
 * Dual-mode support:
 * - RGB mode: Uses sunRadiance_rgb and skyRadiance_rgb
 * - VIS_Fused mode: Uses RGB-to-spectrum conversion (spectral fields are fallback only)
 * - When SolarSpectralLUT available, these RGB values serve as additional fallback
 *
 * Physical units:
 * - RGB radiance: W·sr⁻¹·m⁻² (per RGB channel)
 * - Spectral fallback: W·sr⁻¹·m⁻² (RGB average, NOT per-nm spectral density)
 * - Temperature: Kelvin (K)
 * - worldUnitsToMeters: Conversion factor for scene unit scaling
 *
 * CRITICAL: This struct MUST match GPU LightingParams in common.hlsli exactly!
 * Memory layout validated by static_assert at compile time.
 *
 * @note Size: 64 bytes (16-byte aligned for GPU)
 * @note Uploaded to GPU via storage buffer (binding 2)
 * @note Any layout change requires shader update
 *
 * @see SolarSpectralLUT for wavelength-dependent illumination curves
 * @see AtmosphereNNConfig for the NN atmosphere configuration
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <cstddef>  // For offsetof

// ============================================================================
// LightingParams Data Structure
// ============================================================================

namespace quantiloom {

/**
 * @struct LightingParams
 * @brief Runtime lighting parameters for GPU shaders (80 bytes, 16-byte aligned)
 *
 * Provides sun/sky illumination and atmospheric parameters for all rendering modes.
 * Uploaded to GPU storage buffer and accessed in closest hit shaders.
 *
 * Memory layout (validated by static_assert):
 * @code
 * Offset  0: sunDirection (vec3, 12 bytes) + sunRadiance_spectral (f32, 4 bytes)
 * Offset 16: sunRadiance_rgb (vec3, 12 bytes) + skyRadiance_spectral (f32, 4 bytes)
 * Offset 32: skyRadiance_rgb (vec3, 12 bytes) + skyEmissivityClear (f32, 4 bytes)
 * Offset 48: worldUnitsToMeters + atmosphereTemperature_K + chromaR_correction + chromaB_correction
 * Offset 64: enableShadowRays (u32) + enableEnvironmentMap (u32) +
 *            emissiveTriangleCount (u32) + emissiveTotalPower (f32)
 * Total: 80 bytes
 * @endcode
 *
 * @note MUST match shader struct in common.hlsli (verified at compile time)
 * @note Use RGB values for RGB mode, spectral values for Single/MWIR/LWIR modes
 * @note atmosphereTemperature_K used for IR downwelling thermal radiation
 * @note chromaR/B_correction used for VIS_FUSED mode chromaticity correction
 * @note enableShadowRays: 0 = disabled (debug/GPU-crash workaround), 1 = enabled (default)
 */
struct LightingParams {
    glm::vec3 sunDirection;         // FROM surface TO sun (normalized), offset 0
    // NOTE: sunRadiance_spectral is RGB average fallback, NOT true spectral density
    // Unit: W·sr⁻¹·m⁻² (same as RGB), NOT W·sr⁻¹·m⁻²·nm⁻¹
    // VIS_Fused mode uses RGB-to-spectrum conversion, ignoring this field
    f32 sunRadiance_spectral;       // Spectral radiance fallback (RGB average), offset 12

    glm::vec3 sunRadiance_rgb;      // RGB radiance for RGB mode (fallback), offset 16
    // NOTE: skyRadiance_spectral is RGB average fallback, NOT true spectral density
    // Unit: W·sr⁻¹·m⁻² (same as RGB), NOT W·sr⁻¹·m⁻²·nm⁻¹
    // VIS_Fused mode uses RGB-to-spectrum conversion, ignoring this field
    f32 skyRadiance_spectral;       // Spectral radiance fallback (RGB average), offset 28

    glm::vec3 skyRadiance_rgb;      // RGB radiance for RGB mode (fallback), offset 32

    // Zenith emissivity of a clear sky, [0, 1], offset 44.
    //
    // Zero means the thermal sky is one isotropic blackbody at
    // atmosphereTemperature_K, which is what it was before this field carried
    // anything. Above zero the miss shader takes the flat-slab law instead --
    // eps(theta) = 1 - (1 - eps0)^sec(theta) -- so the sky is coldest at the
    // zenith and approaches the air temperature at the horizon, which is what
    // a thermal camera sees and what drives radiative cooling.
    //
    // Held here rather than the air temperature and humidity it comes from:
    // the Berdahl-Fromberg correlation and the dew point behind it are CPU
    // arithmetic done once per config, and shipping one number instead of two
    // keeps this struct at 80 bytes. Whoever sets it also sets
    // atmosphereTemperature_K to the air temperature.
    //
    // This slot was `transmittance`, deprecated when the NN atmosphere took
    // over the view path; its only reader was a placeholder function with no
    // callers.
    f32 skyEmissivityClear;         // offset 44

    f32 worldUnitsToMeters;         // Conversion factor: world_units × this = meters, offset 48
    f32 atmosphereTemperature_K;    // Effective atmosphere temperature (K) for IR downwelling, offset 52
    f32 chromaR_correction;         // VIS_FUSED chromaticity correction for R channel, offset 56
    f32 chromaB_correction;         // VIS_FUSED chromaticity correction for B channel, offset 60

    u32 enableShadowRays;           // Shadow ray enable flag: 0 = disabled, 1 = enabled, offset 64
    // Image-based lighting from the environment cubemap: 0 = the map contributes
    // nothing, 1 = it lights the scene. Off means off -- the surface gets no
    // environment specular at all, rather than the light of some substitute sky.
    // The background a ray sees when it misses is lighting.sky_radiance either
    // way; the miss shader has never read this map.
    //
    // Took the first of the three padding floats, so the struct is still 80
    // bytes and the shader mirror still matches.
    u32 enableEnvironmentMap;       // 0 = disabled, 1 = enabled, offset 68

    // Emissive geometry, for next-event estimation. Set by the renderer from
    // the scene, not by the caller -- SetLightingParams overwrites whatever it
    // is handed here.
    //
    // Zero triangles disables light sampling entirely, which is what every
    // scene lit only by sun and sky gets, and is why those scenes are
    // unchanged bit for bit.
    u32 emissiveTriangleCount;      // offset 72
    // Sum over emissive triangles of luminance(emissive) * area, in world
    // units squared. The sampling density on a triangle is
    // luminance(emissive) / this, because triangles are chosen in proportion
    // to their power and then uniformly over their area, so the area cancels.
    // That is what lets a surface which turns out to be an emitter compute the
    // density it would have been sampled with, from its own material alone,
    // with no way back to the triangle list.
    f32 emissiveTotalPower;         // offset 76

    // The two padding floats this struct was carrying are now both spoken for.
    // Anything further changes sizeof and therefore the SDK/Studio pairing.
};  // Total: 80 bytes

// ============================================================================
// Compile-time Validation
// ============================================================================
// These static_asserts ensure CPU/GPU struct layout consistency.
// If any fails, the struct layout has diverged from shader expectations.
// ============================================================================

static_assert(sizeof(LightingParams) == 80,
    "LightingParams size mismatch! Expected 80 bytes to match GPU struct");

static_assert(offsetof(LightingParams, sunDirection) == 0,
    "sunDirection offset mismatch");
static_assert(offsetof(LightingParams, sunRadiance_spectral) == 12,
    "sunRadiance_spectral offset mismatch");
static_assert(offsetof(LightingParams, sunRadiance_rgb) == 16,
    "sunRadiance_rgb offset mismatch");
static_assert(offsetof(LightingParams, skyRadiance_spectral) == 28,
    "skyRadiance_spectral offset mismatch");
static_assert(offsetof(LightingParams, skyRadiance_rgb) == 32,
    "skyRadiance_rgb offset mismatch");
static_assert(offsetof(LightingParams, skyEmissivityClear) == 44,
    "skyEmissivityClear offset mismatch");
static_assert(offsetof(LightingParams, worldUnitsToMeters) == 48,
    "worldUnitsToMeters offset mismatch");
static_assert(offsetof(LightingParams, atmosphereTemperature_K) == 52,
    "atmosphereTemperature_K offset mismatch");
static_assert(offsetof(LightingParams, chromaR_correction) == 56,
    "chromaR_correction offset mismatch");
static_assert(offsetof(LightingParams, chromaB_correction) == 60,
    "chromaB_correction offset mismatch");
static_assert(offsetof(LightingParams, enableEnvironmentMap) == 68,
    "enableEnvironmentMap offset mismatch");
static_assert(offsetof(LightingParams, enableShadowRays) == 64,
    "enableShadowRays offset mismatch");
static_assert(offsetof(LightingParams, emissiveTriangleCount) == 72,
    "emissiveTriangleCount offset mismatch");
static_assert(offsetof(LightingParams, emissiveTotalPower) == 76,
    "emissiveTotalPower offset mismatch");

// ============================================================================
// Default Values
// ============================================================================

namespace LightingDefaults {
    // Sun direction: 45° elevation, south-facing
    constexpr glm::vec3 SUN_DIRECTION = glm::vec3(0.0f, 0.707f, 0.707f);

    // Sun radiance: ~100 klux equivalent
    constexpr f32 SUN_RADIANCE_SPECTRAL = 1.0f;
    constexpr glm::vec3 SUN_RADIANCE_RGB = glm::vec3(1.0f, 1.0f, 1.0f);

    // Sky radiance: ~10% of sun
    constexpr f32 SKY_RADIANCE_SPECTRAL = 0.1f;
    constexpr glm::vec3 SKY_RADIANCE_RGB = glm::vec3(0.1f, 0.15f, 0.2f);

    // Clear-sky zenith emissivity: 0 keeps the isotropic blackbody sky, which
    // is what every scene rendered before the flat-slab model existed.
    constexpr f32 SKY_EMISSIVITY_CLEAR = 0.0f;

    // World units: meters
    constexpr f32 WORLD_UNITS_TO_METERS = 1.0f;

    // Atmosphere temperature for IR downwelling radiation
    // 260K = clear mid-latitude sky
    // Range: 240K (cold/dry) to 290K (hot/humid)
    constexpr f32 ATMOSPHERE_TEMPERATURE_K = 260.0f;
    constexpr f32 ATMOSPHERE_TEMPERATURE_K_MIN = 150.0f;  // Extreme cold
    constexpr f32 ATMOSPHERE_TEMPERATURE_K_MAX = 350.0f;  // Extreme hot

    // VIS_FUSED chromaticity correction factors
    // Derived from XYZ→RGB conversion for equal-integral CIE CMF data
    // For flat spectrum (gray input), XYZ ratio = (1:1:1), producing RGB ratio ≈ (1.27:1.0:0.96)
    // Correction factors normalize RGB output to neutral gray
    // NOTE: These values are for EQUAL-INTEGRAL CMF data (∫x̄=∫ȳ=∫z̄≈106.85)
    //       Standard CIE 1931 has UNEQUAL integrals and needs different factors!
    // Identity, and the history is worth keeping. These used to be 0.7872 and
    // 1.0437 -- exactly the G/R and G/B of the equal-energy illuminant in sRGB
    // -- because an RGB light source was upsampled to a FLAT spectrum, which is
    // E rather than D65, and a nominally white sky rendered visibly warm. The
    // fix was a diagonal scale on the final radiance, and it was applied to
    // every VIS_FUSED render including the spectrally correct ones: a scene lit
    // by a real D65 spectrum came out 16.6% short in red and 10.6% long in blue
    // because of it.
    //
    // The convention is now fixed where it was wrong. An RGB illuminant is
    // upsampled against D65, so (1,1,1) integrates back to sRGB white with no
    // correction at all, and a measured solar spectrum is left alone. The keys
    // remain for anyone who wants the old look; the default no longer applies a
    // white balance nobody asked for.
    constexpr f32 CHROMA_R_CORRECTION = 1.0f;
    constexpr f32 CHROMA_B_CORRECTION = 1.0f;
}

// ============================================================================
// Factory Functions
// ============================================================================

// Create LightingParams with default values
inline LightingParams CreateDefaultLightingParams() {
    LightingParams params{};
    params.sunDirection = glm::normalize(LightingDefaults::SUN_DIRECTION);
    params.sunRadiance_spectral = LightingDefaults::SUN_RADIANCE_SPECTRAL;
    params.sunRadiance_rgb = LightingDefaults::SUN_RADIANCE_RGB;
    params.skyRadiance_spectral = LightingDefaults::SKY_RADIANCE_SPECTRAL;
    params.skyRadiance_rgb = LightingDefaults::SKY_RADIANCE_RGB;
    params.skyEmissivityClear = LightingDefaults::SKY_EMISSIVITY_CLEAR;
    params.worldUnitsToMeters = LightingDefaults::WORLD_UNITS_TO_METERS;
    params.atmosphereTemperature_K = LightingDefaults::ATMOSPHERE_TEMPERATURE_K;
    params.chromaR_correction = LightingDefaults::CHROMA_R_CORRECTION;
    params.chromaB_correction = LightingDefaults::CHROMA_B_CORRECTION;
    // On, matching what a config gets when it does not name
    // renderer.enable_shadow_rays. This defaulted off, so a host starting from
    // these defaults -- which is what an interactive context does before any
    // config is applied -- rendered a scene shadowless that the CLI rendered
    // with shadows. The comment here used to cite a driver crash; the escape
    // hatch for that is the config key, which still turns them off.
    params.enableShadowRays = 1u;
    // Off, because a context that has just been constructed has no environment
    // map in it. This flag does not mean "the scene would like image-based
    // lighting"; it means "there is a real map bound at binding 10 and the
    // shader may sample it as a light source". Defaulting it on made a fresh
    // context light every scene with the fallback cubemap, which is a
    // placeholder, not sky. LoadEnvironmentMap raises it when a map actually
    // loads, and ConfigResolve computes it from the config.
    params.enableEnvironmentMap = 0u;
    // No scene yet, so no emitters. Filled in when one is built.
    params.emissiveTriangleCount = 0u;
    params.emissiveTotalPower = 0.0f;
    return params;
}

// Validate atmosphere temperature is within reasonable range
inline bool IsAtmosphereTemperatureValid(f32 temp_K) {
    return temp_K >= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MIN &&
           temp_K <= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MAX;
}

} // namespace quantiloom
