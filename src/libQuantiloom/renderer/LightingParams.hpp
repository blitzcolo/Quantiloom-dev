/**
 * @file LightingParams.hpp
 * @brief Runtime lighting parameters for GPU shaders (sun/sky radiance, atmosphere config)
 *
 * Provides LightingParams struct for runtime lighting configuration:
 * - Sun direction and radiance (direct illumination)
 * - Sky radiance (ambient/diffuse illumination)
 * - Atmospheric transmittance and temperature
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
 * @author wtflmao
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
 * Offset 32: skyRadiance_rgb (vec3, 12 bytes) + transmittance (f32, 4 bytes)
 * Offset 48: worldUnitsToMeters + atmosphereTemperature_K + chromaR_correction + chromaB_correction
 * Offset 64: enableShadowRays (u32) + _padding[3] (12 bytes)
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
    f32 transmittance;              // Atmospheric transmittance τ(λ) [0, 1], offset 44

    f32 worldUnitsToMeters;         // Conversion factor: world_units × this = meters, offset 48
    f32 atmosphereTemperature_K;    // Effective atmosphere temperature (K) for IR downwelling, offset 52
    f32 chromaR_correction;         // VIS_FUSED chromaticity correction for R channel, offset 56
    f32 chromaB_correction;         // VIS_FUSED chromaticity correction for B channel, offset 60

    u32 enableShadowRays;           // Shadow ray enable flag: 0 = disabled, 1 = enabled, offset 64
    f32 _padding[3];                // Padding to 80 bytes (16-byte aligned), offset 68-80
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
static_assert(offsetof(LightingParams, transmittance) == 44,
    "transmittance offset mismatch");
static_assert(offsetof(LightingParams, worldUnitsToMeters) == 48,
    "worldUnitsToMeters offset mismatch");
static_assert(offsetof(LightingParams, atmosphereTemperature_K) == 52,
    "atmosphereTemperature_K offset mismatch");
static_assert(offsetof(LightingParams, chromaR_correction) == 56,
    "chromaR_correction offset mismatch");
static_assert(offsetof(LightingParams, chromaB_correction) == 60,
    "chromaB_correction offset mismatch");
static_assert(offsetof(LightingParams, enableShadowRays) == 64,
    "enableShadowRays offset mismatch");

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

    // Atmospheric transmittance: clear sky
    constexpr f32 TRANSMITTANCE = 0.9f;

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
    constexpr f32 CHROMA_R_CORRECTION = 0.7872f;  // G/R ratio to neutralize red shift
    constexpr f32 CHROMA_B_CORRECTION = 1.0437f;  // G/B ratio to neutralize blue deficit
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
    params.transmittance = LightingDefaults::TRANSMITTANCE;
    params.worldUnitsToMeters = LightingDefaults::WORLD_UNITS_TO_METERS;
    params.atmosphereTemperature_K = LightingDefaults::ATMOSPHERE_TEMPERATURE_K;
    params.chromaR_correction = LightingDefaults::CHROMA_R_CORRECTION;
    params.chromaB_correction = LightingDefaults::CHROMA_B_CORRECTION;
    params.enableShadowRays = 0u;  // Disabled by default (known GPU crash on some drivers)
    return params;
}

// Validate atmosphere temperature is within reasonable range
inline bool IsAtmosphereTemperatureValid(f32 temp_K) {
    return temp_K >= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MIN &&
           temp_K <= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MAX;
}

} // namespace quantiloom
