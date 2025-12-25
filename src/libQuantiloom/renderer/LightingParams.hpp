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
 * - Spectral mode: Uses sunRadiance_spectral and skyRadiance_spectral
 * - When SolarSpectralLUT available, these values serve as fallback
 *
 * Physical units:
 * - Radiance: W·sr⁻¹·m⁻² (RGB mode) or W·sr⁻¹·m⁻²·nm⁻¹ (spectral mode)
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
 * @see AtmosphericConfig for atmospheric scattering parameters
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
 * @brief Runtime lighting parameters for GPU shaders (64 bytes, 16-byte aligned)
 *
 * Provides sun/sky illumination and atmospheric parameters for all rendering modes.
 * Uploaded to GPU storage buffer and accessed in closest hit shaders.
 *
 * Memory layout (validated by static_assert):
 * @code
 * Offset  0: sunDirection (vec3, 12 bytes) + sunRadiance_spectral (f32, 4 bytes)
 * Offset 16: sunRadiance_rgb (vec3, 12 bytes) + skyRadiance_spectral (f32, 4 bytes)
 * Offset 32: skyRadiance_rgb (vec3, 12 bytes) + transmittance (f32, 4 bytes)
 * Offset 48: worldUnitsToMeters (f32, 4 bytes) + atmosphereTemperature_K (f32, 4 bytes) + _padding (vec2, 8 bytes)
 * Total: 64 bytes
 * @endcode
 *
 * @note MUST match shader struct in common.hlsli (verified at compile time)
 * @note Use RGB values for RGB_Fused mode, spectral values for Single/MWIR/LWIR modes
 * @note atmosphereTemperature_K used for IR downwelling thermal radiation
 */
struct LightingParams {
    glm::vec3 sunDirection;         // FROM surface TO sun (normalized), offset 0
    f32 sunRadiance_spectral;       // Spectral radiance at current λ (fallback), offset 12

    glm::vec3 sunRadiance_rgb;      // RGB radiance for RGB mode (fallback), offset 16
    f32 skyRadiance_spectral;       // Spectral radiance at current λ (fallback), offset 28

    glm::vec3 skyRadiance_rgb;      // RGB radiance for RGB mode (fallback), offset 32
    f32 transmittance;              // Atmospheric transmittance τ(λ) [0, 1], offset 44

    f32 worldUnitsToMeters;         // Conversion factor: world_units × this = meters, offset 48
    f32 atmosphereTemperature_K;    // Effective atmosphere temperature (K) for IR downwelling, offset 52
    glm::vec2 _padding;             // Padding for 16-byte alignment, offset 56
};  // Total: 64 bytes

// ============================================================================
// Compile-time Validation
// ============================================================================
// These static_asserts ensure CPU/GPU struct layout consistency.
// If any fails, the struct layout has diverged from shader expectations.
// ============================================================================

static_assert(sizeof(LightingParams) == 64,
    "LightingParams size mismatch! Expected 64 bytes to match GPU struct");

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
static_assert(offsetof(LightingParams, _padding) == 56,
    "_padding offset mismatch");

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
    params._padding = glm::vec2(0.0f);
    return params;
}

// Validate atmosphere temperature is within reasonable range
inline bool IsAtmosphereTemperatureValid(f32 temp_K) {
    return temp_K >= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MIN &&
           temp_K <= LightingDefaults::ATMOSPHERE_TEMPERATURE_K_MAX;
}

} // namespace quantiloom
