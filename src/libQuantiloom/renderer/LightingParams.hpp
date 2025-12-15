#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <cstddef>  // For offsetof

// ============================================================================
// LightingParams Data Structure
// ============================================================================
// Runtime lighting parameters for GPU shading. Provides sun/sky radiance values
// and atmospheric parameters for both RGB and spectral rendering modes.
//
// CRITICAL: This structure MUST match GPU-side LightingParams in common.hlsli!
// Any change here requires corresponding update in the shader.
//
// MEMORY LAYOUT (64 bytes total, 16-byte aligned):
// - Offset  0: sunDirection (12 bytes) + sunRadiance_spectral (4 bytes)
// - Offset 16: sunRadiance_rgb (12 bytes) + skyRadiance_spectral (4 bytes)
// - Offset 32: skyRadiance_rgb (12 bytes) + transmittance (4 bytes)
// - Offset 48: worldUnitsToMeters (4 bytes) + atmosphereTemperature_K (4 bytes) + padding (8 bytes)
//
// USAGE:
// - RGB mode: Use sunRadiance_rgb and skyRadiance_rgb
// - Spectral mode: Use sunRadiance_spectral and skyRadiance_spectral
// - MWIR/LWIR mode: Use atmosphereTemperature_K for downwelling thermal radiation
//
// NOTE: When SolarSpectralLUT is available, the spectral/rgb values serve as fallback.
// ============================================================================

namespace quantiloom {

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
