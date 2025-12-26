/**
 * @file AtmosphericConfig.hpp
 * @brief Atmospheric scattering configuration for volumetric rendering
 *
 * Provides atmospheric rendering configuration:
 * - AtmosphericParamsGPU: GPU-side parameters (64 bytes, matches shader)
 * - AtmosphericConfig: CPU-side configuration with preset factories
 *
 * Atmospheric scattering model:
 * - Rayleigh scattering: Molecular scattering (wavelength^-4 dependence)
 * - Mie scattering: Aerosol scattering (wavelength^-alpha dependence)
 * - Delta-tracking: Unbiased volumetric path tracing
 *
 * Presets:
 * - ClearDay: Standard atmosphere (visibility 23km, sea level)
 * - Hazy: Reduced visibility (10km, high aerosol loading)
 * - PollutedUrban: Heavy pollution (5km visibility)
 * - MountainTop: High altitude (3000m, thin atmosphere)
 * - Mars: Martian atmosphere (CO2, dust storms)
 * - Disabled: No atmospheric scattering (for indoor/small scenes)
 *
 * Physical parameters:
 * - beta_rayleigh_550nm: Rayleigh scattering coefficient at 550nm (m⁻¹)
 * - beta_mie_550nm: Mie scattering coefficient at 550nm (m⁻¹)
 * - scale_height: Exponential falloff with altitude H = H0 * exp(-h/H_scale)
 * - mie_g: Asymmetry parameter (forward scattering bias, -1 to 1)
 * - mie_alpha: Angstrom exponent (wavelength dependence, typically 0-2)
 *
 * Usage example:
 * @code
 * // Use preset
 * AtmosphericConfig config = AtmosphericConfig::ClearDay();
 * AtmosphericParamsGPU gpuParams = config.ToGPU();
 *
 * // Upload to GPU
 * GpuBuffer atmosBuffer(allocator, sizeof(AtmosphericParamsGPU),
 *                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
 *                       VMA_MEMORY_USAGE_CPU_TO_GPU);
 * atmosBuffer.Upload(&gpuParams, sizeof(AtmosphericParamsGPU));
 *
 * // Or load from TOML
 * auto result = AtmosphericConfig::FromTOML("atmosphere.toml");
 * @endcode
 *
 * @note GPU struct size: 64 bytes (verified by static_assert)
 * @note For indoor/small scenes, use Disabled() preset (avoids artifacts)
 * @note Automatic enable/disable based on scene bounding box (see main.cpp)
 *
 * @see LightingParams for basic sun/sky illumination parameters
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <glm/glm.hpp>

namespace quantiloom {

// ============================================================================
// AtmosphericParams - GPU Structure
// ============================================================================
/**
 * @struct AtmosphericParamsGPU
 * @brief GPU-side atmospheric scattering parameters (64 bytes, matches common.hlsli)
 *
 * Contains all parameters for wavelength-dependent volumetric atmospheric rendering:
 * - Rayleigh scattering (molecular): beta, scale height
 * - Mie scattering (aerosol): beta, scale height, phase function
 * - Planet geometry: radius, atmosphere height
 * - Delta-tracking: max distance, max steps, extinction threshold
 *
 * @note MUST match shader struct in common.hlsli (verified by static_assert)
 * @note Size: 64 bytes (4×16-byte aligned blocks)
 */
struct AtmosphericParamsGPU {
    // Rayleigh scattering (molecular) - 16 bytes
    glm::vec3 beta_rayleigh_550nm;      // Scattering coefficient at 550nm (m^-1)
    f32 rayleigh_scale_height;          // Scale height H_r (meters)

    // Mie scattering (aerosol) - 16 bytes
    glm::vec3 beta_mie_550nm;           // Scattering coefficient at 550nm (m^-1)
    f32 mie_scale_height;               // Scale height H_m (meters)

    // Mie phase function - 16 bytes
    f32 mie_g;                          // Asymmetry parameter
    f32 mie_alpha;                      // Angstrom exponent
    f32 planet_radius;                  // Planet radius (meters)
    f32 atmosphere_height;              // Atmosphere top (meters)

    // Delta-Tracking parameters - 16 bytes
    f32 max_distance;                   // Maximum ray march distance (meters)
    u32 max_steps;                      // Maximum steps
    f32 extinction_threshold;           // Early termination threshold
    u32 _padding;                       // 16-byte alignment
};

// Static assertion to ensure size matches GPU expectations
static_assert(sizeof(AtmosphericParamsGPU) == 64, "AtmosphericParamsGPU size mismatch");

// ============================================================================
// AtmosphericConfig - CPU Configuration
// ============================================================================
// High-level configuration for atmospheric rendering
// Provides preset configurations and TOML loading
// ============================================================================

// TODO: Move atmospheric config to Scene API - should be part of scene description
// Current DLL export is temporary for backward compatibility
class QL_API AtmosphericConfig {
public:
    // Constructors
    AtmosphericConfig();
    ~AtmosphericConfig() = default;

    // Preset configurations
    static auto ClearDay() -> AtmosphericConfig;
    static auto Hazy() -> AtmosphericConfig;
    static auto PollutedUrban() -> AtmosphericConfig;
    static auto MountainTop() -> AtmosphericConfig;
    static auto Mars() -> AtmosphericConfig;
    static auto Disabled() -> AtmosphericConfig;

    // Load from TOML configuration
    static auto FromTOML(const String& tomlPath) -> Result<AtmosphericConfig, String>;

    // Convert to GPU structure
    [[nodiscard]] auto ToGPU() const -> AtmosphericParamsGPU;

    // Check if atmospheric rendering is enabled
    [[nodiscard]] auto IsEnabled() const -> bool {
        return rayleigh_enabled || mie_enabled;
    }

    // ========================================================================
    // Configuration Parameters
    // ========================================================================

    // Rayleigh scattering (molecular)
    bool rayleigh_enabled = true;
    f32 rayleigh_beta_550nm = 5.8e-6f;      // m^-1 (typical: 5.8e-6)
    f32 rayleigh_scale_height = 8500.0f;    // meters (typical: 8000-8500m)

    // Mie scattering (aerosol)
    bool mie_enabled = true;
    f32 mie_beta_550nm = 2.0e-6f;           // m^-1 (typical: 2e-6 for clear, 20e-6 for hazy)
    f32 mie_scale_height = 1200.0f;         // meters (typical: 1200m)
    f32 mie_g = 0.76f;                      // Henyey-Greenstein asymmetry (typical: 0.76)
    f32 mie_alpha = 0.84f;                  // Angstrom exponent (typical: 0.84)

    // Planet geometry
    f32 planet_radius = 6.371e6f;           // meters (Earth: 6.371e6m)
    f32 atmosphere_height = 60000.0f;       // meters (typical: 60km)

    // Delta-Tracking parameters
    f32 max_distance = 100000.0f;           // meters (max ray march distance)
    u32 max_steps = 64;                     // Maximum steps (typical: 32-128)
    f32 extinction_threshold = 0.01f;       // Early termination (typical: 0.01 = 1%)
};

} // namespace quantiloom
