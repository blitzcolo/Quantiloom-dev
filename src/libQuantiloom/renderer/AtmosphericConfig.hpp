#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>

namespace quantiloom {

// ============================================================================
// AtmosphericParams - GPU Structure
// ============================================================================
// Must match shader structure in common.hlsli
// Size: 64 bytes (2×vec3 + 10×f32/u32, all 16-byte aligned)
// ============================================================================

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

class AtmosphericConfig {
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
    auto ToGPU() const -> AtmosphericParamsGPU;

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
