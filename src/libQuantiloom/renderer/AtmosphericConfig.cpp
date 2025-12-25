#include "AtmosphericConfig.hpp"
#include "core/Log.hpp"
#include <toml++/toml.h>
#include <fstream>

namespace quantiloom {

// ============================================================================
// Constructor
// ============================================================================

AtmosphericConfig::AtmosphericConfig() {
    // Initialize with ClearDay defaults (inline to avoid recursion)
    rayleigh_enabled = true;
    rayleigh_beta_550nm = 5.8e-6f;
    rayleigh_scale_height = 8500.0f;

    mie_enabled = true;
    mie_beta_550nm = 2.0e-6f;
    mie_scale_height = 1200.0f;
    mie_g = 0.76f;
    mie_alpha = 0.84f;

    planet_radius = 6.371e6f;
    atmosphere_height = 60000.0f;

    max_distance = 100000.0f;
    max_steps = 64;
    extinction_threshold = 0.01f;
}

// ============================================================================
// Preset Configurations
// ============================================================================

auto AtmosphericConfig::ClearDay() -> AtmosphericConfig {
    AtmosphericConfig config;
    config.rayleigh_enabled = true;
    config.rayleigh_beta_550nm = 5.8e-6f;
    config.rayleigh_scale_height = 8500.0f;

    config.mie_enabled = true;
    config.mie_beta_550nm = 2.0e-6f;    // Clear visibility (~23km)
    config.mie_scale_height = 1200.0f;
    config.mie_g = 0.76f;
    config.mie_alpha = 0.84f;

    config.planet_radius = 6.371e6f;
    config.atmosphere_height = 60000.0f;

    config.max_distance = 100000.0f;
    config.max_steps = 64;
    config.extinction_threshold = 0.01f;

    return config;
}

auto AtmosphericConfig::Hazy() -> AtmosphericConfig {
    AtmosphericConfig config = ClearDay();
    config.mie_beta_550nm = 10.0e-6f;   // Reduced visibility (~4.5km)
    config.mie_g = 0.80f;               // More forward scattering
    return config;
}

auto AtmosphericConfig::PollutedUrban() -> AtmosphericConfig {
    AtmosphericConfig config = ClearDay();
    config.mie_beta_550nm = 20.0e-6f;   // Very low visibility (~2km)
    config.mie_scale_height = 800.0f;   // Lower aerosol layer
    config.mie_g = 0.82f;
    return config;
}

auto AtmosphericConfig::MountainTop() -> AtmosphericConfig {
    AtmosphericConfig config = ClearDay();
    config.mie_beta_550nm = 0.5e-6f;    // Extremely clear (~90km visibility)
    config.rayleigh_beta_550nm = 4.0e-6f; // Thinner atmosphere
    return config;
}

auto AtmosphericConfig::Mars() -> AtmosphericConfig {
    AtmosphericConfig config;
    config.rayleigh_enabled = false;    // CO2 atmosphere, different scattering
    config.mie_enabled = true;
    config.mie_beta_550nm = 50.0e-6f;   // Dust-dominated
    config.mie_scale_height = 10000.0f; // Higher dust layer
    config.mie_g = 0.90f;               // Very forward scattering (dust)
    config.mie_alpha = 0.5f;            // Red-shifted

    config.planet_radius = 3.3895e6f;   // Mars radius
    config.atmosphere_height = 50000.0f;
    config.max_distance = 50000.0f;

    return config;
}

auto AtmosphericConfig::Disabled() -> AtmosphericConfig {
    AtmosphericConfig config;
    config.rayleigh_enabled = false;
    config.mie_enabled = false;
    config.rayleigh_beta_550nm = 0.0f;
    config.mie_beta_550nm = 0.0f;
    return config;
}

// ============================================================================
// TOML Loading
// ============================================================================

auto AtmosphericConfig::FromTOML(const String& tomlPath) -> Result<AtmosphericConfig> {
    try {
        // Parse TOML file
        auto table = toml::parse_file(tomlPath);

        // Check if [atmospheric] section exists
        auto atmo_table = table["atmospheric"];
        if (!atmo_table) {
            // No atmospheric config: use default ClearDay
            Log::Info("No [atmospheric] section in {}, using ClearDay preset", tomlPath);
            return Result(ClearDay());
        }

        AtmosphericConfig config;

        // Load preset (optional)
        if (auto preset = atmo_table["preset"].value<std::string>()) {
            if (*preset == "clear_day") {
                config = ClearDay();
            } else if (*preset == "hazy") {
                config = Hazy();
            } else if (*preset == "polluted_urban") {
                config = PollutedUrban();
            } else if (*preset == "mountain_top") {
                config = MountainTop();
            } else if (*preset == "mars") {
                config = Mars();
            } else if (*preset == "disabled") {
                config = Disabled();
            } else {
                Log::Warn("Unknown atmospheric preset '{}', using ClearDay", *preset);
                config = ClearDay();
            }
        }

        // Override with custom values (if provided)
        if (auto enabled = atmo_table["rayleigh_enabled"].value<bool>()) {
            config.rayleigh_enabled = *enabled;
        }
        if (auto beta = atmo_table["rayleigh_beta_550nm"].value<double>()) {
            config.rayleigh_beta_550nm = static_cast<f32>(*beta);
        }
        if (auto height = atmo_table["rayleigh_scale_height"].value<double>()) {
            config.rayleigh_scale_height = static_cast<f32>(*height);
        }

        if (auto enabled = atmo_table["mie_enabled"].value<bool>()) {
            config.mie_enabled = *enabled;
        }
        if (auto beta = atmo_table["mie_beta_550nm"].value<double>()) {
            config.mie_beta_550nm = static_cast<f32>(*beta);
        }
        if (auto height = atmo_table["mie_scale_height"].value<double>()) {
            config.mie_scale_height = static_cast<f32>(*height);
        }
        if (auto g = atmo_table["mie_g"].value<double>()) {
            config.mie_g = static_cast<f32>(*g);
        }
        if (auto alpha = atmo_table["mie_alpha"].value<double>()) {
            config.mie_alpha = static_cast<f32>(*alpha);
        }

        if (auto radius = atmo_table["planet_radius"].value<double>()) {
            config.planet_radius = static_cast<f32>(*radius);
        }
        if (auto height = atmo_table["atmosphere_height"].value<double>()) {
            config.atmosphere_height = static_cast<f32>(*height);
        }

        if (auto dist = atmo_table["max_distance"].value<double>()) {
            config.max_distance = static_cast<f32>(*dist);
        }
        if (auto steps = atmo_table["max_steps"].value<int64_t>()) {
            config.max_steps = static_cast<u32>(*steps);
        }
        if (auto thresh = atmo_table["extinction_threshold"].value<double>()) {
            config.extinction_threshold = static_cast<f32>(*thresh);
        }

        Log::Info("Loaded atmospheric config from {}: Rayleigh={}, Mie={}",
                  tomlPath, config.rayleigh_enabled, config.mie_enabled);

        return Result(config);

    } catch (const toml::parse_error& err) {
        return Result<AtmosphericConfig>(Result<AtmosphericConfig>::Err(
            String("Failed to parse TOML: ") + err.what()));
    } catch (const std::exception& err) {
        return Result<AtmosphericConfig>(Result<AtmosphericConfig>::Err(
            String("Error loading atmospheric config: ") + err.what()));
    }
}

// ============================================================================
// GPU Conversion
// ============================================================================

auto AtmosphericConfig::ToGPU() const -> AtmosphericParamsGPU {
    AtmosphericParamsGPU gpu{};

    // Rayleigh scattering (replicate scalar to RGB for now)
    // For spectral rendering, this will be computed per-wavelength in shader
    f32 beta_r = rayleigh_enabled ? rayleigh_beta_550nm : 0.0f;
    gpu.beta_rayleigh_550nm = glm::vec3(beta_r, beta_r, beta_r);
    gpu.rayleigh_scale_height = rayleigh_scale_height;

    // Mie scattering
    f32 beta_m = mie_enabled ? mie_beta_550nm : 0.0f;
    gpu.beta_mie_550nm = glm::vec3(beta_m, beta_m, beta_m);
    gpu.mie_scale_height = mie_scale_height;

    // Mie phase function
    gpu.mie_g = mie_g;
    gpu.mie_alpha = mie_alpha;

    // Planet geometry
    gpu.planet_radius = planet_radius;
    gpu.atmosphere_height = atmosphere_height;

    // Delta-Tracking parameters
    gpu.max_distance = max_distance;
    gpu.max_steps = max_steps;
    gpu.extinction_threshold = extinction_threshold;
    gpu._padding = 0;

    return gpu;
}

} // namespace quantiloom
