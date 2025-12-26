#pragma once

#include "core/LUT.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>

namespace quantiloom {

// ============================================================================
// LUTLoader - Atmosphere LUT I/O using TOML format
// ============================================================================
// TOML structure:
//   [metadata]
//   solar_zenith_deg = "30"
//   visibility_km = "23"
//
//   [data]
//   wavelengths = [380.0, 390.0, ...]
//   solar_irradiance = [1.0, 1.1, ...]
//   sky_radiance = [0.1, 0.15, ...]
//   transmittance = [0.95, 0.94, ...]
//
// File extension: .lut.toml (recommended) or .toml
// ============================================================================

class QL_API LUTLoader {
public:
    // Load LUT from TOML file
    static std::optional<AtmosphereLUT> LoadTOML(const std::string& filepath);

    // Save LUT to TOML file
    static bool SaveTOML(const std::string& filepath, const AtmosphereLUT& lut);

    // Check if file exists
    static bool FileExists(const std::string& filepath);

    // Get wavelength range without loading full LUT (fast peek)
    static std::optional<std::pair<f32, f32>> GetWavelengthRange(const std::string& filepath);
};

} // namespace quantiloom
