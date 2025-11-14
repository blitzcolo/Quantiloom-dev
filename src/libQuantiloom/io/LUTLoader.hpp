#pragma once

#include "core/LUT.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>

namespace quantiloom {

// ============================================================================
// LUTLoader - MODTRAN LUT loading from HDF5
// ============================================================================
// Expected HDF5 structure:
//   /wavelengths         - 1D dataset [n], float32, nm
//   /solar_irradiance    - 1D dataset [n], float32, W/m^2/nm
//   /sky_radiance        - 1D dataset [n], float32, W/m^2/sr/nm
//   /transmittance       - 1D dataset [n], float32, dimensionless
//   /metadata            - Group with string attributes
//
// This structure is compatible with our dummy LUT generator and
// future MODTRAN export scripts.
// ============================================================================

class QL_API LUTLoader {
public:
    // ========================================================================
    // LUT Loading
    // ========================================================================

    // Load LUT from HDF5 file
    static std::optional<AtmosphereLUT> LoadHDF5(const std::string& filepath);

    // ========================================================================
    // LUT Saving (for test/debug purposes)
    // ========================================================================

    // Save LUT to HDF5 file
    static bool SaveHDF5(const std::string& filepath, const AtmosphereLUT& lut);

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if file exists
    static bool FileExists(const std::string& filepath);

    // Get wavelength range without loading full LUT (fast peek)
    static std::optional<std::pair<f32, f32>> GetWavelengthRange(const std::string& filepath);
};

} // namespace quantiloom
