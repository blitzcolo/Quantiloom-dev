#pragma once

#include "core/SpectralCube.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>

namespace quantiloom {

// ============================================================================
// SpectralIO - HDF5 hyperspectral cube reading/writing
// ============================================================================
// HDF5 structure:
//   /data              - 3D dataset [nbands, height, width], float32
//   /wavelengths       - 1D dataset [nbands], float32
//   /metadata          - Group containing string attributes
//
// Memory layout:
//   C-order (row-major): data[b][y][x]
//   This matches HDF5's default layout and allows per-band processing
//
// Why HDF5 instead of EXR for hyperspectral?
// - EXR has practical channel limit (~1000)
// - HDF5 supports arbitrary dimensions and chunking
// - HDF5 is standard in scientific computing (MODTRAN, hyperspectral sensors)
// ============================================================================

class QL_API SpectralIO {
public:
    // ========================================================================
    // HDF5 Writing
    // ========================================================================

    // Write spectral cube to HDF5 file
    static bool WriteHDF5(const std::string& filepath, const SpectralCube& cube);

    // ========================================================================
    // HDF5 Reading
    // ========================================================================

    // Read spectral cube from HDF5 file
    static std::optional<SpectralCube> ReadHDF5(const std::string& filepath);

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if file exists and is valid HDF5
    static bool FileExists(const std::string& filepath);

    // Get cube dimensions without loading data (fast peek)
    static std::optional<std::tuple<u32, u32, u32>> GetDimensions(const std::string& filepath);
};

} // namespace quantiloom
