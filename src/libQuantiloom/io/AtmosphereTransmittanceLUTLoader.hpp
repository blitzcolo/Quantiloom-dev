#pragma once

#include "core/AtmosphereTransmittanceLUT.hpp"
#include "core/Types.hpp"
#include <filesystem>
#include <optional>

namespace quantiloom {

namespace fs = std::filesystem;

// ============================================================================
// AtmosphereTransmittanceLUTLoader
// ============================================================================
// Loads 3D atmospheric transmittance LUTs from .qlut files.
//
// FILE FORMAT (.qlut):
//   The .qlut format combines TOML metadata with binary data for efficient
//   loading while maintaining human-readable configuration.
//
//   Structure:
//   ┌─────────────────────────────────────┐
//   │  TOML Header (1024 bytes, padded)   │  <- Human-readable metadata
//   ├─────────────────────────────────────┤
//   │  Transmittance Data (float32[])     │  <- Binary, C-order
//   ├─────────────────────────────────────┤
//   │  Path Radiance Data (float32[])     │  <- Binary, C-order (optional)
//   └─────────────────────────────────────┘
//
//   TOML Header Example:
//   ```toml
//   [metadata]
//   name = "Midlatitude Summer"
//   source = "libRadtran 2.0.4"
//   created = "2024-12-25"
//   atmospheric_model = "US_Standard_1976"
//
//   [grid.wavelength]
//   start = 300.0
//   stop = 14000.0
//   step = 10.0
//   count = 1371
//
//   [grid.altitude]
//   start = 0.0
//   stop = 30000.0
//   step = 1000.0
//   count = 31
//
//   [grid.zenith]
//   values = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
//
//   [data]
//   header_size = 2048
//   transmittance_offset = 2048
//   transmittance_count = 297501
//   path_radiance_offset = 2191028
//   path_radiance_count = 297501
//   dtype = "float32"
//   byte_order = "little"
//   ```
//
// USAGE:
//   auto result = AtmosphereTransmittanceLUTLoader::Load("atmosphere.qlut");
//   if (result) {
//       float tau = result->QueryTransmittance(4000.0f, 1000.0f, 30.0f);
//   }
//
// GENERATION:
//   See scripts/atmosphere-qlut-gen/README.md for Python generation tools.
// ============================================================================

class QL_API AtmosphereTransmittanceLUTLoader {
public:
    // ========================================================================
    // Load from .qlut file
    // ========================================================================
    // Returns std::nullopt on failure (with error logged)
    static auto Load(const fs::path& filepath) -> Optional<AtmosphereTransmittanceLUT>;

    // ========================================================================
    // Save to .qlut file
    // ========================================================================
    // For testing and LUT generation from C++
    // Returns true on success
    static auto Save(const fs::path& filepath, const AtmosphereTransmittanceLUT& lut) -> bool;

    // ========================================================================
    // Quick Metadata Peek
    // ========================================================================
    // Read only the TOML header without loading binary data
    // Useful for listing available LUTs and their properties

    struct LUTInfo {
        String name;
        String source;
        String created;
        String atmospheric_model;
        f32 wavelength_min;
        f32 wavelength_max;
        f32 altitude_max;
        usize data_size_bytes;
    };

    static auto PeekInfo(const fs::path& filepath) -> Optional<LUTInfo>;

    // ========================================================================
    // File Format Constants
    // ========================================================================
    static constexpr usize HEADER_SIZE = 2048;  // Fixed TOML header size (bytes)
    static constexpr const char* FILE_EXTENSION = ".qlut";
    static constexpr u32 FORMAT_VERSION = 1;

private:
    // Parse TOML header string into LUT metadata and axis info
    static auto ParseHeader(const String& header, AtmosphereTransmittanceLUT& lut) -> bool;

    // Generate TOML header string from LUT
    static auto GenerateHeader(const AtmosphereTransmittanceLUT& lut) -> String;
};

} // namespace quantiloom
