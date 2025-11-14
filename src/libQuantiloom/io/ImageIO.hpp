#pragma once

#include "core/Image.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>

namespace quantiloom {

// ============================================================================
// ImageIO - EXR image reading/writing using OpenEXR 3.x
// ============================================================================
// Supported formats:
// - Single-channel (grayscale)
// - Multi-channel (arbitrary number of channels)
// - Preserves channel names and metadata
//
// Notes:
// - Always converts to/from f32 internally (OpenEXR uses HALF by default)
// - Metadata is stored as string attributes in EXR header
// ============================================================================

class QL_API ImageIO {
public:
    // ========================================================================
    // EXR Writing
    // ========================================================================

    // Write image to EXR file
    // Returns true on success, false on failure
    static bool WriteEXR(const std::string& filepath, const Image& image);

    // ========================================================================
    // EXR Reading
    // ========================================================================

    // Read image from EXR file
    // Returns std::nullopt on failure
    static std::optional<Image> ReadEXR(const std::string& filepath);

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if file exists and is readable
    static bool FileExists(const std::string& filepath);

    // Get image dimensions without loading full image (fast peek)
    static std::optional<std::tuple<u32, u32, u32>> GetDimensions(const std::string& filepath);
};

} // namespace quantiloom
