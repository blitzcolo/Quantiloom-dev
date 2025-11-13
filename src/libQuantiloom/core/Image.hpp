#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// Image - Generic multi-channel image container
// ============================================================================
// Used for:
// - MS-RT multi-band output (N channels, each named)
// - Single-band intermediate buffers
// - RGB/sRGB preview outputs
//
// Memory layout: Row-major, channel-last
//   data[y * width * channels + x * channels + c]
// This matches OpenEXR's scanline order and allows efficient iteration.
// ============================================================================

struct Image {
    // Dimensions
    u32 width = 0;
    u32 height = 0;
    u32 channels = 0;

    // Pixel data (row-major, channel-last: [y][x][c])
    // Always stored as f32, even if source is f16 or u8
    std::vector<f32> data;

    // Channel metadata (optional, for multi-spectral outputs)
    // e.g., {"VIS_550", "NIR_850", "SWIR_1600"}
    std::vector<std::string> channelNames;

    // Generic metadata (key-value pairs)
    // e.g., {"spp": "64", "mode": "MS-RT", "seconds_per_frame": "2.3"}
    std::unordered_map<std::string, std::string> metadata;

    // ========================================================================
    // Constructors
    // ========================================================================

    Image() = default;

    Image(u32 w, u32 h, u32 c)
        : width(w), height(h), channels(c), data(w * h * c, 0.0f) {
        channelNames.resize(c);
        for (u32 i = 0; i < c; ++i) {
            channelNames[i] = "Channel_" + std::to_string(i);
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get pixel value at (x, y, channel)
    // No bounds checking in release mode for performance
    inline f32& operator()(u32 x, u32 y, u32 c) {
        return data[y * width * channels + x * channels + c];
    }

    inline const f32& operator()(u32 x, u32 y, u32 c) const {
        return data[y * width * channels + x * channels + c];
    }

    // Get pointer to pixel (x, y) - useful for bulk operations
    inline f32* PixelPtr(u32 x, u32 y) {
        return &data[y * width * channels + x * channels];
    }

    inline const f32* PixelPtr(u32 x, u32 y) const {
        return &data[y * width * channels + x * channels];
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    // Total number of pixels
    inline u32 PixelCount() const { return width * height; }

    // Total number of elements (pixels * channels)
    inline u32 TotalElements() const { return width * height * channels; }

    // Check if image is valid
    inline bool IsValid() const {
        return width > 0 && height > 0 && channels > 0 &&
               data.size() == TotalElements();
    }

    // Clear image data (set all to zero)
    void Clear() { std::fill(data.begin(), data.end(), 0.0f); }

    // Resize image (will clear existing data)
    void Resize(u32 w, u32 h, u32 c) {
        width = w;
        height = h;
        channels = c;
        data.resize(w * h * c, 0.0f);
        channelNames.resize(c);
        for (u32 i = 0; i < c; ++i) {
            channelNames[i] = "Channel_" + std::to_string(i);
        }
    }
};

} // namespace quantiloom
