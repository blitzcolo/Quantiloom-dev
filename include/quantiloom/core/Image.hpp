/**
 * @file Image.hpp
 * @brief Generic multi-channel image container for spectral/multi-band rendering
 *
 * Provides Image struct for storing:
 * - RGB/RGBA images (standard rendering output)
 * - Single-channel grayscale images (monochromatic spectral output)
 * - Multi-band spectral cubes (hyperspectral rendering, N channels)
 * - HDR floating-point data (physical radiance units)
 *
 * Memory layout: Row-major, channel-last (matches OpenEXR scanline order):
 *   data[y * width * channels + x * channels + c]
 *
 * This layout enables:
 * - Efficient scanline iteration for I/O
 * - Cache-friendly pixel access
 * - Direct interop with OpenEXR, stb_image, etc.
 *
 * All pixel data is stored as f32, even if source is f16/u8 (converted during load).
 * Supports optional channel naming and key-value metadata for multi-spectral output.
 *
 * @author blitzcolo
 */

#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// Image - Generic multi-channel image container
// ============================================================================
/**
 * @struct Image
 * @brief Generic multi-channel floating-point image container
 *
 * Stores arbitrary-channel images (RGB, RGBA, grayscale, multi-spectral) with:
 * - f32 pixel data (HDR-capable, physical radiance units)
 * - Channel naming (for multi-spectral output identification)
 * - Metadata key-value store (rendering params, spectral config, etc.)
 *
 * Usage examples:
 * @code
 * // Create RGB image
 * Image rgb(1920, 1080, 3);
 * rgb(0, 0, 0) = 1.0f;  // Red pixel at (0,0)
 *
 * // Create multi-spectral cube
 * Image spectral(512, 512, 16);  // 16 bands
 * spectral.channelNames[0] = "VIS_550nm";
 * spectral.channelNames[1] = "NIR_850nm";
 * spectral.metadata["spp"] = "128";
 *
 * // Efficient scanline iteration
 * for (u32 y = 0; y < img.height; ++y) {
 *     f32* scanline = img.PixelPtr(0, y);
 *     for (u32 x = 0; x < img.width; ++x) {
 *         f32 r = scanline[x * img.channels + 0];
 *         // Process pixel...
 *     }
 * }
 * @endcode
 *
 * @note Memory layout: data[y][x][c] (row-major, channel-last)
 * @note Always uses f32 storage (converts u8/f16 to f32 during load)
 * @see ImageIO for EXR/PNG read/write operations
 */
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
    // e.g., {"spp": "64", "mode": "rgb", "seconds_per_frame": "2.3"}
    std::unordered_map<std::string, std::string> metadata;

    // ========================================================================
    // Constructors
    // ========================================================================

    Image() = default;

    Image(const u32 w, const u32 h, const u32 c)
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
    inline f32& operator()(const u32 x, const u32 y, const u32 c) {
        return data[y * width * channels + x * channels + c];
    }

    inline const f32& operator()(const u32 x, const u32 y, const u32 c) const {
        return data[y * width * channels + x * channels + c];
    }

    // Get pointer to pixel (x, y) - useful for bulk operations
    inline f32* PixelPtr(const u32 x, const u32 y) {
        return &data[y * width * channels + x * channels];
    }

    [[nodiscard]] inline const f32* PixelPtr(const u32 x, const u32 y) const {
        return &data[y * width * channels + x * channels];
    }

    // Index of a named channel, or `fallback` if this image does not name one.
    //
    // Interleaved order is NOT the order a writer used. ImageIO::WriteEXR hands
    // its channels to an Imf::ChannelList, which is a name-sorted map, and
    // ImageIO::ReadEXR walks that list back -- so "R","G","B" returns as
    // "B","G","R" and "R","G","B","A" as "A","B","G","R". The per-channel data
    // follows its own name, so nothing is corrupted; only the index moves.
    // Reading index 0 of an EXR that came off disk therefore gets the
    // alphabetically first channel, which for a render output is the alpha.
    //
    // Anything that reads an image it did not itself allocate must ask by name.
    [[nodiscard]] inline u32 ChannelIndex(const std::string_view name,
                                          const u32 fallback = 0) const {
        for (u32 i = 0; i < channelNames.size() && i < channels; ++i) {
            if (channelNames[i] == name) return i;
        }
        return fallback;
    }

    // Index of the channel to read when a monochrome quantity is wanted from an
    // image that may be grey, RGB or RGBA: the red channel by preference, then
    // any of the usual single-channel spellings, and otherwise the first
    // channel that is not the alpha.
    [[nodiscard]] inline u32 LuminanceChannelIndex() const {
        for (const std::string_view name : {"R", "Y", "V", "Gray", "Grey"}) {
            for (u32 i = 0; i < channelNames.size() && i < channels; ++i) {
                if (channelNames[i] == name) return i;
            }
        }
        for (u32 i = 0; i < channelNames.size() && i < channels; ++i) {
            if (channelNames[i] != "A" && channelNames[i] != "Alpha") return i;
        }
        return 0;
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    // Total number of pixels
    [[nodiscard]] inline u32 PixelCount() const { return width * height; }

    // Total number of elements (pixels * channels)
    [[nodiscard]] inline u32 TotalElements() const { return width * height * channels; }

    // Check if image is valid
    [[nodiscard]] inline bool IsValid() const {
        return width > 0 && height > 0 && channels > 0 &&
               data.size() == TotalElements();
    }

    // Clear image data (set all to zero)
    void Clear() { std::fill(data.begin(), data.end(), 0.0f); }

    // Resize image (will clear existing data)
    void Resize(const u32 w, const u32 h, const u32 c) {
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
