/**
 * @file ImageIO.hpp
 * @brief EXR and PNG image file I/O for spectral rendering output
 *
 * Provides ImageIO class with static functions for:
 * - EXR writing: Multi-channel HDR output with metadata (OpenEXR 3.x)
 * - PNG writing: LDR 8-bit sRGB preview output (stb_image_write)
 * - EXR reading: Load HDR images with channel names and metadata
 * - File utilities: Existence checking, dimension queries
 *
 * Supported formats:
 * - EXR: Arbitrary channel count, f32 precision, metadata preservation
 * - PNG: 1 (grayscale), 3 (RGB), 4 (RGBA) channels, 8-bit per channel
 *
 * Format conversions:
 * - EXR: f32 CPU data → HALF (f16) on disk (OpenEXR default)
 * - PNG: f32 HDR → u8 LDR with sRGB gamma encoding
 *
 * Channel naming (EXR only):
 * - Image::channelNames preserved in EXR layer names
 * - Example: ["VIS_550nm", "NIR_850nm", "SWIR_1650nm"]
 *
 * Metadata storage (EXR only):
 * - Image::metadata stored as EXR string attributes
 * - Example: {"spp": "128", "wavelength_nm": "550", "mode": "spectral"}
 *
 * @note All image data stored as f32 in memory (converted from/to disk formats)
 * @note PNG output applies sRGB gamma encoding (not linear RGB)
 * @note EXR supports unlimited channels, PNG limited to 1-4 channels
 *
 * @author wtflmao
 */

#pragma once

#include "core/Image.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>

namespace quantiloom {

// ============================================================================
// ImageIO - EXR image reading/writing using OpenEXR 3.x
// ============================================================================
/**
 * @class ImageIO
 * @brief Static utility class for EXR/PNG image file I/O
 *
 * Handles all image file operations for Quantiloom spectral rendering.
 * Supports multi-channel HDR output (EXR) and LDR preview (PNG).
 *
 * EXR workflow (multi-channel HDR):
 * @code
 * // Create spectral image
 * Image spectral(512, 512, 16);  // 16 spectral bands
 * spectral.channelNames = {"VIS_450", "VIS_500", ..., "NIR_900"};
 * spectral.metadata["spp"] = "128";
 * spectral.metadata["mode"] = "multispectral";
 *
 * // Render and save
 * // ... (render to spectral.data) ...
 * ImageIO::WriteEXR("output.exr", spectral);
 *
 * // Load back
 * auto loaded = ImageIO::ReadEXR("output.exr");
 * if (loaded.has_value()) {
 *     QL_LOG_INFO("Loaded {} channels: {}", loaded->channels, loaded->channelNames[0]);
 * }
 * @endcode
 *
 * PNG workflow (8-bit sRGB preview):
 * @code
 * // Create RGB preview from HDR
 * Image preview(1920, 1080, 3);
 * // ... (tone-map HDR to [0,1] range) ...
 * ImageIO::WritePNG("preview.png", preview);  // Applies sRGB gamma
 * @endcode
 *
 * Fast dimension queries:
 * @code
 * // Check dimensions without loading full image
 * auto dims = ImageIO::GetDimensions("large_file.exr");
 * if (dims.has_value()) {
 *     auto [w, h, c] = *dims;
 *     QL_LOG_INFO("Image: {}x{}, {} channels", w, h, c);
 * }
 * @endcode
 *
 * @note All methods are static (no instance needed)
 * @note WriteEXR/WritePNG return bool (true = success, false = failure)
 * @note ReadEXR returns std::optional<Image> (nullopt on failure)
 * @note PNG writing performs automatic sRGB gamma encoding
 *
 * @see Image for multi-channel image container
 */
class QL_API ImageIO {
public:
    // ========================================================================
    // EXR Writing
    // ========================================================================

    // Write image to EXR file
    // Returns true on success, false on failure
    static bool WriteEXR(const std::string& filepath, const Image& image);

    // Write image to PNG file (8-bit sRGB output)
    // Converts HDR float data to LDR with optional gamma correction
    // Input channels: 1 (grayscale), 3 (RGB), or 4 (RGBA)
    // Returns true on success, false on failure
    static bool WritePNG(const std::string& filepath, const Image& image);

    // ========================================================================
    // EXR Reading
    // ========================================================================

    // Read image from EXR file
    // Returns std::nullopt on failure
    static std::optional<Image> ReadEXR(const std::string& filepath);

    // Read image from any supported format (EXR, PNG, JPEG, BMP, TGA, HDR)
    // Automatically detects format by extension
    // Returns std::nullopt on failure
    static std::optional<Image> ReadImage(const std::string& filepath);

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if file exists and is readable
    static bool FileExists(const std::string& filepath);

    // Get image dimensions without loading full image (fast peek)
    static std::optional<std::tuple<u32, u32, u32>> GetDimensions(const std::string& filepath);
};

} // namespace quantiloom
