/**
 * @file TextureCompressor.hpp
 * @brief GPU texture compression (BC7) for VRAM optimization
 *
 * Provides runtime BC7 compression to reduce VRAM usage by 4-8x:
 * - 2048x2048 RGBA8: 16MB -> 4MB (BC7)
 * - Total savings for 522 textures: 8.4GB -> 2.1GB
 *
 * BC7 format advantages:
 * - High quality (visually lossless for most content)
 * - Hardware decompression on all modern GPUs
 * - Supported by Vulkan via VK_FORMAT_BC7_*_BLOCK
 *
 * Compression time trade-off:
 * - ~100-200ms per 2048x2048 texture (single-threaded)
 * - Can be parallelized with texture loading (Phase 3)
 *
 * Requirements:
 * - QUANTILOOM_USE_BC7ENC=1 compile flag
 * - bc7enc_rdo library (MIT license, header-only)
 *   https://github.com/richgel999/bc7enc_rdo
 *
 * Usage:
 * @code
 * if (TextureCompressor::IsAvailable()) {
 *     auto compressed = TextureCompressor::CompressBC7(texture);
 *     // Upload compressed data with VK_FORMAT_BC7_SRGB_BLOCK
 * }
 * @endcode
 *
 * @note BC7 requires 4x4 block alignment (width/height must be multiple of 4)
 * @note Textures smaller than 4x4 cannot be compressed
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "scene/Texture.hpp"
#include <vector>
#include <optional>

namespace quantiloom {

/**
 * @struct BC7CompressedData
 * @brief Container for BC7 compressed texture data
 */
struct QL_API BC7CompressedData {
    std::vector<u8> data;       ///< Compressed block data
    u32 width;                  ///< Original width (for mipmap generation)
    u32 height;                 ///< Original height
    u32 blockCountX;            ///< Number of 4x4 blocks in X
    u32 blockCountY;            ///< Number of 4x4 blocks in Y
    bool isSRGB;                ///< Use VK_FORMAT_BC7_SRGB_BLOCK vs UNORM

    /// Calculate compressed data size in bytes
    [[nodiscard]] size_t GetCompressedSize() const {
        // BC7: 16 bytes per 4x4 block
        return static_cast<size_t>(blockCountX) * blockCountY * 16;
    }

    /// Calculate compression ratio (uncompressed / compressed)
    [[nodiscard]] float GetCompressionRatio() const {
        size_t uncompressedSize = static_cast<size_t>(width) * height * 4;
        return static_cast<float>(uncompressedSize) / static_cast<float>(GetCompressedSize());
    }
};

/**
 * @class TextureCompressor
 * @brief Static utility class for BC7 texture compression
 *
 * Provides runtime compression of RGBA8 textures to BC7 format.
 * BC7 offers near-lossless quality with 4:1 compression ratio.
 *
 * Example workflow:
 * @code
 * // Check if compression is available
 * if (!TextureCompressor::IsAvailable()) {
 *     QL_LOG_WARN("BC7 compression not available, using uncompressed");
 *     return;
 * }
 *
 * // Check if texture can be compressed
 * if (!TextureCompressor::CanCompress(texture)) {
 *     QL_LOG_INFO("Texture too small for BC7, using uncompressed");
 *     return;
 * }
 *
 * // Compress
 * auto result = TextureCompressor::CompressBC7(texture);
 * if (result.has_value()) {
 *     // Use compressed data for GPU upload
 *     VkFormat format = result->isSRGB
 *         ? VK_FORMAT_BC7_SRGB_BLOCK
 *         : VK_FORMAT_BC7_UNORM_BLOCK;
 *     // ... create VkImage with compressed format ...
 * }
 * @endcode
 */
class QL_API TextureCompressor {
public:
    // ========================================================================
    // Availability Check
    // ========================================================================

    /**
     * @brief Check if BC7 compression is available
     * @return true if bc7enc library is linked, false otherwise
     *
     * Returns false if QUANTILOOM_USE_BC7ENC is not defined or bc7enc
     * failed to initialize. Always call this before attempting compression.
     */
    [[nodiscard]] static bool IsAvailable();

    /**
     * @brief Check if a texture can be BC7 compressed
     * @param texture The texture to check
     * @return true if texture meets BC7 requirements
     *
     * Requirements:
     * - Width and height >= 4 (BC7 block size)
     * - Width and height are multiples of 4 (or will be padded)
     * - 4 channels (RGBA)
     * - Non-empty pixel data
     */
    [[nodiscard]] static bool CanCompress(const Texture& texture);

    // ========================================================================
    // Compression
    // ========================================================================

    /**
     * @brief Compress RGBA8 texture to BC7 format
     * @param texture Source texture (must be RGBA8, 4 channels)
     * @param highQuality Use high quality mode (slower, better quality)
     * @return Compressed data or nullopt on failure
     *
     * Performance (2048x2048 texture):
     * - highQuality=false: ~50-100ms
     * - highQuality=true: ~150-300ms
     *
     * Quality:
     * - BC7 is visually lossless for most photographic content
     * - High quality mode improves gradients and fine detail
     */
    [[nodiscard]] static std::optional<BC7CompressedData> CompressBC7(
        const Texture& texture,
        bool highQuality = false);

    // ========================================================================
    // Utility
    // ========================================================================

    /**
     * @brief Get BC7 compressed size for given dimensions
     * @param width Texture width
     * @param height Texture height
     * @return Size in bytes (16 bytes per 4x4 block)
     */
    [[nodiscard]] static size_t GetBC7CompressedSize(u32 width, u32 height);

    /**
     * @brief Pad dimensions to BC7 block alignment (multiple of 4)
     * @param width Input width
     * @param height Input height
     * @param outWidth Aligned width (>= width, multiple of 4)
     * @param outHeight Aligned height (>= height, multiple of 4)
     */
    static void GetAlignedDimensions(u32 width, u32 height, u32& outWidth, u32& outHeight);

private:
    // Prevent instantiation
    TextureCompressor() = delete;
};

} // namespace quantiloom
