/**
 * @file TextureCompressor.cpp
 * @brief BC7 texture compression implementation
 *
 * Conditionally compiles with bc7enc_rdo when QUANTILOOM_USE_BC7ENC=1.
 * Falls back to stub implementation when bc7enc is not available.
 *
 * To enable BC7 compression:
 * 1. Download bc7enc_rdo: https://github.com/richgel999/bc7enc_rdo
 * 2. Add to vendor/bc7enc_rdo/
 * 3. Build with -DQUANTILOOM_USE_BC7ENC=ON
 *
 * @author wtflmao
 */

#include "TextureCompressor.hpp"
#include "core/Log.hpp"
#include <cstring>
#include <algorithm>

// ============================================================================
// BC7 Compression with bc7enc_rdo
// ============================================================================

#if QUANTILOOM_USE_BC7ENC

// bc7enc_rdo is a single-header library
// Implementation is defined once here
#define BC7ENC_RDO_IMPLEMENTATION
#include <bc7enc_rdo/bc7enc.h>
#include <bc7enc_rdo/rgbcx.h>

namespace quantiloom {

// Initialize bc7enc once
static bool g_bc7encInitialized = false;

static void EnsureBC7EncInitialized() {
    if (!g_bc7encInitialized) {
        bc7enc_compress_block_init();
        rgbcx::init();
        g_bc7encInitialized = true;
        QL_LOG_INFO("BC7 encoder initialized");
    }
}

bool TextureCompressor::IsAvailable() {
    return true;
}

bool TextureCompressor::CanCompress(const Texture& texture) {
    // BC7 requires 4x4 blocks
    if (texture.width < 4 || texture.height < 4) {
        return false;
    }

    // Must be RGBA (4 channels)
    if (texture.channels != 4) {
        return false;
    }

    // Must have pixel data
    if (texture.pixels.empty()) {
        return false;
    }

    // Verify pixel data size
    size_t expectedSize = static_cast<size_t>(texture.width) * texture.height * 4;
    if (texture.pixels.size() != expectedSize) {
        return false;
    }

    return true;
}

std::optional<BC7CompressedData> TextureCompressor::CompressBC7(
    const Texture& texture,
    bool highQuality)
{
    EnsureBC7EncInitialized();

    if (!CanCompress(texture)) {
        QL_LOG_ERROR("TextureCompressor: Texture '{}' cannot be BC7 compressed", texture.name);
        return std::nullopt;
    }

    // Calculate aligned dimensions (multiple of 4)
    u32 alignedWidth, alignedHeight;
    GetAlignedDimensions(texture.width, texture.height, alignedWidth, alignedHeight);

    u32 blockCountX = alignedWidth / 4;
    u32 blockCountY = alignedHeight / 4;
    size_t totalBlocks = static_cast<size_t>(blockCountX) * blockCountY;

    // Prepare result
    BC7CompressedData result;
    result.width = texture.width;
    result.height = texture.height;
    result.blockCountX = blockCountX;
    result.blockCountY = blockCountY;
    result.isSRGB = texture.isSRGB;
    result.data.resize(totalBlocks * 16);  // 16 bytes per BC7 block

    // Create padded source if needed
    std::vector<u8> paddedPixels;
    const u8* srcPixels = texture.pixels.data();
    u32 srcPitch = texture.width * 4;

    if (alignedWidth != texture.width || alignedHeight != texture.height) {
        // Need to pad texture to block alignment
        paddedPixels.resize(static_cast<size_t>(alignedWidth) * alignedHeight * 4, 0);

        for (u32 y = 0; y < texture.height; ++y) {
            std::memcpy(
                paddedPixels.data() + y * alignedWidth * 4,
                texture.pixels.data() + y * texture.width * 4,
                texture.width * 4
            );
            // Edge extend for remaining columns
            if (alignedWidth > texture.width) {
                u8* rowStart = paddedPixels.data() + y * alignedWidth * 4;
                const u8* lastPixel = rowStart + (texture.width - 1) * 4;
                for (u32 x = texture.width; x < alignedWidth; ++x) {
                    std::memcpy(rowStart + x * 4, lastPixel, 4);
                }
            }
        }
        // Edge extend for remaining rows
        if (alignedHeight > texture.height) {
            const u8* lastRow = paddedPixels.data() + (texture.height - 1) * alignedWidth * 4;
            for (u32 y = texture.height; y < alignedHeight; ++y) {
                std::memcpy(paddedPixels.data() + y * alignedWidth * 4, lastRow, alignedWidth * 4);
            }
        }

        srcPixels = paddedPixels.data();
        srcPitch = alignedWidth * 4;
    }

    // Configure encoder
    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);

    if (highQuality) {
        // Higher quality, slower compression
        bc7enc_compress_block_params_init_uber_level(&params, 4);
    } else {
        // Fast mode for real-time loading
        bc7enc_compress_block_params_init_linear_weights(&params);
        params.m_uber_level = 0;
    }

    // Compress each 4x4 block
    u8* dstBlock = result.data.data();

    for (u32 by = 0; by < blockCountY; ++by) {
        for (u32 bx = 0; bx < blockCountX; ++bx) {
            // Extract 4x4 block
            u8 block[64];  // 16 pixels * 4 channels
            for (u32 py = 0; py < 4; ++py) {
                const u8* srcRow = srcPixels + (by * 4 + py) * srcPitch + bx * 16;
                std::memcpy(block + py * 16, srcRow, 16);
            }

            // Compress block
            bc7enc_compress_block(dstBlock, block, &params);
            dstBlock += 16;
        }
    }

    QL_LOG_INFO("TextureCompressor: Compressed '{}' {}x{} -> {} bytes (ratio: {:.1f}x)",
                texture.name, texture.width, texture.height,
                result.GetCompressedSize(), result.GetCompressionRatio());

    return result;
}

size_t TextureCompressor::GetBC7CompressedSize(u32 width, u32 height) {
    u32 alignedWidth, alignedHeight;
    GetAlignedDimensions(width, height, alignedWidth, alignedHeight);
    u32 blockCountX = alignedWidth / 4;
    u32 blockCountY = alignedHeight / 4;
    return static_cast<size_t>(blockCountX) * blockCountY * 16;
}

void TextureCompressor::GetAlignedDimensions(u32 width, u32 height, u32& outWidth, u32& outHeight) {
    // Round up to multiple of 4
    outWidth = (width + 3) & ~3u;
    outHeight = (height + 3) & ~3u;
}

} // namespace quantiloom

#else // QUANTILOOM_USE_BC7ENC not defined

// ============================================================================
// Stub Implementation (BC7 not available)
// ============================================================================

namespace quantiloom {

bool TextureCompressor::IsAvailable() {
    return false;
}

bool TextureCompressor::CanCompress(const Texture& /* texture */) {
    return false;
}

std::optional<BC7CompressedData> TextureCompressor::CompressBC7(
    const Texture& /* texture */,
    bool /* highQuality */)
{
    QL_LOG_WARN("TextureCompressor: BC7 compression not available. "
                "Build with -DQUANTILOOM_USE_BC7ENC=ON and add bc7enc_rdo to vendor/");
    return std::nullopt;
}

size_t TextureCompressor::GetBC7CompressedSize(u32 width, u32 height) {
    u32 alignedWidth = (width + 3) & ~3u;
    u32 alignedHeight = (height + 3) & ~3u;
    u32 blockCountX = alignedWidth / 4;
    u32 blockCountY = alignedHeight / 4;
    return static_cast<size_t>(blockCountX) * blockCountY * 16;
}

void TextureCompressor::GetAlignedDimensions(u32 width, u32 height, u32& outWidth, u32& outHeight) {
    outWidth = (width + 3) & ~3u;
    outHeight = (height + 3) & ~3u;
}

} // namespace quantiloom

#endif // QUANTILOOM_USE_BC7ENC
