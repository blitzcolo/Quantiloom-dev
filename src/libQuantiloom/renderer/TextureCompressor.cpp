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
#include <future>
#include <thread>
#include <chrono>

// ============================================================================
// BC7 Compression with bc7enc_rdo
// ============================================================================

#if QUANTILOOM_USE_BC7ENC

// bc7enc_rdo source files (bc7enc.cpp, rgbcx.cpp) are compiled separately via CMake
// Only include headers here
#include "rgbcx.h"
#include "bc7enc.h"

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
        // Use perceptual weights (already set by _init) and max uber level
        params.m_uber_level = BC7ENC_MAX_UBER_LEVEL;  // 4
        params.m_max_partitions = BC7ENC_MAX_PARTITIONS;
    } else {
        // Fast mode for real-time loading
        bc7enc_compress_block_params_init_linear_weights(&params);
        params.m_uber_level = 0;
        params.m_max_partitions = 16;  // Fewer partitions for faster encoding
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

void TextureCompressor::ParallelCompressTextures(
    std::vector<Texture>& textures,
    bool highQuality,
    unsigned int maxThreads)
{
    if (!IsAvailable()) {
        QL_LOG_WARN("TextureCompressor: BC7 not available, skipping parallel compression");
        return;
    }

    // Count compressible textures
    std::vector<size_t> compressibleIndices;
    compressibleIndices.reserve(textures.size());
    for (size_t i = 0; i < textures.size(); ++i) {
        if (CanCompress(textures[i])) {
            compressibleIndices.push_back(i);
        }
    }

    if (compressibleIndices.empty()) {
        QL_LOG_INFO("TextureCompressor: No textures eligible for BC7 compression");
        return;
    }

    // Determine thread count
    unsigned int numThreads = maxThreads;
    if (numThreads == 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    numThreads = std::min(numThreads, static_cast<unsigned int>(compressibleIndices.size()));
    if (numThreads == 0) numThreads = 4;
    numThreads = std::min(numThreads, 8u);  // Cap at 8 threads

    QL_LOG_INFO("TextureCompressor: Parallel BC7 compression of {} textures using {} threads (quality: {})",
                compressibleIndices.size(), numThreads, highQuality ? "high" : "fast");

    auto startTime = std::chrono::steady_clock::now();

    // Launch async compression tasks
    std::vector<std::future<std::pair<size_t, std::optional<BC7CompressedData>>>> futures;
    futures.reserve(compressibleIndices.size());

    for (size_t idx : compressibleIndices) {
        const Texture& tex = textures[idx];
        futures.push_back(std::async(std::launch::async, [&tex, idx, highQuality]() {
            auto result = CompressBC7(tex, highQuality);
            return std::make_pair(idx, std::move(result));
        }));
    }

    // Collect results and store in texture.bc7Data
    size_t successCount = 0;
    size_t totalCompressedBytes = 0;
    size_t totalUncompressedBytes = 0;

    for (auto& f : futures) {
        auto [idx, compressedOpt] = f.get();
        if (compressedOpt.has_value()) {
            const auto& compressed = compressedOpt.value();
            Texture& tex = textures[idx];

            // Convert BC7CompressedData to PrecompressedBC7
            PrecompressedBC7 precompressed;
            precompressed.data = std::move(compressedOpt->data);
            precompressed.blockCountX = compressed.blockCountX;
            precompressed.blockCountY = compressed.blockCountY;

            tex.bc7Data = std::move(precompressed);
            successCount++;

            totalCompressedBytes += compressed.GetCompressedSize();
            totalUncompressedBytes += static_cast<size_t>(tex.width) * tex.height * 4;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    float ratio = totalUncompressedBytes > 0
        ? static_cast<float>(totalUncompressedBytes) / static_cast<float>(totalCompressedBytes)
        : 0.0f;

    QL_LOG_INFO("TextureCompressor: BC7 compression complete: {}/{} textures, {:.1f}MB -> {:.1f}MB ({:.1f}x), took {}ms",
                successCount, compressibleIndices.size(),
                totalUncompressedBytes / (1024.0 * 1024.0),
                totalCompressedBytes / (1024.0 * 1024.0),
                ratio, elapsed.count());
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

void TextureCompressor::ParallelCompressTextures(
    std::vector<Texture>& /* textures */,
    bool /* highQuality */,
    unsigned int /* maxThreads */)
{
    QL_LOG_WARN("TextureCompressor: BC7 compression not available. "
                "Build with -DQUANTILOOM_USE_BC7ENC=ON and add bc7enc_rdo to vendor/");
}

} // namespace quantiloom

#endif // QUANTILOOM_USE_BC7ENC
