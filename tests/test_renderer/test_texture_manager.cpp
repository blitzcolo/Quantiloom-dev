/**
 * @file test_texture_manager.cpp
 * @brief Unit tests for TextureManager optimization features
 *
 * Tests texture deduplication, CPU memory release, and BC7 compression
 * infrastructure added as part of the texture optimization effort.
 *
 * Optimization goals:
 * - Reduce CPU RAM from 8.4GB to ~500MB (after GPU upload)
 * - Reduce VRAM from 8.4GB to ~1GB (with BC7)
 * - Reduce load time from 56s to ~15-20s (parallel loading)
 *
 * @author wtflmao
 */

#include <gtest/gtest.h>
#include "scene/Texture.hpp"
#include "renderer/TextureCompressor.hpp"
#include "core/Types.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>

using namespace quantiloom;

// ============================================================================
// Test Fixtures and Utilities
// ============================================================================

namespace {

/**
 * @brief Create a test texture with specified dimensions
 */
Texture CreateTestTexture(u32 width, u32 height, const std::string& name = "TestTexture") {
    Texture tex;
    tex.name = name;
    tex.width = width;
    tex.height = height;
    tex.channels = 4;
    tex.isSRGB = true;

    // Fill with gradient pattern for compression quality testing
    tex.pixels.resize(static_cast<size_t>(width) * height * 4);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            tex.pixels[idx + 0] = static_cast<u8>((x * 255) / width);      // R
            tex.pixels[idx + 1] = static_cast<u8>((y * 255) / height);     // G
            tex.pixels[idx + 2] = static_cast<u8>(((x + y) * 127) / (width + height)); // B
            tex.pixels[idx + 3] = 255;  // A
        }
    }

    return tex;
}

/**
 * @brief Simple texture cache for deduplication testing
 * (Mirrors the implementation in UsdLoader.cpp)
 */
struct TextureCache {
    std::unordered_map<std::string, size_t> pathToIndex;
    std::mutex mutex;

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex);
        pathToIndex.clear();
    }

    std::pair<bool, size_t> Find(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = pathToIndex.find(path);
        if (it != pathToIndex.end()) {
            return {true, it->second};
        }
        return {false, 0};
    }

    void Insert(const std::string& path, size_t index) {
        std::lock_guard<std::mutex> lock(mutex);
        pathToIndex[path] = index;
    }
};

} // anonymous namespace

// ============================================================================
// Texture Deduplication Tests
// ============================================================================

/**
 * @test TextureCache properly deduplicates textures by path
 */
TEST(TextureDeduplicationTest, CacheHitForSamePath) {
    TextureCache cache;

    // Insert first texture
    cache.Insert("/path/to/texture1.png", 0);
    cache.Insert("/path/to/texture2.png", 1);

    // Should find existing texture
    auto [found1, idx1] = cache.Find("/path/to/texture1.png");
    EXPECT_TRUE(found1);
    EXPECT_EQ(idx1, 0);

    auto [found2, idx2] = cache.Find("/path/to/texture2.png");
    EXPECT_TRUE(found2);
    EXPECT_EQ(idx2, 1);

    // Should not find non-existent texture
    auto [found3, idx3] = cache.Find("/path/to/texture3.png");
    EXPECT_FALSE(found3);
}

/**
 * @test TextureCache handles case-sensitive paths correctly
 */
TEST(TextureDeduplicationTest, CaseSensitivePaths) {
    TextureCache cache;

    cache.Insert("/path/to/Texture.png", 0);

    // Different case should be different entry
    auto [found, idx] = cache.Find("/path/to/texture.png");
    EXPECT_FALSE(found);  // Case-sensitive on most systems
}

/**
 * @test TextureCache clears properly between loads
 */
TEST(TextureDeduplicationTest, ClearBetweenLoads) {
    TextureCache cache;

    cache.Insert("/texture1.png", 0);
    EXPECT_TRUE(cache.Find("/texture1.png").first);

    cache.Clear();

    // After clear, texture should not be found
    EXPECT_FALSE(cache.Find("/texture1.png").first);
}

/**
 * @test TextureCache is thread-safe for concurrent access
 */
TEST(TextureDeduplicationTest, ThreadSafety) {
    TextureCache cache;
    constexpr int numThreads = 8;
    constexpr int insertionsPerThread = 100;

    std::vector<std::future<void>> futures;

    // Concurrent insertions
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [&cache, t]() {
            for (int i = 0; i < insertionsPerThread; ++i) {
                std::string path = "/texture_" + std::to_string(t) + "_" + std::to_string(i) + ".png";
                cache.Insert(path, static_cast<size_t>(t * insertionsPerThread + i));
            }
        }));
    }

    // Wait for all threads
    for (auto& f : futures) {
        f.get();
    }

    // Verify all insertions
    int foundCount = 0;
    for (int t = 0; t < numThreads; ++t) {
        for (int i = 0; i < insertionsPerThread; ++i) {
            std::string path = "/texture_" + std::to_string(t) + "_" + std::to_string(i) + ".png";
            if (cache.Find(path).first) {
                foundCount++;
            }
        }
    }

    EXPECT_EQ(foundCount, numThreads * insertionsPerThread);
}

// ============================================================================
// CPU Memory Release Tests
// ============================================================================

/**
 * @test Texture pixels can be cleared and memory released
 */
TEST(CPUMemoryReleaseTest, ClearPixelsAfterUpload) {
    Texture tex = CreateTestTexture(256, 256);

    // Verify initial state
    EXPECT_FALSE(tex.pixels.empty());
    EXPECT_EQ(tex.pixels.size(), 256 * 256 * 4);

    // Simulate GPU upload complete - release CPU memory
    tex.pixels.clear();
    tex.pixels.shrink_to_fit();

    // Verify memory released
    EXPECT_TRUE(tex.pixels.empty());
    EXPECT_EQ(tex.pixels.capacity(), 0);  // shrink_to_fit should release allocation
}

/**
 * @test Large texture memory footprint calculation
 */
TEST(CPUMemoryReleaseTest, MemoryFootprintCalculation) {
    // Simulate OldAttic scene: 522 textures @ 2048x2048 RGBA
    constexpr u32 textureCount = 522;
    constexpr u32 texWidth = 2048;
    constexpr u32 texHeight = 2048;
    constexpr u32 channels = 4;

    size_t perTextureBytes = static_cast<size_t>(texWidth) * texHeight * channels;
    size_t totalBytes = perTextureBytes * textureCount;

    // Verify expected footprint
    EXPECT_EQ(perTextureBytes, 16 * 1024 * 1024);  // 16 MB per texture
    EXPECT_GT(totalBytes, 8ULL * 1024 * 1024 * 1024);  // > 8 GB total
}

// ============================================================================
// Parallel Loading Tests
// ============================================================================

/**
 * @test Parallel texture path collection with set deduplication
 */
TEST(ParallelLoadingTest, UniquePathCollection) {
    std::unordered_set<std::string> uniquePaths;

    // Simulate collecting paths from multiple materials
    // Some paths may be duplicated across materials
    std::vector<std::string> materialPaths = {
        "/textures/wood_basecolor.png",
        "/textures/wood_normal.png",
        "/textures/metal_basecolor.png",
        "/textures/wood_basecolor.png",  // Duplicate
        "/textures/glass_basecolor.png",
        "/textures/metal_basecolor.png", // Duplicate
    };

    for (const auto& path : materialPaths) {
        uniquePaths.insert(path);
    }

    // Should have only unique paths
    EXPECT_EQ(uniquePaths.size(), 4);
    EXPECT_TRUE(uniquePaths.count("/textures/wood_basecolor.png") > 0);
    EXPECT_TRUE(uniquePaths.count("/textures/wood_normal.png") > 0);
    EXPECT_TRUE(uniquePaths.count("/textures/metal_basecolor.png") > 0);
    EXPECT_TRUE(uniquePaths.count("/textures/glass_basecolor.png") > 0);
}

/**
 * @test Parallel future collection pattern
 */
TEST(ParallelLoadingTest, AsyncFutureCollection) {
    constexpr int numTasks = 16;
    std::vector<std::future<int>> futures;

    auto startTime = std::chrono::steady_clock::now();

    // Launch parallel tasks
    for (int i = 0; i < numTasks; ++i) {
        futures.push_back(std::async(std::launch::async, [i]() {
            // Simulate work (50ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i * 2;
        }));
    }

    // Collect results
    int sum = 0;
    for (auto& f : futures) {
        sum += f.get();
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Verify results
    EXPECT_EQ(sum, numTasks * (numTasks - 1));  // Sum of 0, 2, 4, ..., 2*(n-1)

    // Should complete faster than sequential (16 * 50ms = 800ms)
    // With parallelism, should be around 50-100ms + overhead
    EXPECT_LT(elapsed.count(), 500);  // Should be much less than 800ms
}

/**
 * @test Thread count determination
 */
TEST(ParallelLoadingTest, ThreadCountDetermination) {
    unsigned int hwThreads = std::thread::hardware_concurrency();
    unsigned int numThreads = std::min(8u, hwThreads);
    if (numThreads == 0) numThreads = 4;

    // Should be between 1 and 8
    EXPECT_GE(numThreads, 1);
    EXPECT_LE(numThreads, 8);
}

// ============================================================================
// BC7 Compression Tests
// ============================================================================

/**
 * @test BC7 availability check
 */
TEST(BC7CompressionTest, AvailabilityCheck) {
    // Just verify the function exists and returns consistently
    bool available1 = TextureCompressor::IsAvailable();
    bool available2 = TextureCompressor::IsAvailable();
    EXPECT_EQ(available1, available2);

    // Note: Actual availability depends on QUANTILOOM_USE_BC7ENC compile flag
    // This test just verifies the API exists
}

/**
 * @test BC7 compressed size calculation
 */
TEST(BC7CompressionTest, CompressedSizeCalculation) {
    // BC7: 16 bytes per 4x4 block
    // For 2048x2048: (2048/4) * (2048/4) * 16 = 512 * 512 * 16 = 4MB

    size_t compressedSize = TextureCompressor::GetBC7CompressedSize(2048, 2048);
    size_t expectedSize = (2048 / 4) * (2048 / 4) * 16;

    EXPECT_EQ(compressedSize, expectedSize);
    EXPECT_EQ(compressedSize, 4 * 1024 * 1024);  // 4 MB
}

/**
 * @test BC7 block alignment calculation
 */
TEST(BC7CompressionTest, BlockAlignment) {
    u32 outWidth, outHeight;

    // Already aligned
    TextureCompressor::GetAlignedDimensions(256, 256, outWidth, outHeight);
    EXPECT_EQ(outWidth, 256);
    EXPECT_EQ(outHeight, 256);

    // Needs padding
    TextureCompressor::GetAlignedDimensions(255, 255, outWidth, outHeight);
    EXPECT_EQ(outWidth, 256);
    EXPECT_EQ(outHeight, 256);

    // Odd dimensions
    TextureCompressor::GetAlignedDimensions(1, 1, outWidth, outHeight);
    EXPECT_EQ(outWidth, 4);
    EXPECT_EQ(outHeight, 4);

    // One dimension aligned, one not
    TextureCompressor::GetAlignedDimensions(512, 513, outWidth, outHeight);
    EXPECT_EQ(outWidth, 512);
    EXPECT_EQ(outHeight, 516);
}

/**
 * @test BC7 compression ratio calculation
 */
TEST(BC7CompressionTest, CompressionRatio) {
    // For 2048x2048 RGBA8:
    // Uncompressed: 2048 * 2048 * 4 = 16 MB
    // BC7: 4 MB
    // Ratio: 4.0x

    size_t uncompressed = 2048 * 2048 * 4;
    size_t compressed = TextureCompressor::GetBC7CompressedSize(2048, 2048);

    float ratio = static_cast<float>(uncompressed) / static_cast<float>(compressed);
    EXPECT_NEAR(ratio, 4.0f, 0.01f);
}

/**
 * @test BC7 CanCompress validation
 */
TEST(BC7CompressionTest, CanCompressValidation) {
    // Valid texture
    Texture validTex = CreateTestTexture(256, 256);
    bool canCompress = TextureCompressor::CanCompress(validTex);
    // Note: Result depends on IsAvailable(), but should not crash

    // Texture too small
    Texture smallTex = CreateTestTexture(2, 2);
    EXPECT_FALSE(TextureCompressor::CanCompress(smallTex));

    // Empty texture
    Texture emptyTex;
    emptyTex.width = 256;
    emptyTex.height = 256;
    emptyTex.channels = 4;
    emptyTex.pixels.clear();
    EXPECT_FALSE(TextureCompressor::CanCompress(emptyTex));

    // Wrong channel count
    Texture wrongChannels = CreateTestTexture(256, 256);
    wrongChannels.channels = 3;  // Not RGBA
    EXPECT_FALSE(TextureCompressor::CanCompress(wrongChannels));
}

/**
 * @test BC7 VRAM savings calculation for realistic scene
 */
TEST(BC7CompressionTest, VRAMSavingsCalculation) {
    // OldAttic scene: 522 textures @ 2048x2048
    constexpr u32 textureCount = 522;
    constexpr u32 texWidth = 2048;
    constexpr u32 texHeight = 2048;

    size_t perTextureUncompressed = static_cast<size_t>(texWidth) * texHeight * 4;
    size_t totalUncompressed = perTextureUncompressed * textureCount;

    size_t perTextureCompressed = TextureCompressor::GetBC7CompressedSize(texWidth, texHeight);
    size_t totalCompressed = perTextureCompressed * textureCount;

    // Verify savings
    size_t savedBytes = totalUncompressed - totalCompressed;
    float savingsPercent = (static_cast<float>(savedBytes) / totalUncompressed) * 100.0f;

    EXPECT_GT(savedBytes, 6ULL * 1024 * 1024 * 1024);  // Should save > 6 GB
    EXPECT_GT(savingsPercent, 70.0f);  // Should save > 70%
}

// ============================================================================
// Integration Tests
// ============================================================================

/**
 * @test Complete texture optimization workflow (without GPU)
 */
TEST(TextureOptimizationWorkflowTest, FullWorkflow) {
    // Step 1: Simulate loading textures with deduplication
    TextureCache cache;
    std::vector<Texture> loadedTextures;
    std::vector<std::string> texturePaths = {
        "/scene/wood.png",
        "/scene/metal.png",
        "/scene/wood.png",  // Duplicate
        "/scene/glass.png",
    };

    for (const auto& path : texturePaths) {
        auto [found, idx] = cache.Find(path);
        if (!found) {
            // Load texture (simulated)
            Texture tex = CreateTestTexture(64, 64, path);
            size_t newIdx = loadedTextures.size();
            loadedTextures.push_back(std::move(tex));
            cache.Insert(path, newIdx);
        }
    }

    // Should have only 3 unique textures
    EXPECT_EQ(loadedTextures.size(), 3);

    // Step 2: Simulate GPU upload with memory release
    for (auto& tex : loadedTextures) {
        // Simulate upload...
        // Then release CPU memory
        tex.pixels.clear();
        tex.pixels.shrink_to_fit();
    }

    // All textures should have released CPU memory
    for (const auto& tex : loadedTextures) {
        EXPECT_TRUE(tex.pixels.empty());
        EXPECT_EQ(tex.pixels.capacity(), 0);
    }
}

/**
 * @test BC7CompressedData struct functionality
 */
TEST(BC7CompressedDataTest, StructFunctionality) {
    BC7CompressedData data;
    data.width = 256;
    data.height = 256;
    data.blockCountX = 64;
    data.blockCountY = 64;
    data.isSRGB = true;
    data.data.resize(64 * 64 * 16);  // 16 bytes per block

    EXPECT_EQ(data.GetCompressedSize(), 64 * 64 * 16);

    // Compression ratio: (256*256*4) / (64*64*16) = 262144 / 65536 = 4.0
    EXPECT_NEAR(data.GetCompressionRatio(), 4.0f, 0.01f);
}
