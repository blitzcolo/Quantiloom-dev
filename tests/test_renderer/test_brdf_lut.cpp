// ============================================================================
// Quantiloom - Unit Tests for renderer/BRDFLutGenerator.hpp
// ============================================================================
// Tests cover:
// - BRDF LUT generation (basic validation)
// - Binary cache save/load roundtrip
// - Cache validation (magic, version, resolution, sampleCount)
// - Cache invalidation on parameter mismatch
// - Error handling for invalid files
// ============================================================================

#include <gtest/gtest.h>
#include "renderer/BRDFLutGenerator.hpp"
#include "core/Image.hpp"
#include <filesystem>
#include <fstream>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temp Directory
// ============================================================================

class BRDFLutGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        testDir = std::filesystem::temp_directory_path() / "quantiloom_brdf_lut_test";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::filesystem::path GetTestPath(const std::string& filename) {
        return testDir / filename;
    }

    std::filesystem::path testDir;
};

// ============================================================================
// Basic Generation Tests
// ============================================================================

TEST_F(BRDFLutGeneratorTest, GenerateSmallLUT) {
    // Use small resolution for fast testing
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    Image lut = BRDFLutGenerator::Generate(config);

    // Verify dimensions
    EXPECT_EQ(lut.width, 32u);
    EXPECT_EQ(lut.height, 32u);
    EXPECT_EQ(lut.channels, 2u);  // scale and bias

    // Verify channel names
    EXPECT_EQ(lut.channelNames.size(), 2u);
    EXPECT_EQ(lut.channelNames[0], "scale");
    EXPECT_EQ(lut.channelNames[1], "bias");
}

TEST_F(BRDFLutGeneratorTest, GenerateLUTValuesInRange) {
    BRDFLutGenerator::Config config;
    config.resolution = 16;
    config.sampleCount = 32;

    Image lut = BRDFLutGenerator::Generate(config);

    // All scale and bias values should be in [0, 1] range
    for (u32 y = 0; y < lut.height; ++y) {
        for (u32 x = 0; x < lut.width; ++x) {
            f32 scale = lut(x, y, 0);
            f32 bias = lut(x, y, 1);

            EXPECT_GE(scale, 0.0f) << "scale at (" << x << ", " << y << ")";
            EXPECT_LE(scale, 1.0f) << "scale at (" << x << ", " << y << ")";
            EXPECT_GE(bias, 0.0f) << "bias at (" << x << ", " << y << ")";
            EXPECT_LE(bias, 1.0f) << "bias at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(BRDFLutGeneratorTest, GenerateLUTPhysicallyPlausible) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 128;

    Image lut = BRDFLutGenerator::Generate(config);

    // At NdotV = 1.0 (perpendicular view), Fresnel effect is minimal
    // scale should be close to 1.0, bias should be close to 0.0
    // NdotV = 1.0 corresponds to x = resolution - 1
    u32 x_perpendicular = config.resolution - 1;
    u32 y_smooth = 0;  // roughness = 0 (smooth surface)

    f32 scale_perp = lut(x_perpendicular, y_smooth, 0);
    f32 bias_perp = lut(x_perpendicular, y_smooth, 1);

    // scale + bias should be close to 1.0 at perpendicular view
    EXPECT_NEAR(scale_perp + bias_perp, 1.0f, 0.1f);

    // At grazing angles (NdotV -> 0), Fresnel effect increases bias
    u32 x_grazing = 0;
    f32 scale_grazing = lut(x_grazing, y_smooth, 0);
    f32 bias_grazing = lut(x_grazing, y_smooth, 1);

    // bias should increase at grazing angles
    EXPECT_GT(bias_grazing, bias_perp);
}

// ============================================================================
// Binary Cache Save/Load Tests
// ============================================================================

TEST_F(BRDFLutGeneratorTest, SaveAndLoadBinaryCache) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    // Generate original LUT
    Image original = BRDFLutGenerator::Generate(config);

    auto cachePath = GetTestPath("brdf_cache.bin");

    // Save to binary
    bool saveSuccess = BRDFLutGenerator::SaveToBinary(cachePath.string(), original, config);
    ASSERT_TRUE(saveSuccess);
    EXPECT_TRUE(std::filesystem::exists(cachePath));

    // Load from binary
    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string(), &config);
    ASSERT_TRUE(loaded.has_value());

    // Verify dimensions
    EXPECT_EQ(loaded->width, original.width);
    EXPECT_EQ(loaded->height, original.height);
    EXPECT_EQ(loaded->channels, original.channels);

    // Verify pixel data (exact match for binary format)
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            EXPECT_FLOAT_EQ((*loaded)(x, y, 0), original(x, y, 0))
                << "scale mismatch at (" << x << ", " << y << ")";
            EXPECT_FLOAT_EQ((*loaded)(x, y, 1), original(x, y, 1))
                << "bias mismatch at (" << x << ", " << y << ")";
        }
    }
}

TEST_F(BRDFLutGeneratorTest, LoadBinaryWithoutValidation) {
    BRDFLutGenerator::Config config;
    config.resolution = 16;
    config.sampleCount = 32;

    Image original = BRDFLutGenerator::Generate(config);

    auto cachePath = GetTestPath("brdf_no_validate.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), original, config));

    // Load without config validation (pass nullptr)
    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string(), nullptr);
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->width, 16u);
    EXPECT_EQ(loaded->height, 16u);
}

// ============================================================================
// Cache Validation Tests
// ============================================================================

TEST_F(BRDFLutGeneratorTest, IsCacheValidTrue) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    Image lut = BRDFLutGenerator::Generate(config);
    auto cachePath = GetTestPath("valid_cache.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config));

    // Cache should be valid with same config
    EXPECT_TRUE(BRDFLutGenerator::IsCacheValid(cachePath.string(), config));
}

TEST_F(BRDFLutGeneratorTest, IsCacheValidFalseNonexistent) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    auto cachePath = GetTestPath("nonexistent.bin");

    EXPECT_FALSE(BRDFLutGenerator::IsCacheValid(cachePath.string(), config));
}

TEST_F(BRDFLutGeneratorTest, IsCacheValidFalseResolutionMismatch) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    Image lut = BRDFLutGenerator::Generate(config);
    auto cachePath = GetTestPath("res_mismatch.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config));

    // Different resolution should invalidate cache
    BRDFLutGenerator::Config differentConfig;
    differentConfig.resolution = 64;  // Different!
    differentConfig.sampleCount = 64;

    EXPECT_FALSE(BRDFLutGenerator::IsCacheValid(cachePath.string(), differentConfig));
}

TEST_F(BRDFLutGeneratorTest, IsCacheValidFalseSampleCountMismatch) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    Image lut = BRDFLutGenerator::Generate(config);
    auto cachePath = GetTestPath("sample_mismatch.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config));

    // Different sample count should invalidate cache
    BRDFLutGenerator::Config differentConfig;
    differentConfig.resolution = 32;
    differentConfig.sampleCount = 128;  // Different!

    EXPECT_FALSE(BRDFLutGenerator::IsCacheValid(cachePath.string(), differentConfig));
}

TEST_F(BRDFLutGeneratorTest, LoadBinaryRejectsResolutionMismatch) {
    BRDFLutGenerator::Config config;
    config.resolution = 32;
    config.sampleCount = 64;

    Image lut = BRDFLutGenerator::Generate(config);
    auto cachePath = GetTestPath("reject_res.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config));

    // Try to load with different expected resolution
    BRDFLutGenerator::Config differentConfig;
    differentConfig.resolution = 64;
    differentConfig.sampleCount = 64;

    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string(), &differentConfig);
    EXPECT_FALSE(loaded.has_value());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(BRDFLutGeneratorTest, LoadBinaryNonexistentFile) {
    auto cachePath = GetTestPath("nonexistent.bin");

    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(BRDFLutGeneratorTest, LoadBinaryInvalidMagic) {
    auto cachePath = GetTestPath("invalid_magic.bin");

    // Create file with wrong magic number
    std::ofstream file(cachePath, std::ios::binary);
    u32 wrongMagic = 0x12345678;
    u32 version = 1;
    u32 resolution = 32;
    u32 sampleCount = 64;

    file.write(reinterpret_cast<char*>(&wrongMagic), sizeof(wrongMagic));
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    file.write(reinterpret_cast<char*>(&resolution), sizeof(resolution));
    file.write(reinterpret_cast<char*>(&sampleCount), sizeof(sampleCount));
    file.close();

    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(BRDFLutGeneratorTest, LoadBinaryInvalidVersion) {
    auto cachePath = GetTestPath("invalid_version.bin");

    // Create file with wrong version
    std::ofstream file(cachePath, std::ios::binary);
    u32 magic = 0x4C444642;  // Correct magic
    u32 wrongVersion = 999;  // Wrong version
    u32 resolution = 32;
    u32 sampleCount = 64;

    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<char*>(&wrongVersion), sizeof(wrongVersion));
    file.write(reinterpret_cast<char*>(&resolution), sizeof(resolution));
    file.write(reinterpret_cast<char*>(&sampleCount), sizeof(sampleCount));
    file.close();

    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(BRDFLutGeneratorTest, LoadBinaryTruncatedFile) {
    auto cachePath = GetTestPath("truncated.bin");

    // Create file with valid header but truncated data
    std::ofstream file(cachePath, std::ios::binary);
    u32 magic = 0x4C444642;
    u32 version = 1;
    u32 resolution = 32;
    u32 sampleCount = 64;

    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    file.write(reinterpret_cast<char*>(&resolution), sizeof(resolution));
    file.write(reinterpret_cast<char*>(&sampleCount), sizeof(sampleCount));
    // Don't write any data - file is truncated
    file.close();

    auto loaded = BRDFLutGenerator::LoadFromBinary(cachePath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(BRDFLutGeneratorTest, SaveBinaryCreatesParentDirectories) {
    // Cache path with non-existent parent directory
    auto cachePath = testDir / "subdir1" / "subdir2" / "cache.bin";

    BRDFLutGenerator::Config config;
    config.resolution = 16;
    config.sampleCount = 32;

    Image lut = BRDFLutGenerator::Generate(config);

    // Should create parent directories automatically
    bool saveSuccess = BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config);
    EXPECT_TRUE(saveSuccess);
    EXPECT_TRUE(std::filesystem::exists(cachePath));
}

// ============================================================================
// File Size Validation
// ============================================================================

TEST_F(BRDFLutGeneratorTest, BinaryCacheFileSize) {
    BRDFLutGenerator::Config config;
    config.resolution = 64;
    config.sampleCount = 128;

    Image lut = BRDFLutGenerator::Generate(config);
    auto cachePath = GetTestPath("size_check.bin");

    ASSERT_TRUE(BRDFLutGenerator::SaveToBinary(cachePath.string(), lut, config));

    // Expected size: header (16 bytes) + data (64 * 64 * 2 * 4 bytes)
    size_t expectedSize = 16 + (64 * 64 * 2 * sizeof(f32));
    size_t actualSize = std::filesystem::file_size(cachePath);

    EXPECT_EQ(actualSize, expectedSize);
}
