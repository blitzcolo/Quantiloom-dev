// ============================================================================
// Quantiloom - Unit Tests for io/ImageIO.hpp
// ============================================================================
// Tests cover:
// - EXR file writing and reading
// - Channel name preservation
// - Metadata preservation
// - Multi-channel support
// - HDR value preservation
// - File existence checking
// - Dimension reading without loading
// ============================================================================

#include <gtest/gtest.h>
#include "io/ImageIO.hpp"
#include "core/Image.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <cstdio>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temp Directory
// ============================================================================

class ImageIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        testDir = std::filesystem::temp_directory_path() / "quantiloom_imageio_test";
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
// Basic Write/Read Tests
// ============================================================================

TEST_F(ImageIOTest, WriteAndReadSingleChannel) {
    // Create a grayscale image
    Image original(10, 10, 1);
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            original(x, y, 0) = static_cast<f32>(x + y) / 20.0f;
        }
    }

    auto filepath = GetTestPath("single_channel.exr");

    // Write
    bool writeSuccess = ImageIO::WriteEXR(filepath.string(), original);
    ASSERT_TRUE(writeSuccess);
    EXPECT_TRUE(ImageIO::FileExists(filepath.string()));

    // Read
    auto readResult = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(readResult.has_value());

    Image loaded = std::move(*readResult);

    // Verify dimensions
    EXPECT_EQ(loaded.width, original.width);
    EXPECT_EQ(loaded.height, original.height);
    EXPECT_EQ(loaded.channels, original.channels);

    // Verify pixel data
    for (u32 y = 0; y < loaded.height; ++y) {
        for (u32 x = 0; x < loaded.width; ++x) {
            EXPECT_NEAR(loaded(x, y, 0), original(x, y, 0), 1e-6f);
        }
    }
}

TEST_F(ImageIOTest, WriteAndReadRGB) {
    // Create an RGB image
    Image original(20, 15, 3);
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            original(x, y, 0) = static_cast<f32>(x) / 20.0f;      // Red
            original(x, y, 1) = static_cast<f32>(y) / 15.0f;      // Green
            original(x, y, 2) = static_cast<f32>(x * y) / 300.0f; // Blue
        }
    }

    auto filepath = GetTestPath("rgb.exr");

    // Write and read
    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Verify all channels
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            EXPECT_NEAR((*loaded)(x, y, 0), original(x, y, 0), 1e-6f);
            EXPECT_NEAR((*loaded)(x, y, 1), original(x, y, 1), 1e-6f);
            EXPECT_NEAR((*loaded)(x, y, 2), original(x, y, 2), 1e-6f);
        }
    }
}

TEST_F(ImageIOTest, WriteAndReadMultispectral) {
    // Create a multi-channel image (simulating multispectral bands)
    Image original(10, 10, 8);

    original.channelNames = {
        "VIS_450", "VIS_550", "VIS_650",
        "NIR_850", "NIR_950",
        "SWIR_1600", "SWIR_2200", "SWIR_2400"
    };

    // Fill with test pattern
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            for (u32 c = 0; c < original.channels; ++c) {
                original(x, y, c) = static_cast<f32>(c) * 0.1f + static_cast<f32>(x + y) / 20.0f;
            }
        }
    }

    auto filepath = GetTestPath("multispectral.exr");

    // Write and read
    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Verify all channels
    EXPECT_EQ(loaded->channels, 8);

    // NOTE: OpenEXR returns channels in alphabetical order
    // We need to match channels by name, not by index
    std::map<std::string, u32> originalChannelIndices;
    for (u32 c = 0; c < original.channels; ++c) {
        originalChannelIndices[original.channelNames[c]] = c;
    }

    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            for (u32 c = 0; c < loaded->channels; ++c) {
                // Find the original index of this channel name
                const std::string& channelName = loaded->channelNames[c];
                u32 originalIndex = originalChannelIndices[channelName];
                EXPECT_NEAR((*loaded)(x, y, c), original(x, y, originalIndex), 1e-6f);
            }
        }
    }
}

// ============================================================================
// HDR Value Tests
// ============================================================================

TEST_F(ImageIOTest, PreserveHDRValues) {
    Image original(5, 5, 3);

    // Set HDR values (much greater than 1.0)
    original(0, 0, 0) = 100.0f;
    original(1, 1, 1) = 1000.0f;
    original(2, 2, 2) = 10000.0f;
    original(3, 3, 0) = 0.00001f;  // Very small value

    auto filepath = GetTestPath("hdr.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // HDR values should be preserved
    EXPECT_NEAR((*loaded)(0, 0, 0), 100.0f, 1e-3f);
    EXPECT_NEAR((*loaded)(1, 1, 1), 1000.0f, 1e-2f);
    EXPECT_NEAR((*loaded)(2, 2, 2), 10000.0f, 1e-1f);
    EXPECT_NEAR((*loaded)(3, 3, 0), 0.00001f, 1e-9f);
}

TEST_F(ImageIOTest, PreserveNegativeValues) {
    Image original(5, 5, 1);

    // EXR can store negative values (useful for certain processing)
    original(0, 0, 0) = -1.0f;
    original(1, 1, 0) = -0.5f;
    original(2, 2, 0) = 0.0f;
    original(3, 3, 0) = 0.5f;
    original(4, 4, 0) = 1.0f;

    auto filepath = GetTestPath("negative.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_NEAR((*loaded)(0, 0, 0), -1.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(1, 1, 0), -0.5f, 1e-6f);
    EXPECT_NEAR((*loaded)(2, 2, 0), 0.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(3, 3, 0), 0.5f, 1e-6f);
    EXPECT_NEAR((*loaded)(4, 4, 0), 1.0f, 1e-6f);
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_F(ImageIOTest, PreserveMetadata) {
    Image original(10, 10, 3);

    original.metadata["spp"] = "64";
    original.metadata["mode"] = "MS-RT";
    original.metadata["seconds_per_frame"] = "2.5";
    original.metadata["renderer"] = "Quantiloom";

    auto filepath = GetTestPath("with_metadata.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Metadata should be preserved
    EXPECT_EQ(loaded->metadata["spp"], "64");
    EXPECT_EQ(loaded->metadata["mode"], "MS-RT");
    EXPECT_EQ(loaded->metadata["seconds_per_frame"], "2.5");
    EXPECT_EQ(loaded->metadata["renderer"], "Quantiloom");
}

// ============================================================================
// Channel Name Tests
// ============================================================================

TEST_F(ImageIOTest, PreserveChannelNames) {
    Image original(10, 10, 3);

    original.channelNames[0] = "VIS_550";
    original.channelNames[1] = "NIR_850";
    original.channelNames[2] = "SWIR_1600";

    auto filepath = GetTestPath("channel_names.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Channel names should be preserved (but OpenEXR returns them alphabetically)
    EXPECT_EQ(loaded->channelNames.size(), 3);
    EXPECT_EQ(loaded->channelNames[0], "NIR_850");    // Alphabetically first
    EXPECT_EQ(loaded->channelNames[1], "SWIR_1600");  // Alphabetically second
    EXPECT_EQ(loaded->channelNames[2], "VIS_550");    // Alphabetically third
}

// ============================================================================
// File Utilities Tests
// ============================================================================

TEST_F(ImageIOTest, FileExistsTrue) {
    Image img(10, 10, 3);
    auto filepath = GetTestPath("exists.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), img));
    EXPECT_TRUE(ImageIO::FileExists(filepath.string()));
}

TEST_F(ImageIOTest, FileExistsFalse) {
    auto filepath = GetTestPath("nonexistent.exr");

    EXPECT_FALSE(ImageIO::FileExists(filepath.string()));
}

TEST_F(ImageIOTest, GetDimensionsWithoutLoading) {
    Image original(123, 456, 7);
    auto filepath = GetTestPath("dimensions.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    // Get dimensions without loading full image
    auto dims = ImageIO::GetDimensions(filepath.string());
    ASSERT_TRUE(dims.has_value());

    auto [width, height, channels] = *dims;
    EXPECT_EQ(width, 123u);
    EXPECT_EQ(height, 456u);
    EXPECT_EQ(channels, 7u);
}

TEST_F(ImageIOTest, GetDimensionsNonexistent) {
    auto filepath = GetTestPath("nonexistent.exr");

    auto dims = ImageIO::GetDimensions(filepath.string());
    EXPECT_FALSE(dims.has_value());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(ImageIOTest, ReadNonexistentFile) {
    auto filepath = GetTestPath("nonexistent.exr");

    auto result = ImageIO::ReadEXR(filepath.string());
    EXPECT_FALSE(result.has_value());
}

TEST_F(ImageIOTest, ReadInvalidEXR) {
    auto filepath = GetTestPath("invalid.exr");

    // Create an invalid file (not actually EXR data)
    std::ofstream file(filepath);
    file << "This is not an EXR file";
    file.close();

    auto result = ImageIO::ReadEXR(filepath.string());
    EXPECT_FALSE(result.has_value());
}

TEST_F(ImageIOTest, WriteToInvalidPath) {
    Image img(10, 10, 3);

    // Try to write to a path that doesn't exist and can't be created
    std::string invalidPath = "/nonexistent_root_directory/test.exr";

    bool result = ImageIO::WriteEXR(invalidPath, img);
    EXPECT_FALSE(result);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ImageIOTest, SinglePixelImage) {
    Image original(1, 1, 3);
    original(0, 0, 0) = 1.0f;
    original(0, 0, 1) = 0.5f;
    original(0, 0, 2) = 0.25f;

    auto filepath = GetTestPath("single_pixel.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->width, 1u);
    EXPECT_EQ(loaded->height, 1u);
    EXPECT_NEAR((*loaded)(0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(0, 0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR((*loaded)(0, 0, 2), 0.25f, 1e-6f);
}

TEST_F(ImageIOTest, LargeImageStressTest) {
    // Test with a reasonably large image
    Image original(1920, 1080, 4);  // Full HD, 4 channels

    // Fill with pattern
    for (u32 y = 0; y < original.height; ++y) {
        for (u32 x = 0; x < original.width; ++x) {
            original(x, y, 0) = static_cast<f32>(x) / 1920.0f;
            original(x, y, 1) = static_cast<f32>(y) / 1080.0f;
            original(x, y, 2) = 0.5f;
            original(x, y, 3) = 1.0f;
        }
    }

    auto filepath = GetTestPath("large.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->width, 1920u);
    EXPECT_EQ(loaded->height, 1080u);
    EXPECT_EQ(loaded->channels, 4u);

    // Spot check a few pixels
    EXPECT_NEAR((*loaded)(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(1919, 1079, 0), 1919.0f / 1920.0f, 1e-4f);
    EXPECT_NEAR((*loaded)(960, 540, 2), 0.5f, 1e-6f);
}

TEST_F(ImageIOTest, ManyChannelsImage) {
    // Test with 200 channels (hyperspectral scenario)
    Image original(50, 50, 200);

    // Use zero-padded channel names so alphabetical order matches numeric order
    for (u32 c = 0; c < original.channels; ++c) {
        char buf[32];
        snprintf(buf, sizeof(buf), "Channel_%03u", c);  // Channel_000, Channel_001, ..., Channel_199
        original.channelNames[c] = buf;
    }

    for (u32 c = 0; c < original.channels; ++c) {
        for (u32 y = 0; y < original.height; ++y) {
            for (u32 x = 0; x < original.width; ++x) {
                original(x, y, c) = static_cast<f32>(c) / 200.0f;
            }
        }
    }

    auto filepath = GetTestPath("many_channels.exr");

    ASSERT_TRUE(ImageIO::WriteEXR(filepath.string(), original));

    auto loaded = ImageIO::ReadEXR(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->channels, 200u);

    // With zero-padded names, alphabetical order matches numeric order
    EXPECT_NEAR((*loaded)(25, 25, 0), 0.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(25, 25, 100), 100.0f / 200.0f, 1e-6f);
    EXPECT_NEAR((*loaded)(25, 25, 199), 199.0f / 200.0f, 1e-6f);
}
