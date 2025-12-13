// ============================================================================
// Quantiloom - Unit Tests for core/Image.hpp
// ============================================================================
// Tests cover:
// - Image construction and initialization
// - Pixel access and manipulation
// - Dimension management
// - Channel naming
// - Metadata handling
// - Memory layout verification
// ============================================================================

#include <gtest/gtest.h>
#include "core/Image.hpp"

using namespace quantiloom;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(ImageTest, DefaultConstruction) {
    Image img;

    EXPECT_EQ(img.width, 0);
    EXPECT_EQ(img.height, 0);
    EXPECT_EQ(img.channels, 0);
    EXPECT_TRUE(img.data.empty());
    EXPECT_FALSE(img.IsValid());
}

TEST(ImageTest, ParameterizedConstruction) {
    Image img(100, 50, 3);

    EXPECT_EQ(img.width, 100);
    EXPECT_EQ(img.height, 50);
    EXPECT_EQ(img.channels, 3);
    EXPECT_EQ(img.data.size(), 100 * 50 * 3);
    EXPECT_TRUE(img.IsValid());

    // Check default initialization to zero
    for (const auto& val : img.data) {
        EXPECT_EQ(val, 0.0f);
    }
}

TEST(ImageTest, ChannelNamesInitialization) {
    Image img(10, 10, 4);

    EXPECT_EQ(img.channelNames.size(), 4);
    EXPECT_EQ(img.channelNames[0], "Channel_0");
    EXPECT_EQ(img.channelNames[1], "Channel_1");
    EXPECT_EQ(img.channelNames[2], "Channel_2");
    EXPECT_EQ(img.channelNames[3], "Channel_3");
}

// ============================================================================
// Pixel Access Tests
// ============================================================================

TEST(ImageTest, PixelWriteRead) {
    Image img(10, 10, 3);

    img(5, 7, 0) = 1.0f;
    img(5, 7, 1) = 2.0f;
    img(5, 7, 2) = 3.0f;

    EXPECT_EQ(img(5, 7, 0), 1.0f);
    EXPECT_EQ(img(5, 7, 1), 2.0f);
    EXPECT_EQ(img(5, 7, 2), 3.0f);
}

TEST(ImageTest, PixelPointerAccess) {
    Image img(10, 10, 3);

    f32* pixel = img.PixelPtr(3, 4);
    pixel[0] = 0.5f;
    pixel[1] = 0.6f;
    pixel[2] = 0.7f;

    EXPECT_EQ(img(3, 4, 0), 0.5f);
    EXPECT_EQ(img(3, 4, 1), 0.6f);
    EXPECT_EQ(img(3, 4, 2), 0.7f);
}

TEST(ImageTest, ConstPixelPointerAccess) {
    Image img(10, 10, 3);
    img(2, 3, 0) = 1.5f;
    img(2, 3, 1) = 2.5f;
    img(2, 3, 2) = 3.5f;

    const Image& constImg = img;
    const f32* pixel = constImg.PixelPtr(2, 3);

    EXPECT_EQ(pixel[0], 1.5f);
    EXPECT_EQ(pixel[1], 2.5f);
    EXPECT_EQ(pixel[2], 3.5f);
}

// ============================================================================
// Memory Layout Tests
// ============================================================================

TEST(ImageTest, RowMajorChannelLastLayout) {
    Image img(3, 2, 2);  // 3x2 image, 2 channels

    // Set values with known pattern
    for (u32 y = 0; y < img.height; ++y) {
        for (u32 x = 0; x < img.width; ++x) {
            img(x, y, 0) = static_cast<f32>(y * 100 + x * 10 + 0);
            img(x, y, 1) = static_cast<f32>(y * 100 + x * 10 + 1);
        }
    }

    // Verify memory layout: [y][x][c]
    // Row 0: [0,1], [10,11], [20,21]
    // Row 1: [100,101], [110,111], [120,121]
    const f32* data = img.data.data();

    EXPECT_EQ(data[0], 0.0f);    // (0, 0, 0)
    EXPECT_EQ(data[1], 1.0f);    // (0, 0, 1)
    EXPECT_EQ(data[2], 10.0f);   // (1, 0, 0)
    EXPECT_EQ(data[3], 11.0f);   // (1, 0, 1)
    EXPECT_EQ(data[4], 20.0f);   // (2, 0, 0)
    EXPECT_EQ(data[5], 21.0f);   // (2, 0, 1)
    EXPECT_EQ(data[6], 100.0f);  // (0, 1, 0)
    EXPECT_EQ(data[7], 101.0f);  // (0, 1, 1)
}

// ============================================================================
// Dimension and Utility Tests
// ============================================================================

TEST(ImageTest, PixelCount) {
    Image img(100, 50, 3);
    EXPECT_EQ(img.PixelCount(), 5000);
}

TEST(ImageTest, TotalElements) {
    Image img(100, 50, 3);
    EXPECT_EQ(img.TotalElements(), 15000);
}

TEST(ImageTest, IsValid) {
    Image img1(10, 10, 3);
    EXPECT_TRUE(img1.IsValid());

    Image img2;
    EXPECT_FALSE(img2.IsValid());

    Image img3(10, 10, 0);
    EXPECT_FALSE(img3.IsValid());

    // Manually corrupt data size
    Image img4(10, 10, 3);
    img4.data.resize(10);  // Wrong size
    EXPECT_FALSE(img4.IsValid());
}

TEST(ImageTest, Clear) {
    Image img(10, 10, 3);

    // Fill with non-zero values
    for (auto& val : img.data) {
        val = 1.0f;
    }

    img.Clear();

    // Verify all zeros
    for (const auto& val : img.data) {
        EXPECT_EQ(val, 0.0f);
    }
}

TEST(ImageTest, Resize) {
    Image img(10, 10, 3);

    // Set some values
    img(5, 5, 0) = 1.0f;

    // Resize
    img.Resize(20, 15, 4);

    EXPECT_EQ(img.width, 20);
    EXPECT_EQ(img.height, 15);
    EXPECT_EQ(img.channels, 4);
    EXPECT_EQ(img.data.size(), 20 * 15 * 4);
    EXPECT_TRUE(img.IsValid());

    // Old data should be cleared
    EXPECT_EQ(img(5, 5, 0), 0.0f);

    // Channel names should be updated
    EXPECT_EQ(img.channelNames.size(), 4);
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST(ImageTest, MetadataStorage) {
    Image img(10, 10, 3);

    img.metadata["spp"] = "64";
    img.metadata["mode"] = "MS-RT";
    img.metadata["seconds_per_frame"] = "2.5";

    EXPECT_EQ(img.metadata["spp"], "64");
    EXPECT_EQ(img.metadata["mode"], "MS-RT");
    EXPECT_EQ(img.metadata["seconds_per_frame"], "2.5");
    EXPECT_EQ(img.metadata.size(), 3);
}

TEST(ImageTest, ChannelNamesCustom) {
    Image img(10, 10, 3);

    img.channelNames[0] = "VIS_550";
    img.channelNames[1] = "NIR_850";
    img.channelNames[2] = "SWIR_1600";

    EXPECT_EQ(img.channelNames[0], "VIS_550");
    EXPECT_EQ(img.channelNames[1], "NIR_850");
    EXPECT_EQ(img.channelNames[2], "SWIR_1600");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ImageTest, SinglePixelImage) {
    Image img(1, 1, 1);

    EXPECT_TRUE(img.IsValid());
    EXPECT_EQ(img.data.size(), 1);

    img(0, 0, 0) = 42.0f;
    EXPECT_EQ(img(0, 0, 0), 42.0f);
}

TEST(ImageTest, LargeChannelCount) {
    // Hyperspectral image with 200 channels
    Image img(100, 100, 200);

    EXPECT_TRUE(img.IsValid());
    EXPECT_EQ(img.TotalElements(), 100 * 100 * 200);
    EXPECT_EQ(img.channelNames.size(), 200);
}

TEST(ImageTest, HDRValues) {
    Image img(10, 10, 3);

    // Test with HDR values
    img(5, 5, 0) = 10000.0f;
    img(5, 5, 1) = -100.0f;  // Negative values should be allowed
    img(5, 5, 2) = 0.00001f;

    EXPECT_EQ(img(5, 5, 0), 10000.0f);
    EXPECT_EQ(img(5, 5, 1), -100.0f);
    EXPECT_EQ(img(5, 5, 2), 0.00001f);
}
