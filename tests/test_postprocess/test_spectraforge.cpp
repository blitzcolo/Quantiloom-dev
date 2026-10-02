#include <gtest/gtest.h>
#include "../../src/libSpectraForge/ColorCluster.hpp"
#include "../../src/libSpectraForge/SpectraForge.hpp"
#include <stdexcept>

using namespace spectraforge;

TEST(SpectraForgeTest, GrayscaleMatchesReplicatedRgbAtTheLastPixel) {
    for (const auto value : {u8{0}, u8{127}, u8{255}}) {
        const u8 gray[] = {value};
        const u8 rgb[] = {value, value, value};
        for (const bool srgb : {false, true}) {
            const auto a = ClusterTextureColors(gray, 1, 1, 1, srgb, 1);
            const auto b = ClusterTextureColors(rgb, 1, 1, 3, srgb, 1);
            ASSERT_EQ(a.centroids.size(), 1u);
            EXPECT_FLOAT_EQ(a.centroids[0].L, b.centroids[0].L);
            EXPECT_FLOAT_EQ(a.centroids[0].a, b.centroids[0].a);
            EXPECT_FLOAT_EQ(a.centroids[0].b, b.centroids[0].b);
        }
    }
}

TEST(SpectraForgeTest, GrayAlphaUsesAlphaOnlyForCoverage) {
    const u8 pixels[] = {255, 0, 64, 255};
    const u8 rgb[] = {64, 64, 64};
    const auto actual = ClusterTextureColors(pixels, 2, 1, 2, false, 1);
    const auto expected = ClusterTextureColors(rgb, 1, 1, 3, false, 1);
    ASSERT_EQ(actual.centroids.size(), 1u);
    EXPECT_FLOAT_EQ(actual.centroids[0].L, expected.centroids[0].L);
    EXPECT_FLOAT_EQ(actual.centroids[0].a, expected.centroids[0].a);
    EXPECT_FLOAT_EQ(actual.centroids[0].b, expected.centroids[0].b);
}

TEST(SpectraForgeTest, RejectsInvalidRawTextureArguments) {
    const u8 pixel = 0;
    EXPECT_THROW(ClusterTextureColors(nullptr, 1, 1, 1, false, 1), std::invalid_argument);
    EXPECT_THROW(ClusterTextureColors(&pixel, 0, 1, 1, false, 1), std::invalid_argument);
    EXPECT_THROW(ClusterTextureColors(&pixel, 1, 1, 0, false, 1), std::invalid_argument);
    EXPECT_THROW(ClusterTextureColors(&pixel, 1, 1, 5, false, 1), std::invalid_argument);
}

TEST(SpectraForgeTest, ProcessSingleAcceptsGrayscaleAndGrayAlpha) {
    for (const u32 channels : {1u, 2u}) {
        quantiloom::Texture texture;
        texture.width = 1;
        texture.height = 1;
        texture.channels = channels;
        texture.pixels = channels == 1 ? std::vector<u8>{128} : std::vector<u8>{128, 255};
        quantiloom::Material material;
        ASSERT_TRUE(SpectraForge::ProcessSingle(material, &texture, 300.0f, 1));
        EXPECT_TRUE(material.HasIRData());
        EXPECT_TRUE(material.ValidateIRKirchhoffLaw());
    }
}
