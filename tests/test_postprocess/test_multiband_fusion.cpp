// ============================================================================
// Quantiloom - Unit Tests for postprocess/MultibandFusion
// ============================================================================
// Tests cover:
// - Laplacian pyramid fusion
// - Weighted average fusion
// - Max response fusion
// - Pseudo color fusion
// - Auto-normalization
// - Error handling (mismatched dimensions)
// ============================================================================

#include <gtest/gtest.h>
#include "postprocess/MultibandFusion.hpp"
#include "core/Image.hpp"
#include <cmath>
#include <limits>
#include "../../src/tools/FusionInput.hpp"

using namespace quantiloom;

// ============================================================================
// Test Fixtures
// ============================================================================

class MultibandFusionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test images (grayscale, 64x64)
        vis = Image(64, 64, 1);
        swir = Image(64, 64, 1);
        mwir = Image(64, 64, 1);

        // Default fusion parameters
        params.method = FusionMethod::WeightedAverage;
        params.autoNormalize = true;
        params.pyramidLevels = 5;
        params.weightVis = 0.4f;
        params.weightSwir = 0.3f;
        params.weightMwir = 0.3f;

        // Fill with test patterns
        FillTestPatterns();
    }

    void FillTestPatterns() {
        // VIS: Bright in top-left
        for (u32 y = 0; y < 64; ++y) {
            for (u32 x = 0; x < 64; ++x) {
                vis(x, y, 0) = (x < 32 && y < 32) ? 1.0f : 0.1f;
            }
        }

        // SWIR: Bright in top-right
        for (u32 y = 0; y < 64; ++y) {
            for (u32 x = 0; x < 64; ++x) {
                swir(x, y, 0) = (x >= 32 && y < 32) ? 1.0f : 0.1f;
            }
        }

        // MWIR: Bright in bottom half
        for (u32 y = 0; y < 64; ++y) {
            for (u32 x = 0; x < 64; ++x) {
                mwir(x, y, 0) = (y >= 32) ? 1.0f : 0.1f;
            }
        }
    }

    Image vis, swir, mwir;
    FusionParams params;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(MultibandFusionTest, ValidInputProducesOutput) {
    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);

    ASSERT_TRUE(result.has_value());
    const Image& fused = result.value();

    EXPECT_EQ(fused.width, 64);
    EXPECT_EQ(fused.height, 64);
    EXPECT_EQ(fused.channels, 1);
}

TEST_F(MultibandFusionTest, MismatchedDimensionsReturnsError) {
    Image wrongSize(32, 32, 1);  // Different size

    auto result = MultibandFusion::Fuse(vis, wrongSize, mwir, params);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("dimensions"), std::string::npos);
}

TEST_F(MultibandFusionTest, MismatchedChannelsReturnsError) {
    Image rgbImage(64, 64, 3);  // Wrong channel count

    auto result = MultibandFusion::Fuse(vis, rgbImage, mwir, params);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("channel"), std::string::npos);
}

TEST_F(MultibandFusionTest, MatchingMultichannelInputsAreRejectedByEveryMethod) {
    const Image rgb(4, 4, 3);
    for (const auto method : {FusionMethod::WeightedAverage, FusionMethod::MaxResponse,
                              FusionMethod::LaplacianPyramid, FusionMethod::PseudoColor}) {
        params.method = method;
        EXPECT_FALSE(MultibandFusion::Fuse(rgb, rgb, rgb, params).has_value());
    }
}

TEST_F(MultibandFusionTest, RejectsZeroAndExcessivePyramidDepth) {
    const Image small(8, 8, 1);
    params.method = FusionMethod::LaplacianPyramid;
    for (const u32 depth : {0u, 5u, std::numeric_limits<u32>::max()}) {
        params.pyramidLevels = depth;
        EXPECT_FALSE(MultibandFusion::Fuse(small, small, small, params).has_value());
    }
    params.method = FusionMethod::WeightedAverage;
    EXPECT_TRUE(MultibandFusion::Fuse(small, small, small, params).has_value());
}

TEST_F(MultibandFusionTest, PyramidAcceptsLastNonemptyLevelAndSinglePixelBands) {
    params.method = FusionMethod::LaplacianPyramid;
    params.autoNormalize = false;
    for (const auto extent : {std::pair{16u, 16u}, std::pair{1u, 8u},
                              std::pair{8u, 1u}, std::pair{1u, 1u}, std::pair{7u, 5u}}) {
        Image input(extent.first, extent.second, 1);
        std::fill(input.data.begin(), input.data.end(), 0.4f);
        params.pyramidLevels = extent.first == 16 ? 5 : (extent.first == 7 ? 3 : 1);
        const auto result = MultibandFusion::Fuse(input, input, input, params);
        ASSERT_TRUE(result.has_value());
        for (const auto value : result.value().data) EXPECT_NEAR(value, 0.4f, 1e-6f);
    }
}

TEST(FusionInputTest, RejectsDifferentExtentsBeforeJointCopy) {
    const Image vis(4, 4, 1), smaller(2, 2, 1), larger(8, 8, 1);
    EXPECT_FALSE(PrepareFusionInputs(vis, smaller, vis).has_value());
    EXPECT_FALSE(PrepareFusionInputs(vis, vis, larger).has_value());
    Image malformed = vis;
    malformed.data.pop_back();
    EXPECT_FALSE(PrepareFusionInputs(vis, malformed, vis).has_value());
}

TEST(FusionInputTest, ExtractsNamedRadianceFromIndependentChannelLayouts) {
    Image rgba(2, 2, 4), gray(2, 2, 1), rgb(2, 2, 3);
    rgba.channelNames = {"A", "B", "G", "R"};
    rgb.channelNames = {"B", "G", "R"};
    for (u32 y = 0; y < 2; ++y) {
        for (u32 x = 0; x < 2; ++x) {
            rgba(x, y, 0) = 1.0f;
            rgba(x, y, 3) = 0.2f;
            gray(x, y, 0) = 0.3f;
            rgb(x, y, 2) = 0.4f;
        }
    }
    const auto bands = PrepareFusionInputs(rgba, gray, rgb);
    ASSERT_TRUE(bands.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(bands.value()[i].channels, 1u);
        for (const auto value : bands.value()[i].data)
            EXPECT_NEAR(value, 0.2f + static_cast<float>(i) * 0.1f, 1e-6f);
    }
}

// ============================================================================
// Weighted Average Fusion Tests
// ============================================================================

TEST_F(MultibandFusionTest, WeightedAveragePreservesStructure) {
    params.method = FusionMethod::WeightedAverage;
    params.weightVis = 1.0f;
    params.weightSwir = 0.0f;
    params.weightMwir = 0.0f;
    params.autoNormalize = false;

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // With 100% VIS weight, output should match VIS input
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            EXPECT_NEAR(fused(x, y, 0), vis(x, y, 0), 0.01f);
        }
    }
}

TEST_F(MultibandFusionTest, WeightedAverageBlendsBands) {
    params.method = FusionMethod::WeightedAverage;
    params.weightVis = 0.5f;
    params.weightSwir = 0.5f;
    params.weightMwir = 0.0f;
    params.autoNormalize = false;

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Check center pixel (should blend VIS + SWIR)
    // VIS top-left bright, SWIR top-right bright
    // Center pixel should have intermediate value
    f32 topLeft = fused(16, 16, 0);   // VIS bright region
    f32 topRight = fused(48, 16, 0);  // SWIR bright region
    f32 center = fused(32, 32, 0);    // Blend region

    EXPECT_GT(topLeft, center);   // VIS bright area > center
    EXPECT_GT(topRight, center);  // SWIR bright area > center
}

// ============================================================================
// Max Response Fusion Tests
// ============================================================================

TEST_F(MultibandFusionTest, MaxResponseTakesMaximum) {
    params.method = FusionMethod::MaxResponse;
    params.autoNormalize = false;

    // Set distinct values for each band
    for (u32 i = 0; i < vis.data.size(); ++i) {
        vis.data[i] = 0.3f;
        swir.data[i] = 0.7f;  // Maximum
        mwir.data[i] = 0.5f;
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // All pixels should equal SWIR (the maximum)
    for (u32 i = 0; i < fused.data.size(); ++i) {
        EXPECT_FLOAT_EQ(fused.data[i], 0.7f);
    }
}

TEST_F(MultibandFusionTest, MaxResponseHandlesSpatialVariation) {
    params.method = FusionMethod::MaxResponse;
    params.autoNormalize = false;

    // Create spatial pattern where different bands dominate in different regions
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            vis(x, y, 0) = (x < 20) ? 1.0f : 0.0f;      // Left edge bright
            swir(x, y, 0) = (x >= 20 && x < 44) ? 1.0f : 0.0f;  // Center bright
            mwir(x, y, 0) = (x >= 44) ? 1.0f : 0.0f;    // Right edge bright
        }
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Check that correct band dominates in each region
    EXPECT_NEAR(fused(10, 32, 0), 1.0f, 0.01f);  // VIS region
    EXPECT_NEAR(fused(32, 32, 0), 1.0f, 0.01f);  // SWIR region
    EXPECT_NEAR(fused(54, 32, 0), 1.0f, 0.01f);  // MWIR region
}

// ============================================================================
// Pseudo Color Fusion Tests
// ============================================================================

TEST_F(MultibandFusionTest, PseudoColorProducesRGB) {
    params.method = FusionMethod::PseudoColor;

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Pseudo color should output 3-channel RGB
    EXPECT_EQ(fused.channels, 3);
    EXPECT_EQ(fused.width, 64);
    EXPECT_EQ(fused.height, 64);
}

TEST_F(MultibandFusionTest, PseudoColorMapsToRGBChannels) {
    params.method = FusionMethod::PseudoColor;
    params.autoNormalize = false;

    // Set distinct values for each band
    for (u32 i = 0; i < vis.data.size(); ++i) {
        vis.data[i] = 0.3f;   // Maps to R
        swir.data[i] = 0.6f;  // Maps to G
        mwir.data[i] = 0.9f;  // Maps to B
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Check RGB mapping
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            EXPECT_FLOAT_EQ(fused(x, y, 0), 0.3f);  // R = VIS
            EXPECT_FLOAT_EQ(fused(x, y, 1), 0.6f);  // G = SWIR
            EXPECT_FLOAT_EQ(fused(x, y, 2), 0.9f);  // B = MWIR
        }
    }
}

// ============================================================================
// Laplacian Pyramid Fusion Tests
// ============================================================================

TEST_F(MultibandFusionTest, LaplacianPyramidProducesValidOutput) {
    params.method = FusionMethod::LaplacianPyramid;
    params.pyramidLevels = 4;

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);

    ASSERT_TRUE(result.has_value());
    const Image& fused = result.value();

    EXPECT_EQ(fused.width, 64);
    EXPECT_EQ(fused.height, 64);
    EXPECT_EQ(fused.channels, 1);

    // Laplacian pyramid reconstruction can have significant overshoot due to interpolation
    // This is normal behavior (Gibbs phenomenon) - allow wide range
    for (const auto& val : fused.data) {
        EXPECT_GE(val, -2.0f);
        EXPECT_LE(val, 3.0f);
        EXPECT_TRUE(std::isfinite(val));  // Main check: no NaN/inf
    }
}

TEST_F(MultibandFusionTest, LaplacianPyramidPreservesDetails) {
    params.method = FusionMethod::LaplacianPyramid;
    params.pyramidLevels = 5;

    // Create image with high-frequency details
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            // Checkerboard pattern in VIS
            vis(x, y, 0) = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
            swir(x, y, 0) = 0.5f;  // Uniform
            mwir(x, y, 0) = 0.5f;  // Uniform
        }
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Fused image should preserve high-frequency checkerboard pattern
    // Check a few pixels to verify pattern preservation
    f32 val1 = fused(10, 10, 0);  // Even position
    f32 val2 = fused(11, 10, 0);  // Odd position

    // Should have visible difference (not averaged out)
    EXPECT_GT(std::abs(val1 - val2), 0.1f);
}

// ============================================================================
// Auto-Normalization Tests
// ============================================================================

TEST_F(MultibandFusionTest, AutoNormalizeEnforcesRange) {
    params.method = FusionMethod::WeightedAverage;
    params.autoNormalize = true;

    // Create images with different dynamic ranges
    for (u32 i = 0; i < vis.data.size(); ++i) {
        vis.data[i] = static_cast<f32>(i) / vis.data.size();  // [0, 1]
        swir.data[i] = static_cast<f32>(i) / vis.data.size() * 0.001f;  // [0, 0.001]
        mwir.data[i] = static_cast<f32>(i) / vis.data.size() * 100.0f;  // [0, 100]
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // After auto-normalization, all bands contribute equally
    // Output should be in [0, 1] range
    for (const auto& val : fused.data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

TEST_F(MultibandFusionTest, NoAutoNormalizePreservesScale) {
    params.method = FusionMethod::WeightedAverage;
    params.autoNormalize = false;
    params.weightVis = 1.0f;
    params.weightSwir = 0.0f;
    params.weightMwir = 0.0f;

    // VIS with values > 1.0
    for (u32 i = 0; i < vis.data.size(); ++i) {
        vis.data[i] = 5.0f;
        swir.data[i] = 0.0f;
        mwir.data[i] = 0.0f;
    }

    auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
    ASSERT_TRUE(result.has_value());

    const Image& fused = result.value();

    // Without auto-normalization, values > 1.0 are preserved
    for (const auto& val : fused.data) {
        EXPECT_NEAR(val, 5.0f, 0.01f);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(MultibandFusionTest, AllMethodsProduceValidOutput) {
    FusionMethod methods[] = {
        FusionMethod::LaplacianPyramid,
        FusionMethod::WeightedAverage,
        FusionMethod::MaxResponse,
        FusionMethod::PseudoColor
    };

    for (auto method : methods) {
        params.method = method;
        params.autoNormalize = true;

        auto result = MultibandFusion::Fuse(vis, swir, mwir, params);

        ASSERT_TRUE(result.has_value()) << "Method failed: " << static_cast<int>(method);

        const Image& fused = result.value();

        // Basic sanity checks
        EXPECT_EQ(fused.width, 64);
        EXPECT_EQ(fused.height, 64);

        // Check channel count (pseudo color has 3, others have 1)
        if (method == FusionMethod::PseudoColor) {
            EXPECT_EQ(fused.channels, 3);
        } else {
            EXPECT_EQ(fused.channels, 1);
        }

        // All values should be finite
        for (const auto& val : fused.data) {
            EXPECT_TRUE(std::isfinite(val));
        }
    }
}
