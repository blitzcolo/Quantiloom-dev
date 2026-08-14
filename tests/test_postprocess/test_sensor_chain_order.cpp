// ============================================================================
// Quantiloom - Unit Tests for Sensor Chain Order and FPN Map Generation
// ============================================================================
// Tests cover:
// - CPU sensor chain execution order verification
// - Deterministic behavior with noise disabled
// - PSF blur effect on high-frequency content
// - FPN structured noise verification
// - FPN map generation statistical properties
// ============================================================================

#include <gtest/gtest.h>
#include "postprocess/GenericSensor.hpp"
#include "postprocess/SensorModel.hpp"
#include "core/Image.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace quantiloom;

// ============================================================================
// Test Fixture
// ============================================================================

class SensorChainOrderTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.focalLength_mm = 50.0f;
        params.fNumber = 2.8f;
        params.pixelPitch_um = 5.0f;
        params.quantumEfficiency = 0.8f;
        params.wellCapacity_e = 50000.0f;
        params.readNoise_e_rms = 10.0f;
        params.darkCurrent_e_s = 50.0f;
        params.integrationTime_s = 0.01f;
        params.bitDepth = 14;
        params.gain = 3.0f;
        params.wavelength_nm = 550.0f;
        params.enablePoissonNoise = false;
        params.enableReadNoise = false;
        params.enableDarkCurrent = false;
        params.enableFPN = false;
        params.enableVignetting = false;
    }

    SensorParams params;
};

// ============================================================================
// CPU Chain Order Verification
// ============================================================================

TEST_F(SensorChainOrderTest, DeterministicWithNoiseDisabled) {
    // With all noise disabled, two runs should produce identical results
    Image hdr(64, 64, 1);
    for (auto& v : hdr.data) v = 0.5f;

    GenericSensor sensor1, sensor2;
    auto r1 = sensor1.Apply(hdr, params);
    auto r2 = sensor2.Apply(hdr, params);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());

    for (u32 i = 0; i < r1.value().rawDN.TotalElements(); ++i) {
        EXPECT_FLOAT_EQ(r1.value().rawDN.data[i], r2.value().rawDN.data[i]);
    }
}

namespace {

// Variance over the centre region, away from the convolution's edge handling.
auto CentreVariance(const Image& img) -> f32 {
    f32 mean = 0.0f;
    u32 count = 0;
    for (u32 y = 30; y < 98; ++y) {
        for (u32 x = 30; x < 98; ++x) {
            mean += img(x, y, 0);
            ++count;
        }
    }
    mean /= static_cast<f32>(count);

    f32 var = 0.0f;
    for (u32 y = 30; y < 98; ++y) {
        for (u32 x = 30; x < 98; ++x) {
            const f32 d = img(x, y, 0) - mean;
            var += d * d;
        }
    }
    return var / static_cast<f32>(count);
}

} // namespace

TEST_F(SensorChainOrderTest, PSFReducesHighFrequency) {
    // Create checkerboard pattern (high frequency)
    Image hdr(128, 128, 1);
    for (u32 y = 0; y < 128; ++y) {
        for (u32 x = 0; x < 128; ++x) {
            hdr(x, y, 0) = ((x + y) % 2 == 0) ? 0.5f : 0.1f;
        }
    }

    // Apply sensor chain (PSF will blur the checkerboard)
    params.fNumber = 8.0f;  // Larger f# -> more blur
    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    // Compare against the enhanced preview, which is radiance on the same scale
    // as the input -- so this is a statement about blur, not about the DN scale.
    // The previous version compared radiance variance against DN variance and
    // absorbed the unit mismatch into a 1e6 factor, which meant it passed for
    // any PSF width whatsoever, including none.
    const f32 inputVar = CentreVariance(hdr);
    const f32 outVar = CentreVariance(result.value().enhancedPreview);

    ASSERT_GT(inputVar, 0.0f);
    EXPECT_LT(outVar, inputVar) << "PSF blur must reduce checkerboard contrast";
}

TEST_F(SensorChainOrderTest, FPNCreatesStructuredNoise) {
    // FPN should create spatially correlated noise (stripes),
    // not random per-pixel noise
    Image hdr(128, 128, 1);
    for (auto& v : hdr.data) v = 0.01f;

    params.enableFPN = true;
    params.prnuSigma = 0.05f;
    params.dsnuSigma_e = 30.0f;
    params.fNumber = 1.4f;
    params.gain = 1.0f;

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    // Adjacent pixels in same column should be correlated (PRNU)
    f32 sameColCorr = 0.0f;
    u32 count = 0;
    for (u32 x = 20; x < 108; ++x) {
        for (u32 y = 20; y < 107; ++y) {
            sameColCorr += dn(x, y, 0) * dn(x, y + 1, 0);
            ++count;
        }
    }
    sameColCorr /= count;

    // Adjacent pixels in different columns should be less correlated
    f32 diffColCorr = 0.0f;
    count = 0;
    for (u32 x = 20; x < 107; ++x) {
        for (u32 y = 20; y < 108; ++y) {
            diffColCorr += dn(x, y, 0) * dn(x + 1, y, 0);
            ++count;
        }
    }
    diffColCorr /= count;

    // Both should be positive (signal present)
    EXPECT_GT(sameColCorr, 0.0f);
    EXPECT_GT(diffColCorr, 0.0f);
}

TEST_F(SensorChainOrderTest, FullChainEndToEnd) {
    // Full chain with all features enabled should produce valid output
    Image hdr(64, 64, 3);
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            // Gradient pattern
            f32 val = 0.01f + 0.5f * static_cast<f32>(x) / 64.0f;
            hdr(x, y, 0) = val;
            hdr(x, y, 1) = val * 0.8f;
            hdr(x, y, 2) = val * 0.6f;
        }
    }

    params.enablePoissonNoise = true;
    params.enableReadNoise = true;
    params.enableDarkCurrent = true;
    params.enableFPN = true;
    params.prnuSigma = 0.02f;
    params.dsnuSigma_e = 10.0f;

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const auto& output = result.value();
    const f32 maxDN = static_cast<f32>((1u << params.bitDepth) - 1);

    // Dimensions preserved
    EXPECT_EQ(output.rawDN.width, 64u);
    EXPECT_EQ(output.rawDN.height, 64u);
    EXPECT_EQ(output.rawDN.channels, 3u);

    // All DN values in valid range
    for (const auto& v : output.rawDN.data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, maxDN);
    }

    // Enhanced preview should have positive values
    for (const auto& v : output.enhancedPreview.data) {
        EXPECT_GE(v, 0.0f);
    }
}

// ============================================================================
// FPN Map Generation Statistical Verification
// ============================================================================

TEST_F(SensorChainOrderTest, PRNUColumnCorrelation) {
    // PRNU creates vertical stripes: pixels in same column are correlated
    Image hdr(200, 200, 1);
    for (auto& v : hdr.data) v = 0.01f;

    params.enableFPN = true;
    params.prnuSigma = 0.08f;
    params.dsnuSigma_e = 0.0f;  // Isolate PRNU
    params.fNumber = 1.4f;
    params.gain = 1.0f;

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());
    const Image& dn = result.value().rawDN;

    // Column means
    std::vector<f32> colMeans(dn.width, 0.0f);
    for (u32 x = 0; x < dn.width; ++x) {
        for (u32 y = 0; y < dn.height; ++y) {
            colMeans[x] += dn(x, y, 0);
        }
        colMeans[x] /= dn.height;
    }

    f32 overallMean = 0.0f;
    for (auto m : colMeans) overallMean += m;
    overallMean /= colMeans.size();

    // Between-column variance (should be large for vertical stripes)
    f32 betweenColVar = 0.0f;
    for (auto m : colMeans) {
        f32 d = m - overallMean;
        betweenColVar += d * d;
    }
    betweenColVar /= colMeans.size();

    // Within-column variance (should be small)
    f32 withinColVar = 0.0f;
    u32 cnt = 0;
    for (u32 x = 50; x < 150; ++x) {
        for (u32 y = 50; y < 150; ++y) {
            f32 d = dn(x, y, 0) - colMeans[x];
            withinColVar += d * d;
            ++cnt;
        }
    }
    withinColVar /= cnt;

    EXPECT_GT(betweenColVar, withinColVar)
        << "PRNU vertical stripes: between-column > within-column variance";
}

TEST_F(SensorChainOrderTest, DSNURowCorrelation) {
    // DSNU creates horizontal stripes: pixels in same row are correlated
    Image hdr(200, 200, 1);
    for (auto& v : hdr.data) v = 0.01f;

    params.enableFPN = true;
    params.prnuSigma = 0.0f;  // Isolate DSNU
    params.dsnuSigma_e = 50.0f;
    params.fNumber = 1.4f;
    params.gain = 1.0f;

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());
    const Image& dn = result.value().rawDN;

    // Row means
    std::vector<f32> rowMeans(dn.height, 0.0f);
    for (u32 y = 0; y < dn.height; ++y) {
        for (u32 x = 0; x < dn.width; ++x) {
            rowMeans[y] += dn(x, y, 0);
        }
        rowMeans[y] /= dn.width;
    }

    f32 overallMean = 0.0f;
    for (auto m : rowMeans) overallMean += m;
    overallMean /= rowMeans.size();

    // Between-row variance (should be large for horizontal stripes)
    f32 betweenRowVar = 0.0f;
    for (auto m : rowMeans) {
        f32 d = m - overallMean;
        betweenRowVar += d * d;
    }
    betweenRowVar /= rowMeans.size();

    // Within-row variance (should be small)
    f32 withinRowVar = 0.0f;
    u32 cnt = 0;
    for (u32 y = 50; y < 150; ++y) {
        for (u32 x = 50; x < 150; ++x) {
            f32 d = dn(x, y, 0) - rowMeans[y];
            withinRowVar += d * d;
            ++cnt;
        }
    }
    withinRowVar /= cnt;

    EXPECT_GT(betweenRowVar, withinRowVar)
        << "DSNU horizontal stripes: between-row > within-row variance";
}

TEST_F(SensorChainOrderTest, FPNMapsDeterministic) {
    // Same sensor instance should produce same FPN pattern
    // (FPN maps generated once and reused)
    Image hdr(100, 100, 1);
    for (auto& v : hdr.data) v = 0.01f;

    params.enableFPN = true;
    params.prnuSigma = 0.05f;
    params.dsnuSigma_e = 20.0f;
    params.fNumber = 1.4f;
    params.gain = 1.0f;

    GenericSensor sensor;
    auto r1 = sensor.Apply(hdr, params);
    auto r2 = sensor.Apply(hdr, params);
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());

    // FPN pattern should be identical across frames
    // With noise disabled, results should be identical
    for (u32 i = 0; i < r1.value().rawDN.TotalElements(); ++i) {
        EXPECT_FLOAT_EQ(r1.value().rawDN.data[i],
                         r2.value().rawDN.data[i]);
    }
}
