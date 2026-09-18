// ============================================================================
// Quantiloom - Unit Tests for Sensor Chain Order and FPN Map Generation
// ============================================================================
// Tests cover:
// - CPU sensor chain execution order verification
// - Deterministic behavior with noise disabled
// - PSF blur effect on high-frequency content
// - fixed per-pixel PRNU and dark nonuniformity
// - fixed-pattern persistence across captures
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
    Image hdr(128, 128, 1);
    for (u32 y = 0; y < 128; ++y)
        for (u32 x = 0; x < 128; ++x)
            hdr(x, y, 0) = ((x + y) % 2 == 0) ? 0.005f : 0.001f;
    params.fNumber = 8.0f;
    params.psfSigma_px = 0.0f;
    GenericSensor sharpSensor;
    auto sharp = sharpSensor.Apply(hdr, params);
    ASSERT_TRUE(sharp.has_value());
    params.psfSigma_px = -1.0f;
    GenericSensor blurredSensor;
    auto blurred = blurredSensor.Apply(hdr, params);
    ASSERT_TRUE(blurred.has_value());
    // Compare two previews in the same encoded display space; input radiance
    // and a display image have different units and cannot be compared directly.
    EXPECT_LT(CentreVariance(blurred.value().enhancedPreview),
              CentreVariance(sharp.value().enhancedPreview));
}

TEST_F(SensorChainOrderTest, FPNCreatesStableSpatialNoise) {
    Image hdr(64, 64, 1);
    for (auto& value : hdr.data) value = 0.01f;
    params.enableFPN = true;
    params.prnuSigma = 0.05f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    auto first = GenericSensor().Apply(hdr, params);
    auto second = GenericSensor().Apply(hdr, params);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first.value().rawDN.data, second.value().rawDN.data);
    const f32 baseline = first.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : first.value().rawDN.data) if (value != baseline) ++different;
    EXPECT_GT(different, first.value().rawDN.data.size() / 4);
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
// Fixed-Pattern Statistical Verification
// ============================================================================

TEST_F(SensorChainOrderTest, PRNUOnlyModulatesIlluminatedCharge) {
    Image dark(64, 64, 1);
    Image lit(64, 64, 1);
    for (auto& value : lit.data) value = 0.01f;
    params.enableFPN = true;
    params.prnuSigma = 0.08f;
    params.dsnuSigma_e = 0.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    GenericSensor sensor;
    auto darkFrame = sensor.Apply(dark, params);
    auto litFrame = sensor.Apply(lit, params);
    ASSERT_TRUE(darkFrame.has_value());
    ASSERT_TRUE(litFrame.has_value());
    for (f32 value : darkFrame.value().rawDN.data) EXPECT_FLOAT_EQ(value, 0.0f);
    const f32 first = litFrame.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : litFrame.value().rawDN.data) if (value != first) ++different;
    EXPECT_GT(different, litFrame.value().rawDN.data.size() / 4);
}

TEST_F(SensorChainOrderTest, DSNUIsDarkChargeNonuniformity) {
    Image dark(64, 64, 1);
    params.enableFPN = true;
    params.prnuSigma = 0.0f;
    params.dsnuSigma_e = 20.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    GenericSensor noDarkCurrent;
    auto absent = noDarkCurrent.Apply(dark, params);
    ASSERT_TRUE(absent.has_value());
    for (f32 value : absent.value().rawDN.data) EXPECT_FLOAT_EQ(value, 0.0f);
    params.enableDarkCurrent = true;
    params.darkCurrent_e_s = 5000.0f;
    GenericSensor darkCurrent;
    auto present = darkCurrent.Apply(dark, params);
    ASSERT_TRUE(present.has_value());
    const f32 first = present.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : present.value().rawDN.data) if (value != first) ++different;
    EXPECT_GT(different, present.value().rawDN.data.size() / 4);
}

TEST_F(SensorChainOrderTest, FPNMapsDeterministic) {
    // Fixed-pattern keys exclude the acquisition index, so a second
    // capture with temporal noise disabled must be identical.
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
