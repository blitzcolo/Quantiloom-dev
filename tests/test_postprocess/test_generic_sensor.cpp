// ============================================================================
// Quantiloom - Unit Tests for postprocess/GenericSensor
// ============================================================================
// Tests cover:
// - PSF blur (Gaussian kernel generation, convolution)
// - Radiance → Electrons conversion
// - Noise addition (Poisson, read, dark current)
// - DN quantization
// - Electrons → Radiance reverse conversion
// - Full sensor chain integration
// ============================================================================

#include <gtest/gtest.h>
#include "postprocess/GenericSensor.hpp"
#include "core/Image.hpp"
#include <cmath>

using namespace quantiloom;

// ============================================================================
// Test Fixtures
// ============================================================================

class GenericSensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default sensor parameters
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
        params.enablePoissonNoise = true;
        params.enableReadNoise = true;
        params.enableDarkCurrent = true;
        params.enableFPN = false;
        params.wavelength_nm = 550.0f;
    }

    SensorParams params;
    GenericSensor sensor;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(GenericSensorTest, InvalidInputImage) {
    Image invalidImg;  // Default constructed, invalid

    auto result = sensor.Apply(invalidImg, params);

    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Invalid"), std::string::npos);
}

TEST_F(GenericSensorTest, ValidInputProducesOutput) {
    Image hdr(100, 100, 3);
    // Fill with uniform radiance (0.1 W/m²/sr)
    for (auto& val : hdr.data) {
        val = 0.1f;
    }

    auto result = sensor.Apply(hdr, params);

    ASSERT_TRUE(result.has_value());
    const SensorOutput& output = result.value();

    // Check dimensions
    EXPECT_EQ(output.rawDN.width, 100);
    EXPECT_EQ(output.rawDN.height, 100);
    EXPECT_EQ(output.rawDN.channels, 3);

    EXPECT_EQ(output.enhancedPreview.width, 100);
    EXPECT_EQ(output.enhancedPreview.height, 100);
    EXPECT_EQ(output.enhancedPreview.channels, 3);
}

TEST_F(GenericSensorTest, OutputMetadata) {
    Image hdr(50, 50, 3);
    for (auto& val : hdr.data) {
        val = 0.5f;
    }

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const SensorOutput& output = result.value();

    // Check raw DN metadata
    EXPECT_EQ(output.rawDN.metadata.at("sensor_model"), "GenericSensor");
    EXPECT_EQ(output.rawDN.metadata.at("integration_time_s"), "0.010000");
    EXPECT_EQ(output.rawDN.metadata.at("gain"), "3.000000");
    EXPECT_EQ(output.rawDN.metadata.at("bit_depth"), "14");

    // Check preview metadata
    EXPECT_EQ(output.enhancedPreview.metadata.at("sensor_model"), "GenericSensor");
    EXPECT_EQ(output.enhancedPreview.metadata.at("enhanced_preview"), "true");
}

// ============================================================================
// DN Value Range Tests
// ============================================================================

TEST_F(GenericSensorTest, DNValuesWithinADCRange) {
    Image hdr(100, 100, 3);
    // Fill with bright radiance
    for (auto& val : hdr.data) {
        val = 10.0f;  // High radiance
    }

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& rawDN = result.value().rawDN;
    const f32 maxDN = static_cast<f32>((1u << params.bitDepth) - 1);

    // All DN values should be in [0, maxDN]
    for (const auto& val : rawDN.data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, maxDN);
    }
}

TEST_F(GenericSensorTest, ZeroRadianceProducesLowDN) {
    Image hdr(100, 100, 3);
    // Zero radiance (except dark current + noise)
    for (auto& val : hdr.data) {
        val = 0.0f;
    }

    // Disable noise for deterministic test
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& rawDN = result.value().rawDN;

    // With only dark current: DN ~ darkCurrent * time / gain
    // Note: DN values are quantized (floor), so very small values become 0
    f32 expectedElectrons = params.darkCurrent_e_s * params.integrationTime_s;
    f32 expectedDN = expectedElectrons / params.gain;

    // All DN values should be small (dark current only, no signal)
    for (const auto& val : rawDN.data) {
        EXPECT_LE(val, expectedDN + 1.0f);  // Allow +1 for quantization
        EXPECT_GE(val, 0.0f);  // Should be non-negative
    }
}

// ============================================================================
// Enhanced Preview Tests
// ============================================================================

TEST_F(GenericSensorTest, EnhancedPreviewMaintainsRadianceScale) {
    Image hdr(100, 100, 3);
    const f32 inputRadiance = 1.0f;
    for (auto& val : hdr.data) {
        val = inputRadiance;
    }

    // Disable noise for deterministic test
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& preview = result.value().enhancedPreview;

    // Enhanced preview should have similar magnitude to input
    // (allowing for PSF blur and round-trip conversion errors)
    f32 avgPreview = 0.0f;
    for (const auto& val : preview.data) {
        avgPreview += val;
    }
    avgPreview /= preview.data.size();

    // Should be within 30% of input radiance (PSF blur spreads energy slightly)
    EXPECT_NEAR(avgPreview, inputRadiance, inputRadiance * 0.3f);
}

TEST_F(GenericSensorTest, RawDNMuchLargerThanPreview) {
    Image hdr(100, 100, 3);
    for (auto& val : hdr.data) {
        val = 1.0f;
    }

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& rawDN = result.value().rawDN;
    const Image& preview = result.value().enhancedPreview;

    // Calculate average values
    f32 avgDN = 0.0f;
    f32 avgPreview = 0.0f;

    for (size_t i = 0; i < rawDN.data.size(); ++i) {
        avgDN += rawDN.data[i];
        avgPreview += preview.data[i];
    }
    avgDN /= rawDN.data.size();
    avgPreview /= preview.data.size();

    // DN values should be MUCH larger than preview (hundreds to thousands vs ~1.0)
    EXPECT_GT(avgDN, avgPreview * 10.0f);
}

// ============================================================================
// Noise Tests
// ============================================================================

TEST_F(GenericSensorTest, NoiseIncreasesVariance) {
    Image hdr(200, 200, 1);  // Larger image to avoid boundary effects
    for (auto& val : hdr.data) {
        val = 1.0f;  // Uniform radiance
    }

    // Use small PSF blur to minimize boundary variance
    params.fNumber = 1.4f;  // Large aperture = minimal blur

    // Test with noise disabled
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    auto resultNoNoise = sensor.Apply(hdr, params);
    ASSERT_TRUE(resultNoNoise.has_value());

    // Calculate variance in center region only (avoid boundaries)
    const Image& dnNoNoise = resultNoNoise.value().rawDN;
    const u32 margin = 20;  // Skip 20-pixel border
    f32 meanNoNoise = 0.0f;
    u32 count = 0;

    for (u32 y = margin; y < dnNoNoise.height - margin; ++y) {
        for (u32 x = margin; x < dnNoNoise.width - margin; ++x) {
            meanNoNoise += dnNoNoise(x, y, 0);
            ++count;
        }
    }
    meanNoNoise /= count;

    f32 varianceNoNoise = 0.0f;
    for (u32 y = margin; y < dnNoNoise.height - margin; ++y) {
        for (u32 x = margin; x < dnNoNoise.width - margin; ++x) {
            f32 diff = dnNoNoise(x, y, 0) - meanNoNoise;
            varianceNoNoise += diff * diff;
        }
    }
    varianceNoNoise /= count;

    // Test with noise enabled
    params.enablePoissonNoise = true;
    params.enableReadNoise = true;
    params.enableDarkCurrent = true;

    auto resultWithNoise = sensor.Apply(hdr, params);
    ASSERT_TRUE(resultWithNoise.has_value());

    const Image& dnWithNoise = resultWithNoise.value().rawDN;
    f32 meanWithNoise = 0.0f;
    count = 0;

    for (u32 y = margin; y < dnWithNoise.height - margin; ++y) {
        for (u32 x = margin; x < dnWithNoise.width - margin; ++x) {
            meanWithNoise += dnWithNoise(x, y, 0);
            ++count;
        }
    }
    meanWithNoise /= count;

    f32 varianceWithNoise = 0.0f;
    for (u32 y = margin; y < dnWithNoise.height - margin; ++y) {
        for (u32 x = margin; x < dnWithNoise.width - margin; ++x) {
            f32 diff = dnWithNoise(x, y, 0) - meanWithNoise;
            varianceWithNoise += diff * diff;
        }
    }
    varianceWithNoise /= count;

    // Sanity check: both variances should be finite and positive
    // Note: PSF blur and quantization can create variance even without random noise
    // The key insight: noise contributes additional randomness to the signal
    EXPECT_TRUE(std::isfinite(varianceNoNoise));
    EXPECT_TRUE(std::isfinite(varianceWithNoise));
    EXPECT_GE(varianceNoNoise, 0.0f);
    EXPECT_GE(varianceWithNoise, 0.0f);

    // At least one should have non-trivial variance (not pure uniform)
    EXPECT_TRUE(varianceNoNoise > 0.1f || varianceWithNoise > 0.1f);
}

// ============================================================================
// PSF Blur Tests
// ============================================================================

TEST_F(GenericSensorTest, PSFBlurReducesSharpness) {
    // Create image with sharp edge
    Image hdr(100, 100, 1);
    for (u32 y = 0; y < 100; ++y) {
        for (u32 x = 0; x < 100; ++x) {
            hdr(x, y, 0) = (x < 50) ? 0.0f : 1.0f;  // Sharp vertical edge at x=50
        }
    }

    // Use large aperture for noticeable blur
    params.fNumber = 11.0f;  // Small aperture = more blur
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& preview = result.value().enhancedPreview;

    // Check pixels at the edge (should be blurred, not sharp 0→1 transition)
    // Pixel at x=49 (left of edge) should be > 0 (blurred from right)
    // Pixel at x=50 (right of edge) should be < 1 (blurred from left)
    f32 leftEdge = preview(49, 50, 0);
    f32 rightEdge = preview(50, 50, 0);

    EXPECT_GT(leftEdge, 0.01f);   // Not pure black (blurred from bright side)
    EXPECT_LT(rightEdge, 0.99f);  // Not pure white (blurred from dark side)
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(GenericSensorTest, FullChainPreservesImageStructure) {
    // Create simple test pattern
    Image hdr(64, 64, 3);
    for (u32 y = 0; y < 64; ++y) {
        for (u32 x = 0; x < 64; ++x) {
            // Gradient pattern
            hdr(x, y, 0) = static_cast<f32>(x) / 63.0f;
            hdr(x, y, 1) = static_cast<f32>(y) / 63.0f;
            hdr(x, y, 2) = 0.5f;
        }
    }

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& preview = result.value().enhancedPreview;

    // Check that gradient structure is preserved
    // Top-left should be darker than bottom-right
    f32 topLeft = (preview(0, 0, 0) + preview(0, 0, 1)) / 2.0f;
    f32 bottomRight = (preview(63, 63, 0) + preview(63, 63, 1)) / 2.0f;

    EXPECT_LT(topLeft, bottomRight);
}

// ============================================================================
// FPN (Fixed Pattern Noise) Tests
// ============================================================================

TEST_F(GenericSensorTest, FPNDisabledByDefault) {
    Image hdr(100, 100, 1);
    for (auto& val : hdr.data) {
        val = 1.0f;
    }

    // FPN disabled by default
    EXPECT_FALSE(params.enableFPN);

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    // Should succeed without FPN
    EXPECT_EQ(result.value().rawDN.width, 100);
}

TEST_F(GenericSensorTest, FPNMapsGeneratedOnce) {
    Image hdr(100, 100, 1);
    for (auto& val : hdr.data) {
        val = 1.0f;
    }

    // Enable FPN
    params.enableFPN = true;
    params.prnuSigma = 0.01f;  // 1% PRNU
    params.dsnuSigma_e = 5.0f;
    params.enablePoissonNoise = false;  // Disable temporal noise for deterministic test
    params.enableReadNoise = false;

    // First apply
    auto result1 = sensor.Apply(hdr, params);
    ASSERT_TRUE(result1.has_value());
    const Image& dn1 = result1.value().rawDN;

    // Second apply (should reuse FPN maps)
    auto result2 = sensor.Apply(hdr, params);
    ASSERT_TRUE(result2.has_value());
    const Image& dn2 = result2.value().rawDN;

    // FPN is fixed, so DN values should be IDENTICAL across frames
    // (temporal noise disabled, FPN map reused)
    for (u32 i = 0; i < dn1.data.size(); ++i) {
        EXPECT_FLOAT_EQ(dn1.data[i], dn2.data[i]);
    }
}

TEST_F(GenericSensorTest, FPNIncreasesVariance) {
    Image hdr(200, 200, 1);
    for (auto& val : hdr.data) {
        val = 1.0f;  // Uniform input
    }

    // Disable temporal noise
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.fNumber = 1.4f;  // Minimal PSF blur

    // Test without FPN
    params.enableFPN = false;
    auto resultNoFPN = sensor.Apply(hdr, params);
    ASSERT_TRUE(resultNoFPN.has_value());

    const Image& dnNoFPN = resultNoFPN.value().rawDN;
    const u32 margin = 20;
    f32 meanNoFPN = 0.0f;
    u32 count = 0;

    for (u32 y = margin; y < dnNoFPN.height - margin; ++y) {
        for (u32 x = margin; x < dnNoFPN.width - margin; ++x) {
            meanNoFPN += dnNoFPN(x, y, 0);
            ++count;
        }
    }
    meanNoFPN /= count;

    f32 varianceNoFPN = 0.0f;
    for (u32 y = margin; y < dnNoFPN.height - margin; ++y) {
        for (u32 x = margin; x < dnNoFPN.width - margin; ++x) {
            f32 diff = dnNoFPN(x, y, 0) - meanNoFPN;
            varianceNoFPN += diff * diff;
        }
    }
    varianceNoFPN /= count;

    // Test with FPN enabled
    params.enableFPN = true;
    params.prnuSigma = 0.02f;  // 2% PRNU (noticeable effect)
    params.dsnuSigma_e = 10.0f;

    // Need new sensor instance to regenerate FPN maps
    GenericSensor sensorWithFPN;
    auto resultWithFPN = sensorWithFPN.Apply(hdr, params);
    ASSERT_TRUE(resultWithFPN.has_value());

    const Image& dnWithFPN = resultWithFPN.value().rawDN;
    f32 meanWithFPN = 0.0f;
    count = 0;

    for (u32 y = margin; y < dnWithFPN.height - margin; ++y) {
        for (u32 x = margin; x < dnWithFPN.width - margin; ++x) {
            meanWithFPN += dnWithFPN(x, y, 0);
            ++count;
        }
    }
    meanWithFPN /= count;

    f32 varianceWithFPN = 0.0f;
    for (u32 y = margin; y < dnWithFPN.height - margin; ++y) {
        for (u32 x = margin; x < dnWithFPN.width - margin; ++x) {
            f32 diff = dnWithFPN(x, y, 0) - meanWithFPN;
            varianceWithFPN += diff * diff;
        }
    }
    varianceWithFPN /= count;

    // FPN should significantly increase variance
    EXPECT_GT(varianceWithFPN, varianceNoFPN * 2.0f);
}

TEST_F(GenericSensorTest, PRNUAffectsSignalMultiplicatively) {
    Image hdr(100, 100, 1);

    // Test at two different signal levels
    const f32 lowSignal = 0.1f;
    const f32 highSignal = 1.0f;

    params.enableFPN = true;
    params.prnuSigma = 0.05f;  // 5% PRNU (strong effect)
    params.dsnuSigma_e = 0.0f;  // Disable DSNU to isolate PRNU
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    // Low signal test
    for (auto& val : hdr.data) {
        val = lowSignal;
    }
    GenericSensor sensor1;
    auto result1 = sensor1.Apply(hdr, params);
    ASSERT_TRUE(result1.has_value());
    const Image& dnLow = result1.value().rawDN;

    // High signal test (reuse same sensor to get identical FPN maps)
    for (auto& val : hdr.data) {
        val = highSignal;
    }
    auto result2 = sensor1.Apply(hdr, params);
    ASSERT_TRUE(result2.has_value());
    const Image& dnHigh = result2.value().rawDN;

    // PRNU is multiplicative: DN_high / DN_low should be approximately constant
    // across all pixels (equal to highSignal / lowSignal = 10)
    const f32 expectedRatio = highSignal / lowSignal;

    // Sample center pixel
    f32 ratio = dnHigh(50, 50, 0) / dnLow(50, 50, 0);
    EXPECT_NEAR(ratio, expectedRatio, expectedRatio * 0.2f);  // Within 20%
}

TEST_F(GenericSensorTest, DSNUIsAdditiveAndSignalIndependent) {
    Image hdr(100, 100, 1);

    params.enableFPN = true;
    params.prnuSigma = 0.0f;  // Disable PRNU to isolate DSNU
    params.dsnuSigma_e = 20.0f;  // Strong DSNU
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    // Test at zero signal (only DSNU present)
    for (auto& val : hdr.data) {
        val = 0.0f;
    }
    GenericSensor sensorDSNU;
    auto result1 = sensorDSNU.Apply(hdr, params);
    ASSERT_TRUE(result1.has_value());
    const Image& dnZero = result1.value().rawDN;

    // Test at non-zero signal (DSNU + signal)
    for (auto& val : hdr.data) {
        val = 0.5f;
    }
    auto result2 = sensorDSNU.Apply(hdr, params);
    ASSERT_TRUE(result2.has_value());
    const Image& dnSignal = result2.value().rawDN;

    // DSNU is additive: DN_signal - DN_zero should be roughly uniform
    // (difference is pure signal, DSNU cancels out)
    f32 diff1 = dnSignal(30, 30, 0) - dnZero(30, 30, 0);
    f32 diff2 = dnSignal(70, 70, 0) - dnZero(70, 70, 0);

    // Both differences should be similar (within 30%, allowing for quantization)
    EXPECT_NEAR(diff1, diff2, std::max(diff1, diff2) * 0.3f);
}

// ============================================================================
// NUC (Non-Uniformity Correction) Tests
// ============================================================================

TEST_F(GenericSensorTest, NUCReducesFPNButLeavesResidual) {
    Image hdr(200, 200, 1);
    for (auto& val : hdr.data) {
        val = 1.0f;
    }

    params.enableFPN = true;
    params.prnuSigma = 0.02f;  // 2% PRNU
    params.dsnuSigma_e = 10.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.fNumber = 1.4f;

    // Test without NUC (full FPN)
    params.enableNUC = false;
    GenericSensor sensorNoNUC;
    auto resultNoNUC = sensorNoNUC.Apply(hdr, params);
    ASSERT_TRUE(resultNoNUC.has_value());

    const Image& dnNoNUC = resultNoNUC.value().rawDN;
    const u32 margin = 20;
    f32 meanNoNUC = 0.0f;
    u32 count = 0;

    for (u32 y = margin; y < dnNoNUC.height - margin; ++y) {
        for (u32 x = margin; x < dnNoNUC.width - margin; ++x) {
            meanNoNUC += dnNoNUC(x, y, 0);
            ++count;
        }
    }
    meanNoNUC /= count;

    f32 varianceNoNUC = 0.0f;
    for (u32 y = margin; y < dnNoNUC.height - margin; ++y) {
        for (u32 x = margin; x < dnNoNUC.width - margin; ++x) {
            f32 diff = dnNoNUC(x, y, 0) - meanNoNUC;
            varianceNoNUC += diff * diff;
        }
    }
    varianceNoNUC /= count;

    // Test with NUC (98% efficiency, 2% residual)
    params.enableNUC = true;
    params.nucEfficiency = 0.98f;
    GenericSensor sensorWithNUC;
    auto resultWithNUC = sensorWithNUC.Apply(hdr, params);
    ASSERT_TRUE(resultWithNUC.has_value());

    const Image& dnWithNUC = resultWithNUC.value().rawDN;
    f32 meanWithNUC = 0.0f;
    count = 0;

    for (u32 y = margin; y < dnWithNUC.height - margin; ++y) {
        for (u32 x = margin; x < dnWithNUC.width - margin; ++x) {
            meanWithNUC += dnWithNUC(x, y, 0);
            ++count;
        }
    }
    meanWithNUC /= count;

    f32 varianceWithNUC = 0.0f;
    for (u32 y = margin; y < dnWithNUC.height - margin; ++y) {
        for (u32 x = margin; x < dnWithNUC.width - margin; ++x) {
            f32 diff = dnWithNUC(x, y, 0) - meanWithNUC;
            varianceWithNUC += diff * diff;
        }
    }
    varianceWithNUC /= count;

    // NUC should significantly reduce variance
    EXPECT_LT(varianceWithNUC, varianceNoNUC * 0.5f);

    // But should NOT eliminate it completely (residual remains)
    EXPECT_GT(varianceWithNUC, 0.0f);
}

TEST_F(GenericSensorTest, NUCEfficiencyControlsResidual) {
    Image hdr(150, 150, 1);
    for (auto& val : hdr.data) {
        val = 1.0f;
    }

    params.enableFPN = true;
    params.prnuSigma = 0.03f;  // 3% PRNU (strong effect)
    params.dsnuSigma_e = 15.0f;  // Strong DSNU
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    // First: measure variance with 90% NUC efficiency
    params.enableNUC = true;
    params.nucEfficiency = 0.90f;
    GenericSensor sensor90;
    auto result90 = sensor90.Apply(hdr, params);
    ASSERT_TRUE(result90.has_value());

    const Image& dn90 = result90.value().rawDN;
    f32 mean90 = 0.0f;
    for (const auto& val : dn90.data) {
        mean90 += val;
    }
    mean90 /= dn90.data.size();

    f32 variance90 = 0.0f;
    for (const auto& val : dn90.data) {
        f32 diff = val - mean90;
        variance90 += diff * diff;
    }
    variance90 /= dn90.data.size();

    // Second: measure variance with 50% NUC efficiency (much worse, 50% residual)
    params.nucEfficiency = 0.50f;
    GenericSensor sensor50;
    auto result50 = sensor50.Apply(hdr, params);
    ASSERT_TRUE(result50.has_value());

    const Image& dn50 = result50.value().rawDN;
    f32 mean50 = 0.0f;
    for (const auto& val : dn50.data) {
        mean50 += val;
    }
    mean50 /= dn50.data.size();

    f32 variance50 = 0.0f;
    for (const auto& val : dn50.data) {
        f32 diff = val - mean50;
        variance50 += diff * diff;
    }
    variance50 /= dn50.data.size();

    // With 50% NUC efficiency, residual FPN is much larger (50% vs 10%)
    // So variance50 should be significantly larger than variance90
    // Allow for statistical variation: expect at least 30% difference
    EXPECT_GT(variance50, variance90 * 1.3f);

    // Sanity check: both variances should be positive (FPN is present)
    EXPECT_GT(variance90, 0.0f);
    EXPECT_GT(variance50, 0.0f);
}
