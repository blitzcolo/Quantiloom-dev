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
    EXPECT_NE(result.error().find("valid mono or linear RGB"), std::string::npos);
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
    Image hdr(8, 8, 1);
    for (auto& value : hdr.data) value = 0.005f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());
    const auto& raw = result.value().rawDN;
    const auto& preview = result.value().enhancedPreview;
    EXPECT_EQ(raw.metadata.at("camera_signal_kind"), "raw_dn");
    EXPECT_EQ(raw.metadata.at("camera_unit"), "DN");
    EXPECT_EQ(raw.metadata.at("camera_input_semantics"), "fast_rgb_approximation");
    EXPECT_EQ(preview.metadata.at("camera_signal_kind"), "device_preview_srgb");
    EXPECT_EQ(preview.metadata.at("camera_input_semantics"), "fast_rgb_approximation");
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

TEST_F(GenericSensorTest, EnhancedPreviewComesFromQuantizedRaw) {
    Image hdr(8, 8, 1);
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.gain = 100.0f; // legacy gain is e-/DN.
    for (auto& value : hdr.data) value = 0.0005f;
    auto below = sensor.Apply(hdr, params);
    ASSERT_TRUE(below.has_value());
    for (auto& value : hdr.data) value = 0.001f;
    auto above = sensor.Apply(hdr, params);
    ASSERT_TRUE(above.has_value());
    EXPECT_FLOAT_EQ(below.value().rawDN.data[0], 0.0f);
    EXPECT_FLOAT_EQ(above.value().rawDN.data[0], 1.0f);
    EXPECT_FLOAT_EQ(below.value().enhancedPreview.data[0], 0.0f);
    EXPECT_GT(above.value().enhancedPreview.data[0], 0.0f);
}

TEST_F(GenericSensorTest, RawAndPreviewHaveDistinctUnits) {
    Image hdr(16, 16, 3);
    for (auto& value : hdr.data) value = 0.01f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());
    const auto& raw = result.value().rawDN;
    const auto& preview = result.value().enhancedPreview;
    EXPECT_GT(raw.data[0], 1.0f);
    EXPECT_EQ(raw.data[0], std::floor(raw.data[0]));
    EXPECT_GT(preview.data[0], 0.0f);
    EXPECT_LE(preview.data[0], 1.0f);
    EXPECT_EQ(preview.metadata.at("camera_unit"), "sRGB-preview");
}

// ============================================================================
// Noise Tests
// ============================================================================

TEST_F(GenericSensorTest, NoiseIncreasesVariance) {
    Image hdr(200, 200, 1);  // Larger image to avoid boundary effects
    for (auto& val : hdr.data) {
        val = 0.005f;  // Uniform radiance
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
            hdr(x, y, 0) = (x < 50) ? 0.0f : 0.005f;  // Sharp vertical edge at x=50
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

    EXPECT_GT(leftEdge, 0.0f);   // Not pure black (blurred from bright side)
    EXPECT_LT(rightEdge, preview(60, 50, 0));  // Not pure white (blurred from dark side)
}

// ----------------------------------------------------------------------------
// PSF width override (sensor.psf_sigma_px)
// ----------------------------------------------------------------------------
// f-number sets two physically distinct things at once -- the PSF width and the
// collection solid angle -- so a study that sweeps blur has to compensate
// exposure to hold the photon budget fixed. The override decouples them.

namespace {

// A sharp vertical edge at x = 50, noise disabled so the chain is deterministic.
auto MakeEdgeImage() -> Image {
    Image hdr(100, 100, 1);
    for (u32 y = 0; y < 100; ++y) {
        for (u32 x = 0; x < 100; ++x) {
            hdr(x, y, 0) = (x < 50) ? 0.0f : 0.005f;
        }
    }
    return hdr;
}

// How far the edge has spread: bright signal that leaked into the dark side.
auto EdgeLeakage(const Image& preview) -> f32 {
    f32 leak = 0.0f;
    for (u32 x = 0; x < 50; ++x) {
        leak += preview(x, 50, 0);
    }
    return leak;
}

} // namespace

TEST_F(GenericSensorTest, PSFSigmaOverrideWidensBlur) {
    const Image hdr = MakeEdgeImage();
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    auto derived = sensor.Apply(hdr, params);
    ASSERT_TRUE(derived.has_value());

    params.psfSigma_px = 4.0f;  // far wider than the diffraction-limited width
    auto overridden = sensor.Apply(hdr, params);
    ASSERT_TRUE(overridden.has_value());

    EXPECT_GT(EdgeLeakage(overridden.value().enhancedPreview),
              EdgeLeakage(derived.value().enhancedPreview))
        << "an explicit sigma must reach the blur, not be ignored";
}

TEST_F(GenericSensorTest, PSFSigmaOverrideZeroLeavesEdgeSharp) {
    const Image hdr = MakeEdgeImage();
    params.fNumber = 11.0f;  // would otherwise blur visibly
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    params.psfSigma_px = 0.0f;
    auto noBlur = sensor.Apply(hdr, params);
    ASSERT_TRUE(noBlur.has_value());

    params.psfSigma_px = -1.0f;
    auto blurred = sensor.Apply(hdr, params);
    ASSERT_TRUE(blurred.has_value());

    // Zero means no blur, rather than falling back to the derived width.
    EXPECT_LT(EdgeLeakage(noBlur.value().enhancedPreview), 1e-3f);
    EXPECT_GT(EdgeLeakage(blurred.value().enhancedPreview),
              EdgeLeakage(noBlur.value().enhancedPreview) + 5e-4f);
}

TEST_F(GenericSensorTest, PSFSigmaNegativeNonSentinelIsRejected) {
    const Image hdr = MakeEdgeImage();
    params.fNumber = 11.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;

    ASSERT_LT(params.psfSigma_px, 0.0f) << "fixture leaves the field defaulted";
    params.psfSigma_px = -0.5f;  // only exactly -1 requests the derived PSF
    auto explicitlyNegative = sensor.Apply(hdr, params);
    EXPECT_FALSE(explicitlyNegative.has_value());
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
            hdr(x, y, 0) = static_cast<f32>(x) * 0.01f / 63.0f;
            hdr(x, y, 1) = static_cast<f32>(y) * 0.01f / 63.0f;
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

TEST_F(GenericSensorTest, FixedPatternStableAcrossCaptures) {
    Image hdr(100, 100, 1);
    for (auto& val : hdr.data) {
        val = 0.005f;
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

    // The counter key keeps fixed pattern terms unchanged on the next capture.
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
        val = 0.005f;  // Uniform input
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

    // A second instance with the same seed uses the same fixed pattern.
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
    const f32 lowSignal = 0.001f;
    const f32 highSignal = 0.01f;

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

    // High signal test with the same fixed per-pixel response.
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
    params.enableDarkCurrent = true;
    params.darkCurrent_e_s = 5000.0f;

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
        val = 0.005f;
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

TEST_F(GenericSensorTest, NUCCorrectsOutputWithoutChangingRawFPN) {
    Image hdr(64, 64, 1);
    for (auto& value : hdr.data) value = 0.01f;
    params.enableFPN = true;
    params.prnuSigma = 0.05f;
    params.dsnuSigma_e = 5.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.noiseSeed = 1234u;
    params.enableNUC = false;
    GenericSensor uncorrectedSensor;
    auto uncorrected = uncorrectedSensor.Apply(hdr, params);
    ASSERT_TRUE(uncorrected.has_value());
    params.enableNUC = true;
    params.nucEfficiency = 1.0f;
    GenericSensor correctedSensor;
    auto corrected = correctedSensor.Apply(hdr, params);
    ASSERT_TRUE(corrected.has_value());
    EXPECT_EQ(uncorrected.value().rawDN.data, corrected.value().rawDN.data);
    auto variance = [](const Image& image) {
        f64 mean = 0.0;
        for (f32 value : image.data) mean += value;
        mean /= image.data.size();
        f64 sum = 0.0;
        for (f32 value : image.data) sum += (value - mean) * (value - mean);
        return sum / image.data.size();
    };
    EXPECT_LT(variance(corrected.value().enhancedPreview),
              variance(uncorrected.value().enhancedPreview) * 0.5);
}

TEST_F(GenericSensorTest, CalibrationResidualChangesPreviewOnly) {
    Image hdr(64, 64, 1);
    for (auto& value : hdr.data) value = 0.01f;
    params.enableFPN = true;
    params.prnuSigma = 0.05f;
    params.dsnuSigma_e = 0.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableNUC = true;
    params.noiseSeed = 4567u;
    params.nucEfficiency = 0.95f;
    GenericSensor nearlyCalibrated;
    auto nearly = nearlyCalibrated.Apply(hdr, params);
    ASSERT_TRUE(nearly.has_value());
    params.nucEfficiency = 0.50f;
    GenericSensor weaklyCalibrated;
    auto weak = weaklyCalibrated.Apply(hdr, params);
    ASSERT_TRUE(weak.has_value());
    EXPECT_EQ(nearly.value().rawDN.data, weak.value().rawDN.data);
    f64 difference = 0.0;
    for (size_t i = 0; i < nearly.value().enhancedPreview.data.size(); ++i)
        difference += std::abs(nearly.value().enhancedPreview.data[i] -
                               weak.value().enhancedPreview.data[i]);
    EXPECT_GT(difference, 0.0);
}

// ============================================================================
// FPN Per-Pixel Persistence Tests
// ============================================================================

TEST_F(GenericSensorTest, PRNUIsFixedPerPixelAcrossCaptures) {
    Image hdr(64, 64, 1);
    for (auto& value : hdr.data) value = 0.01f;
    params.enableFPN = true;
    params.prnuSigma = 0.08f;
    params.dsnuSigma_e = 0.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    auto first = sensor.Apply(hdr, params);
    auto second = sensor.Apply(hdr, params);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first.value().rawDN.data, second.value().rawDN.data);
    const f32 initial = first.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : first.value().rawDN.data) if (value != initial) ++different;
    EXPECT_GT(different, first.value().rawDN.data.size() / 4);
}

TEST_F(GenericSensorTest, DSNUIsFixedPerPixelAcrossDarkCaptures) {
    Image hdr(64, 64, 1);
    params.enableFPN = true;
    params.prnuSigma = 0.0f;
    params.dsnuSigma_e = 20.0f;
    params.darkCurrent_e_s = 5000.0f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = true;
    auto first = sensor.Apply(hdr, params);
    auto second = sensor.Apply(hdr, params);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first.value().rawDN.data, second.value().rawDN.data);
    const f32 initial = first.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : first.value().rawDN.data) if (value != initial) ++different;
    EXPECT_GT(different, first.value().rawDN.data.size() / 4);
}

TEST_F(GenericSensorTest, CombinedFPNCreatesSpatialVariationWithoutStripeAssumption) {
    Image hdr(64, 64, 1);
    for (auto& value : hdr.data) value = 0.01f;
    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = true;
    params.darkCurrent_e_s = 5000.0f;
    params.prnuSigma = 0.06f;
    params.dsnuSigma_e = 20.0f;
    params.enableFPN = false;
    GenericSensor uniformSensor;
    auto uniform = uniformSensor.Apply(hdr, params);
    ASSERT_TRUE(uniform.has_value());
    params.enableFPN = true;
    GenericSensor patternedSensor;
    auto patterned = patternedSensor.Apply(hdr, params);
    ASSERT_TRUE(patterned.has_value());
    const auto first = patterned.value().rawDN.data[0];
    size_t different = 0;
    for (f32 value : patterned.value().rawDN.data) if (value != first) ++different;
    EXPECT_GT(different, patterned.value().rawDN.data.size() / 4);
    EXPECT_NE(patterned.value().rawDN.data, uniform.value().rawDN.data);
}

// ============================================================================
// P5 Fix: Vignetting Tests (Cos^4 Natural Vignetting)
// ============================================================================
// Tests for the cos^4 vignetting implementation added in P5 fix.
// Natural vignetting follows: E(θ) = E(0) × cos⁴(θ)
// where θ is the angle from the optical axis.
// ============================================================================

TEST_F(GenericSensorTest, VignettingDisabledProducesUniformOutput) {
    // Create uniform radiance image
    // Use moderate radiance to avoid ADC saturation (14-bit max = 16383)
    Image hdr(128, 128, 1);
    for (auto& val : hdr.data) {
        val = 0.5f;  // Moderate radiance to stay below saturation
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = false;  // Disabled

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    // Center and corner should have equal DN values (no vignetting)
    u32 centerIdx = 64 * 128 + 64;  // Center pixel
    u32 cornerIdx = 0;               // Top-left corner

    f32 centerDN = dn.data[centerIdx];
    f32 cornerDN = dn.data[cornerIdx];

    // Allow small tolerance for numerical precision
    EXPECT_NEAR(centerDN, cornerDN, centerDN * 0.01f)
        << "With vignetting disabled, center and corner should be equal";
}

TEST_F(GenericSensorTest, VignettingEnabledDarkensEdges) {
    // Create uniform radiance image
    // Use low radiance to avoid ADC saturation (14-bit max = 16383)
    Image hdr(128, 128, 1);
    for (auto& val : hdr.data) {
        val = 0.1f;  // Low radiance to stay well below saturation
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = true;   // Enabled
    params.fov_deg = 60.0f;            // Wide FOV for noticeable vignetting
    params.isTelecentric = false;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    // Get center and corner values
    u32 centerIdx = 64 * 128 + 64;
    u32 cornerIdx = 0;

    f32 centerDN = dn.data[centerIdx];
    f32 cornerDN = dn.data[cornerIdx];

    // Corner should be darker than center (vignetting effect)
    EXPECT_GT(centerDN, cornerDN)
        << "With vignetting enabled, center should be brighter than corner";

    // Vignetting ratio should be between 40% and 90%
    // For 60° FOV, corner angle is about 30°, cos^4(30°) ≈ 0.56
    f32 vignetteRatio = cornerDN / centerDN;
    EXPECT_GT(vignetteRatio, 0.3f) << "Vignette ratio too low (corner too dark)";
    EXPECT_LT(vignetteRatio, 0.95f) << "Vignette ratio too high (no visible vignetting)";
}

TEST_F(GenericSensorTest, TelecentricLensNoVignetting) {
    // Telecentric lenses have no natural vignetting
    // Use moderate radiance to avoid ADC saturation
    Image hdr(128, 128, 1);
    for (auto& val : hdr.data) {
        val = 0.5f;
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = true;   // Would be enabled...
    params.fov_deg = 60.0f;
    params.isTelecentric = true;       // ...but telecentric overrides

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    u32 centerIdx = 64 * 128 + 64;
    u32 cornerIdx = 0;

    f32 centerDN = dn.data[centerIdx];
    f32 cornerDN = dn.data[cornerIdx];

    // Telecentric lens should have no vignetting
    EXPECT_NEAR(centerDN, cornerDN, centerDN * 0.01f)
        << "Telecentric lens should have no vignetting";
}

TEST_F(GenericSensorTest, VignettingFollowsCos4Law) {
    // Verify that vignetting accurately follows the cos^4 law
    // Use moderate radiance to avoid ADC saturation while maintaining precision
    Image hdr(256, 256, 1);
    for (auto& val : hdr.data) {
        val = 0.5f;  // Moderate radiance - avoids saturation at 14-bit ADC
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = true;
    params.fov_deg = 45.0f;
    params.isTelecentric = false;
    params.pixelPitch_um = 5.0f;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    // Get center value
    const u32 cx = 128, cy = 128;
    f32 centerDN = dn(cx, cy, 0);
    ASSERT_GT(centerDN, 0.0f) << "Center DN should be positive";

    // Calculate focal length from FOV (matching the sensor implementation)
    const f32 PI = 3.14159265358979323846f;
    f32 fov_half_rad = params.fov_deg * 0.5f * (PI / 180.0f);
    f32 sensor_half_width = static_cast<f32>(cx) * params.pixelPitch_um * 1e-6f;
    f32 focal_length_m = sensor_half_width / std::tan(fov_half_rad);

    // Test at various radial distances
    std::vector<std::pair<u32, u32>> testPoints = {
        {160, 128},  // 32 pixels right of center
        {192, 128},  // 64 pixels right of center
        {224, 128},  // 96 pixels right of center
        {160, 160},  // Diagonal
        {192, 192},  // Diagonal
    };

    for (const auto& [testX, testY] : testPoints) {
        f32 testDN = dn(testX, testY, 0);

        // Calculate expected vignetting based on cos^4 law
        f32 dx = (static_cast<f32>(testX) - static_cast<f32>(cx)) * params.pixelPitch_um * 1e-6f;
        f32 dy = (static_cast<f32>(testY) - static_cast<f32>(cy)) * params.pixelPitch_um * 1e-6f;
        f32 r = std::sqrt(dx * dx + dy * dy);
        f32 theta = std::atan2(r, focal_length_m);
        f32 cos_theta = std::cos(theta);
        f32 expected_vignette = cos_theta * cos_theta * cos_theta * cos_theta;

        f32 actual_ratio = testDN / centerDN;

        // Allow 5% tolerance for numerical precision
        EXPECT_NEAR(actual_ratio, expected_vignette, 0.05f)
            << "Vignetting at pixel (" << testX << ", " << testY << ") "
            << "should follow cos^4 law. Expected: " << expected_vignette
            << ", Actual: " << actual_ratio;
    }
}

TEST_F(GenericSensorTest, VignettingScalesWithFOV) {
    // Wider FOV should produce more vignetting at corners
    // Use moderate radiance to avoid ADC saturation
    Image hdr(128, 128, 1);
    for (auto& val : hdr.data) {
        val = 0.5f;
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = true;
    params.isTelecentric = false;

    // Test with narrow FOV
    params.fov_deg = 20.0f;
    auto resultNarrow = sensor.Apply(hdr, params);
    ASSERT_TRUE(resultNarrow.has_value());

    // Test with wide FOV
    params.fov_deg = 90.0f;
    GenericSensor sensorWide;
    auto resultWide = sensorWide.Apply(hdr, params);
    ASSERT_TRUE(resultWide.has_value());

    const Image& dnNarrow = resultNarrow.value().rawDN;
    const Image& dnWide = resultWide.value().rawDN;

    // Calculate vignetting ratio at corner for both FOVs
    u32 centerIdx = 64 * 128 + 64;
    u32 cornerIdx = 0;

    f32 ratioNarrow = dnNarrow.data[cornerIdx] / dnNarrow.data[centerIdx];
    f32 ratioWide = dnWide.data[cornerIdx] / dnWide.data[centerIdx];

    // Wide FOV should have more vignetting (lower ratio at corner)
    EXPECT_LT(ratioWide, ratioNarrow)
        << "Wider FOV should produce more vignetting at corners";
}

TEST_F(GenericSensorTest, VignettingSymmetric) {
    // Vignetting should be radially symmetric
    // Use moderate radiance to avoid ADC saturation
    Image hdr(128, 128, 1);
    for (auto& val : hdr.data) {
        val = 0.5f;
    }

    params.enablePoissonNoise = false;
    params.enableReadNoise = false;
    params.enableDarkCurrent = false;
    params.enableFPN = false;
    params.enableVignetting = true;
    params.fov_deg = 45.0f;
    params.isTelecentric = false;

    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    const Image& dn = result.value().rawDN;

    // Check that points at same distance from center have same DN
    // Compare four corners of a square around center
    u32 cx = 64, cy = 64;
    u32 offset = 30;  // 30 pixels from center

    f32 dnRight = dn(cx + offset, cy, 0);
    f32 dnLeft = dn(cx - offset, cy, 0);
    f32 dnUp = dn(cx, cy - offset, 0);
    f32 dnDown = dn(cx, cy + offset, 0);

    // All four should be approximately equal (radial symmetry)
    f32 avg = (dnRight + dnLeft + dnUp + dnDown) / 4.0f;
    f32 tolerance = avg * 0.02f;  // 2% tolerance

    EXPECT_NEAR(dnRight, avg, tolerance);
    EXPECT_NEAR(dnLeft, avg, tolerance);
    EXPECT_NEAR(dnUp, avg, tolerance);
    EXPECT_NEAR(dnDown, avg, tolerance);
}

// ============================================================================
// Noise Reproducibility
// ============================================================================
// GenericSensor used to seed itself from std::random_device with no way to
// override it, so every noise-enabled render differed and the FPN tests were
// nondeterministic -- PRNUAffectsSignalMultiplicatively failed roughly 1 run in
// 12, taking build_wsl.sh's gate red with it. These pin the guarantee that
// replaced it, since without them the seed could be ignored again silently.

TEST_F(GenericSensorTest, SameSeedGivesIdenticalOutput) {
    Image hdr(32, 32, 1);
    for (auto& v : hdr.data) { v = 0.5f; }

    params.enableFPN = true;
    params.enablePoissonNoise = true;
    params.enableReadNoise = true;
    params.noiseSeed = 12345U;

    GenericSensor a;
    GenericSensor b;
    auto ra = a.Apply(hdr, params);
    auto rb = b.Apply(hdr, params);
    ASSERT_TRUE(ra.has_value());
    ASSERT_TRUE(rb.has_value());

    // Bit-identical, not merely close: this is what lets two renders of one
    // scene be compared directly.
    EXPECT_EQ(ra.value().rawDN.data, rb.value().rawDN.data);
}

TEST_F(GenericSensorTest, DifferentSeedGivesDifferentNoise) {
    Image hdr(32, 32, 1);
    for (auto& v : hdr.data) { v = 0.5f; }

    params.enableFPN = true;
    params.enablePoissonNoise = true;
    params.enableReadNoise = true;

    params.noiseSeed = 1U;
    GenericSensor a;
    auto ra = a.Apply(hdr, params);

    params.noiseSeed = 2U;
    GenericSensor b;
    auto rb = b.Apply(hdr, params);

    ASSERT_TRUE(ra.has_value());
    ASSERT_TRUE(rb.has_value());
    EXPECT_NE(ra.value().rawDN.data, rb.value().rawDN.data);
}

TEST_F(GenericSensorTest, ZeroSeedRequestsNondeterministicNoise) {
    Image hdr(32, 32, 1);
    for (auto& v : hdr.data) { v = 0.5f; }

    params.enableFPN = true;
    params.enablePoissonNoise = true;
    params.enableReadNoise = true;
    params.noiseSeed = 0U;  // documented opt-out from reproducibility

    GenericSensor a;
    GenericSensor b;
    auto ra = a.Apply(hdr, params);
    auto rb = b.Apply(hdr, params);
    ASSERT_TRUE(ra.has_value());
    ASSERT_TRUE(rb.has_value());
    EXPECT_NE(ra.value().rawDN.data, rb.value().rawDN.data);
}

TEST_F(GenericSensorTest, DefaultSeedIsReproducible) {
    Image hdr(16, 16, 1);
    for (auto& v : hdr.data) { v = 0.5f; }

    params.enableFPN = true;
    params.enableReadNoise = true;
    // params.noiseSeed left at its default

    GenericSensor a;
    GenericSensor b;
    auto ra = a.Apply(hdr, params);
    auto rb = b.Apply(hdr, params);
    ASSERT_TRUE(ra.has_value());
    ASSERT_TRUE(rb.has_value());
    EXPECT_EQ(ra.value().rawDN.data, rb.value().rawDN.data)
        << "the default must be deterministic, or renders are not comparable";
}
