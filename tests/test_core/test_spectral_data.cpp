// ============================================================================
// Quantiloom - Unit Tests for core/SpectralData.hpp
// ============================================================================
// Tests cover:
// - SpectralCurve construction and evaluation
// - Linear interpolation accuracy
// - Edge case handling (out of range, empty curves)
// - SpectralCurveGPU conversion and memory layout
// - Downsampling algorithms
// ============================================================================

#include <gtest/gtest.h>
#include "core/SpectralData.hpp"
#include <cmath>

using namespace quantiloom;

// ============================================================================
// SpectralCurve Construction Tests
// ============================================================================

TEST(SpectralDataTest, DefaultConstruction) {
    SpectralCurve curve;

    EXPECT_TRUE(curve.samples.empty());
    EXPECT_FALSE(curve.IsValid());
}

TEST(SpectralDataTest, ArrayConstruction) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve curve(wavelengths, values);

    EXPECT_EQ(curve.samples.size(), 3);
    EXPECT_TRUE(curve.IsValid());

    EXPECT_EQ(curve.samples[0].first, 400.0f);
    EXPECT_EQ(curve.samples[0].second, 0.2f);
    EXPECT_EQ(curve.samples[1].first, 500.0f);
    EXPECT_EQ(curve.samples[1].second, 0.5f);
    EXPECT_EQ(curve.samples[2].first, 600.0f);
    EXPECT_EQ(curve.samples[2].second, 0.8f);
}

TEST(SpectralDataTest, MismatchedArrayConstruction) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f};  // Mismatched size

    SpectralCurve curve(wavelengths, values);

    // Should create empty curve
    EXPECT_TRUE(curve.samples.empty());
    EXPECT_FALSE(curve.IsValid());
}

// ============================================================================
// Evaluation Tests (Linear Interpolation)
// ============================================================================

TEST(SpectralDataTest, EvaluateExactSample) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve curve(wavelengths, values);

    // Exact sample points
    EXPECT_EQ(curve.Evaluate(400.0f), 0.2f);
    EXPECT_EQ(curve.Evaluate(500.0f), 0.5f);
    EXPECT_EQ(curve.Evaluate(600.0f), 0.8f);
}

TEST(SpectralDataTest, EvaluateLinearInterpolation) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve curve(wavelengths, values);

    // Midpoint between 400 and 500
    EXPECT_NEAR(curve.Evaluate(450.0f), 0.35f, 1e-6f);

    // Midpoint between 500 and 600
    EXPECT_NEAR(curve.Evaluate(550.0f), 0.65f, 1e-6f);

    // Quarter point
    EXPECT_NEAR(curve.Evaluate(425.0f), 0.275f, 1e-6f);
}

TEST(SpectralDataTest, EvaluateOutOfRangeBelowMin) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve curve(wavelengths, values);

    // Below minimum - should return edge value
    EXPECT_EQ(curve.Evaluate(300.0f), 0.2f);
    EXPECT_EQ(curve.Evaluate(380.0f), 0.2f);
}

TEST(SpectralDataTest, EvaluateOutOfRangeAboveMax) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve curve(wavelengths, values);

    // Above maximum - should return edge value
    EXPECT_EQ(curve.Evaluate(700.0f), 0.8f);
    EXPECT_EQ(curve.Evaluate(1000.0f), 0.8f);
}

TEST(SpectralDataTest, EvaluateEmptyCurve) {
    SpectralCurve curve;

    EXPECT_EQ(curve.Evaluate(500.0f), 0.0f);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(SpectralDataTest, IsValidEmptyCurve) {
    SpectralCurve curve;
    EXPECT_FALSE(curve.IsValid());
}

TEST(SpectralDataTest, IsValidMonotonicIncreasing) {
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.2f}, {500.0f, 0.5f}, {600.0f, 0.8f}};

    EXPECT_TRUE(curve.IsValid());
}

TEST(SpectralDataTest, IsValidNonMonotonic) {
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.2f}, {600.0f, 0.5f}, {500.0f, 0.8f}};  // Out of order

    EXPECT_FALSE(curve.IsValid());
}

TEST(SpectralDataTest, IsValidDuplicateWavelengths) {
    SpectralCurve curve;
    curve.samples = {{400.0f, 0.2f}, {500.0f, 0.5f}, {500.0f, 0.6f}};  // Duplicate

    EXPECT_FALSE(curve.IsValid());
}

TEST(SpectralDataTest, IsValidSingleSample) {
    SpectralCurve curve;
    curve.samples = {{500.0f, 0.5f}};

    EXPECT_TRUE(curve.IsValid());
}

// ============================================================================
// Wavelength Range Tests
// ============================================================================

TEST(SpectralDataTest, GetWavelengthRange) {
    Vector<f32> wavelengths = {380.0f, 550.0f, 760.0f};
    Vector<f32> values = {0.1f, 0.5f, 0.9f};

    SpectralCurve curve(wavelengths, values);

    auto range = curve.GetWavelengthRange();
    EXPECT_EQ(range.first, 380.0f);
    EXPECT_EQ(range.second, 760.0f);
}

TEST(SpectralDataTest, GetWavelengthRangeEmpty) {
    SpectralCurve curve;

    auto range = curve.GetWavelengthRange();
    EXPECT_EQ(range.first, 0.0f);
    EXPECT_EQ(range.second, 0.0f);
}

// ============================================================================
// SpectralCurveGPU Construction Tests
// ============================================================================

TEST(SpectralDataTest, SpectralCurveGPUDefaultConstruction) {
    SpectralCurveGPU gpu;

    EXPECT_EQ(gpu.numSamples, 0);
    EXPECT_EQ(gpu.startWavelength_nm, 0.0f);
    EXPECT_EQ(gpu.stepSize_nm, 0.0f);

    // Verify all values are zero-initialized
    for (u32 i = 0; i < MAX_SPECTRAL_SAMPLES; ++i) {
        EXPECT_EQ(gpu.values[i], 0.0f);
    }
}

TEST(SpectralDataTest, SpectralCurveGPUSizeVerification) {
    // Verify struct size: 64×4 + 4 + 4 + 4 + 4 = 272 bytes
    // CRITICAL: Must match GPU-side SpectralCurveGPU in common.hlsli
    EXPECT_EQ(sizeof(SpectralCurveGPU), 272);
}

TEST(SpectralDataTest, SpectralCurveGPUFromCPUEmpty) {
    SpectralCurve cpu;
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    EXPECT_EQ(gpu.numSamples, 0);
}

TEST(SpectralDataTest, SpectralCurveGPUFromCPUSmall) {
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f};

    SpectralCurve cpu(wavelengths, values);
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    // FromCPU resamples to uniform grid (default MAX_SPECTRAL_SAMPLES)
    EXPECT_EQ(gpu.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_NEAR(gpu.startWavelength_nm, 400.0f, 1e-5f);

    // Verify wavelength range is preserved
    f32 expectedStep = (600.0f - 400.0f) / static_cast<f32>(MAX_SPECTRAL_SAMPLES - 1);
    EXPECT_NEAR(gpu.stepSize_nm, expectedStep, 1e-5f);

    // Verify first and last values (interpolated to uniform grid)
    EXPECT_NEAR(gpu.values[0], 0.2f, 1e-5f);
    EXPECT_NEAR(gpu.values[MAX_SPECTRAL_SAMPLES - 1], 0.8f, 1e-5f);

    // Verify GetWavelength() helper
    EXPECT_NEAR(gpu.GetWavelength(0), 400.0f, 1e-5f);
    EXPECT_NEAR(gpu.GetWavelength(MAX_SPECTRAL_SAMPLES - 1), 600.0f, 1e-5f);
}

TEST(SpectralDataTest, SpectralCurveGPUFromCPUMaxSize) {
    // Test with exactly MAX_SPECTRAL_SAMPLES
    Vector<f32> wavelengths(MAX_SPECTRAL_SAMPLES);
    Vector<f32> values(MAX_SPECTRAL_SAMPLES);

    f32 lambda_min = 380.0f;
    f32 lambda_max = 380.0f + (MAX_SPECTRAL_SAMPLES - 1) * 10.0f;

    for (u32 i = 0; i < MAX_SPECTRAL_SAMPLES; ++i) {
        wavelengths[i] = 380.0f + i * 10.0f;
        values[i] = 0.1f + i * 0.01f;
    }

    SpectralCurve cpu(wavelengths, values);
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    EXPECT_EQ(gpu.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_NEAR(gpu.startWavelength_nm, lambda_min, 1e-5f);

    // Verify wavelength computation via GetWavelength()
    EXPECT_NEAR(gpu.GetWavelength(0), lambda_min, 1e-5f);
    EXPECT_NEAR(gpu.GetWavelength(MAX_SPECTRAL_SAMPLES - 1), lambda_max, 1e-5f);

    // Verify first and last values
    EXPECT_NEAR(gpu.values[0], values[0], 1e-5f);
    EXPECT_NEAR(gpu.values[MAX_SPECTRAL_SAMPLES - 1], values[MAX_SPECTRAL_SAMPLES - 1], 1e-5f);
}

TEST(SpectralDataTest, SpectralCurveGPUFromCPUDownsampling) {
    // Test with more than MAX_SPECTRAL_SAMPLES (requires downsampling)
    const u32 oversizedCount = MAX_SPECTRAL_SAMPLES + 20;
    Vector<f32> wavelengths(oversizedCount);
    Vector<f32> values(oversizedCount);

    f32 lambda_min = 380.0f;
    f32 lambda_max = 760.0f;

    for (u32 i = 0; i < oversizedCount; ++i) {
        f32 t = static_cast<f32>(i) / static_cast<f32>(oversizedCount - 1);
        wavelengths[i] = lambda_min + t * (lambda_max - lambda_min);
        values[i] = 0.1f + t * 0.8f;  // Linear ramp
    }

    SpectralCurve cpu(wavelengths, values);
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    EXPECT_EQ(gpu.numSamples, MAX_SPECTRAL_SAMPLES);

    // Verify wavelength range is preserved via uniform sampling
    EXPECT_NEAR(gpu.startWavelength_nm, lambda_min, 1e-5f);
    EXPECT_NEAR(gpu.GetWavelength(MAX_SPECTRAL_SAMPLES - 1), lambda_max, 1e-5f);

    // Verify uniform spacing
    f32 expectedStep = (lambda_max - lambda_min) / static_cast<f32>(MAX_SPECTRAL_SAMPLES - 1);
    EXPECT_NEAR(gpu.stepSize_nm, expectedStep, 1e-5f);

    // Verify monotonically increasing (via GetWavelength)
    for (u32 i = 1; i < MAX_SPECTRAL_SAMPLES; ++i) {
        EXPECT_GT(gpu.GetWavelength(i), gpu.GetWavelength(i - 1));
    }
}

TEST(SpectralDataTest, SpectralCurveGPUEvaluate) {
    // Test the Evaluate() function for O(1) lookup
    Vector<f32> wavelengths = {400.0f, 500.0f, 600.0f, 700.0f};
    Vector<f32> values = {0.2f, 0.5f, 0.8f, 0.3f};

    SpectralCurve cpu(wavelengths, values);
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    // Verify evaluation at endpoints (should match original curve edges)
    EXPECT_NEAR(gpu.Evaluate(400.0f), 0.2f, 1e-5f);
    EXPECT_NEAR(gpu.Evaluate(700.0f), 0.3f, 1e-5f);

    // Verify interpolation works
    f32 mid_value = gpu.Evaluate(550.0f);
    EXPECT_GT(mid_value, 0.0f);
    EXPECT_LT(mid_value, 1.0f);

    // Out of range should clamp to edge values
    EXPECT_NEAR(gpu.Evaluate(300.0f), gpu.values[0], 1e-5f);
    EXPECT_NEAR(gpu.Evaluate(800.0f), gpu.values[gpu.numSamples - 1], 1e-5f);
}

TEST(SpectralDataTest, SpectralCurveGPUGetWavelengthRange) {
    Vector<f32> wavelengths = {380.0f, 550.0f, 760.0f};
    Vector<f32> values = {0.1f, 0.5f, 0.9f};

    SpectralCurve cpu(wavelengths, values);
    SpectralCurveGPU gpu = SpectralCurveGPU::FromCPU(cpu);

    auto range = gpu.GetWavelengthRange();
    EXPECT_NEAR(range.first, 380.0f, 1e-5f);
    EXPECT_NEAR(range.second, 760.0f, 1e-5f);
}

TEST(SpectralDataTest, SpectralCurveGPUCreateUniform) {
    // Test the CreateUniform() factory function
    SpectralCurveGPU gpu = SpectralCurveGPU::CreateUniform(380.0f, 780.0f, 41, 0.5f);

    EXPECT_EQ(gpu.numSamples, 41);
    EXPECT_NEAR(gpu.startWavelength_nm, 380.0f, 1e-5f);
    EXPECT_NEAR(gpu.stepSize_nm, 10.0f, 1e-5f);  // (780-380)/(41-1) = 10

    // All values should be 0.5
    for (u32 i = 0; i < gpu.numSamples; ++i) {
        EXPECT_NEAR(gpu.values[i], 0.5f, 1e-5f);
    }

    // Verify wavelength range
    EXPECT_NEAR(gpu.GetWavelength(0), 380.0f, 1e-5f);
    EXPECT_NEAR(gpu.GetWavelength(40), 780.0f, 1e-5f);
}

// ============================================================================
// Physical Realism Tests
// ============================================================================

TEST(SpectralDataTest, VisibleSpectrumReflectance) {
    // Simulate a typical visible spectrum reflectance curve
    Vector<f32> wavelengths = {380.0f, 450.0f, 550.0f, 650.0f, 760.0f};
    Vector<f32> values = {0.05f, 0.15f, 0.60f, 0.20f, 0.10f};  // Green peak

    SpectralCurve curve(wavelengths, values);

    EXPECT_TRUE(curve.IsValid());

    // Peak should be around 550nm (green)
    EXPECT_GT(curve.Evaluate(550.0f), curve.Evaluate(450.0f));
    EXPECT_GT(curve.Evaluate(550.0f), curve.Evaluate(650.0f));
}

TEST(SpectralDataTest, IREmissivityCurve) {
    // Simulate infrared emissivity curve (3-12 µm = 3000-12000 nm)
    Vector<f32> wavelengths = {3000.0f, 5000.0f, 8000.0f, 12000.0f};
    Vector<f32> values = {0.85f, 0.90f, 0.92f, 0.88f};

    SpectralCurve curve(wavelengths, values);

    EXPECT_TRUE(curve.IsValid());

    // All emissivity values should be in [0, 1]
    for (u32 i = 0; i < wavelengths.size(); ++i) {
        f32 emissivity = curve.Evaluate(wavelengths[i]);
        EXPECT_GE(emissivity, 0.0f);
        EXPECT_LE(emissivity, 1.0f);
    }
}
