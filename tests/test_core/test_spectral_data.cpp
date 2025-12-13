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

// ============================================================================
// ComplexRefractiveIndex Construction Tests
// ============================================================================

TEST(SpectralDataTest, ComplexRefractiveIndexDefaultConstruction) {
    ComplexRefractiveIndex cri;

    EXPECT_TRUE(cri.wavelengths_nm.empty());
    EXPECT_TRUE(cri.n.empty());
    EXPECT_TRUE(cri.k.empty());
    EXPECT_FALSE(cri.IsValid());
}

TEST(SpectralDataTest, ComplexRefractiveIndexConstruction) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 500.0f, 600.0f};
    cri.n = {0.12f, 0.18f, 0.21f};
    cri.k = {1.80f, 2.50f, 3.10f};

    EXPECT_TRUE(cri.IsValid());
    EXPECT_EQ(cri.wavelengths_nm.size(), 3);
}

TEST(SpectralDataTest, ComplexRefractiveIndexMismatchedSizes) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 500.0f, 600.0f};
    cri.n = {0.12f, 0.18f};  // Mismatched
    cri.k = {1.80f, 2.50f, 3.10f};

    EXPECT_FALSE(cri.IsValid());
}

// ============================================================================
// ComplexRefractiveIndex Evaluation Tests
// ============================================================================

TEST(SpectralDataTest, ComplexRefractiveIndexEvaluateExact) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 500.0f, 600.0f};
    cri.n = {0.12f, 0.18f, 0.21f};
    cri.k = {1.80f, 2.50f, 3.10f};

    // Test exact sample points
    auto [n_400, k_400] = cri.Evaluate(400.0f);
    EXPECT_NEAR(n_400, 0.12f, 1e-6f);
    EXPECT_NEAR(k_400, 1.80f, 1e-6f);

    auto [n_600, k_600] = cri.Evaluate(600.0f);
    EXPECT_NEAR(n_600, 0.21f, 1e-6f);
    EXPECT_NEAR(k_600, 3.10f, 1e-6f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexEvaluateInterpolation) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 600.0f};
    cri.n = {0.10f, 0.20f};
    cri.k = {2.00f, 3.00f};

    // Midpoint interpolation
    auto [n_mid, k_mid] = cri.Evaluate(500.0f);
    EXPECT_NEAR(n_mid, 0.15f, 1e-6f);
    EXPECT_NEAR(k_mid, 2.50f, 1e-6f);

    // Quarter point
    auto [n_q1, k_q1] = cri.Evaluate(450.0f);
    EXPECT_NEAR(n_q1, 0.125f, 1e-6f);
    EXPECT_NEAR(k_q1, 2.25f, 1e-6f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexEvaluateOutOfRange) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {400.0f, 500.0f, 600.0f};
    cri.n = {0.10f, 0.15f, 0.20f};
    cri.k = {2.00f, 2.50f, 3.00f};

    // Below minimum
    auto [n_below, k_below] = cri.Evaluate(300.0f);
    EXPECT_NEAR(n_below, 0.10f, 1e-6f);
    EXPECT_NEAR(k_below, 2.00f, 1e-6f);

    // Above maximum
    auto [n_above, k_above] = cri.Evaluate(800.0f);
    EXPECT_NEAR(n_above, 0.20f, 1e-6f);
    EXPECT_NEAR(k_above, 3.00f, 1e-6f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexEvaluateEmpty) {
    ComplexRefractiveIndex cri;

    auto [n, k] = cri.Evaluate(500.0f);
    EXPECT_NEAR(n, 1.0f, 1e-6f);  // Default: air
    EXPECT_NEAR(k, 0.0f, 1e-6f);
}

// ============================================================================
// ComplexRefractiveIndex Fresnel R0 Tests
// ============================================================================

TEST(SpectralDataTest, ComplexRefractiveIndexFresnelR0Dielectric) {
    // Glass-like dielectric: n=1.5, k≈0
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {550.0f};
    cri.n = {1.5f};
    cri.k = {0.0f};

    // F0 = [(1.5-1)² + 0²] / [(1.5+1)² + 0²] = 0.25 / 6.25 = 0.04
    f32 F0 = cri.FresnelR0(550.0f);
    EXPECT_NEAR(F0, 0.04f, 1e-3f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexFresnelR0Metal) {
    // Gold-like metal: n=0.2, k=3.0 (typical visible spectrum)
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {550.0f};
    cri.n = {0.2f};
    cri.k = {3.0f};

    // F0 = [(0.2-1)² + 3²] / [(0.2+1)² + 3²] = 9.64 / 10.44 ≈ 0.9234
    f32 F0 = cri.FresnelR0(550.0f);
    EXPECT_NEAR(F0, 0.9234f, 0.01f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexFresnelR0Air) {
    // Air: n=1.0, k=0.0 → F0=0
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {550.0f};
    cri.n = {1.0f};
    cri.k = {0.0f};

    f32 F0 = cri.FresnelR0(550.0f);
    EXPECT_NEAR(F0, 0.0f, 1e-6f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGetWavelengthRange) {
    ComplexRefractiveIndex cri;
    cri.wavelengths_nm = {380.0f, 550.0f, 760.0f};
    cri.n = {0.1f, 0.2f, 0.3f};
    cri.k = {1.0f, 2.0f, 3.0f};

    auto [min_wl, max_wl] = cri.GetWavelengthRange();
    EXPECT_NEAR(min_wl, 380.0f, 1e-6f);
    EXPECT_NEAR(max_wl, 760.0f, 1e-6f);
}

// ============================================================================
// ComplexRefractiveIndexGPU Tests
// ============================================================================

TEST(SpectralDataTest, ComplexRefractiveIndexGPUDefaultConstruction) {
    ComplexRefractiveIndexGPU gpu;

    EXPECT_EQ(gpu.numSamples, 0);
    EXPECT_EQ(gpu.startWavelength_nm, 0.0f);
    EXPECT_EQ(gpu.stepSize_nm, 0.0f);

    // All arrays should be zero-initialized
    for (u32 i = 0; i < MAX_SPECTRAL_SAMPLES; ++i) {
        EXPECT_EQ(gpu.n[i], 0.0f);
        EXPECT_EQ(gpu.k[i], 0.0f);
    }
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUSizeVerification) {
    // Verify struct size: 64×4 (n) + 64×4 (k) + 4 + 4 + 4 + 4 = 528 bytes
    // CRITICAL: Must match GPU-side ComplexRefractiveIndexGPU in common.hlsli
    EXPECT_EQ(sizeof(ComplexRefractiveIndexGPU), 528);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUFromCPUEmpty) {
    ComplexRefractiveIndex cpu;
    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    EXPECT_EQ(gpu.numSamples, 0);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUFromCPUBasic) {
    ComplexRefractiveIndex cpu;
    cpu.wavelengths_nm = {400.0f, 500.0f, 600.0f, 700.0f};
    cpu.n = {0.10f, 0.15f, 0.20f, 0.25f};
    cpu.k = {2.00f, 2.50f, 3.00f, 3.50f};

    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    EXPECT_EQ(gpu.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_NEAR(gpu.startWavelength_nm, 400.0f, 1e-5f);

    // Verify step size
    f32 expectedStep = (700.0f - 400.0f) / static_cast<f32>(MAX_SPECTRAL_SAMPLES - 1);
    EXPECT_NEAR(gpu.stepSize_nm, expectedStep, 1e-5f);

    // Verify endpoint values
    EXPECT_NEAR(gpu.n[0], 0.10f, 1e-5f);
    EXPECT_NEAR(gpu.k[0], 2.00f, 1e-5f);
    EXPECT_NEAR(gpu.n[MAX_SPECTRAL_SAMPLES - 1], 0.25f, 1e-5f);
    EXPECT_NEAR(gpu.k[MAX_SPECTRAL_SAMPLES - 1], 3.50f, 1e-5f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUGetWavelength) {
    ComplexRefractiveIndex cpu;
    cpu.wavelengths_nm = {400.0f, 700.0f};
    cpu.n = {0.10f, 0.25f};
    cpu.k = {2.00f, 3.50f};

    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    // First wavelength
    EXPECT_NEAR(gpu.GetWavelength(0), 400.0f, 1e-5f);

    // Last wavelength
    EXPECT_NEAR(gpu.GetWavelength(MAX_SPECTRAL_SAMPLES - 1), 700.0f, 1e-5f);

    // Middle wavelength
    f32 mid = gpu.GetWavelength(MAX_SPECTRAL_SAMPLES / 2);
    EXPECT_GT(mid, 400.0f);
    EXPECT_LT(mid, 700.0f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUEvaluate) {
    ComplexRefractiveIndex cpu;
    cpu.wavelengths_nm = {400.0f, 600.0f};
    cpu.n = {0.10f, 0.20f};
    cpu.k = {2.00f, 3.00f};

    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    // Test endpoints
    auto [n_start, k_start] = gpu.Evaluate(400.0f);
    EXPECT_NEAR(n_start, 0.10f, 1e-5f);
    EXPECT_NEAR(k_start, 2.00f, 1e-5f);

    auto [n_end, k_end] = gpu.Evaluate(600.0f);
    EXPECT_NEAR(n_end, 0.20f, 1e-5f);
    EXPECT_NEAR(k_end, 3.00f, 1e-5f);

    // Test interpolation at midpoint
    auto [n_mid, k_mid] = gpu.Evaluate(500.0f);
    EXPECT_NEAR(n_mid, 0.15f, 0.01f);  // Allow small tolerance due to resampling
    EXPECT_NEAR(k_mid, 2.50f, 0.05f);

    // Test out-of-range clamping
    auto [n_below, k_below] = gpu.Evaluate(300.0f);
    EXPECT_NEAR(n_below, gpu.n[0], 1e-5f);
    EXPECT_NEAR(k_below, gpu.k[0], 1e-5f);

    auto [n_above, k_above] = gpu.Evaluate(800.0f);
    EXPECT_NEAR(n_above, gpu.n[gpu.numSamples - 1], 1e-5f);
    EXPECT_NEAR(k_above, gpu.k[gpu.numSamples - 1], 1e-5f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUFresnelR0) {
    // GPU conversion requires at least 2 data points for stepSize calculation
    ComplexRefractiveIndex cpu;
    cpu.wavelengths_nm = {500.0f, 600.0f};
    cpu.n = {0.2f, 0.2f};  // Same n,k for consistent F0 across range
    cpu.k = {3.0f, 3.0f};

    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    // F0 = [(0.2-1)² + 3²] / [(0.2+1)² + 3²] = 9.64 / 10.44 ≈ 0.9234
    f32 F0 = gpu.FresnelR0(550.0f);
    EXPECT_NEAR(F0, 0.9234f, 0.01f);
}

TEST(SpectralDataTest, ComplexRefractiveIndexGPUUniformSampling) {
    // Test that GPU representation produces uniform wavelength spacing
    ComplexRefractiveIndex cpu;
    cpu.wavelengths_nm = {380.0f, 400.0f, 500.0f, 600.0f, 760.0f};  // Non-uniform
    cpu.n = {0.08f, 0.10f, 0.15f, 0.20f, 0.28f};
    cpu.k = {1.50f, 2.00f, 2.50f, 3.00f, 4.00f};

    ComplexRefractiveIndexGPU gpu = ComplexRefractiveIndexGPU::FromCPU(cpu);

    // Verify uniform spacing (relax tolerance for float accumulation)
    for (u32 i = 1; i < gpu.numSamples; ++i) {
        f32 step = gpu.GetWavelength(i) - gpu.GetWavelength(i - 1);
        EXPECT_NEAR(step, gpu.stepSize_nm, 1e-4f);
    }
}

// ============================================================================
// Physical Material Tests
// ============================================================================

TEST(SpectralDataTest, GoldOpticalConstants) {
    // Typical gold optical constants in visible spectrum
    // Reference: Johnson & Christy (1972)
    ComplexRefractiveIndex gold;
    gold.wavelengths_nm = {500.0f, 550.0f, 600.0f, 650.0f, 700.0f};
    gold.n = {0.93f, 0.40f, 0.18f, 0.14f, 0.13f};
    gold.k = {1.95f, 2.36f, 3.00f, 3.48f, 3.88f};

    EXPECT_TRUE(gold.IsValid());

    // Gold has moderate F0 at 500nm (~0.5), increases to high F0 at 700nm (~0.97)
    // This wavelength-dependent reflectance gives gold its characteristic color
    EXPECT_GT(gold.FresnelR0(500.0f), 0.4f);  // At blue/green: moderate reflectance
    EXPECT_GT(gold.FresnelR0(700.0f), 0.9f);  // At red: high reflectance

    // F0 should increase with wavelength (gold color: absorbs blue, reflects red)
    EXPECT_LT(gold.FresnelR0(500.0f), gold.FresnelR0(700.0f));
}

TEST(SpectralDataTest, SilverOpticalConstants) {
    // Silver has more uniform reflectance across visible spectrum
    ComplexRefractiveIndex silver;
    silver.wavelengths_nm = {400.0f, 500.0f, 600.0f, 700.0f};
    silver.n = {0.05f, 0.05f, 0.06f, 0.07f};
    silver.k = {2.00f, 2.87f, 3.75f, 4.62f};

    EXPECT_TRUE(silver.IsValid());

    // Silver should have very high reflectance (> 0.9) at all visible wavelengths
    for (f32 wl : silver.wavelengths_nm) {
        f32 F0 = silver.FresnelR0(wl);
        EXPECT_GT(F0, 0.9f);
    }
}

TEST(SpectralDataTest, CopperOpticalConstants) {
    // Copper has characteristic red/orange color
    ComplexRefractiveIndex copper;
    copper.wavelengths_nm = {500.0f, 600.0f, 700.0f};
    copper.n = {1.04f, 0.27f, 0.21f};
    copper.k = {2.59f, 3.42f, 4.38f};

    EXPECT_TRUE(copper.IsValid());

    // Copper should have lower F0 at short wavelengths (absorbs blue/green)
    EXPECT_LT(copper.FresnelR0(500.0f), copper.FresnelR0(700.0f));
}
