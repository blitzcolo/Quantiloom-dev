// ============================================================================
// Quantiloom - Unit Tests for hs_core/SpectralReconstructor.hpp
// ============================================================================
// Tests cover:
// - InterpolationMethod enum values
// - SpectralReconstructor::Reconstruct with various methods
// - SpectralReconstructor::InterpolatePixelSpectrum
// - SpectralReconstructor::EstimateMaxError
// - Linear interpolation accuracy
// - Catmull-Rom interpolation smoothness
// - Akima interpolation robustness
// - Edge cases: empty cubes, single-band, boundary extrapolation
// - Reconstruction accuracy verification
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/SpectralReconstructor.hpp"
#include "hs_core/AdaptiveGridGenerator.hpp"
#include "hs_core/HyperspectralConfig.hpp"
#include "core/SpectralCube.hpp"
#include <cmath>
#include <numeric>

using namespace quantiloom;

// ============================================================================
// Helper Functions for Generating Test Data
// ============================================================================

namespace {

/**
 * @brief Create a simple SpectralCube for testing
 */
SpectralCube CreateTestCube(u32 width, u32 height, u32 nbands,
                            f32 minWl, f32 maxWl) {
    SpectralCube cube(width, height, nbands, minWl, maxWl);
    return cube;
}

/**
 * @brief Fill cube with linear ramp in spectral dimension
 */
void FillLinearRamp(SpectralCube& cube, f32 startVal, f32 endVal) {
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 t = static_cast<f32>(b) / std::max(1u, cube.nbands - 1);
        f32 val = startVal + t * (endVal - startVal);
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                cube(x, y, b) = val;
            }
        }
    }
}

/**
 * @brief Fill cube with Gaussian spectrum
 */
void FillGaussianSpectrum(SpectralCube& cube, f32 centerWl, f32 sigma,
                          f32 amplitude, f32 baseline) {
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 wl = cube.wavelengths[b];
        f32 x = wl - centerWl;
        f32 val = baseline + amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x_px = 0; x_px < cube.width; ++x_px) {
                cube(x_px, y, b) = val;
            }
        }
    }
}

/**
 * @brief Fill cube with sinusoidal spectrum
 */
void FillSinusoidalSpectrum(SpectralCube& cube, f32 frequency, f32 amplitude,
                            f32 baseline) {
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 wl = cube.wavelengths[b];
        f32 val = baseline + amplitude * std::sin(frequency * wl);
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x_px = 0; x_px < cube.width; ++x_px) {
                cube(x_px, y, b) = val;
            }
        }
    }
}

/**
 * @brief Calculate RMS error between two cubes
 */
f32 CalculateRMSError(const SpectralCube& a, const SpectralCube& b) {
    if (a.width != b.width || a.height != b.height || a.nbands != b.nbands) {
        return std::numeric_limits<f32>::max();
    }

    f64 sumSq = 0.0;
    u64 count = 0;

    for (u32 y = 0; y < a.height; ++y) {
        for (u32 x = 0; x < a.width; ++x) {
            for (u32 b_idx = 0; b_idx < a.nbands; ++b_idx) {
                f32 diff = a(x, y, b_idx) - b(x, y, b_idx);
                sumSq += static_cast<f64>(diff * diff);
                ++count;
            }
        }
    }

    return static_cast<f32>(std::sqrt(sumSq / count));
}

/**
 * @brief Calculate max absolute error between two cubes
 */
f32 CalculateMaxError(const SpectralCube& a, const SpectralCube& b) {
    if (a.width != b.width || a.height != b.height || a.nbands != b.nbands) {
        return std::numeric_limits<f32>::max();
    }

    f32 maxErr = 0.0f;

    for (u32 y = 0; y < a.height; ++y) {
        for (u32 x = 0; x < a.width; ++x) {
            for (u32 b_idx = 0; b_idx < a.nbands; ++b_idx) {
                f32 err = std::abs(a(x, y, b_idx) - b(x, y, b_idx));
                maxErr = std::max(maxErr, err);
            }
        }
    }

    return maxErr;
}

}  // anonymous namespace

// ============================================================================
// InterpolationMethod Tests
// ============================================================================

TEST(InterpolationMethodTest, EnumValues) {
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Linear), 0u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::CatmullRom), 1u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Akima), 2u);
}

// ============================================================================
// SpectralReconstructor::InterpolatePixelSpectrum Tests
// ============================================================================

TEST(SpectralReconstructorTest, InterpolatePixelLinearConstant) {
    // Sparse cube with constant value
    SpectralCube sparse(2, 2, 3, 3000.0f, 5000.0f);
    for (u32 b = 0; b < 3; ++b) {
        sparse(0, 0, b) = 0.5f;
    }

    Vector<f32> targetWl = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    // All values should be 0.5
    EXPECT_EQ(result.size(), 5u);
    for (f32 val : result) {
        EXPECT_NEAR(val, 0.5f, 1e-5f);
    }
}

TEST(SpectralReconstructorTest, InterpolatePixelLinearRamp) {
    // Sparse cube with linear ramp: [0.0, 0.5, 1.0] at wavelengths [3000, 4000, 5000]
    SpectralCube sparse(2, 2, 3, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 0.5f;
    sparse(0, 0, 2) = 1.0f;

    Vector<f32> targetWl = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    EXPECT_EQ(result.size(), 5u);
    EXPECT_NEAR(result[0], 0.0f, 1e-5f);   // 3000nm
    EXPECT_NEAR(result[1], 0.25f, 1e-5f);  // 3500nm (interpolated)
    EXPECT_NEAR(result[2], 0.5f, 1e-5f);   // 4000nm
    EXPECT_NEAR(result[3], 0.75f, 1e-5f);  // 4500nm (interpolated)
    EXPECT_NEAR(result[4], 1.0f, 1e-5f);   // 5000nm
}

TEST(SpectralReconstructorTest, InterpolatePixelCatmullRom) {
    // Sparse cube with values at [3000, 4000, 5000, 6000]
    SpectralCube sparse(1, 1, 4, 3000.0f, 6000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f, 6000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 1.0f;
    sparse(0, 0, 2) = 1.0f;
    sparse(0, 0, 3) = 0.0f;

    Vector<f32> targetWl = {3500.0f, 4500.0f, 5500.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::CatmullRom
    );

    // Catmull-Rom should produce smooth curve
    EXPECT_EQ(result.size(), 3u);
    // Midpoint values should be reasonable (Catmull-Rom may overshoot)
    EXPECT_GT(result[0], 0.0f);  // Rising
    EXPECT_GT(result[1], 0.5f);  // Near peak
    EXPECT_GT(result[2], 0.0f);  // Falling
}

TEST(SpectralReconstructorTest, InterpolatePixelAkima) {
    // Same test as Catmull-Rom
    SpectralCube sparse(1, 1, 4, 3000.0f, 6000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f, 6000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 1.0f;
    sparse(0, 0, 2) = 1.0f;
    sparse(0, 0, 3) = 0.0f;

    Vector<f32> targetWl = {3500.0f, 4500.0f, 5500.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Akima
    );

    // Akima should also produce smooth curve, but less overshoot
    EXPECT_EQ(result.size(), 3u);
    EXPECT_GT(result[0], 0.0f);
    EXPECT_GT(result[1], 0.5f);
    EXPECT_GT(result[2], 0.0f);
}

// ============================================================================
// SpectralReconstructor::Reconstruct Tests
// ============================================================================

TEST(SpectralReconstructorTest, ReconstructUniformToUniform) {
    // Sparse cube is actually full resolution
    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;

    auto grid = AdaptiveGridGenerator::GenerateUniform(config);

    SpectralCube sparse(2, 2, config.GetNumBands(), 3000.0f, 3200.0f);
    FillLinearRamp(sparse, 0.0f, 1.0f);

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);

    // Should be identical (no actual interpolation needed)
    EXPECT_EQ(full.width, sparse.width);
    EXPECT_EQ(full.height, sparse.height);
    EXPECT_EQ(full.nbands, sparse.nbands);

    f32 maxErr = CalculateMaxError(sparse, full);
    EXPECT_LT(maxErr, 1e-5f);
}

TEST(SpectralReconstructorTest, ReconstructSparseToFull) {
    // Create sparse cube (every other wavelength)
    SpectralCube sparse(2, 2, 3, 3000.0f, 3200.0f);
    sparse.wavelengths = {3000.0f, 3100.0f, 3200.0f};
    FillLinearRamp(sparse, 0.0f, 1.0f);

    AdaptiveGridInfo grid;
    grid.wavelengths = {3000.0f, 3100.0f, 3200.0f};
    grid.isCritical = {true, true, true};
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;  // Target: 5 bands

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);

    EXPECT_EQ(full.nbands, 5u);
    EXPECT_NEAR(full.wavelengths[0], 3000.0f, 1e-3f);
    EXPECT_NEAR(full.wavelengths[4], 3200.0f, 1e-3f);

    // Check interpolated values (should be linear)
    // For pixel (0,0): band 0 = 0.0, band 2 = 0.5, band 4 = 1.0
    // So band 1 (3050nm) should be ~0.25
    EXPECT_NEAR(full(0, 0, 1), 0.25f, 0.05f);
}

TEST(SpectralReconstructorTest, ReconstructPreservesMetadata) {
    SpectralCube sparse(2, 2, 3, 3000.0f, 3200.0f);
    sparse.metadata["test_key"] = "test_value";

    AdaptiveGridInfo grid;
    grid.wavelengths = {3000.0f, 3100.0f, 3200.0f};
    grid.isCritical = {true, true, true};
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 100.0f;

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config);

    // Check metadata preserved
    EXPECT_EQ(full.metadata["test_key"], "test_value");
    // Should also have reconstruction method
    EXPECT_FALSE(full.metadata["reconstruction_method"].empty());
}

TEST(SpectralReconstructorTest, ReconstructWithMapping) {
    SpectralCube sparse(2, 2, 3, 3000.0f, 3200.0f);
    sparse.wavelengths = {3000.0f, 3100.0f, 3200.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 0.5f;
    sparse(0, 0, 2) = 1.0f;

    // Pre-compute mapping
    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;

    auto mapping = GenerateReconstructionMapping(grid, config);

    SpectralReconstructor reconstructor;
    auto full = reconstructor.ReconstructWithMapping(sparse, mapping, config);

    EXPECT_EQ(full.nbands, config.GetNumBands());
}

// ============================================================================
// Interpolation Accuracy Tests
// ============================================================================

TEST(SpectralReconstructorTest, LinearInterpolationExact) {
    // For truly linear data, linear interpolation should be exact
    SpectralCube sparse(1, 1, 3, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 0.5f;
    sparse(0, 0, 2) = 1.0f;

    // Ground truth: also linear
    SpectralCube groundTruth(1, 1, 5, 3000.0f, 5000.0f);
    FillLinearRamp(groundTruth, 0.0f, 1.0f);

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 500.0f;

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);

    f32 maxErr = CalculateMaxError(full, groundTruth);
    EXPECT_LT(maxErr, 1e-4f);  // Nearly exact for linear data
}

TEST(SpectralReconstructorTest, CubicInterpolationSmoothCurve) {
    // For smooth curved data, cubic should be better than linear
    SpectralCube sparse(1, 1, 5, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};
    // Parabola: f(x) = -(x-4000)^2/1000000 + 1
    for (u32 b = 0; b < 5; ++b) {
        f32 wl = sparse.wavelengths[b];
        f32 x = wl - 4000.0f;
        sparse(0, 0, b) = -x * x / 1000000.0f + 1.0f;
    }

    // Ground truth at finer resolution
    SpectralCube groundTruth(1, 1, 21, 3000.0f, 5000.0f);
    for (u32 b = 0; b < 21; ++b) {
        f32 wl = groundTruth.wavelengths[b];
        f32 x = wl - 4000.0f;
        groundTruth(0, 0, b) = -x * x / 1000000.0f + 1.0f;
    }

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 5;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 100.0f;

    SpectralReconstructor reconstructor;

    auto fullLinear = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);
    auto fullCubic = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::CatmullRom);

    f32 linearErr = CalculateRMSError(fullLinear, groundTruth);
    f32 cubicErr = CalculateRMSError(fullCubic, groundTruth);

    // Cubic should have lower error for smooth curves
    EXPECT_LT(cubicErr, linearErr);
}

// ============================================================================
// SpectralReconstructor::EstimateMaxError Tests
// ============================================================================

TEST(SpectralReconstructorTest, EstimateMaxErrorCoarse) {
    AdaptiveGridInfo grid;
    grid.compressionRatio = 2.0f;

    HyperspectralConfig config;
    config.wavelengthStep_nm = 50.0f;
    config.adaptiveCoarseMultiplier = 2.0f;

    f32 error = SpectralReconstructor::EstimateMaxError(grid, config);

    // Error should be positive and finite
    EXPECT_GT(error, 0.0f);
    EXPECT_LT(error, 1.0f);
}

TEST(SpectralReconstructorTest, EstimateMaxErrorFine) {
    AdaptiveGridInfo grid;
    grid.compressionRatio = 1.0f;

    HyperspectralConfig config;
    config.wavelengthStep_nm = 10.0f;
    config.adaptiveCoarseMultiplier = 1.0f;

    f32 error = SpectralReconstructor::EstimateMaxError(grid, config);

    // Fine sampling should have lower estimated error
    EXPECT_GT(error, 0.0f);
}

TEST(SpectralReconstructorTest, EstimateMaxErrorScalesWithStep) {
    AdaptiveGridInfo grid;
    grid.compressionRatio = 1.0f;

    HyperspectralConfig config1;
    config1.wavelengthStep_nm = 10.0f;
    config1.adaptiveCoarseMultiplier = 2.0f;

    HyperspectralConfig config2;
    config2.wavelengthStep_nm = 50.0f;
    config2.adaptiveCoarseMultiplier = 2.0f;

    f32 error1 = SpectralReconstructor::EstimateMaxError(grid, config1);
    f32 error2 = SpectralReconstructor::EstimateMaxError(grid, config2);

    // Larger step should have larger error estimate
    EXPECT_GT(error2, error1);
}

// The estimate is driven by the widest gap actually present in the grid, not by
// the config's nominal coarse step. These two cases pin that down; without them
// gridInfo could go unread again and nothing would fail.

TEST(SpectralReconstructorTest, EstimateMaxErrorUsesWidestGridGap) {
    HyperspectralConfig config;
    config.wavelengthStep_nm = 10.0f;
    config.adaptiveCoarseMultiplier = 2.0f;  // nominal coarse step = 20 nm

    // A grid whose real gaps (10, 10, 80) exceed the nominal coarse step.
    AdaptiveGridInfo grid;
    grid.wavelengths = {1000.0f, 1010.0f, 1020.0f, 1100.0f};

    AdaptiveGridInfo empty;

    f32 measured = SpectralReconstructor::EstimateMaxError(grid, config);
    f32 nominal = SpectralReconstructor::EstimateMaxError(empty, config);

    // 80 nm gap vs a 20 nm assumption: 16x in a quadratic bound.
    EXPECT_GT(measured, nominal);
    EXPECT_NEAR(measured / nominal, 16.0f, 0.1f);
}

TEST(SpectralReconstructorTest, EstimateMaxErrorUniformGridMatchesItsOwnSpacing) {
    HyperspectralConfig config;
    config.wavelengthStep_nm = 10.0f;
    config.adaptiveCoarseMultiplier = 1.0f;

    // Uniform 10 nm grid: measured spacing and nominal step agree, so the
    // estimate must be the same either way.
    AdaptiveGridInfo grid;
    grid.wavelengths = {500.0f, 510.0f, 520.0f, 530.0f};

    AdaptiveGridInfo empty;

    EXPECT_NEAR(SpectralReconstructor::EstimateMaxError(grid, config),
                SpectralReconstructor::EstimateMaxError(empty, config), 1e-12f);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST(SpectralReconstructorTest, InterpolateEmptyWavelengthList) {
    SpectralCube sparse(1, 1, 3, 3000.0f, 5000.0f);
    sparse(0, 0, 0) = 0.5f;

    Vector<f32> targetWl;  // Empty

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(sparse, 0, 0, targetWl);

    EXPECT_TRUE(result.empty());
}

TEST(SpectralReconstructorTest, InterpolateSingleSourceBand) {
    SpectralCube sparse(1, 1, 1, 4000.0f, 4000.0f);
    sparse.wavelengths = {4000.0f};
    sparse(0, 0, 0) = 0.5f;

    Vector<f32> targetWl = {3500.0f, 4000.0f, 4500.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    // With single source, all targets should get that value
    EXPECT_EQ(result.size(), 3u);
    for (f32 val : result) {
        EXPECT_NEAR(val, 0.5f, 1e-5f);
    }
}

TEST(SpectralReconstructorTest, InterpolateTwoSourceBands) {
    SpectralCube sparse(1, 1, 2, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 5000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 1.0f;

    Vector<f32> targetWl = {3000.0f, 4000.0f, 5000.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    EXPECT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 0.0f, 1e-5f);
    EXPECT_NEAR(result[1], 0.5f, 1e-5f);  // Midpoint
    EXPECT_NEAR(result[2], 1.0f, 1e-5f);
}

TEST(SpectralReconstructorTest, ExtrapolationBeyondRange) {
    SpectralCube sparse(1, 1, 3, 3500.0f, 4500.0f);
    sparse.wavelengths = {3500.0f, 4000.0f, 4500.0f};
    sparse(0, 0, 0) = 0.3f;
    sparse(0, 0, 1) = 0.5f;
    sparse(0, 0, 2) = 0.7f;

    // Target wavelengths outside source range
    Vector<f32> targetWl = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    EXPECT_EQ(result.size(), 5u);
    // Values at edges should be reasonable (clamped or extrapolated)
    EXPECT_GT(result[0], -1.0f);  // Not wildly negative
    EXPECT_LT(result[4], 2.0f);   // Not wildly positive
}

TEST(SpectralReconstructorTest, ReconstructLargeCube) {
    // Test with larger cube dimensions
    SpectralCube sparse(64, 64, 10, 3000.0f, 5000.0f);
    FillGaussianSpectrum(sparse, 4000.0f, 500.0f, 0.5f, 0.2f);

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 10;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 100.0f;

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config);

    EXPECT_EQ(full.width, 64u);
    EXPECT_EQ(full.height, 64u);
    EXPECT_EQ(full.nbands, 21u);  // (5000-3000)/100 + 1
}

TEST(SpectralReconstructorTest, ReconstructSinglePixel) {
    SpectralCube sparse(1, 1, 3, 3000.0f, 5000.0f);
    sparse(0, 0, 0) = 0.1f;
    sparse(0, 0, 1) = 0.5f;
    sparse(0, 0, 2) = 0.9f;

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 500.0f;

    SpectralReconstructor reconstructor;
    auto full = reconstructor.Reconstruct(sparse, grid, config);

    EXPECT_EQ(full.width, 1u);
    EXPECT_EQ(full.height, 1u);
}

// ============================================================================
// Interpolation Method Comparison Tests
// ============================================================================

TEST(SpectralReconstructorTest, CompareMethodsOnSmoothData) {
    // Create smooth Gaussian spectrum
    SpectralCube sparse(1, 1, 5, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};
    FillGaussianSpectrum(sparse, 4000.0f, 600.0f, 0.8f, 0.1f);

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 5;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 100.0f;

    SpectralReconstructor reconstructor;

    auto linearResult = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);
    auto catmullResult = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::CatmullRom);
    auto akimaResult = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Akima);

    // All should produce valid results
    EXPECT_EQ(linearResult.nbands, config.GetNumBands());
    EXPECT_EQ(catmullResult.nbands, config.GetNumBands());
    EXPECT_EQ(akimaResult.nbands, config.GetNumBands());

    // All values should be positive (Gaussian + baseline)
    for (u32 b = 0; b < linearResult.nbands; ++b) {
        EXPECT_GT(linearResult(0, 0, b), 0.0f);
        EXPECT_GT(catmullResult(0, 0, b), 0.0f);
        EXPECT_GT(akimaResult(0, 0, b), 0.0f);
    }
}

TEST(SpectralReconstructorTest, CubicDoesNotOscillateOnMonotonic) {
    // Monotonic increasing data - cubic should not introduce oscillations
    SpectralCube sparse(1, 1, 5, 3000.0f, 5000.0f);
    sparse.wavelengths = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};
    sparse(0, 0, 0) = 0.0f;
    sparse(0, 0, 1) = 0.25f;
    sparse(0, 0, 2) = 0.5f;
    sparse(0, 0, 3) = 0.75f;
    sparse(0, 0, 4) = 1.0f;

    Vector<f32> targetWl = {3250.0f, 3750.0f, 4250.0f, 4750.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::CatmullRom
    );

    // Interpolated values should be monotonic (or nearly so)
    for (usize i = 1; i < result.size(); ++i) {
        // Allow small tolerance for numerical issues
        EXPECT_GE(result[i], result[i-1] - 0.01f);
    }
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST(SpectralReconstructorTest, MetadataRecordsMethod) {
    SpectralCube sparse(1, 1, 3, 3000.0f, 5000.0f);

    AdaptiveGridInfo grid;
    grid.wavelengths = sparse.wavelengths;
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 1000.0f;

    SpectralReconstructor reconstructor;

    auto linear = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Linear);
    EXPECT_EQ(linear.metadata["reconstruction_method"], "linear");

    auto catmull = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::CatmullRom);
    EXPECT_EQ(catmull.metadata["reconstruction_method"], "catmull_rom");

    auto akima = reconstructor.Reconstruct(sparse, grid, config, InterpolationMethod::Akima);
    EXPECT_EQ(akima.metadata["reconstruction_method"], "akima");
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

TEST(SpectralReconstructorTest, InterpolateAtExactSourceWavelengths) {
    SpectralCube sparse(1, 1, 4, 3000.0f, 6000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f, 6000.0f};
    sparse(0, 0, 0) = 0.1f;
    sparse(0, 0, 1) = 0.4f;
    sparse(0, 0, 2) = 0.7f;
    sparse(0, 0, 3) = 1.0f;

    // Query at exact source wavelengths
    Vector<f32> targetWl = {3000.0f, 4000.0f, 5000.0f, 6000.0f};

    SpectralReconstructor reconstructor;
    auto result = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::CatmullRom
    );

    // Should return exact values
    EXPECT_NEAR(result[0], 0.1f, 1e-4f);
    EXPECT_NEAR(result[1], 0.4f, 1e-4f);
    EXPECT_NEAR(result[2], 0.7f, 1e-4f);
    EXPECT_NEAR(result[3], 1.0f, 1e-4f);
}

TEST(SpectralReconstructorTest, InterpolateFirstAndLastSegment) {
    SpectralCube sparse(1, 1, 4, 3000.0f, 6000.0f);
    sparse.wavelengths = {3000.0f, 4000.0f, 5000.0f, 6000.0f};
    sparse(0, 0, 0) = 0.2f;
    sparse(0, 0, 1) = 0.4f;
    sparse(0, 0, 2) = 0.6f;
    sparse(0, 0, 3) = 0.8f;

    // Query in first and last segments
    Vector<f32> targetWl = {3200.0f, 5800.0f};

    SpectralReconstructor reconstructor;

    auto linearResult = reconstructor.InterpolatePixelSpectrum(
        sparse, 0, 0, targetWl, InterpolationMethod::Linear
    );

    // First segment: 3200 is 20% into [3000, 4000]
    EXPECT_NEAR(linearResult[0], 0.2f + 0.2f * 0.2f, 0.02f);

    // Last segment: 5800 is 80% into [5000, 6000]
    EXPECT_NEAR(linearResult[1], 0.6f + 0.2f * 0.8f, 0.02f);
}
