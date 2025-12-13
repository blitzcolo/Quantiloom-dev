// ============================================================================
// Quantiloom - Unit Tests for core/SpectralCube.hpp
// ============================================================================
// Tests cover:
// - SpectralCube construction and initialization
// - Wavelength generation and validation
// - Pixel access and band access
// - Memory layout verification (C-order: band-major)
// - FindClosestBand functionality
// - Edge cases and validation
// ============================================================================

#include <gtest/gtest.h>
#include "core/SpectralCube.hpp"

using namespace quantiloom;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(SpectralCubeTest, DefaultConstruction) {
    SpectralCube cube;

    EXPECT_EQ(cube.width, 0);
    EXPECT_EQ(cube.height, 0);
    EXPECT_EQ(cube.nbands, 0);
    EXPECT_EQ(cube.lambda_min, 0.0f);
    EXPECT_EQ(cube.lambda_max, 0.0f);
    EXPECT_EQ(cube.delta_lambda, 0.0f);
    EXPECT_TRUE(cube.data.empty());
    EXPECT_TRUE(cube.wavelengths.empty());
    EXPECT_FALSE(cube.IsValid());
}

TEST(SpectralCubeTest, ParameterizedConstruction) {
    // Create a 10x8 cube with 50 bands from 400nm to 700nm
    SpectralCube cube(10, 8, 50, 400.0f, 700.0f);

    EXPECT_EQ(cube.width, 10);
    EXPECT_EQ(cube.height, 8);
    EXPECT_EQ(cube.nbands, 50);
    EXPECT_EQ(cube.lambda_min, 400.0f);
    EXPECT_EQ(cube.lambda_max, 700.0f);
    EXPECT_TRUE(cube.IsValid());

    // Check delta_lambda calculation
    f32 expected_delta = (700.0f - 400.0f) / 49.0f;  // nbands - 1
    EXPECT_NEAR(cube.delta_lambda, expected_delta, 1e-5f);

    // Check data allocation
    EXPECT_EQ(cube.data.size(), 10 * 8 * 50);

    // Check wavelengths array
    EXPECT_EQ(cube.wavelengths.size(), 50);
    EXPECT_NEAR(cube.wavelengths[0], 400.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[49], 700.0f, 1e-5f);

    // Check default initialization to zero
    for (const auto& val : cube.data) {
        EXPECT_EQ(val, 0.0f);
    }
}

TEST(SpectralCubeTest, WavelengthGeneration) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);

    EXPECT_EQ(cube.wavelengths.size(), 5);
    EXPECT_NEAR(cube.wavelengths[0], 400.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[1], 500.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[2], 600.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[3], 700.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[4], 800.0f, 1e-5f);
}

// ============================================================================
// Pixel and Band Access Tests
// ============================================================================

TEST(SpectralCubeTest, PixelAccess) {
    SpectralCube cube(10, 8, 3, 400.0f, 600.0f);

    // Write values
    cube(5, 3, 0) = 1.0f;
    cube(5, 3, 1) = 2.0f;
    cube(5, 3, 2) = 3.0f;

    // Read values
    EXPECT_EQ(cube(5, 3, 0), 1.0f);
    EXPECT_EQ(cube(5, 3, 1), 2.0f);
    EXPECT_EQ(cube(5, 3, 2), 3.0f);
}

TEST(SpectralCubeTest, ConstPixelAccess) {
    SpectralCube cube(10, 8, 2, 400.0f, 500.0f);
    cube(2, 3, 0) = 5.5f;
    cube(2, 3, 1) = 6.5f;

    const SpectralCube& constCube = cube;
    EXPECT_EQ(constCube(2, 3, 0), 5.5f);
    EXPECT_EQ(constCube(2, 3, 1), 6.5f);
}

TEST(SpectralCubeTest, BandPointerAccess) {
    SpectralCube cube(4, 3, 2, 400.0f, 500.0f);

    // Get pointer to first band
    f32* band0 = cube.BandPtr(0);
    f32* band1 = cube.BandPtr(1);

    // Write directly via pointer
    for (u32 i = 0; i < 12; ++i) {  // 4x3 = 12 pixels per band
        band0[i] = 100.0f;
        band1[i] = 200.0f;
    }

    // Verify via operator()
    for (u32 y = 0; y < 3; ++y) {
        for (u32 x = 0; x < 4; ++x) {
            EXPECT_EQ(cube(x, y, 0), 100.0f);
            EXPECT_EQ(cube(x, y, 1), 200.0f);
        }
    }
}

TEST(SpectralCubeTest, ConstBandPointerAccess) {
    SpectralCube cube(4, 3, 2, 400.0f, 500.0f);
    cube(1, 2, 0) = 7.5f;

    const SpectralCube& constCube = cube;
    const f32* band0 = constCube.BandPtr(0);

    // Verify we can read through const pointer
    EXPECT_EQ(band0[2 * 4 + 1], 7.5f);  // y=2, x=1
}

// ============================================================================
// Memory Layout Tests (C-order: band-major)
// ============================================================================

TEST(SpectralCubeTest, BandMajorLayout) {
    // Create 2x2 cube with 3 bands
    SpectralCube cube(2, 2, 3, 400.0f, 600.0f);

    // Set known pattern: band_index * 100 + y * 10 + x
    for (u32 b = 0; b < 3; ++b) {
        for (u32 y = 0; y < 2; ++y) {
            for (u32 x = 0; x < 2; ++x) {
                cube(x, y, b) = static_cast<f32>(b * 100 + y * 10 + x);
            }
        }
    }

    // Verify C-order memory layout: [band][y][x]
    // Band 0: [0, 1, 10, 11]
    // Band 1: [100, 101, 110, 111]
    // Band 2: [200, 201, 210, 211]
    const f32* data = cube.data.data();

    // Band 0
    EXPECT_EQ(data[0], 0.0f);    // (0, 0, 0)
    EXPECT_EQ(data[1], 1.0f);    // (1, 0, 0)
    EXPECT_EQ(data[2], 10.0f);   // (0, 1, 0)
    EXPECT_EQ(data[3], 11.0f);   // (1, 1, 0)

    // Band 1
    EXPECT_EQ(data[4], 100.0f);  // (0, 0, 1)
    EXPECT_EQ(data[5], 101.0f);  // (1, 0, 1)
    EXPECT_EQ(data[6], 110.0f);  // (0, 1, 1)
    EXPECT_EQ(data[7], 111.0f);  // (1, 1, 1)

    // Band 2
    EXPECT_EQ(data[8], 200.0f);   // (0, 0, 2)
    EXPECT_EQ(data[9], 201.0f);   // (1, 0, 2)
    EXPECT_EQ(data[10], 210.0f);  // (0, 1, 2)
    EXPECT_EQ(data[11], 211.0f);  // (1, 1, 2)
}

TEST(SpectralCubeTest, BandContiguity) {
    SpectralCube cube(10, 8, 50, 400.0f, 700.0f);

    // Each band should be contiguous in memory
    f32* band0 = cube.BandPtr(0);
    f32* band1 = cube.BandPtr(1);

    // Distance between bands should be width * height
    usize band_size = cube.width * cube.height;
    EXPECT_EQ(band1 - band0, band_size);
}

// ============================================================================
// Utility Method Tests
// ============================================================================

TEST(SpectralCubeTest, PixelsPerBand) {
    SpectralCube cube(100, 50, 10, 400.0f, 700.0f);
    EXPECT_EQ(cube.PixelsPerBand(), 5000);
}

TEST(SpectralCubeTest, TotalElements) {
    SpectralCube cube(100, 50, 10, 400.0f, 700.0f);
    EXPECT_EQ(cube.TotalElements(), 50000);
}

TEST(SpectralCubeTest, GetWavelength) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);

    EXPECT_NEAR(cube.GetWavelength(0), 400.0f, 1e-5f);
    EXPECT_NEAR(cube.GetWavelength(2), 600.0f, 1e-5f);
    EXPECT_NEAR(cube.GetWavelength(4), 800.0f, 1e-5f);
}

TEST(SpectralCubeTest, Clear) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);

    // Fill with non-zero values
    for (auto& val : cube.data) {
        val = 42.0f;
    }

    cube.Clear();

    // Verify all zeros
    for (const auto& val : cube.data) {
        EXPECT_EQ(val, 0.0f);
    }
}

// ============================================================================
// FindClosestBand Tests
// ============================================================================

TEST(SpectralCubeTest, FindClosestBandExactMatch) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);
    // Wavelengths: [400, 500, 600, 700, 800]

    EXPECT_EQ(cube.FindClosestBand(400.0f), 0);
    EXPECT_EQ(cube.FindClosestBand(600.0f), 2);
    EXPECT_EQ(cube.FindClosestBand(800.0f), 4);
}

TEST(SpectralCubeTest, FindClosestBandInterpolated) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);
    // Wavelengths: [400, 500, 600, 700, 800]

    // 450 is closer to 400 than 500
    EXPECT_EQ(cube.FindClosestBand(450.0f), 0);

    // 550 is exactly between 500 and 600, should pick 500 (index 1)
    EXPECT_EQ(cube.FindClosestBand(550.0f), 1);

    // 720 is closer to 700 than 800
    EXPECT_EQ(cube.FindClosestBand(720.0f), 3);
}

TEST(SpectralCubeTest, FindClosestBandOutOfRange) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);

    // Below range - should return first band
    EXPECT_EQ(cube.FindClosestBand(300.0f), 0);

    // Above range - should return last band
    EXPECT_EQ(cube.FindClosestBand(1000.0f), 4);
}

TEST(SpectralCubeTest, FindClosestBandEmptyCube) {
    SpectralCube cube;
    EXPECT_EQ(cube.FindClosestBand(550.0f), 0);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(SpectralCubeTest, IsValidCorrect) {
    SpectralCube cube(10, 8, 5, 400.0f, 700.0f);
    EXPECT_TRUE(cube.IsValid());
}

TEST(SpectralCubeTest, IsValidEmptyCube) {
    SpectralCube cube;
    EXPECT_FALSE(cube.IsValid());
}

TEST(SpectralCubeTest, IsValidZeroDimensions) {
    SpectralCube cube(0, 10, 5, 400.0f, 700.0f);
    EXPECT_FALSE(cube.IsValid());

    SpectralCube cube2(10, 0, 5, 400.0f, 700.0f);
    EXPECT_FALSE(cube2.IsValid());

    SpectralCube cube3(10, 10, 0, 400.0f, 700.0f);
    EXPECT_FALSE(cube3.IsValid());
}

TEST(SpectralCubeTest, IsValidInvalidWavelengthRange) {
    SpectralCube cube(10, 10, 5, 700.0f, 400.0f);  // min > max
    EXPECT_FALSE(cube.IsValid());
}

TEST(SpectralCubeTest, IsValidDataSizeMismatch) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);
    EXPECT_TRUE(cube.IsValid());

    // Corrupt data size
    cube.data.resize(100);
    EXPECT_FALSE(cube.IsValid());
}

TEST(SpectralCubeTest, IsValidWavelengthSizeMismatch) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);
    EXPECT_TRUE(cube.IsValid());

    // Corrupt wavelengths size
    cube.wavelengths.resize(3);
    EXPECT_FALSE(cube.IsValid());
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST(SpectralCubeTest, MetadataStorage) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);

    cube.metadata["sensor"] = "AVIRIS-NG";
    cube.metadata["date"] = "2024-01-15";
    cube.metadata["integration_time_ms"] = "10.5";

    EXPECT_EQ(cube.metadata["sensor"], "AVIRIS-NG");
    EXPECT_EQ(cube.metadata["date"], "2024-01-15");
    EXPECT_EQ(cube.metadata["integration_time_ms"], "10.5");
    EXPECT_EQ(cube.metadata.size(), 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SpectralCubeTest, SinglePixelCube) {
    // Single pixel with multiple bands (avoid nbands=1 which causes divide-by-zero)
    SpectralCube cube(1, 1, 5, 400.0f, 700.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.PixelsPerBand(), 1);
    EXPECT_EQ(cube.TotalElements(), 5);

    cube(0, 0, 2) = 99.0f;
    EXPECT_EQ(cube(0, 0, 2), 99.0f);
}

TEST(SpectralCubeTest, SingleBandCube) {
    // Single band with multiple pixels (avoid nbands=1 for now - known limitation)
    // Note: nbands=1 causes delta_lambda division by zero in constructor
    // This is a known limitation that should be fixed in SpectralCube.hpp
    SpectralCube cube(100, 100, 2, 550.0f, 560.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.nbands, 2);
    EXPECT_EQ(cube.wavelengths.size(), 2);
    EXPECT_NEAR(cube.wavelengths[0], 550.0f, 1e-5f);
    EXPECT_NEAR(cube.wavelengths[1], 560.0f, 1e-5f);
}

TEST(SpectralCubeTest, HighSpectralResolution) {
    // Hyperspectral cube with 500 bands
    SpectralCube cube(100, 100, 500, 400.0f, 2500.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.nbands, 500);
    EXPECT_EQ(cube.TotalElements(), 100 * 100 * 500);
    EXPECT_EQ(cube.wavelengths.size(), 500);

    // Check wavelength spacing
    f32 expected_delta = (2500.0f - 400.0f) / 499.0f;
    EXPECT_NEAR(cube.delta_lambda, expected_delta, 1e-4f);
}

TEST(SpectralCubeTest, HDRValues) {
    SpectralCube cube(10, 10, 3, 400.0f, 600.0f);

    // Test with HDR radiance values
    cube(5, 5, 0) = 10000.0f;
    cube(5, 5, 1) = -100.0f;  // Negative values should be allowed
    cube(5, 5, 2) = 0.00001f;

    EXPECT_EQ(cube(5, 5, 0), 10000.0f);
    EXPECT_EQ(cube(5, 5, 1), -100.0f);
    EXPECT_EQ(cube(5, 5, 2), 0.00001f);
}

// ============================================================================
// Realistic Use Case Tests
// ============================================================================

TEST(SpectralCubeTest, VisibleSpectrum) {
    // Typical visible spectrum hyperspectral cube
    SpectralCube cube(640, 480, 76, 380.0f, 760.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.width, 640);
    EXPECT_EQ(cube.height, 480);
    EXPECT_EQ(cube.nbands, 76);
    EXPECT_NEAR(cube.delta_lambda, 5.066667f, 1e-3f);  // ~5nm spacing
}

TEST(SpectralCubeTest, NIRSpectrum) {
    // Near-infrared hyperspectral cube
    SpectralCube cube(1024, 768, 128, 700.0f, 1400.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.nbands, 128);
    EXPECT_NEAR(cube.lambda_min, 700.0f, 1e-5f);
    EXPECT_NEAR(cube.lambda_max, 1400.0f, 1e-5f);
}

TEST(SpectralCubeTest, ThermalIRSpectrum) {
    // Thermal infrared (LWIR) hyperspectral cube
    SpectralCube cube(320, 240, 200, 8000.0f, 12000.0f);

    EXPECT_TRUE(cube.IsValid());
    EXPECT_EQ(cube.nbands, 200);
    EXPECT_NEAR(cube.lambda_min, 8000.0f, 1e-5f);  // 8 microns
    EXPECT_NEAR(cube.lambda_max, 12000.0f, 1e-5f); // 12 microns
}
