// ============================================================================
// Quantiloom - Unit Tests for io/SpectralCubeIO.hpp
// ============================================================================
// Tests cover:
// - ENVI format writing and reading (BSQ, BIL, BIP)
// - EXR multipart format writing and reading
// - Interleave conversion correctness
// - Wavelength metadata preservation
// - Round-trip data integrity
// - Error handling
// ============================================================================

#include <gtest/gtest.h>
#include "io/SpectralCubeIO.hpp"
#include "core/SpectralCube.hpp"

#include <filesystem>
#include <cmath>

using namespace quantiloom;

// Helper to create test directory
class SpectralCubeIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "quantiloom_test";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all(testDir);
    }

    std::filesystem::path testDir;

    // Create a test cube with known values
    SpectralCube CreateTestCube(u32 width, u32 height, u32 nbands,
                                 f32 lambdaMin, f32 lambdaMax) {
        SpectralCube cube(width, height, nbands, lambdaMin, lambdaMax);

        // Fill with test pattern: value = band * 100 + y * 10 + x
        for (u32 b = 0; b < nbands; ++b) {
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    cube(x, y, b) = static_cast<f32>(b * 100 + y * 10 + x);
                }
            }
        }

        return cube;
    }

    // Compare two cubes for equality
    bool CompareCubes(const SpectralCube& a, const SpectralCube& b, f32 tolerance = 1e-5f) {
        if (a.width != b.width || a.height != b.height || a.nbands != b.nbands) {
            return false;
        }

        for (u32 b_idx = 0; b_idx < a.nbands; ++b_idx) {
            for (u32 y = 0; y < a.height; ++y) {
                for (u32 x = 0; x < a.width; ++x) {
                    if (std::abs(a(x, y, b_idx) - b(x, y, b_idx)) > tolerance) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
};

// ============================================================================
// ENVIInterleave Helper Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, InterleaveToString) {
    EXPECT_STREQ(ENVIInterleaveToString(ENVIInterleave::BSQ), "bsq");
    EXPECT_STREQ(ENVIInterleaveToString(ENVIInterleave::BIL), "bil");
    EXPECT_STREQ(ENVIInterleaveToString(ENVIInterleave::BIP), "bip");
}

// ============================================================================
// ENVI BSQ Format Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, WriteReadENVI_BSQ_Small) {
    auto cube = CreateTestCube(4, 3, 5, 400.0f, 800.0f);
    String basePath = (testDir / "test_bsq").string();

    // Write
    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BSQ));

    // Check files exist
    EXPECT_TRUE(std::filesystem::exists(basePath + ".hdr"));
    EXPECT_TRUE(std::filesystem::exists(basePath + ".dat"));

    // Read back
    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value()) << result.error();

    const SpectralCube& loaded = result.value();

    // Verify dimensions
    EXPECT_EQ(loaded.width, cube.width);
    EXPECT_EQ(loaded.height, cube.height);
    EXPECT_EQ(loaded.nbands, cube.nbands);

    // Verify wavelength range
    EXPECT_NEAR(loaded.lambda_min, cube.lambda_min, 1e-2f);
    EXPECT_NEAR(loaded.lambda_max, cube.lambda_max, 1e-2f);

    // Verify data integrity
    EXPECT_TRUE(CompareCubes(cube, loaded));
}

TEST_F(SpectralCubeIOTest, WriteReadENVI_BSQ_Large) {
    auto cube = CreateTestCube(64, 48, 128, 400.0f, 2500.0f);
    String basePath = (testDir / "test_bsq_large").string();

    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BSQ));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value()) << result.error();

    EXPECT_TRUE(CompareCubes(cube, result.value()));
}

// ============================================================================
// ENVI BIL Format Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, WriteReadENVI_BIL) {
    auto cube = CreateTestCube(4, 3, 5, 400.0f, 800.0f);
    String basePath = (testDir / "test_bil").string();

    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BIL));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value()) << result.error();

    // BIL should produce identical data after round-trip
    EXPECT_TRUE(CompareCubes(cube, result.value()));
}

// ============================================================================
// ENVI BIP Format Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, WriteReadENVI_BIP) {
    auto cube = CreateTestCube(4, 3, 5, 400.0f, 800.0f);
    String basePath = (testDir / "test_bip").string();

    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BIP));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value()) << result.error();

    // BIP should produce identical data after round-trip
    EXPECT_TRUE(CompareCubes(cube, result.value()));
}

// ============================================================================
// ENVI Wavelength Metadata Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, ENVIWavelengthPreservation) {
    SpectralCube cube(10, 10, 5, 400.0f, 800.0f);

    // Custom wavelengths (non-uniform)
    cube.wavelengths = {400.0f, 450.0f, 550.0f, 700.0f, 800.0f};

    String basePath = (testDir / "test_wavelengths").string();
    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BSQ));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value());

    const SpectralCube& loaded = result.value();
    ASSERT_EQ(loaded.wavelengths.size(), 5);

    EXPECT_NEAR(loaded.wavelengths[0], 400.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[1], 450.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[2], 550.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[3], 700.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[4], 800.0f, 1e-2f);
}

// ============================================================================
// EXR Format Tests
// ============================================================================
// NOTE: EXR multipart format is temporarily disabled due to static initialization
// conflict with VMA/Vulkan. These tests are skipped until the issue is resolved.
// See SpectralCubeIO.cpp for details.

TEST_F(SpectralCubeIOTest, WriteReadEXR_Small) {
    GTEST_SKIP() << "EXR multipart disabled due to VMA static initialization conflict";

    auto cube = CreateTestCube(16, 12, 8, 400.0f, 700.0f);
    String path = (testDir / "test_cube.exr").string();

    ASSERT_TRUE(SpectralCubeIO::WriteEXR(cube, path));
    EXPECT_TRUE(std::filesystem::exists(path));

    auto result = SpectralCubeIO::ReadEXR(path);
    ASSERT_TRUE(result.has_value()) << result.error();

    const SpectralCube& loaded = result.value();
    EXPECT_EQ(loaded.width, cube.width);
    EXPECT_EQ(loaded.height, cube.height);
    EXPECT_EQ(loaded.nbands, cube.nbands);

    EXPECT_TRUE(CompareCubes(cube, loaded));
}

TEST_F(SpectralCubeIOTest, WriteReadEXR_Wavelengths) {
    GTEST_SKIP() << "EXR multipart disabled due to VMA static initialization conflict";

    SpectralCube cube(8, 8, 4, 3000.0f, 5000.0f);
    cube.wavelengths = {3000.0f, 3500.0f, 4000.0f, 5000.0f};

    String path = (testDir / "test_exr_wl.exr").string();
    ASSERT_TRUE(SpectralCubeIO::WriteEXR(cube, path));

    auto result = SpectralCubeIO::ReadEXR(path);
    ASSERT_TRUE(result.has_value());

    // EXR stores wavelength as float attribute
    const SpectralCube& loaded = result.value();
    EXPECT_EQ(loaded.nbands, 4);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, WriteInvalidCube) {
    SpectralCube invalid;  // Default construction = invalid
    String basePath = (testDir / "test_invalid").string();

    EXPECT_FALSE(SpectralCubeIO::WriteENVI(invalid, basePath));
    EXPECT_FALSE(SpectralCubeIO::WriteEXR(invalid, basePath + ".exr"));
}

TEST_F(SpectralCubeIOTest, ReadNonexistentENVI) {
    auto result = SpectralCubeIO::ReadENVI("/nonexistent/path/file");
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralCubeIOTest, ReadNonexistentEXR) {
    auto result = SpectralCubeIO::ReadEXR("/nonexistent/path/file.exr");
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, DetectFormatENVI) {
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.hdr"), "envi");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.dat"), "envi");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.raw"), "envi");
}

TEST_F(SpectralCubeIOTest, DetectFormatGeoTIFF) {
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.tif"), "geotiff");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.tiff"), "geotiff");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.TIF"), "geotiff");
}

TEST_F(SpectralCubeIOTest, DetectFormatEXR) {
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.exr"), "exr");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.EXR"), "exr");
}

TEST_F(SpectralCubeIOTest, DetectFormatUnknown) {
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.png"), "unknown");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("file.jpg"), "unknown");
    EXPECT_EQ(SpectralCubeIO::DetectFormat("noextension"), "unknown");
}

// ============================================================================
// Interleave Conversion Correctness Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, InterleaveConversionBSQ_BIL_RoundTrip) {
    // Create cube and write as BIL
    auto cube = CreateTestCube(8, 6, 10, 400.0f, 800.0f);
    String basePath = (testDir / "interleave_bil").string();

    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BIL));

    // Read back (should convert back to BSQ internally)
    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value());

    // Verify every pixel matches
    const SpectralCube& loaded = result.value();
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                EXPECT_NEAR(cube(x, y, b), loaded(x, y, b), 1e-5f)
                    << "Mismatch at (" << x << "," << y << "," << b << ")";
            }
        }
    }
}

TEST_F(SpectralCubeIOTest, InterleaveConversionBSQ_BIP_RoundTrip) {
    auto cube = CreateTestCube(8, 6, 10, 400.0f, 800.0f);
    String basePath = (testDir / "interleave_bip").string();

    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath, ENVIInterleave::BIP));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value());

    const SpectralCube& loaded = result.value();
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                EXPECT_NEAR(cube(x, y, b), loaded(x, y, b), 1e-5f)
                    << "Mismatch at (" << x << "," << y << "," << b << ")";
            }
        }
    }
}

// ============================================================================
// HDR Value Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, HDRValuePreservation) {
    SpectralCube cube(4, 4, 3, 400.0f, 600.0f);

    // Test various HDR values
    cube(0, 0, 0) = 0.0f;
    cube(1, 0, 0) = 1.0f;
    cube(2, 0, 0) = 10000.0f;  // High dynamic range
    cube(3, 0, 0) = 1e-6f;     // Very small
    cube(0, 1, 0) = -0.5f;     // Negative (might occur with certain processing)

    String basePath = (testDir / "hdr_test").string();
    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value());

    const SpectralCube& loaded = result.value();
    EXPECT_NEAR(loaded(0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(loaded(1, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(loaded(2, 0, 0), 10000.0f, 1.0f);
    EXPECT_NEAR(loaded(3, 0, 0), 1e-6f, 1e-7f);
    EXPECT_NEAR(loaded(0, 1, 0), -0.5f, 1e-6f);
}

// ============================================================================
// Realistic Use Case Tests
// ============================================================================

TEST_F(SpectralCubeIOTest, RealisticMWIRCube) {
    // Typical MWIR thermal imaging cube
    SpectralCube cube(320, 240, 41, 3000.0f, 5000.0f);

    // Fill with blackbody-like radiance pattern
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 wavelength = cube.wavelengths[b];
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                // Simple gradient for test
                cube(x, y, b) = 100.0f + wavelength * 0.01f + y * 0.1f;
            }
        }
    }

    cube.metadata["sensor"] = "MWIR_Test";
    cube.metadata["temperature_K"] = "300";

    String basePath = (testDir / "mwir_realistic").string();
    ASSERT_TRUE(SpectralCubeIO::WriteENVI(cube, basePath));

    auto result = SpectralCubeIO::ReadENVI(basePath);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().width, 320);
    EXPECT_EQ(result.value().height, 240);
    EXPECT_EQ(result.value().nbands, 41);
}
