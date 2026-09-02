// ============================================================================
// Quantiloom - Unit Tests for io/SpectralCubeIO.hpp
// ============================================================================
// Tests cover:
// - ENVI format writing and reading (BSQ, BIL, BIP)
// - Spectral EXR writing and reading (one channel per band)
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
#include <fstream>

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
// Single part, one channel per band, wavelength in the channel name: the
// spectral layout of Fichet et al. 2021. See SpectralCubeIO.cpp.

TEST_F(SpectralCubeIOTest, WriteReadEXR_Small) {
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

    // The channel name is the only record of a wavelength, and these are not
    // round numbers: 300 nm over seven intervals.
    ASSERT_EQ(loaded.wavelengths.size(), cube.wavelengths.size());
    for (u32 b = 0; b < cube.nbands; ++b) {
        EXPECT_NEAR(loaded.wavelengths[b], cube.wavelengths[b], 1e-3f) << "band " << b;
    }
}

TEST_F(SpectralCubeIOTest, WriteReadEXR_Wavelengths) {
    SpectralCube cube(8, 8, 4, 3000.0f, 5000.0f);
    cube.wavelengths = {3000.0f, 3500.0f, 4000.0f, 5000.0f};

    String path = (testDir / "test_exr_wl.exr").string();
    ASSERT_TRUE(SpectralCubeIO::WriteEXR(cube, path));

    auto result = SpectralCubeIO::ReadEXR(path);
    ASSERT_TRUE(result.has_value()) << result.error();

    const SpectralCube& loaded = result.value();
    EXPECT_EQ(loaded.nbands, 4);

    // A non-uniform axis survives, because each band carries its own wavelength
    // rather than a start and a step.
    ASSERT_EQ(loaded.wavelengths.size(), 4);
    EXPECT_NEAR(loaded.wavelengths[0], 3000.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[1], 3500.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[2], 4000.0f, 1e-2f);
    EXPECT_NEAR(loaded.wavelengths[3], 5000.0f, 1e-2f);

    // The file says which layout it is, so another renderer can tell.
    ASSERT_TRUE(loaded.metadata.contains("spectralLayoutVersion"));
    EXPECT_EQ(loaded.metadata.at("spectralLayoutVersion"), "1.0");
    ASSERT_TRUE(loaded.metadata.contains("emissiveUnits"));
}

TEST_F(SpectralCubeIOTest, EXRBandOrderComesFromWavelengthNotChannelName) {
    // An OpenEXR channel list is name-sorted, and these four names sort
    // "S0.1000nm", "S0.1650nm", "S0.400nm", "S0.550nm" -- a different order
    // from the one the cube was written in.
    SpectralCube cube(4, 4, 4, 400.0f, 1650.0f);
    cube.wavelengths = {400.0f, 550.0f, 1000.0f, 1650.0f};

    // One constant per band, so a permuted band is visible in the data.
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                cube(x, y, b) = static_cast<f32>((b + 1) * 10);
            }
        }
    }

    String path = (testDir / "test_exr_order.exr").string();
    ASSERT_TRUE(SpectralCubeIO::WriteEXR(cube, path));

    auto result = SpectralCubeIO::ReadEXR(path);
    ASSERT_TRUE(result.has_value()) << result.error();

    const SpectralCube& loaded = result.value();
    ASSERT_EQ(loaded.nbands, 4);
    for (u32 b = 0; b < 4; ++b) {
        EXPECT_NEAR(loaded.wavelengths[b], cube.wavelengths[b], 1e-2f) << "band " << b;
        EXPECT_NEAR(loaded(0, 0, b), static_cast<f32>((b + 1) * 10), 1e-5f) << "band " << b;
    }
}

TEST_F(SpectralCubeIOTest, EXRRejectsTwoBandsAtOneWavelength) {
    // Their channel names would collide in the header's name-keyed channel
    // list, and the file would come back a band short with no error anywhere.
    SpectralCube cube(4, 4, 3, 400.0f, 600.0f);
    cube.wavelengths = {400.0f, 500.0f, 500.0f};

    String path = (testDir / "test_exr_dup.exr").string();
    EXPECT_FALSE(SpectralCubeIO::WriteEXR(cube, path));
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

// ============================================================================
// TIFF
// ============================================================================
// Written by hand rather than through libtiff, so the round trip is the only
// thing standing between the byte layout and a file nobody can open.

TEST_F(SpectralCubeIOTest, WriteReadTIFF_Small) {
    SpectralCube cube(4, 3, 5, 400.0f, 800.0f);
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                cube(x, y, b) = static_cast<f32>(b * 100 + y * 10 + x) * 0.125f;
            }
        }
    }

    const std::string path = (testDir / "small.tif").string();
    ASSERT_TRUE(SpectralCubeIO::WriteGeoTIFF(cube, path));

    auto read = SpectralCubeIO::ReadGeoTIFF(path);
    ASSERT_TRUE(read.has_value()) << read.error();
    const SpectralCube& back = read.value();

    EXPECT_EQ(back.width, cube.width);
    EXPECT_EQ(back.height, cube.height);
    EXPECT_EQ(back.nbands, cube.nbands);
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                // Float32 in, float32 out, no compression: exact, or the layout
                // is wrong somewhere and "close" would hide it.
                EXPECT_FLOAT_EQ(back(x, y, b), cube(x, y, b))
                    << "at (" << x << ", " << y << ", band " << b << ")";
            }
        }
    }
}

TEST_F(SpectralCubeIOTest, TIFFCarriesTheWavelengths) {
    // The reason a cube is a cube rather than a stack of images. GDAL reads
    // them out of tag 42112 as band descriptions; this reads them back as
    // numbers, which is what a round trip has to preserve.
    SpectralCube cube(2, 2, 4, 8000.0f, 12000.0f);
    cube.wavelengths = {8000.0f, 9333.5f, 10667.25f, 12000.0f};
    std::fill(cube.data.begin(), cube.data.end(), 1.0f);

    const std::string path = (testDir / "lwir.tif").string();
    ASSERT_TRUE(SpectralCubeIO::WriteGeoTIFF(cube, path));

    auto read = SpectralCubeIO::ReadGeoTIFF(path);
    ASSERT_TRUE(read.has_value()) << read.error();
    ASSERT_EQ(read.value().wavelengths.size(), 4u);
    for (size_t b = 0; b < 4; ++b) {
        EXPECT_NEAR(read.value().wavelengths[b], cube.wavelengths[b], 1e-3f)
            << "band " << b;
    }
    EXPECT_EQ(read.value().metadata.at("wavelength_units"), "nm");
}

TEST_F(SpectralCubeIOTest, TIFFKeepsANonUniformWavelengthAxis) {
    // The case a uniform lambda_min/delta cannot express, and the one a sensor
    // with unevenly spaced bands actually is.
    SpectralCube cube(3, 2, 4, 3000.0f, 5000.0f);
    cube.wavelengths = {3000.0f, 3500.0f, 4000.0f, 5000.0f};
    for (usize i = 0; i < cube.data.size(); ++i) {
        cube.data[i] = static_cast<f32>(i);
    }

    const std::string path = (testDir / "nonuniform.tif").string();
    ASSERT_TRUE(SpectralCubeIO::WriteGeoTIFF(cube, path));

    auto read = SpectralCubeIO::ReadGeoTIFF(path);
    ASSERT_TRUE(read.has_value()) << read.error();
    EXPECT_NEAR(read.value().wavelengths[3] - read.value().wavelengths[2], 1000.0f, 1e-3f)
        << "the last gap is twice the others and has to survive";
}

TEST_F(SpectralCubeIOTest, TIFFHoldsHDRValues) {
    // A thermal band's radiance is around 1e-2 and a visible highlight can be
    // thousands. Neither may be clipped, which is the whole reason the samples
    // are float rather than the 16-bit integers a TIFF usually carries.
    SpectralCube cube(2, 2, 3, 400.0f, 700.0f);
    cube(0, 0, 0) = 1.0e-6f;
    cube(1, 0, 0) = 1.0e5f;
    cube(0, 1, 1) = -2.5f;   // a residual may be negative
    cube(1, 1, 2) = 0.0f;

    const std::string path = (testDir / "hdr.tif").string();
    ASSERT_TRUE(SpectralCubeIO::WriteGeoTIFF(cube, path));

    auto read = SpectralCubeIO::ReadGeoTIFF(path);
    ASSERT_TRUE(read.has_value()) << read.error();
    EXPECT_FLOAT_EQ(read.value()(0, 0, 0), 1.0e-6f);
    EXPECT_FLOAT_EQ(read.value()(1, 0, 0), 1.0e5f);
    EXPECT_FLOAT_EQ(read.value()(0, 1, 1), -2.5f);
}

TEST_F(SpectralCubeIOTest, ReadNonexistentTIFF) {
    EXPECT_FALSE(SpectralCubeIO::ReadGeoTIFF((testDir / "nope.tif").string()).has_value());
}

TEST_F(SpectralCubeIOTest, TIFFRejectsWhatItCannotRead) {
    // Saying which of the several ways a TIFF can be unreadable this one is
    // beats "failed to read": a compressed file and an integer file are
    // different problems with different answers.
    const std::string path = (testDir / "notatiff.tif").string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "this is not a tiff at all, it is a sentence";
    }
    auto read = SpectralCubeIO::ReadGeoTIFF(path);
    ASSERT_FALSE(read.has_value());
    EXPECT_NE(read.error().find("Not a TIFF"), std::string::npos) << read.error();

    // A BigTIFF header, which is a different format rather than a bigger one.
    const std::string big = (testDir / "big.tif").string();
    {
        std::ofstream f(big, std::ios::binary);
        const unsigned char header[] = {'I', 'I', 43, 0, 8, 0, 0, 0};
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
    }
    auto bigRead = SpectralCubeIO::ReadGeoTIFF(big);
    ASSERT_FALSE(bigRead.has_value());
    EXPECT_NE(bigRead.error().find("BigTIFF"), std::string::npos) << bigRead.error();
}
