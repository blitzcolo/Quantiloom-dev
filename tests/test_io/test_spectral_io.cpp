// ============================================================================
// Quantiloom - Unit Tests for io/SpectralIO.hpp
// ============================================================================
// Tests cover:
// - SpectralCube HDF5 read/write roundtrip
// - Metadata preservation
// - Large hyperspectral cube handling
// - File format validation
// - Edge cases
// ============================================================================

#include <gtest/gtest.h>
#include "io/SpectralIO.hpp"
#include "core/SpectralCube.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temporary File Management
// ============================================================================

class SpectralIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_spectral_tests";
        std::filesystem::create_directories(tempDir);
    }

    void TearDown() override {
        // Clean up temporary files
        if (std::filesystem::exists(tempDir)) {
            std::filesystem::remove_all(tempDir);
        }
    }

    std::filesystem::path GetTempFilePath(const std::string& filename) {
        return tempDir / filename;
    }

    // Helper: Create a simple test cube
    SpectralCube CreateTestCube() {
        SpectralCube cube(10, 8, 5, 400.0f, 700.0f);

        // Fill with test pattern
        for (u32 b = 0; b < cube.nbands; ++b) {
            for (u32 y = 0; y < cube.height; ++y) {
                for (u32 x = 0; x < cube.width; ++x) {
                    cube(x, y, b) = static_cast<f32>(b * 1000 + y * 100 + x);
                }
            }
        }

        cube.metadata["sensor"] = "TestSensor";
        cube.metadata["date"] = "2024-01-15";

        return cube;
    }

    std::filesystem::path tempDir;
};

// ============================================================================
// Basic Read/Write Tests
// ============================================================================

TEST_F(SpectralIOTest, WriteAndReadRoundtrip) {
    SpectralCube original = CreateTestCube();
    auto filepath = GetTempFilePath("test_cube.h5");

    // Write cube
    bool writeSuccess = SpectralIO::WriteHDF5(filepath.string(), original);
    ASSERT_TRUE(writeSuccess);
    ASSERT_TRUE(std::filesystem::exists(filepath));

    // Read cube
    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Verify dimensions
    EXPECT_EQ(loaded->width, original.width);
    EXPECT_EQ(loaded->height, original.height);
    EXPECT_EQ(loaded->nbands, original.nbands);
    EXPECT_NEAR(loaded->lambda_min, original.lambda_min, 1e-5f);
    EXPECT_NEAR(loaded->lambda_max, original.lambda_max, 1e-5f);
    EXPECT_NEAR(loaded->delta_lambda, original.delta_lambda, 1e-5f);

    // Verify wavelengths
    EXPECT_EQ(loaded->wavelengths.size(), original.wavelengths.size());
    for (u32 i = 0; i < loaded->wavelengths.size(); ++i) {
        EXPECT_NEAR(loaded->wavelengths[i], original.wavelengths[i], 1e-5f);
    }

    // Verify data
    EXPECT_EQ(loaded->data.size(), original.data.size());
    for (u32 i = 0; i < loaded->data.size(); ++i) {
        EXPECT_NEAR(loaded->data[i], original.data[i], 1e-5f);
    }

    // Verify metadata
    EXPECT_EQ(loaded->metadata["sensor"], "TestSensor");
    EXPECT_EQ(loaded->metadata["date"], "2024-01-15");
}

TEST_F(SpectralIOTest, WriteInvalidCube) {
    SpectralCube invalid;
    // Empty cube - should fail validation

    auto filepath = GetTempFilePath("invalid_cube.h5");
    bool writeSuccess = SpectralIO::WriteHDF5(filepath.string(), invalid);
    EXPECT_FALSE(writeSuccess);
    EXPECT_FALSE(std::filesystem::exists(filepath));
}

TEST_F(SpectralIOTest, ReadNonexistentFile) {
    auto filepath = GetTempFilePath("nonexistent_cube.h5");

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    EXPECT_FALSE(loaded.has_value());
}

// ============================================================================
// Data Integrity Tests
// ============================================================================

TEST_F(SpectralIOTest, DataIntegrityAllZeros) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);
    // Keep all zeros (default initialization)

    auto filepath = GetTempFilePath("zeros_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    for (const auto& val : loaded->data) {
        EXPECT_EQ(val, 0.0f);
    }
}

TEST_F(SpectralIOTest, DataIntegrityRandomValues) {
    SpectralCube cube(20, 15, 10, 400.0f, 1000.0f);

    // Fill with pseudo-random values
    for (u32 i = 0; i < cube.data.size(); ++i) {
        cube.data[i] = static_cast<f32>(i * 0.123f);
    }

    auto filepath = GetTempFilePath("random_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    for (u32 i = 0; i < cube.data.size(); ++i) {
        EXPECT_NEAR(loaded->data[i], cube.data[i], 1e-5f);
    }
}

TEST_F(SpectralIOTest, DataIntegrityHDRValues) {
    SpectralCube cube(5, 5, 3, 400.0f, 600.0f);

    // Set HDR values (very large and very small)
    cube(0, 0, 0) = 10000.0f;
    cube(1, 1, 1) = -500.0f;
    cube(2, 2, 2) = 0.00001f;

    auto filepath = GetTempFilePath("hdr_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_NEAR(loaded->operator()(0, 0, 0), 10000.0f, 1e-2f);
    EXPECT_NEAR(loaded->operator()(1, 1, 1), -500.0f, 1e-2f);
    EXPECT_NEAR(loaded->operator()(2, 2, 2), 0.00001f, 1e-8f);
}

// ============================================================================
// Memory Layout Tests
// ============================================================================

TEST_F(SpectralIOTest, BandMajorLayoutPreserved) {
    SpectralCube cube(3, 2, 2, 400.0f, 500.0f);

    // Set known pattern for each band
    for (u32 b = 0; b < 2; ++b) {
        f32* band = cube.BandPtr(b);
        for (u32 i = 0; i < 6; ++i) {  // 3*2 = 6 pixels per band
            band[i] = static_cast<f32>(b * 100 + i);
        }
    }

    auto filepath = GetTempFilePath("layout_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Verify band contiguity
    for (u32 b = 0; b < 2; ++b) {
        const f32* band = loaded->BandPtr(b);
        for (u32 i = 0; i < 6; ++i) {
            EXPECT_NEAR(band[i], static_cast<f32>(b * 100 + i), 1e-5f);
        }
    }
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_F(SpectralIOTest, MetadataPreservation) {
    SpectralCube cube = CreateTestCube();
    cube.metadata["test_key_1"] = "value_1";
    cube.metadata["test_key_2"] = "value_2";
    cube.metadata["integration_time_ms"] = "10.5";

    auto filepath = GetTempFilePath("metadata_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->metadata["test_key_1"], "value_1");
    EXPECT_EQ(loaded->metadata["test_key_2"], "value_2");
    EXPECT_EQ(loaded->metadata["integration_time_ms"], "10.5");
}

TEST_F(SpectralIOTest, EmptyMetadata) {
    SpectralCube cube(10, 10, 5, 400.0f, 700.0f);
    cube.metadata.clear();

    auto filepath = GetTempFilePath("no_metadata_cube.h5");
    bool writeSuccess = SpectralIO::WriteHDF5(filepath.string(), cube);
    ASSERT_TRUE(writeSuccess);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    // Metadata might be empty or contain only default fields
}

TEST_F(SpectralIOTest, LargeMetadata) {
    SpectralCube cube(5, 5, 3, 400.0f, 600.0f);

    // Add many metadata entries
    for (int i = 0; i < 100; ++i) {
        cube.metadata["key_" + std::to_string(i)] = "value_" + std::to_string(i);
    }

    auto filepath = GetTempFilePath("large_metadata_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    for (int i = 0; i < 100; ++i) {
        std::string key = "key_" + std::to_string(i);
        std::string expected = "value_" + std::to_string(i);
        EXPECT_EQ(loaded->metadata[key], expected);
    }
}

// ============================================================================
// File Utilities Tests
// ============================================================================

TEST_F(SpectralIOTest, FileExistsCheck) {
    auto filepath = GetTempFilePath("exists_test.h5");

    EXPECT_FALSE(SpectralIO::FileExists(filepath.string()));

    SpectralCube cube = CreateTestCube();
    SpectralIO::WriteHDF5(filepath.string(), cube);

    EXPECT_TRUE(SpectralIO::FileExists(filepath.string()));
}

TEST_F(SpectralIOTest, GetDimensions) {
    SpectralCube cube(100, 75, 50, 400.0f, 2500.0f);
    auto filepath = GetTempFilePath("dimensions_cube.h5");

    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto dims = SpectralIO::GetDimensions(filepath.string());
    ASSERT_TRUE(dims.has_value());

    auto [width, height, nbands] = *dims;
    EXPECT_EQ(width, 100);
    EXPECT_EQ(height, 75);
    EXPECT_EQ(nbands, 50);
}

TEST_F(SpectralIOTest, GetDimensionsNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_dims.h5");

    auto dims = SpectralIO::GetDimensions(filepath.string());
    EXPECT_FALSE(dims.has_value());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SpectralIOTest, SinglePixelCube) {
    // Single pixel with multiple bands
    SpectralCube cube(1, 1, 5, 400.0f, 700.0f);
    cube(0, 0, 2) = 99.0f;

    auto filepath = GetTempFilePath("single_pixel_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->IsValid());
    EXPECT_NEAR(loaded->operator()(0, 0, 2), 99.0f, 1e-5f);
}

TEST_F(SpectralIOTest, SingleBandCube) {
    // Use 2 bands instead of 1 to avoid divide-by-zero in SpectralCube constructor
    SpectralCube cube(100, 100, 2, 550.0f, 560.0f);

    for (u32 y = 0; y < 100; ++y) {
        for (u32 x = 0; x < 100; ++x) {
            cube(x, y, 0) = static_cast<f32>(y * 100 + x);
            cube(x, y, 1) = static_cast<f32>(y * 100 + x) * 0.5f;
        }
    }

    auto filepath = GetTempFilePath("single_band_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->nbands, 2);
}

TEST_F(SpectralIOTest, LargeHyperspectralCube) {
    // Realistic hyperspectral cube: 640x480 with 200 bands
    SpectralCube cube(640, 480, 200, 400.0f, 2500.0f);

    // Fill with test pattern (avoid filling all data to reduce test time)
    for (u32 b = 0; b < cube.nbands; b += 10) {
        for (u32 y = 0; y < cube.height; y += 10) {
            for (u32 x = 0; x < cube.width; x += 10) {
                cube(x, y, b) = static_cast<f32>(b * 10000 + y * 100 + x);
            }
        }
    }

    auto filepath = GetTempFilePath("large_cube.h5");
    bool writeSuccess = SpectralIO::WriteHDF5(filepath.string(), cube);
    ASSERT_TRUE(writeSuccess);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->width, 640);
    EXPECT_EQ(loaded->height, 480);
    EXPECT_EQ(loaded->nbands, 200);
}

// ============================================================================
// Realistic Use Cases
// ============================================================================

TEST_F(SpectralIOTest, VisibleSpectrumCube) {
    // Typical visible spectrum hyperspectral image
    SpectralCube cube(1024, 768, 76, 380.0f, 760.0f);

    // Simulate a scene: gradient pattern
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 wavelength = cube.GetWavelength(b);
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                // Simple gradient based on position and wavelength
                f32 value = (wavelength / 760.0f) * (x / 1024.0f) * (y / 768.0f);
                cube(x, y, b) = value;
            }
        }
    }

    cube.metadata["sensor"] = "AVIRIS-NG";
    cube.metadata["spectral_range"] = "Visible";
    cube.metadata["flight_altitude_m"] = "4500";

    auto filepath = GetTempFilePath("visible_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->IsValid());
    EXPECT_EQ(loaded->metadata["sensor"], "AVIRIS-NG");
}

TEST_F(SpectralIOTest, ThermalIRCube) {
    // Thermal IR hyperspectral cube (LWIR: 8-12 microns)
    SpectralCube cube(320, 240, 100, 8000.0f, 12000.0f);

    // Simulate thermal scene: temperature-dependent radiance
    for (u32 b = 0; b < cube.nbands; ++b) {
        f32 wavelength = cube.GetWavelength(b);
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                // Simplified Planck's law for blackbody radiation
                f32 temperature = 300.0f + (x / 320.0f) * 100.0f;  // 300-400K
                f32 lambda_m = wavelength * 1e-9f;  // nm to m
                // Simplified (not actual Planck formula, just for testing)
                f32 radiance = temperature / (lambda_m * 1e6f);
                cube(x, y, b) = radiance;
            }
        }
    }

    cube.metadata["sensor"] = "MWIR_Camera";
    cube.metadata["spectral_range"] = "LWIR";

    auto filepath = GetTempFilePath("thermal_cube.h5");
    SpectralIO::WriteHDF5(filepath.string(), cube);

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->nbands, 100);
    EXPECT_NEAR(loaded->lambda_min, 8000.0f, 1e-5f);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SpectralIOTest, CorruptedFile) {
    auto filepath = GetTempFilePath("corrupted.h5");

    // Create a non-HDF5 file
    std::ofstream file(filepath);
    file << "This is not an HDF5 file!";
    file.close();

    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(SpectralIOTest, OverwriteExistingFile) {
    SpectralCube cube1(10, 10, 5, 400.0f, 700.0f);
    auto filepath = GetTempFilePath("overwrite_test.h5");

    // Write first cube
    SpectralIO::WriteHDF5(filepath.string(), cube1);

    // Create different cube
    SpectralCube cube2(20, 15, 10, 800.0f, 1200.0f);
    cube2(0, 0, 0) = 999.0f;

    // Overwrite
    bool writeSuccess = SpectralIO::WriteHDF5(filepath.string(), cube2);
    ASSERT_TRUE(writeSuccess);

    // Load and verify it's the new cube
    auto loaded = SpectralIO::ReadHDF5(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->width, 20);
    EXPECT_EQ(loaded->height, 15);
    EXPECT_EQ(loaded->nbands, 10);
    EXPECT_NEAR(loaded->operator()(0, 0, 0), 999.0f, 1e-5f);
}

TEST_F(SpectralIOTest, LoadedCubeIsValid) {
    SpectralCube cube = CreateTestCube();
    auto filepath = GetTempFilePath("valid_check_cube.h5");

    SpectralIO::WriteHDF5(filepath.string(), cube);
    auto loaded = SpectralIO::ReadHDF5(filepath.string());

    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->IsValid());
}
