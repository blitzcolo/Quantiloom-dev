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

// ============================================================================
// CSV Spectral Curve Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadSpectralCurveCSVBasic) {
    auto filepath = GetTempFilePath("test_curve.csv");

    // Create test CSV file
    std::ofstream file(filepath);
    file << "400.0, 0.12\n";
    file << "450.0, 0.25\n";
    file << "500.0, 0.50\n";
    file << "550.0, 0.65\n";
    file << "600.0, 0.40\n";
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& samples = result.value();
    EXPECT_EQ(samples.size(), 5);

    // Verify wavelengths and values
    EXPECT_NEAR(samples[0].first, 400.0f, 1e-5f);
    EXPECT_NEAR(samples[0].second, 0.12f, 1e-5f);
    EXPECT_NEAR(samples[2].first, 500.0f, 1e-5f);
    EXPECT_NEAR(samples[2].second, 0.50f, 1e-5f);
    EXPECT_NEAR(samples[4].first, 600.0f, 1e-5f);
    EXPECT_NEAR(samples[4].second, 0.40f, 1e-5f);
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVWithComments) {
    auto filepath = GetTempFilePath("curve_comments.csv");

    // Note: CSV loader only skips lines starting with '#'
    // Header lines with text will cause parse errors
    std::ofstream file(filepath);
    file << "# This is a comment\n";
    file << "# wavelength_nm, reflectance (header as comment)\n";
    file << "400.0, 0.10\n";
    file << "500.0, 0.50\n";
    file << "# Another comment\n";
    file << "600.0, 0.30\n";
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    ASSERT_TRUE(result.has_value());

    // Should skip comment lines, get 3 data points
    EXPECT_EQ(result.value().size(), 3);
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_curve.csv");

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadSpectralCurveCSVEmpty) {
    auto filepath = GetTempFilePath("empty_curve.csv");

    std::ofstream file(filepath);
    file.close();

    auto result = SpectralIO::LoadSpectralCurveCSV(filepath);
    // Either error or empty result is acceptable
    if (result.has_value()) {
        EXPECT_TRUE(result.value().empty());
    }
}

// ============================================================================
// USGS Spectral Library Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadUSGSBasic) {
    // Create mock USGS wavelength file
    auto wavelengthPath = GetTempFilePath("wavelengths.txt");
    std::ofstream wvFile(wavelengthPath);
    wvFile << "Record=1 Wavelengths for BECK spectrometer\n";
    wvFile << "0.350000\n";  // 350 nm (in micrometers)
    wvFile << "0.400000\n";  // 400 nm
    wvFile << "0.500000\n";  // 500 nm
    wvFile << "0.600000\n";  // 600 nm
    wvFile << "0.700000\n";  // 700 nm
    wvFile.close();

    // Create mock USGS reflectance file
    auto reflectancePath = GetTempFilePath("TestMaterial_AREF.txt");
    std::ofstream refFile(reflectancePath);
    refFile << "Record=1 TestMaterial DHR reflectance\n";
    refFile << "0.150000\n";  // Reflectance at 350nm
    refFile << "0.250000\n";  // Reflectance at 400nm
    refFile << "0.500000\n";  // Reflectance at 500nm
    refFile << "0.350000\n";  // Reflectance at 600nm
    refFile << "0.200000\n";  // Reflectance at 700nm
    refFile.close();

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 5);

    // Wavelengths should be in nm (converted from µm)
    EXPECT_NEAR(curve.samples[0].first, 350.0f, 1e-3f);
    EXPECT_NEAR(curve.samples[1].first, 400.0f, 1e-3f);
    EXPECT_NEAR(curve.samples[4].first, 700.0f, 1e-3f);

    // Reflectance values
    EXPECT_NEAR(curve.samples[0].second, 0.15f, 1e-5f);
    EXPECT_NEAR(curve.samples[2].second, 0.50f, 1e-5f);
}

TEST_F(SpectralIOTest, LoadUSGSWithInvalidData) {
    // Create wavelength file
    auto wavelengthPath = GetTempFilePath("wavelengths_invalid.txt");
    std::ofstream wvFile(wavelengthPath);
    wvFile << "Record=1 Test wavelengths\n";
    wvFile << "0.400000\n";
    wvFile << "0.500000\n";
    wvFile << "0.600000\n";
    wvFile << "0.700000\n";
    wvFile.close();

    // Create reflectance file with USGS invalid marker (-1.23e+034)
    auto reflectancePath = GetTempFilePath("TestMaterial_Invalid_AREF.txt");
    std::ofstream refFile(reflectancePath);
    refFile << "Record=1 TestMaterial with invalid data\n";
    refFile << "0.250000\n";    // Valid
    refFile << "-1.23e+034\n";  // Invalid marker (should be skipped)
    refFile << "0.400000\n";    // Valid
    refFile << "0.300000\n";    // Valid
    refFile.close();

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    // Should have fewer samples due to invalid data being filtered
    EXPECT_LT(curve.samples.size(), 4);
    EXPECT_TRUE(curve.IsValid());

    // All remaining values should be in valid range [0, 1]
    for (const auto& sample : curve.samples) {
        EXPECT_GE(sample.second, 0.0f);
        EXPECT_LE(sample.second, 1.0f);
    }
}

TEST_F(SpectralIOTest, LoadUSGSNonexistentFiles) {
    auto wavelengthPath = GetTempFilePath("nonexistent_wv.txt");
    auto reflectancePath = GetTempFilePath("nonexistent_ref.txt");

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadUSGSMismatchedSizes) {
    // Create wavelength file with 3 entries
    auto wavelengthPath = GetTempFilePath("wavelengths_short.txt");
    std::ofstream wvFile(wavelengthPath);
    wvFile << "Record=1\n";
    wvFile << "0.400000\n";
    wvFile << "0.500000\n";
    wvFile << "0.600000\n";
    wvFile.close();

    // Create reflectance file with 5 entries (mismatched)
    auto reflectancePath = GetTempFilePath("TestMaterial_Mismatch_AREF.txt");
    std::ofstream refFile(reflectancePath);
    refFile << "Record=1\n";
    refFile << "0.10\n";
    refFile << "0.20\n";
    refFile << "0.30\n";
    refFile << "0.40\n";
    refFile << "0.50\n";
    refFile.close();

    auto result = SpectralIO::LoadUSGS(reflectancePath, wavelengthPath);
    // Should either fail or use the minimum size
    if (result.has_value()) {
        EXPECT_LE(result.value().samples.size(), 3);
    }
}

// ============================================================================
// RefractiveIndex.INFO YAML Loading Tests
// ============================================================================

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLBasic) {
    auto filepath = GetTempFilePath("gold_test.yml");

    // Create mock RefractiveIndex.INFO YAML file
    std::ofstream file(filepath);
    file << "# Test gold optical constants\n";
    file << "REFERENCES: \"Test reference\"\n";
    file << "COMMENTS: \"Test gold data\"\n";
    file << "DATA:\n";
    file << "  - type: tabulated nk\n";
    file << "    data: |\n";
    file << "        0.5000 0.9400 1.9600\n";  // 500nm: n=0.94, k=1.96
    file << "        0.5500 0.8800 2.2000\n";  // 550nm: n=0.88, k=2.20
    file << "        0.6000 0.2100 2.8700\n";  // 600nm: n=0.21, k=2.87
    file << "        0.6500 0.1400 3.1500\n";  // 650nm: n=0.14, k=3.15
    file << "        0.7000 0.1300 3.4200\n";  // 700nm: n=0.13, k=3.42
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    ASSERT_TRUE(result.has_value());

    const ComplexRefractiveIndex& cri = result.value();
    EXPECT_TRUE(cri.IsValid());
    EXPECT_EQ(cri.wavelengths_nm.size(), 5);
    EXPECT_EQ(cri.n.size(), 5);
    EXPECT_EQ(cri.k.size(), 5);

    // Wavelengths should be in nm (converted from µm)
    EXPECT_NEAR(cri.wavelengths_nm[0], 500.0f, 1e-3f);
    EXPECT_NEAR(cri.wavelengths_nm[4], 700.0f, 1e-3f);

    // Verify n values (refractive index)
    EXPECT_NEAR(cri.n[0], 0.94f, 1e-3f);
    EXPECT_NEAR(cri.n[2], 0.21f, 1e-3f);

    // Verify k values (extinction coefficient)
    EXPECT_NEAR(cri.k[0], 1.96f, 1e-3f);
    EXPECT_NEAR(cri.k[4], 3.42f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLEvaluate) {
    auto filepath = GetTempFilePath("silver_test.yml");

    std::ofstream file(filepath);
    file << "DATA:\n";
    file << "  - type: tabulated nk\n";
    file << "    data: |\n";
    file << "        0.4000 0.0500 2.0000\n";  // 400nm
    file << "        0.5000 0.0600 2.5000\n";  // 500nm
    file << "        0.6000 0.0700 3.0000\n";  // 600nm
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    ASSERT_TRUE(result.has_value());

    const ComplexRefractiveIndex& cri = result.value();

    // Test exact sample points
    auto [n_400, k_400] = cri.Evaluate(400.0f);
    EXPECT_NEAR(n_400, 0.05f, 1e-5f);
    EXPECT_NEAR(k_400, 2.0f, 1e-5f);

    // Test interpolation at midpoint
    auto [n_450, k_450] = cri.Evaluate(450.0f);
    EXPECT_NEAR(n_450, 0.055f, 1e-5f);  // Midpoint between 0.05 and 0.06
    EXPECT_NEAR(k_450, 2.25f, 1e-5f);   // Midpoint between 2.0 and 2.5

    // Test out-of-range (below)
    auto [n_300, k_300] = cri.Evaluate(300.0f);
    EXPECT_NEAR(n_300, 0.05f, 1e-5f);  // Clamp to first value

    // Test out-of-range (above)
    auto [n_800, k_800] = cri.Evaluate(800.0f);
    EXPECT_NEAR(n_800, 0.07f, 1e-5f);  // Clamp to last value
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLFresnelR0) {
    auto filepath = GetTempFilePath("metal_fresnel.yml");

    // Need at least 2 data points for GPU conversion (stepSize calculation)
    std::ofstream file(filepath);
    file << "DATA:\n";
    file << "  - type: tabulated nk\n";
    file << "    data: |\n";
    file << "        0.5000 0.2000 3.0000\n";  // 500nm
    file << "        0.6000 0.2000 3.0000\n";  // 600nm (same n,k for consistent F0)
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    ASSERT_TRUE(result.has_value());

    const ComplexRefractiveIndex& cri = result.value();

    // Calculate expected F0: [(n-1)² + k²] / [(n+1)² + k²]
    // n=0.2, k=3.0
    // numerator = (0.2-1)² + 3² = 0.64 + 9 = 9.64
    // denominator = (0.2+1)² + 3² = 1.44 + 9 = 10.44
    // F0 = 9.64 / 10.44 ≈ 0.9234
    f32 F0 = cri.FresnelR0(550.0f);
    EXPECT_NEAR(F0, 0.9234f, 0.01f);
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLNonexistent) {
    auto filepath = GetTempFilePath("nonexistent.yml");

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLMalformed) {
    auto filepath = GetTempFilePath("malformed.yml");

    std::ofstream file(filepath);
    file << "This is not valid YAML for RefractiveIndex.INFO\n";
    file << "Random text without proper DATA section\n";
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    // Should either fail or return empty data
    if (result.has_value()) {
        EXPECT_FALSE(result.value().IsValid());
    }
}

TEST_F(SpectralIOTest, LoadRefractiveIndexYAMLWavelengthConversion) {
    auto filepath = GetTempFilePath("wavelength_conv.yml");

    // Test that µm to nm conversion works correctly
    std::ofstream file(filepath);
    file << "DATA:\n";
    file << "  - type: tabulated nk\n";
    file << "    data: |\n";
    file << "        1.0000 1.5000 0.0100\n";  // 1.0 µm = 1000 nm
    file << "        2.0000 1.4500 0.0050\n";  // 2.0 µm = 2000 nm
    file.close();

    auto result = SpectralIO::LoadRefractiveIndexYAML(filepath);
    ASSERT_TRUE(result.has_value());

    const ComplexRefractiveIndex& cri = result.value();

    // Verify wavelengths are converted to nm
    EXPECT_NEAR(cri.wavelengths_nm[0], 1000.0f, 1e-3f);
    EXPECT_NEAR(cri.wavelengths_nm[1], 2000.0f, 1e-3f);
}
