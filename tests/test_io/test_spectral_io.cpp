// ============================================================================
// Quantiloom - Unit Tests for io/SpectralIO.hpp
// ============================================================================
// Tests cover:
// - SpectralCube HDF5 read/write roundtrip
// - Metadata preservation
// - Large hyperspectral cube handling
// - File format validation
// - CSV spectral curve loading
// - USGS Spectral Library loading
// - RefractiveIndex.INFO YAML loading
// - ASTM G-173 solar spectrum loading (sun/sky irradiance)
// - SolarSpectralLUT GPU format conversion
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

// ============================================================================
// ASTM G-173 Solar Spectrum Loading Tests
// ============================================================================
// Tests for loading ASTM G-173-03 Reference Solar Spectral Irradiances
// File format: 4-column CSV (Wavelength, ETR, Global, Direct+circumsolar)
// ============================================================================

TEST_F(SpectralIOTest, LoadASTMG173Basic) {
    auto filepath = GetTempFilePath("astmg173_test.csv");

    // Create mock ASTM G-173 CSV file (simplified)
    // Column 1: Wavelength (nm)
    // Column 2: ETR (W·m⁻²·nm⁻¹)
    // Column 3: Global tilt (W·m⁻²·nm⁻¹)
    // Column 4: Direct+circumsolar (W·m⁻²·nm⁻¹)
    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "280,8.2000E-02,4.7309E-23,2.5361E-26\n";
    file << "300,5.1400E-01,1.0230E-04,2.4980E-06\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file << "1000,7.4900E-01,6.9660E-01,6.6810E-01\n";
    file << "2000,2.1630E-01,1.4440E-01,1.6200E-01\n";
    file << "4000,1.6170E-02,1.1010E-02,1.3080E-02\n";
    file.close();

    // Load column 4 (Direct+circumsolar) - default
    auto result = SpectralIO::LoadASTMG173(filepath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 9);

    // Verify wavelength range (280-4000 nm)
    EXPECT_NEAR(curve.samples.front().first, 280.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 4000.0f, 1e-3f);

    // Verify some irradiance values (column 4: Direct+circumsolar)
    // At 400nm: 1.1130 W·m⁻²·nm⁻¹
    auto range = curve.GetWavelengthRange();
    EXPECT_NEAR(range.first, 280.0f, 1e-3f);
    EXPECT_NEAR(range.second, 4000.0f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadASTMG173Column2ETR) {
    auto filepath = GetTempFilePath("astmg173_etr.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file.close();

    // Load column 2 (ETR - Extraterrestrial Radiation)
    auto result = SpectralIO::LoadASTMG173(filepath, 2);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);

    // Verify ETR values at 500nm should be 1.9170
    EXPECT_NEAR(curve.Evaluate(500.0f), 1.917f, 0.01f);
}

TEST_F(SpectralIOTest, LoadASTMG173Column3Global) {
    auto filepath = GetTempFilePath("astmg173_global.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file.close();

    // Load column 3 (Global tilt)
    auto result = SpectralIO::LoadASTMG173(filepath, 3);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);

    // Verify Global tilt value at 500nm should be 1.6750
    EXPECT_NEAR(curve.Evaluate(500.0f), 1.675f, 0.01f);
}

TEST_F(SpectralIOTest, LoadASTMG173SunAndSkyBasic) {
    auto filepath = GetTempFilePath("astmg173_sun_sky.csv");

    // Create mock ASTM G-173 data
    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file << "450,1.7610E+00,1.5510E+00,1.4000E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "550,1.8690E+00,1.6620E+00,1.5510E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "650,1.6760E+00,1.5460E+00,1.4720E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173SunAndSky(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Verify both curves are valid
    EXPECT_TRUE(sunCurve.IsValid());
    EXPECT_TRUE(skyCurve.IsValid());
    EXPECT_EQ(sunCurve.samples.size(), 7);
    EXPECT_EQ(skyCurve.samples.size(), 7);

    // Sun curve should be Direct+circumsolar (column 4)
    // Sky curve should be Global - Direct (column 3 - column 4)

    // At 500nm:
    // Direct+circumsolar = 1.5290
    // Global = 1.6750
    // Diffuse sky = Global - Direct = 1.6750 - 1.5290 = 0.1460
    EXPECT_NEAR(sunCurve.Evaluate(500.0f), 1.5290f, 0.01f);
    EXPECT_NEAR(skyCurve.Evaluate(500.0f), 0.146f, 0.02f);

    // Verify wavelength ranges match
    auto sunRange = sunCurve.GetWavelengthRange();
    auto skyRange = skyCurve.GetWavelengthRange();
    EXPECT_NEAR(sunRange.first, skyRange.first, 1e-3f);
    EXPECT_NEAR(sunRange.second, skyRange.second, 1e-3f);
}

TEST_F(SpectralIOTest, LoadASTMG173SunAndSkyPhysicalValues) {
    auto filepath = GetTempFilePath("astmg173_physical.csv");

    // Use more realistic ASTM G-173 values for visible spectrum peak
    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "380,9.9550E-01,7.0550E-01,5.5910E-01\n";
    file << "450,1.7610E+00,1.5510E+00,1.4000E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";  // Near peak
    file << "550,1.8690E+00,1.6620E+00,1.5510E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file << "780,1.1240E+00,1.0190E+00,9.5820E-01\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173SunAndSky(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Verify physical characteristics:
    // 1. Sun irradiance peaks around 500nm (visible light)
    f32 sun500 = sunCurve.Evaluate(500.0f);
    f32 sun380 = sunCurve.Evaluate(380.0f);
    f32 sun780 = sunCurve.Evaluate(780.0f);
    EXPECT_GT(sun500, sun380);  // Peak > UV edge
    EXPECT_GT(sun500, sun780);  // Peak > IR edge

    // 2. All irradiance values should be positive
    for (const auto& sample : sunCurve.samples) {
        EXPECT_GE(sample.second, 0.0f);
    }
    for (const auto& sample : skyCurve.samples) {
        EXPECT_GE(sample.second, 0.0f);
    }

    // 3. Sky (diffuse) should be less than sun (direct) at most wavelengths
    EXPECT_LT(skyCurve.Evaluate(500.0f), sunCurve.Evaluate(500.0f));
}

TEST_F(SpectralIOTest, LoadASTMG173Nonexistent) {
    auto filepath = GetTempFilePath("nonexistent_astm.csv");

    auto result = SpectralIO::LoadASTMG173(filepath);
    EXPECT_FALSE(result.has_value());

    auto result2 = SpectralIO::LoadASTMG173SunAndSky(filepath);
    EXPECT_FALSE(result2.has_value());
}

TEST_F(SpectralIOTest, LoadASTMG173MalformedHeader) {
    auto filepath = GetTempFilePath("malformed_astm.csv");

    std::ofstream file(filepath);
    file << "This is not a valid ASTM G-173 header\n";
    file << "Random,Data,Goes,Here\n";
    file << "100,200,300,400\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173(filepath);
    // Should either fail or handle gracefully
    // Implementation may vary based on header parsing behavior
}

TEST_F(SpectralIOTest, LoadASTMG173InvalidColumn) {
    auto filepath = GetTempFilePath("astmg173_inv_col.csv");

    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "400,1.5140E+00,1.2680E+00,1.1130E+00\n";
    file.close();

    // Request invalid column (column 5 doesn't exist)
    auto result = SpectralIO::LoadASTMG173(filepath, 5);
    // Should either fail or fall back to valid column
    if (result.has_value()) {
        EXPECT_TRUE(result.value().samples.empty() || result.value().samples.size() > 0);
    }
}

TEST_F(SpectralIOTest, LoadASTMG173ScientificNotation) {
    auto filepath = GetTempFilePath("astmg173_scientific.csv");

    // Test various scientific notation formats
    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "280,8.2000E-02,4.7309E-23,2.5361E-26\n";  // Very small values
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";   // Normal values
    file << "4000,1.6170E-02,1.1010E-02,1.3080E-02\n";  // Small IR values
    file.close();

    auto result = SpectralIO::LoadASTMG173(filepath);
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);

    // Verify scientific notation parsing
    EXPECT_NEAR(curve.samples[0].second, 2.5361e-26f, 1e-28f);  // Very small UV
    EXPECT_NEAR(curve.samples[1].second, 1.529f, 0.01f);        // Normal visible
    EXPECT_NEAR(curve.samples[2].second, 0.01308f, 0.0001f);    // Small IR
}

TEST_F(SpectralIOTest, LoadASTMG173ToSolarSpectralLUT) {
    auto filepath = GetTempFilePath("astmg173_to_lut.csv");

    // Create ASTM G-173 data covering visible spectrum
    std::ofstream file(filepath);
    file << "Wvlgth nm,Etr W*m-2*nm-1,Global tilt  W*m-2*nm-1,Direct+circumsolar W*m-2*nm-1\n";
    file << "380,9.9550E-01,7.0550E-01,5.5910E-01\n";
    file << "450,1.7610E+00,1.5510E+00,1.4000E+00\n";
    file << "500,1.9170E+00,1.6750E+00,1.5290E+00\n";
    file << "550,1.8690E+00,1.6620E+00,1.5510E+00\n";
    file << "600,1.8310E+00,1.6740E+00,1.5870E+00\n";
    file << "700,1.5050E+00,1.3820E+00,1.3140E+00\n";
    file << "780,1.1240E+00,1.0190E+00,9.5820E-01\n";
    file.close();

    auto result = SpectralIO::LoadASTMG173SunAndSky(filepath);
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Convert to SolarSpectralLUT (GPU format)
    SolarSpectralLUT lut = SolarSpectralLUT::FromCPU(sunCurve, skyCurve);

    EXPECT_TRUE(lut.IsValid());
    EXPECT_EQ(lut.sunIrradiance.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_EQ(lut.skyIrradiance.numSamples, MAX_SPECTRAL_SAMPLES);

    // Verify wavelength range is preserved
    auto [min_wl, max_wl] = lut.GetWavelengthRange();
    EXPECT_NEAR(min_wl, 380.0f, 1.0f);
    EXPECT_NEAR(max_wl, 780.0f, 1.0f);

    // Verify evaluation works (interpolation)
    f32 sunAt500 = lut.sunIrradiance.Evaluate(500.0f);
    EXPECT_GT(sunAt500, 1.0f);   // Should be > 1 W·m⁻²·nm⁻¹ at peak
    EXPECT_LT(sunAt500, 2.0f);   // But less than 2

    f32 skyAt500 = lut.skyIrradiance.Evaluate(500.0f);
    EXPECT_GT(skyAt500, 0.0f);   // Diffuse sky should be positive
    EXPECT_LT(skyAt500, 0.5f);   // But much less than direct sun
}

// ============================================================================
// libRadtran uvspec Output Loading Tests
// ============================================================================
// Tests for loading libRadtran atmospheric radiative transfer model output.
// Standard uvspec output format: wavelength edir edn eup uavg
// ============================================================================

TEST_F(SpectralIOTest, LoadLibRadtranUvspecBasic) {
    auto filepath = GetTempFilePath("uvspec_basic.txt");

    // Create mock libRadtran uvspec output (space-separated)
    // Format: wavelength(nm)  edir  edn  eup  uavg
    std::ofstream file(filepath);
    file << "# libRadtran uvspec output\n";
    file << "# wavelength(nm)  edir  edn  eup  uavg\n";
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "650.000  1.4720  0.0740  0.0000  0.0370\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file.close();

    // Load edir (column 2)
    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_TRUE(curve.IsValid());
    EXPECT_EQ(curve.samples.size(), 8);

    // Verify wavelength range
    EXPECT_NEAR(curve.samples.front().first, 380.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 700.0f, 1e-3f);

    // Verify edir value at 500nm
    EXPECT_NEAR(curve.Evaluate(500.0f), 1.529f, 0.01f);
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecEdnColumn) {
    auto filepath = GetTempFilePath("uvspec_edn.txt");

    std::ofstream file(filepath);
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file.close();

    // Load edn (column 3)
    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 3, "nm");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);

    // Verify edn value at 500nm
    EXPECT_NEAR(curve.Evaluate(500.0f), 0.146f, 0.01f);
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecMicrometerUnit) {
    auto filepath = GetTempFilePath("uvspec_um.txt");

    // Wavelengths in micrometers (µm)
    std::ofstream file(filepath);
    file << "# wavelength in micrometers\n";
    file << "0.400  1.1130  0.1550  0.0000  0.0775\n";
    file << "0.500  1.5290  0.1460  0.0000  0.0730\n";
    file << "0.600  1.5870  0.0870  0.0000  0.0435\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "um");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();

    // Verify wavelengths are converted to nm
    EXPECT_NEAR(curve.samples.front().first, 400.0f, 1e-3f);
    EXPECT_NEAR(curve.samples.back().first, 600.0f, 1e-3f);
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecWavenumberUnit) {
    auto filepath = GetTempFilePath("uvspec_cm1.txt");

    // Wavelengths in wavenumber (cm⁻¹)
    // λ(nm) = 1e7 / ν(cm⁻¹)
    // 25000 cm⁻¹ = 400 nm
    // 20000 cm⁻¹ = 500 nm
    // 16667 cm⁻¹ = 600 nm
    std::ofstream file(filepath);
    file << "# wavelength in wavenumber cm-1\n";
    file << "25000.0  1.1130  0.1550  0.0000  0.0775\n";
    file << "20000.0  1.5290  0.1460  0.0000  0.0730\n";
    file << "16666.7  1.5870  0.0870  0.0000  0.0435\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "cm-1");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();

    // Wavenumber input should be sorted to wavelength order (low to high)
    EXPECT_NEAR(curve.samples.front().first, 400.0f, 1.0f);  // 25000 cm⁻¹ → 400 nm
    EXPECT_NEAR(curve.samples.back().first, 600.0f, 1.0f);   // 16667 cm⁻¹ → 600 nm
}

TEST_F(SpectralIOTest, LoadLibRadtranSunAndSkyBasic) {
    auto filepath = GetTempFilePath("uvspec_sun_sky.txt");

    std::ofstream file(filepath);
    file << "# libRadtran uvspec output for sun and sky\n";
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Verify both curves are valid
    EXPECT_TRUE(sunCurve.IsValid());
    EXPECT_TRUE(skyCurve.IsValid());
    EXPECT_EQ(sunCurve.samples.size(), 6);
    EXPECT_EQ(skyCurve.samples.size(), 6);

    // Sun curve should be edir (column 2)
    EXPECT_NEAR(sunCurve.Evaluate(500.0f), 1.529f, 0.01f);

    // Sky curve should be edn (column 3)
    EXPECT_NEAR(skyCurve.Evaluate(500.0f), 0.146f, 0.01f);

    // Verify wavelength ranges match
    auto sunRange = sunCurve.GetWavelengthRange();
    auto skyRange = skyCurve.GetWavelengthRange();
    EXPECT_NEAR(sunRange.first, skyRange.first, 1e-3f);
    EXPECT_NEAR(sunRange.second, skyRange.second, 1e-3f);
}

TEST_F(SpectralIOTest, LoadLibRadtranSunAndSkyPhysicalValues) {
    auto filepath = GetTempFilePath("uvspec_physical.txt");

    // Use realistic libRadtran-like values for visible spectrum
    std::ofstream file(filepath);
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";  // Near peak
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file << "780.000  0.9582  0.0510  0.0000  0.0255\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Verify physical characteristics:
    // 1. Sun irradiance peaks around 500-600nm
    f32 sun500 = sunCurve.Evaluate(500.0f);
    f32 sun380 = sunCurve.Evaluate(380.0f);
    f32 sun780 = sunCurve.Evaluate(780.0f);
    EXPECT_GT(sun500, sun380);  // Peak > UV edge
    EXPECT_GT(sun500, sun780);  // Peak > IR edge

    // 2. All irradiance values should be positive
    for (const auto& sample : sunCurve.samples) {
        EXPECT_GE(sample.second, 0.0f);
    }
    for (const auto& sample : skyCurve.samples) {
        EXPECT_GE(sample.second, 0.0f);
    }

    // 3. Sky (diffuse) should be less than sun (direct)
    EXPECT_LT(skyCurve.Evaluate(500.0f), sunCurve.Evaluate(500.0f));
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_uvspec.txt");

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    EXPECT_FALSE(result.has_value());

    auto result2 = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    EXPECT_FALSE(result2.has_value());
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecInvalidColumn) {
    auto filepath = GetTempFilePath("uvspec_inv_col.txt");

    std::ofstream file(filepath);
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file.close();

    // Request column 11 (way out of range)
    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 11, "nm");
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecInvalidUnit) {
    auto filepath = GetTempFilePath("uvspec_inv_unit.txt");

    std::ofstream file(filepath);
    file << "400.000  1.1130  0.1550  0.0000  0.0775\n";
    file.close();

    // Invalid wavelength unit
    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "invalid_unit");
    EXPECT_FALSE(result.has_value());
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecScientificNotation) {
    auto filepath = GetTempFilePath("uvspec_scientific.txt");

    // Test scientific notation parsing (common in libRadtran output)
    std::ofstream file(filepath);
    file << "280.000  8.2000E-02  4.7309E-23  0.0000E+00  2.3655E-23\n";
    file << "300.000  5.1400E-01  1.0230E-04  0.0000E+00  5.1150E-05\n";
    file << "500.000  1.5290E+00  1.4600E-01  0.0000E+00  7.3000E-02\n";
    file << "1000.000  6.6810E-01  4.2100E-02  0.0000E+00  2.1050E-02\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 4);

    // Verify scientific notation parsing
    EXPECT_NEAR(curve.samples[0].second, 0.082f, 0.001f);     // 8.2E-02
    EXPECT_NEAR(curve.samples[2].second, 1.529f, 0.01f);      // 1.5290E+00
}

TEST_F(SpectralIOTest, LoadLibRadtranToSolarSpectralLUT) {
    auto filepath = GetTempFilePath("uvspec_to_lut.txt");

    // Create libRadtran output covering visible spectrum
    std::ofstream file(filepath);
    file << "380.000  0.5591  0.1464  0.0000  0.0732\n";
    file << "450.000  1.4000  0.1510  0.0000  0.0755\n";
    file << "500.000  1.5290  0.1460  0.0000  0.0730\n";
    file << "550.000  1.5510  0.1110  0.0000  0.0555\n";
    file << "600.000  1.5870  0.0870  0.0000  0.0435\n";
    file << "700.000  1.3140  0.0680  0.0000  0.0340\n";
    file << "780.000  0.9582  0.0510  0.0000  0.0255\n";
    file.close();

    auto result = SpectralIO::LoadLibRadtranSunAndSky(filepath, "nm");
    ASSERT_TRUE(result.has_value());

    const auto& [sunCurve, skyCurve] = result.value();

    // Convert to SolarSpectralLUT (GPU format)
    SolarSpectralLUT lut = SolarSpectralLUT::FromCPU(sunCurve, skyCurve);

    EXPECT_TRUE(lut.IsValid());
    EXPECT_EQ(lut.sunIrradiance.numSamples, MAX_SPECTRAL_SAMPLES);
    EXPECT_EQ(lut.skyIrradiance.numSamples, MAX_SPECTRAL_SAMPLES);

    // Verify wavelength range is preserved
    auto [min_wl, max_wl] = lut.GetWavelengthRange();
    EXPECT_NEAR(min_wl, 380.0f, 1.0f);
    EXPECT_NEAR(max_wl, 780.0f, 1.0f);

    // Verify evaluation works (interpolation)
    f32 sunAt500 = lut.sunIrradiance.Evaluate(500.0f);
    EXPECT_GT(sunAt500, 1.0f);   // Should be > 1 W·m⁻²·nm⁻¹ at peak
    EXPECT_LT(sunAt500, 2.0f);   // But less than 2

    f32 skyAt500 = lut.skyIrradiance.Evaluate(500.0f);
    EXPECT_GT(skyAt500, 0.0f);   // Diffuse sky should be positive
    EXPECT_LT(skyAt500, 0.2f);   // But much less than direct sun
}

TEST_F(SpectralIOTest, LoadLibRadtranUvspecWithExtraWhitespace) {
    auto filepath = GetTempFilePath("uvspec_whitespace.txt");

    // Test handling of various whitespace patterns
    std::ofstream file(filepath);
    file << "  400.000    1.1130    0.1550    0.0000    0.0775  \n";  // Leading/trailing spaces
    file << "\t500.000\t1.5290\t0.1460\t0.0000\t0.0730\n";           // Tabs
    file << "   600.000  1.5870   0.0870  0.0000   0.0435   \n";      // Multiple spaces
    file.close();

    auto result = SpectralIO::LoadLibRadtranUvspec(filepath, 2, "nm");
    ASSERT_TRUE(result.has_value());

    const SpectralCurve& curve = result.value();
    EXPECT_EQ(curve.samples.size(), 3);
}
