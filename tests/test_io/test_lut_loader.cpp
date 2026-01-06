// ============================================================================
// Quantiloom - Unit Tests for io/LUTLoader.hpp
// ============================================================================
// Tests cover:
// - AtmosphereLUT TOML read/write roundtrip
// - Interpolation correctness
// - Validation checks
// - Metadata preservation
// - File existence checks
// - Edge cases
// ============================================================================

#include <gtest/gtest.h>
#include "io/LUTLoader.hpp"
#include "core/LUT.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temporary File Management
// ============================================================================

class LUTLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_lut_tests";
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

    // Helper: Create a simple test LUT
    AtmosphereLUT CreateTestLUT() {
        AtmosphereLUT lut;

        // Wavelengths from 400nm to 2500nm (5 samples)
        lut.wavelengths = {400.0f, 800.0f, 1200.0f, 1600.0f, 2500.0f};

        // Solar irradiance (decreasing with wavelength - simplified model)
        lut.solar_irradiance = {1800.0f, 1400.0f, 1000.0f, 600.0f, 200.0f};

        // Sky radiance (Rayleigh scattering - stronger at short wavelengths)
        lut.sky_radiance = {80.0f, 20.0f, 5.0f, 2.0f, 0.5f};

        // Transmittance (atmospheric absorption)
        lut.transmittance = {0.85f, 0.90f, 0.80f, 0.75f, 0.70f};

        // Metadata
        lut.metadata["solar_zenith_deg"] = "30";
        lut.metadata["visibility_km"] = "23";
        lut.metadata["model"] = "US_Standard";

        return lut;
    }

    std::filesystem::path tempDir;
};

// ============================================================================
// Basic Read/Write Tests
// ============================================================================

TEST_F(LUTLoaderTest, SaveAndLoadRoundtrip) {
    AtmosphereLUT original = CreateTestLUT();
    auto filepath = GetTempFilePath("test_lut.toml");

    // Save LUT
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), original);
    ASSERT_TRUE(saveSuccess);
    ASSERT_TRUE(std::filesystem::exists(filepath));

    // Load LUT
    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    // Verify data integrity
    EXPECT_EQ(loaded->wavelengths.size(), original.wavelengths.size());
    for (size_t i = 0; i < original.wavelengths.size(); ++i) {
        EXPECT_NEAR(loaded->wavelengths[i], original.wavelengths[i], 1e-4f);
        EXPECT_NEAR(loaded->solar_irradiance[i], original.solar_irradiance[i], 1e-4f);
        EXPECT_NEAR(loaded->sky_radiance[i], original.sky_radiance[i], 1e-4f);
        EXPECT_NEAR(loaded->transmittance[i], original.transmittance[i], 1e-4f);
    }

    // Verify metadata
    EXPECT_EQ(loaded->metadata["solar_zenith_deg"], "30");
    EXPECT_EQ(loaded->metadata["visibility_km"], "23");
    EXPECT_EQ(loaded->metadata["model"], "US_Standard");
}

TEST_F(LUTLoaderTest, SaveInvalidLUT) {
    AtmosphereLUT invalid;
    // Empty LUT - should fail validation

    auto filepath = GetTempFilePath("invalid_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), invalid);
    EXPECT_FALSE(saveSuccess);
    EXPECT_FALSE(std::filesystem::exists(filepath));
}

TEST_F(LUTLoaderTest, LoadNonexistentFile) {
    auto filepath = GetTempFilePath("nonexistent.toml");

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    EXPECT_FALSE(loaded.has_value());
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(LUTLoaderTest, LoadedLUTIsValid) {
    AtmosphereLUT original = CreateTestLUT();
    auto filepath = GetTempFilePath("valid_lut.toml");

    LUTLoader::SaveTOML(filepath.string(), original);
    auto loaded = LUTLoader::LoadTOML(filepath.string());

    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->IsValid());
}

TEST_F(LUTLoaderTest, MonotonicWavelengthPreserved) {
    AtmosphereLUT original = CreateTestLUT();
    auto filepath = GetTempFilePath("monotonic_lut.toml");

    LUTLoader::SaveTOML(filepath.string(), original);
    auto loaded = LUTLoader::LoadTOML(filepath.string());

    ASSERT_TRUE(loaded.has_value());

    // Check monotonicity
    for (usize i = 1; i < loaded->wavelengths.size(); ++i) {
        EXPECT_GT(loaded->wavelengths[i], loaded->wavelengths[i - 1])
            << "Wavelengths must be monotonically increasing";
    }
}

// ============================================================================
// Interpolation Tests
// ============================================================================

TEST_F(LUTLoaderTest, InterpolationExactMatch) {
    AtmosphereLUT lut = CreateTestLUT();

    // Test exact wavelength matches
    EXPECT_NEAR(lut.GetSolarIrradiance(400.0f), 1800.0f, 1e-5f);
    EXPECT_NEAR(lut.GetSolarIrradiance(1200.0f), 1000.0f, 1e-5f);
    EXPECT_NEAR(lut.GetSolarIrradiance(2500.0f), 200.0f, 1e-5f);
}

TEST_F(LUTLoaderTest, InterpolationLinear) {
    AtmosphereLUT lut = CreateTestLUT();
    // Wavelengths: [400, 800, 1200, 1600, 2500]
    // Solar irradiance: [1800, 1400, 1000, 600, 200]

    // Test midpoint between 400 and 800: should be (1800 + 1400) / 2 = 1600
    f32 midpoint_irr = lut.GetSolarIrradiance(600.0f);
    EXPECT_NEAR(midpoint_irr, 1600.0f, 1e-4f);

    // Test 3/4 between 800 and 1200 (at 1100nm)
    // t = (1100 - 800) / (1200 - 800) = 300 / 400 = 0.75
    // result = 1400 * 0.25 + 1000 * 0.75 = 350 + 750 = 1100
    f32 interpolated = lut.GetSolarIrradiance(1100.0f);
    EXPECT_NEAR(interpolated, 1100.0f, 1e-4f);
}

TEST_F(LUTLoaderTest, InterpolationClamping) {
    AtmosphereLUT lut = CreateTestLUT();

    // Below range - should clamp to first value
    EXPECT_NEAR(lut.GetSolarIrradiance(300.0f), 1800.0f, 1e-5f);

    // Above range - should clamp to last value
    EXPECT_NEAR(lut.GetSolarIrradiance(3000.0f), 200.0f, 1e-5f);
}

TEST_F(LUTLoaderTest, InterpolationAllChannels) {
    AtmosphereLUT lut = CreateTestLUT();

    f32 lambda = 1000.0f;  // Between 800 and 1200

    // t = (1000 - 800) / (1200 - 800) = 200 / 400 = 0.5
    f32 expected_irr = 1400.0f * 0.5f + 1000.0f * 0.5f;    // = 1200
    f32 expected_sky = 20.0f * 0.5f + 5.0f * 0.5f;         // = 12.5
    f32 expected_trans = 0.90f * 0.5f + 0.80f * 0.5f;      // = 0.85

    EXPECT_NEAR(lut.GetSolarIrradiance(lambda), expected_irr, 1e-4f);
    EXPECT_NEAR(lut.GetSkyRadiance(lambda), expected_sky, 1e-4f);
    EXPECT_NEAR(lut.GetTransmittance(lambda), expected_trans, 1e-4f);
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_F(LUTLoaderTest, MetadataPreservation) {
    AtmosphereLUT original = CreateTestLUT();
    original.metadata["test_key_1"] = "value_1";
    original.metadata["test_key_2"] = "value_2";

    auto filepath = GetTempFilePath("metadata_lut.toml");
    LUTLoader::SaveTOML(filepath.string(), original);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_EQ(loaded->metadata["test_key_1"], "value_1");
    EXPECT_EQ(loaded->metadata["test_key_2"], "value_2");
}

TEST_F(LUTLoaderTest, EmptyMetadata) {
    AtmosphereLUT lut = CreateTestLUT();
    lut.metadata.clear();

    auto filepath = GetTempFilePath("no_metadata_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut);
    ASSERT_TRUE(saveSuccess);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->metadata.empty());
}

// ============================================================================
// File Utilities Tests
// ============================================================================

TEST_F(LUTLoaderTest, FileExistsCheck) {
    auto filepath = GetTempFilePath("exists_test.toml");

    EXPECT_FALSE(LUTLoader::FileExists(filepath.string()));

    AtmosphereLUT lut = CreateTestLUT();
    LUTLoader::SaveTOML(filepath.string(), lut);

    EXPECT_TRUE(LUTLoader::FileExists(filepath.string()));
}

TEST_F(LUTLoaderTest, GetWavelengthRange) {
    AtmosphereLUT lut = CreateTestLUT();
    auto filepath = GetTempFilePath("wavelength_range_test.toml");

    LUTLoader::SaveTOML(filepath.string(), lut);

    auto range = LUTLoader::GetWavelengthRange(filepath.string());
    ASSERT_TRUE(range.has_value());

    EXPECT_NEAR(range->first, 400.0f, 1e-5f);
    EXPECT_NEAR(range->second, 2500.0f, 1e-5f);
}

TEST_F(LUTLoaderTest, GetWavelengthRangeNonexistent) {
    auto filepath = GetTempFilePath("nonexistent_range.toml");

    auto range = LUTLoader::GetWavelengthRange(filepath.string());
    EXPECT_FALSE(range.has_value());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LUTLoaderTest, SingleWavelengthSample) {
    AtmosphereLUT lut;
    lut.wavelengths = {550.0f};
    lut.solar_irradiance = {1500.0f};
    lut.sky_radiance = {50.0f};
    lut.transmittance = {0.85f};

    auto filepath = GetTempFilePath("single_sample_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut);
    ASSERT_TRUE(saveSuccess);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->IsValid());

    // Interpolation should always return the single value
    EXPECT_NEAR(loaded->GetSolarIrradiance(400.0f), 1500.0f, 1e-5f);
    EXPECT_NEAR(loaded->GetSolarIrradiance(550.0f), 1500.0f, 1e-5f);
    EXPECT_NEAR(loaded->GetSolarIrradiance(700.0f), 1500.0f, 1e-5f);
}

TEST_F(LUTLoaderTest, HighResolutionLUT) {
    // Create a high-resolution LUT with 1000 samples
    AtmosphereLUT lut;
    lut.wavelengths.resize(1000);
    lut.solar_irradiance.resize(1000);
    lut.sky_radiance.resize(1000);
    lut.transmittance.resize(1000);

    for (u32 i = 0; i < 1000; ++i) {
        f32 lambda = 300.0f + i * 2.5f;  // 300nm to 2800nm
        lut.wavelengths[i] = lambda;
        lut.solar_irradiance[i] = 2000.0f * std::exp(-lambda / 1000.0f);
        lut.sky_radiance[i] = 100.0f / (lambda * lambda);
        lut.transmittance[i] = 0.7f + 0.3f * std::exp(-lambda / 500.0f);
    }

    ASSERT_TRUE(lut.IsValid());

    auto filepath = GetTempFilePath("highres_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut);
    ASSERT_TRUE(saveSuccess);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->Size(), 1000);
}

TEST_F(LUTLoaderTest, ZeroValues) {
    AtmosphereLUT lut;
    lut.wavelengths = {400.0f, 800.0f, 1200.0f};
    lut.solar_irradiance = {0.0f, 0.0f, 0.0f};
    lut.sky_radiance = {0.0f, 0.0f, 0.0f};
    lut.transmittance = {0.0f, 0.0f, 0.0f};

    auto filepath = GetTempFilePath("zero_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut);
    ASSERT_TRUE(saveSuccess);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());

    EXPECT_NEAR(loaded->GetSolarIrradiance(600.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(loaded->GetSkyRadiance(600.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(loaded->GetTransmittance(600.0f), 0.0f, 1e-5f);
}

// ============================================================================
// Realistic MODTRAN-style LUT Test
// ============================================================================

TEST_F(LUTLoaderTest, RealisticMODTRANLUT) {
    AtmosphereLUT lut;

    // Visible to SWIR range (400nm to 2500nm, 10nm spacing)
    u32 nsamples = 211;
    lut.wavelengths.resize(nsamples);
    lut.solar_irradiance.resize(nsamples);
    lut.sky_radiance.resize(nsamples);
    lut.transmittance.resize(nsamples);

    for (u32 i = 0; i < nsamples; ++i) {
        f32 lambda = 400.0f + i * 10.0f;
        lut.wavelengths[i] = lambda;

        // Simplified solar spectrum model
        if (lambda < 550.0f) {
            lut.solar_irradiance[i] = 1.5f * lambda;  // Rising in blue
        } else if (lambda < 1000.0f) {
            lut.solar_irradiance[i] = 1800.0f;        // Peak in visible/NIR
        } else {
            lut.solar_irradiance[i] = 1800.0f * std::exp(-(lambda - 1000.0f) / 1000.0f);
        }

        // Rayleigh scattering (1/lambda^4)
        lut.sky_radiance[i] = 1e10f / (lambda * lambda * lambda * lambda);

        // Atmospheric transmittance with water vapor absorption
        f32 base_trans = 0.85f;
        if (lambda > 900.0f && lambda < 1000.0f) {
            base_trans = 0.50f;  // Water vapor band
        }
        if (lambda > 1350.0f && lambda < 1450.0f) {
            base_trans = 0.30f;  // Strong water vapor band
        }
        lut.transmittance[i] = base_trans;
    }

    lut.metadata["model"] = "MODTRAN6";
    lut.metadata["atmosphere"] = "US_Standard";
    lut.metadata["solar_zenith"] = "30";
    lut.metadata["visibility_km"] = "23";

    ASSERT_TRUE(lut.IsValid());

    auto filepath = GetTempFilePath("modtran_realistic_lut.toml");
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut);
    ASSERT_TRUE(saveSuccess);

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->Size(), nsamples);
    EXPECT_EQ(loaded->metadata["model"], "MODTRAN6");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(LUTLoaderTest, CorruptedFile) {
    auto filepath = GetTempFilePath("corrupted.toml");

    // Create a malformed TOML file
    std::ofstream file(filepath);
    file << "[data]\nwavelengths = \"not an array\"\n";
    file.close();

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(LUTLoaderTest, MissingDataSection) {
    auto filepath = GetTempFilePath("missing_data.toml");

    // Create TOML with only metadata
    std::ofstream file(filepath);
    file << "[metadata]\nmodel = \"test\"\n";
    file.close();

    auto loaded = LUTLoader::LoadTOML(filepath.string());
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(LUTLoaderTest, OverwriteExistingFile) {
    AtmosphereLUT lut1 = CreateTestLUT();
    auto filepath = GetTempFilePath("overwrite_test.toml");

    // Save first LUT
    LUTLoader::SaveTOML(filepath.string(), lut1);

    // Create different LUT
    AtmosphereLUT lut2;
    lut2.wavelengths = {500.0f, 600.0f};
    lut2.solar_irradiance = {1000.0f, 900.0f};
    lut2.sky_radiance = {30.0f, 25.0f};
    lut2.transmittance = {0.88f, 0.85f};

    // Overwrite
    bool saveSuccess = LUTLoader::SaveTOML(filepath.string(), lut2);
    ASSERT_TRUE(saveSuccess);

    // Load and verify it's the new LUT
    auto loaded = LUTLoader::LoadTOML(filepath.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->Size(), 2);
    EXPECT_NEAR(loaded->wavelengths[0], 500.0f, 1e-5f);
}
