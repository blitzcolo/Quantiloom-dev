// ============================================================================
// Quantiloom - Unit Tests for AtmosphereTransmittanceLUT and Loader
// ============================================================================
// Tests cover:
// - AtmosphereTransmittanceLUT data structure
// - UniformAxis and NonUniformAxis functionality
// - Trilinear interpolation
// - .qlut file read/write roundtrip
// - TOML header parsing and generation
// - Edge cases and error handling
// ============================================================================

#include <gtest/gtest.h>
#include "core/AtmosphereTransmittanceLUT.hpp"
#include "io/AtmosphereTransmittanceLUTLoader.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>

using namespace quantiloom;

// ============================================================================
// Test Fixture
// ============================================================================

class AtmosphereTransmittanceLUTTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempDir = std::filesystem::temp_directory_path() / "quantiloom_atmo_lut_tests";
        std::filesystem::create_directories(tempDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(tempDir)) {
            std::filesystem::remove_all(tempDir);
        }
    }

    std::filesystem::path GetTempFilePath(const std::string& filename) {
        return tempDir / filename;
    }

    // Helper: Create a minimal valid LUT for testing
    AtmosphereTransmittanceLUT CreateMinimalLUT() {
        AtmosphereTransmittanceLUT lut;

        lut.name = "Test LUT";
        lut.source = "Unit Test";
        lut.created = "2024-12-25";
        lut.atmospheric_model = "Test_Model";

        // 3 wavelengths: 400, 450, 500 nm
        lut.wavelength = {400.0f, 500.0f, 50.0f, 3};

        // 2 altitudes: 0, 1000 m
        lut.altitude = {0.0f, 1000.0f, 1000.0f, 2};

        // 2 zenith angles: 0, 45 deg
        lut.zenith.values = {0.0f, 45.0f};

        // Total: 3 * 2 * 2 = 12 elements
        lut.transmittance.resize(12);
        for (usize i = 0; i < 12; ++i) {
            lut.transmittance[i] = 0.5f + 0.04f * static_cast<f32>(i);  // 0.5 to 0.94
        }

        return lut;
    }

    // Helper: Create a realistic LUT for comprehensive testing
    AtmosphereTransmittanceLUT CreateRealisticLUT() {
        AtmosphereTransmittanceLUT lut;

        lut.name = "Midlatitude Summer (Test)";
        lut.source = "Unit Test Generator";
        lut.created = "2024-12-25T12:00:00Z";
        lut.atmospheric_model = "US_Standard_1976";

        // Wavelength: 300-14000 nm, 100nm step (138 samples)
        lut.wavelength = {300.0f, 14000.0f, 100.0f, 138};

        // Altitude: 0-30000 m, 5000m step (7 samples)
        lut.altitude = {0.0f, 30000.0f, 5000.0f, 7};

        // Zenith: 0, 30, 60, 85 deg (4 samples)
        lut.zenith.values = {0.0f, 30.0f, 60.0f, 85.0f};

        // Total: 138 * 7 * 4 = 3864 elements
        const usize total = 138 * 7 * 4;
        lut.transmittance.resize(total);
        lut.path_radiance.resize(total);

        for (u32 i_wave = 0; i_wave < 138; ++i_wave) {
            f32 lambda = 300.0f + i_wave * 100.0f;

            for (u32 i_alt = 0; i_alt < 7; ++i_alt) {
                f32 alt = i_alt * 5000.0f;
                f32 density_factor = std::exp(-alt / 8500.0f);

                for (u32 i_zen = 0; i_zen < 4; ++i_zen) {
                    f32 zen = lut.zenith.values[i_zen];
                    f32 air_mass = 1.0f / std::max(0.1f, std::cos(zen * 3.14159f / 180.0f));

                    usize idx = lut.GetDataIndex(i_wave, i_alt, i_zen);

                    // Simplified transmittance model
                    f32 optical_depth = density_factor * air_mass * (1.0f + 1000.0f / lambda);
                    lut.transmittance[idx] = std::exp(-optical_depth * 0.1f);

                    // Simplified path radiance (thermal emission proxy)
                    lut.path_radiance[idx] = density_factor * (1.0f - lut.transmittance[idx]) * 0.01f;
                }
            }
        }

        return lut;
    }

    std::filesystem::path tempDir;
};

// ============================================================================
// UniformAxis Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, UniformAxisBasic) {
    UniformAxis axis{100.0f, 500.0f, 100.0f, 5};  // 100, 200, 300, 400, 500

    EXPECT_TRUE(axis.IsValid());
    EXPECT_FLOAT_EQ(axis.GetValue(0), 100.0f);
    EXPECT_FLOAT_EQ(axis.GetValue(2), 300.0f);
    EXPECT_FLOAT_EQ(axis.GetValue(4), 500.0f);
}

TEST_F(AtmosphereTransmittanceLUTTest, UniformAxisFractionalIndex) {
    UniformAxis axis{0.0f, 100.0f, 25.0f, 5};  // 0, 25, 50, 75, 100

    // Exact matches
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(50.0f), 2.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(100.0f), 4.0f);

    // Interpolated
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(12.5f), 0.5f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(37.5f), 1.5f);

    // Clamping
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(-50.0f), 0.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(200.0f), 4.0f);
}

TEST_F(AtmosphereTransmittanceLUTTest, UniformAxisInvalid) {
    UniformAxis empty{0.0f, 0.0f, 1.0f, 0};
    EXPECT_FALSE(empty.IsValid());

    UniformAxis zero_step{0.0f, 100.0f, 0.0f, 5};
    EXPECT_FALSE(zero_step.IsValid());

    UniformAxis negative_step{0.0f, 100.0f, -10.0f, 5};
    EXPECT_FALSE(negative_step.IsValid());
}

// ============================================================================
// NonUniformAxis Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, NonUniformAxisBasic) {
    NonUniformAxis axis;
    axis.values = {0.0f, 15.0f, 30.0f, 45.0f, 60.0f, 75.0f, 85.0f};

    EXPECT_TRUE(axis.IsValid());
    EXPECT_EQ(axis.Count(), 7);
}

TEST_F(AtmosphereTransmittanceLUTTest, NonUniformAxisFractionalIndex) {
    NonUniformAxis axis;
    axis.values = {0.0f, 30.0f, 60.0f, 85.0f};

    // Exact matches
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(30.0f), 1.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(85.0f), 3.0f);

    // Interpolated
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(15.0f), 0.5f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(45.0f), 1.5f);

    // Non-uniform spacing: 60 to 85 (25 degree span)
    // 72.5 is midpoint -> index 2.5
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(72.5f), 2.5f);

    // Clamping
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(-10.0f), 0.0f);
    EXPECT_FLOAT_EQ(axis.GetFractionalIndex(90.0f), 3.0f);
}

TEST_F(AtmosphereTransmittanceLUTTest, NonUniformAxisInvalid) {
    NonUniformAxis empty;
    EXPECT_FALSE(empty.IsValid());

    NonUniformAxis not_monotonic;
    not_monotonic.values = {0.0f, 30.0f, 20.0f, 60.0f};  // 20 < 30
    EXPECT_FALSE(not_monotonic.IsValid());
}

// ============================================================================
// AtmosphereTransmittanceLUT Validation Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, MinimalLUTIsValid) {
    auto lut = CreateMinimalLUT();
    EXPECT_TRUE(lut.IsValid());
}

TEST_F(AtmosphereTransmittanceLUTTest, RealisticLUTIsValid) {
    auto lut = CreateRealisticLUT();
    EXPECT_TRUE(lut.IsValid());
    EXPECT_EQ(lut.TotalDataSize(), 138 * 7 * 4);
}

TEST_F(AtmosphereTransmittanceLUTTest, EmptyLUTIsInvalid) {
    AtmosphereTransmittanceLUT lut;
    EXPECT_FALSE(lut.IsValid());
}

TEST_F(AtmosphereTransmittanceLUTTest, MismatchedDataSizeIsInvalid) {
    auto lut = CreateMinimalLUT();
    lut.transmittance.resize(5);  // Should be 12
    EXPECT_FALSE(lut.IsValid());
}

// ============================================================================
// Trilinear Interpolation Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, InterpolationExactCorners) {
    auto lut = CreateMinimalLUT();

    // First corner: wavelength=400, altitude=0, zenith=0
    f32 first = lut.QueryTransmittance(400.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(first, lut.transmittance[0]);

    // Last corner: wavelength=500, altitude=1000, zenith=45
    f32 last = lut.QueryTransmittance(500.0f, 1000.0f, 45.0f);
    EXPECT_FLOAT_EQ(last, lut.transmittance[11]);
}

TEST_F(AtmosphereTransmittanceLUTTest, InterpolationMidpoint) {
    AtmosphereTransmittanceLUT lut;

    lut.wavelength = {0.0f, 100.0f, 100.0f, 2};
    lut.altitude = {0.0f, 100.0f, 100.0f, 2};
    lut.zenith.values = {0.0f, 90.0f};

    // 2x2x2 = 8 elements, set to known values
    lut.transmittance = {
        0.0f, 1.0f,   // [0][0][0], [0][0][1]
        0.0f, 1.0f,   // [0][1][0], [0][1][1]
        0.0f, 1.0f,   // [1][0][0], [1][0][1]
        0.0f, 1.0f    // [1][1][0], [1][1][1]
    };

    // At zenith=0, all values are 0
    EXPECT_FLOAT_EQ(lut.QueryTransmittance(50.0f, 50.0f, 0.0f), 0.0f);

    // At zenith=90, all values are 1
    EXPECT_FLOAT_EQ(lut.QueryTransmittance(50.0f, 50.0f, 90.0f), 1.0f);

    // At zenith=45 (midpoint), result should be 0.5
    EXPECT_FLOAT_EQ(lut.QueryTransmittance(50.0f, 50.0f, 45.0f), 0.5f);
}

TEST_F(AtmosphereTransmittanceLUTTest, InterpolationClamping) {
    auto lut = CreateMinimalLUT();

    // Below minimum wavelength
    f32 below = lut.QueryTransmittance(300.0f, 0.0f, 0.0f);
    f32 at_min = lut.QueryTransmittance(400.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(below, at_min);

    // Above maximum wavelength
    f32 above = lut.QueryTransmittance(600.0f, 1000.0f, 45.0f);
    f32 at_max = lut.QueryTransmittance(500.0f, 1000.0f, 45.0f);
    EXPECT_FLOAT_EQ(above, at_max);
}

// ============================================================================
// File I/O Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, SaveAndLoadRoundtrip) {
    auto original = CreateMinimalLUT();
    auto filepath = GetTempFilePath("test_roundtrip.qlut");

    // Save
    bool saved = AtmosphereTransmittanceLUTLoader::Save(filepath, original);
    ASSERT_TRUE(saved);
    ASSERT_TRUE(std::filesystem::exists(filepath));

    // Load
    auto loaded = AtmosphereTransmittanceLUTLoader::Load(filepath);
    ASSERT_TRUE(loaded.has_value());

    // Verify metadata
    EXPECT_EQ(loaded->name, original.name);
    EXPECT_EQ(loaded->source, original.source);
    EXPECT_EQ(loaded->atmospheric_model, original.atmospheric_model);

    // Verify axes
    EXPECT_FLOAT_EQ(loaded->wavelength.start, original.wavelength.start);
    EXPECT_FLOAT_EQ(loaded->wavelength.stop, original.wavelength.stop);
    EXPECT_FLOAT_EQ(loaded->wavelength.step, original.wavelength.step);
    EXPECT_EQ(loaded->wavelength.count, original.wavelength.count);

    EXPECT_FLOAT_EQ(loaded->altitude.start, original.altitude.start);
    EXPECT_EQ(loaded->altitude.count, original.altitude.count);

    EXPECT_EQ(loaded->zenith.values.size(), original.zenith.values.size());

    // Verify data
    ASSERT_EQ(loaded->transmittance.size(), original.transmittance.size());
    for (usize i = 0; i < original.transmittance.size(); ++i) {
        EXPECT_FLOAT_EQ(loaded->transmittance[i], original.transmittance[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(AtmosphereTransmittanceLUTTest, SaveAndLoadWithPathRadiance) {
    auto original = CreateRealisticLUT();
    auto filepath = GetTempFilePath("test_with_path_radiance.qlut");

    bool saved = AtmosphereTransmittanceLUTLoader::Save(filepath, original);
    ASSERT_TRUE(saved);

    auto loaded = AtmosphereTransmittanceLUTLoader::Load(filepath);
    ASSERT_TRUE(loaded.has_value());

    // Verify path radiance was loaded
    ASSERT_EQ(loaded->path_radiance.size(), original.path_radiance.size());
    for (usize i = 0; i < std::min<usize>(100, original.path_radiance.size()); ++i) {
        EXPECT_NEAR(loaded->path_radiance[i], original.path_radiance[i], 1e-6f)
            << "Path radiance mismatch at index " << i;
    }
}

TEST_F(AtmosphereTransmittanceLUTTest, LoadNonexistentFile) {
    auto filepath = GetTempFilePath("nonexistent.qlut");
    auto loaded = AtmosphereTransmittanceLUTLoader::Load(filepath);
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(AtmosphereTransmittanceLUTTest, SaveInvalidLUT) {
    AtmosphereTransmittanceLUT invalid;
    auto filepath = GetTempFilePath("invalid.qlut");

    bool saved = AtmosphereTransmittanceLUTLoader::Save(filepath, invalid);
    EXPECT_FALSE(saved);
    EXPECT_FALSE(std::filesystem::exists(filepath));
}

TEST_F(AtmosphereTransmittanceLUTTest, LoadCorruptedFile) {
    auto filepath = GetTempFilePath("corrupted.qlut");

    // Create a file with invalid content
    std::ofstream file(filepath, std::ios::binary);
    file << "This is not a valid QLUT file!";
    file.close();

    auto loaded = AtmosphereTransmittanceLUTLoader::Load(filepath);
    EXPECT_FALSE(loaded.has_value());
}

// ============================================================================
// PeekInfo Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, PeekInfoBasic) {
    auto original = CreateRealisticLUT();
    auto filepath = GetTempFilePath("peek_test.qlut");

    AtmosphereTransmittanceLUTLoader::Save(filepath, original);

    auto info = AtmosphereTransmittanceLUTLoader::PeekInfo(filepath);
    ASSERT_TRUE(info.has_value());

    EXPECT_EQ(info->name, original.name);
    EXPECT_EQ(info->source, original.source);
    EXPECT_EQ(info->atmospheric_model, original.atmospheric_model);
    EXPECT_FLOAT_EQ(info->wavelength_min, 300.0f);
    EXPECT_FLOAT_EQ(info->wavelength_max, 14000.0f);
    EXPECT_FLOAT_EQ(info->altitude_max, 30000.0f);
}

TEST_F(AtmosphereTransmittanceLUTTest, PeekInfoNonexistent) {
    auto info = AtmosphereTransmittanceLUTLoader::PeekInfo("nonexistent.qlut");
    EXPECT_FALSE(info.has_value());
}

// ============================================================================
// Query Performance (Sanity Check)
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, QueryPerformanceSanity) {
    auto lut = CreateRealisticLUT();

    // Perform many queries to ensure no crashes or obvious performance issues
    f32 sum = 0.0f;
    for (int i = 0; i < 10000; ++i) {
        f32 wave = 300.0f + (i % 1000) * 13.7f;
        f32 alt = (i % 7) * 5000.0f;
        f32 zen = (i % 4) * 25.0f;

        sum += lut.QueryTransmittance(wave, alt, zen);
    }

    // Just verify we got reasonable results (not NaN or Inf)
    EXPECT_TRUE(std::isfinite(sum));
    EXPECT_GT(sum, 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, SingleElementLUT) {
    AtmosphereTransmittanceLUT lut;

    lut.name = "Single Element";
    lut.source = "Test";
    lut.wavelength = {550.0f, 550.0f, 1.0f, 1};
    lut.altitude = {0.0f, 0.0f, 1.0f, 1};
    lut.zenith.values = {0.0f};
    lut.transmittance = {0.85f};

    EXPECT_TRUE(lut.IsValid());

    // Any query should return the single value
    EXPECT_FLOAT_EQ(lut.QueryTransmittance(400.0f, 1000.0f, 45.0f), 0.85f);
    EXPECT_FLOAT_EQ(lut.QueryTransmittance(700.0f, 0.0f, 0.0f), 0.85f);
}

TEST_F(AtmosphereTransmittanceLUTTest, LargeZenithAngle) {
    auto lut = CreateRealisticLUT();

    // Query at high zenith angle (near horizon)
    f32 trans_85 = lut.QueryTransmittance(550.0f, 0.0f, 85.0f);
    f32 trans_0 = lut.QueryTransmittance(550.0f, 0.0f, 0.0f);

    // Transmittance should be lower at high zenith (longer path)
    EXPECT_LT(trans_85, trans_0);
}

TEST_F(AtmosphereTransmittanceLUTTest, HighAltitudeTransmittance) {
    auto lut = CreateRealisticLUT();

    // At high altitude, transmittance should be higher (less atmosphere)
    f32 trans_0 = lut.QueryTransmittance(550.0f, 0.0f, 0.0f);
    f32 trans_30km = lut.QueryTransmittance(550.0f, 30000.0f, 0.0f);

    EXPECT_GT(trans_30km, trans_0);
}

// ============================================================================
// Wavelength Dependence Tests
// ============================================================================

TEST_F(AtmosphereTransmittanceLUTTest, RayleighWavelengthDependence) {
    auto lut = CreateRealisticLUT();

    // At sea level, blue light should have lower transmittance than red
    // due to Rayleigh scattering (~1/λ^4)
    f32 trans_blue = lut.QueryTransmittance(450.0f, 0.0f, 0.0f);
    f32 trans_red = lut.QueryTransmittance(700.0f, 0.0f, 0.0f);

    EXPECT_LT(trans_blue, trans_red);
}
