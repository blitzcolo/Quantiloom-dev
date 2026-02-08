// ============================================================================
// Quantiloom - Unit Tests for Sensor Configuration Parsing
// ============================================================================
// Tests cover:
// - SensorParams default construction values
// - PostprocessConfig::ParseSensorParams from TOML
// - FPN subtable parsing
// - IsSensorEnabled flag
// - Partial config with defaults
// ============================================================================

#include <gtest/gtest.h>
#include "postprocess/PostprocessConfig.hpp"
#include "postprocess/SensorModel.hpp"
#include "core/Config.hpp"
#include <fstream>
#include <filesystem>

using namespace quantiloom;

// ============================================================================
// Test Fixture
// ============================================================================

class SensorConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "quantiloom_sensor_test";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::filesystem::path CreateTOML(const std::string& name, const std::string& content) {
        auto path = testDir / name;
        std::ofstream file(path);
        file << content;
        file.close();
        return path;
    }

    std::filesystem::path testDir;
};

// ============================================================================
// SensorParams Default Construction Tests
// ============================================================================

TEST(SensorParamsTest, DefaultConstruction) {
    SensorParams p;

    // Optics
    EXPECT_FLOAT_EQ(p.focalLength_mm, 50.0f);
    EXPECT_FLOAT_EQ(p.fNumber, 2.8f);
    EXPECT_FLOAT_EQ(p.pixelPitch_um, 5.0f);

    // Detector
    EXPECT_FLOAT_EQ(p.quantumEfficiency, 0.8f);
    EXPECT_FLOAT_EQ(p.wellCapacity_e, 50000.0f);
    EXPECT_FLOAT_EQ(p.readNoise_e_rms, 10.0f);
    EXPECT_FLOAT_EQ(p.darkCurrent_e_s, 50.0f);
    EXPECT_FLOAT_EQ(p.integrationTime_s, 0.01f);

    // ADC
    EXPECT_EQ(p.bitDepth, 14u);
    EXPECT_FLOAT_EQ(p.gain, 3.0f);

    // Noise flags
    EXPECT_TRUE(p.enablePoissonNoise);
    EXPECT_TRUE(p.enableReadNoise);
    EXPECT_TRUE(p.enableDarkCurrent);
    EXPECT_FALSE(p.enableFPN);
}

TEST(SensorParamsTest, FPNParameterDefaults) {
    SensorParams p;

    EXPECT_FLOAT_EQ(p.prnuSigma, 0.01f);
    EXPECT_FLOAT_EQ(p.dsnuSigma_e, 5.0f);
    EXPECT_FALSE(p.enableNUC);
    EXPECT_FLOAT_EQ(p.nucEfficiency, 0.98f);
    EXPECT_FLOAT_EQ(p.wavelength_nm, 550.0f);
    EXPECT_FLOAT_EQ(p.detectorTemperature_K, 77.0f);
}

// ============================================================================
// ParseSensorParams Tests
// ============================================================================

TEST_F(SensorConfigTest, DefaultValues) {
    // Empty config should return all defaults
    auto path = CreateTOML("empty.toml", "");
    auto result = Config::Load(path);
    ASSERT_TRUE(result.has_value());

    SensorParams p = PostprocessConfig::ParseSensorParams(*result);

    EXPECT_FLOAT_EQ(p.focalLength_mm, 50.0f);
    EXPECT_FLOAT_EQ(p.fNumber, 2.8f);
    EXPECT_FLOAT_EQ(p.pixelPitch_um, 5.0f);
    EXPECT_FLOAT_EQ(p.quantumEfficiency, 0.8f);
    EXPECT_EQ(p.bitDepth, 14u);
    EXPECT_FLOAT_EQ(p.gain, 0.5f);  // ParseSensorParams default is 0.5
    EXPECT_TRUE(p.enablePoissonNoise);
    EXPECT_FALSE(p.enableFPN);
}

TEST_F(SensorConfigTest, AllFieldsParsed) {
    auto path = CreateTOML("full.toml", R"(
[sensor]
focal_length_mm = 35.0
f_number = 4.0
pixel_pitch_um = 3.5
quantum_efficiency = 0.65
well_capacity_e = 30000.0
read_noise_e_rms = 8.0
dark_current_e_s = 25.0
integration_time_s = 0.02
bit_depth = 12
gain = 2.0
enable_poisson_noise = false
enable_read_noise = false
enable_dark_current = false
enable_fpn = true
detector_temperature_k = 300.0

[sensor.fpn]
prnu_sigma = 0.03
dsnu_sigma_e = 15.0
enable_nuc = true
nuc_efficiency = 0.95

[spectral]
wavelength_nm = 850.0
    )");

    auto result = Config::Load(path);
    ASSERT_TRUE(result.has_value());

    SensorParams p = PostprocessConfig::ParseSensorParams(*result);

    EXPECT_FLOAT_EQ(p.focalLength_mm, 35.0f);
    EXPECT_FLOAT_EQ(p.fNumber, 4.0f);
    EXPECT_FLOAT_EQ(p.pixelPitch_um, 3.5f);
    EXPECT_FLOAT_EQ(p.quantumEfficiency, 0.65f);
    EXPECT_FLOAT_EQ(p.wellCapacity_e, 30000.0f);
    EXPECT_FLOAT_EQ(p.readNoise_e_rms, 8.0f);
    EXPECT_FLOAT_EQ(p.darkCurrent_e_s, 25.0f);
    EXPECT_FLOAT_EQ(p.integrationTime_s, 0.02f);
    EXPECT_EQ(p.bitDepth, 12u);
    EXPECT_FLOAT_EQ(p.gain, 2.0f);
    EXPECT_FALSE(p.enablePoissonNoise);
    EXPECT_FALSE(p.enableReadNoise);
    EXPECT_FALSE(p.enableDarkCurrent);
    EXPECT_TRUE(p.enableFPN);
    EXPECT_FLOAT_EQ(p.detectorTemperature_K, 300.0f);
    EXPECT_FLOAT_EQ(p.wavelength_nm, 850.0f);
}

TEST_F(SensorConfigTest, FPNSubtableParsed) {
    auto path = CreateTOML("fpn.toml", R"(
[sensor]
enable_fpn = true

[sensor.fpn]
prnu_sigma = 0.03
dsnu_sigma_e = 15.0
enable_nuc = true
nuc_efficiency = 0.95
    )");

    auto result = Config::Load(path);
    ASSERT_TRUE(result.has_value());

    SensorParams p = PostprocessConfig::ParseSensorParams(*result);

    EXPECT_TRUE(p.enableFPN);
    EXPECT_FLOAT_EQ(p.prnuSigma, 0.03f);
    EXPECT_FLOAT_EQ(p.dsnuSigma_e, 15.0f);
    EXPECT_TRUE(p.enableNUC);
    EXPECT_FLOAT_EQ(p.nucEfficiency, 0.95f);
}

TEST_F(SensorConfigTest, IsSensorEnabled) {
    // Enabled
    {
        auto path = CreateTOML("enabled.toml", R"(
[sensor]
enabled = true
        )");
        auto result = Config::Load(path);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(PostprocessConfig::IsSensorEnabled(*result));
    }
    // Disabled
    {
        auto path = CreateTOML("disabled.toml", R"(
[sensor]
enabled = false
        )");
        auto result = Config::Load(path);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(PostprocessConfig::IsSensorEnabled(*result));
    }
    // Missing (default false)
    {
        auto path = CreateTOML("missing.toml", "");
        auto result = Config::Load(path);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(PostprocessConfig::IsSensorEnabled(*result));
    }
}

TEST_F(SensorConfigTest, PartialConfig) {
    // Only some fields specified, rest should use defaults
    auto path = CreateTOML("partial.toml", R"(
[sensor]
f_number = 5.6
bit_depth = 16
    )");

    auto result = Config::Load(path);
    ASSERT_TRUE(result.has_value());

    SensorParams p = PostprocessConfig::ParseSensorParams(*result);

    // Specified values
    EXPECT_FLOAT_EQ(p.fNumber, 5.6f);
    EXPECT_EQ(p.bitDepth, 16u);

    // Defaults for unspecified
    EXPECT_FLOAT_EQ(p.focalLength_mm, 50.0f);
    EXPECT_FLOAT_EQ(p.pixelPitch_um, 5.0f);
    EXPECT_FLOAT_EQ(p.quantumEfficiency, 0.8f);
    EXPECT_FLOAT_EQ(p.prnuSigma, 0.01f);
}
