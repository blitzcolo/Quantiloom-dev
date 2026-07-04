// ============================================================================
// Quantiloom - Unit Tests for renderer/AtmosphericConfig
// ============================================================================
// Tests cover:
// - Preset configurations
// - TOML loading
// - GPU structure conversion
// - Size validation
// ============================================================================

#include <gtest/gtest.h>
#include "renderer/AtmosphericConfig.hpp"
#include <fstream>
#include <filesystem>

using namespace quantiloom;

namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class AtmosphericConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default config
        config = AtmosphericConfig::ClearDay();
    }

    AtmosphericConfig config;
};

// ============================================================================
// Preset Tests
// ============================================================================

TEST_F(AtmosphericConfigTest, DefaultConstructorUsesClearDay) {
    AtmosphericConfig default_config;
    AtmosphericConfig clear_day = AtmosphericConfig::ClearDay();

    EXPECT_EQ(default_config.rayleigh_beta_550nm, clear_day.rayleigh_beta_550nm);
    EXPECT_EQ(default_config.mie_beta_550nm, clear_day.mie_beta_550nm);
}

TEST_F(AtmosphericConfigTest, ClearDayPresetHasReasonableValues) {
    auto clearDay = AtmosphericConfig::ClearDay();

    EXPECT_TRUE(clearDay.rayleigh_enabled);
    EXPECT_TRUE(clearDay.mie_enabled);
    EXPECT_NEAR(clearDay.rayleigh_beta_550nm, 5.8e-6f, 1e-7f);
    // 1.70e-4 /m -> Koschmieder visibility ~23 km, a realistic clear day.
    // (The old pinned value 2.0e-6 implied 1956 km visibility; see ATM-01/ATM-21.)
    EXPECT_NEAR(clearDay.mie_beta_550nm, 1.70e-4f, 1e-6f);
    EXPECT_NEAR(clearDay.planet_radius, 6.371e6f, 1e3f);
}

TEST_F(AtmosphericConfigTest, HazyPresetIncreasesAerosol) {
    auto clear = AtmosphericConfig::ClearDay();
    auto hazy = AtmosphericConfig::Hazy();

    // Hazy should have more aerosol scattering
    EXPECT_GT(hazy.mie_beta_550nm, clear.mie_beta_550nm);
}

TEST_F(AtmosphericConfigTest, MarsPresetDisablesRayleigh) {
    auto mars = AtmosphericConfig::Mars();

    EXPECT_FALSE(mars.rayleigh_enabled);
    EXPECT_TRUE(mars.mie_enabled);
    EXPECT_LT(mars.planet_radius, 6.0e6f);  // Smaller than Earth
}

TEST_F(AtmosphericConfigTest, DisabledPresetZerosCoefficients) {
    auto disabled = AtmosphericConfig::Disabled();

    EXPECT_FALSE(disabled.rayleigh_enabled);
    EXPECT_FALSE(disabled.mie_enabled);
    EXPECT_FLOAT_EQ(disabled.rayleigh_beta_550nm, 0.0f);
    EXPECT_FLOAT_EQ(disabled.mie_beta_550nm, 0.0f);
    EXPECT_FALSE(disabled.IsEnabled());
}

// ============================================================================
// GPU Conversion Tests
// ============================================================================

TEST_F(AtmosphericConfigTest, ToGPUProducesCorrectSize) {
    auto gpu = config.ToGPU();

    // Verify size matches shader expectations (64 bytes)
    EXPECT_EQ(sizeof(gpu), 64);
}

TEST_F(AtmosphericConfigTest, ToGPUPreservesRayleighCoefficient) {
    config.rayleigh_beta_550nm = 7.0e-6f;
    config.rayleigh_enabled = true;

    auto gpu = config.ToGPU();

    EXPECT_NEAR(gpu.beta_rayleigh_550nm.x, 7.0e-6f, 1e-9f);
    EXPECT_NEAR(gpu.beta_rayleigh_550nm.y, 7.0e-6f, 1e-9f);
    EXPECT_NEAR(gpu.beta_rayleigh_550nm.z, 7.0e-6f, 1e-9f);
}

TEST_F(AtmosphericConfigTest, ToGPUZerosDisabledScattering) {
    config.rayleigh_enabled = false;
    config.mie_enabled = false;

    auto gpu = config.ToGPU();

    EXPECT_FLOAT_EQ(gpu.beta_rayleigh_550nm.x, 0.0f);
    EXPECT_FLOAT_EQ(gpu.beta_mie_550nm.x, 0.0f);
}

TEST_F(AtmosphericConfigTest, ToGPUPreservesPlanetGeometry) {
    config.planet_radius = 1.234e6f;
    config.atmosphere_height = 56789.0f;

    auto gpu = config.ToGPU();

    EXPECT_FLOAT_EQ(gpu.planet_radius, 1.234e6f);
    EXPECT_FLOAT_EQ(gpu.atmosphere_height, 56789.0f);
}

// ============================================================================
// TOML Loading Tests
// ============================================================================

TEST_F(AtmosphericConfigTest, FromTOMLHandlesMissingFile) {
    auto result = AtmosphericConfig::FromTOML("/nonexistent/path.toml");

    EXPECT_FALSE(result.has_value());
}

TEST_F(AtmosphericConfigTest, FromTOMLUsesDefaultIfNoAtmosphericSection) {
    // Create temporary TOML without [atmospheric] section
    fs::path temp_path = fs::temp_directory_path() / "quantiloom_test_no_atmo.toml";
    std::ofstream file(temp_path);
    file << "[scene]\n";
    file << "name = \"test\"\n";
    file.close();

    auto result = AtmosphericConfig::FromTOML(temp_path.string());

    ASSERT_TRUE(result.has_value());
    auto loaded = result.value();
    EXPECT_TRUE(loaded.rayleigh_enabled);  // Should use ClearDay default

    // Cleanup
    fs::remove(temp_path);
}

TEST_F(AtmosphericConfigTest, FromTOMLLoadsPreset) {
    fs::path temp_path = fs::temp_directory_path() / "quantiloom_test_preset.toml";
    std::ofstream file(temp_path);
    file << "[atmospheric]\n";
    file << "preset = \"hazy\"\n";
    file.close();

    auto result = AtmosphericConfig::FromTOML(temp_path.string());

    ASSERT_TRUE(result.has_value());
    auto loaded = result.value();

    auto hazy = AtmosphericConfig::Hazy();
    EXPECT_FLOAT_EQ(loaded.mie_beta_550nm, hazy.mie_beta_550nm);

    // Cleanup
    fs::remove(temp_path);
}

TEST_F(AtmosphericConfigTest, FromTOMLOverridesPresetWithCustomValues) {
    fs::path temp_path = fs::temp_directory_path() / "quantiloom_test_override.toml";
    std::ofstream file(temp_path);
    file << "[atmospheric]\n";
    file << "preset = \"clear_day\"\n";
    file << "mie_beta_550nm = 1.5e-5\n";
    file << "max_steps = 128\n";
    file.close();

    auto result = AtmosphericConfig::FromTOML(temp_path.string());

    ASSERT_TRUE(result.has_value());
    auto loaded = result.value();

    EXPECT_NEAR(loaded.mie_beta_550nm, 1.5e-5f, 1e-8f);
    EXPECT_EQ(loaded.max_steps, 128);

    // Cleanup
    fs::remove(temp_path);
}

TEST_F(AtmosphericConfigTest, FromTOMLHandlesDisabled) {
    fs::path temp_path = fs::temp_directory_path() / "quantiloom_test_disabled.toml";
    std::ofstream file(temp_path);
    file << "[atmospheric]\n";
    file << "rayleigh_enabled = false\n";
    file << "mie_enabled = false\n";
    file.close();

    auto result = AtmosphericConfig::FromTOML(temp_path.string());

    ASSERT_TRUE(result.has_value());
    auto loaded = result.value();

    EXPECT_FALSE(loaded.IsEnabled());

    // Cleanup
    fs::remove(temp_path);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(AtmosphericConfigTest, IsEnabledReturnsTrueWhenRayleighEnabled) {
    config.rayleigh_enabled = true;
    config.mie_enabled = false;

    EXPECT_TRUE(config.IsEnabled());
}

TEST_F(AtmosphericConfigTest, IsEnabledReturnsTrueWhenMieEnabled) {
    config.rayleigh_enabled = false;
    config.mie_enabled = true;

    EXPECT_TRUE(config.IsEnabled());
}

TEST_F(AtmosphericConfigTest, IsEnabledReturnsFalseWhenBothDisabled) {
    config.rayleigh_enabled = false;
    config.mie_enabled = false;

    EXPECT_FALSE(config.IsEnabled());
}
