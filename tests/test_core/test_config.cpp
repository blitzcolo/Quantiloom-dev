// ============================================================================
// Quantiloom - Unit Tests for core/Config.hpp
// ============================================================================
// Tests cover:
// - TOML file loading and parsing
// - Key navigation (dot-separated paths)
// - Type conversion (i32, f32, String, bool)
// - Default values
// - Required key validation
// - Array handling
// - Table extraction
// ============================================================================

#include <gtest/gtest.h>
#include "core/Config.hpp"
#include <fstream>
#include <filesystem>

using namespace quantiloom;

// ============================================================================
// Test Fixture with Temp TOML Files
// ============================================================================

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        testDir = std::filesystem::temp_directory_path() / "quantiloom_test";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    // Helper: Create a TOML file with given content
    std::filesystem::path CreateTOML(const std::string& filename, const std::string& content) {
        auto filepath = testDir / filename;
        std::ofstream file(filepath);
        file << content;
        file.close();
        return filepath;
    }

    std::filesystem::path testDir;
};

// ============================================================================
// Loading Tests
// ============================================================================

TEST_F(ConfigTest, LoadValidFile) {
    auto filepath = CreateTOML("valid.toml", R"(
        [renderer]
        width = 1280
        height = 720
        spp = 4
    )");

    auto result = Config::Load(filepath);
    ASSERT_TRUE(result.has_value());

    Config config = std::move(*result);
    EXPECT_TRUE(config.Has("renderer.width"));
    EXPECT_EQ(config.Get<i32>("renderer.width", 0), 1280);
}

TEST_F(ConfigTest, LoadNonexistentFile) {
    auto filepath = testDir / "nonexistent.toml";

    auto result = Config::Load(filepath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(ConfigTest, LoadInvalidTOML) {
    auto filepath = CreateTOML("invalid.toml", R"(
        [renderer
        width = 1280  # Missing closing bracket
    )");

    auto result = Config::Load(filepath);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Key Navigation Tests
// ============================================================================

TEST_F(ConfigTest, HasKeyTopLevel) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
        height = 720
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_TRUE(config.Has("width"));
    EXPECT_TRUE(config.Has("height"));
    EXPECT_FALSE(config.Has("nonexistent"));
}

TEST_F(ConfigTest, HasKeyNested) {
    auto filepath = CreateTOML("test.toml", R"(
        [renderer]
        width = 1280

        [renderer.settings]
        quality = "high"
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_TRUE(config.Has("renderer.width"));
    EXPECT_TRUE(config.Has("renderer.settings.quality"));
    EXPECT_FALSE(config.Has("renderer.settings.nonexistent"));
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

TEST_F(ConfigTest, GetInteger) {
    auto filepath = CreateTOML("test.toml", R"(
        value_i32 = 42
        value_i64 = 9999999999
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_EQ(config.Get<i32>("value_i32", 0), 42);
    EXPECT_EQ(config.Get<i64>("value_i64", 0), 9999999999);
}

TEST_F(ConfigTest, GetUnsignedInteger) {
    auto filepath = CreateTOML("test.toml", R"(
        value_u32 = 123
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_EQ(config.Get<u32>("value_u32", 0), 123u);
}

TEST_F(ConfigTest, GetFloat) {
    auto filepath = CreateTOML("test.toml", R"(
        value_f32 = 3.14
        value_f64 = 2.718281828
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_NEAR(config.Get<f32>("value_f32", 0.0f), 3.14f, 1e-6f);
    EXPECT_NEAR(config.Get<f64>("value_f64", 0.0), 2.718281828, 1e-9);
}

TEST_F(ConfigTest, GetString) {
    auto filepath = CreateTOML("test.toml", R"(
        name = "Quantiloom"
        mode = "HS-OFF"
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_EQ(config.Get<String>("name", ""), "Quantiloom");
    EXPECT_EQ(config.Get<String>("mode", ""), "HS-OFF");
}

TEST_F(ConfigTest, GetBoolean) {
    auto filepath = CreateTOML("test.toml", R"(
        enable_denoising = true
        enable_debug = false
    )");

    auto config = Config::Load(filepath).value();

    EXPECT_TRUE(config.Get<bool>("enable_denoising", false));
    EXPECT_FALSE(config.Get<bool>("enable_debug", true));
}

// ============================================================================
// Default Value Tests
// ============================================================================

TEST_F(ConfigTest, GetWithDefault) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    // Existing key
    EXPECT_EQ(config.Get<i32>("width", 0), 1280);

    // Missing key - should return default
    EXPECT_EQ(config.Get<i32>("height", 720), 720);
    EXPECT_EQ(config.Get<String>("name", "default"), "default");
    EXPECT_TRUE(config.Get<bool>("missing_bool", true));
}

// ============================================================================
// Required Key Tests
// ============================================================================

TEST_F(ConfigTest, GetRequiredSuccess) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    auto result = config.GetRequired<i32>("width");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1280);
}

TEST_F(ConfigTest, GetRequiredMissingKey) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    auto result = config.GetRequired<i32>("height");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ConfigTest, GetRequiredTypeMismatch) {
    auto filepath = CreateTOML("test.toml", R"(
        width = "not a number"
    )");

    auto config = Config::Load(filepath).value();

    auto result = config.GetRequired<i32>("width");
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Array Tests
// ============================================================================

TEST_F(ConfigTest, GetArrayInteger) {
    auto filepath = CreateTOML("test.toml", R"(
        resolution = [1280, 720]
    )");

    auto config = Config::Load(filepath).value();

    auto arr = config.GetArray<i32>("resolution");
    ASSERT_EQ(arr.size(), 2);
    EXPECT_EQ(arr[0], 1280);
    EXPECT_EQ(arr[1], 720);
}

TEST_F(ConfigTest, GetArrayFloat) {
    auto filepath = CreateTOML("test.toml", R"(
        position = [1.0, 2.5, -3.7]
    )");

    auto config = Config::Load(filepath).value();

    auto arr = config.GetArray<f32>("position");
    ASSERT_EQ(arr.size(), 3);
    EXPECT_NEAR(arr[0], 1.0f, 1e-6f);
    EXPECT_NEAR(arr[1], 2.5f, 1e-6f);
    EXPECT_NEAR(arr[2], -3.7f, 1e-6f);
}

TEST_F(ConfigTest, GetArrayString) {
    auto filepath = CreateTOML("test.toml", R"(
        bands = ["VIS_550", "NIR_850", "SWIR_1600"]
    )");

    auto config = Config::Load(filepath).value();

    auto arr = config.GetArray<String>("bands");
    ASSERT_EQ(arr.size(), 3);
    EXPECT_EQ(arr[0], "VIS_550");
    EXPECT_EQ(arr[1], "NIR_850");
    EXPECT_EQ(arr[2], "SWIR_1600");
}

TEST_F(ConfigTest, GetArrayMissing) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    auto arr = config.GetArray<i32>("missing_array");
    EXPECT_TRUE(arr.empty());
}

// ============================================================================
// Table Tests
// ============================================================================

TEST_F(ConfigTest, GetTableSuccess) {
    auto filepath = CreateTOML("test.toml", R"(
        [renderer]
        width = 1280
        height = 720

        [renderer.settings]
        quality = "high"
    )");

    auto config = Config::Load(filepath).value();

    auto tableResult = config.GetTable("renderer");
    ASSERT_TRUE(tableResult.has_value());

    Config rendererConfig = std::move(*tableResult);
    EXPECT_EQ(rendererConfig.Get<i32>("width", 0), 1280);
    EXPECT_EQ(rendererConfig.Get<i32>("height", 0), 720);
}

TEST_F(ConfigTest, GetTableMissing) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    auto result = config.GetTable("nonexistent");
    EXPECT_FALSE(result.has_value());
}

TEST_F(ConfigTest, GetTableNotATable) {
    auto filepath = CreateTOML("test.toml", R"(
        width = 1280
    )");

    auto config = Config::Load(filepath).value();

    auto result = config.GetTable("width");  // width is an integer, not a table
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Complex Nested Structure Tests
// ============================================================================

TEST_F(ConfigTest, ComplexNestedStructure) {
    auto filepath = CreateTOML("complex.toml", R"(
        [renderer]
        resolution = [1280, 720]
        spp = 4
        preset = "MS-RT"
        seconds_per_frame = true

        [spectral]
        band_samples = 4

        [[spectral.bands]]
        name = "VIS_550"
        center_nm = 550.0
        fwhm_nm = 40.0

        [[spectral.bands]]
        name = "NIR_850"
        center_nm = 850.0
        fwhm_nm = 30.0

        [atmosphere]
        mode = "LUT_FAST"
        lut = "modtran_fast.h5"
    )");

    auto config = Config::Load(filepath).value();

    // Test top-level access
    EXPECT_EQ(config.Get<i32>("renderer.spp", 0), 4);
    EXPECT_EQ(config.Get<String>("renderer.preset", ""), "MS-RT");
    EXPECT_TRUE(config.Get<bool>("renderer.seconds_per_frame", false));

    // Test array access
    auto resolution = config.GetArray<i32>("renderer.resolution");
    ASSERT_EQ(resolution.size(), 2);
    EXPECT_EQ(resolution[0], 1280);
    EXPECT_EQ(resolution[1], 720);

    // Test nested tables
    EXPECT_EQ(config.Get<i32>("spectral.band_samples", 0), 4);
    EXPECT_EQ(config.Get<String>("atmosphere.mode", ""), "LUT_FAST");
}
