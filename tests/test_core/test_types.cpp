// ============================================================================
// Quantiloom - Unit Tests for core/Types.hpp
// ============================================================================
// Tests cover:
// - Result type operations (success/error handling)
// - SpectralMode parsing
// - ErrorCode conversion
// - Constants validation
// ============================================================================

#include <gtest/gtest.h>
#include "core/Types.hpp"

using namespace quantiloom;

// ============================================================================
// Result Type Tests
// ============================================================================

TEST(TypesTest, ResultSuccessConstruction) {
    Result<i32, String> res = 42;

    EXPECT_TRUE(res.has_value());
    EXPECT_TRUE(res);
    EXPECT_EQ(res.value(), 42);
    EXPECT_EQ(*res, 42);
}

TEST(TypesTest, ResultErrorConstruction) {
    Result<i32, String> res = Result<i32, String>::Err("Test error");

    EXPECT_FALSE(res.has_value());
    EXPECT_FALSE(res);
    EXPECT_EQ(res.error(), "Test error");
}

TEST(TypesTest, ResultMoveSemantics) {
    Result<String, i32> res = String("Success value");

    EXPECT_TRUE(res.has_value());
    String moved = std::move(res).value();
    EXPECT_EQ(moved, "Success value");
}

TEST(TypesTest, ResultCopySemantics) {
    Result<i32, String> res1 = 100;
    Result<i32, String> res2 = res1;

    EXPECT_TRUE(res1.has_value());
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res1.value(), 100);
    EXPECT_EQ(res2.value(), 100);
}

TEST(TypesTest, ResultErrorHelper) {
    auto makeError = []() -> Result<i32, String> {
        return Result<i32, String>::Err("Failed");
    };

    auto res = makeError();
    EXPECT_FALSE(res.has_value());
    EXPECT_EQ(res.error(), "Failed");
}

// ============================================================================
// SpectralMode Parsing Tests
// ============================================================================

TEST(TypesTest, ParseSpectralModeSingle) {
    auto res1 = ParseSpectralMode("single");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::Single);

    auto res2 = ParseSpectralMode("single_wavelength");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::Single);
}

TEST(TypesTest, ParseSpectralModeRGBFused) {
    auto res1 = ParseSpectralMode("rgb_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::RGB_Fused);

    // Legacy name for backward compatibility
    auto res2 = ParseSpectralMode("rgb");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::RGB_Fused);

    auto res3 = ParseSpectralMode("RGB");
    EXPECT_TRUE(res3.has_value());
    EXPECT_EQ(res3.value(), SpectralMode::RGB_Fused);
}

TEST(TypesTest, ParseSpectralModeMultispectral) {
    auto res = ParseSpectralMode("multispectral");
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), SpectralMode::Multispectral);
}

TEST(TypesTest, ParseSpectralModeMWIR) {
    auto res1 = ParseSpectralMode("mwir_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::MWIR_Fused);

    auto res2 = ParseSpectralMode("MWIR");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::MWIR_Fused);
}

TEST(TypesTest, ParseSpectralModeLWIR) {
    auto res1 = ParseSpectralMode("lwir_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::LWIR_Fused);

    auto res2 = ParseSpectralMode("LWIR");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::LWIR_Fused);
}

TEST(TypesTest, ParseSpectralModeSWIR) {
    auto res1 = ParseSpectralMode("swir_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::SWIR_Fused);

    auto res2 = ParseSpectralMode("SWIR");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::SWIR_Fused);
}

TEST(TypesTest, ParseSpectralModeInvalid) {
    auto res = ParseSpectralMode("invalid_mode");
    EXPECT_FALSE(res.has_value());
    EXPECT_FALSE(res.error().empty());
}

// ============================================================================
// SpectralMode Enum Value Tests
// ============================================================================
// These tests verify that SpectralMode enum values match GPU shader defines
// in common.hlsli. If these fail, CPU/GPU rendering mode selection will break.
// ============================================================================

TEST(TypesTest, SpectralModeEnumValues) {
    // CRITICAL: These values MUST match #define SPECTRAL_MODE_* in common.hlsli
    // Shader defines:
    //   #define SPECTRAL_MODE_SINGLE       0
    //   #define SPECTRAL_MODE_RGB_FUSED    1
    //   #define SPECTRAL_MODE_MULTISPECTRAL 2
    //   #define SPECTRAL_MODE_MWIR_FUSED   3
    //   #define SPECTRAL_MODE_LWIR_FUSED   4
    //   #define SPECTRAL_MODE_SWIR_FUSED   5

    EXPECT_EQ(static_cast<u32>(SpectralMode::Single), 0u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::RGB_Fused), 1u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::Multispectral), 2u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::MWIR_Fused), 3u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::LWIR_Fused), 4u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::SWIR_Fused), 5u);
}

TEST(TypesTest, SpectralModeEnumSize) {
    // SpectralMode is backed by u32 for GPU compatibility
    EXPECT_EQ(sizeof(SpectralMode), sizeof(u32));
}

TEST(TypesTest, SpectralModeAllModesAccepted) {
    // Verify all modes have at least one valid string representation
    const char* modeStrings[] = {
        "single",        // Single
        "rgb_fused",     // RGB_Fused
        "multispectral", // Multispectral
        "mwir_fused",    // MWIR_Fused
        "lwir_fused",    // LWIR_Fused
        "swir_fused"     // SWIR_Fused
    };

    for (const char* modeStr : modeStrings) {
        auto res = ParseSpectralMode(modeStr);
        EXPECT_TRUE(res.has_value()) << "Failed to parse: " << modeStr;
    }
}

// ============================================================================
// ErrorCode Tests
// ============================================================================

TEST(TypesTest, ErrorCodeToStringSuccess) {
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::Success), "Success");
}

TEST(TypesTest, ErrorCodeToStringFileErrors) {
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::FileNotFound), "File not found");
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::FileReadError), "File read error");
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::FileWriteError), "File write error");
}

TEST(TypesTest, ErrorCodeToStringConfigErrors) {
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::ConfigParseError), "Config parse error");
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::ConfigMissingKey), "Config missing key");
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::ConfigInvalidValue), "Config invalid value");
}

TEST(TypesTest, ErrorCodeToStringVulkanErrors) {
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::VulkanInitFailed), "Vulkan initialization failed");
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::VulkanDeviceNotFound), "Vulkan device not found");
}

TEST(TypesTest, ErrorCodeToStringUnknown) {
    EXPECT_STREQ(ErrorCodeToString(ErrorCode::Unknown), "Unknown error");
}

// ============================================================================
// Constants Tests
// ============================================================================

TEST(TypesTest, MathConstants) {
    using namespace constants;

    EXPECT_NEAR(PI, 3.14159265358979323846, 1e-15);
    EXPECT_NEAR(TWO_PI, 2.0 * PI, 1e-15);
    EXPECT_NEAR(INV_PI, 1.0 / PI, 1e-15);
    EXPECT_NEAR(PI * INV_PI, 1.0, 1e-15);
}

TEST(TypesTest, SpectralRangeConstants) {
    using namespace constants;

    // Visible spectrum (CIE standard)
    EXPECT_EQ(WAVELENGTH_MIN_VISIBLE, 380.0f);
    EXPECT_EQ(WAVELENGTH_MAX_VISIBLE, 780.0f);

    // SWIR band (Short-Wave Infrared)
    EXPECT_EQ(WAVELENGTH_MIN_SWIR, 1000.0f);
    EXPECT_EQ(WAVELENGTH_MAX_SWIR, 2500.0f);

    // MWIR band (Mid-Wave Infrared)
    EXPECT_EQ(WAVELENGTH_MIN_MWIR, 3000.0f);
    EXPECT_EQ(WAVELENGTH_MAX_MWIR, 5000.0f);

    // LWIR band (Long-Wave Infrared)
    EXPECT_EQ(WAVELENGTH_MIN_LWIR, 8000.0f);
    EXPECT_EQ(WAVELENGTH_MAX_LWIR, 12000.0f);

    // Bands should be in increasing order
    EXPECT_LT(WAVELENGTH_MAX_VISIBLE, WAVELENGTH_MIN_SWIR);
    EXPECT_LT(WAVELENGTH_MAX_SWIR, WAVELENGTH_MIN_MWIR);
    EXPECT_LT(WAVELENGTH_MAX_MWIR, WAVELENGTH_MIN_LWIR);
}

TEST(TypesTest, PhysicalConstants) {
    using namespace constants;

    EXPECT_EQ(SPEED_OF_LIGHT, 299792458.0);
    EXPECT_EQ(PLANCK_CONSTANT, 6.62607015e-34);
}

// ============================================================================
// Type Alias Tests (compile-time verification)
// ============================================================================

TEST(TypesTest, IntegerTypeSizes) {
    EXPECT_EQ(sizeof(i8), 1);
    EXPECT_EQ(sizeof(i16), 2);
    EXPECT_EQ(sizeof(i32), 4);
    EXPECT_EQ(sizeof(i64), 8);

    EXPECT_EQ(sizeof(u8), 1);
    EXPECT_EQ(sizeof(u16), 2);
    EXPECT_EQ(sizeof(u32), 4);
    EXPECT_EQ(sizeof(u64), 8);
}

TEST(TypesTest, FloatTypeSizes) {
    EXPECT_EQ(sizeof(f32), 4);
    EXPECT_EQ(sizeof(f64), 8);
}

TEST(TypesTest, WavelengthType) {
    static_assert(std::is_same_v<Wavelength, f32>, "Wavelength must be f32");
}

// ============================================================================
// Concept Tests (C++20)
// ============================================================================

TEST(TypesTest, ArithmeticConcept) {
    static_assert(Arithmetic<i32>, "i32 should satisfy Arithmetic concept");
    static_assert(Arithmetic<f32>, "f32 should satisfy Arithmetic concept");
    static_assert(!Arithmetic<String>, "String should not satisfy Arithmetic concept");
}

TEST(TypesTest, NumericConcept) {
    static_assert(Numeric<i32>, "i32 should satisfy Numeric concept");
    static_assert(Numeric<f32>, "f32 should satisfy Numeric concept");
    static_assert(Numeric<f64>, "f64 should satisfy Numeric concept");
}
