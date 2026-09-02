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

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>

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

TEST(TypesTest, ParseSpectralModeRGB) {
    // "rgb" and "RGB" map to the fast RGB mode (default)
    auto res1 = ParseSpectralMode("rgb");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::RGB);

    auto res2 = ParseSpectralMode("RGB");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::RGB);
}

TEST(TypesTest, ParseSpectralModeVISFused) {
    // Both visible modes are reachable by their own name.
    auto res1 = ParseSpectralMode("vis_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::VIS_Fused);

    auto res2 = ParseSpectralMode("vis_hero");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::VIS_Hero);
}

TEST(TypesTest, TheBandAliasNamesTheSampledMode) {
    // "VIS" is the band, and asking for a band rather than an estimator gets
    // the one to render with: four sampled wavelengths a path, which follows
    // n(lambda) through a dispersive interface. vis_fused stays reachable, and
    // only by name, because what it is for is being the reference.
    auto res = ParseSpectralMode("VIS");
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), SpectralMode::VIS_Hero);
    EXPECT_TRUE(IsVisMode(res.value()));
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

TEST(TypesTest, ParseSpectralModeNIR) {
    auto res1 = ParseSpectralMode("nir_fused");
    EXPECT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), SpectralMode::NIR_Fused);

    auto res2 = ParseSpectralMode("NIR");
    EXPECT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), SpectralMode::NIR_Fused);
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
    //   #define SPECTRAL_MODE_VIS_FUSED    1
    //   #define SPECTRAL_MODE_MULTISPECTRAL 2
    //   #define SPECTRAL_MODE_MWIR_FUSED   3
    //   #define SPECTRAL_MODE_LWIR_FUSED   4
    //   #define SPECTRAL_MODE_SWIR_FUSED   5
    //   #define SPECTRAL_MODE_NIR_FUSED    6
    //   #define SPECTRAL_MODE_RGB          7
    //   #define SPECTRAL_MODE_VIS_HERO     8

    EXPECT_EQ(static_cast<u32>(SpectralMode::Single), 0u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::VIS_Fused), 1u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::Multispectral), 2u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::MWIR_Fused), 3u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::LWIR_Fused), 4u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::SWIR_Fused), 5u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::NIR_Fused), 6u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::RGB), 7u);
    EXPECT_EQ(static_cast<u32>(SpectralMode::VIS_Hero), 8u);
}

// The comment above is a promise; this reads the shader and checks it. The
// value travels as a specialization constant, so a renumbering that compiles
// on both sides silently renders one mode's scene with another's estimator.
TEST(TypesTest, ShaderSpectralModeDefinesMatchTheEnum) {
    const std::filesystem::path hlsl =
        std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "src" / "shaders" / "common.hlsli";
    std::ifstream in(hlsl);
    ASSERT_TRUE(in.is_open()) << "cannot open " << hlsl.string();

    std::unordered_map<std::string, u32> defines;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string hash, name, value;
        if (!(ls >> hash >> name >> value)) continue;
        if (hash != "#define") continue;
        if (name.rfind("SPECTRAL_MODE_", 0) != 0) continue;
        defines[name] = static_cast<u32>(std::stoul(value));
    }

    const std::pair<const char*, SpectralMode> modes[] = {
        {"SPECTRAL_MODE_SINGLE",        SpectralMode::Single},
        {"SPECTRAL_MODE_VIS_FUSED",     SpectralMode::VIS_Fused},
        {"SPECTRAL_MODE_MULTISPECTRAL", SpectralMode::Multispectral},
        {"SPECTRAL_MODE_MWIR_FUSED",    SpectralMode::MWIR_Fused},
        {"SPECTRAL_MODE_LWIR_FUSED",    SpectralMode::LWIR_Fused},
        {"SPECTRAL_MODE_SWIR_FUSED",    SpectralMode::SWIR_Fused},
        {"SPECTRAL_MODE_NIR_FUSED",     SpectralMode::NIR_Fused},
        {"SPECTRAL_MODE_RGB",           SpectralMode::RGB},
        {"SPECTRAL_MODE_VIS_HERO",      SpectralMode::VIS_Hero},
    };
    EXPECT_EQ(defines.size(), std::size(modes))
        << "common.hlsli defines a mode the enum does not list, or the reverse";
    for (const auto& [name, mode] : modes) {
        ASSERT_TRUE(defines.count(name)) << "missing " << name;
        EXPECT_EQ(defines[name], static_cast<u32>(mode)) << name;
    }
}

TEST(TypesTest, SpectralModeEnumSize) {
    // SpectralMode is backed by u32 for GPU compatibility
    EXPECT_EQ(sizeof(SpectralMode), sizeof(u32));
}

TEST(TypesTest, SpectralModeAllModesAccepted) {
    // Verify all modes have at least one valid string representation
    const char* modeStrings[] = {
        "single",        // Single
        "rgb",           // RGB (default fast mode)
        "vis_fused",     // VIS_Fused (spectral integration)
        "multispectral", // Multispectral
        "mwir_fused",    // MWIR_Fused
        "lwir_fused",    // LWIR_Fused
        "swir_fused",    // SWIR_Fused
        "nir_fused",     // NIR_Fused
        "vis_hero"       // VIS_Hero (hero-wavelength sampling of the same band)
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

    // NIR band (Near-Infrared) - reflected solar
    EXPECT_EQ(WAVELENGTH_MIN_NIR, 780.0f);
    EXPECT_EQ(WAVELENGTH_MAX_NIR, 1400.0f);

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
    EXPECT_EQ(WAVELENGTH_MAX_VISIBLE, WAVELENGTH_MIN_NIR);  // NIR starts where visible ends
    EXPECT_LT(WAVELENGTH_MAX_NIR, WAVELENGTH_MAX_SWIR);     // NIR overlaps with SWIR
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

// ============================================================================
// Fused Band Info Tests
// ============================================================================
// GetFusedBandInfo() is the single source of truth for fused-band wavelength
// ranges on the CPU side. Values MUST match the shader constants in
// closesthit.rchit / miss.rmiss exactly.
// ============================================================================

TEST(TypesTest, FusedBandInfoLWIR) {
    auto band = GetFusedBandInfo(SpectralMode::LWIR_Fused);
    ASSERT_TRUE(band.has_value());
    EXPECT_FLOAT_EQ(band->lambdaMinNm, 8000.0f);
    EXPECT_FLOAT_EQ(band->lambdaMaxNm, 12000.0f);
    EXPECT_FLOAT_EQ(band->CenterNm(), 10000.0f);
    EXPECT_FLOAT_EQ(band->WidthNm(), 4000.0f);
}

TEST(TypesTest, FusedBandInfoMWIR) {
    auto band = GetFusedBandInfo(SpectralMode::MWIR_Fused);
    ASSERT_TRUE(band.has_value());
    EXPECT_FLOAT_EQ(band->lambdaMinNm, 3000.0f);
    EXPECT_FLOAT_EQ(band->lambdaMaxNm, 5000.0f);
    EXPECT_FLOAT_EQ(band->CenterNm(), 4000.0f);
    EXPECT_FLOAT_EQ(band->WidthNm(), 2000.0f);
}

TEST(TypesTest, FusedBandInfoSWIR) {
    auto band = GetFusedBandInfo(SpectralMode::SWIR_Fused);
    ASSERT_TRUE(band.has_value());
    EXPECT_FLOAT_EQ(band->lambdaMinNm, 1400.0f);
    EXPECT_FLOAT_EQ(band->lambdaMaxNm, 2400.0f);
    EXPECT_FLOAT_EQ(band->CenterNm(), 1900.0f);
    EXPECT_FLOAT_EQ(band->WidthNm(), 1000.0f);
}

TEST(TypesTest, FusedBandInfoNIR) {
    auto band = GetFusedBandInfo(SpectralMode::NIR_Fused);
    ASSERT_TRUE(band.has_value());
    EXPECT_FLOAT_EQ(band->lambdaMinNm, 930.0f);
    EXPECT_FLOAT_EQ(band->lambdaMaxNm, 1200.0f);
}

TEST(TypesTest, FusedBandInfoVIS) {
    auto band = GetFusedBandInfo(SpectralMode::VIS_Fused);
    ASSERT_TRUE(band.has_value());
    EXPECT_FLOAT_EQ(band->lambdaMinNm, 400.0f);
    EXPECT_FLOAT_EQ(band->lambdaMaxNm, 780.0f);
}

TEST(TypesTest, FusedBandInfoNoneForNonFusedModes) {
    EXPECT_FALSE(GetFusedBandInfo(SpectralMode::RGB).has_value());
    EXPECT_FALSE(GetFusedBandInfo(SpectralMode::Single).has_value());
    EXPECT_FALSE(GetFusedBandInfo(SpectralMode::Multispectral).has_value());
}

// The tests above pin the C++ values. This one pins the HLSL side to them by
// reading common.hlsli, which is the only way to catch the two drifting apart:
// the shaders cannot include Types.hpp, and a mismatch changes rendered output
// without failing to compile. The SWIR and NIR ranges were documented wrong for
// exactly this reason.
TEST(TypesTest, ShaderBandConstantsMatchGetFusedBandInfo) {
    const std::filesystem::path hlsl =
        std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "src" / "shaders" / "common.hlsli";
    std::ifstream in(hlsl);
    ASSERT_TRUE(in.is_open()) << "cannot open " << hlsl.string();

    std::unordered_map<std::string, float> defines;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string hash, name, value;
        if (!(ls >> hash >> name >> value)) continue;
        if (hash != "#define") continue;
        if (name.rfind("SPECTRAL_", 0) != 0 || name.find("_LAMBDA_") == std::string::npos) continue;
        defines[name] = std::stof(value);
    }
    ASSERT_EQ(defines.size(), 10u) << "expected 10 SPECTRAL_*_LAMBDA_* defines in common.hlsli";

    const std::pair<SpectralMode, const char*> bands[] = {
        {SpectralMode::VIS_Fused,  "VIS"},
        {SpectralMode::NIR_Fused,  "NIR"},
        {SpectralMode::SWIR_Fused, "SWIR"},
        {SpectralMode::MWIR_Fused, "MWIR"},
        {SpectralMode::LWIR_Fused, "LWIR"},
    };
    for (const auto& [mode, name] : bands) {
        auto band = GetFusedBandInfo(mode);
        ASSERT_TRUE(band.has_value()) << name;
        const std::string lo = std::string("SPECTRAL_") + name + "_LAMBDA_MIN";
        const std::string hi = std::string("SPECTRAL_") + name + "_LAMBDA_MAX";
        ASSERT_TRUE(defines.count(lo)) << "missing " << lo;
        ASSERT_TRUE(defines.count(hi)) << "missing " << hi;
        EXPECT_FLOAT_EQ(defines[lo], band->lambdaMinNm) << name << " min";
        EXPECT_FLOAT_EQ(defines[hi], band->lambdaMaxNm) << name << " max";
    }
}

TEST(TypesTest, IsThermalIRFusedMode) {
    // Thermal-capable fused bands: emission term matters, sensor needs
    // band-integrated radiance and band-center photon wavelength.
    EXPECT_TRUE(IsIRFusedMode(SpectralMode::SWIR_Fused));
    EXPECT_TRUE(IsIRFusedMode(SpectralMode::MWIR_Fused));
    EXPECT_TRUE(IsIRFusedMode(SpectralMode::LWIR_Fused));
    EXPECT_TRUE(IsIRFusedMode(SpectralMode::NIR_Fused));
    EXPECT_FALSE(IsIRFusedMode(SpectralMode::RGB));
    EXPECT_FALSE(IsIRFusedMode(SpectralMode::VIS_Fused));
    EXPECT_FALSE(IsIRFusedMode(SpectralMode::VIS_Hero));
    EXPECT_FALSE(IsIRFusedMode(SpectralMode::Single));
}

TEST(TypesTest, TheTwoVisibleModesShareABand) {
    EXPECT_TRUE(IsVisMode(SpectralMode::VIS_Fused));
    EXPECT_TRUE(IsVisMode(SpectralMode::VIS_Hero));
    EXPECT_FALSE(IsVisMode(SpectralMode::RGB));
    EXPECT_FALSE(IsVisMode(SpectralMode::Single));
    EXPECT_FALSE(IsVisMode(SpectralMode::NIR_Fused));

    // Same band edges, or the two are not comparable and neither can serve as
    // the other's reference.
    const auto fused = GetFusedBandInfo(SpectralMode::VIS_Fused);
    const auto hero = GetFusedBandInfo(SpectralMode::VIS_Hero);
    ASSERT_TRUE(fused.has_value());
    ASSERT_TRUE(hero.has_value());
    EXPECT_FLOAT_EQ(fused->lambdaMinNm, hero->lambdaMinNm);
    EXPECT_FLOAT_EQ(fused->lambdaMaxNm, hero->lambdaMaxNm);
}
