#include <gtest/gtest.h>

#include "postprocess/CameraIsp.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

CameraConfig MonoPhotonConfig(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
    config.device.detector = DetectorKind::Photon;
    config.device.cfa = CfaPattern::Mono;
    ResponseCurve qe;
    qe.kind = ResponseKind::AbsoluteQE;
    qe.wavelengthNm = {500.0, 600.0};
    qe.value = {0.5, 0.5};
    ResponseStack response;
    response.quantumEfficiency = qe;
    config.device.channels.push_back({"Mono", response});
    config.optics.sensorWidthPx = width;
    config.optics.sensorHeightPx = height;
    config.optics.fNumber = 2.0;
    config.optics.pixelPitchUm = 5.0;
    config.readout.exposureSeconds = 0.01;
    config.readout.framePeriodSeconds = 0.1;
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    // The display chain normalizes the corrected signal by the well capacity;
    // 16383 keeps DN and electrons 1:1 for the synthetic frames below.
    config.photon.fullWellElectrons = 16383.0;
    config.quality.noiseFree = true;
    config.products.display = true;
    return config;
}

CameraConfig BayerPhotonConfig(u32 width, u32 height) {
    CameraConfig config = MonoPhotonConfig(width, height);
    config.device.cfa = CfaPattern::RGGB;
    ResponseCurve qe;
    qe.kind = ResponseKind::AbsoluteQE;
    qe.wavelengthNm = {500.0, 600.0};
    qe.value = {0.5, 0.5};
    for (const char* name : {"R", "G", "B"}) {
        ResponseStack response;
        response.quantumEfficiency = qe;
        config.device.channels.push_back({name, response});
    }
    return config;
}

CameraConfig ThermalMonoConfig(u32 width, u32 height) {
    CameraConfig config = MonoPhotonConfig(width, height);
    config.device.detector = DetectorKind::Thermal;
    ResponseCurve absorptance;
    absorptance.kind = ResponseKind::ThermalAbsorptance;
    absorptance.wavelengthNm = {8000.0, 14000.0};
    absorptance.value = {1.0, 1.0};
    config.device.channels.clear();
    ResponseStack response;
    response.thermalAbsorptance = absorptance;
    config.device.channels.push_back({"Mono", response});
    config.thermal.timeConstantSeconds = 0.008;
    config.thermal.responsivityDnPerWatt = 1e13;
    return config;
}

f64 FullScale(const CameraConfig& config) {
    // The photon display branch normalizes the corrected device signal (in
    // electrons) by the well capacity; the synthetic tests keep the two equal
    // so DN values read directly as well fractions.
    return config.photon.fullWellElectrons;
}

// Builds the RAW DN product shape RunIsp expects.
Image MakeRaw(const CameraConfig& config,
              const std::function<f64(u32, u32, u32)>& value) {
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    const u32 channels = config.device.cfa == CfaPattern::MultiChannel ?
        static_cast<u32>(config.device.channels.size()) : 1u;
    Image raw(width, height, channels);
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x)
            for (u32 c = 0; c < channels; ++c)
                raw(x, y, c) = static_cast<f32>(value(x, y, c));
    return raw;
}

Result<Image, String> RunOnRaw(const CameraConfig& config, const Image& raw) {
    CaptureState state;
    return RunIsp(config, raw, raw, state, 0.0);
}

f64 ExpectedSrgb(f64 linear) {
    const f64 clamped = std::clamp(linear, 0.0, 1.0);
    return clamped <= 0.0031308 ? 12.92 * clamped :
           1.055 * std::pow(clamped, 1.0 / 2.4) - 0.055;
}

// ---------------------------------------------------------------------------
// sRGB decode and CIE 1976 for the colorimetry tests
// ---------------------------------------------------------------------------

std::array<f64, 3> SrgbToLinear(f64 r, f64 g, f64 b) {
    const auto channel = [](f64 v) {
        v = std::clamp(v, 0.0, 1.0);
        return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
    };
    return {channel(r), channel(g), channel(b)};
}

f64 DeltaE76(const std::array<f64, 3>& a, const std::array<f64, 3>& b) {
    const auto labOf = [](const std::array<f64, 3>& rgb) {
        const f64 x = 0.4124564 * rgb[0] + 0.3575761 * rgb[1] + 0.1804375 * rgb[2];
        const f64 y = 0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.0721750 * rgb[2];
        const f64 z = 0.0193339 * rgb[0] + 0.1191920 * rgb[1] + 0.9503041 * rgb[2];
        const auto f = [](f64 v) {
            const f64 epsilon = 216.0 / 24389.0;
            const f64 kappa = 24389.0 / 27.0;
            return v > epsilon ? std::cbrt(v) : (kappa * v + 16.0) / 116.0;
        };
        const f64 fx = f(x / 0.95047), fy = f(y), fz = f(z / 1.08883);
        return std::array<f64, 3>{116.0 * fy - 16.0, 500.0 * (fx - fy),
                                  200.0 * (fy - fz)};
    };
    const auto la = labOf(a), lb = labOf(b);
    return std::sqrt((la[0] - lb[0]) * (la[0] - lb[0]) +
                     (la[1] - lb[1]) * (la[1] - lb[1]) +
                     (la[2] - lb[2]) * (la[2] - lb[2]));
}

// ---------------------------------------------------------------------------
// HLSL palette parsing: the CPU ramps must be the HLSL control points
// ---------------------------------------------------------------------------

std::filesystem::path ShaderSource(const char* name) {
    return std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "src" / "shaders" / name;
}

// Extracts the float3 control-point literals of one palette function from the
// HLSL text. Prefers the shared display_palettes.hlsli once it exists, and
// falls back to clahe.comp.hlsl where the ramps live today.
std::vector<std::array<f64, 3>> ParseHlslPalette(StringView functionName) {
    std::filesystem::path path = ShaderSource("display_palettes.hlsli");
    if (!std::filesystem::exists(path)) path = ShaderSource("clahe.comp.hlsl");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open palette shader source");
    std::ostringstream text;
    text << input.rdbuf();
    const String source = text.str();
    const size_t functionAt = source.find(String(functionName));
    if (functionAt == String::npos)
        throw std::runtime_error("palette function not found in HLSL");
    const size_t tableAt = source.find("static const float3", functionAt);
    if (tableAt == String::npos)
        throw std::runtime_error("palette table not found in HLSL");
    const size_t endAt = source.find("};", tableAt);
    const String table = source.substr(tableAt, endAt - tableAt);
    static const std::regex point(
        R"(float3\(\s*([-0-9.eE+]+)f?\s*,\s*([-0-9.eE+]+)f?\s*,\s*([-0-9.eE+]+)f?\s*\))");
    std::vector<std::array<f64, 3>> points;
    for (auto it = std::sregex_iterator(table.begin(), table.end(), point);
         it != std::sregex_iterator(); ++it) {
        points.push_back({std::stod((*it)[1]), std::stod((*it)[2]),
                          std::stod((*it)[3])});
    }
    if (points.size() < 2) throw std::runtime_error("no palette points parsed");
    return points;
}

std::filesystem::path UniqueTemporaryDirectory(const char* name) {
    std::random_device random;
    for (u32 attempt = 0; attempt < 32; ++attempt) {
        const auto path = std::filesystem::temp_directory_path() /
            (String(name) + "_" + std::to_string(random()) + "_" +
             std::to_string(attempt));
        if (std::filesystem::create_directory(path)) return path;
    }
    throw std::runtime_error("could not create a unique camera test directory");
}

} // namespace

// ---------------------------------------------------------------------------
// MHC golden values, hand-derived from the published 5x5 kernels
// ---------------------------------------------------------------------------

TEST(CameraIspTest, MalvarHeCutlerGoldenPixels) {
    // 8x8 RGGB frame, dn(x,y) = (3x + 7y) mod 16, 4-bit ADC (full scale 15).
    // Hand-computed expectations at the pixels below (paper kernels, /8):
    //   (2,2) R site:     R = 4,     G = 6,     B = 5
    //   (3,2) G in R row: R = 7,     G = 7,     B = 7
    //   (3,3) B site:     R = 12,    G = 10,    B = 14
    CameraConfig config = BayerPhotonConfig(8, 8);
    config.readout.adcBits = 4;
    config.readout.outputBits = 4;
    config.photon.fullWellElectrons = 15.0;
    const auto raw = MakeRaw(config, [](u32 x, u32 y, u32) {
        return static_cast<f64>((3 * x + 7 * y) % 16);
    });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    const auto expect = [&display](u32 x, u32 y, u32 c, f64 linear) {
        EXPECT_NEAR(display.value().operator()(x, y, c), ExpectedSrgb(linear / 15.0),
                    1e-6)
            << "pixel (" << x << "," << y << ") channel " << c;
    };
    expect(2, 2, 0, 4.0);     // R at R (the sample itself)
    expect(2, 2, 1, 6.0);     // G at R
    expect(2, 2, 2, 5.0);     // B at R
    expect(3, 2, 0, 7.0);     // R at G in an R row
    expect(3, 2, 1, 7.0);     // G at G (the sample itself)
    expect(3, 2, 2, 7.0);     // B at G in an R row
    expect(3, 3, 0, 12.0);    // R at B
    expect(3, 3, 1, 10.0);    // G at B
    expect(3, 3, 2, 14.0);    // B at B (the sample itself)
}

// ---------------------------------------------------------------------------
// Single-step unit tests on a 1x1 mono frame
// ---------------------------------------------------------------------------

TEST(CameraIspTest, BlackLevelIsRemovedBeforeTheDisplayChain) {
    CameraConfig config = MonoPhotonConfig(2, 2);
    config.readout.blackLevelDn = 40.0;
    config.photon.fullWellElectrons = 255.0;
    // 600 e-/s over the 0.01 s exposure gives 6 e-, which quantizes to
    // 6 + 40 = 46 DN; the display must see the corrected 6, not the 46.
    Image measured(2, 2, 1);
    std::fill(measured.data.begin(), measured.data.end(), 600.0f);
    CaptureState state;
    const auto output =
        CpuCameraPipeline(config).CaptureMeasured(state, 0.0, measured);
    ASSERT_TRUE(output.has_value());
    EXPECT_FLOAT_EQ(output.value().rawDn.value().image.data[0], 46.0f);
    for (size_t i = 0; i < output.value().display.value().image.data.size(); ++i)
        EXPECT_NEAR(output.value().display.value().image.data[i],
                    ExpectedSrgb(6.0 / 255.0), 1e-6);
}

TEST(CameraIspTest, WhiteBalanceScalesTheMonoPlane) {
    CameraConfig config = MonoPhotonConfig(1, 1);
    config.isp.whiteBalance = {2.0, 1.0, 1.0};
    const auto raw = MakeRaw(config, [](u32, u32, u32) { return 5.0; });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    EXPECT_NEAR(display.value().data[0],
                ExpectedSrgb(10.0 / FullScale(config)), 1e-7);
}

TEST(CameraIspTest, ColorMatrixMapsEachOutputChannel) {
    CameraConfig config = MonoPhotonConfig(1, 1);
    // R' = 2R, G' = 0.5G, B' = 0.25B over the replicated mono triple.
    config.isp.deviceToLinearSrgb = {2.0, 0.0, 0.0,
                                     0.0, 0.5, 0.0,
                                     0.0, 0.0, 0.25};
    const auto raw = MakeRaw(config, [](u32, u32, u32) { return 4.0; });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    const f64 base = 4.0 / FullScale(config);
    EXPECT_NEAR(display.value().data[0], ExpectedSrgb(2.0 * base), 1e-7);
    EXPECT_NEAR(display.value().data[1], ExpectedSrgb(0.5 * base), 1e-7);
    EXPECT_NEAR(display.value().data[2], ExpectedSrgb(0.25 * base), 1e-7);
}

TEST(CameraIspTest, ToneGammaAppliesAPowerCurve) {
    CameraConfig config = MonoPhotonConfig(1, 1);
    config.isp.toneGamma = 2.0;
    const auto raw = MakeRaw(config, [](u32, u32, u32) { return 5.0; });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    EXPECT_NEAR(display.value().data[0],
                ExpectedSrgb(std::sqrt(5.0 / FullScale(config))), 1e-7);
}

// ---------------------------------------------------------------------------
// Defect pixel repair
// ---------------------------------------------------------------------------

TEST(CameraIspTest, DefectPixelsAreReplacedByNeighborMean) {
    CameraConfig config = MonoPhotonConfig(5, 5);
    auto raw = MakeRaw(config, [](u32, u32, u32) { return 10.0; });
    raw(2, 2, 0) = 999.0f;
    raw(3, 2, 0) = 999.0f; // adjacent defect: must be skipped as a neighbor
    raw(0, 0, 0) = 999.0f; // border defect: only two in-range neighbors
    config.isp.defectPixels = {{2, 2}, {3, 2}, {0, 0}};
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    const f64 repaired = ExpectedSrgb(10.0 / FullScale(config));
    for (const auto [x, y] : {std::pair{2u, 2u}, {3u, 2u}, {0u, 0u}})
        EXPECT_NEAR(display.value().operator()(x, y, 0), repaired, 1e-7)
            << "defect pixel (" << x << "," << y << ")";
}

// ---------------------------------------------------------------------------
// Bayer diagonal edge: no directional error blow-up
// ---------------------------------------------------------------------------

TEST(CameraIspTest, BayerDiagonalEdgeKeepsInterpolationErrorBounded) {
    constexpr u32 kSize = 24;
    CameraConfig config = BayerPhotonConfig(kSize, kSize);
    // A smooth grey step running along the diagonal x - y = 12.
    const auto truth = [](u32 x, u32 y) {
        return 0.5 + 0.5 * std::tanh((static_cast<f64>(x) - y - 12) / 2.0);
    };
    const auto raw = MakeRaw(config, [&](u32 x, u32 y, u32) {
        return std::lround(truth(x, y) * FullScale(config));
    });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    // Error in linear light at G sites on each side of the edge.
    const auto sideError = [&](bool upperSide) {
        f64 worst = 0.0;
        for (u32 y = 4; y + 4 < kSize; ++y)
            for (u32 x = 4; x + 4 < kSize; ++x) {
                const bool side = static_cast<i32>(x) - static_cast<i32>(y) > 12;
                if (side != upperSide) continue;
                // Decode the encoded display value back to linear.
                const f64 encoded = display.value().operator()(x, y, 1);
                const f64 linear = encoded <= 0.04045 ?
                    encoded / 12.92 :
                    std::pow((encoded + 0.055) / 1.055, 2.4);
                worst = std::max(worst, std::fabs(linear - truth(x, y)));
            }
        return worst;
    };
    const f64 upper = sideError(true);
    const f64 lower = sideError(false);
    EXPECT_LT(upper, 0.05);
    EXPECT_LT(lower, 0.05);
    // No directionality: neither side's error may dominate the other's.
    EXPECT_LT(std::max(upper, lower),
              2.0 * (std::min(upper, lower) + 1e-6));
}

// ---------------------------------------------------------------------------
// Colorimetry: a self-calibrated chart is exact, a misplaced CCM is bounded
// ---------------------------------------------------------------------------

TEST(CameraIspTest, ColorCheckerDeltaEIsZeroForSelfCalibratedCcm) {
    // 24 chart patches spanning hue and lightness.
    std::vector<std::array<f64, 3>> reference;
    for (u32 i = 0; i < 24; ++i) {
        const f64 hue = static_cast<f64>(i % 12) / 12.0;
        const f64 value = 0.25 + 0.6 * static_cast<f64>(i / 12);
        const f64 sat = 0.65;
        // HSV -> linear sRGB.
        const auto sector = [&](f64 v) {
            const f64 h = hue * 6.0;
            const f64 c = v * sat;
            const f64 x = c * (1.0 - std::fabs(std::fmod(h, 2.0) - 1.0));
            const f64 m = v - c;
            if (h < 1.0) return std::array<f64, 3>{c + m, x + m, m};
            if (h < 2.0) return std::array<f64, 3>{x + m, c + m, m};
            if (h < 3.0) return std::array<f64, 3>{m, c + m, x + m};
            if (h < 4.0) return std::array<f64, 3>{m, x + m, c + m};
            if (h < 5.0) return std::array<f64, 3>{x + m, m, c + m};
            return std::array<f64, 3>{c + m, m, x + m};
        };
        // sRGB-encode the linear value first so patches sit in display gamut.
        const auto encode = [](f64 v) {
            return v <= 0.0031308 ? 12.92 * v :
                   1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
        };
        const auto patch = sector(encode(value));
        reference.push_back(SrgbToLinear(patch[0], patch[1], patch[2]));
    }
    // A plausible device transform (all-positive mixing).
    const std::array<f64, 9> forward = {0.90, 0.05, 0.02,
                                        0.03, 0.85, 0.05,
                                        0.01, 0.04, 0.80};
    const auto multiply = [](const std::array<f64, 9>& m,
                             const std::array<f64, 3>& v) {
        std::array<f64, 3> out{};
        for (u32 row = 0; row < 3; ++row)
            for (u32 col = 0; col < 3; ++col)
                out[row] += m[row * 3 + col] * v[col];
        return out;
    };
    const auto invert3 = [](const std::array<f64, 9>& m) {
        const f64 a = m[0], b = m[1], c = m[2];
        const f64 d = m[3], e = m[4], f = m[5];
        const f64 g = m[6], h = m[7], i = m[8];
        const f64 det = a * (e * i - f * h) - b * (d * i - f * g) +
                        c * (d * h - e * g);
        return std::array<f64, 9>{
            (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
            (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
            (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
    };
    CameraConfig config = MonoPhotonConfig(6, 4);
    const auto inverse = invert3(forward);
    config.isp.deviceToLinearSrgb = inverse;
    const auto raw = MakeRaw(config, [&](u32 x, u32 y, u32) {
        const auto device = multiply(forward, reference[y * 6 + x]);
        return std::lround(device[0] * FullScale(config));
    });
    // Feed the per-channel device values through a three-channel device so the
    // CCM is exercised on each channel.
    Image rawRgb(6, 4, 3);
    for (u32 y = 0; y < 4; ++y)
        for (u32 x = 0; x < 6; ++x) {
            const auto device = multiply(forward, reference[y * 6 + x]);
            for (u32 c = 0; c < 3; ++c)
                rawRgb(x, y, c) =
                    static_cast<f32>(std::lround(device[c] * FullScale(config)));
        }
    config.device.cfa = CfaPattern::MultiChannel;
    config.device.channels = {config.device.channels.front(),
                              config.device.channels.front(),
                              config.device.channels.front()};
    const auto display = RunOnRaw(config, rawRgb);
    ASSERT_TRUE(display.has_value());
    f64 worst = 0.0;
    for (u32 i = 0; i < 24; ++i) {
        const u32 x = i % 6, y = i / 6;
        const auto decoded = SrgbToLinear(display.value().operator()(x, y, 0),
                                          display.value().operator()(x, y, 1),
                                          display.value().operator()(x, y, 2));
        worst = std::max(worst, DeltaE76(decoded, reference[i]));
    }
    EXPECT_LT(worst, 2.0) << "self-calibrated CCM should invert to near zero dE";
}

TEST(CameraIspTest, MisplacedCcmKeepsDeltaELargeButFinite) {
    std::vector<std::array<f64, 3>> reference;
    for (u32 i = 0; i < 24; ++i)
        reference.push_back({0.1 + 0.8 * ((i % 4) / 3.0),
                             0.1 + 0.8 * (((i / 4) % 4) / 3.0),
                             0.1 + 0.8 * ((i / 16) / 2.0)});
    CameraConfig config = MonoPhotonConfig(6, 4);
    config.readout.adcBits = 14;
    // Deliberately wrong matrix: swaps red and green rows.
    config.isp.deviceToLinearSrgb = {0.0, 1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0};
    const auto raw = MakeRaw(config, [&](u32 x, u32 y, u32) {
        return std::lround(reference[y * 6 + x][0] * FullScale(config));
    });
    (void)raw;
    Image rawRgb(6, 4, 3);
    for (u32 y = 0; y < 4; ++y)
        for (u32 x = 0; x < 6; ++x)
            for (u32 c = 0; c < 3; ++c)
                rawRgb(x, y, c) = static_cast<f32>(
                    std::lround(reference[y * 6 + x][c] * FullScale(config)));
    config.device.cfa = CfaPattern::MultiChannel;
    config.device.channels = {config.device.channels.front(),
                              config.device.channels.front(),
                              config.device.channels.front()};
    const auto display = RunOnRaw(config, rawRgb);
    ASSERT_TRUE(display.has_value());
    f64 worst = 0.0;
    for (u32 i = 0; i < 24; ++i) {
        const u32 x = i % 6, y = i / 6;
        const auto decoded = SrgbToLinear(display.value().operator()(x, y, 0),
                                          display.value().operator()(x, y, 1),
                                          display.value().operator()(x, y, 2));
        worst = std::max(worst, DeltaE76(decoded, reference[i]));
    }
    EXPECT_GT(worst, 10.0) << "a swapped CCM must move the colors";
    EXPECT_LT(worst, 150.0) << "but stay within a bounded region of Lab";
}

// ---------------------------------------------------------------------------
// Metamerism: different spectra, same device response, same display
// ---------------------------------------------------------------------------

TEST(CameraIspTest, MetamericSpectraGiveIdenticalDisplay) {
    CameraConfig config = MonoPhotonConfig(4, 4);
    config.quality.wavelengthSamples = 32;
    config.quality.pixelSamples = 1;
    config.products.bandMeasurement = true;
    // The internal wavelength grid: response knots plus a uniform sweep.
    std::vector<f64> grid;
    for (u32 i = 0; i <= config.quality.wavelengthSamples; ++i)
        grid.push_back(500.0 + 100.0 * i / config.quality.wavelengthSamples);
    const auto area = PixelCollectionAreaM2(config.optics);
    ASSERT_TRUE(area.has_value());
    // Per-wavelength photon weight of this device, built exactly the way the
    // pipeline builds its measurement matrix: the full grid span with one
    // basis sample set to unit irradiance.
    std::vector<f64> weights(grid.size());
    for (size_t k = 0; k < grid.size(); ++k) {
        std::vector<SpectralIrradianceSample> basis(grid.size());
        for (size_t i = 0; i < grid.size(); ++i)
            basis[i] = {grid[i], i == k ? 1.0 : 0.0};
        const auto measured = IntegratePhoton(basis,
            config.device.channels.front().response, area.value());
        ASSERT_TRUE(measured.has_value());
        weights[k] = measured.value().electronRatePerSecond;
    }
    const f64 meanWeight = std::accumulate(weights.begin(), weights.end(), 0.0) /
                           static_cast<f64>(weights.size());
    // Sampler B inverts the photon weight: a genuinely different spectrum that
    // integrates to the same electron rate at every pixel.
    std::map<f64, f64> metamer;
    for (size_t k = 0; k < grid.size(); ++k) metamer[grid[k]] = meanWeight / weights[k];
    const SpectralFrameSampler flat = [](double, double) -> Result<Image, String> {
        Image sample(4, 4, 1);
        std::fill(sample.data.begin(), sample.data.end(), 1.0f);
        return sample;
    };
    const SpectralFrameSampler inverted = [&metamer](double, double wavelengthNm)
        -> Result<Image, String> {
        const auto it = metamer.find(wavelengthNm);
        if (it == metamer.end())
            return Result<Image, String>::Err("unexpected wavelength");
        Image sample(4, 4, 1);
        std::fill(sample.data.begin(), sample.data.end(),
                  static_cast<f32>(it->second));
        return sample;
    };
    // The two spectra really are different.
    ASSERT_NE(flat(0.0, grid.front()).value().data[0],
              inverted(0.0, grid.front()).value().data[0]);
    CaptureState stateA, stateB;
    const auto captureA = CpuCameraPipeline(config).Capture(stateA, 0.0, flat);
    const auto captureB = CpuCameraPipeline(config).Capture(stateB, 0.0, inverted);
    ASSERT_TRUE(captureA.has_value());
    ASSERT_TRUE(captureB.has_value());
    ASSERT_TRUE(captureA.value().bandMeasurement.has_value());
    ASSERT_TRUE(captureB.value().bandMeasurement.has_value());
    ASSERT_TRUE(captureA.value().display.has_value());
    ASSERT_TRUE(captureB.value().display.has_value());
    const auto& a = captureA.value().bandMeasurement->image.data;
    const auto& b = captureB.value().bandMeasurement->image.data;
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], 1e-3 * std::max(1.0f, std::fabs(a[i])));
    // The display chain is downstream of the device response: metameric pairs
    // are indistinguishable there. RAW DN may differ by at most one count
    // where a value sat exactly on a quantization boundary, which is at most
    // a 1-DN step in the encoded display.
    ASSERT_TRUE(captureA.value().rawDn.has_value());
    ASSERT_TRUE(captureB.value().rawDn.has_value());
    for (size_t i = 0; i < captureA.value().rawDn->image.data.size(); ++i)
        EXPECT_NEAR(captureA.value().rawDn->image.data[i],
                    captureB.value().rawDn->image.data[i], 1.0f);
    for (size_t i = 0; i < captureA.value().display.value().image.data.size(); ++i)
        EXPECT_NEAR(captureA.value().display.value().image.data[i],
                    captureB.value().display.value().image.data[i], 2e-3f);
}

// ---------------------------------------------------------------------------
// Infrared display branch
// ---------------------------------------------------------------------------

TEST(CameraIspTest, InfraredPaletteDoesNotTouchApparentTemperature) {
    const auto captureWith = [](DisplayPalette palette) {
        CameraConfig config = ThermalMonoConfig(8, 8);
        config.products.apparentTemperature = true;
        config.isp.infraredPalette = palette;
        config.isp.infraredTone = DisplayToneMode::Equalize;
        CaptureState state;
        const auto measured = [](double, double) -> Result<Image, String> {
            Image sample(8, 8, 1);
            for (u32 y = 0; y < 8; ++y)
                for (u32 x = 0; x < 8; ++x)
                    sample(x, y, 0) = static_cast<f32>(100.0 + x + 16 * y);
            return sample;
        };
        return CpuCameraPipeline(config).Capture(state, 0.0, measured);
    };
    const auto grey = captureWith(DisplayPalette::Grey);
    const auto ironbow = captureWith(DisplayPalette::Ironbow);
    ASSERT_TRUE(grey.has_value());
    ASSERT_TRUE(ironbow.has_value());
    ASSERT_TRUE(grey.value().apparentTemperature.has_value());
    ASSERT_TRUE(ironbow.value().apparentTemperature.has_value());
    // The temperature product is computed from the corrected signal, upstream
    // of the display branch: it must not move when the palette changes.
    EXPECT_EQ(grey.value().apparentTemperature->image.data,
              ironbow.value().apparentTemperature->image.data);
    // The displays, of course, differ.
    EXPECT_NE(grey.value().display.value().image.data, ironbow.value().display.value().image.data);
}

TEST(CameraIspTest, ClaheToneDegradesToEqualizeWithMetadata) {
    const auto runWith = [](DisplayToneMode tone) {
        CameraConfig config = ThermalMonoConfig(4, 4);
        config.isp.infraredTone = tone;
        const auto raw = MakeRaw(config, [](u32 x, u32 y, u32) {
            return 100.0 + x + 16.0 * y;
        });
        Image corrected = raw;
        CaptureState state;
        return RunIsp(config, raw, corrected, state, 0.0);
    };
    const auto clahe = runWith(DisplayToneMode::Clahe);
    const auto equalize = runWith(DisplayToneMode::Equalize);
    ASSERT_TRUE(clahe.has_value());
    ASSERT_TRUE(equalize.has_value());
    EXPECT_EQ(clahe.value().metadata.at("camera_ir_clahe_fallback"), "equalize");
    EXPECT_EQ(equalize.value().metadata.count("camera_ir_clahe_fallback"), 0u);
    for (size_t i = 0; i < clahe.value().data.size(); ++i)
        EXPECT_FLOAT_EQ(clahe.value().data[i], equalize.value().data[i]);
}

// ---------------------------------------------------------------------------
// Default chain is identity up to the sRGB encoding
// ---------------------------------------------------------------------------

TEST(CameraIspTest, DefaultChainIsIdentityUpToSrgbEncoding) {
    CameraConfig config = MonoPhotonConfig(6, 6);
    const auto raw = MakeRaw(config, [](u32 x, u32 y, u32) {
        return static_cast<f64>((x * 11 + y * 5) % 200);
    });
    const auto display = RunOnRaw(config, raw);
    ASSERT_TRUE(display.has_value());
    for (u32 y = 0; y < 6; ++y)
        for (u32 x = 0; x < 6; ++x) {
            const f64 linear = raw(x, y, 0) / FullScale(config);
            for (u32 c = 0; c < 3; ++c)
                EXPECT_NEAR(display.value().operator()(x, y, c), ExpectedSrgb(linear),
                            1e-5)
                    << "pixel (" << x << "," << y << ")";
        }
}

// ---------------------------------------------------------------------------
// Palette consistency with the HLSL ramps
// ---------------------------------------------------------------------------

TEST(CameraIspTest, CpuPalettesMatchTheHlslControlPoints) {
    const auto expectTable = [](StringView functionName,
                                const auto& table) {
        const auto parsed = ParseHlslPalette(functionName);
        ASSERT_EQ(parsed.size(), table.size()) << functionName;
        for (size_t i = 0; i < parsed.size(); ++i)
            for (u32 c = 0; c < 3; ++c)
                EXPECT_DOUBLE_EQ(parsed[i][c], table[i][c])
                    << functionName << " point " << i << " channel " << c;
    };
    expectTable("IronbowPalette", isp::kIronbowPoints);
    expectTable("RainbowPalette", isp::kRainbowPoints);
    expectTable("ViridisPalette", isp::kViridisPoints);
}

TEST(CameraIspTest, CpuPaletteSamplesMatchTheHlslFormula) {
    // Recomputes each ramp with the exact HLSL float semantics (saturate,
    // uint truncation of t*(N-1), lerp) from the parsed control points.
    const auto sampleHlsl = [](const std::vector<std::array<f64, 3>>& points,
                               f64 t) {
        const f64 x = std::clamp(t, 0.0, 1.0) *
                      static_cast<f64>(points.size() - 1);
        const size_t i = std::min(static_cast<size_t>(x), points.size() - 2);
        const f64 fraction = x - static_cast<f64>(i);
        std::array<f64, 3> out{};
        for (u32 c = 0; c < 3; ++c)
            out[c] = points[i][c] + (points[i + 1][c] - points[i][c]) * fraction;
        return out;
    };
    const auto compare = [&](StringView functionName, DisplayPalette palette) {
        const auto parsed = ParseHlslPalette(functionName);
        for (u32 step = 0; step <= 20; ++step) {
            const f64 t = static_cast<f64>(step) / 20.0;
            const auto hlsl = sampleHlsl(parsed, t);
            const auto cpu = ApplyDisplayPalette(t, palette);
            for (u32 c = 0; c < 3; ++c)
                EXPECT_NEAR(cpu[c], hlsl[c], 1e-12)
                    << functionName << " at t=" << t;
        }
    };
    compare("IronbowPalette", DisplayPalette::Ironbow);
    compare("RainbowPalette", DisplayPalette::Rainbow);
    compare("ViridisPalette", DisplayPalette::Viridis);
    // Grey ramps are exact identities.
    for (u32 step = 0; step <= 20; ++step) {
        const f64 t = static_cast<f64>(step) / 20.0;
        const auto grey = ApplyDisplayPalette(t, DisplayPalette::Grey);
        const auto inverted = ApplyDisplayPalette(t, DisplayPalette::GreyInverted);
        EXPECT_DOUBLE_EQ(grey[0], t);
        EXPECT_DOUBLE_EQ(inverted[0], 1.0 - t);
    }
}

// ---------------------------------------------------------------------------
// Product integration
// ---------------------------------------------------------------------------

TEST(CameraIspTest, DisplayProductCarriesAcquisitionMetadata) {
    CameraConfig config = MonoPhotonConfig(2, 2);
    const auto raw = MakeRaw(config, [](u32 x, u32 y, u32) { return 50.0 + x + y; });
    CaptureState state;
    state.acquisitionIndex = 41;
    const auto display = RunIsp(config, raw, raw, state, 0.0);
    ASSERT_TRUE(display.has_value());
    EXPECT_EQ(display.value().metadata.at("camera_acquisition_index"), "41");
}
