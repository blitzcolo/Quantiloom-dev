/**
 * @file colour_lab.cpp
 * @brief The RGB-to-spectrum conventions, measured rather than asserted
 *
 * Two questions, both about what an RGB triple is taken to mean, and both
 * answered by numbers that appear in the manual and in the paper:
 *
 *   --lut-sweep    how much the coefficient table's resolution costs in
 *                  colour accuracy. Fits a table at each requested resolution
 *                  and measures the interpolation error against the colours it
 *                  claims to reproduce.
 *
 *   --illuminant   what an authored emissive triple comes back as after a
 *                  round trip through the illuminant convention. Host-side:
 *                  fit, spectrum, D65, the 1 nm observer, XYZ, linear sRGB.
 *
 * Why a tool and not a unit test. Fitting 3*64^3 lattice points is seconds of
 * work per resolution, and three resolutions is most of a minute -- against a
 * suite that runs in nine. The unit tests pin the table's *behaviour*; this
 * measures its *accuracy*, which is a different question asked much less often.
 *
 * Output is JSON on stdout, so the experiment script that consumes it does not
 * have to parse a human table. Progress goes to stderr.
 */

#include "core/CIE_CMF_Data.hpp"
#include "core/D65Illuminant.hpp"
#include "core/Log.hpp"
#include "core/RgbToSpectrum.hpp"
#include "core/SpectralData.hpp"
#include "io/SpectralIO.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace quantiloom;

namespace {

// The sample the accuracy numbers are drawn from. Fixed here rather than passed
// in so that a rerun reproduces the published figure without an argument nobody
// remembers; --samples and --seed override it for a sensitivity check.
//
// 200k rather than 20k because of the p99 specifically. Mean and worst are
// settled at 20k -- mean to three digits, worst because the eight cube corners
// are always in the sample and are what sets it -- but the 99th percentile of
// 20k is the 200th-largest of a heavy tail, and it moves by 8 % between seeds.
// At 200k it is stable to the digit published. The whole sweep costs two
// seconds, so there is nothing to trade against.
constexpr u32 kDefaultSamples = 200000;
constexpr u64 kDefaultSeed = 0x5CULL << 56 | 0x01A11B1EULL;  // "colour lab", once

void PrintUsage() {
    std::fprintf(stderr,
                 "colour_lab --lut-sweep [--resolutions 32,48,64] [--samples N] [--seed S]\n"
                 "colour_lab --illuminant [--rgb R,G,B]\n");
}

std::vector<u32> ParseResolutions(const char* text) {
    std::vector<u32> out;
    u32 value = 0;
    bool any = false;
    for (const char* p = text;; ++p) {
        if (*p >= '0' && *p <= '9') {
            value = value * 10 + static_cast<u32>(*p - '0');
            any = true;
        } else {
            if (any) {
                out.push_back(value);
            }
            value = 0;
            any = false;
            if (*p == '\0') {
                break;
            }
        }
    }
    return out;
}

bool ParseTriple(const char* text, f64 out[3]) {
    return std::sscanf(text, "%lf,%lf,%lf", &out[0], &out[1], &out[2]) == 3;
}

// ============================================================================
// (a) resolution sweep
// ============================================================================

int LutSweep(const std::vector<u32>& resolutions, u32 samples, u64 seed) {
    std::printf("{\n  \"measurement\": \"rgb2spec table interpolation error\",\n");
    std::printf("  \"metric\": \"CIE76 dE in CIELab, against the colour the fit targets\",\n");
    std::printf("  \"sampling\": \"8 sRGB cube corners + %u uniform, mt19937_64\",\n", samples);
    std::printf("  \"seed\": \"0x%llX\",\n", static_cast<unsigned long long>(seed));
    std::printf("  \"jnd\": 2.3,\n  \"rows\": [\n");

    bool first = true;
    for (const u32 res : resolutions) {
        std::fprintf(stderr, "fitting res %u ...\n", res);
        auto built = RgbToSpectrumTable::Build(res);
        if (!built.has_value()) {
            std::fprintf(stderr, "  failed: %s\n", built.error().c_str());
            return 1;
        }
        const RgbToSpectrumAccuracy a = MeasureAccuracy(built.value(), samples, seed);
        std::printf("%s    {\"resolution\": %u, \"coefficient_bytes\": %llu, "
                    "\"coefficient_mb\": %.2f, \"samples\": %u, "
                    "\"mean_dE\": %.4f, \"p99_dE\": %.4f, \"worst_dE\": %.4f, "
                    "\"worst_colour\": [%.4f, %.4f, %.4f]}",
                    first ? "" : ",\n", a.resolution,
                    static_cast<unsigned long long>(a.coefficientBytes),
                    static_cast<f64>(a.coefficientBytes) / (1024.0 * 1024.0), a.samples,
                    a.meanDeltaE, a.p99DeltaE, a.worstDeltaE,
                    static_cast<f64>(a.worstColour.r), static_cast<f64>(a.worstColour.g),
                    static_cast<f64>(a.worstColour.b));
        first = false;
    }
    std::printf("\n  ]\n}\n");
    return 0;
}

// ============================================================================
// (b) illuminant round trip
// ============================================================================

// The convention under test, from SpectralConversion.hlsli:
//
//     L(lambda) = scale * s(c; lambda) * d65rel(lambda),   scale = 2 * max(rgb)
//
// The 2 puts the fitted triple at or below 0.5, inside the sigmoid's
// well-conditioned interior, which is what makes the error independent of the
// authored magnitude.
SpectralCurve EmissiveCurve(const RgbToSpectrumTable& table, const f64 rgb[3], bool useD65) {
    const f64 maxComponent = std::max(rgb[0], std::max(rgb[1], rgb[2]));
    const f64 scale = 2.0 * maxComponent;

    SpectralCurve curve;
    curve.samples.reserve(CIE_CMF_LUT_SIZE);
    if (scale <= 0.0) {
        return curve;
    }
    const glm::vec3 chroma(static_cast<f32>(rgb[0] / scale), static_cast<f32>(rgb[1] / scale),
                           static_cast<f32>(rgb[2] / scale));
    const RgbSpectrumCoeffs coeffs = table.Lookup(chroma);

    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f32 lambda = CIE_CMF_LAMBDA_MIN + static_cast<f32>(i);
        // A flat illuminant is the equal-energy illuminant E, not D65. That is
        // the whole of the retired convention's error, and the reason it needed
        // a downstream correction at all.
        const f32 illuminant = useD65 ? D65Relative(lambda) : 1.0f;
        curve.samples.push_back(
            {lambda, static_cast<f32>(scale) * EvaluateRgbSpectrum(coeffs, lambda) * illuminant});
    }
    return curve;
}

int Illuminant(const f64 rgb[3]) {
    auto built = RgbToSpectrumTable::Build(kRgbToSpectrumResolution);
    if (!built.has_value()) {
        std::fprintf(stderr, "table build failed: %s\n", built.error().c_str());
        return 1;
    }
    const RgbToSpectrumTable& table = built.value();

    // The equal-energy illuminant's own colour, which is where the retired
    // convention's two correction constants came from: they are E's G/R and
    // G/B in linear sRGB.
    SpectralCurve flat;
    flat.samples.reserve(CIE_CMF_LUT_SIZE);
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        flat.samples.push_back({CIE_CMF_LAMBDA_MIN + static_cast<f32>(i), 1.0f});
    }
    const glm::vec3 eRgb = EmissionSpectrumToRenderedLinearSrgb(flat);
    const f64 wbR = static_cast<f64>(eRgb.g) / static_cast<f64>(eRgb.r);
    const f64 wbB = static_cast<f64>(eRgb.g) / static_cast<f64>(eRgb.b);

    const glm::vec3 chosen = EmissionSpectrumToRenderedLinearSrgb(EmissiveCurve(table, rgb, true));
    glm::vec3 flatConvention = EmissionSpectrumToRenderedLinearSrgb(EmissiveCurve(table, rgb, false));
    flatConvention.r *= static_cast<f32>(wbR);
    flatConvention.b *= static_cast<f32>(wbB);

    auto ratio = [](const glm::vec3& v, f64 out[3]) {
        const f64 r = static_cast<f64>(v.r);
        out[0] = 1.0;
        out[1] = (r != 0.0) ? static_cast<f64>(v.g) / r : 0.0;
        out[2] = (r != 0.0) ? static_cast<f64>(v.b) / r : 0.0;
    };
    // Deviation is the largest per-channel departure from the authored ratio,
    // which is what "3.57 % off" meant: a channel that is wrong by that much,
    // not an average that hides it.
    auto deviation = [](const f64 got[3], const f64 want[3]) {
        f64 worst = 0.0;
        for (int i = 1; i < 3; ++i) {
            worst = std::max(worst, std::fabs(got[i] - want[i]));
        }
        return 100.0 * worst;
    };

    f64 authored[3];
    ratio(glm::vec3(static_cast<f32>(rgb[0]), static_cast<f32>(rgb[1]), static_cast<f32>(rgb[2])),
          authored);
    f64 chosenRatio[3], flatRatio[3];
    ratio(chosen, chosenRatio);
    ratio(flatConvention, flatRatio);

    std::printf("{\n  \"measurement\": \"emissive RGB round trip, host-side\",\n");
    std::printf("  \"path\": \"fit -> sigmoid x illuminant -> 1 nm CIE observer -> XYZ -> linear sRGB\",\n");
    std::printf("  \"authored_rgb\": [%.4f, %.4f, %.4f],\n", rgb[0], rgb[1], rgb[2]);
    std::printf("  \"equal_energy_srgb\": [%.4f, %.4f, %.4f],\n",
                static_cast<f64>(eRgb.r) / static_cast<f64>(eRgb.g), 1.0,
                static_cast<f64>(eRgb.b) / static_cast<f64>(eRgb.g));
    std::printf("  \"retired_white_balance\": {\"r\": %.4f, \"b\": %.4f},\n", wbR, wbB);
    std::printf("  \"rows\": [\n");
    std::printf("    {\"convention\": \"authored values\", \"ratio\": [1.0, %.4f, %.4f], "
                "\"deviation_pct\": null},\n", authored[1], authored[2]);
    std::printf("    {\"convention\": \"flat spectrum + downstream white balance\", "
                "\"ratio\": [1.0, %.4f, %.4f], \"deviation_pct\": %.2f},\n",
                flatRatio[1], flatRatio[2], deviation(flatRatio, authored));
    std::printf("    {\"convention\": \"sigmoid x D65\", \"ratio\": [1.0, %.4f, %.4f], "
                "\"deviation_pct\": %.2f}\n", chosenRatio[1], chosenRatio[2],
                deviation(chosenRatio, authored));
    std::printf("  ],\n");

    // The argument that outlives the Cornell number, and the one the design
    // rests on. The two constants are correct for exactly one illuminant -- the
    // flat one they were derived from -- and the retired code applied them
    // unconditionally. So the quantity of interest is what they did to a scene
    // whose light was ALREADY right: a real D65 spectrum, or measured sunlight.
    // Equal-energy E is deliberately not listed; there the change is the whole
    // purpose, not an error, and reporting it beside these would conflate the
    // two.
    //
    // Luminance-normalised, because a correction that darkened everything by
    // one factor would be an exposure choice rather than a colour error.
    std::printf("  \"fixed_correction_applied_where_unneeded\": [\n");
    struct Case {
        const char* name;
        SpectralCurve curve;
    };
    std::vector<Case> cases;

    SpectralCurve d65;
    d65.samples.reserve(CIE_CMF_LUT_SIZE);
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f32 lambda = CIE_CMF_LAMBDA_MIN + static_cast<f32>(i);
        d65.samples.push_back({lambda, D65Relative(lambda)});
    }
    cases.push_back({"CIE D65", std::move(d65)});

    // Column 3 is the global tilt spectrum, the same column the solar LUT binds.
    auto solar = SpectralIO::LoadASTMG173("assets/luts/astmg173.csv", 3);
    if (solar.has_value()) {
        cases.push_back({"ASTM G-173 global tilt", std::move(solar.value())});
    } else {
        std::fprintf(stderr, "note: %s -- solar row omitted\n", solar.error().c_str());
    }

    bool firstCase = true;
    for (const Case& c : cases) {
        const glm::vec3 clean = EmissionSpectrumToRenderedLinearSrgb(c.curve);
        glm::vec3 corrected = clean;
        corrected.r *= static_cast<f32>(wbR);
        corrected.b *= static_cast<f32>(wbB);

        auto luminance = [](const glm::vec3& v) {  // Rec. 709, as the display path uses
            return 0.2126 * static_cast<f64>(v.r) + 0.7152 * static_cast<f64>(v.g) +
                   0.0722 * static_cast<f64>(v.b);
        };
        const f64 k = luminance(clean) / luminance(corrected);
        const f64 dR = 100.0 * (k * static_cast<f64>(corrected.r) / static_cast<f64>(clean.r) - 1.0);
        const f64 dG = 100.0 * (k * static_cast<f64>(corrected.g) / static_cast<f64>(clean.g) - 1.0);
        const f64 dB = 100.0 * (k * static_cast<f64>(corrected.b) / static_cast<f64>(clean.b) - 1.0);
        std::printf("%s    {\"illuminant\": \"%s\", \"channel_error_pct\": [%.1f, %.1f, %.1f]}",
                    firstCase ? "" : ",\n", c.name, dR, dG, dB);
        firstCase = false;
    }
    std::printf("\n  ]\n}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool doSweep = false;
    bool doIlluminant = false;
    std::vector<u32> resolutions{32, 48, 64};
    u32 samples = kDefaultSamples;
    u64 seed = kDefaultSeed;
    f64 rgb[3] = {15.0, 15.0, 12.0};  // the Cornell box's light panel

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasNext = i + 1 < argc;
        if (arg == "--lut-sweep") {
            doSweep = true;
        } else if (arg == "--illuminant") {
            doIlluminant = true;
        } else if (arg == "--resolutions" && hasNext) {
            resolutions = ParseResolutions(argv[++i]);
        } else if (arg == "--samples" && hasNext) {
            samples = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--seed" && hasNext) {
            seed = std::strtoull(argv[++i], nullptr, 0);
        } else if (arg == "--rgb" && hasNext) {
            if (!ParseTriple(argv[++i], rgb)) {
                std::fprintf(stderr, "--rgb wants R,G,B\n");
                return 2;
            }
        } else {
            PrintUsage();
            return 2;
        }
    }

    if (doSweep == doIlluminant) {
        PrintUsage();
        return 2;
    }
    if (doSweep && resolutions.empty()) {
        std::fprintf(stderr, "--resolutions parsed to nothing\n");
        return 2;
    }
    return doSweep ? LutSweep(resolutions, samples, seed) : Illuminant(rgb);
}
