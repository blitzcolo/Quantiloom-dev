/**
 * @file test_rgb_to_spectrum.cpp
 * @brief Jakob & Hanika spectral upsampling
 *
 * The claim being pinned is not "the code runs" but "the fit is exact": for a
 * colour inside the sRGB gamut the sigmoid spectrum reproduces it to zero
 * CIELab error under D65. That is what makes this an upgrade rather than a
 * different approximation, and it is checked here against the same observer
 * table the renderer integrates against rather than against a textbook.
 *
 * Everything runs at a small resolution. The production table is 64 per axis
 * and takes seconds to fit; these want to stay inside a 4 s suite, and none of
 * the properties under test depend on the table being fine.
 */

#include <gtest/gtest.h>

#include "core/RgbToSpectrum.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "core/D65Illuminant.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace quantiloom;

namespace {

constexpr u32 kTestRes = 24;

// Integrate a spectrum against the observer under D65, exactly as the fitter
// does, and convert to linear sRGB. Independent arithmetic from the library's,
// so a sign error in one does not cancel in the other.
glm::dvec3 ReflectanceToRgb(const RgbSpectrumCoeffs& c) {
    double xyz[3] = {0.0, 0.0, 0.0};
    double yNorm = 0.0;
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f32 lambda = CIE_CMF_LAMBDA_MIN + static_cast<f32>(i);
        const double w = (i == 0 || i == CIE_CMF_LUT_SIZE - 1) ? 0.5 : 1.0;
        const double d65 = static_cast<double>(D65Relative(lambda));
        const double r = static_cast<double>(EvaluateRgbSpectrum(c, lambda));
        for (int k = 0; k < 3; ++k) {
            xyz[k] += w * d65 * CIE_1931_2DEG[i][k] * r;
        }
        yNorm += w * d65 * CIE_1931_2DEG[i][1];
    }
    for (double& v : xyz) {
        v /= yNorm;
    }
    return {3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2],
            -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2],
            0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0570 * xyz[2]};
}

double LabF(double x) {
    constexpr double d = 6.0 / 29.0;
    return (x > d * d * d) ? std::cbrt(x) : x / (3.0 * d * d) + 4.0 / 29.0;
}

glm::dvec3 RgbToLab(const glm::dvec3& rgb) {
    // linear sRGB -> XYZ, D65 white point
    const double X = 0.4124564 * rgb.x + 0.3575761 * rgb.y + 0.1804375 * rgb.z;
    const double Y = 0.2126729 * rgb.x + 0.7151522 * rgb.y + 0.0721750 * rgb.z;
    const double Z = 0.0193339 * rgb.x + 0.1191920 * rgb.y + 0.9503041 * rgb.z;
    const double fx = LabF(X / 0.9504559);
    const double fy = LabF(Y / 1.0);
    const double fz = LabF(Z / 1.0890578);
    return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

double DeltaE(const glm::dvec3& a, const glm::dvec3& b) {
    const glm::dvec3 la = RgbToLab(a);
    const glm::dvec3 lb = RgbToLab(b);
    const glm::dvec3 d = la - lb;
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

const RgbToSpectrumTable& SharedTable() {
    static const RgbToSpectrumTable table = []() {
        auto built = RgbToSpectrumTable::Build(kTestRes);
        EXPECT_TRUE(built.has_value()) << (built.has_value() ? "" : built.error());
        return std::move(built).value();
    }();
    return table;
}

}  // namespace

// ============================================================================
// The sigmoid itself
// ============================================================================

// s(x) = 1/2 + x / (2 sqrt(1 + x^2)) is bounded by construction, which is the
// property the Gaussian upsampler it replaces did not have: that one needed a
// clamp to 1.5 and could return a reflectance above 1 for an HDR base colour.
TEST(RgbToSpectrum, ReflectanceIsBoundedForAnyCoefficients) {
    const std::vector<RgbSpectrumCoeffs> wild = {
        {0.0f, 0.0f, 0.0f},      {1e6f, -1e6f, 1e6f},   {-1e6f, 1e6f, -1e6f},
        {100.0f, -50.0f, 3.0f},  {-0.001f, 0.0f, -30.0f}, {0.0f, 1e-8f, 1e-8f},
    };
    for (const RgbSpectrumCoeffs& c : wild) {
        for (f32 lambda = 380.0f; lambda <= 780.0f; lambda += 1.0f) {
            const f32 r = EvaluateRgbSpectrum(c, lambda);
            EXPECT_GE(r, 0.0f);
            EXPECT_LE(r, 1.0f);
            EXPECT_TRUE(std::isfinite(r));
        }
    }
}

// Outside the fitted band the polynomial keeps growing and the sigmoid saturates
// -- toward 1 for a saturated warm colour, which in a thermal render is a
// mirror where a wall should be. The clamp is what stops that from leaving the
// function, and this pins it: past 780 nm the answer is frozen, not extrapolated.
TEST(RgbToSpectrum, OutsideTheFittedBandTheAnswerIsFrozen) {
    const RgbSpectrumCoeffs mango = SharedTable().Lookup({1.0f, 0.329f, 0.1f});
    const f32 at780 = EvaluateRgbSpectrum(mango, 780.0f);
    for (const f32 lambda : {781.0f, 900.0f, 1200.0f, 2400.0f, 10000.0f}) {
        EXPECT_FLOAT_EQ(EvaluateRgbSpectrum(mango, lambda), at780) << "lambda " << lambda;
    }
    const f32 at380 = EvaluateRgbSpectrum(mango, 380.0f);
    for (const f32 lambda : {379.0f, 200.0f, 0.0f}) {
        EXPECT_FLOAT_EQ(EvaluateRgbSpectrum(mango, lambda), at380) << "lambda " << lambda;
    }
}

// ============================================================================
// Achromatic: the closed form, and why it has to be exact
// ============================================================================

// s(c2) = g exactly when c2 = (g - 1/2) / sqrt(g (1 - g)). Every dielectric
// without KHR_materials_specular has F0 = 0.04 grey and both render gates use
// grey scenes, so this path carries them; if it drifted, "bit-identical" would
// stop being provable.
TEST(RgbToSpectrum, AchromaticClosedFormIsFlatAndExact) {
    for (const f32 g : {0.0001f, 0.04f, 0.18f, 0.5f, 0.65f, 0.85f, 0.9999f}) {
        const RgbSpectrumCoeffs c = AchromaticRgbSpectrum(g);
        EXPECT_FLOAT_EQ(c.c0, 0.0f);
        EXPECT_FLOAT_EQ(c.c1, 0.0f);
        for (f32 lambda = 380.0f; lambda <= 780.0f; lambda += 20.0f) {
            EXPECT_NEAR(EvaluateRgbSpectrum(c, lambda), g, 1e-6f) << "g " << g;
        }
    }
}

// A grey triple must not reach the table at all -- the same reasoning, one
// level up. Checked through Lookup so the early-out cannot be removed without
// this failing.
TEST(RgbToSpectrum, GreyLookupTakesTheClosedFormNotTheTable) {
    for (const f32 g : {0.04f, 0.25f, 0.5f, 0.8f}) {
        const RgbSpectrumCoeffs c = SharedTable().Lookup({g, g, g});
        EXPECT_FLOAT_EQ(c.c0, 0.0f);
        EXPECT_FLOAT_EQ(c.c1, 0.0f);
        EXPECT_FLOAT_EQ(c.c2, AchromaticRgbSpectrum(g).c2);
        EXPECT_NEAR(EvaluateRgbSpectrum(c, 550.0f), g, 1e-6f);
    }
}

// The two extremes are limits, not values: c2 runs to +-infinity. What matters
// is that they stay finite and land on the right side of the interval.
TEST(RgbToSpectrum, BlackAndWhiteStayFinite) {
    const RgbSpectrumCoeffs black = AchromaticRgbSpectrum(0.0f);
    const RgbSpectrumCoeffs white = AchromaticRgbSpectrum(1.0f);
    EXPECT_TRUE(std::isfinite(black.c2));
    EXPECT_TRUE(std::isfinite(white.c2));
    EXPECT_LT(EvaluateRgbSpectrum(black, 550.0f), 1e-5f);
    EXPECT_GT(EvaluateRgbSpectrum(white, 550.0f), 1.0f - 1e-5f);
}

// ============================================================================
// The fit is exact inside the gamut
// ============================================================================

// This is the headline property and the reason for the change. A table node is
// a colour the solver was actually run on, so there is no interpolation in the
// way: the residual should be zero to the solver's own tolerance.
TEST(RgbToSpectrum, FitIsExactAtTableNodes) {
    const RgbToSpectrumTable& table = SharedTable();
    const u32 res = table.Resolution();
    double worst = 0.0;
    glm::vec3 worstRgb{};
    for (u32 l = 0; l < 3; ++l) {
        for (u32 k = 4; k < res; k += 3) {
            for (u32 j = 0; j < res; j += 5) {
                for (u32 i = 0; i < res; i += 5) {
                    const f32 b = RgbToSpectrumTable::ZNode(k, res);
                    const f32 x = static_cast<f32>(i) / static_cast<f32>(res - 1);
                    const f32 y = static_cast<f32>(j) / static_cast<f32>(res - 1);
                    glm::vec3 rgb{};
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;
                    if (b < 1e-3f) {
                        continue;  // black, where Lab has no chroma to speak of
                    }
                    const double dE = DeltaE(ReflectanceToRgb(table.Lookup(rgb)), rgb);
                    if (dE > worst) {
                        worst = dE;
                        worstRgb = rgb;
                    }
                }
            }
        }
    }
    // A just-noticeable difference is about 2.3. At a node the fit itself is the
    // only error and it is orders below that.
    EXPECT_LT(worst, 0.05) << "worst at rgb (" << worstRgb.r << ", " << worstRgb.g << ", "
                           << worstRgb.b << ")";
}

// Between nodes the trilinear blend adds its own error, and that is what the
// table resolution buys down. At 24 per axis it is already well inside a JND;
// production runs 64.
TEST(RgbToSpectrum, InterpolatedColoursStayInsideAJustNoticeableDifference) {
    const RgbToSpectrumTable& table = SharedTable();
    const std::vector<glm::vec3> colours = {
        {1.0f, 0.329f, 0.1f},    // SheenChair mango velvet
        {0.05f, 0.17f, 0.5f},    // GlamVelvetSofa navy sheen
        {0.04f, 0.01f, 0.08f},   // a purple dielectric F0
        {0.65f, 0.15f, 0.1f},    // cornell box red brick
        {0.25f, 0.4f, 0.15f},    // cornell box olive green
        {0.85f, 0.85f, 0.82f},   // near-neutral marble, just off the grey path
        {0.7f, 0.3f, 0.9f},  {0.13f, 0.77f, 0.42f}, {0.9f, 0.85f, 0.2f},
    };
    for (const glm::vec3& rgb : colours) {
        const double dE = DeltaE(ReflectanceToRgb(table.Lookup(rgb)), rgb);
        EXPECT_LT(dE, 2.3) << "rgb (" << rgb.r << ", " << rgb.g << ", " << rgb.b << ")";
    }
}

// The upsampled spectrum has to be a reflectance, not just the right colour:
// a fit that reproduced the tristimulus by going above 1 somewhere would break
// energy conservation in the furnace.
TEST(RgbToSpectrum, FittedSpectraAreValidReflectances) {
    const RgbToSpectrumTable& table = SharedTable();
    for (const glm::vec3& rgb : std::vector<glm::vec3>{
             {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
             {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f},
             {0.99f, 0.02f, 0.5f}}) {
        const RgbSpectrumCoeffs c = table.Lookup(rgb);
        for (f32 lambda = 380.0f; lambda <= 780.0f; lambda += 5.0f) {
            const f32 r = EvaluateRgbSpectrum(c, lambda);
            EXPECT_GE(r, 0.0f);
            EXPECT_LE(r, 1.0f);
        }
    }
}

// ============================================================================
// The z-axis warp and its inverse are a matched pair
// ============================================================================

// The shader inverts smoothstep-squared in closed form instead of searching the
// node array. If that inverse and ZNode ever disagreed the table would be read
// off by a texel with no other symptom, so the round trip is pinned here.
TEST(RgbToSpectrum, ZWarpRoundTripsThroughItsClosedFormInverse) {
    constexpr u32 res = 64;
    auto invSmoothstep = [](double v) {
        return 0.5 - std::sin(std::asin(std::clamp(1.0 - 2.0 * v, -1.0, 1.0)) / 3.0);
    };
    for (u32 i = 0; i < res; ++i) {
        const double z = RgbToSpectrumTable::ZNode(i, res);
        const double u = invSmoothstep(invSmoothstep(z)) * (res - 1);
        EXPECT_NEAR(u, static_cast<double>(i), 2e-3) << "node " << i;
    }
    // Monotone, which is what makes the inverse single-valued.
    for (u32 i = 1; i < res; ++i) {
        EXPECT_GT(RgbToSpectrumTable::ZNode(i, res), RgbToSpectrumTable::ZNode(i - 1, res));
    }
    EXPECT_FLOAT_EQ(RgbToSpectrumTable::ZNode(0, res), 0.0f);
    EXPECT_FLOAT_EQ(RgbToSpectrumTable::ZNode(res - 1, res), 1.0f);
}

// ============================================================================
// HDR, which the illuminant path relies on
// ============================================================================

// A light source is not bounded by 1, so it is fitted at rgb / (2 max) and
// multiplied back. The factor 2 keeps the fitted value at or below 0.5, well
// inside the sigmoid's well-conditioned interior. The property that matters is
// that the error does not depend on the scale.
TEST(RgbToSpectrum, HdrColoursScaleWithoutLosingAccuracy) {
    const RgbToSpectrumTable& table = SharedTable();
    const glm::vec3 hue{1.0f, 0.9f, 0.7f};
    double reference = -1.0;
    for (const f32 s : {1.0f, 4.0f, 50.0f, 5000.0f}) {
        const glm::vec3 rgb = hue * s;
        const f32 scale = 2.0f * std::max({rgb.r, rgb.g, rgb.b});
        const RgbSpectrumCoeffs c = table.Lookup(rgb / scale);
        const glm::dvec3 got = ReflectanceToRgb(c) * static_cast<double>(scale);
        const double rel = glm::length(got - glm::dvec3(rgb)) / glm::length(glm::dvec3(rgb));
        if (reference < 0.0) {
            reference = rel;
        }
        EXPECT_LT(rel, 0.01) << "scale " << s;
        EXPECT_NEAR(rel, reference, 1e-6) << "scale " << s;
    }
}

// ============================================================================
// Build guards
// ============================================================================

TEST(RgbToSpectrum, BuildRejectsUnusableResolutions) {
    EXPECT_FALSE(RgbToSpectrumTable::Build(0).has_value());
    EXPECT_FALSE(RgbToSpectrumTable::Build(1).has_value());
    EXPECT_FALSE(RgbToSpectrumTable::Build(1000).has_value());
}

TEST(RgbToSpectrum, TableHasTheExpectedShape) {
    const RgbToSpectrumTable& table = SharedTable();
    EXPECT_EQ(table.Resolution(), kTestRes);
    EXPECT_EQ(table.Data().size(),
              static_cast<size_t>(3) * kTestRes * kTestRes * kTestRes * 3);
    for (const f32 v : table.Data()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}
