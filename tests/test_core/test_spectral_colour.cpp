/**
 * @file test_spectral_colour.cpp
 * @brief Illuminant spectrum -> linear sRGB
 *
 * These pin numbers that are otherwise only visible by rendering something and
 * squinting at it. The sun's colour is now derived rather than typed into a
 * config, so it is worth being able to fail here instead of in an image.
 */

#include <gtest/gtest.h>

#include "core/SpectralData.hpp"

using namespace quantiloom;

namespace {

// Chromaticity, which is the part that does not depend on exposure.
void Chromaticity(const glm::vec3& rgb, double& x, double& y) {
    // linear sRGB -> XYZ, the inverse of the matrix the conversion applies
    const double X = 0.4124 * rgb.r + 0.3576 * rgb.g + 0.1805 * rgb.b;
    const double Y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    const double Z = 0.0193 * rgb.r + 0.1192 * rgb.g + 0.9505 * rgb.b;
    const double s = X + Y + Z;
    x = X / s;
    y = Y / s;
}

}  // namespace

// Illuminant E sits at the equal-energy point by definition. sRGB, however, is
// referenced to D65 at (0.3127, 0.3290), so a flat spectrum does NOT come out
// as (1, 1, 1) -- it lands slightly warm. That is colour science, not an error,
// and it is pinned here because "flat spectrum" and "white in sRGB" are easy to
// assume are the same thing. The illuminant that is exactly (1, 1, 1) is D65.
TEST(SpectralColour, EqualEnergySitsAtTheEqualEnergyPointNotAtSrgbWhite) {
    const auto rgb = SpectralIrradianceToLinearSrgb(MakeEqualEnergyIlluminant());

    EXPECT_NEAR(rgb.r, 1.2048f, 2e-3f);
    EXPECT_NEAR(rgb.g, 0.9484f, 2e-3f);
    EXPECT_NEAR(rgb.b, 0.9086f, 2e-3f);

    // Luminance is 1 by construction, which is what makes it an exposure-neutral
    // reference regardless of the chromaticity above.
    const double Y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    EXPECT_NEAR(Y, 1.0, 2e-3);

    double x = 0.0, y = 0.0;
    Chromaticity(rgb, x, y);
    EXPECT_NEAR(x, 1.0 / 3.0, 1e-3);
    EXPECT_NEAR(y, 1.0 / 3.0, 1e-3);
}

// Scaling the spectrum scales the colour and leaves the chromaticity alone.
// This is what lets exposure be a separate concern from white balance.
TEST(SpectralColour, MagnitudeIsLinearAndChromaticityIsNot) {
    auto curve = MakeEqualEnergyIlluminant();
    const auto unit = SpectralIrradianceToLinearSrgb(curve);

    for (auto& s : curve.samples) s.second *= 7.0f;
    const auto scaled = SpectralIrradianceToLinearSrgb(curve);

    EXPECT_NEAR(scaled.g / unit.g, 7.0f, 1e-3f);

    double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
    Chromaticity(unit, x0, y0);
    Chromaticity(scaled, x1, y1);
    EXPECT_NEAR(x0, x1, 1e-6);
    EXPECT_NEAR(y0, y1, 1e-6);
}

// SpectralCurve::Evaluate clamps to its endpoints rather than returning zero
// outside its range, so an infrared-only curve is NOT dark in the visible --
// it holds its edge value all the way down. Worth pinning, because "my curve
// starts at 3 um so it cannot affect colour" is exactly the wrong intuition,
// and it means a measured curve should span the band it will be asked about.
TEST(SpectralColour, EvaluateClampsSoAnInfraredCurveStillHasColour) {
    SpectralCurve infraredOnly;
    infraredOnly.samples.emplace_back(3000.0f, 500.0f);
    infraredOnly.samples.emplace_back(15000.0f, 500.0f);

    const auto rgb = SpectralIrradianceToLinearSrgb(infraredOnly);
    EXPECT_GT(rgb.g, 100.0f);  // 500 W/m^2/nm held flat across 380-780 nm

    // Genuinely zero in the visible means zero out.
    SpectralCurve dark;
    dark.samples.emplace_back(300.0f, 0.0f);
    dark.samples.emplace_back(2999.0f, 0.0f);
    dark.samples.emplace_back(3000.0f, 500.0f);
    const auto none = SpectralIrradianceToLinearSrgb(dark);
    EXPECT_NEAR(none.g, 0.0f, 1e-4f);
}

TEST(SpectralColour, EmptyCurveIsBlackRatherThanUndefined) {
    const auto rgb = SpectralIrradianceToLinearSrgb(SpectralCurve{});
    EXPECT_FLOAT_EQ(rgb.r, 0.0f);
    EXPECT_FLOAT_EQ(rgb.g, 0.0f);
    EXPECT_FLOAT_EQ(rgb.b, 0.0f);
}

// Sunlight is warm, and a renderer that says otherwise is describing a lamp
// nobody has. Values are for a spectrum shaped like the ASTM G-173 direct
// beam; the tolerance is wide because the point is the direction of the shift,
// not the third digit.
TEST(SpectralColour, DaylightIsWarmerThanEqualEnergy) {
    // Rayleigh scattering removes blue from the direct beam, so approximate
    // it as equal energy tilted down towards short wavelengths.
    SpectralCurve warm;
    for (int nm = 380; nm <= 780; nm += 10) {
        const float t = (nm - 380) / 400.0f;   // 0 at blue, 1 at red
        warm.samples.emplace_back(static_cast<f32>(nm), 0.6f + 0.8f * t);
    }

    const auto rgb = SpectralIrradianceToLinearSrgb(warm);
    EXPECT_GT(rgb.r, rgb.g);
    EXPECT_GT(rgb.g, rgb.b);

    double x = 0.0, y = 0.0;
    Chromaticity(rgb, x, y);
    EXPECT_GT(x, 1.0 / 3.0);  // warmer than the equal-energy point
}
