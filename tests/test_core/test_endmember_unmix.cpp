/**
 * @file test_endmember_unmix.cpp
 * @brief Reflectance colour under D65, and the NNLS that inverts it
 *
 * These two functions are what decide, per texel, how much of each endmember a
 * surface is made of. Both are pure and cheap to check here, and both fail
 * quietly if they are wrong: a bad colour or a bad weight does not crash, it
 * just renders a slightly wrong material, which is the hardest kind of error
 * to notice in a spectral image.
 */

#include <gtest/gtest.h>

#include "core/NNLS.hpp"
#include "core/SpectralData.hpp"

using namespace quantiloom;

namespace {

SpectralCurve FlatCurve(f32 value, f32 lambdaMin = 360.0f, f32 lambdaMax = 800.0f) {
    SpectralCurve curve;
    for (f32 lambda = lambdaMin; lambda <= lambdaMax; lambda += 5.0f) {
        curve.samples.emplace_back(lambda, value);
    }
    return curve;
}

// A step: `value` below the edge, `other` above it. Crude, but enough to make a
// curve that is unambiguously blue-ish or red-ish.
SpectralCurve StepCurve(f32 below, f32 above, f32 edge_nm) {
    SpectralCurve curve;
    for (f32 lambda = 360.0f; lambda <= 800.0f; lambda += 5.0f) {
        curve.samples.emplace_back(lambda, lambda < edge_nm ? below : above);
    }
    return curve;
}

}  // namespace

// ============================================================================
// ReflectanceToLinearSrgbD65
// ============================================================================

// The definition of the normalisation: a perfect diffuser under the illuminant
// sRGB is referenced to is sRGB white. This is the one number in the whole
// unmixing chain that is fixed by definition rather than measured, so if it
// drifts, everything downstream is scaled by the drift without saying so.
TEST(EndmemberUnmix, PerfectDiffuserIsWhite) {
    const auto rgb = ReflectanceToLinearSrgbD65(FlatCurve(1.0f));

    EXPECT_NEAR(rgb.r, 1.0f, 2e-3f);
    EXPECT_NEAR(rgb.g, 1.0f, 2e-3f);
    EXPECT_NEAR(rgb.b, 1.0f, 2e-3f);
}

// Reflectance scales linearly: half the light back is half the colour. Worth
// pinning because it is what lets a single endmember act as a brightness
// modulation -- the k == 1 unmix is exactly this relation, inverted.
TEST(EndmemberUnmix, GreyScalesLinearly) {
    const auto rgb = ReflectanceToLinearSrgbD65(FlatCurve(0.5f));

    EXPECT_NEAR(rgb.r, 0.5f, 2e-3f);
    EXPECT_NEAR(rgb.g, 0.5f, 2e-3f);
    EXPECT_NEAR(rgb.b, 0.5f, 2e-3f);
}

TEST(EndmemberUnmix, EmptyCurveIsBlackNotGarbage) {
    const auto rgb = ReflectanceToLinearSrgbD65(SpectralCurve{});

    EXPECT_FLOAT_EQ(rgb.r, 0.0f);
    EXPECT_FLOAT_EQ(rgb.g, 0.0f);
    EXPECT_FLOAT_EQ(rgb.b, 0.0f);
}

// A curve that reflects only the long wavelengths must read red, and one that
// reflects only the short ones must read blue. This is the check that the
// illuminant and the observer are lined up the right way round -- swapping
// them, or reversing the table, still produces plausible-looking greys.
//
// Only the dominant channel is asserted. The other two are near zero for a
// spectrally saturated colour and their order is an artifact of the sRGB
// primaries, not of the spectrum: this red comes out g=0.021, b=0.038.
TEST(EndmemberUnmix, SpectrumShapeReachesTheRightChannel) {
    const auto red = ReflectanceToLinearSrgbD65(StepCurve(0.05f, 0.9f, 600.0f));
    const auto blue = ReflectanceToLinearSrgbD65(StepCurve(0.9f, 0.05f, 500.0f));

    EXPECT_GT(red.r, 4.0f * red.g);
    EXPECT_GT(red.r, 4.0f * red.b);
    EXPECT_GT(blue.b, 4.0f * blue.r);
    EXPECT_GT(blue.b, 4.0f * blue.g);
}

TEST(EndmemberUnmix, GamutIsClamped) {
    const auto rgb = ReflectanceToLinearSrgbD65(StepCurve(0.0f, 1.0f, 620.0f));

    EXPECT_GE(rgb.r, 0.0f);
    EXPECT_GE(rgb.g, 0.0f);
    EXPECT_GE(rgb.b, 0.0f);
    EXPECT_LE(rgb.r, 1.0f);
    EXPECT_LE(rgb.g, 1.0f);
    EXPECT_LE(rgb.b, 1.0f);
}

// ============================================================================
// SolveNNLS
// ============================================================================

TEST(EndmemberUnmix, SingleEndmemberIsAProjection) {
    const glm::vec3 c[1] = {glm::vec3(0.4f, 0.5f, 0.2f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 1, c[0] * 1.7f, w);
    EXPECT_NEAR(w[0], 1.7f, 1e-3f);

    // A texel darker than the endmember is that endmember, dimmer. This is the
    // brightness modulation that makes a curve-bound surface show its texture.
    SolveNNLS(c, 1, c[0] * 0.35f, w);
    EXPECT_NEAR(w[0], 0.35f, 1e-3f);
}

// The direction that must be refused: no surface contains a negative amount of
// a material, and a negative weight would drive the mixed reflectance below
// zero wherever it appeared.
TEST(EndmemberUnmix, NegativeProjectionIsClampedToZero) {
    const glm::vec3 c[1] = {glm::vec3(0.4f, 0.5f, 0.2f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 1, glm::vec3(-0.4f, -0.5f, -0.2f), w);
    EXPECT_FLOAT_EQ(w[0], 0.0f);
}

TEST(EndmemberUnmix, TwoOrthogonalEndmembersAreRecoveredExactly) {
    const glm::vec3 c[2] = {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 2, glm::vec3(0.3f, 0.0f, 0.7f), w);

    EXPECT_NEAR(w[0], 0.3f, 2e-3f);
    EXPECT_NEAR(w[1], 0.7f, 2e-3f);
}

// The mixture reconstructs the target even when the endmembers are not
// orthogonal, which is the realistic case -- two natural reflectance curves
// are usually well correlated.
TEST(EndmemberUnmix, ObliqueEndmembersReconstructTheTarget) {
    const glm::vec3 c[2] = {glm::vec3(0.35f, 0.42f, 0.18f), glm::vec3(0.55f, 0.48f, 0.30f)};
    const glm::vec3 target = 0.4f * c[0] + 0.45f * c[1];
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 2, target, w);
    const glm::vec3 fit = w[0] * c[0] + w[1] * c[1];

    EXPECT_NEAR(fit.r, target.r, 5e-3f);
    EXPECT_NEAR(fit.g, target.g, 5e-3f);
    EXPECT_NEAR(fit.b, target.b, 5e-3f);
}

// A target that only one endmember can explain must not borrow a negative
// amount of the other to fit better.
TEST(EndmemberUnmix, ActiveSetDropsAnEndmemberRatherThanGoNegative) {
    const glm::vec3 c[2] = {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 2, glm::vec3(0.8f, -0.5f, 0.0f), w);

    EXPECT_NEAR(w[0], 0.8f, 2e-3f);
    EXPECT_FLOAT_EQ(w[1], 0.0f);
}

// Four endmembers against three colour equations is underdetermined; the ridge
// term is what stops it from picking an arbitrary point on the solution line.
// The requirement is not a particular w, it is that the fit is good, the
// weights are non-negative and finite, and that near-identical colours do not
// get wildly different answers.
TEST(EndmemberUnmix, UnderdeterminedFourWayStaysFiniteAndClose) {
    const glm::vec3 c[4] = {
        glm::vec3(0.60f, 0.30f, 0.20f), glm::vec3(0.25f, 0.55f, 0.20f),
        glm::vec3(0.20f, 0.25f, 0.65f), glm::vec3(0.40f, 0.40f, 0.40f)};
    const glm::vec3 target(0.42f, 0.38f, 0.33f);

    f32 w[MAX_ENDMEMBERS] = {};
    SolveNNLS(c, 4, target, w);

    glm::vec3 fit(0.0f);
    for (i32 i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(w[i]));
        EXPECT_GE(w[i], 0.0f);
        fit += w[i] * c[i];
    }
    EXPECT_NEAR(fit.r, target.r, 1e-2f);
    EXPECT_NEAR(fit.g, target.g, 1e-2f);
    EXPECT_NEAR(fit.b, target.b, 1e-2f);

    f32 wNudged[MAX_ENDMEMBERS] = {};
    SolveNNLS(c, 4, target + glm::vec3(1e-4f), wNudged);
    for (i32 i = 0; i < 4; ++i) {
        EXPECT_NEAR(w[i], wNudged[i], 5e-2f);
    }
}

// Two identical endmembers are the degenerate case the ridge term exists for:
// the split between them is arbitrary, but the answer must still be finite and
// must still reconstruct the target.
TEST(EndmemberUnmix, CollinearEndmembersDoNotBlowUp) {
    const glm::vec3 c[2] = {glm::vec3(0.5f, 0.4f, 0.3f), glm::vec3(0.5f, 0.4f, 0.3f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 2, glm::vec3(0.5f, 0.4f, 0.3f), w);

    EXPECT_TRUE(std::isfinite(w[0]));
    EXPECT_TRUE(std::isfinite(w[1]));
    EXPECT_GE(w[0], 0.0f);
    EXPECT_GE(w[1], 0.0f);
    EXPECT_NEAR(w[0] + w[1], 1.0f, 2e-2f);
}

TEST(EndmemberUnmix, BlackTargetGivesNoWeight) {
    const glm::vec3 c[2] = {glm::vec3(0.5f, 0.4f, 0.3f), glm::vec3(0.2f, 0.6f, 0.4f)};
    f32 w[MAX_ENDMEMBERS] = {};

    SolveNNLS(c, 2, glm::vec3(0.0f), w);

    EXPECT_FLOAT_EQ(w[0], 0.0f);
    EXPECT_FLOAT_EQ(w[1], 0.0f);
}

// The round trip the unmixer actually performs: measured curves in, endmember
// colours out, a texel that IS one of those colours back to a unit weight.
TEST(EndmemberUnmix, CurveColoursUnmixBackToUnitWeights) {
    const glm::vec3 colors[2] = {
        ReflectanceToLinearSrgbD65(StepCurve(0.05f, 0.7f, 600.0f)),   // reddish
        ReflectanceToLinearSrgbD65(StepCurve(0.7f, 0.05f, 500.0f))};  // bluish

    f32 w[MAX_ENDMEMBERS] = {};
    SolveNNLS(colors, 2, colors[0], w);
    EXPECT_NEAR(w[0], 1.0f, 2e-2f);
    EXPECT_NEAR(w[1], 0.0f, 2e-2f);

    SolveNNLS(colors, 2, colors[1], w);
    EXPECT_NEAR(w[0], 0.0f, 2e-2f);
    EXPECT_NEAR(w[1], 1.0f, 2e-2f);
}
