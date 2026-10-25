// ============================================================================
// Quantiloom - Unit Tests for renderer/SkyDome.hpp
// ============================================================================
// The clear sky is only allowed to move the sky's light around, never to add
// or remove any: per wavelength the factor the miss shader applies must
// integrate, cosine-weighted over the upper hemisphere, to pi -- the uniform
// dome's own integral. These check that on the CPU twin, with a quadrature
// independent of the one FillSkyDome uses, and check that the light moved the
// way a clear sky moves it.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/SkyDome.hpp"

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::rendercore;

namespace {

constexpr double kPi = 3.14159265358979323846;

glm::vec3 SunAtElevation(float degrees) {
    const float e = degrees * static_cast<float>(kPi) / 180.0f;
    return glm::vec3(std::cos(e) * 0.6f, std::sin(e), std::cos(e) * 0.8f);
}

// (pi / N) * shape, cosine-weighted over the upper hemisphere on a theta-phi
// grid finer than, and laid out differently from, ClearSkyNode's mu-phi one.
double IntegratedFactor(float lambdaNm, const glm::vec3& sun, float aod) {
    const ClearSkyNodeValues node = ClearSkyNode(lambdaNm, sun, aod);
    constexpr int kTheta = 256;
    constexpr int kPhi = 512;
    double sum = 0.0;
    for (int i = 0; i < kTheta; ++i) {
        const double theta = (i + 0.5) / kTheta * kPi / 2.0;
        for (int j = 0; j < kPhi; ++j) {
            const double phi = (j + 0.5) / kPhi * 2.0 * kPi;
            const glm::vec3 dir(static_cast<float>(std::sin(theta) * std::cos(phi)),
                                static_cast<float>(std::cos(theta)),
                                static_cast<float>(std::sin(theta) * std::sin(phi)));
            const double shape =
                ClearSkyShape(lambdaNm, dir, sun, aod, node.multipleScattering);
            sum += (kPi / node.cosineIntegral) * shape * std::cos(theta) * std::sin(theta);
        }
    }
    return sum * (kPi / 2.0 / kTheta) * (2.0 * kPi / kPhi);
}

}  // namespace

TEST(SkyDomeTest, ExponentialIntegralE3MatchesTabulatedValues) {
    // Abramowitz & Stegun table 5.4 / scipy.special.expn(3, x).
    EXPECT_NEAR(ExponentialIntegralE3(0.0), 0.5, 1e-12);
    EXPECT_NEAR(ExponentialIntegralE3(0.1), 0.4162914579, 1e-9);
    EXPECT_NEAR(ExponentialIntegralE3(0.5), 0.2216043643, 1e-9);
    EXPECT_NEAR(ExponentialIntegralE3(1.0), 0.1096919672, 1e-9);
    EXPECT_NEAR(ExponentialIntegralE3(3.0), 0.0089306466, 1e-9);
}

TEST(SkyDomeTest, TheClearSkyDeliversTheIlluminantsIrradiance) {
    // Low, middle and high sun; blue, green, red; G-173's aerosol and a hazy one.
    for (const float elevation : {10.0f, 45.0f, 80.0f}) {
        const glm::vec3 sun = SkySunDirection(SunAtElevation(elevation));
        for (const float lambda : {400.0f, 550.0f, 700.0f}) {
            for (const float aod : {0.084f, 0.5f}) {
                EXPECT_NEAR(IntegratedFactor(lambda, sun, aod), kPi, kPi * 2e-3)
                    << "elevation " << elevation << " deg, " << lambda << " nm, aod " << aod;
            }
        }
    }
}

TEST(SkyDomeTest, TheZenithIsBluerThanTheHorizon) {
    // Relative to the average each wavelength keeps: overhead the short
    // wavelengths are over-represented, at the horizon under-represented.
    const glm::vec3 sun = SkySunDirection(SunAtElevation(45.0f));
    const glm::vec3 zenith(0.0f, 1.0f, 0.0f);
    const glm::vec3 horizon = glm::normalize(glm::vec3(-0.6f, 0.03f, -0.8f));  // away from the sun

    const auto factor = [&](float lambda, const glm::vec3& dir) {
        const ClearSkyNodeValues node = ClearSkyNode(lambda, sun, 0.084f);
        return (kPi / node.cosineIntegral) *
               ClearSkyShape(lambda, dir, sun, 0.084f, node.multipleScattering);
    };
    const double zenithBlueOverRed = factor(450.0f, zenith) / factor(650.0f, zenith);
    const double horizonBlueOverRed = factor(450.0f, horizon) / factor(650.0f, horizon);
    EXPECT_GT(zenithBlueOverRed, 1.0);
    EXPECT_LT(horizonBlueOverRed, 1.0);

    // And the horizon is brighter than the zenith, which is the other half of
    // what a clear sky looks like.
    EXPECT_GT(factor(550.0f, horizon), factor(550.0f, zenith));
}

TEST(SkyDomeTest, AUniformSkyGetsNoTables) {
    LightingParams params = CreateDefaultLightingParams();
    params.skyDistribution = SkyDistributionId::Uniform;
    params.skyDomeN[3] = 1.0f;
    FillSkyDome(params);
    for (float n : params.skyDomeN) EXPECT_EQ(n, 0.0f);
    for (float j : params.skyDomeJ) EXPECT_EQ(j, 0.0f);
}

TEST(SkyDomeTest, ASunBelowTheHorizonStillGivesAFiniteSky) {
    // The plane-parallel sky has no twilight; the shape holds the sun at three
    // degrees, and E_sky decides how much light there is.
    LightingParams params = CreateDefaultLightingParams();
    params.sunDirection = glm::vec3(0.3f, -0.4f, 0.2f);
    FillSkyDome(params);
    for (u32 k = 0; k < skydome::kNodes; ++k) {
        EXPECT_TRUE(std::isfinite(params.skyDomeN[k]) && params.skyDomeN[k] > 0.0f) << k;
        EXPECT_TRUE(std::isfinite(params.skyDomeJ[k]) && params.skyDomeJ[k] >= 0.0f) << k;
    }
    const glm::vec3 sun = SkySunDirection(params.sunDirection);
    EXPECT_NEAR(sun.y, skydome::kMinSunSin, 1e-6f);
    EXPECT_NEAR(glm::length(sun), 1.0f, 1e-5f);
}
