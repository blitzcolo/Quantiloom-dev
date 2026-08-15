// ============================================================================
// Quantiloom - Unit Tests for core/SkyThermal.hpp
// ============================================================================
// The clear-sky correlation, checked against the two things about it that are
// not fits: the dew point is exact at saturation, and the sky can never
// radiate more than a blackbody at the air temperature. The correlation's own
// coefficients are pinned at a handful of points so a typo in them fails here
// rather than in a render nobody has a reference for.
// ============================================================================

#include <gtest/gtest.h>

#include "core/SkyThermal.hpp"

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::skythermal;

// ============================================================================
// Dew point
// ============================================================================

TEST(SkyThermalTest, SaturatedAirHasTheAirTemperatureAsItsDewPoint) {
    // Exact, not fitted: at RH = 100 the logarithm vanishes and Magnus
    // reduces to an identity.
    for (const f64 T : {-10.0, 0.0, 15.0, 30.0}) {
        EXPECT_NEAR(DewPointC(T, 100.0), T, 1e-9) << "at " << T << " C";
    }
}

TEST(SkyThermalTest, DewPointIsBelowTheAirTemperatureAndFallsWithHumidity) {
    const f64 airC = 20.0;
    f64 previous = airC;
    for (const f64 rh : {90.0, 70.0, 50.0, 30.0, 10.0}) {
        const f64 dp = DewPointC(airC, rh);
        EXPECT_LT(dp, airC);
        EXPECT_LT(dp, previous) << "not monotone at RH " << rh;
        previous = dp;
    }
}

TEST(SkyThermalTest, DewPointAtTwentyCelsiusAndFiftyPercent) {
    // The textbook value for this pair is 9.3 C; anything that moves the
    // Magnus coefficients moves this.
    EXPECT_NEAR(DewPointC(20.0, 50.0), 9.3, 0.1);
}

TEST(SkyThermalTest, VeryDryAirIsClampedRatherThanRunningAway) {
    // ln(RH/100) goes to minus infinity as the humidity does to zero, and no
    // measured sky is that dry. The clamp is what keeps a config typo from
    // producing a NaN emissivity.
    EXPECT_TRUE(std::isfinite(DewPointC(20.0, 0.0)));
    EXPECT_DOUBLE_EQ(DewPointC(20.0, 0.0), DewPointC(20.0, 1.0));
}

// ============================================================================
// Emissivity
// ============================================================================

TEST(SkyThermalTest, EmissivityFollowsBerdahlFromberg) {
    // eps = 0.711 + 0.56 (Tdp/100) + 0.73 (Tdp/100)^2, spelled out here so a
    // changed coefficient fails rather than agreeing with itself.
    for (const f64 dp : {-20.0, -5.0, 0.0, 10.0, 25.0}) {
        const f64 t = dp / 100.0;
        EXPECT_NEAR(ClearSkyEmissivity(dp), 0.711 + 0.56 * t + 0.73 * t * t, 1e-12)
            << "at dew point " << dp << " C";
    }
}

TEST(SkyThermalTest, AWetterSkyRadiatesMore) {
    f64 previous = 0.0;
    for (const f64 dp : {-20.0, -10.0, 0.0, 10.0, 20.0, 30.0}) {
        const f64 eps = ClearSkyEmissivity(dp);
        EXPECT_GT(eps, previous) << "not monotone at dew point " << dp;
        previous = eps;
    }
}

TEST(SkyThermalTest, EmissivityStaysAPhysicalFraction) {
    // The quadratic leaves [0, 1] outside the range it was fitted over, and a
    // sky that emits more than a blackbody would put energy into a scene from
    // nowhere.
    for (const f64 dp : {-100.0, -50.0, 60.0, 200.0}) {
        const f64 eps = ClearSkyEmissivity(dp);
        EXPECT_GE(eps, 0.0);
        EXPECT_LE(eps, 1.0);
    }
}

// ============================================================================
// Effective temperature
// ============================================================================

TEST(SkyThermalTest, EffectiveSkyTemperatureIsColderThanTheAir) {
    const f64 airK = 288.15;
    const f64 eps = ClearSkyEmissivity(DewPointC(15.0, 50.0));
    const f64 skyK = EffectiveSkyTemperatureK(airK, eps);

    EXPECT_LT(skyK, airK);
    // A clear sky at moderate humidity runs 15-25 K below the air; much less
    // and there would be no radiative cooling, much more and the correlation
    // is not the one being used.
    EXPECT_GT(airK - skyK, 10.0);
    EXPECT_LT(airK - skyK, 35.0);
}

TEST(SkyThermalTest, AUnitEmissivitySkyIsTheAirTemperature) {
    EXPECT_DOUBLE_EQ(EffectiveSkyTemperatureK(288.15, 1.0), 288.15);
}

TEST(SkyThermalTest, SigmaTFourthIsWhatTheQuarterPowerInverts) {
    // T_sky = T_air eps^(1/4) is the temperature whose blackbody flux equals
    // the sky's, so the fluxes must agree.
    const f64 airK = 300.0;
    const f64 eps = 0.8;
    const f64 skyK = EffectiveSkyTemperatureK(airK, eps);
    EXPECT_NEAR(std::pow(skyK, 4.0), eps * std::pow(airK, 4.0), eps * std::pow(airK, 4.0) * 1e-12);
}
