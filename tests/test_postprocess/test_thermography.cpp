// ============================================================================
// Quantiloom - Unit Tests for postprocess/Thermography.hpp
// ============================================================================
// The camera model, which is what separates a rendered radiance field from a
// rendered thermogram. Two surfaces at one temperature and two emissivities
// read differently through it, which is exactly the effect a measurement
// campaign has to live with and a simulation has to reproduce before it can be
// compared with one.
//
// The NETD case is the one worth keeping honest: its responsivity is a second
// copy of the chain GenericSensor applies, so it is asserted against that
// chain rather than against itself.
// ============================================================================

#include <gtest/gtest.h>

#include "postprocess/Thermography.hpp"

#include "postprocess/GenericSensor.hpp"

#include <cmath>

using namespace quantiloom;

namespace {

constexpr f64 kLwirMin = 8000.0, kLwirMax = 12000.0;
constexpr f64 kMwirMin = 3000.0, kMwirMax = 5000.0;

/// A sensor with one noise source at a time, so a NETD can be attributed.
SensorParams QuietSensor() {
    SensorParams p;
    p.enablePoissonNoise = false;
    p.enableReadNoise = false;
    p.enableDarkCurrent = false;
    p.enableFPN = false;
    p.wavelength_nm = 10000.0f;
    p.integrationTime_s = 0.01f;
    return p;
}

}  // namespace

// ============================================================================
// Apparent temperature: what a camera assuming a blackbody would report
// ============================================================================

TEST(ThermographyTest, ABlackbodyReadsItsOwnTemperature) {
    for (const f64 T : {250.0, 300.0, 400.0, 800.0}) {
        const f64 L = BlackbodyBandRadiance(kLwirMin, kLwirMax, T);
        EXPECT_NEAR(InvertApparentTemperatureK(L, kLwirMin, kLwirMax), T, 1e-6);
    }
}

TEST(ThermographyTest, AGreyBodyReadsColderThanItIs) {
    // The reason a thermogram is not a temperature field. A surface at 320 K
    // with emissivity 0.9, reflecting a 250 K sky, sends less than a 320 K
    // blackbody would, and a camera told nothing reports the difference as
    // temperature.
    const f64 T_surface = 320.0;
    const f64 T_sky = 250.0;
    const f64 eps = 0.9;

    const f64 measured = eps * BlackbodyBandRadiance(kLwirMin, kLwirMax, T_surface) +
                         (1.0 - eps) * BlackbodyBandRadiance(kLwirMin, kLwirMax, T_sky);

    const f64 apparent = InvertApparentTemperatureK(measured, kLwirMin, kLwirMax);
    EXPECT_LT(apparent, T_surface);
    EXPECT_GT(apparent, T_sky);
}

// ============================================================================
// The correction, Aguerre eq. 8 taken per band
// ============================================================================

TEST(ThermographyTest, TellingTheCameraTheEmissivityRecoversTheSurface) {
    const f64 T_surface = 320.0;
    const f64 T_refl = 250.0;

    for (const f64 eps : {0.5, 0.75, 0.9, 0.98}) {
        const f64 measured = eps * BlackbodyBandRadiance(kLwirMin, kLwirMax, T_surface) +
                             (1.0 - eps) * BlackbodyBandRadiance(kLwirMin, kLwirMax, T_refl);

        ThermographyParams params;
        params.emissivity = static_cast<f32>(eps);
        params.reflectedTemperature_K = static_cast<f32>(T_refl);

        EXPECT_NEAR(InvertSurfaceTemperatureK(measured, kLwirMin, kLwirMax, params), T_surface,
                    1e-3)
            << "at emissivity " << eps;
    }
}

TEST(ThermographyTest, TheAtmosphericTermRecoversTheSurfaceToo) {
    // L = tau [eps B(Ts) + (1-eps) B(Trefl)] + (1-tau) B(Tatm)
    const f64 T_surface = 350.0, T_refl = 260.0, T_atm = 285.0;
    const f64 eps = 0.85, tau = 0.7;

    const f64 measured = tau * (eps * BlackbodyBandRadiance(kMwirMin, kMwirMax, T_surface) +
                                (1.0 - eps) * BlackbodyBandRadiance(kMwirMin, kMwirMax, T_refl)) +
                         (1.0 - tau) * BlackbodyBandRadiance(kMwirMin, kMwirMax, T_atm);

    ThermographyParams params;
    params.emissivity = static_cast<f32>(eps);
    params.reflectedTemperature_K = static_cast<f32>(T_refl);
    params.atmosphereTransmittance = static_cast<f32>(tau);
    params.atmosphereTemperature_K = static_cast<f32>(T_atm);

    EXPECT_NEAR(InvertSurfaceTemperatureK(measured, kMwirMin, kMwirMax, params), T_surface, 1e-3);
}

TEST(ThermographyTest, DefaultParamsAreExactlyApparentTemperature) {
    const f64 L = BlackbodyBandRadiance(kLwirMin, kLwirMax, 305.0) * 0.8;
    EXPECT_DOUBLE_EQ(InvertSurfaceTemperatureK(L, kLwirMin, kLwirMax, ThermographyParams{}),
                     InvertApparentTemperatureK(L, kLwirMin, kLwirMax));
}

TEST(ThermographyTest, ParametersThatOversubtractGiveTheFloorRatherThanNaN) {
    // Claiming a reflected source hotter than the whole measurement leaves a
    // negative blackbody. That is the parameters disagreeing with the image,
    // and it has to stay finite so the disagreement is visible in the map.
    ThermographyParams params;
    params.emissivity = 0.1f;
    params.reflectedTemperature_K = 1000.0f;

    const f64 measured = BlackbodyBandRadiance(kLwirMin, kLwirMax, 300.0);
    const f64 T = InvertSurfaceTemperatureK(measured, kLwirMin, kLwirMax, params);
    EXPECT_TRUE(std::isfinite(T));
    EXPECT_LT(T, 200.0);
}

TEST(ThermographyTest, AZeroEmissivitySurfaceCarriesNoTemperature) {
    ThermographyParams params;
    params.emissivity = 0.0f;
    const f64 T = InvertSurfaceTemperatureK(1.0, kLwirMin, kLwirMax, params);
    EXPECT_TRUE(std::isfinite(T));
}

// ============================================================================
// NETD
// ============================================================================

TEST(ThermographyTest, NetdIsZeroWithoutNoiseAndFiniteWithIt) {
    SensorParams p = QuietSensor();
    EXPECT_DOUBLE_EQ(NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0), 0.0);

    p.enableReadNoise = true;
    const f64 netd = NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0);
    EXPECT_GT(netd, 0.0);
    EXPECT_TRUE(std::isfinite(netd));
}

TEST(ThermographyTest, NetdImprovesWithIntegrationTime) {
    // Read noise is per frame while the signal accumulates, so a longer
    // integration buys sensitivity. The direction is the physics; the exact
    // factor depends on which noise dominates.
    SensorParams p = QuietSensor();
    p.enableReadNoise = true;

    const f64 shortT = NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0);
    p.integrationTime_s = 0.04f;
    const f64 longT = NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0);

    EXPECT_LT(longT, shortT);
    EXPECT_NEAR(longT, shortT / 4.0, shortT * 1e-6);  // read-noise-limited: 1/t
}

TEST(ThermographyTest, NetdResponsivityIsTheSensorChainsOwn) {
    // The one number in this file that is a second copy of something: the
    // radiance-to-electrons factor. Recovered here by running the actual
    // sensor with every noise source off, so a divergence between the two
    // fails rather than quietly changing every NETD ever reported.
    SensorParams p = QuietSensor();
    p.quantumEfficiency = 0.6f;
    p.fNumber = 2.0f;
    p.pixelPitch_um = 15.0f;
    p.integrationTime_s = 0.02f;
    p.psfSigma_px = 0.0f;  // no blur, so a flat field stays flat
    p.wellCapacity_e = 1e9f;
    p.gain = 1.0f;
    p.bitDepth = 24;

    // A flat band-integrated radiance, which is what the chain is fed. Small
    // enough that the electrons it produces stay inside the ADC: at unit
    // radiance this sensor makes 2.5e7 of them, and a 24-bit converter clips
    // at 1.7e7, which reads as a responsivity mismatch rather than as
    // saturation.
    constexpr f32 kRadiance = 0.1f;
    Image flat(4, 4, 3);
    for (auto& v : flat.data) {
        v = kRadiance;
    }

    GenericSensor sensor;
    auto result = sensor.Apply(flat, p);
    ASSERT_TRUE(result.has_value()) << result.error();

    // DN back to electrons through the gain, which is what the chain divided
    // by. Noise is off, so this is the signal exactly.
    const f64 electrons = static_cast<f64>(result.value().rawDN(2, 2, 0)) * p.gain;

    // NETD's model, run forward: sigma_e of 1 electron would be worth
    // 1/responsivity of radiance, so responsivity = electrons / radiance.
    SensorParams noiseOnly = p;
    noiseOnly.enableReadNoise = true;
    noiseOnly.readNoise_e_rms = 1.0f;
    const f64 netd =
        NoiseEquivalentTemperatureDifferenceK(noiseOnly, kLwirMin, kLwirMax, 300.0);
    const f64 dLdT = BandRadianceDerivativePerK(kLwirMin, kLwirMax, 300.0) *
                     (kLwirMax - kLwirMin);
    const f64 impliedResponsivity = 1.0 / (netd * dLdT);

    EXPECT_NEAR(impliedResponsivity, electrons / kRadiance, electrons * 1e-3);
}

TEST(ThermographyTest, NetdIsWorseWhereTheBandHasLessSensitivity) {
    // The MWIR's dL/dT at room temperature is far smaller than the LWIR's, so
    // the same detector noise buys a coarser temperature there. This is why a
    // NETD without a band and a scene temperature attached means nothing.
    SensorParams p = QuietSensor();
    p.enableReadNoise = true;

    const f64 lwir = NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0);
    const f64 mwir = NoiseEquivalentTemperatureDifferenceK(p, kMwirMin, kMwirMax, 300.0);
    EXPECT_GT(mwir, lwir);
}
