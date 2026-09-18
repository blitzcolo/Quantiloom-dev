// ============================================================================
// Quantiloom - Unit Tests for postprocess/Thermography.hpp
// ============================================================================
// The camera model, which is what separates a rendered radiance field from a
// rendered thermogram. Two surfaces at one temperature and two emissivities
// read differently through it, which is exactly the effect a measurement
// campaign has to live with and a simulation has to reproduce before it can be
// compared with one.
//
// This legacy helper describes an explicitly generic fixed-band estimate.
// A configured device uses its own full response and temperature derivative
// in CpuCameraPipeline; those tests live in test_camera_cpu.cpp.
// ============================================================================

#include <gtest/gtest.h>

#include "postprocess/Thermography.hpp"

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

TEST(ThermographyTest, GenericFixedBandNetdMatchesReadNoiseClosedForm) {
    // The compatibility estimate uses one documented representative
    // wavelength for the entire band. It must not be compared to a true
    // response integral or to GenericSensor's 1 nm fast-RGB proxy.
    SensorParams p = QuietSensor();
    p.quantumEfficiency = 0.6f;
    p.fNumber = 2.0f;
    p.pixelPitch_um = 15.0f;
    p.integrationTime_s = 0.02f;
    p.enableReadNoise = true;
    p.readNoise_e_rms = 1.0f;
    const f64 omega = 3.14159265358979323846 / (1.0 + 4.0 * 2.0 * 2.0);
    const f64 area = std::pow(15e-6, 2.0);
    const f64 photonEnergy = 6.62607015e-34 * 299792458.0 / 10e-6;
    const f64 electronsPerBandRadiance =
        omega * area * 0.02 * 0.6 / photonEnergy;
    const f64 dBandRadiancePerK =
        BandRadianceDerivativePerK(kLwirMin, kLwirMax, 300.0) *
        (kLwirMax - kLwirMin);
    const f64 expected = 1.0 / (electronsPerBandRadiance * dBandRadiancePerK);
    const f64 measured =
        NoiseEquivalentTemperatureDifferenceK(p, kLwirMin, kLwirMax, 300.0);
    EXPECT_NEAR(measured, expected, expected * 1e-6);
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
