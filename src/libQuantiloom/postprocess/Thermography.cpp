/**
 * @file Thermography.cpp
 * @brief Band radiance back to temperature
 */

#include "postprocess/Thermography.hpp"

#include "core/Blackbody.hpp"

#include <cmath>
#include <limits>

namespace quantiloom {

namespace {

// The same two constants GenericSensor computes photon energy from.
constexpr f64 kPlanckConstant = 6.62607015e-34;  // J s
constexpr f64 kSpeedOfLight = 299792458.0;       // m/s

/// Electrons per unit band-integrated radiance (W sr^-1 m^-2), on axis.
///
/// GenericSensor::RadianceToElectrons chains E = L*Omega, then E*A*t joules,
/// then divides by the photon energy and multiplies by QE. Collected here as
/// one factor because a noise-equivalent radiance is that chain run backwards,
/// and because the vignetting and well clamp the pixel loop also applies are
/// not part of a sensitivity figure -- NETD is quoted on axis and below
/// saturation.
f64 ResponsivityElectronsPerRadiance(const SensorParams& p) {
    const f64 pixelPitchM = static_cast<f64>(p.pixelPitch_um) * 1e-6;
    const f64 pixelArea_m2 = pixelPitchM * pixelPitchM;
    const f64 wavelengthM = static_cast<f64>(p.wavelength_nm) * 1e-9;
    if (wavelengthM <= 0.0) {
        return 0.0;
    }
    const f64 photonEnergy_J = (kPlanckConstant * kSpeedOfLight) / wavelengthM;
    const f64 solidAngle_sr = ApertureSolidAngleSr(p.fNumber);

    return solidAngle_sr * pixelArea_m2 * static_cast<f64>(p.integrationTime_s) *
           static_cast<f64>(p.quantumEfficiency) / photonEnergy_J;
}

}  // namespace

f64 InvertApparentTemperatureK(const f64 radiance, const f64 lambdaMinNm,
                               const f64 lambdaMaxNm) {
    return blackbody::InvertBandAverageRadiance(radiance, lambdaMinNm, lambdaMaxNm);
}

f64 InvertSurfaceTemperatureK(const f64 radiance, const f64 lambdaMinNm,
                              const f64 lambdaMaxNm, const ThermographyParams& params) {
    const f64 eps = static_cast<f64>(params.emissivity);
    const f64 tau = static_cast<f64>(params.atmosphereTransmittance);

    // An emissivity of zero says the surface emits nothing, so no measurement
    // of it can carry a temperature. A non-positive transmittance says nothing
    // reaches the lens. Both are configuration errors rather than cold scenes.
    if (eps <= 0.0 || tau <= 0.0) {
        return blackbody::kMinInvertibleK;
    }

    f64 surfaceBlackbody = radiance / tau;

    if (eps < 1.0 && params.reflectedTemperature_K > 0.0) {
        surfaceBlackbody -= (1.0 - eps) *
                            blackbody::BandAverageRadiance(
                                lambdaMinNm, lambdaMaxNm,
                                static_cast<f64>(params.reflectedTemperature_K));
    }
    if (tau < 1.0 && params.atmosphereTemperature_K > 0.0) {
        surfaceBlackbody -= ((1.0 - tau) / tau) *
                            blackbody::BandAverageRadiance(
                                lambdaMinNm, lambdaMaxNm,
                                static_cast<f64>(params.atmosphereTemperature_K));
    }

    surfaceBlackbody /= eps;

    // Negative here means the reflected and path terms account for more than
    // the pixel received: the parameters disagree with the image. Reporting
    // the floor keeps the map finite and the disagreement visible, which a
    // NaN would not.
    return blackbody::InvertBandAverageRadiance(surfaceBlackbody, lambdaMinNm, lambdaMaxNm);
}

f64 BandRadianceDerivativePerK(const f64 lambdaMinNm, const f64 lambdaMaxNm,
                               const f64 temperatureK) {
    return blackbody::BandAverageRadianceDerivative(lambdaMinNm, lambdaMaxNm, temperatureK);
}

f64 BlackbodyBandRadiance(const f64 lambdaMinNm, const f64 lambdaMaxNm,
                          const f64 temperatureK) {
    return blackbody::BandAverageRadiance(lambdaMinNm, lambdaMaxNm, temperatureK);
}

f64 NoiseEquivalentTemperatureDifferenceK(const SensorParams& sensor, const f64 lambdaMinNm,
                                          const f64 lambdaMaxNm, const f64 sceneTemperatureK) {
    const f64 responsivity = ResponsivityElectronsPerRadiance(sensor);
    if (responsivity <= 0.0 || lambdaMaxNm <= lambdaMinNm) {
        return std::numeric_limits<f64>::infinity();
    }

    const f64 bandWidthNm = lambdaMaxNm - lambdaMinNm;

    // The signal the scene puts on the detector, band-integrated because the
    // responsivity is per band-integrated radiance while the blackbody
    // routines are per nm.
    const f64 sceneRadiance =
        blackbody::BandAverageRadiance(lambdaMinNm, lambdaMaxNm, sceneTemperatureK) *
        bandWidthNm;
    const f64 signalElectrons = responsivity * sceneRadiance;

    f64 varianceElectrons = 0.0;
    if (sensor.enablePoissonNoise) {
        varianceElectrons += signalElectrons;  // shot noise: variance = mean
    }
    if (sensor.enableReadNoise) {
        varianceElectrons +=
            static_cast<f64>(sensor.readNoise_e_rms) * static_cast<f64>(sensor.readNoise_e_rms);
    }
    if (sensor.enableDarkCurrent) {
        varianceElectrons +=
            static_cast<f64>(sensor.darkCurrent_e_s) * static_cast<f64>(sensor.integrationTime_s);
    }
    if (varianceElectrons <= 0.0) {
        return 0.0;  // a noiseless sensor resolves any difference
    }

    // Referred back to the scene: electrons to radiance through the same
    // responsivity, radiance to kelvin through the band's own slope.
    const f64 noiseRadiance = std::sqrt(varianceElectrons) / responsivity;
    const f64 dRadiance_dT =
        blackbody::BandAverageRadianceDerivative(lambdaMinNm, lambdaMaxNm, sceneTemperatureK) *
        bandWidthNm;
    if (dRadiance_dT <= 0.0) {
        return std::numeric_limits<f64>::infinity();
    }
    return noiseRadiance / dRadiance_dT;
}

}  // namespace quantiloom
