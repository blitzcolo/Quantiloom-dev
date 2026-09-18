#pragma once

#include "postprocess/CameraPipeline.hpp"

#include <span>

namespace quantiloom::camera {

inline constexpr f64 kPlanckJs = 6.62607015e-34;
inline constexpr f64 kLightSpeedMps = 299792458.0;
inline constexpr f64 kBoltzmannJPerK = 1.380649e-23;

enum class NoiseClass : u32 {
    PhotonShot, DarkShot, Read, Bias, FixedPrnu, FixedDsnu,
    ThermalRead, ThermalDrift, EmpiricalEffect
};

[[nodiscard]] Result<void, String> ValidateResponse(const ResponseCurve& response);
[[nodiscard]] Result<void, String> ValidateResponseStack(const ResponseStack& response,
                                                         DetectorKind detector);
[[nodiscard]] Result<void, String> ValidateCameraConfig(const CameraConfig& config);
[[nodiscard]] Result<CameraPoseKey, String> CameraPoseAt(
    const CameraMotionConfig& motion, f64 timeSeconds);

// The sole radiance-to-irradiance boundary. The input radiance is W/m^2/sr/nm.
// The result is W/m^2/nm BEFORE any lens/filter/QE response is applied.
// On-axis collection uses the projected pupil solid-angle factor pi/(1+4N^2),
// not the geometric cone angle; off-axis multiplies cos^4(angle).
[[nodiscard]] Result<std::vector<SpectralIrradianceSample>, String>
RadianceToIrradiance(std::span<const SpectralRadianceSample> radiance,
                     f64 fNumber, f64 fieldAngleRad,
                     bool applyCosFourth = true);
[[nodiscard]] Result<f64, String> ApertureSolidAngleSr(f64 fNumber);
[[nodiscard]] Result<f64, String> PixelCollectionAreaM2(const OpticsConfig& optics);
[[nodiscard]] Result<f64, String> HorizontalFovRadians(f64 focalLengthMm,
                                                       f64 pixelPitchUm, u32 widthPx);
[[nodiscard]] Result<f64, String> EffectiveFocalLengthMm(f64 horizontalFovRad,
                                                         f64 pixelPitchUm, u32 widthPx);

// Samples are pre-response irradiance in W/m^2/nm. The detector response
// defines the integration range; all samples and ancillary curves must cover it.
// Electron rate is e-/s; thermal absorbed power is W. No silent zero on error.
[[nodiscard]] Result<PhotonMeasurement, String>
IntegratePhoton(std::span<const SpectralIrradianceSample> samples,
                const ResponseStack& response, f64 pixelAreaM2);
[[nodiscard]] Result<ThermalMeasurement, String>
IntegrateThermal(std::span<const SpectralIrradianceSample> samples,
                 const ResponseStack& response, f64 pixelAreaM2);

// AiryIntensityNormalized is a dimensionless peak-normalized profile:
// [2 J1(pi*r/(lambda*N)) / (pi*r/(lambda*N))]^2.
// AiryPsfPerSquareMeter is normalized to unit integral over the image plane.
[[nodiscard]] Result<f64, String> AiryIntensityNormalized(f64 radiusM,
                                                          f64 wavelengthNm, f64 fNumber);
[[nodiscard]] Result<f64, String> AiryPsfPerSquareMeter(f64 radiusM,
                                                        f64 wavelengthNm, f64 fNumber);
[[nodiscard]] Result<f64, String> AiryPixelFraction(f64 centerXM, f64 centerYM,
                                                    f64 pixelPitchM, f64 wavelengthNm,
                                                    f64 fNumber, u32 samplesPerSide);

[[nodiscard]] Result<f64, String> PlanckRadianceWm2SrNm(f64 wavelengthNm,
                                                        f64 temperatureK);
[[nodiscard]] Result<f64, String> PlanckDerivativeWm2SrNmPerK(f64 wavelengthNm,
                                                              f64 temperatureK);
[[nodiscard]] Result<PhotonMeasurement, String>
IntegrateBlackbodyPhoton(f64 temperatureK, const ResponseStack& response,
                         f64 fNumber, f64 pixelAreaM2);
[[nodiscard]] Result<ThermalMeasurement, String>
IntegrateBlackbodyThermal(f64 temperatureK, const ResponseStack& response,
                          f64 fNumber, f64 pixelAreaM2);
[[nodiscard]] Result<f64, String>
BlackbodyThermalDerivativeWPerK(f64 temperatureK, const ResponseStack& response,
                                f64 fNumber, f64 pixelAreaM2);

// Exact constant-forcing solution of tau ds/dt + s = absorbed power.
// dt=0 preserves state; tau=0 gives instantaneous response. Negative or
// non-finite inputs fail.
[[nodiscard]] Result<f64, String> StepThermalResponse(f64 previousW,
                                                       f64 absorbedPowerW,
                                                       f64 dtSeconds, f64 tauSeconds);

// Portable 32-bit counter stream. Acquisition is ignored inside this function
// for FixedPrnu/FixedDsnu; all other classes include it. GPU code must mirror
// these integer operations and folding of the 64-bit capture index.
[[nodiscard]] u32 CounterRandomU32(u32 deviceSeed, u32 pixelIndex,
                                   u64 acquisitionIndex, NoiseClass noiseClass,
                                   u32 counter);
[[nodiscard]] f64 CounterUniform01(u32 deviceSeed, u32 pixelIndex,
                                   u64 acquisitionIndex, NoiseClass noiseClass,
                                   u32 counter);

// Checks that spectral products carry a wavelength per image channel and
// measurement products carry the detector profile/units, then stamps the
// portable metadata keys used by EXR/CLI/Qt exports.
[[nodiscard]] Result<void, String> AnnotateProductMetadata(CameraProduct& product);

} // namespace quantiloom::camera
