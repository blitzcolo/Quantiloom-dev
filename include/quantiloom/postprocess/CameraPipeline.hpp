#pragma once

#include "../core/Image.hpp"
#include "../core/Types.hpp"
#include "../renderer/DisplayControl.hpp"

#include <array>
#include <limits>
#include <optional>
#include <vector>

namespace quantiloom::camera {

inline constexpr u32 kCameraConfigVersion = 1;

// Storage shape never defines physical meaning.
enum class SignalKind : u32 {
    SpectralRadiance, BandMeasurement, DeviceLinear, CieLinearSrgb,
    DisplaySrgb, RawDN, ApparentTemperature, FastRgbApproximation
};
enum class DetectorKind : u32 { Photon, Thermal };
enum class CalibrationStatus : u32 { GenericAssumption, HardwareReference, Calibrated };
enum class ValueProvenance : u32 { Manufacturer, Digitized, Derived, Assumed };
enum class ResponseKind : u32 {
    AbsoluteQE, RelativeQE, LensTransmission, FilterTransmission,
    ThermalAbsorptance, SystemPhotonQE, SystemThermalAbsorptance
};
enum class RelativeNormalization : u32 { None, PeakOne, AreaOne };
enum class ShutterKind : u32 { Global, Rolling };
enum class ProcessingBackend : u32 { CpuReference, GpuPreview };
enum class CfaPattern : u32 { Mono, RGGB, GRBG, GBRG, BGGR };
enum class OutputColorSpace : u32 { DeviceNative, CieLinearSrgb, DisplaySrgb };

struct ParameterSource {
    String parameterPath;
    ValueProvenance provenance = ValueProvenance::Assumed;
    String document;
    String documentVersion;
    String condition;
};

struct ResponseCurve {
    ResponseKind kind = ResponseKind::AbsoluteQE;
    RelativeNormalization normalization = RelativeNormalization::None;
    std::vector<f64> wavelengthNm;
    std::vector<f64> value;
    // RelativeQE PeakOne: dimensionless peak QE. AreaOne: QE*nm.
    // A relative curve requires both amplitude and amplitudeSource.
    f64 amplitude = 1.0;
    String amplitudeSource;
    // If measured QE/absorptance already includes active pixel area, the
    // OpticsConfig fillFactor must stay at one.
    bool includesPixelFillFactor = false;
    ParameterSource source;
    [[nodiscard]] f64 MinNm() const { return wavelengthNm.empty() ? 0.0 : wavelengthNm.front(); }
    [[nodiscard]] f64 MaxNm() const { return wavelengthNm.empty() ? 0.0 : wavelengthNm.back(); }
};

// A system response is mutually exclusive with component curves. Missing
// lens/filter means unit throughput; QE or absorptance is never optional.
struct ResponseStack {
    std::optional<ResponseCurve> lensTransmission;
    std::optional<ResponseCurve> filterTransmission;
    std::optional<ResponseCurve> quantumEfficiency;
    std::optional<ResponseCurve> thermalAbsorptance;
    std::optional<ResponseCurve> systemResponse;
};

struct ChannelProfile {
    String name;
    ResponseStack response;
};

struct OpticsConfig {
    f64 focalLengthMm = 50.0;
    f64 fNumber = 2.8;
    f64 pixelPitchUm = 5.0;
    // Only used when the QE/absorptance explicitly excludes the dead area.
    // A measured system response already includes it and must keep this at 1.
    f64 fillFactor = 1.0;
    u32 sensorWidthPx = 0;
    u32 sensorHeightPx = 0;
    f64 focusDistanceM = std::numeric_limits<f64>::infinity();
    bool cosFourthVignetting = false;
    String knownPsfPath;
};

struct ReadoutConfig {
    ShutterKind shutter = ShutterKind::Global;
    // Frame time is the first row's exposure midpoint.
    f64 exposureSeconds = 0.01;
    f64 rowDelaySeconds = 0.0;
    f64 framePeriodSeconds = 1.0 / 30.0;
    f64 analogGain = 1.0;
    f64 electronsPerDn = 1.0;
    f64 blackLevelDn = 0.0;
    u32 adcBits = 14;
    u32 outputBits = 14;
    f64 effectiveBits = 0.0; // 0 = undocumented, not inferred from outputBits.
};

struct PhotonDetectorConfig {
    f64 fullWellElectrons = 50000.0;
    f64 darkCurrentElectronsPerSecond = 0.0;
    f64 readNoiseElectronsRms = 0.0;
    f64 prnuSigma = 0.0;
    f64 dsnuElectronsRms = 0.0;
    f64 biasDnRms = 0.0;
    bool applyNuc = false;
    f64 nucResidualFraction = 0.0;
};

struct ThermalDetectorConfig {
    f64 timeConstantSeconds = 0.008;
    f64 responsivityDnPerWatt = 1.0;
    f64 readNoiseDnRms = 0.0;
    f64 driftDnPerSecond = 0.0;
    f64 readoutWindowSeconds = 0.0;
    f64 netdKelvin = 0.0;
    f64 netdReferenceTemperatureK = 0.0;
    f64 netdNoiseBandwidthHz = 0.0;
    String netdOpticalCondition;
};

struct DeviceProfile {
    String id;
    String displayName;
    String documentVersion;
    String documentUrl;
    CalibrationStatus calibration = CalibrationStatus::GenericAssumption;
    DetectorKind detector = DetectorKind::Photon;
    CfaPattern cfa = CfaPattern::Mono;
    std::vector<ChannelProfile> channels;
    f64 effectiveMinNm = 0.0;
    f64 effectiveMaxNm = 0.0;
    String readoutMode;
    std::vector<ParameterSource> parameterSources;
};

struct HsvConfig {
    f64 hueOffsetDegrees = 0.0;
    f64 saturationScale = 1.0;
    f64 valueGamma = 1.0;
    bool empiricalNoise = false;
    bool temporalDrift = false;
};

struct IspConfig {
    bool autoExposure = false;
    bool autoWhiteBalance = false;
    std::array<f64, 3> whiteBalance = {1.0, 1.0, 1.0};
    std::array<f64, 9> deviceToLinearSrgb = {1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0};
    bool denoise = false;
    f64 denoiseStrength = 0.0;
    bool sharpen = false;
    f64 sharpenStrength = 0.0;
    f64 toneGamma = 1.0;
    bool clipOutOfGamut = true;
    DisplayToneMode infraredTone = DisplayToneMode::Linear;
    DisplayPalette infraredPalette = DisplayPalette::Grey;
    f64 contrastLowPercentile = 1.0;
    f64 contrastHighPercentile = 99.0;
    HsvConfig hsv;
};

struct ProcessingQuality {
    ProcessingBackend backend = ProcessingBackend::CpuReference;
    u32 wavelengthSamples = 32;
    u32 timeSamples = 1;
    u32 pixelSamples = 1;
    u32 gpuTimePositions = 8;
};

struct CameraPoseKey {
    f64 timeSeconds = 0.0;
    std::array<f64, 3> position = {};
    std::array<f64, 3> lookAt = {};
    bool operator==(const CameraPoseKey&) const = default;
};
enum class MotionInterpolation : u32 { Linear };
enum class MotionExtrapolation : u32 { Hold };
struct CameraMotionConfig {
    std::vector<CameraPoseKey> keys;
    MotionInterpolation interpolation = MotionInterpolation::Linear;
    MotionExtrapolation extrapolation = MotionExtrapolation::Hold;
    bool operator==(const CameraMotionConfig&) const = default;
};

struct ProductRequest {
    bool tracedRadiance = false;
    bool cieLinearSrgb = false;
    bool bandMeasurement = false;
    bool rawDn = true;
    bool correctedDeviceSignal = false;
    bool apparentTemperature = false;
    bool display = true;
};

struct CameraConfig {
    u32 version = kCameraConfigVersion;
    DeviceProfile device;
    OpticsConfig optics;
    ReadoutConfig readout;
    PhotonDetectorConfig photon;
    ThermalDetectorConfig thermal;
    IspConfig isp;
    ProcessingQuality quality;
    CameraMotionConfig motion;
    ProductRequest products;
    OutputColorSpace outputColorSpace = OutputColorSpace::DisplaySrgb;
    u32 randomSeed = 0x548CU; // Same 32-bit counter key on CPU and GPU.
    bool calibratedFastRgbInput = false;
    f64 fastRgbRadianceScale = 0.0; // 0 = labelled generic approximation.
    // For a Mono detector only row 0 is used. This is an approximation,
    // never a substitute for the device's true spectral response.
    std::array<f64, 9> fastRgbToDevice = {1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0};
};

// Only explicit acquisition advances this snapshot. Rendering spp, GUI redraw
// and display-only processing never change acquisitionIndex.
struct CaptureState {
    u32 version = kCameraConfigVersion;
    u64 acquisitionIndex = 0;
    f64 frameTimeSeconds = 0.0;
    f64 nextExposureSeconds = 0.0;
    f64 nextAnalogGain = 1.0;
    std::array<f64, 3> nextWhiteBalance = {1.0, 1.0, 1.0};
    std::vector<f64> thermalPixelStateW;
};

struct SignalDescriptor {
    SignalKind kind = SignalKind::SpectralRadiance;
    String unit; // E.g. W/m^2/sr/nm, e-/s, W, DN, K, encoded sRGB.
    String responseProfileId;
    f64 responseMinNm = 0.0;
    f64 responseMaxNm = 0.0;
    // Optional per-channel response identity and span when a detector reads
    // several channels at each pixel. A Bayer RAW still has one per pixel;
    // the CFA enum maps that scalar to one of the device's R/G/B responses.
    std::vector<String> channelResponseIds;
    std::vector<std::array<f64, 2>> channelResponseSpanNm;
    // One wavelength per spectral cube channel. Empty for non-spectral data.
    std::vector<f64> channelWavelengthNm;
    CalibrationStatus calibration = CalibrationStatus::GenericAssumption;
    u32 algorithmVersion = kCameraConfigVersion;
    u64 acquisitionIndex = 0;
    f64 exposureStartSeconds = 0.0;
    f64 exposureEndSeconds = 0.0;
    // A CFA image has one scalar per pixel; this identifies the pattern.
    CfaPattern cfa = CfaPattern::Mono;
    u32 channelsPerPixel = 1;
};
struct CameraProduct { Image image; SignalDescriptor signal; };
struct CameraOutput {
    std::optional<CameraProduct> tracedRadiance;
    std::optional<CameraProduct> cieLinearSrgb;
    std::optional<CameraProduct> bandMeasurement;
    std::optional<CameraProduct> rawDn;
    std::optional<CameraProduct> correctedDeviceSignal;
    std::optional<CameraProduct> apparentTemperature;
    std::optional<CameraProduct> display;
};

// The only L_lambda -> E_lambda conversion belongs to the physics kernel.
// This irradiance is PRE-TRANSMISSION: the stack's lens/filter throughput has
// not been applied. The plan's E_lambda in its photon formula is the later
// detector-plane irradiance after those optical factors.
struct SpectralRadianceSample { f64 wavelengthNm; f64 radianceWm2SrNm; };
struct SpectralIrradianceSample { f64 wavelengthNm; f64 irradianceWm2Nm; };
struct PhotonMeasurement { f64 electronRatePerSecond = 0.0; };
struct ThermalMeasurement { f64 absorbedPowerW = 0.0; };

} // namespace quantiloom::camera
