#include "postprocess/GenericSensor.hpp"

#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <algorithm>
#include <numbers>
#include <random>

namespace quantiloom {

struct GenericSensor::Impl {
    camera::CaptureState capture;
    u32 requestedSeed = 0;
    u32 effectiveSeed = 0;
    bool seeded = false;
};

GenericSensor::GenericSensor() : m_impl(std::make_unique<Impl>()) {}
GenericSensor::~GenericSensor() = default;

auto GenericSensor::Apply(const Image& hdr, const SensorParams& params)
    -> Result<SensorOutput, String> {
    if (!hdr.IsValid() || (hdr.channels != 1 && hdr.channels != 3 &&
                           hdr.channels != 4))
        return Result<SensorOutput, String>::Err(
            "GenericSensor needs a valid mono or linear RGB image");
    size_t displayCount = 0;
    if (!Image::TryElementCount(hdr.width, hdr.height, 3, displayCount))
        return Result<SensorOutput, String>::Err("sensor display exceeds image storage limits");
    if (!m_impl->seeded || m_impl->requestedSeed != params.noiseSeed) {
        m_impl->requestedSeed = params.noiseSeed;
        m_impl->effectiveSeed = params.noiseSeed != 0 ?
            params.noiseSeed : std::random_device{}();
        m_impl->capture = {};
        m_impl->seeded = true;
    }
    camera::CameraConfig config;
    config.enabled = true;
    config.device.id = "legacy_generic_photon";
    config.device.displayName = "Legacy generic photon sensor";
    config.device.detector = camera::DetectorKind::Photon;
    config.device.calibration = camera::CalibrationStatus::GenericAssumption;
    config.device.cfa = hdr.channels == 1 ? camera::CfaPattern::Mono :
                                          camera::CfaPattern::MultiChannel;
    config.optics.sensorWidthPx = hdr.width;
    config.optics.sensorHeightPx = hdr.height;
    config.optics.focalLengthMm = params.focalLength_mm;
    config.optics.fNumber = params.fNumber;
    config.optics.pixelPitchUm = params.pixelPitch_um;
    config.optics.psfSigmaPixelsOverride = params.psfSigma_px;
    config.optics.cosFourthVignetting =
        params.enableVignetting && !params.isTelecentric;
    if (config.optics.cosFourthVignetting && params.fov_deg > 0.0f) {
        const auto focal = camera::EffectiveFocalLengthMm(
            static_cast<f64>(params.fov_deg) * std::numbers::pi_v<f64> / 180.0,
            params.pixelPitch_um, hdr.width);
        if (!focal) return Result<SensorOutput, String>::Err(focal.error());
        config.optics.focalLengthMm = focal.value();
    }
    config.readout.exposureSeconds = params.integrationTime_s;
    config.readout.electronsPerDn = params.gain; // Old gain was e-/DN.
    config.readout.analogGain = 1.0;
    config.readout.adcBits = params.bitDepth;
    config.readout.outputBits = params.bitDepth;
    config.photon.fullWellElectrons = params.wellCapacity_e;
    config.photon.darkCurrentElectronsPerSecond = params.darkCurrent_e_s;
    config.photon.readNoiseElectronsRms = params.readNoise_e_rms;
    config.photon.prnuSigma = params.prnuSigma;
    config.photon.dsnuElectronsRms = params.dsnuSigma_e;
    config.photon.dsnuReferenceExposureSeconds = params.integrationTime_s;
    config.photon.enableShotNoise = params.enablePoissonNoise;
    config.photon.enableDarkShotNoise = params.enablePoissonNoise;
    config.photon.enableReadNoise = params.enableReadNoise;
    config.photon.enableDarkCurrent = params.enableDarkCurrent;
    config.photon.enableFpn = params.enableFPN;
    config.photon.applyNuc = params.enableNUC;
    config.photon.nucResidualFraction =
        std::clamp(1.0 - static_cast<f64>(params.nucEfficiency), 0.0, 1.0);
    config.randomSeed = m_impl->effectiveSeed;
    config.fastRgbRadianceScale = 1.0; // Explicit generic 1 nm flat surrogate.
    config.calibratedFastRgbInput = false;
    config.quality.wavelengthSamples = 2;
    config.quality.timeSamples = 1;
    config.quality.pixelSamples = 1;
    config.products.rawDn = true;
    config.products.display = true;
    config.products.bandMeasurement = false;
    config.products.correctedDeviceSignal = false;
    const f64 centerNm = params.wavelength_nm;
    for (u32 channel = 0; channel < (hdr.channels == 1 ? 1u : 3u); ++channel) {
        camera::ResponseCurve qe;
        qe.kind = camera::ResponseKind::AbsoluteQE;
        qe.wavelengthNm = {centerNm - 0.5, centerNm + 0.5};
        qe.value = {params.quantumEfficiency, params.quantumEfficiency};
        qe.source.parameterPath = "legacy.sensor.quantum_efficiency";
        qe.source.provenance = camera::ValueProvenance::Assumed;
        camera::ResponseStack response;
        response.quantumEfficiency = std::move(qe);
        config.device.channels.push_back({
            hdr.channels == 1 ? "Mono" : (channel == 0 ? "R" :
                                             channel == 1 ? "G" : "B"),
            std::move(response)});
    }
    if (params.enableNUC && params.enableFPN) {
        const size_t count = static_cast<size_t>(hdr.width) * hdr.height *
                             (hdr.channels == 1 ? 1u : 3u);
        config.photon.nucGainMap.resize(count);
        config.photon.nucOffsetElectronsMap.resize(count);
        const u32 calibrationSeed = camera::DeviceRandomSeed(config);
        for (size_t i = 0; i < count; ++i) {
            const f64 prnu = params.prnuSigma * camera::CounterGaussian(
                calibrationSeed, static_cast<u32>(i), 0,
                camera::NoiseClass::FixedPrnu);
            const f64 dsnu = params.dsnuSigma_e * camera::CounterGaussian(
                calibrationSeed, static_cast<u32>(i), 0,
                camera::NoiseClass::FixedDsnu);
            config.photon.nucGainMap[i] = 1.0 / std::max(0.01, 1.0 + prnu);
            config.photon.nucOffsetElectronsMap[i] = -dsnu;
        }
    }
    Image rgb(hdr.width, hdr.height, 3);
    for (u32 y = 0; y < hdr.height; ++y)
        for (u32 x = 0; x < hdr.width; ++x)
            for (u32 channel = 0; channel < 3; ++channel)
                rgb(x, y, channel) = hdr(x, y, hdr.channels == 1 ? 0u : channel);
    camera::CpuCameraPipeline pipeline(std::move(config));
    auto output = pipeline.CaptureFastRgb(
        m_impl->capture, static_cast<f64>(m_impl->capture.acquisitionIndex) *
                             params.integrationTime_s, rgb);
    if (!output) return Result<SensorOutput, String>::Err(output.error());
    if (!output.value().rawDn || !output.value().display)
        return Result<SensorOutput, String>::Err(
            "legacy camera adapter did not return RAW and preview");
    SensorOutput legacy;
    legacy.rawDN = std::move(output.value().rawDn->image);
    legacy.enhancedPreview = std::move(output.value().display->image);
    if (hdr.channels == 1) {
        Image gray(hdr.width, hdr.height, 1);
        for (u32 y = 0; y < hdr.height; ++y)
            for (u32 x = 0; x < hdr.width; ++x)
                gray(x, y, 0) = legacy.enhancedPreview(x, y, 0);
        gray.metadata = legacy.enhancedPreview.metadata;
        legacy.enhancedPreview = std::move(gray);
    }
    legacy.rawDN.metadata["camera_input_semantics"] = "fast_rgb_approximation";
    legacy.enhancedPreview.metadata["camera_input_semantics"] =
        "fast_rgb_approximation";
    return legacy;
}

} // namespace quantiloom
