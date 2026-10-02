#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CameraConfigIO.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace quantiloom::camera {
namespace {

template<class T> Result<T, String> Fail(const char* message) {
    return typename Result<T, String>::Err(String(message));
}

bool FiniteNonnegative(f64 x) { return std::isfinite(x) && x >= 0.0; }
bool FinitePositive(f64 x) { return std::isfinite(x) && x > 0.0; }

const ResponseCurve* BaseCurve(const ResponseStack& stack, DetectorKind detector) {
    if (stack.systemResponse) return &*stack.systemResponse;
    if (detector == DetectorKind::Photon && stack.quantumEfficiency)
        return &*stack.quantumEfficiency;
    if (detector == DetectorKind::Thermal && stack.thermalAbsorptance)
        return &*stack.thermalAbsorptance;
    return nullptr;
}

f64 CurveAt(const ResponseCurve& curve, f64 wavelengthNm) {
    if (wavelengthNm < curve.MinNm() || wavelengthNm > curve.MaxNm()) return 0.0;
    const auto hi = std::lower_bound(curve.wavelengthNm.begin(), curve.wavelengthNm.end(),
                                     wavelengthNm);
    if (hi == curve.wavelengthNm.begin()) return curve.amplitude * curve.value.front();
    if (hi == curve.wavelengthNm.end()) return curve.amplitude * curve.value.back();
    const auto index = static_cast<size_t>(hi - curve.wavelengthNm.begin());
    const f64 t = (wavelengthNm - curve.wavelengthNm[index - 1]) /
                  (curve.wavelengthNm[index] - curve.wavelengthNm[index - 1]);
    return curve.amplitude *
        (curve.value[index - 1] + t * (curve.value[index] - curve.value[index - 1]));
}

f64 EffectiveResponse(const ResponseStack& stack, DetectorKind detector, f64 wavelengthNm) {
    f64 response = CurveAt(*BaseCurve(stack, detector), wavelengthNm);
    if (stack.lensTransmission) response *= CurveAt(*stack.lensTransmission, wavelengthNm);
    if (stack.filterTransmission) response *= CurveAt(*stack.filterTransmission, wavelengthNm);
    return response;
}

template<class Sample> bool ValidSamples(std::span<const Sample> samples) {
    if (samples.size() < 2) return false;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!FinitePositive(samples[i].wavelengthNm)) return false;
        if (i && samples[i].wavelengthNm <= samples[i - 1].wavelengthNm) return false;
    }
    return true;
}

f64 IrradianceAt(std::span<const SpectralIrradianceSample> samples, f64 wavelengthNm) {
    const auto hi = std::lower_bound(samples.begin(), samples.end(), wavelengthNm,
        [](const SpectralIrradianceSample& item, f64 nm) { return item.wavelengthNm < nm; });
    if (hi == samples.begin()) return hi->irradianceWm2Nm;
    if (hi == samples.end()) return samples.back().irradianceWm2Nm;
    const auto& lo = *(hi - 1);
    const f64 t = (wavelengthNm - lo.wavelengthNm) / (hi->wavelengthNm - lo.wavelengthNm);
    return lo.irradianceWm2Nm + t * (hi->irradianceWm2Nm - lo.irradianceWm2Nm);
}

std::vector<f64> ResponseKnots(const ResponseStack& stack, DetectorKind detector) {
    const ResponseCurve& base = *BaseCurve(stack, detector);
    std::vector<f64> knots = base.wavelengthNm;
    const auto append = [&](const ResponseCurve& curve) {
        for (const f64 nm : curve.wavelengthNm)
            if (nm > base.MinNm() && nm < base.MaxNm()) knots.push_back(nm);
    };
    if (stack.lensTransmission) append(*stack.lensTransmission);
    if (stack.filterTransmission) append(*stack.filterTransmission);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    return knots;
}

std::vector<f64> IntegrationKnots(std::span<const SpectralIrradianceSample> samples,
                                  const ResponseStack& stack, DetectorKind detector) {
    const ResponseCurve& base = *BaseCurve(stack, detector);
    std::vector<f64> knots = ResponseKnots(stack, detector);
    for (const auto& sample : samples)
        if (sample.wavelengthNm > base.MinNm() && sample.wavelengthNm < base.MaxNm())
            knots.push_back(sample.wavelengthNm);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    return knots;
}

// Four-point Gauss-Legendre exactly integrates the product of four linear
// response/source interpolants and the extra wavelength in photon conversion.
template<class Density> f64 Gauss4(f64 a, f64 b, Density&& density) {
    constexpr f64 nodes[4] = {-0.8611363115940526, -0.3399810435848563,
                               0.3399810435848563,  0.8611363115940526};
    constexpr f64 weights[4] = {0.3478548451374539, 0.6521451548625461,
                                 0.6521451548625461, 0.3478548451374539};
    const f64 mid = 0.5 * (a + b), half = 0.5 * (b - a);
    f64 sum = 0.0;
    for (int i = 0; i < 4; ++i) sum += weights[i] * density(mid + half * nodes[i]);
    return half * sum;
}

Result<void, String> ValidateInput(std::span<const SpectralIrradianceSample> samples,
                                  const ResponseStack& stack, DetectorKind detector,
                                  f64 pixelAreaM2) {
    if (!FinitePositive(pixelAreaM2)) return Result<void, String>::Err("pixel area must be finite and positive");
    const auto responseOk = ValidateResponseStack(stack, detector);
    if (!responseOk) return responseOk;
    if (!ValidSamples(samples)) return Result<void, String>::Err("wavelength samples must be finite, positive and strictly increasing");
    for (const auto& sample : samples)
        if (!FiniteNonnegative(sample.irradianceWm2Nm))
            return Result<void, String>::Err("spectral irradiance must be finite and nonnegative");
    const ResponseCurve& base = *BaseCurve(stack, detector);
    if (samples.front().wavelengthNm > base.MinNm() ||
        samples.back().wavelengthNm < base.MaxNm())
        return Result<void, String>::Err("source spectrum does not cover the full detector response");
    return Result<void, String>::Ok();
}

u32 Mix32(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

} // namespace

Result<void, String> ValidateResponse(const ResponseCurve& response) {
    if (response.wavelengthNm.size() < 2 ||
        response.wavelengthNm.size() != response.value.size())
        return Result<void, String>::Err("response needs at least two matching wavelength and value samples");
    if (!FiniteNonnegative(response.amplitude))
        return Result<void, String>::Err("response amplitude must be finite and nonnegative");
    for (size_t i = 0; i < response.value.size(); ++i) {
        if (!FinitePositive(response.wavelengthNm[i]) ||
            (i && response.wavelengthNm[i] <= response.wavelengthNm[i - 1]))
            return Result<void, String>::Err("response wavelengths must be finite, positive and increasing");
        if (!FiniteNonnegative(response.value[i]))
            return Result<void, String>::Err("response values must be finite and nonnegative");
    }
    if (response.kind == ResponseKind::RelativeQE) {
        if (response.normalization == RelativeNormalization::None ||
            response.amplitudeSource.empty() || response.amplitude <= 0.0)
            return Result<void, String>::Err("relative QE needs a normalization, amplitude and amplitude source");
        if (response.normalization == RelativeNormalization::PeakOne) {
            if (std::abs(*std::max_element(response.value.begin(), response.value.end()) - 1.0) > 1e-6)
                return Result<void, String>::Err("PeakOne response must have unit peak");
        } else {
            f64 area = 0.0;
            for (size_t i = 1; i < response.value.size(); ++i)
                area += 0.5 * (response.value[i - 1] + response.value[i]) *
                        (response.wavelengthNm[i] - response.wavelengthNm[i - 1]);
            if (std::abs(area - 1.0) > 1e-6)
                return Result<void, String>::Err("AreaOne response must integrate to one per nm");
        }
    } else {
        if (response.normalization != RelativeNormalization::None || response.amplitude != 1.0)
            return Result<void, String>::Err("absolute responses cannot use relative normalization or amplitude");
    }
    for (size_t i = 0; i < response.value.size(); ++i)
        if (response.amplitude * response.value[i] > 1.0 + 1e-12)
            return Result<void, String>::Err("effective response exceeds one");
    return Result<void, String>::Ok();
}

Result<void, String> ValidateResponseStack(const ResponseStack& stack, DetectorKind detector) {
    if (stack.systemResponse) {
        if (stack.lensTransmission || stack.filterTransmission ||
            stack.quantumEfficiency || stack.thermalAbsorptance)
            return Result<void, String>::Err("system response cannot be multiplied by component responses");
        const auto expected = detector == DetectorKind::Photon ?
            ResponseKind::SystemPhotonQE : ResponseKind::SystemThermalAbsorptance;
        if (stack.systemResponse->kind != expected)
            return Result<void, String>::Err("system response does not match detector technology");
    } else if (detector == DetectorKind::Photon) {
        if (!stack.quantumEfficiency || stack.thermalAbsorptance)
            return Result<void, String>::Err("photon detector requires QE and forbids thermal absorptance");
        if (stack.quantumEfficiency->kind != ResponseKind::AbsoluteQE &&
            stack.quantumEfficiency->kind != ResponseKind::RelativeQE)
            return Result<void, String>::Err("photon detector response must be QE");
    } else {
        if (!stack.thermalAbsorptance || stack.quantumEfficiency)
            return Result<void, String>::Err("thermal detector requires absorptance and forbids QE");
        if (stack.thermalAbsorptance->kind != ResponseKind::ThermalAbsorptance)
            return Result<void, String>::Err("thermal detector response must be absorptance");
    }
    if (stack.lensTransmission && stack.lensTransmission->kind != ResponseKind::LensTransmission)
        return Result<void, String>::Err("lens response has wrong kind");
    if (stack.filterTransmission && stack.filterTransmission->kind != ResponseKind::FilterTransmission)
        return Result<void, String>::Err("filter response has wrong kind");
    const ResponseCurve* curves[] = {stack.lensTransmission ? &*stack.lensTransmission : nullptr,
        stack.filterTransmission ? &*stack.filterTransmission : nullptr,
        stack.quantumEfficiency ? &*stack.quantumEfficiency : nullptr,
        stack.thermalAbsorptance ? &*stack.thermalAbsorptance : nullptr,
        stack.systemResponse ? &*stack.systemResponse : nullptr};
    for (const auto* curve : curves) if (curve) {
        const auto valid = ValidateResponse(*curve);
        if (!valid) return valid;
    }
    const ResponseCurve& base = *BaseCurve(stack, detector);
    for (const auto* ancillary : {curves[0], curves[1]}) if (ancillary &&
        (ancillary->MinNm() > base.MinNm() || ancillary->MaxNm() < base.MaxNm()))
        return Result<void, String>::Err("lens/filter response does not cover detector response");
    return Result<void, String>::Ok();
}

Result<void, String> ValidateCameraConfig(const CameraConfig& config) {
    if (config.version != kCameraConfigVersion)
        return Result<void, String>::Err("unsupported camera config version");
    if (!FinitePositive(config.optics.focalLengthMm) ||
        !FinitePositive(config.optics.fNumber) ||
        !FinitePositive(config.optics.pixelPitchUm) ||
        !FinitePositive(config.optics.fillFactor) ||
        config.optics.fillFactor > 1.0 ||
        !std::isfinite(config.optics.psfSigmaPixelsOverride) ||
        (config.optics.psfSigmaPixelsOverride < 0.0 &&
         config.optics.psfSigmaPixelsOverride != -1.0) ||
        config.optics.sensorWidthPx == 0 || config.optics.sensorHeightPx == 0)
        return Result<void, String>::Err("invalid physical camera geometry");
    if (!FinitePositive(config.readout.exposureSeconds) ||
        !FinitePositive(config.readout.framePeriodSeconds) ||
        !FiniteNonnegative(config.readout.rowDelaySeconds) ||
        !FinitePositive(config.readout.analogGain) ||
        !FinitePositive(config.readout.electronsPerDn) ||
        !FiniteNonnegative(config.readout.blackLevelDn) ||
        config.readout.adcBits == 0 || config.readout.adcBits > 24 ||
        config.readout.outputBits < config.readout.adcBits || config.readout.outputBits > 32 ||
        !FiniteNonnegative(config.readout.effectiveBits) ||
        config.readout.effectiveBits > config.readout.adcBits)
        return Result<void, String>::Err("invalid readout or ADC parameters");
    if (config.device.channels.empty())
        return Result<void, String>::Err("camera needs at least one device channel");
    if (config.device.cfa == CfaPattern::Mono && config.device.channels.size() != 1)
        return Result<void, String>::Err("monochrome detector needs one response channel");
    if (config.device.cfa == CfaPattern::MultiChannel &&
        config.device.channels.size() < 2)
        return Result<void, String>::Err("multi-channel detector needs multiple responses");
    if (config.device.cfa != CfaPattern::Mono &&
        config.device.cfa != CfaPattern::MultiChannel &&
        config.device.channels.size() != 3)
        return Result<void, String>::Err("Bayer detector needs R, G and B response channels");
    if (config.device.channels.size() > std::numeric_limits<u32>::max())
        return Result<void, String>::Err("camera device channel count is too large");
    const u32 rawChannels = config.device.cfa == CfaPattern::MultiChannel ?
        static_cast<u32>(config.device.channels.size()) : 1u;
    size_t calibratedSamples = 0, displaySamples = 0;
    if (!Image::TryElementCount(config.optics.sensorWidthPx,
            config.optics.sensorHeightPx, rawChannels, calibratedSamples) ||
        !Image::TryElementCount(config.optics.sensorWidthPx,
            config.optics.sensorHeightPx, 3, displaySamples))
        return Result<void, String>::Err("camera sensor array exceeds image storage limits");
    for (const auto& channel : config.device.channels) {
        const auto valid = ValidateResponseStack(channel.response, config.device.detector);
        if (!valid) return valid;
        const auto* base = BaseCurve(channel.response, config.device.detector);
        if (config.optics.fillFactor != 1.0 &&
            (channel.response.systemResponse || base->includesPixelFillFactor))
            return Result<void, String>::Err("pixel fill factor is already included in detector response");
        if (config.device.effectiveMinNm > 0.0 &&
            (config.device.effectiveMinNm > base->MinNm() ||
             config.device.effectiveMaxNm < base->MaxNm()))
            return Result<void, String>::Err("device effective span truncates response curve");
    }
    if ((config.device.effectiveMinNm != 0.0 || config.device.effectiveMaxNm != 0.0) &&
        (!FinitePositive(config.device.effectiveMinNm) ||
         !FinitePositive(config.device.effectiveMaxNm) ||
         config.device.effectiveMinNm >= config.device.effectiveMaxNm))
        return Result<void, String>::Err("invalid device effective spectral span");
    if (config.quality.wavelengthSamples == 0 || config.quality.timeSamples == 0 ||
        config.quality.pixelSamples == 0 || config.quality.gpuTimePositions == 0)
        return Result<void, String>::Err("camera sample counts must be positive");
    if (!FiniteNonnegative(config.warmup.seconds))
        return Result<void, String>::Err("camera warmup seconds must be finite and nonnegative");

    const auto validCalibration = [calibratedSamples](
        const std::vector<f64>& values, bool mustBePositive) {
        if (!values.empty() && values.size() != calibratedSamples) return false;
        return std::all_of(values.begin(), values.end(), [mustBePositive](f64 value) {
            return std::isfinite(value) && (mustBePositive ? value > 0.0 : true);
        });
    };
    if (config.device.detector == DetectorKind::Photon) {
        if (!FinitePositive(config.photon.fullWellElectrons) ||
            !FiniteNonnegative(config.photon.darkCurrentElectronsPerSecond) ||
            !FiniteNonnegative(config.photon.readNoiseElectronsRms) ||
            !FiniteNonnegative(config.photon.prnuSigma) ||
            !FiniteNonnegative(config.photon.dsnuElectronsRms) ||
            !FinitePositive(config.photon.dsnuReferenceExposureSeconds) ||
            !FiniteNonnegative(config.photon.biasDnRms) ||
            !FiniteNonnegative(config.photon.nucResidualFraction))
            return Result<void, String>::Err("invalid photon detector parameters");
        if (!validCalibration(config.photon.nucGainMap, true) ||
            !validCalibration(config.photon.nucOffsetElectronsMap, false))
            return Result<void, String>::Err("photon NUC maps must match device pixels");
    } else {
        if (!FiniteNonnegative(config.thermal.timeConstantSeconds) ||
            !FinitePositive(config.thermal.responsivityDnPerWatt) ||
            !FiniteNonnegative(config.thermal.readNoiseDnRms) ||
            !FiniteNonnegative(config.thermal.driftDnPerSecond) ||
            !FiniteNonnegative(config.thermal.readoutWindowSeconds) ||
            !FiniteNonnegative(config.thermal.netdKelvin))
            return Result<void, String>::Err("invalid thermal detector parameters");
        if (config.thermal.netdKelvin > 0.0 &&
            (!FinitePositive(config.thermal.netdReferenceTemperatureK) ||
             !FinitePositive(config.thermal.netdNoiseBandwidthHz) ||
             config.thermal.netdOpticalCondition.empty()))
            return Result<void, String>::Err("NETD requires reference temperature, optical condition and bandwidth");
        if (config.thermal.netdKelvin > 0.0 &&
            config.thermal.readNoiseDnRms > 0.0)
            return Result<void, String>::Err("NETD and independent read noise would double-count one output noise");
        if (!validCalibration(config.thermal.nucGainMap, true) ||
            !validCalibration(config.thermal.nucOffsetDnMap, false))
            return Result<void, String>::Err("thermal NUC maps must match device pixels");
    }
    if (!FiniteNonnegative(config.fastRgbRadianceScale) ||
        (config.calibratedFastRgbInput && !FinitePositive(config.fastRgbRadianceScale)))
        return Result<void, String>::Err("calibrated fast RGB needs an absolute radiance scale");
    if (auto validMotion = ValidateCameraMotion(config.motion); !validMotion)
        return validMotion;
    if (config.isp.defectPixels.size() > 1) {
        std::vector<std::pair<u32, u32>> pixels;
        pixels.reserve(config.isp.defectPixels.size());
        for (const auto& pixel : config.isp.defectPixels)
            pixels.emplace_back(pixel[0], pixel[1]);
        std::sort(pixels.begin(), pixels.end());
        if (std::adjacent_find(pixels.begin(), pixels.end()) != pixels.end())
            return Result<void, String>::Err("defect pixel list has a duplicate entry");
    }
    for (const auto& [x, y] : config.isp.defectPixels) {
        if (x >= config.optics.sensorWidthPx || y >= config.optics.sensorHeightPx)
            return Result<void, String>::Err(
                "defect pixel lies outside the sensor array");
    }
    if (!std::isfinite(config.isp.contrastLowPercentile) ||
        !std::isfinite(config.isp.contrastHighPercentile) ||
        config.isp.contrastLowPercentile < 0.0 ||
        config.isp.contrastHighPercentile > 100.0 ||
        config.isp.contrastLowPercentile >= config.isp.contrastHighPercentile)
        return Result<void, String>::Err("AGC percentiles must be finite and satisfy 0 <= low < high <= 100");
    const auto& hsv = config.isp.hsv;
    if (!std::isfinite(hsv.hueOffsetDegrees) ||
        !std::isfinite(hsv.saturationScale) ||
        !std::isfinite(hsv.valueGamma) || hsv.valueGamma <= 0.0 ||
        !FiniteNonnegative(hsv.empiricalNoiseSigma) ||
        !FiniteNonnegative(hsv.temporalDriftSigma))
        return Result<void, String>::Err("invalid HSV effect parameters");
    const auto& autoControl = config.isp.autoControl;
    if (!FinitePositive(autoControl.targetLuminance) ||
        !std::isfinite(autoControl.smoothing) ||
        autoControl.smoothing < 0.0 || autoControl.smoothing > 1.0 ||
        !FinitePositive(autoControl.minExposureSeconds) ||
        !FinitePositive(autoControl.maxExposureSeconds) ||
        autoControl.minExposureSeconds > autoControl.maxExposureSeconds ||
        !FinitePositive(autoControl.maxGain) || autoControl.maxGain < 1.0)
        return Result<void, String>::Err("invalid auto-control parameters");
    return Result<void, String>::Ok();
}

Result<void, String> ValidateCameraMotion(const CameraMotionConfig& motion) {
    for (size_t i = 0; i < motion.keys.size(); ++i) {
        const auto& key = motion.keys[i];
        if (!std::isfinite(key.timeSeconds) ||
            (i && key.timeSeconds <= motion.keys[i - 1].timeSeconds))
            return Result<void, String>::Err("camera motion times must be finite and increasing");
        for (const f64 coord : key.position) if (!std::isfinite(coord))
            return Result<void, String>::Err("camera motion position must be finite");
        for (const f64 coord : key.lookAt) if (!std::isfinite(coord))
            return Result<void, String>::Err("camera motion target must be finite");
        f64 distanceSquared = 0.0;
        for (size_t axis = 0; axis < 3; ++axis) {
            const f64 delta = key.lookAt[axis] - key.position[axis];
            distanceSquared += delta * delta;
        }
        if (!FinitePositive(distanceSquared) || distanceSquared < 1e-24)
            return Result<void, String>::Err("camera motion target must differ from position");
        if (i) {
            const auto& previous = motion.keys[i - 1];
            f64 dot = 0.0, deltaNormSquared = 0.0, currentNormSquared = 0.0;
            f64 priorDirection[3], delta[3];
            for (size_t axis = 0; axis < 3; ++axis) {
                priorDirection[axis] = previous.lookAt[axis] - previous.position[axis];
                delta[axis] = (key.lookAt[axis] - key.position[axis]) - priorDirection[axis];
                dot += priorDirection[axis] * delta[axis];
                deltaNormSquared += delta[axis] * delta[axis];
                currentNormSquared += priorDirection[axis] * priorDirection[axis];
            }
            if (deltaNormSquared > 0.0) {
                const f64 zeroAt = -dot / deltaNormSquared;
                if (zeroAt >= 0.0 && zeroAt <= 1.0) {
                    f64 minimumNormSquared = 0.0;
                    for (size_t axis = 0; axis < 3; ++axis) {
                        const f64 component = priorDirection[axis] + zeroAt * delta[axis];
                        minimumNormSquared += component * component;
                    }
                    if (minimumNormSquared <= 1e-24 * std::max(1.0, currentNormSquared))
                        return Result<void, String>::Err(
                            "interpolated camera motion has a zero viewing direction");
                }
            }
        }
    }
    return Result<void, String>::Ok();
}

Result<CameraPoseKey, String> CameraPoseAt(const CameraMotionConfig& motion,
                                          f64 timeSeconds) {
    if (!std::isfinite(timeSeconds) || motion.keys.empty())
        return Fail<CameraPoseKey>("camera motion needs finite time and at least one key");
    if (auto valid = ValidateCameraMotion(motion); !valid)
        return Result<CameraPoseKey, String>::Err(valid.error());
    if (timeSeconds <= motion.keys.front().timeSeconds) return motion.keys.front();
    if (timeSeconds >= motion.keys.back().timeSeconds) return motion.keys.back();
    const auto high = std::upper_bound(motion.keys.begin(), motion.keys.end(), timeSeconds,
        [](f64 t, const CameraPoseKey& key) { return t < key.timeSeconds; });
    const auto& lo = *(high - 1);
    const f64 fraction = (timeSeconds - lo.timeSeconds) / (high->timeSeconds - lo.timeSeconds);
    CameraPoseKey pose;
    pose.timeSeconds = timeSeconds;
    f64 lengthSquared = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
        pose.position[axis] = lo.position[axis] +
                              fraction * (high->position[axis] - lo.position[axis]);
        pose.lookAt[axis] = lo.lookAt[axis] +
                            fraction * (high->lookAt[axis] - lo.lookAt[axis]);
        const f64 direction = pose.lookAt[axis] - pose.position[axis];
        lengthSquared += direction * direction;
    }
    if (!FinitePositive(lengthSquared) || lengthSquared < 1e-24)
        return Fail<CameraPoseKey>("interpolated camera direction is invalid");
    return pose;
}

Result<u64, String> CameraAcquisitionIndexAt(
    f64 firstTimeSeconds, f64 framePeriodSeconds, f64 sceneTimeSeconds) {
    if (!std::isfinite(firstTimeSeconds) || !FinitePositive(framePeriodSeconds) ||
        !std::isfinite(sceneTimeSeconds) || sceneTimeSeconds < firstTimeSeconds)
        return Result<u64, String>::Err("invalid camera acquisition grid or scene time");
    long double slots =
        (static_cast<long double>(sceneTimeSeconds) - firstTimeSeconds) /
        framePeriodSeconds;
    const long double nearest = std::round(slots);
    const long double tolerance =
        8.0L * std::numeric_limits<f64>::epsilon() *
        std::max(1.0L, std::abs(slots));
    if (std::abs(slots - nearest) <= tolerance) slots = nearest;
    if (slots >= static_cast<long double>(std::numeric_limits<u64>::max()))
        return Result<u64, String>::Err("camera acquisition index overflows");
    return static_cast<u64>(std::floor(slots));
}

Result<f64, String> CameraAcquisitionTimeAt(
    f64 firstTimeSeconds, f64 framePeriodSeconds, u64 acquisitionIndex) {
    if (!std::isfinite(firstTimeSeconds) || !FinitePositive(framePeriodSeconds))
        return Result<f64, String>::Err("invalid camera acquisition grid");
    const long double time = static_cast<long double>(firstTimeSeconds) +
        static_cast<long double>(framePeriodSeconds) * acquisitionIndex;
    if (time > std::numeric_limits<f64>::max() ||
        time < -std::numeric_limits<f64>::max())
        return Result<f64, String>::Err("camera acquisition time overflows");
    return static_cast<f64>(time);
}

Result<f64, String> ApertureSolidAngleSr(f64 fNumber) {
    if (!FinitePositive(fNumber)) return Fail<f64>("f-number must be finite and positive");
    return std::numbers::pi_v<f64> / (1.0 + 4.0 * fNumber * fNumber);
}

Result<f64, String> PixelCollectionAreaM2(const OpticsConfig& optics) {
    if (!FinitePositive(optics.pixelPitchUm) ||
        !FinitePositive(optics.fillFactor) || optics.fillFactor > 1.0)
        return Fail<f64>("pixel pitch and fill factor must be valid");
    const f64 pitchM = optics.pixelPitchUm * 1e-6;
    const f64 area = pitchM * pitchM * optics.fillFactor;
    if (!FinitePositive(area)) return Fail<f64>("pixel area overflowed");
    return area;
}

Result<std::vector<SpectralIrradianceSample>, String>
RadianceToIrradiance(std::span<const SpectralRadianceSample> radiance,
                     f64 fNumber, f64 fieldAngleRad, bool applyCosFourth) {
    if (!ValidSamples(radiance)) return Fail<std::vector<SpectralIrradianceSample>>(
        "radiance wavelengths must be finite, positive and increasing");
    if (!std::isfinite(fieldAngleRad) || std::abs(fieldAngleRad) >= std::numbers::pi_v<f64> / 2)
        return Fail<std::vector<SpectralIrradianceSample>>("field angle must be finite and inside the image hemisphere");
    const auto omega = ApertureSolidAngleSr(fNumber);
    if (!omega) return Fail<std::vector<SpectralIrradianceSample>>("invalid f-number");
    const f64 cosine = std::cos(fieldAngleRad);
    const f64 vignette = applyCosFourth ? cosine * cosine * cosine * cosine : 1.0;
    std::vector<SpectralIrradianceSample> result;
    result.reserve(radiance.size());
    for (const auto& sample : radiance) {
        if (!FiniteNonnegative(sample.radianceWm2SrNm))
            return Fail<std::vector<SpectralIrradianceSample>>("radiance must be finite and nonnegative");
        result.push_back({sample.wavelengthNm, sample.radianceWm2SrNm * omega.value() * vignette});
    }
    return result;
}

Result<f64, String> HorizontalFovRadians(f64 focalLengthMm, f64 pixelPitchUm, u32 widthPx) {
    if (!FinitePositive(focalLengthMm) || !FinitePositive(pixelPitchUm) || widthPx == 0)
        return Fail<f64>("focal length, pixel pitch and width must be positive");
    return 2.0 * std::atan(static_cast<f64>(widthPx) * pixelPitchUm * 1e-3 /
                           (2.0 * focalLengthMm));
}

Result<f64, String> EffectiveFocalLengthMm(f64 horizontalFovRad, f64 pixelPitchUm, u32 widthPx) {
    if (!std::isfinite(horizontalFovRad) || horizontalFovRad <= 0.0 ||
        horizontalFovRad >= std::numbers::pi_v<f64> ||
        !FinitePositive(pixelPitchUm) || widthPx == 0)
        return Fail<f64>("invalid horizontal FOV, pitch or array width");
    return static_cast<f64>(widthPx) * pixelPitchUm * 1e-3 /
           (2.0 * std::tan(horizontalFovRad / 2.0));
}

Result<PhotonMeasurement, String>
IntegratePhoton(std::span<const SpectralIrradianceSample> samples,
                const ResponseStack& response, f64 pixelAreaM2) {
    const auto valid = ValidateInput(samples, response, DetectorKind::Photon, pixelAreaM2);
    if (!valid) return Fail<PhotonMeasurement>(valid.error().c_str());
    const auto knots = IntegrationKnots(samples, response, DetectorKind::Photon);
    f64 rate = 0.0;
    for (size_t i = 1; i < knots.size(); ++i)
        rate += Gauss4(knots[i - 1], knots[i], [&](f64 nm) {
            return IrradianceAt(samples, nm) *
                   EffectiveResponse(response, DetectorKind::Photon, nm) *
                   (nm * 1e-9) / (kPlanckJs * kLightSpeedMps);
        });
    rate *= pixelAreaM2;
    if (!FiniteNonnegative(rate)) return Fail<PhotonMeasurement>("photon integral overflowed");
    return PhotonMeasurement{rate};
}

Result<ThermalMeasurement, String>
IntegrateThermal(std::span<const SpectralIrradianceSample> samples,
                 const ResponseStack& response, f64 pixelAreaM2) {
    const auto valid = ValidateInput(samples, response, DetectorKind::Thermal, pixelAreaM2);
    if (!valid) return Fail<ThermalMeasurement>(valid.error().c_str());
    const auto knots = IntegrationKnots(samples, response, DetectorKind::Thermal);
    f64 power = 0.0;
    for (size_t i = 1; i < knots.size(); ++i)
        power += Gauss4(knots[i - 1], knots[i], [&](f64 nm) {
            return IrradianceAt(samples, nm) *
                   EffectiveResponse(response, DetectorKind::Thermal, nm);
        });
    power *= pixelAreaM2;
    if (!FiniteNonnegative(power)) return Fail<ThermalMeasurement>("thermal integral overflowed");
    return ThermalMeasurement{power};
}

Result<f64, String> AiryIntensityNormalized(f64 radiusM, f64 wavelengthNm, f64 fNumber) {
    if (!FiniteNonnegative(radiusM) || !FinitePositive(wavelengthNm) || !FinitePositive(fNumber))
        return Fail<f64>("invalid Airy radius, wavelength or f-number");
    const f64 x = std::numbers::pi_v<f64> * radiusM / (wavelengthNm * 1e-9 * fNumber);
    if (x == 0.0) return 1.0;
    const f64 ratio = 2.0 * std::cyl_bessel_j(1.0, x) / x;
    return ratio * ratio;
}

Result<f64, String> AiryPsfPerSquareMeter(f64 radiusM, f64 wavelengthNm, f64 fNumber) {
    const auto intensity = AiryIntensityNormalized(radiusM, wavelengthNm, fNumber);
    if (!intensity) return Fail<f64>(intensity.error().c_str());
    const f64 scale = wavelengthNm * 1e-9 * fNumber;
    return intensity.value() * std::numbers::pi_v<f64> / (4.0 * scale * scale);
}

Result<f64, String> AiryPixelFraction(f64 centerXM, f64 centerYM, f64 pixelPitchM,
                                     f64 wavelengthNm, f64 fNumber, u32 samplesPerSide) {
    if (!std::isfinite(centerXM) || !std::isfinite(centerYM) ||
        !FinitePositive(pixelPitchM) || samplesPerSide == 0 || samplesPerSide > 1024)
        return Fail<f64>("invalid Airy pixel integration geometry");
    f64 sum = 0.0;
    for (u32 y = 0; y < samplesPerSide; ++y)
        for (u32 x = 0; x < samplesPerSide; ++x) {
            const f64 px = centerXM + ((x + 0.5) / samplesPerSide - 0.5) * pixelPitchM;
            const f64 py = centerYM + ((y + 0.5) / samplesPerSide - 0.5) * pixelPitchM;
            const auto density = AiryPsfPerSquareMeter(std::hypot(px, py), wavelengthNm, fNumber);
            if (!density) return Fail<f64>(density.error().c_str());
            sum += density.value();
        }
    return std::min(1.0, sum * pixelPitchM * pixelPitchM /
                        (static_cast<f64>(samplesPerSide) * samplesPerSide));
}

Result<f64, String> PlanckRadianceWm2SrNm(f64 wavelengthNm, f64 temperatureK) {
    if (!FinitePositive(wavelengthNm) || !FinitePositive(temperatureK))
        return Fail<f64>("Planck wavelength and temperature must be finite and positive");
    const f64 lambdaM = wavelengthNm * 1e-9;
    const f64 x = kPlanckJs * kLightSpeedMps / (lambdaM * kBoltzmannJPerK * temperatureK);
    if (x > 700.0) return 0.0;
    const f64 radiance = 2.0 * kPlanckJs * kLightSpeedMps * kLightSpeedMps /
                         (std::pow(lambdaM, 5.0) * std::expm1(x)) * 1e-9;
    if (!FiniteNonnegative(radiance)) return Fail<f64>("Planck radiance overflowed");
    return radiance;
}

Result<f64, String> PlanckDerivativeWm2SrNmPerK(f64 wavelengthNm, f64 temperatureK) {
    const auto radiance = PlanckRadianceWm2SrNm(wavelengthNm, temperatureK);
    if (!radiance) return Fail<f64>(radiance.error().c_str());
    if (radiance.value() == 0.0) return 0.0;
    const f64 x = kPlanckJs * kLightSpeedMps /
                  (wavelengthNm * 1e-9 * kBoltzmannJPerK * temperatureK);
    const f64 derivative = radiance.value() * (x / temperatureK) / (-std::expm1(-x));
    if (!FiniteNonnegative(derivative)) return Fail<f64>("Planck derivative overflowed");
    return derivative;
}

namespace {
Result<f64, String> IntegrateBlackbody(f64 temperatureK, const ResponseStack& response,
                                      DetectorKind detector, f64 fNumber,
                                      f64 pixelAreaM2, bool derivative) {
    const auto valid = ValidateResponseStack(response, detector);
    if (!valid) return Fail<f64>(valid.error().c_str());
    const auto omega = ApertureSolidAngleSr(fNumber);
    if (!omega || !FinitePositive(pixelAreaM2) || !FinitePositive(temperatureK))
        return Fail<f64>("invalid blackbody temperature, aperture or pixel area");
    const auto knots = ResponseKnots(response, detector);
    f64 total = 0.0;
    for (size_t i = 1; i < knots.size(); ++i) {
        const f64 start = knots[i - 1];
        const f64 end = knots[i];
        const u32 parts = static_cast<u32>(std::ceil((end - start) / 5.0));
        for (u32 part = 0; part < parts; ++part) {
            const f64 a = start + (end - start) * part / parts;
            const f64 b = start + (end - start) * (part + 1) / parts;
            total += Gauss4(a, b, [&](f64 nm) {
                const auto blackbody = derivative ?
                    PlanckDerivativeWm2SrNmPerK(nm, temperatureK) :
                    PlanckRadianceWm2SrNm(nm, temperatureK);
                if (!blackbody) return std::numeric_limits<f64>::quiet_NaN();
                const f64 photonFactor = detector == DetectorKind::Photon ?
                    nm * 1e-9 / (kPlanckJs * kLightSpeedMps) : 1.0;
                return blackbody.value() * EffectiveResponse(response, detector, nm) *
                       photonFactor;
            });
        }
    }
    total *= omega.value() * pixelAreaM2;
    if (!FiniteNonnegative(total)) return Fail<f64>("blackbody response integral overflowed");
    return total;
}
} // namespace

Result<PhotonMeasurement, String> IntegrateBlackbodyPhoton(f64 temperatureK,
    const ResponseStack& response, f64 fNumber, f64 pixelAreaM2) {
    const auto result = IntegrateBlackbody(temperatureK, response, DetectorKind::Photon,
                                           fNumber, pixelAreaM2, false);
    if (!result) return Fail<PhotonMeasurement>(result.error().c_str());
    return PhotonMeasurement{result.value()};
}

Result<ThermalMeasurement, String> IntegrateBlackbodyThermal(f64 temperatureK,
    const ResponseStack& response, f64 fNumber, f64 pixelAreaM2) {
    const auto result = IntegrateBlackbody(temperatureK, response, DetectorKind::Thermal,
                                           fNumber, pixelAreaM2, false);
    if (!result) return Fail<ThermalMeasurement>(result.error().c_str());
    return ThermalMeasurement{result.value()};
}

Result<f64, String> BlackbodyThermalDerivativeWPerK(f64 temperatureK,
    const ResponseStack& response, f64 fNumber, f64 pixelAreaM2) {
    return IntegrateBlackbody(temperatureK, response, DetectorKind::Thermal,
                              fNumber, pixelAreaM2, true);
}

Result<f64, String> StepThermalResponse(f64 previousW, f64 absorbedPowerW,
                                       f64 dtSeconds, f64 tauSeconds) {
    if (!FiniteNonnegative(previousW) || !FiniteNonnegative(absorbedPowerW) ||
        !FiniteNonnegative(dtSeconds) || !FiniteNonnegative(tauSeconds))
        return Fail<f64>("thermal response needs finite nonnegative state, power and times");
    if (dtSeconds == 0.0) return previousW;
    if (tauSeconds == 0.0) return absorbedPowerW;
    const f64 gain = -std::expm1(-dtSeconds / tauSeconds);
    return previousW + gain * (absorbedPowerW - previousW);
}

u32 CounterRandomU32(u32 deviceSeed, u32 pixelIndex, u64 acquisitionIndex,
                     NoiseClass noiseClass, u32 counter) {
    const bool fixed = noiseClass == NoiseClass::FixedPrnu ||
                       noiseClass == NoiseClass::FixedDsnu;
    const u32 lo = fixed ? 0U : static_cast<u32>(acquisitionIndex);
    const u32 hi = fixed ? 0U : static_cast<u32>(acquisitionIndex >> 32);
    u32 state = Mix32(deviceSeed ^ 0x9e3779b9U);
    state = Mix32(state ^ Mix32(pixelIndex + 0x85ebca6bU));
    state = Mix32(state ^ Mix32(lo + 0xc2b2ae35U));
    state = Mix32(state ^ Mix32(hi + 0x27d4eb2fU));
    state = Mix32(state ^ Mix32(static_cast<u32>(noiseClass) + 0x165667b1U));
    return Mix32(state ^ Mix32(counter + 0xd3a2646cU));
}

u32 DeviceRandomSeed(const CameraConfig& config) {
    u32 idHash = 2166136261u;
    for (unsigned char byte : config.device.id) {
        idHash ^= byte;
        idHash *= 16777619u;
    }
    return Mix32(config.randomSeed ^ idHash);
}

f64 CounterUniform01(u32 deviceSeed, u32 pixelIndex, u64 acquisitionIndex,
                     NoiseClass noiseClass, u32 counter) {
    return (static_cast<f64>(CounterRandomU32(deviceSeed, pixelIndex, acquisitionIndex,
                                               noiseClass, counter)) + 0.5) / 4294967296.0;
}

f64 CounterGaussian(u32 deviceSeed, u32 pixelIndex, u64 acquisitionIndex,
                    NoiseClass noiseClass) {
    const f64 u1 = CounterUniform01(deviceSeed, pixelIndex, acquisitionIndex,
                                    noiseClass, 0);
    const f64 u2 = CounterUniform01(deviceSeed, pixelIndex, acquisitionIndex,
                                    noiseClass, 1);
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * std::numbers::pi_v<f64> * u2);
}

Result<void, String> AnnotateProductMetadata(CameraProduct& product) {
    if (!product.image.IsValid())
        return Result<void, String>::Err("camera product image is invalid");
    const auto& signal = product.signal;
    if (signal.unit.empty())
        return Result<void, String>::Err("camera product requires an explicit unit");
    if (signal.algorithmVersion != kCameraConfigVersion)
        return Result<void, String>::Err("unsupported camera product algorithm version");
    if (!std::isfinite(signal.exposureStartSeconds) ||
        !std::isfinite(signal.exposureEndSeconds) ||
        signal.exposureStartSeconds > signal.exposureEndSeconds)
        return Result<void, String>::Err("invalid camera product exposure window");
    if (signal.kind == SignalKind::SpectralRadiance) {
        if (signal.unit != "W/m^2/sr/nm" ||
            signal.channelWavelengthNm.size() != product.image.channels)
            return Result<void, String>::Err("spectral radiance needs units and one wavelength per channel");
        for (size_t i = 0; i < signal.channelWavelengthNm.size(); ++i)
            if (!FinitePositive(signal.channelWavelengthNm[i]) ||
                (i && signal.channelWavelengthNm[i] <= signal.channelWavelengthNm[i - 1]))
                return Result<void, String>::Err("spectral channel wavelengths must increase");
    } else if (!signal.channelWavelengthNm.empty()) {
        return Result<void, String>::Err("non-spectral product cannot claim spectral channel coordinates");
    }
    if (signal.kind == SignalKind::BandMeasurement) {
        if (signal.unit != "e-/s" && signal.unit != "W")
            return Result<void, String>::Err("band measurement must carry e-/s or W units");
        if (signal.responseProfileId.empty() ||
            !FinitePositive(signal.responseMinNm) ||
            !FinitePositive(signal.responseMaxNm) ||
            signal.responseMinNm >= signal.responseMaxNm ||
            signal.channelsPerPixel == 0 ||
            product.image.channels != signal.channelsPerPixel)
            return Result<void, String>::Err("band measurement needs response span and declared channel count");
        if ((!signal.channelResponseIds.empty() &&
             signal.channelResponseIds.size() != signal.channelsPerPixel) ||
            (!signal.channelResponseSpanNm.empty() &&
             signal.channelResponseSpanNm.size() != signal.channelsPerPixel))
            return Result<void, String>::Err("band measurement channel response metadata has wrong count");
        for (const auto& span : signal.channelResponseSpanNm)
            if (!FinitePositive(span[0]) || !FinitePositive(span[1]) ||
                span[0] >= span[1])
                return Result<void, String>::Err("band measurement channel response span is invalid");
    }
    if (signal.kind == SignalKind::RawDN && signal.unit != "DN")
        return Result<void, String>::Err("RAW product must use DN units");
    if (signal.kind == SignalKind::ApparentTemperature && signal.unit != "K")
        return Result<void, String>::Err("temperature product must use kelvin");
    String kind;
    switch (signal.kind) {
    case SignalKind::SpectralRadiance: kind = "spectral_radiance"; break;
    case SignalKind::BandMeasurement: kind = "band_measurement"; break;
    case SignalKind::DeviceLinear: kind = "device_linear"; break;
    case SignalKind::CieLinearSrgb: kind = "cie_linear_srgb"; break;
    case SignalKind::DisplaySrgb: kind = "display_srgb"; break;
    case SignalKind::RawDN: kind = "raw_dn"; break;
    case SignalKind::ApparentTemperature: kind = "apparent_temperature"; break;
    case SignalKind::FastRgbApproximation: kind = "fast_rgb_approximation"; break;
    }
    auto& metadata = product.image.metadata;
    metadata["camera_signal_kind"] = kind;
    metadata["camera_unit"] = signal.unit;
    metadata["camera_profile_id"] = signal.responseProfileId;
    metadata["camera_response_min_nm"] = std::to_string(signal.responseMinNm);
    metadata["camera_response_max_nm"] = std::to_string(signal.responseMaxNm);
    metadata["camera_calibration_status"] = std::to_string(static_cast<u32>(signal.calibration));
    metadata["camera_algorithm_version"] = std::to_string(signal.algorithmVersion);
    metadata["camera_acquisition_index"] = std::to_string(signal.acquisitionIndex);
    metadata["camera_exposure_start_s"] = std::to_string(signal.exposureStartSeconds);
    metadata["camera_exposure_end_s"] = std::to_string(signal.exposureEndSeconds);
    metadata["camera_cfa"] = std::to_string(static_cast<u32>(signal.cfa));
    metadata["camera_channels_per_pixel"] = std::to_string(signal.channelsPerPixel);
    // EXR sorts channels by name on read. Store each coordinate with its
    // original name, so consumers can use Image::ChannelIndex(name) after
    // round-trip instead of assuming the original interleaved index survived.
    if (!signal.channelWavelengthNm.empty() ||
        !signal.channelResponseIds.empty() ||
        !signal.channelResponseSpanNm.empty()) {
        if (product.image.channelNames.size() != product.image.channels)
            return Result<void, String>::Err("camera channel metadata requires channel names");
        for (u32 i = 0; i < product.image.channels; ++i) {
            const auto& name = product.image.channelNames[i];
            if (name.empty() ||
                std::find(product.image.channelNames.begin(),
                          product.image.channelNames.begin() + i, name) !=
                    product.image.channelNames.begin() + i)
                return Result<void, String>::Err("camera channel names must be nonempty and unique");
            const String prefix = "camera_channel_" + std::to_string(i) + "_";
            metadata[prefix + "name"] = name;
            if (!signal.channelWavelengthNm.empty())
                metadata[prefix + "wavelength_nm"] =
                    std::to_string(signal.channelWavelengthNm[i]);
            if (!signal.channelResponseIds.empty())
                metadata[prefix + "response_id"] = signal.channelResponseIds[i];
            if (!signal.channelResponseSpanNm.empty()) {
                metadata[prefix + "response_min_nm"] =
                    std::to_string(signal.channelResponseSpanNm[i][0]);
                metadata[prefix + "response_max_nm"] =
                    std::to_string(signal.channelResponseSpanNm[i][1]);
            }
        }
    }
    return Result<void, String>::Ok();
}

} // namespace quantiloom::camera
