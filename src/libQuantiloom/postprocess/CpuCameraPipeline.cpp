#include "postprocess/CpuCameraPipeline.hpp"

#include "io/ImageIO.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>

namespace quantiloom::camera {
namespace {

template<class T> Result<T, String> Fail(const String& message) {
    return typename Result<T, String>::Err(message);
}

const ResponseCurve& DetectorCurve(const ResponseStack& stack, DetectorKind kind) {
    if (stack.systemResponse) return *stack.systemResponse;
    return kind == DetectorKind::Photon ? *stack.quantumEfficiency :
                                         *stack.thermalAbsorptance;
}

u32 ChannelAt(CfaPattern cfa, u32 x, u32 y) {
    const bool px = (x & 1u) != 0, py = (y & 1u) != 0;
    switch (cfa) {
    case CfaPattern::RGGB: return !py ? (px ? 1u : 0u) : (px ? 2u : 1u);
    case CfaPattern::GRBG: return !py ? (px ? 0u : 1u) : (px ? 1u : 2u);
    case CfaPattern::GBRG: return !py ? (px ? 2u : 1u) : (px ? 1u : 0u);
    case CfaPattern::BGGR: return !py ? (px ? 1u : 2u) : (px ? 0u : 1u);
    default: return 0u;
    }
}

u32 OutputChannels(const CameraConfig& config) {
    return config.device.cfa == CfaPattern::MultiChannel ?
           static_cast<u32>(config.device.channels.size()) : 1u;
}

std::vector<f64> WavelengthGrid(const CameraConfig& config) {
    std::vector<f64> grid;
    for (const auto& channel : config.device.channels) {
        const ResponseStack& stack = channel.response;
        const auto& detector = DetectorCurve(stack, config.device.detector);
        grid.insert(grid.end(), detector.wavelengthNm.begin(), detector.wavelengthNm.end());
        for (const auto* curve : {stack.lensTransmission ? &*stack.lensTransmission : nullptr,
                                  stack.filterTransmission ? &*stack.filterTransmission : nullptr}) {
            if (curve) {
                for (f64 nm : curve->wavelengthNm)
                    if (nm > detector.MinNm() && nm < detector.MaxNm())
                        grid.push_back(nm);
            }
        }
        // Incrementing quality densifies each actual response span, avoiding
        // wasted samples in gaps between unrelated visible and IR channels.
        for (u32 i = 0; i <= config.quality.wavelengthSamples; ++i)
            grid.push_back(detector.MinNm() +
                           (detector.MaxNm() - detector.MinNm()) *
                           static_cast<f64>(i) / config.quality.wavelengthSamples);
    }
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    return grid;
}

struct Kernel {
    u32 radius = 0;
    std::vector<f64> weight;
};

Result<Kernel, String> BuildKernel(const CameraConfig& config, f64 wavelengthNm,
                                  const std::optional<Image>& knownPsf) {
    Kernel kernel;
    if (knownPsf) {
        const Image& image = *knownPsf;
        if (!image.IsValid() || image.width != image.height ||
            image.width % 2u == 0u || image.channels == 0)
            return Fail<Kernel>("known PSF must be a valid odd square image");
        kernel.radius = image.width / 2u;
        kernel.weight.resize(static_cast<size_t>(image.width) * image.height);
        f64 sum = 0.0;
        for (u32 y = 0; y < image.height; ++y)
            for (u32 x = 0; x < image.width; ++x) {
                const f64 value = image(x, y, 0);
                if (!std::isfinite(value) || value < 0.0)
                    return Fail<Kernel>("known PSF weights must be finite and nonnegative");
                kernel.weight[static_cast<size_t>(y) * image.width + x] = value;
                sum += value;
            }
        if (!(sum > 0.0) || !std::isfinite(sum))
            return Fail<Kernel>("known PSF has no finite energy");
        for (auto& weight : kernel.weight) weight /= sum;
        return kernel;
    }
    if (config.optics.psfSigmaPixelsOverride >= 0.0) {
        const f64 sigma = config.optics.psfSigmaPixelsOverride;
        if (sigma < 0.1) {
            kernel.radius = 0;
            kernel.weight = {1.0};
            return kernel;
        }
        if (!std::isfinite(sigma) || sigma > 64.0)
            return Fail<Kernel>("Gaussian PSF override is out of range");
        kernel.radius = static_cast<u32>(std::ceil(3.0 * sigma));
        const u32 side = kernel.radius * 2u + 1u;
        kernel.weight.resize(static_cast<size_t>(side) * side);
        f64 sum = 0.0;
        for (u32 y = 0; y < side; ++y)
            for (u32 x = 0; x < side; ++x) {
                const f64 dx = static_cast<i32>(x) - static_cast<i32>(kernel.radius);
                const f64 dy = static_cast<i32>(y) - static_cast<i32>(kernel.radius);
                const f64 weight = std::exp(-0.5 * (dx * dx + dy * dy) /
                                             (sigma * sigma));
                kernel.weight[static_cast<size_t>(y) * side + x] = weight;
                sum += weight;
            }
        for (auto& weight : kernel.weight) weight /= sum;
        return kernel;
    }
    const f64 pitchM = config.optics.pixelPitchUm * 1e-6;
    const f64 scalePx = wavelengthNm * 1e-9 * config.optics.fNumber / pitchM;
    const f64 requested = std::ceil(8.0 * scalePx *
                                     std::sqrt(static_cast<f64>(config.quality.pixelSamples)));
    if (!std::isfinite(requested) || requested > 1024.0)
        return Fail<Kernel>("Airy PSF radius exceeds CPU reference limit");
    kernel.radius = static_cast<u32>(std::max(1.0, requested));
    const u32 side = kernel.radius * 2u + 1u;
    kernel.weight.resize(static_cast<size_t>(side) * side);
    f64 sum = 0.0;
    for (u32 y = 0; y < side; ++y)
        for (u32 x = 0; x < side; ++x) {
            const f64 dx = (static_cast<i32>(x) - static_cast<i32>(kernel.radius)) * pitchM;
            const f64 dy = (static_cast<i32>(y) - static_cast<i32>(kernel.radius)) * pitchM;
            const auto fraction = AiryPixelFraction(
                dx, dy, pitchM, wavelengthNm, config.optics.fNumber,
                std::max(2u, config.quality.pixelSamples));
            if (!fraction) return Fail<Kernel>(fraction.error());
            const f64 weight = fraction.value();
            kernel.weight[static_cast<size_t>(y) * side + x] = weight;
            sum += weight;
        }
    if (!(sum > 0.0) || !std::isfinite(sum))
        return Fail<Kernel>("Airy PSF has no finite energy");
    for (auto& weight : kernel.weight) weight /= sum;
    return kernel;
}

Image Convolve(const Image& image, const Kernel& kernel) {
    Image result(image.width, image.height, 1);
    const i32 radius = static_cast<i32>(kernel.radius);
    const i32 side = radius * 2 + 1;
    for (u32 y = 0; y < image.height; ++y)
        for (u32 x = 0; x < image.width; ++x) {
            f64 sum = 0.0, normalizer = 0.0;
            for (i32 ky = -radius; ky <= radius; ++ky) {
                const i32 sy = static_cast<i32>(y) + ky;
                if (sy < 0 || sy >= static_cast<i32>(image.height)) continue;
                for (i32 kx = -radius; kx <= radius; ++kx) {
                    const i32 sx = static_cast<i32>(x) + kx;
                    if (sx < 0 || sx >= static_cast<i32>(image.width)) continue;
                    const f64 weight = kernel.weight[
                        static_cast<size_t>(ky + radius) * side + (kx + radius)];
                    sum += weight * image(static_cast<u32>(sx), static_cast<u32>(sy), 0);
                    normalizer += weight;
                }
            }
            result(x, y, 0) = static_cast<f32>(sum / normalizer);
        }
    return result;
}

Result<std::vector<std::vector<f64>>, String>
MeasurementWeights(const CameraConfig& config, const std::vector<f64>& wavelengths) {
    const auto area = PixelCollectionAreaM2(config.optics);
    if (!area) return Fail<std::vector<std::vector<f64>>>(area.error());
    std::vector<std::vector<f64>> weights(
        config.device.channels.size(), std::vector<f64>(wavelengths.size(), 0.0));
    std::vector<SpectralIrradianceSample> basis(wavelengths.size());
    for (size_t i = 0; i < wavelengths.size(); ++i)
        basis[i] = {wavelengths[i], 0.0};
    for (size_t channel = 0; channel < config.device.channels.size(); ++channel)
        for (size_t sample = 0; sample < wavelengths.size(); ++sample) {
            basis[sample].irradianceWm2Nm = 1.0;
            if (config.device.detector == DetectorKind::Photon) {
                const auto measured = IntegratePhoton(
                    basis, config.device.channels[channel].response, area.value());
                if (!measured) return Fail<std::vector<std::vector<f64>>>(measured.error());
                weights[channel][sample] = measured.value().electronRatePerSecond;
            } else {
                const auto measured = IntegrateThermal(
                    basis, config.device.channels[channel].response, area.value());
                if (!measured) return Fail<std::vector<std::vector<f64>>>(measured.error());
                weights[channel][sample] = measured.value().absorbedPowerW;
            }
            basis[sample].irradianceWm2Nm = 0.0;
        }
    return weights;
}

f64 FixedGaussian(u32 seed, u32 pixel, NoiseClass kind) {
    return CounterGaussian(seed, pixel, 0, kind);
}

f64 TemporalGaussian(u32 seed, u32 pixel, u64 capture, NoiseClass kind) {
    return CounterGaussian(seed, pixel, capture, kind);
}

f64 Poisson(f64 mean, u32 seed, u32 pixel, u64 capture, NoiseClass kind) {
    if (mean <= 0.0) return 0.0;
    // std::poisson_distribution is a true Poisson sampler. Its algorithm may
    // differ across standard libraries; the counter seed and statistics do not.
    std::mt19937 engine(CounterRandomU32(seed, pixel, capture, kind, 0));
    std::poisson_distribution<u64> distribution(mean);
    return static_cast<f64>(distribution(engine));
}

f64 Quantize(f64 analogDn, u32 bits) {
    const f64 upper = std::ldexp(1.0, static_cast<int>(bits)) - 1.0;
    return std::clamp(std::floor(analogDn + 0.5), 0.0, upper);
}

f32 EncodeSrgb(f32 linear) {
    const f32 clamped = std::clamp(linear, 0.0f, 1.0f);
    return clamped <= 0.0031308f ? 12.92f * clamped :
           1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

String Precise(f64 value) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(17) << value;
    return out.str();
}

SignalDescriptor Descriptor(const CameraConfig& config, SignalKind kind,
                            const String& unit, u64 acquisitionIndex,
                            f64 frameTimeSeconds) {
    SignalDescriptor signal;
    signal.kind = kind;
    signal.unit = unit;
    signal.responseProfileId = config.device.id.empty() ? "generic" : config.device.id;
    signal.calibration = config.device.calibration;
    signal.acquisitionIndex = acquisitionIndex;
    signal.exposureStartSeconds = frameTimeSeconds - config.readout.exposureSeconds / 2.0;
    signal.exposureEndSeconds = frameTimeSeconds + config.readout.exposureSeconds / 2.0 +
        (config.readout.shutter == ShutterKind::Rolling ?
         (config.optics.sensorHeightPx - 1) * config.readout.rowDelaySeconds : 0.0);
    signal.cfa = config.device.cfa;
    signal.channelsPerPixel = OutputChannels(config);
    signal.responseMinNm = std::numeric_limits<f64>::max();
    signal.responseMaxNm = 0.0;
    for (const auto& channel : config.device.channels) {
        const auto& response = DetectorCurve(channel.response, config.device.detector);
        signal.responseMinNm = std::min(signal.responseMinNm, response.MinNm());
        signal.responseMaxNm = std::max(signal.responseMaxNm, response.MaxNm());
    }
    if (config.device.cfa == CfaPattern::MultiChannel)
        for (const auto& channel : config.device.channels) {
            const auto& response = DetectorCurve(channel.response, config.device.detector);
            signal.channelResponseIds.push_back(channel.name);
            signal.channelResponseSpanNm.push_back({response.MinNm(), response.MaxNm()});
        }
    return signal;
}

Result<CameraProduct, String> Product(Image image, SignalDescriptor signal) {
    CameraProduct product{std::move(image), std::move(signal)};
    const auto annotated = AnnotateProductMetadata(product);
    if (!annotated) return Fail<CameraProduct>(annotated.error());
    return product;
}

Result<f64, String> ApparentTemperature(f64 measurement, const CameraConfig& config,
                                        u32 channel, f64 pixelAreaM2) {
    const auto& response = config.device.channels[channel].response;
    f64 lo = 10.0, hi = 5000.0;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const f64 middle = (lo + hi) / 2.0;
        f64 reference = 0.0;
        if (config.device.detector == DetectorKind::Photon) {
            const auto rate = IntegrateBlackbodyPhoton(
                middle, response, config.optics.fNumber, pixelAreaM2);
            if (!rate) return Fail<f64>(rate.error());
            reference = rate.value().electronRatePerSecond;
        } else {
            const auto power = IntegrateBlackbodyThermal(
                middle, response, config.optics.fNumber, pixelAreaM2);
            if (!power) return Fail<f64>(power.error());
            reference = power.value().absorbedPowerW;
        }
        if (reference < measurement) lo = middle;
        else hi = middle;
    }
    return (lo + hi) / 2.0;
}

} // namespace

CpuCameraPipeline::CpuCameraPipeline(CameraConfig config) : m_config(std::move(config)) {}

Result<CameraOutput, String>
CpuCameraPipeline::Capture(CaptureState& state, f64 firstRowMidpointSeconds,
                           const SpectralFrameSampler& sampler) const {
    if (m_config.products.tracedRadiance || m_config.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(m_config);
    if (!valid) return Fail<CameraOutput>(valid.error());
    if (!std::isfinite(firstRowMidpointSeconds) || !sampler)
        return Fail<CameraOutput>("camera capture needs finite frame time and a sampler");
    const u32 width = m_config.optics.sensorWidthPx;
    const u32 height = m_config.optics.sensorHeightPx;
    const u32 channels = OutputChannels(m_config);
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t elementCount = pixelCount * channels;
    const auto wavelengths = WavelengthGrid(m_config);
    const auto weights = MeasurementWeights(m_config, wavelengths);
    if (!weights) return Fail<CameraOutput>(weights.error());
    const auto omega = ApertureSolidAngleSr(m_config.optics.fNumber);
    const auto area = PixelCollectionAreaM2(m_config.optics);
    if (!omega || !area) return Fail<CameraOutput>("invalid camera aperture or pixel area");
    std::optional<Image> knownPsf;
    if (!m_config.optics.knownPsfPath.empty()) {
        knownPsf = ImageIO::ReadImage(m_config.optics.knownPsfPath);
        if (!knownPsf) return Fail<CameraOutput>("could not load known PSF image");
    }
    std::vector<f64> vignette(pixelCount, 1.0);
    if (m_config.optics.cosFourthVignetting) {
        const f64 pitchM = m_config.optics.pixelPitchUm * 1e-6;
        const f64 focalM = m_config.optics.focalLengthMm * 1e-3;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const f64 dx = (x + 0.5 - width * 0.5) * pitchM;
                const f64 dy = (y + 0.5 - height * 0.5) * pitchM;
                const f64 cosine = 1.0 / std::sqrt(1.0 + (dx * dx + dy * dy) /
                                                            (focalM * focalM));
                vignette[static_cast<size_t>(y) * width + x] = std::pow(cosine, 4.0);
            }
    }
    std::vector<f64> expected(elementCount, 0.0);
    const bool rolling = m_config.readout.shutter == ShutterKind::Rolling;
    const u32 rowGroups = rolling ? height : 1u;
    struct ExposureEvent { f64 time; u32 firstY; u32 lastY; };
    std::vector<ExposureEvent> events;
    events.reserve(static_cast<size_t>(rowGroups) * m_config.quality.timeSamples);
    for (u32 group = 0; group < rowGroups; ++group) {
        for (u32 timeSample = 0; timeSample < m_config.quality.timeSamples; ++timeSample) {
            const f64 time = firstRowMidpointSeconds +
                (rolling ? group * m_config.readout.rowDelaySeconds : 0.0) -
                m_config.readout.exposureSeconds / 2.0 +
                (timeSample + 0.5) * m_config.readout.exposureSeconds /
                    m_config.quality.timeSamples;
            events.push_back({time, rolling ? group : 0u,
                              rolling ? group + 1u : height});
        }
    }
    std::stable_sort(events.begin(), events.end(),
                     [](const ExposureEvent& a, const ExposureEvent& b) {
                         return a.time < b.time;
                     });
    for (const auto& event : events) {
            for (size_t wavelengthIndex = 0; wavelengthIndex < wavelengths.size();
                 ++wavelengthIndex) {
                auto sampled = sampler(event.time, wavelengths[wavelengthIndex]);
                if (!sampled) return Fail<CameraOutput>(sampled.error());
                const Image& radiance = sampled.value();
                if (!radiance.IsValid() || radiance.width != width ||
                    radiance.height != height || radiance.channels != 1)
                    return Fail<CameraOutput>("spectral sampler must return one correctly sized radiance channel");
                for (const f32 value : radiance.data)
                    if (!std::isfinite(value))
                        return Fail<CameraOutput>("spectral sampler returned nonfinite radiance");
                const auto kernel = BuildKernel(m_config, wavelengths[wavelengthIndex], knownPsf);
                if (!kernel) return Fail<CameraOutput>(kernel.error());
                const Image blurred = Convolve(radiance, kernel.value());
                for (u32 y = event.firstY; y < event.lastY; ++y)
                    for (u32 x = 0; x < width; ++x) {
                        const size_t pixel = static_cast<size_t>(y) * width + x;
                        const f64 irradiance = blurred(x, y, 0) * omega.value() * vignette[pixel];
                        if (m_config.device.cfa == CfaPattern::MultiChannel) {
                            for (u32 channel = 0; channel < channels; ++channel)
                                expected[pixel * channels + channel] += irradiance *
                                    weights.value()[channel][wavelengthIndex] /
                                    m_config.quality.timeSamples;
                        } else {
                            const u32 channel = ChannelAt(m_config.device.cfa, x, y);
                            expected[pixel] += irradiance *
                                weights.value()[channel][wavelengthIndex] /
                                m_config.quality.timeSamples;
                        }
                    }
            }
    }
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   false, true);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureMeasured(CaptureState& state, f64 firstRowMidpointSeconds,
                                   const Image& measuredRate) const {
    if (m_config.products.tracedRadiance || m_config.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(m_config);
    if (!valid) return Fail<CameraOutput>(valid.error());
    if (!std::isfinite(firstRowMidpointSeconds) || !measuredRate.IsValid() ||
        measuredRate.width != m_config.optics.sensorWidthPx ||
        measuredRate.height != m_config.optics.sensorHeightPx ||
        measuredRate.channels != OutputChannels(m_config))
        return Fail<CameraOutput>("measured-input image has invalid time, shape or channels");
    std::vector<f64> expected(measuredRate.data.begin(), measuredRate.data.end());
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   false, false);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureFastRgb(CaptureState& state, f64 firstRowMidpointSeconds,
                                  const Image& linearRgb) const {
    if (m_config.products.tracedRadiance || m_config.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(m_config);
    if (!valid) return Fail<CameraOutput>(valid.error());
    const u32 width = m_config.optics.sensorWidthPx;
    const u32 height = m_config.optics.sensorHeightPx;
    const u32 outputChannels = OutputChannels(m_config);
    if (!std::isfinite(firstRowMidpointSeconds) || !linearRgb.IsValid() ||
        linearRgb.width != width || linearRgb.height != height ||
        linearRgb.channels < 3 || m_config.device.channels.size() > 3)
        return Fail<CameraOutput>("fast RGB input needs three linear channels and matching geometry");
    for (f32 value : linearRgb.data)
        if (!std::isfinite(value) || value < 0.0f)
            return Fail<CameraOutput>("fast RGB input must be finite and nonnegative");
    const auto grid = WavelengthGrid(m_config);
    const auto weights = MeasurementWeights(m_config, grid);
    const auto omega = ApertureSolidAngleSr(m_config.optics.fNumber);
    if (!weights || !omega) return Fail<CameraOutput>(
        weights ? omega.error() : weights.error());
    std::optional<Image> knownPsf;
    if (!m_config.optics.knownPsfPath.empty()) {
        knownPsf = ImageIO::ReadImage(m_config.optics.knownPsfPath);
        if (!knownPsf) return Fail<CameraOutput>("could not load known PSF image");
    }
    const f64 radianceScale = m_config.fastRgbRadianceScale > 0.0 ?
                              m_config.fastRgbRadianceScale : 1.0;
    const f64 pitchM = m_config.optics.pixelPitchUm * 1e-6;
    const f64 focalM = m_config.optics.focalLengthMm * 1e-3;
    std::vector<f64> expected(static_cast<size_t>(width) * height * outputChannels, 0.0);
    for (u32 deviceChannel = 0;
         deviceChannel < m_config.device.channels.size(); ++deviceChannel) {
        Image mapped(width, height, 1);
        const size_t row = static_cast<size_t>(deviceChannel) * 3;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                f64 value = 0.0;
                for (u32 rgb = 0; rgb < 3; ++rgb)
                    value += m_config.fastRgbToDevice[row + rgb] *
                             linearRgb(x, y, rgb);
                mapped(x, y, 0) = static_cast<f32>(std::max(0.0, value));
            }
        const auto& curve = DetectorCurve(
            m_config.device.channels[deviceChannel].response,
            m_config.device.detector);
        const f64 effectiveNm = 0.5 * (curve.MinNm() + curve.MaxNm());
        const auto kernel = BuildKernel(m_config, effectiveNm, knownPsf);
        if (!kernel) return Fail<CameraOutput>(kernel.error());
        const Image blurred = Convolve(mapped, kernel.value());
        f64 responseWeight = 0.0;
        for (f64 weight : weights.value()[deviceChannel]) responseWeight += weight;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                if (m_config.device.cfa != CfaPattern::MultiChannel &&
                    ChannelAt(m_config.device.cfa, x, y) != deviceChannel)
                    continue;
                f64 vignette = 1.0;
                if (m_config.optics.cosFourthVignetting) {
                    const f64 dx = (x + 0.5 - width * 0.5) * pitchM;
                    const f64 dy = (y + 0.5 - height * 0.5) * pitchM;
                    const f64 cosine = 1.0 / std::sqrt(
                        1.0 + (dx * dx + dy * dy) / (focalM * focalM));
                    vignette = std::pow(cosine, 4.0);
                }
                const size_t pixel = static_cast<size_t>(y) * width + x;
                const size_t index = pixel * outputChannels +
                    (m_config.device.cfa == CfaPattern::MultiChannel ?
                     deviceChannel : 0u);
                expected[index] = blurred(x, y, 0) * radianceScale *
                                  omega.value() * vignette * responseWeight;
            }
    }
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   true, false);
}

Result<CameraOutput, String>
CpuCameraPipeline::Readout(CaptureState& state, f64 firstRowMidpointSeconds,
                           std::vector<f64> expected,
                           bool fastRgbApproximation,
                           bool allowSignedMonteCarloResidual) const {
    const u32 width = m_config.optics.sensorWidthPx;
    const u32 height = m_config.optics.sensorHeightPx;
    const u32 channels = OutputChannels(m_config);
    const size_t elementCount = static_cast<size_t>(width) * height * channels;
    if (expected.size() != elementCount)
        return Fail<CameraOutput>("measured-input element count is invalid");
    size_t negativeMcElements = 0;
    f64 negativeMcSum = 0.0;
    f64 negativeMcMinimum = 0.0;
    for (auto& value : expected) {
        if (!std::isfinite(value))
            return Fail<CameraOutput>("camera measurement overflowed");
        if (value < 0.0) {
            if (!allowSignedMonteCarloResidual)
                return Fail<CameraOutput>("measured input must be nonnegative");
            negativeMcSum += -value;
            negativeMcMinimum = std::min(negativeMcMinimum, value);
            value = 0.0;
            ++negativeMcElements;
        }
    }
    const auto area = PixelCollectionAreaM2(m_config.optics);
    if (!area) return Fail<CameraOutput>(area.error());
    std::vector<f64> thermalReadNoiseDn(m_config.device.channels.size(),
                                         m_config.thermal.readNoiseDnRms);
    if (m_config.device.detector == DetectorKind::Thermal &&
        m_config.thermal.netdKelvin > 0.0) {
        for (size_t channel = 0; channel < thermalReadNoiseDn.size(); ++channel) {
            const auto slope = BlackbodyThermalDerivativeWPerK(
                m_config.thermal.netdReferenceTemperatureK,
                m_config.device.channels[channel].response,
                m_config.optics.fNumber, area.value());
            if (!slope) return Fail<CameraOutput>(slope.error());
            thermalReadNoiseDn[channel] = m_config.thermal.netdKelvin *
                slope.value() * m_config.thermal.responsivityDnPerWatt;
            // White output noise, rectangular readout window, B = 1/(2T).
            // Zero window retains the documented reference bandwidth.
            if (m_config.thermal.readoutWindowSeconds > 0.0) {
                const f64 bandwidthHz =
                    1.0 / (2.0 * m_config.thermal.readoutWindowSeconds);
                thermalReadNoiseDn[channel] *= std::sqrt(
                    bandwidthHz / m_config.thermal.netdNoiseBandwidthHz);
            }
        }
    }
    Image measured(width, height, channels);
    Image raw(width, height, channels);
    Image corrected(width, height, channels);
    for (u32 channel = 0; channel < channels; ++channel) {
        const String name = channels == 1 ? "Measurement" :
            m_config.device.channels[channel].name;
        measured.channelNames[channel] = name;
        raw.channelNames[channel] = channels == 1 ? "Raw" : name;
        corrected.channelNames[channel] = channels == 1 ? "Signal" : name;
    }
    CaptureState nextState = state;
    if (m_config.device.detector == DetectorKind::Thermal &&
        nextState.thermalPixelStateW.size() != elementCount)
        nextState.thermalPixelStateW = expected; // first-frame steady state.
    const f64 elapsed = std::max(0.0, firstRowMidpointSeconds - state.frameTimeSeconds);
    for (size_t index = 0; index < elementCount; ++index) {
        const u32 pixel = static_cast<u32>(index / channels);
        const u32 noisePixel = static_cast<u32>(index);
        const u32 x = pixel % width, y = pixel / width;
        const f64 rateOrPower = expected[index];
        if (!std::isfinite(rateOrPower) || rateOrPower < 0.0)
            return Fail<CameraOutput>("camera measurement overflowed");
        measured.data[index] = static_cast<f32>(rateOrPower);
        f64 analogDn = 0.0, correctedValue = 0.0;
        if (m_config.device.detector == DetectorKind::Photon) {
            const auto& detector = m_config.photon;
            const f64 exposure = m_config.readout.exposureSeconds;
            const f64 prnu = (!m_config.quality.noiseFree && detector.enableFpn) ?
                detector.prnuSigma * FixedGaussian(
                    m_config.randomSeed, noisePixel, NoiseClass::FixedPrnu) : 0.0;
            const f64 expectedLight = std::max(0.0, rateOrPower * exposure * (1.0 + prnu));
            const f64 darkNonuniform = (!m_config.quality.noiseFree && detector.enableFpn) ?
                detector.dsnuElectronsRms *
                (exposure / detector.dsnuReferenceExposureSeconds) *
                FixedGaussian(m_config.randomSeed, noisePixel, NoiseClass::FixedDsnu) : 0.0;
            const f64 expectedDark = detector.enableDarkCurrent ?
                std::max(0.0, detector.darkCurrentElectronsPerSecond * exposure +
                              darkNonuniform) : 0.0;
            const f64 light = (!m_config.quality.noiseFree && detector.enableShotNoise) ?
                Poisson(expectedLight, m_config.randomSeed, noisePixel, state.acquisitionIndex,
                        NoiseClass::PhotonShot) : expectedLight;
            const f64 dark = (!m_config.quality.noiseFree && detector.enableDarkShotNoise) ?
                Poisson(expectedDark, m_config.randomSeed, noisePixel, state.acquisitionIndex,
                        NoiseClass::DarkShot) : expectedDark;
            f64 electrons = std::min(detector.fullWellElectrons, light + dark);
            if (!m_config.quality.noiseFree && detector.enableReadNoise)
                electrons += detector.readNoiseElectronsRms *
                    TemporalGaussian(m_config.randomSeed, noisePixel, state.acquisitionIndex,
                                     NoiseClass::Read);
            const f64 fixedBias = (!m_config.quality.noiseFree && detector.enableFpn) ?
                detector.biasDnRms * FixedGaussian(
                    m_config.randomSeed, noisePixel, NoiseClass::Bias) : 0.0;
            analogDn = m_config.readout.analogGain * electrons /
                       m_config.readout.electronsPerDn +
                       m_config.readout.blackLevelDn + fixedBias;
            const f64 dn = Quantize(analogDn, m_config.readout.adcBits);
            raw.data[index] = static_cast<f32>(dn);
            correctedValue = (dn - m_config.readout.blackLevelDn - fixedBias) *
                             m_config.readout.electronsPerDn /
                             m_config.readout.analogGain;
            correctedValue -= detector.enableDarkCurrent ?
                detector.darkCurrentElectronsPerSecond * exposure : 0.0;
            if (detector.applyNuc) {
                const f64 gain = detector.nucGainMap.empty() ? 1.0 :
                                 detector.nucGainMap[index];
                const f64 offset = detector.nucOffsetElectronsMap.empty() ? 0.0 :
                                   detector.nucOffsetElectronsMap[index];
                correctedValue = correctedValue * gain + offset;
                if (!m_config.quality.noiseFree)
                    correctedValue += detector.nucResidualFraction *
                                      (prnu * rateOrPower * exposure +
                                       darkNonuniform);
            }
            corrected.data[index] = static_cast<f32>(std::max(0.0, correctedValue));
        } else {
            const auto& detector = m_config.thermal;
            const auto stepped = StepThermalResponse(nextState.thermalPixelStateW[index],
                rateOrPower, elapsed, detector.timeConstantSeconds);
            if (!stepped) return Fail<CameraOutput>(stepped.error());
            nextState.thermalPixelStateW[index] = stepped.value();
            analogDn = stepped.value() * detector.responsivityDnPerWatt +
                       m_config.readout.blackLevelDn +
                       detector.driftDnPerSecond * firstRowMidpointSeconds;
            if (!m_config.quality.noiseFree) {
                const u32 deviceChannel = m_config.device.cfa == CfaPattern::MultiChannel ?
                                          static_cast<u32>(index % channels) :
                                          ChannelAt(m_config.device.cfa, x, y);
                analogDn += thermalReadNoiseDn[deviceChannel] *
                    TemporalGaussian(m_config.randomSeed, noisePixel, state.acquisitionIndex,
                                     NoiseClass::ThermalRead);
            }
            const f64 dn = Quantize(analogDn, m_config.readout.adcBits);
            raw.data[index] = static_cast<f32>(dn);
            correctedValue = (dn - m_config.readout.blackLevelDn -
                detector.driftDnPerSecond * firstRowMidpointSeconds) /
                detector.responsivityDnPerWatt;
            if (!detector.nucGainMap.empty())
                correctedValue *= detector.nucGainMap[index];
            if (!detector.nucOffsetDnMap.empty())
                correctedValue += detector.nucOffsetDnMap[index] /
                                  detector.responsivityDnPerWatt;
            corrected.data[index] = static_cast<f32>(std::max(0.0, correctedValue));
        }
        (void)x;
        (void)y;
    }
    Image display(width, height, 3);
    display.channelNames = {"R", "G", "B"};
    f64 minThermal = std::numeric_limits<f64>::max(), maxThermal = 0.0;
    if (m_config.device.detector == DetectorKind::Thermal) {
        for (const f32 value : corrected.data) {
            minThermal = std::min(minThermal, static_cast<f64>(value));
            maxThermal = std::max(maxThermal, static_cast<f64>(value));
        }
    }
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            if (m_config.device.detector == DetectorKind::Photon) {
                const f64 scale = 1.0 / m_config.photon.fullWellElectrons;
                if (m_config.device.cfa == CfaPattern::MultiChannel && channels >= 3) {
                    for (u32 c = 0; c < 3; ++c)
                        display(x, y, c) = static_cast<f32>(std::clamp(
                            corrected.data[pixel * channels + c] * scale, 0.0, 1.0));
                } else {
                    const f32 value = static_cast<f32>(std::clamp(
                        corrected.data[pixel] * scale, 0.0, 1.0));
                    for (u32 c = 0; c < 3; ++c) display(x, y, c) = value;
                }
            } else {
                const f32 value = static_cast<f32>(std::clamp(
                    (corrected.data[pixel * channels] - minThermal) /
                    std::max(1e-30, maxThermal - minThermal), 0.0, 1.0));
                for (u32 c = 0; c < 3; ++c) display(x, y, c) = value;
            }
        }
    for (auto& value : display.data) value = EncodeSrgb(value);
    CameraOutput output;
    const u64 acquisition = state.acquisitionIndex;
    if (m_config.products.bandMeasurement) {
        auto product = Product(std::move(measured), Descriptor(m_config,
            SignalKind::BandMeasurement,
            m_config.device.detector == DetectorKind::Photon ? "e-/s" : "W",
            acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.bandMeasurement = std::move(product.value());
    }
    if (m_config.products.rawDn) {
        auto product = Product(std::move(raw), Descriptor(m_config,
            SignalKind::RawDN, "DN", acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        product.value().image.metadata["camera_adc_bits"] =
            std::to_string(m_config.readout.adcBits);
        product.value().image.metadata["camera_output_bits"] =
            std::to_string(m_config.readout.outputBits);
        if (m_config.device.detector == DetectorKind::Thermal &&
            m_config.thermal.netdKelvin > 0.0) {
            product.value().image.metadata["camera_netd_reference_k"] =
                std::to_string(m_config.thermal.netdReferenceTemperatureK);
            product.value().image.metadata["camera_netd_reference_bandwidth_hz"] =
                std::to_string(m_config.thermal.netdNoiseBandwidthHz);
            product.value().image.metadata["camera_netd_noise_model"] =
                "white_noise_rectangular_readout";
            product.value().image.metadata["camera_readout_window_s"] =
                std::to_string(m_config.thermal.readoutWindowSeconds);
        }
        output.rawDn = std::move(product.value());
    }
    if (m_config.products.display) {
        auto product = Product(std::move(display), Descriptor(m_config,
            SignalKind::DevicePreviewSrgb, "sRGB-preview", acquisition,
            firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.display = std::move(product.value());
    }
    if (m_config.products.apparentTemperature) {
        Image temperature(width, height, channels);
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                for (u32 c = 0; c < channels; ++c) {
                    const u32 channel = m_config.device.cfa == CfaPattern::MultiChannel ?
                                        c : ChannelAt(m_config.device.cfa, x, y);
                    const f64 measurement = m_config.device.detector == DetectorKind::Photon ?
                        corrected.data[pixel * channels + c] /
                            m_config.readout.exposureSeconds :
                        corrected.data[pixel * channels + c];
                    const auto kelvin = ApparentTemperature(
                        measurement, m_config, channel, area.value());
                    if (!kelvin) return Fail<CameraOutput>(kelvin.error());
                    temperature.data[pixel * channels + c] =
                        static_cast<f32>(kelvin.value());
                }
            }
        auto product = Product(std::move(temperature), Descriptor(m_config,
            SignalKind::ApparentTemperature, "K", acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.apparentTemperature = std::move(product.value());
    }
    if (m_config.products.correctedDeviceSignal) {
        auto product = Product(std::move(corrected), Descriptor(m_config,
            SignalKind::DeviceLinear,
            m_config.device.detector == DetectorKind::Photon ? "e-" : "W",
            acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.correctedDeviceSignal = std::move(product.value());
    }
    if (fastRgbApproximation) {
        const auto annotateApproximation = [this](
            std::optional<CameraProduct>& product) {
            if (!product) return;
            auto& metadata = product->image.metadata;
            metadata["camera_input_semantics"] = "fast_rgb_approximation";
            metadata["camera_fast_rgb_scale_Wm2SrNm"] =
                std::to_string(m_config.fastRgbRadianceScale > 0.0 ?
                               m_config.fastRgbRadianceScale : 1.0);
            metadata["camera_fast_rgb_mapping"] =
                m_config.calibratedFastRgbInput ? "calibrated_matrix" :
                                                 "generic_identity_assumption";
        };
        annotateApproximation(output.bandMeasurement);
        annotateApproximation(output.rawDn);
        annotateApproximation(output.correctedDeviceSignal);
        annotateApproximation(output.apparentTemperature);
        annotateApproximation(output.display);
    }
    if (negativeMcElements > 0) {
        const auto annotate = [negativeMcElements, negativeMcSum,
                               negativeMcMinimum](
            std::optional<CameraProduct>& product) {
            if (product) {
                product->image.metadata["camera_negative_mc_measurement_elements"] =
                    std::to_string(negativeMcElements);
                product->image.metadata["camera_negative_mc_rate_sum"] =
                    Precise(negativeMcSum);
                product->image.metadata["camera_negative_mc_rate_minimum"] =
                    Precise(negativeMcMinimum);
            }
        };
        annotate(output.bandMeasurement);
        annotate(output.rawDn);
        annotate(output.correctedDeviceSignal);
        annotate(output.apparentTemperature);
        annotate(output.display);
    }
    nextState.acquisitionIndex = acquisition + 1;
    nextState.frameTimeSeconds = firstRowMidpointSeconds;
    nextState.nextExposureSeconds = m_config.readout.exposureSeconds;
    nextState.nextAnalogGain = m_config.readout.analogGain;
    state = std::move(nextState);
    return output;
}

} // namespace quantiloom::camera
