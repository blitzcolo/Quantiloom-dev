#include "postprocess/CpuCameraPipeline.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/CameraAutoControl.hpp"
#include "postprocess/CameraIsp.hpp"
#include "postprocess/CameraDemosaic.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <type_traits>

namespace quantiloom::camera {
namespace {


const ResponseCurve& DetectorCurve(const ResponseStack& stack, DetectorKind kind) {
    if (stack.systemResponse) return *stack.systemResponse;
    return kind == DetectorKind::Photon ? *stack.quantumEfficiency :
                                         *stack.thermalAbsorptance;
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
    const bool displayProduct = kind == SignalKind::DisplaySrgb ||
        kind == SignalKind::CieLinearSrgb;
    signal.channelsPerPixel = displayProduct ? 3u : OutputChannels(config);
    signal.responseMinNm = std::numeric_limits<f64>::max();
    signal.responseMaxNm = 0.0;
    for (const auto& channel : config.device.channels) {
        const auto& response = DetectorCurve(channel.response, config.device.detector);
        signal.responseMinNm = std::min(signal.responseMinNm, response.MinNm());
        signal.responseMaxNm = std::max(signal.responseMaxNm, response.MaxNm());
    }
    if (!displayProduct && config.device.cfa == CfaPattern::MultiChannel)
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

CpuCameraPipeline::CpuCameraPipeline(CameraConfig config,bool samplerAppliesVignetting,
    const std::array<Image,4>* sampled,std::array<Image,5>* integrated)
    : m_config(std::move(config)),m_samplerAppliesVignetting(samplerAppliesVignetting),
      m_sampledContributions(sampled),m_integratedContributions(integrated) {}

CameraConfig CpuCameraPipeline::EffectiveConfig(const CaptureState& state,
                                                bool commitState) const {
    // The auto controller writes its output into state.next*; a committed
    // acquisition consumes it as its config. A fresh state (nextExposure 0)
    // means "never captured": start from the authored manual values. Each
    // rail only overrides when its own loop is enabled, so with auto off the
    // effective config is the authored one and legacy behavior is unchanged.
    CameraConfig effective = m_config;
    if (!commitState)
        return effective;
    const IspConfig& isp = m_config.isp;
    const bool exposureFresh = state.nextExposureSeconds <= 0.0;
    if (isp.autoExposure && !exposureFresh) {
        effective.readout.exposureSeconds = state.nextExposureSeconds;
        if (state.nextAnalogGain > 0.0)
            effective.readout.analogGain = state.nextAnalogGain;
    }
    if (isp.autoWhiteBalance && !exposureFresh)
        effective.isp.whiteBalance = state.nextWhiteBalance;
    return effective;
}

Result<CameraOutput, String>
CpuCameraPipeline::Capture(CaptureState& state, f64 firstRowMidpointSeconds,
                           const SpectralFrameSampler& sampler) const {
    return CaptureImpl(state, firstRowMidpointSeconds, sampler,
                       /*commitState=*/true);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureReprocess(const CaptureState& state,
                                    f64 firstRowMidpointSeconds,
                                    const SpectralFrameSampler& sampler) const {
    CaptureState working = state;
    return CaptureImpl(working, firstRowMidpointSeconds, sampler,
                       /*commitState=*/false);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureImpl(CaptureState& state, f64 firstRowMidpointSeconds,
                               const SpectralFrameSampler& sampler,
                               bool commitState) const {
    // A committed acquisition consumes the feedback written by the previous
    // one's auto controller; a reprocess (commitState=false) replays against
    // the authored config so it cannot advance or even move the loop.
    const CameraConfig captureConfig = EffectiveConfig(state, commitState);
    if (captureConfig.products.tracedRadiance || captureConfig.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(captureConfig);
    if (!valid) return Fail<CameraOutput>(valid.error());
    if (!std::isfinite(firstRowMidpointSeconds) || !sampler)
        return Fail<CameraOutput>("camera capture needs finite frame time and a sampler");
    const u32 width = captureConfig.optics.sensorWidthPx;
    const u32 height = captureConfig.optics.sensorHeightPx;
    const u32 channels = OutputChannels(captureConfig);
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t elementCount = pixelCount * channels;
    const auto wavelengths = WavelengthGrid(captureConfig);
    const auto weights = MeasurementWeights(captureConfig, wavelengths);
    if (!weights) return Fail<CameraOutput>(weights.error());
    const auto omega = ApertureSolidAngleSr(captureConfig.optics.fNumber);
    const auto area = PixelCollectionAreaM2(captureConfig.optics);
    if (!omega || !area) return Fail<CameraOutput>("invalid camera aperture or pixel area");
    std::optional<Image> knownPsf;
    if (!captureConfig.optics.knownPsfPath.empty()) {
        knownPsf = ImageIO::ReadImage(captureConfig.optics.knownPsfPath);
        if (!knownPsf) return Fail<CameraOutput>("could not load known PSF image");
    }
    std::vector<f64> vignette(pixelCount, 1.0);
    if (captureConfig.optics.cosFourthVignetting && !m_samplerAppliesVignetting) {
        const auto projection=ResolveProjection(captureConfig.optics.projection,width,height,
            captureConfig.optics.focalLengthMm,captureConfig.optics.pixelPitchUm);
        if(!projection)return Fail<CameraOutput>(projection.error());
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const auto ray=UnprojectPixel(*projection,{x+.5,y+.5});
                const f64 cosine=ray.valid ? ray.direction.z : 0;
                vignette[static_cast<size_t>(y) * width + x] = std::pow(cosine, 4.0);
            }
    }
    std::vector<f64> expected(elementCount, 0.0);
    std::array<std::vector<f64>,4> components;
    if(m_integratedContributions)for(auto& v:components)v.resize(elementCount,0);
    const bool rolling = captureConfig.readout.shutter == ShutterKind::Rolling;
    const u32 rowGroups = rolling ? height : 1u;
    struct ExposureEvent { f64 time; u32 firstY; u32 lastY; };
    std::vector<ExposureEvent> events;
    events.reserve(static_cast<size_t>(rowGroups) * captureConfig.quality.timeSamples);
    for (u32 group = 0; group < rowGroups; ++group) {
        for (u32 timeSample = 0; timeSample < captureConfig.quality.timeSamples; ++timeSample) {
            const f64 time = firstRowMidpointSeconds +
                (rolling ? group * captureConfig.readout.rowDelaySeconds : 0.0) -
                captureConfig.readout.exposureSeconds / 2.0 +
                (timeSample + 0.5) * captureConfig.readout.exposureSeconds /
                    captureConfig.quality.timeSamples;
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
                const auto kernel = BuildKernel(captureConfig, wavelengths[wavelengthIndex], knownPsf);
                if (!kernel) return Fail<CameraOutput>(kernel.error());
                Image irradianceInput=radiance;
                for(size_t i=0;i<pixelCount;++i)irradianceInput.data[i]*=static_cast<f32>(vignette[i]);
                const Image blurred = Convolve(irradianceInput, kernel.value());
                std::array<Image,4> blurredComponents;
                if(m_integratedContributions) {
                    if(!m_sampledContributions)return Fail<CameraOutput>("missing fusion components");
                    for(size_t c=0;c<4;++c) {
                        auto input=(*m_sampledContributions)[c];
                        if(input.width!=width || input.height!=height || input.channels!=1)
                            return Fail<CameraOutput>("fusion component grid mismatch");
                        for(size_t i=0;i<pixelCount;++i)input.data[i]*=static_cast<f32>(vignette[i]);
                        blurredComponents[c]=Convolve(input,kernel.value());
                    }
                }
                for (u32 y = event.firstY; y < event.lastY; ++y)
                    for (u32 x = 0; x < width; ++x) {
                        const size_t pixel = static_cast<size_t>(y) * width + x;
                        const f64 irradiance = blurred(x, y, 0) * omega.value();
                        if (captureConfig.device.cfa == CfaPattern::MultiChannel) {
                            for (u32 channel = 0; channel < channels; ++channel)
                                expected[pixel * channels + channel] += irradiance *
                                    weights.value()[channel][wavelengthIndex] /
                                    captureConfig.quality.timeSamples;
                            if(m_integratedContributions)for(u32 channel=0;channel<channels;++channel)
                                for(size_t c=0;c<4;++c)components[c][pixel*channels+channel]+=
                                    blurredComponents[c](x,y,0)*omega.value()*weights.value()[channel][wavelengthIndex]/captureConfig.quality.timeSamples;
                        } else {
                            const u32 channel = CfaChannelAt(captureConfig.device.cfa, x, y);
                            expected[pixel] += irradiance *
                                weights.value()[channel][wavelengthIndex] /
                                captureConfig.quality.timeSamples;
                            if(m_integratedContributions)for(size_t c=0;c<4;++c)components[c][pixel]+=
                                blurredComponents[c](x,y,0)*omega.value()*weights.value()[channel][wavelengthIndex]/captureConfig.quality.timeSamples;
                        }
                    }
            }
    }
    if(m_integratedContributions) {
        for(size_t c=0;c<5;++c) {
            auto& image=(*m_integratedContributions)[c];image=Image(width,height,channels);
            for(u32 channel=0;channel<channels;++channel)image.channelNames[channel]=channels==1 ? "Measurement" : captureConfig.device.channels[channel].name;
            for(size_t i=0;i<elementCount;++i)image.data[i]=static_cast<f32>(c==4 ? expected[i] : components[c][i]);
        }
    }
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   false, true, commitState, captureConfig);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureMeasured(CaptureState& state, f64 firstRowMidpointSeconds,
                                   const Image& measuredRate) const {
    const CameraConfig captureConfig = EffectiveConfig(state, /*commitState=*/true);
    if (captureConfig.products.tracedRadiance || captureConfig.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(captureConfig);
    if (!valid) return Fail<CameraOutput>(valid.error());
    if (!std::isfinite(firstRowMidpointSeconds) || !measuredRate.IsValid() ||
        measuredRate.width != captureConfig.optics.sensorWidthPx ||
        measuredRate.height != captureConfig.optics.sensorHeightPx ||
        measuredRate.channels != OutputChannels(captureConfig))
        return Fail<CameraOutput>("measured-input image has invalid time, shape or channels");
    std::vector<f64> expected(measuredRate.data.begin(), measuredRate.data.end());
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   false, false, /*commitState=*/true, captureConfig);
}

Result<CameraOutput, String>
CpuCameraPipeline::CaptureFastRgb(CaptureState& state, f64 firstRowMidpointSeconds,
                                  const Image& linearRgb) const {
    const CameraConfig captureConfig = EffectiveConfig(state, /*commitState=*/true);
    if (captureConfig.products.tracedRadiance || captureConfig.products.cieLinearSrgb)
        return Fail<CameraOutput>(
            "traced and CIE products must be supplied by the renderer facade");
    const auto valid = ValidateCameraConfig(captureConfig);
    if (!valid) return Fail<CameraOutput>(valid.error());
    const u32 width = captureConfig.optics.sensorWidthPx;
    const u32 height = captureConfig.optics.sensorHeightPx;
    const u32 outputChannels = OutputChannels(captureConfig);
    if (!std::isfinite(firstRowMidpointSeconds) || !linearRgb.IsValid() ||
        linearRgb.width != width || linearRgb.height != height ||
        linearRgb.channels < 3 || captureConfig.device.channels.size() > 3)
        return Fail<CameraOutput>("fast RGB input needs three linear channels and matching geometry");
    for (f32 value : linearRgb.data)
        if (!std::isfinite(value) || value < 0.0f)
            return Fail<CameraOutput>("fast RGB input must be finite and nonnegative");
    const auto grid = WavelengthGrid(captureConfig);
    const auto weights = MeasurementWeights(captureConfig, grid);
    const auto omega = ApertureSolidAngleSr(captureConfig.optics.fNumber);
    if (!weights || !omega) return Fail<CameraOutput>(
        weights ? omega.error() : weights.error());
    std::optional<Image> knownPsf;
    if (!captureConfig.optics.knownPsfPath.empty()) {
        knownPsf = ImageIO::ReadImage(captureConfig.optics.knownPsfPath);
        if (!knownPsf) return Fail<CameraOutput>("could not load known PSF image");
    }
    const f64 radianceScale = captureConfig.fastRgbRadianceScale > 0.0 ?
                              captureConfig.fastRgbRadianceScale : 1.0;
    const f64 pitchM = captureConfig.optics.pixelPitchUm * 1e-6;
    const f64 focalM = captureConfig.optics.focalLengthMm * 1e-3;
    std::vector<f64> expected(static_cast<size_t>(width) * height * outputChannels, 0.0);
    for (u32 deviceChannel = 0;
         deviceChannel < captureConfig.device.channels.size(); ++deviceChannel) {
        Image mapped(width, height, 1);
        const size_t row = static_cast<size_t>(deviceChannel) * 3;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                f64 value = 0.0;
                for (u32 rgb = 0; rgb < 3; ++rgb)
                    value += captureConfig.fastRgbToDevice[row + rgb] *
                             linearRgb(x, y, rgb);
                mapped(x, y, 0) = static_cast<f32>(std::max(0.0, value));
            }
        const auto& curve = DetectorCurve(
            captureConfig.device.channels[deviceChannel].response,
            captureConfig.device.detector);
        const f64 effectiveNm = 0.5 * (curve.MinNm() + curve.MaxNm());
        const auto kernel = BuildKernel(captureConfig, effectiveNm, knownPsf);
        if (!kernel) return Fail<CameraOutput>(kernel.error());
        const Image blurred = Convolve(mapped, kernel.value());
        f64 responseWeight = 0.0;
        for (f64 weight : weights.value()[deviceChannel]) responseWeight += weight;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                if (captureConfig.device.cfa != CfaPattern::MultiChannel &&
                    CfaChannelAt(captureConfig.device.cfa, x, y) != deviceChannel)
                    continue;
                f64 vignette = 1.0;
                if (captureConfig.optics.cosFourthVignetting) {
                    const f64 dx = (x + 0.5 - width * 0.5) * pitchM;
                    const f64 dy = (y + 0.5 - height * 0.5) * pitchM;
                    const f64 cosine = 1.0 / std::sqrt(
                        1.0 + (dx * dx + dy * dy) / (focalM * focalM));
                    vignette = std::pow(cosine, 4.0);
                }
                const size_t pixel = static_cast<size_t>(y) * width + x;
                const size_t index = pixel * outputChannels +
                    (captureConfig.device.cfa == CfaPattern::MultiChannel ?
                     deviceChannel : 0u);
                expected[index] = blurred(x, y, 0) * radianceScale *
                                  omega.value() * vignette * responseWeight;
            }
    }
    return Readout(state, firstRowMidpointSeconds, std::move(expected),
                   true, false, /*commitState=*/true, captureConfig);
}

Result<CameraOutput, String>
CpuCameraPipeline::Readout(CaptureState& state, f64 firstRowMidpointSeconds,
                           std::vector<f64> expected,
                           bool fastRgbApproximation,
                           bool allowSignedMonteCarloResidual,
                           bool commitState,
                           const CameraConfig& config) const {
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    const u32 channels = OutputChannels(config);
    const size_t elementCount = static_cast<size_t>(width) * height * channels;
    const u32 effectiveSeed = DeviceRandomSeed(config);
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
    const auto area = PixelCollectionAreaM2(config.optics);
    if (!area) return Fail<CameraOutput>(area.error());
    std::vector<f64> thermalReadNoiseDn(config.device.channels.size(),
                                         config.thermal.readNoiseDnRms);
    if (config.device.detector == DetectorKind::Thermal &&
        config.thermal.netdKelvin > 0.0) {
        for (size_t channel = 0; channel < thermalReadNoiseDn.size(); ++channel) {
            const auto slope = BlackbodyThermalDerivativeWPerK(
                config.thermal.netdReferenceTemperatureK,
                config.device.channels[channel].response,
                config.optics.fNumber, area.value());
            if (!slope) return Fail<CameraOutput>(slope.error());
            thermalReadNoiseDn[channel] = config.thermal.netdKelvin *
                slope.value() * config.thermal.responsivityDnPerWatt;
            // White output noise, rectangular readout window, B = 1/(2T).
            // Zero window retains the documented reference bandwidth.
            if (config.thermal.readoutWindowSeconds > 0.0) {
                const f64 bandwidthHz =
                    1.0 / (2.0 * config.thermal.readoutWindowSeconds);
                thermalReadNoiseDn[channel] *= std::sqrt(
                    bandwidthHz / config.thermal.netdNoiseBandwidthHz);
            }
        }
    }
    Image measured(width, height, channels);
    Image raw(width, height, channels);
    Image corrected(width, height, channels);
    for (u32 channel = 0; channel < channels; ++channel) {
        const String name = channels == 1 ? "Measurement" :
            config.device.channels[channel].name;
        measured.channelNames[channel] = name;
        raw.channelNames[channel] = channels == 1 ? "Raw" : name;
        corrected.channelNames[channel] = channels == 1 ? "Signal" : name;
    }
    CaptureState nextState = state;
    if (config.device.detector == DetectorKind::Thermal &&
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
        if (config.device.detector == DetectorKind::Photon) {
            const auto& detector = config.photon;
            const f64 exposure = config.readout.exposureSeconds;
            const f64 prnu = (!config.quality.noiseFree && detector.enableFpn) ?
                detector.prnuSigma * FixedGaussian(
                    effectiveSeed, noisePixel, NoiseClass::FixedPrnu) : 0.0;
            const f64 expectedLight = std::max(0.0, rateOrPower * exposure * (1.0 + prnu));
            const f64 darkNonuniform = (!config.quality.noiseFree && detector.enableFpn) ?
                detector.dsnuElectronsRms *
                (exposure / detector.dsnuReferenceExposureSeconds) *
                FixedGaussian(effectiveSeed, noisePixel, NoiseClass::FixedDsnu) : 0.0;
            const f64 expectedDark = detector.enableDarkCurrent ?
                std::max(0.0, detector.darkCurrentElectronsPerSecond * exposure +
                              darkNonuniform) : 0.0;
            const f64 light = (!config.quality.noiseFree && detector.enableShotNoise) ?
                Poisson(expectedLight, effectiveSeed, noisePixel, state.acquisitionIndex,
                        NoiseClass::PhotonShot) : expectedLight;
            const f64 dark = (!config.quality.noiseFree && detector.enableDarkShotNoise) ?
                Poisson(expectedDark, effectiveSeed, noisePixel, state.acquisitionIndex,
                        NoiseClass::DarkShot) : expectedDark;
            f64 electrons = std::min(detector.fullWellElectrons, light + dark);
            if (!config.quality.noiseFree && detector.enableReadNoise)
                electrons += detector.readNoiseElectronsRms *
                    TemporalGaussian(effectiveSeed, noisePixel, state.acquisitionIndex,
                                     NoiseClass::Read);
            const f64 fixedBias = (!config.quality.noiseFree && detector.enableFpn) ?
                detector.biasDnRms * FixedGaussian(
                    effectiveSeed, noisePixel, NoiseClass::Bias) : 0.0;
            analogDn = config.readout.analogGain * electrons /
                       config.readout.electronsPerDn +
                       config.readout.blackLevelDn + fixedBias;
            const f64 dn = Quantize(analogDn, config.readout.adcBits);
            raw.data[index] = static_cast<f32>(dn);
            correctedValue = (dn - config.readout.blackLevelDn - fixedBias) *
                             config.readout.electronsPerDn /
                             config.readout.analogGain;
            correctedValue -= detector.enableDarkCurrent ?
                detector.darkCurrentElectronsPerSecond * exposure : 0.0;
            if (detector.applyNuc) {
                const f64 gain = detector.nucGainMap.empty() ? 1.0 :
                                 detector.nucGainMap[index];
                const f64 offset = detector.nucOffsetElectronsMap.empty() ? 0.0 :
                                   detector.nucOffsetElectronsMap[index];
                correctedValue = correctedValue * gain + offset;
                if (!config.quality.noiseFree)
                    correctedValue += detector.nucResidualFraction *
                                      (prnu * rateOrPower * exposure +
                                       darkNonuniform);
            }
            corrected.data[index] = static_cast<f32>(std::max(0.0, correctedValue));
        } else {
            const auto& detector = config.thermal;
            const auto stepped = StepThermalResponse(nextState.thermalPixelStateW[index],
                rateOrPower, elapsed, detector.timeConstantSeconds);
            if (!stepped) return Fail<CameraOutput>(stepped.error());
            nextState.thermalPixelStateW[index] = stepped.value();
            analogDn = stepped.value() * detector.responsivityDnPerWatt +
                       config.readout.blackLevelDn +
                       detector.driftDnPerSecond * firstRowMidpointSeconds;
            if (!config.quality.noiseFree) {
                const u32 deviceChannel = config.device.cfa == CfaPattern::MultiChannel ?
                                          static_cast<u32>(index % channels) :
                                          CfaChannelAt(config.device.cfa, x, y);
                analogDn += thermalReadNoiseDn[deviceChannel] *
                    TemporalGaussian(effectiveSeed, noisePixel, state.acquisitionIndex,
                                     NoiseClass::ThermalRead);
            }
            const f64 dn = Quantize(analogDn, config.readout.adcBits);
            raw.data[index] = static_cast<f32>(dn);
            correctedValue = (dn - config.readout.blackLevelDn -
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
    // Acquisition statistics for the AE/AWB loop, computed while raw and
    // corrected are still intact (the products below move out of them).
    std::optional<AcquisitionStats> controlStats;
    if (commitState &&
        config.device.detector == DetectorKind::Photon &&
        (config.isp.autoExposure || config.isp.autoWhiteBalance))
        controlStats = ComputeAcquisitionStats(config, raw, corrected);
    Image display;
    if (config.products.display) {
        auto isp = RunIsp(config, raw, corrected, state, firstRowMidpointSeconds);
        if (!isp) return Fail<CameraOutput>(isp.error());
        display = std::move(isp.value());
    }
    CameraOutput output;
    const u64 acquisition = state.acquisitionIndex;
    if (config.products.bandMeasurement) {
        auto product = Product(std::move(measured), Descriptor(config,
            SignalKind::BandMeasurement,
            config.device.detector == DetectorKind::Photon ? "e-/s" : "W",
            acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.bandMeasurement = std::move(product.value());
    }
    if (config.products.rawDn) {
        auto product = Product(std::move(raw), Descriptor(config,
            SignalKind::RawDN, "DN", acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        product.value().image.metadata["camera_adc_bits"] =
            std::to_string(config.readout.adcBits);
        product.value().image.metadata["camera_output_bits"] =
            std::to_string(config.readout.outputBits);
        if (config.device.detector == DetectorKind::Thermal &&
            config.thermal.netdKelvin > 0.0) {
            product.value().image.metadata["camera_netd_reference_k"] =
                std::to_string(config.thermal.netdReferenceTemperatureK);
            product.value().image.metadata["camera_netd_reference_bandwidth_hz"] =
                std::to_string(config.thermal.netdNoiseBandwidthHz);
            product.value().image.metadata["camera_netd_noise_model"] =
                "white_noise_rectangular_readout";
            product.value().image.metadata["camera_readout_window_s"] =
                std::to_string(config.thermal.readoutWindowSeconds);
        }
        output.rawDn = std::move(product.value());
    }
    if (config.products.display) {
        auto product = Product(std::move(display), Descriptor(config,
            SignalKind::DisplaySrgb, "encoded sRGB", acquisition,
            firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.display = std::move(product.value());
    }
    if (config.products.apparentTemperature) {
        Image temperature(width, height, channels);
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                for (u32 c = 0; c < channels; ++c) {
                    const u32 channel = config.device.cfa == CfaPattern::MultiChannel ?
                                        c : CfaChannelAt(config.device.cfa, x, y);
                    const f64 measurement = config.device.detector == DetectorKind::Photon ?
                        corrected.data[pixel * channels + c] /
                            config.readout.exposureSeconds :
                        corrected.data[pixel * channels + c];
                    const auto kelvin = ApparentTemperature(
                        measurement, config, channel, area.value());
                    if (!kelvin) return Fail<CameraOutput>(kelvin.error());
                    temperature.data[pixel * channels + c] =
                        static_cast<f32>(kelvin.value());
                }
            }
        auto product = Product(std::move(temperature), Descriptor(config,
            SignalKind::ApparentTemperature, "K", acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.apparentTemperature = std::move(product.value());
    }
    if (config.products.correctedDeviceSignal) {
        auto product = Product(std::move(corrected), Descriptor(config,
            SignalKind::DeviceLinear,
            config.device.detector == DetectorKind::Photon ? "e-" : "W",
            acquisition, firstRowMidpointSeconds));
        if (!product) return Fail<CameraOutput>(product.error());
        output.correctedDeviceSignal = std::move(product.value());
    }
    if (fastRgbApproximation) {
        const auto annotateApproximation = [this, &config](
            std::optional<CameraProduct>& product) {
            if (!product) return;
            auto& metadata = product->image.metadata;
            metadata["camera_input_semantics"] = "fast_rgb_approximation";
            metadata["camera_fast_rgb_scale_Wm2SrNm"] =
                std::to_string(config.fastRgbRadianceScale > 0.0 ?
                               config.fastRgbRadianceScale : 1.0);
            metadata["camera_fast_rgb_mapping"] =
                config.calibratedFastRgbInput ? "calibrated_matrix" :
                                                 "generic_identity_assumption";
        };
        annotateApproximation(output.bandMeasurement);
        annotateApproximation(output.rawDn);
        annotateApproximation(output.correctedDeviceSignal);
        annotateApproximation(output.apparentTemperature);
        annotateApproximation(output.display);
    }
    const auto annotateSeed = [this, &config, effectiveSeed](
        std::optional<CameraProduct>& product) {
        if (!product) return;
        product->image.metadata["camera_random_seed"] =
            std::to_string(config.randomSeed);
        product->image.metadata["camera_effective_device_seed"] =
            std::to_string(effectiveSeed);
        product->image.metadata["camera_random_algorithm"] =
            "fnv1a_device_id_mix32_counter_v1";
    };
    annotateSeed(output.bandMeasurement);
    annotateSeed(output.rawDn);
    annotateSeed(output.correctedDeviceSignal);
    annotateSeed(output.apparentTemperature);
    annotateSeed(output.display);
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
    nextState.nextExposureSeconds = config.readout.exposureSeconds;
    nextState.nextAnalogGain = config.readout.analogGain;
    if (commitState) {
        // AE/AWB closed loop. Only a committed photon-chain acquisition runs
        // the controller: CaptureReprocess and the commitState=false path
        // never reach this block, so a same-tick reprocess cannot advance the
        // feedback. A thermal detector responds to absorbed power, not scene
        // luminance -- AE has no physical meaning there, so the auto_* flags
        // are treated as off and the write-back keeps the authored config.
        const bool photon = config.device.detector == DetectorKind::Photon;
        const bool autoWanted = config.isp.autoExposure || config.isp.autoWhiteBalance;
        if (photon && controlStats) {
            AutoControlState previous;
            const bool freshState = state.nextExposureSeconds <= 0.0;
            previous.exposure = freshState ? config.readout.exposureSeconds :
                                             state.nextExposureSeconds;
            previous.analogGain = freshState ? config.readout.analogGain :
                                               state.nextAnalogGain;
            previous.whiteBalance = freshState ? config.isp.whiteBalance :
                                                 state.nextWhiteBalance;
            AutoControlInput controlInput;
            controlInput.lumaMean = controlStats->lumaMean;
            const f64 total = static_cast<f64>(controlStats->saturatedCount +
                                               controlStats->unsaturatedCount);
            controlInput.saturatedFraction = total > 0.0 ?
                static_cast<f64>(controlStats->saturatedCount) / total : 0.0;
            controlInput.channelMeans = controlStats->channelMeans;
            const AutoControlState next = StepAutoControl(
                config.isp, previous, controlInput,
                config.isp.autoExposure, config.isp.autoWhiteBalance);
            nextState.nextExposureSeconds = next.exposure;
            nextState.nextAnalogGain = next.analogGain;
            nextState.nextWhiteBalance = next.whiteBalance;
        } else if (!photon && autoWanted && !m_autoThermalNoted) {
            m_autoThermalNoted = true;
            QL_LOG_WARN(
                "camera auto_exposure/auto_white_balance ignored: the thermal "
                "detector responds to absorbed power, not scene luminance");
        }
        state = std::move(nextState);
    }
    return output;
}

Result<CaptureCheckpoint, String>
CheckpointCamera(const CaptureState& state) {
    if (!std::isfinite(state.frameTimeSeconds) ||
        !std::isfinite(state.nextExposureSeconds) ||
        !std::isfinite(state.nextAnalogGain))
        return Fail<CaptureCheckpoint>("capture state has non-finite history fields");
    CaptureCheckpoint checkpoint;
    checkpoint.acquisitionIndex = state.acquisitionIndex;
    checkpoint.frameTimeSeconds = state.frameTimeSeconds;
    checkpoint.historyEpoch = state.historyEpoch;
    checkpoint.thermalPixelStateW = state.thermalPixelStateW;
    checkpoint.nextExposureSeconds = state.nextExposureSeconds;
    checkpoint.nextAnalogGain = state.nextAnalogGain;
    checkpoint.nextWhiteBalance = state.nextWhiteBalance;
    return checkpoint;
}

Result<void, String>
RestoreCamera(CaptureState& state, const CaptureCheckpoint& checkpoint) {
    if (!std::isfinite(checkpoint.frameTimeSeconds) ||
        !std::isfinite(checkpoint.nextExposureSeconds) ||
        !std::isfinite(checkpoint.nextAnalogGain))
        return Result<void, String>::Err("capture checkpoint has non-finite history fields");
    state.acquisitionIndex = checkpoint.acquisitionIndex;
    state.frameTimeSeconds = checkpoint.frameTimeSeconds;
    state.thermalPixelStateW = checkpoint.thermalPixelStateW;
    state.nextExposureSeconds = checkpoint.nextExposureSeconds;
    state.nextAnalogGain = checkpoint.nextAnalogGain;
    state.nextWhiteBalance = checkpoint.nextWhiteBalance;
    // A restore rewinds history: the epoch moves on so replayed ticks can be
    // told apart from the originals by hosts and by the GPU twin state.
    state.historyEpoch = checkpoint.historyEpoch + 1;
    return Result<void, String>::Ok();
}

Result<void, String>
AdvanceCameraState(CaptureState& state, f64 timeSeconds,
                   const CameraAdvanceFn& advance) {
    if (!std::isfinite(timeSeconds))
        return Result<void, String>::Err("camera advance needs a finite frame time");
    if (!advance)
        return Result<void, String>::Err("camera advance needs an acquisition step");
    const u64 before = state.acquisitionIndex;
    auto stepped = advance(state, timeSeconds);
    if (!stepped) return Result<void, String>::Err(stepped.error());
    if (state.acquisitionIndex != before + 1)
        return Result<void, String>::Err("camera advance did not advance the acquisition index");
    return Result<void, String>::Ok();
}

Result<void, String>
WarmUpCamera(CaptureState& state, f64 seconds, f64 framePeriodSeconds,
             const CameraAdvanceFn& advance) {
    if (!std::isfinite(seconds) || seconds < 0.0)
        return Result<void, String>::Err("camera warmup seconds must be finite and nonnegative");
    if (!std::isfinite(framePeriodSeconds) || framePeriodSeconds <= 0.0)
        return Result<void, String>::Err("camera warmup frame period must be finite and positive");
    if (!std::isfinite(state.frameTimeSeconds))
        return Result<void, String>::Err("camera warmup needs a finite current frame time");
    const u64 steps = static_cast<u64>(std::llround(seconds / framePeriodSeconds));
    if (steps == 0) return Result<void, String>::Ok();
    const f64 now = state.frameTimeSeconds;
    // The warmup acquisitions sit on the frame grid that ends at now:
    // nominal times now - K*T, ..., now - T. The grid start clamps to the
    // clock origin; a clamped step lands on the same instant as its
    // predecessor, measures an elapsed of zero, and so moves the acquisition
    // index and the noise streams but not the thermal state.
    const f64 start = std::max(now - static_cast<f64>(steps) * framePeriodSeconds,
                               0.0);
    // Rewind the frame-time anchor one period below the grid so the first
    // advance measures a full period. Without this every warmup step would
    // see a negative elapsed and the thermal state would stay frozen.
    state.frameTimeSeconds = start - framePeriodSeconds;
    for (u64 i = 0; i < steps; ++i) {
        const f64 nominal = now - static_cast<f64>(steps) * framePeriodSeconds +
                            static_cast<f64>(i) * framePeriodSeconds;
        const f64 time = std::max(nominal, start);
        auto stepped = AdvanceCameraState(state, time, advance);
        if (!stepped) return stepped;
    }
    return Result<void, String>::Ok();
}

} // namespace quantiloom::camera
