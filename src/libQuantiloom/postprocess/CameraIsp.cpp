#include "postprocess/CameraIsp.hpp"

#include "postprocess/CameraDemosaic.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace quantiloom::camera {
namespace {

// CFA channel index at a pixel; mirrors ChannelAt in CpuCameraPipeline.cpp.
// The CFA enum maps one scalar per pixel onto the device's R/G/B responses.
u32 CfaChannelAt(CfaPattern cfa, u32 x, u32 y) {
    const bool px = (x & 1u) != 0, py = (y & 1u) != 0;
    switch (cfa) {
    case CfaPattern::RGGB: return !py ? (px ? 1u : 0u) : (px ? 2u : 1u);
    case CfaPattern::GRBG: return !py ? (px ? 0u : 1u) : (px ? 1u : 2u);
    case CfaPattern::GBRG: return !py ? (px ? 2u : 1u) : (px ? 1u : 0u);
    case CfaPattern::BGGR: return !py ? (px ? 1u : 2u) : (px ? 0u : 1u);
    default: return 0u;
    }
}

f64 Clamp01(f64 value) { return std::clamp(value, 0.0, 1.0); }

f64 EncodeSrgb(f64 linear) {
    const f64 clamped = Clamp01(linear);
    return clamped <= 0.0031308 ? 12.92 * clamped :
           1.055 * std::pow(clamped, 1.0 / 2.4) - 0.055;
}

// Piecewise-linear control-point ramp, matching the HLSL palette functions
// (saturate, truncate, lerp).
template<size_t N>
std::array<f64, 3> SampleRamp(const std::array<std::array<f64, 3>, N>& points,
                              f64 t) {
    const f64 x = Clamp01(t) * static_cast<f64>(N - 1);
    const size_t i = std::min(static_cast<size_t>(x), N - 2);
    const f64 fraction = x - static_cast<f64>(i);
    std::array<f64, 3> out{};
    for (size_t c = 0; c < 3; ++c)
        out[c] = points[i][c] +
                 (points[i + 1][c] - points[i][c]) * fraction;
    return out;
}

} // namespace

std::array<f64, 3> ApplyDisplayPalette(f64 t, DisplayPalette palette) {
    switch (palette) {
    case DisplayPalette::GreyInverted:
        return {1.0 - Clamp01(t), 1.0 - Clamp01(t), 1.0 - Clamp01(t)};
    case DisplayPalette::Ironbow: return SampleRamp(isp::kIronbowPoints, t);
    case DisplayPalette::Rainbow: return SampleRamp(isp::kRainbowPoints, t);
    case DisplayPalette::Viridis: return SampleRamp(isp::kViridisPoints, t);
    case DisplayPalette::Grey: break;
    }
    return {Clamp01(t), Clamp01(t), Clamp01(t)};
}

namespace {

// One scalar display value per pixel. Linear is the percentile window (what a
// thermal camera calls linear AGC); Equalize is a 256-bin global histogram
// equalization (plateau AGC without the clip).
std::vector<f64> AgcTone(const std::vector<f64>& values,
                         const IspConfig& isp, bool* usedEqualizeFallback) {
    const size_t count = values.size();
    std::vector<f64> tone(values.size(), 0.5);
    if (count == 0) return tone;
    if (isp.infraredTone == DisplayToneMode::Linear) {
        std::vector<f64> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&sorted](f64 p) {
            const f64 rank = Clamp01(p / 100.0) *
                             static_cast<f64>(sorted.size() - 1);
            if (!std::isfinite(rank) || rank < 0.0 ||
                rank > static_cast<f64>(sorted.size() - 1))
                return sorted.front();
            const size_t index = std::min(static_cast<size_t>(std::floor(rank + 0.5)),
                                          sorted.size() - 1);
            return sorted[index];
        };
        const f64 lo = percentile(isp.contrastLowPercentile);
        const f64 hi = percentile(isp.contrastHighPercentile);
        if (!(hi > lo)) {
            std::fill(tone.begin(), tone.end(), 0.5);
            return tone;
        }
        for (size_t i = 0; i < count; ++i)
            tone[i] = Clamp01((values[i] - lo) / (hi - lo));
        return tone;
    }
    // Equalize, and CLAHE on the CPU chain: per-tile equalization is a GPU/
    // viewport path, so a CPU product degrades to the global operator and
    // records that in the product metadata.
    *usedEqualizeFallback = isp.infraredTone == DisplayToneMode::Clahe;
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const f64 lo = *minIt, hi = *maxIt;
    if (!(hi > lo)) {
        std::fill(tone.begin(), tone.end(), 0.5);
        return tone;
    }
    constexpr u32 kBins = 256;
    std::array<u64, kBins> histogram{};
    const auto binOf = [lo, hi](f64 v) {
        const u32 bin = static_cast<u32>(
            Clamp01((v - lo) / (hi - lo)) * (kBins - 1) + 0.5);
        return std::min(bin, kBins - 1);
    };
    for (f64 v : values) ++histogram[binOf(v)];
    std::array<f64, kBins> cdf{};
    f64 cumulative = 0.0;
    for (u32 b = 0; b < kBins; ++b) {
        cumulative += static_cast<f64>(histogram[b]);
        cdf[b] = cumulative / static_cast<f64>(count);
    }
    for (size_t i = 0; i < count; ++i) tone[i] = cdf[binOf(values[i])];
    return tone;
}

f64 Convolve5(const std::vector<f64>& plane, u32 width, u32 height,
              u32 x, u32 y, const std::array<std::array<f64, 5>, 5>& kernel) {
    f64 sum = 0.0;
    for (i32 dy = -2; dy <= 2; ++dy) {
        const u32 sy = static_cast<u32>(std::clamp(
            static_cast<i32>(y) + dy, 0, static_cast<i32>(height) - 1));
        for (i32 dx = -2; dx <= 2; ++dx) {
            const u32 sx = static_cast<u32>(std::clamp(
                static_cast<i32>(x) + dx, 0, static_cast<i32>(width) - 1));
            sum += kernel[dy + 2][dx + 2] *
                   plane[static_cast<size_t>(sy) * width + sx];
        }
    }
    return sum;
}

// 3x3 Gaussian (sigma 1), normalized, replicated border.
f64 Gaussian3(const std::vector<f64>& plane, u32 width, u32 height,
              u32 x, u32 y) {
    static constexpr f64 kWeight[3][3] = {
        {1.0, 2.0, 1.0}, {2.0, 4.0, 2.0}, {1.0, 2.0, 1.0}};
    f64 sum = 0.0;
    for (i32 dy = -1; dy <= 1; ++dy) {
        const u32 sy = static_cast<u32>(std::clamp(
            static_cast<i32>(y) + dy, 0, static_cast<i32>(height) - 1));
        for (i32 dx = -1; dx <= 1; ++dx) {
            const u32 sx = static_cast<u32>(std::clamp(
                static_cast<i32>(x) + dx, 0, static_cast<i32>(width) - 1));
            sum += kWeight[dy + 1][dx + 1] *
                   plane[static_cast<size_t>(sy) * width + sx];
        }
    }
    return sum / 16.0;
}

Result<std::vector<f64>, String> BuildDefectMask(
    const CameraConfig& config, u32 width, u32 height) {
    std::vector<f64> mask(static_cast<size_t>(width) * height, 0.0);
    for (const auto& [x, y] : config.isp.defectPixels) {
        if (x >= width || y >= height)
            return Result<std::vector<f64>, String>::Err(
                "defect pixel lies outside the sensor array");
        mask[static_cast<size_t>(y) * width + x] = 1.0;
    }
    return mask;
}

// Black-level-free preprocessing and white balance over one scalar plane of
// the corrected device frame. The corrected signal already has its black level
// subtracted and its NUC tables applied (that correction lives in the readout
// so the calibrated and RAW products can differ); the display chain consumes
// it directly, normalized by the well capacity. For a Bayer plane channelAt
// supplies the CFA channel per pixel; for MultiChannel it is the plane index.
// A defect pixel is replaced by the mean of its non-defect four-neighbors.
Result<std::vector<f64>, String> PreprocessPlane(
    const Image& corrected, u32 planeIndex, u32 planeCount,
    const CameraConfig& config, const std::vector<f64>& defectMask) {
    const u32 width = corrected.width, height = corrected.height;
    const f64 fullWell = config.photon.fullWellElectrons;
    std::vector<f64> plane(static_cast<size_t>(width) * height, 0.0);
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            const f64 signal = corrected.data[pixel * planeCount + planeIndex];
            f64 value = std::max(0.0, signal) / fullWell;
            const u32 channel = planeCount == 1 ?
                CfaChannelAt(config.device.cfa, x, y) : planeIndex;
            value *= config.isp.whiteBalance[std::min(channel, 2u)];
            plane[pixel] = value;
        }
    if (!config.isp.defectPixels.empty()) {
        const std::vector<f64> original = plane;
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                if (defectMask[pixel] == 0.0) continue;
                f64 sum = 0.0;
                u32 neighbors = 0;
                for (const auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0},
                                            std::pair{0, 1}, std::pair{0, -1}}) {
                    const i32 nx = static_cast<i32>(x) + dx;
                    const i32 ny = static_cast<i32>(y) + dy;
                    if (nx < 0 || ny < 0 || nx >= static_cast<i32>(width) ||
                        ny >= static_cast<i32>(height))
                        continue;
                    const size_t neighbor = static_cast<size_t>(ny) * width +
                                            static_cast<u32>(nx);
                    if (defectMask[neighbor] != 0.0) continue;
                    sum += original[neighbor];
                    ++neighbors;
                }
                plane[pixel] = neighbors > 0 ? sum / neighbors : 0.0;
            }
    }
    return plane;
}

Result<Image, String> RunVisibleIsp(const CameraConfig& config,
                                    const Image& corrected) {
    const u32 width = corrected.width, height = corrected.height;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const auto defectMask = BuildDefectMask(config, width, height);
    if (!defectMask) return Result<Image, String>::Err(defectMask.error());

    // Demosaic into linear per-pixel RGB (before CCM), 3 planes.
    std::vector<std::array<f64, 3>> linear(pixelCount);
    const bool bayer = corrected.channels == 1 &&
        config.device.cfa != CfaPattern::Mono;
    if (bayer) {
        const auto plane = PreprocessPlane(corrected, 0, 1, config, *defectMask);
        if (!plane) return Result<Image, String>::Err(plane.error());
        for (u32 y = 0; y < height; ++y)
            for (u32 x = 0; x < width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                const u32 here = CfaChannelAt(config.device.cfa, x, y);
                std::array<f64, 3> rgb{};
                for (u32 c = 0; c < 3; ++c) {
                    if (c == here) {
                        rgb[c] = (*plane)[pixel];
                        continue;
                    }
                    const std::array<std::array<f64, 5>, 5>* kernel;
                    if (c == 1) {
                        // Green at a red or blue site.
                        kernel = &mhc::kGreenAtRedBlue;
                    } else if (here == 1) {
                        // Color at a green site: orientation follows which
                        // color sits left and right of this green pixel.
                        const u32 side = x + 1 < width ?
                            CfaChannelAt(config.device.cfa, x + 1, y) : 3u;
                        kernel = side == c ? &mhc::kColorAtGreenHorizontal :
                                             &mhc::kColorAtGreenVertical;
                    } else {
                        // Red at a blue site or blue at a red site.
                        kernel = &mhc::kColorAtOpposite;
                    }
                    rgb[c] = Convolve5(*plane, width, height, x, y, *kernel);
                }
                linear[pixel] = rgb;
            }
    } else {
        // Mono: one scalar plane replicated; MultiChannel: up to three device
        // channels mapped onto display R/G/B.
        const u32 planes = corrected.channels;
        const u32 used = std::min(planes, 3u);
        for (u32 p = 0; p < used; ++p) {
            const auto plane = PreprocessPlane(corrected, p, planes, config, *defectMask);
            if (!plane) return Result<Image, String>::Err(plane.error());
            for (size_t pixel = 0; pixel < pixelCount; ++pixel)
                linear[pixel][p] = (*plane)[pixel];
        }
        for (size_t pixel = 0; pixel < pixelCount; ++pixel)
            for (u32 c = used; c < 3; ++c)
                linear[pixel][c] = linear[pixel][used - 1];
    }

    const auto& isp = config.isp;
    std::vector<std::array<f64, 3>> rgb = linear;
    // Color correction matrix.
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
        std::array<f64, 3> corrected{};
        for (u32 row = 0; row < 3; ++row)
            for (u32 col = 0; col < 3; ++col)
                corrected[row] += isp.deviceToLinearSrgb[row * 3 + col] *
                                  rgb[pixel][col];
        rgb[pixel] = corrected;
    }
    // Optional 3x3 Gaussian denoise, blended by denoiseStrength.
    if (isp.denoise && isp.denoiseStrength > 0.0) {
        const f64 blend = Clamp01(isp.denoiseStrength);
        for (u32 c = 0; c < 3; ++c) {
            std::vector<f64> plane(pixelCount);
            for (size_t i = 0; i < pixelCount; ++i) plane[i] = rgb[i][c];
            for (u32 y = 0; y < height; ++y)
                for (u32 x = 0; x < width; ++x) {
                    const size_t pixel = static_cast<size_t>(y) * width + x;
                    rgb[pixel][c] = plane[pixel] +
                        blend * (Gaussian3(plane, width, height, x, y) -
                                 plane[pixel]);
                }
        }
    }
    // Optional unsharp mask.
    if (isp.sharpen && isp.sharpenStrength > 0.0) {
        const f64 amount = std::max(0.0, isp.sharpenStrength);
        for (u32 c = 0; c < 3; ++c) {
            std::vector<f64> plane(pixelCount);
            for (size_t i = 0; i < pixelCount; ++i) plane[i] = rgb[i][c];
            for (u32 y = 0; y < height; ++y)
                for (u32 x = 0; x < width; ++x) {
                    const size_t pixel = static_cast<size_t>(y) * width + x;
                    rgb[pixel][c] = std::max(0.0, plane[pixel] +
                        amount * (plane[pixel] -
                                  Gaussian3(plane, width, height, x, y)));
                }
        }
    }
    // Tone, gamut handling, encode. pow needs a nonnegative base; the sharpen
    // stage already clamps below zero and tone is applied after it.
    Image display(width, height, 3);
    display.channelNames = {"R", "G", "B"};
    const f64 inverseGamma = 1.0 / isp.toneGamma;
    for (size_t pixel = 0; pixel < pixelCount; ++pixel)
        for (u32 c = 0; c < 3; ++c) {
            f64 value = std::pow(std::max(0.0, rgb[pixel][c]), inverseGamma);
            value = isp.clipOutOfGamut ? Clamp01(value) : value / (1.0 + value);
            display.data[pixel * 3 + c] = static_cast<f32>(EncodeSrgb(value));
        }
    return display;
}

Result<Image, String> RunInfraredDisplay(const CameraConfig& config,
                                         const Image& corrected,
                                         bool* usedEqualizeFallback) {
    const u32 width = corrected.width, height = corrected.height;
    const u32 channel = corrected.ChannelIndex("Signal", 0);
    std::vector<f64> values(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = corrected.data[i * corrected.channels + channel];
    const std::vector<f64> tone = AgcTone(values, config.isp, usedEqualizeFallback);
    Image display(width, height, 3);
    display.channelNames = {"R", "G", "B"};
    for (size_t i = 0; i < values.size(); ++i) {
        const std::array<f64, 3> rgb =
            ApplyDisplayPalette(tone[i], config.isp.infraredPalette);
        for (u32 c = 0; c < 3; ++c)
            display.data[i * 3 + c] = static_cast<f32>(EncodeSrgb(rgb[c]));
    }
    return display;
}

} // namespace

AcquisitionStats ComputeAcquisitionStats(const CameraConfig& config,
                                         const Image& rawDn,
                                         const Image& corrected) {
    AcquisitionStats stats;
    const u32 width = corrected.width, height = corrected.height;
    const u32 channels = corrected.channels;
    const size_t elementCount = static_cast<size_t>(width) * height * channels;
    const bool photon = config.device.detector == DetectorKind::Photon;
    const f64 fullWell = config.photon.fullWellElectrons;
    const f64 adcMax =
        std::ldexp(1.0, static_cast<int>(config.readout.adcBits)) - 1.0;
    const f64 satThreshold = 0.98 * adcMax;
    const bool rawUsable = rawDn.data.size() == elementCount;
    f64 lumaSum = 0.0;
    std::array<f64, 3> channelSums{0.0, 0.0, 0.0};
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x)
            for (u32 c = 0; c < channels; ++c) {
                const size_t index =
                    (static_cast<size_t>(y) * width + x) * channels + c;
                if (rawUsable &&
                    static_cast<f64>(rawDn.data[index]) >= satThreshold) {
                    ++stats.saturatedCount;
                    continue;
                }
                const f64 signal = corrected.data[index];
                const f64 value =
                    photon ? std::max(0.0, signal) / fullWell : signal;
                ++stats.unsaturatedCount;
                lumaSum += value;
                const u32 channel = channels == 1 ?
                    CfaChannelAt(config.device.cfa, x, y) :
                    std::min(c, 2u);
                channelSums[channel] += value;
                ++stats.channelCounts[channel];
            }
    if (stats.unsaturatedCount > 0)
        stats.lumaMean = lumaSum / static_cast<f64>(stats.unsaturatedCount);
    for (u32 c = 0; c < 3; ++c)
        if (stats.channelCounts[c] > 0)
            stats.channelMeans[c] =
                channelSums[c] / static_cast<f64>(stats.channelCounts[c]);
    return stats;
}

namespace {

// Hue in turns ([0,1) around the circle); value and saturation in [0,1].
std::array<f64, 3> RgbToHsv(const std::array<f64, 3>& rgb) {
    const f64 r = Clamp01(rgb[0]), g = Clamp01(rgb[1]), b = Clamp01(rgb[2]);
    const f64 maxc = std::max({r, g, b});
    const f64 minc = std::min({r, g, b});
    const f64 chroma = maxc - minc;
    f64 hue = 0.0;
    if (chroma > 0.0) {
        if (maxc == r) {
            hue = (g - b) / chroma / 6.0;
        } else if (maxc == g) {
            hue = (b - r) / chroma / 6.0 + 2.0 / 6.0;
        } else {
            hue = (r - g) / chroma / 6.0 + 4.0 / 6.0;
        }
        hue -= std::floor(hue); // wrap into [0,1)
    }
    const f64 saturation = maxc > 0.0 ? chroma / maxc : 0.0;
    return {hue, saturation, maxc};
}

std::array<f64, 3> HsvToRgb(const std::array<f64, 3>& hsv) {
    const f64 h = hsv[0] - std::floor(hsv[0]);
    const f64 s = Clamp01(hsv[1]);
    const f64 v = Clamp01(hsv[2]);
    const f64 sector = h * 6.0;
    const u32 i = static_cast<u32>(sector) % 6u;
    const f64 f = sector - std::floor(sector);
    const f64 p = v * (1.0 - s);
    const f64 q = v * (1.0 - s * f);
    const f64 t = v * (1.0 - s * (1.0 - f));
    switch (i) {
    case 0: return {v, t, p};
    case 1: return {q, v, p};
    case 2: return {p, v, t};
    case 3: return {p, q, v};
    case 4: return {t, p, v};
    default: return {v, p, q};
    }
}

f64 SmoothStep(f64 edge0, f64 edge1, f64 x) {
    const f64 t = Clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

} // namespace

Result<Image, String> ApplyHsv(const CameraConfig& config, Image display,
                               const u64 acquisitionIndex) {
    const HsvConfig& hsv = config.isp.hsv;
    if (display.channels != 3)
        return Result<Image, String>::Err("HSV stage needs a 3-channel display image");
    const u32 seed = DeviceRandomSeed(config);
    const f64 offsetTurns = hsv.hueOffsetDegrees / 360.0;
    const size_t pixelCount = static_cast<size_t>(display.width) * display.height;
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
        std::array<f64, 3> rgb{display.data[pixel * 3], display.data[pixel * 3 + 1],
                               display.data[pixel * 3 + 2]};
        std::array<f64, 3> hsvValue = RgbToHsv(rgb);
        f64 hue = hsvValue[0];
        if (hsv.hueOffsetDegrees != 0.0) {
            // Grey has no hue: fade the offset out below S=0.05 and keep it
            // fully above S=0.2, so an IR-grey frame stays exactly grey.
            const f64 weight = SmoothStep(0.05, 0.2, hsvValue[1]);
            if (weight > 0.0) {
                const f64 shifted = hue + offsetTurns;
                hue = hue + weight * (shifted - std::floor(shifted) - hue);
            }
        }
        f64 saturation = Clamp01(hsvValue[1] * hsv.saturationScale);
        f64 value = std::pow(Clamp01(hsvValue[2]),
                             1.0 / hsv.valueGamma);
        if (hsv.empiricalNoise)
            value += CounterGaussian(seed, static_cast<u32>(pixel),
                                     acquisitionIndex * 2u,
                                     NoiseClass::EmpiricalNoise) *
                     hsv.empiricalNoiseSigma;
        if (hsv.temporalDrift)
            value += CounterGaussian(seed, static_cast<u32>(pixel),
                                     acquisitionIndex >> 4,
                                     NoiseClass::EmpiricalDrift) *
                     hsv.temporalDriftSigma;
        const std::array<f64, 3> out =
            HsvToRgb({hue, saturation, Clamp01(value)});
        for (u32 c = 0; c < 3; ++c)
            display.data[pixel * 3 + c] = static_cast<f32>(out[c]);
    }
    return display;
}

Result<Image, String> RunIsp(const CameraConfig& config, const Image& rawDn,
                             const Image& corrected, const CaptureState& state,
                             f64 frameTimeSeconds) {
    (void)frameTimeSeconds; // Reserved for per-frame metadata; stages are stateless.
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    const u32 rawChannels = config.device.cfa == CfaPattern::MultiChannel ?
        static_cast<u32>(config.device.channels.size()) : 1u;
    if (width == 0 || height == 0)
        return Result<Image, String>::Err("camera sensor array has zero extent");
    if (!rawDn.IsValid() || rawDn.width != width || rawDn.height != height ||
        rawDn.channels != rawChannels)
        return Result<Image, String>::Err(
            "RAW DN image does not match the sensor array shape");
    const bool thermal = config.device.detector == DetectorKind::Thermal;
    if (!corrected.IsValid() || corrected.width != width ||
        corrected.height != height || corrected.channels != rawChannels)
        return Result<Image, String>::Err(
            "corrected device image does not match the sensor array shape");
    size_t displayCount = 0;
    if (!Image::TryElementCount(width, height, 3, displayCount))
        return Result<Image, String>::Err("ISP display exceeds image storage limits");
    if (!std::isfinite(config.isp.contrastLowPercentile) ||
        !std::isfinite(config.isp.contrastHighPercentile) ||
        config.isp.contrastLowPercentile < 0.0 ||
        config.isp.contrastHighPercentile > 100.0 ||
        config.isp.contrastLowPercentile >= config.isp.contrastHighPercentile)
        return Result<Image, String>::Err("invalid AGC percentile window");
    for (f32 value : corrected.data)
        if (!std::isfinite(value))
            return Result<Image, String>::Err("ISP input contains non-finite values");
    bool usedEqualizeFallback = false;
    Result<Image, String> display = thermal ?
        RunInfraredDisplay(config, corrected, &usedEqualizeFallback) :
        RunVisibleIsp(config, corrected);
    if (!display) return display;
    Image& image = display.value();
    // The HSV/empirical stage runs on the final encoded-sRGB display (visible
    // encode or infrared palette+encode), so pseudo-colour frames take the hue
    // path and grey IR frames are protected by the low-saturation suppression.
    // Every effect at its default means the stage is skipped entirely and the
    // product stays bit-identical to a chain without it.
    const HsvConfig& hsv = config.isp.hsv;
    if (hsv.hueOffsetDegrees != 0.0 || hsv.saturationScale != 1.0 ||
        hsv.valueGamma != 1.0 || hsv.empiricalNoise || hsv.temporalDrift) {
        auto effected = ApplyHsv(config, std::move(image), state.acquisitionIndex);
        if (!effected) return effected;
        image = std::move(effected.value());
    }
    for (f32 value : image.data)
        if (!std::isfinite(value))
            return Result<Image, String>::Err("ISP produced non-finite display values");
    image.metadata["camera_acquisition_index"] =
        std::to_string(state.acquisitionIndex);
    if (usedEqualizeFallback) {
        image.metadata["camera_ir_clahe_fallback"] = "equalize";
        image.metadata["camera_ir_tone_used"] = "equalize";
    }
    return display;
}

} // namespace quantiloom::camera
