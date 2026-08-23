// ============================================================================
// Quantiloom - Sensor Lab
// ============================================================================
// The sensor chain run many times over one radiance field, so that the two
// kinds of noise it produces can be told apart.
//
// They separate by how they behave across frames, and that is the whole reason
// this tool exists rather than a loop over renders:
//
//   * Temporal noise -- shot, read, dark -- is redrawn every frame.  Its size
//     is the per-pixel standard deviation across frames, and it is what an
//     empirical NETD is measured from.
//   * Fixed-pattern noise -- PRNU, DSNU -- is one fixed map.  Averaging frames
//     drives the temporal part down as 1/sqrt(N) and leaves the pattern
//     standing; the spatial standard deviation of that average is the residual
//     the non-uniformity correction failed to remove.
//
// Rendering N times cannot measure either.  Each render builds a fresh
// GenericSensor, and a fresh sensor with a fresh seed draws fresh FPN maps --
// so the pattern moves with the noise and nothing separates.  One sensor
// instance applied N times is what keeps the maps fixed while the stream
// advances, which is exactly the physical situation: one detector, many frames.
//
// Usage:
//   sensor_lab <config.toml> [--input in.exr | --uniform L] [options]
//
//     --input FILE     radiance EXR to run the chain over
//     --uniform L      synthesise a uniform field at radiance L instead
//                      (the dark field a real characterisation shutters for)
//     --size WxH       size of the synthesised field (default 128x128)
//     --frames N       how many frames to draw (default 100)
//     --out-prefix P   writes P_mean.exr, P_std.exr and P_stats.json
//
// The [sensor] section of the config is read exactly as a scene's is, so a
// characterisation runs against the same parameters the render used.
// ============================================================================

#include "core/Config.hpp"
#include "core/Image.hpp"
#include "core/Log.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/GenericSensor.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "postprocess/SensorModel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace quantiloom;

namespace {

/// Median of a copy, for reporting a per-pixel statistic as one number. The
/// median rather than the mean because a hot pixel or a saturated corner
/// should not set the figure the paper quotes.
f64 Median(std::vector<f64> values) {
    if (values.empty()) return 0.0;
    const usize mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<isize>(mid), values.end());
    return values[mid];
}

f64 Mean(const std::vector<f64>& values) {
    if (values.empty()) return 0.0;
    f64 sum = 0.0;
    for (const f64 v : values) sum += v;
    return sum / static_cast<f64>(values.size());
}

/// A string safe to put between JSON quotes.
///
/// The only string this file emits is a file path, and on Windows a path is
/// full of backslashes -- `H:\quantiloom` opens with an invalid JSON escape and
/// makes the whole stats file unparseable, which is a poor way for a
/// measurement to fail.
std::string JsonEscaped(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

f64 StdDev(const std::vector<f64>& values) {
    if (values.size() < 2) return 0.0;
    const f64 mean = Mean(values);
    f64 sum = 0.0;
    for (const f64 v : values) sum += (v - mean) * (v - mean);
    return std::sqrt(sum / static_cast<f64>(values.size() - 1));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: sensor_lab <config.toml> [--input in.exr | --uniform L]\n"
                     "                  [--size WxH] [--frames N] [--out-prefix P]\n"
                     "\n"
                     "  Runs one sensor instance over one radiance field N times and\n"
                     "  separates temporal noise from the fixed pattern.\n";
        return 1;
    }

    const std::string configPath = argv[1];
    std::string inputPath;
    std::string outPrefix;
    f64 uniformRadiance = -1.0;
    u32 width = 128;
    u32 height = 128;
    u32 frames = 100;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (arg == "--uniform" && i + 1 < argc) {
            uniformRadiance = std::stod(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = static_cast<u32>(std::stoul(argv[++i]));
        } else if (arg == "--out-prefix" && i + 1 < argc) {
            outPrefix = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            const std::string size = argv[++i];
            const auto x = size.find('x');
            if (x == std::string::npos) {
                std::cerr << "--size wants WxH, e.g. 256x256\n";
                return 1;
            }
            width = static_cast<u32>(std::stoul(size.substr(0, x)));
            height = static_cast<u32>(std::stoul(size.substr(x + 1)));
        } else {
            std::cerr << "unrecognised argument: " << arg << "\n";
            return 1;
        }
    }

    if (inputPath.empty() && uniformRadiance < 0.0) {
        std::cerr << "one of --input or --uniform is required\n";
        return 1;
    }
    if (frames < 1) {
        std::cerr << "--frames must be at least 1\n";
        return 1;
    }
    // One frame is a legitimate request -- it is what a figure showing the
    // chain's effect on an image wants -- but it separates nothing: the
    // per-pixel spread over a single sample is zero by construction, so the
    // temporal and residual-FPN columns below are meaningless rather than
    // small, and saying so is cheaper than a reader deducing it from a row of
    // zeros.
    const bool temporalDefined = frames >= 2;

    Log::Init();

    auto loaded = Config::Load(configPath);
    if (!loaded.has_value()) {
        QL_LOG_ERROR("cannot read config '{}': {}", configPath, loaded.error());
        return 1;
    }
    const SensorParams params = PostprocessConfig::ParseSensorParams(loaded.value());

    // ------------------------------------------------------------------
    // The radiance field
    // ------------------------------------------------------------------
    Image hdr;
    if (!inputPath.empty()) {
        auto image = ImageIO::ReadEXR(inputPath);
        if (!image.has_value()) {
            QL_LOG_ERROR("cannot read '{}'", inputPath);
            return 1;
        }
        hdr = std::move(image.value());

        // Pick the radiance channel BY NAME. ImageIO::ReadEXR fills its channel
        // list by walking OpenEXR's ChannelList, which iterates alphabetically,
        // so an RGBA file arrives ordered A, B, G, R -- index 0 is the ALPHA.
        // Reading index 0 here gave a constant 1.0 for every scene: first as a
        // signal four thousand times too small, so every cavity read as the same
        // dark frame, and then, once the band scaling was added, as 4000
        // W/sr/m^2, which saturates the well for every cavity alike. Both look
        // like a sensor result and neither is.
        //
        // Every band this tool serves is monochrome, so the radiance channel is
        // copied across three and the alpha is dropped.
        usize source = 0;
        for (usize c = 0; c < hdr.channelNames.size(); ++c) {
            const std::string& name = hdr.channelNames[c];
            if (name == "R" || name == "Y" || name == "V" || name == "Gray") {
                source = c;
                break;
            }
            if (name != "A" && name != "Alpha") source = c;
        }
        if (hdr.channels != 3 || source != 0) {
            const std::string picked = source < hdr.channelNames.size()
                                           ? hdr.channelNames[source] : "0";
            Image mono(hdr.width, hdr.height, 3);
            for (usize p = 0; p < static_cast<usize>(hdr.width) * hdr.height; ++p) {
                const f32 value = hdr.data[p * hdr.channels + source];
                mono.data[p * 3] = value;
                mono.data[p * 3 + 1] = value;
                mono.data[p * 3 + 2] = value;
            }
            mono.metadata = hdr.metadata;
            QL_LOG_INFO("sensor_lab: {} input channels, taking '{}' (index {}) as "
                        "the radiance", hdr.channels, picked, source);
            hdr = std::move(mono);
        }
    } else {
        hdr = Image(width, height, 3);
        std::fill(hdr.data.begin(), hdr.data.end(), static_cast<f32>(uniformRadiance));
    }

    // ------------------------------------------------------------------
    // Into the units the sensor chain expects
    // ------------------------------------------------------------------
    // A fused IR render writes the band AVERAGE, in W/sr/m^2/nm, and the
    // detector collects the band INTEGRAL. RenderJob multiplies by the band
    // width on the way in and divides it back out on the way to the EXR;
    // anything else driving GenericSensor has to do the same.
    //
    // Skipping it is not a small error. At 10 um the band is 4000 nm wide, so
    // the photon signal arrives 4000x too small -- for a 300 K cavity that is
    // about two thousand electrons against two hundred thousand from dark
    // current, and every scene reads as the same dark frame. Measured that way,
    // a 250 K and a 350 K cavity both returned 1357.2 DN.
    const auto parsedMode = ParseSpectralMode(loaded.value().GetString("spectral.mode", "rgb"));
    const SpectralMode mode = parsedMode.has_value() ? parsedMode.value() : SpectralMode::RGB;
    f64 radianceScale = 1.0;
    if (const auto band = GetFusedBandInfo(mode); band.has_value() && IsIRFusedMode(mode)) {
        radianceScale = static_cast<f64>(band->WidthNm());
        for (f32& value : hdr.data) {
            value = static_cast<f32>(static_cast<f64>(value) * radianceScale);
        }
        QL_LOG_INFO("sensor_lab: band {:.0f}-{:.0f} nm, radiance scaled by {:.0f} nm "
                    "to the band integral the detector sees",
                    band->lambdaMinNm, band->lambdaMaxNm, radianceScale);
    }

    {
        const auto [lo, hi] = std::minmax_element(hdr.data.begin(), hdr.data.end());
        QL_LOG_INFO("sensor_lab: radiance into the chain {:.6e} .. {:.6e} W/sr/m^2",
                    *lo, *hi);
    }

    QL_LOG_INFO("sensor_lab: {}x{} x{} channels, {} frames, seed {}, FPN {}, NUC {} @ {:.1f}%",
                hdr.width, hdr.height, hdr.channels, frames, params.noiseSeed,
                params.enableFPN ? "on" : "off", params.enableNUC ? "on" : "off",
                params.nucEfficiency * 100.0f);

    // ------------------------------------------------------------------
    // N frames through one sensor
    // ------------------------------------------------------------------
    // Welford would save the second pass, but N is small and the running form
    // hides a mistake that a two-pass sum does not: what is being separated
    // here is two standard deviations that differ by 1/sqrt(N), and that
    // margin should not also be carrying a numerical one.
    GenericSensor sensor;
    const usize pixels = static_cast<usize>(hdr.width) * hdr.height;

    std::vector<f64> sum(pixels, 0.0);
    std::vector<f64> sumSquares(pixels, 0.0);
    std::vector<f64> firstFrame(pixels, 0.0);

    for (u32 frame = 0; frame < frames; ++frame) {
        auto out = sensor.Apply(hdr, params);
        if (!out.has_value()) {
            QL_LOG_ERROR("sensor chain failed on frame {}: {}", frame, out.error());
            return 1;
        }
        const Image& dn = out.value().rawDN;

        // Channel 0 only. Every band this tool is used for is monochrome, and
        // the three channels of an IR render carry the same scalar.
        for (usize p = 0; p < pixels; ++p) {
            const f64 value = dn.data[p * dn.channels];
            sum[p] += value;
            sumSquares[p] += value * value;
            if (frame == 0) firstFrame[p] = value;
        }
    }

    // ------------------------------------------------------------------
    // Separate the two
    // ------------------------------------------------------------------
    const f64 n = static_cast<f64>(frames);
    std::vector<f64> perPixelMean(pixels);
    std::vector<f64> perPixelStd(pixels);
    for (usize p = 0; p < pixels; ++p) {
        perPixelMean[p] = sum[p] / n;
        const f64 variance =
            std::max(0.0, (sumSquares[p] - sum[p] * sum[p] / n) / (n - 1.0));
        perPixelStd[p] = std::sqrt(variance);
    }

    const f64 temporalMedian = Median(perPixelStd);
    const f64 temporalMean = Mean(perPixelStd);
    const f64 residualFpn = StdDev(perPixelMean);
    const f64 singleFrameSpatial = StdDev(firstFrame);
    const f64 meanDN = Mean(perPixelMean);

    // The spatial spread of a single frame contains both; the frame average
    // has driven the temporal part down by sqrt(N) and left the pattern. This
    // is the number that should track (1 - NUC efficiency) x injected FPN.
    QL_LOG_INFO("  mean DN                       {:.4f}", meanDN);
    if (temporalDefined) {
        QL_LOG_INFO("  temporal sigma (median px)    {:.4f} DN", temporalMedian);
        QL_LOG_INFO("  temporal sigma (mean px)      {:.4f} DN", temporalMean);
        QL_LOG_INFO("  residual FPN (spatial sigma)  {:.4f} DN", residualFpn);
    } else {
        QL_LOG_INFO("  temporal sigma                undefined at one frame");
        QL_LOG_INFO("  residual FPN                  undefined at one frame "
                    "(nothing has averaged out)");
    }
    QL_LOG_INFO("  single-frame spatial sigma    {:.4f} DN", singleFrameSpatial);

    if (outPrefix.empty()) {
        return 0;
    }

    Image meanImage(hdr.width, hdr.height, 1);
    Image stdImage(hdr.width, hdr.height, 1);
    meanImage.channelNames[0] = "DN";
    stdImage.channelNames[0] = "DN";
    for (usize p = 0; p < pixels; ++p) {
        meanImage.data[p] = static_cast<f32>(perPixelMean[p]);
        stdImage.data[p] = static_cast<f32>(perPixelStd[p]);
    }
    meanImage.metadata["frames"] = std::to_string(frames);
    meanImage.metadata["units"] = "DN";
    stdImage.metadata["frames"] = std::to_string(frames);
    stdImage.metadata["units"] = "DN";

    if (!ImageIO::WriteEXR(outPrefix + "_mean.exr", meanImage) ||
        !ImageIO::WriteEXR(outPrefix + "_std.exr", stdImage)) {
        QL_LOG_ERROR("cannot write EXRs under prefix '{}'", outPrefix);
        return 1;
    }

    std::ofstream stats(outPrefix + "_stats.json");
    if (!stats) {
        QL_LOG_ERROR("cannot write '{}_stats.json'", outPrefix);
        return 1;
    }
    stats << std::setprecision(9) << "{\n"
          << "  \"frames\": " << frames << ",\n"
          << "  \"width\": " << hdr.width << ",\n"
          << "  \"height\": " << hdr.height << ",\n"
          << "  \"source\": \""
          << JsonEscaped(inputPath.empty()
                             ? ("uniform:" + std::to_string(uniformRadiance))
                             : inputPath)
          << "\",\n"
          << "  \"noise_seed\": " << params.noiseSeed << ",\n"
          << "  \"enable_fpn\": " << (params.enableFPN ? "true" : "false") << ",\n"
          << "  \"prnu_sigma\": " << params.prnuSigma << ",\n"
          << "  \"dsnu_sigma_e\": " << params.dsnuSigma_e << ",\n"
          << "  \"enable_nuc\": " << (params.enableNUC ? "true" : "false") << ",\n"
          << "  \"nuc_efficiency\": " << params.nucEfficiency << ",\n"
          << "  \"bit_depth\": " << params.bitDepth << ",\n"
          << "  \"integration_time_s\": " << params.integrationTime_s << ",\n"
          << "  \"f_number\": " << params.fNumber << ",\n"
          << "  \"quantum_efficiency\": " << params.quantumEfficiency << ",\n"
          << "  \"well_capacity_e\": " << params.wellCapacity_e << ",\n"
          << "  \"wavelength_nm\": " << params.wavelength_nm << ",\n"
          << "  \"temporal_statistics_defined\": "
          << (temporalDefined ? "true" : "false") << ",\n"
          << "  \"mean_dn\": " << meanDN << ",\n"
          << "  \"temporal_sigma_dn_median\": " << temporalMedian << ",\n"
          << "  \"temporal_sigma_dn_mean\": " << temporalMean << ",\n"
          << "  \"residual_fpn_dn\": " << residualFpn << ",\n"
          << "  \"single_frame_spatial_sigma_dn\": " << singleFrameSpatial << "\n"
          << "}\n";

    QL_LOG_INFO("sensor_lab: wrote {}_mean.exr, {}_std.exr, {}_stats.json", outPrefix,
                outPrefix, outPrefix);
    return 0;
}
