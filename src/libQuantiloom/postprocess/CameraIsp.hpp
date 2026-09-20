#pragma once

/**
 * @file CameraIsp.hpp
 * @brief CPU image signal processor: RAW DN in, display sRGB out (internal)
 *
 * RunIsp is the full classic ISP display chain for one captured frame. The
 * visible (photon) path runs black-level subtraction, defect-pixel repair,
 * white balance, Malvar-He-Cutler demosaic, color correction, optional
 * denoise/sharpen, tone, gamut handling and sRGB encoding. The infrared
 * (thermal) path takes the corrected scalar, applies an AGC tone operator
 * (Linear/Equalize; CLAHE degrades to Equalize on the CPU chain and says so in
 * the product metadata) and one of the shared display palettes.
 *
 * The palette ramps are control-point tables shared with the HLSL in
 * src/shaders/clahe.comp.hlsl (to become display_palettes.hlsli); the test
 * suite parses the shader text and compares them point for point.
 */

#include "core/Image.hpp"
#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"

#include <array>

namespace quantiloom::camera {

namespace isp {

// Evenly spaced control points, linearly interpolated, exactly as in the HLSL
// ApplyPalette family. A ramp rather than a polynomial fit because the control
// points are the specification.
inline constexpr std::array<std::array<f64, 3>, 8> kIronbowPoints = {{
    {{0.000, 0.000, 0.000}},
    {{0.110, 0.020, 0.260}},
    {{0.300, 0.030, 0.430}},
    {{0.510, 0.060, 0.430}},
    {{0.730, 0.170, 0.310}},
    {{0.900, 0.350, 0.130}},
    {{0.990, 0.640, 0.010}},
    {{1.000, 1.000, 0.850}},
}};

inline constexpr std::array<std::array<f64, 3>, 6> kRainbowPoints = {{
    {{0.0, 0.0, 0.5}},
    {{0.0, 0.0, 1.0}},
    {{0.0, 1.0, 1.0}},
    {{1.0, 1.0, 0.0}},
    {{1.0, 0.0, 0.0}},
    {{0.5, 0.0, 0.0}},
}};

inline constexpr std::array<std::array<f64, 3>, 11> kViridisPoints = {{
    {{0.267, 0.005, 0.329}},
    {{0.283, 0.141, 0.458}},
    {{0.254, 0.265, 0.530}},
    {{0.207, 0.372, 0.553}},
    {{0.164, 0.471, 0.558}},
    {{0.128, 0.567, 0.551}},
    {{0.135, 0.659, 0.518}},
    {{0.267, 0.749, 0.441}},
    {{0.478, 0.821, 0.318}},
    {{0.741, 0.873, 0.150}},
    {{0.993, 0.906, 0.144}},
}};

} // namespace isp

// Maps a normalized [0,1] display scalar to palette RGB. Grey is the identity
// ramp and the only palette that leaves a grey image grey; GreyInverted is
// 1 - t. The ramps are control-point tables shared with the HLSL ApplyPalette
// family (src/shaders/clahe.comp.hlsl, to become display_palettes.hlsli).
[[nodiscard]] std::array<f64, 3> ApplyDisplayPalette(f64 t, DisplayPalette palette);

// Acquisition statistics for the AE/AWB closed loop. All means run over the
// UNSATURATED elements only (raw dn < 0.98 * adcMax), and the value is the
// display-domain scalar -- max(corrected, 0) / fullWell for a photon
// detector, the corrected W for a thermal one -- exactly the quantity
// camera_stats.comp.hlsl reports in ISP_STAT_MEAN. Channel means gather the
// same scalar per CFA/device channel BEFORE any white-balance gain, which is
// what makes grey-world AWB a measurement rather than a feedback identity.
struct AcquisitionStats {
    u64 saturatedCount = 0;
    u64 unsaturatedCount = 0;
    f64 lumaMean = 0.0; // mean over unsaturated elements
    std::array<f64, 3> channelMeans{0.0, 0.0, 0.0}; // per-channel unsaturated means
    std::array<u64, 3> channelCounts{0, 0, 0};
};
[[nodiscard]] AcquisitionStats ComputeAcquisitionStats(const CameraConfig& config,
                                                       const Image& rawDn,
                                                       const Image& corrected);

// Applies the HSV/empirical-effect stage to an encoded-sRGB display image
// in place. Runs after the full display chain (visible sRGB encode or
// infrared palette+encode), so it covers pseudo-colour too; hue offsets on a
// greyscale frame are suppressed by the low-saturation weight and are a no-op
// in practice. With every effect at its default the call is skipped by the
// caller, so an untouched chain stays bit-identical. The empirical effects
// draw on independent counter streams keyed on the acquisition index:
// noise keys on acquisition*2, drift on acquisition>>4, so the same
// acquisition always reproduces the same pattern (a same-tick reprocess is
// bit-identical) and two acquisitions never share one. Never touches RAW,
// corrected or apparent-temperature products -- display only.
[[nodiscard]] Result<Image, String> ApplyHsv(const CameraConfig& config, Image display,
                                             u64 acquisitionIndex);

// Runs the display ISP over one acquisition. rawDn is the quantized CFA/device
// frame (one scalar per pixel for Bayer/Mono, one per channel for
// MultiChannel); it is shape-checked against the corrected signal. The photon
// display branch consumes the corrected device signal -- black level already
// subtracted and NUC tables applied upstream in the readout, so calibration
// stays visible in the display -- normalized by the well capacity; the
// infrared (thermal) branch reads its scalar from the corrected signal too.
// The result is a 3-channel encoded-sRGB image whose metadata records the
// acquisition and any display-chain fallback (e.g. the CLAHE-to-Equalize
// degradation on the CPU chain).
[[nodiscard]] Result<Image, String> RunIsp(
    const CameraConfig& config, const Image& rawDn,
    const Image& corrected, const CaptureState& state, f64 frameTimeSeconds);

} // namespace quantiloom::camera
