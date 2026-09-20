#pragma once

/**
 * @file CameraAutoControl.hpp
 * @brief Closed-loop AE/AWB controller for the camera acquisition chain (internal)
 *
 * StepAutoControl is a pure function from (config, previous state, measured
 * statistics) to the next state. One call per committed acquisition advances
 * the feedback; CaptureReprocess and the commitState=false path never call it,
 * so a same-tick reprocess cannot move the loop.
 *
 * AE drives the mean of the UNSATURATED display-domain scalar (the photon
 * chain's well fraction, exactly what camera_stats.comp.hlsl reports in
 * ISP_STAT_MEAN) toward targetLuminance. Exposure is adjusted first, clamped
 * to [minExposureSeconds, maxExposureSeconds]; only the residual the exposure
 * rail cannot absorb moves analogGain, clamped to [1, maxGain]. A first-order
 * IIR with fraction `smoothing` bounds the per-acquisition step.
 *
 * AWB is grey-world: each channel gain is meanG/meanC over the unsaturated
 * pixels, smoothed by the same IIR.
 *
 * The controller is a photon-chain feature. A thermal detector responds to
 * absorbed power rather than scene luminance, so AE has no physical meaning
 * there; the CPU pipeline simply never runs the controller for a thermal
 * device and logs the auto_* flags as ignored.
 */

#include "postprocess/CameraPipeline.hpp"

namespace quantiloom::camera {

// Measured statistics of one acquisition. All means run over the UNSATURATED
// pixels only (dn < 0.98 * adcMax), mirroring camera_stats.comp.hlsl: the
// luma mean is the display-domain scalar mean (ISP_STAT_MEAN) and the
// channel means are the same scalar gathered per CFA/device channel, before
// any white-balance gain is applied.
struct AutoControlInput {
    f64 lumaMean = 0.0;
    f64 saturatedFraction = 0.0;
    std::array<f64, 3> channelMeans{0.0, 0.0, 0.0};
};

struct AutoControlState {
    f64 exposure = 0.0;
    f64 analogGain = 1.0;
    std::array<f64, 3> whiteBalance{1.0, 1.0, 1.0};
};

// enableAe=false leaves exposure and analogGain at `previous`;
// enableAwb=false leaves whiteBalance at `previous`. A non-finite or
// non-positive measured mean is "no information": the corresponding rail
// holds its previous value rather than extrapolating toward infinity.
[[nodiscard]] AutoControlState StepAutoControl(const IspConfig& isp,
                                               const AutoControlState& previous,
                                               const AutoControlInput& input,
                                               bool enableAe, bool enableAwb);

} // namespace quantiloom::camera
