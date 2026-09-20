#include "postprocess/CameraAutoControl.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::camera {
namespace {

f64 ClampFinite(f64 value, f64 lo, f64 hi) {
    if (!std::isfinite(value)) return lo;
    return std::clamp(value, lo, hi);
}

} // namespace

AutoControlState StepAutoControl(const IspConfig& isp,
                                 const AutoControlState& previous,
                                 const AutoControlInput& input,
                                 const bool enableAe, const bool enableAwb) {
    const AutoControlConfig& config = isp.autoControl;
    AutoControlState next = previous;

    if (enableAe) {
        // Total optical scaling needed to land the unsaturated mean on the
        // target. The ratio is multiplicative: exposure * gain * mean ~ target.
        f64 ratio = 1.0;
        if (std::isfinite(input.lumaMean) && input.lumaMean > 1e-12)
            ratio = config.targetLuminance / input.lumaMean;
        if (std::isfinite(ratio) && ratio > 0.0) {
            // Exposure rail first: try to absorb the whole correction with
            // exposure alone, clamped to its range.
            const f64 exposureTarget =
                ClampFinite(previous.exposure * ratio,
                            config.minExposureSeconds,
                            config.maxExposureSeconds);
            // Residual the exposure rail could not absorb moves the gain
            // rail, so gain only ever changes once exposure sits at a limit.
            f64 gainTarget = previous.analogGain;
            if (exposureTarget > 0.0 && previous.exposure > 0.0)
                gainTarget = ClampFinite(
                    previous.analogGain * ratio *
                        (previous.exposure / exposureTarget),
                    1.0, config.maxGain);
            const f64 s = config.smoothing;
            next.exposure = std::isfinite(previous.exposure) ?
                previous.exposure + s * (exposureTarget - previous.exposure) :
                exposureTarget;
            next.analogGain = std::isfinite(previous.analogGain) ?
                previous.analogGain + s * (gainTarget - previous.analogGain) :
                gainTarget;
            // The IIR is a convex combination, but clamp once more so a
            // pathological previous value can never escape the rails.
            next.exposure = ClampFinite(next.exposure,
                                        config.minExposureSeconds,
                                        config.maxExposureSeconds);
            next.analogGain = ClampFinite(next.analogGain, 1.0, config.maxGain);
        }
    }

    if (enableAwb) {
        const f64 green = input.channelMeans[1];
        const f64 s = config.smoothing;
        for (size_t c = 0; c < 3; ++c) {
            const f64 mean = input.channelMeans[c];
            if (!std::isfinite(mean) || mean <= 1e-12 ||
                !std::isfinite(green) || green <= 1e-12)
                continue; // no information for this channel: hold
            const f64 target = ClampFinite(green / mean, 1e-3, 1e3);
            next.whiteBalance[c] = previous.whiteBalance[c] +
                s * (target - previous.whiteBalance[c]);
            if (!std::isfinite(next.whiteBalance[c]) ||
                next.whiteBalance[c] <= 0.0)
                next.whiteBalance[c] = previous.whiteBalance[c];
        }
    }

    return next;
}

} // namespace quantiloom::camera
