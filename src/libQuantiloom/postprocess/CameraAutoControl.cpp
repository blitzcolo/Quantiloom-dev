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
        // Total optical scaling needed to land the metered mean on the
        // target. The ratio is multiplicative: exposure * gain * mean ~ target.
        //
        // The metered mean counts a clipped pixel at full scale. The
        // unsaturated mean alone is the right measurement while clipping is
        // incidental -- a glint, the sun disc -- and the wrong one once it is
        // not: with most of the frame clipped, the pixels left are the
        // darkest, their mean is low, and the loop opened up, clipping more.
        // With all of it clipped there was no unsaturated pixel at all, the
        // mean read as no information, and the loop held a white frame
        // forever -- every visible preset in sunlight. Full scale is a lower
        // bound on what a clipped pixel saw, so the loop now always stops
        // down from a clipped frame, and with nothing clipped it is the
        // unsaturated mean exactly as before.
        const f64 clipped = std::isfinite(input.saturatedFraction)
                                ? std::clamp(input.saturatedFraction, 0.0, 1.0)
                                : 0.0;
        const f64 unsaturated =
            std::isfinite(input.lumaMean) ? std::max(input.lumaMean, 0.0) : 0.0;
        const f64 metered = (1.0 - clipped) * unsaturated + clipped;
        f64 ratio = 1.0;
        if (metered > 1e-12)
            ratio = config.targetLuminance / metered;
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
