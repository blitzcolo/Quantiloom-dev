/**
 * @file TimelineControl.hpp
 * @brief What a host needs to know about the clock a scene is rendered against
 *
 * Plain data, passed by value, with the tick arithmetic inline so that the SDK
 * and every host agree on which tick a second is without any of them
 * re-deriving it. Seconds are canonical; a tick is a frame of the grid those
 * seconds are sampled on, exactly as USD's `timeCodesPerSecond` relates a
 * timecode to a second.
 *
 * The rate is a double because the span is not always a few seconds. A month
 * of thermal history sampled every hundred and fifty seconds is 0.006667 ticks
 * per second, and refusing that in the name of round numbers would mean
 * writing the same scene as 17280 ticks of something else.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

#include <cmath>

namespace quantiloom {

/**
 * @brief The clock, where it stands, and what it drives
 *
 * `present` false means the scene named no `[timeline]` and nothing in it
 * moves. Every other field is then meaningless and a host should show no
 * transport at all rather than a disabled one over made-up numbers.
 */
struct TimelineInfo {
    bool present = false;

    f64 start_s = 0.0;
    f64 end_s = 0.0;
    f64 current_s = 0.0;
    f64 ticksPerSecond = 20.0;

    /// Whether the timeline drives the thermal hour. When it does,
    /// `thermal.time_h` means the hour at `start_s` rather than the hour being
    /// rendered, and the hour being rendered is `currentThermalHour`.
    bool thermalMapped = false;
    f64 thermalHourAtStart = 0.0;
    /// Seconds of thermal simulation per second of timeline.
    f64 thermalTimeScale = 1.0;
    f64 currentThermalHour = 0.0;

    u32 animatedNodeCount = 0;
    u32 modelCount = 0;
    /// Piecewise-static spans the thermal solve is cut into. One means the
    /// geometry never changed, or the scene asked for reference mode.
    u32 thermalEpochCount = 0;
    u32 currentThermalEpoch = 0;

    [[nodiscard]] i64 TickOf(f64 t_s) const {
        return static_cast<i64>(std::llround((t_s - start_s) * ticksPerSecond));
    }

    [[nodiscard]] f64 TimeOfTick(i64 tick) const {
        return start_s + static_cast<f64>(tick) / ticksPerSecond;
    }

    /// Ticks in [start_s, end_s] inclusive of both ends -- a one-second
    /// timeline at 20 tick/s has 21 of them, and rendering it produces 21
    /// frames.
    [[nodiscard]] i64 TickCount() const {
        const i64 last = TickOf(end_s);
        return (last < 0) ? 1 : last + 1;
    }
};

}  // namespace quantiloom
