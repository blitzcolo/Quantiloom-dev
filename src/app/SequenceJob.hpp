/**
 * @file SequenceJob.hpp
 * @brief Many frames of one scene, on one device
 *
 * `batch` renders a list of configurations, and each of them gets its own
 * renderer: its own device, its own scene load, its own thermal setup. That is
 * right for a list of different scenes and wrong for a list of instants of the
 * same one -- a hundred ticks of a truck driving past would build the same
 * acceleration structure a hundred times and restart the same trajectory a
 * hundred times.
 *
 * `sequence` is the other shape. One config, one renderer, the clock moved
 * between frames. The thermal solve in particular is stepped forward rather
 * than replayed, so rendering a day in order costs about what rendering its
 * last hour costs.
 *
 * A config with no `[timeline]` has nothing to sequence and is refused rather
 * than rendered once.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

#include <optional>

namespace quantiloom::app {

struct SequenceOptions {
    String configPath;

    /// Inclusive tick range. Absent means the whole of the timeline.
    std::optional<i64> fromTick;
    std::optional<i64> toTick;
    /// Render every Nth tick. One is every tick.
    u32 every = 1;

    /// Where the frames go. Empty takes `renderer.output` and inserts
    /// `_{tick:05}` before its extension.
    String outputTemplate;

    String atmosphereModelPackFallback;

    /// List the frames and stop, before rendering any of them. The renderer is
    /// still built: the tick range and the default template come out of the
    /// resolved config, and reading those keys a second way here is the
    /// divergence this project keeps closing.
    bool dryRun = false;
};

/**
 * @brief Fill in a frame's name
 *
 * `{tick}` and `{time_s}`, each optionally with a format: `{tick:05}` pads to
 * five digits, `{time_s:.3f}` gives three decimals. Anything else is copied
 * through, so a template with no placeholder in it produces the same name for
 * every frame -- which is a mistake the caller is warned about rather than one
 * this function invents a placeholder to prevent.
 */
[[nodiscard]] String FormatFrameName(StringView tmpl, i64 tick, f64 time_s);

/// @return the process exit code: 0 when every frame was written.
int RunSequence(const SequenceOptions& options);

}  // namespace quantiloom::app
