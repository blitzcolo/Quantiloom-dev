/**
 * @file MotionSpec.hpp
 * @brief Turning a `[…motion]` table into a MotionSpec
 *
 * Separate from Motion.cpp because evaluation has no business knowing what
 * TOML is, and separate from ConfigResolve.cpp because both `[models.motion]`
 * and `[nodes.motion]` are the same grammar read in two places.
 *
 * Nothing here fails: an unreadable trajectory yields no spec and a list of
 * warnings, and the caller decides what severity that is. The one thing it
 * will not do is guess -- a key it does not recognise is reported, never
 * quietly dropped.
 */

#pragma once

#include "core/Config.hpp"
#include "scene/Motion.hpp"

#include <functional>
#include <optional>

namespace quantiloom::scene {

/**
 * @brief A time, written as seconds or with a unit
 *
 * `4`, `4.5`, `"15s"`, `"90min"`, `"36h"`, `"2.5d"`. Units exist because a
 * timeline that spans a month is as ordinary here as one that spans ten
 * seconds, and `end_s = 2592000` is a number nobody checks.
 */
[[nodiscard]] Result<f64, String> ParseDurationString(StringView text);

/**
 * @brief Read one duration-valued key
 *
 * @return the value in seconds, or nullopt when the key is absent or
 *         malformed. A malformed one appends to @p warnings.
 */
[[nodiscard]] std::optional<f64> ReadDuration(const Config& table, StringView key,
                                              const String& owner, Vector<String>& warnings);

/// How a path a motion table named is turned into one that can be opened.
/// Supplied by the caller so this file stays clear of the resolver's
/// config-directory convention.
using MotionPathResolver = std::function<String(const String&)>;

/**
 * @brief Read a motion table
 *
 * The three forms are alternatives; when more than one is present the order
 * keys > segments > engines decides, with a warning.
 *
 * @param origin_s  the timeline start, which is the zero of the engines' clock
 * @return nullopt when the table describes no motion at all
 */
[[nodiscard]] std::optional<MotionSpec> ParseMotionSpec(const Config& table, f64 origin_s,
                                                        const String& owner,
                                                        const MotionPathResolver& resolvePath,
                                                        Vector<String>& warnings);

/**
 * @brief Read keyframes out of a CSV
 *
 * Columns `t,x,y,z` and optionally `qx,qy,qz,qw`. Blank lines and lines
 * starting with `#` are skipped, and a first line whose first field is not a
 * number is taken for a header. Times are plain seconds -- a file with a
 * thousand rows in it was written by a program, and a program does not need
 * `"2.5d"`.
 */
[[nodiscard]] Result<Vector<MotionKey>, String> LoadMotionKeysCsv(const String& path);

}  // namespace quantiloom::scene
