/**
 * @file LogTap.hpp
 * @brief A hook on the log stream, for tests that assert what was said
 *
 * Internal. A warning is part of a loader's contract -- "this input is not
 * applied, and here is why" -- and a contract nobody can test drifts. The tap
 * sees every message before the level filter, so a test can capture an INFO
 * line while the suite runs at WARN.
 *
 * Not in include/quantiloom/ and not exported: the suite links the object
 * library, and no host has a reason to read the log by callback.
 */

#pragma once

#include "core/Log.hpp"

#include <functional>
#include <string_view>

namespace quantiloom::logtap {

using Fn = std::function<void(Log::Level, std::string_view)>;

/// Install a tap, or clear it with an empty function. One at a time; the
/// caller owns the lifetime and must clear it before what it captures into
/// goes away.
void Set(Fn fn);

}  // namespace quantiloom::logtap
