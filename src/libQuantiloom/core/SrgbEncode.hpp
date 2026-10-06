/**
 * @file SrgbEncode.hpp
 * @brief IEC 61966-2-1 linear-to-sRGB transfer curve
 *
 * Four writers encoded PNG/display pixels through their own copy of this
 * curve -- ImageIO, the MCP image packer, the USD texture bank, and the
 * camera ISP's display path -- which is how one fixed drifted out of the
 * others' sight. One curve lives here now.
 *
 * The f32 overload applies the curve and nothing else: every f32 caller
 * already clamps to [0, 1] itself, and a value outside the domain should
 * stay visible downstream rather than be silently pinned here. The f64
 * overload keeps the camera ISP's own contract, which clamped inside the
 * encode.
 */

#pragma once

#include "core/Types.hpp"
#include <algorithm>
#include <cmath>

namespace quantiloom {

/// @param linear caller-clamped linear-light value, expected [0, 1]
inline f32 LinearToSrgb(const f32 linear) {
    return linear <= 0.0031308f
        ? 12.92f * linear
        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

/// Same curve in f64, clamping inside as the camera ISP's encoder always did.
inline f64 LinearToSrgb(const f64 linear) {
    const f64 clamped = std::clamp(linear, 0.0, 1.0);
    return clamped <= 0.0031308
        ? 12.92 * clamped
        : 1.055 * std::pow(clamped, 1.0 / 2.4) - 0.055;
}

} // namespace quantiloom
