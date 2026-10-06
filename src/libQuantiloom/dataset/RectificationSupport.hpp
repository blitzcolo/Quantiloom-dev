#pragma once
#include "scene/CameraProjection.hpp"
#include <algorithm>
#include <cmath>

namespace quantiloom::dataset::detail {
// All consumers use the same support as the bilinear image resampler. At an
// exact last pixel centre the repeated neighbour is the same in-bounds texel.
inline bool HasRectificationSupport(const camera::CameraProjection& native, glm::dvec2 pixel,
                                    u32 width, u32 height) {
    if (!width || !height || !std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < .5 ||
        pixel.y < .5 || pixel.x > width - .5 || pixel.y > height - .5)
        return false;
    const auto x0 = static_cast<u32>(pixel.x - .5), y0 = static_cast<u32>(pixel.y - .5);
    const auto x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    return camera::UnprojectPixel(native, {x0 + .5, y0 + .5}).valid &&
           camera::UnprojectPixel(native, {x1 + .5, y0 + .5}).valid &&
           camera::UnprojectPixel(native, {x0 + .5, y1 + .5}).valid &&
           camera::UnprojectPixel(native, {x1 + .5, y1 + .5}).valid;
}
}
