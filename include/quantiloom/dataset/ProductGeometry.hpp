#pragma once
#include "core/Platform.hpp"
#include "scene/Camera.hpp"

namespace quantiloom::dataset {

/// Native product grid. Right/down/forward camera coordinates; top-left image
/// origin, pixel centres at (x+0.5,y+0.5), row-major matrices, world-space pose.
/// No exposure integration, rectification or resampling is implied.
struct QL_API ProductGeometry {
    u32 version = 1;
    u32 width = 0, height = 0;
    f64 referenceTimeSeconds = 0.0;
    f64 worldUnitsToMeters = 1.0;
    CameraData camera{};

    [[nodiscard]] Result<String, String> ToJson() const;
};
} // namespace quantiloom::dataset
