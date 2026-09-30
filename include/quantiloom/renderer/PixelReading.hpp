#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>

namespace quantiloom {

/// A raw accumulation pixel and the image it was copied from. Coordinates
/// are in the host's target extent, before mapping to the render extent.
struct PixelReading {
    u64 requestId = 0;
    u64 imageGeneration = 0;
    u64 acquisitionIndex = 0;
    u32 frameIndex = 0;
    u32 accumulatedSamples = 0;
    u32 x = 0;
    u32 y = 0;
    glm::vec4 value{};
};

} // namespace quantiloom
