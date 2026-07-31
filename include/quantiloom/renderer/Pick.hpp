// ============================================================================
// Quantiloom - Viewport pick result
// ============================================================================
// Returned by ExternalRenderContext::Pick. A plain aggregate on purpose: the
// frontend needs the answer to "what is under this pixel", not an object.
// No QL_API -- nothing here is code (see ConfigApply.hpp for the pattern).
// ============================================================================

#pragma once

#include "../core/Types.hpp"

#include <glm/glm.hpp>

namespace quantiloom {

/**
 * @brief What the primary camera ray through a pixel hit, if anything
 *
 * Produced by a 1x1 inline ray-query dispatch against the live TLAS, using
 * exactly the raygen shader's ray for that pixel (pixel center, no jitter),
 * so the answer agrees with what is on screen.
 */
struct PickResult {
    bool hit = false;          ///< false: the ray reached TMax without a hit
    u32 nodeIndex = 0;         ///< Index into Scene::nodes (host-mapped)
    u32 instanceIndex = 0;     ///< TLAS instance index (node x primitive)
    u32 primitiveIndex = 0;    ///< Triangle index within the instance
    f32 hitT = 0.0f;           ///< Distance along the normalized primary ray
    glm::vec3 worldPosition{0.0f};  ///< origin + direction * hitT
};

}  // namespace quantiloom
