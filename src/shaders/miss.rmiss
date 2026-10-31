// ============================================================================
// Quantiloom - Miss Shader
// ============================================================================
// Says that a ray missed, and nothing else: raygen evaluates the sky itself
// (ShadeMiss in shade_miss.hlsli, called from path_driver.hlsli) with the ray
// it traced, whose TMax is what RayTCurrent() would report here.
// ============================================================================

#include "common.hlsli"

[shader("miss")]
void main(inout Payload payload) {
    payload.hit.t = -1.0;
}
