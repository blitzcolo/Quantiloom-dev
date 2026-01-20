// ============================================================================
// Quantiloom - Shadow Miss Shader
// ============================================================================
// Called when a shadow ray misses all geometry (ray reached light source)
// This shader sets the isShadowed flag to 0 to indicate the point is lit
// ============================================================================

#include "common.hlsli"

// ============================================================================
// Shadow Miss Entry Point
// ============================================================================
// When a shadow ray misses all geometry, it means:
// - The ray traveled from hit point toward light without occlusion
// - The surface point is NOT in shadow (fully lit by this light)
//
// We set isShadowed = 0 to indicate "not shadowed"
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    // Ray reached light without hitting anything - NOT shadowed
    payload.isShadowed = 0;
}
