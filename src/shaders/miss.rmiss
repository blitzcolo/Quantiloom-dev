// ============================================================================
// Quantiloom M1 - Miss Shader
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// ============================================================================

#include "common.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LUTData> skyLUT;

// ============================================================================
// Miss Entry Point
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    // Fetch sky radiance from LUT
    // M1: Single entry LUT (index 0), hemispherical average
    LUTData lut = skyLUT[0];

    // Return sky background
    payload.radiance = lut.skyRadiance;
}
