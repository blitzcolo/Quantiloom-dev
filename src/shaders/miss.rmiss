// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Uses spectral radiance from LUT (single wavelength)
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
    // Fetch sky spectral radiance from LUT
    // M1: Single entry LUT (index 0), hemispherical average
    LUTData lut = skyLUT[0];

    // Return sky background (spectral mode: scalar to grayscale RGB)
    float radiance_spectral = lut.skyRadiance_spectral;
    payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
}
