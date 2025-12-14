// ============================================================================
// Quantiloom M1 - Miss Shader (Spectral Rendering)
// ============================================================================
// Returns sky background radiance when ray misses all geometry
// Supports both RGB and spectral modes
// ============================================================================

#include "common.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LUTData> skyLUT;

// ============================================================================
// Push Constants
// ============================================================================
// Camera data for accessing spectral mode
// ============================================================================

[[vk::push_constant]] CameraData camera;

// ============================================================================
// Miss Entry Point
// ============================================================================

[shader("miss")]
void main(inout Payload payload) {
    // Fetch sky radiance from LUT
    LUTData lut = skyLUT[0];

    // Choose sky radiance based on spectral mode
    if (camera.spectral_mode == SPECTRAL_MODE_RGB_FUSED) {
        // RGB mode: Use full RGB sky radiance
        payload.radiance = lut.skyRadiance_rgb;
    } else {
        // Spectral modes: Use scalar spectral radiance (replicate to RGB)
        float radiance_spectral = lut.skyRadiance_spectral;
        payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }
}
