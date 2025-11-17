// ============================================================================
// Quantiloom M1 - Closest Hit Shader
// ============================================================================
// Computes Lambert BRDF shading with direct sun lighting from LUT
// ============================================================================

#include "common.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LUTData> skyLUT;

// ============================================================================
// Hit Attributes
// ============================================================================
// Barycentric coordinates of hit point within triangle
// ============================================================================

struct HitAttributes {
    float2 bary;  // Barycentric coordinates (b1, b2), where b0 = 1 - b1 - b2
};

// ============================================================================
// Closest Hit Entry Point
// ============================================================================

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    // M1: Hardcoded surface properties
    // TODO M2+: Fetch from material buffer using InstanceCustomIndex
    float3 albedo = float3(0.8, 0.8, 0.8);  // Diffuse albedo (gray)

    // M1 Simplification: Use camera-facing normal for visible surfaces
    // This ensures all visible faces receive lighting
    // TODO M2+: Compute geometric normal from triangle vertices
    float3 normal = normalize(-WorldRayDirection());

    // Fetch sun/sky data from LUT
    // M1: Single entry LUT (index 0), no wavelength dimension
    LUTData lut = skyLUT[0];

    float3 sunDir = normalize(lut.sunDirection);
    float3 sunRadiance = lut.sunRadiance;

    // Lambert BRDF: f = albedo / pi
    float3 brdf = albedo / 3.14159265;

    // Direct lighting: L_out = BRDF * L_in * (N · L)
    float NdotL = max(dot(normal, sunDir), 0.0);
    float3 directLight = brdf * sunRadiance * NdotL;

    // M1: No shadow rays, no indirect lighting
    payload.radiance = directLight;
}
