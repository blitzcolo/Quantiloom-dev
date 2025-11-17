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
[[vk::binding(3, 0)]] StructuredBuffer<float3> vertexBuffer;  // Vertex positions
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;      // Triangle indices

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

    // ========================================================================
    // Compute geometric normal from triangle vertices
    // ========================================================================

    // Get triangle primitive ID
    uint primitiveID = PrimitiveIndex();

    // Read triangle indices (3 indices per triangle)
    uint idx0 = indexBuffer[primitiveID * 3 + 0];
    uint idx1 = indexBuffer[primitiveID * 3 + 1];
    uint idx2 = indexBuffer[primitiveID * 3 + 2];

    // Read vertex positions
    float3 v0 = vertexBuffer[idx0];
    float3 v1 = vertexBuffer[idx1];
    float3 v2 = vertexBuffer[idx2];

    // Compute edge vectors
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;

    // Compute geometric normal via cross product (CCW winding)
    float3 geometricNormal = normalize(cross(edge1, edge2));

    // Ensure normal faces the ray (front-facing)
    // If ray hits back face, flip the normal
    float3 rayDir = WorldRayDirection();
    if (dot(geometricNormal, rayDir) > 0.0) {
        geometricNormal = -geometricNormal;
    }

    float3 normal = geometricNormal;

    // ========================================================================
    // Fetch sun/sky data from LUT
    // ========================================================================

    LUTData lut = skyLUT[0];
    float3 sunDir = normalize(lut.sunDirection);
    float3 sunRadiance = lut.sunRadiance;

    // ========================================================================
    // Lambert BRDF shading
    // ========================================================================

    // Lambert BRDF: f = albedo / pi
    float3 brdf = albedo / 3.14159265;

    // Direct lighting: L_out = BRDF * L_in * (N · L)
    float NdotL = max(dot(normal, sunDir), 0.0);
    float3 directLight = brdf * sunRadiance * NdotL;

    // M1: No shadow rays, no indirect lighting
    payload.radiance = directLight;
}
