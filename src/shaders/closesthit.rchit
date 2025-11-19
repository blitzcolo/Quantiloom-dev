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
[[vk::binding(3, 0)]] StructuredBuffer<float3> vertexBuffer;    // Vertex positions
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties

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
    // Fetch material properties from buffer using instance ID
    // InstanceID() returns the instanceCustomIndex set in TLAS (see main_m1_test.cpp)
    // Note: In Vulkan HLSL, InstanceID() corresponds to gl_InstanceCustomIndexEXT
    uint materialID = InstanceID();
    MaterialData material = materials[materialID];
    float3 albedo = material.albedo;

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

    // Use geometric normal directly (trust CCW winding order)
    // With correct winding, cross(edge1, edge2) produces outward-facing normals
    float3 normal = geometricNormal;

    // ========================================================================
    // Fetch sun/sky data from LUT
    // ========================================================================

    LUTData lut = skyLUT[0];
    float3 sunDir = normalize(lut.sunDirection);
    float3 sunRadiance = lut.sunRadiance;
    float3 skyRadiance = lut.skyRadiance;

    // ========================================================================
    // Lambert BRDF shading
    // ========================================================================

    // Lambert BRDF: f = albedo / pi
    float3 brdf = albedo / 3.14159265;

    // Direct sun lighting: L_out = BRDF * L_sun * (N · L)
    float NdotL = max(dot(normal, sunDir), 0.0);
    float3 directSun = brdf * sunRadiance * NdotL;

    // Sky ambient lighting (hemispherical integration approximation)
    // For uniform sky: ∫(albedo/π) * L_sky * cos(θ) dω ≈ albedo * L_sky
    float3 skyAmbient = albedo * skyRadiance;

    // Total outgoing radiance: direct sun + sky ambient
    // M1: No shadow rays (all surfaces receive sun), no indirect bounces
    payload.radiance = directSun + skyAmbient;
}
