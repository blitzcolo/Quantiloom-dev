// ============================================================================
// Quantiloom - Closest Hit Shader
// ============================================================================
// Computes Lambert BRDF shading with direct sun lighting from LUT
//
// SPECTRAL RENDERING:
// - Currently: Lambert BRDF is wavelength-independent (albedo / π)
// - Future (M2+): Support wavelength-dependent BRDF (albedo(λ) / π)
// - Wavelength available via camera.wavelength_nm (push constants)
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
    // InstanceID() returns the instanceCustomIndex set in TLAS (see main.cpp)
    // Note: In Vulkan HLSL, InstanceID() corresponds to gl_InstanceCustomIndexEXT
    uint materialID = InstanceID();
    MaterialData material = materials[materialID];
    float albedo_spectral = material.albedo_spectral;  // Spectral reflectance at current λ

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
    // Note: This normal is in Object Space (vertex positions are in object space)
    float3 objectNormal = normalize(cross(edge1, edge2));

    // Transform normal from Object Space to World Space
    // Use transpose of WorldToObject for proper normal transformation
    // (Normals transform by inverse-transpose of model matrix)
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldNormal = normalize(mul(objectNormal, normalTransform));

    // M1: Single-sided geometry, use world normal directly
    // (No faceforward needed - that's for double-sided materials in M2+)
    float3 normal = worldNormal;

    // ========================================================================
    // Fetch sun/sky spectral data from LUT
    // ========================================================================

    LUTData lut = skyLUT[0];
    float3 sunDir = normalize(lut.sunDirection);
    float sunRadiance_spectral = lut.sunRadiance_spectral;  // Spectral radiance at current λ
    float skyRadiance_spectral = lut.skyRadiance_spectral;  // Spectral radiance at current λ

    // ========================================================================
    // Spectral Lambert BRDF shading
    // ========================================================================

    // Lambert BRDF: f(λ) = albedo(λ) / π
    float brdf_spectral = albedo_spectral / 3.14159265;

    // Direct sun lighting: L_out(λ) = BRDF(λ) * L_sun(λ) * (N · L)
    float NdotL = max(dot(normal, sunDir), 0.0);
    float directSun_spectral = brdf_spectral * sunRadiance_spectral * NdotL;

    // Sky ambient lighting (hemispherical integration approximation)
    // For uniform sky: ∫(albedo(λ)/π) * L_sky(λ) * cos(θ) dω ≈ albedo(λ) * L_sky(λ)
    float skyAmbient_spectral = albedo_spectral * skyRadiance_spectral;

    // Total outgoing spectral radiance: direct sun + sky ambient
    // Single-wavelength mode: No shadow rays, no indirect bounces
    float radiance_spectral = directSun_spectral + skyAmbient_spectral;

    // Output as grayscale RGB (all three channels have same value)
    // This allows visualization of single-wavelength renders
    payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
}
