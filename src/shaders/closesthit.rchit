// ============================================================================
// Quantiloom - Closest Hit Shader (PBR with Textures)
// ============================================================================
// Computes Cook-Torrance PBR shading with:
// - Texture sampling (base color, metallic-roughness, normal, emissive)
// - Direct sun lighting from LUT
// - Sky ambient lighting (hemispherical integration approximation)
//
// SPECTRAL RENDERING (M1 compatibility):
// - Uses spectralAlbedo for single-wavelength rendering
// - Future (M2+): Support wavelength-dependent BRDF
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LUTData> skyLUT;
[[vk::binding(3, 0)]] StructuredBuffer<float3> vertexBuffer;    // Vertex positions
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties
[[vk::binding(6, 0)]] Texture2D textures[];                     // Bindless texture array
[[vk::binding(7, 0)]] SamplerState samplers[];                  // Bindless sampler array

// ============================================================================
// Hit Attributes
// ============================================================================
// Barycentric coordinates of hit point within triangle
// ============================================================================

struct HitAttributes {
    float2 bary;  // Barycentric coordinates (b1, b2), where b0 = 1 - b1 - b2
};

// ============================================================================
// Helper Functions
// ============================================================================

// Sample texture with fallback for invalid indices
// Note: Use SampleLevel instead of Sample for ray tracing shaders (explicit LOD required)
float4 SampleTexture(int textureIndex, int samplerIndex, float2 uv, float4 fallback) {
    if (textureIndex < 0) {
        return fallback;
    }
    return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        samplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0  // LOD 0 (no mipmapping in M1)
    );
}

// Compute TBN matrix for normal mapping (Gram-Schmidt orthogonalization)
// N: geometric normal, T: tangent, returns orthonormal TBN matrix
float3x3 ComputeTBN(float3 N, float3 T) {
    // Orthogonalize tangent with respect to normal (Gram-Schmidt)
    T = normalize(T - N * dot(N, T));

    // Compute bitangent
    float3 B = cross(N, T);

    return float3x3(T, B, N);
}

// Transform normal from tangent space to world space
float3 ApplyNormalMap(float3 tangentNormal, float3 worldNormal, float3 worldTangent) {
    // Build TBN matrix
    float3x3 TBN = ComputeTBN(worldNormal, worldTangent);

    // Transform tangent-space normal to world space
    float3 normal = mul(tangentNormal, TBN);

    return normalize(normal);
}

// ============================================================================
// Closest Hit Entry Point
// ============================================================================

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    // TEMPORARY DEBUG: Simplified closesthit - just return red color
    #if 0
    // ========================================================================
    // Fetch material properties
    // ========================================================================

    uint materialID = InstanceID();
    MaterialData material = materials[materialID];

    // ========================================================================
    // Compute geometric normal from triangle vertices
    // ========================================================================

    uint primitiveID = PrimitiveIndex();

    // Read triangle indices
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

    // Compute geometric normal (object space)
    float3 objectNormal = normalize(cross(edge1, edge2));

    // Transform normal to world space
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldNormal = normalize(mul(objectNormal, normalTransform));

    // TODO (Phase 3.5): UVs and normals are not yet in vertex buffer
    // For M1, we use geometric normals and fake UVs
    // This will be fixed when vertex buffer includes full vertex attributes

    // Fake UVs (planar projection for testing)
    float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float2 uv = hitPoint.xy * 0.1;  // Simple planar mapping

    // Fake tangent (will be replaced with proper vertex tangent in M2+)
    // CRITICAL: Choose reference vector based on normal direction to avoid degenerate cross product
    float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 worldTangent = normalize(cross(worldNormal, refVector));

    // ========================================================================
    // Sample textures
    // ========================================================================

    // Base color texture
    float4 baseColor = SampleTexture(
        material.baseColorTextureIndex,
        material.baseColorTextureIndex,  // Use same index for sampler (1:1 mapping)
        uv,
        material.baseColorFactor
    );

    // Modulate with base color factor
    baseColor *= material.baseColorFactor;

    // Metallic-Roughness texture (G=roughness, B=metallic)
    float4 metallicRoughness = SampleTexture(
        material.metallicRoughnessTextureIndex,
        material.metallicRoughnessTextureIndex,
        uv,
        float4(1.0, material.roughnessFactor, material.metallicFactor, 1.0)
    );

    float roughness = material.roughnessFactor * metallicRoughness.g;
    float metallic = material.metallicFactor * metallicRoughness.b;

    // Normal map (tangent space, [0,1] -> [-1,1])
    float3 normal = worldNormal;
    if (material.normalTextureIndex >= 0) {
        float3 tangentNormal = SampleTexture(
            material.normalTextureIndex,
            material.normalTextureIndex,
            uv,
            float4(0.5, 0.5, 1.0, 1.0)  // Default: pointing up in tangent space
        ).xyz;

        // Convert [0,1] to [-1,1]
        tangentNormal = tangentNormal * 2.0 - 1.0;
        tangentNormal.xy *= material.normalScale;
        tangentNormal = normalize(tangentNormal);

        // Transform to world space
        normal = ApplyNormalMap(tangentNormal, worldNormal, worldTangent);
    }

    // Emissive texture
    float3 emissive = material.emissiveFactor;
    if (material.emissiveTextureIndex >= 0) {
        emissive *= SampleTexture(
            material.emissiveTextureIndex,
            material.emissiveTextureIndex,
            uv,
            float4(1.0, 1.0, 1.0, 1.0)
        ).rgb;
    }

    // ========================================================================
    // Fetch sun/sky spectral data from LUT
    // ========================================================================

    LUTData lut = skyLUT[0];
    float3 sunDir = normalize(lut.sunDirection);
    float sunRadiance_spectral = lut.sunRadiance_spectral;
    float skyRadiance_spectral = lut.skyRadiance_spectral;

    // ========================================================================
    // PBR Shading
    // ========================================================================

    // View direction (FROM surface TO camera)
    float3 V = -normalize(WorldRayDirection());

    // Light direction (FROM surface TO sun)
    float3 L = sunDir;

    // Compute PBR BRDF (Cook-Torrance)
    float3 albedo = baseColor.rgb;
    float3 brdf = CookTorranceBRDF(normal, V, L, albedo, metallic, roughness);

    // Direct sun lighting: L_out = BRDF * L_sun * (N · L)
    float NdotL = max(dot(normal, L), 0.0);
    float3 directSun = brdf * sunRadiance_spectral * NdotL;

    // Sky ambient lighting (approximate hemispherical integration)
    // For PBR, we use the diffuse term only (specular requires IBL in M2+)
    float3 kD = (1.0 - FresnelSchlick(
        lerp(float3(0.04, 0.04, 0.04), albedo, metallic),
        max(dot(normal, V), 0.0)
    )) * (1.0 - metallic);
    float3 skyAmbient = kD * albedo / PI * skyRadiance_spectral;

    // Total outgoing radiance: direct sun + sky ambient + emissive
    // For M1: Single-wavelength mode, output as grayscale RGB
    float3 radiance = directSun + skyAmbient + emissive;

    // Spectral mode: Convert to grayscale for visualization
    // (All channels should have similar values for spectral rendering)
    float radiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0;

    payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    #else
    // DEBUG LEVEL 3: Test texture sampling with simple lighting
    uint materialID = InstanceID();
    MaterialData material = materials[materialID];

    uint primitiveID = PrimitiveIndex();

    // Read triangle indices
    uint idx0 = indexBuffer[primitiveID * 3 + 0];
    uint idx1 = indexBuffer[primitiveID * 3 + 1];
    uint idx2 = indexBuffer[primitiveID * 3 + 2];

    // Read vertex positions
    float3 v0 = vertexBuffer[idx0];
    float3 v1 = vertexBuffer[idx1];
    float3 v2 = vertexBuffer[idx2];

    // Compute geometric normal
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 objectNormal = normalize(cross(edge1, edge2));

    // Transform to world space
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldNormal = normalize(mul(objectNormal, normalTransform));

    // Compute UVs (fake planar projection)
    float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float2 uv = hitPoint.xy * 0.1;

    // Sample base color texture
    float3 albedo = material.baseColorFactor.rgb;
    if (material.baseColorTextureIndex >= 0) {
        float4 texColor = SampleTexture(
            material.baseColorTextureIndex,
            material.baseColorTextureIndex,
            uv,
            material.baseColorFactor
        );
        albedo = texColor.rgb * material.baseColorFactor.rgb;
    }

    // Get metallic and roughness (no texture sampling for now to isolate issue)
    float roughness = material.roughnessFactor;
    float metallic = material.metallicFactor;

    // Fetch sun/sky data
    LUTData lut = skyLUT[0];
    float3 sunDir = normalize(lut.sunDirection);
    float sunRadiance_spectral = lut.sunRadiance_spectral;
    float skyRadiance_spectral = lut.skyRadiance_spectral;

    // View direction
    float3 V = -normalize(WorldRayDirection());
    float3 L = sunDir;

    // CRITICAL TEST: Full PBR BRDF - this likely causes GPU timeout
    float3 brdf = CookTorranceBRDF(worldNormal, V, L, albedo, metallic, roughness);

    // Direct sun lighting
    float NdotL = max(dot(worldNormal, L), 0.0);
    float3 directSun = brdf * sunRadiance_spectral * NdotL;

    // Sky ambient (simplified)
    float3 kD = (1.0 - FresnelSchlick(
        lerp(float3(0.04, 0.04, 0.04), albedo, metallic),
        max(dot(worldNormal, V), 0.0)
    )) * (1.0 - metallic);
    float3 skyAmbient = kD * albedo / PI * skyRadiance_spectral;

    // Total radiance
    float3 radiance = directSun + skyAmbient;
    float radiance_spectral_out = (radiance.r + radiance.g + radiance.b) / 3.0;

    payload.radiance = float3(radiance_spectral_out, radiance_spectral_out, radiance_spectral_out);
    #endif
}
