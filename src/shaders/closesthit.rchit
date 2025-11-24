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

// Safe normalize: returns fallback if vector is near-zero to prevent NaN/Inf
// This is CRITICAL for GPU stability - normalize() on zero vectors causes crashes
float3 SafeNormalize(float3 v, float3 fallback) {
    float lenSq = dot(v, v);
    if (lenSq < 1e-8) {
        return fallback;
    }
    return v * rsqrt(lenSq);
}

// Safe normalize with default fallback to up vector
float3 SafeNormalize(float3 v) {
    return SafeNormalize(v, float3(0.0, 1.0, 0.0));
}

// Maximum valid texture index (must match MAX_TEXTURES in RayTracingPipeline.cpp)
// CRITICAL: This bounds check prevents GPU hangs from invalid descriptor access
static const int MAX_TEXTURE_INDEX = 1024;

// Sample texture with fallback for invalid indices
// Note: Use SampleLevel instead of Sample for ray tracing shaders (explicit LOD required)
// FIXED: Added upper bound check to prevent access to unbound descriptors
// If texture index is garbage (e.g., due to struct misalignment), this prevents GPU hang
float4 SampleTexture(int textureIndex, int samplerIndex, float2 uv, float4 fallback) {
    // Check both lower AND upper bounds to prevent invalid descriptor access
    // Invalid indices (negative or out-of-range) can cause GPU hangs with PARTIALLY_BOUND descriptors
    if (textureIndex < 0 || textureIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    // Ensure sampler index is also valid (use same index as texture for 1:1 mapping)
    if (samplerIndex < 0 || samplerIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        samplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0  // LOD 0 (no mipmapping in M1)
    );
}

// Compute TBN matrix for normal mapping (Gram-Schmidt orthogonalization)
// N: geometric normal, T: tangent, returns orthonormal TBN matrix
// FIXED: Use SafeNormalize to prevent NaN when vectors are near-parallel
float3x3 ComputeTBN(float3 N, float3 T) {
    // Orthogonalize tangent with respect to normal (Gram-Schmidt)
    // Use SafeNormalize to handle edge case where T is parallel to N
    float3 T_ortho = T - N * dot(N, T);
    T = SafeNormalize(T_ortho, T);  // Fallback to original T if orthogonalized is zero

    // Compute bitangent and NORMALIZE it
    // FIXED: cross(N, T) must be normalized for correct TBN transform
    float3 B = SafeNormalize(cross(N, T), cross(N, float3(1.0, 0.0, 0.0)));

    return float3x3(T, B, N);
}

// Transform normal from tangent space to world space
// FIXED: Added safety checks to prevent NaN propagation
float3 ApplyNormalMap(float3 tangentNormal, float3 worldNormal, float3 worldTangent) {
    // Validate tangent normal before transformation
    // If tangent normal is degenerate, return geometric normal
    float tangentLenSq = dot(tangentNormal, tangentNormal);
    if (tangentLenSq < 1e-8 || !isfinite(tangentLenSq)) {
        return worldNormal;
    }

    // Build TBN matrix
    float3x3 TBN = ComputeTBN(worldNormal, worldTangent);

    // Transform tangent-space normal to world space
    float3 normal = mul(tangentNormal, TBN);

    // Use SafeNormalize with geometric normal as fallback
    return SafeNormalize(normal, worldNormal);
}

// ============================================================================
// Closest Hit Entry Point
// ============================================================================

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
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
    // FIXED: Use SafeNormalize to handle degenerate triangles (collinear vertices)
    float3 crossProduct = cross(edge1, edge2);
    float3 objectNormal = SafeNormalize(crossProduct, float3(0.0, 1.0, 0.0));

    // Transform normal to world space
    // FIXED: Use SafeNormalize to prevent NaN propagation
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldNormal = SafeNormalize(mul(objectNormal, normalTransform), float3(0.0, 1.0, 0.0));

    // TODO (Phase 3.5): UVs and normals are not yet in vertex buffer
    // For M1, we use geometric normals and fake UVs
    // This will be fixed when vertex buffer includes full vertex attributes

    // Fake UVs (planar projection for testing)
    float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float2 uv = hitPoint.xy * 0.1;  // Simple planar mapping

    // Fake tangent (will be replaced with proper vertex tangent in M2+)
    // CRITICAL: Choose reference vector based on normal direction to avoid degenerate cross product
    float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    // FIXED: Use SafeNormalize to prevent crash when cross product is near-zero
    float3 worldTangent = SafeNormalize(cross(worldNormal, refVector), float3(1.0, 0.0, 0.0));

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

        // Clamp normalScale to reasonable range to prevent extreme values
        float clampedNormalScale = clamp(material.normalScale, 0.0, 10.0);
        tangentNormal.xy *= clampedNormalScale;

        // FIXED: Use SafeNormalize to handle edge case where tangent normal becomes near-zero
        // This can happen with extreme normalScale values or degenerate texture data
        tangentNormal = SafeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));

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
    // FIXED: Use SafeNormalize in case LUT data is invalid
    float3 sunDir = SafeNormalize(lut.sunDirection, float3(0.0, 1.0, 0.0));
    float sunRadiance_spectral = lut.sunRadiance_spectral;
    float skyRadiance_spectral = lut.skyRadiance_spectral;

    // ========================================================================
    // PBR Shading
    // ========================================================================

    // View direction (FROM surface TO camera)
    // FIXED: Use SafeNormalize to handle edge cases
    float3 V = SafeNormalize(-WorldRayDirection(), float3(0.0, 0.0, 1.0));

    // Light direction (FROM surface TO sun)
    float3 L = sunDir;

    // FIXED: Final validation of normal - if still invalid, use geometric normal
    if (!isfinite(dot(normal, normal)) || dot(normal, normal) < 1e-8) {
        normal = worldNormal;
    }

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

    // FIXED: Final validation - clamp and sanitize output to prevent NaN/Inf propagation
    // NaN/Inf values can cause GPU hangs or corrupt the entire output image
    if (!isfinite(radiance_spectral)) {
        radiance_spectral = 0.0;  // Fallback to black for invalid pixels
    }
    radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);  // Reasonable HDR range

    payload.radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
}
