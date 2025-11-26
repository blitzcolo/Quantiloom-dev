// ============================================================================
// Quantiloom - Closest Hit Shader (PBR with Textures)
// ============================================================================
// Computes Cook-Torrance PBR shading with:
// - Texture sampling (base color, metallic-roughness, normal, emissive)
// - Direct sun lighting from LUT
// - Sky ambient lighting (hemispherical integration approximation)
//
// SPECTRAL RENDERING:
// - Supports multiple rendering modes: single, RGB, MWIR, LWIR
// - RGB mode: Physically-based spectral upsampling + XYZ integration
// - Single mode: Uses spectralAlbedo for single-wavelength rendering
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LUTData> skyLUT;
[[vk::binding(3, 0)]] StructuredBuffer<float3> vertexBuffer;    // Vertex positions
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties
[[vk::binding(6, 0)]] Texture2D textures[];                     // Bindless texture array
[[vk::binding(7, 0)]] SamplerState samplers[];                  // Bindless sampler array
[[vk::binding(8, 0)]] StructuredBuffer<float2> uvBuffer;        // UV coordinates (optional)
[[vk::binding(9, 0)]] StructuredBuffer<float4> tangentBuffer;   // Tangent vectors (optional)
[[vk::binding(10, 0)]] Texture2D<float2> brdfLUT;               // BRDF integration LUT for IBL
[[vk::binding(11, 0)]] SamplerState brdfLUTSampler;             // Sampler for BRDF LUT

// ============================================================================
// Push Constants
// ============================================================================
// Camera data for accessing spectral mode and wavelength
// ============================================================================

[[vk::push_constant]] CameraData camera;

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
// NOTE: In ray tracing, we cannot use automatic LOD (Sample), must use explicit LOD (SampleLevel)
// - Ray tracing shaders don't have screen-space derivatives for automatic LOD selection
// - Currently using LOD 0, but mipmap infrastructure is enabled (trilinear filtering ready)
// - TODO (M2+): Implement ray differentials for accurate texture filtering
//   See: "Ray Differentials" in PBRT-v4 or "Texture Level of Detail Strategies for Real-Time Ray Tracing"
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
        samplers[NonUniformResourceIndex(samplerIndex)], uv, 0.0  // TODO (M2+): Compute LOD from ray differential
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

    // UV coordinates: Use real UVs from buffer (interpolated with barycentrics)
    // Barycentric interpolation: uv = u0 * (1 - b1 - b2) + u1 * b1 + u2 * b2
    float2 uv0 = uvBuffer[idx0];
    float2 uv1 = uvBuffer[idx1];
    float2 uv2 = uvBuffer[idx2];
    float2 uv = uv0 * (1.0 - attribs.bary.x - attribs.bary.y) + uv1 * attribs.bary.x + uv2 * attribs.bary.y;

    // Read tangent from buffer (or fallback to fake tangent)
    float3 worldTangent;
    if (material.normalTextureIndex >= 0) {  // Only compute tangent if normal map is used
        // TODO: Check if tangent buffer is bound (requires push constant or flag)
        // For now, attempt to read from buffer and fall back to fake tangent if data is invalid
        float4 tangent4_0 = tangentBuffer[idx0];
        float4 tangent4_1 = tangentBuffer[idx1];
        float4 tangent4_2 = tangentBuffer[idx2];

        // Barycentric interpolation of tangents
        float4 tangent4 = tangent4_0 * (1.0 - attribs.bary.x - attribs.bary.y) +
                          tangent4_1 * attribs.bary.x +
                          tangent4_2 * attribs.bary.y;

        float3 tangent = tangent4.xyz;
        float handedness = tangent4.w;  // ±1 for bitangent orientation

        // Transform tangent to world space
        float3x3 objectToWorld = (float3x3)ObjectToWorld3x4();
        float3 tangentWorld = mul(tangent, objectToWorld);

        // Validate tangent - if invalid, fall back to fake tangent
        if (isfinite(dot(tangent, tangent)) && dot(tangent, tangent) > 1e-8) {
            worldTangent = SafeNormalize(tangentWorld, float3(1.0, 0.0, 0.0));
        } else {
            // Fallback: fake tangent (for backward compatibility)
            float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
            worldTangent = SafeNormalize(cross(worldNormal, refVector), float3(1.0, 0.0, 0.0));
        }
    } else {
        // No normal map: use fake tangent (doesn't matter since it won't be used)
        float3 refVector = abs(worldNormal.y) > 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
        worldTangent = SafeNormalize(cross(worldNormal, refVector), float3(1.0, 0.0, 0.0));
    }

    // ========================================================================
    // Sample textures
    // ========================================================================

    // Base color texture (glTF: baseColor = baseColorTexture * baseColorFactor)
    // Use white (1,1,1,1) as fallback so multiplication with factor works correctly
    float4 baseColor = SampleTexture(
        material.baseColorTextureIndex,
        material.baseColorTextureIndex,  // Use same index for sampler (1:1 mapping)
        uv,
        float4(1.0, 1.0, 1.0, 1.0)  // White fallback for correct factor multiplication
    );

    // Modulate with base color factor (texture * factor, or 1 * factor if no texture)
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
    // NOTE: glTF 2.0 spec allows emissiveFactor to exceed 1.0 (HDR emissive)
    // This is intentional for self-luminous surfaces (e.g., lights, displays, neon signs)
    // No clamping is applied here; emissive can be arbitrarily high for physically-based rendering
    // The final radiance will be clamped in the validation step to prevent NaN/Inf
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
    // Fetch sun/sky lighting data from LUT
    // ========================================================================

    LUTData lut = skyLUT[0];
    // FIXED: Use SafeNormalize in case LUT data is invalid
    float3 sunDir = SafeNormalize(lut.sunDirection, float3(0.0, 1.0, 0.0));

    // Choose lighting based on spectral mode
    float3 sunRadiance;
    float3 skyRadiance;

    if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
        // RGB mode: Use full RGB lighting
        sunRadiance = lut.sunRadiance_rgb;
        skyRadiance = lut.skyRadiance_rgb;
    } else {
        // Spectral modes: Use scalar spectral radiance (replicate to RGB)
        sunRadiance = float3(lut.sunRadiance_spectral, lut.sunRadiance_spectral, lut.sunRadiance_spectral);
        skyRadiance = float3(lut.skyRadiance_spectral, lut.skyRadiance_spectral, lut.skyRadiance_spectral);
    }

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
    float3 directSun = brdf * sunRadiance * NdotL;

    // ========================================================================
    // Image-Based Lighting (IBL) - Diffuse and Specular
    // ========================================================================

    // Compute F0 (reflectance at normal incidence) for Fresnel calculations
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // ------------------------------------------------------------------------
    // IBL Diffuse (Sky Ambient)
    // ------------------------------------------------------------------------
    // For diffuse IBL, use hemispherical integration approximation
    // kD = energy not reflected specularly (energy conservation)
    float3 kD = (1.0 - FresnelSchlick(F0, max(dot(normal, V), 0.0))) * (1.0 - metallic);
    float3 skyAmbient = kD * albedo / PI * skyRadiance;

    // ------------------------------------------------------------------------
    // IBL Specular (Environment Reflection)
    // ------------------------------------------------------------------------
    // For specular IBL, use split-sum approximation with BRDF LUT
    // Since we don't have a full environment cubemap, we use skyRadiance
    // as a uniform environment (simplified but physically plausible)

    // Compute reflection direction (mirror reflection around normal)
    float3 R = ComputeReflectionDirection(V, normal);

    // Sample environment at reflection direction
    // NOTE: For full IBL, this would sample a prefiltered environment cubemap
    // based on roughness. Here we use uniform skyRadiance (simplified).
    // For rough surfaces, the reflection should be more diffuse, but without
    // a proper environment map, we approximate with uniform sky color.
    float3 prefilteredColor = skyRadiance;  // Simplified: uniform environment

    // Evaluate IBL specular using split-sum approximation with BRDF LUT
    float3 iblSpecular = EvaluateIBLSpecular(
        normal,              // Surface normal
        V,                   // View direction
        F0,                  // Reflectance at normal incidence
        roughness,           // Surface roughness
        prefilteredColor,    // Environment radiance (simplified: uniform sky)
        brdfLUT,             // BRDF integration lookup table
        brdfLUTSampler       // Sampler for BRDF LUT
    );

    // Total outgoing radiance: direct sun + sky ambient + IBL specular + emissive
    float3 radiance = directSun + skyAmbient + iblSpecular + emissive;

    // ========================================================================
    // Spectral Mode Selection: Choose rendering pipeline based on mode
    // ========================================================================

    float3 output_radiance;

    if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
        // ====================================================================
        // RGB Mode: Physically-correct RGB rendering
        // ====================================================================
        // For RGB mode, we directly output the PBR-computed RGB radiance.
        // The radiance is already in linear RGB space (from PBR calculations).
        //
        // NOTE: Full spectral RGB pipeline (with RGB→Spectrum→XYZ→RGB) is
        // TBD as a future enhancement. Current implementation outputs linear RGB.
        // ====================================================================

        // DEBUG: Directly output baseColor texture to verify texture sampling
        // Uncomment to bypass all lighting and see raw texture color
        // output_radiance = baseColor.rgb;

        output_radiance = radiance;

        // FIXED: Validation - clamp and sanitize to prevent NaN/Inf
        if (!isfinite(output_radiance.r) || !isfinite(output_radiance.g) || !isfinite(output_radiance.b)) {
            output_radiance = float3(0.0, 0.0, 0.0);  // Fallback to black
        }
        output_radiance = clamp(output_radiance, 0.0, 1000.0);  // Reasonable HDR range

    } else if (camera.spectral_mode == SPECTRAL_MODE_SINGLE) {
        // ====================================================================
        // Single Wavelength Mode: Grayscale spectral rendering
        // ====================================================================
        // Convert RGB radiance to grayscale for single-wavelength visualization
        // ====================================================================

        float radiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0;

        // Validation
        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);

        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);

    } else if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED || camera.spectral_mode == SPECTRAL_MODE_LWIR_FUSED) {
        // ====================================================================
        // MWIR/LWIR Fusion Mode: Infrared band fusion with blackbody emission
        // ====================================================================
        // Compute self-emission (ε × L_blackbody) + reflected radiance (ρ × L_incident)
        // Uses simplified model: emissivity ≈ spectralAlbedo (Kirchhoff's law approximation)
        // For full quantitative IR, use measured ε(λ)/ρ(λ)/τ(λ) curves (future work)
        // ====================================================================

        float lambda_nm = camera.wavelength_nm;

        // Use spectralAlbedo as IR emissivity (simplified approximation)
        // For quantitative IR, replace with GetIREmissivity(lambda_nm) from IR curve
        float emissivity = material.spectralAlbedo;
        float reflectance = material.spectralAlbedo;

        // Self-emission: ε(λ) × L_blackbody(T, λ)
        float selfEmission = 0.0;
        if (material.irTemperature_K > 0.0) {
            float blackbodyRadiance = IRPlanckRadiance(material.irTemperature_K, lambda_nm);
            selfEmission = emissivity * blackbodyRadiance;
        }

        // Reflected radiance: ρ(λ) × L_incident
        // Convert RGB radiance to scalar for IR (simple average)
        float reflected = (radiance.r + radiance.g + radiance.b) / 3.0 * reflectance;

        float radiance_spectral = selfEmission + reflected;

        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1e6);  // Allow high dynamic range for IR

        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);

    } else {
        // ====================================================================
        // Fallback: Unknown mode or MULTISPECTRAL (TBD)
        // ====================================================================

        float radiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0;

        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);

        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);
    }

    payload.radiance = output_radiance;
}
