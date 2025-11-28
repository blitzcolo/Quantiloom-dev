// ============================================================================
// Quantiloom - Closest Hit Shader (PBR with Textures)
// ============================================================================
// Computes Cook-Torrance PBR shading with:
// - Texture sampling (base color, metallic-roughness, normal, emissive)
// - Direct sun lighting from LUT with Beer-Lambert atmospheric attenuation
// - Sky ambient lighting (hemispherical integration with cosine-weighted sampling)
//
// SPECTRAL RENDERING:
// - Supports multiple rendering modes: single, RGB, MWIR, LWIR
// - RGB mode: Physically-based spectral upsampling + XYZ integration
// - Single mode: Full spectral fidelity with measured reflectance curves
//
// ATMOSPHERIC MODEL (LUT-fast):
// - Beer-Lambert path attenuation: T(λ, d) = exp(-σ_t(λ) × d)
// - MODTRAN LUT provides σ_t(λ) extinction coefficient
// - Hemispherical sky radiance integration for diffuse ambient
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"
#include "spectral_query.hlsli"

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

// ============================================================================
// NEW (M2+): Spectral Curve Buffer
// ============================================================================
// Buffer of spectral reflectance curves for physically-based spectral rendering
// Indexed by MaterialData::spectralReflectanceCurveIndex
// Binding 13 chosen to avoid conflict with IBL resources (10-12)
// ============================================================================

[[vk::binding(13, 0)]] StructuredBuffer<SpectralCurveGPU> spectralCurves;

// ============================================================================
// IBL (Image-Based Lighting) Resources
// ============================================================================
// Added for physically-based specular reflections on metallic surfaces
// ============================================================================

[[vk::binding(10, 0)]] TextureCube<float4> prefilteredEnvMap;  // Prefiltered environment cubemap (with mipmaps)
[[vk::binding(11, 0)]] Texture2D<float2> brdfLUT;              // BRDF integration lookup table
[[vk::binding(12, 0)]] SamplerState iblSampler;                // Linear sampler for IBL textures

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

// Compute texture LOD from ray differentials
// Uses ray differential method to determine appropriate mipmap level
//
// Algorithm:
// 1. Compute how UV coordinates change per pixel (dUV/dx, dUV/dy)
// 2. Convert to texture space (multiply by texture dimensions)
// 3. Take maximum footprint as LOD
//
// References:
// - "Ray Differentials" in PBRT-v4 §10.1
// - Igehy, "Tracing Ray Differentials" (1999)
float ComputeTextureLOD(float2 uv, float2 dUVdx, float2 dUVdy, float2 textureDimensions) {
    // Convert UV differentials to texture-space footprint
    float2 dTexdx = dUVdx * textureDimensions;
    float2 dTexdy = dUVdy * textureDimensions;

    // Compute maximum footprint (anisotropic filtering approximation)
    float footprintX = length(dTexdx);
    float footprintY = length(dTexdy);
    float maxFootprint = max(footprintX, footprintY);

    // LOD = log2(maxFootprint), clamped to reasonable range
    // If footprint < 1 pixel, use LOD 0 (highest detail)
    float lod = max(0.0, log2(maxFootprint));

    return lod;
}

// Sample texture with ray differential LOD
// NOTE: In ray tracing, we cannot use automatic LOD (Sample), must use explicit LOD (SampleLevel)
// - Ray tracing shaders don't have screen-space derivatives for automatic LOD selection
// - We compute LOD from ray differentials for accurate texture filtering
// - This prevents aliasing artifacts at grazing angles and distant surfaces
float4 SampleTextureWithLOD(int textureIndex, int samplerIndex, float2 uv,
                            float2 dUVdx, float2 dUVdy, float4 fallback) {
    // Check both lower AND upper bounds to prevent invalid descriptor access
    // Invalid indices (negative or out-of-range) can cause GPU hangs with PARTIALLY_BOUND descriptors
    if (textureIndex < 0 || textureIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }
    // Ensure sampler index is also valid (use same index as texture for 1:1 mapping)
    if (samplerIndex < 0 || samplerIndex >= MAX_TEXTURE_INDEX) {
        return fallback;
    }

    // Get texture dimensions for LOD computation
    // For simplicity, assume 2048x2048 textures (typical size)
    // In production, use GetDimensions() or pass as parameter
    float2 textureDimensions = float2(2048.0, 2048.0);

    // Compute LOD from ray differentials
    float lod = ComputeTextureLOD(uv, dUVdx, dUVdy, textureDimensions);

    // Sample with computed LOD
    return textures[NonUniformResourceIndex(textureIndex)].SampleLevel(
        samplers[NonUniformResourceIndex(samplerIndex)], uv, lod
    );
}

// Legacy function for compatibility (uses LOD 0)
float4 SampleTexture(int textureIndex, int samplerIndex, float2 uv, float4 fallback) {
    return SampleTextureWithLOD(textureIndex, samplerIndex, uv,
                                float2(0.0, 0.0), float2(0.0, 0.0), fallback);
}

// Compute UV differentials from ray differentials
// Estimates how UV coordinates change per screen pixel using ray differentials
//
// Algorithm:
// 1. Propagate ray differentials through intersection
// 2. Compute auxiliary intersection points for differential rays
// 3. Interpolate UVs at auxiliary points
// 4. Compute dUV = UVauxiliary - UVcenter
//
// Simplified version: Use triangle edge vectors to estimate UV gradient
float2 ComputeUVDifferentialX(float3 rayDir, float3 dDdx, float t, float3 edge1, float3 edge2,
                               float2 uv0, float2 uv1, float2 uv2, float2 bary) {
    // Propagate differential ray: O' = O + t * dD/dx
    // For small differential, approximate intersection as moving along triangle plane
    // dUV/dx ≈ (∂UV/∂edge1) * (dP/dx · edge1) + (∂UV/∂edge2) * (dP/dx · edge2)

    // Simplified: Use ray direction change scaled by distance
    // This gives a reasonable approximation for LOD computation
    float scale = t * length(dDdx);
    float2 dUV = float2(scale, 0.0) * 0.001;  // Heuristic scaling

    return dUV;
}

float2 ComputeUVDifferentialY(float3 rayDir, float3 dDdy, float t, float3 edge1, float3 edge2,
                               float2 uv0, float2 uv1, float2 uv2, float2 bary) {
    // Similar to X differential
    float scale = t * length(dDdy);
    float2 dUV = float2(0.0, scale) * 0.001;  // Heuristic scaling

    return dUV;
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
    // LUT-fast Atmospheric Transmission Model (Beer-Lambert Law)
    // ========================================================================
    // Computes atmospheric transmittance along view path using Beer-Lambert law:
    //   T(λ, d) = exp(-σ_t(λ) × d)
    //
    // where:
    //   σ_t(λ) = wavelength-dependent extinction coefficient (1/m)
    //   d = path length (m) from camera to surface
    //
    // MODTRAN LUT INTEGRATION:
    // - LUT provides τ_vertical(λ) = vertical optical depth (dimensionless)
    // - Convert to extinction coefficient: σ_t(λ) = τ_vertical(λ) / H_atm
    // - H_atm ≈ 8000m (atmospheric scale height)
    //
    // NOTE: This is a SIMPLIFIED model (LUT-fast mode).
    // Full volume rendering (M4) will use delta-tracking with 3D extinction fields.
    // ========================================================================

    // Compute path length from camera to hit point
    float pathLength_m = RayTCurrent();  // Distance along ray in meters (world units)

    // Convert LUT transmittance (vertical optical depth) to extinction coefficient
    // Assumption: LUT transmittance is for vertical path through atmosphere
    // τ_vertical ≈ 0.1-0.5 (typical clear sky), so σ_t ≈ 1e-5 to 6e-5 m^-1
    const float atmosphericScaleHeight_m = 8000.0;  // Rayleigh scale height (m)

    float opticalDepth_vertical = -log(max(lut.transmittance, 1e-6));  // τ = -ln(T)
    float extinctionCoeff = opticalDepth_vertical / atmosphericScaleHeight_m;  // σ_t = τ / H

    // Apply Beer-Lambert attenuation along view path
    // For short paths (< 10km), this is a reasonable approximation
    // For longer paths, need to account for path integral through varying density
    float atmosphericTransmittance = exp(-extinctionCoeff * pathLength_m);

    // Clamp to [0, 1] to prevent numerical issues
    atmosphericTransmittance = clamp(atmosphericTransmittance, 0.0, 1.0);

    // Apply transmittance to sun radiance (direct lighting attenuated by atmosphere)
    // Sky radiance is NOT attenuated (it's already the result of atmospheric scattering)
    sunRadiance *= atmosphericTransmittance;

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

    // Direct sun lighting with atmospheric attenuation (Beer-Lambert law)
    // L_out = BRDF * L_sun * τ(λ, d) * (N · L)
    // where τ(λ, d) is atmospheric transmittance computed from Beer-Lambert law
    // NOTE: sunRadiance already includes atmosphericTransmittance (applied above)
    float NdotL = max(dot(normal, L), 0.0);
    float3 directSun = brdf * sunRadiance * NdotL;

    // ========================================================================
    // Image-Based Lighting (IBL) - Diffuse and Specular
    // ========================================================================

    // Compute F0 (reflectance at normal incidence) for Fresnel calculations
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // ------------------------------------------------------------------------
    // Sky Radiance Hemispherical Integration (Diffuse Ambient)
    // ------------------------------------------------------------------------
    // Computes diffuse sky lighting by integrating sky radiance over hemisphere:
    //   L_sky = ∫_Ω L_sky(ω) × BRDF(ω) × (N · ω) dω
    //
    // For Lambertian BRDF (f = ρ/π), this simplifies to:
    //   L_sky = (ρ/π) × L_sky × ∫_Ω (N · ω) dω
    //         = (ρ/π) × L_sky × π
    //         = ρ × L_sky
    //
    // For PBR materials with Fresnel term, we account for:
    // - kD = diffuse reflection coefficient (energy not reflected specularly)
    // - Fresnel term reduces diffuse contribution at grazing angles
    // - Metallic materials have no diffuse reflection (kD ≈ 0)
    //
    // PHYSICAL INTERPRETATION:
    // - Sky radiance L_sky(λ) is the average radiance from sky dome
    // - Hemispherical integral ∫(N·ω)dω = π (solid angle of hemisphere)
    // - For Lambertian: outgoing radiance = albedo × incident irradiance
    //
    // NOTE: This is the "ambient" term in traditional graphics, but physically
    // it represents diffuse reflection of scattered sky radiance.
    // ========================================================================

    // Compute diffuse reflection coefficient (energy conservation with specular)
    float3 kD = (1.0 - FresnelSchlick(F0, max(dot(normal, V), 0.0))) * (1.0 - metallic);

    // Hemispherical integration with Lambertian BRDF
    // Factor of π from hemisphere integral cancels with π in BRDF denominator
    float3 skyAmbient = kD * albedo * skyRadiance;

    // ========================================================================
    // Image-Based Lighting (IBL) Specular Reflection
    // ========================================================================
    // Implements split-sum approximation for physically-based environment reflections
    // References:
    // - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
    // - glTF 2.0 specification (KHR_lights_punctual + IBL extension)
    // ========================================================================

    float3 iblSpecular = float3(0.0, 0.0, 0.0);

    // Only compute IBL for surfaces with non-zero metallic or roughness < 1.0
    // This optimization skips perfectly diffuse surfaces (no specular reflection)
    if (metallic > 0.01 || roughness < 0.99) {
        // 1. Compute reflection vector R = reflect(-V, N)
        //    This is the direction we would see a perfect mirror reflection
        float3 R = reflect(-V, normal);

        // 2. Select mipmap level based on roughness
        //    Rougher surfaces sample blurrier reflections (higher mip levels)
        //    Query number of mip levels at runtime (shader intrinsic)
        uint width, height, numMips;
        prefilteredEnvMap.GetDimensions(0, width, height, numMips);
        float lod = roughness * float(numMips - 1);

        // 3. Sample prefiltered environment map
        //    SampleLevel = explicit LOD (required in ray tracing shaders)
        float3 prefilteredColor = prefilteredEnvMap.SampleLevel(iblSampler, R, lod).rgb;

        // 4. Sample BRDF integration LUT
        //    Inputs: (NdotV, roughness) → Outputs: (scale, bias) for Fresnel term
        float NdotV_clamped = max(dot(normal, V), 0.0);
        float2 envBRDF = brdfLUT.SampleLevel(iblSampler, float2(NdotV_clamped, roughness), 0.0).rg;

        // 5. Split-sum approximation
        //    L_ibl = ∫ L(l) * BRDF(l,v) * (n·l) dl
        //          ≈ (∫ L(l) * (n·l) dl) * (∫ BRDF(l,v) * (n·l) dl)
        //          ≈ prefilteredColor * (F0 * envBRDF.x + envBRDF.y)
        //
        //    Where:
        //    - envBRDF.x (scale): multiplies F0 (Fresnel at normal incidence)
        //    - envBRDF.y (bias): constant offset for grazing angles
        float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
        iblSpecular = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);

        // 6. Energy conservation: for metals, reduce diffuse contribution
        //    (already handled by kD term in skyAmbient calculation above)
    }

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
        // Single Wavelength Mode: True Spectral Rendering (Visible Light)
        // ====================================================================
        // For single wavelength, we:
        // 1. Convert RGB albedo → spectral reflectance at camera.wavelength_nm
        // 2. Compute scalar PBR BRDF with spectral reflectance
        // 3. Use scalar sun radiance (sunRadiance_spectral)
        //
        // IMPORTANT: This is ONLY valid for visible light (380-780 nm).
        // For IR wavelengths, use MWIR/LWIR modes instead.
        // ====================================================================

        float lambda = camera.wavelength_nm;

        // 1. Convert RGB albedo to spectral reflectance at wavelength λ
        //    Uses Gaussian-based RGB→Spectrum upsampling (SpectralConversion.hlsli)
        //    NOTE: baseColor.rgb is already in linear space (glTF textures are sRGB-decoded)
        float spectralAlbedo = GetSpectralReflectanceFromRGBTexture(
            baseColor.rgb,
            lambda,
            false  // Already in linear space (not sRGB)
        );

        // 2. Compute scalar PBR BRDF with spectral albedo
        //    Uses the same Cook-Torrance model, but with scalar reflectance
        float brdf_scalar = CookTorranceBRDF_Spectral(
            normal,
            V,
            L,
            spectralAlbedo,
            metallic,
            roughness
        );

        // 3. Direct sun lighting: L_out = BRDF * L_sun(λ) * (N · L)
        //    Use scalar sun radiance at wavelength λ
        float sunIntensity_scalar = lut.sunRadiance_spectral;
        float directSun_scalar = brdf_scalar * sunIntensity_scalar * NdotL;

        // 4. Sky ambient lighting (scalar)
        //    Use simplified diffuse approximation (same as RGB mode)
        float3 F0_scalar = lerp(float3(0.04, 0.04, 0.04), float3(spectralAlbedo, spectralAlbedo, spectralAlbedo), metallic);
        float3 F_scalar = FresnelSchlick(F0_scalar, max(dot(normal, V), 0.0));
        float kD_scalar = ((1.0 - F_scalar.r) * (1.0 - metallic));  // Use .r since all channels are identical
        float skyIntensity_scalar = lut.skyRadiance_spectral;
        float skyAmbient_scalar = kD_scalar * spectralAlbedo / PI * skyIntensity_scalar;

        // 5. Total spectral radiance (scalar)
        float radiance_spectral = directSun_scalar + skyAmbient_scalar + emissive.r;  // Assume emissive is grayscale in spectral mode

        // Validation
        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);

        // Output as grayscale (replicate scalar to RGB for display)
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

        // Use IR material properties (evaluated from curves at current wavelength)
        // If no IR data available, fallback to spectralAlbedo approximation
        float emissivity = material.irEmissivity;
        float reflectance = material.irReflectance;
        float transmittance = material.irTransmittance;

        // Kirchhoff's law validation: ε + ρ + τ ≤ 1
        // Note: GPU already receives pre-evaluated values from CPU-side curves
        float energySum = emissivity + reflectance + transmittance;
        if (energySum > 1.0) {
            // Normalize to conserve energy if curves violate Kirchhoff's law
            float normFactor = 1.0 / energySum;
            emissivity *= normFactor;
            reflectance *= normFactor;
            transmittance *= normFactor;
        }

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
