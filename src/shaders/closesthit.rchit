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
//ATMOSPHERIC MODEL (Enhanced with Delta-Tracking support):
// - Beer-Lambert path attenuation: T(λ, d) = exp(-σ_t(λ) × d)
// - MODTRAN LUT provides σ_t(λ) extinction coefficient (fallback)
// - AtmosphericParams provides physical Rayleigh + Mie coefficients (enhanced)
// - Hemispherical sky radiance integration for diffuse ambient
// ============================================================================

#include "common.hlsli"
#include "pbr.hlsli"
#include "SpectralConversion.hlsli"
#include "blackbody.hlsli"
#include "spectral_query.hlsli"
#include "atmospheric.hlsli"

// ============================================================================
// Bindings
// ============================================================================

[[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;
[[vk::binding(3, 0)]] StructuredBuffer<float3> vertexBuffer;    // Vertex positions
[[vk::binding(4, 0)]] StructuredBuffer<uint> indexBuffer;       // Triangle indices
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materials; // Material properties
[[vk::binding(6, 0)]] Texture2D textures[];                     // Bindless texture array
[[vk::binding(7, 0)]] SamplerState samplers[];                  // Bindless sampler array
[[vk::binding(8, 0)]] StructuredBuffer<float2> uvBuffer;        // UV coordinates (optional)
[[vk::binding(9, 0)]] StructuredBuffer<float4> tangentBuffer;   // Tangent vectors (optional)
[[vk::binding(16, 0)]] StructuredBuffer<float3> normalBuffer;   // Normal vectors (required for smooth shading)

// ============================================================================
// NEW (M2+): Spectral Curve Buffer
// ============================================================================
// Buffer of spectral reflectance curves for physically-based spectral rendering
// Indexed by MaterialData::spectralReflectanceCurveIndex
// Binding 13 chosen to avoid conflict with IBL resources (10-12)
// ============================================================================

[[vk::binding(13, 0)]] StructuredBuffer<SpectralCurveGPU> spectralCurves;

// ============================================================================
// NEW (M2+): Complex Refractive Index Buffer
// ============================================================================
// Buffer of complex refractive index (n, k) for physical Fresnel calculation
// Indexed by MaterialData::complexRefractiveIndexIndex
// Data source: RefractiveIndex.INFO database (measured metal optical constants)
//
// PHYSICS:
// - Metal surfaces: Use measured n,k for accurate wavelength-dependent Fresnel
// - Dielectrics: k ≈ 0 (transparent), n determines refraction
// - Semiconductors: Wavelength-dependent n,k (silicon, germanium)
//
// When complexRefractiveIndexIndex >= 0, use physical Fresnel equation:
//   F = FresnelConductor(cosθ, n(λ), k(λ))
// Otherwise, use standard PBR approximation:
//   F = F0 + (1-F0) * (1-cosθ)^5
// ============================================================================

[[vk::binding(14, 0)]] StructuredBuffer<ComplexRefractiveIndexGPU> complexRefractiveIndices;

// ============================================================================
// NEW (M2+): Solar Spectral LUT Buffer
// ============================================================================
// Contains ASTM G-173 solar irradiance curves for spectral rendering
// Enables physically-accurate wavelength-dependent sun/sky illumination
//
// DATA:
// - sunIrradiance: Direct+circumsolar spectral irradiance (W·m⁻²·nm⁻¹)
// - skyIrradiance: Diffuse sky spectral irradiance (W·m⁻²·nm⁻¹)
//
// USAGE:
// - Query sun irradiance: SampleSunIrradiance(solarLUT, wavelength_nm)
// - Query sky irradiance: SampleSkyIrradiance(solarLUT, wavelength_nm)
// - Convert to radiance: L = E / SUN_SOLID_ANGLE_SR
//
// When solarSpectralLUT[0].sunIrradiance.numSamples == 0, fall back to LightingParams RGB values
// ============================================================================

[[vk::binding(15, 0)]] StructuredBuffer<SolarSpectralLUT> solarSpectralLUT;

// ============================================================================
// Atmospheric Parameters Buffer (Binding 17)
// ============================================================================
// Physical parameters for enhanced atmospheric transmittance calculation
// Provides Rayleigh + Mie coefficients for wavelength-dependent extinction
// When beta_rayleigh_550nm.x > 0, use physical model; otherwise use LUT fallback
// ============================================================================

[[vk::binding(17, 0)]] StructuredBuffer<AtmosphericParams> atmosphericParams;

// ============================================================================
// Instance Geometry Info Buffer (Binding 18)
// ============================================================================
// Per-TLAS-instance geometry offset information for multi-BLAS support.
// When scene has multiple BLAS, each instance's geometry data is merged into
// global buffers. This buffer tells us where each instance's data starts.
//
// USAGE:
//   uint instIdx = InstanceIndex();
//   InstanceGeometryInfo geo = instanceGeometryInfo[instIdx];
//   uint idx = indexBuffer[geo.indexOffset + PrimitiveIndex() * 3 + i];
//   float3 pos = vertexBuffer[geo.vertexOffset + idx];
//   float3 nrm = normalBuffer[geo.normalOffset + idx];
// ============================================================================

[[vk::binding(18, 0)]] StructuredBuffer<InstanceGeometryInfo> instanceGeometryInfo;

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
    [[vk::location(0)]] float2 bary : SV_Barycentrics;  // Barycentric coordinates (b1, b2), where b0 = 1 - b1 - b2
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
// Physical Fresnel F0 Computation
// ============================================================================
// Computes F0 (normal incidence reflectance) using physical n,k data when
// available, falling back to standard PBR approximation otherwise.
//
// PHYSICAL PATH (complexRefractiveIndexIndex >= 0):
//   - Query measured complex refractive index at current wavelength
//   - Use exact Fresnel equation: F0 = [(n-1)² + k²] / [(n+1)² + k²]
//   - Provides wavelength-dependent specular reflection (gold, copper, etc.)
//
// PBR PATH (complexRefractiveIndexIndex < 0):
//   - Use standard approximation: F0 = lerp(0.04, albedo, metallic)
//   - 0.04 = typical dielectric F0 (glass, plastic)
//   - Albedo used for metals (color tinting at normal incidence)
//
// PARAMETERS:
//   material: MaterialData with complexRefractiveIndexIndex
//   albedo: Base color (used for PBR metallic tinting)
//   metallic: Metalness factor [0, 1]
//   wavelength_nm: Current wavelength for spectral lookup
//
// RETURNS:
//   float3 F0 - normal incidence reflectance (replicated to RGB for non-spectral)
// ============================================================================

float3 ComputePhysicalF0(MaterialData material, float3 albedo, float metallic, float wavelength_nm) {
    if (material.complexRefractiveIndexIndex >= 0) {
        // PHYSICAL PATH: Use measured n,k data from RefractiveIndex.INFO
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[material.complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        float n = nk.x;
        float k = nk.y;

        // Fresnel at normal incidence: F0 = [(n-1)² + k²] / [(n+1)² + k²]
        float F0_physical = FresnelF0(n, k);

        // For spectral mode, return scalar F0 replicated to RGB
        // For RGB mode, this is an approximation (should sample at R/G/B wavelengths)
        return float3(F0_physical, F0_physical, F0_physical);
    }

    // PBR PATH: Standard approximation
    // Dielectrics: F0 ≈ 0.04 (glass, plastic, water)
    // Metals: F0 = albedo (color tinting from base color)
    return lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
}

// Compute full Fresnel reflectance at arbitrary angle using physical n,k data
// Used for specular highlight computation (not just F0)
float ComputePhysicalFresnel(MaterialData material, float cosTheta, float wavelength_nm) {
    if (material.complexRefractiveIndexIndex >= 0) {
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[material.complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        return FresnelConductor(cosTheta, nk.x, nk.y);
    }

    // Fallback: Use Schlick approximation with typical dielectric F0
    return FresnelSchlick(cosTheta, 0.04);
}

// ============================================================================
// Closest Hit Entry Point
// ============================================================================

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attribs) {
    // ========================================================================
    // Get instance geometry info for multi-BLAS support
    // ========================================================================
    // InstanceIndex() returns the TLAS instance index
    // instanceGeometryInfo provides offsets into merged global geometry buffers
    // This allows correct geometry access when scene has multiple BLAS
    // ========================================================================

    uint instanceIdx = InstanceIndex();
    InstanceGeometryInfo geoInfo = instanceGeometryInfo[instanceIdx];

    // Material ID is stored in InstanceGeometryInfo (not InstanceID anymore)
    uint materialID = geoInfo.materialId;
    MaterialData material = materials[materialID];

    // ========================================================================
    // Compute geometric normal from triangle vertices
    // ========================================================================

    uint primitiveID = PrimitiveIndex();

    // Read triangle indices with offset into global index buffer
    uint idx0 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 0];
    uint idx1 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 1];
    uint idx2 = indexBuffer[geoInfo.indexOffset + primitiveID * 3 + 2];

    // Read vertex positions with offset into global vertex buffer
    float3 v0 = vertexBuffer[geoInfo.vertexOffset + idx0];
    float3 v1 = vertexBuffer[geoInfo.vertexOffset + idx1];
    float3 v2 = vertexBuffer[geoInfo.vertexOffset + idx2];

    // ========================================================================
    // Compute TRUE geometric normal from triangle edges (flat normal)
    // This is the actual surface normal, NOT the interpolated shading normal
    // ========================================================================
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 objectGeometricNormal = SafeNormalize(cross(edge1, edge2), float3(0.0, 1.0, 0.0));

    // Transform geometric normal to world space using inverse-transpose
    // mul(v, M) = v * M = transpose(M) * v, and we want transpose(inverse(ObjectToWorld)) * v
    // WorldToObject = inverse(ObjectToWorld), so mul(v, WorldToObject) = transpose(WorldToObject) * v ✓
    float3x3 normalTransform = (float3x3)WorldToObject3x4();
    float3 worldGeometricNormal = SafeNormalize(mul(objectGeometricNormal, normalTransform), float3(0.0, 1.0, 0.0));

    // Face forward: ensure geometric normal points toward camera (opposite to ray direction)
    float3 rayDir = WorldRayDirection();
    if (dot(worldGeometricNormal, rayDir) > 0.0) {
        worldGeometricNormal = -worldGeometricNormal;
    }

    // Read per-vertex normals with offset into global normal buffer
    // This gives smooth shading (Gouraud/Phong) instead of flat shading
    float3 n0 = normalBuffer[geoInfo.normalOffset + idx0];
    float3 n1 = normalBuffer[geoInfo.normalOffset + idx1];
    float3 n2 = normalBuffer[geoInfo.normalOffset + idx2];

    // Barycentric interpolation: n = n0 * w0 + n1 * w1 + n2 * w2
    // where w0 = (1 - bary.x - bary.y), w1 = bary.x, w2 = bary.y
    float3 objectNormal = n0 * (1.0 - attribs.bary.x - attribs.bary.y)
                        + n1 * attribs.bary.x
                        + n2 * attribs.bary.y;
    objectNormal = SafeNormalize(objectNormal, float3(0.0, 1.0, 0.0));

    // Transform shading normal to world space
    // FIXED: Use SafeNormalize to prevent NaN propagation
    float3 worldNormal = SafeNormalize(mul(objectNormal, normalTransform), float3(0.0, 1.0, 0.0));

    // UV coordinates with offset into global UV buffer
    // Barycentric interpolation: uv = u0 * (1 - b1 - b2) + u1 * b1 + u2 * b2
    float2 uv0 = uvBuffer[geoInfo.uvOffset + idx0];
    float2 uv1 = uvBuffer[geoInfo.uvOffset + idx1];
    float2 uv2 = uvBuffer[geoInfo.uvOffset + idx2];
    float2 uv = uv0 * (1.0 - attribs.bary.x - attribs.bary.y) + uv1 * attribs.bary.x + uv2 * attribs.bary.y;

    // Read tangent from buffer with offset (or fallback to fake tangent)
    float3 worldTangent;
    if (material.normalTextureIndex >= 0) {  // Only compute tangent if normal map is used
        // Read tangents with offset into global tangent buffer
        float4 tangent4_0 = tangentBuffer[geoInfo.tangentOffset + idx0];
        float4 tangent4_1 = tangentBuffer[geoInfo.tangentOffset + idx1];
        float4 tangent4_2 = tangentBuffer[geoInfo.tangentOffset + idx2];

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

    LightingParams lut = lightingParams[0];
    // FIXED: Use SafeNormalize in case LUT data is invalid
    float3 sunDir = SafeNormalize(lut.sunDirection, float3(0.0, 1.0, 0.0));

    // Choose lighting based on spectral mode
    float3 sunRadiance;
    float3 skyRadiance;

    if (camera.spectral_mode == SPECTRAL_MODE_RGB || camera.spectral_mode == SPECTRAL_MODE_VIS_FUSED) {
        // RGB and VIS_FUSED modes: Use full RGB lighting
        sunRadiance = lut.sunRadiance_rgb;
        skyRadiance = lut.skyRadiance_rgb;
    } else {
        // Spectral modes: Use scalar spectral radiance (replicate to RGB)
        sunRadiance = float3(lut.sunRadiance_spectral, lut.sunRadiance_spectral, lut.sunRadiance_spectral);
        skyRadiance = float3(lut.skyRadiance_spectral, lut.skyRadiance_spectral, lut.skyRadiance_spectral);
    }

    // ========================================================================
    // Enhanced Atmospheric Transmission Model
    // ========================================================================
    // Computes atmospheric transmittance along view path
    //   T(λ, d) = exp(-σ_t(λ) × d)
    //
    // TWO MODES:
    // 1. Physical mode (AtmosphericParams available):
    //    - Uses Rayleigh + Mie scattering coefficients
    //    - Wavelength-dependent extinction: σ_t(λ) = β_r(λ) + β_m(λ)
    //    - Assumes constant density (simplified for closesthit performance)
    //
    // 2. LUT-fast mode (fallback):
    //    - Uses MODTRAN LUT transmittance
    //    - Converts vertical optical depth to extinction coefficient
    // ========================================================================

    // Compute path length from camera to hit point (convert to meters)
    float pathLength_m = RayTCurrent() * lut.worldUnitsToMeters;

    float atmosphericTransmittance;
    AtmosphericParams atmo = atmosphericParams[0];

    // Check if physical atmospheric model is enabled
    if (atmo.beta_rayleigh_550nm.x > 1e-9) {
        // PHYSICAL MODE: Use Rayleigh + Mie coefficients
        // Use SCALAR versions for single-wavelength computation
        float beta_r = RayleighScatteringCoeff_Scalar(camera.wavelength_nm, atmo.beta_rayleigh_550nm.x);
        float beta_m = MieScatteringCoeff_Scalar(camera.wavelength_nm, atmo.beta_mie_550nm.x, atmo.mie_alpha);

        // Total extinction (scalar for single wavelength)
        float extinction = beta_r + beta_m;

        // Transmittance along path
        atmosphericTransmittance = exp(-extinction * pathLength_m);
    } else {
        // LUT-FAST MODE (fallback): Use MODTRAN LUT
        const float atmosphericScaleHeight_m = 8000.0;
        float opticalDepth_vertical = -log(max(lut.transmittance, 1e-6));
        float extinctionCoeff = opticalDepth_vertical / atmosphericScaleHeight_m;
        atmosphericTransmittance = exp(-extinctionCoeff * pathLength_m);
    }

    // Clamp to [0, 1] to prevent numerical issues
    atmosphericTransmittance = clamp(atmosphericTransmittance, 0.0, 1.0);

    // Apply transmittance to sun radiance for RGB mode only
    // VIS_FUSED mode computes wavelength-dependent transmittance inside the spectral loop
    // Sky radiance is NOT attenuated (it's already the result of atmospheric scattering)
    if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
        sunRadiance *= atmosphericTransmittance;
    }
    // Note: For VIS_FUSED, transmittance is applied per-wavelength in the loop below

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
    // Uses physical n,k data when available for wavelength-accurate metal reflections
    float3 F0 = ComputePhysicalF0(material, albedo, metallic, camera.wavelength_nm);

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

    // Declare IBL variables outside conditional for use in spectral integration
    float3 prefilteredColor = float3(0.0, 0.0, 0.0);
    float2 envBRDF = float2(0.0, 0.0);

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
        prefilteredColor = prefilteredEnvMap.SampleLevel(iblSampler, R, lod).rgb;

        // 4. Sample BRDF integration LUT
        //    Inputs: (NdotV, roughness) → Outputs: (scale, bias) for Fresnel term
        float NdotV_clamped = max(dot(normal, V), 0.0);
        envBRDF = brdfLUT.SampleLevel(iblSampler, float2(NdotV_clamped, roughness), 0.0).rg;

        // 5. Split-sum approximation
        //    L_ibl = ∫ L(l) * BRDF(l,v) * (n·l) dl
        //          ≈ (∫ L(l) * (n·l) dl) * (∫ BRDF(l,v) * (n·l) dl)
        //          ≈ prefilteredColor * (F0 * envBRDF.x + envBRDF.y)
        //
        //    Where:
        //    - envBRDF.x (scale): multiplies F0 (Fresnel at normal incidence)
        //    - envBRDF.y (bias): constant offset for grazing angles
        //    NOTE: F0 is computed outside this block using physical n,k data when available
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
        // RGB Mode: Fast Pure-RGB Pipeline (No Spectral Integration)
        // ====================================================================
        // Direct RGB rendering without spectral wavelength sampling.
        // Uses standard PBR calculations already computed above.
        //
        // This is the FASTEST mode with no spectral integration overhead.
        // For physically-correct spectral rendering, use VIS_FUSED mode.
        // ====================================================================
        output_radiance = radiance;

        // Validation: clamp and sanitize to prevent NaN/Inf
        if (!isfinite(output_radiance.r) || !isfinite(output_radiance.g) || !isfinite(output_radiance.b)) {
            output_radiance = float3(0.0, 0.0, 0.0);
        }
        output_radiance = clamp(output_radiance, 0.0, 1000.0);

    } else if (camera.spectral_mode == SPECTRAL_MODE_VIS_FUSED) {
        // ====================================================================
        // VIS_Fused Mode: True 32-Wavelength Spectral Integration
        // ====================================================================
        // Physically-correct spectral rendering with full wavelength sampling:
        //   1. Sample 32 wavelengths uniformly across visible spectrum (380-780nm)
        //   2. Compute spectral radiance L(λ) at each wavelength
        //   3. Integrate via CIE XYZ color matching functions
        //   4. Convert XYZ → Linear RGB (sRGB D65)
        //
        // This is the TRUE HS-OFF spectral rendering for visible light.
        // Performance: ~10-15x slower than RGB mode, but physically accurate.
        //
        // SPECTRAL REFLECTANCE SOURCE (priority order):
        //   1. Measured spectral curve (spectralReflectanceCurveIndex >= 0)
        //   2. RGB texture upsampling via Gaussian basis (fallback)
        //
        // ILLUMINATION SOURCE (priority order):
        //   1. SolarSpectralLUT with measured ASTM G-173 spectra (preferred)
        //   2. RGB values converted to colored spectrum via ConvertLinearRGBToSpectrum
        // ====================================================================

        // Spectral integration parameters
        const uint   NUM_WAVELENGTH_SAMPLES = 32;
        const float  LAMBDA_MIN_VIS = 380.0;  // nm
        const float  LAMBDA_MAX_VIS = 780.0;  // nm
        const float  LAMBDA_STEP = (LAMBDA_MAX_VIS - LAMBDA_MIN_VIS) / float(NUM_WAVELENGTH_SAMPLES - 1);

        // Accumulate XYZ tristimulus values
        float3 XYZ_accum = float3(0.0, 0.0, 0.0);

        // Check if we have physical spectral irradiance data
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Precompute IBL parameters for spectral integration
        // F0 scalar: average of RGB F0 for spectral Fresnel approximation
        float F0_scalar = (F0.r + F0.g + F0.b) / 3.0;
        bool useIBL = (metallic > 0.01 || roughness < 0.99);

        // Loop over wavelengths
        // NOTE: Removed [unroll] to reduce shader compilation time (was 50+ seconds)
        // Modern GPUs handle small loops efficiently without forced unrolling
        for (uint i = 0; i < NUM_WAVELENGTH_SAMPLES; ++i) {
            float lambda = LAMBDA_MIN_VIS + float(i) * LAMBDA_STEP;

            // ================================================================
            // Query Sun/Sky Spectral Radiance at Wavelength λ
            // ================================================================
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                // PHYSICAL PATH: Query ASTM G-173 spectral irradiance curves
                // Convert irradiance (W·m⁻²·nm⁻¹) to radiance (W·sr⁻¹·m⁻²·nm⁻¹)
                float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);

                // Sun disk: L = E / Ω_sun (radiance from irradiance)
                sun_radiance_lambda = SunIrradianceToRadiance(sun_irr);
                // Sky: diffuse hemispherical, already in radiance-like units (W·m⁻²·nm⁻¹·sr⁻¹ approximated)
                // For sky dome, we assume uniform sky approximation: L_sky ≈ E_sky / π
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // CORRECTED FALLBACK: Convert RGB to colored illuminant spectrum
                // Use illuminant function (no clamp) instead of reflectance function (clamped to 1.5)
                // This preserves HDR intensity of light sources
                sun_radiance_lambda = ConvertLinearRGBToIlluminantSpectrum(lut.sunRadiance_rgb, lambda);
                sky_radiance_lambda = ConvertLinearRGBToIlluminantSpectrum(lut.skyRadiance_rgb, lambda);
            }

            // ================================================================
            // Wavelength-Dependent Atmospheric Transmittance (Fix 3)
            // ================================================================
            // Rayleigh scattering: β(λ) ∝ λ^-4 (strong wavelength dependence)
            // Mie scattering: β(λ) ∝ λ^-α where α ≈ 0.84 (weaker dependence)
            //
            // This causes blue light (400nm) to be scattered ~9x more than red (700nm),
            // producing the familiar reddening of distant objects and sunset colors.
            // ================================================================
            float transmittance_lambda = 1.0;
            if (atmo.beta_rayleigh_550nm.x > 1e-9) {
                // Physical mode: compute wavelength-dependent extinction
                float beta_r = RayleighScatteringCoeff_Scalar(lambda, atmo.beta_rayleigh_550nm.x);
                float beta_m = MieScatteringCoeff_Scalar(lambda, atmo.beta_mie_550nm.x, atmo.mie_alpha);
                float extinction_lambda = beta_r + beta_m;
                transmittance_lambda = exp(-extinction_lambda * pathLength_m);
            } else {
                // LUT fallback: use pre-computed scalar transmittance (wavelength-independent)
                transmittance_lambda = atmosphericTransmittance;
            }
            transmittance_lambda = clamp(transmittance_lambda, 0.0, 1.0);

            // Apply wavelength-dependent transmittance to direct sunlight
            // Note: sky_radiance is already scattered light, don't attenuate twice
            sun_radiance_lambda *= transmittance_lambda;

            // 1. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                // Quantitative path: measured spectral curve
                rho_lambda = EvaluateSpectralCurve(spectralCurves, material.spectralReflectanceCurveIndex, lambda);
            } else {
                // Fallback path: RGB → Spectrum upsampling
                rho_lambda = ConvertLinearRGBToSpectrum(baseColor.rgb, lambda);
            }

            // 2. Compute BRDF at this wavelength (scalar Cook-Torrance)
            float brdf_lambda = CookTorranceBRDF_Spectral(normal, V, L, rho_lambda, metallic, roughness);

            // 3. Compute spectral radiance: L(λ) = BRDF(λ) × L_sun(λ) × (N·L) + kD × ρ(λ)/π × L_sky(λ)
            float L_direct = brdf_lambda * sun_radiance_lambda * NdotL;

            // Diffuse ambient (simplified Fresnel for diffuse coefficient)
            float kD_lambda = (1.0 - metallic);  // Metals have no diffuse
            float L_ambient = kD_lambda * rho_lambda / PI * sky_radiance_lambda;

            // 4. Emissive contribution (spectrally integrated)
            // Convert emissive RGB to spectral radiance at this wavelength
            // Use illuminant function (no clamp) to preserve HDR emissive intensity
            float L_emissive = ConvertLinearRGBToIlluminantSpectrum(emissive, lambda);

            // 5. IBL specular contribution (spectrally integrated)
            // Convert prefiltered environment RGB to spectrum at this wavelength
            // Use illuminant function (no clamp) to preserve HDR environment intensity
            float L_ibl = 0.0;
            if (useIBL) {
                float ibl_spectrum = ConvertLinearRGBToIlluminantSpectrum(prefilteredColor, lambda);
                L_ibl = ibl_spectrum * (F0_scalar * envBRDF.x + envBRDF.y);
            }

            float L_lambda = L_direct + L_ambient + L_emissive + L_ibl;

            // 6. Weight by CIE XYZ color matching functions
            float x_bar = CIE_X(lambda);
            float y_bar = CIE_Y(lambda);
            float z_bar = CIE_Z(lambda);

            // Riemann sum integration: ∫L(λ)×CMF(λ)dλ ≈ Σ L(λᵢ)×CMF(λᵢ)×Δλ
            XYZ_accum.x += L_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += L_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += L_lambda * z_bar * LAMBDA_STEP;
        }

        // NOTE: XYZ_accum from Riemann sum is already correctly normalized.
        // The integration XYZ_accum += L(λ) × CMF(λ) × Δλ already has proper units.
        // DO NOT divide by CIE_Y_INTEGRAL (106.9), as that would make output 107x too dark!
        // The constant 106.9 is for 1nm sampling, but we use LAMBDA_STEP ≈ 12.9nm.
        // Riemann sum normalization is: XYZ = Σ[L(λᵢ) × CMF(λᵢ) × Δλ] (already correct)

        // XYZ → Linear RGB (sRGB D65)
        output_radiance = ConvertXYZToLinearRGB(XYZ_accum);

        // ====================================================================
        // CHROMATICITY CORRECTION for Equal-Energy → D65 Adaptation
        // ====================================================================
        // Problem: CIE color matching functions have different integrals:
        //   ∫x̄(λ)dλ ≈ 95.05, ∫ȳ(λ)dλ ≈ 106.9, ∫z̄(λ)dλ ≈ 108.89
        //
        // For FLAT spectrum material (gray colors), XYZ ratio is (95:107:109)
        // After sRGB matrix, this produces GREEN-BIASED output:
        //   RGB ∝ (89.3, 113.0, 98.6) ≈ (0.79, 1.00, 0.87)
        //
        // This correction neutralizes the chromaticity shift by scaling
        // R and B channels to match G, ensuring flat spectrum → neutral gray.
        //
        // Correction factors are configurable via LightingParams (default 1.266, 1.146)
        // ====================================================================
        output_radiance.r *= lut.chromaR_correction;
        output_radiance.b *= lut.chromaB_correction;

        // NOTE: IBL is now integrated in the spectral loop above (L_ibl term)
        // No need to add iblSpecular separately

        // Validation: clamp and sanitize to prevent NaN/Inf
        if (!isfinite(output_radiance.r) || !isfinite(output_radiance.g) || !isfinite(output_radiance.b)) {
            output_radiance = float3(0.0, 0.0, 0.0);  // Fallback to black
        }
        output_radiance = clamp(output_radiance, 0.0, 1000.0);  // Reasonable HDR range

    } else if (camera.spectral_mode == SPECTRAL_MODE_SINGLE) {
        // ====================================================================
        // Single Wavelength Mode: True Spectral Rendering (HS-OFF Quantitative)
        // ====================================================================
        // For single wavelength, we:
        // 1. Query physical spectral reflectance curve if available
        // 2. Fallback to RGB upsampling if no measured curve
        // 3. Compute scalar PBR BRDF with spectral reflectance
        // 4. Query sun/sky radiance from SolarSpectralLUT or LightingParams fallback
        //
        // SPECTRAL DATA PRIORITY:
        // 1. SolarSpectralLUT (ASTM G-173) → True spectral illumination
        // 2. LightingParams.sunRadiance_spectral → Scalar fallback
        // ====================================================================

        float lambda = camera.wavelength_nm;

        // ================================================================
        // Query Sun/Sky Spectral Radiance at Wavelength λ
        // ================================================================
        float sunRadiance_lambda;
        float skyRadiance_lambda;

        if (solarSpectralLUT[0].sunIrradiance.numSamples > 0) {
            // PHYSICAL PATH: Query ASTM G-173 spectral irradiance curves
            float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], lambda);
            float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);

            // Convert irradiance to radiance
            sunRadiance_lambda = SunIrradianceToRadiance(sun_irr);
            skyRadiance_lambda = sky_irr / PI;  // Diffuse sky: L ≈ E / π
        } else {
            // FALLBACK: Use LightingParams scalar values
            sunRadiance_lambda = lut.sunRadiance_spectral;
            skyRadiance_lambda = lut.skyRadiance_spectral;
        }

        // 1. Query spectral reflectance: prefer measured curve, fallback to RGB upsampling
        float spectralAlbedo;

        if (material.spectralReflectanceCurveIndex >= 0) {
            // QUANTITATIVE PATH: Use physically-measured spectral reflectance curve
            // This enables true HS-OFF mode with physical accuracy
            spectralAlbedo = EvaluateSpectralCurve(
                spectralCurves,
                material.spectralReflectanceCurveIndex,
                lambda
            );
        } else {
            // FALLBACK PATH: RGB → Spectrum upsampling (approximate, ~70-80% accuracy)
            // WARNING: This path does NOT guarantee physical accuracy
            // For quantitative rendering, materials MUST have measured spectral curves
            spectralAlbedo = GetSpectralReflectanceFromRGBTexture(
                baseColor.rgb,
                lambda,
                false  // Already in linear space (not sRGB)
            );
        }

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
        //    Use spectral sun radiance at wavelength λ
        float directSun_scalar = brdf_scalar * sunRadiance_lambda * NdotL;

        // 4. Sky ambient lighting (scalar)
        //    Use simplified diffuse approximation (same as RGB mode)
        float3 F0_scalar = lerp(float3(0.04, 0.04, 0.04), float3(spectralAlbedo, spectralAlbedo, spectralAlbedo), metallic);
        float3 F_scalar = FresnelSchlick(F0_scalar, max(dot(normal, V), 0.0));
        float kD_scalar = ((1.0 - F_scalar.r) * (1.0 - metallic));  // Use .r since all channels are identical
        float skyAmbient_scalar = kD_scalar * spectralAlbedo / PI * skyRadiance_lambda;

        // 5. Total spectral radiance (scalar)
        float radiance_spectral = directSun_scalar + skyAmbient_scalar + emissive.r;  // Assume emissive is grayscale in spectral mode

        // Validation
        if (!isfinite(radiance_spectral)) {
            radiance_spectral = 0.0;
        }
        radiance_spectral = clamp(radiance_spectral, 0.0, 1000.0);

        // Output as grayscale (replicate scalar to RGB for display)
        output_radiance = float3(radiance_spectral, radiance_spectral, radiance_spectral);

    } else if (camera.spectral_mode == SPECTRAL_MODE_SWIR_FUSED) {
        // ====================================================================
        // SWIR Fused Mode: Short-Wave IR Band Integration (1000-2500nm)
        // ====================================================================
        // In SWIR band, solar radiation is still significant (unlike MWIR/LWIR).
        // Physics model combines:
        //   1. Reflected solar irradiance (dominant for passive imaging)
        //   2. Minor thermal emission (only for very hot objects T > 500K)
        //
        // L_total(λ) = ρ(λ) × [L_sun(λ) + L_sky(λ)] + ε(λ) × L_bb(T,λ)
        //
        // For typical outdoor scenes at ambient temperature (~300K), thermal
        // emission in SWIR is negligible (Planck peak at ~10μm, not 1-2.5μm).
        // ====================================================================

        const float SWIR_LAMBDA_MIN = 1000.0;   // nm
        const float SWIR_LAMBDA_MAX = 2500.0;   // nm
        const uint  NUM_SWIR_SAMPLES = 16;
        const float lambda_step = (SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN) / float(NUM_SWIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Check if we have spectral solar LUT for accurate SWIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Fallback: Convert RGB radiance to spectral density for SWIR band
        // Note: This is an approximation. Real solar spectrum extends into SWIR,
        // but visible RGB only covers 380-780nm. We extrapolate the luminance.
        float sun_luminance = 0.2126 * lut.sunRadiance_rgb.r +
                              0.7152 * lut.sunRadiance_rgb.g +
                              0.0722 * lut.sunRadiance_rgb.b;
        float sky_luminance = 0.2126 * lut.skyRadiance_rgb.r +
                              0.7152 * lut.skyRadiance_rgb.g +
                              0.0722 * lut.skyRadiance_rgb.b;
        float swir_bandwidth = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;  // 1500nm
        float sun_power_rgb = sun_luminance / swir_bandwidth;  // Per nm (approximate)
        float sky_power_rgb = sky_luminance / swir_bandwidth;  // Per nm (approximate)

        // Material IR properties (for thermal contribution, usually negligible in SWIR)
        // Use effective emissivity that derives from metallic factor when not set (P1 fix)
        float emissivity = GetEffectiveIREmissivity(material);
        float reflectance = GetEffectiveIRReflectance(material);

        // NOTE: Removed [unroll] to reduce shader compilation time
        for (uint i = 0; i < NUM_SWIR_SAMPLES; ++i) {
            float lambda = SWIR_LAMBDA_MIN + float(i) * lambda_step;

            // 1. Query solar/sky irradiance at this SWIR wavelength
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);
                sun_radiance_lambda = SunIrradianceToRadiance(sun_irr);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // Fallback: use RGB average (approximation)
                sun_radiance_lambda = sun_power_rgb;
                sky_radiance_lambda = sky_power_rgb;
            }

            // 2. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                rho_lambda = EvaluateSpectralCurve(spectralCurves, material.spectralReflectanceCurveIndex, lambda);
            } else {
                // Fallback: use IR reflectance from energy conservation
                rho_lambda = reflectance;
            }

            // 3. Reflected solar radiance: ρ(λ) × (L_sun(λ) × NdotL + L_sky(λ))
            float L_reflected = rho_lambda * (sun_radiance_lambda * NdotL + sky_radiance_lambda);

            // 4. Thermal emission (minor in SWIR for T < 500K)
            float L_emission = 0.0;
            if (material.irTemperature_K > 400.0) {
                // Only compute if object is hot enough for SWIR emission
                float L_blackbody = IRPlanckRadiance(material.irTemperature_K, lambda);
                L_emission = emissivity * L_blackbody;
            }

            // 5. Total spectral radiance
            float L_lambda = L_reflected + L_emission;

            radiance_accum += L_lambda * lambda_step;
        }

        // Normalize by band width
        float band_width = SWIR_LAMBDA_MAX - SWIR_LAMBDA_MIN;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (camera.spectral_mode == SPECTRAL_MODE_NIR_FUSED) {
        // ====================================================================
        // NIR Fused Mode: Near-Infrared Band Integration (780-1400nm)
        // ====================================================================
        // NIR is "reflected infrared" - behaves almost identically to visible light.
        // Solar radiation is the dominant source; thermal emission is negligible
        // for objects below ~600K (327°C).
        //
        // Physics model:
        //   L_total(λ) = ρ(λ) × [L_sun(λ) × cos(θ) + L_sky(λ)]
        //
        // Key properties of NIR (780-1400nm):
        //   - ~46-55% of solar energy is in NIR/SWIR bands
        //   - Can be focused with standard glass optics (unlike thermal IR)
        //   - Commonly used in: vegetation analysis (chlorophyll reflection),
        //     night vision (active illumination), material identification
        //   - Water absorbs strongly at 970nm and 1200nm (moisture detection)
        //
        // Reference: ISO 20473 classifies NIR as IR-A (780nm - 1.4μm)
        // ====================================================================

        const float NIR_LAMBDA_MIN = 780.0;    // nm (start of IR-A band)
        const float NIR_LAMBDA_MAX = 1400.0;   // nm (end of IR-A band)
        const uint  NUM_NIR_SAMPLES = 16;
        const float lambda_step = (NIR_LAMBDA_MAX - NIR_LAMBDA_MIN) / float(NUM_NIR_SAMPLES - 1);

        float radiance_accum = 0.0;

        // Check if we have spectral solar LUT for accurate NIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Fallback: Convert RGB radiance to spectral density for NIR band
        // Note: This is an approximation. Real solar spectrum extends into NIR,
        // but visible RGB only covers 380-780nm. We extrapolate the luminance.
        float sun_luminance = 0.2126 * lut.sunRadiance_rgb.r +
                              0.7152 * lut.sunRadiance_rgb.g +
                              0.0722 * lut.sunRadiance_rgb.b;
        float sky_luminance = 0.2126 * lut.skyRadiance_rgb.r +
                              0.7152 * lut.skyRadiance_rgb.g +
                              0.0722 * lut.skyRadiance_rgb.b;
        float nir_bandwidth = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;  // 620nm
        float sun_power_rgb = sun_luminance / nir_bandwidth;  // Per nm (approximate)
        float sky_power_rgb = sky_luminance / nir_bandwidth;  // Per nm (approximate)

        // NOTE: Removed [unroll] to reduce shader compilation time
        for (uint i = 0; i < NUM_NIR_SAMPLES; ++i) {
            float lambda = NIR_LAMBDA_MIN + float(i) * lambda_step;

            // 1. Query solar/sky irradiance at this NIR wavelength
            float sun_radiance_lambda;
            float sky_radiance_lambda;

            if (hasSpectralSolarLUT) {
                float sun_irr = SampleSunIrradiance(solarSpectralLUT[0], lambda);
                float sky_irr = SampleSkyIrradiance(solarSpectralLUT[0], lambda);
                sun_radiance_lambda = SunIrradianceToRadiance(sun_irr);
                sky_radiance_lambda = sky_irr / PI;
            } else {
                // Fallback: use RGB average (approximation)
                sun_radiance_lambda = sun_power_rgb;
                sky_radiance_lambda = sky_power_rgb;
            }

            // 2. Get spectral reflectance at this wavelength
            float rho_lambda;
            if (material.spectralReflectanceCurveIndex >= 0) {
                // Quantitative path: use measured spectral curve
                rho_lambda = EvaluateSpectralCurve(spectralCurves, material.spectralReflectanceCurveIndex, lambda);
            } else {
                // Fallback: RGB upsampling (NIR is close enough to visible for this to be reasonable)
                // This uses Gaussian basis functions centered at R/G/B wavelengths
                rho_lambda = ConvertLinearRGBToSpectrum(baseColor.rgb, lambda);
            }

            // 3. Reflected solar radiance: ρ(λ) × (L_sun(λ) × NdotL + L_sky(λ))
            float L_reflected = rho_lambda * (sun_radiance_lambda * NdotL + sky_radiance_lambda);

            // Note: Thermal emission is negligible in NIR for T < 600K
            // A 600K object peaks at ~4800nm (Wien's law), far from NIR band
            // Skip thermal calculation for performance

            radiance_accum += L_reflected * lambda_step;
        }

        // Normalize by band width
        float band_width = NIR_LAMBDA_MAX - NIR_LAMBDA_MIN;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);

        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

    } else if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED || camera.spectral_mode == SPECTRAL_MODE_LWIR_FUSED) {
        // ====================================================================
        // MWIR/LWIR Fused Mode: Multi-Wavelength IR Band Integration
        // ====================================================================
        // True spectral integration across infrared bands:
        //   - MWIR: 3000-5000nm (Mid-Wave Infrared)
        //   - LWIR: 8000-12000nm (Long-Wave Infrared)
        //
        // Physics model (updated with solar reflection for MWIR - P2 fix):
        //   L_total(λ) = ε(λ) × L_bb(T,λ) + ρ(λ) × [L_atm↓(λ) + L_sun(λ)×cos(θ)]
        //
        // where:
        //   ε(λ) = emissivity (derived from metallicFactor if not set - P1 fix)
        //   L_bb(T,λ) = Planck blackbody radiance at temperature T
        //   ρ(λ) = reflectance = 1 - ε - τ (Kirchhoff's law)
        //   L_atm↓ = atmospheric downwelling radiance (sky thermal)
        //   L_sun = solar irradiance (significant in MWIR 3-5μm, negligible in LWIR)
        //
        // MWIR Solar Contribution (P2 fix):
        //   At 4μm, solar irradiance ≈ 5 W·m⁻²·μm⁻¹ (AM1.5)
        //   For T=300K surface: thermal emission ≈ 0.24 W·sr⁻¹·m⁻²·μm⁻¹
        //   Solar reflection can contribute 5-20% of total radiance for sunlit surfaces!
        //
        // Output: Single grayscale value (band-integrated radiance)
        // ====================================================================

        // Determine wavelength range and whether to include solar term
        float lambda_min, lambda_max;
        bool includeSolarReflection = false;  // Only for MWIR during daytime

        if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED) {
            lambda_min = 3000.0;   // nm
            lambda_max = 5000.0;   // nm
            // MWIR: solar contributes 5-20% for sunlit surfaces (P2 fix)
            // Only include if surface is facing sun and sun is above horizon
            includeSolarReflection = (NdotL > 0.0);
        } else {  // LWIR
            lambda_min = 8000.0;   // nm
            lambda_max = 12000.0;  // nm
            // LWIR: solar contribution < 0.1%, skip for performance
            includeSolarReflection = false;
        }

        // Integration parameters
        const uint  NUM_IR_SAMPLES = 16;  // Fewer samples than visible (smoother spectra)
        const float lambda_step = (lambda_max - lambda_min) / float(NUM_IR_SAMPLES - 1);

        // Accumulate band-integrated radiance
        float radiance_accum = 0.0;

        // Material IR properties (P1 fix: use effective emissivity from metallic factor)
        float emissivity = GetEffectiveIREmissivity(material);
        float reflectance = GetEffectiveIRReflectance(material);

        // Atmospheric downwelling radiation temperature
        float T_atmosphere = lut.atmosphereTemperature_K;

        // Check if we have spectral solar LUT for accurate MWIR illumination
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Sun solid angle for converting surface radiance to irradiance
        // Ω_sun ≈ 6.8e-5 sr (subtends ~0.53° angular diameter)
        const float SUN_SOLID_ANGLE_SR = 6.8e-5;

        // Loop over wavelengths in IR band
        // NOTE: Removed [unroll] to reduce shader compilation time
        for (uint i = 0; i < NUM_IR_SAMPLES; ++i) {
            float lambda = lambda_min + float(i) * lambda_step;

            // 1. Self-emission: ε(λ) × L_blackbody(T_surface, λ)
            float L_emission = 0.0;
            if (material.irTemperature_K > 0.0) {
                float L_blackbody = IRPlanckRadiance(material.irTemperature_K, lambda);
                L_emission = emissivity * L_blackbody;
            }

            // 2. Reflected atmospheric downwelling radiation: ρ(λ) × L_atmosphere(T_atm, λ)
            float L_downwelling = IRPlanckRadiance(T_atmosphere, lambda);
            float L_reflected_atm = reflectance * L_downwelling;

            // 3. Reflected solar radiance (P2 fix: MWIR daytime solar contribution)
            float L_reflected_sun = 0.0;
            if (includeSolarReflection && NdotL > 0.0) {
                // Get solar spectral irradiance at this MWIR wavelength
                float sun_irr_lambda = 0.0;

                if (hasSpectralSolarLUT) {
                    // Query ASTM G-173 or similar (if data extends to MWIR)
                    sun_irr_lambda = SampleSunIrradiance(solarSpectralLUT[0], lambda);
                } else {
                    // Fallback: Planck approximation for sun at 5778K
                    // L_sun(λ) = B(T_sun, λ) × Ω_sun × (R_sun / D_earth-sun)²
                    // The irradiance at Earth is already factored into typical solar data
                    // Here we use Planck at 5778K scaled to match AM0 solar constant
                    float L_sun_surface = IRPlanckRadiance(5778.0, lambda);
                    // Scale by sun solid angle to get approximate irradiance
                    // This is a rough approximation; use spectral LUT for accuracy
                    sun_irr_lambda = L_sun_surface * SUN_SOLID_ANGLE_SR * 1e4;  // W/m²/μm approximate
                }

                // Convert irradiance to radiance and apply Lambertian BRDF
                // L_reflected = ρ/π × E_sun × cos(θ) for diffuse surfaces
                // For simplicity, using ρ × (E/π) × NdotL
                float sun_radiance_lambda = sun_irr_lambda / PI;
                L_reflected_sun = reflectance * sun_radiance_lambda * NdotL;

                // Apply atmospheric transmittance on sun-surface path (if available)
                // TODO: Query AtmosphereTransmittanceLUT for accurate path transmittance
                // For now, assume typical MWIR transmittance of ~0.8 for clear sky
                const float MWIR_ATM_TRANSMITTANCE_APPROX = 0.8;
                L_reflected_sun *= MWIR_ATM_TRANSMITTANCE_APPROX;
            }

            // 4. Total spectral radiance at this wavelength
            float L_lambda = L_emission + L_reflected_atm + L_reflected_sun;

            // Accumulate (Riemann sum)
            radiance_accum += L_lambda * lambda_step;
        }

        // Normalize by wavelength range to get average radiance over band
        float band_width = lambda_max - lambda_min;
        float radiance_avg = radiance_accum / band_width;

        // Validation
        if (!isfinite(radiance_avg)) {
            radiance_avg = 0.0;
        }
        radiance_avg = clamp(radiance_avg, 0.0, 1e6);  // Allow high dynamic range for IR

        // Output as grayscale (IR images are single-channel)
        output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);

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

    // ========================================================================
    // Debug Visualization Output
    // ========================================================================
    // If debug mode is enabled, output debug visualization instead of normal radiance
    // This allows inspecting intermediate rendering data for debugging pipeline issues
    // ========================================================================

    if (camera.debug_mode != DEBUG_MODE_NONE) {
        float3 debug_output = float3(1.0, 0.0, 1.0);  // Magenta = unhandled mode

        switch (camera.debug_mode) {
            // ----------------------------------------------------------------
            // Geometry Debug (1-9)
            // ----------------------------------------------------------------
            case DEBUG_MODE_WORLD_POSITION: {
                // World position (use frac for visibility, scaled by 0.1)
                float3 worldPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
                debug_output = frac(worldPos * 0.1);
                break;
            }
            case DEBUG_MODE_GEOMETRIC_NORMAL:
                // TRUE geometric normal from triangle edges (flat normal, remap [-1,1] to [0,1])
                // This is NOT the interpolated shading normal - it's the actual surface orientation
                debug_output = worldGeometricNormal * 0.5 + 0.5;
                break;

            case DEBUG_MODE_SHADED_NORMAL:
                // Final shaded normal with normal map (remap [-1,1] to [0,1])
                debug_output = normal * 0.5 + 0.5;
                break;

            case DEBUG_MODE_TANGENT:
                // Tangent vector (remap [-1,1] to [0,1])
                debug_output = worldTangent * 0.5 + 0.5;
                break;

            case DEBUG_MODE_UV:
                // UV coordinates (use frac for tiling visibility)
                debug_output = float3(frac(uv), 0.0);
                break;

            case DEBUG_MODE_MATERIAL_ID:
                // Material index hashed to color
                debug_output = HashToColor(materialID);
                break;

            case DEBUG_MODE_TRIANGLE_ID:
                // Triangle/primitive index hashed to color
                debug_output = HashToColor(PrimitiveIndex());
                break;

            case DEBUG_MODE_BARYCENTRIC:
                // Barycentric coordinates (b0, b1, b2) where b0 = 1 - b1 - b2
                debug_output = float3(1.0 - attribs.bary.x - attribs.bary.y,
                                      attribs.bary.x,
                                      attribs.bary.y);
                break;

            // ----------------------------------------------------------------
            // Material Debug (10-19)
            // ----------------------------------------------------------------
            case DEBUG_MODE_BASE_COLOR:
                // Albedo/base color RGB
                debug_output = baseColor.rgb;
                break;

            case DEBUG_MODE_METALLIC:
                // Metallic factor (grayscale)
                debug_output = float3(metallic, metallic, metallic);
                break;

            case DEBUG_MODE_ROUGHNESS:
                // Roughness factor (grayscale)
                debug_output = float3(roughness, roughness, roughness);
                break;

            case DEBUG_MODE_NORMAL_MAP_DELTA: {
                // Normal map contribution (difference from geometric normal)
                float3 delta = normal - worldNormal;
                debug_output = delta * 0.5 + 0.5;
                break;
            }

            case DEBUG_MODE_EMISSIVE:
                // Emissive RGB (tone map for visibility)
                debug_output = emissive / (1.0 + emissive);  // Simple Reinhard
                break;

            case DEBUG_MODE_ALPHA:
                // Alpha channel (grayscale)
                debug_output = float3(baseColor.a, baseColor.a, baseColor.a);
                break;

            // ----------------------------------------------------------------
            // Lighting Debug (20-29)
            // ----------------------------------------------------------------
            case DEBUG_MODE_NDOTL:
                // N dot L (grayscale)
                debug_output = float3(NdotL, NdotL, NdotL);
                break;

            case DEBUG_MODE_NDOTV: {
                // N dot V (grayscale)
                float NdotV_val = max(dot(normal, V), 0.0);
                debug_output = float3(NdotV_val, NdotV_val, NdotV_val);
                break;
            }

            case DEBUG_MODE_DIRECT_SUN:
                // Direct sun contribution (tone map for visibility)
                debug_output = directSun / (1.0 + directSun);
                break;

            case DEBUG_MODE_DIFFUSE:
                // Diffuse component (kD * albedo)
                debug_output = kD * albedo;
                break;

            case DEBUG_MODE_ATMOSPHERIC_TRANS:
                // Atmospheric transmittance (grayscale)
                debug_output = float3(atmosphericTransmittance, atmosphericTransmittance, atmosphericTransmittance);
                break;

            // ----------------------------------------------------------------
            // BRDF Debug (30-39)
            // ----------------------------------------------------------------
            case DEBUG_MODE_FRESNEL_F0:
                // Fresnel at normal incidence (F0)
                debug_output = F0;
                break;

            case DEBUG_MODE_FRESNEL: {
                // Fresnel at current viewing angle
                float NdotV_fresnel = max(dot(normal, V), 0.0);
                debug_output = FresnelSchlick(F0, NdotV_fresnel);
                break;
            }

            case DEBUG_MODE_BRDF:
                // Full Cook-Torrance BRDF (scale for visibility)
                debug_output = brdf * 0.1;  // Scale down since BRDF can be large
                break;

            // ----------------------------------------------------------------
            // IBL Debug (40-49)
            // ----------------------------------------------------------------
            case DEBUG_MODE_REFLECTION_DIR: {
                // Reflection direction (remap [-1,1] to [0,1])
                float3 R = reflect(-V, normal);
                debug_output = R * 0.5 + 0.5;
                break;
            }

            case DEBUG_MODE_PREFILTERED_ENV:
                // Prefiltered environment map sample
                debug_output = prefilteredColor;
                break;

            case DEBUG_MODE_BRDF_LUT:
                // BRDF LUT sample (scale, bias, 0)
                debug_output = float3(envBRDF, 0.0);
                break;

            case DEBUG_MODE_IBL_SPECULAR:
                // IBL specular contribution
                debug_output = iblSpecular;
                break;

            case DEBUG_MODE_SKY_AMBIENT:
                // Sky ambient/diffuse contribution
                debug_output = skyAmbient;
                break;

            // ----------------------------------------------------------------
            // Spectral Debug (50-59)
            // ----------------------------------------------------------------
            case DEBUG_MODE_XYZ:
                // For spectral modes, this would show XYZ values
                // For RGB mode, show a placeholder
                debug_output = output_radiance;  // Current output as fallback
                break;

            case DEBUG_MODE_BEFORE_CHROMA:
                // RGB before chromaticity correction (same as XYZ for now)
                debug_output = output_radiance;
                break;

            case DEBUG_MODE_SPECTRAL_REFL: {
                // Spectral reflectance at 550nm (green)
                float refl550 = (albedo.r + albedo.g + albedo.b) / 3.0;  // Approximate
                debug_output = float3(refl550, refl550, refl550);
                break;
            }

            // ----------------------------------------------------------------
            // IR Debug (60-69)
            // ----------------------------------------------------------------
            case DEBUG_MODE_TEMPERATURE: {
                // Surface temperature (colormap 200K - 500K for visibility)
                float temp_K = material.irTemperature_K;
                if (temp_K <= 0.0) temp_K = 300.0;  // Default to room temp
                debug_output = TemperatureToColor(temp_K, 200.0, 500.0);
                break;
            }

            case DEBUG_MODE_IR_EMISSIVITY: {
                // IR emissivity (grayscale)
                float emissivity = GetEffectiveIREmissivity(material);
                debug_output = float3(emissivity, emissivity, emissivity);
                break;
            }

            case DEBUG_MODE_IR_EMISSION: {
                // Thermal emission component (grayscale, scaled)
                float temp_K = material.irTemperature_K;
                if (temp_K <= 0.0) temp_K = 300.0;
                float emission = GetEffectiveIREmissivity(material) * IRPlanckRadiance(temp_K, 10000.0);
                emission = emission / (1.0 + emission);  // Tone map
                debug_output = float3(emission, emission, emission);
                break;
            }

            case DEBUG_MODE_IR_REFLECTION: {
                // IR reflection component (grayscale)
                float refl = GetEffectiveIRReflectance(material);
                debug_output = float3(refl, refl, refl);
                break;
            }

            default:
                // Unknown mode: show magenta error color
                debug_output = float3(1.0, 0.0, 1.0);
                break;
        }

        // Apply debug output and return early
        payload.radiance = debug_output;
        return;
    }

    payload.radiance = output_radiance;
}
