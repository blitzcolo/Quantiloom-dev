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

    // Read per-vertex normals and interpolate using barycentric coordinates
    // This gives smooth shading (Gouraud/Phong) instead of flat shading
    float3 n0 = normalBuffer[idx0];
    float3 n1 = normalBuffer[idx1];
    float3 n2 = normalBuffer[idx2];

    // Barycentric interpolation: n = n0 * w0 + n1 * w1 + n2 * w2
    // where w0 = (1 - bary.x - bary.y), w1 = bary.x, w2 = bary.y
    float3 objectNormal = n0 * (1.0 - attribs.bary.x - attribs.bary.y)
                        + n1 * attribs.bary.x
                        + n2 * attribs.bary.y;
    objectNormal = SafeNormalize(objectNormal, float3(0.0, 1.0, 0.0));

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

    LightingParams lut = lightingParams[0];
    // FIXED: Use SafeNormalize in case LUT data is invalid
    float3 sunDir = SafeNormalize(lut.sunDirection, float3(0.0, 1.0, 0.0));

    // Choose lighting based on spectral mode
    float3 sunRadiance;
    float3 skyRadiance;

    if (camera.spectral_mode == SPECTRAL_MODE_RGB_FUSED) {
        // RGB mode: Use full RGB lighting
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

    if (camera.spectral_mode == SPECTRAL_MODE_RGB_FUSED) {
        // ====================================================================
        // RGB_Fused Mode: True 32-Wavelength Spectral Integration
        // ====================================================================
        // Physically-correct spectral rendering with full wavelength sampling:
        //   1. Sample 32 wavelengths uniformly across visible spectrum (380-780nm)
        //   2. Compute spectral radiance L(λ) at each wavelength
        //   3. Integrate via CIE XYZ color matching functions
        //   4. Convert XYZ → Linear RGB (sRGB D65)
        //
        // This is the TRUE HS-OFF spectral rendering for visible light.
        // Performance: ~10-15x slower than single wavelength, but physically accurate.
        //
        // SPECTRAL REFLECTANCE SOURCE (priority order):
        //   1. Measured spectral curve (spectralReflectanceCurveIndex >= 0)
        //   2. RGB texture upsampling via Gaussian basis (fallback)
        // ====================================================================

        // Spectral integration parameters
        const uint   NUM_WAVELENGTH_SAMPLES = 32;
        const float  LAMBDA_MIN_VIS = 380.0;  // nm
        const float  LAMBDA_MAX_VIS = 780.0;  // nm
        const float  LAMBDA_STEP = (LAMBDA_MAX_VIS - LAMBDA_MIN_VIS) / float(NUM_WAVELENGTH_SAMPLES - 1);

        // CIE XYZ normalization factor
        // For equal-energy white (E), Y should integrate to 1.0
        // ∫ȳ(λ)dλ ≈ 106.9 over 380-780nm, so normalize by this
        const float CIE_Y_INTEGRAL = 106.9;

        // Accumulate XYZ tristimulus values
        float3 XYZ_accum = float3(0.0, 0.0, 0.0);

        // ====================================================================
        // Solar Spectral LUT: Use true spectral irradiance when available
        // ====================================================================
        // Priority:
        // 1. SolarSpectralLUT with measured ASTM G-173 spectra (preferred)
        // 2. LightingParams RGB values approximated as flat spectrum (fallback)
        // ====================================================================
        bool hasSpectralSolarLUT = (solarSpectralLUT[0].sunIrradiance.numSamples > 0);

        // Fallback: approximate RGB as flat spectrum (for legacy compatibility)
        float sun_power_rgb = (lut.sunRadiance_rgb.r + lut.sunRadiance_rgb.g + lut.sunRadiance_rgb.b) / 3.0;
        float sky_power_rgb = (lut.skyRadiance_rgb.r + lut.skyRadiance_rgb.g + lut.skyRadiance_rgb.b) / 3.0;

        // Loop over wavelengths
        [unroll]
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
                // FALLBACK: Use flat spectrum approximation from RGB values
                sun_radiance_lambda = sun_power_rgb;
                sky_radiance_lambda = sky_power_rgb;
            }

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
            // This ensures proper color reproduction for self-luminous surfaces
            float L_emissive = ConvertLinearRGBToSpectrum(emissive, lambda);

            float L_lambda = L_direct + L_ambient + L_emissive;

            // 5. Weight by CIE XYZ color matching functions
            float x_bar = CIE_X(lambda);
            float y_bar = CIE_Y(lambda);
            float z_bar = CIE_Z(lambda);

            // Riemann sum integration: ∫L(λ)×CMF(λ)dλ ≈ Σ L(λᵢ)×CMF(λᵢ)×Δλ
            XYZ_accum.x += L_lambda * x_bar * LAMBDA_STEP;
            XYZ_accum.y += L_lambda * y_bar * LAMBDA_STEP;
            XYZ_accum.z += L_lambda * z_bar * LAMBDA_STEP;
        }

        // Normalize by CIE Y integral for proper luminance scaling
        XYZ_accum /= CIE_Y_INTEGRAL;

        // XYZ → Linear RGB (sRGB D65)
        output_radiance = ConvertXYZToLinearRGB(XYZ_accum);

        // Add IBL specular reflection (already computed in RGB)
        // NOTE: IBL uses prefiltered environment map which is already in RGB space.
        // For full spectral correctness, IBL would need spectral environment maps,
        // but this is computationally prohibitive and rarely done in practice.
        output_radiance += iblSpecular;

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

        // Fallback: flat spectrum from RGB average
        float sun_power_rgb = (lut.sunRadiance_rgb.r + lut.sunRadiance_rgb.g + lut.sunRadiance_rgb.b) / 3.0;
        float sky_power_rgb = (lut.skyRadiance_rgb.r + lut.skyRadiance_rgb.g + lut.skyRadiance_rgb.b) / 3.0;

        // Material IR properties (for thermal contribution, usually negligible in SWIR)
        // Use effective emissivity that derives from metallic factor when not set (P1 fix)
        float emissivity = GetEffectiveIREmissivity(material);
        float reflectance = GetEffectiveIRReflectance(material);

        [unroll]
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
        [unroll]
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

    payload.radiance = output_radiance;
}
