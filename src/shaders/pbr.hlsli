// ============================================================================
// Quantiloom - PBR Utilities
// ============================================================================
// Cook-Torrance microfacet BRDF implementation
// Based on Disney/Epic Games PBR model with GGX distribution
//
// References:
// - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
// - "Physically Based Shading at Disney" (Brent Burley, Disney, 2012)
// - glTF 2.0 specification (KHR_materials_pbrMetallicRoughness)
//
// BRDF Model:
// - Microfacet specular: Cook-Torrance with GGX/Trowbridge-Reitz NDF
// - Diffuse: Lambertian (albedo / π)
// - Energy conservation: diffuse term scaled by (1 - F) for metals
// ============================================================================

#ifndef QUANTILOOM_PBR_HLSLI
#define QUANTILOOM_PBR_HLSLI

#include "common.hlsli"

// ============================================================================
// Complex Refractive Index Buffer (binding 14)
// ============================================================================
// Complex refractive index (n, k) for physical conductor Fresnel, indexed by
// MaterialData::complexRefractiveIndexIndex. Declared here because the BRDF
// functions below read it directly; entry shaders must not redeclare it.
// Data source: RefractiveIndex.INFO database (measured metal optical constants)
//
// When complexRefractiveIndexIndex >= 0, use physical Fresnel equation:
//   F = FresnelConductor(cosθ, n(λ), k(λ))
// Otherwise, use standard PBR approximation:
//   F = F0 + (1-F0) * (1-cosθ)^5
// ============================================================================

[[vk::binding(14, 0)]] StructuredBuffer<ComplexRefractiveIndexGPU> complexRefractiveIndices;

static const float PI = 3.14159265358979323846;
static const float EPSILON = 1e-6;

// ============================================================================
// Fresnel Term (Schlick Approximation)
// ============================================================================
// Both variants live in common.hlsli, and both take the cosine FIRST:
//
//     float  FresnelSchlick   (float cosTheta, float  F0)
//     float3 FresnelSchlickRGB(float cosTheta, float3 F0)
//
// This file used to declare a float3 overload of FresnelSchlick with the
// arguments the other way round. HLSL overload resolution then bound scalar
// calls written in THAT order to common.hlsli's exact (float, float) match --
// silently, with the arguments swapped -- which broke every spectral mode:
// F came out ~0.94 on a dielectric instead of ~0.04 and buried the measured
// reflectance under a specular pedestal. Do not reintroduce a same-named
// overload with a different argument order.
// ============================================================================

// ============================================================================
// GGX Normal Distribution Function (Trowbridge-Reitz)
// ============================================================================
// Distribution of microfacet normals (D term in Cook-Torrance)
// Determines shape of specular highlights
//
// alpha: Roughness parameter (roughness^2 for perceptually linear)
// NdotH: dot(n, h) where n=surface normal, h=half vector
// ============================================================================

float DistributionGGX(float NdotH, float alpha) {
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (alpha2 - 1.0) + 1.0;
    denom = PI * denom * denom;

    // Avoid division by zero
    return alpha2 / max(denom, EPSILON);
}

// ============================================================================
// Smith GGX Geometric Shadowing (Height-Correlated)
// ============================================================================
// Accounts for microfacet self-shadowing and masking
// G1: Single-direction shadowing term (Schlick-GGX approximation)
// G: Combined shadowing-masking term for view and light directions
//
// k: Remapping of alpha for direct lighting (Epic Games formula)
//    k = (alpha + 1)^2 / 8 for direct lighting
// ============================================================================

float GeometrySchlickGGX(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k + EPSILON);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    // Remap roughness for direct lighting (Epic Games approach)
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float ggx1 = GeometrySchlickGGX(NdotL, k);  // Light direction
    float ggx2 = GeometrySchlickGGX(NdotV, k);  // View direction

    return ggx1 * ggx2;
}

// The environment variant, k = alpha/2 with alpha = roughness^2. Same function,
// different remap, and which one is correct depends on what the result is
// weighed against rather than on the surface: the split-sum envBRDF LUT is
// built with this k (BRDFLutGenerator::GeometrySmith_GGX_IBL and
// ibl_brdf_lut.comp, which agree), so a traced environment bounce has to use it
// too. Weighting a bounce with the direct-lighting remap and subtracting a
// base term built from the LUT leaves a roughness-dependent residue that looks
// like a physical effect and is not one.
float GeometrySmith_IBL(float NdotV, float NdotL, float roughness) {
    float k = (roughness * roughness) / 2.0;

    float ggx1 = GeometrySchlickGGX(NdotL, k);
    float ggx2 = GeometrySchlickGGX(NdotV, k);

    return ggx1 * ggx2;
}

// ============================================================================
// Optimized Visibility Term: Vis = G / (4 * NdotV * NdotL)
// ============================================================================
// Combines the geometric shadowing term and Cook-Torrance denominator
// into a single, optimized calculation that avoids division by NdotV/NdotL.
//
// PERFORMANCE BENEFITS:
// - Fewer instructions (eliminates redundant multiplications)
// - Better numerical stability (cancels out NdotV/NdotL in numerator/denominator)
// - Reduces "fireflies" artifacts at grazing angles
//
// MATHEMATICAL DERIVATION:
// For Schlick-GGX with k = (roughness+1)²/8 (direct) or k = roughness²/2 (IBL):
//   G = G₁(NdotL) × G₁(NdotV)
//   G₁(x) = x / (x(1-k) + k)
//
// Therefore:
//   Vis = G / (4·NdotV·NdotL)
//       = [NdotL/(NdotL(1-k)+k)] × [NdotV/(NdotV(1-k)+k)] / (4·NdotV·NdotL)
//       = 1 / [4 × (NdotL(1-k)+k) × (NdotV(1-k)+k)]
//
// This formulation completely eliminates NdotV and NdotL from the numerator!
//
// TWO VERSIONS PROVIDED:
// - VisibilitySmithGGXCorrelated: For direct lighting, k = (roughness+1)²/8
// - VisibilitySmithGGXCorrelatedIBL: For IBL, k = roughness²/2
//
// References:
// - "Optimizing PBR" (Sébastien Lagarde, 2014)
// - "Moving Frostbite to PBR" (EA Frostbite, 2014)
// - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
// ============================================================================

float VisibilitySmithGGXCorrelated(float NdotV, float NdotL, float roughness) {
    // Remap roughness for direct lighting (Epic Games approach)
    // k = (roughness + 1)² / 8
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    // Optimized visibility term: 1 / [4 × (NdotL(1-k)+k) × (NdotV(1-k)+k)]
    // This formulation avoids division by NdotV/NdotL, improving stability
    float oneMinusK = 1.0 - k;
    float denomL = NdotL * oneMinusK + k;
    float denomV = NdotV * oneMinusK + k;

    // Combined denominator with safety clamp
    float denominator = 4.0 * denomL * denomV;

    return 1.0 / max(denominator, EPSILON);
}

// ============================================================================
// IBL-Specific Visibility Term
// ============================================================================
// For Image-Based Lighting, the k remapping differs from direct lighting:
//   k_ibl = roughness² / 2
//
// This is because IBL integrates over the entire hemisphere, not a single
// light direction. The different k value better approximates the geometry
// term for environment lighting.
//
// Reference: "Real Shading in Unreal Engine 4" (Brian Karis, 2013)
// ============================================================================

float VisibilitySmithGGXCorrelatedIBL(float NdotV, float NdotL, float roughness) {
    // Remap roughness for IBL
    // k = roughness² / 2
    float alpha = roughness * roughness;
    float k = alpha / 2.0;

    // Same optimized formulation as direct lighting
    float oneMinusK = 1.0 - k;
    float denomL = NdotL * oneMinusK + k;
    float denomV = NdotV * oneMinusK + k;

    float denominator = 4.0 * denomL * denomV;

    return 1.0 / max(denominator, EPSILON);
}

// ============================================================================
// Helper: Compute F0 (normal incidence reflectance)
// ============================================================================
// Computes F0 for both RGB and spectral paths, with optional physical n,k data.
//
// PHYSICAL PATH (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0):
//   - Query measured complex refractive index at current wavelength
//   - Use exact Fresnel equation: F0 = [(n-1)^2 + k^2] / [(n+1)^2 + k^2]
//
// PBR PATH (index < 0 or wavelength invalid):
//   - Standard approximation: F0 = lerp(0.04, albedo, metallic)
// ============================================================================

float3 ComputeF0(float3 albedo, float metallic,
                 int complexRefractiveIndexIndex, float wavelength_nm) {
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        float F0_physical = FresnelF0(nk.x, nk.y);
        return float3(F0_physical, F0_physical, F0_physical);
    }
    return lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
}

float ComputeF0_Scalar(float spectralAlbedo, float metallic,
                       int complexRefractiveIndexIndex, float wavelength_nm) {
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        return FresnelF0(nk.x, nk.y);
    }
    return lerp(0.04, spectralAlbedo, metallic);
}

// ============================================================================
// Refraction Index at a Wavelength
// ============================================================================
// The n that Snell's law and the dielectric Fresnel term should use, in
// priority order:
//
//   1. MEASURED n(λ) from the material's refractive-index table (binding 14,
//      loaded from a refractiveindex.info YAML). This is real data over the
//      wavelengths it covers, and it is right where a Cauchy fit is not:
//      the infrared, and any region of anomalous dispersion.
//   2. Two-term Cauchy from n_d and the Abbe number.
//
// The table was already being loaded and bound -- it is what gives metals
// their wavelength-dependent F0 above -- but refraction ignored it and used
// `material.ior` with a Cauchy fit. For a renderer covering 380-15000 nm that
// is backwards: the fit is the fallback, the measurement is the answer.
//
// Only the real part is used. k is absorption, which belongs to Fresnel and to
// Beer-Lambert, not to the refraction angle.
//
// @param wavelength_nm  Pass 0 to mean "no wavelength in this mode" (RGB),
//                       which falls through to the material's n_d.
// ============================================================================

float RefractionIOR(MaterialData material, float wavelength_nm) {
    if (wavelength_nm <= 0.0) {
        return material.ior;
    }
    if (material.complexRefractiveIndexIndex >= 0) {
        ComplexRefractiveIndexGPU cri =
            complexRefractiveIndices[material.complexRefractiveIndexIndex];
        return SampleComplexRefractiveIndex(cri, wavelength_nm).x;
    }
    // Degenerates to material.ior when dispersion is 0, so this is safe as the
    // single entry point for every refraction site.
    return CauchyIOR(material.ior, material.dispersion, wavelength_nm);
}

// ============================================================================
// Cook-Torrance Microfacet BRDF
// ============================================================================
// Full PBR BRDF combining diffuse and specular terms
//
// Inputs:
// - N: Surface normal (world space, normalized)
// - V: View direction (FROM surface TO camera, normalized)
// - L: Light direction (FROM surface TO light, normalized)
// - albedo: Base color (linear RGB, [0,1])
// - metallic: Metalness [0,1] (0=dielectric, 1=metal)
// - roughness: Roughness [0,1] (0=smooth, 1=rough)
// - complexRefractiveIndexIndex: Index into gComplexRefractiveIndices (-1 = none)
// - wavelength_nm: Current wavelength for spectral lookup (>0 = valid)
//
// Output:
// - BRDF value (unitless, multiply by incident radiance and NdotL for final color)
// ============================================================================

// Safe half-vector computation: returns fallback (normal) if V and L are opposite
// This prevents NaN from normalize(zero_vector) which can cause GPU hangs
float3 SafeHalfVector(float3 V, float3 L, float3 N) {
    float3 sum = V + L;
    float lenSq = dot(sum, sum);
    // If V and L are nearly opposite, fall back to surface normal
    // This is physically plausible (grazing angle case)
    if (lenSq < 1e-8) {
        return N;
    }
    return sum * rsqrt(lenSq);
}

float3 CookTorranceBRDF(
    float3 N,
    float3 V,
    float3 L,
    float3 albedo,
    float metallic,
    float roughness,
    int complexRefractiveIndexIndex,
    float wavelength_nm
) {
    // OPTIMIZATION: Clamp minimum roughness to prevent numerical instability
    // Perfectly smooth surfaces (roughness=0) lead to Dirac delta distribution
    // which causes NaN and fireflies. Minimum value of 0.045 is perceptually smooth.
    // References: UE4, Unity HDRP, Frostbite all use similar clamping
    const float MIN_ROUGHNESS = 0.045;
    roughness = max(roughness, MIN_ROUGHNESS);

    // Compute half vector (with safety check for opposite V and L)
    // FIXED: Use SafeHalfVector to prevent NaN when V + L is near-zero
    float3 H = SafeHalfVector(V, L, N);

    // Compute dot products (clamped to avoid negative values)
    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Compute F0 (reflectance at normal incidence)
    // Uses physical n,k data when available for wavelength-accurate metals
    float3 F0 = ComputeF0(albedo, metallic, complexRefractiveIndexIndex, wavelength_nm);

    // ========================================================================
    // Specular Term (Cook-Torrance microfacet BRDF) - OPTIMIZED
    // ========================================================================

    // Fresnel term: use exact conductor Fresnel when n,k data is available
    float3 F;
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        F = float3(FresnelConductor(VdotH, nk.x, nk.y),
                   FresnelConductor(VdotH, nk.x, nk.y),
                   FresnelConductor(VdotH, nk.x, nk.y));
    } else {
        F = FresnelSchlickRGB(VdotH, F0);
    }

    // Normal distribution function (GGX)
    float alpha = roughness * roughness;  // Perceptually linear roughness
    float D = DistributionGGX(NdotH, alpha);

    // OPTIMIZATION: Use combined visibility term instead of G/(4*NdotV*NdotL)
    // This eliminates NdotV/NdotL divisions, improving performance and stability
    float Vis = VisibilitySmithGGXCorrelated(NdotV, NdotL, roughness);

    // Cook-Torrance specular BRDF: D * F * Vis
    // Note: Vis already includes the 1/(4*NdotV*NdotL) term
    float3 specular = D * F * Vis;

    // ========================================================================
    // Diffuse Term (Lambertian)
    // ========================================================================

    // Energy conservation: kD = 1 - kS (where kS = F)
    // For metals, diffuse contribution is zero (kD = 0)
    float3 kD = (1.0 - F) * (1.0 - metallic);

    // Lambertian diffuse BRDF: albedo / π
    float3 diffuse = kD * albedo / PI;

    // ========================================================================
    // Combined BRDF (diffuse + specular)
    // ========================================================================
    // NOTE: For future enhancement, consider multi-scattering energy compensation
    // for rough surfaces to prevent darkening. See Kulla-Conty 2017 or
    // Turquin 2019 practical approximations.

    return diffuse + specular;
}

// ============================================================================
// Native Scalar Spectral Cook-Torrance BRDF
// ============================================================================
// Computes the BRDF as a scalar directly without constructing float3 and
// averaging channels. Uses exact conductor Fresnel when n,k data is available.
// ============================================================================

float CookTorranceBRDF_Spectral(
    float3 N,
    float3 V,
    float3 L,
    float spectralAlbedo,
    float metallic,
    float roughness,
    int complexRefractiveIndexIndex,
    float wavelength_nm
) {
    const float MIN_ROUGHNESS = 0.045;
    roughness = max(roughness, MIN_ROUGHNESS);

    float3 H = SafeHalfVector(V, L, N);

    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Compute F0 and Fresnel term
    float F0 = ComputeF0_Scalar(spectralAlbedo, metallic, complexRefractiveIndexIndex, wavelength_nm);

    float F;
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        ComplexRefractiveIndexGPU cri = complexRefractiveIndices[complexRefractiveIndexIndex];
        float2 nk = SampleComplexRefractiveIndex(cri, wavelength_nm);
        F = FresnelConductor(VdotH, nk.x, nk.y);
    } else {
        F = FresnelSchlick(VdotH, F0);
    }

    float alpha = roughness * roughness;
    float D = DistributionGGX(NdotH, alpha);
    float Vis = VisibilitySmithGGXCorrelated(NdotV, NdotL, roughness);

    float specular = D * F * Vis;

    // Diffuse term
    float kD = (1.0 - F) * (1.0 - metallic);
    float diffuse = kD * spectralAlbedo / PI;

    return diffuse + specular;
}

// ============================================================================
// Image-Based Lighting (IBL) Functions
// ============================================================================
// Split-sum approximation for environment map specular reflection
//
// References:
// - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
// - "Moving Frostbite to PBR" (Sébastien Lagarde, EA Frostbite, 2014)
//
// Split-Sum Approximation:
//   ∫ L(ωi) * f(ωo, ωi) * (n·ωi) dωi
//   ≈ ∫ L(ωi) dωi * ∫ f(ωo, ωi) * (n·ωi) dωi
//   = prefilteredColor * BRDF_LUT(NdotV, roughness)
//
// BRDF_LUT stores pre-integrated Fresnel term:
//   BRDF_LUT.r = scale term (for F0)
//   BRDF_LUT.g = bias term (for (1-F0)^5)
//
// Final IBL specular:
//   L_ibl = prefilteredColor * (F0 * brdfLUT.r + brdfLUT.g)
// ============================================================================

// Fresnel-Schlick with roughness correction for IBL
// Accounts for energy loss at grazing angles for rough surfaces
// Cosine first, matching FresnelSchlick / FresnelSchlickRGB in common.hlsli.
float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    cosTheta = saturate(cosTheta);
    float oneMinusCos = 1.0 - cosTheta;

    // OPTIMIZATION: Replace pow(x, 5.0) with multiplication chain
    // This is significantly faster on GPU (3 multiplications vs expensive pow)
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;

    // Roughness correction: interpolate between F0 and 1 based on roughness
    // At grazing angles, rough surfaces still show full Fresnel reflection
    float3 maxReflection = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxReflection - F0) * oneMinusCos5;
}

// Compute specular reflection direction (mirror reflection)
// Reflects V (view direction) around N (surface normal)
// Returns normalized reflection vector
float3 ComputeReflectionDirection(float3 V, float3 N) {
    // R = 2 * (N·V) * N - V
    return reflect(-V, N);  // Note: reflect() expects incident vector pointing TO surface
}

// Sample BRDF integration LUT for IBL
// Returns (scale, bias) for split-sum approximation
//
// Texture format: RG16F or RG32F
//   R channel: scale term (multiply by F0)
//   G channel: bias term (add as is)
//
// Inputs:
//   NdotV: dot(N, V), range [0, 1]
//   roughness: surface roughness, range [0, 1]
float2 SampleBRDF_LUT(Texture2D<float2> brdfLUT, SamplerState sampler, float NdotV, float roughness) {
    // BRDF LUT is parameterized by (NdotV, roughness)
    // x-axis: NdotV (0 = grazing, 1 = normal incidence)
    // y-axis: roughness (0 = smooth, 1 = rough)
    float2 uv = float2(saturate(NdotV), saturate(roughness));
    return brdfLUT.SampleLevel(sampler, uv, 0.0);
}

// Evaluate IBL specular contribution (split-sum approximation)
//
// Inputs:
//   N: surface normal (world space, normalized)
//   V: view direction (FROM surface TO camera, normalized)
//   F0: reflectance at normal incidence (specular color)
//   roughness: surface roughness [0, 1]
//   prefilteredColor: environment radiance from reflection direction
//   brdfLUT: pre-integrated BRDF lookup texture
//   samplerState: sampler for BRDF LUT
//
// Output:
//   IBL specular radiance (W·sr⁻¹·m⁻²)
float3 EvaluateIBLSpecular(
    float3 N,
    float3 V,
    float3 F0,
    float roughness,
    float3 prefilteredColor,
    Texture2D<float2> brdfLUT,
    SamplerState samplerState
) {
    float NdotV = max(dot(N, V), 0.0);

    // Sample BRDF LUT (pre-integrated Fresnel term)
    float2 brdf = SampleBRDF_LUT(brdfLUT, samplerState, NdotV, roughness);

    // Split-sum approximation: prefilteredColor * (F0 * scale + bias)
    // brdf.r = scale term (multiply by F0)
    // brdf.g = bias term (add directly)
    float3 iblSpecular = prefilteredColor * (F0 * brdf.r + brdf.g);

    return iblSpecular;
}

// ============================================================================
// Multi-Scattering Energy Compensation (Future Enhancement)
// ============================================================================
// The standard Cook-Torrance BRDF models SINGLE-SCATTERING only: light bounces
// once off the microfacet surface. In reality, rough surfaces exhibit
// MULTI-SCATTERING: light bounces multiple times between microfacets before
// exiting.
//
// PROBLEM: Energy Loss (Darkening)
// At high roughness, single-scattering loses energy because light that bounces
// into valleys between microfacets is not accounted for. This causes rough
// materials (especially metals) to appear darker than they should physically.
//
// Example: A rough aluminum ball looks darker than it should because the BRDF
// doesn't account for light bouncing multiple times in the micro-grooves.
//
// SOLUTION: Energy Compensation
// Several methods exist to approximate multi-scattering:
//
// 1. **Kulla-Conty 2017** (Most Accurate, Expensive)
//    - Pre-computes energy loss in a 2D LUT: E(μ, α) where μ=NdotV, α=roughness
//    - Adds compensation term: (1 - E) × F_avg × albedo
//    - Requires additional texture lookup and storage
//    - Reference: "Revisiting Physically Based Shading at Imageworks" (Kulla & Conty, 2017)
//
// 2. **Turquin 2019** (Practical Approximation)
//    - Analytical approximation without LUT
//    - Adds term: k × (1 - F) × roughness² where k ≈ 0.5-1.0
//    - Simpler but less accurate than Kulla-Conty
//    - Reference: "Practical multiple scattering compensation for microfacet models" (Turquin, 2019)
//
// 3. **Fdez-Agüera 2021** (LUT-free Analytical)
//    - High-quality analytical fit to multi-scattering
//    - No texture lookups, polynomial approximation
//    - Reference: "A Multiple-Scattering Microfacet Model for Real-Time IBL" (Fdez-Agüera, 2021)
//
// WHEN TO IMPLEMENT:
// - If rough metallic materials look too dark (especially at grazing angles)
// - If physical accuracy is critical (product visualization, material design)
// - If rendering budget allows for extra texture lookup or computation
//
// IMPLEMENTATION NOTES:
// - Multi-scattering affects BOTH direct lighting and IBL
// - For metals: effect is most visible (colored multi-scattering)
// - For dielectrics: effect is subtle but measurable
// - Can be implemented as post-process or integrated into BRDF
//
// PSEUDOCODE (Kulla-Conty):
// ```
// float E_o = energyLUT.Sample(NdotV, roughness).r;  // Outgoing energy loss
// float E_avg = energyLUT.Sample(0.5, roughness).g;  // Average energy loss
// float F_avg = F0;  // Simplified, or use average Fresnel
//
// float3 energyCompensation = (1.0 - E_o) * F_avg * albedo / (1.0 - F_avg * (1.0 - E_avg));
// float3 finalBRDF = standardBRDF + energyCompensation;
// ```
//
// REFERENCES:
// - Kulla & Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH 2017
// - Turquin, "Practical multiple scattering compensation for microfacet models", 2019
// - Fdez-Agüera, "A Multiple-Scattering Microfacet Model for Real-Time IBL", JCGT 2021

// ============================================================================
// PCG Random + Importance Sampling for IR Monte Carlo Reflection
// ============================================================================

uint pcg_hash(uint s) { s = s*747796405u + 2891336453u; s = ((s>>((s>>28)+4))^s)*277803737u; return (s>>22)^s; }
float pcg_float(inout uint s) { s = pcg_hash(s); return float(s) * (1.0/4294967296.0); }

// Build orthonormal basis around N
void BuildBasis(float3 N, out float3 T, out float3 B) {
    float3 up = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// GGX importance sample — returns world-space wi, outputs pdf
// The roughness floor every bounce-side function applies before doing anything
// with it. Matches the one CookTorranceBRDF_Spectral already imposes on the
// shading side. Without it a perfectly smooth surface reaches DistributionGGX
// with alpha = 0, whose denominator is zero at NdotH = 1, and the resulting NaN
// slips through a `pdf <= 1e-6` guard because every comparison against NaN is
// false. It surfaced as black speckle on polished surfaces.
//
// Load-bearing that all three of BsdfMixturePdf, EvalBounceBrdf and
// TraceEnvBounceResidual apply the SAME floor: they are a sampling density, a
// BRDF evaluation and a weight for one and the same lobe, and MIS only
// partitions correctly if they agree on which lobe that is.
static const float MIN_BOUNCE_ROUGHNESS = 0.045;

// Smith's masking function for GGX, exact rather than the Schlick fit the
// shading terms use. This one is the sampling density's, so it has to be the
// real thing: it is what SampleGGXVNDF_PCG below actually draws from.
float SmithG1_GGX(float NdotV, float alpha) {
    const float a2 = alpha * alpha;
    const float c  = max(NdotV, 1e-4);
    return 2.0 * c / (c + sqrt(a2 + (1.0 - a2) * c * c));
}

// Sample a half vector from the distribution of VISIBLE normals (Heitz 2018,
// "Sampling the GGX Distribution of Visible Normals", JCGT 7(4)).
//
// The previous sampler drew from D alone, which proposes half vectors the view
// direction cannot actually see. Those come back with a weight of
// G * VdotH / (NdotV * NdotH) -- a ratio with nothing bounding it, since NdotH
// is floored at 1e-4 and NDF sampling is exactly what puts H far from N. The
// result is the classic firefly: one sample worth thousands of the ones around
// it, which the running average then carries for a long time. Sampling visible
// normals instead makes the weight G2/G1, which is at most 1.
//
// Returns the reflected direction and its solid-angle density.
float3 SampleGGXVNDF_PCG(float3 N, float3 V, float alpha, inout uint rng, out float pdf) {
    float3 T, B;
    BuildBasis(N, T, B);

    // Into the tangent frame, where the normal is +Z.
    const float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));

    // Section 3.2: stretch the view direction to the hemisphere configuration.
    const float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, max(Ve.z, 1e-6)));

    // Section 4.1: an orthonormal basis around Vh.
    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq)
                                  : float3(1.0, 0.0, 0.0);
    const float3 T2 = cross(Vh, T1);

    // Section 4.2: a uniform point on the projected disk, squashed to match the
    // visible area.
    const float u1 = pcg_float(rng);
    const float u2 = pcg_float(rng);
    const float r   = sqrt(u1);
    const float phi = 2.0 * PI * u2;
    const float t1 = r * cos(phi);
    float       t2 = r * sin(phi);
    const float s  = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(1.0 - t1 * t1, 0.0)) + s * t2;

    // Section 4.3: back onto the hemisphere, then unstretch.
    const float3 Nh = t1 * T1 + t2 * T2 +
                      sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0)) * Vh;
    const float3 Ne = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(Nh.z, 0.0)));

    const float3 H  = normalize(T * Ne.x + B * Ne.y + N * Ne.z);
    const float3 wi = reflect(-V, H);

    // p(wi) = D_vis(H) / (4 VdotH), and D_vis = G1 VdotH D / NdotV, so the
    // VdotH cancels and this needs no half-vector term of its own.
    const float NdotV = max(dot(N, V), 1e-4);
    const float NdotH = max(dot(N, H), 0.0);
    pdf = SmithG1_GGX(NdotV, alpha) * DistributionGGX(NdotH, alpha) / (4.0 * NdotV);
    return wi;
}

// Cosine-weighted hemisphere sample — for rough/Lambertian surfaces
float3 CosineSampleHemisphere_PCG(float3 N, inout uint rng, out float pdf) {
    float u1 = pcg_float(rng), u2 = pcg_float(rng);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float3 wi_local = float3(r*cos(phi), r*sin(phi), sqrt(max(1.0-u1, 0.0)));
    float3 T, B;
    BuildBasis(N, T, B);
    float3 wi = normalize(T*wi_local.x + B*wi_local.y + N*wi_local.z);
    pdf = max(dot(N, wi), 0.0) / PI;
    return wi;
}
// - Heitz et al., "Multiple-scattering microfacet BSDFs with the Smith model", SIGGRAPH 2016
// ============================================================================

#endif // QUANTILOOM_PBR_HLSLI
