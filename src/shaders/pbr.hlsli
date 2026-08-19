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
                 int complexRefractiveIndexIndex, float wavelength_nm,
                 float3 dielectricF0) {
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, complexRefractiveIndexIndex, wavelength_nm);
        float F0_physical = FresnelF0(nk.x, nk.y);
        return float3(F0_physical, F0_physical, F0_physical);
    }
    return lerp(dielectricF0, albedo, metallic);
}

float ComputeF0_Scalar(float spectralAlbedo, float metallic,
                       int complexRefractiveIndexIndex, float wavelength_nm,
                       float dielectricF0) {
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, complexRefractiveIndexIndex, wavelength_nm);
        return FresnelF0(nk.x, nk.y);
    }
    return lerp(dielectricF0, spectralAlbedo, metallic);
}

// ============================================================================
// KHR_materials_specular: the dielectric F0 and F90
// ============================================================================
// What used to be the hardcoded 0.04 above. The specification composes it as
//
//   f0_ior = ((ior - 1) / (ior + 1))^2          -- exactly 0.04 at ior 1.5
//   F0     = min(f0_ior * specularColor, 1) * specularWeight
//   F90    = specularWeight
//
// Two things about that expression are load-bearing.
//
// The clamp comes BEFORE the specularWeight multiply. specularColor is allowed
// to exceed 1 and assets use it -- SpecularSilkPouf authors [10, 0.6, 0] to
// saturate the red channel's F0 while leaving green at 0.024 and blue at zero.
// Clamping the product instead lets red out at more than a mirror.
//
// F90 is separate because specular scales the whole Fresnel curve rather than
// just its normal-incidence end. A surface with specularWeight 0 reflects
// nothing even at grazing, where the usual Schlick form would still go to 1.
// That is why FresnelSchlickF90 exists as a distinct function rather than as an
// overload of FresnelSchlick -- pbr.hlsli's own header records what happened
// the last time a same-named Fresnel overload was added.
//
// Also note this is the first time material.ior reaches the reflective F0 at
// all. It used to drive only refraction, so a glass with ior 1.33 reflected as
// if it were 1.5. At the default ior these agree to within a bit.
// ============================================================================

float3 DielectricF0RGB(float ior, float3 specularColor, float specularWeight) {
    float f = (ior - 1.0) / (ior + 1.0);
    return min(f * f * specularColor, 1.0) * specularWeight;
}

float DielectricF0Scalar(float ior, float specularColor, float specularWeight) {
    float f = (ior - 1.0) / (ior + 1.0);
    return min(f * f * specularColor, 1.0) * specularWeight;
}

// Schlick with an explicit F90. Deliberately NOT an overload of FresnelSchlick:
// that one takes (cosTheta, F0) and a same-named three-argument sibling is
// exactly the shape that silently bound the wrong way once already.
float FresnelSchlickF90(float cosTheta, float F0, float F90) {
    float f = pow(saturate(1.0 - cosTheta), 5.0);
    return F0 + (F90 - F0) * f;
}

float3 FresnelSchlickF90RGB(float cosTheta, float3 F0, float F90) {
    float f = pow(saturate(1.0 - cosTheta), 5.0);
    return F0 + (F90.xxx - F0) * f;
}

// ============================================================================
// KHR_materials_anisotropy: a stretched GGX lobe
// ============================================================================
// The extension does not add a lobe, it reshapes the one already there:
//
//   alpha_t = lerp(alpha, 1, strength^2)   along the anisotropy direction
//   alpha_b = alpha                        across it
//
// so alpha_t >= alpha_b always. This only ever roughens one axis; it never
// sharpens the other, which is why no energy bookkeeping is needed -- the lobe
// is redistributed, not enlarged.
//
// Unlike sheen, this cannot be folded into a cosine lobe with a matching
// directional albedo, because what changes IS the angular shape. So the whole
// MIS quartet -- the sampler, the mixture density, the NEE evaluation and the
// bounce weight -- takes the frame below and stretches together. Getting one of
// the four wrong does not look like a wrong highlight; it looks like a scene
// that is uniformly too bright or too dark at grazing angles, which is the
// failure EvalBounceBrdf's header records.
//
// active == false is not "strength 0" -- it routes every consumer back to the
// original isotropic expressions, bit for bit. Algebraic equality is not enough
// here: the anisotropic D and V are written in a different (better conditioned)
// form, and a scene with no anisotropy has to render exactly as it did.
// ============================================================================

struct AnisoFrame {
    float3 T;       // the direction highlights stretch along, world space
    float3 B;       // across it, world space
    float  alphaT;
    float  alphaB;
    bool   active;
};

AnisoFrame IsotropicFrame() {
    AnisoFrame f;
    f.T = float3(1.0, 0.0, 0.0);
    f.B = float3(0.0, 1.0, 0.0);
    f.alphaT = 0.0;
    f.alphaB = 0.0;
    f.active = false;
    return f;
}

/// The two alphas, from a roughness already floored by the caller.
void AnisotropyAlphas(float alpha, float strength, out float alphaT, out float alphaB) {
    alphaT = lerp(alpha, 1.0, strength * strength);
    alphaB = alpha;
}

// Burley's anisotropic GGX, in the sample viewer's algebraically equivalent but
// better conditioned arrangement:
//
//   D = 1 / (pi a_t a_b ((h.t)^2/a_t^2 + (h.b)^2/a_b^2 + (h.n)^2)^2)
float DistributionGGXAniso(float TdotH, float BdotH, float NdotH,
                           float alphaT, float alphaB) {
    const float a2 = alphaT * alphaB;
    const float3 f = float3(alphaB * TdotH, alphaT * BdotH, a2 * NdotH);
    const float  d = dot(f, f);
    if (d <= 0.0) {
        return 0.0;
    }
    const float w2 = a2 / d;
    return a2 * w2 * w2 / PI;
}

// Height-correlated Smith visibility, anisotropic. Already carries the
// 1/(4 NdotV NdotL) the specular denominator would otherwise need, exactly as
// VisibilitySmithGGXCorrelated does for the isotropic case.
float VisibilitySmithGGXCorrelatedAniso(float TdotV, float BdotV, float NdotV,
                                        float TdotL, float BdotL, float NdotL,
                                        float alphaT, float alphaB) {
    const float lambdaV = NdotL * length(float3(alphaT * TdotV, alphaB * BdotV, NdotV));
    const float lambdaL = NdotV * length(float3(alphaT * TdotL, alphaB * BdotL, NdotL));
    const float v = lambdaV + lambdaL;
    return (v > 0.0) ? (0.5 / v) : 0.0;
}

// Smith's masking for one direction, anisotropic. This is the density's G1 --
// what SampleGGXVNDFAniso actually draws from -- so like its isotropic sibling
// it is the exact form and not a fit.
float SmithG1_GGXAniso(float TdotV, float BdotV, float NdotV,
                       float alphaT, float alphaB) {
    const float c = max(NdotV, 1e-4);
    return 2.0 * c / (c + length(float3(alphaT * TdotV, alphaB * BdotV, c)));
}

// ============================================================================
// KHR_materials_clearcoat: an infinitely thin dielectric coat
// ============================================================================
// The specification layers it with a single scalar weight:
//
//   coated = mix(base, clearcoat_brdf, clearcoat * F_c)
//   F_c    = 0.04 + 0.96 (1 - |N_c . V|)^5
//
// Two things about that are easy to get wrong and both are deliberate in the
// specification.
//
// F_c is taken at N.V, not at V.H. Every other Fresnel in this file uses the
// half vector; this one does not, because the layering operator above is a
// plain lerp and only a view-dependent (not light-dependent) weight keeps it
// energy conserving as the light direction sweeps the hemisphere.
//
// The weight multiplies the WHOLE base, emission included. A coat is over the
// emitter, not under it, so an emissive surface under a coat is dimmed by
// exactly what the coat reflects away.
//
// The lobe itself is colourless -- F is the weight above, not something inside
// the lobe -- so this returns D * Vis and the caller supplies the rest. In the
// thermal bands the 0.04 is replaced by a measured curve, since a dielectric
// coat's visible reflectance is fiction at 10 microns.
// ============================================================================

float ClearcoatFresnel(float ccNdotV) {
    const float f = pow(saturate(1.0 - ccNdotV), 5.0);
    return 0.04 + 0.96 * f;
}

float ClearcoatBRDF(float ccNdotH, float ccNdotV, float ccNdotL, float ccRoughness) {
    const float r = max(ccRoughness, 0.045);
    const float alpha = r * r;
    const float D = DistributionGGX(ccNdotH, alpha);
    const float Vis = VisibilitySmithGGXCorrelated(max(ccNdotV, EPSILON),
                                                   max(ccNdotL, EPSILON), r);
    return D * Vis;
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
        return SampleComplexRefractiveIndex(complexRefractiveIndices, material.complexRefractiveIndexIndex, wavelength_nm).x;
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

// ============================================================================
// Sheen (KHR_materials_sheen): the Charlie microfibre lobe
// ============================================================================
// Velvet, felt and brushed cloth do not look like a rough dielectric because
// the scattering happens off fibres standing away from the surface rather than
// off facets lying in it. The resulting lobe peaks at grazing angles instead of
// around the mirror direction -- the bright rim on a velvet cushion -- which is
// not a shape any roughness setting on GGX can produce.
//
// D is the Charlie distribution and V is the Estevez-Kulla shadowing fit, which
// is what the glTF specification's reference implementation pairs it with. The
// simpler Ashikhmin visibility the specification also permits was measured
// against numerical integration here and is not energy conserving: its
// directional albedo reaches 1.74 at grazing incidence for roughness 0.1, which
// this renderer's furnace gate would report as a surface emitting light.
//
// Sheen layers on top of the base BRDF and the base pays for it through
// SheenAlbedoScaling. The caller does the layering, so that the site which
// knows whether the band has a diffuse/specular split at all is the site that
// decides how the two combine.
// ============================================================================

// The glTF reference implementation clamps sheen roughness up off zero for the
// same reason MIN_ROUGHNESS exists above: alpha = 0 is a Dirac delta.
static const float MIN_SHEEN_ROUGHNESS = 0.07;

/// Charlie distribution (Estevez & Kulla 2017), normalised over the hemisphere.
float SheenD_Charlie(float NdotH, float sheenRoughness) {
    const float alphaG = max(sheenRoughness * sheenRoughness,
                             MIN_SHEEN_ROUGHNESS * MIN_SHEEN_ROUGHNESS);
    const float invAlpha = 1.0 / alphaG;
    const float cos2h = NdotH * NdotH;
    const float sin2h = max(1.0 - cos2h, 1e-7);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}

/// Estevez-Kulla shadowing helper: their analytic fit to the sheen lambda term.
float SheenLambdaHelper(float x, float alphaG) {
    const float oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
    const float a = lerp(21.5473, 25.3245, oneMinusAlphaSq);
    const float b = lerp(3.82987, 3.32435, oneMinusAlphaSq);
    const float c = lerp(0.19823, 0.16801, oneMinusAlphaSq);
    const float d = lerp(-1.97760, -1.27393, oneMinusAlphaSq);
    const float e = lerp(-4.32054, -4.85967, oneMinusAlphaSq);
    return a / (1.0 + b * pow(max(x, 1e-7), c)) + d * x + e;
}

float SheenLambda(float cosTheta, float alphaG) {
    // The fit is stated for cos < 0.5 and mirrored above it; evaluating the
    // raw form past 0.5 diverges.
    const float x = abs(cosTheta);
    if (x < 0.5) {
        return exp(SheenLambdaHelper(x, alphaG));
    }
    return exp(2.0 * SheenLambdaHelper(0.5, alphaG) - SheenLambdaHelper(1.0 - x, alphaG));
}

/// Sheen visibility: G / (4 NdotV NdotL), already carrying the 1/4 and the
/// cosine denominators the way VisibilitySmithGGXCorrelated does.
float SheenV_Charlie(float NdotV, float NdotL, float sheenRoughness) {
    const float alphaG = max(sheenRoughness * sheenRoughness,
                             MIN_SHEEN_ROUGHNESS * MIN_SHEEN_ROUGHNESS);
    const float lambdaV = SheenLambda(NdotV, alphaG);
    const float lambdaL = SheenLambda(NdotL, alphaG);
    return clamp(1.0 / ((1.0 + lambdaV + lambdaL) * (4.0 * NdotV * NdotL)), 0.0, 1.0);
}

// ----------------------------------------------------------------------------
// Sheen directional albedo
// ----------------------------------------------------------------------------
//   E(mu_v, r) = integral over the hemisphere of D * V * mu_l dw_l
//
// There is no usable closed form and no usable polynomial fit: the surface has
// a ridge along low roughness at grazing incidence that a degree-6 polynomial
// misses by 0.18. Everyone who ships this ships a table, and this is the table,
// integrated offline at 600x600 quadrature.
//
// Both axes are warped by a square so the nodes crowd where the ridge is:
//   mu = (i/15)^2                     i = 0..15
//   r  = 0.07 + 0.93 * (j/15)^2       j = 0..15
// Worst bilinear error against the reference is 0.036, and that is at
// mu = 0.001 -- a surface seen exactly edge-on, whose projected area is zero.
// Away from the last row it is under 0.005.
//
// THE CLAMP IS BAKED IN, not applied at the call sites. Charlie x Estevez-Kulla
// integrates to 1.63 as mu_v approaches zero, which is a surface returning more
// light than reached it. Clamping here makes "E <= 1" a property of the data,
// so every consumer -- the albedo scaling, the infrared carve-out, the bounce
// throughput -- inherits it without having to remember to ask for it.
static const int SHEEN_E_DIM = 16;
static const float SHEEN_E_TABLE[256] = {
    1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.91543, 0.78780, 0.69201, 0.62572, 0.58558, 0.56855, 0.57188, 0.59194, 0.62177, 0.64739, 0.64483,
    1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.91403, 0.81368, 0.74378, 0.70179, 0.68456, 0.68844, 0.70837, 0.73590, 0.75692, 0.74993,
    1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.98863, 0.87956, 0.79424, 0.73482, 0.70039, 0.68876, 0.69671, 0.71924, 0.74784, 0.76880, 0.76186,
    1.00000, 1.00000, 1.00000, 0.99803, 0.93358, 0.85528, 0.78041, 0.71952, 0.67724, 0.65476, 0.65138, 0.66486, 0.69082, 0.72129, 0.74334, 0.73796,
    0.64751, 0.66550, 0.70145, 0.72310, 0.71628, 0.68795, 0.65202, 0.61975, 0.59789, 0.58974, 0.59627, 0.61645, 0.64674, 0.67990, 0.70391, 0.70097,
    0.33029, 0.35220, 0.40523, 0.46112, 0.49833, 0.51325, 0.51366, 0.50942, 0.50820, 0.51484, 0.53178, 0.55915, 0.59428, 0.63069, 0.65724, 0.65739,
    0.12880, 0.14562, 0.19215, 0.25397, 0.31165, 0.35401, 0.38137, 0.39990, 0.41639, 0.43617, 0.46257, 0.49664, 0.53645, 0.57622, 0.60561, 0.60926,
    0.03811, 0.04686, 0.07501, 0.12174, 0.17682, 0.22779, 0.26897, 0.30167, 0.33030, 0.35948, 0.39281, 0.43187, 0.47527, 0.51782, 0.54992, 0.55725,
    0.00801, 0.01108, 0.02314, 0.04952, 0.08970, 0.13569, 0.17963, 0.21874, 0.25436, 0.28946, 0.32689, 0.36833, 0.41316, 0.45707, 0.49112, 0.50201,
    0.00106, 0.00173, 0.00522, 0.01625, 0.03942, 0.07331, 0.11204, 0.15096, 0.18883, 0.22667, 0.26623, 0.30863, 0.35324, 0.39655, 0.43117, 0.44513,
    0.00007, 0.00015, 0.00077, 0.00400, 0.01433, 0.03487, 0.06397, 0.09777, 0.13372, 0.17123, 0.21081, 0.25287, 0.29653, 0.33861, 0.37280, 0.38912,
    0.00000, 0.00001, 0.00006, 0.00066, 0.00401, 0.01396, 0.03247, 0.05824, 0.08906, 0.12351, 0.16109, 0.20146, 0.24340, 0.28390, 0.31753, 0.33593,
    0.00000, 0.00000, 0.00000, 0.00006, 0.00077, 0.00438, 0.01404, 0.03114, 0.05506, 0.08458, 0.11871, 0.15647, 0.19626, 0.23510, 0.26821, 0.28857,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00008, 0.00092, 0.00466, 0.01392, 0.03006, 0.05306, 0.08219, 0.11623, 0.15327, 0.19026, 0.22285, 0.24507,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00009, 0.00091, 0.00436, 0.01283, 0.02802, 0.05043, 0.07933, 0.11278, 0.14761, 0.17964, 0.20361,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00002, 0.00036, 0.00249, 0.00922, 0.02321, 0.04532, 0.07418, 0.10657, 0.13810, 0.16377,
};

/// Fraction of incident light the sheen lobe alone returns, for a view at
/// NdotV. Always in [0, 1]; exactly 0 is not guaranteed, so callers gate on the
/// sheen colour rather than on this.
float SheenAlbedo(float NdotV, float sheenRoughness) {
    const float r = clamp(sheenRoughness, MIN_SHEEN_ROUGHNESS, 1.0);

    // Invert the node warps. Both are squares, so the inverse is a sqrt.
    const float fi = sqrt(saturate(NdotV)) * float(SHEEN_E_DIM - 1);
    const float fj = sqrt((r - MIN_SHEEN_ROUGHNESS) / (1.0 - MIN_SHEEN_ROUGHNESS))
                     * float(SHEEN_E_DIM - 1);

    const int i0 = clamp(int(floor(fi)), 0, SHEEN_E_DIM - 2);
    const int j0 = clamp(int(floor(fj)), 0, SHEEN_E_DIM - 2);
    const float a = saturate(fi - float(i0));
    const float b = saturate(fj - float(j0));

    const float e00 = SHEEN_E_TABLE[i0 * SHEEN_E_DIM + j0];
    const float e10 = SHEEN_E_TABLE[(i0 + 1) * SHEEN_E_DIM + j0];
    const float e01 = SHEEN_E_TABLE[i0 * SHEEN_E_DIM + j0 + 1];
    const float e11 = SHEEN_E_TABLE[(i0 + 1) * SHEEN_E_DIM + j0 + 1];

    return lerp(lerp(e00, e10, a), lerp(e01, e11, a), b);
}

/// What the base BRDF must be multiplied by to pay for the sheen layered over
/// it -- the albedo-scaling approximation the glTF specification prescribes.
///
/// `sheenReflectance` is the largest component of the sheen colour in RGB, or
/// the scalar sheen reflectance in a spectral band. Returns exactly 1.0 when
/// that is 0, with no rounding on the path, which is what keeps a scene with no
/// sheen rendering bit-identically.
float SheenAlbedoScaling(float sheenReflectance, float NdotV, float sheenRoughness) {
    if (sheenReflectance <= 0.0) {
        return 1.0;
    }
    return max(1.0 - sheenReflectance * SheenAlbedo(NdotV, sheenRoughness), 0.0);
}

/// The sheen BRDF without its colour: D * V, to be multiplied by the sheen
/// reflectance (RGB or per-wavelength scalar) by the caller.
float SheenBRDF(float NdotH, float NdotV, float NdotL, float sheenRoughness) {
    return SheenD_Charlie(NdotH, sheenRoughness) *
           SheenV_Charlie(NdotV, NdotL, sheenRoughness);
}

float3 CookTorranceBRDF(
    float3 N,
    float3 V,
    float3 L,
    float3 albedo,
    float metallic,
    float roughness,
    int complexRefractiveIndexIndex,
    float wavelength_nm,
    float3 dielectricF0,
    float F90,
    AnisoFrame aniso
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
    float3 F0 = ComputeF0(albedo, metallic, complexRefractiveIndexIndex, wavelength_nm,
                          dielectricF0);

    // ========================================================================
    // Specular Term (Cook-Torrance microfacet BRDF) - OPTIMIZED
    // ========================================================================

    // Fresnel term: use exact conductor Fresnel when n,k data is available.
    // A measured n,k wins over the specular extension outright -- the extension
    // is an authoring control over a fitted dielectric, and there is nothing to
    // fit when the material carries a measurement.
    float3 F;
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, complexRefractiveIndexIndex, wavelength_nm);
        F = float3(FresnelConductor(VdotH, nk.x, nk.y),
                   FresnelConductor(VdotH, nk.x, nk.y),
                   FresnelConductor(VdotH, nk.x, nk.y));
    } else {
        F = FresnelSchlickF90RGB(VdotH, F0, F90);
    }

    // Normal distribution function (GGX), stretched along the material tangent
    // when KHR_materials_anisotropy is in play. The inactive branch is the
    // original expression untouched, so a scene without the extension renders
    // bit for bit as it did.
    float alpha = roughness * roughness;  // Perceptually linear roughness
    float D, Vis;
    if (aniso.active) {
        D = DistributionGGXAniso(dot(aniso.T, H), dot(aniso.B, H), NdotH,
                                 aniso.alphaT, aniso.alphaB);
        Vis = VisibilitySmithGGXCorrelatedAniso(dot(aniso.T, V), dot(aniso.B, V), NdotV,
                                                dot(aniso.T, L), dot(aniso.B, L), NdotL,
                                                aniso.alphaT, aniso.alphaB);
    } else {
        D = DistributionGGX(NdotH, alpha);
        // OPTIMIZATION: Use combined visibility term instead of G/(4*NdotV*NdotL)
        // This eliminates NdotV/NdotL divisions, improving performance and stability
        Vis = VisibilitySmithGGXCorrelated(NdotV, NdotL, roughness);
    }

    // Cook-Torrance specular BRDF: D * F * Vis
    // Note: Vis already includes the 1/(4*NdotV*NdotL) term
    float3 specular = D * F * Vis;

    // ========================================================================
    // Diffuse Term (Lambertian)
    // ========================================================================

    // Energy conservation: kD = 1 - kS (where kS = F)
    // For metals, diffuse contribution is zero (kD = 0)
    //
    // Per channel, where the glTF specular specification says
    // 1 - max_value(fresnel). That max exists so an RGB engine does not shift
    // hue when specularColor is chromatic; this renderer's authoritative paths
    // are per-wavelength scalars, where per-channel IS the physical answer and
    // a max over three primaries would be the approximation. Keeping it per
    // channel also leaves every existing render untouched.
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
    float wavelength_nm,
    float dielectricF0,
    float F90,
    AnisoFrame aniso
) {
    const float MIN_ROUGHNESS = 0.045;
    roughness = max(roughness, MIN_ROUGHNESS);

    float3 H = SafeHalfVector(V, L, N);

    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Compute F0 and Fresnel term
    float F0 = ComputeF0_Scalar(spectralAlbedo, metallic, complexRefractiveIndexIndex,
                                wavelength_nm, dielectricF0);

    float F;
    if (complexRefractiveIndexIndex >= 0 && wavelength_nm > 0.0) {
        float2 nk = SampleComplexRefractiveIndex(complexRefractiveIndices, complexRefractiveIndexIndex, wavelength_nm);
        F = FresnelConductor(VdotH, nk.x, nk.y);
    } else {
        F = FresnelSchlickF90(VdotH, F0, F90);
    }

    float alpha = roughness * roughness;
    float D, Vis;
    if (aniso.active) {
        D = DistributionGGXAniso(dot(aniso.T, H), dot(aniso.B, H), NdotH,
                                 aniso.alphaT, aniso.alphaB);
        Vis = VisibilitySmithGGXCorrelatedAniso(dot(aniso.T, V), dot(aniso.B, V), NdotV,
                                                dot(aniso.T, L), dot(aniso.B, L), NdotL,
                                                aniso.alphaT, aniso.alphaB);
    } else {
        D = DistributionGGX(NdotH, alpha);
        Vis = VisibilitySmithGGXCorrelated(NdotV, NdotL, roughness);
    }

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
// Importance Sampling for Monte Carlo Reflection
// ============================================================================
// The samplers below take their uniform numbers as arguments rather than
// drawing them, because where those numbers come from is the caller's business:
// on the first bounce they are stratified, deeper they are PCG. See
// sampling.hlsli, which owns both streams.
// ============================================================================

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
// real thing: it is what SampleGGXVNDF below actually draws from.
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
float3 SampleGGXVNDF(float3 N, float3 V, float alpha, float2 u, out float pdf) {
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
    const float r   = sqrt(u.x);
    const float phi = 2.0 * PI * u.y;
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

// The same sampler with two alphas and the material's own tangent frame.
//
// Heitz's derivation stretches the view direction by alpha before working on
// the hemisphere and unstretches the result afterwards; nothing in it assumes
// the two axes stretch equally, so the anisotropic case is the same algorithm
// with (alphaT, alphaB) in place of (alpha, alpha) -- and with the frame coming
// from the material rather than from BuildBasis, since which way the lobe
// stretches is now observable.
float3 SampleGGXVNDFAniso(float3 N, float3 T, float3 B, float3 V,
                          float alphaT, float alphaB, float2 u, out float pdf) {
    const float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));

    const float3 Vh = normalize(float3(alphaT * Ve.x, alphaB * Ve.y, max(Ve.z, 1e-6)));

    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq)
                                  : float3(1.0, 0.0, 0.0);
    const float3 T2 = cross(Vh, T1);

    const float r   = sqrt(u.x);
    const float phi = 2.0 * PI * u.y;
    const float t1 = r * cos(phi);
    float       t2 = r * sin(phi);
    const float s  = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(1.0 - t1 * t1, 0.0)) + s * t2;

    const float3 Nh = t1 * T1 + t2 * T2 +
                      sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0)) * Vh;
    const float3 Ne = normalize(float3(alphaT * Nh.x, alphaB * Nh.y, max(Nh.z, 0.0)));

    const float3 H  = normalize(T * Ne.x + B * Ne.y + N * Ne.z);
    const float3 wi = reflect(-V, H);

    const float NdotV = max(dot(N, V), 1e-4);
    pdf = SmithG1_GGXAniso(dot(T, V), dot(B, V), NdotV, alphaT, alphaB) *
          DistributionGGXAniso(dot(T, H), dot(B, H), max(dot(N, H), 0.0), alphaT, alphaB) /
          (4.0 * NdotV);
    return wi;
}

// Cosine-weighted hemisphere sample — for rough/Lambertian surfaces
float3 CosineSampleHemisphere(float3 N, float2 u, out float pdf) {
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float3 wi_local = float3(r*cos(phi), r*sin(phi), sqrt(max(1.0-u.x, 0.0)));
    float3 T, B;
    BuildBasis(N, T, B);
    float3 wi = normalize(T*wi_local.x + B*wi_local.y + N*wi_local.z);
    pdf = max(dot(N, wi), 0.0) / PI;
    return wi;
}
// - Heitz et al., "Multiple-scattering microfacet BSDFs with the Smith model", SIGGRAPH 2016
// ============================================================================

#endif // QUANTILOOM_PBR_HLSLI
