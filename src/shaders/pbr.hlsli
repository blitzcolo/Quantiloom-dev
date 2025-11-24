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

static const float PI = 3.14159265358979323846;
static const float EPSILON = 1e-6;

// ============================================================================
// Fresnel Term (Schlick Approximation)
// ============================================================================
// Fresnel reflectance for dielectrics and metals
// F0: Reflectance at normal incidence (specular color)
//     - For dielectrics: ~0.04 (grayscale)
//     - For metals: baseColor (colored)
// cosTheta: dot(v, h) where v=view direction, h=half vector
// ============================================================================

float3 FresnelSchlick(float3 F0, float cosTheta) {
    // Clamp to avoid artifacts when cosTheta < 0
    cosTheta = saturate(cosTheta);

    // Schlick approximation: F0 + (1 - F0) * (1 - cosTheta)^5
    float oneMinusCos = 1.0 - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;

    return F0 + (1.0 - F0) * oneMinusCos5;
}

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
    float roughness
) {
    // Compute half vector (with safety check for opposite V and L)
    // FIXED: Use SafeHalfVector to prevent NaN when V + L is near-zero
    float3 H = SafeHalfVector(V, L, N);

    // Compute dot products (clamped to avoid negative values)
    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Compute F0 (reflectance at normal incidence)
    // For dielectrics: F0 = 0.04 (approximate for common materials)
    // For metals: F0 = albedo (colored reflection)
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // ========================================================================
    // Specular Term (Cook-Torrance microfacet BRDF)
    // ========================================================================

    // Fresnel term
    float3 F = FresnelSchlick(F0, VdotH);

    // Normal distribution function (GGX)
    float alpha = roughness * roughness;  // Perceptually linear roughness
    float D = DistributionGGX(NdotH, alpha);

    // Geometric shadowing term (Smith GGX)
    float G = GeometrySmith(NdotV, NdotL, roughness);

    // Cook-Torrance specular BRDF: (D * F * G) / (4 * NdotV * NdotL)
    float3 numerator = D * F * G;
    float denominator = 4.0 * NdotV * NdotL;
    float3 specular = numerator / max(denominator, EPSILON);

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

    return diffuse + specular;
}

// ============================================================================
// Simplified PBR for Spectral Mode (M1 Compatibility)
// ============================================================================
// Evaluates PBR using scalar spectralAlbedo instead of RGB
// Useful for single-wavelength rendering
// ============================================================================

float CookTorranceBRDF_Spectral(
    float3 N,
    float3 V,
    float3 L,
    float spectralAlbedo,
    float metallic,
    float roughness
) {
    // Use grayscale albedo for spectral mode
    float3 albedo = float3(spectralAlbedo, spectralAlbedo, spectralAlbedo);

    float3 brdf = CookTorranceBRDF(N, V, L, albedo, metallic, roughness);

    // Return average of RGB channels (they should be identical for grayscale input)
    return (brdf.r + brdf.g + brdf.b) / 3.0;
}

#endif // QUANTILOOM_PBR_HLSLI
