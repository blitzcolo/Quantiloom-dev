// ============================================================================
// Quantiloom - Volumetric Rendering Library
// ============================================================================
// Medium properties, Beer-Lambert transmittance and the Henyey-Greenstein
// phase function for the single-scattering block in closesthit.rchit.
//
// PHYSICS:
//   σ_t(x) = σ_a(x) + σ_s(x)  (extinction = absorption + scattering)
//   T(d) = exp(-∫σ_t(x)dx)    (Beer-Lambert transmittance)
//
// Stochastic tracking (Woodcock/delta-tracking) is not implemented here:
// in a homogeneous medium the null-collision loop accepts every candidate
// (sigma_t equals the majorant everywhere), which reduces it to the
// analytic Beer-Lambert this file already provides. A heterogeneous
// implementation would need a spatial majorant to be worth having.
//
// REFERENCES:
//   - Novák et al., "Monte Carlo Methods for Volumetric Light Transport" (2018)
//   - PBRT-v4 §14: Light Transport I: Surface Reflection
//   - Fong et al., "Production Volume Rendering" (SIGGRAPH 2017)
// ============================================================================

#ifndef QUANTILOOM_VOLUMETRIC_HLSLI
#define QUANTILOOM_VOLUMETRIC_HLSLI

#include "common.hlsli"

// ============================================================================
// Constants
// ============================================================================

static const float VOLUME_EPSILON = 1e-7;   // Minimum extinction threshold

// ============================================================================
// Medium Properties Structure
// ============================================================================
// Describes the optical properties of a participating medium at a point.
// Can be constant (homogeneous) or spatially varying (heterogeneous).
// ============================================================================

struct MediumProperties {
    float3 sigma_s;          // Scattering coefficient (m⁻¹) - probability of scattering per unit distance
    float3 sigma_a;          // Absorption coefficient (m⁻¹) - probability of absorption per unit distance
    float3 sigma_t;          // Extinction coefficient = σ_s + σ_a (computed)
    float  g;                // Henyey-Greenstein asymmetry parameter [-1, 1]
                             //   g > 0: forward scattering (smoke, fog)
                             //   g = 0: isotropic scattering
                             //   g < 0: backward scattering (clouds)
    float3 albedo;           // Single-scattering albedo ω = σ_s / σ_t [0, 1]
};

// ============================================================================
// Create Medium Properties from Material Data
// ============================================================================
// Converts MaterialData volume properties to MediumProperties structure.
//
// Input:
//   material: MaterialData with volume properties
//
// Returns:
//   MediumProperties for use in delta-tracking
// ============================================================================

MediumProperties CreateMediumFromMaterial(MaterialData material) {
    MediumProperties medium;

    // Scale coefficients by density
    float density = max(material.volumeDensity, 0.0);
    medium.sigma_s = float3(material.scatteringCoeff, material.scatteringCoeff, material.scatteringCoeff) * density;
    medium.sigma_a = float3(material.absorptionCoeff, material.absorptionCoeff, material.absorptionCoeff) * density;
    medium.sigma_t = medium.sigma_s + medium.sigma_a;
    medium.g = clamp(material.phaseG, -0.999, 0.999);

    // Compute single-scattering albedo (avoid division by zero)
    float sigma_t_max = max(max(medium.sigma_t.x, medium.sigma_t.y), medium.sigma_t.z);
    if (sigma_t_max > VOLUME_EPSILON) {
        medium.albedo = medium.sigma_s / medium.sigma_t;
    } else {
        medium.albedo = float3(0.0, 0.0, 0.0);
    }

    return medium;
}

// ============================================================================
// Create Medium from Transmission Material (for internal volume)
// ============================================================================
// Creates medium properties for the interior of a transmissive object
// using Beer-Lambert attenuation parameters.
//
// Input:
//   attenuationColor: Color after traveling attenuationDistance
//   attenuationDistance: Reference distance for attenuation
//
// Returns:
//   MediumProperties for Beer-Lambert absorption (no scattering)
// ============================================================================

MediumProperties CreateMediumFromAttenuation(float3 attenuationColor, float attenuationDistance) {
    MediumProperties medium;

    // Initialize to no scattering/absorption
    medium.sigma_s = float3(0.0, 0.0, 0.0);
    medium.sigma_a = float3(0.0, 0.0, 0.0);
    medium.sigma_t = float3(0.0, 0.0, 0.0);
    medium.g = 0.0;
    medium.albedo = float3(0.0, 0.0, 0.0);

    if (attenuationDistance > 0.0) {
        // Compute absorption coefficient from Beer-Lambert
        // attenuationColor = exp(-σ_a × attenuationDistance)
        // σ_a = -ln(attenuationColor) / attenuationDistance
        float3 safeColor = max(attenuationColor, float3(0.001, 0.001, 0.001));
        medium.sigma_a = -log(safeColor) / attenuationDistance;
        medium.sigma_t = medium.sigma_a;  // No scattering, only absorption
    }

    return medium;
}

// ============================================================================
// Henyey-Greenstein Phase Function
// ============================================================================
// Probability density for scattering direction given asymmetry parameter g.
//
// PHYSICS:
//   p(θ; g) = (1 - g²) / [4π × (1 + g² - 2g×cos(θ))^(3/2)]
//
// Integrates to 1 over the sphere.
//
// Input:
//   cosTheta: cos(angle between incident and scattered directions)
//   g: asymmetry parameter [-1, 1]
//
// Returns:
//   Phase function value (probability density per steradian)
// ============================================================================

float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    float denom32 = denom * sqrt(denom);  // denom^(3/2)

    // Avoid division by zero when g = 1 and cosTheta = 1
    if (denom32 < 1e-6) {
        return 1.0 / (4.0 * 3.14159265359);
    }

    return (1.0 - g2) / (4.0 * 3.14159265359 * denom32);
}

// ============================================================================
// Sample Henyey-Greenstein Phase Function
// ============================================================================
// Importance samples a scattering direction from HG distribution.
//
// Input:
//   g: asymmetry parameter [-1, 1]
//   u1, u2: uniform random numbers in [0, 1)
//
// Returns:
//   Sampled direction in local frame (z = forward)
// ============================================================================

float3 SampleHenyeyGreenstein(float g, float u1, float u2) {
    float cosTheta;

    if (abs(g) < 1e-3) {
        // Isotropic: uniform sphere sampling
        cosTheta = 1.0 - 2.0 * u1;
    } else {
        // HG importance sampling
        float g2 = g * g;
        float sqTerm = (1.0 - g2) / (1.0 - g + 2.0 * g * u1);
        cosTheta = (1.0 + g2 - sqTerm * sqTerm) / (2.0 * g);
    }

    // Clamp to valid range
    cosTheta = clamp(cosTheta, -1.0, 1.0);

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * 3.14159265359 * u2;

    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// ============================================================================
// Transform Local Direction to World Frame
// ============================================================================
// Transforms a direction from local frame (z = forward) to world frame.
//
// Input:
//   localDir: direction in local frame
//   forward: forward direction in world frame (normalized)
//
// Returns:
//   Direction in world frame (normalized)
// ============================================================================

float3 LocalToWorld(float3 localDir, float3 forward) {
    // Build orthonormal basis with forward as z-axis
    float3 up = abs(forward.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(up, forward));
    up = cross(forward, right);

    // Transform
    return right * localDir.x + up * localDir.y + forward * localDir.z;
}

// ============================================================================
// Simple Beer-Lambert Transmittance (Analytic)
// ============================================================================
// Direct computation for homogeneous media without stochastic sampling.
// Use when exact solution is preferred over unbiased estimation.
//
// Input:
//   distance: path length through medium
//   medium: medium properties
//
// Returns:
//   Transmittance [0, 1] per channel
// ============================================================================

float3 BeerLambertTransmittance(float distance, MediumProperties medium) {
    return exp(-medium.sigma_t * distance);
}

#endif // QUANTILOOM_VOLUMETRIC_HLSLI
