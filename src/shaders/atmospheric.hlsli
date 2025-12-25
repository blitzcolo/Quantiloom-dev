// ============================================================================
// Quantiloom - Atmospheric Scattering Library
// ============================================================================
// Complete physical model for Delta-Tracking volumetric atmospheric rendering
// Implements Rayleigh + Mie scattering with wavelength dependence
//
// REFERENCES:
// - Bruneton & Neyret, "Precomputed Atmospheric Scattering" (2008)
// - PBRT-v4 §11.4: Participating Media
// - Novák et al., "Monte Carlo Methods for Volumetric Light Transport Simulation" (2018)
// ============================================================================

#ifndef QUANTILOOM_ATMOSPHERIC_HLSLI
#define QUANTILOOM_ATMOSPHERIC_HLSLI

#include "common.hlsli"

// ============================================================================
// Physical Constants
// ============================================================================
// IMPORTANT: PI is defined in pbr.hlsli
// Files using atmospheric.hlsli must include pbr.hlsli BEFORE atmospheric.hlsli

static const float INV_PI = 0.31830988618379067154;
static const float INV_4PI = 0.07957747154594766788;

// Reference wavelength for normalization (green, human eye peak sensitivity)
static const float REFERENCE_WAVELENGTH_NM = 550.0;

// ============================================================================
// Wavelength-Dependent Scattering Coefficients
// ============================================================================
// Physical scattering follows power laws:
//   - Rayleigh (molecular): β(λ) ∝ λ⁻⁴ (strong wavelength dependence)
//   - Mie (aerosol): β(λ) ∝ λ⁻ᵅ where α ≈ 0.84 (weaker dependence)
//
// TWO VERSIONS PROVIDED:
//   1. Scalar version: For single-wavelength spectral rendering
//   2. RGB version: For RGB mode with three separate wavelength calculations
// ============================================================================

/// Compute Rayleigh scattering coefficient at single wavelength (SCALAR)
/// β_r(λ) = β_r(550nm) × (550/λ)⁴
/// @param wavelength_nm Query wavelength (nanometers)
/// @param beta_r_550nm Reference Rayleigh coefficient at 550nm (m⁻¹)
/// @return Rayleigh scattering coefficient at wavelength (m⁻¹)
float RayleighScatteringCoeff_Scalar(float wavelength_nm, float beta_r_550nm) {
    float ratio = REFERENCE_WAVELENGTH_NM / wavelength_nm;
    float power4 = ratio * ratio * ratio * ratio;
    return beta_r_550nm * power4;
}

/// Compute Rayleigh scattering coefficient at three RGB wavelengths
/// Evaluates β_r(λ) separately for R, G, B wavelengths
/// @param wavelengths_rgb Three wavelengths for R, G, B channels (nanometers)
/// @param beta_r_550nm Reference Rayleigh coefficient at 550nm (m⁻¹)
/// @return Rayleigh scattering coefficients for RGB (m⁻¹)
float3 RayleighScatteringCoeff_RGB(float3 wavelengths_rgb, float beta_r_550nm) {
    float3 ratio = float3(REFERENCE_WAVELENGTH_NM, REFERENCE_WAVELENGTH_NM, REFERENCE_WAVELENGTH_NM) / wavelengths_rgb;
    float3 power4 = ratio * ratio * ratio * ratio;
    return float3(beta_r_550nm, beta_r_550nm, beta_r_550nm) * power4;
}

/// Compute Mie scattering coefficient at single wavelength (SCALAR)
/// β_m(λ) = β_m(550nm) × (550/λ)ᵅ
/// @param wavelength_nm Query wavelength (nanometers)
/// @param beta_m_550nm Reference Mie coefficient at 550nm (m⁻¹)
/// @param alpha Angstrom exponent (typically ~0.84)
/// @return Mie scattering coefficient at wavelength (m⁻¹)
float MieScatteringCoeff_Scalar(float wavelength_nm, float beta_m_550nm, float alpha) {
    float ratio = REFERENCE_WAVELENGTH_NM / wavelength_nm;
    float power_alpha = pow(ratio, alpha);
    return beta_m_550nm * power_alpha;
}

/// Compute Mie scattering coefficient at three RGB wavelengths
/// Evaluates β_m(λ) separately for R, G, B wavelengths
/// @param wavelengths_rgb Three wavelengths for R, G, B channels (nanometers)
/// @param beta_m_550nm Reference Mie coefficient at 550nm (m⁻¹)
/// @param alpha Angstrom exponent (typically ~0.84)
/// @return Mie scattering coefficients for RGB (m⁻¹)
float3 MieScatteringCoeff_RGB(float3 wavelengths_rgb, float beta_m_550nm, float alpha) {
    float3 ratio = float3(REFERENCE_WAVELENGTH_NM, REFERENCE_WAVELENGTH_NM, REFERENCE_WAVELENGTH_NM) / wavelengths_rgb;
    float3 power_alpha = pow(ratio, float3(alpha, alpha, alpha));
    return float3(beta_m_550nm, beta_m_550nm, beta_m_550nm) * power_alpha;
}

// ============================================================================
// Atmospheric Density Models
// ============================================================================

/// Compute atmospheric density at altitude using exponential decay
/// ρ(h) = exp(-h / H)
/// @param altitude_m Altitude above sea level (meters)
/// @param scale_height Scale height H (meters)
/// @return Normalized density [0, 1]
float AtmosphericDensity(float altitude_m, float scale_height) {
    return exp(-max(altitude_m, 0.0) / scale_height);
}

/// Compute altitude from ray position (assuming spherical planet)
/// @param pos World position (meters)
/// @param planet_center Planet center position (meters)
/// @param planet_radius Planet radius (meters)
/// @return Altitude above surface (meters)
float GetAltitude(float3 pos, float3 planet_center, float planet_radius) {
    float dist_from_center = length(pos - planet_center);
    return dist_from_center - planet_radius;
}

// ============================================================================
// Phase Functions
// ============================================================================

/// Rayleigh phase function (isotropic with weak angular dependence)
/// p_r(θ) = 3/(16π) × (1 + cos²θ)
/// @param cosTheta cos(angle between incident and scattered direction)
/// @return Phase function value (probability density)
float RayleighPhaseFunction(float cosTheta) {
    float cos2 = cosTheta * cosTheta;
    return (3.0 / (16.0 * PI)) * (1.0 + cos2);
}

/// Henyey-Greenstein phase function (Mie scattering approximation)
/// p_m(θ; g) = (1-g²) / [4π(1 + g² - 2g·cosθ)^(3/2)]
/// @param cosTheta cos(angle between incident and scattered direction)
/// @param g Asymmetry parameter (g ∈ [-1, 1], 0 = isotropic, >0 = forward)
/// @return Phase function value (probability density)
float HenyeyGreensteinPhaseFunction(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

// ============================================================================
// Random Number Generator (PCG Hash)
// ============================================================================
// Must be defined BEFORE DeltaTracking to avoid forward reference errors

/// Generate random float in [0, 1) using PCG hash
/// @param state Random state (modified in-place)
/// @return Random float in [0, 1)
float RandomFloat01(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) / 4294967296.0;
}

// ============================================================================
// Ray-Sphere Intersection
// ============================================================================

/// Compute ray-sphere intersection
/// @param ray_origin Ray origin (meters)
/// @param ray_dir Ray direction (normalized)
/// @param sphere_center Sphere center (meters)
/// @param sphere_radius Sphere radius (meters)
/// @param[out] t_near Near intersection distance (meters)
/// @param[out] t_far Far intersection distance (meters)
/// @return true if ray intersects sphere, false otherwise
bool RaySphereIntersection(float3 ray_origin, float3 ray_dir,
                            float3 sphere_center, float sphere_radius,
                            out float t_near, out float t_far) {
    float3 oc = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        t_near = 0.0;
        t_far = 0.0;
        return false;
    }

    float sqrt_disc = sqrt(discriminant);
    float inv_2a = 0.5 / a;
    t_near = (-b - sqrt_disc) * inv_2a;
    t_far = (-b + sqrt_disc) * inv_2a;

    return true;
}

// ============================================================================
// Delta-Tracking Core Algorithm
// ============================================================================

/// Perform Delta-Tracking through participating medium
/// Samples scattering events using Woodcock tracking (null-collision method)
///
/// ALGORITHM:
/// 1. Sample free-flight distance: t = -ln(ξ) / σ_maj
/// 2. At sampled point, compute real extinction σ_t
/// 3. Accept collision with probability σ_t / σ_maj (real event)
/// 4. Reject with probability 1 - σ_t / σ_maj (null event, continue)
/// 5. Repeat until real collision or ray exits medium
///
/// @param ray_origin Ray origin (world space, meters)
/// @param ray_dir Ray direction (normalized)
/// @param t_min Entry distance into medium (meters)
/// @param t_max Exit distance from medium (meters)
/// @param wavelength_nm Current wavelength (nanometers)
/// @param atmosphere Atmospheric parameters
/// @param planet_center Planet center position (meters)
/// @param random_state Random state for sampling (modified in-place)
/// @param[out] t_scatter Distance to scattering event (meters), or -1 if none
/// @param[out] transmittance Accumulated transmittance along path [0, 1]
/// @return true if scattering event occurred, false if ray exited medium
bool DeltaTracking(
    float3 ray_origin, float3 ray_dir,
    float t_min, float t_max,
    float wavelength_nm,
    AtmosphericParams atmosphere,
    float3 planet_center,
    inout uint random_state,
    out float t_scatter,
    out float transmittance) {

    // Compute majorant extinction coefficient (max over entire path)
    // For exponentially decaying atmosphere, majorant is at lowest altitude (t_min)
    float3 pos_entry = ray_origin + ray_dir * t_min;
    float altitude_entry = GetAltitude(pos_entry, planet_center, atmosphere.planet_radius);

    // Use SCALAR versions for single-wavelength volumetric rendering
    float beta_r = RayleighScatteringCoeff_Scalar(wavelength_nm, atmosphere.beta_rayleigh_550nm.x);
    float beta_m = MieScatteringCoeff_Scalar(wavelength_nm, atmosphere.beta_mie_550nm.x, atmosphere.mie_alpha);

    float rho_r_entry = AtmosphericDensity(altitude_entry, atmosphere.rayleigh_scale_height);
    float rho_m_entry = AtmosphericDensity(altitude_entry, atmosphere.mie_scale_height);

    // Total extinction = scattering (assuming negligible absorption)
    float sigma_t_entry = (beta_r * rho_r_entry) + (beta_m * rho_m_entry);

    // Use as majorant (conservative bound)
    float sigma_maj = sigma_t_entry;

    // Safety check: if majorant is near zero, atmosphere is negligible
    // Threshold: 1e-7 m⁻¹ means mean free path > 10,000 km
    // At this density, scattering probability < 0.01% per km
    // This prevents numerical overflow when computing free-flight distance
    if (sigma_maj < 1e-7) {
        t_scatter = -1.0;
        transmittance = 1.0;
        return false;
    }

    // Delta-Tracking loop
    float t = t_min;
    transmittance = 1.0;

    for (uint step = 0; step < atmosphere.max_steps; ++step) {
        // Sample free-flight distance: t_free = -ln(ξ) / σ_maj
        // Note: log(xi) is negative for xi ∈ (0,1), so -log(xi) is positive
        float xi = RandomFloat01(random_state);
        float free_flight = -log(max(xi, 1e-6)) / sigma_maj;

        // Clamp free-flight distance to prevent numerical overflow
        // Max reasonable distance is 2x the path length through atmosphere
        float max_free_flight = (t_max - t_min) * 2.0;
        free_flight = min(free_flight, max_free_flight);

        float t_sample = t + free_flight;

        // Check if ray exited medium
        if (t_sample > t_max) {
            t_scatter = -1.0;
            return false;
        }

        // Compute real extinction at sampled point
        float3 pos_sample = ray_origin + ray_dir * t_sample;
        float altitude_sample = GetAltitude(pos_sample, planet_center, atmosphere.planet_radius);

        float rho_r = AtmosphericDensity(altitude_sample, atmosphere.rayleigh_scale_height);
        float rho_m = AtmosphericDensity(altitude_sample, atmosphere.mie_scale_height);

        float sigma_t_sample = (beta_r * rho_r) + (beta_m * rho_m);
        float sigma_t = sigma_t_sample;

        // Collision probability
        float p_collision = sigma_t / sigma_maj;

        // Accept or reject collision
        float xi2 = RandomFloat01(random_state);
        if (xi2 < p_collision) {
            // Real collision: scattering event
            t_scatter = t_sample;
            return true;
        }

        // Null collision: continue tracking
        t = t_sample;
    }

    // Max steps exceeded: treat as no collision
    // Use path-integrated transmittance approximation instead of single-point estimate
    //
    // IMPROVED APPROXIMATION:
    // For exponentially decaying atmosphere, using only the entry point extinction
    // overestimates absorption. We use trapezoidal integration with entry/exit points
    // and midpoint for better accuracy:
    //   τ ≈ (σ_entry + 4×σ_mid + σ_exit) / 6 × path_length (Simpson's rule)
    //
    // This reduces error from O(path_length) to O(path_length³) for smooth profiles.
    t_scatter = -1.0;

    float3 pos_exit = ray_origin + ray_dir * t_max;
    float altitude_exit = GetAltitude(pos_exit, planet_center, atmosphere.planet_radius);

    float3 pos_mid = ray_origin + ray_dir * ((t_min + t_max) * 0.5);
    float altitude_mid = GetAltitude(pos_mid, planet_center, atmosphere.planet_radius);

    // Compute extinction at exit and midpoint
    float rho_r_exit = AtmosphericDensity(altitude_exit, atmosphere.rayleigh_scale_height);
    float rho_m_exit = AtmosphericDensity(altitude_exit, atmosphere.mie_scale_height);
    float sigma_exit = (beta_r * rho_r_exit) + (beta_m * rho_m_exit);

    float rho_r_mid = AtmosphericDensity(altitude_mid, atmosphere.rayleigh_scale_height);
    float rho_m_mid = AtmosphericDensity(altitude_mid, atmosphere.mie_scale_height);
    float sigma_mid = (beta_r * rho_r_mid) + (beta_m * rho_m_mid);

    // Simpson's rule: (f(a) + 4f(m) + f(b)) / 6 × (b - a)
    float sigma_avg = (sigma_t_entry + 4.0 * sigma_mid + sigma_exit) / 6.0;
    float path_length = t_max - t_min;
    float optical_depth = sigma_avg * path_length;

    transmittance = exp(-optical_depth);
    return false;
}

// ============================================================================
// Single Scattering Computation
// ============================================================================

/// Compute single scattering radiance at a point in the atmosphere
/// @param pos Scattering point (world space, meters)
/// @param view_dir View direction (FROM point TO camera, normalized)
/// @param sun_dir Sun direction (FROM point TO sun, normalized)
/// @param wavelength_nm Current wavelength (nanometers)
/// @param atmosphere Atmospheric parameters
/// @param planet_center Planet center position (meters)
/// @param sun_radiance Sun spectral radiance (W·sr⁻¹·m⁻²·nm⁻¹)
/// @return Scattered radiance (W·sr⁻¹·m⁻²·nm⁻¹)
float3 SingleScattering(
    float3 pos, float3 view_dir, float3 sun_dir,
    float wavelength_nm,
    AtmosphericParams atmosphere,
    float3 planet_center,
    float sun_radiance) {

    float altitude = GetAltitude(pos, planet_center, atmosphere.planet_radius);

    // Compute scattering coefficients at this altitude
    // Use SCALAR versions for single-wavelength computation
    float beta_r = RayleighScatteringCoeff_Scalar(wavelength_nm, atmosphere.beta_rayleigh_550nm.x);
    float beta_m = MieScatteringCoeff_Scalar(wavelength_nm, atmosphere.beta_mie_550nm.x, atmosphere.mie_alpha);

    float rho_r = AtmosphericDensity(altitude, atmosphere.rayleigh_scale_height);
    float rho_m = AtmosphericDensity(altitude, atmosphere.mie_scale_height);

    float sigma_s_r = beta_r * rho_r;
    float sigma_s_m = beta_m * rho_m;

    // Phase functions
    float cosTheta = dot(-view_dir, sun_dir);  // Scattering angle
    float phase_r = RayleighPhaseFunction(cosTheta);
    float phase_m = HenyeyGreensteinPhaseFunction(cosTheta, atmosphere.mie_g);

    // Compute transmittance from scattering point to sun
    // (Simplified: assume exponential decay, no shadowing)
    // For full accuracy, would need another ray trace to sun
    float t_sun_near, t_sun_far;
    bool hits_atmo = RaySphereIntersection(
        pos, sun_dir, planet_center,
        atmosphere.planet_radius + atmosphere.atmosphere_height,
        t_sun_near, t_sun_far);

    float transmittance_sun = 1.0;
    if (hits_atmo && t_sun_far > 0.0) {
        // Approximate transmittance using altitude-averaged extinction
        float avg_altitude = altitude + 0.5 * atmosphere.atmosphere_height;
        float rho_avg_r = AtmosphericDensity(avg_altitude, atmosphere.rayleigh_scale_height);
        float rho_avg_m = AtmosphericDensity(avg_altitude, atmosphere.mie_scale_height);
        float sigma_t_avg = (beta_r * rho_avg_r) + (beta_m * rho_avg_m);
        float sigma_avg = sigma_t_avg;
        transmittance_sun = exp(-sigma_avg * max(t_sun_far - max(t_sun_near, 0.0), 0.0));
    }

    // Single scattering integral (simplified: point source sun)
    // L_s = β_s × p(θ) × L_sun × T(point → sun)
    float L_scattered_r = sigma_s_r * phase_r * sun_radiance * transmittance_sun;
    float L_scattered_m = sigma_s_m * phase_m * sun_radiance * transmittance_sun;

    return float3(L_scattered_r + L_scattered_m, L_scattered_r + L_scattered_m, L_scattered_r + L_scattered_m);
}

#endif // QUANTILOOM_ATMOSPHERIC_HLSLI
