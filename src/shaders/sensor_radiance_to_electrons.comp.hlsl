/**
 * @file sensor_radiance_to_electrons.comp.hlsl
 * @brief GPU Sensor Simulation - Pass 1: Radiance → Photo-electrons
 *
 * Converts HDR radiance (W·sr⁻¹·m⁻²) to photo-electrons (e⁻) using:
 * - Quantum efficiency (QE)
 * - Pixel area (from pixel pitch)
 * - Solid angle (from f-number)
 * - Integration time
 * - Optional: Dark current, vignetting (cos^4 law)
 *
 * Physics:
 *   1. Irradiance: E = L × Ω (W·m⁻²)
 *   2. Energy: E_total = E × A × t (J)
 *   3. Photons: N_photons = E_total / E_photon
 *   4. Electrons: N_e = N_photons × QE
 *
 * @author blitzcolo
 */

// ============================================================================
// Push Constants
// ============================================================================

struct SensorRadianceParams {
    float quantumEfficiency;      // QE [0-1]
    float pixelPitch_um;          // Pixel pitch (μm)
    float focalLength_mm;         // Focal length (mm)
    float fNumber;                // Aperture (f-stop)

    float integrationTime_s;      // Integration time (s)
    float wellCapacity_e;         // Full well capacity (e⁻)
    float wavelength_nm;          // Peak wavelength (nm)
    float darkCurrent_e_s;        // Dark current (e⁻/s)

    uint enableDarkCurrent;       // Flag: add dark current
    uint enableVignetting;        // Flag: apply cos^4 vignetting
    float fov_deg;                // Horizontal FOV (degrees)
    uint isTelecentric;           // Flag: telecentric lens (no vignetting)

    uint imageWidth;              // Image width
    uint imageHeight;             // Image height

    // Band-integrated radiance is what this pass wants; a fused-band render writes
    // per-nm average radiance instead, so the host passes the band width here.
    // 1.0 for modes that already write integrated radiance.
    float radianceScale;
    uint padding;
};

[[vk::push_constant]] SensorRadianceParams params;

// ============================================================================
// Bindings
// ============================================================================

// Input: HDR radiance from ray tracer (W·sr⁻¹·m⁻²)
[[vk::binding(0, 0)]] Texture2D<float4> inputRadiance;

// Output: Photo-electrons (e⁻)
[[vk::binding(1, 0)]] RWTexture2D<float4> outputElectrons;

// ============================================================================
// Physical Constants
// ============================================================================

static const float kPlanckConstant = 6.62607015e-34;  // J·s
static const float kSpeedOfLight = 299792458.0;       // m/s
static const float PI = 3.14159265359;

// ============================================================================
// Main Compute Shader
// ============================================================================

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 coord = dispatchThreadID.xy;

    // Bounds check
    if (coord.x >= params.imageWidth || coord.y >= params.imageHeight) {
        return;
    }

    // Sample input radiance and recover band-integrated units (W·sr⁻¹·m⁻²)
    float3 radiance = inputRadiance[coord].rgb * params.radianceScale;

    // 1. Calculate pixel area (m²)
    float pixelArea_m2 = pow(params.pixelPitch_um * 1e-6, 2.0);

    // 2. Calculate solid angle subtended by lens aperture
    // Ω = π / (4 × f#²)
    float solidAngle_sr = PI / (4.0 * params.fNumber * params.fNumber);

    // 3. Apply vignetting (cos^4 law) if enabled
    float vignetteFactor = 1.0;
    if (params.enableVignetting != 0 && params.isTelecentric == 0) {
        // Calculate distance from optical axis (normalized)
        float2 center = float2(params.imageWidth, params.imageHeight) * 0.5;
        float2 offset = float2(coord) - center;
        float r = length(offset);
        float maxR = length(center);  // Distance to corner

        // Convert to angle: θ = atan(r / focal_length)
        float focalLength_m = params.focalLength_mm * 1e-3;
        float r_m = (r / maxR) * tan(radians(params.fov_deg * 0.5)) * focalLength_m;
        float theta = atan(r_m / focalLength_m);

        // cos^4 vignetting
        float cosTheta = cos(theta);
        vignetteFactor = pow(cosTheta, 4.0);
    }

    float effectiveSolidAngle = solidAngle_sr * vignetteFactor;

    // 4. Irradiance: E = L · Ω (W·m⁻²)
    float3 irradiance = radiance * effectiveSolidAngle;

    // 5. Energy collected: E_total = E · A · t (J)
    float3 energy_J = irradiance * pixelArea_m2 * params.integrationTime_s;

    // 6. Photon energy: E_photon = h·c / λ
    float wavelength_m = params.wavelength_nm * 1e-9;
    float photonEnergy_J = (kPlanckConstant * kSpeedOfLight) / wavelength_m;

    // 7. Number of photons: N_photons = E_total / E_photon
    float3 numPhotons = energy_J / photonEnergy_J;

    // 8. Photo-electrons: N_e = N_photons · QE
    float3 numElectrons = numPhotons * params.quantumEfficiency;

    // 9. Add dark current if enabled
    if (params.enableDarkCurrent != 0) {
        numElectrons += float3(params.darkCurrent_e_s * params.integrationTime_s,
                               params.darkCurrent_e_s * params.integrationTime_s,
                               params.darkCurrent_e_s * params.integrationTime_s);
    }

    // 10. Clamp to well capacity
    numElectrons = min(numElectrons, float3(params.wellCapacity_e,
                                             params.wellCapacity_e,
                                             params.wellCapacity_e));

    // Store result
    outputElectrons[coord] = float4(numElectrons, 1.0);
}
