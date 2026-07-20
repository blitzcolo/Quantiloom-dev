/**
 * @file sensor_quantize_to_radiance.comp.hlsl
 * @brief GPU Sensor Simulation - Pass 5: Quantize → Radiance
 *
 * Final sensor chain pass:
 * 1. Quantize electrons to DN (Digital Numbers): DN = electrons / gain
 * 2. Clamp to ADC bit depth: [0, 2^bitDepth - 1]
 * 3. Convert DN back to electrons: electrons = DN × gain
 * 4. Convert electrons back to radiance for display
 *
 * This simulates the full sensor readout chain while preserving
 * the radiance units needed for display.
 *
 * @author blitzcolo
 */

// ============================================================================
// Push Constants
// ============================================================================

struct SensorQuantizeParams {
    float gain;                   // Gain (e⁻/DN)
    uint bitDepth;                // ADC bit depth (12/14/16)
    float quantumEfficiency;      // QE (for reverse conversion)
    float pixelPitch_um;          // Pixel pitch (μm)

    float focalLength_mm;         // Focal length (mm)
    float fNumber;                // Aperture (f-stop)
    float integrationTime_s;      // Integration time (s)
    float wavelength_nm;          // Wavelength (nm)

    float darkCurrent_e_s;        // Dark current (e⁻/s)
    uint enableDarkCurrent;       // Flag
    uint imageWidth;              // Image width
    uint imageHeight;             // Image height
};

[[vk::push_constant]] SensorQuantizeParams params;

// ============================================================================
// Bindings
// ============================================================================

// Input: Photo-electrons (e⁻)
[[vk::binding(0, 0)]] Texture2D<float4> electronsImage;

// Output: Radiance (W·sr⁻¹·m⁻²) for display
[[vk::binding(1, 0)]] RWTexture2D<float4> outputRadiance;

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

    // Load electron count
    float3 numElectrons = electronsImage[coord].rgb;

    // 1. Quantize to DN: DN = electrons / gain
    float maxDN = float((1u << params.bitDepth) - 1u);
    float3 dnValue = floor(clamp(numElectrons / params.gain,
                                  float3(0.0, 0.0, 0.0),
                                  float3(maxDN, maxDN, maxDN)));

    // 2. Convert DN back to electrons (simulating readout)
    float3 readoutElectrons = dnValue * params.gain;

    // 3. Remove dark current contribution (for radiance conversion)
    if (params.enableDarkCurrent != 0) {
        float darkContribution = params.darkCurrent_e_s * params.integrationTime_s;
        readoutElectrons -= float3(darkContribution, darkContribution, darkContribution);
        readoutElectrons = max(readoutElectrons, float3(0.0, 0.0, 0.0));
    }

    // 4. Electrons → Photons: N_photons = N_e / QE
    float3 numPhotons = readoutElectrons / max(params.quantumEfficiency, 1e-10);

    // 5. Photons → Energy: E_total = N_photons × E_photon
    float wavelength_m = params.wavelength_nm * 1e-9;
    float photonEnergy_J = (kPlanckConstant * kSpeedOfLight) / wavelength_m;
    float3 energy_J = numPhotons * photonEnergy_J;

    // 6. Energy → Irradiance: E = E_total / (A × t)
    float pixelArea_m2 = pow(params.pixelPitch_um * 1e-6, 2.0);
    float3 irradiance = energy_J / (pixelArea_m2 * params.integrationTime_s);

    // 7. Irradiance → Radiance: L = E / Ω
    float solidAngle_sr = PI / (4.0 * params.fNumber * params.fNumber);
    float3 radiance = irradiance / solidAngle_sr;

    // Store result
    outputRadiance[coord] = float4(radiance, 1.0);
}
