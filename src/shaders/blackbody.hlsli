// ============================================================================
// Quantiloom - Blackbody Radiation (Planck's Law)
// ============================================================================
// Implements Planck's law for thermal self-emission in infrared rendering
//
// References:
// - Planck, M. (1901). "On the Law of Distribution of Energy in the Normal Spectrum"
// - PBRT-v4 Chapter 12: Light Sources
// - "Physically Based Rendering of Thermal Emissions" (various sources)
// ============================================================================

#ifndef QUANTILOOM_BLACKBODY_HLSLI
#define QUANTILOOM_BLACKBODY_HLSLI

// ============================================================================
// Physical Constants
// ============================================================================

static const float PLANCK_H = 6.62607015e-34;  // Planck constant (J·s)
static const float SPEED_OF_LIGHT_C = 299792458.0;   // Speed of light (m/s)
static const float BOLTZMANN_K = 1.380649e-23;       // Boltzmann constant (J/K)

// ============================================================================
// Planck's Law: Spectral Radiance of Blackbody
// ============================================================================
// Computes the spectral radiance L_λ(T, λ) of a perfect blackbody at
// temperature T and wavelength λ.
//
// Formula:
//   L_λ(T, λ) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
//
// Where:
//   h = 6.62607015×10⁻³⁴ J·s (Planck constant)
//   c = 299792458 m/s (speed of light)
//   k = 1.380649×10⁻²³ J/K (Boltzmann constant)
//   T = temperature (K)
//   λ = wavelength (m)
//
// Units:
//   temperature_K: Kelvin [0, ∞)
//   wavelength_nm: nanometers [100, 100000] (covers UV to far-IR)
//   Returns: L_λ in W·sr⁻¹·m⁻²·nm⁻¹
//
// Typical temperatures:
//   - Human body: 310 K (98.6°F)
//   - Room temperature: 300 K (80°F)
//   - Boiling water: 373 K (212°F)
//   - Engine: 500-1000 K
//   - Molten metal: 1500-3000 K
//   - Daylight sun: ~5800 K
// ============================================================================

float IRPlanckRadiance(float temperature_K, float wavelength_nm) {
    // Guard against invalid inputs
    if (temperature_K <= 0.0 || wavelength_nm <= 0.0) {
        return 0.0;
    }

    // Convert wavelength from nm to meters
    float lambda_m = wavelength_nm * 1e-9;

    // Precompute constants
    float c1 = 2.0 * PLANCK_H * SPEED_OF_LIGHT_C * SPEED_OF_LIGHT_C;  // 2hc²
    float c2 = PLANCK_H * SPEED_OF_LIGHT_C / BOLTZMANN_K;             // hc/k

    // Planck's law
    float numerator = c1 / pow(lambda_m, 5.0);
    float exponent = c2 / (lambda_m * temperature_K);

    // Avoid overflow: exp(x) overflows for x > ~88
    // For very short wavelengths or low temperatures, exp(exponent) >> 1,
    // so we can approximate: exp(x) - 1 ≈ exp(x)
    float denominator;
    if (exponent > 50.0) {
        // exp(50) ≈ 5e21, denominator ≈ exp(exponent)
        denominator = exp(exponent);
    } else {
        denominator = exp(exponent) - 1.0;
    }

    // L_λ in W·sr⁻¹·m⁻²·m⁻¹
    float L_lambda_per_m = numerator / max(denominator, 1e-30);

    // Convert from /m to /nm
    float L_lambda_per_nm = L_lambda_per_m * 1e-9;

    return L_lambda_per_nm;
}

// ============================================================================
// Helper: Wien's Displacement Law (Peak Wavelength)
// ============================================================================
// Computes the wavelength at which blackbody radiation peaks for a given
// temperature.
//
// Formula:
//   λ_peak = b / T
//
// Where:
//   b = 2.897771955 × 10⁻³ m·K (Wien's displacement constant)
//
// Examples:
//   - Human body (310 K): λ_peak ≈ 9.3 μm (LWIR)
//   - Sun (5800 K): λ_peak ≈ 500 nm (visible green)
//
// Returns: Peak wavelength in nanometers
// ============================================================================

float IRWienPeakWavelength(float temperature_K) {
    const float WIEN_CONSTANT = 2.897771955e-3;  // m·K
    float lambda_peak_m = WIEN_CONSTANT / max(temperature_K, 1.0);
    return lambda_peak_m * 1e9;  // Convert to nm
}

#endif // QUANTILOOM_BLACKBODY_HLSLI
