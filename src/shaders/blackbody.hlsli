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
static const float STEFAN_BOLTZMANN = 5.670374419e-8;  // Stefan-Boltzmann constant (W·m⁻²·K⁻⁴)

// ============================================================================
// Pre-computed Constants for Planck's Law (Performance Optimization)
// ============================================================================
// These constants combine physical constants with unit conversion factors
// to minimize runtime computation in IRPlanckRadiance()
//
// C1_NM = 2hc² × 10³⁶ (converts λ from nm to m, and dλ from m to nm)
// C2 = hc/k (exponent factor)
//
// DERIVATION:
// Standard Planck formula (λ in meters):
//   L_λ(m⁻¹) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)  [W·sr⁻¹·m⁻²·m⁻¹]
//
// Convert wavelength to nanometers:
//   λ_m = λ_nm × 10⁻⁹
//   λ_m⁵ = λ_nm⁵ × 10⁻⁴⁵
//
// Convert spectral radiance from per-meter to per-nanometer:
//   L_λ(nm⁻¹) = L_λ(m⁻¹) × dλ_m/dλ_nm = L_λ(m⁻¹) × 10⁻⁹
//
// Combine:
//   L_λ(nm⁻¹) = (2hc²/[λ_nm⁵ × 10⁻⁴⁵]) / (...) × 10⁻⁹
//             = (2hc² × 10³⁶/λ_nm⁵) / (...)
//
// Therefore: C1_NM = 2hc² × 10³⁶ = 1.191042972 × 10²⁰
// ============================================================================

static const float C1_NM = 1.191042972e20;  // 2 × h × c² × 1e36 (W·nm⁴·sr⁻¹·m⁻²)
static const float C2 = 1.438776877e-2;      // h × c / k (m·K)

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

    // Optimized: Replace pow(lambda_nm, 5) with explicit multiplication
    // This is significantly faster on GPU hardware
    float lambda_nm_2 = wavelength_nm * wavelength_nm;
    float lambda_nm_4 = lambda_nm_2 * lambda_nm_2;
    float lambda_nm_5 = lambda_nm_4 * wavelength_nm;

    // Optimized: Use pre-computed constant C1_NM which already includes
    // the 2hc² term and the nm to m conversion factor
    float numerator = C1_NM / lambda_nm_5;

    // Convert wavelength to meters for exponent calculation
    float lambda_m = wavelength_nm * 1e-9;
    float exponent = C2 / (lambda_m * temperature_K);

    // Clamp exponent to prevent float32 overflow in exp()
    // exp(88) ≈ 1.65e38 ≈ FLT_MAX; beyond this exp() returns INF
    // When exponent > 80, the result is effectively zero anyway
    // (Wien limit: radiance drops exponentially for hc/λkT >> 1)
    exponent = min(exponent, 80.0);

    float denominator = exp(exponent) - 1.0;

    // L_λ in W·sr⁻¹·m⁻²·nm⁻¹
    // The 1e-30 prevents division by zero and handles edge cases
    float L_lambda_per_nm = numerator / max(denominator, 1e-30);

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

// ============================================================================
// Stefan-Boltzmann Law: Total Emissive Power
// ============================================================================
// Computes the total radiant exitance (power per unit area) emitted by a
// blackbody across all wavelengths.
//
// Formula:
//   M = σ × T⁴
//
// Where:
//   σ = 5.670374419 × 10⁻⁸ W·m⁻²·K⁻⁴ (Stefan-Boltzmann constant)
//   T = temperature (K)
//
// This is the integral of Planck's law over all wavelengths:
//   M = ∫₀^∞ L_λ(T, λ) dλ = σT⁴
//
// Units:
//   temperature_K: Kelvin [0, ∞)
//   Returns: Total emissive power in W·m⁻²
//
// Use cases:
//   - Determining overall thermal emission brightness
//   - Computing total radiative heat transfer
//   - Scaling spectral radiance for physically-based rendering
//
// Examples:
//   - Human body (310 K): M ≈ 524 W/m²
//   - Room temp object (300 K): M ≈ 459 W/m²
//   - Hot engine (600 K): M ≈ 7348 W/m²
//   - Molten steel (1800 K): M ≈ 597 kW/m²
// ============================================================================

float IRStefanBoltzmannPower(float temperature_K) {
    // Guard against invalid input
    if (temperature_K <= 0.0) {
        return 0.0;
    }

    // Optimized: Compute T⁴ using two squaring operations
    float T_squared = temperature_K * temperature_K;
    float T_fourth = T_squared * T_squared;

    // Total emissive power: M = σT⁴
    return STEFAN_BOLTZMANN * T_fourth;
}

// ============================================================================
// Helper: Emissivity-Adjusted Total Power (Graybody)
// ============================================================================
// Computes total emissive power for a graybody (non-ideal emitter) by
// applying an emissivity factor ε ∈ [0, 1].
//
// Formula:
//   M = ε × σ × T⁴
//
// Where:
//   ε = emissivity [0, 1] (dimensionless)
//       ε = 1.0: perfect blackbody
//       ε = 0.0: perfect reflector (no emission)
//
// Typical emissivities:
//   - Polished aluminum: 0.04
//   - Oxidized aluminum: 0.20
//   - Human skin: 0.98
//   - Water: 0.96
//   - Asphalt: 0.93
//   - Concrete: 0.92
//   - Paint (most colors): 0.90-0.95
//
// Returns: Total emissive power in W·m⁻²
// ============================================================================

float IRGraybodyPower(float temperature_K, float emissivity) {
    return emissivity * IRStefanBoltzmannPower(temperature_K);
}

#endif // QUANTILOOM_BLACKBODY_HLSLI
