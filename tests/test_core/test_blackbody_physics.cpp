// ============================================================================
// Quantiloom - Physical Validation Tests for Blackbody Radiation
// ============================================================================
// Tests verify that Planck's law constants and formulas are physically correct
//
// Physical validation points:
// 1. Planck constant C1_NM matches theoretical value
// 2. 300K blackbody radiance at Wien peak (~9.6μm)
// 3. Stefan-Boltzmann law integration
// 4. Wien's displacement law
// 5. Rayleigh scattering wavelength dependence (λ⁻⁴)
// ============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// Physical Constants (from blackbody.hlsli)
// ============================================================================

constexpr double PLANCK_H = 6.62607015e-34;  // J·s
constexpr double SPEED_OF_LIGHT_C = 299792458.0;  // m/s
constexpr double BOLTZMANN_K = 1.380649e-23;  // J/K
constexpr double STEFAN_BOLTZMANN = 5.670374419e-8;  // W·m⁻²·K⁻⁴

// Pre-computed constants (CORRECTED)
constexpr double C1_NM = 1.191042972e20;  // 2 × h × c² × 1e36 (W·nm⁴·sr⁻¹·m⁻²)
constexpr double C2 = 1.438776877e-2;      // h × c / k (m·K)

// Wien's displacement constant
constexpr double WIEN_CONSTANT = 2.897771955e-3;  // m·K

// ============================================================================
// Helper: Planck's Law Implementation (matches shader)
// ============================================================================

double IRPlanckRadiance(double temperature_K, double wavelength_nm) {
    if (temperature_K <= 0.0 || wavelength_nm <= 0.0) {
        return 0.0;
    }

    // Optimized: λ^5 via explicit multiplication
    double lambda_nm_2 = wavelength_nm * wavelength_nm;
    double lambda_nm_4 = lambda_nm_2 * lambda_nm_2;
    double lambda_nm_5 = lambda_nm_4 * wavelength_nm;

    double numerator = C1_NM / lambda_nm_5;

    // Convert wavelength to meters for exponent
    double lambda_m = wavelength_nm * 1e-9;
    double exponent = C2 / (lambda_m * temperature_K);

    double denominator = std::exp(exponent) - 1.0;

    return numerator / std::max(denominator, 1e-30);
}

// ============================================================================
// Helper: Stefan-Boltzmann Total Power
// ============================================================================

double IRStefanBoltzmannPower(double temperature_K) {
    if (temperature_K <= 0.0) {
        return 0.0;
    }

    double T_squared = temperature_K * temperature_K;
    double T_fourth = T_squared * T_squared;

    return STEFAN_BOLTZMANN * T_fourth;
}

// ============================================================================
// Helper: Wien's Peak Wavelength
// ============================================================================

double IRWienPeakWavelength(double temperature_K) {
    double lambda_peak_m = WIEN_CONSTANT / std::max(temperature_K, 1.0);
    return lambda_peak_m * 1e9;  // Convert to nm
}

// ============================================================================
// Physical Constant Validation Tests
// ============================================================================

TEST(BlackbodyPhysicsTest, PlanckConstantC1Verification) {
    // Verify C1_NM = 2hc² × 10³⁶
    double expected_C1 = 2.0 * PLANCK_H * SPEED_OF_LIGHT_C * SPEED_OF_LIGHT_C * 1e36;

    // Allow small tolerance due to floating point precision
    EXPECT_NEAR(C1_NM, expected_C1, expected_C1 * 1e-6);
}

TEST(BlackbodyPhysicsTest, PlanckConstantC2Verification) {
    // Verify C2 = hc/k — tightened to 1e-9 relative error to catch constant drift
    double expected_C2 = (PLANCK_H * SPEED_OF_LIGHT_C) / BOLTZMANN_K;

    EXPECT_NEAR(C2, expected_C2, expected_C2 * 1e-9);
}

TEST(BlackbodyPhysicsTest, WienConstantVerification) {
    // Wien's displacement constant: b = hc/(4.965114231·k)
    // where 4.965114231 is the numerical solution to x·exp(x)/(exp(x)-1) = 5
    double expected_wien = (PLANCK_H * SPEED_OF_LIGHT_C) / (4.965114231 * BOLTZMANN_K);

    EXPECT_NEAR(WIEN_CONSTANT, expected_wien, expected_wien * 1e-6);
}

// ============================================================================
// 300K Blackbody Tests (Room Temperature)
// ============================================================================

TEST(BlackbodyPhysicsTest, RoomTemperature300K_WienPeak) {
    // Wien's displacement law: λ_peak = b/T
    // For 300K: λ_peak = 2.8978×10⁻³ / 300 = 9.659 μm
    double peak_wavelength_nm = IRWienPeakWavelength(300.0);

    EXPECT_NEAR(peak_wavelength_nm, 9659.24, 1.0);  // 9.659 μm (tolerance: 1 nm)
}

TEST(BlackbodyPhysicsTest, RoomTemperature300K_RadianceAtPeak) {
    // At Wien peak, Planck radiance reaches maximum
    // For 300K at λ_peak ≈ 9.6 μm, L_λ ≈ 0.01 W·sr⁻¹·m⁻²·nm⁻¹
    // Verified via Python: 9.9525e-03 W·sr⁻¹·m⁻²·nm⁻¹
    double peak_wavelength_nm = IRWienPeakWavelength(300.0);
    double radiance_at_peak = IRPlanckRadiance(300.0, peak_wavelength_nm);

    // Expected: ~0.01 W·sr⁻¹·m⁻²·nm⁻¹ (allow 20% tolerance)
    EXPECT_GT(radiance_at_peak, 0.008);
    EXPECT_LT(radiance_at_peak, 0.012);
}

TEST(BlackbodyPhysicsTest, RoomTemperature300K_StefanBoltzmannTotal) {
    // Stefan-Boltzmann law: M = σT⁴
    // For 300K: M = 5.670374419e-8 × 300⁴ = 459.3 W/m²
    double total_power = IRStefanBoltzmannPower(300.0);

    EXPECT_NEAR(total_power, 459.3, 0.1);  // Tolerance: 0.1 W/m²
}

TEST(BlackbodyPhysicsTest, RoomTemperature300K_LWIRDominance) {
    // For 300K, emission should be dominated by LWIR (8-12 μm)
    double radiance_lwir = IRPlanckRadiance(300.0, 10000.0);  // 10 μm
    double radiance_mwir = IRPlanckRadiance(300.0, 4000.0);   // 4 μm
    double radiance_visible = IRPlanckRadiance(300.0, 550.0); // 550 nm

    // LWIR >> MWIR >> Visible
    EXPECT_GT(radiance_lwir, radiance_mwir * 10.0);
    EXPECT_GT(radiance_mwir, radiance_visible * 1e10);
}

// ============================================================================
// Temperature Scaling Tests
// ============================================================================

TEST(BlackbodyPhysicsTest, HumanBodyTemperature310K) {
    // Human body temperature: 310K (98.6°F)
    // Wien peak: λ_peak ≈ 9.35 μm
    // Total power: M ≈ 524 W/m²
    double peak_wavelength_nm = IRWienPeakWavelength(310.0);
    double total_power = IRStefanBoltzmannPower(310.0);

    EXPECT_NEAR(peak_wavelength_nm, 9348.0, 10.0);  // 9.348 μm
    EXPECT_NEAR(total_power, 524.0, 1.0);           // 524 W/m²
}

TEST(BlackbodyPhysicsTest, BoilingWater373K) {
    // Boiling water: 373K (212°F)
    // Wien peak: λ_peak ≈ 7.77 μm
    // Total power: M ≈ 1099 W/m²
    double peak_wavelength_nm = IRWienPeakWavelength(373.0);
    double total_power = IRStefanBoltzmannPower(373.0);

    EXPECT_NEAR(peak_wavelength_nm, 7767.0, 10.0);  // 7.767 μm
    EXPECT_NEAR(total_power, 1099.0, 2.0);          // 1099 W/m²
}

TEST(BlackbodyPhysicsTest, HotEngine600K) {
    // Hot engine: 600K
    // Wien peak: λ_peak ≈ 4.83 μm (MWIR)
    // Total power: M ≈ 7348 W/m²
    double peak_wavelength_nm = IRWienPeakWavelength(600.0);
    double total_power = IRStefanBoltzmannPower(600.0);

    EXPECT_NEAR(peak_wavelength_nm, 4830.0, 10.0);  // 4.830 μm
    EXPECT_NEAR(total_power, 7348.0, 10.0);         // 7348 W/m²
}

TEST(BlackbodyPhysicsTest, MoltenSteel1800K) {
    // Molten steel: 1800K
    // Wien peak: λ_peak ≈ 1.61 μm (SWIR)
    // Total power: M ≈ 597 kW/m²
    double peak_wavelength_nm = IRWienPeakWavelength(1800.0);
    double total_power = IRStefanBoltzmannPower(1800.0);

    EXPECT_NEAR(peak_wavelength_nm, 1610.0, 5.0);  // 1.610 μm
    EXPECT_NEAR(total_power, 597000.0, 2000.0);    // 597 kW/m² (relaxed tolerance)
}

TEST(BlackbodyPhysicsTest, Sun5800K) {
    // Sun surface temperature: 5800K
    // Wien peak: λ_peak ≈ 500 nm (visible green)
    // Total power: M ≈ 64 MW/m²
    double peak_wavelength_nm = IRWienPeakWavelength(5800.0);
    double total_power = IRStefanBoltzmannPower(5800.0);

    EXPECT_NEAR(peak_wavelength_nm, 500.0, 1.0);           // 500 nm
    EXPECT_NEAR(total_power, 64.0e6, 0.5e6);               // 64 MW/m²
}

// ============================================================================
// Wavelength Dependence Tests
// ============================================================================

TEST(BlackbodyPhysicsTest, WavelengthScaling_T4Law) {
    // For fixed wavelength, radiance scales as T⁴ at low energies (Rayleigh-Jeans limit)
    // This test verifies the temperature dependence
    double lambda_nm = 10000.0;  // 10 μm (far from Wien peak for both temperatures)

    double L_300 = IRPlanckRadiance(300.0, lambda_nm);
    double L_600 = IRPlanckRadiance(600.0, lambda_nm);

    // At long wavelengths (hc/λkT << 1), exp(hc/λkT) ≈ 1 + hc/λkT
    // So L ∝ T (Rayleigh-Jeans approximation)
    // But for moderate wavelengths, scaling is between T and T⁴
    double ratio = L_600 / L_300;

    // Expect ratio > 2 (linear scaling) but < 16 (T⁴ scaling)
    EXPECT_GT(ratio, 2.0);
    EXPECT_LT(ratio, 16.0);
}

TEST(BlackbodyPhysicsTest, PlanckLaw_ViennaSideCondition) {
    // For high frequencies (hc/λkT >> 1), Planck law reduces to Wien approximation
    // L_λ ≈ (2hc²/λ⁵) × exp(-hc/λkT)
    // Test that short-wavelength behavior follows exponential decay

    double T = 300.0;
    double lambda1_nm = 1000.0;  // 1 μm
    double lambda2_nm = 500.0;   // 0.5 μm (half wavelength)

    double L1 = IRPlanckRadiance(T, lambda1_nm);
    double L2 = IRPlanckRadiance(T, lambda2_nm);

    // At short wavelengths, L_λ decays exponentially with 1/λ
    // L(λ/2) / L(λ) should be extremely small (Wien limit)
    EXPECT_LT(L2 / L1, 1e-10);  // Expect many orders of magnitude smaller
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(BlackbodyPhysicsTest, ZeroTemperature) {
    // At T=0, radiance should be zero
    double radiance = IRPlanckRadiance(0.0, 10000.0);
    EXPECT_EQ(radiance, 0.0);

    double power = IRStefanBoltzmannPower(0.0);
    EXPECT_EQ(power, 0.0);
}

TEST(BlackbodyPhysicsTest, NegativeTemperature) {
    // Negative temperature is non-physical, should return zero
    double radiance = IRPlanckRadiance(-300.0, 10000.0);
    EXPECT_EQ(radiance, 0.0);

    double power = IRStefanBoltzmannPower(-300.0);
    EXPECT_EQ(power, 0.0);
}

TEST(BlackbodyPhysicsTest, NegativeWavelength) {
    // Negative wavelength is non-physical, should return zero
    double radiance = IRPlanckRadiance(300.0, -10000.0);
    EXPECT_EQ(radiance, 0.0);
}

TEST(BlackbodyPhysicsTest, VeryLongWavelength) {
    // At very long wavelengths (radio), Rayleigh-Jeans approximation should apply
    // L_λ ≈ (2ckT)/λ⁴
    double T = 300.0;
    double lambda_nm = 1e9;  // 1 meter = 1e9 nm

    double radiance = IRPlanckRadiance(T, lambda_nm);

    // Rayleigh-Jeans: L ≈ 2ckT/λ⁴
    double lambda_m = lambda_nm * 1e-9;
    double expected_RJ = (2.0 * SPEED_OF_LIGHT_C * BOLTZMANN_K * T) / std::pow(lambda_m, 4.0);
    expected_RJ *= 1e-9;  // Convert to per-nm

    // At very long wavelengths, Planck should match Rayleigh-Jeans
    EXPECT_NEAR(radiance, expected_RJ, expected_RJ * 0.01);  // 1% tolerance
}

// ============================================================================
// Atmospheric Scattering Physics Tests
// ============================================================================

TEST(BlackbodyPhysicsTest, RayleighScattering_Lambda4Dependence) {
    // Rayleigh scattering: β(λ) ∝ λ⁻⁴
    // Blue light (450nm) should scatter more than red light (650nm)
    // Ratio = (650/450)⁴ = 4.35

    double beta_550nm = 5.8e-6;  // m⁻¹ (typical value)

    // Simplified Rayleigh coefficient calculation (matching shader)
    auto rayleigh_coeff = [beta_550nm](double lambda_nm) -> double {
        double ratio = 550.0 / lambda_nm;
        double power4 = ratio * ratio * ratio * ratio;
        return beta_550nm * power4;
    };

    double beta_450 = rayleigh_coeff(450.0);  // Blue
    double beta_650 = rayleigh_coeff(650.0);  // Red

    double ratio = beta_450 / beta_650;

    // Expected: (650/450)⁴ = 4.35
    EXPECT_NEAR(ratio, 4.35, 0.01);
}

TEST(BlackbodyPhysicsTest, MieScattering_WeakerWavelengthDependence) {
    // Mie scattering: β(λ) ∝ λ⁻ᵅ where α ≈ 0.84
    // Wavelength dependence much weaker than Rayleigh

    double beta_550nm = 2.0e-6;  // m⁻¹
    double alpha = 0.84;

    auto mie_coeff = [beta_550nm, alpha](double lambda_nm) -> double {
        double ratio = 550.0 / lambda_nm;
        return beta_550nm * std::pow(ratio, alpha);
    };

    double beta_450 = mie_coeff(450.0);  // Blue
    double beta_650 = mie_coeff(650.0);  // Red

    double ratio_mie = beta_450 / beta_650;

    // Expected: (650/450)^0.84 ≈ 1.36 (much weaker than Rayleigh's 4.35)
    EXPECT_NEAR(ratio_mie, 1.36, 0.05);  // Relaxed tolerance
    EXPECT_LT(ratio_mie, 2.0);  // Must be weaker than λ⁻² scaling
}

// ============================================================================
// NIR Band Physics Tests (780-1400nm)
// ============================================================================
// NIR is "reflected infrared" - thermal emission is negligible for T < 600K
// These tests verify the physical basis for NIR_Fused mode implementation
// ============================================================================

TEST(BlackbodyPhysicsTest, NIR_RoomTemperature300K_NegligibleEmission) {
    // At 300K, thermal emission in NIR (780-1400nm) is essentially zero
    // Wien peak at 300K is ~9660nm, far from NIR band
    double radiance_nir_low = IRPlanckRadiance(300.0, 780.0);   // NIR start
    double radiance_nir_mid = IRPlanckRadiance(300.0, 1000.0);  // NIR mid
    double radiance_nir_high = IRPlanckRadiance(300.0, 1400.0); // NIR end
    double radiance_peak = IRPlanckRadiance(300.0, 9660.0);     // Wien peak

    // NIR emission should be many orders of magnitude smaller than peak
    // Relaxed thresholds based on actual Planck law calculations
    EXPECT_LT(radiance_nir_low, radiance_peak * 1e-18);   // ~8e-22 vs ~1e-20
    EXPECT_LT(radiance_nir_mid, radiance_peak * 1e-13);   // ~2e-16 vs ~1e-15
    EXPECT_LT(radiance_nir_high, radiance_peak * 1e-8);   // ~3e-11 vs ~1e-10
}

TEST(BlackbodyPhysicsTest, NIR_ThresholdTemperature600K) {
    // At 600K, Wien peak is ~4830nm (MWIR)
    // NIR emission starts to become measurable but still weak
    double radiance_nir = IRPlanckRadiance(600.0, 1000.0);  // 1μm
    double radiance_mwir = IRPlanckRadiance(600.0, 4000.0); // 4μm (near peak)

    // NIR should be much weaker than MWIR at 600K
    EXPECT_LT(radiance_nir, radiance_mwir * 0.01);  // < 1% of MWIR
}

TEST(BlackbodyPhysicsTest, NIR_HighTemperature1800K_SignificantEmission) {
    // At 1800K (molten steel), Wien peak is ~1610nm (SWIR/NIR boundary)
    // NIR emission becomes significant
    double radiance_nir = IRPlanckRadiance(1800.0, 1000.0);   // 1μm (NIR)
    double radiance_peak = IRPlanckRadiance(1800.0, 1610.0);  // Wien peak

    // At high temperature, NIR emission is substantial (> 10% of peak)
    EXPECT_GT(radiance_nir, radiance_peak * 0.1);
}

TEST(BlackbodyPhysicsTest, NIR_WavelengthRange_Verification) {
    // Verify NIR band spans 780-1400nm (ISO 20473 IR-A classification)
    constexpr double NIR_MIN = 780.0;   // nm
    constexpr double NIR_MAX = 1400.0;  // nm
    constexpr double VIS_MAX = 780.0;   // nm (visible light ends)

    // NIR should start where visible ends
    EXPECT_EQ(NIR_MIN, VIS_MAX);

    // NIR band width
    double nir_bandwidth = NIR_MAX - NIR_MIN;
    EXPECT_NEAR(nir_bandwidth, 620.0, 1.0);  // 620nm bandwidth
}

TEST(BlackbodyPhysicsTest, NIR_SunTemperature5800K_StrongEmission) {
    // Sun at 5800K peaks at ~500nm (visible)
    // NIR receives substantial solar radiation
    double radiance_visible = IRPlanckRadiance(5800.0, 550.0);  // Green peak
    double radiance_nir = IRPlanckRadiance(5800.0, 1000.0);     // 1μm NIR

    // NIR should be significant fraction of visible (solar spectrum is broad)
    // At 5800K, ratio should be roughly 0.3-0.5 (Wien side of peak)
    double ratio = radiance_nir / radiance_visible;
    EXPECT_GT(ratio, 0.2);
    EXPECT_LT(ratio, 0.6);
}

TEST(BlackbodyPhysicsTest, NIR_ReflectedVsThermal_Comparison) {
    // For outdoor scenes, compare solar NIR irradiance vs thermal emission
    // Sun: 5800K, object: 300K

    // Solar NIR radiance at object (scaled by solid angle, approximated)
    double sun_nir = IRPlanckRadiance(5800.0, 1000.0);

    // Thermal NIR emission from 300K object
    double thermal_nir = IRPlanckRadiance(300.0, 1000.0);

    // Solar should dominate by many orders of magnitude
    // This is why NIR_Fused mode uses reflected solar model, not thermal emission
    // Actual ratio is ~6e19, so use 1e19 as threshold
    EXPECT_GT(sun_nir / thermal_nir, 1e19);
}

TEST(BlackbodyPhysicsTest, NIR_BandIntegration_vs_SWIR) {
    // Compare average radiance across NIR vs SWIR bands at 300K
    // Both should be negligible at room temperature

    auto integrateRadiance = [](double T, double lambda_min, double lambda_max, int samples) {
        double sum = 0.0;
        double step = (lambda_max - lambda_min) / (samples - 1);
        for (int i = 0; i < samples; ++i) {
            double lambda = lambda_min + i * step;
            sum += IRPlanckRadiance(T, lambda);
        }
        return sum * step;
    };

    double nir_integral = integrateRadiance(300.0, 780.0, 1400.0, 16);
    double swir_integral = integrateRadiance(300.0, 1000.0, 2500.0, 16);
    double lwir_integral = integrateRadiance(300.0, 8000.0, 12000.0, 16);

    // LWIR should dominate at room temperature (Wien peak near 10μm)
    // NIR and SWIR have negligible thermal emission at 300K
    // Actual ratios: LWIR/NIR ~ 1e12, LWIR/SWIR ~ 3e4
    EXPECT_GT(lwir_integral, nir_integral * 1e8);
    EXPECT_GT(lwir_integral, swir_integral * 1e3);
}

// ============================================================================
// Angle-dependent IR emissivity: tests removed with the code
// ============================================================================
// A CPU mirror of GetAngleDependentIREmissivity / GetAngleDependentIRReflectance
// lived here with six tests. Both shader functions are gone (see the tombstone
// in src/shaders/common.hlsli), so the mirror described code that no longer
// exists, and the two load-bearing tests among the six --
// AngleEmissivity_MetalGrazingIncrease and _DielectricGrazingDecrease --
// asserted the strict monotonicity of the fabricated law itself, not physics.
// They would have to be deleted to fix the defect, which is a fair description
// of what they were protecting.
//
// The other four asserted only sums and clamps (eps + rho == 1, eps == eps0 at
// normal incidence, output in [0,1]) and were insensitive to the defect. They
// go too: a mirror of a deleted function is a claim about nothing.
//
// What replaces them is a render gate rather than a unit test, because the
// defect was never in the formula in isolation -- it was in feeding a
// view-dependent albedo to a Lambertian lobe, which no test of the formula
// alone can see. assets/configs/viewangle_{swir,lwir}_*.toml render one surface
// from two directions and require both to equal a closed form;
// scripts/render-tests/check_view_independence.py is the check, and it reads
// 16.5 % (reflective) and 28.9 % (thermal) against the old code.
// ============================================================================
