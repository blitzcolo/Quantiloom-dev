// ============================================================================
// Quantiloom - Unit Tests for GPU Sensor Math Formulas
// ============================================================================
// Tests verify the mathematical formulas used by both CPU (GenericSensor)
// and GPU (compute shader) sensor chains. No Vulkan hardware required.
//
// Covers:
// - PSF sigma calculation (Airy disk → Gaussian sigma)
// - Radiance → Photo-electrons conversion
// - ADC quantization (electrons → DN)
// - FPN math (PRNU multiplicative, DSNU additive, NUC residual)
// ============================================================================

#include <gtest/gtest.h>
#include "postprocess/SensorModel.hpp"
#include "postprocess/GenericSensor.hpp"
#include "core/Image.hpp"
#include <algorithm>  // std::clamp -- used to arrive via GenericSensor.hpp
#include <cmath>
#include <numbers>

using namespace quantiloom;

// Physical constants (must match GenericSensor.cpp)
static constexpr f64 kPlanckConstant = 6.62607015e-34;
static constexpr f64 kSpeedOfLight = 299792458.0;

// ============================================================================
// Test Fixture
// ============================================================================

class GPUSensorMathTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.focalLength_mm = 50.0f;
        params.fNumber = 2.8f;
        params.pixelPitch_um = 5.0f;
        params.quantumEfficiency = 0.8f;
        params.wellCapacity_e = 50000.0f;
        params.readNoise_e_rms = 10.0f;
        params.darkCurrent_e_s = 50.0f;
        params.integrationTime_s = 0.01f;
        params.bitDepth = 14;
        params.gain = 3.0f;
        params.wavelength_nm = 550.0f;
        params.enablePoissonNoise = false;
        params.enableReadNoise = false;
        params.enableDarkCurrent = false;
        params.enableFPN = false;
        params.enableVignetting = false;
    }

    SensorParams params;
};

// ============================================================================
// PSF Sigma Calculation Tests
// ============================================================================
// GPU shader uses: σ = 1.22 × λ(m) × f# / pixel_pitch(m)
// CPU uses: airyRadius_um = 1.22 × λ(m) × 1e6 × f#, then σ = airyRadius / pitch
// Both are mathematically identical.

TEST_F(GPUSensorMathTest, PSFSigmaFormula) {
    // σ = 1.22 × λ × f# / pixel_pitch
    // With λ=550nm, f#=2.8, pitch=5μm:
    // σ = 1.22 × 550e-9 × 2.8 / (5e-6) = 0.37576 pixels
    const f32 lambda_m = 550e-9f;
    const f32 fNumber = 2.8f;
    const f32 pitch_m = 5e-6f;

    const f32 expected = 1.22f * lambda_m * fNumber / pitch_m;

    // Also compute via CPU path: airyRadius_um then divide by pitch_um
    const f32 airyRadius_um = 1.22f * lambda_m * 1e6f * fNumber;
    const f32 sigma_cpu = airyRadius_um / 5.0f;

    EXPECT_NEAR(expected, sigma_cpu, 1e-6f)
        << "GPU and CPU PSF sigma formulas must be equivalent";
    EXPECT_NEAR(expected, 0.37576f, 0.001f);
}

TEST_F(GPUSensorMathTest, PSFSigmaClampRange) {
    // GPU clamps sigma to [0.1, 10.0]
    // Test with very small f# → small sigma
    {
        f32 sigma = 1.22f * (550e-9f) * 1.0f / (5e-6f);  // f#=1.0
        sigma = std::max(0.1f, std::min(sigma, 10.0f));
        EXPECT_GE(sigma, 0.1f);
        EXPECT_LE(sigma, 10.0f);
    }
    // Test with extreme values that would exceed 10
    {
        f32 sigma = 1.22f * (2000e-9f) * 22.0f / (1e-6f);  // λ=2μm, f#=22, pitch=1μm
        sigma = std::max(0.1f, std::min(sigma, 10.0f));
        EXPECT_FLOAT_EQ(sigma, 10.0f);
    }
}

TEST_F(GPUSensorMathTest, PSFKernelRadius) {
    // radius = ceil(3 × σ)
    f32 sigma = 1.22f * (550e-9f) * 2.8f / (5e-6f);
    u32 radius = static_cast<u32>(std::ceil(3.0f * sigma));
    EXPECT_EQ(radius, 2u);  // ceil(3 × 0.376) = ceil(1.128) = 2

    // Larger sigma
    sigma = 3.5f;
    radius = static_cast<u32>(std::ceil(3.0f * sigma));
    EXPECT_EQ(radius, 11u);  // ceil(10.5) = 11
}

TEST_F(GPUSensorMathTest, PSFSigmaEdgeCases) {
    // Very fast lens (f/1.0) with short wavelength
    f32 sigma1 = 1.22f * (400e-9f) * 1.0f / (5e-6f);
    EXPECT_GT(sigma1, 0.0f);
    EXPECT_LT(sigma1, 1.0f);

    // Very slow lens (f/22) with long wavelength (SWIR)
    f32 sigma2 = 1.22f * (1500e-9f) * 22.0f / (5e-6f);
    EXPECT_GT(sigma2, sigma1);
    EXPECT_GT(sigma2, 5.0f);
}

// ============================================================================
// Radiance → Photo-electrons Conversion Tests
// ============================================================================

TEST_F(GPUSensorMathTest, SolidAngleFormula) {
    // Ω = π / (4 × f#²)
    const f64 omega_28 = std::numbers::pi / (4.0 * 2.8 * 2.8);
    EXPECT_NEAR(omega_28, 0.1001, 0.001);

    const f64 omega_14 = std::numbers::pi / (4.0 * 1.4 * 1.4);
    EXPECT_NEAR(omega_14, 0.4006, 0.001);

    // Faster lens → larger solid angle → more light
    EXPECT_GT(omega_14, omega_28);
    EXPECT_NEAR(omega_14 / omega_28, 4.0, 0.01);  // f#² ratio = (2.8/1.4)² = 4
}

TEST_F(GPUSensorMathTest, PhotonEnergyFormula) {
    // E = hc/λ
    const f64 lambda_m = 550e-9;
    const f64 E = (kPlanckConstant * kSpeedOfLight) / lambda_m;
    EXPECT_NEAR(E, 3.61e-19, 0.01e-19);

    // Shorter wavelength → higher energy
    const f64 E_blue = (kPlanckConstant * kSpeedOfLight) / 450e-9;
    EXPECT_GT(E_blue, E);
}

TEST_F(GPUSensorMathTest, RadianceToElectronsFormula) {
    // N_e = L × Ω × A × t × QE / E_photon
    // Use low radiance to avoid well capacity saturation (50000 e-)
    const f64 L = 0.5;  // 0.5 W/m²/sr (below saturation)
    const f64 omega = std::numbers::pi / (4.0 * 2.8 * 2.8);
    const f64 A = (5e-6) * (5e-6);  // pixel area m²
    const f64 t = 0.01;  // integration time
    const f64 QE = 0.8;
    const f64 E_photon = (kPlanckConstant * kSpeedOfLight) / 550e-9;

    const f64 expected_e = L * omega * A * t * QE / E_photon;

    // Use large uniform image so center pixel has no PSF boundary effects
    Image hdr(64, 64, 1);
    for (auto& v : hdr.data) v = static_cast<f32>(L);

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    // Sample center pixel (no PSF boundary effects)
    const u32 cx = 32, cy = 32;
    const f32 dn = result.value().rawDN(cx, cy, 0);
    const f64 electrons_approx = dn * params.gain;

    // Tolerance: quantization error < gain
    EXPECT_NEAR(electrons_approx, expected_e, params.gain + 1.0);
}

TEST_F(GPUSensorMathTest, DarkCurrentAddition) {
    // With dark current enabled: electrons += darkCurrent × integrationTime
    params.enableDarkCurrent = true;
    const f64 darkElectrons = params.darkCurrent_e_s * params.integrationTime_s;
    EXPECT_NEAR(darkElectrons, 0.5, 1e-6);

    // Zero radiance + dark current should produce small but nonzero DN
    Image hdr(64, 64, 1);
    for (auto& v : hdr.data) v = 0.0f;

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    // Center pixel DN should be floor(darkElectrons / gain)
    const u32 cx = 32, cy = 32;
    const f32 centerDN = result.value().rawDN(cx, cy, 0);
    const f32 expectedDN = std::floor(static_cast<f32>(darkElectrons / params.gain));
    EXPECT_NEAR(centerDN, expectedDN, 1.0f);
}

TEST_F(GPUSensorMathTest, WellCapacityClamp) {
    // Very bright radiance should saturate at well capacity
    Image hdr(64, 64, 1);
    for (auto& v : hdr.data) v = 1000.0f;  // Very bright

    GenericSensor sensor;
    auto result = sensor.Apply(hdr, params);
    ASSERT_TRUE(result.has_value());

    // Max DN = floor(wellCapacity / gain), clamped to ADC max
    const f32 maxDN = static_cast<f32>((1u << params.bitDepth) - 1);
    const f32 wellDN = std::floor(params.wellCapacity_e / params.gain);
    const f32 expectedMax = std::min(wellDN, maxDN);

    const u32 cx = 32, cy = 32;
    const f32 centerDN = result.value().rawDN(cx, cy, 0);
    EXPECT_NEAR(centerDN, expectedMax, 1.0f);
}

// ============================================================================
// ADC Quantization Tests
// ============================================================================

TEST_F(GPUSensorMathTest, QuantizationFormula) {
    // DN = floor(electrons / gain), clamped to [0, 2^bitDepth - 1]
    const f32 gain = 3.0f;

    EXPECT_FLOAT_EQ(std::floor(100.0f / gain), 33.0f);
    EXPECT_FLOAT_EQ(std::floor(99.0f / gain), 33.0f);
    EXPECT_FLOAT_EQ(std::floor(9.0f / gain), 3.0f);
    EXPECT_FLOAT_EQ(std::floor(2.0f / gain), 0.0f);
}

TEST_F(GPUSensorMathTest, ADCClipping) {
    // DN must be clamped to [0, 2^bitDepth - 1]
    for (u32 bits : {12u, 14u, 16u}) {
        const f32 maxDN = static_cast<f32>((1u << bits) - 1);
        // Electrons exceeding maxDN * gain should clip
        const f32 electrons = maxDN * 3.0f * 2.0f;  // 2x over max
        f32 dn = std::floor(electrons / 3.0f);
        dn = std::clamp(dn, 0.0f, maxDN);
        EXPECT_FLOAT_EQ(dn, maxDN);
    }
}

TEST_F(GPUSensorMathTest, BitDepthVariants) {
    // Verify ADC range for different bit depths
    Image hdr(64, 64, 1);
    for (auto& v : hdr.data) v = 0.5f;

    for (u32 bits : {12u, 14u, 16u}) {
        params.bitDepth = bits;
        const f32 maxDN = static_cast<f32>((1u << bits) - 1);

        GenericSensor sensor;
        auto result = sensor.Apply(hdr, params);
        ASSERT_TRUE(result.has_value());

        for (const auto& val : result.value().rawDN.data) {
            EXPECT_GE(val, 0.0f);
            EXPECT_LE(val, maxDN);
        }
    }
}

// ============================================================================
// FPN Math Tests
// ============================================================================

TEST_F(GPUSensorMathTest, PRNUMultiplicative) {
    // PRNU: signal' = signal × (1 + prnu)
    const f32 signal = 10000.0f;
    const f32 prnu = 0.02f;

    const f32 result = signal * (1.0f + prnu);
    EXPECT_FLOAT_EQ(result, 10200.0f);

    const f32 result_neg = signal * (1.0f + (-0.02f));
    EXPECT_FLOAT_EQ(result_neg, 9800.0f);
}

TEST_F(GPUSensorMathTest, DSNUAdditive) {
    // DSNU: signal' = signal + dsnu
    const f32 signal = 10000.0f;
    const f32 dsnu = 15.0f;

    EXPECT_FLOAT_EQ(signal + dsnu, 10015.0f);
    EXPECT_FLOAT_EQ(signal + (-dsnu), 9985.0f);

    // DSNU is signal-independent (same offset regardless of signal level)
    const f32 low = 100.0f;
    const f32 high = 40000.0f;
    EXPECT_FLOAT_EQ((low + dsnu) - low, dsnu);
    EXPECT_FLOAT_EQ((high + dsnu) - high, dsnu);
}

TEST_F(GPUSensorMathTest, NUCResidual) {
    // NUC reduces FPN by nucEfficiency factor:
    // residual_prnu = prnu × (1 - nucEfficiency)
    // residual_dsnu = dsnu × (1 - nucEfficiency)
    const f32 prnu = 0.02f;
    const f32 dsnu = 15.0f;
    const f32 efficiency = 0.98f;

    const f32 residual_prnu = prnu * (1.0f - efficiency);
    const f32 residual_dsnu = dsnu * (1.0f - efficiency);

    EXPECT_NEAR(residual_prnu, 0.0004f, 1e-6f);  // 2% of original
    EXPECT_NEAR(residual_dsnu, 0.3f, 1e-6f);

    // Apply NUC-corrected FPN to signal
    const f32 signal = 10000.0f;
    const f32 corrected = signal * (1.0f + residual_prnu) + residual_dsnu;
    // Much closer to original than without NUC
    EXPECT_NEAR(corrected, signal, 10.0f);
}

TEST_F(GPUSensorMathTest, NUCFullEfficiency) {
    // 100% NUC efficiency should completely eliminate FPN
    const f32 prnu = 0.05f;
    const f32 dsnu = 30.0f;
    const f32 efficiency = 1.0f;

    const f32 residual_prnu = prnu * (1.0f - efficiency);
    const f32 residual_dsnu = dsnu * (1.0f - efficiency);

    EXPECT_FLOAT_EQ(residual_prnu, 0.0f);
    EXPECT_FLOAT_EQ(residual_dsnu, 0.0f);

    // Signal should be completely unchanged
    const f32 signal = 10000.0f;
    const f32 corrected = signal * (1.0f + residual_prnu) + residual_dsnu;
    EXPECT_FLOAT_EQ(corrected, signal);
}