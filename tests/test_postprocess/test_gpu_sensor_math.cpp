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
// Both chains call PSFSigmaPixels() / PSFKernelRadiusPixels() from
// postprocess/SensorModel.hpp, so these tests exercise the production formula
// rather than a third copy of it -- an earlier version of this file restated
// the arithmetic inline, which meant a wrong constant in the renderer could be
// matched by the same wrong constant here and the suite would stay green.
//
// The expectations below are pinned to closed forms, not to decimal literals.

// Half-width at half maximum of the Airy pattern, in units of lambda*fNumber.
// The intensity is [2*J1(x)/x]^2 with x = pi*r/(lambda*N); bisect for the half
// maximum, then convert x back to a radius.
static auto AiryHWHM_LambdaN() -> f64 {
    const auto airyIntensity = [](f64 x) {
        const f64 t = 2.0 * std::cyl_bessel_j(1, x) / x;
        return t * t;
    };
    // The central lobe falls from 1 at x -> 0 to its first zero at x = 3.8317.
    f64 lo = 1.0, hi = 2.0;  // airyIntensity(1) > 1/2 > airyIntensity(2)
    for (int i = 0; i < 200; ++i) {
        const f64 mid = 0.5 * (lo + hi);
        if (airyIntensity(mid) > 0.5) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi) / std::numbers::pi;
}

// FWHM of a Gaussian is 2*sqrt(2*ln2) sigma.
static constexpr f64 kGaussianFWHMPerSigma = 2.3548200450309493;

TEST_F(GPUSensorMathTest, PSFSigmaMatchesAiryFWHM) {
    const f64 airyFWHM_lambdaN = 2.0 * AiryHWHM_LambdaN();

    // Sanity: the derivation reproduces the textbook 1.029 lambda*N.
    EXPECT_NEAR(airyFWHM_lambdaN, 1.02899, 1e-4);

    // The production constant is the Gaussian sigma of that same FWHM.
    EXPECT_NEAR(static_cast<f64>(kAiryGaussianSigmaFactor),
                airyFWHM_lambdaN / kGaussianFWHMPerSigma, 5e-4)
        << "PSF width must be a Gaussian matched to the Airy core's FWHM, not "
           "the Rayleigh radius 1.22 lambda*N";

    // And the derived sigma, in pixels, carries the relation through.
    params.wavelength_nm = 550.0f;
    params.fNumber = 2.8f;
    params.pixelPitch_um = 5.0f;
    const f64 lambdaN_px = (550e-9 * 2.8) / 5e-6;
    EXPECT_NEAR(PSFSigmaPixels(params) * kGaussianFWHMPerSigma,
                airyFWHM_lambdaN * lambdaN_px, 1e-3);
}

TEST_F(GPUSensorMathTest, PSFSigmaOverrideReplacesDerivedWidth) {
    const f32 derived = PSFSigmaPixels(params);
    ASSERT_GT(derived, 0.0f);

    // A supplied width is used verbatim, in pixels.
    params.psfSigma_px = 2.5f;
    EXPECT_FLOAT_EQ(PSFSigmaPixels(params), 2.5f);

    // Zero is a real width -- no blur -- and must not read as "unset".
    params.psfSigma_px = 0.0f;
    EXPECT_FLOAT_EQ(PSFSigmaPixels(params), 0.0f);
    EXPECT_EQ(PSFKernelRadiusPixels(PSFSigmaPixels(params)), 0u);

    // Any negative value falls back to the derived width.
    params.psfSigma_px = -1.0f;
    EXPECT_FLOAT_EQ(PSFSigmaPixels(params), derived);
    EXPECT_FLOAT_EQ(SensorParams{}.psfSigma_px, -1.0f) << "default is the sentinel";
}

TEST_F(GPUSensorMathTest, PSFSigmaClampRange) {
    // Below the floor, both chains skip the blur: radius 0 is a copy.
    params.psfSigma_px = kMinPSFSigmaPixels * 0.5f;
    EXPECT_EQ(PSFKernelRadiusPixels(PSFSigmaPixels(params)), 0u);

    // At the floor it blurs.
    params.psfSigma_px = kMinPSFSigmaPixels;
    EXPECT_GT(PSFKernelRadiusPixels(PSFSigmaPixels(params)), 0u);

    // Above the ceiling the kernel stops growing.
    params.psfSigma_px = 1000.0f;
    EXPECT_EQ(PSFKernelRadiusPixels(PSFSigmaPixels(params)),
              PSFKernelRadiusPixels(kMaxPSFSigmaPixels));
}

TEST_F(GPUSensorMathTest, PSFKernelRadius) {
    // radius = ceil(3 sigma), the 99.7% support of the Gaussian.
    EXPECT_EQ(PSFKernelRadiusPixels(3.5f), 11u);   // ceil(10.5)
    EXPECT_EQ(PSFKernelRadiusPixels(0.5f), 2u);    // ceil(1.5)

    // The fixture (550 nm, f/2.8, 5 um) is a sub-pixel PSF but still blurs.
    const f32 sigma = PSFSigmaPixels(params);
    EXPECT_LT(sigma, 1.0f);
    EXPECT_EQ(PSFKernelRadiusPixels(sigma),
              static_cast<u32>(std::ceil(3.0f * sigma)));
}

TEST_F(GPUSensorMathTest, PSFSigmaEdgeCases) {
    // Very fast lens (f/1.0) with short wavelength: well under a pixel.
    params.wavelength_nm = 400.0f;
    params.fNumber = 1.0f;
    params.pixelPitch_um = 5.0f;
    const f32 sigma1 = PSFSigmaPixels(params);
    EXPECT_GT(sigma1, 0.0f);
    EXPECT_LT(sigma1, 1.0f);

    // Very slow lens (f/22) with long wavelength (SWIR): several pixels.
    params.wavelength_nm = 1500.0f;
    params.fNumber = 22.0f;
    const f32 sigma2 = PSFSigmaPixels(params);
    EXPECT_GT(sigma2, sigma1);
    EXPECT_GT(sigma2, 2.0f);

    // Sigma is linear in both lambda and f#.
    params.fNumber = 44.0f;
    EXPECT_NEAR(PSFSigmaPixels(params), 2.0f * sigma2, 1e-5f);
}

// ============================================================================
// Radiance → Photo-electrons Conversion Tests
// ============================================================================

TEST_F(GPUSensorMathTest, SolidAngleFormula) {
    // Ω = π·sin²θ for a cone of half-angle θ with tanθ = 1/(2·f#).
    // Pinned to that relation rather than to a decimal, so the small-angle
    // form π/(4·f#²) -- which overstates collection by 1 + 1/(4·f#²), i.e.
    // 6.25% at f/2 -- cannot creep back in unnoticed.
    // The f-number arrives as f32, so widen theta from the same rounded value --
    // otherwise this compares the formula against a different aperture and the
    // ~1e-8 relative difference reads as a formula error.
    for (const f32 nf : {1.0f, 1.4f, 2.0f, 2.8f, 5.6f, 11.0f}) {
        const f64 n = static_cast<f64>(nf);
        const f64 theta = std::atan(1.0 / (2.0 * n));
        const f64 exact = std::numbers::pi * std::sin(theta) * std::sin(theta);
        EXPECT_NEAR(ApertureSolidAngleSr(nf), exact, 1e-15) << "at f/" << n;
    }

    // Faster lens → larger solid angle → more light.
    const f64 omega_28 = ApertureSolidAngleSr(2.8f);
    const f64 omega_14 = ApertureSolidAngleSr(1.4f);
    EXPECT_GT(omega_14, omega_28);

    // The exact law is not a pure 1/f#² ratio; the shortfall against 4.0 is the
    // bias the small-angle form used to hide -- ~8.5% between these two stops.
    const f64 n28 = static_cast<f64>(2.8f), n14 = static_cast<f64>(1.4f);
    EXPECT_NEAR(omega_14 / omega_28,
                (1.0 + 4.0 * n28 * n28) / (1.0 + 4.0 * n14 * n14), 1e-12);
    EXPECT_LT(omega_14 / omega_28, 4.0);
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
    const f64 omega = ApertureSolidAngleSr(2.8f);
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