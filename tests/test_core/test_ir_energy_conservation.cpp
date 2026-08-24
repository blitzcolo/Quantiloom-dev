// IR energy conservation tests — guards Phase C/D/E changes.
// CPU mirrors of shader functions from common.hlsli.

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <array>

// Physical constants (same as test_blackbody_physics.cpp / blackbody.hlsli)
static constexpr double C1_NM = 1.191042972e20;
static constexpr double C2    = 1.438776877e-2;

static double IRPlanckRadiance(double T_K, double lambda_nm) {
    if (T_K <= 0.0 || lambda_nm <= 0.0) return 0.0;
    double l2 = lambda_nm * lambda_nm;
    double l5 = l2 * l2 * lambda_nm;
    double lambda_m = lambda_nm * 1e-9;
    double exponent = C2 / (lambda_m * T_K);
    if (exponent > 700.0) return 0.0;
    double denom = std::exp(exponent) - 1.0;
    if (denom <= 0.0) return 0.0;
    return C1_NM / l5 / denom;
}

// ---- CPU mirrors of common.hlsli IR functions ----

static double GetEffectiveIREmissivity(double irEmissivity, double metallic, double roughness) {
    if (irEmissivity < 0.0) {
        double e = 0.95 - 0.90 * metallic;
        e += metallic * roughness * 0.15;
        return std::clamp(e, 0.0, 1.0);
    }
    return irEmissivity;
}

// Mirror of the shipping shader: emissivity is hemispherical, so there is no
// view argument to pass. This replaced a mirror of GetAngleDependentIR*, which
// modulated eps by cos^0.7(theta) and derived rho from the result -- a
// view-dependent albedo that was then used in Lambertian lobes. Both shader
// functions are deleted; see the tombstone in src/shaders/common.hlsli.
static double IREmissivityOf(double baseEmissivity) { return baseEmissivity; }

static double IRReflectanceOf(double baseEmissivity, double transmittance) {
    return std::clamp(1.0 - baseEmissivity - transmittance, 0.0, 1.0);
}

// ---- Energy conservation: ε(θ) + ρ(θ) + τ = 1 ----

TEST(IREnergyConservation, EpsilonPlusRhoPlusTauEqualsOne_Opaque) {
    // Opaque materials (τ=0): strict ε+ρ=1 at all angles
    constexpr double eps0_vals[]  = {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    constexpr double NdotV_vals[] = {1.0, 0.8, 0.5, 0.2, 0.05};
    constexpr double metal_vals[] = {0.0, 1.0};
    constexpr double tau = 0.0;

    for (double eps0 : eps0_vals) {
        double e = IREmissivityOf(eps0);
        double r = IRReflectanceOf(eps0, tau);
        // Exact, not within 0.01: with no angular modulation this is an
        // identity rather than an approximation.
        EXPECT_NEAR(e + r, 1.0, 1e-12) << "eps0=" << eps0;
        // And it does not depend on the view. The sweep is kept for that
        // reason alone -- the values above are angle-free now, so the loop
        // documents the invariant the render gate measures end to end
        // (scripts/render-tests/check_view_independence.py).
        for (double NdotV : NdotV_vals) {
            for (double metallic : metal_vals) {
                EXPECT_DOUBLE_EQ(IREmissivityOf(eps0), e)
                    << "eps0=" << eps0 << " NdotV=" << NdotV
                    << " metallic=" << metallic;
            }
        }
    }
}

TEST(IREnergyConservation, EpsilonPlusRhoPlusTauEqualsOne_Transmissive) {
    // This used to tolerate 15%, with the comment "metal Hagen-Rubens boost can
    // push ε+τ>1; reflectance clamps to 0, so sum exceeds 1 -- a known
    // approximation in the current shader". That approximation is gone, so the
    // sum is now exact and the tolerance says so.
    constexpr double eps0_vals[]  = {0.3, 0.5, 0.9};
    constexpr double tau_vals[]   = {0.05, 0.1};
    constexpr double NdotV_vals[] = {1.0, 0.5, 0.05};
    constexpr double metal_vals[] = {0.0, 1.0};

    for (double eps0 : eps0_vals) {
        for (double tau : tau_vals) {
            if (eps0 + tau > 1.0) continue;
            for (double NdotV : NdotV_vals) {
                for (double metallic : metal_vals) {
                    double e = IREmissivityOf(eps0);
                    double r = IRReflectanceOf(eps0, tau);
                    double sum = e + r + tau;
                    EXPECT_NEAR(sum, 1.0, 1e-12)
                        << "eps0=" << eps0 << " tau=" << tau
                        << " NdotV=" << NdotV << " metallic=" << metallic;
                }
            }
        }
    }
}

// ---- Isothermal cavity invariant: sum of all terms = B(T,λ) ----

TEST(IREnergyConservation, IsothermalCavityInvariant) {
    constexpr double temps[]   = {250.0, 300.0, 400.0, 600.0};
    constexpr double lambdas[] = {3000.0, 5000.0, 8000.0, 10000.0, 12000.0, 14000.0};
    constexpr double eps0 = 0.6;
    constexpr double tau  = 0.0;

    for (double T : temps) {
        for (double lam : lambdas) {
            double B = IRPlanckRadiance(T, lam);
            ASSERT_GT(B, 0.0);

            // In an isothermal cavity every surface sees B(T) from all directions.
            // Emitted + reflected + transmitted = ε·B + ρ·B + τ·B = (ε+ρ+τ)·B = B
            double e = IREmissivityOf(eps0);
            double r = IRReflectanceOf(eps0, tau);
            double L = e * B + r * B + tau * B;
            EXPECT_NEAR(L / B, 1.0, 1e-6)
                << "T=" << T << " lambda=" << lam;
        }
    }
}

// Cross-check against harness.py pre-baked values (planck_blackbody)
TEST(IREnergyConservation, PlanckCrossCheckHarness) {
    // 300K @ 10µm: near Wien peak, ~0.01 W/sr/m²/nm
    double L = IRPlanckRadiance(300.0, 10000.0);
    EXPECT_NEAR(L, 0.00992, 5e-04);

    // 600K @ Wien peak (~4830nm): strong MWIR emitter
    double L2 = IRPlanckRadiance(600.0, 4830.0);
    EXPECT_GT(L2, 0.1);
    EXPECT_LT(L2, 0.5);
}

// ---- Sentinel behavior: irEmissivity = -1 → metallic heuristic ----
// The sentinel is implemented; these run and pass. (They previously carried a
// note saying they were DISABLED_ pending a "Phase C" fix, which had long since
// landed -- the tests were never disabled.)
//
// Note what these can and cannot catch: GetEffectiveIREmissivity above is a CPU
// transcription of the HLSL in common.hlsli:927, so they pin the formula as
// written here. Editing the shader without editing the mirror leaves both green
// and the two silently different. Change them together.

TEST(IREnergyConservation, SentinelDerivedEmissivity_Dielectric) {
    double e = GetEffectiveIREmissivity(-1.0, 0.0, 0.5);
    // Dielectric: 0.95 - 0.90*0 + 0*0.5*0.15 = 0.95
    EXPECT_NEAR(e, 0.95, 0.01);
}

TEST(IREnergyConservation, SentinelDerivedEmissivity_Metal) {
    double e = GetEffectiveIREmissivity(-1.0, 1.0, 0.5);
    // Metal: 0.95 - 0.90*1 + 1*0.5*0.15 = 0.125
    EXPECT_NEAR(e, 0.125, 0.01);
}

TEST(IREnergyConservation, SentinelDerivedEmissivity_HalfMetal) {
    double e = GetEffectiveIREmissivity(-1.0, 0.5, 0.5);
    // 0.95 - 0.90*0.5 + 0.5*0.5*0.15 = 0.5375
    EXPECT_NEAR(e, 0.5375, 0.01);
}
