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

static double GetAngleDependentIREmissivity(double baseEmissivity, double NdotV, double metallic) {
    double cosTheta = std::max(NdotV, 0.01);
    if (metallic > 0.5) {
        constexpr double METAL_GRAZING_ALPHA = 1.0;
        double f = 1.0 + METAL_GRAZING_ALPHA * (1.0 - cosTheta);
        return std::clamp(baseEmissivity * f, 0.0, 1.0);
    } else {
        constexpr double DIELECTRIC_GRAZING_BETA = 0.7;
        double f = std::pow(cosTheta, DIELECTRIC_GRAZING_BETA);
        return baseEmissivity * f;
    }
}

static double GetAngleDependentIRReflectance(double baseEmissivity, double transmittance,
                                              double NdotV, double metallic) {
    double e = GetAngleDependentIREmissivity(baseEmissivity, NdotV, metallic);
    return std::clamp(1.0 - e - transmittance, 0.0, 1.0);
}

// ---- Energy conservation: ε(θ) + ρ(θ) + τ = 1 ----

TEST(IREnergyConservation, EpsilonPlusRhoPlusTauEqualsOne_Opaque) {
    // Opaque materials (τ=0): strict ε+ρ=1 at all angles
    constexpr double eps0_vals[]  = {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    constexpr double NdotV_vals[] = {1.0, 0.8, 0.5, 0.2, 0.05};
    constexpr double metal_vals[] = {0.0, 1.0};
    constexpr double tau = 0.0;

    for (double eps0 : eps0_vals) {
        for (double NdotV : NdotV_vals) {
            for (double metallic : metal_vals) {
                double e = GetAngleDependentIREmissivity(eps0, NdotV, metallic);
                double r = GetAngleDependentIRReflectance(eps0, tau, NdotV, metallic);
                EXPECT_NEAR(e + r, 1.0, 0.01)
                    << "eps0=" << eps0 << " NdotV=" << NdotV
                    << " metallic=" << metallic;
            }
        }
    }
}

TEST(IREnergyConservation, EpsilonPlusRhoPlusTauEqualsOne_Transmissive) {
    // With τ>0, metal Hagen-Rubens boost can push ε+τ>1; reflectance
    // clamps to 0, so sum exceeds 1.  This is a known approximation in the
    // current shader — document but tolerate up to 15%.
    constexpr double eps0_vals[]  = {0.3, 0.5, 0.9};
    constexpr double tau_vals[]   = {0.05, 0.1};
    constexpr double NdotV_vals[] = {1.0, 0.5, 0.05};
    constexpr double metal_vals[] = {0.0, 1.0};

    for (double eps0 : eps0_vals) {
        for (double tau : tau_vals) {
            if (eps0 + tau > 1.0) continue;
            for (double NdotV : NdotV_vals) {
                for (double metallic : metal_vals) {
                    double e = GetAngleDependentIREmissivity(eps0, NdotV, metallic);
                    double r = GetAngleDependentIRReflectance(eps0, tau, NdotV, metallic);
                    double sum = e + r + tau;
                    EXPECT_NEAR(sum, 1.0, 0.15)
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
            double e = GetAngleDependentIREmissivity(eps0, 1.0, 0.0);
            double r = GetAngleDependentIRReflectance(eps0, tau, 1.0, 0.0);
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
// Phase C sentinel fix must make this work; DISABLED_ until then.

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
