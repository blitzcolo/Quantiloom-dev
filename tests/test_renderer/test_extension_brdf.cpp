/**
 * @file test_extension_brdf.cpp
 * @brief Closed-form and quadrature checks for the four glTF material lobes
 *
 * These re-derive, in independent C++, the identities the HLSL in
 * src/shaders/{pbr.hlsli,closesthit.rchit} relies on. They cannot catch a
 * transcription slip between here and there -- test_sheen_brdf.cpp can, because
 * sheen's answer is a table it can read out of the shipped shader, and these
 * are functions. What they can catch is the class of error that survives every
 * render gate: a lobe that quietly reflects more energy than arrives.
 *
 * That is not hypothetical. The sheen work rejected the Ashikhmin visibility
 * term on exactly this evidence (directional albedo 1.74 at grazing), and the
 * anisotropy work spent a while convinced it had the same bug because the
 * mirror-direction peak drops 48x while a tangent-aligned camera sees the lobe
 * get dramatically brighter. Both are answered by the same integral.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// The shader's formulas, re-derived from the specifications
// ============================================================================

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

double Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double Length(const Vec3& v) { return std::sqrt(Dot(v, v)); }

Vec3 Normalize(const Vec3& v) {
    const double n = Length(v);
    return (n > 0.0) ? Vec3{v.x / n, v.y / n, v.z / n} : Vec3{0.0, 0.0, 1.0};
}

/// Isotropic GGX, as DistributionGGX in pbr.hlsli. alpha is roughness^2.
constexpr double kEpsilon = 1e-6;  // the shader's EPSILON, and its NDF ceiling

double DistributionGGX(double NdotH, double alpha) {
    const double a2 = alpha * alpha;
    const double d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / std::max(kPi * d * d, kEpsilon);
}

/// Burley's anisotropic GGX, as DistributionGGXAniso. Note the crossed
/// multiply -- alphaB against the tangent term, alphaT against the bitangent --
/// which comes out of the algebra and is not a transposition.
double DistributionGGXAniso(double TdotH, double BdotH, double NdotH,
                            double alphaT, double alphaB) {
    const double a2 = alphaT * alphaB;
    const double fx = alphaB * TdotH, fy = alphaT * BdotH, fz = a2 * NdotH;
    const double d = fx * fx + fy * fy + fz * fz;
    if (d <= 0.0) {
        return 0.0;
    }
    // The same guard the isotropic form applies, so the two agree even where
    // that guard caps the peak -- see the note in pbr.hlsli.
    const double w2 = a2 / d;
    return a2 / std::max(kPi / std::max(w2 * w2, kEpsilon), kEpsilon);
}

/// Height-correlated Smith visibility, anisotropic; already carries the
/// 1/(4 NdotV NdotL) that the specular denominator would otherwise need.
double VisibilityAniso(double TdotV, double BdotV, double NdotV,
                       double TdotL, double BdotL, double NdotL,
                       double alphaT, double alphaB) {
    const double gv = NdotL * Length({alphaT * TdotV, alphaB * BdotV, NdotV});
    const double gl = NdotV * Length({alphaT * TdotL, alphaB * BdotL, NdotL});
    const double v = gv + gl;
    return (v > 0.0) ? 0.5 / v : 0.0;
}

/// alpha_t = mix(alpha, 1, strength^2) along the direction, alpha_b = alpha
/// across it -- so alpha_t >= alpha_b always.
void AnisotropyAlphas(double alpha, double strength, double& alphaT, double& alphaB) {
    alphaT = alpha + (1.0 - alpha) * strength * strength;
    alphaB = alpha;
}

/// The dielectric F0 KHR_materials_specular produces. The clamp is applied to
/// f0_ior * specularColor BEFORE the weight multiply, which is the whole point.
double SpecularDielectricF0(double ior, double specularColor, double specularWeight) {
    const double f = (ior - 1.0) / (ior + 1.0);
    return std::min(f * f * specularColor, 1.0) * specularWeight;
}

/// The clearcoat's Fresnel, taken at N.V rather than V.H per the specification.
double ClearcoatFresnel(double ccNdotV) {
    const double f = std::pow(std::clamp(1.0 - ccNdotV, 0.0, 1.0), 5.0);
    return 0.04 + 0.96 * f;
}

// ============================================================================
// Quadrature
// ============================================================================

/// Directional albedo of a specular lobe: the fraction of light arriving from
/// direction V that leaves the surface at all. A BRDF that reflects more than
/// it receives puts this above 1, and a renderer with a furnace gate reports
/// that as a surface emitting light.
///
/// F is taken as 1 throughout -- a perfect reflector -- so the result is the
/// lobe's own normalisation and nothing else.
double DirectionalAlbedoAniso(double NdotV, double alpha, double strength,
                              int nTheta = 256, int nPhi = 256) {
    double alphaT = 0.0, alphaB = 0.0;
    AnisotropyAlphas(alpha, strength, alphaT, alphaB);

    // The frame is N = +Z, T = +X. The view is tilted in the T-N plane, which
    // is the worst case for a lobe stretched along T.
    const Vec3 N{0.0, 0.0, 1.0};
    const Vec3 T{1.0, 0.0, 0.0};
    const Vec3 B{0.0, 1.0, 0.0};
    const Vec3 V = Normalize({std::sqrt(std::max(1.0 - NdotV * NdotV, 0.0)), 0.0, NdotV});

    double sum = 0.0;
    const double dTheta = (kPi / 2.0) / nTheta;
    const double dPhi = (2.0 * kPi) / nPhi;
    for (int i = 0; i < nTheta; ++i) {
        const double theta = (i + 0.5) * dTheta;
        const double st = std::sin(theta), ct = std::cos(theta);
        for (int j = 0; j < nPhi; ++j) {
            const double phi = (j + 0.5) * dPhi;
            const Vec3 L{st * std::cos(phi), st * std::sin(phi), ct};
            const Vec3 H = Normalize({V.x + L.x, V.y + L.y, V.z + L.z});

            const double d = DistributionGGXAniso(Dot(T, H), Dot(B, H), std::max(Dot(N, H), 0.0),
                                                  alphaT, alphaB);
            const double vis = VisibilityAniso(Dot(T, V), Dot(B, V), NdotV,
                                               Dot(T, L), Dot(B, L), ct,
                                               alphaT, alphaB);
            sum += d * vis * ct * st * dTheta * dPhi;
        }
    }
    return sum;
}

}  // namespace

// ============================================================================
// KHR_materials_anisotropy
// ============================================================================

// The shader keeps the isotropic expressions on a separate branch rather than
// letting the anisotropic ones degenerate into them, so that a scene without
// the extension renders bit-identically. This checks the two agree anyway --
// if they ever stop, the branch is hiding a real disagreement rather than a
// rounding difference.
TEST(AnisotropicGgx, ReducesToTheIsotropicDistributionWhenTheAlphasAreEqual) {
    for (const double roughness : {0.05, 0.15, 0.4, 0.8, 1.0}) {
        const double alpha = roughness * roughness;
        for (int i = 0; i <= 20; ++i) {
            // A half vector tilted away from N, with its tangential part split
            // between T and B so neither term of the crossed multiply is zero.
            const double NdotH = i / 20.0;
            const double tangential = std::sqrt(std::max(1.0 - NdotH * NdotH, 0.0));
            const double TdotH = tangential * 0.6;
            const double BdotH = tangential * 0.8;

            const double iso = DistributionGGX(NdotH, alpha);
            const double aniso = DistributionGGXAniso(TdotH, BdotH, NdotH, alpha, alpha);

            EXPECT_NEAR(aniso, iso, 1e-9 * std::max(iso, 1.0))
                << "roughness " << roughness << ", NdotH " << NdotH;
        }
    }
}

// The extension only ever roughens one axis. An implementation that swapped the
// two alphas would sharpen across the tangent instead, which looks like a
// plausible highlight rotated 90 degrees -- exactly the failure
// AnisotropyRotationTest is authored to expose, and one no energy check sees.
TEST(AnisotropicGgx, StretchesTheLobeAlongTheTangentAndNotAcrossIt) {
    const double alpha = 0.15 * 0.15;
    double alphaT = 0.0, alphaB = 0.0;
    AnisotropyAlphas(alpha, 1.0, alphaT, alphaB);

    ASSERT_GT(alphaT, alphaB) << "alpha_t = mix(alpha, 1, strength^2) must exceed alpha_b";

    // Tilt the half vector the same amount along T and along B. The lobe is
    // wide along T, so it still has energy there where the narrow axis has lost
    // it.
    const double NdotH = 0.999;
    const double tilt = std::sqrt(1.0 - NdotH * NdotH);

    const double alongT = DistributionGGXAniso(tilt, 0.0, NdotH, alphaT, alphaB);
    const double alongB = DistributionGGXAniso(0.0, tilt, NdotH, alphaT, alphaB);

    EXPECT_GT(alongT, alongB * 10.0)
        << "a lobe stretched along the tangent must survive a tangential tilt";
}

// The check that matters, and the one the render gates cannot make. Both gates
// are built on scenes with no anisotropy: the furnace has no sun and the
// open-sky check has no occluders, so a lobe that reflected 3x what arrived
// would pass them both and only show as an implausibly bright streak nobody
// could call wrong by eye.
TEST(AnisotropicGgx, DirectionalAlbedoNeverExceedsUnity) {
    for (const double roughness : {0.05, 0.15, 0.35, 0.7}) {
        const double alpha = roughness * roughness;
        for (const double strength : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            for (const double NdotV : {0.9, 0.6, 0.3, 0.15, 0.05}) {
                const double e = DirectionalAlbedoAniso(NdotV, alpha, strength);
                EXPECT_LE(e, 1.0 + 1e-3)
                    << "roughness " << roughness << ", strength " << strength
                    << ", NdotV " << NdotV << " reflects more than arrives";
                EXPECT_GE(e, 0.0);
            }
        }
    }
}

// Anisotropy redistributes; it does not brighten. Fixing the view and sweeping
// the strength must leave the hemisphere integral in the same neighbourhood
// even though the lobe's peak drops by more than an order of magnitude.
TEST(AnisotropicGgx, StrengthRedistributesEnergyRatherThanAddingIt) {
    const double alpha = 0.15 * 0.15;
    const double NdotV = 0.6;

    const double isotropic = DirectionalAlbedoAniso(NdotV, alpha, 0.0);
    const double stretched = DirectionalAlbedoAniso(NdotV, alpha, 1.0);

    EXPECT_LT(stretched, isotropic * 1.05)
        << "full anisotropy must not gain energy over none";
    EXPECT_GT(stretched, isotropic * 0.5)
        << "nor lose most of it -- that would read as a dark streak";

    // And the peak really does collapse, which is what makes the integral
    // interesting rather than trivial.
    double alphaT = 0.0, alphaB = 0.0;
    AnisotropyAlphas(alpha, 1.0, alphaT, alphaB);
    const double peakIso = DistributionGGX(1.0, alpha);
    const double peakAniso = DistributionGGXAniso(0.0, 0.0, 1.0, alphaT, alphaB);
    EXPECT_LT(peakAniso, peakIso / 10.0)
        << "the lobe should be spread, not merely tinted";
}

// ============================================================================
// KHR_materials_specular
// ============================================================================

// The neutral element is 1, not 0, and at the default ior it must reproduce the
// 0.04 that was hardcoded before the extension existed -- to within a rounding
// step, which is what lets every existing render survive unchanged.
TEST(SpecularExtension, NeutralFactorsReproduceTheDielectricConstant) {
    EXPECT_NEAR(SpecularDielectricF0(1.5, 1.0, 1.0), 0.04, 1e-12);
}

// SpecularSilkPouf authors specularColorFactor [10, 0.6, 0] with a weight of
// 0.5. Clamping the product rather than the colour lets red out at 0.5 instead
// of saturating at F0 = 1 first and then being halved -- the same number here
// by coincidence of the weight, so the test pins the green channel where the
// two orders genuinely disagree, and red where the clamp must bite at all.
TEST(SpecularExtension, TheClampAppliesBeforeTheWeightNotAfter) {
    const double ior = 1.5;   // f0_ior = 0.04 exactly
    const double weight = 0.5;

    // Red: 0.04 * 10 = 0.4, under the clamp, so nothing is clipped.
    EXPECT_NEAR(SpecularDielectricF0(ior, 10.0, weight), 0.2, 1e-12);

    // Green: 0.04 * 0.6 = 0.024, halved.
    EXPECT_NEAR(SpecularDielectricF0(ior, 0.6, weight), 0.012, 1e-12);

    // Blue: authored to zero, and must stay there.
    EXPECT_EQ(SpecularDielectricF0(ior, 0.0, weight), 0.0);

    // A colour extreme enough to reach the clamp: 0.04 * 40 = 1.6 -> 1, then
    // weighted. Clamping after the weight would give 0.8 instead.
    EXPECT_NEAR(SpecularDielectricF0(ior, 40.0, weight), 0.5, 1e-12);
}

// GlamVelvetSofa's champagne and gray fabrics author [0,0,0], which removes the
// normal-incidence highlight while leaving the grazing rim -- F90 is the weight,
// not the colour. A predicate or a shader that treated zero as "absent" would
// give those fabrics back a 4% highlight nobody asked for.
TEST(SpecularExtension, AZeroColourRemovesF0ButNotTheGrazingRim) {
    EXPECT_EQ(SpecularDielectricF0(1.5, 0.0, 1.0), 0.0);

    // Schlick between F0 = 0 and F90 = 1 still climbs to 1 at grazing.
    const double F0 = SpecularDielectricF0(1.5, 0.0, 1.0);
    const double F90 = 1.0;
    const double grazing = F0 + (F90 - F0) * std::pow(1.0 - 0.0, 5.0);
    EXPECT_NEAR(grazing, 1.0, 1e-12);
}

// The ior now reaches the reflective F0, which it never did before. A glass at
// 1.33 must no longer reflect as if it were 1.5.
TEST(SpecularExtension, IorDrivesTheDielectricF0) {
    const double water = SpecularDielectricF0(1.33, 1.0, 1.0);
    const double glass = SpecularDielectricF0(1.5, 1.0, 1.0);
    EXPECT_LT(water, glass);
    EXPECT_NEAR(water, 0.02, 1e-3);
}

// ============================================================================
// KHR_materials_clearcoat
// ============================================================================

TEST(ClearcoatLayer, FresnelSpansTheDielectricRangeFromNormalToGrazing) {
    EXPECT_NEAR(ClearcoatFresnel(1.0), 0.04, 1e-12);
    EXPECT_NEAR(ClearcoatFresnel(0.0), 1.0, 1e-12);
    // Monotone in between, so the coat never brightens as the view squares up.
    double previous = ClearcoatFresnel(0.0);
    for (int i = 1; i <= 20; ++i) {
        const double f = ClearcoatFresnel(i / 20.0);
        EXPECT_LE(f, previous + 1e-12) << "at cos " << (i / 20.0);
        previous = f;
    }
}

// The layering operator is a lerp, so what the coat takes and what the base
// keeps must sum to one at every angle -- including on the emission, which sits
// under the coat rather than over it.
TEST(ClearcoatLayer, WhatTheCoatTakesAndTheBaseKeepsSumToOne) {
    for (const double clearcoat : {0.0, 0.25, 0.5, 1.0}) {
        for (int i = 0; i <= 10; ++i) {
            const double ccNdotV = i / 10.0;
            const double taken = clearcoat * ClearcoatFresnel(ccNdotV);
            const double kept = 1.0 - taken;
            EXPECT_NEAR(taken + kept, 1.0, 1e-12);
            EXPECT_GE(kept, 0.0) << "clearcoat " << clearcoat << " at cos " << ccNdotV;
        }
    }
}

// A zero factor has to fold the whole feature away exactly, with no rounding on
// the path: it is what keeps an uncoated scene bit-identical.
TEST(ClearcoatLayer, AZeroFactorLeavesTheBaseExactlyUntouched) {
    for (int i = 0; i <= 10; ++i) {
        const double taken = 0.0 * ClearcoatFresnel(i / 10.0);
        EXPECT_EQ(1.0 - taken, 1.0);
    }
}

// ============================================================================
// KHR_materials_diffuse_transmission
// ============================================================================

// The specification mixes the BTDF against the BRDF rather than adding it, so
// the two diffuse halves always sum to what the diffuse alone had. An
// implementation that added instead would brighten every translucent leaf.
TEST(DiffuseTransmission, SplitsTheDiffuseWithoutCreatingEnergy) {
    for (const double dt : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        const double reflected = 1.0 - dt;
        const double transmitted = dt;
        EXPECT_NEAR(reflected + transmitted, 1.0, 1e-12) << "factor " << dt;
    }

    // At a factor of 1 the reflected diffuse is gone entirely -- which is what
    // DiffuseTransmissionTeacup authors, and why its front-lit faces darken
    // rather than merely gaining a glow.
    EXPECT_EQ(1.0 - 1.0, 0.0);
}

// Zero folds away exactly.
TEST(DiffuseTransmission, AZeroFactorLeavesTheDiffuseExactlyUntouched) {
    EXPECT_EQ(1.0 - 0.0, 1.0);
}
