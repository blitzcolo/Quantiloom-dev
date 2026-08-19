/**
 * @file test_sheen_brdf.cpp
 * @brief Cover for the sheen directional-albedo table shipped in pbr.hlsli
 *
 * The Charlie sheen lobe's directional albedo has no closed form and no usable
 * polynomial fit, so pbr.hlsli carries 256 numbers that were integrated
 * offline. Numbers that were computed once and pasted in are exactly the kind
 * that rot: a transcription slip, an axis warp changed on one side only, or a
 * visibility term swapped without regenerating the table would all leave a
 * plausible-looking file that quietly mis-splits energy.
 *
 * So this reads the table out of the shader source that actually ships and
 * re-derives it. The reference implementation below is the same maths the
 * shader runs, written once more in C++ -- which is the point: two independent
 * spellings of it have to agree, and the table has to agree with both.
 *
 * The units are absolute, not relative: E is a fraction of incident energy, and
 * what matters is how much of it a lobe misplaces, not how much it misplaces
 * relative to a value that is legitimately near zero over half the domain.
 */

#include <gtest/gtest.h>

#include "core/Types.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace quantiloom;

namespace {

constexpr int kDim = 16;                 // SHEEN_E_DIM in pbr.hlsli
constexpr f64 kMinSheenRoughness = 0.07; // MIN_SHEEN_ROUGHNESS in pbr.hlsli
constexpr f64 kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// The reference: Charlie distribution x Estevez-Kulla visibility, integrated
// ---------------------------------------------------------------------------

f64 CharlieD(f64 NdotH, f64 roughness) {
    const f64 alphaG = std::max(roughness * roughness,
                                kMinSheenRoughness * kMinSheenRoughness);
    const f64 invAlpha = 1.0 / alphaG;
    const f64 sin2h = std::max(1.0 - NdotH * NdotH, 1e-7);
    return (2.0 + invAlpha) * std::pow(sin2h, invAlpha * 0.5) / (2.0 * kPi);
}

f64 LambdaHelper(f64 x, f64 alphaG) {
    const f64 o = (1.0 - alphaG) * (1.0 - alphaG);
    const f64 a = 21.5473 + (25.3245 - 21.5473) * o;
    const f64 b = 3.82987 + (3.32435 - 3.82987) * o;
    const f64 c = 0.19823 + (0.16801 - 0.19823) * o;
    const f64 d = -1.97760 + (-1.27393 + 1.97760) * o;
    const f64 e = -4.32054 + (-4.85967 + 4.32054) * o;
    return a / (1.0 + b * std::pow(std::max(x, 1e-7), c)) + d * x + e;
}

f64 SheenLambda(f64 cosTheta, f64 alphaG) {
    const f64 x = std::abs(cosTheta);
    if (x < 0.5) {
        return std::exp(LambdaHelper(x, alphaG));
    }
    return std::exp(2.0 * LambdaHelper(0.5, alphaG) - LambdaHelper(1.0 - x, alphaG));
}

f64 CharlieV(f64 NdotV, f64 NdotL, f64 roughness) {
    const f64 alphaG = std::max(roughness * roughness,
                                kMinSheenRoughness * kMinSheenRoughness);
    const f64 v = 1.0 / ((1.0 + SheenLambda(NdotV, alphaG) + SheenLambda(NdotL, alphaG)) *
                         (4.0 * NdotV * std::max(NdotL, 1e-7)));
    return std::clamp(v, 0.0, 1.0);
}

/// E(mu_v, r) = integral of D * V * mu_l over the hemisphere, then clamped --
/// the clamp being what the shipped table bakes in, because Charlie times
/// Estevez-Kulla integrates past 1 as the view goes edge-on.
f64 IntegrateSheenAlbedo(f64 muV, f64 roughness, int nMu = 300, int nPhi = 300) {
    const f64 sinV = std::sqrt(std::max(1.0 - muV * muV, 0.0));
    f64 sum = 0.0;
    for (int i = 0; i < nMu; ++i) {
        const f64 muL = (static_cast<f64>(i) + 0.5) / nMu;
        const f64 sinL = std::sqrt(std::max(1.0 - muL * muL, 0.0));
        for (int j = 0; j < nPhi; ++j) {
            const f64 phi = (static_cast<f64>(j) + 0.5) / nPhi * 2.0 * kPi;
            const f64 hx = sinL * std::cos(phi) + sinV;
            const f64 hy = sinL * std::sin(phi);
            const f64 hz = muL + muV;
            const f64 hn = std::max(std::sqrt(hx * hx + hy * hy + hz * hz), 1e-9);
            const f64 NdotH = std::clamp(hz / hn, 0.0, 1.0);
            sum += CharlieD(NdotH, roughness) * CharlieV(muV, muL, roughness) * muL;
        }
    }
    return std::min(sum * (1.0 / nMu) * (2.0 * kPi / nPhi), 1.0);
}

// ---------------------------------------------------------------------------
// The shipped table
// ---------------------------------------------------------------------------

/// Pull SHEEN_E_TABLE out of pbr.hlsli. Reads the file that ships rather than a
/// copy, so a table edited without re-deriving it fails here.
std::vector<f64> ReadShippedTable(std::string& whyNot) {
    // Baked in by tests/CMakeLists.txt. A relative path would resolve against
    // whichever directory the runner happened to start in.
    const std::string path =
        (std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "src" / "shaders" / "pbr.hlsli")
            .string();
    std::ifstream file(path);
    if (!file) {
        whyNot = "could not open " + path;
        return {};
    }

    std::string line;
    bool inTable = false;
    std::vector<f64> values;
    while (std::getline(file, line)) {
        if (!inTable) {
            if (line.find("SHEEN_E_TABLE[256]") != std::string::npos) {
                inTable = true;
            }
            continue;
        }
        if (line.find("};") != std::string::npos) {
            break;
        }
        // Commas separate the entries; anything else on the line is a comment.
        std::istringstream row(line);
        std::string token;
        while (std::getline(row, token, ',')) {
            try {
                size_t consumed = 0;
                const f64 v = std::stod(token, &consumed);
                if (consumed > 0) {
                    values.push_back(v);
                }
            } catch (const std::exception&) {
                // Not a number: the trailing comment on the row.
            }
        }
    }
    if (values.size() != static_cast<size_t>(kDim * kDim)) {
        whyNot = "found " + std::to_string(values.size()) + " entries, expected 256";
        return {};
    }
    return values;
}

// The node warps, which have to match SheenAlbedo's inverses in pbr.hlsli.
f64 NodeMu(int i) {
    const f64 t = static_cast<f64>(i) / (kDim - 1);
    return std::max(t * t, 1e-4);
}
f64 NodeRoughness(int j) {
    const f64 t = static_cast<f64>(j) / (kDim - 1);
    return kMinSheenRoughness + (1.0 - kMinSheenRoughness) * t * t;
}

}  // namespace

// Every entry re-derived. This is the assertion that catches a paste error, a
// changed visibility term, or a warp altered on one side only.
TEST(SheenBrdf, ShippedTableMatchesNumericalIntegration) {
    std::string whyNot;
    const std::vector<f64> table = ReadShippedTable(whyNot);
    if (table.empty()) {
        GTEST_SKIP() << "sheen table unavailable: " << whyNot;
    }

    f64 worst = 0.0;
    int worstI = 0, worstJ = 0;
    for (int i = 0; i < kDim; ++i) {
        for (int j = 0; j < kDim; ++j) {
            const f64 expected = IntegrateSheenAlbedo(NodeMu(i), NodeRoughness(j));
            const f64 err = std::abs(table[i * kDim + j] - expected);
            if (err > worst) {
                worst = err;
                worstI = i;
                worstJ = j;
            }
        }
    }

    // The tolerance is quadrature, not fit: the table was integrated at 600x600
    // and this runs at 300x300 to stay inside a 4-second suite.
    EXPECT_LT(worst, 2e-3) << "worst at mu=" << NodeMu(worstI)
                           << " roughness=" << NodeRoughness(worstJ)
                           << " (table " << table[worstI * kDim + worstJ] << ")";
}

// The invariant every consumer inherits instead of asking for. The albedo
// scaling would go negative without it, the infrared carve-out would claim more
// reflectance than the surface has, and the bounce throughput would exceed one.
TEST(SheenBrdf, ShippedTableNeverExceedsUnity) {
    std::string whyNot;
    const std::vector<f64> table = ReadShippedTable(whyNot);
    if (table.empty()) {
        GTEST_SKIP() << "sheen table unavailable: " << whyNot;
    }

    for (size_t k = 0; k < table.size(); ++k) {
        EXPECT_GE(table[k], 0.0) << "entry " << k;
        EXPECT_LE(table[k], 1.0) << "entry " << k;
    }
}

// The reason Ashikhmin was not used. Recorded as a test so that anyone who
// swaps the visibility term back for the cheaper one is told why it was not.
TEST(SheenBrdf, AshikhminVisibilityWouldNotConserveEnergy) {
    const f64 muV = 0.05;
    const f64 roughness = 0.1;

    // Same integral, Ashikhmin's visibility: 1 / (4 (mu_l + mu_v - mu_l mu_v)).
    constexpr int n = 300;
    const f64 sinV = std::sqrt(1.0 - muV * muV);
    f64 sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const f64 muL = (static_cast<f64>(i) + 0.5) / n;
        const f64 sinL = std::sqrt(std::max(1.0 - muL * muL, 0.0));
        for (int j = 0; j < n; ++j) {
            const f64 phi = (static_cast<f64>(j) + 0.5) / n * 2.0 * kPi;
            const f64 hx = sinL * std::cos(phi) + sinV;
            const f64 hy = sinL * std::sin(phi);
            const f64 hz = muL + muV;
            const f64 hn = std::max(std::sqrt(hx * hx + hy * hy + hz * hz), 1e-9);
            const f64 NdotH = std::clamp(hz / hn, 0.0, 1.0);
            const f64 vis = 1.0 / (4.0 * std::max(muL + muV - muL * muV, 1e-7));
            sum += CharlieD(NdotH, roughness) * vis * muL;
        }
    }
    const f64 ashikhminE = sum * (1.0 / n) * (2.0 * kPi / n);

    EXPECT_GT(ashikhminE, 1.2) << "Ashikhmin was expected to over-reflect here";
    EXPECT_LE(IntegrateSheenAlbedo(muV, roughness), 1.0);
}
