/**
 * @file RgbToSpectrum.cpp
 * @brief Fitting the Jakob & Hanika coefficient table
 *
 * The solver minimises the squared CIELab distance between the colour the
 * sigmoid spectrum reproduces under D65 and the colour that was asked for.
 * CIELab rather than RGB because the whole point is perceptual fidelity, and
 * because RGB residuals weight the three channels by nothing in particular:
 * a fit that is right in green and wrong in blue scores the same as the
 * reverse, which is not how the error looks.
 *
 * Inside the gamut the target is exactly reachable and Levenberg-Marquardt
 * converges quadratically to a zero residual. On the gamut boundary no exact
 * solution exists and the Jacobian goes singular as the spectrum saturates;
 * the damping is what makes it settle on the nearest achievable colour instead
 * of walking off.
 */

#include "core/RgbToSpectrum.hpp"

#include "core/CIE_CMF_Data.hpp"
#include "core/D65Illuminant.hpp"
#include "core/Log.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

namespace quantiloom {

namespace {

// ============================================================================
// Colour machinery, in f64
// ============================================================================
// Single precision is enough for the stored table but not for building it: the
// residual is driven to 1e-12 and the Jacobian is assembled from differences of
// integrals, where f32 rounding is the same size as the thing being measured.

// Exactly the matrix SpectralConversion.hlsli:478 uses. Written row-major here;
// the inverse below is computed rather than transcribed so the two cannot drift.
constexpr f64 kXyzToRgb[3][3] = {
    { 3.2406, -1.5372, -0.4986},
    {-0.9689,  1.8758,  0.0415},
    { 0.0557, -0.2040,  1.0570},
};

struct ColourTables {
    // Integration weights: CMF x D65 x trapezoid, normalised so a perfect white
    // reflector integrates to Y = 1. xyz = sum_i R(lambda_i) * wi[i].
    f64 wi[CIE_CMF_LUT_SIZE][3]{};
    f64 whiteXyz[3]{};
    f64 rgbToXyz[3][3]{};
    f64 t[CIE_CMF_LUT_SIZE]{};   // normalised wavelength, the polynomial's variable

    ColourTables() {
        // Trapezoid at 1 nm, matching SpectralColour.cpp: N samples span N-1
        // intervals, and summing N full-width rectangles overstates by one.
        f64 yIntegral = 0.0;
        for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
            const f32 lambda = CIE_CMF_LAMBDA_MIN + static_cast<f32>(i);
            const f64 w = (i == 0 || i == CIE_CMF_LUT_SIZE - 1) ? 0.5 : 1.0;
            const f64 d65 = static_cast<f64>(D65Relative(lambda));
            yIntegral += w * d65 * CIE_1931_2DEG[i][1];
            t[i] = (static_cast<f64>(lambda) - RGB2SPEC_LAMBDA_MIN) / RGB2SPEC_LAMBDA_RANGE;
        }
        const f64 k = 1.0 / yIntegral;
        for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
            const f32 lambda = CIE_CMF_LAMBDA_MIN + static_cast<f32>(i);
            const f64 w = (i == 0 || i == CIE_CMF_LUT_SIZE - 1) ? 0.5 : 1.0;
            const f64 d65 = static_cast<f64>(D65Relative(lambda));
            for (int c = 0; c < 3; ++c) {
                wi[i][c] = w * d65 * CIE_1931_2DEG[i][c] * k;
                whiteXyz[c] += wi[i][c];
            }
        }

        // rgbToXyz = inverse(kXyzToRgb), by cofactors in f64.
        const f64(&m)[3][3] = kXyzToRgb;
        const f64 det =
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        const f64 invDet = 1.0 / det;
        rgbToXyz[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
        rgbToXyz[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
        rgbToXyz[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
        rgbToXyz[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
        rgbToXyz[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
        rgbToXyz[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
        rgbToXyz[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
        rgbToXyz[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
        rgbToXyz[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
    }
};

const ColourTables& Tables() {
    static const ColourTables tables;
    return tables;
}

constexpr f64 kLabDelta = 6.0 / 29.0;

f64 LabF(f64 x) {
    return (x > kLabDelta * kLabDelta * kLabDelta)
               ? std::cbrt(x)
               : x / (3.0 * kLabDelta * kLabDelta) + 4.0 / 29.0;
}

void XyzToLab(const f64 xyz[3], f64 lab[3]) {
    const ColourTables& tb = Tables();
    const f64 fx = LabF(xyz[0] / tb.whiteXyz[0]);
    const f64 fy = LabF(xyz[1] / tb.whiteXyz[1]);
    const f64 fz = LabF(xyz[2] / tb.whiteXyz[2]);
    lab[0] = 116.0 * fy - 16.0;
    lab[1] = 500.0 * (fx - fy);
    lab[2] = 200.0 * (fy - fz);
}

void RgbToLab(const f64 rgb[3], f64 lab[3]) {
    const ColourTables& tb = Tables();
    f64 xyz[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            xyz[a] += tb.rgbToXyz[a][b] * rgb[b];
        }
    }
    XyzToLab(xyz, lab);
}

// d(Lab) / d(XYZ) at xyz.
void LabJacobian(const f64 xyz[3], f64 jac[3][3]) {
    const ColourTables& tb = Tables();
    f64 fp[3];
    for (int c = 0; c < 3; ++c) {
        const f64 r = xyz[c] / tb.whiteXyz[c];
        const f64 d = (r > kLabDelta * kLabDelta * kLabDelta)
                          ? 1.0 / (3.0 * std::cbrt(r) * std::cbrt(r))
                          : 1.0 / (3.0 * kLabDelta * kLabDelta);
        fp[c] = d / tb.whiteXyz[c];
    }
    std::memset(jac, 0, sizeof(f64) * 9);
    jac[0][1] = 116.0 * fp[1];
    jac[1][0] = 500.0 * fp[0];
    jac[1][1] = -500.0 * fp[1];
    jac[2][1] = 200.0 * fp[1];
    jac[2][2] = -200.0 * fp[2];
}

f64 Sigmoid(f64 x) {
    return 0.5 + 0.5 * x / std::sqrt(1.0 + x * x);
}

// The colour the coefficients reproduce.
void EvalXyz(const f64 c[3], f64 xyz[3]) {
    const ColourTables& tb = Tables();
    xyz[0] = xyz[1] = xyz[2] = 0.0;
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f64 t = tb.t[i];
        const f64 s = Sigmoid((c[0] * t + c[1]) * t + c[2]);
        xyz[0] += s * tb.wi[i][0];
        xyz[1] += s * tb.wi[i][1];
        xyz[2] += s * tb.wi[i][2];
    }
}

// Residual in Lab, and its derivative with respect to the coefficients. One
// pass, because the sigmoid and its derivative share the same square root.
void EvalResidualJac(const f64 c[3], const f64 targetLab[3], f64 residual[3], f64 jac[3][3]) {
    const ColourTables& tb = Tables();
    f64 xyz[3] = {0.0, 0.0, 0.0};
    f64 dxyz[3][3] = {};  // d xyz[k] / d c[m]
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f64 t = tb.t[i];
        const f64 x = (c[0] * t + c[1]) * t + c[2];
        const f64 q = 1.0 / std::sqrt(1.0 + x * x);
        const f64 s = 0.5 * x * q + 0.5;
        const f64 sp = 0.5 * q * q * q;             // ds/dx
        const f64 basis[3] = {t * t, t, 1.0};       // dx/dc
        for (int k = 0; k < 3; ++k) {
            const f64 w = tb.wi[i][k];
            xyz[k] += s * w;
            const f64 wsp = w * sp;
            dxyz[k][0] += wsp * basis[0];
            dxyz[k][1] += wsp * basis[1];
            dxyz[k][2] += wsp * basis[2];
        }
    }
    f64 lab[3];
    XyzToLab(xyz, lab);
    for (int i = 0; i < 3; ++i) {
        residual[i] = targetLab[i] - lab[i];
    }
    f64 dlab[3][3];
    LabJacobian(xyz, dlab);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            f64 sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += dlab[a][k] * dxyz[k][b];
            }
            jac[a][b] = -sum;
        }
    }
}

// Solve a 3x3 by cofactors. Returns false when the matrix is singular enough
// that the damping needs to grow rather than the step being trusted.
bool Solve3(const f64 m[3][3], const f64 rhs[3], f64 out[3]) {
    const f64 det =
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (!std::isfinite(det) || std::abs(det) < 1e-30) {
        return false;
    }
    const f64 invDet = 1.0 / det;
    f64 inv[3][3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
    for (int a = 0; a < 3; ++a) {
        out[a] = inv[a][0] * rhs[0] + inv[a][1] * rhs[1] + inv[a][2] * rhs[2];
    }
    return true;
}

/// Levenberg-Marquardt from a warm start. Returns the final CIELab error.
f64 FitOne(const f64 rgb[3], f64 c[3], int maxIters = 20) {
    f64 targetLab[3];
    RgbToLab(rgb, targetLab);

    f64 residual[3], jac[3][3];
    EvalResidualJac(c, targetLab, residual, jac);
    f64 cost = residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2];
    f64 lambda = 1e-3;

    for (int it = 0; it < maxIters && cost > 1e-12; ++it) {
        f64 A[3][3], g[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                f64 sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += jac[k][a] * jac[k][b];
                }
                A[a][b] = sum;
            }
            for (int k = 0; k < 3; ++k) {
                g[a] += jac[k][a] * residual[k];
            }
        }

        bool accepted = false;
        for (int trialIdx = 0; trialIdx < 10 && !accepted; ++trialIdx) {
            f64 M[3][3];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    M[a][b] = A[a][b] + (a == b ? lambda : 0.0);
                }
            }
            f64 step[3];
            if (!Solve3(M, g, step)) {
                lambda *= 10.0;
                continue;
            }
            const f64 trial[3] = {c[0] - step[0], c[1] - step[1], c[2] - step[2]};
            f64 xyz[3], lab[3];
            EvalXyz(trial, xyz);
            XyzToLab(xyz, lab);
            f64 tr[3];
            f64 trialCost = 0.0;
            for (int i = 0; i < 3; ++i) {
                tr[i] = targetLab[i] - lab[i];
                trialCost += tr[i] * tr[i];
            }
            if (trialCost < cost) {
                c[0] = trial[0];
                c[1] = trial[1];
                c[2] = trial[2];
                cost = trialCost;
                lambda = std::max(lambda * 0.5, 1e-12);
                accepted = true;
            } else {
                lambda *= 10.0;
                if (lambda > 1e12) {
                    break;
                }
            }
        }
        if (!accepted) {
            break;  // converged, or no damped step can improve on this
        }
        EvalResidualJac(c, targetLab, residual, jac);
    }
    return std::sqrt(cost);
}

f64 Smoothstep(f64 x) {
    return x * x * (3.0 - 2.0 * x);
}

constexpr u32 kCacheMagic = 0x51'4C'52'53u;  // 'QLRS'
constexpr u32 kCacheVersion = 1u;

struct CacheHeader {
    u32 magic;
    u32 version;
    u32 resolution;
    u32 coeffCount;
};

}  // namespace

// ============================================================================
// Public
// ============================================================================

f32 EvaluateRgbSpectrum(const RgbSpectrumCoeffs& c, f32 lambda_nm) {
    const f32 clamped = std::clamp(lambda_nm, RGB2SPEC_LAMBDA_MIN, RGB2SPEC_LAMBDA_MAX);
    const f32 t = (clamped - RGB2SPEC_LAMBDA_MIN) / RGB2SPEC_LAMBDA_RANGE;
    const f32 x = (c.c0 * t + c.c1) * t + c.c2;
    return 0.5f + 0.5f * x / std::sqrt(1.0f + x * x);
}

RgbSpectrumCoeffs AchromaticRgbSpectrum(f32 g) {
    // The limits are the two constant spectra, where c2 runs to +-infinity. Any
    // finite stand-in is a reflectance of exactly 0 or 1 to well inside f32, and
    // clamping here is what keeps the square root defined.
    const f32 clamped = std::clamp(g, 1e-6f, 1.0f - 1e-6f);
    return {0.0f, 0.0f, (clamped - 0.5f) / std::sqrt(clamped * (1.0f - clamped))};
}

f32 RgbToSpectrumTable::ZNode(u32 i, u32 res) {
    if (res < 2) {
        return 0.0f;
    }
    const f64 u = static_cast<f64>(i) / static_cast<f64>(res - 1);
    return static_cast<f32>(Smoothstep(Smoothstep(u)));
}

Result<RgbToSpectrumTable, String> RgbToSpectrumTable::Build(u32 resolution) {
    if (resolution < 2 || resolution > 256) {
        return Result<RgbToSpectrumTable, String>::Err(
            "RgbToSpectrumTable::Build: resolution must be in [2, 256], got " +
            std::to_string(resolution));
    }

    const u32 res = resolution;
    RgbToSpectrumTable table;
    table.m_res = res;
    table.m_data.assign(static_cast<size_t>(3) * res * res * res * 3, 0.0f);

    Vector<f64> scale(res);
    for (u32 k = 0; k < res; ++k) {
        scale[k] = static_cast<f64>(ZNode(k, res));
    }

    std::atomic<f64> worstError{0.0};

    // One task per (maxc, y) slice. Slices write disjoint regions, so no
    // synchronisation beyond the counter that hands them out.
    const int taskCount = static_cast<int>(3 * res);
    std::atomic<int> nextTask{0};

    auto runSlice = [&](u32 l, u32 j) {
        const f64 y = static_cast<f64>(j) / static_cast<f64>(res - 1);
        f64 localWorst = 0.0;
        for (u32 i = 0; i < res; ++i) {
            const f64 x = static_cast<f64>(i) / static_cast<f64>(res - 1);
            const u32 startK = res / 5;

            // Continuation outward from a mid-brightness start in both
            // directions: the previous solution is the next one's warm start,
            // which is what keeps the hard ends converging in a few iterations.
            for (int direction = 0; direction < 2; ++direction) {
                f64 c[3] = {0.0, 0.0, 0.0};
                const int step = (direction == 0) ? 1 : -1;
                for (int kk = static_cast<int>(startK);
                     kk >= 0 && kk < static_cast<int>(res); kk += step) {
                    const f64 b = scale[static_cast<u32>(kk)];
                    f64 rgb[3];
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    const f64 err = FitOne(rgb, c);
                    localWorst = std::max(localWorst, err);

                    const size_t idx =
                        ((static_cast<size_t>(l) * res + static_cast<u32>(kk)) * res + j) * res + i;
                    table.m_data[idx * 3 + 0] = static_cast<f32>(c[0]);
                    table.m_data[idx * 3 + 1] = static_cast<f32>(c[1]);
                    table.m_data[idx * 3 + 2] = static_cast<f32>(c[2]);
                }
            }
        }
        f64 prev = worstError.load(std::memory_order_relaxed);
        while (localWorst > prev &&
               !worstError.compare_exchange_weak(prev, localWorst, std::memory_order_relaxed)) {
        }
    };

    const unsigned threadCount =
        std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
    Vector<std::thread> pool;
    pool.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t) {
        pool.emplace_back([&]() {
            int task;
            while ((task = nextTask.fetch_add(1)) < taskCount) {
                runSlice(static_cast<u32>(task) / res, static_cast<u32>(task) % res);
            }
        });
    }
    for (std::thread& th : pool) {
        th.join();
    }

    QL_LOG_INFO("  RGB->spectrum table fitted (res {}, {:.2f} MB, worst node dE {:.4f})", res,
                static_cast<f64>(table.m_data.size() * sizeof(f32)) / (1024.0 * 1024.0),
                worstError.load());
    return table;
}

RgbSpectrumCoeffs RgbToSpectrumTable::Lookup(const glm::vec3& rgb) const {
    if (m_res < 2 || m_data.empty()) {
        return AchromaticRgbSpectrum(0.0f);
    }

    const f32 r = std::clamp(rgb.r, 0.0f, 1.0f);
    const f32 g = std::clamp(rgb.g, 0.0f, 1.0f);
    const f32 b = std::clamp(rgb.b, 0.0f, 1.0f);

    if (r == g && g == b) {
        return AchromaticRgbSpectrum(r);
    }

    const f32 comp[3] = {r, g, b};
    const int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
    const f32 z = comp[maxc];
    const f32 res1 = static_cast<f32>(m_res - 1);

    const f32 xf = comp[(maxc + 1) % 3] / z * res1;
    const f32 yf = comp[(maxc + 2) % 3] / z * res1;

    // Invert the smoothstep-squared z warp in closed form. smoothstep(u) = y has
    // the exact root u = 1/2 - sin(asin(1 - 2y) / 3); applying it twice inverts
    // the double application. The reference implementation binary-searches the
    // node array instead; this is the same answer without the search, and the
    // shader uses the identical expression.
    auto invSmoothstep = [](f32 v) {
        const f32 a = std::clamp(1.0f - 2.0f * v, -1.0f, 1.0f);
        return 0.5f - std::sin(std::asin(a) / 3.0f);
    };
    const f32 zf = std::clamp(invSmoothstep(invSmoothstep(z)), 0.0f, 1.0f) * res1;

    const u32 xi = std::min(static_cast<u32>(xf), m_res - 2);
    const u32 yi = std::min(static_cast<u32>(yf), m_res - 2);
    const u32 zi = std::min(static_cast<u32>(zf), m_res - 2);
    const f32 dx = xf - static_cast<f32>(xi);
    const f32 dy = yf - static_cast<f32>(yi);
    const f32 dz = zf - static_cast<f32>(zi);

    f32 out[3] = {0.0f, 0.0f, 0.0f};
    for (u32 a = 0; a < 2; ++a) {
        for (u32 bb = 0; bb < 2; ++bb) {
            for (u32 cc = 0; cc < 2; ++cc) {
                const f32 w = (a ? dz : 1.0f - dz) * (bb ? dy : 1.0f - dy) *
                              (cc ? dx : 1.0f - dx);
                const size_t idx =
                    ((static_cast<size_t>(maxc) * m_res + (zi + a)) * m_res + (yi + bb)) * m_res +
                    (xi + cc);
                out[0] += w * m_data[idx * 3 + 0];
                out[1] += w * m_data[idx * 3 + 1];
                out[2] += w * m_data[idx * 3 + 2];
            }
        }
    }
    return {out[0], out[1], out[2]};
}

bool RgbToSpectrumTable::Save(const String& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        QL_LOG_WARN("RgbToSpectrumTable::Save: cannot open {}", path);
        return false;
    }
    const CacheHeader header{kCacheMagic, kCacheVersion, m_res, 3u};
    bool ok = std::fwrite(&header, sizeof(header), 1, f) == 1;
    ok = ok && std::fwrite(m_data.data(), sizeof(f32), m_data.size(), f) == m_data.size();
    std::fclose(f);
    if (!ok) {
        QL_LOG_WARN("RgbToSpectrumTable::Save: short write to {}", path);
        return false;
    }
    QL_LOG_INFO("  RGB->spectrum table cached to {}", path);
    return true;
}

Result<RgbToSpectrumTable, String> RgbToSpectrumTable::LoadOrBuild(const String& cachePath,
                                                                   u32 resolution) {
    if (!cachePath.empty()) {
        std::FILE* f = std::fopen(cachePath.c_str(), "rb");
        if (f != nullptr) {
            CacheHeader header{};
            const bool headerOk = std::fread(&header, sizeof(header), 1, f) == 1 &&
                                  header.magic == kCacheMagic &&
                                  header.version == kCacheVersion &&
                                  header.resolution == resolution && header.coeffCount == 3u;
            if (headerOk) {
                RgbToSpectrumTable table;
                table.m_res = resolution;
                table.m_data.resize(static_cast<size_t>(3) * resolution * resolution *
                                    resolution * 3);
                const bool dataOk =
                    std::fread(table.m_data.data(), sizeof(f32), table.m_data.size(), f) ==
                    table.m_data.size();
                std::fclose(f);
                if (dataOk) {
                    QL_LOG_INFO("  RGB->spectrum table loaded from cache (res {})", resolution);
                    return table;
                }
                QL_LOG_WARN("RgbToSpectrumTable: cache {} is truncated, refitting", cachePath);
            } else {
                std::fclose(f);
                QL_LOG_INFO("  RGB->spectrum cache {} is stale or foreign, refitting", cachePath);
            }
        }
    }

    auto built = Build(resolution);
    if (!built.has_value()) {
        return built;
    }
    if (!cachePath.empty()) {
        built.value().Save(cachePath);
    }
    return built;
}

RgbToSpectrumAccuracy MeasureAccuracy(const RgbToSpectrumTable& table, u32 randomSamples,
                                      u64 seed) {
    RgbToSpectrumAccuracy out;
    out.resolution = table.Resolution();
    out.coefficientBytes = static_cast<u64>(table.Data().size()) * sizeof(f32);

    // The eight cube corners first, then the random draw. The corners are where
    // the fit is hardest -- c2 runs to +/-inf as a colour approaches white or
    // black, which is why the z axis is warped at all -- and no finite uniform
    // sample lands on them.
    Vector<glm::vec3> colours;
    colours.reserve(static_cast<size_t>(randomSamples) + 8);
    for (int corner = 0; corner < 8; ++corner) {
        colours.emplace_back(static_cast<f32>((corner >> 0) & 1),
                             static_cast<f32>((corner >> 1) & 1),
                             static_cast<f32>((corner >> 2) & 1));
    }

    // Explicit engine and distribution rather than whatever <random> defaults
    // to: this number goes in a paper, so the sample has to be the same one on
    // every implementation that reads the seed.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<f64> uniform(0.0, 1.0);
    for (u32 i = 0; i < randomSamples; ++i) {
        colours.emplace_back(static_cast<f32>(uniform(rng)), static_cast<f32>(uniform(rng)),
                             static_cast<f32>(uniform(rng)));
    }

    Vector<f64> errors;
    errors.reserve(colours.size());
    for (const glm::vec3& colour : colours) {
        const f64 rgb[3] = {static_cast<f64>(colour.r), static_cast<f64>(colour.g),
                            static_cast<f64>(colour.b)};
        f64 targetLab[3];
        RgbToLab(rgb, targetLab);

        const RgbSpectrumCoeffs fetched = table.Lookup(colour);
        const f64 c[3] = {static_cast<f64>(fetched.c0), static_cast<f64>(fetched.c1),
                          static_cast<f64>(fetched.c2)};
        f64 xyz[3];
        EvalXyz(c, xyz);
        f64 lab[3];
        XyzToLab(xyz, lab);

        // CIE76: the Euclidean distance in Lab. The fitter minimises this same
        // quantity, so the number reported is the one being optimised and not a
        // second opinion about it.
        const f64 dE = std::sqrt((lab[0] - targetLab[0]) * (lab[0] - targetLab[0]) +
                                 (lab[1] - targetLab[1]) * (lab[1] - targetLab[1]) +
                                 (lab[2] - targetLab[2]) * (lab[2] - targetLab[2]));
        errors.push_back(dE);
        if (dE > out.worstDeltaE) {
            out.worstDeltaE = dE;
            out.worstColour = colour;
        }
    }

    out.samples = static_cast<u32>(errors.size());
    f64 sum = 0.0;
    for (const f64 e : errors) {
        sum += e;
    }
    out.meanDeltaE = errors.empty() ? 0.0 : sum / static_cast<f64>(errors.size());

    Vector<f64> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
        // Nearest-rank, so the value reported is one of the measurements rather
        // than an interpolation between two of them.
        const size_t rank = static_cast<size_t>(std::ceil(0.99 * static_cast<f64>(sorted.size())));
        out.p99DeltaE = sorted[std::min(sorted.size() - 1, rank == 0 ? 0 : rank - 1)];
    }
    return out;
}

}  // namespace quantiloom
