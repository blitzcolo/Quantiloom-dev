/**
 * @file NNLS.cpp
 * @brief Lawson-Hanson active set NNLS, specialised for k <= 4 and 3 rows
 */

#include "core/NNLS.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom {

namespace {

// Solve M s = rhs for a symmetric positive definite M of order n <= 4, by
// Gaussian elimination with partial pivoting. Returns false if M is singular
// to working precision, which the caller treats as "this subset cannot be
// fitted" and backs out of.
bool SolveSmallSpd(f64 M[MAX_ENDMEMBERS][MAX_ENDMEMBERS], f64 rhs[MAX_ENDMEMBERS],
                   i32 n, f64 out[MAX_ENDMEMBERS]) {
    for (i32 col = 0; col < n; ++col) {
        i32 pivot = col;
        for (i32 row = col + 1; row < n; ++row) {
            if (std::abs(M[row][col]) > std::abs(M[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(M[pivot][col]) < 1e-12) {
            return false;
        }
        if (pivot != col) {
            for (i32 c = 0; c < n; ++c) {
                std::swap(M[col][c], M[pivot][c]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        for (i32 row = col + 1; row < n; ++row) {
            const f64 factor = M[row][col] / M[col][col];
            if (factor == 0.0) {
                continue;
            }
            for (i32 c = col; c < n; ++c) {
                M[row][c] -= factor * M[col][c];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    for (i32 row = n - 1; row >= 0; --row) {
        f64 sum = rhs[row];
        for (i32 c = row + 1; c < n; ++c) {
            sum -= M[row][c] * out[c];
        }
        out[row] = sum / M[row][row];
    }
    return true;
}

// Unconstrained least squares restricted to the passive set, written back into
// full-length s (active entries zero).
bool SolvePassive(const glm::vec3* colors, const bool* passive, i32 k,
                  const glm::vec3& b, f64 s[MAX_ENDMEMBERS]) {
    i32 idx[MAX_ENDMEMBERS];
    i32 n = 0;
    for (i32 i = 0; i < k; ++i) {
        if (passive[i]) {
            idx[n++] = i;
        }
    }
    for (i32 i = 0; i < MAX_ENDMEMBERS; ++i) {
        s[i] = 0.0;
    }
    if (n == 0) {
        return true;
    }

    f64 M[MAX_ENDMEMBERS][MAX_ENDMEMBERS] = {};
    f64 rhs[MAX_ENDMEMBERS] = {};
    for (i32 r = 0; r < n; ++r) {
        const glm::vec3& cr = colors[idx[r]];
        rhs[r] = static_cast<f64>(glm::dot(cr, b));
        for (i32 c = 0; c < n; ++c) {
            M[r][c] = static_cast<f64>(glm::dot(cr, colors[idx[c]]));
        }
        M[r][r] += NNLS_RIDGE;
    }

    f64 sol[MAX_ENDMEMBERS] = {};
    if (!SolveSmallSpd(M, rhs, n, sol)) {
        return false;
    }
    for (i32 r = 0; r < n; ++r) {
        s[idx[r]] = sol[r];
    }
    return true;
}

}  // namespace

void SolveNNLS(const glm::vec3* colors, i32 k, const glm::vec3& b, f32* wOut) {
    k = std::clamp(k, 1, MAX_ENDMEMBERS);
    for (i32 i = 0; i < MAX_ENDMEMBERS; ++i) {
        wOut[i] = 0.0f;
    }

    // k == 1 is the whole problem: project b onto c0 and clamp. Worth spelling
    // out because it is the common case (every material bound to one measured
    // curve, which is what the mixture degenerates to) and it is exactly the
    // brightness modulation the flat-reflectance behaviour was missing.
    if (k == 1) {
        const f64 denom = static_cast<f64>(glm::dot(colors[0], colors[0]));
        if (denom > 1e-12) {
            wOut[0] = static_cast<f32>(std::max(0.0, static_cast<f64>(glm::dot(colors[0], b)) / denom));
        }
        return;
    }

    bool passive[MAX_ENDMEMBERS] = {};
    f64 w[MAX_ENDMEMBERS] = {};

    // At most k additions, each with at most k backtracking steps; the extra
    // factor of two is slack against a cycle from rounding, not a real bound.
    const i32 maxOuter = 2 * k;
    for (i32 outer = 0; outer < maxOuter; ++outer) {
        // Gradient of the residual: which inactive endmember would most reduce
        // the error if it were allowed in.
        glm::vec3 residual = b;
        for (i32 i = 0; i < k; ++i) {
            residual -= static_cast<f32>(w[i]) * colors[i];
        }

        i32 best = -1;
        f64 bestGrad = 1e-9;
        for (i32 i = 0; i < k; ++i) {
            if (passive[i]) {
                continue;
            }
            const f64 g = static_cast<f64>(glm::dot(colors[i], residual));
            if (g > bestGrad) {
                bestGrad = g;
                best = i;
            }
        }
        if (best < 0) {
            break;  // KKT satisfied
        }
        passive[best] = true;

        for (i32 inner = 0; inner < 2 * k; ++inner) {
            f64 s[MAX_ENDMEMBERS] = {};
            if (!SolvePassive(colors, passive, k, b, s)) {
                passive[best] = false;
                break;
            }

            f64 worst = 0.0;
            bool anyNegative = false;
            for (i32 i = 0; i < k; ++i) {
                if (passive[i] && s[i] <= 0.0) {
                    anyNegative = true;
                    worst = std::min(worst, s[i]);
                }
            }
            if (!anyNegative) {
                for (i32 i = 0; i < k; ++i) {
                    w[i] = passive[i] ? s[i] : 0.0;
                }
                break;
            }

            // Step as far toward s as non-negativity allows, then drop
            // whatever landed on zero back into the active set.
            f64 alpha = 1.0;
            for (i32 i = 0; i < k; ++i) {
                if (passive[i] && s[i] <= 0.0) {
                    const f64 denom = w[i] - s[i];
                    if (denom > 1e-12) {
                        alpha = std::min(alpha, w[i] / denom);
                    } else {
                        alpha = 0.0;
                    }
                }
            }
            for (i32 i = 0; i < k; ++i) {
                w[i] = w[i] + alpha * (s[i] - w[i]);
                if (passive[i] && w[i] <= 1e-12) {
                    w[i] = 0.0;
                    passive[i] = false;
                }
            }
        }
    }

    for (i32 i = 0; i < k; ++i) {
        wOut[i] = static_cast<f32>(std::max(0.0, w[i]));
    }
}

}  // namespace quantiloom
