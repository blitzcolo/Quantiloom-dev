/**
 * @file NNLS.hpp
 * @brief Non-negative least squares for small endmember mixtures
 *
 * Internal. Solves argmin ||A w - b||, w >= 0, for A with at most four columns
 * and exactly three rows -- the shape that comes out of unmixing an RGB texel
 * against K <= 4 endmember colours (SpectralUnmixer).
 *
 * Non-negativity is not a nicety here: a negative weight would mean a surface
 * containing a negative amount of a material, and the mixture it feeds
 * (rho = sum w_i rho_i) would leave the physical range wherever it happened.
 */

#pragma once

#include "core/Types.hpp"
#include "scene/Material.hpp"

#include <glm/glm.hpp>

namespace quantiloom {

// Maximum endmembers in one mixture, from the material model that defines it
// (scene/Material.hpp). Aliased rather than redeclared: two constants that
// must agree are one constant with extra steps.
inline constexpr i32 MAX_ENDMEMBERS = Material::MAX_ENDMEMBERS;

// Ridge term added to the normal equations. With k = 4 the system is
// underdetermined (three equations, four unknowns) and the unregularised
// solution is one arbitrary point on a line of equally good fits, which makes
// neighbouring texels pick wildly different mixtures for near-identical
// colours -- visible as noise in the weight map. This picks the
// smallest-norm fit among them, which is stable and is also the least
// committal reading of an ambiguous colour.
inline constexpr f64 NNLS_RIDGE = 1e-4;

/**
 * @brief Solve argmin ||sum_i w_i c_i - b||^2 subject to w >= 0
 *
 * @param colors   endmember colours in linear sRGB, k entries used
 * @param k        endmember count, clamped to [1, MAX_ENDMEMBERS]
 * @param b        target colour in linear sRGB
 * @param wOut     receives MAX_ENDMEMBERS weights; unused slots are zero
 *
 * Lawson-Hanson active set. Weights are NOT normalised to sum to one: a texel
 * darker than every endmember is a legitimately dimmer patch of the same
 * material, and forcing the sum would erase exactly the brightness variation
 * this exists to recover. Callers clamp the reconstructed reflectance instead.
 */
void SolveNNLS(const glm::vec3* colors, i32 k, const glm::vec3& b, f32* wOut);

}  // namespace quantiloom
