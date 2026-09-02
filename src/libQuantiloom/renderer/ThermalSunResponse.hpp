/**
 * @file ThermalSunResponse.hpp
 * @brief The buffer that lets a shadow be finer than the triangle it falls on
 *
 * The solver runs on triangles. `BuildThermalMesh` makes one element per
 * triangle, `RunSunVisibility` decides from the element's centroid whether the
 * sun reaches it, and the closest-hit shader reads one temperature per
 * `PrimitiveIndex()`. So a thermal shadow can only have edges where the mesh
 * has edges: on a 120 m desert ground tessellated 201 x 201, that is a 0.6 m
 * triangle, and a sphere 0.7 m across casts a shadow shaped like one triangle.
 *
 * The physics is not what is coarse. Dry sand diffuses heat about 3 cm in an
 * hour, and the model gives each element an independent one-dimensional column
 * with no lateral conduction at all -- so the temperature field it describes
 * has a shadow edge as sharp as the geometry's. Only the discretisation is
 * coarse.
 *
 * What this buffer carries is the first-order fix. Beside each element's
 * temperature it ships dT/dv -- how far that temperature would move per unit
 * of its own sun visibility, taken from the trajectory's tangent rather than
 * from a steady-state formula -- and v_element, the visibility the solve
 * actually used. The shader traces its own sun ray and evaluates
 *
 *     T(x) = T_element + (v(x) - v_element) * dT/dv
 *
 * which is the same field at the element's mean and resolves the shadow at
 * whatever resolution the ray tracer has. Where nothing is shadowed, or where
 * the sun is down, dT/dv is zero and the correction and its ray both vanish.
 *
 * The layout is one float4 per record with a header at index 0, so the whole
 * thing is one binding and needs no element count: see MakeThermalSunResponse.
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>

namespace quantiloom::rendercore {

/**
 * @brief One record of binding 26
 *
 * Index 0 is the header: `a` is the sun direction the solve used at the
 * rendered instant (from surface toward the sun) and `b` is 0 when the buffer
 * is inert and `1 + M` otherwise, where M is how many past sun columns the
 * buffer also carries. The direction is carried rather than read from
 * LightingParams because the forcing CSV owns it and it need not match
 * `[lighting] sun_direction`.
 *
 * Records 1..M, when M > 0, are those past columns: `a` is where the sun was
 * for each, in the same convention.
 *
 * After them the elements start, `1 + M` records each, so element i begins at
 * `(1 + M) * (1 + thermalElementBase + PrimitiveIndex())`. The first of its
 * records is the present sun -- `a.x` the sensitivity not attributed to any
 * carried column, `a.y` the visibility the solve used -- and record j is
 * column j's own (dT/dv_j, v_j). With M = 0 that formula is `1 + element` and
 * the layout is byte for byte the one that existed before columns did.
 */
struct ThermalSunResponseGpu {
    glm::vec3 a{0.0f};
    f32 b = 0.0f;
};
static_assert(sizeof(ThermalSunResponseGpu) == 16);

/**
 * @brief Pack the solver's per-element sun response into that layout
 *
 * Returns a single zeroed record -- header w = 0, correction off -- when the
 * two per-element arrays are missing or disagree about their length. The
 * descriptor has to be valid whether or not a solve ran, and one inert record
 * is cheaper than a buffer of zeros nobody reads.
 *
 * The present-sun record carries `sunSensitivity_K` MINUS what the carried
 * columns hold, because the shader adds the terms rather than choosing between
 * them: the sum over the records is the whole day's sensitivity either way,
 * and the split decides only how much of it is traced at its own hour rather
 * than assumed to look like now.
 *
 * @param lagSensitivity_K  slot-major, M * elements, or empty for none
 * @param lagVisibility     slot-major, the same size
 * @param lagDirection      M entries; a slot whose direction is zero is dropped
 */
[[nodiscard]] Vector<ThermalSunResponseGpu> MakeThermalSunResponse(
    const Vector<f32>& sunSensitivity_K, const Vector<f32>& sunVisibility,
    const glm::vec3& sunDirection, const Vector<f32>& lagSensitivity_K = {},
    const Vector<f32>& lagVisibility = {}, const Vector<glm::vec3>& lagDirection = {});

}  // namespace quantiloom::rendercore
