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
 * rendered instant (from surface toward the sun) and `b` is 1 when the rest of
 * the buffer is real. The direction is carried rather than read from
 * LightingParams because the forcing CSV owns it and it need not match
 * `[lighting] sun_direction`.
 *
 * Index `1 + thermalElementBase + PrimitiveIndex()` is an element: `a.x` is
 * dT/dv in kelvin per unit visibility, `a.y` is the visibility the solve used,
 * and the rest is padding. The +1 is confined to this buffer -- the element
 * bases themselves stay exactly what the exchange precompute uses.
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
 */
[[nodiscard]] Vector<ThermalSunResponseGpu> MakeThermalSunResponse(
    const Vector<f32>& sunSensitivity_K, const Vector<f32>& sunVisibility,
    const glm::vec3& sunDirection);

}  // namespace quantiloom::rendercore
