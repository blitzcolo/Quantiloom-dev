/**
 * @file ShortwaveGains.hpp
 * @brief Sunlight that arrived off something else
 *
 * The direct term answers "is this element in the sun", and for a lone plate
 * under an open sky that is the whole of the short wave. It is not the whole
 * of it anywhere a second surface exists. A north-facing wall is never in the
 * sun and is nonetheless warm all afternoon, because the road in front of it
 * is bright and Lambertian; a courtyard floor gets the sky's diffuse light
 * twice, once from above and once off the walls around it.
 *
 * Both paths are gathered with the same cosine-weighted view factors the
 * long-wave exchange already uses -- radiosity and radiative exchange are the
 * same integral over the same hemisphere, one in the visible and one in the
 * infrared. So this costs a matrix pass, not a second precompute.
 *
 * Both are baked rather than stepped. The reflected gain per unit direct
 * irradiance depends only on where the sun is, and the sun is only at K places
 * in the table; the diffuse gain does not depend on the sun at all. Neither
 * depends on temperature, which is what makes them constants of the trajectory
 * instead of terms in it -- a step reads them and pays nothing.
 */

#pragma once

#include "thermal/ThermalTypes.hpp"

namespace quantiloom::thermal {

/**
 * @brief Fill @p table's reflectedGain and diffuseGain from the geometry
 *
 * One bounce, no more: the light an element sends on has already been counted
 * once, and a second bounce off surfaces that absorb 70% of what hits them
 * moves a temperature by less than the view factors are accurate to.
 *
 * Every element reflects, including the ones the solver does not step -- a
 * wall does not stop lighting the ground because nobody asked for its
 * temperature. Their absorptivity is whatever the config gave them, default
 * included.
 *
 * @param exchange   view factors and sky fractions, as the long wave uses them
 * @param elements   surface elements, indexed as the exchange rows are
 * @param materials  indexed by ThermalElement::materialId
 * @param table      sun visibility per column; the gains are written into it.
 *                   Needs sampleDirection to bake the reflected part -- without
 *                   it only the diffuse gain is written.
 */
void BakeShortwaveGains(const ExchangeGeometry& exchange,
                        const Vector<ThermalElement>& elements,
                        const Vector<ThermalMaterial>& materials,
                        SunVisibilityTable& table);

}  // namespace quantiloom::thermal
