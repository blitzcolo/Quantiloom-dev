/**
 * @file CpuCrankNicolsonStepper.hpp
 * @brief One-dimensional conduction through each element, on the CPU
 */

#pragma once

#include "thermal/ThermalStepper.hpp"

namespace quantiloom::thermal {

/**
 * @brief Crank-Nicolson through the slab, explicit in the radiative coupling
 *
 * Each element is a slab of its material, discretised into nodeCount nodes
 * from the exposed face to the back. Within the slab:
 *
 *   rho c dT/dt = k d2T/dx2
 *
 * discretised with the trapezoid in time -- second-order accurate and
 * unconditionally stable, unlike the explicit scheme whose stability limit at
 * the node spacing this uses would be a fraction of a second. The tridiagonal
 * system that produces is solved directly by Thomas elimination, which is
 * exact and O(nodes).
 *
 * At the exposed face:
 *
 *   -k dT/dx = alpha_s E_sun v_i max(cos theta, 0)
 *              + eps sigma (sum_j F_ij T_j^4 + s_i T_sky^4 - T_i^4)
 *              + h (T_air - T_i)
 *
 * The radiative term uses the PREVIOUS step's temperatures for every T on the
 * right, including the element's own. That is the linearisation Aguerre et al.
 * describe, and it is why the timestep matters: it is accurate while the
 * surface does not move far within one step, which for a masonry wall at a
 * minute per step it does not, and for a thin metal sheet in direct sun it
 * might. The scale to compare against is rho c d / (h + 4 eps sigma T^3),
 * which the solver logs.
 *
 * The back face is adiabatic or held, per material.
 */
class CpuCrankNicolsonStepper final : public IThermalStepper {
public:
    void Step(ThermalState& state, const Vector<ThermalElement>& elements,
              const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
              const ThermalForcing& forcing, f64 dt_s,
              std::span<const f32> sunVisibility) override;

    [[nodiscard]] const char* Name() const override { return "CPU Crank-Nicolson"; }

    /// Shortest time constant across the participating materials, in seconds:
    /// rho c d / (h + 4 eps sigma T^3), the scale a timestep should stay under
    /// for the explicit radiative coupling to hold. Reported rather than
    /// enforced -- a step twice this is inaccurate rather than unstable, and
    /// which one matters is the caller's judgement.
    [[nodiscard]] static f64 ShortestTimeConstantSeconds(
        const Vector<ThermalElement>& elements, const Vector<ThermalMaterial>& materials,
        f64 referenceTemperature_K);
};

}  // namespace quantiloom::thermal
