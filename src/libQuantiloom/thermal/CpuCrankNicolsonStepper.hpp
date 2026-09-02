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
 *   -k dT/dx = alpha_s (E_sun v_i max(cos theta, 0) + E_sun R_i + E_diff G_i)
 *              + eps sigma (sum_j F_ij T_j^4 + s_i T_sky^4 - T_i^4)
 *              + h (T_air - T_i)
 *              - f_wet (h / c_p) L_v (q_sat(T_i) - RH q_sat(T_air))
 *
 * where R_i and G_i are the baked short-wave gains -- sunlight and skylight
 * that reached the element off a neighbour, which the direct term cannot see.
 *
 * The radiative term uses the PREVIOUS step's temperatures for every T on the
 * right, including the element's own. That is the linearisation Aguerre et al.
 * describe, and it is why the timestep matters: it is accurate while the
 * surface does not move far within one step, which for a masonry wall at a
 * minute per step it does not, and for a thin metal sheet in direct sun it
 * might. The scale to compare against is rho c d / (h + 4 eps sigma T^3),
 * which the solver logs.
 *
 * The latent term does not get that treatment. Its slope in T is several times
 * the radiative one -- saturation humidity roughly doubles every eleven
 * degrees -- so it is linearised into the matrix beside the convection
 * instead, half implicit as Crank-Nicolson wants. Left explicit it oscillates
 * at the timesteps this runs at.
 *
 * The back face is adiabatic or held, per material.
 */
class CpuCrankNicolsonStepper final : public IThermalStepper {
public:
    CpuCrankNicolsonStepper() = default;
    explicit CpuCrankNicolsonStepper(const ConvectionLaw& law) : m_convection(law) {}

    void Step(ThermalState& state, const Vector<ThermalElement>& elements,
              const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
              const ThermalForcing& forcing, f64 dt_s,
              const ShortwaveSample& shortwave) override;

    /// Which correlation supplies h when the forcing does not. Held here
    /// rather than in the forcing because it says how the balance is modelled
    /// rather than what the weather is doing, and because it is fixed for a
    /// whole run.
    void SetConvection(const ConvectionLaw& law) { m_convection = law; }
    [[nodiscard]] ConvectionLaw Convection() const override { return m_convection; }
    [[nodiscard]] bool CarriesLateralConduction() const override { return true; }

    /// Answered from the same EvaluateSurfaceBalance the step uses, so what a
    /// probe shows is what the trajectory was built from rather than a second
    /// opinion about it.
    [[nodiscard]] bool SurfaceFluxesAt(const ThermalState& state,
                                       const Vector<ThermalElement>& elements,
                                       const Vector<ThermalMaterial>& materials,
                                       const ExchangeGeometry& exchange,
                                       const ThermalForcing& forcing,
                                       const ShortwaveSample& shortwave, u32 element,
                                       SurfaceFluxes& out) const override;

    /// Also reachable without an instance, because a caller has to name the
    /// stepper for the solve cache key before it has decided to build one.
    static constexpr const char* kName = "CPU Crank-Nicolson";
    [[nodiscard]] const char* Name() const override { return kName; }

    /// Shortest time constant across the participating materials, in seconds:
    /// rho c d / (h + 4 eps sigma T^3 + latent), the scale a timestep should stay under
    /// for the explicit radiative coupling to hold. Reported rather than
    /// enforced -- a step twice this is inaccurate rather than unstable, and
    /// which one matters is the caller's judgement.
    ///
    /// @param law             which correlation supplies h. Under a wind or
    ///                        stability law the material's own coefficient is
    ///                        not what the run will use, and the estimate
    ///                        takes the larger of the two: too small an h
    ///                        makes this advisory quieter than it should be.
    /// @param windSpeed_m_s   the fastest wind the forcing reaches, since that
    ///                        is where the coefficient peaks
    [[nodiscard]] static f64 ShortestTimeConstantSeconds(
        const Vector<ThermalElement>& elements, const Vector<ThermalMaterial>& materials,
        f64 referenceTemperature_K, const ConvectionLaw& law = {},
        f64 windSpeed_m_s = 0.0);

private:
    ConvectionLaw m_convection;
};

}  // namespace quantiloom::thermal
