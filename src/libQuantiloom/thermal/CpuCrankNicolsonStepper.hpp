/**
 * @file CpuCrankNicolsonStepper.hpp
 * @brief One-dimensional conduction through each element, on the CPU
 */

#pragma once

#include "thermal/ThermalMesh.hpp"
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
 *              + eps_i (H_i - sigma T_i^4)
 *              + h (T_air - T_i)
 *              - f_wet (h / c_p) L_v (q_sat(T_i) - RH q_sat(T_air))
 *
 * where R_i and G_i are the baked short-wave gains -- sunlight and skylight
 * that reached the element off a neighbour, which the direct term cannot see.
 *
 * The long wave is diffuse-grey exchange. H_i is the irradiance the element's
 * hemisphere sends it,
 *
 *   H_i = sum_j F_ij J_j + s_i sigma T_sky^4,   J_j = eps_j sigma T_j^4 + (1 - eps_j) H_j
 *
 * J_j the neighbour's radiosity: what it emits plus what it reflects. Of H_i
 * the element absorbs eps_i H_i, by Kirchhoff, and it emits eps_i sigma T_i^4.
 * Treating every neighbour as a blackbody instead -- J_j = sigma T_j^4 -- would
 * over-state a neighbour's emission by 1/eps_j and drop its reflection, which
 * for a polished metal is twenty times the wrong number.
 *
 * The radiosity is reached by lagging the reflected part: J_j takes H_j from
 * the previous step (ThermalState::longwaveIrradiance_W_m2), so every step is
 * one Jacobi sweep of the radiosity system started from the last one. The
 * sweep contracts by (1 - eps) per bounce and the temperatures it reads move
 * slowly against it, so it tracks the fixed point to well within what the
 * linearisation below already costs.
 *
 * The neighbours' radiosity uses the PREVIOUS step's temperatures. The
 * element's own emission does not: it is Newton-linearised and split half
 * implicit, like the evaporation. That is accurate while the surface does not
 * move far within one step, which for a masonry wall at a minute per step it
 * does not, and for a thin metal sheet in direct sun it might. The scale to
 * compare against is rho c d / (h + 4 eps sigma T^3), which the solver logs.
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

    /// The two faces of each thin shell, from ThermalMesh::shellPartner.
    ///
    /// A pair shares one column: the lower index owns it and is stepped with a
    /// full surface balance at BOTH ends, the higher index is not stepped at
    /// all, and the owner's back-node temperature is written into the
    /// partner's surface slot at the end of the step. Every consumer -- the
    /// radiative exchange, the render's temperature buffer, a dump, a probe --
    /// then reads the partner's real temperature without knowing a shell is
    /// involved.
    ///
    /// Empty is every scene that declares no shells, and the rows below are
    /// then exactly the rows they were.
    void SetShellPartners(Vector<u32> partners) { m_shellPartner = std::move(partners); }
    [[nodiscard]] bool CarriesShells() const override { return true; }
    [[nodiscard]] bool CarriesInteriorSources() const override { return true; }
    [[nodiscard]] ConvectionLaw Convection() const override { return m_convection; }
    [[nodiscard]] bool CarriesLateralConduction() const override { return true; }
    [[nodiscard]] bool CarriesLagSensitivity() const override { return true; }
    [[nodiscard]] bool CarriesParameterSensitivity() const override { return true; }

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
    /// rho c d / (h + 4 eps sigma T^3 + latent): how fast the slowest-to-react
    /// mode of the thinnest surface relaxes. Every exchange at the face is
    /// half implicit, so a longer step is stable; it under-resolves that
    /// surface's transient (and Crank-Nicolson rings on it for a while before
    /// settling), which is why this is reported rather than enforced.
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
    /// Storage whose size depends only on the largest step this instance has
    /// seen. A timeline calls Step thousands of times with the same shape, so
    /// retaining these capacities removes allocator work from every thermal
    /// timestep while keeping the numerical loop and its order unchanged.
    struct Workspace {
        Vector<f64> surfacePrevious;
        Vector<f64> radiosity;
        Vector<f64> irradiance;
        Vector<f64> lateralPrevious;
        Vector<f64> lateralPreviousSensitivity;
        Vector<f64> lateralPreviousLag;
        Vector<f64> lateralPreviousParameter;

        Vector<f64> lower;
        Vector<f64> diag;
        Vector<f64> upper;
        Vector<f64> rhs;
        Vector<f64> rhsTangent;
        Vector<f64> rhsLag;
        Vector<f64> rhsParameter;

        Vector<f64> lagWeight;
        Vector<f64> lateralRate;
        Vector<f64> lateralRateTangent;
        Vector<f64> lateralRateLag;
        Vector<f64> lateralRateParameter;

        Vector<std::span<f64>> rightHandSides;
        Vector<std::span<f64>> parameterSides;
    };

    ConvectionLaw m_convection;
    /// Element -> the triangle on the other side of the same shell, or
    /// ThermalMesh::kNoShellPartner. Empty when nothing is a shell.
    Vector<u32> m_shellPartner;
    Workspace m_workspace;
};

}  // namespace quantiloom::thermal
