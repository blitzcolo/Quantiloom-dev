/**
 * @file ThermalStepper.hpp
 * @brief Advancing the surface temperatures by one timestep
 *
 * An interface with one implementation, which is usually a mistake. It is here
 * because of where the second one would go: the per-element slabs are
 * independent given the previous step's temperatures, so this is a compute
 * dispatch waiting to happen, and the boundary between "what a step is" and
 * "who runs it" is the only thing that would have to be right for that to
 * drop in.
 *
 * The CPU one is not a placeholder. At the sizes this renders -- ten thousand
 * elements, ten nodes each -- a step is a few hundred thousand flops and runs
 * in under a millisecond, against tens of milliseconds for one sample of the
 * frame it precedes. What would justify the GPU is a scene an order of
 * magnitude larger, and the interface is what keeps that from being a rewrite.
 */

#pragma once

#include "thermal/ThermalTypes.hpp"

#include <span>

namespace quantiloom::thermal {

/**
 * @brief One step of the surface energy balance
 *
 * The radiative term is non-linear -- it goes as T^4, and every element's
 * temperature appears in every other element's balance -- so a step
 * linearises it against the temperatures it was given, exactly as Aguerre et
 * al. do. That is only accurate for a timestep short enough that the surface
 * does not move far within it; the solver's own comment says what "short
 * enough" works out to.
 */
class IThermalStepper {
public:
    virtual ~IThermalStepper() = default;

    /**
     * @brief Advance every element by @p dt seconds, in place
     *
     * @param state      node temperatures, updated
     * @param elements   the surface elements, indexed as the state is
     * @param materials  thermal properties, indexed by ThermalElement::materialId
     * @param exchange   who sees whom (view factors and sky fraction)
     * @param forcing    the outside world for this step
     * @param dt_s       timestep
     * @param shortwave  per-element sun visibility and the baked gains for this
     *                   step, interpolated out of the sun table rather than
     *                   read from the exchange
     */
    virtual void Step(ThermalState& state, const Vector<ThermalElement>& elements,
                      const Vector<ThermalMaterial>& materials,
                      const ExchangeGeometry& exchange, const ThermalForcing& forcing,
                      f64 dt_s, const ShortwaveSample& shortwave) = 0;

    /**
     * @brief Run a batch of steps without per-step host readback
     *
     * The default implementation loops Step with the sun column interpolated
     * from the table into a scratch buffer. A GPU implementation dispatches
     * the whole batch in one submit.
     */
    virtual void StepMany(ThermalState& state, const Vector<ThermalElement>& elements,
                          const Vector<ThermalMaterial>& materials,
                          const ExchangeGeometry& exchange,
                          const SunVisibilityTable& sunTable,
                          std::span<const ThermalBatchStep> steps);

    /// Which convection correlation this stepper evaluates. The default is the
    /// constant one, which is what an implementation that has not been taught
    /// the others honestly reports. The host compares it against what the
    /// config asked for and picks a stepper that matches, rather than letting a
    /// step quietly use a different h -- that would be a wrong trajectory
    /// rather than a slower one.
    [[nodiscard]] virtual ConvectionLaw Convection() const { return {}; }

    /// Whether this stepper evaluates ExchangeGeometry::lateral. False is the
    /// honest answer for an implementation that has not been taught it, and
    /// the host picks one that has rather than letting the rows be ignored.
    [[nodiscard]] virtual bool CarriesLateralConduction() const { return false; }

    /**
     * @brief Decompose one element's surface balance at the state it is in
     *
     * A query rather than a side effect of stepping: it reads the state and
     * changes nothing, so a probe can ask about any element at any replayed
     * hour without the trajectory being aware it was asked. What it must NOT
     * be is a second reading of the balance -- an implementation answers this
     * from the same code its own step uses, or it declines.
     *
     * @return false when this stepper does not decompose its own balance, in
     *         which case @p out is untouched. Declining is the honest answer,
     *         and a caller has to say so rather than show six zeros as if they
     *         were fluxes.
     */
    [[nodiscard]] virtual bool SurfaceFluxesAt(const ThermalState& /*state*/,
                                               const Vector<ThermalElement>& /*elements*/,
                                               const Vector<ThermalMaterial>& /*materials*/,
                                               const ExchangeGeometry& /*exchange*/,
                                               const ThermalForcing& /*forcing*/,
                                               const ShortwaveSample& /*shortwave*/,
                                               u32 /*element*/,
                                               SurfaceFluxes& /*out*/) const {
        return false;
    }

    /// For the log line that says which one ran -- and for the solve cache,
    /// which keys on it: two steppers give answers that differ in the last
    /// bits, so an entry is only valid for the one that produced it. Renaming
    /// one is therefore a cache invalidation, which is harmless; giving two of
    /// them the same name would let their results be served for each other,
    /// which is not.
    [[nodiscard]] virtual const char* Name() const = 0;
};

}  // namespace quantiloom::thermal
