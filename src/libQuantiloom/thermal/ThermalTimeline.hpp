/**
 * @file ThermalTimeline.hpp
 * @brief A fixed-grid trajectory with checkpointing
 *
 * The trajectory is pinned to a regular grid: t_k = startTime_h + k * dt / 3600.
 * The timestep dt does not depend on the end time, so a query at one hour reuses
 * every step that led to a query at an earlier hour, and a checkpoint taken at
 * grid step k is valid for any future query that passes through k.
 *
 * A scrub to a time before the latest checkpoint replays from the nearest one
 * instead of from the start — the cost of dragging the slider backwards is
 * proportional to the checkpoint stride, not to the total time.
 *
 * Steady-state relaxation is computed once at construction and cached as
 * checkpoint 0. It is the most expensive single operation (up to 200 relaxation
 * steps), and it never has to be repeated unless the geometry or materials change.
 */

#pragma once

#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalStepper.hpp"

#include <map>
#include <memory>

namespace quantiloom::thermal {

class ThermalTimeline {
public:
    struct Desc {
        f64 startTime_h = 0.0;
        f64 timestep_s = 60.0;
        f64 checkpointStride_h = 1.0;
        u32 nodeCount = 10;
        InitialCondition initial = InitialCondition::Steady;
        f64 initialTemperature_K = 288.15;
        /// Carry dT/dv beside the temperature, so a shading pass can correct a
        /// per-triangle temperature to a per-pixel one. Doubles the state and
        /// therefore the checkpoints, and adds an elimination pass per step;
        /// on by default because every renderer wants it and only a caller
        /// that just needs bulk temperatures should turn it off.
        bool carrySunSensitivity = true;
        /// How many of the sun's most recent columns to carry a tangent of
        /// their own, beside the whole-day one. Zero is the old behaviour;
        /// each slot costs another state vector, another elimination pass per
        /// step, and lets the shading pass trace one more hour of the shadow's
        /// history instead of assuming it looked like now.
        u32 sunMemoryLags = 0;
        /// Material parameters to carry a tangent of, beside the sun's.
        Vector<ThermalParameter> parameters;
    };

    /**
     * @brief A trajectory over a geometry that changes in steps
     *
     * Every epoch must carry the same number of elements in the same order --
     * rigid motion, so only centroids and normals differ. That is what lets
     * one state vector cross a boundary: the temperature of element i means
     * the same surface on both sides of it.
     *
     * @param schedule  outlives this, like everything else here. Epoch zero's
     *                  geometry is what the steady-state relaxation, the
     *                  participation count and the time-constant estimate all
     *                  use, since they describe the world the trajectory
     *                  starts in.
     */
    ThermalTimeline(const Desc& desc, const ThermalGeometrySchedule& schedule,
                    const Vector<ThermalMaterial>& materials,
                    const Vector<std::pair<f64, ThermalForcing>>& forcingSeries,
                    const ThermalForcing& constantForcing, IThermalStepper& stepper);

    /// One geometry for the whole run.
    ///
    /// Copies @p elements into an owned single-epoch schedule -- the one place
    /// here that copies rather than referencing. Kept because it is what every
    /// caller with nothing moving wants to write, and what the tests do write.
    ThermalTimeline(const Desc& desc, const Vector<ThermalElement>& elements,
                    const Vector<ThermalMaterial>& materials,
                    const ExchangeGeometry& exchange,
                    const SunVisibilityTable& sunTable,
                    const Vector<std::pair<f64, ThermalForcing>>& forcingSeries,
                    const ThermalForcing& constantForcing, IThermalStepper& stepper);

    /// The state at the given time. Steps from the nearest checkpoint, creating
    /// new checkpoints as it goes. Off-grid times use a partial step into a
    /// scratch state that is discarded on the next call.
    const ThermalState& StateAt(f64 time_h);

    /// One element's surface balance at a time, term by term.
    ///
    /// Replayed exactly as StateAt replays; the forcing and the shortwave
    /// gains are assembled here because they are the ones a step at that
    /// instant would have used.
    ///
    /// @param decomposer  which stepper answers. Not necessarily the one that
    ///        built the trajectory: the balance is a pure function of the
    ///        state, so decomposing with one implementation keeps a panel's
    ///        numbers from depending on which stepper a machine happened to
    ///        choose. What is being reported either way is the balance AT the
    ///        state the trajectory reached.
    ///
    /// @return false when the decomposer does not decompose a balance, or the
    ///         element is out of range. `out` is untouched then, so a caller
    ///         cannot mistake a refusal for six zero fluxes.
    [[nodiscard]] bool SurfaceFluxesAt(f64 time_h, u32 element, SurfaceFluxes& out,
                                       const IThermalStepper& decomposer);

    [[nodiscard]] u32 LastStepCount() const { return m_lastStepCount; }
    [[nodiscard]] u32 CheckpointCount() const {
        return static_cast<u32>(m_checkpoints.size());
    }
    [[nodiscard]] usize CheckpointBytes() const;
    [[nodiscard]] u32 ParticipatingElements() const { return m_participatingElements; }
    [[nodiscard]] f64 ShortestTimeConstant_s() const { return m_shortestTau; }

    [[nodiscard]] u32 EpochCount() const {
        return static_cast<u32>(m_schedule->Count());
    }
    [[nodiscard]] u32 EpochAt(f64 time_h) const {
        return static_cast<u32>(m_schedule->EpochAt(time_h));
    }
    /// The geometry in force at @p time_h -- what SurfaceFluxesAt decomposed
    /// against, and what a field extracted at that hour must read its normals
    /// from.
    [[nodiscard]] const ThermalGeometryEpoch& GeometryAt(f64 time_h) const {
        return m_schedule->At(time_h);
    }

private:
    i64 GridIndex(f64 time_h) const;
    f64 GridTime(i64 k) const;
    void StepRange(ThermalState& state, i64 from, i64 to);
    /// Which epoch a grid step belongs to, from the boundaries precomputed at
    /// construction. Asked per batch rather than per step.
    usize EpochForStep(i64 k) const;
    void Initialise();

    Desc m_desc;

    /// Only the single-geometry constructor uses this; the other one binds
    /// straight to the caller's. Declared first so it outlives the reference.
    std::unique_ptr<ThermalGeometrySchedule> m_ownedSchedule;
    const ThermalGeometrySchedule* m_schedule = nullptr;

    const Vector<ThermalMaterial>& m_materials;
    const Vector<std::pair<f64, ThermalForcing>>& m_forcingSeries;
    ThermalForcing m_constantForcing;
    IThermalStepper& m_stepper;

    /// Grid step at which each epoch begins. Entry 0 is always 0; entry e is
    /// ceil((from_h - startTime_h) * 3600 / dt), so a boundary that falls
    /// inside a step belongs to the step after it.
    Vector<i64> m_epochBoundaryStep;

    std::map<i64, ThermalState> m_checkpoints;
    i64 m_checkpointStride = 1;  // in grid steps
    ThermalState m_scratch;
    u32 m_lastStepCount = 0;
    u32 m_participatingElements = 0;
    f64 m_shortestTau = 0.0;
};

}  // namespace quantiloom::thermal
