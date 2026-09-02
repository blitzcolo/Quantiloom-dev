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

    [[nodiscard]] u32 LastStepCount() const { return m_lastStepCount; }
    [[nodiscard]] u32 CheckpointCount() const {
        return static_cast<u32>(m_checkpoints.size());
    }
    [[nodiscard]] usize CheckpointBytes() const;
    [[nodiscard]] u32 ParticipatingElements() const { return m_participatingElements; }
    [[nodiscard]] f64 ShortestTimeConstant_s() const { return m_shortestTau; }

private:
    i64 GridIndex(f64 time_h) const;
    f64 GridTime(i64 k) const;
    void StepRange(ThermalState& state, i64 from, i64 to);

    Desc m_desc;

    const Vector<ThermalElement>& m_elements;
    const Vector<ThermalMaterial>& m_materials;
    const ExchangeGeometry& m_exchange;
    const SunVisibilityTable& m_sunTable;
    const Vector<std::pair<f64, ThermalForcing>>& m_forcingSeries;
    ThermalForcing m_constantForcing;
    IThermalStepper& m_stepper;

    std::map<i64, ThermalState> m_checkpoints;
    i64 m_checkpointStride;  // in grid steps
    ThermalState m_scratch;
    u32 m_lastStepCount = 0;
    u32 m_participatingElements = 0;
    f64 m_shortestTau = 0.0;
};

}  // namespace quantiloom::thermal
