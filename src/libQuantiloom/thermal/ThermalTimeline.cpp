/**
 * @file ThermalTimeline.cpp
 * @brief Fixed-grid trajectory with checkpointing
 */

#include "thermal/ThermalTimeline.hpp"

#include "core/Log.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::thermal {

namespace {

/// The sun table interpolated at one time, into caller-owned scratch. The
/// reflected column comes along on the same indices as the visibility it was
/// baked from, so the two never disagree about where the sun is. With no table
/// this falls back to the exchange's single sun column, and with neither to
/// full sun -- which is what a scene with no shadowing precompute is.
void SampleShortwaveAt(const SunVisibilityTable& table, const ExchangeGeometry& exchange,
                       const f64 time_h, const usize n, Vector<f32>& visibility,
                       Vector<f32>& reflected, ShortwaveSample* columns = nullptr) {
    // The visibility is the same interpolation the renderer asks for by name
    // when it needs v_element, so it lives in one place -- the two disagreeing
    // would put the shading correction's reference point somewhere the solve
    // never was.
    visibility = SampleSunVisibilityAt(table, exchange, time_h, n);
    reflected.clear();

    if (table.SampleCount() > 0 && table.ElementCount() == n) {
        usize a = 0;
        usize b = 0;
        f64 blend = 0.0;
        table.SampleIndices(time_h, a, b, blend);
        const f32 bf = static_cast<f32>(blend);

        if (columns != nullptr) {
            columns->columnA = a;
            columns->columnB = b;
            columns->columnBlend = blend;
            columns->columnsKnown = true;
        }

        const f32* reflectedA = table.ReflectedColumn(a);
        const f32* reflectedB = table.ReflectedColumn(b);
        if (reflectedA != nullptr && reflectedB != nullptr) {
            reflected.resize(n);
            for (usize e = 0; e < n; ++e) {
                reflected[e] = reflectedA[e] + bf * (reflectedB[e] - reflectedA[e]);
            }
        }
    }
}

void RelaxToSteadyState(ThermalState& state, const Vector<ThermalElement>& elements,
                        const Vector<ThermalMaterial>& materials,
                        const ExchangeGeometry& exchange,
                        const ShortwaveSample& shortwave,
                        const ThermalForcing& forcing, IThermalStepper& stepper) {
    constexpr f64 kRelaxStep_s = 3600.0;
    constexpr i32 kMaxIterations = 200;
    constexpr f64 kSettled_K = 0.01;
    constexpr i32 kStaleLimit = 10;

    Vector<f64> previous(elements.size());
    f64 bestMove = std::numeric_limits<f64>::max();
    i32 staleTicks = 0;

    for (i32 iteration = 0; iteration < kMaxIterations; ++iteration) {
        for (usize e = 0; e < elements.size(); ++e) {
            previous[e] = state.Surface(e);
        }
        stepper.Step(state, elements, materials, exchange, forcing,
                     kRelaxStep_s, shortwave);

        f64 largestMove = 0.0;
        for (usize e = 0; e < elements.size(); ++e) {
            largestMove = std::max(largestMove, std::abs(state.Surface(e) - previous[e]));
        }
        if (largestMove < kSettled_K) {
            QL_LOG_INFO("  Thermal: steady state after {} relaxation steps", iteration + 1);
            return;
        }
        if (largestMove < bestMove) {
            bestMove = largestMove;
            staleTicks = 0;
        } else {
            ++staleTicks;
            if (staleTicks >= kStaleLimit) {
                QL_LOG_WARN("  Thermal: relaxation stalled at {:.4f} K after {} steps",
                            bestMove, iteration + 1);
                return;
            }
        }
    }
    QL_LOG_WARN("  Thermal: steady state did not settle in {} steps; starting from "
                "where it got to", kMaxIterations);
}

}  // namespace

ThermalTimeline::ThermalTimeline(const Desc& desc, const ThermalGeometrySchedule& schedule,
                                 const Vector<ThermalMaterial>& materials,
                                 const Vector<std::pair<f64, ThermalForcing>>& forcingSeries,
                                 const ThermalForcing& constantForcing,
                                 IThermalStepper& stepper)
    : m_desc(desc)
    , m_schedule(&schedule)
    , m_materials(materials)
    , m_forcingSeries(forcingSeries)
    , m_constantForcing(constantForcing)
    , m_stepper(stepper) {
    Initialise();
}

ThermalTimeline::ThermalTimeline(const Desc& desc, const Vector<ThermalElement>& elements,
                                 const Vector<ThermalMaterial>& materials,
                                 const ExchangeGeometry& exchange,
                                 const SunVisibilityTable& sunTable,
                                 const Vector<std::pair<f64, ThermalForcing>>& forcingSeries,
                                 const ThermalForcing& constantForcing,
                                 IThermalStepper& stepper)
    : m_desc(desc)
    , m_ownedSchedule(std::make_unique<ThermalGeometrySchedule>(
          ThermalGeometrySchedule::Single(exchange, sunTable, elements)))
    , m_schedule(m_ownedSchedule.get())
    , m_materials(materials)
    , m_forcingSeries(forcingSeries)
    , m_constantForcing(constantForcing)
    , m_stepper(stepper) {
    Initialise();
}

void ThermalTimeline::Initialise() {
    const f64 strideSteps =
        (m_desc.checkpointStride_h * 3600.0) / std::max(m_desc.timestep_s, 1e-6);
    m_checkpointStride = std::max(static_cast<i64>(std::round(strideSteps)), i64{1});

    // Where each epoch begins on the step grid. A boundary that falls inside a
    // step belongs to the step AFTER it: a step is integrated with one
    // geometry, and the one it spends most of itself in is the one it should
    // be -- but ceil is the choice that keeps a boundary from being applied
    // before it happens, which matters more than half a step of accuracy.
    const ThermalGeometrySchedule& schedule = *m_schedule;
    m_epochBoundaryStep.assign(schedule.Count(), 0);
    for (usize e = 1; e < schedule.Count(); ++e) {
        const f64 elapsed_s = (schedule.epochs[e].from_h - m_desc.startTime_h) * 3600.0;
        m_epochBoundaryStep[e] =
            std::max(static_cast<i64>(std::ceil(elapsed_s / m_desc.timestep_s)), i64{0});
    }

    // Everything below describes the world the trajectory STARTS in, which is
    // epoch zero: it reaches back forever, so a query before the timeline
    // begins is a query about it.
    const Vector<ThermalElement>& elements = schedule.epochs.front().elements;
    const ExchangeGeometry& exchange = schedule.epochs.front().exchange;
    const SunVisibilityTable& sunTable = schedule.epochs.front().sunTable;

    for (const ThermalElement& el : elements) {
        if (el.area_m2 > 0.0f && el.materialId < m_materials.size() &&
            m_materials[el.materialId].ParticipatesInSolve()) {
            ++m_participatingElements;
        }
    }
    m_shortestTau = CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
        elements, m_materials, m_desc.initialTemperature_K);

    // Initial state
    ThermalState initial;
    initial.nodeCount = std::max(2u, m_desc.nodeCount);
    initial.temperature_K.assign(elements.size() * initial.nodeCount,
                                 m_desc.initialTemperature_K);
    // Zero, and it means what it says: at t = 0 nothing that has happened yet
    // depends on the sun, so the trajectory's derivative with respect to sun
    // visibility starts at nothing and accumulates. Sizing it is what turns
    // the tangent on for every stepper downstream.
    if (m_desc.carrySunSensitivity) {
        initial.sunSensitivity_K.assign(initial.temperature_K.size(), 0.0);
        // One tangent per tracked column, and no column tracked yet: at t = 0
        // the whole of the answer is in the whole-day tangent, which is where
        // the shading pass looks when a column is not tracked.
        if (m_desc.sunMemoryLags > 0 && sunTable.SampleCount() > 1) {
            initial.lagColumn.assign(m_desc.sunMemoryLags, ThermalState::kNoLagColumn);
            initial.lagSensitivity_K.assign(
                initial.temperature_K.size() * m_desc.sunMemoryLags, 0.0);
        }
    }
    // Independent of the sun's tangent: a fit for a material property wants
    // these whether or not a shadow is being resolved.
    if (!m_desc.parameters.empty()) {
        initial.parameters = m_desc.parameters;
        initial.parameterSensitivity.assign(
            initial.temperature_K.size() * m_desc.parameters.size(), 0.0);
    }

    if (m_desc.initial == InitialCondition::Steady) {
        const ThermalForcing startForcing =
            SampleForcing(m_forcingSeries, m_desc.startTime_h, m_constantForcing);

        Vector<f32> startSunVis;
        Vector<f32> startReflected;
        SampleShortwaveAt(sunTable, exchange, m_desc.startTime_h, elements.size(),
                          startSunVis, startReflected);

        RelaxToSteadyState(initial, elements, m_materials, exchange,
                           {startSunVis, startReflected, sunTable.diffuseGain},
                           startForcing, m_stepper);
    }

    m_checkpoints[0] = std::move(initial);
}

usize ThermalTimeline::EpochForStep(const i64 k) const {
    usize found = 0;
    for (usize e = 1; e < m_epochBoundaryStep.size(); ++e) {
        if (m_epochBoundaryStep[e] <= k) {
            found = e;
        } else {
            break;
        }
    }
    return found;
}

const ThermalState& ThermalTimeline::StateAt(const f64 time_h) {
    const f64 clamped = std::max(time_h, m_desc.startTime_h);
    const i64 target = GridIndex(clamped);

    // Find nearest checkpoint at or before target
    auto it = m_checkpoints.upper_bound(target);
    if (it != m_checkpoints.begin()) --it;
    // it now points to the largest key <= target

    const i64 from = it->first;
    if (from == target) {
        m_lastStepCount = 0;
        return it->second;
    }

    // Step from checkpoint to target, creating new checkpoints along the way
    m_scratch = it->second;
    StepRange(m_scratch, from, target);

    // If target sits on a checkpoint boundary, store it
    if (target % m_checkpointStride == 0) {
        m_checkpoints[target] = m_scratch;
        return m_checkpoints[target];
    }

    // Check if the partial-step time doesn't align perfectly with the grid
    const f64 gridTime = GridTime(target);
    const f64 remainder_s = (clamped - gridTime) * 3600.0;
    if (remainder_s > 0.01) {
        // Off-grid: do a partial step into scratch (discarded next call)
        const f64 t_mid = gridTime + 0.5 * remainder_s / 3600.0;
        const ThermalGeometryEpoch& epoch = m_schedule->At(t_mid);
        const ThermalForcing forcing =
            SampleForcing(m_forcingSeries, t_mid, m_constantForcing);
        Vector<f32> sunVis;
        Vector<f32> reflected;
        ShortwaveSample sample;
        SampleShortwaveAt(epoch.sunTable, epoch.exchange, t_mid, epoch.elements.size(),
                          sunVis, reflected, &sample);
        sample.sunVisibility = sunVis;
        sample.reflectedGain = reflected;
        sample.diffuseGain = epoch.sunTable.diffuseGain;
        m_stepper.Step(m_scratch, epoch.elements, m_materials, epoch.exchange, forcing,
                       remainder_s, sample);
        ++m_lastStepCount;
    }

    return m_scratch;
}

bool ThermalTimeline::SurfaceFluxesAt(const f64 time_h, const u32 element,
                                      SurfaceFluxes& out,
                                      const IThermalStepper& decomposer) {
    const ThermalGeometryEpoch& epoch = m_schedule->At(time_h);
    if (element >= epoch.elements.size()) return false;

    const ThermalState& state = StateAt(time_h);
    const ThermalForcing forcing = SampleForcing(m_forcingSeries, time_h, m_constantForcing);

    Vector<f32> sunVis;
    Vector<f32> reflected;
    ShortwaveSample sample;
    SampleShortwaveAt(epoch.sunTable, epoch.exchange, time_h, epoch.elements.size(), sunVis,
                      reflected, &sample);
    sample.sunVisibility = sunVis;
    sample.reflectedGain = reflected;
    sample.diffuseGain = epoch.sunTable.diffuseGain;

    return decomposer.SurfaceFluxesAt(state, epoch.elements, m_materials, epoch.exchange,
                                      forcing, sample, element, out);
}

usize ThermalTimeline::CheckpointBytes() const {
    usize total = 0;
    for (const auto& [k, state] : m_checkpoints) {
        total += state.ByteSize();
    }
    return total;
}

i64 ThermalTimeline::GridIndex(const f64 time_h) const {
    const f64 elapsed_s = (time_h - m_desc.startTime_h) * 3600.0;
    return std::max(static_cast<i64>(std::floor(elapsed_s / m_desc.timestep_s)), i64{0});
}

f64 ThermalTimeline::GridTime(const i64 k) const {
    return m_desc.startTime_h + static_cast<f64>(k) * m_desc.timestep_s / 3600.0;
}

void ThermalTimeline::StepRange(ThermalState& state, const i64 from, const i64 to) {
    if (from >= to) {
        m_lastStepCount = 0;
        return;
    }

    Vector<ThermalBatchStep> batch;
    batch.reserve(static_cast<usize>(std::min(to - from, i64{512})));

    i64 k = from;
    m_lastStepCount = 0;

    while (k < to) {
        // Build a batch up to the next checkpoint boundary or to `to`
        i64 batchEnd = to;
        // Next checkpoint after k
        const i64 nextCp = ((k / m_checkpointStride) + 1) * m_checkpointStride;
        if (nextCp < batchEnd) batchEnd = nextCp;

        // ...and never across a geometry change. A batch is stepped against
        // one exchange and one sun table, so an epoch boundary inside it would
        // integrate part of the span with the wrong world.
        const usize epochIndex = EpochForStep(k);
        for (usize e = epochIndex + 1; e < m_epochBoundaryStep.size(); ++e) {
            if (m_epochBoundaryStep[e] > k) {
                if (m_epochBoundaryStep[e] < batchEnd) batchEnd = m_epochBoundaryStep[e];
                break;
            }
        }
        const ThermalGeometryEpoch& epoch = m_schedule->epochs[epochIndex];

        batch.clear();
        for (i64 step = k; step < batchEnd; ++step) {
            const f64 t_mid = GridTime(step) + 0.5 * m_desc.timestep_s / 3600.0;
            ThermalBatchStep bs;
            bs.forcing = SampleForcing(m_forcingSeries, t_mid, m_constantForcing);
            bs.dt_s = m_desc.timestep_s;
            // The column indices are into THIS epoch's table, which may be a
            // subset of the forcing file's rows -- so they are taken from the
            // epoch rather than from any shared one.
            if (epoch.sunTable.SampleCount() > 0) {
                epoch.sunTable.SampleIndices(t_mid, bs.sunSampleA, bs.sunSampleB, bs.sunBlend);
            }
            batch.push_back(bs);
        }

        m_stepper.StepMany(state, epoch.elements, m_materials, epoch.exchange,
                           epoch.sunTable, batch);
        m_lastStepCount += static_cast<u32>(batch.size());
        k = batchEnd;

        // Store checkpoint if we landed on one and it's not already stored
        if (k % m_checkpointStride == 0 && k < to) {
            if (m_checkpoints.find(k) == m_checkpoints.end()) {
                m_checkpoints[k] = state;
            }
        }
    }

    // Store checkpoint if target is on a boundary
    if (to % m_checkpointStride == 0) {
        if (m_checkpoints.find(to) == m_checkpoints.end()) {
            m_checkpoints[to] = state;
        }
    }
}

}  // namespace quantiloom::thermal
