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

    // Every node, not only the surfaces: a thick slab's face can sit still
    // while its interior is still drifting towards the profile the face will
    // follow, and a start taken there replays that drift into the first hours
    // of the day as if it were weather.
    Vector<f64> previous;
    f64 bestMove = std::numeric_limits<f64>::max();
    f64 lastMove = std::numeric_limits<f64>::max();
    i32 staleTicks = 0;

    for (i32 iteration = 0; iteration < kMaxIterations; ++iteration) {
        previous = state.temperature_K;
        stepper.Step(state, elements, materials, exchange, forcing,
                     kRelaxStep_s, shortwave);

        f64 stepMove = 0.0;
        for (usize i = 0; i < previous.size(); ++i) {
            stepMove = std::max(stepMove, std::abs(state.temperature_K[i] - previous[i]));
        }
        // Judged on the envelope of two consecutive moves, not on one. At an
        // hour a step Crank-Nicolson rings on the face's stiff mode -- an
        // amplification near -0.7 -- on top of the slow decay towards steady,
        // so single moves alternate large and small (1.0, 0.08, 0.75, 0.15,
        // ...) and a valley could read as settled, or as a stall once later
        // valleys came out shallower. The pairwise maximum decreases
        // monotonically while the iteration is converging.
        const f64 largestMove = std::max(stepMove, lastMove);
        lastMove = stepMove;
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
    // ResolveRenderConfig refuses a config with any of these wrong. A host that
    // fills a Desc itself -- the interactive preview's SetParams, a tool --
    // does not pass through it, and everything below divides by the step: a
    // zero, negative or NaN one makes every grid index a NaN cast to an
    // integer and a clock that never advances, a NaN stride or start does the
    // same to the checkpoints and the grid. Such a value is reported and the
    // Desc default stands in for it.
    const Desc defaults;
    if (!(std::isfinite(m_desc.timestep_s) && m_desc.timestep_s > 0.0)) {
        QL_LOG_ERROR("  Thermal: timestep {} s is not a positive number; using {} s",
                     m_desc.timestep_s, defaults.timestep_s);
        m_desc.timestep_s = defaults.timestep_s;
    }
    if (!(std::isfinite(m_desc.checkpointStride_h) && m_desc.checkpointStride_h > 0.0)) {
        QL_LOG_ERROR("  Thermal: checkpoint stride {} h is not a positive number; "
                     "using {} h",
                     m_desc.checkpointStride_h, defaults.checkpointStride_h);
        m_desc.checkpointStride_h = defaults.checkpointStride_h;
    }
    if (!std::isfinite(m_desc.startTime_h)) {
        QL_LOG_ERROR("  Thermal: start time {} h is not a number; using {} h",
                     m_desc.startTime_h, defaults.startTime_h);
        m_desc.startTime_h = defaults.startTime_h;
    }
    if (m_desc.nodeCount < 2) {
        // A one-node slab has no interior to conduct through; the state below
        // was always sized with two, and now the Desc says so too.
        QL_LOG_ERROR("  Thermal: {} layers cannot conduct; using 2", m_desc.nodeCount);
        m_desc.nodeCount = 2;
    }

    const f64 strideSteps = (m_desc.checkpointStride_h * 3600.0) / m_desc.timestep_s;
    m_checkpointStride = std::max(static_cast<i64>(std::round(strideSteps)), i64{1});

    // Where each epoch begins on the step grid. A boundary that falls inside a
    // step belongs to the step AFTER it: a step is integrated with one
    // geometry, and the one it spends most of itself in is the one it should
    // be -- but ceil is the choice that keeps a boundary from being applied
    // before it happens, which matters more than half a step of accuracy.
    const ThermalGeometrySchedule& schedule = *m_schedule;
    m_epochBoundaryStep.assign(schedule.Count(), 0);
    // A geometry with no sun visibility of its own is stepped as shadowed (see
    // SampleSunVisibilityAt). Said once here, where it is decided, rather than
    // on every step that reads it.
    for (usize e = 0; e < schedule.Count(); ++e) {
        const ThermalGeometryEpoch& epoch = schedule.epochs[e];
        const usize n = epoch.elements.size();
        const bool tableFits =
            epoch.sunTable.SampleCount() > 0 && epoch.sunTable.ElementCount() == n;
        if (n > 0 && !tableFits && epoch.exchange.sunVisibility.size() != n) {
            QL_LOG_WARN("  Thermal: epoch {} has no sun visibility for its {} elements "
                        "(table {} x {}, exchange {}); the direct beam is treated as "
                        "shadowed",
                        e, n, epoch.sunTable.SampleCount(), epoch.sunTable.ElementCount(),
                        epoch.exchange.sunVisibility.size());
        }
    }
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
    const f64 gridTime = GridTime(target);
    const f64 remainder_s = (clamped - gridTime) * 3600.0;

    // Find the nearest checkpoint at or before the complete grid state the
    // query needs. A more recent complete state can be closer still: this is
    // the common timeline/trajectory case, where each query is one or a few
    // steps after the previous one.
    auto it = m_checkpoints.upper_bound(target);
    if (it != m_checkpoints.begin()) --it;

    // A query landing exactly on a checkpoint needs no working copy at all:
    // the checkpoint is the answer. (Checkpoints only sit on stride
    // multiples, so this also covers the boundary return below.)
    if (it->first == target && remainder_s <= 0.01) {
        m_lastStepCount = 0;
        return it->second;
    }

    // An equal step can reuse the recent state directly: m_gridState holds
    // that step's content already, so `>=` avoids a pointless copy.
    const bool useRecent = m_hasGridState && m_gridStateStep <= target &&
                           m_gridStateStep >= it->first;

    if (!useRecent) {
        m_gridState = it->second;
        m_gridStateStep = it->first;
        m_hasGridState = true;
    }

    if (m_gridStateStep < target) {
        StepRange(m_gridState, m_gridStateStep, target);
        m_gridStateStep = target;
    } else {
        m_lastStepCount = 0;
    }

    if (remainder_s <= 0.01) {
        // StepRange stored a boundary target. Return that stable copy when one
        // exists; otherwise the recent full-grid cache is the answer itself.
        if (target % m_checkpointStride == 0) {
            return m_checkpoints.at(target);
        }
        return m_gridState;
    }

    // Off-grid: branch from the complete fixed-grid cache into disposable
    // scratch. In particular, never advance m_gridState by this partial dt:
    // the next complete step must still begin on the regular grid.
    m_scratch = m_gridState;
    const f64 t_mid = gridTime + 0.5 * remainder_s / 3600.0;
    // The geometry of the grid step this is the first part of, by the same
    // rule the complete steps follow -- not the one at the partial step's
    // midpoint. A boundary inside the step belongs to the step after it, so
    // the midpoint would apply the new geometry from gridTime on, before the
    // boundary, and the partial answer would stop converging to the complete
    // step it is a part of.
    const ThermalGeometryEpoch& epoch = m_schedule->epochs[EpochForStep(target)];
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
