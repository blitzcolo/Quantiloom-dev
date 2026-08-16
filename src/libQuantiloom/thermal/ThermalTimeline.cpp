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

void RelaxToSteadyState(ThermalState& state, const Vector<ThermalElement>& elements,
                        const Vector<ThermalMaterial>& materials,
                        const ExchangeGeometry& exchange,
                        std::span<const f32> sunVisibility,
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
                     kRelaxStep_s, sunVisibility);

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

ThermalTimeline::ThermalTimeline(const Desc& desc,
                                 const Vector<ThermalElement>& elements,
                                 const Vector<ThermalMaterial>& materials,
                                 const ExchangeGeometry& exchange,
                                 const SunVisibilityTable& sunTable,
                                 const Vector<std::pair<f64, ThermalForcing>>& forcingSeries,
                                 const ThermalForcing& constantForcing,
                                 IThermalStepper& stepper)
    : m_desc(desc)
    , m_elements(elements)
    , m_materials(materials)
    , m_exchange(exchange)
    , m_sunTable(sunTable)
    , m_forcingSeries(forcingSeries)
    , m_constantForcing(constantForcing)
    , m_stepper(stepper) {
    const f64 strideSteps =
        (desc.checkpointStride_h * 3600.0) / std::max(desc.timestep_s, 1e-6);
    m_checkpointStride = std::max(static_cast<i64>(std::round(strideSteps)), i64{1});

    for (const ThermalElement& el : elements) {
        if (el.area_m2 > 0.0f && el.materialId < materials.size() &&
            materials[el.materialId].ParticipatesInSolve()) {
            ++m_participatingElements;
        }
    }
    m_shortestTau = CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
        elements, materials, desc.initialTemperature_K);

    // Initial state
    ThermalState initial;
    initial.nodeCount = std::max(2u, desc.nodeCount);
    initial.temperature_K.assign(elements.size() * initial.nodeCount,
                                 desc.initialTemperature_K);

    if (desc.initial == InitialCondition::Steady) {
        const ThermalForcing startForcing =
            SampleForcing(forcingSeries, desc.startTime_h, constantForcing);

        // Sun visibility at start time
        Vector<f32> startSunVis;
        if (sunTable.SampleCount() > 0) {
            usize a, b;
            f64 blend;
            sunTable.SampleIndices(desc.startTime_h, a, b, blend);
            startSunVis.resize(elements.size());
            const f32* colA = sunTable.Column(a);
            const f32* colB = sunTable.Column(b);
            const f32 bf = static_cast<f32>(blend);
            for (usize e = 0; e < elements.size(); ++e) {
                startSunVis[e] = colA[e] + bf * (colB[e] - colA[e]);
            }
        } else if (!exchange.sunVisibility.empty()) {
            startSunVis = exchange.sunVisibility;
        } else {
            startSunVis.assign(elements.size(), 1.0f);
        }

        RelaxToSteadyState(initial, elements, materials, exchange,
                           startSunVis, startForcing, stepper);
    }

    m_checkpoints[0] = std::move(initial);
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
        const ThermalForcing forcing =
            SampleForcing(m_forcingSeries, t_mid, m_constantForcing);
        Vector<f32> sunVis(m_elements.size());
        if (m_sunTable.SampleCount() > 0) {
            usize a, b;
            f64 blend;
            m_sunTable.SampleIndices(t_mid, a, b, blend);
            const f32* colA = m_sunTable.Column(a);
            const f32* colB = m_sunTable.Column(b);
            const f32 bf = static_cast<f32>(blend);
            for (usize e = 0; e < m_elements.size(); ++e) {
                sunVis[e] = colA[e] + bf * (colB[e] - colA[e]);
            }
        } else if (!m_exchange.sunVisibility.empty()) {
            sunVis = m_exchange.sunVisibility;
        } else {
            std::fill(sunVis.begin(), sunVis.end(), 1.0f);
        }
        m_stepper.Step(m_scratch, m_elements, m_materials, m_exchange,
                       forcing, remainder_s, sunVis);
        ++m_lastStepCount;
    }

    return m_scratch;
}

usize ThermalTimeline::CheckpointBytes() const {
    usize total = 0;
    for (const auto& [k, state] : m_checkpoints) {
        total += state.temperature_K.size() * sizeof(f64);
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

        batch.clear();
        for (i64 step = k; step < batchEnd; ++step) {
            const f64 t_mid = GridTime(step) + 0.5 * m_desc.timestep_s / 3600.0;
            ThermalBatchStep bs;
            bs.forcing = SampleForcing(m_forcingSeries, t_mid, m_constantForcing);
            bs.dt_s = m_desc.timestep_s;
            if (m_sunTable.SampleCount() > 0) {
                m_sunTable.SampleIndices(t_mid, bs.sunSampleA, bs.sunSampleB, bs.sunBlend);
            }
            batch.push_back(bs);
        }

        m_stepper.StepMany(state, m_elements, m_materials, m_exchange,
                           m_sunTable, batch);
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
