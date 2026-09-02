/**
 * @file ThermalStepper.cpp
 * @brief Default StepMany: loop Step with the sun column interpolated
 */

#include "thermal/ThermalStepper.hpp"

#include <algorithm>

namespace quantiloom::thermal {

namespace {

/// Interpolate one sample-major column pair into @p out. @p columnA and
/// @p columnB may be null, which is how a table with no baked gain says so.
void BlendColumn(const f32* columnA, const f32* columnB, const f32 blend, const usize n,
                 Vector<f32>& out) {
    if (columnA == nullptr) {
        std::fill(out.begin(), out.end(), 0.0f);
        return;
    }
    if (columnB == nullptr || columnA == columnB || blend <= 0.0f) {
        std::copy(columnA, columnA + n, out.begin());
        return;
    }
    if (blend >= 1.0f) {
        std::copy(columnB, columnB + n, out.begin());
        return;
    }
    for (usize e = 0; e < n; ++e) {
        out[e] = columnA[e] + blend * (columnB[e] - columnA[e]);
    }
}

}  // namespace

void IThermalStepper::StepMany(ThermalState& state,
                               const Vector<ThermalElement>& elements,
                               const Vector<ThermalMaterial>& materials,
                               const ExchangeGeometry& exchange,
                               const SunVisibilityTable& sunTable,
                               std::span<const ThermalBatchStep> steps) {
    const usize n = elements.size();
    Vector<f32> sunVis(n);
    Vector<f32> reflected;
    if (!sunTable.reflectedGain.empty()) reflected.resize(n);

    for (const ThermalBatchStep& step : steps) {
        const bool haveTable = sunTable.SampleCount() > 0 && sunTable.ElementCount() == n;
        const f32 blend = static_cast<f32>(step.sunBlend);

        if (!haveTable) {
            std::fill(sunVis.begin(), sunVis.end(), 0.0f);
        } else {
            BlendColumn(sunTable.Column(step.sunSampleA), sunTable.Column(step.sunSampleB),
                        blend, n, sunVis);
        }
        // The bounce is interpolated on the same indices as the visibility it
        // was baked from, so the two never disagree about where the sun is.
        if (!reflected.empty()) {
            BlendColumn(haveTable ? sunTable.ReflectedColumn(step.sunSampleA) : nullptr,
                        haveTable ? sunTable.ReflectedColumn(step.sunSampleB) : nullptr,
                        blend, n, reflected);
        }

        // The columns travel with the sample rather than only their blend, so
        // a step can attribute its short wave to the hours that produced it.
        ShortwaveSample sample{sunVis, reflected, sunTable.diffuseGain};
        sample.columnA = step.sunSampleA;
        sample.columnB = step.sunSampleB;
        sample.columnBlend = step.sunBlend;
        sample.columnsKnown = haveTable;

        Step(state, elements, materials, exchange, step.forcing, step.dt_s, sample);
    }
}

}  // namespace quantiloom::thermal
