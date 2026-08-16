/**
 * @file ThermalStepper.cpp
 * @brief Default StepMany: loop Step with sun visibility interpolation
 */

#include "thermal/ThermalStepper.hpp"

namespace quantiloom::thermal {

void IThermalStepper::StepMany(ThermalState& state,
                               const Vector<ThermalElement>& elements,
                               const Vector<ThermalMaterial>& materials,
                               const ExchangeGeometry& exchange,
                               const SunVisibilityTable& sunTable,
                               std::span<const ThermalBatchStep> steps) {
    const usize n = elements.size();
    Vector<f32> sunVis(n);

    for (const ThermalBatchStep& step : steps) {
        if (sunTable.SampleCount() == 0) {
            std::fill(sunVis.begin(), sunVis.end(), 0.0f);
        } else if (step.sunSampleA == step.sunSampleB || step.sunBlend <= 0.0) {
            const f32* col = sunTable.Column(step.sunSampleA);
            std::copy(col, col + n, sunVis.begin());
        } else if (step.sunBlend >= 1.0) {
            const f32* col = sunTable.Column(step.sunSampleB);
            std::copy(col, col + n, sunVis.begin());
        } else {
            const f32* colA = sunTable.Column(step.sunSampleA);
            const f32* colB = sunTable.Column(step.sunSampleB);
            const f32 b = static_cast<f32>(step.sunBlend);
            for (usize e = 0; e < n; ++e) {
                sunVis[e] = colA[e] + b * (colB[e] - colA[e]);
            }
        }
        Step(state, elements, materials, exchange, step.forcing, step.dt_s, sunVis);
    }
}

}  // namespace quantiloom::thermal
