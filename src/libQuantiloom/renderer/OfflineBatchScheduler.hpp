#pragma once

#include "core/Types.hpp"

#include <algorithm>
#include <cmath>
#include <span>

namespace quantiloom::rendercore {

/// Chooses how many path samples belong in the next offline GPU submit.
///
/// The budget is deliberately far below Windows' TDR interval.  A newly seen
/// slower sample replaces the estimate immediately, while faster samples move
/// it down gradually; batch growth is capped separately so one unusually fast
/// submit cannot turn into a long, uninterruptible one.  Invalid or unsupported
/// GPU timestamps leave the historical two-sample batch in place.
class OfflineBatchScheduler {
public:
    static constexpr f32 kTargetGpuMs = 100.0f;
    static constexpr u32 kFallbackSamples = 2;
    static constexpr u32 kMaxSamples = 32;

    [[nodiscard]] u32 NextBatchSize(const u32 remainingSamples) const {
        if (remainingSamples == 0) return 0;
        if (!m_hasEstimate) {
            return std::min(remainingSamples, kFallbackSamples);
        }

        const auto budgetSamples = static_cast<u32>(std::clamp(
            std::floor(kTargetGpuMs / m_estimatedSampleGpuMs), 1.0f,
            static_cast<f32>(kMaxSamples)));
        const u32 growthLimit = std::min(kMaxSamples, m_previousBatchSize * 2u);
        return std::min({remainingSamples, budgetSamples, growthLimit});
    }

    void ObserveCompletedBatch(const std::span<const f32> sampleGpuMs) {
        if (sampleGpuMs.empty()) return;
        m_previousBatchSize = static_cast<u32>(sampleGpuMs.size());

        f32 slowestValidSampleMs = 0.0f;
        for (const f32 elapsedMs : sampleGpuMs) {
            if (std::isfinite(elapsedMs) && elapsedMs > 0.0f) {
                slowestValidSampleMs = std::max(slowestValidSampleMs, elapsedMs);
            }
        }
        if (slowestValidSampleMs == 0.0f) return;

        if (!m_hasEstimate || slowestValidSampleMs >= m_estimatedSampleGpuMs) {
            m_estimatedSampleGpuMs = slowestValidSampleMs;
        } else {
            constexpr f32 kPreviousEstimateWeight = 0.75f;
            m_estimatedSampleGpuMs =
                kPreviousEstimateWeight * m_estimatedSampleGpuMs +
                (1.0f - kPreviousEstimateWeight) * slowestValidSampleMs;
        }
        m_hasEstimate = true;
    }

private:
    f32 m_estimatedSampleGpuMs = 0.0f;
    u32 m_previousBatchSize = kFallbackSamples;
    bool m_hasEstimate = false;
};

}  // namespace quantiloom::rendercore
