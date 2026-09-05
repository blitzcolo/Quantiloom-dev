#include "thermal/ThermalEpochs.hpp"

#include "core/Log.hpp"
#include "scene/Motion.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::thermal {

namespace {

/// A guard on the periodic candidates, so that a stride someone typed as a
/// millisecond over a month does not try to allocate a hundred million
/// doubles. Hit means the stride was nonsense, and saying so is better than
/// either obeying it or silently rounding it.
constexpr usize kMaxPeriodicCandidates = 100000;

}  // namespace

Vector<f64> PlanEpochTimes(const EpochPlanInput& input) {
    Vector<f64> kept;
    kept.push_back(input.start_s);

    if (input.nodes.empty() || input.end_s <= input.start_s) {
        return kept;
    }

    // ------------------------------------------------------------------
    // Candidates
    // ------------------------------------------------------------------
    Vector<f64> candidates;
    for (const EpochPlanInput::Node& node : input.nodes) {
        for (const f64 t : node.changeTimes) {
            if (t > input.start_s && t <= input.end_s) candidates.push_back(t);
        }
    }

    if (input.stride_s > 0.0) {
        const f64 span = input.end_s - input.start_s;
        const f64 count = std::floor(span / input.stride_s);
        if (count > static_cast<f64>(kMaxPeriodicCandidates)) {
            QL_LOG_WARN("  Thermal epochs: a stride of {:g} s over {:g} s would be {:g} "
                        "candidates; using {} instead",
                        input.stride_s, span, count, kMaxPeriodicCandidates);
        }
        const usize steps =
            static_cast<usize>(std::min(count, static_cast<f64>(kMaxPeriodicCandidates)));
        for (usize k = 1; k <= steps; ++k) {
            const f64 t = input.start_s + static_cast<f64>(k) * input.stride_s;
            if (t > input.start_s && t <= input.end_s) candidates.push_back(t);
        }
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    // ------------------------------------------------------------------
    // Keep only the ones something actually moved for
    // ------------------------------------------------------------------
    Vector<glm::mat4> lastPose;
    lastPose.reserve(input.nodes.size());
    for (const EpochPlanInput::Node& node : input.nodes) {
        lastPose.push_back(node.poseAt ? node.poseAt(input.start_s) : glm::mat4(1.0f));
    }

    Vector<glm::mat4> here(input.nodes.size());
    for (const f64 candidate : candidates) {
        f32 furthest = 0.0f;
        for (usize n = 0; n < input.nodes.size(); ++n) {
            const EpochPlanInput::Node& node = input.nodes[n];
            here[n] = node.poseAt ? node.poseAt(candidate) : lastPose[n];
            furthest = std::max(furthest,
                                scene::PoseDisplacement(lastPose[n], here[n],
                                                        node.boundRadius_m));
        }
        if (furthest < input.minMove_m) continue;

        kept.push_back(candidate);
        lastPose = here;
    }

    return kept;
}

}  // namespace quantiloom::thermal
