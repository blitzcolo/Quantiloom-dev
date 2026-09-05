/**
 * @file ThermalEpochs.hpp
 * @brief Where to cut a moving scene into pieces that stand still
 *
 * The expensive part of a thermal solve is finding out who sees whom, and that
 * answer only changes when something moves. So the question this file answers
 * is: at which instants is it worth asking again?
 *
 * Three sources of candidates, in order of how much they are worth:
 *
 *  - The trajectory's own discontinuities -- keyframe times, segment
 *    boundaries. A path that turns a corner at t = 4 should have a boundary at
 *    t = 4, whatever the stride says.
 *  - A regular stride while something is actually moving, so a smooth path is
 *    sampled rather than treated as one jump.
 *  - The start, always, because epoch zero is the world the solve relaxes into.
 *
 * And one filter: a candidate at which nothing has moved more than
 * `minMove_m` since the last kept boundary is dropped. A convoy that parks for
 * six hours costs one epoch for those six hours, not one per stride.
 *
 * No GPU here and no Vulkan: this decides *when*, and ThermalEpochBuilder does
 * the expensive part of deciding *what*.
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>

#include <functional>

namespace quantiloom::thermal {

/**
 * @brief What the planner needs to know about the things that move
 */
struct EpochPlanInput {
    f64 start_s = 0.0;
    f64 end_s = 0.0;

    /// The longest an epoch may run while something is moving. Zero or less
    /// means no regular candidates at all, so only the trajectories'
    /// discontinuities produce boundaries.
    ///
    /// A stride finer than one thermal timestep buys nothing: the solver
    /// integrates one geometry per step, so two boundaries inside one step are
    /// indistinguishable from one.
    f64 stride_s = 0.0;

    /// A candidate at which nothing moved further than this since the last
    /// kept boundary is discarded.
    f32 minMove_m = 0.05f;

    struct Node {
        f32 boundRadius_m = 0.0f;
        /// The node's world transform at a time. Called O(candidates * nodes)
        /// times, which is tens by tens.
        std::function<glm::mat4(f64)> poseAt;
        /// Instants at which this node's trajectory stops being smooth.
        Vector<f64> changeTimes;
    };
    Vector<Node> nodes;
};

/**
 * @brief The instants at which the geometry is worth re-measuring
 *
 * @return sorted timeline seconds, always beginning with `start_s`. A scene
 *         with nothing moving gets exactly one entry, which is the
 *         single-geometry case spelled the same way.
 */
[[nodiscard]] Vector<f64> PlanEpochTimes(const EpochPlanInput& input);

}  // namespace quantiloom::thermal
