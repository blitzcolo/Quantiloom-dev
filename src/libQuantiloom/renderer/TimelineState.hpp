/**
 * @file TimelineState.hpp
 * @brief Which nodes move, how, and where they would stand if they did not
 *
 * A side table, deliberately. Scene, SceneNode, Material, Texture and
 * BRDFModels are the layout contract between the SDK and Quantiloom Studio:
 * they are compiled into both binaries and neither may grow a field. So the
 * animation lives here, keyed by node index, and a node's `transform` stays
 * what it always was -- where the node is right now.
 *
 * ## The composition
 *
 *     world(t) = M_model(t) * R_model * M_node(t) * R_node
 *
 * `R_model` is the `[[models]]` rest pose, `R_node` what the file placed the
 * node at (possibly overwritten by `[[nodes]]`). The model's track moves the
 * whole model in world space; the node's track moves the node within its
 * model, which is what makes a wheel spin about its own axle rather than about
 * the world origin.
 *
 * ## Rest poses, and why `[[nodes]]` writes one
 *
 * `RestOf` returns `R_model * R_node` -- where the node would stand with every
 * trajectory at the identity. That, not the pose at `time_s`, is what a saved
 * config records for a moving node: a pose that meant "where it is at 2.5 s"
 * would be wrong at every other instant, and dragging the transport would have
 * to mark the document modified. Dragging a gizmo at time t therefore goes
 * through SetWorldPose, which solves the composition above for `R_node` and
 * keeps the edit at every t.
 */

#pragma once

#include "renderer/ConfigResolve.hpp"
#include "renderer/TimelineControl.hpp"
#include "renderer/RenderCore.hpp"
#include "scene/Motion.hpp"

#include <memory>

namespace quantiloom::rendercore {

/// One node the clock moves, and everything needed to move it without
/// touching the scene graph.
struct AnimatedNode {
    u32 node = 0;
    u32 model = SceneLoadInfo::kNoModel;

    glm::mat4 modelRest{1.0f};  ///< R_model
    glm::mat4 nodeRest{1.0f};   ///< R_node

    /// Shared: one compiled track per `[[models]]` entry, however many nodes
    /// that model brought.
    std::shared_ptr<const scene::MotionTrack> modelTrack;
    std::shared_ptr<const scene::MotionTrack> nodeTrack;

    /// Radius of the node's bounding sphere in world units. Only the epoch
    /// planner reads it, to weigh a rotation against a translation.
    f32 boundRadius_m = 0.0f;

    /// Whether any of this node's primitives emits. A moving emitter means the
    /// emissive triangle list has to be rebuilt for the frame; a moving rock
    /// does not.
    bool emissive = false;
};

class TimelineState {
public:
    TimelineState() = default;

    /**
     * @brief Compile every trajectory and pair it with its nodes
     *
     * Called after `[[nodes]]` has had its say, so the transforms it snapshots
     * as rest poses are the final ones.
     *
     * @param report  motionTracks, modelsLoaded and timelinePresent are filled
     */
    static TimelineState Build(const Scene& scene, const TimelineConfig& config,
                               const Vector<ModelEntry>& models, const SceneLoadInfo& info,
                               const Vector<std::pair<u32, scene::MotionSpec>>& nodeMotion,
                               ConfigApplyReport& report);

    /// True when there is a clock to move: either the config declared one or
    /// something in the scene has a trajectory.
    [[nodiscard]] bool Present() const { return m_present; }
    [[nodiscard]] bool HasMotion() const { return !m_animated.empty(); }

    /**
     * @brief Put every animated node where it belongs at @p t_s
     *
     * @return the node indices whose transform actually changed, so a caller
     *         can skip the TLAS refit and the emissive rebuild when a scrub
     *         landed on the same pose.
     */
    Vector<u32> Apply(Scene& scene, f64 t_s);

    /// The pose at @p t_s without touching the scene. Used by epoch planning,
    /// which asks about times other than the one on screen.
    [[nodiscard]] glm::mat4 PoseAt(const AnimatedNode& node, f64 t_s) const;

    /// Reconcile a gizmo drag: the node is at @p world now, so its rest pose
    /// is whatever makes the composition come out that way at the current time.
    void SetWorldPose(u32 node, const glm::mat4& world);

    /// Where a node would stand with every trajectory at the identity. For a
    /// node with no trajectory that is simply its transform.
    [[nodiscard]] glm::mat4 RestOf(const Scene& scene, u32 node) const;

    /// The thermal hour @p t_s maps to. Meaningless unless SetThermalMapping
    /// has been called.
    [[nodiscard]] f64 HourAt(f64 t_s) const;

    /// `thermal.time_h` becomes the hour at the timeline's start, and the
    /// scale says how fast the clock runs from there.
    void SetThermalMapping(f64 hourAtStart_h, f64 scale);
    [[nodiscard]] bool ThermalMapped() const { return m_thermalMapped; }
    [[nodiscard]] f64 ThermalHourAtStart() const { return m_thermalHourAtStart; }

    /// Reported by the epoch builder once it knows; purely informational.
    void SetEpochCounts(u32 count, u32 current);

    [[nodiscard]] TimelineInfo Info() const;
    [[nodiscard]] f64 Current_s() const { return m_current_s; }
    void SetCurrent(f64 t_s) { m_current_s = t_s; }

    [[nodiscard]] const Vector<AnimatedNode>& Animated() const { return m_animated; }
    [[nodiscard]] const TimelineConfig& Config() const { return m_config; }

    /// Every instant at which some trajectory stops being smooth, within
    /// [from_s, to_s]. Sorted and deduplicated.
    [[nodiscard]] Vector<f64> ChangeTimes(f64 from_s, f64 to_s) const;

private:
    TimelineConfig m_config;
    Vector<AnimatedNode> m_animated;
    bool m_present = false;

    f64 m_current_s = 0.0;
    bool m_thermalMapped = false;
    f64 m_thermalHourAtStart = 0.0;

    u32 m_modelCount = 0;
    u32 m_epochCount = 0;
    u32 m_currentEpoch = 0;
};

}  // namespace quantiloom::rendercore
