/**
 * @file Motion.hpp
 * @brief What a node's transform is at a time, and nothing else
 *
 * A MotionSpec is what a TOML said; a MotionTrack is what that means at an
 * instant. The split is the same one the rest of the resolver keeps: parsing
 * happens once and reports its problems as text, evaluation happens per frame
 * and cannot fail. A track that failed to compile is a track that evaluates to
 * identity, which leaves its node standing at its rest pose -- the honest
 * outcome for a trajectory nobody could read, and one that still renders.
 *
 * ## The three grammars, and why there are three
 *
 * Keyframes are what every interchange format has (glTF samplers, USD
 * timeSamples, DIRSIG waypoints) and what an exporter can write. Piecewise
 * expressions are what someone writes by hand when the path is a formula
 * rather than a list of points. The parametric engines -- straight line,
 * circle, spin, face-the-way-you-are-going, look-at -- are the handful of
 * motions that occur so often in sensor simulation that spelling them as
 * either of the other two is busywork.
 *
 * They are alternatives, not layers: a spec is in exactly one of the three
 * forms. What *does* layer is the model track and the node track, which
 * compose as
 *
 *     world(t) = M_model(t) * R_model * M_node(t) * R_node
 *
 * with R the rest poses. Every track here produces one of the M factors: a
 * rigid offset from rest, so an absent channel means "no offset" rather than
 * "at the origin".
 *
 * ## Rigid only
 *
 * Translation and rotation. No scale, no shear, no per-vertex deformation --
 * the thermal side depends on it: a rigid epoch produces the same elements in
 * the same order with only their centroids and normals moved, which is what
 * lets one solve carry its state across a geometry change.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>

namespace quantiloom::scene {

/// Which of the three grammars a spec is written in.
enum class MotionForm : u8 { None, Keys, Segments, Engine };

/// How a value between two keys is found. Cubic is Catmull-Rom on position
/// and an eased slerp on rotation -- deliberately not squad, which would need
/// tangent quaternions nobody authors and buys smoothness this does not need.
enum class MotionInterp : u8 { Step, Linear, Cubic };

/// What happens outside the authored range. Hold clamps to the nearest end;
/// Loop wraps over the whole span, so the last pose meets the first.
enum class MotionExtrap : u8 { Hold, Loop };

/// One authored instant. The two channels are independent: a key may carry a
/// position, a rotation, or both, and a channel with no keys at all is absent
/// rather than zero.
struct MotionKey {
    f64 t_s = 0.0;
    bool hasPosition = false;
    bool hasRotation = false;
    glm::dvec3 position{0.0};
    glm::dquat rotation{1.0, 0.0, 0.0, 0.0};
};

/// One piece of a piecewise definition. The strings are expressions in `t`
/// (global seconds) and `s` (seconds since this segment began); they are
/// compiled once, by MotionTrack::Compile, and never re-parsed.
struct MotionSegment {
    f64 from_s = 0.0;
    f64 to_s = 0.0;
    bool hasPosition = false;
    bool hasRotation = false;
    String position[3];
    String rotationEulerDeg[3];
};

enum class MotionLocationType : u8 { None, Linear, Circle };

/// Where the parametric engine puts a thing. Angles in degrees, distances in
/// world units, time measured from MotionSpec::origin_s.
struct MotionLocation {
    MotionLocationType type = MotionLocationType::None;

    // Linear
    glm::dvec3 start{0.0};
    glm::dvec3 velocity{0.0};

    // Circle
    glm::dvec3 center{0.0};
    f64 radius = 0.0;
    f64 angularSpeedDeg_s = 0.0;
    f64 startAngleDeg = 0.0;
    glm::dvec3 axis{0.0, 1.0, 0.0};
};

enum class MotionOrientationType : u8 { None, Fixed, Spin, AlongVelocity, LookAt };

/// Which way it faces while it does that.
struct MotionOrientation {
    MotionOrientationType type = MotionOrientationType::None;

    glm::dvec3 eulerDeg{0.0};               ///< Fixed
    glm::dvec3 axis{0.0, 1.0, 0.0};         ///< Spin
    f64 rateDeg_s = 0.0;                    ///< Spin
    glm::dvec3 forward{0.0, 0.0, 1.0};      ///< AlongVelocity: the model's own nose
    glm::dvec3 target{0.0};                 ///< LookAt
    glm::dvec3 up{0.0, 1.0, 0.0};           ///< LookAt
};

/**
 * @brief A trajectory as written, before anything is compiled
 *
 * Plain data on purpose: the config layer fills it, the tests build it
 * directly, and neither has to link an expression parser to do so.
 */
struct MotionSpec {
    MotionForm form = MotionForm::None;
    MotionInterp interpolation = MotionInterp::Linear;
    MotionExtrap extrapolate = MotionExtrap::Hold;

    Vector<MotionKey> keys;
    Vector<MotionSegment> segments;
    MotionLocation location;
    MotionOrientation orientation;

    /// The zero of the engines' clock -- `timeline.start_s`. Keyframe and
    /// segment times are absolute timeline seconds and ignore this.
    f64 origin_s = 0.0;
};

/// Up to six compiled scalar expressions over `t` and `s`, in the order
/// px, py, pz, rx, ry, rz. Defined in MotionExpr.cpp, which is the only
/// translation unit that includes ExprTk -- it costs the better part of a
/// minute to compile and there is no reason to pay it twice.
class SegmentExpressions {
public:
    /// @param expressions  exactly as authored; an empty string compiles to 0
    /// @return the compiled set, or the parser's own message with the offending
    ///         expression appended
    static Result<std::unique_ptr<SegmentExpressions>, String> Compile(
        const Vector<String>& expressions);

    ~SegmentExpressions();
    SegmentExpressions(const SegmentExpressions&) = delete;
    SegmentExpressions& operator=(const SegmentExpressions&) = delete;

    /// Writes one f64 per compiled expression into @p out. Not re-entrant:
    /// the bound variables are members.
    void Evaluate(f64 t_s, f64 s_s, f64* out) const;
    [[nodiscard]] usize Count() const;

private:
    SegmentExpressions();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief A compiled trajectory
 *
 * One of these per distinct MotionSpec, shared by every node it drives -- a
 * model's own track is evaluated once per frame no matter how many nodes the
 * model brought.
 *
 * Immutable after Compile, but not re-entrant: the expression form writes the
 * time into a scratch the compiled expressions read, so two threads must not
 * evaluate the same track at once. Nothing here does; the timeline applies
 * poses from one thread.
 */
class MotionTrack {
public:
    /**
     * @brief Compile a spec, keeping whatever parts of it are usable
     *
     * Only a spec that cannot produce any motion at all is an error. A segment
     * whose expression will not parse is dropped with a warning and the rest of
     * the track survives, because losing one piece of a path is recoverable and
     * losing the scene is not.
     *
     * @param warnings  appended to, never cleared; may be null
     */
    static Result<std::unique_ptr<MotionTrack>, String> Compile(
        const MotionSpec& spec, Vector<String>* warnings = nullptr);

    ~MotionTrack();
    MotionTrack(const MotionTrack&) = delete;
    MotionTrack& operator=(const MotionTrack&) = delete;

    /// The rigid offset from rest at @p t_s: translation composed with rotation.
    [[nodiscard]] glm::mat4 Evaluate(f64 t_s) const;

    [[nodiscard]] glm::dvec3 PositionAt(f64 t_s) const;

    /// Analytic where an engine knows its own derivative, and a central
    /// difference over 1 ms everywhere else. Only the orientation engines and
    /// the epoch planner ask.
    [[nodiscard]] glm::dvec3 VelocityAt(f64 t_s) const;

    /// The instants at which this track stops being smooth, within
    /// [from_s, to_s]: key times, or segment boundaries. Empty for the engines,
    /// which are smooth everywhere. Sorted, and free of duplicates.
    [[nodiscard]] Vector<f64> ChangeTimes(f64 from_s, f64 to_s) const;

    /// True when evaluation is the identity at every time, so a caller can skip
    /// the node entirely.
    [[nodiscard]] bool IsStatic() const;

    /// The last authored instant -- the last key or the end of the last
    /// segment. Zero for the engines, which do not end. A timeline with no
    /// `end_s` of its own takes the largest of these.
    [[nodiscard]] f64 AuthoredEnd_s() const;

private:
    MotionTrack();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief How far a rigid body moved, counting rotation as arc length
 *
 * `|dt| + r * angle`, with r the radius of the node's bounding sphere: a
 * degree of yaw on something ten metres wide has to count for more than a
 * degree of yaw on a bolt, because what the thermal side cares about is
 * whether any surface went somewhere new.
 */
[[nodiscard]] f32 PoseDisplacement(const glm::mat4& a, const glm::mat4& b, f32 boundRadius_m);

}  // namespace quantiloom::scene
