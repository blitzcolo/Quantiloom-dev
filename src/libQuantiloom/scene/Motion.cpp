#include "scene/Motion.hpp"

#include "core/Log.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace quantiloom::scene {

namespace {

constexpr f64 kVelocityEpsilon = 1e-9;
/// Half-width of the central difference that stands in for a derivative the
/// track cannot state. One millisecond: fine enough that a 100 m/s trajectory
/// resolves to a tenth of a metre, coarse enough that f64 subtraction of two
/// nearby positions still has bits left.
constexpr f64 kFiniteDifference_s = 1e-3;

/// The same X-then-Y-then-Z composition ParseTransformKeys uses for
/// `rotation_euler_degrees`, so the two places a scene spells a rotation mean
/// the same rotation.
glm::dquat EulerDegreesToQuat(const glm::dvec3& degrees) {
    const glm::dvec3 r = glm::radians(degrees);
    return glm::angleAxis(r.x, glm::dvec3(1.0, 0.0, 0.0)) *
           glm::angleAxis(r.y, glm::dvec3(0.0, 1.0, 0.0)) *
           glm::angleAxis(r.z, glm::dvec3(0.0, 0.0, 1.0));
}

/// Shortest-arc slerp. glm will happily interpolate the long way round when
/// two keys are more than half a turn apart in their raw components, which is
/// a thing an exporter produces and nobody means.
glm::dquat ShortestSlerp(const glm::dquat& a, glm::dquat b, f64 u) {
    if (glm::dot(a, b) < 0.0) b = -b;
    return glm::normalize(glm::slerp(a, b, u));
}

/// The shortest rotation taking one direction onto another. Written out
/// rather than taken from glm's gtx/quaternion.hpp, which is an experimental
/// extension this project does not enable.
glm::dquat RotationBetween(const glm::dvec3& from, const glm::dvec3& to) {
    const glm::dvec3 a = glm::normalize(from);
    const glm::dvec3 b = glm::normalize(to);
    const f64 d = glm::dot(a, b);
    if (d > 1.0 - 1e-12) return glm::dquat(1.0, 0.0, 0.0, 0.0);
    if (d < -1.0 + 1e-12) {
        // Antiparallel: the axis is any perpendicular, and which one is
        // genuinely arbitrary -- there is no shortest arc between opposites.
        glm::dvec3 axis = glm::cross(glm::dvec3(1.0, 0.0, 0.0), a);
        if (glm::length(axis) < 1e-9) axis = glm::cross(glm::dvec3(0.0, 1.0, 0.0), a);
        return glm::angleAxis(glm::pi<f64>(), glm::normalize(axis));
    }
    const glm::dvec3 axis = glm::cross(a, b);
    const f64 s = std::sqrt((1.0 + d) * 2.0);
    return glm::normalize(glm::dquat(s * 0.5, axis / s));
}

/// The rotation that points @p forward along @p dir, with @p up deciding the
/// roll. Falls back to the minimal rotation when the direction and the up
/// vector are parallel, which is the one case a frame cannot be built.
glm::dquat FrameRotation(const glm::dvec3& dir, const glm::dvec3& up,
                         const glm::dvec3& forward) {
    const f64 len = glm::length(dir);
    if (len < kVelocityEpsilon) return glm::dquat(1.0, 0.0, 0.0, 0.0);
    const glm::dvec3 f = dir / len;
    const glm::dvec3 nf = glm::normalize(forward);

    glm::dvec3 right = glm::cross(up, f);
    if (glm::length(right) < 1e-6) {
        // dir is along up: no frame to build, so take the shortest rotation
        // that gets the nose there and leave the roll wherever it lands.
        return RotationBetween(nf, f);
    }
    right = glm::normalize(right);
    const glm::dvec3 realUp = glm::cross(f, right);

    // Columns map +X, +Y, +Z onto the frame, so this rotation takes +Z to f.
    glm::dmat3 basis(right, realUp, f);
    // ...and forward is not necessarily +Z, so undo that first.
    return glm::normalize(glm::dquat(basis) * RotationBetween(nf, glm::dvec3(0.0, 0.0, 1.0)));
}

/// An orthonormal pair spanning the plane a circle engine turns in. Chosen so
/// that the default axis +Y starts the body at +Z and swings it toward +X --
/// the same "zero degrees is +Z, clockwise from above" convention the heading
/// form of the linear engine uses.
void CirclePlane(const glm::dvec3& axis, glm::dvec3& u, glm::dvec3& w) {
    glm::dvec3 n = axis;
    if (glm::length(n) < 1e-9) n = glm::dvec3(0.0, 1.0, 0.0);
    n = glm::normalize(n);
    const glm::dvec3 helper =
        (std::abs(n.y) < 0.9) ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0);
    u = glm::normalize(glm::cross(helper, n));
    w = glm::cross(n, u);
}

}  // namespace

// ============================================================================
// MotionTrack
// ============================================================================

struct MotionTrack::Impl {
    MotionForm form = MotionForm::None;
    MotionInterp interp = MotionInterp::Linear;
    MotionExtrap extrap = MotionExtrap::Hold;

    // Keyframes, split into the two independent channels and sorted.
    Vector<f64> posTimes;
    Vector<glm::dvec3> posValues;
    Vector<f64> rotTimes;
    Vector<glm::dquat> rotValues;

    struct CompiledSegment {
        f64 from_s = 0.0;
        f64 to_s = 0.0;
        bool hasPosition = false;
        bool hasRotation = false;
        std::unique_ptr<SegmentExpressions> expressions;
    };
    Vector<CompiledSegment> segments;

    MotionLocation location;
    MotionOrientation orientation;
    f64 origin_s = 0.0;

    f64 domainFrom_s = 0.0;
    f64 domainTo_s = 0.0;
    bool hasDomain = false;
    f64 authoredEnd_s = 0.0;
    bool isStatic = true;

    [[nodiscard]] f64 RemapTime(f64 t) const {
        if (!hasDomain) return t;
        if (extrap == MotionExtrap::Loop && domainTo_s > domainFrom_s) {
            const f64 span = domainTo_s - domainFrom_s;
            f64 x = std::fmod(t - domainFrom_s, span);
            if (x < 0.0) x += span;
            return domainFrom_s + x;
        }
        return std::clamp(t, domainFrom_s, domainTo_s);
    }

    [[nodiscard]] glm::dvec3 SamplePosition(f64 t) const;
    [[nodiscard]] glm::dquat SampleRotation(f64 t) const;
    [[nodiscard]] const CompiledSegment* SegmentAt(f64 t) const;
    [[nodiscard]] glm::dvec3 EnginePosition(f64 t) const;
    [[nodiscard]] glm::dvec3 EngineVelocity(f64 t) const;
    [[nodiscard]] glm::dquat EngineOrientation(f64 t, const glm::dvec3& position) const;
};

glm::dvec3 MotionTrack::Impl::SamplePosition(f64 t) const {
    const usize n = posTimes.size();
    if (n == 0) return glm::dvec3(0.0);
    if (n == 1 || t <= posTimes.front()) return posValues.front();
    if (t >= posTimes.back()) return posValues.back();

    const usize hi = static_cast<usize>(
        std::upper_bound(posTimes.begin(), posTimes.end(), t) - posTimes.begin());
    const usize lo = hi - 1;
    if (interp == MotionInterp::Step) return posValues[lo];

    const f64 h = posTimes[hi] - posTimes[lo];
    const f64 u = (h > 0.0) ? (t - posTimes[lo]) / h : 0.0;
    if (interp == MotionInterp::Linear) {
        return glm::mix(posValues[lo], posValues[hi], u);
    }

    // Catmull-Rom with one-sided tangents at the ends, spaced by the actual
    // key times rather than assuming a uniform grid.
    auto tangent = [&](usize i) -> glm::dvec3 {
        const usize prev = (i == 0) ? 0 : i - 1;
        const usize next = (i + 1 >= n) ? n - 1 : i + 1;
        const f64 dt = posTimes[next] - posTimes[prev];
        if (dt <= 0.0) return glm::dvec3(0.0);
        return (posValues[next] - posValues[prev]) / dt;
    };
    const glm::dvec3 m0 = tangent(lo);
    const glm::dvec3 m1 = tangent(hi);
    const f64 u2 = u * u;
    const f64 u3 = u2 * u;
    return (2.0 * u3 - 3.0 * u2 + 1.0) * posValues[lo] + (u3 - 2.0 * u2 + u) * h * m0 +
           (-2.0 * u3 + 3.0 * u2) * posValues[hi] + (u3 - u2) * h * m1;
}

glm::dquat MotionTrack::Impl::SampleRotation(f64 t) const {
    const usize n = rotTimes.size();
    if (n == 0) return glm::dquat(1.0, 0.0, 0.0, 0.0);
    if (n == 1 || t <= rotTimes.front()) return rotValues.front();
    if (t >= rotTimes.back()) return rotValues.back();

    const usize hi = static_cast<usize>(
        std::upper_bound(rotTimes.begin(), rotTimes.end(), t) - rotTimes.begin());
    const usize lo = hi - 1;
    if (interp == MotionInterp::Step) return rotValues[lo];

    const f64 h = rotTimes[hi] - rotTimes[lo];
    f64 u = (h > 0.0) ? (t - rotTimes[lo]) / h : 0.0;
    // Cubic on rotation is an eased slerp, not squad: the parameter gets the
    // smooth-step so velocity is continuous at the keys, while the path stays
    // the great circle. Squad would need tangent quaternions no exporter
    // writes and no config author would want to.
    if (interp == MotionInterp::Cubic) u = u * u * (3.0 - 2.0 * u);
    return ShortestSlerp(rotValues[lo], rotValues[hi], u);
}

const MotionTrack::Impl::CompiledSegment* MotionTrack::Impl::SegmentAt(f64 t) const {
    if (segments.empty()) return nullptr;
    for (const auto& seg : segments) {
        if (t >= seg.from_s && t < seg.to_s) return &seg;
    }
    if (t >= segments.back().to_s) return &segments.back();
    return &segments.front();
}

glm::dvec3 MotionTrack::Impl::EnginePosition(f64 t) const {
    const f64 dt = t - origin_s;
    switch (location.type) {
        case MotionLocationType::Linear:
            return location.start + location.velocity * dt;
        case MotionLocationType::Circle: {
            glm::dvec3 u, w;
            CirclePlane(location.axis, u, w);
            const f64 theta =
                glm::radians(location.startAngleDeg + location.angularSpeedDeg_s * dt);
            return location.center + location.radius * (std::cos(theta) * u + std::sin(theta) * w);
        }
        case MotionLocationType::None:
            break;
    }
    return glm::dvec3(0.0);
}

glm::dvec3 MotionTrack::Impl::EngineVelocity(f64 t) const {
    const f64 dt = t - origin_s;
    switch (location.type) {
        case MotionLocationType::Linear:
            return location.velocity;
        case MotionLocationType::Circle: {
            glm::dvec3 u, w;
            CirclePlane(location.axis, u, w);
            const f64 omega = glm::radians(location.angularSpeedDeg_s);
            const f64 theta =
                glm::radians(location.startAngleDeg + location.angularSpeedDeg_s * dt);
            return location.radius * omega * (-std::sin(theta) * u + std::cos(theta) * w);
        }
        case MotionLocationType::None:
            break;
    }
    return glm::dvec3(0.0);
}

glm::dquat MotionTrack::Impl::EngineOrientation(f64 t, const glm::dvec3& position) const {
    switch (orientation.type) {
        case MotionOrientationType::Fixed:
            return EulerDegreesToQuat(orientation.eulerDeg);
        case MotionOrientationType::Spin: {
            glm::dvec3 axis = orientation.axis;
            if (glm::length(axis) < 1e-9) axis = glm::dvec3(0.0, 1.0, 0.0);
            return glm::angleAxis(glm::radians(orientation.rateDeg_s * (t - origin_s)),
                                  glm::normalize(axis));
        }
        case MotionOrientationType::AlongVelocity:
            return FrameRotation(EngineVelocity(t), orientation.up, orientation.forward);
        case MotionOrientationType::LookAt:
            return FrameRotation(orientation.target - position, orientation.up,
                                 orientation.forward);
        case MotionOrientationType::None:
            break;
    }
    return glm::dquat(1.0, 0.0, 0.0, 0.0);
}

MotionTrack::MotionTrack() : m_impl(std::make_unique<Impl>()) {}
MotionTrack::~MotionTrack() = default;

Result<std::unique_ptr<MotionTrack>, String> MotionTrack::Compile(const MotionSpec& spec,
                                                                  Vector<String>* warnings) {
    auto track = std::unique_ptr<MotionTrack>(new MotionTrack());
    Impl& impl = *track->m_impl;
    impl.form = spec.form;
    impl.interp = spec.interpolation;
    impl.extrap = spec.extrapolate;
    impl.location = spec.location;
    impl.orientation = spec.orientation;
    impl.origin_s = spec.origin_s;

    auto warn = [&](String text) {
        if (warnings) warnings->push_back(std::move(text));
    };

    switch (spec.form) {
        case MotionForm::Keys: {
            Vector<MotionKey> keys = spec.keys;
            std::stable_sort(keys.begin(), keys.end(),
                             [](const MotionKey& a, const MotionKey& b) { return a.t_s < b.t_s; });
            for (const auto& key : keys) {
                if (key.hasPosition) {
                    impl.posTimes.push_back(key.t_s);
                    impl.posValues.push_back(key.position);
                }
                if (key.hasRotation) {
                    impl.rotTimes.push_back(key.t_s);
                    impl.rotValues.push_back(glm::normalize(key.rotation));
                }
            }
            if (impl.posTimes.empty() && impl.rotTimes.empty()) {
                return Result<std::unique_ptr<MotionTrack>, String>::Err(
                    "motion has keyframes but none of them carries a position or a rotation");
            }
            if (!keys.empty()) {
                impl.hasDomain = true;
                impl.domainFrom_s = keys.front().t_s;
                impl.domainTo_s = keys.back().t_s;
                impl.authoredEnd_s = keys.back().t_s;
            }
            impl.isStatic = (impl.posTimes.size() + impl.rotTimes.size()) == 0;
            break;
        }
        case MotionForm::Segments: {
            Vector<MotionSegment> segments = spec.segments;
            std::stable_sort(
                segments.begin(), segments.end(),
                [](const MotionSegment& a, const MotionSegment& b) { return a.from_s < b.from_s; });
            f64 previousEnd = -std::numeric_limits<f64>::infinity();
            for (const auto& seg : segments) {
                if (seg.to_s <= seg.from_s) {
                    warn("motion segment [" + std::to_string(seg.from_s) + ", " +
                         std::to_string(seg.to_s) + "] ends before it starts; ignored");
                    continue;
                }
                if (seg.from_s < previousEnd) {
                    warn("motion segment starting at " + std::to_string(seg.from_s) +
                         " s overlaps the previous one, which wins; ignored");
                    continue;
                }
                Vector<String> expressions{
                    seg.hasPosition ? seg.position[0] : String("0"),
                    seg.hasPosition ? seg.position[1] : String("0"),
                    seg.hasPosition ? seg.position[2] : String("0"),
                    seg.hasRotation ? seg.rotationEulerDeg[0] : String("0"),
                    seg.hasRotation ? seg.rotationEulerDeg[1] : String("0"),
                    seg.hasRotation ? seg.rotationEulerDeg[2] : String("0"),
                };
                auto compiled = SegmentExpressions::Compile(expressions);
                if (!compiled) {
                    warn("motion segment [" + std::to_string(seg.from_s) + ", " +
                         std::to_string(seg.to_s) + "]: " + compiled.error() + "; segment ignored");
                    continue;
                }
                Impl::CompiledSegment out;
                out.from_s = seg.from_s;
                out.to_s = seg.to_s;
                out.hasPosition = seg.hasPosition;
                out.hasRotation = seg.hasRotation;
                out.expressions = std::move(*compiled);
                impl.segments.push_back(std::move(out));
                previousEnd = seg.to_s;
            }
            if (impl.segments.empty()) {
                return Result<std::unique_ptr<MotionTrack>, String>::Err(
                    "motion has no segment that could be compiled");
            }
            impl.hasDomain = true;
            impl.domainFrom_s = impl.segments.front().from_s;
            impl.domainTo_s = impl.segments.back().to_s;
            impl.authoredEnd_s = impl.domainTo_s;
            impl.isStatic = false;
            break;
        }
        case MotionForm::Engine: {
            if (spec.location.type == MotionLocationType::None &&
                spec.orientation.type == MotionOrientationType::None) {
                return Result<std::unique_ptr<MotionTrack>, String>::Err(
                    "motion names neither a location engine nor an orientation engine");
            }
            impl.isStatic = false;
            break;
        }
        case MotionForm::None:
            impl.isStatic = true;
            break;
    }

    return Result<std::unique_ptr<MotionTrack>, String>(std::move(track));
}

glm::dvec3 MotionTrack::PositionAt(f64 t_s) const {
    const Impl& impl = *m_impl;
    switch (impl.form) {
        case MotionForm::Keys:
            return impl.SamplePosition(impl.RemapTime(t_s));
        case MotionForm::Segments: {
            const f64 t = impl.RemapTime(t_s);
            const auto* seg = impl.SegmentAt(t);
            if (seg == nullptr || !seg->hasPosition) return glm::dvec3(0.0);
            f64 values[6] = {};
            seg->expressions->Evaluate(t, t - seg->from_s, values);
            return glm::dvec3(values[0], values[1], values[2]);
        }
        case MotionForm::Engine:
            return impl.EnginePosition(t_s);
        case MotionForm::None:
            break;
    }
    return glm::dvec3(0.0);
}

glm::dvec3 MotionTrack::VelocityAt(f64 t_s) const {
    const Impl& impl = *m_impl;
    if (impl.form == MotionForm::Engine) return impl.EngineVelocity(t_s);
    if (impl.form == MotionForm::None) return glm::dvec3(0.0);
    const glm::dvec3 back = PositionAt(t_s - kFiniteDifference_s);
    const glm::dvec3 forward = PositionAt(t_s + kFiniteDifference_s);
    return (forward - back) / (2.0 * kFiniteDifference_s);
}

glm::mat4 MotionTrack::Evaluate(f64 t_s) const {
    const Impl& impl = *m_impl;
    glm::dvec3 position(0.0);
    glm::dquat rotation(1.0, 0.0, 0.0, 0.0);

    switch (impl.form) {
        case MotionForm::Keys: {
            const f64 t = impl.RemapTime(t_s);
            position = impl.SamplePosition(t);
            rotation = impl.SampleRotation(t);
            break;
        }
        case MotionForm::Segments: {
            const f64 t = impl.RemapTime(t_s);
            const auto* seg = impl.SegmentAt(t);
            if (seg != nullptr) {
                f64 values[6] = {};
                seg->expressions->Evaluate(t, t - seg->from_s, values);
                if (seg->hasPosition) position = glm::dvec3(values[0], values[1], values[2]);
                if (seg->hasRotation) {
                    rotation = EulerDegreesToQuat(glm::dvec3(values[3], values[4], values[5]));
                }
            }
            break;
        }
        case MotionForm::Engine:
            position = impl.EnginePosition(t_s);
            rotation = impl.EngineOrientation(t_s, position);
            break;
        case MotionForm::None:
            break;
    }

    glm::dmat4 out = glm::mat4_cast(rotation);
    out[3] = glm::dvec4(position, 1.0);
    return glm::mat4(out);
}

Vector<f64> MotionTrack::ChangeTimes(f64 from_s, f64 to_s) const {
    const Impl& impl = *m_impl;
    Vector<f64> times;
    auto keep = [&](f64 t) {
        if (t >= from_s && t <= to_s) times.push_back(t);
    };

    if (impl.form == MotionForm::Keys) {
        for (const f64 t : impl.posTimes) keep(t);
        for (const f64 t : impl.rotTimes) keep(t);
    } else if (impl.form == MotionForm::Segments) {
        for (const auto& seg : impl.segments) {
            keep(seg.from_s);
            keep(seg.to_s);
        }
    }

    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    return times;
}

bool MotionTrack::IsStatic() const { return m_impl->isStatic; }

f64 MotionTrack::AuthoredEnd_s() const { return m_impl->authoredEnd_s; }

// ============================================================================
// PoseDisplacement
// ============================================================================

f32 PoseDisplacement(const glm::mat4& a, const glm::mat4& b, f32 boundRadius_m) {
    const glm::vec3 da = glm::vec3(b[3]) - glm::vec3(a[3]);
    f32 moved = glm::length(da);

    // The rotation part, read straight off the upper 3x3. These are rigid, so
    // the columns are orthonormal and the quaternion cast is exact; a scale
    // that slipped in would only make the angle nonsense in the last digits,
    // and this feeds a threshold rather than a transform.
    const glm::quat qa = glm::quat_cast(glm::mat3(a));
    const glm::quat qb = glm::quat_cast(glm::mat3(b));
    f32 cosHalf = std::abs(glm::dot(qa, qb));
    cosHalf = std::min(cosHalf, 1.0f);
    const f32 angle = 2.0f * std::acos(cosHalf);
    moved += boundRadius_m * angle;
    return moved;
}

}  // namespace quantiloom::scene
