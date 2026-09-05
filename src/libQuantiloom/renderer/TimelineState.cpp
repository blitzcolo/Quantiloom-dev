#include "renderer/TimelineState.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace quantiloom::rendercore {

namespace {

constexpr glm::vec3 kLuminance{0.2126f, 0.7152f, 0.0722f};

/// Radius of a sphere around the node in world units.
///
/// Half the diagonal of the mesh's object-space bounding box, scaled by the
/// largest the rest pose stretches an axis. It feeds a "did this move far
/// enough to matter" threshold, so an over-estimate costs an extra thermal
/// epoch and an under-estimate costs a missed one -- and over is the safe way
/// to be wrong.
f32 NodeBoundRadius(const Scene& scene, const SceneNode& node, const glm::mat4& rest) {
    if (node.meshIndex >= scene.meshes.size()) return 0.0f;

    glm::vec3 lo(std::numeric_limits<f32>::max());
    glm::vec3 hi(std::numeric_limits<f32>::lowest());
    bool any = false;
    for (const GeometryPrimitive& primitive : scene.meshes[node.meshIndex].primitives) {
        for (const glm::vec3& position : primitive.positions) {
            lo = glm::min(lo, position);
            hi = glm::max(hi, position);
            any = true;
        }
    }
    if (!any) return 0.0f;

    const f32 localRadius = 0.5f * glm::length(hi - lo);
    const f32 scale = std::max({glm::length(glm::vec3(rest[0])), glm::length(glm::vec3(rest[1])),
                                glm::length(glm::vec3(rest[2]))});
    return localRadius * scale;
}

/// Whether moving this node changes the emissive triangle list. The test
/// matches CollectEmissiveTriangles exactly, including its refusal of textured
/// emitters -- a node that is not in the list cannot change it by moving.
bool NodeEmits(const Scene& scene, const SceneNode& node) {
    if (node.meshIndex >= scene.meshes.size()) return false;
    for (const GeometryPrimitive& primitive : scene.meshes[node.meshIndex].primitives) {
        if (primitive.materialId >= scene.materials.size()) continue;
        const Material& material = scene.materials[primitive.materialId];
        if (material.emissiveTextureIndex >= 0) continue;
        if (glm::dot(material.emissiveFactor, kLuminance) > 0.0f) return true;
    }
    return false;
}

/// Compile one spec, routing whatever it complained about into the report.
std::shared_ptr<const scene::MotionTrack> CompileTrack(const scene::MotionSpec& spec,
                                                       const String& owner,
                                                       ConfigApplyReport& report) {
    Vector<String> warnings;
    auto compiled = scene::MotionTrack::Compile(spec, &warnings);
    for (const String& text : warnings) {
        QL_LOG_WARN("{}", text);
        report.messages.push_back(
            {ConfigApplyMessage::Severity::Warning, "motion", owner + ": " + text});
    }
    if (!compiled) {
        const String text = owner + ": " + compiled.error() + "; it will not move";
        QL_LOG_WARN("{}", text);
        report.messages.push_back({ConfigApplyMessage::Severity::Warning, "motion", text});
        return nullptr;
    }
    auto track = std::shared_ptr<const scene::MotionTrack>(std::move(*compiled));
    if (track->IsStatic()) return nullptr;
    ++report.motionTracks;
    return track;
}

}  // namespace

TimelineState TimelineState::Build(const Scene& scene, const TimelineConfig& config,
                                   const Vector<ModelEntry>& models, const SceneLoadInfo& info,
                                   const Vector<std::pair<u32, scene::MotionSpec>>& nodeMotion,
                                   ConfigApplyReport& report) {
    TimelineState state;
    state.m_config = config;
    state.m_current_s = config.present ? config.time_s : 0.0;
    state.m_modelCount = static_cast<u32>(models.size());

    Vector<std::shared_ptr<const scene::MotionTrack>> modelTracks(models.size());
    for (usize i = 0; i < models.size(); ++i) {
        if (!models[i].motion) continue;
        modelTracks[i] = CompileTrack(*models[i].motion, "models." + models[i].name, report);
    }

    std::unordered_map<u32, std::shared_ptr<const scene::MotionTrack>> nodeTracks;
    for (const auto& [nodeIndex, spec] : nodeMotion) {
        const String owner = (nodeIndex < scene.nodes.size())
                                 ? ("nodes." + scene.nodes[nodeIndex].name)
                                 : ("nodes[" + std::to_string(nodeIndex) + "]");
        if (auto track = CompileTrack(spec, owner, report)) {
            nodeTracks.emplace(nodeIndex, std::move(track));
        }
    }

    for (u32 i = 0; i < static_cast<u32>(scene.nodes.size()); ++i) {
        const u32 model = (i < info.nodeModel.size()) ? info.nodeModel[i] : SceneLoadInfo::kNoModel;

        std::shared_ptr<const scene::MotionTrack> modelTrack;
        if (model != SceneLoadInfo::kNoModel && model < modelTracks.size()) {
            modelTrack = modelTracks[model];
        }
        std::shared_ptr<const scene::MotionTrack> nodeTrack;
        if (const auto found = nodeTracks.find(i); found != nodeTracks.end()) {
            nodeTrack = found->second;
        }
        if (!modelTrack && !nodeTrack) continue;

        AnimatedNode animated;
        animated.node = i;
        animated.model = model;
        animated.modelTrack = std::move(modelTrack);
        animated.nodeTrack = std::move(nodeTrack);
        animated.modelRest = (model != SceneLoadInfo::kNoModel && model < models.size())
                                 ? models[model].rest
                                 : glm::mat4(1.0f);
        // R_node is whatever is left of the node's current world transform once
        // the model's rest pose is taken out of it. Taken now, once, so that
        // evaluating a pose never inverts a matrix.
        animated.nodeRest = glm::inverse(animated.modelRest) * scene.nodes[i].transform;
        animated.boundRadius_m = NodeBoundRadius(scene, scene.nodes[i], scene.nodes[i].transform);
        animated.emissive = NodeEmits(scene, scene.nodes[i]);
        state.m_animated.push_back(std::move(animated));
    }

    state.m_present = config.present || !state.m_animated.empty();
    state.m_epochCount = state.m_present ? 1u : 0u;

    report.timelinePresent = state.m_present;
    report.modelsLoaded = info.modelsLoaded;

    if (state.m_present) {
        QL_LOG_INFO("  Timeline: {} animated node(s) over {} model(s)", state.m_animated.size(),
                    state.m_modelCount);
    }
    return state;
}

glm::mat4 TimelineState::PoseAt(const AnimatedNode& node, f64 t_s) const {
    glm::mat4 out = node.modelRest;
    if (node.modelTrack) out = node.modelTrack->Evaluate(t_s) * out;
    if (node.nodeTrack) out = out * node.nodeTrack->Evaluate(t_s);
    return out * node.nodeRest;
}

Vector<u32> TimelineState::Apply(Scene& scene, f64 t_s) {
    m_current_s = t_s;

    Vector<u32> moved;
    for (const AnimatedNode& animated : m_animated) {
        if (animated.node >= scene.nodes.size()) continue;
        const glm::mat4 world = PoseAt(animated, t_s);
        glm::mat4& current = scene.nodes[animated.node].transform;
        // Bitwise, not approximate: two ticks that produce the same pose must
        // not cost a TLAS refit, and "the same pose" here means the same bits
        // because it came out of the same arithmetic.
        if (std::memcmp(&current, &world, sizeof(glm::mat4)) != 0) {
            current = world;
            moved.push_back(animated.node);
        }
    }
    return moved;
}

void TimelineState::SetWorldPose(u32 node, const glm::mat4& world) {
    for (AnimatedNode& animated : m_animated) {
        if (animated.node != node) continue;
        glm::mat4 lead = animated.modelRest;
        if (animated.modelTrack) lead = animated.modelTrack->Evaluate(m_current_s) * lead;
        if (animated.nodeTrack) lead = lead * animated.nodeTrack->Evaluate(m_current_s);
        animated.nodeRest = glm::inverse(lead) * world;
        return;
    }
}

glm::mat4 TimelineState::RestOf(const Scene& scene, u32 node) const {
    for (const AnimatedNode& animated : m_animated) {
        if (animated.node == node) return animated.modelRest * animated.nodeRest;
    }
    if (node < scene.nodes.size()) return scene.nodes[node].transform;
    return glm::mat4(1.0f);
}

f64 TimelineState::HourAt(f64 t_s) const {
    return m_thermalHourAtStart + (t_s - m_config.start_s) * m_config.thermalTimeScale / 3600.0;
}

void TimelineState::SetThermalMapping(f64 hourAtStart_h, f64 scale) {
    m_thermalHourAtStart = hourAtStart_h;
    m_config.thermalTimeScale = (scale > 0.0) ? scale : m_config.thermalTimeScale;
    m_thermalMapped = true;
}

void TimelineState::SetEpochCounts(u32 count, u32 current) {
    m_epochCount = count;
    m_currentEpoch = current;
}

TimelineInfo TimelineState::Info() const {
    TimelineInfo info;
    info.present = m_present;
    info.start_s = m_config.start_s;
    info.end_s = m_config.end_s;
    info.current_s = m_current_s;
    info.ticksPerSecond = m_config.ticksPerSecond;
    info.thermalMapped = m_thermalMapped;
    info.thermalHourAtStart = m_thermalHourAtStart;
    info.thermalTimeScale = m_config.thermalTimeScale;
    info.currentThermalHour = HourAt(m_current_s);
    info.animatedNodeCount = static_cast<u32>(m_animated.size());
    info.modelCount = m_modelCount;
    info.thermalEpochCount = m_epochCount;
    info.currentThermalEpoch = m_currentEpoch;
    return info;
}

Vector<f64> TimelineState::ChangeTimes(f64 from_s, f64 to_s) const {
    Vector<f64> times;
    for (const AnimatedNode& animated : m_animated) {
        if (animated.modelTrack) {
            const auto own = animated.modelTrack->ChangeTimes(from_s, to_s);
            times.insert(times.end(), own.begin(), own.end());
        }
        if (animated.nodeTrack) {
            const auto own = animated.nodeTrack->ChangeTimes(from_s, to_s);
            times.insert(times.end(), own.begin(), own.end());
        }
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    return times;
}

}  // namespace quantiloom::rendercore
