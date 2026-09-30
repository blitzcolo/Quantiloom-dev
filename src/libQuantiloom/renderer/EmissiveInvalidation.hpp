#pragma once

#include "scene/Scene.hpp"

namespace quantiloom::rendercore {

/// The material classification used by the sampled emitter list. Textured
/// emission is reached by BSDF sampling and contributes no entry to that list.
inline bool HasSampledEmission(const Material& material) {
    constexpr glm::vec3 luminance{0.2126f, 0.7152f, 0.0722f};
    return material.emissiveTextureIndex < 0 &&
           glm::dot(material.emissiveFactor, luminance) > 0.0f;
}

inline bool NodeHasSampledEmission(const Scene& scene, const SceneNode& node) {
    if (node.meshIndex >= scene.meshes.size()) return false;
    for (const auto& primitive : scene.meshes[node.meshIndex].primitives) {
        if (primitive.materialId < scene.materials.size() &&
            HasSampledEmission(scene.materials[primitive.materialId])) return true;
    }
    return false;
}

inline bool SampledEmissionChanged(const Material& previous, const Material& next) {
    return previous.emissiveFactor != next.emissiveFactor ||
           previous.emissiveTextureIndex != next.emissiveTextureIndex ||
           previous.emissiveRadianceCurveIndex != next.emissiveRadianceCurveIndex;
}

} // namespace quantiloom::rendercore
