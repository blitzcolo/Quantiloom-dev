/**
 * @file ThermalMesh.cpp
 * @brief The scene's triangles, as things that exchange heat
 */

#include "thermal/ThermalMesh.hpp"

#include <glm/gtc/matrix_inverse.hpp>

namespace quantiloom::thermal {

ThermalMesh BuildThermalMesh(const Scene& scene) {
    ThermalMesh mesh;

    // Primitives in walk order, so the instance loop below is a lookup rather
    // than a scan. Same table SceneGeometry builds, for the same reason.
    Vector<const GeometryPrimitive*> primitives;
    for (const auto& sceneMesh : scene.meshes) {
        for (const auto& prim : sceneMesh.primitives) {
            primitives.push_back(&prim);
        }
    }

    Vector<usize> meshToFirstPrim;
    meshToFirstPrim.reserve(scene.meshes.size());
    usize firstPrim = 0;
    for (const auto& sceneMesh : scene.meshes) {
        meshToFirstPrim.push_back(firstPrim);
        firstPrim += sceneMesh.primitives.size();
    }

    for (usize nodeIndex = 0; nodeIndex < scene.nodes.size(); ++nodeIndex) {
        const SceneNode& node = scene.nodes[nodeIndex];
        // A tombstoned node reserves its index and contributes no instances,
        // which is exactly what SceneGeometry does with it.
        if (!node.active) continue;
        if (node.meshIndex >= scene.meshes.size()) continue;

        const Mesh& sceneMesh = scene.meshes[node.meshIndex];
        const usize base = meshToFirstPrim[node.meshIndex];

        // The normal transform: a non-uniform scale rotates a normal
        // differently from a position, and a surface whose normal is wrong
        // absorbs the sun at the wrong angle.
        const glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(node.transform));

        for (usize prim = 0; prim < sceneMesh.primitives.size(); ++prim) {
            const GeometryPrimitive* primitive = primitives[base + prim];
            mesh.instanceElementBase.push_back(static_cast<u32>(mesh.elements.size()));
            if (!primitive) continue;

            for (usize i = 0; i + 2 < primitive->indices.size(); i += 3) {
                const glm::vec3 p0 = glm::vec3(
                    node.transform * glm::vec4(primitive->positions[primitive->indices[i]], 1.0f));
                const glm::vec3 p1 = glm::vec3(
                    node.transform * glm::vec4(primitive->positions[primitive->indices[i + 1]], 1.0f));
                const glm::vec3 p2 = glm::vec3(
                    node.transform * glm::vec4(primitive->positions[primitive->indices[i + 2]], 1.0f));

                const glm::vec3 edge1 = p1 - p0;
                const glm::vec3 edge2 = p2 - p0;
                const glm::vec3 cross = glm::cross(edge1, edge2);
                const f32 doubleArea = glm::length(cross);

                ThermalElement element;
                element.centroid = (p0 + p1 + p2) / 3.0f;
                element.area_m2 = 0.5f * doubleArea;
                element.materialId = primitive->materialId;

                if (doubleArea > 0.0f) {
                    // The geometric normal, not the shading one: what a
                    // triangle absorbs and radiates through is its own plane,
                    // and an interpolated normal would let a smooth-shaded
                    // sphere exchange more than its area.
                    element.normal = cross / doubleArea;

                    // Where the primitive brings vertex normals, they decide
                    // which side is the outside -- a triangle wound the other
                    // way is still the same surface.
                    if (!primitive->normals.empty()) {
                        const glm::vec3 authored = glm::normalize(
                            normalTransform * primitive->normals[primitive->indices[i]]);
                        if (glm::dot(authored, element.normal) < 0.0f) {
                            element.normal = -element.normal;
                        }
                    }
                } else {
                    ++mesh.degenerateCount;
                }

                mesh.elements.push_back(element);
            }
        }
    }

    return mesh;
}

}  // namespace quantiloom::thermal
