/**
 * @file ThermalMesh.cpp
 * @brief The scene's triangles, as things that exchange heat
 */

#include "thermal/ThermalMesh.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace quantiloom::thermal {

namespace {

/// A tenth of a millimetre. Two vertices of one object that are meant to be
/// the same vertex are usually bit-identical -- they came off the same source
/// position through the same transform -- so this only has to catch the case
/// where a mesh was authored with the seam duplicated.
constexpr f64 kWeldGrid_m = 1.0e-4;

struct EdgeKey {
    i64 a[3] = {0, 0, 0};
    i64 b[3] = {0, 0, 0};

    bool operator==(const EdgeKey& other) const {
        return a[0] == other.a[0] && a[1] == other.a[1] && a[2] == other.a[2] &&
               b[0] == other.b[0] && b[1] == other.b[1] && b[2] == other.b[2];
    }
};

struct EdgeKeyHash {
    usize operator()(const EdgeKey& key) const {
        usize h = 1469598103934665603ull;  // FNV-1a
        for (const i64 v : {key.a[0], key.a[1], key.a[2], key.b[0], key.b[1], key.b[2]}) {
            h ^= static_cast<usize>(v);
            h *= 1099511628211ull;
        }
        return h;
    }
};

/// The edge between two positions, snapped to the weld grid and put in a
/// canonical order so that the two triangles sharing it produce one key.
EdgeKey MakeEdgeKey(const glm::vec3& p, const glm::vec3& q) {
    const auto snap = [](const f32 v) {
        return static_cast<i64>(std::llround(static_cast<f64>(v) / kWeldGrid_m));
    };
    i64 first[3] = {snap(p.x), snap(p.y), snap(p.z)};
    i64 second[3] = {snap(q.x), snap(q.y), snap(q.z)};

    const bool swap = std::lexicographical_compare(std::begin(second), std::end(second),
                                                   std::begin(first), std::end(first));
    EdgeKey key;
    for (i32 i = 0; i < 3; ++i) {
        key.a[i] = swap ? second[i] : first[i];
        key.b[i] = swap ? first[i] : second[i];
    }
    return key;
}

}  // namespace

ThermalMesh BuildThermalMesh(const Scene& scene, const ThermalMeshOptions& options) {
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

    // Which element owns each edge, within one object. Scoped to the node
    // rather than to the whole scene, so a box resting on the ground does not
    // acquire a conduction path into it: heat does cross a contact, but
    // through a contact conductance nobody has supplied, and joining two
    // objects because they touch would be inventing one.
    constexpr u32 kEdgeTaken = 0xFFFFFFFFu;
    std::unordered_map<EdgeKey, u32, EdgeKeyHash> edgeOwner;

    for (usize nodeIndex = 0; nodeIndex < scene.nodes.size(); ++nodeIndex) {
        const SceneNode& node = scene.nodes[nodeIndex];
        edgeOwner.clear();
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

                if (!options.contacts || doubleArea <= 0.0f) continue;

                // The three edges of the triangle just added. The first
                // triangle to present an edge takes it; the second joins to it
                // and marks the edge used, so a third only counts.
                const u32 index = static_cast<u32>(mesh.elements.size() - 1);
                const glm::vec3 corners[3] = {p0, p1, p2};
                for (i32 e = 0; e < 3; ++e) {
                    const glm::vec3& u = corners[e];
                    const glm::vec3& v = corners[(e + 1) % 3];
                    const f32 edgeLength = glm::length(v - u);
                    if (!(edgeLength > 0.0f)) continue;

                    auto [slot, inserted] = edgeOwner.try_emplace(MakeEdgeKey(u, v), index);
                    if (inserted) continue;
                    if (slot->second == kEdgeTaken) {
                        ++mesh.nonManifoldEdgeCount;
                        continue;
                    }

                    const u32 other = slot->second;
                    slot->second = kEdgeTaken;

                    ThermalContact contact;
                    contact.a = other;
                    contact.b = index;
                    contact.sharedEdge_m = edgeLength;
                    contact.centroidDistance_m =
                        glm::length(element.centroid - mesh.elements[other].centroid);
                    mesh.contacts.push_back(contact);
                }
            }
        }
    }

    return mesh;
}

CsrMatrix BuildLateralConduction(const ThermalMesh& mesh,
                                 const Vector<ThermalMaterial>& materials) {
    CsrMatrix lateral;
    if (mesh.contacts.empty() || mesh.elements.empty()) return lateral;

    const usize n = mesh.elements.size();

    // The conductance of each surviving contact, and how many each element has.
    Vector<f32> conductance(mesh.contacts.size(), 0.0f);
    Vector<u32> degree(n, 0u);
    usize kept = 0;

    for (usize c = 0; c < mesh.contacts.size(); ++c) {
        const ThermalContact& contact = mesh.contacts[c];
        if (contact.a >= n || contact.b >= n) continue;
        if (!(contact.centroidDistance_m > 0.0f) || !(contact.sharedEdge_m > 0.0f)) continue;

        const u32 idA = mesh.elements[contact.a].materialId;
        const u32 idB = mesh.elements[contact.b].materialId;
        if (idA >= materials.size() || idB >= materials.size()) continue;
        const f64 kA = materials[idA].conductivity_W_mK;
        const f64 kB = materials[idB].conductivity_W_mK;
        if (!(kA > 0.0) || !(kB > 0.0)) continue;

        // Two conductors in series across the join, so the harmonic mean --
        // the arithmetic one would let a conductor next to an insulator carry
        // half the insulator's own heat away.
        const f64 harmonic = 2.0 * kA * kB / (kA + kB);
        conductance[c] = static_cast<f32>(harmonic * contact.sharedEdge_m /
                                          contact.centroidDistance_m);
        ++degree[contact.a];
        ++degree[contact.b];
        ++kept;
    }

    if (kept == 0) return lateral;

    lateral.rowStart.assign(n + 1, 0u);
    for (usize e = 0; e < n; ++e) {
        lateral.rowStart[e + 1] = lateral.rowStart[e] + degree[e];
    }
    lateral.column.assign(lateral.rowStart[n], 0u);
    lateral.value.assign(lateral.rowStart[n], 0.0f);

    Vector<u32> cursor(lateral.rowStart.begin(), lateral.rowStart.end() - 1);
    for (usize c = 0; c < mesh.contacts.size(); ++c) {
        if (conductance[c] <= 0.0f) continue;
        const ThermalContact& contact = mesh.contacts[c];
        lateral.column[cursor[contact.a]] = contact.b;
        lateral.value[cursor[contact.a]] = conductance[c];
        ++cursor[contact.a];
        lateral.column[cursor[contact.b]] = contact.a;
        lateral.value[cursor[contact.b]] = conductance[c];
        ++cursor[contact.b];
    }

    return lateral;
}

f64 LateralTimeConstantSeconds(const CsrMatrix& lateral,
                               const Vector<ThermalElement>& elements,
                               const Vector<ThermalMaterial>& materials) {
    f64 shortest = std::numeric_limits<f64>::infinity();
    if (lateral.RowCount() != elements.size()) return shortest;

    for (usize e = 0; e < elements.size(); ++e) {
        const u32 id = elements[e].materialId;
        if (id >= materials.size()) continue;
        const ThermalMaterial& material = materials[id];
        if (!material.ParticipatesInSolve()) continue;
        if (!(elements[e].area_m2 > 0.0f)) continue;

        f64 sum = 0.0;
        for (u32 i = lateral.rowStart[e]; i < lateral.rowStart[e + 1]; ++i) {
            sum += lateral.value[i];
        }
        if (!(sum > 0.0)) continue;

        // The slab's thickness cancels: a node's own cell height scales its
        // capacity and its lateral conductance alike.
        const f64 rhoC = static_cast<f64>(material.density_kg_m3) *
                         material.specificHeat_J_kgK;
        shortest = std::min(shortest, rhoC * elements[e].area_m2 / sum);
    }
    return shortest;
}

}  // namespace quantiloom::thermal
