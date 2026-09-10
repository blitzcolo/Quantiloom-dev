/**
 * @file TangentGenerator.cpp
 * @brief Implementation of UV-derived tangent frames
 */

#include "scene/TangentGenerator.hpp"

#include "core/Log.hpp"

#include <cmath>

namespace quantiloom {

namespace {

/// Any unit vector perpendicular to n. Used only where the UVs said nothing.
glm::vec3 AnyPerpendicular(const glm::vec3& n) {
    const glm::vec3 axis = std::abs(n.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f)
                                                : glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::normalize(glm::cross(axis, n));
}

}  // namespace

bool TangentGenerator::FromUv(GeometryPrimitive& primitive) {
    const usize vertexCount = primitive.positions.size();
    if (vertexCount == 0 || primitive.uvs.size() != vertexCount ||
        primitive.normals.size() != vertexCount || primitive.indices.size() < 3) {
        return false;
    }

    std::vector<glm::vec3> alongU(vertexCount, glm::vec3(0.0f));
    std::vector<glm::vec3> alongV(vertexCount, glm::vec3(0.0f));

    const usize triangleCount = primitive.indices.size() / 3;
    usize degenerate = 0;

    for (usize triangle = 0; triangle < triangleCount; ++triangle) {
        const u32 i0 = primitive.indices[triangle * 3 + 0];
        const u32 i1 = primitive.indices[triangle * 3 + 1];
        const u32 i2 = primitive.indices[triangle * 3 + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
            continue;
        }

        const glm::vec3 edge1 = primitive.positions[i1] - primitive.positions[i0];
        const glm::vec3 edge2 = primitive.positions[i2] - primitive.positions[i0];
        const glm::vec2 duv1 = primitive.uvs[i1] - primitive.uvs[i0];
        const glm::vec2 duv2 = primitive.uvs[i2] - primitive.uvs[i0];

        // The determinant of the UV edge matrix: zero means the triangle is a
        // line or a point in texture space and says nothing about direction.
        const f32 determinant = duv1.x * duv2.y - duv2.x * duv1.y;
        if (std::abs(determinant) < 1e-12f) {
            ++degenerate;
            continue;
        }
        const f32 inverse = 1.0f / determinant;

        const glm::vec3 tangent = (edge1 * duv2.y - edge2 * duv1.y) * inverse;
        const glm::vec3 bitangent = (edge2 * duv1.x - edge1 * duv2.x) * inverse;

        // Unweighted accumulation: the triangle areas are already in the edge
        // vectors, so a large triangle contributes proportionally more.
        for (const u32 index : {i0, i1, i2}) {
            alongU[index] += tangent;
            alongV[index] += bitangent;
        }
    }

    primitive.tangents.resize(vertexCount);
    usize fallbacks = 0;

    for (usize vertex = 0; vertex < vertexCount; ++vertex) {
        const glm::vec3 normal = primitive.normals[vertex];
        glm::vec3 tangent = alongU[vertex];

        // Gram-Schmidt: the part of dP/du that lies in the surface. The shader
        // re-orthogonalises too, but a tangent parallel to the normal would
        // already have been lost by then.
        tangent -= normal * glm::dot(normal, tangent);
        const f32 length = glm::length(tangent);
        if (length > 1e-8f) {
            tangent /= length;
        } else {
            tangent = AnyPerpendicular(normal);
            ++fallbacks;
        }

        // Handedness, so a mirrored UV island's bitangent points the right way.
        const f32 handedness =
            glm::dot(glm::cross(normal, tangent), alongV[vertex]) < 0.0f ? -1.0f : 1.0f;

        primitive.tangents[vertex] = glm::vec4(tangent, handedness);
    }

    if (degenerate > 0) {
        QL_LOG_DEBUG("      {} of {} triangles have degenerate UVs and contributed no "
                     "tangent", degenerate, triangleCount);
    }
    if (fallbacks > 0) {
        QL_LOG_DEBUG("      {} of {} vertices fell back to an arbitrary tangent",
                     fallbacks, vertexCount);
    }

    return true;
}

}  // namespace quantiloom
