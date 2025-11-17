#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <string>

// ============================================================================
// Mesh - Triangle mesh geometry
// ============================================================================
// Contains:
// - Vertex positions, normals, UVs
// - Triangle indices
// - Material ID assignment
//
// Memory layout:
// - All data stored in CPU memory (std::vector)
// - Upload to GPU buffers happens in Renderer (via Upload helper)
// - Suitable for BLAS construction in Vulkan Ray Tracing
// ============================================================================

namespace quantiloom {

struct Mesh {
    // Geometry data
    std::vector<glm::vec3> positions;  // Vertex positions (world space)
    std::vector<glm::vec3> normals;    // Vertex normals (normalized)
    std::vector<glm::vec2> uvs;        // Texture coordinates [0, 1]
    std::vector<u32> indices;          // Triangle indices (3 per triangle)

    // Metadata
    String name;       // Mesh name (for debugging)
    u32 materialId = 0;  // Index into Scene::materials

    // ========================================================================
    // Utilities
    // ========================================================================

    // Get number of vertices
    u32 GetVertexCount() const {
        return static_cast<u32>(positions.size());
    }

    // Get number of triangles
    u32 GetTriangleCount() const {
        return static_cast<u32>(indices.size()) / 3;
    }

    // Check if mesh is valid
    bool IsValid() const {
        // Must have at least 3 vertices forming 1 triangle
        if (positions.size() < 3 || indices.size() < 3) {
            return false;
        }

        // Normals and UVs must match vertex count (if present)
        if (!normals.empty() && normals.size() != positions.size()) {
            return false;
        }
        if (!uvs.empty() && uvs.size() != positions.size()) {
            return false;
        }

        // Indices must be divisible by 3
        if (indices.size() % 3 != 0) {
            return false;
        }

        return true;
    }

    // Compute bounding box (AABB)
    void ComputeBounds(glm::vec3& outMin, glm::vec3& outMax) const {
        if (positions.empty()) {
            outMin = outMax = glm::vec3(0.0f);
            return;
        }

        outMin = positions[0];
        outMax = positions[0];

        for (const auto& pos : positions) {
            outMin = glm::min(outMin, pos);
            outMax = glm::max(outMax, pos);
        }
    }
};

} // namespace quantiloom
