#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>

// ============================================================================
// GeometryPrimitive - Minimal rendering unit (single draw call)
// ============================================================================
// Represents a contiguous block of geometry with:
// - Vertex attributes (positions, normals, UVs)
// - Triangle indices
// - Single material binding
//
// In glTF terminology:
// - This maps to a single glTF "primitive" (subset of a mesh)
// - Each primitive has its own material and vertex attributes
//
// In Vulkan Ray Tracing:
// - Each primitive becomes one BLAS (Bottom-Level Acceleration Structure)
// - BLAS is instanced in TLAS with transform and material ID
//
// Memory layout:
// - All data stored in CPU memory (std::vector)
// - Upload to GPU happens in AccelerationStructure::BuildBLAS()
// ============================================================================

namespace quantiloom {

struct GeometryPrimitive {
    // Vertex attributes
    std::vector<glm::vec3> positions;  // Vertex positions (object space)
    std::vector<glm::vec3> normals;    // Vertex normals (normalized, object space)
    std::vector<glm::vec2> uvs;        // Texture coordinates [0, 1]

    // Triangle indices (3 per triangle)
    std::vector<u32> indices;

    // Material binding
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

    // Check if primitive is valid
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

        // All indices must be in valid range
        for (u32 idx : indices) {
            if (idx >= positions.size()) {
                return false;
            }
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

// ============================================================================
// Mesh - Container for multiple geometry primitives
// ============================================================================
// Represents a logical mesh object composed of one or more primitives.
// Each primitive can have its own material and vertex attributes.
//
// In glTF terminology:
// - This maps directly to a glTF "mesh" object
// - A mesh contains an array of "primitives"
//
// For procedural geometry:
// - Simple objects (sphere, cube) have only one primitive
// - Complex objects can be split into multiple primitives for different materials
//
// Lifecycle:
// - Created during scene loading (GltfLoader or SceneBuilder)
// - Stored in Scene::meshes
// - Referenced by SceneNode via meshIndex
// ============================================================================

struct Mesh {
    // Geometry primitives (at least one required)
    std::vector<GeometryPrimitive> primitives;

    // Metadata
    String name;  // Mesh name (for debugging)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if mesh is valid
    bool IsValid() const {
        // Must have at least one primitive
        if (primitives.empty()) {
            return false;
        }

        // All primitives must be valid
        for (const auto& prim : primitives) {
            if (!prim.IsValid()) {
                return false;
            }
        }

        return true;
    }

    // Get total number of primitives
    u32 GetPrimitiveCount() const {
        return static_cast<u32>(primitives.size());
    }

    // Get total triangle count (across all primitives)
    u32 GetTotalTriangleCount() const {
        u32 total = 0;
        for (const auto& prim : primitives) {
            total += prim.GetTriangleCount();
        }
        return total;
    }

    // Get total vertex count (across all primitives)
    u32 GetTotalVertexCount() const {
        u32 total = 0;
        for (const auto& prim : primitives) {
            total += prim.GetVertexCount();
        }
        return total;
    }

    // Compute bounding box (AABB) for entire mesh
    void ComputeBounds(glm::vec3& outMin, glm::vec3& outMax) const {
        if (primitives.empty()) {
            outMin = outMax = glm::vec3(0.0f);
            return;
        }

        primitives[0].ComputeBounds(outMin, outMax);

        for (size_t i = 1; i < primitives.size(); ++i) {
            glm::vec3 primMin, primMax;
            primitives[i].ComputeBounds(primMin, primMax);
            outMin = glm::min(outMin, primMin);
            outMax = glm::max(outMax, primMax);
        }
    }
};

// ============================================================================
// SceneNode - Instance of a mesh in the scene with transform
// ============================================================================
// Represents a placed instance of a mesh in the scene hierarchy.
// Each node references a mesh and applies a local transform.
//
// In glTF terminology:
// - This maps to a glTF "node" object
// - In full glTF, nodes can form a tree hierarchy
// - For M2, we use a simplified flat list (all transforms are world-space)
//
// In Vulkan Ray Tracing:
// - Each node creates one or more TLAS instances (one per primitive in the mesh)
// - Transform is uploaded to TLAS instance data
//
// Future (M3+):
// - Add parent/child hierarchy support
// - Add animation/skinning support
// ============================================================================

struct SceneNode {
    u32 meshIndex = 0;  // Index into Scene::meshes
    glm::mat4 transform = glm::mat4(1.0f);  // Local-to-world transform

    // Metadata
    String name;  // Node name (for debugging)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if node is valid
    bool IsValid() const {
        // Transform matrix should not contain NaN or Inf
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::isnan(transform[i][j]) || std::isinf(transform[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }
};

} // namespace quantiloom
