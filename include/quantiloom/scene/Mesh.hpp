/**
 * @file Mesh.hpp
 * @brief Geometry primitives and mesh containers for ray tracing scenes
 *
 * Provides core geometry data structures:
 * - GeometryPrimitive: Single renderable unit (vertex/index data + material)
 * - Mesh: Container for multiple primitives (logical mesh object)
 * - SceneNode: Mesh instance with world-space transform
 *
 * Geometry hierarchy:
 * @code
 * SceneNode (instance)
 *   └── meshIndex → Mesh (logical object)
 *       └── primitives[] → GeometryPrimitive[] (draw calls)
 *           └── materialId → Material
 * @endcode
 *
 * Ray tracing mapping:
 * - Each GeometryPrimitive → one BLAS (Bottom-Level Acceleration Structure)
 * - Each SceneNode → one TLAS instance (references BLAS with transform)
 * - Material ID embedded in TLAS instance for shader access
 *
 * Vertex attributes:
 * - Positions (vec3): Required, object-space coordinates
 * - Normals (vec3): Optional, smooth shading normals
 * - UVs (vec2): Optional, texture coordinates [0,1]
 * - Tangents (vec4): Optional, normal mapping (xyz=tangent, w=handedness ±1)
 *
 * @note All vertex data stored CPU-side (uploaded to GPU during BLAS build)
 * @note Indices use u32 type (supports up to 4B vertices per primitive)
 * @note Memory layout: Structure-of-Arrays (separate vectors per attribute)
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <algorithm>

// ============================================================================
// GeometryPrimitive - Minimal rendering unit (single draw call)
// ============================================================================

namespace quantiloom {

/**
 * @struct GeometryPrimitive
 * @brief Single contiguous geometry block with vertex attributes and one material
 *
 * Represents the smallest renderable unit in Quantiloom (equivalent to one draw call).
 * Each primitive contains:
 * - Vertex attribute arrays (positions, normals, UVs, tangents)
 * - Triangle index buffer
 * - Material reference
 *
 * In glTF terminology:
 * - Maps to a single glTF "primitive" (subset of a mesh with one material)
 *
 * In Vulkan Ray Tracing:
 * - Each primitive becomes one BLAS (Bottom-Level Acceleration Structure)
 * - BLAS instanced in TLAS with transform and material ID
 *
 * Usage example:
 * @code
 * GeometryPrimitive cube;
 * cube.positions = { 8 vertices };
 * cube.indices = { 36 indices (12 triangles) };
 * cube.normals = { 8 normals };
 * cube.uvs = { 8 UVs };
 * cube.materialId = 0;  // References scene.materials[0]
 *
 * // Build BLAS from primitive
 * BLAS blas(context, cube);
 * blas.Build(cmd);
 * @endcode
 *
 * @note Positions are required, all other attributes optional
 * @note If normals empty, flat shading used (face normals computed from triangles)
 * @note If UVs empty, texture sampling falls back to vertex colors
 * @note If tangents empty, normal mapping disabled
 *
 * @see Mesh for container of multiple primitives
 * @see BLAS for GPU acceleration structure
 */
struct GeometryPrimitive {
    // Vertex attributes
    std::vector<glm::vec3> positions;  // Vertex positions (object space)
    std::vector<glm::vec3> normals;    // Vertex normals (normalized, object space)
    std::vector<glm::vec2> uvs;        // Texture coordinates [0, 1]
    std::vector<glm::vec4> tangents;   // Tangent vectors (xyz = tangent, w = handedness ±1)

    // Triangle indices (3 per triangle)
    std::vector<u32> indices;

    // Material binding
    u32 materialId = 0;  // Index into Scene::materials

    // ========================================================================
    // Utilities
    // ========================================================================

    // Get number of vertices
    [[nodiscard]] u32 GetVertexCount() const {
        return static_cast<u32>(positions.size());
    }

    // Get number of triangles
    [[nodiscard]] u32 GetTriangleCount() const {
        return static_cast<u32>(indices.size()) / 3;
    }

    // Check if primitive has tangent data
    [[nodiscard]] bool HasTangents() const {
        return !tangents.empty();
    }

    // Check if primitive is valid
    [[nodiscard]] bool IsValid() const {
        // Must have at least 3 vertices forming 1 triangle
        if (positions.size() < 3 || indices.size() < 3) {
            return false;
        }

        // Normals, UVs, and tangents must match vertex count (if present)
        if (!normals.empty() && normals.size() != positions.size()) {
            return false;
        }
        if (!uvs.empty() && uvs.size() != positions.size()) {
            return false;
        }
        if (!tangents.empty() && tangents.size() != positions.size()) {
            return false;
        }

        // Indices must be divisible by 3
        if (indices.size() % 3 != 0) {
            return false;
        }

        // All indices must be in valid range
        if (!std::ranges::all_of(indices, [this](const u32 idx) {
            return idx < positions.size();
        })) {
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
    [[nodiscard]] bool IsValid() const {
        // Must have at least one primitive
        if (primitives.empty()) {
            return false;
        }

        // All primitives must be valid
        return std::ranges::all_of(primitives, [](const auto& prim) {
            return prim.IsValid();
        });
    }

    // Get total number of primitives
    [[nodiscard]] u32 GetPrimitiveCount() const {
        return static_cast<u32>(primitives.size());
    }

    // Get total triangle count (across all primitives)
    [[nodiscard]] u32 GetTotalTriangleCount() const {
        u32 total = 0;
        for (const auto& prim : primitives) {
            total += prim.GetTriangleCount();
        }
        return total;
    }

    // Get total vertex count (across all primitives)
    [[nodiscard]] u32 GetTotalVertexCount() const {
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

    // Tombstone for runtime removal. A removed node stays in Scene::nodes
    // with active == false so that node indices held by callers (selection,
    // undo stacks, instance->node maps) never shift; RestoreNode flips it
    // back for undo. Inactive nodes contribute no TLAS instances.
    bool active = true;

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if node is valid
    [[nodiscard]] bool IsValid() const {
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
