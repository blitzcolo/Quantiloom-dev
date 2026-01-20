/**
 * @file NormalGenerator.hpp
 * @brief Dihedral angle-based automatic normal generation for meshes
 *
 * Implements industry-standard "Auto Smooth" functionality (similar to Blender/Maya):
 * - Dihedral angle > threshold → Hard edge → Flat normal (vertex duplication)
 * - Dihedral angle ≤ threshold → Smooth edge → Area-weighted accumulation
 *
 * Algorithm:
 * 1. Build edge topology (O(F))
 * 2. Compute face normals (O(F))
 * 3. Detect hard edges via dihedral angle comparison (O(E))
 * 4. For each vertex, collect smooth-connected faces via flood fill
 * 5. Duplicate vertices at hard edge boundaries
 * 6. Compute area-weighted normals per smooth region
 *
 * @note Default threshold: 60° (industry standard)
 * @note Area weighting: cross product magnitude = 2 * triangle area
 *
 * @author wtflmao
 */

#pragma once

#include "Mesh.hpp"
#include <unordered_map>
#include <unordered_set>

namespace quantiloom {

/**
 * @struct NormalGenerationConfig
 * @brief Configuration for dihedral-based normal generation
 */
struct NormalGenerationConfig {
    float dihedralAngleThreshold = 60.0f;  // Degrees, edges sharper than this are hard
    bool areaWeighted = true;               // Use area-weighted normal accumulation
};

/**
 * @struct EdgeKey
 * @brief Canonical edge identifier (v0 < v1 for consistent hashing)
 */
struct EdgeKey {
    u32 v0, v1;

    EdgeKey(u32 a, u32 b) : v0(std::min(a, b)), v1(std::max(a, b)) {}

    bool operator==(const EdgeKey& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }
};

/**
 * @struct EdgeKeyHash
 * @brief Hash function for EdgeKey (64-bit combined hash)
 */
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& e) const {
        return std::hash<u64>{}(static_cast<u64>(e.v0) << 32 | e.v1);
    }
};

/**
 * @struct EdgeInfo
 * @brief Stores adjacent face indices for an edge
 */
struct EdgeInfo {
    u32 face0 = UINT32_MAX;  // First adjacent face
    u32 face1 = UINT32_MAX;  // Second adjacent face (UINT32_MAX = boundary edge)
};

/**
 * @class NormalGenerator
 * @brief Static utility for generating normals with dihedral angle-based hard/smooth edge detection
 *
 * Usage:
 * @code
 * GeometryPrimitive primitive = LoadMeshWithoutNormals();
 * NormalGenerator::GenerateWithDihedralAngle(primitive);
 * // primitive.normals now populated, vertices duplicated at hard edges
 * @endcode
 */
class NormalGenerator {
public:
    /**
     * @brief Main entry point: generates normals with dihedral-based hard/smooth detection
     *
     * If primitive already has normals, returns immediately (no-op).
     * May duplicate vertices for hard edges (modifies positions, uvs, tangents, indices).
     *
     * @param primitive Geometry to process (modified in-place)
     * @param config Generation parameters (threshold, weighting)
     */
    static void GenerateWithDihedralAngle(
        GeometryPrimitive& primitive,
        const NormalGenerationConfig& config = {}
    );

private:
    using EdgeTopology = std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash>;
    using HardEdgeSet = std::unordered_set<EdgeKey, EdgeKeyHash>;

    /**
     * @brief Build edge-to-faces adjacency map from triangle indices
     * @param indices Triangle index buffer (size must be multiple of 3)
     * @return Map from each edge to its adjacent face indices
     */
    static EdgeTopology BuildEdgeTopology(const std::vector<u32>& indices);

    /**
     * @brief Compute un-normalized face normals (cross product of edges)
     * @param positions Vertex positions
     * @param indices Triangle indices
     * @return Face normals (one per triangle, magnitude = 2 * area)
     */
    static std::vector<glm::vec3> ComputeFaceNormals(
        const std::vector<glm::vec3>& positions,
        const std::vector<u32>& indices
    );

    /**
     * @brief Find edges where dihedral angle exceeds threshold
     * @param topology Edge-to-faces map
     * @param faceNormals Normalized face normals
     * @param thresholdRadians Angle threshold in radians
     * @return Set of hard edges
     */
    static HardEdgeSet FindHardEdges(
        const EdgeTopology& topology,
        const std::vector<glm::vec3>& faceNormals,
        float thresholdRadians
    );

    /**
     * @brief Build vertex-to-adjacent-faces map
     * @param indices Triangle indices
     * @param vertexCount Number of vertices
     * @return For each vertex, list of face indices sharing that vertex
     */
    static std::vector<std::vector<u32>> BuildVertexToFaces(
        const std::vector<u32>& indices,
        size_t vertexCount
    );
};

} // namespace quantiloom
