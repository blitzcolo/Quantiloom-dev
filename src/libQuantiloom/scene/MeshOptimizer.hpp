/**
 * @file MeshOptimizer.hpp
 * @brief Mesh optimization utilities for vertex deduplication and cache optimization
 *
 * Provides static utility functions for mesh optimization:
 * - DeduplicateVertices: Removes duplicate vertices to reduce memory and improve cache efficiency
 * - OptimizeVertexCache: Reorders indices for better GPU vertex cache utilization (future)
 *
 * Vertex deduplication is especially important for USD files where face-varying
 * attributes (normals, UVs) cause geometry expansion, creating many duplicate vertices.
 *
 * Example before deduplication (simple quad with 2 triangles):
 *   Vertices: 6 (3 per triangle, shared edge duplicated)
 *   After: 4 unique vertices
 *
 * OldAttic scene estimate:
 *   Before: ~10M vertices (face-varying expanded)
 *   After: ~6-7M vertices (30-40% reduction)
 *
 * Usage:
 * @code
 * GeometryPrimitive primitive;
 * // ... fill primitive with positions, normals, uvs, indices ...
 *
 * MeshOptimizer::DeduplicateVertices(primitive);
 * // primitive now has optimized vertex/index buffers
 * @endcode
 *
 * @note Deduplication preserves triangle winding order
 * @note Vertices with different normals/UVs at same position are NOT merged (hard edge/UV seam)
 * @note Material ID is preserved
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "scene/Mesh.hpp"
#include <vector>

namespace quantiloom {

/**
 * @struct MeshOptimizationStats
 * @brief Statistics from mesh optimization operations
 */
struct MeshOptimizationStats {
    u32 originalVertexCount = 0;    ///< Vertex count before optimization
    u32 optimizedVertexCount = 0;   ///< Vertex count after optimization
    u32 originalIndexCount = 0;     ///< Index count before optimization
    u32 optimizedIndexCount = 0;    ///< Index count after optimization (unchanged for dedup)
    f32 vertexReductionPercent = 0.0f;  ///< Percentage of vertices removed

    /// Check if any optimization was performed
    [[nodiscard]] bool WasOptimized() const {
        return optimizedVertexCount < originalVertexCount;
    }
};

/**
 * @class MeshOptimizer
 * @brief Static utility class for mesh optimization operations
 *
 * Provides vertex deduplication and other mesh optimization utilities.
 * All operations modify the primitive in-place for efficiency.
 *
 * Vertex deduplication algorithm:
 * 1. Create hash key from (position, normal, uv, tangent) tuple
 * 2. Build hash map of unique vertices
 * 3. Remap indices to point to unique vertices
 * 4. Replace vertex arrays with deduplicated versions
 *
 * Hash key design:
 * - Uses spatial hashing with epsilon tolerance for floating-point comparison
 * - Different normals at same position = different vertex (hard edge)
 * - Different UVs at same position = different vertex (UV seam)
 * - Tangent differences also create separate vertices
 *
 * Thread safety:
 * - All static methods are thread-safe (no shared state)
 * - Can be called concurrently on different primitives
 */
class MeshOptimizer {
public:
    // ========================================================================
    // Vertex Deduplication
    // ========================================================================

    /**
     * @brief Remove duplicate vertices from a geometry primitive
     *
     * Identifies vertices with identical (position, normal, uv, tangent) tuples
     * and merges them, updating the index buffer accordingly.
     *
     * This is especially useful after USD face-varying attribute expansion,
     * which creates many duplicate vertices at shared edges.
     *
     * @param primitive The geometry primitive to optimize (modified in-place)
     * @return Statistics about the optimization
     *
     * @note Does NOT merge vertices with same position but different attributes
     * @note Preserves triangle winding order
     * @note Material ID is preserved
     * @note Empty primitives are handled gracefully (no-op)
     */
    static MeshOptimizationStats DeduplicateVertices(GeometryPrimitive& primitive);

    /**
     * @brief Deduplicate vertices in all primitives of a mesh
     *
     * Convenience method that calls DeduplicateVertices on each primitive.
     *
     * @param mesh The mesh to optimize (modified in-place)
     * @return Combined statistics for all primitives
     */
    static MeshOptimizationStats DeduplicateMesh(Mesh& mesh);

    // ========================================================================
    // Vertex Cache Optimization (Future Enhancement)
    // ========================================================================

    /**
     * @brief Reorder indices for better vertex cache utilization
     *
     * Uses the Forsyth algorithm or similar to reorder triangle indices
     * for optimal GPU vertex cache usage. Can improve rendering performance
     * by 10-20% on cache-sensitive workloads.
     *
     * @param primitive The geometry primitive to optimize (modified in-place)
     *
     * @note Currently not implemented - placeholder for future enhancement
     */
    static void OptimizeVertexCache(GeometryPrimitive& primitive);

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * @brief Check if a primitive could benefit from deduplication
     *
     * Quick heuristic check without performing full deduplication.
     * Returns true if vertex count > index count / 3 (suggesting shared vertices).
     *
     * @param primitive The primitive to check
     * @return true if deduplication may reduce vertex count
     */
    [[nodiscard]] static bool ShouldDeduplicate(const GeometryPrimitive& primitive);

    /**
     * @brief Estimate memory savings from deduplication
     *
     * @param stats Optimization statistics from DeduplicateVertices
     * @return Estimated memory saved in bytes (assuming vec3 pos + vec3 norm + vec2 uv)
     */
    [[nodiscard]] static size_t EstimateMemorySaved(const MeshOptimizationStats& stats);

private:
    // Prevent instantiation
    MeshOptimizer() = delete;

    // ========================================================================
    // Internal Hash Key Structure
    // ========================================================================

    /**
     * @brief Hash key for vertex deduplication
     *
     * Combines position, normal, UV, and tangent into a single hashable structure.
     * Uses quantized values for stable floating-point comparison.
     */
    struct VertexKey {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 tangent;

        bool operator==(const VertexKey& other) const;
    };

    /**
     * @brief Hash functor for VertexKey
     */
    struct VertexKeyHash {
        size_t operator()(const VertexKey& key) const;
    };

    // ========================================================================
    // Internal Helper Functions
    // ========================================================================

    /**
     * @brief Quantize float for stable hashing
     *
     * Rounds to fixed precision to handle floating-point epsilon differences.
     * Default precision: 1e-5 (5 decimal places)
     */
    [[nodiscard]] static float Quantize(float value, float epsilon = 1e-5f);
};

} // namespace quantiloom
