/**
 * @file MeshOptimizer.cpp
 * @brief Implementation of mesh optimization utilities
 *
 * @author blitzcolo
 */

#include "MeshOptimizer.hpp"
#include "core/Log.hpp"
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <functional>

namespace quantiloom {

// ============================================================================
// VertexKey Implementation
// ============================================================================

bool MeshOptimizer::VertexKey::operator==(const VertexKey& other) const {
    // Use exact comparison since values are pre-quantized
    return position == other.position &&
           normal == other.normal &&
           uv == other.uv &&
           tangent == other.tangent;
}

// ============================================================================
// VertexKeyHash Implementation
// ============================================================================

size_t MeshOptimizer::VertexKeyHash::operator()(const VertexKey& key) const {
    // Combine hashes of all components using FNV-1a inspired mixing
    auto hashFloat = [](float f) -> size_t {
        // Reinterpret as integer bits for stable hashing
        u32 bits;
        std::memcpy(&bits, &f, sizeof(f));
        return std::hash<u32>{}(bits);
    };

    auto hashVec2 = [&](const glm::vec2& v) -> size_t {
        return hashFloat(v.x) ^ (hashFloat(v.y) << 1);
    };

    auto hashVec3 = [&](const glm::vec3& v) -> size_t {
        return hashFloat(v.x) ^ (hashFloat(v.y) << 1) ^ (hashFloat(v.z) << 2);
    };

    auto hashVec4 = [&](const glm::vec4& v) -> size_t {
        return hashFloat(v.x) ^ (hashFloat(v.y) << 1) ^ (hashFloat(v.z) << 2) ^ (hashFloat(v.w) << 3);
    };

    size_t h = 0;
    h ^= hashVec3(key.position) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hashVec3(key.normal) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hashVec2(key.uv) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hashVec4(key.tangent) + 0x9e3779b9 + (h << 6) + (h >> 2);

    return h;
}

// ============================================================================
// Quantize Helper
// ============================================================================

float MeshOptimizer::Quantize(float value, float epsilon) {
    // Round to nearest multiple of epsilon
    return std::round(value / epsilon) * epsilon;
}

// ============================================================================
// DeduplicateVertices Implementation
// ============================================================================

MeshOptimizationStats MeshOptimizer::DeduplicateVertices(GeometryPrimitive& primitive) {
    MeshOptimizationStats stats;

    // Record original counts
    stats.originalVertexCount = static_cast<u32>(primitive.positions.size());
    stats.originalIndexCount = static_cast<u32>(primitive.indices.size());
    stats.optimizedIndexCount = stats.originalIndexCount;  // Index count unchanged

    // Handle empty or invalid primitive
    if (primitive.positions.empty() || primitive.indices.empty()) {
        stats.optimizedVertexCount = stats.originalVertexCount;
        return stats;
    }

    // Verify attribute array sizes match
    bool hasNormals = !primitive.normals.empty();
    bool hasUVs = !primitive.uvs.empty();
    bool hasTangents = !primitive.tangents.empty();

    if (hasNormals && primitive.normals.size() != primitive.positions.size()) {
        QL_LOG_WARN("MeshOptimizer: Normal count ({}) != position count ({}), skipping dedup",
                    primitive.normals.size(), primitive.positions.size());
        stats.optimizedVertexCount = stats.originalVertexCount;
        return stats;
    }
    if (hasUVs && primitive.uvs.size() != primitive.positions.size()) {
        QL_LOG_WARN("MeshOptimizer: UV count ({}) != position count ({}), skipping dedup",
                    primitive.uvs.size(), primitive.positions.size());
        stats.optimizedVertexCount = stats.originalVertexCount;
        return stats;
    }
    if (hasTangents && primitive.tangents.size() != primitive.positions.size()) {
        QL_LOG_WARN("MeshOptimizer: Tangent count ({}) != position count ({}), skipping dedup",
                    primitive.tangents.size(), primitive.positions.size());
        stats.optimizedVertexCount = stats.originalVertexCount;
        return stats;
    }

    // Build hash map of unique vertices
    std::unordered_map<VertexKey, u32, VertexKeyHash> uniqueVertices;
    uniqueVertices.reserve(primitive.positions.size() / 2);  // Estimate 50% unique

    // New optimized arrays
    std::vector<glm::vec3> newPositions;
    std::vector<glm::vec3> newNormals;
    std::vector<glm::vec2> newUVs;
    std::vector<glm::vec4> newTangents;
    std::vector<u32> newIndices;

    newPositions.reserve(primitive.positions.size() / 2);
    if (hasNormals) newNormals.reserve(primitive.positions.size() / 2);
    if (hasUVs) newUVs.reserve(primitive.positions.size() / 2);
    if (hasTangents) newTangents.reserve(primitive.positions.size() / 2);
    newIndices.reserve(primitive.indices.size());

    // Quantization epsilon for floating-point comparison
    constexpr float epsilon = 1e-5f;

    // Process each index, building unique vertex list
    for (u32 idx : primitive.indices) {
        if (idx >= primitive.positions.size()) {
            QL_LOG_ERROR("MeshOptimizer: Index {} out of bounds (max {})",
                         idx, primitive.positions.size() - 1);
            stats.optimizedVertexCount = stats.originalVertexCount;
            return stats;
        }

        // Build vertex key with quantized values for stable comparison
        VertexKey key;
        key.position = glm::vec3(
            Quantize(primitive.positions[idx].x, epsilon),
            Quantize(primitive.positions[idx].y, epsilon),
            Quantize(primitive.positions[idx].z, epsilon)
        );

        if (hasNormals) {
            key.normal = glm::vec3(
                Quantize(primitive.normals[idx].x, epsilon),
                Quantize(primitive.normals[idx].y, epsilon),
                Quantize(primitive.normals[idx].z, epsilon)
            );
        } else {
            key.normal = glm::vec3(0.0f);
        }

        if (hasUVs) {
            key.uv = glm::vec2(
                Quantize(primitive.uvs[idx].x, epsilon),
                Quantize(primitive.uvs[idx].y, epsilon)
            );
        } else {
            key.uv = glm::vec2(0.0f);
        }

        if (hasTangents) {
            key.tangent = glm::vec4(
                Quantize(primitive.tangents[idx].x, epsilon),
                Quantize(primitive.tangents[idx].y, epsilon),
                Quantize(primitive.tangents[idx].z, epsilon),
                Quantize(primitive.tangents[idx].w, epsilon)
            );
        } else {
            key.tangent = glm::vec4(0.0f);
        }

        // Check if vertex already exists
        auto it = uniqueVertices.find(key);
        if (it != uniqueVertices.end()) {
            // Vertex already exists, reuse its index
            newIndices.push_back(it->second);
        } else {
            // New unique vertex
            u32 newIdx = static_cast<u32>(newPositions.size());
            uniqueVertices[key] = newIdx;

            // Copy original vertex data (not quantized values, to preserve precision)
            newPositions.push_back(primitive.positions[idx]);
            if (hasNormals) newNormals.push_back(primitive.normals[idx]);
            if (hasUVs) newUVs.push_back(primitive.uvs[idx]);
            if (hasTangents) newTangents.push_back(primitive.tangents[idx]);

            newIndices.push_back(newIdx);
        }
    }

    // Replace original arrays with optimized versions
    primitive.positions = std::move(newPositions);
    primitive.normals = std::move(newNormals);
    primitive.uvs = std::move(newUVs);
    primitive.tangents = std::move(newTangents);
    primitive.indices = std::move(newIndices);

    // Update statistics
    stats.optimizedVertexCount = static_cast<u32>(primitive.positions.size());

    if (stats.originalVertexCount > 0) {
        stats.vertexReductionPercent =
            100.0f * (1.0f - static_cast<f32>(stats.optimizedVertexCount) /
                             static_cast<f32>(stats.originalVertexCount));
    }

    // ========================================================================
    // Validation: Verify triangle integrity after deduplication
    // ========================================================================
    if (primitive.indices.size() >= 3) {
        QL_LOG_DEBUG("=== MeshOptimizer Post-Dedup Validation ===");
        QL_LOG_DEBUG("  Vertices: {} -> {}, Indices: {}",
                    stats.originalVertexCount, stats.optimizedVertexCount,
                    primitive.indices.size());

        // Check first 12 triangles (if cube, this is all of them)
        size_t numTriangles = primitive.indices.size() / 3;
        for (size_t triIdx = 0; triIdx < std::min(numTriangles, size_t(12)); ++triIdx) {
            u32 idx0 = primitive.indices[triIdx * 3 + 0];
            u32 idx1 = primitive.indices[triIdx * 3 + 1];
            u32 idx2 = primitive.indices[triIdx * 3 + 2];

            if (idx0 >= primitive.positions.size() ||
                idx1 >= primitive.positions.size() ||
                idx2 >= primitive.positions.size()) {
                QL_LOG_ERROR("  Triangle {}: INVALID post-dedup indices [{}, {}, {}] (max={})",
                            triIdx, idx0, idx1, idx2, primitive.positions.size() - 1);
                continue;
            }

            glm::vec3 v0 = primitive.positions[idx0];
            glm::vec3 v1 = primitive.positions[idx1];
            glm::vec3 v2 = primitive.positions[idx2];

            // Compute geometric normal
            glm::vec3 e0 = v1 - v0;
            glm::vec3 e1 = v2 - v0;
            glm::vec3 cross = glm::cross(e0, e1);
            float len = glm::length(cross);

            if (len < 1e-8f) {
                QL_LOG_WARN("  Triangle {}: DEGENERATE (near-zero area)", triIdx);
                continue;
            }

            glm::vec3 geoNormal = cross / len;

            // Check if axis-aligned (cube validation)
            bool axisAligned =
                (std::abs(std::abs(geoNormal.x) - 1.0f) < 0.01f &&
                 std::abs(geoNormal.y) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                (std::abs(geoNormal.x) < 0.01f &&
                 std::abs(std::abs(geoNormal.y) - 1.0f) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                (std::abs(geoNormal.x) < 0.01f && std::abs(geoNormal.y) < 0.01f &&
                 std::abs(std::abs(geoNormal.z) - 1.0f) < 0.01f);

            QL_LOG_DEBUG("  Tri {}: [{},{},{}] -> geoN=({:.3f},{:.3f},{:.3f}) {}",
                        triIdx, idx0, idx1, idx2,
                        geoNormal.x, geoNormal.y, geoNormal.z,
                        axisAligned ? "FINE" : "SKEWED");
        }
        QL_LOG_DEBUG("=== End Post-Dedup Validation ===");
    }

    return stats;
}

// ============================================================================
// DeduplicateMesh Implementation
// ============================================================================

MeshOptimizationStats MeshOptimizer::DeduplicateMesh(Mesh& mesh) {
    MeshOptimizationStats totalStats;

    for (auto& primitive : mesh.primitives) {
        auto stats = DeduplicateVertices(primitive);
        totalStats.originalVertexCount += stats.originalVertexCount;
        totalStats.optimizedVertexCount += stats.optimizedVertexCount;
        totalStats.originalIndexCount += stats.originalIndexCount;
        totalStats.optimizedIndexCount += stats.optimizedIndexCount;
    }

    if (totalStats.originalVertexCount > 0) {
        totalStats.vertexReductionPercent =
            100.0f * (1.0f - static_cast<f32>(totalStats.optimizedVertexCount) /
                             static_cast<f32>(totalStats.originalVertexCount));
    }

    return totalStats;
}

// ============================================================================
// OptimizeVertexCache Implementation (Placeholder)
// ============================================================================

void MeshOptimizer::OptimizeVertexCache(GeometryPrimitive& /* primitive */) {
    // TODO: Implement Forsyth algorithm or similar for vertex cache optimization
    // This is a future enhancement - for now, just log a message
    QL_LOG_DEBUG("MeshOptimizer::OptimizeVertexCache not yet implemented");
}

// ============================================================================
// Utility Functions
// ============================================================================

bool MeshOptimizer::ShouldDeduplicate(const GeometryPrimitive& primitive) {
    if (primitive.positions.empty() || primitive.indices.empty()) {
        return false;
    }

    // Heuristic: If vertex count equals triangle count * 3, vertices are likely not shared
    // (This happens after face-varying expansion)
    u32 vertexCount = static_cast<u32>(primitive.positions.size());
    u32 triangleCount = static_cast<u32>(primitive.indices.size()) / 3;

    // If vertices per triangle is close to 3.0, there's little sharing
    float verticesPerTriangle = static_cast<float>(vertexCount) / static_cast<float>(triangleCount);

    // If > 2.5 vertices per triangle on average, deduplication may help
    // (Optimal mesh has ~1.5-2.0 vertices per triangle due to sharing)
    return verticesPerTriangle > 2.5f;
}

size_t MeshOptimizer::EstimateMemorySaved(const MeshOptimizationStats& stats) {
    if (!stats.WasOptimized()) {
        return 0;
    }

    // Estimate per-vertex memory:
    // - vec3 position: 12 bytes
    // - vec3 normal: 12 bytes
    // - vec2 uv: 8 bytes
    // - vec4 tangent: 16 bytes (optional, assume present)
    // Total: 48 bytes per vertex
    constexpr size_t bytesPerVertex = 12 + 12 + 8 + 16;

    u32 vertexReduction = stats.originalVertexCount - stats.optimizedVertexCount;
    return static_cast<size_t>(vertexReduction) * bytesPerVertex;
}

} // namespace quantiloom
