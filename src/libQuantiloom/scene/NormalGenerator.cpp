/**
 * @file NormalGenerator.cpp
 * @brief Implementation of dihedral angle-based normal generation
 */

#include "NormalGenerator.hpp"
#include "core/Log.hpp"
#include <glm/gtc/constants.hpp>
#include <queue>
#include <algorithm>
#include <cmath>

namespace quantiloom {

// ============================================================================
// BuildEdgeTopology
// ============================================================================

NormalGenerator::EdgeTopology NormalGenerator::BuildEdgeTopology(
    const std::vector<u32>& indices)
{
    EdgeTopology topology;
    const size_t faceCount = indices.size() / 3;

    for (size_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        const u32 i0 = indices[faceIdx * 3 + 0];
        const u32 i1 = indices[faceIdx * 3 + 1];
        const u32 i2 = indices[faceIdx * 3 + 2];

        // Three edges per triangle
        EdgeKey edges[3] = {
            EdgeKey(i0, i1),
            EdgeKey(i1, i2),
            EdgeKey(i2, i0)
        };

        for (const auto& edge : edges) {
            auto& info = topology[edge];
            if (info.face0 == UINT32_MAX) {
                info.face0 = static_cast<u32>(faceIdx);
            } else if (info.face1 == UINT32_MAX) {
                info.face1 = static_cast<u32>(faceIdx);
            }
            // If both faces already assigned, this is a non-manifold edge (ignore)
        }
    }

    return topology;
}

// ============================================================================
// ComputeFaceNormals
// ============================================================================

std::vector<glm::vec3> NormalGenerator::ComputeFaceNormals(
    const std::vector<glm::vec3>& positions,
    const std::vector<u32>& indices)
{
    const size_t faceCount = indices.size() / 3;
    std::vector<glm::vec3> faceNormals(faceCount);

    for (size_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        const u32 i0 = indices[faceIdx * 3 + 0];
        const u32 i1 = indices[faceIdx * 3 + 1];
        const u32 i2 = indices[faceIdx * 3 + 2];

        const glm::vec3& v0 = positions[i0];
        const glm::vec3& v1 = positions[i1];
        const glm::vec3& v2 = positions[i2];

        // Cross product: magnitude = 2 * triangle area (used for area weighting)
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 normal = glm::cross(edge1, edge2);

        float len = glm::length(normal);
        if (len > 1e-8f) {
            faceNormals[faceIdx] = normal / len;  // Normalized for angle comparison
        } else {
            faceNormals[faceIdx] = glm::vec3(0.0f, 1.0f, 0.0f);  // Degenerate triangle
        }
    }

    return faceNormals;
}

// ============================================================================
// FindHardEdges
// ============================================================================

NormalGenerator::HardEdgeSet NormalGenerator::FindHardEdges(
    const EdgeTopology& topology,
    const std::vector<glm::vec3>& faceNormals,
    float thresholdRadians)
{
    HardEdgeSet hardEdges;

    for (const auto& [edge, info] : topology) {
        // Boundary edges are always considered hard (no smooth blending across open edges)
        if (info.face1 == UINT32_MAX) {
            hardEdges.insert(edge);
            continue;
        }

        const glm::vec3& n1 = faceNormals[info.face0];
        const glm::vec3& n2 = faceNormals[info.face1];

        float cosAngle = glm::dot(n1, n2);
        // Clamp to [-1, 1] to handle floating-point errors
        cosAngle = std::clamp(cosAngle, -1.0f, 1.0f);
        float dihedralAngle = std::acos(cosAngle);

        if (dihedralAngle > thresholdRadians) {
            hardEdges.insert(edge);
        }
    }

    return hardEdges;
}

// ============================================================================
// BuildVertexToFaces
// ============================================================================

std::vector<std::vector<u32>> NormalGenerator::BuildVertexToFaces(
    const std::vector<u32>& indices,
    size_t vertexCount)
{
    std::vector<std::vector<u32>> vertexToFaces(vertexCount);
    const size_t faceCount = indices.size() / 3;

    for (size_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        const u32 i0 = indices[faceIdx * 3 + 0];
        const u32 i1 = indices[faceIdx * 3 + 1];
        const u32 i2 = indices[faceIdx * 3 + 2];

        vertexToFaces[i0].push_back(static_cast<u32>(faceIdx));
        vertexToFaces[i1].push_back(static_cast<u32>(faceIdx));
        vertexToFaces[i2].push_back(static_cast<u32>(faceIdx));
    }

    return vertexToFaces;
}

// ============================================================================
// GenerateWithDihedralAngle - Main Algorithm
// ============================================================================

void NormalGenerator::GenerateWithDihedralAngle(
    GeometryPrimitive& primitive,
    const NormalGenerationConfig& config)
{
    // Early exit if already has normals
    if (!primitive.normals.empty()) {
        return;
    }

    // Validate input
    if (primitive.positions.empty() || primitive.indices.empty()) {
        QL_LOG_WARN("NormalGenerator: Empty geometry, skipping");
        return;
    }

    if (primitive.indices.size() % 3 != 0) {
        QL_LOG_WARN("NormalGenerator: Index count not divisible by 3, skipping");
        return;
    }

    const float thresholdRad = glm::radians(config.dihedralAngleThreshold);
    const size_t originalVertexCount = primitive.positions.size();
    const size_t faceCount = primitive.indices.size() / 3;

    QL_LOG_DEBUG("NormalGenerator: Processing {} vertices, {} triangles, threshold {}°",
                 originalVertexCount, faceCount, config.dihedralAngleThreshold);

    // Step 1: Build edge topology
    EdgeTopology edgeTopology = BuildEdgeTopology(primitive.indices);

    // Step 2: Compute face normals (normalized for angle comparison)
    std::vector<glm::vec3> faceNormals = ComputeFaceNormals(
        primitive.positions, primitive.indices);

    // Step 3: Find hard edges
    HardEdgeSet hardEdges = FindHardEdges(edgeTopology, faceNormals, thresholdRad);

    QL_LOG_DEBUG("NormalGenerator: Found {} hard edges out of {} total edges",
                 hardEdges.size(), edgeTopology.size());

    // Step 4: Build vertex-to-faces adjacency
    std::vector<std::vector<u32>> vertexToFaces = BuildVertexToFaces(
        primitive.indices, originalVertexCount);

    // Step 5: For each vertex, find smooth-connected face groups
    // Each group will get a separate vertex with its own averaged normal
    //
    // Strategy: For each vertex, flood-fill through adjacent faces,
    // stopping at hard edges. Each connected component becomes a "smooth region".

    // Build face-to-face adjacency (faces sharing an edge)
    // For each face, store which faces are adjacent via smooth edges
    std::vector<std::vector<u32>> faceAdjacency(faceCount);
    for (const auto& [edge, info] : edgeTopology) {
        if (info.face1 == UINT32_MAX) continue;  // Boundary
        if (hardEdges.count(edge) > 0) continue;  // Hard edge

        // Smooth edge: faces are connected
        faceAdjacency[info.face0].push_back(info.face1);
        faceAdjacency[info.face1].push_back(info.face0);
    }

    // For each vertex, find smooth regions via BFS
    // smoothRegions[vertexIdx] = list of face groups (each group = vector of face indices)
    std::vector<std::vector<std::vector<u32>>> smoothRegions(originalVertexCount);

    for (u32 vertexIdx = 0; vertexIdx < originalVertexCount; ++vertexIdx) {
        const auto& adjacentFaces = vertexToFaces[vertexIdx];
        if (adjacentFaces.empty()) continue;

        std::unordered_set<u32> visited;

        for (u32 startFace : adjacentFaces) {
            if (visited.count(startFace) > 0) continue;

            // BFS to find all faces in this smooth region
            std::vector<u32> region;
            std::queue<u32> queue;
            queue.push(startFace);
            visited.insert(startFace);

            while (!queue.empty()) {
                u32 currentFace = queue.front();
                queue.pop();
                region.push_back(currentFace);

                // Check adjacent faces (via smooth edges only)
                for (u32 neighborFace : faceAdjacency[currentFace]) {
                    // Only consider faces that also share this vertex
                    bool sharesVertex = false;
                    for (u32 f : adjacentFaces) {
                        if (f == neighborFace) {
                            sharesVertex = true;
                            break;
                        }
                    }

                    if (sharesVertex && visited.count(neighborFace) == 0) {
                        visited.insert(neighborFace);
                        queue.push(neighborFace);
                    }
                }
            }

            smoothRegions[vertexIdx].push_back(std::move(region));
        }
    }

    // Step 6: Build new vertex/normal arrays with duplication at hard edges
    std::vector<glm::vec3> newPositions;
    std::vector<glm::vec3> newNormals;
    std::vector<glm::vec2> newUVs;
    std::vector<glm::vec4> newTangents;
    std::vector<u32> newIndices = primitive.indices;

    const bool hasUVs = !primitive.uvs.empty();
    const bool hasTangents = !primitive.tangents.empty();

    // Map: (originalVertex, regionIndex) -> newVertexIndex
    std::vector<std::vector<u32>> vertexRegionToNew(originalVertexCount);

    for (u32 vertexIdx = 0; vertexIdx < originalVertexCount; ++vertexIdx) {
        const auto& regions = smoothRegions[vertexIdx];

        for (size_t regionIdx = 0; regionIdx < regions.size(); ++regionIdx) {
            const auto& facesInRegion = regions[regionIdx];

            // Create new vertex for this region
            u32 newIdx = static_cast<u32>(newPositions.size());
            vertexRegionToNew[vertexIdx].push_back(newIdx);

            newPositions.push_back(primitive.positions[vertexIdx]);
            if (hasUVs) {
                newUVs.push_back(primitive.uvs[vertexIdx]);
            }
            if (hasTangents) {
                newTangents.push_back(primitive.tangents[vertexIdx]);
            }

            // Compute area-weighted normal for this region
            glm::vec3 accumulatedNormal(0.0f);
            for (u32 faceIdx : facesInRegion) {
                const u32 i0 = primitive.indices[faceIdx * 3 + 0];
                const u32 i1 = primitive.indices[faceIdx * 3 + 1];
                const u32 i2 = primitive.indices[faceIdx * 3 + 2];

                const glm::vec3& v0 = primitive.positions[i0];
                const glm::vec3& v1 = primitive.positions[i1];
                const glm::vec3& v2 = primitive.positions[i2];

                // Un-normalized cross product for area weighting
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 faceNormal = glm::cross(edge1, edge2);

                if (config.areaWeighted) {
                    accumulatedNormal += faceNormal;  // Magnitude = 2 * area
                } else {
                    float len = glm::length(faceNormal);
                    if (len > 1e-8f) {
                        accumulatedNormal += faceNormal / len;  // Unit weight
                    }
                }
            }

            // Normalize final normal
            float len = glm::length(accumulatedNormal);
            if (len > 1e-8f) {
                newNormals.push_back(accumulatedNormal / len);
            } else {
                newNormals.push_back(glm::vec3(0.0f, 1.0f, 0.0f));  // Fallback
            }
        }
    }

    // Step 7: Update indices to point to new vertices
    // For each face, find which region each vertex belongs to
    for (size_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        for (int corner = 0; corner < 3; ++corner) {
            u32 oldIdx = primitive.indices[faceIdx * 3 + corner];
            const auto& regions = smoothRegions[oldIdx];

            // Find which region contains this face
            for (size_t regionIdx = 0; regionIdx < regions.size(); ++regionIdx) {
                const auto& facesInRegion = regions[regionIdx];
                bool found = false;
                for (u32 f : facesInRegion) {
                    if (f == static_cast<u32>(faceIdx)) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    newIndices[faceIdx * 3 + corner] = vertexRegionToNew[oldIdx][regionIdx];
                    break;
                }
            }
        }
    }

    // Step 8: Replace primitive data
    primitive.positions = std::move(newPositions);
    primitive.normals = std::move(newNormals);
    primitive.indices = std::move(newIndices);
    if (hasUVs) {
        primitive.uvs = std::move(newUVs);
    }
    if (hasTangents) {
        primitive.tangents = std::move(newTangents);
    }

    QL_LOG_DEBUG("NormalGenerator: {} -> {} vertices ({} duplicated for hard edges)",
                 originalVertexCount, primitive.positions.size(),
                 primitive.positions.size() - originalVertexCount);
}

} // namespace quantiloom
