/**
 * @file test_mesh_optimizer.cpp
 * @brief Comprehensive unit tests for MeshOptimizer vertex deduplication
 *
 * Tests all MeshOptimizer functionality including:
 * - Basic vertex deduplication (simple quad)
 * - No-op when already optimized
 * - Hard edges (same position, different normals)
 * - UV seams (same position, different UVs)
 * - Empty mesh handling
 * - Large mesh performance
 * - Material ID preservation
 * - Triangle winding order preservation
 *
 * @author wtflmao
 */

#include <gtest/gtest.h>
#include "scene/MeshOptimizer.hpp"
#include "scene/Mesh.hpp"
#include "core/Types.hpp"
#include <vector>
#include <cmath>

using namespace quantiloom;

// ============================================================================
// Test Utilities
// ============================================================================

namespace {

/**
 * @brief Create a simple quad primitive with duplicate vertices
 *
 * A quad made of 2 triangles with 6 vertices (shared edge duplicated):
 *   v0---v1      After dedup should be 4 unique vertices
 *   | \ / |
 *   |  X  |      Triangles: (0,1,2) and (3,4,5)
 *   | / \ |      where v2==v3 and v1==v4
 *   v3---v4
 */
GeometryPrimitive CreateQuadWithDuplicates() {
    GeometryPrimitive prim;

    // 6 vertices for 2 triangles (duplicated at shared edge)
    // Triangle 1: (0, 1, 2)
    prim.positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));  // v0 - bottom left
    prim.positions.push_back(glm::vec3(1.0f, 0.0f, 0.0f));  // v1 - bottom right
    prim.positions.push_back(glm::vec3(1.0f, 1.0f, 0.0f));  // v2 - top right

    // Triangle 2: (3, 4, 5)
    prim.positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));  // v3 - bottom left (duplicate of v0)
    prim.positions.push_back(glm::vec3(1.0f, 1.0f, 0.0f));  // v4 - top right (duplicate of v2)
    prim.positions.push_back(glm::vec3(0.0f, 1.0f, 0.0f));  // v5 - top left

    // All have same normal (facing +Z)
    glm::vec3 faceNormal(0.0f, 0.0f, 1.0f);
    for (int i = 0; i < 6; ++i) {
        prim.normals.push_back(faceNormal);
    }

    // UV coordinates matching positions (0,0 at bottom-left)
    prim.uvs.push_back(glm::vec2(0.0f, 0.0f));  // v0
    prim.uvs.push_back(glm::vec2(1.0f, 0.0f));  // v1
    prim.uvs.push_back(glm::vec2(1.0f, 1.0f));  // v2
    prim.uvs.push_back(glm::vec2(0.0f, 0.0f));  // v3 (duplicate)
    prim.uvs.push_back(glm::vec2(1.0f, 1.0f));  // v4 (duplicate)
    prim.uvs.push_back(glm::vec2(0.0f, 1.0f));  // v5

    // Sequential indices (before deduplication)
    prim.indices = {0, 1, 2, 3, 4, 5};

    prim.materialId = 42;

    return prim;
}

/**
 * @brief Create an already-optimized quad (4 unique vertices)
 */
GeometryPrimitive CreateOptimizedQuad() {
    GeometryPrimitive prim;

    // 4 unique vertices
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0 - bottom left
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1 - bottom right
        glm::vec3(1.0f, 1.0f, 0.0f),  // 2 - top right
        glm::vec3(0.0f, 1.0f, 0.0f)   // 3 - top left
    };

    glm::vec3 faceNormal(0.0f, 0.0f, 1.0f);
    prim.normals = {faceNormal, faceNormal, faceNormal, faceNormal};

    prim.uvs = {
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(1.0f, 1.0f),
        glm::vec2(0.0f, 1.0f)
    };

    // Indices reference shared vertices
    prim.indices = {0, 1, 2, 0, 2, 3};

    prim.materialId = 7;

    return prim;
}

/**
 * @brief Create a cube with hard edges (different normals at same vertex)
 */
GeometryPrimitive CreateCubeWithHardEdges() {
    GeometryPrimitive prim;

    // Cube with 6 faces, each face has 4 unique vertices (24 total due to hard edges)
    // Each face has its own normal, so vertices at corners have different normals

    // Front face (+Z)
    glm::vec3 frontNorm(0.0f, 0.0f, 1.0f);
    prim.positions.push_back(glm::vec3(-1, -1, 1)); prim.normals.push_back(frontNorm);
    prim.positions.push_back(glm::vec3( 1, -1, 1)); prim.normals.push_back(frontNorm);
    prim.positions.push_back(glm::vec3( 1,  1, 1)); prim.normals.push_back(frontNorm);
    prim.positions.push_back(glm::vec3(-1,  1, 1)); prim.normals.push_back(frontNorm);

    // Back face (-Z)
    glm::vec3 backNorm(0.0f, 0.0f, -1.0f);
    prim.positions.push_back(glm::vec3( 1, -1, -1)); prim.normals.push_back(backNorm);
    prim.positions.push_back(glm::vec3(-1, -1, -1)); prim.normals.push_back(backNorm);
    prim.positions.push_back(glm::vec3(-1,  1, -1)); prim.normals.push_back(backNorm);
    prim.positions.push_back(glm::vec3( 1,  1, -1)); prim.normals.push_back(backNorm);

    // Right face (+X)
    glm::vec3 rightNorm(1.0f, 0.0f, 0.0f);
    prim.positions.push_back(glm::vec3(1, -1,  1)); prim.normals.push_back(rightNorm);
    prim.positions.push_back(glm::vec3(1, -1, -1)); prim.normals.push_back(rightNorm);
    prim.positions.push_back(glm::vec3(1,  1, -1)); prim.normals.push_back(rightNorm);
    prim.positions.push_back(glm::vec3(1,  1,  1)); prim.normals.push_back(rightNorm);

    // Left face (-X)
    glm::vec3 leftNorm(-1.0f, 0.0f, 0.0f);
    prim.positions.push_back(glm::vec3(-1, -1, -1)); prim.normals.push_back(leftNorm);
    prim.positions.push_back(glm::vec3(-1, -1,  1)); prim.normals.push_back(leftNorm);
    prim.positions.push_back(glm::vec3(-1,  1,  1)); prim.normals.push_back(leftNorm);
    prim.positions.push_back(glm::vec3(-1,  1, -1)); prim.normals.push_back(leftNorm);

    // Top face (+Y)
    glm::vec3 topNorm(0.0f, 1.0f, 0.0f);
    prim.positions.push_back(glm::vec3(-1, 1,  1)); prim.normals.push_back(topNorm);
    prim.positions.push_back(glm::vec3( 1, 1,  1)); prim.normals.push_back(topNorm);
    prim.positions.push_back(glm::vec3( 1, 1, -1)); prim.normals.push_back(topNorm);
    prim.positions.push_back(glm::vec3(-1, 1, -1)); prim.normals.push_back(topNorm);

    // Bottom face (-Y)
    glm::vec3 bottomNorm(0.0f, -1.0f, 0.0f);
    prim.positions.push_back(glm::vec3(-1, -1, -1)); prim.normals.push_back(bottomNorm);
    prim.positions.push_back(glm::vec3( 1, -1, -1)); prim.normals.push_back(bottomNorm);
    prim.positions.push_back(glm::vec3( 1, -1,  1)); prim.normals.push_back(bottomNorm);
    prim.positions.push_back(glm::vec3(-1, -1,  1)); prim.normals.push_back(bottomNorm);

    // Indices for 6 faces (2 triangles per face)
    for (int face = 0; face < 6; ++face) {
        int base = face * 4;
        prim.indices.push_back(base + 0);
        prim.indices.push_back(base + 1);
        prim.indices.push_back(base + 2);
        prim.indices.push_back(base + 0);
        prim.indices.push_back(base + 2);
        prim.indices.push_back(base + 3);
    }

    prim.materialId = 99;

    return prim;
}

/**
 * @brief Create a quad with UV seam (same position, different UVs)
 */
GeometryPrimitive CreateQuadWithUVSeam() {
    GeometryPrimitive prim;

    // Two vertices at same position but different UVs (like a texture seam)
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(1.0f, 1.0f, 0.0f),  // 2
        glm::vec3(1.0f, 1.0f, 0.0f),  // 3 - same position as 2, different UV
        glm::vec3(0.0f, 1.0f, 0.0f),  // 4
        glm::vec3(0.0f, 0.0f, 0.0f),  // 5 - same position as 0, different UV
    };

    glm::vec3 norm(0.0f, 0.0f, 1.0f);
    prim.normals = {norm, norm, norm, norm, norm, norm};

    // UVs: vertices 2 and 3 have same position but different UVs
    prim.uvs = {
        glm::vec2(0.0f, 0.0f),  // 0
        glm::vec2(0.5f, 0.0f),  // 1
        glm::vec2(0.5f, 1.0f),  // 2
        glm::vec2(0.6f, 1.0f),  // 3 - different UV than 2!
        glm::vec2(1.0f, 1.0f),  // 4
        glm::vec2(0.1f, 0.0f),  // 5 - different UV than 0!
    };

    prim.indices = {0, 1, 2, 3, 4, 5};

    return prim;
}

/**
 * @brief Create a large mesh for performance testing
 */
GeometryPrimitive CreateLargeMesh(u32 gridSize = 100) {
    GeometryPrimitive prim;

    // Create grid with duplicate vertices (like face-varying USD data)
    for (u32 y = 0; y < gridSize; ++y) {
        for (u32 x = 0; x < gridSize; ++x) {
            // Each cell is a quad with 6 vertices (2 triangles, 4 unique corners)
            float fx = static_cast<float>(x);
            float fy = static_cast<float>(y);

            glm::vec3 norm(0.0f, 1.0f, 0.0f);

            // Triangle 1: (x,y), (x+1,y), (x+1,y+1)
            prim.positions.push_back(glm::vec3(fx, 0.0f, fy));
            prim.positions.push_back(glm::vec3(fx + 1.0f, 0.0f, fy));
            prim.positions.push_back(glm::vec3(fx + 1.0f, 0.0f, fy + 1.0f));

            prim.normals.push_back(norm);
            prim.normals.push_back(norm);
            prim.normals.push_back(norm);

            prim.uvs.push_back(glm::vec2(fx / gridSize, fy / gridSize));
            prim.uvs.push_back(glm::vec2((fx + 1) / gridSize, fy / gridSize));
            prim.uvs.push_back(glm::vec2((fx + 1) / gridSize, (fy + 1) / gridSize));

            // Triangle 2: (x,y), (x+1,y+1), (x,y+1)
            prim.positions.push_back(glm::vec3(fx, 0.0f, fy));  // duplicate
            prim.positions.push_back(glm::vec3(fx + 1.0f, 0.0f, fy + 1.0f));  // duplicate
            prim.positions.push_back(glm::vec3(fx, 0.0f, fy + 1.0f));

            prim.normals.push_back(norm);
            prim.normals.push_back(norm);
            prim.normals.push_back(norm);

            prim.uvs.push_back(glm::vec2(fx / gridSize, fy / gridSize));  // duplicate
            prim.uvs.push_back(glm::vec2((fx + 1) / gridSize, (fy + 1) / gridSize));  // duplicate
            prim.uvs.push_back(glm::vec2(fx / gridSize, (fy + 1) / gridSize));
        }
    }

    // Sequential indices
    prim.indices.resize(prim.positions.size());
    for (size_t i = 0; i < prim.indices.size(); ++i) {
        prim.indices[i] = static_cast<u32>(i);
    }

    return prim;
}

/**
 * @brief Compute triangle normal from indices
 */
glm::vec3 ComputeTriangleNormal(const GeometryPrimitive& prim, u32 triIdx) {
    u32 i0 = prim.indices[triIdx * 3 + 0];
    u32 i1 = prim.indices[triIdx * 3 + 1];
    u32 i2 = prim.indices[triIdx * 3 + 2];

    glm::vec3 v0 = prim.positions[i0];
    glm::vec3 v1 = prim.positions[i1];
    glm::vec3 v2 = prim.positions[i2];

    return glm::normalize(glm::cross(v1 - v0, v2 - v0));
}

} // anonymous namespace

// ============================================================================
// Basic Deduplication Tests
// ============================================================================

/**
 * @test Simple quad with duplicate vertices is properly deduplicated
 */
TEST(MeshOptimizerTest, DeduplicateSimpleQuad) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    // Before dedup: 6 vertices
    EXPECT_EQ(prim.positions.size(), 6);
    EXPECT_EQ(prim.normals.size(), 6);
    EXPECT_EQ(prim.uvs.size(), 6);
    EXPECT_EQ(prim.indices.size(), 6);

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // After dedup: 4 unique vertices
    EXPECT_EQ(prim.positions.size(), 4);
    EXPECT_EQ(prim.normals.size(), 4);
    EXPECT_EQ(prim.uvs.size(), 4);
    EXPECT_EQ(prim.indices.size(), 6);  // Index count unchanged

    // Verify statistics
    EXPECT_EQ(stats.originalVertexCount, 6);
    EXPECT_EQ(stats.optimizedVertexCount, 4);
    EXPECT_TRUE(stats.WasOptimized());
    EXPECT_GT(stats.vertexReductionPercent, 30.0f);  // ~33% reduction
}

/**
 * @test Already optimized mesh doesn't change
 */
TEST(MeshOptimizerTest, DeduplicateNoChange) {
    GeometryPrimitive prim = CreateOptimizedQuad();

    u32 originalVertexCount = static_cast<u32>(prim.positions.size());
    u32 originalMaterialId = prim.materialId;

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // Should be unchanged (or minimal change due to floating point)
    EXPECT_EQ(prim.positions.size(), originalVertexCount);
    EXPECT_EQ(stats.optimizedVertexCount, originalVertexCount);
    EXPECT_FALSE(stats.WasOptimized());
    EXPECT_EQ(prim.materialId, originalMaterialId);
}

/**
 * @test Hard edges are preserved (same position, different normals)
 */
TEST(MeshOptimizerTest, DeduplicateWithNormals) {
    GeometryPrimitive prim = CreateCubeWithHardEdges();

    // Before dedup: 24 vertices (4 per face * 6 faces)
    EXPECT_EQ(prim.positions.size(), 24);

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // After dedup: should still be 24 (no sharing because normals differ at corners)
    // Vertices at cube corners have same position but 3 different normals
    EXPECT_EQ(prim.positions.size(), 24);
    EXPECT_FALSE(stats.WasOptimized());
}

/**
 * @test UV seams are preserved (same position, different UVs)
 */
TEST(MeshOptimizerTest, DeduplicateWithUVs) {
    GeometryPrimitive prim = CreateQuadWithUVSeam();

    // Before dedup: 6 vertices
    EXPECT_EQ(prim.positions.size(), 6);

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // After dedup: should still be 6 because UVs differ at same positions
    EXPECT_EQ(prim.positions.size(), 6);
    EXPECT_FALSE(stats.WasOptimized());
}

// ============================================================================
// Edge Cases
// ============================================================================

/**
 * @test Empty mesh doesn't crash
 */
TEST(MeshOptimizerTest, DeduplicateEmptyMesh) {
    GeometryPrimitive prim;

    // Empty primitive
    EXPECT_TRUE(prim.positions.empty());
    EXPECT_TRUE(prim.indices.empty());

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // Should not crash, no change
    EXPECT_TRUE(prim.positions.empty());
    EXPECT_FALSE(stats.WasOptimized());
}

/**
 * @test Primitive with positions only (no normals/UVs)
 */
TEST(MeshOptimizerTest, DeduplicatePositionsOnly) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    // Remove normals and UVs
    prim.normals.clear();
    prim.uvs.clear();

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // Should deduplicate based on position alone
    EXPECT_EQ(prim.positions.size(), 4);
    EXPECT_TRUE(prim.normals.empty());
    EXPECT_TRUE(prim.uvs.empty());
    EXPECT_TRUE(stats.WasOptimized());
}

/**
 * @test Mismatched attribute arrays are handled gracefully
 */
TEST(MeshOptimizerTest, DeduplicateMismatchedArrays) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    // Create mismatched arrays
    prim.normals.pop_back();  // 5 normals, 6 positions

    u32 originalCount = static_cast<u32>(prim.positions.size());
    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // Should skip deduplication due to mismatch
    EXPECT_EQ(prim.positions.size(), originalCount);
    EXPECT_FALSE(stats.WasOptimized());
}

// ============================================================================
// Performance Tests
// ============================================================================

/**
 * @test Large mesh performance and correctness
 */
TEST(MeshOptimizerTest, DeduplicateLargeMesh) {
    constexpr u32 gridSize = 100;
    GeometryPrimitive prim = CreateLargeMesh(gridSize);

    // Before: gridSize^2 * 6 vertices = 60,000 vertices
    u32 expectedBefore = gridSize * gridSize * 6;
    EXPECT_EQ(prim.positions.size(), expectedBefore);

    auto stats = MeshOptimizer::DeduplicateVertices(prim);

    // After: approximately (gridSize+1)^2 unique vertices = ~10,201
    // Due to shared edges between grid cells
    u32 expectedAfter = (gridSize + 1) * (gridSize + 1);
    EXPECT_NEAR(prim.positions.size(), expectedAfter, expectedAfter * 0.1);  // 10% tolerance

    // Should have significant reduction
    EXPECT_TRUE(stats.WasOptimized());
    EXPECT_GT(stats.vertexReductionPercent, 80.0f);  // Should reduce by ~80%
}

// ============================================================================
// Correctness Tests
// ============================================================================

/**
 * @test Material ID is preserved
 */
TEST(MeshOptimizerTest, PreserveMaterialId) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    u32 originalMaterialId = prim.materialId;
    EXPECT_EQ(originalMaterialId, 42);

    MeshOptimizer::DeduplicateVertices(prim);

    // Material ID should be unchanged
    EXPECT_EQ(prim.materialId, originalMaterialId);
}

/**
 * @test Triangle winding order is preserved
 */
TEST(MeshOptimizerTest, PreserveWindingOrder) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    // Compute expected normals (from winding order) before dedup
    glm::vec3 normalBefore1 = ComputeTriangleNormal(prim, 0);
    glm::vec3 normalBefore2 = ComputeTriangleNormal(prim, 1);

    MeshOptimizer::DeduplicateVertices(prim);

    // Compute normals after dedup
    glm::vec3 normalAfter1 = ComputeTriangleNormal(prim, 0);
    glm::vec3 normalAfter2 = ComputeTriangleNormal(prim, 1);

    // Normals should be same direction (same winding)
    EXPECT_GT(glm::dot(normalBefore1, normalAfter1), 0.99f);
    EXPECT_GT(glm::dot(normalBefore2, normalAfter2), 0.99f);
}

/**
 * @test Primitive validity is maintained
 */
TEST(MeshOptimizerTest, PreservePrimitiveValidity) {
    GeometryPrimitive prim = CreateQuadWithDuplicates();

    EXPECT_TRUE(prim.IsValid());

    MeshOptimizer::DeduplicateVertices(prim);

    // Primitive should still be valid
    EXPECT_TRUE(prim.IsValid());
}

// ============================================================================
// Mesh-level Tests
// ============================================================================

/**
 * @test DeduplicateMesh works on entire mesh
 */
TEST(MeshOptimizerTest, DeduplicateMeshMultiplePrimitives) {
    Mesh mesh;
    mesh.name = "TestMesh";

    // Add multiple primitives
    mesh.primitives.push_back(CreateQuadWithDuplicates());
    mesh.primitives.push_back(CreateQuadWithDuplicates());
    mesh.primitives.push_back(CreateQuadWithDuplicates());

    auto stats = MeshOptimizer::DeduplicateMesh(mesh);

    // All primitives should be deduplicated
    for (const auto& prim : mesh.primitives) {
        EXPECT_EQ(prim.positions.size(), 4);
    }

    EXPECT_EQ(stats.originalVertexCount, 18);  // 6 * 3
    EXPECT_EQ(stats.optimizedVertexCount, 12); // 4 * 3
    EXPECT_TRUE(stats.WasOptimized());
}

// ============================================================================
// Utility Function Tests
// ============================================================================

/**
 * @test ShouldDeduplicate heuristic
 */
TEST(MeshOptimizerTest, ShouldDeduplicateHeuristic) {
    // Unoptimized mesh: 6 vertices / 2 triangles = 3.0 vertices per triangle
    GeometryPrimitive unoptimized = CreateQuadWithDuplicates();
    EXPECT_TRUE(MeshOptimizer::ShouldDeduplicate(unoptimized));

    // Optimized mesh: 4 vertices / 2 triangles = 2.0 vertices per triangle
    GeometryPrimitive optimized = CreateOptimizedQuad();
    EXPECT_FALSE(MeshOptimizer::ShouldDeduplicate(optimized));
}

/**
 * @test Memory savings estimation
 */
TEST(MeshOptimizerTest, EstimateMemorySaved) {
    MeshOptimizationStats stats;
    stats.originalVertexCount = 1000;
    stats.optimizedVertexCount = 600;
    stats.vertexReductionPercent = 40.0f;

    size_t savedBytes = MeshOptimizer::EstimateMemorySaved(stats);

    // 400 vertices removed * 48 bytes per vertex = 19,200 bytes
    EXPECT_EQ(savedBytes, 400 * 48);
}

/**
 * @test No memory saved when not optimized
 */
TEST(MeshOptimizerTest, EstimateMemorySavedNotOptimized) {
    MeshOptimizationStats stats;
    stats.originalVertexCount = 100;
    stats.optimizedVertexCount = 100;
    stats.vertexReductionPercent = 0.0f;

    size_t savedBytes = MeshOptimizer::EstimateMemorySaved(stats);

    EXPECT_EQ(savedBytes, 0);
}
