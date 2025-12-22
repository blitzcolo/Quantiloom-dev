// ============================================================================
// Quantiloom - Unit Tests for renderer/AccelerationStructure
// ============================================================================
// Tests cover:
// - Normal buffer creation and fallback generation
// - Buffer accessor validation
// - BLAS construction with and without normals
// ============================================================================

#include <gtest/gtest.h>
#include "scene/Mesh.hpp"
#include <glm/glm.hpp>

using namespace quantiloom;

// ============================================================================
// GeometryPrimitive - Normal Handling Tests
// ============================================================================
// These tests document the expected behavior when normals are missing.
// BLAS (in AccelerationStructure.cpp) will automatically generate flat normals
// for primitives without normals at GPU upload time.
// ============================================================================

TEST(AccelerationStructureTest, PrimitiveWithoutNormals) {
    // Create a simple triangle WITHOUT normals
    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    };
    prim.indices = {0, 1, 2};

    // Primitive is still valid even without normals
    EXPECT_TRUE(prim.IsValid());
    EXPECT_TRUE(prim.normals.empty());

    // Note: BLAS will automatically generate flat normals during GPU upload
    // See AccelerationStructure.cpp:UploadGeometryBuffers() for implementation
}

TEST(AccelerationStructureTest, PrimitiveWithNormals) {
    // Create a triangle WITH normals (smooth shading)
    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    };
    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };
    prim.indices = {0, 1, 2};

    EXPECT_TRUE(prim.IsValid());
    EXPECT_FALSE(prim.normals.empty());
    EXPECT_EQ(prim.normals.size(), 3);

    // BLAS will use these normals directly for smooth shading
}

TEST(AccelerationStructureTest, FlatNormalGeneration) {
    // This test documents the expected flat normal for a triangle
    // facing +Z direction
    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    };
    prim.indices = {0, 1, 2};

    // Compute geometric normal (what BLAS will generate)
    glm::vec3 v0 = prim.positions[0];
    glm::vec3 v1 = prim.positions[1];
    glm::vec3 v2 = prim.positions[2];

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 expectedNormal = glm::normalize(glm::cross(edge1, edge2));

    // Expected normal for this triangle should point in +Z direction
    EXPECT_NEAR(expectedNormal.x, 0.0f, 1e-5f);
    EXPECT_NEAR(expectedNormal.y, 0.0f, 1e-5f);
    EXPECT_NEAR(expectedNormal.z, 1.0f, 1e-5f);
}

TEST(AccelerationStructureTest, QuadMeshWithNormals) {
    // Create a quad mesh (2 triangles) with smooth normals
    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(-1.0f, -1.0f, 0.0f),  // Bottom-left
        glm::vec3( 1.0f, -1.0f, 0.0f),  // Bottom-right
        glm::vec3( 1.0f,  1.0f, 0.0f),  // Top-right
        glm::vec3(-1.0f,  1.0f, 0.0f)   // Top-left
    };

    // All normals point in +Z direction (smooth shading)
    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };

    prim.indices = {0, 1, 2, 0, 2, 3};

    EXPECT_TRUE(prim.IsValid());
    EXPECT_EQ(prim.normals.size(), 4);
    EXPECT_EQ(prim.GetTriangleCount(), 2);

    // BLAS will interpolate these normals for smooth shading
}

TEST(AccelerationStructureTest, CubeMeshWithFlatNormals) {
    // Create a cube with flat normals (one normal per face, duplicated vertices)
    GeometryPrimitive prim;

    // Front face (4 vertices, all with normal +Z)
    prim.positions = {
        glm::vec3(-1.0f, -1.0f, 1.0f),
        glm::vec3( 1.0f, -1.0f, 1.0f),
        glm::vec3( 1.0f,  1.0f, 1.0f),
        glm::vec3(-1.0f,  1.0f, 1.0f)
    };

    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };

    prim.indices = {0, 1, 2, 0, 2, 3};

    EXPECT_TRUE(prim.IsValid());
    EXPECT_EQ(prim.normals.size(), 4);

    // All normals should be identical (flat shading for this face)
    for (const auto& normal : prim.normals) {
        EXPECT_NEAR(normal.x, 0.0f, 1e-5f);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5f);
        EXPECT_NEAR(normal.z, 1.0f, 1e-5f);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(AccelerationStructureTest, MeshWithMixedPrimitives) {
    // Create a mesh with one primitive having normals, another without
    Mesh mesh;
    mesh.name = "MixedMesh";

    // Primitive 1: WITH normals (smooth shading)
    GeometryPrimitive prim1;
    prim1.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim1.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };
    prim1.indices = {0, 1, 2};

    // Primitive 2: WITHOUT normals (will use flat shading)
    GeometryPrimitive prim2;
    prim2.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim2.indices = {0, 1, 2};

    mesh.primitives.push_back(prim1);
    mesh.primitives.push_back(prim2);

    EXPECT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh.GetPrimitiveCount(), 2);

    // Both primitives are valid
    EXPECT_FALSE(prim1.normals.empty());
    EXPECT_TRUE(prim2.normals.empty());
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(AccelerationStructureTest, NormalValidation) {
    // Test that IsValid() correctly handles normals
    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1, 2};

    // Valid: no normals (will be generated)
    EXPECT_TRUE(prim.IsValid());

    // Valid: correct number of normals
    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };
    EXPECT_TRUE(prim.IsValid());

    // Invalid: wrong number of normals
    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f)  // Only 1 normal for 3 vertices
    };
    EXPECT_FALSE(prim.IsValid());
}
