// ============================================================================
// Quantiloom - Unit Tests for scene/Mesh.hpp
// ============================================================================
// Tests cover:
// - GeometryPrimitive construction and validation
// - Mesh construction and validation
// - ComputeBounds (AABB calculation) for primitives and meshes
// - Vertex/triangle counting
// - Edge cases and validation
// ============================================================================

#include <gtest/gtest.h>
#include "scene/Mesh.hpp"
#include <glm/glm.hpp>

using namespace quantiloom;

// ============================================================================
// GeometryPrimitive - Basic Construction Tests
// ============================================================================

TEST(GeometryPrimitiveTest, EmptyPrimitive) {
    GeometryPrimitive prim;

    EXPECT_TRUE(prim.positions.empty());
    EXPECT_TRUE(prim.normals.empty());
    EXPECT_TRUE(prim.uvs.empty());
    EXPECT_TRUE(prim.tangents.empty());
    EXPECT_TRUE(prim.indices.empty());
    EXPECT_EQ(prim.materialId, 0);
    EXPECT_FALSE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, SimpleTriangle) {
    GeometryPrimitive prim;

    // Create a simple triangle
    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    };
    prim.indices = {0, 1, 2};

    EXPECT_TRUE(prim.IsValid());
    EXPECT_EQ(prim.GetVertexCount(), 3);
    EXPECT_EQ(prim.GetTriangleCount(), 1);
}

TEST(GeometryPrimitiveTest, InvalidPrimitiveNotEnoughVertices) {
    GeometryPrimitive prim;

    prim.positions = {glm::vec3(0.0f)};  // Only 1 vertex
    prim.indices = {0, 0, 0};

    EXPECT_FALSE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, InvalidPrimitiveNotEnoughIndices) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1};  // Only 2 indices, need 3 for a triangle

    EXPECT_FALSE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, InvalidIndicesOutOfRange) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1, 5};  // Index 5 out of range

    EXPECT_FALSE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, InvalidIndicesNotDivisibleBy3) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1, 2, 0};  // 4 indices, not divisible by 3

    EXPECT_FALSE(prim.IsValid());
}

// ============================================================================
// GeometryPrimitive - Attribute Tests
// ============================================================================

TEST(GeometryPrimitiveTest, WithNormals) {
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
}

TEST(GeometryPrimitiveTest, InvalidNormalsMismatch) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f)  // Only 1 normal for 3 vertices
    };
    prim.indices = {0, 1, 2};

    EXPECT_FALSE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, WithUVs) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.uvs = {
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 1.0f)
    };
    prim.indices = {0, 1, 2};

    EXPECT_TRUE(prim.IsValid());
}

TEST(GeometryPrimitiveTest, WithTangents) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.tangents = {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
    };
    prim.indices = {0, 1, 2};

    EXPECT_TRUE(prim.IsValid());
    EXPECT_TRUE(prim.HasTangents());
}

TEST(GeometryPrimitiveTest, NoTangents) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1, 2};

    EXPECT_FALSE(prim.HasTangents());
}

// ============================================================================
// GeometryPrimitive - ComputeBounds Tests
// ============================================================================

TEST(GeometryPrimitiveTest, ComputeBoundsSimpleTriangle) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(2.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 3.0f, 0.0f)
    };
    prim.indices = {0, 1, 2};

    glm::vec3 minBound, maxBound;
    prim.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(0.0f, 0.0f, 0.0f));
    EXPECT_EQ(maxBound, glm::vec3(2.0f, 3.0f, 0.0f));
}

TEST(GeometryPrimitiveTest, ComputeBoundsCube) {
    GeometryPrimitive prim;

    // Unit cube vertices
    prim.positions = {
        glm::vec3(-1.0f, -1.0f, -1.0f),
        glm::vec3( 1.0f, -1.0f, -1.0f),
        glm::vec3( 1.0f,  1.0f, -1.0f),
        glm::vec3(-1.0f,  1.0f, -1.0f),
        glm::vec3(-1.0f, -1.0f,  1.0f),
        glm::vec3( 1.0f, -1.0f,  1.0f),
        glm::vec3( 1.0f,  1.0f,  1.0f),
        glm::vec3(-1.0f,  1.0f,  1.0f)
    };

    // Two triangles per face (12 triangles total)
    prim.indices = {
        0, 1, 2, 0, 2, 3,  // Front
        4, 6, 5, 4, 7, 6,  // Back
        0, 4, 5, 0, 5, 1,  // Bottom
        2, 6, 7, 2, 7, 3,  // Top
        0, 3, 7, 0, 7, 4,  // Left
        1, 5, 6, 1, 6, 2   // Right
    };

    glm::vec3 minBound, maxBound;
    prim.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(-1.0f, -1.0f, -1.0f));
    EXPECT_EQ(maxBound, glm::vec3(1.0f, 1.0f, 1.0f));
}

TEST(GeometryPrimitiveTest, ComputeBoundsNegativeCoordinates) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(-5.0f, -10.0f, -2.0f),
        glm::vec3(-3.0f, -8.0f, -1.0f),
        glm::vec3(-4.0f, -9.0f, -3.0f)
    };
    prim.indices = {0, 1, 2};

    glm::vec3 minBound, maxBound;
    prim.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(-5.0f, -10.0f, -3.0f));
    EXPECT_EQ(maxBound, glm::vec3(-3.0f, -8.0f, -1.0f));
}

TEST(GeometryPrimitiveTest, ComputeBoundsEmptyPrimitive) {
    GeometryPrimitive prim;

    glm::vec3 minBound, maxBound;
    prim.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(0.0f));
    EXPECT_EQ(maxBound, glm::vec3(0.0f));
}

TEST(GeometryPrimitiveTest, ComputeBoundsSinglePoint) {
    GeometryPrimitive prim;

    prim.positions = {
        glm::vec3(5.0f, 3.0f, -2.0f),
        glm::vec3(5.0f, 3.0f, -2.0f),
        glm::vec3(5.0f, 3.0f, -2.0f)
    };
    prim.indices = {0, 1, 2};

    glm::vec3 minBound, maxBound;
    prim.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(5.0f, 3.0f, -2.0f));
    EXPECT_EQ(maxBound, glm::vec3(5.0f, 3.0f, -2.0f));
}

// ============================================================================
// Mesh - Basic Tests
// ============================================================================

TEST(MeshTest, EmptyMesh) {
    Mesh mesh;

    EXPECT_TRUE(mesh.primitives.empty());
    EXPECT_TRUE(mesh.name.empty());
    EXPECT_FALSE(mesh.IsValid());
    EXPECT_EQ(mesh.GetPrimitiveCount(), 0);
    EXPECT_EQ(mesh.GetTotalTriangleCount(), 0);
    EXPECT_EQ(mesh.GetTotalVertexCount(), 0);
}

TEST(MeshTest, SinglePrimitiveMesh) {
    Mesh mesh;
    mesh.name = "TestMesh";

    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim.indices = {0, 1, 2};

    mesh.primitives.push_back(prim);

    EXPECT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh.GetPrimitiveCount(), 1);
    EXPECT_EQ(mesh.GetTotalTriangleCount(), 1);
    EXPECT_EQ(mesh.GetTotalVertexCount(), 3);
    EXPECT_EQ(mesh.name, "TestMesh");
}

TEST(MeshTest, MultiplePrimitivesMesh) {
    Mesh mesh;

    // First primitive: 3 vertices, 1 triangle
    GeometryPrimitive prim1;
    prim1.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    prim1.indices = {0, 1, 2};

    // Second primitive: 4 vertices, 2 triangles
    GeometryPrimitive prim2;
    prim2.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f), glm::vec3(3.0f)
    };
    prim2.indices = {0, 1, 2, 0, 2, 3};

    mesh.primitives.push_back(prim1);
    mesh.primitives.push_back(prim2);

    EXPECT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh.GetPrimitiveCount(), 2);
    EXPECT_EQ(mesh.GetTotalTriangleCount(), 3);   // 1 + 2
    EXPECT_EQ(mesh.GetTotalVertexCount(), 7);     // 3 + 4
}

TEST(MeshTest, InvalidMeshWithInvalidPrimitive) {
    Mesh mesh;

    GeometryPrimitive validPrim;
    validPrim.positions = {
        glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(2.0f)
    };
    validPrim.indices = {0, 1, 2};

    GeometryPrimitive invalidPrim;
    invalidPrim.positions = {glm::vec3(0.0f)};  // Not enough vertices

    mesh.primitives.push_back(validPrim);
    mesh.primitives.push_back(invalidPrim);

    EXPECT_FALSE(mesh.IsValid());
}

// ============================================================================
// Mesh - ComputeBounds Tests
// ============================================================================

TEST(MeshTest, ComputeBoundsSinglePrimitive) {
    Mesh mesh;

    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(1.0f, 2.0f, 3.0f),
        glm::vec3(4.0f, 5.0f, 6.0f),
        glm::vec3(7.0f, 8.0f, 9.0f)
    };
    prim.indices = {0, 1, 2};
    mesh.primitives.push_back(prim);

    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(maxBound, glm::vec3(7.0f, 8.0f, 9.0f));
}

TEST(MeshTest, ComputeBoundsMultiplePrimitives) {
    Mesh mesh;

    // First primitive
    GeometryPrimitive prim1;
    prim1.positions = {
        glm::vec3(-1.0f, -1.0f, -1.0f),
        glm::vec3(1.0f, -1.0f, -1.0f),
        glm::vec3(0.0f, 1.0f, -1.0f)
    };
    prim1.indices = {0, 1, 2};

    // Second primitive (extends bounds)
    GeometryPrimitive prim2;
    prim2.positions = {
        glm::vec3(-2.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 2.0f)
    };
    prim2.indices = {0, 1, 2};

    mesh.primitives.push_back(prim1);
    mesh.primitives.push_back(prim2);

    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(-2.0f, -1.0f, -1.0f));
    EXPECT_EQ(maxBound, glm::vec3(1.0f, 2.0f, 2.0f));
}

TEST(MeshTest, ComputeBoundsEmptyMesh) {
    Mesh mesh;

    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(0.0f));
    EXPECT_EQ(maxBound, glm::vec3(0.0f));
}

// ============================================================================
// SceneNode Tests
// ============================================================================

TEST(SceneNodeTest, DefaultConstruction) {
    SceneNode node;

    EXPECT_EQ(node.meshIndex, 0);
    EXPECT_EQ(node.transform, glm::mat4(1.0f));  // Identity matrix
    EXPECT_TRUE(node.name.empty());
    EXPECT_TRUE(node.IsValid());
}

TEST(SceneNodeTest, WithTransform) {
    SceneNode node;

    glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    node.transform = translation;

    EXPECT_TRUE(node.IsValid());
}

TEST(SceneNodeTest, InvalidTransformWithNaN) {
    SceneNode node;

    node.transform[0][0] = std::numeric_limits<float>::quiet_NaN();

    EXPECT_FALSE(node.IsValid());
}

TEST(SceneNodeTest, InvalidTransformWithInf) {
    SceneNode node;

    node.transform[1][2] = std::numeric_limits<float>::infinity();

    EXPECT_FALSE(node.IsValid());
}

// ============================================================================
// Realistic Geometry Tests
// ============================================================================

TEST(MeshTest, RealisticQuadMesh) {
    // Create a quad mesh (2 triangles)
    Mesh mesh;
    mesh.name = "Quad";

    GeometryPrimitive prim;
    prim.positions = {
        glm::vec3(-1.0f, -1.0f, 0.0f),  // Bottom-left
        glm::vec3( 1.0f, -1.0f, 0.0f),  // Bottom-right
        glm::vec3( 1.0f,  1.0f, 0.0f),  // Top-right
        glm::vec3(-1.0f,  1.0f, 0.0f)   // Top-left
    };

    prim.normals = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    };

    prim.uvs = {
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(1.0f, 1.0f),
        glm::vec2(0.0f, 1.0f)
    };

    prim.indices = {0, 1, 2, 0, 2, 3};
    prim.materialId = 0;

    mesh.primitives.push_back(prim);

    EXPECT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh.GetTotalTriangleCount(), 2);
    EXPECT_EQ(mesh.GetTotalVertexCount(), 4);

    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(-1.0f, -1.0f, 0.0f));
    EXPECT_EQ(maxBound, glm::vec3(1.0f, 1.0f, 0.0f));
}

TEST(MeshTest, RealisticSphereMesh) {
    // Create a simplified sphere (octahedron approximation)
    Mesh mesh;
    mesh.name = "Sphere";

    GeometryPrimitive prim;

    // 6 vertices (axis points)
    prim.positions = {
        glm::vec3( 0.0f,  1.0f,  0.0f),  // Top
        glm::vec3( 0.0f, -1.0f,  0.0f),  // Bottom
        glm::vec3( 1.0f,  0.0f,  0.0f),  // Right
        glm::vec3(-1.0f,  0.0f,  0.0f),  // Left
        glm::vec3( 0.0f,  0.0f,  1.0f),  // Front
        glm::vec3( 0.0f,  0.0f, -1.0f)   // Back
    };

    // 8 triangular faces
    prim.indices = {
        0, 4, 2,  // Top-front-right
        0, 2, 5,  // Top-right-back
        0, 5, 3,  // Top-back-left
        0, 3, 4,  // Top-left-front
        1, 2, 4,  // Bottom-right-front
        1, 5, 2,  // Bottom-back-right
        1, 3, 5,  // Bottom-left-back
        1, 4, 3   // Bottom-front-left
    };

    mesh.primitives.push_back(prim);

    EXPECT_TRUE(mesh.IsValid());
    EXPECT_EQ(mesh.GetTotalTriangleCount(), 8);
    EXPECT_EQ(mesh.GetTotalVertexCount(), 6);

    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    EXPECT_EQ(minBound, glm::vec3(-1.0f, -1.0f, -1.0f));
    EXPECT_EQ(maxBound, glm::vec3(1.0f, 1.0f, 1.0f));
}
