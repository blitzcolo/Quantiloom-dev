/**
 * @file test_scene_geometry.cpp
 * @brief Cover for rendercore::SceneGeometry
 *
 * Offsets into the merged buffers are exactly the kind of thing that goes wrong
 * silently: a wrong one renders geometry from the neighbouring primitive rather than
 * crashing, which looks like a modelling problem. The cases below pin the offset
 * table and the instance walk, neither of which is visible in an image.
 */

#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "renderer/RenderCore.hpp"
#include "renderer/VulkanContext.hpp"
#include "scene/Scene.hpp"

#include <glm/gtc/matrix_transform.hpp>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

namespace {

// A triangle with the attributes named, so a primitive can be given only some.
GeometryPrimitive MakeTriangle(u32 materialId, bool withNormals, bool withUVs, bool withTangents) {
    GeometryPrimitive prim;
    prim.positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    prim.indices = {0, 1, 2};
    prim.materialId = materialId;

    if (withNormals) {
        prim.normals.assign(3, glm::vec3(0.0f, 0.0f, 1.0f));
    }
    if (withUVs) {
        prim.uvs.assign(3, glm::vec2(0.5f, 0.5f));
    }
    if (withTangents) {
        prim.tangents.assign(3, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    }
    return prim;
}

Mesh MakeMesh(std::initializer_list<GeometryPrimitive> primitives) {
    Mesh mesh;
    mesh.primitives.assign(primitives.begin(), primitives.end());
    return mesh;
}

SceneNode MakeNode(u32 meshIndex, const glm::mat4& transform = glm::mat4(1.0f)) {
    SceneNode node;
    node.meshIndex = meshIndex;
    node.transform = transform;
    return node;
}

}  // namespace

TEST_F(VulkanDeviceTest, SceneGeometryConcatenatesPrimitivesInOrder) {
    Scene scene;
    scene.meshes.push_back(MakeMesh({MakeTriangle(0, true, true, true),
                                     MakeTriangle(1, true, true, true)}));
    scene.nodes.push_back(MakeNode(0));

    auto geometry = rendercore::SceneGeometry::Build(Device(), scene);

    ASSERT_TRUE(geometry.IsValid());
    EXPECT_EQ(geometry.BlasCount(), 2u);
    EXPECT_EQ(geometry.VertexCount(), 6u);
    EXPECT_EQ(geometry.IndexCount(), 6u);

    ASSERT_EQ(geometry.Instances().size(), 2u);
    EXPECT_EQ(geometry.Instances()[0].vertexOffset, 0u);
    EXPECT_EQ(geometry.Instances()[0].indexOffset, 0u);
    EXPECT_EQ(geometry.Instances()[1].vertexOffset, 3u);
    EXPECT_EQ(geometry.Instances()[1].indexOffset, 3u);
    EXPECT_EQ(geometry.Instances()[1].materialId, 1u);
}

// A primitive that supplies no normals still occupies a slice sized by its vertex
// count, so every later primitive's normalOffset stays aligned with its vertexOffset.
// Skipping the slice instead would shift every subsequent primitive's normals.
TEST_F(VulkanDeviceTest, SceneGeometryReservesSlicesForAbsentAttributes) {
    Scene scene;
    scene.meshes.push_back(MakeMesh({MakeTriangle(0, /*normals=*/false, /*uvs=*/false,
                                                  /*tangents=*/false),
                                     MakeTriangle(0, true, true, true)}));
    scene.nodes.push_back(MakeNode(0));

    auto geometry = rendercore::SceneGeometry::Build(Device(), scene);

    ASSERT_TRUE(geometry.IsValid());
    ASSERT_EQ(geometry.Instances().size(), 2u);

    const auto& second = geometry.Instances()[1];
    EXPECT_EQ(second.vertexOffset, 3u);
    EXPECT_EQ(second.normalOffset, 3u) << "absent normals still reserve three slots";
    EXPECT_EQ(second.uvOffset, 3u);
    EXPECT_EQ(second.tangentOffset, 3u);
}

// One BLAS per primitive, one instance per node-primitive pair. Two nodes sharing a
// mesh must reuse its structures rather than rebuilding them.
TEST_F(VulkanDeviceTest, SceneGeometryInstancesNodesWithoutDuplicatingBlas) {
    Scene scene;
    scene.meshes.push_back(MakeMesh({MakeTriangle(0, true, true, true)}));
    scene.nodes.push_back(MakeNode(0));
    scene.nodes.push_back(MakeNode(0, glm::translate(glm::mat4(1.0f), {5.0f, 0.0f, 0.0f})));

    auto geometry = rendercore::SceneGeometry::Build(Device(), scene);

    ASSERT_TRUE(geometry.IsValid());
    EXPECT_EQ(geometry.BlasCount(), 1u) << "one primitive, one structure";
    EXPECT_EQ(geometry.InstanceCount(), 2u) << "two nodes reference it";
    EXPECT_EQ(geometry.Instances()[0].vertexOffset, geometry.Instances()[1].vertexOffset)
        << "both instances address the same geometry";
}

// InstanceIndex() in the shader is a position in the node-then-primitive walk, so
// Build and RebuildTlas have to agree on that order. A transform change must not
// renumber the instances.
TEST_F(VulkanDeviceTest, SceneGeometryKeepsInstanceOrderAcrossATlasRebuild) {
    Scene scene;
    scene.meshes.push_back(MakeMesh({MakeTriangle(0, true, true, true),
                                     MakeTriangle(1, true, true, true)}));
    scene.nodes.push_back(MakeNode(0));
    scene.nodes.push_back(MakeNode(0));

    auto geometry = rendercore::SceneGeometry::Build(Device(), scene);
    ASSERT_TRUE(geometry.IsValid());

    const auto before = geometry.Instances();
    ASSERT_EQ(before.size(), 4u) << "two nodes times two primitives";

    scene.nodes[1].transform = glm::translate(glm::mat4(1.0f), {0.0f, 2.0f, 0.0f});
    geometry.RebuildTlas(Device(), scene);

    ASSERT_TRUE(geometry.IsValid());
    ASSERT_EQ(geometry.Instances().size(), before.size());
    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(geometry.Instances()[i].vertexOffset, before[i].vertexOffset) << "instance " << i;
        EXPECT_EQ(geometry.Instances()[i].materialId, before[i].materialId) << "instance " << i;
    }
}

TEST_F(VulkanDeviceTest, SceneGeometryRejectsASceneWithNoPrimitives) {
    Scene scene;
    scene.meshes.push_back(Mesh{});

    auto geometry = rendercore::SceneGeometry::Build(Device(), scene);

    EXPECT_FALSE(geometry.IsValid()) << "nothing to trace is not an empty TLAS";
}
