// ============================================================================
// Quantiloom - Unit Tests for scene/SceneMerge.hpp
// ============================================================================
// Tests cover:
// - Index offsetting: nodes into meshes, primitives into materials, materials
//   into textures
// - Node names carrying their model's prefix
// - Material names kept bare unless they collide
// - The rest pose landing on every node of the appended model
// ============================================================================

#include <gtest/gtest.h>

#include "scene/SceneMerge.hpp"

#include <glm/gtc/matrix_transform.hpp>

using namespace quantiloom;
using namespace quantiloom::scene;

namespace {

/// One node, one mesh, one primitive, one material, one texture -- the
/// smallest thing whose indices can be wrong.
Scene MakeScene(const String& materialName, const String& nodeName, i32 baseColorTexture) {
    Scene scene;

    Texture texture;
    texture.width = 1;
    texture.height = 1;
    scene.textures.push_back(std::move(texture));

    Material material;
    material.name = materialName;
    material.baseColorTextureIndex = baseColorTexture;
    scene.materials.push_back(std::move(material));

    GeometryPrimitive primitive;
    primitive.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    primitive.indices = {0, 1, 2};
    primitive.materialId = 0;

    Mesh mesh;
    mesh.name = "mesh";
    mesh.primitives.push_back(std::move(primitive));
    scene.meshes.push_back(std::move(mesh));

    SceneNode node;
    node.meshIndex = 0;
    node.name = nodeName;
    scene.nodes.push_back(node);

    return scene;
}

}  // namespace

TEST(SceneMergeTest, IndicesAreOffsetIntoTheMergedArrays) {
    Scene into = MakeScene("Ground", "Plane", 0);
    Scene part = MakeScene("Paint", "Body", 0);

    const AppendReport report = AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    ASSERT_EQ(into.nodes.size(), 2u);
    ASSERT_EQ(into.meshes.size(), 2u);
    ASSERT_EQ(into.materials.size(), 2u);
    ASSERT_EQ(into.textures.size(), 2u);

    EXPECT_EQ(report.firstNode, 1u);
    EXPECT_EQ(report.nodeCount, 1u);
    EXPECT_EQ(report.firstMaterial, 1u);

    // The appended node points at the appended mesh...
    EXPECT_EQ(into.nodes[1].meshIndex, 1u);
    // ...whose primitive points at the appended material...
    EXPECT_EQ(into.meshes[1].primitives[0].materialId, 1u);
    // ...which points at the appended texture.
    EXPECT_EQ(into.materials[1].baseColorTextureIndex, 1);
    // And the original is untouched.
    EXPECT_EQ(into.nodes[0].meshIndex, 0u);
    EXPECT_EQ(into.materials[0].baseColorTextureIndex, 0);
}

TEST(SceneMergeTest, AbsentTextureIndicesStayAbsent) {
    Scene into = MakeScene("Ground", "Plane", 0);
    Scene part = MakeScene("Paint", "Body", -1);

    AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    EXPECT_EQ(into.materials[1].baseColorTextureIndex, -1);
    EXPECT_EQ(into.materials[1].normalTextureIndex, -1);
}

TEST(SceneMergeTest, NodeNamesCarryTheirModelPrefix) {
    Scene into;
    Scene part = MakeScene("Paint", "Body", -1);

    AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    ASSERT_EQ(into.nodes.size(), 1u);
    EXPECT_EQ(into.nodes[0].name, "car/Body");
}

TEST(SceneMergeTest, AnUnnamedNodeStillGetsAUniqueName) {
    Scene into;
    Scene part = MakeScene("Paint", "", -1);

    AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    ASSERT_EQ(into.nodes.size(), 1u);
    EXPECT_EQ(into.nodes[0].name, "car/node_0");
}

TEST(SceneMergeTest, MaterialNamesStayBareUnlessTheyCollide) {
    Scene into = MakeScene("Paint", "Plane", -1);
    Scene part = MakeScene("Paint", "Body", -1);

    const AppendReport report = AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    EXPECT_EQ(report.renamedMaterials, 1u);
    EXPECT_EQ(into.materials[0].name, "Paint");
    EXPECT_EQ(into.materials[1].name, "car/Paint");
}

TEST(SceneMergeTest, ANameThatDoesNotCollideIsLeftAlone) {
    Scene into = MakeScene("Ground", "Plane", -1);
    Scene part = MakeScene("Paint", "Body", -1);

    const AppendReport report = AppendScene(into, std::move(part), "car", glm::mat4(1.0f));

    EXPECT_EQ(report.renamedMaterials, 0u);
    EXPECT_EQ(into.materials[1].name, "Paint");
}

TEST(SceneMergeTest, TheRestPosePremultipliesEveryAppendedNode) {
    Scene into;
    Scene part = MakeScene("Paint", "Body", -1);
    part.nodes[0].transform = glm::translate(glm::mat4(1.0f), glm::vec3(1, 0, 0));

    const glm::mat4 rest = glm::translate(glm::mat4(1.0f), glm::vec3(0, 5, 0));
    AppendScene(into, std::move(part), "car", rest);

    const glm::vec3 placed = glm::vec3(into.nodes[0].transform[3]);
    EXPECT_FLOAT_EQ(placed.x, 1.0f);
    EXPECT_FLOAT_EQ(placed.y, 5.0f);
}

TEST(SceneMergeTest, TheFirstModelSaysWhatTheSceneIs) {
    Scene into;
    Scene first = MakeScene("Paint", "Body", -1);
    first.name = "Car";
    first.lambda_min = 400.0f;

    Scene second = MakeScene("Ground", "Plane", -1);
    second.name = "Terrain";
    second.lambda_min = 8000.0f;

    AppendScene(into, std::move(first), "car", glm::mat4(1.0f));
    AppendScene(into, std::move(second), "ground", glm::mat4(1.0f));

    EXPECT_EQ(into.name, "Car");
    EXPECT_FLOAT_EQ(into.lambda_min, 400.0f);
}
