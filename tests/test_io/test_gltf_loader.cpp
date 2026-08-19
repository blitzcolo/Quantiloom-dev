// ============================================================================
// Quantiloom - Unit Tests for io/GltfLoader.hpp
// ============================================================================
// Tests cover:
// - Loading basic GLTF files
// - Mesh parsing and validation
// - Material parsing
// - Node/transform parsing
// - Error handling for missing files
// - Scene graph construction
// ============================================================================

#include <gtest/gtest.h>
#include "io/GltfLoader.hpp"
#include "scene/Scene.hpp"
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include <filesystem>
#include <fstream>

using namespace quantiloom;

// ============================================================================
// Test Fixture
// ============================================================================

class GltfLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Resolved from the repo root baked in at configure time. The previous
        // two candidates could never match: current_path()/".." resolves to the
        // parent of the repo when run from the root as documented, and the
        // /mnt/d fallback is a WSL path that this Windows .exe cannot see. Both
        // failing meant the whole suite reported "assets not found" and skipped,
        // although the submodule was checked out the entire time.
        assetsPath = std::filesystem::path(QUANTILOOM_SOURCE_ROOT)
                    / "assets" / "models" / "glTF-Sample-Assets" / "Models";

        hasTestAssets = std::filesystem::exists(assetsPath);
    }

    std::filesystem::path GetModelPath(const std::string& modelName) {
        return assetsPath / modelName / "glTF" / (modelName + ".gltf");
    }

    bool hasTestAssets;
    std::filesystem::path assetsPath;
};

// ============================================================================
// Basic Loading Tests
// ============================================================================

TEST_F(GltfLoaderTest, LoadNonexistentFile) {
    auto result = GltfLoader::LoadFromFile("nonexistent_file.gltf");
    EXPECT_FALSE(result.has_value());
}

TEST_F(GltfLoaderTest, LoadInvalidFile) {
    // Create a temporary invalid file
    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "invalid.gltf";
    std::ofstream file(tempPath);
    file << "This is not valid GLTF JSON!";
    file.close();

    auto result = GltfLoader::LoadFromFile(tempPath.string());
    EXPECT_FALSE(result.has_value());

    // Cleanup
    std::filesystem::remove(tempPath);
}

// ============================================================================
// Simple Model Tests (requires glTF-Sample-Assets)
// ============================================================================

TEST_F(GltfLoaderTest, LoadSimpleBox) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found at: " << modelPath;
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load Box.gltf: " << result.error();

    Scene& scene = *result;

    // Box model should have at least one mesh
    EXPECT_GT(scene.meshes.size(), 0) << "Box model should have at least one mesh";

    if (!scene.meshes.empty()) {
        const Mesh& mesh = scene.meshes[0];
        EXPECT_TRUE(mesh.IsValid()) << "Loaded mesh should be valid";
        EXPECT_GT(mesh.GetPrimitiveCount(), 0) << "Mesh should have at least one primitive";

        if (!mesh.primitives.empty()) {
            const GeometryPrimitive& prim = mesh.primitives[0];
            EXPECT_GT(prim.GetVertexCount(), 0) << "Primitive should have vertices";
            EXPECT_GT(prim.GetTriangleCount(), 0) << "Primitive should have triangles";

            // Box has 24 vertices (not 8) because vertices are duplicated per face
            // for proper per-face normals in glTF. Each face has 4 vertices.
            EXPECT_EQ(prim.GetVertexCount(), 24) << "Box should have 24 vertices (4 per face)";
            EXPECT_EQ(prim.GetTriangleCount(), 12) << "Box should have 12 triangles (2 per face)";
        }
    }

    // Box model should have at least one node
    EXPECT_GT(scene.nodes.size(), 0) << "Box model should have at least one node";

    // Box model should have at least one material
    EXPECT_GT(scene.materials.size(), 0) << "Box model should have at least one material";
}

TEST_F(GltfLoaderTest, LoadedSceneIsValid) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // Note: Scene::IsValid() requires spectral configuration which GLTF doesn't provide
    // So we check individual components instead

    for (const auto& mesh : scene.meshes) {
        EXPECT_TRUE(mesh.IsValid()) << "All meshes should be valid";
    }

    for (const auto& material : scene.materials) {
        EXPECT_TRUE(material.IsValid()) << "All materials should be valid";
    }

    for (const auto& node : scene.nodes) {
        EXPECT_TRUE(node.IsValid()) << "All nodes should be valid";
    }
}

TEST_F(GltfLoaderTest, LoadBoxWithBounds) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.meshes.size(), 0);

    const Mesh& mesh = scene.meshes[0];
    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    // Box should have reasonable bounds
    EXPECT_LE(minBound.x, maxBound.x);
    EXPECT_LE(minBound.y, maxBound.y);
    EXPECT_LE(minBound.z, maxBound.z);

    // Bounds should not be at origin (unless the box is degenerate)
    bool hasVolume = (maxBound.x > minBound.x) &&
                     (maxBound.y > minBound.y) &&
                     (maxBound.z > minBound.z);
    EXPECT_TRUE(hasVolume) << "Box should have non-zero volume";
}

TEST_F(GltfLoaderTest, LoadBoxTextured) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("BoxTextured");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "BoxTextured model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load BoxTextured.gltf: " << result.error();

    Scene& scene = *result;

    // BoxTextured should have textures
    EXPECT_GT(scene.textures.size(), 0) << "BoxTextured should have at least one texture";

    // Materials should reference textures
    if (!scene.materials.empty()) {
        bool hasTexturedMaterial = false;
        for (const auto& material : scene.materials) {
            if (material.HasTextures()) {
                hasTexturedMaterial = true;
                break;
            }
        }
        EXPECT_TRUE(hasTexturedMaterial) << "BoxTextured should have at least one material with textures";
    }
}

TEST_F(GltfLoaderTest, LoadBoxVertexColors) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("BoxVertexColors");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "BoxVertexColors model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load BoxVertexColors.gltf: " << result.error();

    Scene& scene = *result;
    ASSERT_GT(scene.meshes.size(), 0);

    const Mesh& mesh = scene.meshes[0];
    EXPECT_TRUE(mesh.IsValid());
}

// ============================================================================
// Material Parsing Tests
// ============================================================================

TEST_F(GltfLoaderTest, MaterialPBRProperties) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.materials.size(), 0);

    const Material& material = scene.materials[0];

    // Check PBR properties are in valid range
    EXPECT_GE(material.metallicFactor, 0.0f);
    EXPECT_LE(material.metallicFactor, 1.0f);

    EXPECT_GE(material.roughnessFactor, 0.0f);
    EXPECT_LE(material.roughnessFactor, 1.0f);

    // Base color should be non-negative
    EXPECT_GE(material.baseColorFactor.r, 0.0f);
    EXPECT_GE(material.baseColorFactor.g, 0.0f);
    EXPECT_GE(material.baseColorFactor.b, 0.0f);
    EXPECT_GE(material.baseColorFactor.a, 0.0f);
}

// ============================================================================
// Mesh Validation Tests
// ============================================================================

TEST_F(GltfLoaderTest, MeshTriangleIndices) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.meshes.size(), 0);

    for (const auto& mesh : scene.meshes) {
        for (const auto& prim : mesh.primitives) {
            // All indices should be valid
            u32 vertexCount = prim.GetVertexCount();
            for (u32 idx : prim.indices) {
                EXPECT_LT(idx, vertexCount)
                    << "Index " << idx << " out of range (vertex count: " << vertexCount << ")";
            }

            // Index count should be divisible by 3
            EXPECT_EQ(prim.indices.size() % 3, 0)
                << "Index count should be divisible by 3 (triangles)";
        }
    }
}

TEST_F(GltfLoaderTest, MeshAttributeConsistency) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.meshes.size(), 0);

    for (const auto& mesh : scene.meshes) {
        for (const auto& prim : mesh.primitives) {
            u32 vertexCount = prim.GetVertexCount();

            // If normals exist, count should match positions
            if (!prim.normals.empty()) {
                EXPECT_EQ(prim.normals.size(), vertexCount)
                    << "Normal count should match vertex count";
            }

            // If UVs exist, count should match positions
            if (!prim.uvs.empty()) {
                EXPECT_EQ(prim.uvs.size(), vertexCount)
                    << "UV count should match vertex count";
            }

            // If tangents exist, count should match positions
            if (!prim.tangents.empty()) {
                EXPECT_EQ(prim.tangents.size(), vertexCount)
                    << "Tangent count should match vertex count";
            }
        }
    }
}

// ============================================================================
// Node/Transform Tests
// ============================================================================

TEST_F(GltfLoaderTest, NodeTransformsValid) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.nodes.size(), 0);

    for (const auto& node : scene.nodes) {
        EXPECT_TRUE(node.IsValid()) << "Node transform should be valid";

        // Check that transform matrix doesn't contain NaN or Inf
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                EXPECT_FALSE(std::isnan(node.transform[i][j]))
                    << "Transform should not contain NaN";
                EXPECT_FALSE(std::isinf(node.transform[i][j]))
                    << "Transform should not contain Inf";
            }
        }
    }
}

TEST_F(GltfLoaderTest, NodeMeshReferences) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // All node mesh references should be valid
    for (const auto& node : scene.nodes) {
        EXPECT_LT(node.meshIndex, scene.meshes.size())
            << "Node mesh index should be within bounds";
    }
}

// ============================================================================
// Binary glTF (.glb) Tests
// ============================================================================

TEST_F(GltfLoaderTest, LoadBinaryGLB) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto glbPath = assetsPath / "Box" / "glTF-Binary" / "Box.glb";
    if (!std::filesystem::exists(glbPath)) {
        GTEST_SKIP() << "Box.glb not found";
    }

    auto result = GltfLoader::LoadFromFile(glbPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load Box.glb: " << result.error();

    Scene& scene = *result;
    EXPECT_GT(scene.meshes.size(), 0) << "GLB should have meshes";
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(GltfLoaderTest, HandleEmptyPath) {
    auto result = GltfLoader::LoadFromFile("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(GltfLoaderTest, HandleDirectoryPath) {
    // Try to load a directory instead of a file
    auto result = GltfLoader::LoadFromFile(std::filesystem::temp_directory_path().string());
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(GltfLoaderTest, LoadMultiplePrimitives) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // Count total primitives across all meshes
    u32 totalPrimitives = 0;
    u32 totalTriangles = 0;
    u32 totalVertices = 0;

    for (const auto& mesh : scene.meshes) {
        totalPrimitives += mesh.GetPrimitiveCount();
        totalTriangles += mesh.GetTotalTriangleCount();
        totalVertices += mesh.GetTotalVertexCount();
    }

    EXPECT_GT(totalPrimitives, 0) << "Scene should have at least one primitive";
    EXPECT_GT(totalTriangles, 0) << "Scene should have at least one triangle";
    EXPECT_GT(totalVertices, 0) << "Scene should have at least one vertex";
}

TEST_F(GltfLoaderTest, SceneResourceCounts) {
    if (!hasTestAssets) {
        GTEST_SKIP() << "glTF-Sample-Assets not found, skipping test";
    }

    auto modelPath = GetModelPath("Box");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "Box model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // Log resource counts for debugging
    u32 meshCount = static_cast<u32>(scene.meshes.size());
    u32 nodeCount = static_cast<u32>(scene.nodes.size());
    u32 materialCount = static_cast<u32>(scene.materials.size());

    EXPECT_GT(meshCount, 0) << "Should have at least one mesh";
    EXPECT_GT(nodeCount, 0) << "Should have at least one node";
    EXPECT_GT(materialCount, 0) << "Should have at least one material";
    // Textures are optional, so we don't check texture count
}

// ============================================================================
// KHR_materials_sheen and KHR_texture_transform
// ============================================================================
// No extension had a parsing test before these -- not transmission, not ior,
// not volume, not dispersion. These two get one because the sample assets pin
// down exactly the cases that are easy to get wrong: two texture slots sharing
// one image, and two slots of one material disagreeing about their transform.
// ============================================================================

TEST_F(GltfLoaderTest, ParsesSheenFactorsFromGlamVelvetSofa) {
    auto modelPath = GetModelPath("GlamVelvetSofa");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "GlamVelvetSofa model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());
    Scene& scene = *result;

    // The champagne fabric carries sheenColorFactor [0.9, 0.7, 0.6] and
    // sheenRoughnessFactor 0.6, and no sheen texture at all -- the factor-only
    // path, which is what most authored sheen looks like.
    const Material* champagne = nullptr;
    for (const auto& mat : scene.materials) {
        if (mat.name.find("champagne") != std::string::npos) {
            champagne = &mat;
            break;
        }
    }
    ASSERT_NE(champagne, nullptr) << "expected a champagne fabric material";

    EXPECT_NEAR(champagne->sheenColorFactor.r, 0.9f, 1e-3f);
    EXPECT_NEAR(champagne->sheenColorFactor.g, 0.7f, 1e-3f);
    EXPECT_NEAR(champagne->sheenColorFactor.b, 0.6f, 1e-3f);
    EXPECT_NEAR(champagne->sheenRoughnessFactor, 0.6f, 1e-3f);
    EXPECT_EQ(champagne->sheenColorTextureIndex, -1);
    EXPECT_EQ(champagne->sheenRoughnessTextureIndex, -1);
    EXPECT_TRUE(champagne->HasSheen());

    // Its normal map is scaled and rotated, and nothing else in the material
    // is -- the per-slot case, in the asset the user actually asked for.
    EXPECT_NEAR(champagne->normalUv.scale.x, 5.0f, 1e-3f);
    EXPECT_NEAR(champagne->normalUv.scale.y, 5.0f, 1e-3f);
    EXPECT_TRUE(champagne->baseColorUv.IsIdentity())
        << "base colour has no transform in this asset";
}

TEST_F(GltfLoaderTest, ParsesSheenTexturesAndTilingFromSheenCloth) {
    auto modelPath = GetModelPath("SheenCloth");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "SheenCloth model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());
    Scene& scene = *result;
    ASSERT_FALSE(scene.materials.empty());

    const Material& cloth = scene.materials[0];

    // Both sheen slots point at the same image: RGB carries the colour and
    // ALPHA the roughness, which is the packing the specification recommends.
    EXPECT_GE(cloth.sheenColorTextureIndex, 0);
    EXPECT_EQ(cloth.sheenColorTextureIndex, cloth.sheenRoughnessTextureIndex);

    // That shared image must still be marked sRGB. It is safe because sRGB
    // decoding leaves alpha alone, and this is the assertion that says so.
    ASSERT_LT(static_cast<size_t>(cloth.sheenColorTextureIndex), scene.textures.size());
    EXPECT_TRUE(scene.textures[cloth.sheenColorTextureIndex].isSRGB);

    // Every slot tiles 30x. Without KHR_texture_transform the weave renders as
    // one enormous smear, which is the whole reason it is supported here.
    EXPECT_NEAR(cloth.sheenColorUv.scale.x, 30.0f, 1e-3f);
    EXPECT_NEAR(cloth.sheenRoughnessUv.scale.x, 30.0f, 1e-3f);
    EXPECT_NEAR(cloth.baseColorUv.scale.x, 30.0f, 1e-3f);
    EXPECT_NEAR(cloth.normalUv.scale.x, 30.0f, 1e-3f);
}

// A material with no sheen extension must come back with the glTF defaults,
// which are also the values every sheen term in the shader folds away on.
TEST_F(GltfLoaderTest, MaterialWithoutSheenKeepsTheDefaults) {
    auto modelPath = GetModelPath("BoxTextured");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "BoxTextured model not found";
    }

    auto result = GltfLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());
    Scene& scene = *result;
    ASSERT_FALSE(scene.materials.empty());

    for (const auto& mat : scene.materials) {
        EXPECT_EQ(mat.sheenColorFactor, glm::vec3(0.0f));
        EXPECT_EQ(mat.sheenRoughnessFactor, 0.0f);
        EXPECT_FALSE(mat.HasSheen());
        EXPECT_TRUE(mat.baseColorUv.IsIdentity());
        EXPECT_TRUE(mat.normalUv.IsIdentity());
    }
}
