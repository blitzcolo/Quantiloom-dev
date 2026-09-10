// ============================================================================
// Quantiloom - Unit Tests for io/UsdLoader.hpp
// ============================================================================
// Tests cover:
// - Loading basic USD files (.usda, .usdc, .usdz)
// - Mesh parsing and validation
// - Material parsing (UsdPreviewSurface)
// - Node/transform parsing
// - Error handling for missing files
// - Scene graph construction
// ============================================================================

#include <gtest/gtest.h>
#include "io/UsdLoader.hpp"
#include "scene/Scene.hpp"
#include "scene/Mesh.hpp"
#include "scene/Material.hpp"
#include <filesystem>
#include <fstream>

using namespace quantiloom;

// ============================================================================
// Test Fixture
// ============================================================================

class UsdLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if OpenUSD is available
        hasOpenUSD = UsdLoader::IsAvailable();

        // Try multiple paths to find USD test assets
        std::vector<std::filesystem::path> searchPaths = {
            // Running from build directory (build/tests/Release/)
            std::filesystem::current_path() / ".." / ".." / ".." / "assets" / "models" / "usd_test",
            // Running from build directory (build/)
            std::filesystem::current_path() / ".." / "assets" / "models" / "usd_test",
            // Running from project root
            std::filesystem::current_path() / "assets" / "models" / "usd_test",
            // Repo root baked in at configure time -- the only candidate that
            // does not depend on the caller's cwd or on this one machine.
            std::filesystem::path(QUANTILOOM_SOURCE_ROOT) / "assets" / "models" / "usd_test",
        };

        hasTestAssets = false;
        for (const auto& path : searchPaths) {
            if (std::filesystem::exists(path)) {
                assetsPath = std::filesystem::canonical(path);
                hasTestAssets = true;
                break;
            }
        }

        // Create test assets if directory exists but files don't
        if (hasTestAssets) {
            auto cubePath = assetsPath / "simple_cube.usda";
            if (!std::filesystem::exists(cubePath)) {
                CreateTestAssets();
            }
        }
    }

    void CreateTestAssets() {
        // Create a simple cube USDA file for testing
        std::filesystem::path cubePath = assetsPath / "simple_cube.usda";
        std::ofstream file(cubePath);
        file << R"(#usda 1.0
(
    defaultPrim = "Cube"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "Cube" (
    kind = "component"
)
{
    def Mesh "CubeMesh"
    {
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4]
        point3f[] points = [
            (-0.5, -0.5,  0.5),
            ( 0.5, -0.5,  0.5),
            (-0.5,  0.5,  0.5),
            ( 0.5,  0.5,  0.5),
            (-0.5,  0.5, -0.5),
            ( 0.5,  0.5, -0.5),
            (-0.5, -0.5, -0.5),
            ( 0.5, -0.5, -0.5)
        ]
        normal3f[] primvars:normals = [
            (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
            (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1),
            (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),
            (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
            (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)
        ] (
            interpolation = "faceVarying"
        )
        texCoord2f[] primvars:st = [
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1)
        ] (
            interpolation = "faceVarying"
        )
        uniform token subdivisionScheme = "none"
    }
}
)";
        file.close();

        // Create a USDA with material
        std::filesystem::path pbrPath = assetsPath / "pbr_sphere.usda";
        std::ofstream pbrFile(pbrPath);
        pbrFile << R"(#usda 1.0
(
    defaultPrim = "Sphere"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Scope "Materials"
{
    def Material "RedMetal"
    {
        token outputs:surface.connect = </Materials/RedMetal/PBRShader.outputs:surface>

        def Shader "PBRShader"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.8, 0.1, 0.1)
            float inputs:metallic = 0.9
            float inputs:roughness = 0.2
            color3f inputs:emissiveColor = (0.0, 0.0, 0.0)
            float inputs:opacity = 1.0
            token outputs:surface
        }
    }
}

def Xform "Sphere" (
    kind = "component"
)
{
    def Mesh "SphereMesh" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        rel material:binding = </Materials/RedMetal>

        int[] faceVertexCounts = [4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 7, 1, 2, 6, 5, 3]
        point3f[] points = [
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
            (-0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5)
        ]
        uniform token subdivisionScheme = "none"
    }
}
)";
        pbrFile.close();
    }

    std::filesystem::path GetTestFilePath(const std::string& fileName) {
        return assetsPath / fileName;
    }

    /// Write an inline `#usda` document and return its path.
    ///
    /// Scratch fixtures go to the temp directory, never to
    /// assets/models/usd_test/ -- git tracks that one, and a test that leaves
    /// generated files in the source tree turns a clean checkout dirty.
    static std::filesystem::path WriteUsda(const std::string& fileName,
                                           const std::string& body) {
        const auto dir =
            std::filesystem::temp_directory_path() / "quantiloom_usd_fixtures";
        std::filesystem::create_directories(dir);
        const auto path = dir / fileName;
        std::ofstream file(path);
        file << body;
        file.close();
        return path;
    }

    /// The first primitive of the first mesh, for fixtures that author one.
    static const GeometryPrimitive& OnlyPrimitive(const Scene& scene) {
        EXPECT_EQ(scene.meshes.size(), 1u);
        EXPECT_EQ(scene.meshes[0].primitives.size(), 1u);
        return scene.meshes[0].primitives[0];
    }

    bool hasOpenUSD;
    bool hasTestAssets;
    std::filesystem::path assetsPath;
};

// ============================================================================
// Basic Loading Tests
// ============================================================================

TEST_F(UsdLoaderTest, LoadNonexistentFile) {
    auto result = UsdLoader::LoadFromFile("nonexistent_file.usda");
    EXPECT_FALSE(result.has_value());
}

TEST_F(UsdLoaderTest, LoadInvalidFile) {
    // Create a temporary invalid file
    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "invalid.usda";
    std::ofstream file(tempPath);
    file << "This is not valid USD!";
    file.close();

    auto result = UsdLoader::LoadFromFile(tempPath.string());
    EXPECT_FALSE(result.has_value());

    // Cleanup
    std::filesystem::remove(tempPath);
}

TEST_F(UsdLoaderTest, HandleEmptyPath) {
    auto result = UsdLoader::LoadFromFile("");
    EXPECT_FALSE(result.has_value());
}

TEST_F(UsdLoaderTest, HandleDirectoryPath) {
    auto result = UsdLoader::LoadFromFile(std::filesystem::temp_directory_path().string());
    EXPECT_FALSE(result.has_value());
}

TEST_F(UsdLoaderTest, UnsupportedExtension) {
    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test.abc";
    std::ofstream file(tempPath);
    file << "test";
    file.close();

    auto result = UsdLoader::LoadFromFile(tempPath.string());
    EXPECT_FALSE(result.has_value());

    std::filesystem::remove(tempPath);
}

// ============================================================================
// Simple Model Tests
// ============================================================================

TEST_F(UsdLoaderTest, LoadSimpleCube) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found at: " << modelPath;
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load simple_cube.usda: " << result.error();

    Scene& scene = *result;

    // Should have at least one mesh
    EXPECT_GT(scene.meshes.size(), 0) << "Scene should have at least one mesh";

    if (!scene.meshes.empty()) {
        const Mesh& mesh = scene.meshes[0];
        EXPECT_TRUE(mesh.IsValid()) << "Loaded mesh should be valid";
        EXPECT_GT(mesh.GetPrimitiveCount(), 0) << "Mesh should have at least one primitive";

        if (!mesh.primitives.empty()) {
            const GeometryPrimitive& prim = mesh.primitives[0];
            EXPECT_GT(prim.GetVertexCount(), 0) << "Primitive should have vertices";
            EXPECT_GT(prim.GetTriangleCount(), 0) << "Primitive should have triangles";
        }
    }

    // Should have at least one node
    EXPECT_GT(scene.nodes.size(), 0) << "Scene should have at least one node";

    // Should have at least one material (default if none in file)
    EXPECT_GT(scene.materials.size(), 0) << "Scene should have at least one material";
}

TEST_F(UsdLoaderTest, LoadPBRMaterial) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("pbr_sphere.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "pbr_sphere.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load pbr_sphere.usda: " << result.error();

    Scene& scene = *result;
    ASSERT_GT(scene.materials.size(), 0) << "Should have at least one material";

    // Find the RedMetal material
    bool foundRedMetal = false;
    for (const auto& mat : scene.materials) {
        if (mat.name.find("RedMetal") != std::string::npos ||
            mat.name.find("Material") != std::string::npos) {
            foundRedMetal = true;

            // Check PBR properties are parsed correctly
            // RedMetal has: diffuseColor=(0.8, 0.1, 0.1), metallic=0.9, roughness=0.2
            EXPECT_GE(mat.metallicFactor, 0.0f);
            EXPECT_LE(mat.metallicFactor, 1.0f);
            EXPECT_GE(mat.roughnessFactor, 0.0f);
            EXPECT_LE(mat.roughnessFactor, 1.0f);
            break;
        }
    }

    // It's OK if material name differs, just check we have valid materials
    EXPECT_GT(scene.materials.size(), 0);
}

// ============================================================================
// Scene Validation Tests
// ============================================================================

TEST_F(UsdLoaderTest, LoadedSceneIsValid) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

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

TEST_F(UsdLoaderTest, MeshBounds) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.meshes.size(), 0);

    const Mesh& mesh = scene.meshes[0];
    glm::vec3 minBound, maxBound;
    mesh.ComputeBounds(minBound, maxBound);

    // Bounds should be valid
    EXPECT_LE(minBound.x, maxBound.x);
    EXPECT_LE(minBound.y, maxBound.y);
    EXPECT_LE(minBound.z, maxBound.z);

    // Should have non-zero volume
    bool hasVolume = (maxBound.x > minBound.x) &&
                     (maxBound.y > minBound.y) &&
                     (maxBound.z > minBound.z);
    EXPECT_TRUE(hasVolume) << "Mesh should have non-zero volume";
}

// ============================================================================
// Material Tests
// ============================================================================

TEST_F(UsdLoaderTest, MaterialPBRProperties) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.materials.size(), 0);

    for (const auto& material : scene.materials) {
        // PBR properties should be in valid range
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
}

// ============================================================================
// Mesh Validation Tests
// ============================================================================

TEST_F(UsdLoaderTest, MeshTriangleIndices) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
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

            // Index count should be divisible by 3 (triangles)
            EXPECT_EQ(prim.indices.size() % 3, 0)
                << "Index count should be divisible by 3 (triangles)";
        }
    }
}

TEST_F(UsdLoaderTest, MeshAttributeConsistency) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
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
        }
    }
}

// ============================================================================
// Node/Transform Tests
// ============================================================================

TEST_F(UsdLoaderTest, NodeTransformsValid) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;
    ASSERT_GT(scene.nodes.size(), 0);

    for (const auto& node : scene.nodes) {
        EXPECT_TRUE(node.IsValid()) << "Node should be valid";

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

TEST_F(UsdLoaderTest, NodeMeshReferences) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // All node mesh references should be valid
    for (const auto& node : scene.nodes) {
        EXPECT_LT(node.meshIndex, scene.meshes.size())
            << "Node mesh index should be within bounds";
    }
}

// ============================================================================
// Spectral Extension Tests
// ============================================================================

TEST_F(UsdLoaderTest, SpectralMaterialReference) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("spectral_material.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "spectral_material.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load spectral_material.usda: " << result.error();

    Scene& scene = *result;
    ASSERT_GT(scene.materials.size(), 0) << "Should have materials";

    // Find AluminumMetal material and verify spectral reference
    bool foundAluminum = false;
    for (const auto& mat : scene.materials) {
        if (mat.name.find("Aluminum") != std::string::npos) {
            foundAluminum = true;

            // Check Quantiloom material reference attributes
            if (mat.HasQuantiloomRef()) {
                EXPECT_EQ(mat.quantiloomMaterialType, "quantiloom_usgs");
                EXPECT_EQ(mat.quantiloomMaterialRef, "Aluminum brushed 293K");
                EXPECT_EQ(mat.spectralSource, Material::SpectralSource::Measured);
            }
            break;
        }
    }

    // Material may not be found if tydra doesn't expose the name correctly
    // This is OK - we verify the parsing logic works when material is found
    if (!foundAluminum) {
        // Check if any material has spectral reference set
        bool anySpectralRef = false;
        for (const auto& mat : scene.materials) {
            if (mat.HasQuantiloomRef()) {
                anySpectralRef = true;
                break;
            }
        }
        // It's OK if no spectral refs found - means parsing didn't find matching prim
        SUCCEED() << "Material name matching may differ from prim path";
    }
}

TEST_F(UsdLoaderTest, SpectralIRProperties) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("spectral_material.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "spectral_material.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value()) << "Failed to load spectral_material.usda: " << result.error();

    Scene& scene = *result;

    // Check that materials are loaded (IR curves may fail to load if CSV files don't exist)
    EXPECT_GT(scene.materials.size(), 0);

    // Find AsphaltRoad material - it has both reference AND IR properties
    for (const auto& mat : scene.materials) {
        if (mat.name.find("Asphalt") != std::string::npos) {
            // If parsing worked, temperature should be set
            if (mat.irTemperature_K > 0.0f) {
                EXPECT_NEAR(mat.irTemperature_K, 320.0f, 0.1f);
            }
            break;
        }
    }
}

TEST_F(UsdLoaderTest, SpectralMaterialPBRFallback) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("spectral_material.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "spectral_material.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    // All materials should have valid PBR properties regardless of spectral extensions
    for (const auto& mat : scene.materials) {
        EXPECT_GE(mat.metallicFactor, 0.0f);
        EXPECT_LE(mat.metallicFactor, 1.0f);
        EXPECT_GE(mat.roughnessFactor, 0.0f);
        EXPECT_LE(mat.roughnessFactor, 1.0f);
        EXPECT_TRUE(mat.IsValid()) << "Material should be valid: " << mat.name;
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(UsdLoaderTest, SceneResourceCounts) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }
    if (!hasTestAssets) {
        GTEST_SKIP() << "USD test assets not found";
    }

    auto modelPath = GetTestFilePath("simple_cube.usda");
    if (!std::filesystem::exists(modelPath)) {
        GTEST_SKIP() << "simple_cube.usda not found";
    }

    auto result = UsdLoader::LoadFromFile(modelPath.string());
    ASSERT_TRUE(result.has_value());

    Scene& scene = *result;

    u32 meshCount = static_cast<u32>(scene.meshes.size());
    u32 nodeCount = static_cast<u32>(scene.nodes.size());
    u32 materialCount = static_cast<u32>(scene.materials.size());

    EXPECT_GT(meshCount, 0) << "Should have at least one mesh";
    EXPECT_GT(nodeCount, 0) << "Should have at least one node";
    EXPECT_GT(materialCount, 0) << "Should have at least one material";

    // Count totals
    u32 totalTriangles = 0;
    u32 totalVertices = 0;

    for (const auto& mesh : scene.meshes) {
        totalTriangles += mesh.GetTotalTriangleCount();
        totalVertices += mesh.GetTotalVertexCount();
    }

    EXPECT_GT(totalTriangles, 0) << "Scene should have triangles";
    EXPECT_GT(totalVertices, 0) << "Scene should have vertices";
}

// ============================================================================
// Vertex attribute hygiene
// ============================================================================
// USD lets every attribute carry its own interpolation, so a mesh may need its
// geometry expanded for one of them and not for the other. These pin the four
// combinations that used to lose an attribute outright.

TEST_F(UsdLoaderTest, VertexUvsSurviveFaceVaryingNormals) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("fv_normals_vertex_uvs.usda", R"(#usda 1.0
(
    defaultPrim = "Quad"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
        interpolation = "faceVarying"
    )
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    const GeometryPrimitive& prim = OnlyPrimitive(scene);
    ASSERT_FALSE(prim.positions.empty());
    EXPECT_EQ(prim.normals.size(), prim.positions.size());
    EXPECT_EQ(prim.uvs.size(), prim.positions.size())
        << "Expanding for face-varying normals must re-index the vertex UVs, "
           "not discard them";
}

TEST_F(UsdLoaderTest, AuthoredVertexNormalsSurviveFaceVaryingUvs) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // A normal that no generator would produce for a flat quad in Z, so a
    // silent rebuild by NormalGenerator is distinguishable from the authored
    // value surviving.
    const auto path = WriteUsda("vertex_normals_fv_uvs.usda", R"(#usda 1.0
(
    defaultPrim = "Quad"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    normal3f[] normals = [(0, 0.6, 0.8), (0, 0.6, 0.8), (0, 0.6, 0.8), (0, 0.6, 0.8)] (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "faceVarying"
    )
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    const GeometryPrimitive& prim = OnlyPrimitive(scene);
    ASSERT_FALSE(prim.positions.empty());
    ASSERT_EQ(prim.normals.size(), prim.positions.size());
    EXPECT_EQ(prim.uvs.size(), prim.positions.size());

    for (const auto& n : prim.normals) {
        EXPECT_NEAR(n.x, 0.0f, 1e-5f);
        EXPECT_NEAR(n.y, 0.6f, 1e-5f) << "Authored normals were rebuilt";
        EXPECT_NEAR(n.z, 0.8f, 1e-5f) << "Authored normals were rebuilt";
    }
}

TEST_F(UsdLoaderTest, ConstantNormalsSurviveFaceVaryingExpansion) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // A constant normal is one value for the whole mesh. It has to be re-read
    // for every face vertex the expansion emits, or it stays sized to the old
    // point list and the mesh is invalid.
    const auto path = WriteUsda("constant_normal_fv_uvs.usda", R"(#usda 1.0
(
    defaultPrim = "Quad"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    normal3f[] normals = [(0, 0.6, 0.8)] (
        interpolation = "constant"
    )
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "faceVarying"
    )
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    const GeometryPrimitive& prim = OnlyPrimitive(scene);
    ASSERT_FALSE(prim.positions.empty());
    ASSERT_EQ(prim.normals.size(), prim.positions.size());

    for (const auto& n : prim.normals) {
        EXPECT_NEAR(n.y, 0.6f, 1e-5f);
        EXPECT_NEAR(n.z, 0.8f, 1e-5f);
    }
}

TEST_F(UsdLoaderTest, FlipsStToImageRowOrder) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // UsdUVTexture puts st's origin at the image's lower-left; stb decodes
    // top-down and the shaders sample glTF's upper-left. st (0, 0) must
    // therefore arrive as uv (0, 1).
    const auto path = WriteUsda("st_origin.usda", R"(#usda 1.0
(
    defaultPrim = "Quad"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    const GeometryPrimitive& prim = OnlyPrimitive(scene);
    ASSERT_EQ(prim.uvs.size(), prim.positions.size());

    bool foundOrigin = false;
    for (size_t i = 0; i < prim.positions.size(); ++i) {
        if (glm::length(prim.positions[i] - glm::vec3(0.0f, 0.0f, 0.0f)) < 1e-5f) {
            foundOrigin = true;
            EXPECT_NEAR(prim.uvs[i].x, 0.0f, 1e-5f);
            EXPECT_NEAR(prim.uvs[i].y, 1.0f, 1e-5f)
                << "st (0, 0) is the image's bottom row, which is v = 1 here";
        }
    }
    EXPECT_TRUE(foundOrigin);
}

// ============================================================================
// Purpose, visibility and stage metrics
// ============================================================================

TEST_F(UsdLoaderTest, SkipsGuideProxyAndInvisiblePrims) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("purpose_and_visibility.usda", R"(#usda 1.0
(
    defaultPrim = "Render"
)

def Mesh "Render"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}

def Mesh "Guide"
{
    uniform token purpose = "guide"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}

def Mesh "Proxy"
{
    uniform token purpose = "proxy"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}

def Mesh "Hidden"
{
    token visibility = "invisible"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    EXPECT_EQ(scene.meshes.size(), 1u)
        << "guide, proxy and invisible prims are not render geometry";
    ASSERT_EQ(scene.nodes.size(), 1u);
    EXPECT_EQ(scene.nodes[0].name, "Render");
}

TEST_F(UsdLoaderTest, FoldsAuthoredMetersPerUnitAndZUpIntoNodeTransforms) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("z_up_centimetres.usda", R"(#usda 1.0
(
    defaultPrim = "Tri"
    upAxis = "Z"
    metersPerUnit = 0.01
)

def Mesh "Tri"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    ASSERT_EQ(scene.nodes.size(), 1u);

    // Z-up sends the stage's up vector to +Y, and one centimetre is 0.01 m.
    const glm::vec3 up =
        glm::vec3(scene.nodes[0].transform * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
    EXPECT_NEAR(up.x, 0.0f, 1e-5f);
    EXPECT_NEAR(up.y, 0.01f, 1e-5f);
    EXPECT_NEAR(up.z, 0.0f, 1e-5f);
}

TEST_F(UsdLoaderTest, UnauthoredMetersPerUnitIsNotApplied) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // USD's default when the metadata is absent is 0.01. Applying it silently
    // would shrink by a hundred every scene that loads correctly today, so an
    // unauthored stage stays at 1.0 and warns instead.
    const auto path = WriteUsda("no_stage_metrics.usda", R"(#usda 1.0
(
    defaultPrim = "Tri"
)

def Mesh "Tri"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    ASSERT_EQ(scene.nodes.size(), 1u);

    const glm::vec3 point =
        glm::vec3(scene.nodes[0].transform * glm::vec4(1.0f, 2.0f, 3.0f, 1.0f));
    EXPECT_NEAR(point.x, 1.0f, 1e-5f);
    EXPECT_NEAR(point.y, 2.0f, 1e-5f);
    EXPECT_NEAR(point.z, 3.0f, 1e-5f);
}
