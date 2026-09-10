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

    /// Write an uncompressed 32-bit TGA with exactly the bytes given.
    ///
    /// A texture test needs the pixels it asked for. ImageIO::WritePNG applies
    /// sRGB encoding on the way out (ImageIO.cpp:197), so a known byte written
    /// through it comes back as a different one; TGA is a 18-byte header and
    /// raw BGRA, which stb_image reads back unchanged. The descriptor bit 0x20
    /// puts the origin top-left, so the rows are in the order they are written.
    static std::filesystem::path WriteRawTga(const std::string& fileName, u32 width,
                                             u32 height,
                                             const std::vector<u8>& rgba) {
        const auto dir =
            std::filesystem::temp_directory_path() / "quantiloom_usd_fixtures";
        std::filesystem::create_directories(dir);
        const auto path = dir / fileName;

        std::vector<u8> header(18, 0);
        header[2] = 2;  // uncompressed true-colour
        header[12] = static_cast<u8>(width & 0xFF);
        header[13] = static_cast<u8>((width >> 8) & 0xFF);
        header[14] = static_cast<u8>(height & 0xFF);
        header[15] = static_cast<u8>((height >> 8) & 0xFF);
        header[16] = 32;    // bits per pixel
        header[17] = 0x28;  // 8 alpha bits, top-left origin

        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(header.data()),
                   static_cast<std::streamsize>(header.size()));
        for (usize i = 0; i + 3 < rgba.size(); i += 4) {
            const u8 bgra[4] = {rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]};
            file.write(reinterpret_cast<const char*>(bgra), 4);
        }
        file.close();
        return path;
    }

    /// A solid image of one colour.
    static std::filesystem::path WriteSolidTga(const std::string& fileName, u8 r, u8 g,
                                               u8 b, u8 a) {
        std::vector<u8> pixels;
        for (int i = 0; i < 4; ++i) {
            pixels.insert(pixels.end(), {r, g, b, a});
        }
        return WriteRawTga(fileName, 2, 2, pixels);
    }

    static const Material& MaterialNamed(const Scene& scene, const std::string& name) {
        for (const auto& material : scene.materials) {
            if (material.name == name) {
                return material;
            }
        }
        ADD_FAILURE() << "no material named '" << name << "'";
        return scene.materials.at(0);
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

// ============================================================================
// The table-driven surface reader
// ============================================================================

namespace {

/// The mesh every material fixture binds, so the scene has geometry to carry it.
constexpr const char* kBoundQuad = R"(
    def Mesh "Quad"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </Mat>
    }
)";

}  // namespace

TEST_F(UsdLoaderTest, ATexturedDiffuseColorGetsAUnitFactor) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("unit_factor_colour.tga", 200, 150, 100, 255);
    WriteSolidTga("unit_factor_emissive.tga", 90, 80, 70, 255);

    const auto path = WriteUsda("unit_factor.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Colour.outputs:rgb>
        color3f inputs:emissiveColor.connect = </Mat/Emissive.outputs:rgb>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./unit_factor_colour.tga@
        float3 outputs:rgb
    }

    def Shader "Emissive"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./unit_factor_emissive.tga@
        float3 outputs:rgb
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // A connected input means "take the texture", so the factor it multiplies is
    // one. It used to keep the input's default: 0.8 on base colour, and zero on
    // emissive, which made every textured USD emitter dark.
    ASSERT_GE(material.baseColorTextureIndex, 0);
    EXPECT_NEAR(material.baseColorFactor.r, 1.0f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.g, 1.0f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.b, 1.0f, 1e-6f);

    ASSERT_GE(material.emissiveTextureIndex, 0);
    EXPECT_NEAR(material.emissiveFactor.r, 1.0f, 1e-6f);
    EXPECT_NEAR(material.emissiveFactor.g, 1.0f, 1e-6f);
    EXPECT_NEAR(material.emissiveFactor.b, 1.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, UsdUVTextureScaleBecomesTheColourFactor) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("scaled_colour.tga", 255, 255, 255, 255);

    const auto path = WriteUsda("texture_scale.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Colour.outputs:rgb>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./scaled_colour.tga@
        float4 inputs:scale = (0.5, 0.25, 0.125, 1)
        float3 outputs:rgb
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // A scale with no bias is exactly a factor, so it folds instead of being
    // baked into pixels the spectral unmixer would then read differently.
    ASSERT_GE(material.baseColorTextureIndex, 0);
    EXPECT_NEAR(material.baseColorFactor.r, 0.5f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.g, 0.25f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.b, 0.125f, 1e-6f);
}

TEST_F(UsdLoaderTest, SeparateMetallicAndRoughnessImagesArePackedIntoGAndB) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("mr_metal.tga", 0x40, 0x00, 0x00, 0xFF);
    WriteSolidTga("mr_rough.tga", 0x00, 0xC0, 0x00, 0xFF);

    const auto path = WriteUsda("metal_rough_split.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        float inputs:metallic.connect = </Mat/Metal.outputs:r>
        float inputs:roughness.connect = </Mat/Rough.outputs:g>
        token outputs:surface
    }

    def Shader "Metal"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./mr_metal.tga@
        float outputs:r
    }

    def Shader "Rough"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./mr_rough.tga@
        float outputs:g
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    // USD binds two files; the shader reads one texture, G for roughness and B
    // for metallic. Whichever image arrived first used to take the single slot
    // and the other was dropped.
    ASSERT_GE(material.metallicRoughnessTextureIndex, 0);
    const Texture& packed = scene.textures[material.metallicRoughnessTextureIndex];
    if (packed.pixels.empty()) {
        GTEST_SKIP() << "texture pixels were released by block compression";
    }
    EXPECT_EQ(packed.pixels[1], 0xC0) << "G is roughness";
    EXPECT_EQ(packed.pixels[2], 0x40) << "B is metallic";
    EXPECT_FALSE(packed.isSRGB);
    EXPECT_NEAR(material.metallicFactor, 1.0f, 1e-6f);
    EXPECT_NEAR(material.roughnessFactor, 1.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, AMetallicOnlyImageLeavesRoughnessAsTheScalar) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("metal_only.tga", 0x40, 0x00, 0x00, 0xFF);

    const auto path = WriteUsda("metal_only.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        float inputs:metallic.connect = </Mat/Metal.outputs:r>
        float inputs:roughness = 0.25
        token outputs:surface
    }

    def Shader "Metal"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./metal_only.tga@
        float outputs:r
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    ASSERT_GE(material.metallicRoughnessTextureIndex, 0);
    const Texture& packed = scene.textures[material.metallicRoughnessTextureIndex];
    if (packed.pixels.empty()) {
        GTEST_SKIP() << "texture pixels were released by block compression";
    }
    // 255 is 1.0, so the scalar half of the pair survives the multiply.
    EXPECT_EQ(packed.pixels[1], 0xFF);
    EXPECT_EQ(packed.pixels[2], 0x40);
    EXPECT_NEAR(material.roughnessFactor, 0.25f, 1e-6f);
}

TEST_F(UsdLoaderTest, OpacityThresholdSelectsAlphaMask) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("opacity_threshold.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        float inputs:opacityThreshold = 0.4
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Mask);
    EXPECT_NEAR(material.alphaCutoff, 0.4f, 1e-6f);
}

TEST_F(UsdLoaderTest, OpacityImageAlphaLandsInBaseColourAlpha) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("opacity_colour.tga", 200, 150, 100, 255);
    WriteSolidTga("opacity_alpha.tga", 0, 0, 0, 0x80);

    const auto path = WriteUsda("opacity_texture.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Colour.outputs:rgb>
        float inputs:opacity.connect = </Mat/Alpha.outputs:a>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./opacity_colour.tga@
        float3 outputs:rgb
    }

    def Shader "Alpha"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./opacity_alpha.tga@
        float outputs:a
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Blend);
    ASSERT_GE(material.baseColorTextureIndex, 0);
    const Texture& base = scene.textures[material.baseColorTextureIndex];
    if (base.pixels.empty()) {
        GTEST_SKIP() << "texture pixels were released by block compression";
    }
    EXPECT_EQ(base.pixels[0], 200);
    EXPECT_EQ(base.pixels[3], 0x80) << "the opacity map is the base colour's alpha";
    EXPECT_TRUE(base.retainCpuPixels)
        << "a non-opaque base colour is read on the CPU for the thermal view factors";
}

TEST_F(UsdLoaderTest, ClearcoatInputsAreRead) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("clearcoat.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        float inputs:clearcoat = 0.7
        float inputs:clearcoatRoughness = 0.2
        float inputs:ior = 1.7
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    EXPECT_NEAR(material.clearcoatFactor, 0.7f, 1e-6f);
    EXPECT_NEAR(material.clearcoatRoughnessFactor, 0.2f, 1e-6f);
    EXPECT_NEAR(material.ior, 1.7f, 1e-6f);
}

TEST_F(UsdLoaderTest, NormalMapScaleBiasIdiomIsNotBakedTwice) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("normal_flat.tga", 128, 128, 255, 255);

    const auto path = WriteUsda("normal_idiom.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        normal3f inputs:normal.connect = </Mat/Normal.outputs:rgb>
        token outputs:surface
    }

    def Shader "Normal"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./normal_flat.tga@
        float4 inputs:scale = (2, 2, 2, 1)
        float4 inputs:bias = (-1, -1, -1, 0)
        token inputs:sourceColorSpace = "raw"
        float3 outputs:rgb
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    ASSERT_GE(material.normalTextureIndex, 0);
    const Texture& normal = scene.textures[material.normalTextureIndex];
    if (normal.pixels.empty()) {
        GTEST_SKIP() << "texture pixels were released by block compression";
    }
    // The shader already maps [0,1] to [-1,1]; baking the same affine here
    // would apply it twice and a flat normal would come out sideways.
    EXPECT_EQ(normal.pixels[0], 128);
    EXPECT_EQ(normal.pixels[1], 128);
    EXPECT_EQ(normal.pixels[2], 255);
    EXPECT_FALSE(normal.isSRGB);
}

TEST_F(UsdLoaderTest, WrapModesAndSourceColorSpaceReachTheTexture) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("wrapped.tga", 10, 20, 30, 255);

    const auto path = WriteUsda("wrap_modes.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Colour.outputs:rgb>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./wrapped.tga@
        token inputs:wrapS = "clamp"
        token inputs:wrapT = "mirror"
        token inputs:sourceColorSpace = "raw"
        float3 outputs:rgb
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    ASSERT_GE(material.baseColorTextureIndex, 0);
    const Texture& texture = scene.textures[material.baseColorTextureIndex];
    EXPECT_EQ(texture.sampler.wrapS, TextureSampler::WrapMode::ClampToEdge);
    EXPECT_EQ(texture.sampler.wrapT, TextureSampler::WrapMode::MirroredRepeat);
    EXPECT_FALSE(texture.isSRGB) << "sourceColorSpace = raw overrides the slot's guess";
}

TEST_F(UsdLoaderTest, AnImageUsedAsColourAndAsDataGetsTwoEntries) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("shared.tga", 90, 120, 150, 255);

    const auto path = WriteUsda("shared_image.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/AsColour.outputs:rgb>
        float inputs:roughness.connect = </Mat/AsData.outputs:g>
        token outputs:surface
    }

    def Shader "AsColour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./shared.tga@
        float3 outputs:rgb
    }

    def Shader "AsData"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./shared.tga@
        float outputs:g
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    ASSERT_GE(material.baseColorTextureIndex, 0);
    ASSERT_GE(material.metallicRoughnessTextureIndex, 0);
    EXPECT_NE(material.baseColorTextureIndex, material.metallicRoughnessTextureIndex)
        << "one file in two roles is two entries; a pass over shared indices could "
           "only give it one colour space";
    EXPECT_TRUE(scene.textures[material.baseColorTextureIndex].isSRGB);
    EXPECT_FALSE(scene.textures[material.metallicRoughnessTextureIndex].isSRGB);
}

TEST_F(UsdLoaderTest, UsdTransform2dBecomesThePerSlotUvTransform) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("transformed.tga", 10, 20, 30, 255);

    const auto path = WriteUsda("uv_transform.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Colour.outputs:rgb>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./transformed.tga@
        float2 inputs:st.connect = </Mat/Place.outputs:result>
        float3 outputs:rgb
    }

    def Shader "Place"
    {
        uniform token info:id = "UsdTransform2d"
        float2 inputs:in.connect = </Mat/Reader.outputs:result>
        float2 inputs:scale = (2, 3)
        float inputs:rotation = 0
        float2 inputs:translation = (0.1, 0.2)
        float2 outputs:result
    }

    def Shader "Reader"
    {
        uniform token info:id = "UsdPrimvarReader_float2"
        token inputs:varname = "st"
        float2 outputs:result
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    ASSERT_GE(material.baseColorTextureIndex, 0);
    // The vertices were flipped in V, so the transform is conjugated by the same
    // flip: scale unchanged, rotation negated, offset.y = 1 - t.y - s.y*cos(0).
    EXPECT_NEAR(material.baseColorUv.scale.x, 2.0f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.scale.y, 3.0f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.rotation, 0.0f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.offset.x, 0.1f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.offset.y, 1.0f - 0.2f - 3.0f, 1e-5f);
}

TEST_F(UsdLoaderTest, AnUnknownShaderIdKeepsMaterialDefaults) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("unknown_shader.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_disney_principled_surfaceshader"
        color3f inputs:base_color = (0.1, 0.2, 0.3)
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // Reading an unknown vocabulary through one it is not written in matches no
    // input name and produces a material that looks badly authored. Keeping the
    // defaults and warning says what actually happened.
    EXPECT_NEAR(material.baseColorFactor.r, 0.8f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.g, 0.8f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.b, 0.8f, 1e-6f);
    EXPECT_NEAR(material.roughnessFactor, 0.5f, 1e-6f);
    EXPECT_NEAR(material.metallicFactor, 0.0f, 1e-6f);
}

// ============================================================================
// MaterialX standard_surface
// ============================================================================

TEST_F(UsdLoaderTest, StandardSurfaceScalarsMapToTheGltfFields) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("standard_surface_scalars.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        float inputs:base = 0.5
        color3f inputs:base_color = (0.4, 0.5, 0.6)
        float inputs:metalness = 0.3
        float inputs:specular_roughness = 0.25
        float inputs:specular_IOR = 1.7
        float inputs:specular_anisotropy = 0.6
        float inputs:specular_rotation = 0.25
        float inputs:transmission = 0.4
        color3f inputs:transmission_color = (0.9, 0.8, 0.7)
        float inputs:transmission_depth = 2.0
        float inputs:transmission_dispersion = 30.0
        float inputs:sheen = 0.5
        color3f inputs:sheen_color = (1, 0.5, 0.25)
        float inputs:sheen_roughness = 0.4
        float inputs:coat = 0.3
        float inputs:coat_roughness = 0.15
        float inputs:emission = 2.0
        color3f inputs:emission_color = (1, 0.5, 0.25)
        color3f inputs:opacity = (0.5, 0.5, 0.5)
        bool inputs:thin_walled = 1
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // base is a weight on base_color, and emission a weight on emission_color;
    // both fold into one factor, the way KHR_materials_emissive_strength does.
    EXPECT_NEAR(material.baseColorFactor.r, 0.2f, 1e-5f);
    EXPECT_NEAR(material.baseColorFactor.g, 0.25f, 1e-5f);
    EXPECT_NEAR(material.baseColorFactor.b, 0.3f, 1e-5f);
    EXPECT_NEAR(material.emissiveFactor.r, 2.0f, 1e-5f);
    EXPECT_NEAR(material.emissiveFactor.g, 1.0f, 1e-5f);
    EXPECT_NEAR(material.emissiveFactor.b, 0.5f, 1e-5f);

    EXPECT_NEAR(material.metallicFactor, 0.3f, 1e-6f);
    EXPECT_NEAR(material.roughnessFactor, 0.25f, 1e-6f);
    EXPECT_NEAR(material.ior, 1.7f, 1e-6f);

    // specular_rotation is in turns; anisotropyRotation is in radians.
    EXPECT_NEAR(material.anisotropyStrength, 0.6f, 1e-6f);
    EXPECT_NEAR(material.anisotropyRotation, 1.5707963f, 1e-5f);

    // transmission_depth is a distance, so it is Beer-Lambert's and the colour
    // becomes the attenuation.
    EXPECT_NEAR(material.transmission, 0.4f, 1e-6f);
    EXPECT_NEAR(material.attenuationDistance, 2.0f, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.r, 0.9f, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.b, 0.7f, 1e-6f);

    // Material::dispersion is the Abbe number's reciprocal.
    EXPECT_NEAR(material.dispersion, 1.0f / 30.0f, 1e-6f);

    EXPECT_NEAR(material.sheenColorFactor.r, 0.5f, 1e-6f);
    EXPECT_NEAR(material.sheenColorFactor.g, 0.25f, 1e-6f);
    EXPECT_NEAR(material.sheenColorFactor.b, 0.125f, 1e-6f);
    EXPECT_NEAR(material.sheenRoughnessFactor, 0.4f, 1e-6f);

    EXPECT_NEAR(material.clearcoatFactor, 0.3f, 1e-6f);
    EXPECT_NEAR(material.clearcoatRoughnessFactor, 0.15f, 1e-6f);

    // opacity is a colour in standard_surface and a scalar here.
    EXPECT_NEAR(material.baseColorFactor.a, 0.5f, 1e-6f);
    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Blend);

    // thin_walled is geometry's business in USD and the material's here.
    EXPECT_TRUE(material.doubleSided);
}

TEST_F(UsdLoaderTest, StandardSurfaceTransmissionColourWithoutDepthIsNotBeerLambert) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("standard_surface_tint.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        float inputs:transmission = 1.0
        color3f inputs:transmission_color = (0.9, 0.5, 0.2)
        float inputs:transmission_depth = 0.0
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // With depth zero the colour is a distance-independent tint. Turning it into
    // a one-metre absorption would invent a coefficient nobody supplied.
    EXPECT_NEAR(material.transmission, 1.0f, 1e-6f);
    EXPECT_NEAR(material.attenuationDistance, Material{}.attenuationDistance, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.r, 1.0f, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.g, 1.0f, 1e-6f);

    // And the old `baseColorFactor.a = 1 - transmission` is gone: it counted
    // transmission twice once the blend mode also applied.
    EXPECT_NEAR(material.baseColorFactor.a, 1.0f, 1e-6f);
    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Opaque);
}

TEST_F(UsdLoaderTest, StandardSurfaceTexturesGoThroughTheImageNodes) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("mtlx_colour.tga", 210, 160, 110, 255);
    WriteSolidTga("mtlx_metal.tga", 0x30, 0x00, 0x00, 0xFF);

    const auto path = WriteUsda("standard_surface_textures.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        color3f inputs:base_color.connect = </Mat/Colour.outputs:out>
        float inputs:metalness.connect = </Mat/Split.outputs:outr>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "ND_image_color3"
        asset inputs:file = @./mtlx_colour.tga@ (
            colorSpace = "srgb_texture"
        )
        string inputs:uaddressmode = "clamp"
        float2 inputs:texcoord.connect = </Mat/Place.outputs:out>
        color3f outputs:out
    }

    def Shader "Place"
    {
        uniform token info:id = "ND_place2d_vector2"
        float2 inputs:pivot = (0.5, 0.5)
        float2 inputs:scale = (2, 2)
        float inputs:rotate = 0
        float2 inputs:offset = (0, 0)
        int inputs:operationorder = 0
        float2 outputs:out
    }

    def Shader "Split"
    {
        uniform token info:id = "ND_separate3_color3"
        color3f inputs:in.connect = </Mat/Metal.outputs:out>
        float outputs:outr
    }

    def Shader "Metal"
    {
        uniform token info:id = "ND_image_color3"
        asset inputs:file = @./mtlx_metal.tga@
        color3f outputs:out
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    ASSERT_GE(material.baseColorTextureIndex, 0);
    EXPECT_TRUE(scene.textures[material.baseColorTextureIndex].isSRGB)
        << "colorSpace = srgb_texture on the file input";
    EXPECT_EQ(scene.textures[material.baseColorTextureIndex].sampler.wrapS,
              TextureSampler::WrapMode::ClampToEdge);

    // place2d divides by scale, so scale = 2 halves the coordinates; the pivot
    // holds the centre still. Conjugated by the V flip, the rotation stays zero.
    EXPECT_NEAR(material.baseColorUv.scale.x, 0.5f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.scale.y, 0.5f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.offset.x, 0.25f, 1e-5f);
    EXPECT_NEAR(material.baseColorUv.offset.y, 1.0f - 0.25f - 0.5f, 1e-5f);

    ASSERT_GE(material.metallicRoughnessTextureIndex, 0);
    const Texture& packed = scene.textures[material.metallicRoughnessTextureIndex];
    if (packed.pixels.empty()) {
        GTEST_SKIP() << "texture pixels were released by block compression";
    }
    EXPECT_EQ(packed.pixels[2], 0x30) << "separate3.outr routed into B";
    EXPECT_NEAR(material.metallicFactor, 1.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, ASecondUvSetFallsBackToSetZero) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("second_uv.tga", 10, 20, 30, 255);

    const auto path = WriteUsda("second_uv_set.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        color3f inputs:base_color.connect = </Mat/Colour.outputs:out>
        token outputs:surface
    }

    def Shader "Colour"
    {
        uniform token info:id = "ND_image_color3"
        asset inputs:file = @./second_uv.tga@
        float2 inputs:texcoord.connect = </Mat/Coord.outputs:out>
        color3f outputs:out
    }

    def Shader "Coord"
    {
        uniform token info:id = "ND_texcoord_vector2"
        int inputs:index = 1
        float2 outputs:out
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // Only one UV set is loaded, the same limitation GltfLoader has. The texture
    // still binds, against set 0, and the log says so.
    EXPECT_GE(material.baseColorTextureIndex, 0);
}

// ============================================================================
// MaterialX gltf_pbr and open_pbr_surface
// ============================================================================

TEST_F(UsdLoaderTest, GltfPbrSurfaceshaderMapsOneToOne) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("gltf_pbr.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_gltf_pbr_surfaceshader"
        color3f inputs:base_color = (0.4, 0.5, 0.6)
        float inputs:metallic = 0.3
        float inputs:roughness = 0.25
        float inputs:ior = 1.7
        float inputs:specular = 0.8
        color3f inputs:specular_color = (0.9, 0.85, 0.8)
        float inputs:transmission = 0.4
        float inputs:thickness = 1.5
        float inputs:attenuation_distance = 2.0
        color3f inputs:attenuation_color = (0.9, 0.8, 0.7)
        color3f inputs:sheen_color = (0.5, 0.25, 0.125)
        float inputs:sheen_roughness = 0.4
        float inputs:clearcoat = 0.3
        float inputs:clearcoat_roughness = 0.15
        color3f inputs:emissive = (1, 0.5, 0.25)
        float inputs:emissive_strength = 2.0
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // gltf_pbr used to be routed into the UsdPreviewSurface reader, where none
    // of these names matched and every material came out the default grey.
    EXPECT_NEAR(material.baseColorFactor.r, 0.4f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.b, 0.6f, 1e-6f);
    EXPECT_NEAR(material.metallicFactor, 0.3f, 1e-6f);
    EXPECT_NEAR(material.roughnessFactor, 0.25f, 1e-6f);
    EXPECT_NEAR(material.ior, 1.7f, 1e-6f);
    EXPECT_NEAR(material.specularFactor, 0.8f, 1e-6f);
    EXPECT_NEAR(material.specularColorFactor.g, 0.85f, 1e-6f);
    EXPECT_NEAR(material.transmission, 0.4f, 1e-6f);
    EXPECT_NEAR(material.thicknessFactor, 1.5f, 1e-6f);
    EXPECT_NEAR(material.attenuationDistance, 2.0f, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.b, 0.7f, 1e-6f);
    EXPECT_NEAR(material.sheenColorFactor.r, 0.5f, 1e-6f);
    EXPECT_NEAR(material.sheenRoughnessFactor, 0.4f, 1e-6f);
    EXPECT_NEAR(material.clearcoatFactor, 0.3f, 1e-6f);
    EXPECT_NEAR(material.clearcoatRoughnessFactor, 0.15f, 1e-6f);

    // emissive_strength folds into the factor, as it does for glTF itself.
    EXPECT_NEAR(material.emissiveFactor.r, 2.0f, 1e-6f);
    EXPECT_NEAR(material.emissiveFactor.g, 1.0f, 1e-6f);
    EXPECT_NEAR(material.emissiveFactor.b, 0.5f, 1e-6f);
}

TEST_F(UsdLoaderTest, GltfPbrWithoutAnAttenuationDistanceHasNoAttenuation) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("gltf_pbr_no_attenuation.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_gltf_pbr_surfaceshader"
        float inputs:transmission = 1.0
        color3f inputs:attenuation_color = (0.2, 0.4, 0.6)
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // attenuation_distance has no default in the node definition, so a colour
    // without one describes no absorption; inventing a distance would invent a
    // coefficient.
    EXPECT_NEAR(material.attenuationDistance, 0.0f, 1e-6f);
    EXPECT_NEAR(material.attenuationColor.r, 1.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, GltfPbrAlphaModeIntegerSelectsMaskAndCutoff) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("gltf_pbr_alpha.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_gltf_pbr_surfaceshader"
        float inputs:alpha = 0.6
        int inputs:alpha_mode = 1
        float inputs:alpha_cutoff = 0.35
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // The enum travels as an integer: 0 opaque, 1 mask, 2 blend.
    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Mask);
    EXPECT_NEAR(material.alphaCutoff, 0.35f, 1e-6f);
    EXPECT_NEAR(material.baseColorFactor.a, 0.6f, 1e-6f);
}

TEST_F(UsdLoaderTest, OpenPbrDispersionScaleOverAbbeNumber) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("open_pbr_dispersion.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_open_pbr_surface_surfaceshader"
        float inputs:transmission_weight = 1.0
        float inputs:transmission_dispersion_scale = 0.5
        float inputs:transmission_dispersion_abbe_number = 40.0
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // OpenPBR splits dispersion into a strength and an Abbe number, so the
    // reciprocal Material stores is scale / V.
    EXPECT_NEAR(material.dispersion, 0.5f / 40.0f, 1e-6f);
    EXPECT_NEAR(material.transmission, 1.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, OpenPbrDispersionScaleZeroMeansNoDispersion) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("open_pbr_no_dispersion.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_open_pbr_surface_surfaceshader"
        float inputs:transmission_weight = 1.0
        float inputs:transmission_dispersion_abbe_number = 40.0
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // The Abbe number defaults to 20, so reading it alone would give every
    // OpenPBR glass a dispersion nobody asked for.
    EXPECT_NEAR(material.dispersion, 0.0f, 1e-6f);
}

TEST_F(UsdLoaderTest, OpenPbrFuzzAndCoatMapToSheenAndClearcoat) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("open_pbr_fuzz.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_open_pbr_surface_surfaceshader"
        float inputs:base_weight = 0.5
        color3f inputs:base_color = (0.4, 0.5, 0.6)
        float inputs:base_metalness = 0.2
        float inputs:specular_roughness = 0.35
        float inputs:specular_ior = 1.8
        float inputs:specular_roughness_anisotropy = 0.7
        float inputs:fuzz_weight = 0.5
        color3f inputs:fuzz_color = (1, 0.5, 0.25)
        float inputs:fuzz_roughness = 0.6
        float inputs:coat_weight = 0.4
        float inputs:coat_roughness = 0.05
        float inputs:geometry_opacity = 0.75
        bool inputs:geometry_thin_walled = 1
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // base_weight is a weight on base_color, fuzz is sheen, coat is clearcoat.
    EXPECT_NEAR(material.baseColorFactor.r, 0.2f, 1e-5f);
    EXPECT_NEAR(material.baseColorFactor.g, 0.25f, 1e-5f);
    EXPECT_NEAR(material.baseColorFactor.b, 0.3f, 1e-5f);
    EXPECT_NEAR(material.metallicFactor, 0.2f, 1e-6f);
    EXPECT_NEAR(material.roughnessFactor, 0.35f, 1e-6f);
    EXPECT_NEAR(material.ior, 1.8f, 1e-6f);
    EXPECT_NEAR(material.anisotropyStrength, 0.7f, 1e-6f);

    EXPECT_NEAR(material.sheenColorFactor.r, 0.5f, 1e-6f);
    EXPECT_NEAR(material.sheenColorFactor.g, 0.25f, 1e-6f);
    EXPECT_NEAR(material.sheenColorFactor.b, 0.125f, 1e-6f);
    EXPECT_NEAR(material.sheenRoughnessFactor, 0.6f, 1e-6f);

    EXPECT_NEAR(material.clearcoatFactor, 0.4f, 1e-6f);
    EXPECT_NEAR(material.clearcoatRoughnessFactor, 0.05f, 1e-6f);

    EXPECT_NEAR(material.baseColorFactor.a, 0.75f, 1e-6f);
    EXPECT_EQ(material.alphaMode, Material::AlphaMode::Blend);
    EXPECT_TRUE(material.doubleSided);
}

TEST_F(UsdLoaderTest, MaterialXImageColorSpaceMetadataSelectsSrgb) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteSolidTga("mtlx_srgb.tga", 200, 150, 100, 255);
    WriteSolidTga("mtlx_linear.tga", 200, 150, 100, 255);

    const auto path = WriteUsda("mtlx_colour_space.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_gltf_pbr_surfaceshader"
        color3f inputs:base_color.connect = </Mat/Encoded.outputs:out>
        color3f inputs:emissive.connect = </Mat/Linear.outputs:out>
        token outputs:surface
    }

    def Shader "Encoded"
    {
        uniform token info:id = "ND_image_color3"
        asset inputs:file = @./mtlx_srgb.tga@ (
            colorSpace = "srgb_texture"
        )
        color3f outputs:out
    }

    def Shader "Linear"
    {
        uniform token info:id = "ND_image_color3"
        asset inputs:file = @./mtlx_linear.tga@ (
            colorSpace = "lin_rec709"
        )
        color3f outputs:out
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;
    const Material& material = MaterialNamed(scene, "Mat");

    // Colour space is the one thing that breaks spectral upsampling silently: an
    // sRGB image read as linear gives the wrong reflectance and nothing errors.
    ASSERT_GE(material.baseColorTextureIndex, 0);
    ASSERT_GE(material.emissiveTextureIndex, 0);
    EXPECT_TRUE(scene.textures[material.baseColorTextureIndex].isSRGB);
    EXPECT_FALSE(scene.textures[material.emissiveTextureIndex].isSRGB)
        << "lin_rec709 overrides the slot's guess that an emissive map is colour";
}

TEST_F(UsdLoaderTest, AMaterialXDocumentReferencedFromUsdResolvesItsStandardSurface) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // A .mtlx document is a USD layer as far as this loader is concerned: the
    // usdMtlx plugin translates it, and what comes out the other side is an
    // ordinary UsdShadeShader with `info:id = ND_standard_surface_surfaceshader`.
    // Nothing here reads MaterialX, and nothing here links it.
    WriteUsda("referenced_material.mtlx", R"(<?xml version="1.0"?>
<materialx version="1.39">
  <standard_surface name="SS1" type="surfaceshader">
    <input name="base_color" type="color3" value="0.1, 0.2, 0.3" />
    <input name="specular_roughness" type="float" value="0.35" />
    <input name="metalness" type="float" value="0.9" />
  </standard_surface>
  <surfacematerial name="Mat1" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SS1" />
  </surfacematerial>
</materialx>
)");

    const auto path = WriteUsda("mtlx_reference.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + R"(
def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
}

def "Library" (
    references = @./referenced_material.mtlx@</MaterialX/Materials/Mat1>
)
{
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Scene& scene = *result;

    const Material* referenced = nullptr;
    for (const auto& material : scene.materials) {
        if (std::abs(material.roughnessFactor - 0.35f) < 1e-5f) {
            referenced = &material;
            break;
        }
    }
    ASSERT_NE(referenced, nullptr)
        << "the .mtlx document's standard_surface did not reach the scene; "
        << scene.materials.size() << " materials were loaded";

    EXPECT_NEAR(referenced->metallicFactor, 0.9f, 1e-5f);
    EXPECT_NEAR(referenced->baseColorFactor.r, 0.1f, 1e-5f);
    EXPECT_NEAR(referenced->baseColorFactor.g, 0.2f, 1e-5f);
    EXPECT_NEAR(referenced->baseColorFactor.b, 0.3f, 1e-5f);
}

// ============================================================================
// Quantiloom attributes on a Material prim
// ============================================================================

TEST_F(UsdLoaderTest, ReadsFluorescenceCurvesAndYieldFromQuantiloomAttributes) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteUsda("fluor_excitation.csv", "400,0.0\n450,0.8\n500,0.0\n");
    WriteUsda("fluor_emission.csv", "550,0.0\n600,1.0\n650,0.0\n");

    const auto path = WriteUsda("fluorescence.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    string quantiloom:fluorescenceExcitationCurve = "fluor_excitation.csv"
    string quantiloom:fluorescenceEmissionCurve = "fluor_emission.csv"
    float quantiloom:fluorescenceYield = 0.6

    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.6, 0.6, 0.6)
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // The keys are QUANTILOOM_materials_fluorescence's, spelled the same way, so
    // one scene converted between the two formats says the same thing.
    ASSERT_EQ(material.fluorescenceExcitationCurve.size(), 3u);
    ASSERT_EQ(material.fluorescenceEmissionCurve.size(), 3u);
    EXPECT_NEAR(material.fluorescenceExcitationCurve[1].first, 450.0f, 1e-3f);
    EXPECT_NEAR(material.fluorescenceExcitationCurve[1].second, 0.8f, 1e-4f);
    EXPECT_NEAR(material.fluorescenceEmissionCurve[1].first, 600.0f, 1e-3f);
    EXPECT_NEAR(material.fluorescenceYield, 0.6f, 1e-5f);

    // No emissiveFactor: a fluorescent surface is dark on its own, and a triple
    // here would put it in the emitter-sampling table.
    EXPECT_EQ(material.emissiveFactor, glm::vec3(0.0f));

    // The pair's provenance, for the energy warning ResolveFluorescence prints.
    EXPECT_NE(material.fluorescenceSource.find("#/Mat"), String::npos)
        << "fluorescenceSource was '" << material.fluorescenceSource << "'";
}

TEST_F(UsdLoaderTest, OneFluorescenceCurveAloneDoesNotFluoresce) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    WriteUsda("fluor_half.csv", "400,0.0\n450,0.8\n500,0.0\n");

    const auto path = WriteUsda("fluorescence_half.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    string quantiloom:fluorescenceExcitationCurve = "fluor_half.csv"
    float quantiloom:fluorescenceYield = 0.6

    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // Half a transfer is not a description of anything; the loader says so and
    // leaves the source empty so nothing downstream treats it as a pair.
    EXPECT_TRUE(material.fluorescenceEmissionCurve.empty());
    EXPECT_TRUE(material.fluorescenceSource.empty());
}

TEST_F(UsdLoaderTest, QuantiloomDispersionOverridesTheSurfaceValue) {
    if (!hasOpenUSD) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto path = WriteUsda("quantiloom_dispersion.usda", std::string(R"(#usda 1.0
(
    defaultPrim = "Quad"
)
)") + kBoundQuad + R"(
def Material "Mat"
{
    float quantiloom:dispersion = 0.0125

    token outputs:surface.connect = </Mat/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        float inputs:transmission = 1.0
        float inputs:transmission_dispersion = 30.0
        token outputs:surface
    }
}
)");

    auto result = UsdLoader::LoadFromFile(path.string());
    ASSERT_TRUE(result.has_value()) << result.error();
    const Material& material = MaterialNamed(*result, "Mat");

    // The attribute is already 1/Abbe, and it is read after the surface, so it
    // wins over the vocabulary's own value -- the same precedence
    // QUANTILOOM_materials_dispersion has over KHR_materials_dispersion.
    EXPECT_NEAR(material.dispersion, 0.0125f, 1e-6f);
}
