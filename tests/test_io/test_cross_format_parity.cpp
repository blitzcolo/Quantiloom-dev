// ============================================================================
// Quantiloom - Cross-format material parity
// ============================================================================
// The strongest guarantee available for the USD reader: the same material,
// written once as glTF with its KHR extensions and once as USD, must arrive as
// the same Material. Everything downstream -- spectral upsampling, the BRDF,
// the IR energy balance -- is one implementation for both formats, so a USD
// material that matches its glTF twin field for field is right for the same
// reasons the glTF one is.
//
// Where a vocabulary parameterises a quantity differently (a weight times a
// colour, an Abbe number instead of its reciprocal, turns instead of radians)
// the conversion is stated in the fixture and the expectation is derived from
// it, so a change to the conversion has to change this file too.
// ============================================================================

#include <gtest/gtest.h>

#include "io/GltfLoader.hpp"
#include "io/UsdLoader.hpp"
#include "scene/Material.hpp"
#include "scene/Scene.hpp"

#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

std::filesystem::path FixtureDir() {
    const auto dir = std::filesystem::temp_directory_path() / "quantiloom_parity_fixtures";
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path WriteFixture(const std::string& fileName, const std::string& body) {
    const auto path = FixtureDir() / fileName;
    std::ofstream file(path);
    file << body;
    file.close();
    return path;
}

/// A glTF document with one material on one triangle. The buffer is the same
/// three vertices every fixture here uses; only the material differs.
std::string GltfWithMaterial(const std::string& extensionsUsed,
                             const std::string& material) {
    return R"JSON({
  "asset": { "version": "2.0" },
  "extensionsUsed": [ )JSON" + extensionsUsed + R"JSON( ],
  "materials": [ )JSON" + material + R"JSON( ],
  "meshes": [ { "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ] } ],
  "nodes": [ { "mesh": 0 } ],
  "scenes": [ { "nodes": [ 0 ] } ],
  "scene": 0,
  "accessors": [
    {
      "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
      "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
    }
  ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36 } ],
  "buffers": [
    {
      "byteLength": 36,
      "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"
    }
  ]
})JSON";
}

const Material& OnlyMaterial(const Scene& scene) {
    EXPECT_FALSE(scene.materials.empty());
    return scene.materials.at(0);
}

void ExpectSameColour(const glm::vec3& usd, const glm::vec3& gltf, const char* what) {
    EXPECT_NEAR(usd.r, gltf.r, 1e-6f) << what;
    EXPECT_NEAR(usd.g, gltf.g, 1e-6f) << what;
    EXPECT_NEAR(usd.b, gltf.b, 1e-6f) << what;
}

}  // namespace

// ============================================================================
// UsdPreviewSurface against the glTF core
// ============================================================================

TEST(CrossFormatParity, UsdPreviewSurfaceMatchesTheGltfCore) {
    if (!UsdLoader::IsAvailable()) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto gltfPath = WriteFixture("preview_core.gltf", GltfWithMaterial(
        R"("KHR_materials_ior", "KHR_materials_clearcoat")",
        R"({
      "name": "M",
      "pbrMetallicRoughness": {
        "baseColorFactor": [ 0.4, 0.5, 0.6, 1.0 ],
        "metallicFactor": 0.3,
        "roughnessFactor": 0.25
      },
      "emissiveFactor": [ 0.1, 0.2, 0.3 ],
      "alphaMode": "MASK",
      "alphaCutoff": 0.4,
      "extensions": {
        "KHR_materials_ior": { "ior": 1.7 },
        "KHR_materials_clearcoat": {
          "clearcoatFactor": 0.3,
          "clearcoatRoughnessFactor": 0.15
        }
      }
    })"));

    const auto usdPath = WriteFixture("preview_core.usda", R"(#usda 1.0
(
    defaultPrim = "Tri"
)

def Mesh "Tri"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    rel material:binding = </M>
}

def Material "M"
{
    token outputs:surface.connect = </M/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.4, 0.5, 0.6)
        float inputs:metallic = 0.3
        float inputs:roughness = 0.25
        color3f inputs:emissiveColor = (0.1, 0.2, 0.3)
        float inputs:opacityThreshold = 0.4
        float inputs:ior = 1.7
        float inputs:clearcoat = 0.3
        float inputs:clearcoatRoughness = 0.15
        token outputs:surface
    }
}
)");

    auto gltfScene = GltfLoader::LoadFromFile(gltfPath.string());
    ASSERT_TRUE(gltfScene.has_value()) << gltfScene.error();
    auto usdScene = UsdLoader::LoadFromFile(usdPath.string());
    ASSERT_TRUE(usdScene.has_value()) << usdScene.error();

    const Material& gltf = OnlyMaterial(*gltfScene);
    const Material& usd = OnlyMaterial(*usdScene);

    ExpectSameColour(glm::vec3(usd.baseColorFactor), glm::vec3(gltf.baseColorFactor),
                     "baseColorFactor");
    EXPECT_NEAR(usd.baseColorFactor.a, gltf.baseColorFactor.a, 1e-6f);
    EXPECT_NEAR(usd.metallicFactor, gltf.metallicFactor, 1e-6f);
    EXPECT_NEAR(usd.roughnessFactor, gltf.roughnessFactor, 1e-6f);
    ExpectSameColour(usd.emissiveFactor, gltf.emissiveFactor, "emissiveFactor");
    EXPECT_NEAR(usd.ior, gltf.ior, 1e-6f);
    EXPECT_NEAR(usd.clearcoatFactor, gltf.clearcoatFactor, 1e-6f);
    EXPECT_NEAR(usd.clearcoatRoughnessFactor, gltf.clearcoatRoughnessFactor, 1e-6f);

    // opacityThreshold is glTF's alphaCutoff by another name.
    EXPECT_EQ(usd.alphaMode, gltf.alphaMode);
    EXPECT_NEAR(usd.alphaCutoff, gltf.alphaCutoff, 1e-6f);
}

// ============================================================================
// MaterialX standard_surface against the glTF extensions
// ============================================================================

TEST(CrossFormatParity, StandardSurfaceMatchesTheGltfLoaderWithinTheDocumentedConversions) {
    if (!UsdLoader::IsAvailable()) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // The conversions this fixture exercises, one per differing parameterisation:
    //   sheen        : sheen (0.5) x sheen_color (1, 0.5, 0.25) = sheenColorFactor
    //   emission     : emission (2) x emission_color (1, 0.5, 0.25) = emissiveFactor,
    //                  the same fold KHR_materials_emissive_strength gets
    //   anisotropy   : specular_rotation is in turns, anisotropyRotation in radians
    //   dispersion   : transmission_dispersion is an Abbe number V and
    //                  Material::dispersion is 1/V; glTF's value is 20/V, so
    //                  V = 20 pairs with a glTF dispersion of 1
    //   volume       : transmission_color with transmission_depth is
    //                  attenuationColor over attenuationDistance
    const auto gltfPath = WriteFixture("standard_surface.gltf", GltfWithMaterial(
        R"("KHR_materials_ior", "KHR_materials_specular", "KHR_materials_anisotropy",
           "KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_sheen",
           "KHR_materials_clearcoat", "KHR_materials_dispersion",
           "KHR_materials_emissive_strength")",
        R"({
      "name": "M",
      "pbrMetallicRoughness": {
        "baseColorFactor": [ 0.4, 0.5, 0.6, 1.0 ],
        "metallicFactor": 0.3,
        "roughnessFactor": 0.25
      },
      "emissiveFactor": [ 1.0, 0.5, 0.25 ],
      "extensions": {
        "KHR_materials_ior": { "ior": 1.7 },
        "KHR_materials_specular": {
          "specularFactor": 0.8,
          "specularColorFactor": [ 0.9, 0.85, 0.8 ]
        },
        "KHR_materials_anisotropy": {
          "anisotropyStrength": 0.6,
          "anisotropyRotation": 1.5707963
        },
        "KHR_materials_transmission": { "transmissionFactor": 0.4 },
        "KHR_materials_volume": {
          "attenuationColor": [ 0.9, 0.8, 0.7 ],
          "attenuationDistance": 2.0
        },
        "KHR_materials_sheen": {
          "sheenColorFactor": [ 0.5, 0.25, 0.125 ],
          "sheenRoughnessFactor": 0.4
        },
        "KHR_materials_clearcoat": {
          "clearcoatFactor": 0.3,
          "clearcoatRoughnessFactor": 0.15
        },
        "KHR_materials_dispersion": { "dispersion": 1.0 },
        "KHR_materials_emissive_strength": { "emissiveStrength": 2.0 }
      }
    })"));

    const auto usdPath = WriteFixture("standard_surface.usda", R"(#usda 1.0
(
    defaultPrim = "Tri"
)

def Mesh "Tri"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    rel material:binding = </M>
}

def Material "M"
{
    token outputs:surface.connect = </M/Surface.outputs:surface>

    def Shader "Surface"
    {
        uniform token info:id = "ND_standard_surface_surfaceshader"
        float inputs:base = 1.0
        color3f inputs:base_color = (0.4, 0.5, 0.6)
        float inputs:metalness = 0.3
        float inputs:specular_roughness = 0.25
        float inputs:specular_IOR = 1.7
        float inputs:specular = 0.8
        color3f inputs:specular_color = (0.9, 0.85, 0.8)
        float inputs:specular_anisotropy = 0.6
        float inputs:specular_rotation = 0.25
        float inputs:transmission = 0.4
        color3f inputs:transmission_color = (0.9, 0.8, 0.7)
        float inputs:transmission_depth = 2.0
        float inputs:transmission_dispersion = 20.0
        float inputs:sheen = 0.5
        color3f inputs:sheen_color = (1.0, 0.5, 0.25)
        float inputs:sheen_roughness = 0.4
        float inputs:coat = 0.3
        float inputs:coat_roughness = 0.15
        float inputs:emission = 2.0
        color3f inputs:emission_color = (1.0, 0.5, 0.25)
        token outputs:surface
    }
}
)");

    auto gltfScene = GltfLoader::LoadFromFile(gltfPath.string());
    ASSERT_TRUE(gltfScene.has_value()) << gltfScene.error();
    auto usdScene = UsdLoader::LoadFromFile(usdPath.string());
    ASSERT_TRUE(usdScene.has_value()) << usdScene.error();

    const Material& gltf = OnlyMaterial(*gltfScene);
    const Material& usd = OnlyMaterial(*usdScene);

    ExpectSameColour(glm::vec3(usd.baseColorFactor), glm::vec3(gltf.baseColorFactor),
                     "baseColorFactor");
    EXPECT_NEAR(usd.metallicFactor, gltf.metallicFactor, 1e-6f);
    EXPECT_NEAR(usd.roughnessFactor, gltf.roughnessFactor, 1e-6f);
    EXPECT_NEAR(usd.ior, gltf.ior, 1e-6f);

    EXPECT_NEAR(usd.specularFactor, gltf.specularFactor, 1e-6f);
    ExpectSameColour(usd.specularColorFactor, gltf.specularColorFactor,
                     "specularColorFactor");

    EXPECT_NEAR(usd.anisotropyStrength, gltf.anisotropyStrength, 1e-6f);
    EXPECT_NEAR(usd.anisotropyRotation, gltf.anisotropyRotation, 1e-5f);

    EXPECT_NEAR(usd.transmission, gltf.transmission, 1e-6f);
    ExpectSameColour(usd.attenuationColor, gltf.attenuationColor, "attenuationColor");
    EXPECT_NEAR(usd.attenuationDistance, gltf.attenuationDistance, 1e-6f);

    EXPECT_NEAR(usd.dispersion, gltf.dispersion, 1e-6f);

    ExpectSameColour(usd.sheenColorFactor, gltf.sheenColorFactor, "sheenColorFactor");
    EXPECT_NEAR(usd.sheenRoughnessFactor, gltf.sheenRoughnessFactor, 1e-6f);

    EXPECT_NEAR(usd.clearcoatFactor, gltf.clearcoatFactor, 1e-6f);
    EXPECT_NEAR(usd.clearcoatRoughnessFactor, gltf.clearcoatRoughnessFactor, 1e-6f);

    ExpectSameColour(usd.emissiveFactor, gltf.emissiveFactor, "emissiveFactor");

    // Not compared: thicknessFactor, which standard_surface has no input for,
    // and iridescence and subsurface, which this renderer has no model for.
}

// ============================================================================
// MaterialX gltf_pbr against the glTF loader
// ============================================================================

TEST(CrossFormatParity, GltfPbrInUsdMatchesTheGltfLoaderFieldForField) {
    if (!UsdLoader::IsAvailable()) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    // ND_gltf_pbr_surfaceshader is the glTF material expressed as a MaterialX
    // node, so this is the closest thing to a byte-for-byte comparison the two
    // formats admit: the same numbers under the same names.
    //
    // Two extensions are absent from the comparison because they are absent
    // from the node definition. MaterialX 1.39's gltf_pbr has no anisotropy and
    // no dispersion input -- glTF has KHR_materials_anisotropy and
    // KHR_materials_dispersion, this node models neither -- so there is nothing
    // on the USD side to compare against and nothing the reader could invent.
    const auto gltfPath = WriteFixture("gltf_pbr.gltf", GltfWithMaterial(
        R"("KHR_materials_ior", "KHR_materials_specular", "KHR_materials_transmission",
           "KHR_materials_volume", "KHR_materials_sheen", "KHR_materials_clearcoat",
           "KHR_materials_emissive_strength")",
        R"({
      "name": "M",
      "pbrMetallicRoughness": {
        "baseColorFactor": [ 0.4, 0.5, 0.6, 0.6 ],
        "metallicFactor": 0.3,
        "roughnessFactor": 0.25
      },
      "emissiveFactor": [ 1.0, 0.5, 0.25 ],
      "alphaMode": "MASK",
      "alphaCutoff": 0.35,
      "extensions": {
        "KHR_materials_ior": { "ior": 1.7 },
        "KHR_materials_specular": {
          "specularFactor": 0.8,
          "specularColorFactor": [ 0.9, 0.85, 0.8 ]
        },
        "KHR_materials_transmission": { "transmissionFactor": 0.4 },
        "KHR_materials_volume": {
          "thicknessFactor": 1.5,
          "attenuationColor": [ 0.9, 0.8, 0.7 ],
          "attenuationDistance": 2.0
        },
        "KHR_materials_sheen": {
          "sheenColorFactor": [ 0.5, 0.25, 0.125 ],
          "sheenRoughnessFactor": 0.4
        },
        "KHR_materials_clearcoat": {
          "clearcoatFactor": 0.3,
          "clearcoatRoughnessFactor": 0.15
        },
        "KHR_materials_emissive_strength": { "emissiveStrength": 2.0 }
      }
    })"));

    const auto usdPath = WriteFixture("gltf_pbr.usda", R"(#usda 1.0
(
    defaultPrim = "Tri"
)

def Mesh "Tri"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    rel material:binding = </M>
}

def Material "M"
{
    token outputs:surface.connect = </M/Surface.outputs:surface>

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
        color3f inputs:emissive = (1.0, 0.5, 0.25)
        float inputs:emissive_strength = 2.0
        float inputs:alpha = 0.6
        int inputs:alpha_mode = 1
        float inputs:alpha_cutoff = 0.35
        token outputs:surface
    }
}
)");

    auto gltfScene = GltfLoader::LoadFromFile(gltfPath.string());
    ASSERT_TRUE(gltfScene.has_value()) << gltfScene.error();
    auto usdScene = UsdLoader::LoadFromFile(usdPath.string());
    ASSERT_TRUE(usdScene.has_value()) << usdScene.error();

    const Material& gltf = OnlyMaterial(*gltfScene);
    const Material& usd = OnlyMaterial(*usdScene);

    ExpectSameColour(glm::vec3(usd.baseColorFactor), glm::vec3(gltf.baseColorFactor),
                     "baseColorFactor");
    EXPECT_NEAR(usd.baseColorFactor.a, gltf.baseColorFactor.a, 1e-6f);
    EXPECT_NEAR(usd.metallicFactor, gltf.metallicFactor, 1e-6f);
    EXPECT_NEAR(usd.roughnessFactor, gltf.roughnessFactor, 1e-6f);
    EXPECT_NEAR(usd.ior, gltf.ior, 1e-6f);

    EXPECT_NEAR(usd.specularFactor, gltf.specularFactor, 1e-6f);
    ExpectSameColour(usd.specularColorFactor, gltf.specularColorFactor,
                     "specularColorFactor");

    EXPECT_NEAR(usd.transmission, gltf.transmission, 1e-6f);
    EXPECT_NEAR(usd.thicknessFactor, gltf.thicknessFactor, 1e-6f);
    ExpectSameColour(usd.attenuationColor, gltf.attenuationColor, "attenuationColor");
    EXPECT_NEAR(usd.attenuationDistance, gltf.attenuationDistance, 1e-6f);

    ExpectSameColour(usd.sheenColorFactor, gltf.sheenColorFactor, "sheenColorFactor");
    EXPECT_NEAR(usd.sheenRoughnessFactor, gltf.sheenRoughnessFactor, 1e-6f);

    EXPECT_NEAR(usd.clearcoatFactor, gltf.clearcoatFactor, 1e-6f);
    EXPECT_NEAR(usd.clearcoatRoughnessFactor, gltf.clearcoatRoughnessFactor, 1e-6f);

    ExpectSameColour(usd.emissiveFactor, gltf.emissiveFactor, "emissiveFactor");

    EXPECT_EQ(usd.alphaMode, gltf.alphaMode);
    EXPECT_NEAR(usd.alphaCutoff, gltf.alphaCutoff, 1e-6f);
}
