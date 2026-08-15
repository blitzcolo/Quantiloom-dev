// ============================================================================
// Quantiloom - Unit Tests for renderer/TemperatureTextureLoader.hpp
// ============================================================================
// The mount is the only step between a config naming a temperature map and the
// shader sampling one. What it must get right is narrow and easy to get wrong
// silently: the index it writes, the encoding the shader decodes, and what
// happens to a material whose path does not load.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/TemperatureTextureLoader.hpp"

#include "io/ImageIO.hpp"

#include <filesystem>

using namespace quantiloom;
using namespace quantiloom::rendercore;

namespace {

class TemperatureTextureTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "quantiloom_temperature_texture";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    /// A single-row map with the given normalised values. EXR rather than PNG:
    /// ReadImage divides PNG bytes by 255 without decoding sRGB, while WritePNG
    /// encodes it, so a PNG round trip inside a test would assert against the
    /// gamma curve rather than against the mount.
    void WriteMap(const std::string& name, const std::vector<f32>& values) {
        Image img(static_cast<u32>(values.size()), 1, 1);
        img.channelNames = {"Y"};
        for (u32 x = 0; x < img.width; ++x) {
            img(x, 0, 0) = values[x];
        }
        ASSERT_TRUE(ImageIO::WriteEXR((testDir / name).string(), img));
    }

    Scene SceneWithMaterial(const String& texturePath) {
        Scene scene;
        Material m = Material::CreateLambertian(glm::vec3(0.5f), "Panel");
        m.temperatureTexturePath = texturePath;
        scene.materials.push_back(m);
        return scene;
    }

    std::filesystem::path testDir;
    ConfigApplyReport report;
};

}  // namespace

TEST_F(TemperatureTextureTest, MountsTheMapAndPointsTheMaterialAtIt) {
    WriteMap("temp.exr", {0.0f, 0.5f, 1.0f});

    Scene scene = SceneWithMaterial("temp.exr");
    MountTemperatureTextures(scene, testDir.string(), report);

    ASSERT_EQ(scene.textures.size(), 1u);
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, 0);

    const Texture& tex = scene.textures[0];
    EXPECT_EQ(tex.width, 3u);
    EXPECT_EQ(tex.height, 1u);
    EXPECT_EQ(tex.channels, 4u);

    // Not colour, and not compressible: BC7 would smear a field across blocks.
    EXPECT_FALSE(tex.isSRGB);
    EXPECT_TRUE(tex.skipBlockCompression);

    // The R channel is what the shader decodes; 0.5 lands on 128 rather than
    // 127 because the encode rounds.
    ASSERT_EQ(tex.pixels.size(), 3u * 4u);
    EXPECT_EQ(tex.pixels[0], 0u);
    EXPECT_EQ(tex.pixels[4], 128u);
    EXPECT_EQ(tex.pixels[8], 255u);
}

TEST_F(TemperatureTextureTest, MaterialsWithNoPathAreLeftAlone) {
    Scene scene = SceneWithMaterial("");
    scene.materials[0].temperatureTextureIndex = -1;

    MountTemperatureTextures(scene, testDir.string(), report);

    EXPECT_TRUE(scene.textures.empty());
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, -1);
}

TEST_F(TemperatureTextureTest, APathThatDoesNotLoadLeavesTheSceneFilesMapInPlace) {
    // A config typo costs the override, never the render: whatever the loader
    // put on the material stays.
    Scene scene = SceneWithMaterial("no_such_map.exr");
    scene.materials[0].temperatureTextureIndex = 7;

    MountTemperatureTextures(scene, testDir.string(), report);

    EXPECT_TRUE(scene.textures.empty());
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, 7);
}

TEST_F(TemperatureTextureTest, TheConfigsMapOverridesOneTheSceneFileProvided) {
    WriteMap("temp.exr", {0.25f});

    Scene scene = SceneWithMaterial("temp.exr");
    scene.materials[0].temperatureTextureIndex = 7;  // as glTF extras would leave it

    MountTemperatureTextures(scene, testDir.string(), report);

    ASSERT_EQ(scene.textures.size(), 1u);
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, 0);
}

TEST_F(TemperatureTextureTest, ValuesOutsideTheNormalisedRangeAreClampedAndReported) {
    // A map authored in kelvin rather than [0, 1] saturates at the top of the
    // range, which looks like a working render of a flat surface. The warning
    // is the only thing that says otherwise.
    WriteMap("temp.exr", {-0.5f, 1.5f});

    Scene scene = SceneWithMaterial("temp.exr");
    MountTemperatureTextures(scene, testDir.string(), report);

    ASSERT_EQ(scene.textures.size(), 1u);
    EXPECT_EQ(scene.textures[0].pixels[0], 0u);
    EXPECT_EQ(scene.textures[0].pixels[4], 255u);
}

TEST_F(TemperatureTextureTest, EachMountedMaterialGetsItsOwnTexture) {
    WriteMap("a.exr", {0.1f});
    WriteMap("b.exr", {0.9f});

    Scene scene;
    for (const auto* name : {"A", "B"}) {
        Material m = Material::CreateLambertian(glm::vec3(0.5f), name);
        m.temperatureTexturePath = String(name) == "A" ? "a.exr" : "b.exr";
        scene.materials.push_back(m);
    }

    MountTemperatureTextures(scene, testDir.string(), report);

    ASSERT_EQ(scene.textures.size(), 2u);
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, 0);
    EXPECT_EQ(scene.materials[1].temperatureTextureIndex, 1);
    EXPECT_EQ(scene.textures[0].pixels[0], 26u);   // 0.1 * 255 rounded
    EXPECT_EQ(scene.textures[1].pixels[0], 230u);  // 0.9 * 255 rounded
}
