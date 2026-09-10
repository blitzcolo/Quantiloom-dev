/**
 * @file test_render_core.cpp
 * @brief Tests for renderer/RenderCore -- orchestration shared by the CLI and the GUI
 *
 * These exist because the two equirectangular-to-cubemap implementations this code
 * replaced were vertical mirrors of each other, and nothing caught it: the GUI
 * rendered every environment map upside down while the CLI got it right. The
 * orientation cases below fail on either mistake.
 *
 * RenderCore is internal -- no QL_API, not in include/quantiloom/. These tests reach
 * it because the suite links quantiloom_core rather than the DLL.
 */

#include <gtest/gtest.h>

#include "renderer/RenderCore.hpp"

#include "io/UsdLoader.hpp"
#include "support/LogCapture.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

// Equirectangular map whose value is the row index normalised to [0, 1]:
// row 0 (the zenith, per OpenEXR scanline order) is 0.0, the last row is 1.0.
Image MakeVerticalRamp(const u32 width, const u32 height) {
    Image img(width, height, 3);
    for (u32 y = 0; y < height; ++y) {
        const f32 t = static_cast<f32>(y) / static_cast<f32>(height - 1);
        for (u32 x = 0; x < width; ++x) {
            img(x, y, 0) = t;
            img(x, y, 1) = t;
            img(x, y, 2) = t;
        }
    }
    return img;
}

f32 FaceMean(const Image& face) {
    f64 sum = 0.0;
    for (u32 y = 0; y < face.height; ++y) {
        for (u32 x = 0; x < face.width; ++x) {
            sum += face(x, y, 0);
        }
    }
    return static_cast<f32>(sum / (face.width * face.height));
}

constexpr u32 kPlusY = 2;
constexpr u32 kMinusY = 3;

}  // namespace

// An .exr environment map comes back from ImageIO in OpenEXR's name-sorted
// channel order, so "R","G","B" arrives as "B","G","R". Sampling positions
// 0,1,2 therefore swapped red and blue on every EXR HDRI in the repository,
// and the tests above never saw it because MakeVerticalRamp builds its image
// in memory with the default "Channel_0".."Channel_2" names, which sort back
// into the order they were written and are grey besides.
TEST(RenderCoreEquirectToCubemap, HonoursChannelNamesRatherThanPositions) {
    Image src(16, 8, 3);
    src.channelNames = {"B", "G", "R"};  // what a read-back .exr looks like
    for (u32 y = 0; y < src.height; ++y) {
        for (u32 x = 0; x < src.width; ++x) {
            src(x, y, 0) = 0.0f;  // B
            src(x, y, 1) = 0.0f;  // G
            src(x, y, 2) = 1.0f;  // R
        }
    }

    const auto faces = rendercore::EquirectToCubemap(src, 8);
    ASSERT_EQ(faces.size(), 6u);
    for (const Image& face : faces) {
        EXPECT_NEAR(face(4, 4, 0), 1.0f, 1e-5f) << "red belongs in the red slot";
        EXPECT_NEAR(face(4, 4, 1), 0.0f, 1e-5f);
        EXPECT_NEAR(face(4, 4, 2), 0.0f, 1e-5f) << "blue must not inherit red";
    }
}

TEST(RenderCoreEquirectToCubemap, ProducesSixSquareRgbFaces) {
    const Image src = MakeVerticalRamp(64, 32);
    const auto faces = rendercore::EquirectToCubemap(src, 16);

    ASSERT_EQ(faces.size(), 6u);
    for (const auto& face : faces) {
        EXPECT_EQ(face.width, 16u);
        EXPECT_EQ(face.height, 16u);
        EXPECT_EQ(face.channels, 3u);
    }
}

// The regression that motivated RenderCore. Row 0 of an equirectangular map is the
// zenith, so +Y must sample near it and -Y near the last row. Swapping the mapping --
// as the GUI's copy did -- inverts both means and fails here.
TEST(RenderCoreEquirectToCubemap, TopOfSourceLandsOnPlusYFace) {
    const Image src = MakeVerticalRamp(128, 64);
    const auto faces = rendercore::EquirectToCubemap(src, 32);

    const f32 up = FaceMean(faces[kPlusY]);
    const f32 down = FaceMean(faces[kMinusY]);

    EXPECT_LT(up, 0.25f) << "+Y should sample the top of the source (value near 0)";
    EXPECT_GT(down, 0.75f) << "-Y should sample the bottom of the source (value near 1)";
    EXPECT_LT(up, down);
}

// Complements the case above: the four side faces straddle the equator, so each
// averages near the middle of the ramp. A mirrored mapping leaves these unchanged,
// which is exactly why the orientation case is needed as well.
TEST(RenderCoreEquirectToCubemap, SideFacesAverageNearTheEquator) {
    const Image src = MakeVerticalRamp(128, 64);
    const auto faces = rendercore::EquirectToCubemap(src, 32);

    for (const u32 face : {0u, 1u, 4u, 5u}) {
        EXPECT_NEAR(FaceMean(faces[face]), 0.5f, 0.05f)
            << "side face " << face << " should straddle the equator";
    }
}

// Azimuth: a bright meridian at u = 0.5 maps to +X, since u = (atan2(z, x) + pi)/2pi.
TEST(RenderCoreEquirectToCubemap, BrightMeridianLandsOnPlusXFace) {
    const u32 width = 128;
    Image src(width, 64, 3);
    const u32 bright = width / 2;
    for (u32 y = 0; y < src.height; ++y) {
        for (u32 c = 0; c < 3; ++c) {
            src(bright, y, c) = 1.0f;
            src(bright - 1, y, c) = 1.0f;
        }
    }

    const auto faces = rendercore::EquirectToCubemap(src, 32);

    EXPECT_GT(FaceMean(faces[0]), FaceMean(faces[1])) << "+X brighter than -X";
    EXPECT_GT(FaceMean(faces[0]), FaceMean(faces[4])) << "+X brighter than +Z";
    EXPECT_GT(FaceMean(faces[0]), FaceMean(faces[5])) << "+X brighter than -Z";
}

// ============================================================================
// LoadSceneFromConfig
// ============================================================================
// The two implementations this merged had diverged, and the cases below pin how each
// difference was resolved. Only the file-less paths are covered here: loading a glTF
// or USD is the loaders' own contract, tested in test_io.

namespace {

Config ConfigFrom(const std::string& toml) {
    const auto path = std::filesystem::temp_directory_path() /
                      "quantiloom_rendercore_scene_test.toml";
    std::ofstream(path) << toml;
    auto cfg = Config::Load(path);
    EXPECT_TRUE(cfg.has_value()) << (cfg.has_value() ? "" : cfg.error());
    return std::move(cfg.value());
}

}  // namespace

// A config that names no scene is an error, not a scene. Both implementations this
// merged had a way of carrying on -- the CLI built a procedural Cornell box and
// warned -- which turned a misspelt key into a wrong picture instead of a message.
TEST(RenderCoreLoadSceneFromConfig, RejectsAConfigThatNamesNoScene) {
    const auto cfg = ConfigFrom("[render]\nwidth = 64\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("scene.usd"), std::string::npos) << scene.error();
    EXPECT_NE(scene.error().find("scene.gltf"), std::string::npos) << scene.error();
}

// scene.preset used to select one of three meshes built in code. It is gone, so a
// config carrying one is a config naming no scene.
TEST(RenderCoreLoadSceneFromConfig, DoesNotResurrectScenePreset) {
    const auto cfg = ConfigFrom("[scene]\npreset = \"cornell_box\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    EXPECT_FALSE(scene.has_value());
}

// Precedence: the CLI took the USD when a config named both, the render context took
// the glTF. The CLI's order survived, so a config naming both reports a USD failure
// rather than silently rendering the glTF.
TEST(RenderCoreLoadSceneFromConfig, UsdTakesPrecedenceOverGltf) {
    const auto cfg = ConfigFrom(
        "[scene]\nusd = \"no_such_file.usdc\"\ngltf = \"no_such_file.gltf\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("USD"), std::string::npos) << scene.error();
}

// A variant spec that does not parse fails the load rather than rendering the
// default variant. An unknown variant *name* is a typo in the data and warns;
// the syntax is the contract between the config and the loader.
TEST(RenderCoreLoadSceneFromConfig, AMalformedUsdVariantSpecFailsTheLoad) {
    const auto cfg = ConfigFrom(
        "[scene]\nusd = \"no_such_file.usdc\"\nvariant = \"/Root/Car{color\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("variant"), std::string::npos) << scene.error();
    // And it fails on the spec, before it ever looks for the file.
    EXPECT_EQ(scene.error().find("Failed to load USD"), std::string::npos) << scene.error();
}

TEST(RenderCoreLoadSceneFromConfig, AMalformedModelVariantSpecNamesTheModel) {
    const auto cfg = ConfigFrom(
        "[[models]]\nfile = \"no_such_file.usdc\"\nname = \"rig\"\n"
        "variant = \"lod\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("rig"), std::string::npos) << scene.error();
}

// The positive half of the contract: what [scene] says about a USD file --
// the variant, the time code, the stage metrics, the payload policy -- arrives
// at the loader as the options it means. The loader's own tests pin what each
// option does; this one pins that the config reaches it, which nothing between
// the CLI and Studio can check any other way.
TEST(RenderCoreLoadSceneFromConfig, UsdVariantAndTimeCodeReachTheLoader) {
    if (!UsdLoader::IsAvailable()) {
        GTEST_SKIP() << "OpenUSD support not available";
    }

    const auto dir = std::filesystem::temp_directory_path() / "quantiloom_rendercore_usd";
    std::filesystem::create_directories(dir);

    // A Z-up, centimetre stage whose "low" variant is the only one animated.
    // The default variant is "high" and its triangle never reaches x = 5, so a
    // vertex at 5 proves both the variant and the time code were applied.
    const auto stagePath = dir / "options.usda";
    std::ofstream(stagePath) << R"(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 0.01
)

def Xform "World"
{
    def Mesh "Tri" (
        variants = { string lod = "high" }
        prepend variantSets = "lod"
    )
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        variantSet "lod" = {
            "high" {
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            }
            "low" {
                point3f[] points.timeSamples = {
                    0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                    10: [(0, 0, 0), (5, 0, 0), (0, 5, 0)],
                }
            }
        }
    }
}
)";

    const std::string usdKey = "[scene]\nusd = \"" + stagePath.generic_string() + "\"\n";

    {
        const auto cfg = ConfigFrom(usdKey + "variant = \"lod=low\"\nusd_time_code = 10.0\n"
                                             "usd_stage_metrics = false\n");
        auto scene = rendercore::LoadSceneFromConfig(cfg);
        ASSERT_TRUE(scene.has_value()) << scene.error();
        const Scene& loaded = *scene;
        ASSERT_EQ(loaded.meshes.size(), 1u);
        ASSERT_EQ(loaded.meshes[0].primitives.size(), 1u);

        f32 maxX = 0.0f;
        for (const auto& position : loaded.meshes[0].primitives[0].positions) {
            maxX = std::max(maxX, position.x);
        }
        EXPECT_NEAR(maxX, 5.0f, 1e-5f) << "the low variant, at time code 10";

        ASSERT_EQ(loaded.nodes.size(), 1u);
        const glm::vec3 up =
            glm::vec3(loaded.nodes[0].transform * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        EXPECT_NEAR(up.z, 1.0f, 1e-5f) << "usd_stage_metrics = false leaves the stage's axes";
    }

    {
        // Nothing said: the default variant, the default time, and the stage's
        // Z-up centimetres folded in, so the stage's Z lands on +Y at 0.01.
        const auto cfg = ConfigFrom(usdKey);
        auto scene = rendercore::LoadSceneFromConfig(cfg);
        ASSERT_TRUE(scene.has_value()) << scene.error();
        const Scene& loaded = *scene;
        ASSERT_EQ(loaded.meshes.size(), 1u);

        f32 maxX = 0.0f;
        for (const auto& position : loaded.meshes[0].primitives[0].positions) {
            maxX = std::max(maxX, position.x);
        }
        EXPECT_NEAR(maxX, 1.0f, 1e-5f) << "the high variant";

        ASSERT_EQ(loaded.nodes.size(), 1u);
        const glm::vec3 up =
            glm::vec3(loaded.nodes[0].transform * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
        EXPECT_NEAR(up.y, 0.01f, 1e-6f) << "Z-up and metersPerUnit folded in by default";
        EXPECT_NEAR(up.z, 0.0f, 1e-6f);
    }

    // Payloads: a root whose only geometry hangs off a payload.
    std::ofstream(dir / "body.usda") << R"(#usda 1.0
(
    defaultPrim = "Body"
)

def Mesh "Body"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
)";
    const auto rootPath = dir / "root.usda";
    std::ofstream(rootPath) << R"(#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def "Heavy" (
        payload = @./body.usda@</Body>
    )
    {
    }
}
)";
    const std::string rootKey = "[scene]\nusd = \"" + rootPath.generic_string() + "\"\n";
    {
        auto scene = rendercore::LoadSceneFromConfig(ConfigFrom(rootKey + "usd_payloads = \"none\"\n"));
        ASSERT_TRUE(scene.has_value()) << scene.error();
        EXPECT_TRUE(scene.value().meshes.empty()) << "usd_payloads = \"none\" reached the loader";
    }
    {
        auto scene = rendercore::LoadSceneFromConfig(ConfigFrom(rootKey));
        ASSERT_TRUE(scene.has_value()) << scene.error();
        EXPECT_EQ(scene.value().meshes.size(), 1u) << "payloads load by default";
    }
}

// A value that is neither "all" nor "none" is a typo, and [[models]].payloads
// already says so; [scene] says the same and loads everything.
TEST(RenderCoreLoadSceneFromConfig, AnUnknownUsdPayloadsValueWarnsAndLoadsAll) {
    support::ScopedLogCapture log;
    const auto cfg = ConfigFrom(
        "[scene]\nusd = \"no_such_file.usdc\"\nusd_payloads = \"some\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    EXPECT_FALSE(scene.has_value());
    EXPECT_TRUE(log.HasWarning("scene.usd_payloads is \"some\"")) << log.Dump();
}

TEST(RenderCoreLoadSceneFromConfig, ReportsAMissingSceneFile) {
    const auto cfg = ConfigFrom("[scene]\ngltf = \"no_such_file.gltf\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("glTF"), std::string::npos) << scene.error();
}
