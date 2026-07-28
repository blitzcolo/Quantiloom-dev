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

TEST(RenderCoreLoadSceneFromConfig, ReportsAMissingSceneFile) {
    const auto cfg = ConfigFrom("[scene]\ngltf = \"no_such_file.gltf\"\n");
    auto scene = rendercore::LoadSceneFromConfig(cfg);

    ASSERT_FALSE(scene.has_value());
    EXPECT_NE(scene.error().find("glTF"), std::string::npos) << scene.error();
}
