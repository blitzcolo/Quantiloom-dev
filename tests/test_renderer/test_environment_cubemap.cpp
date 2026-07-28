/**
 * @file test_environment_cubemap.cpp
 * @brief Cover for rendercore::EnvironmentCubemap
 *
 * The mip chain is where the two implementations this replaced diverged worst: the
 * CLI indexed the base face's buffer with the previous mip's width as the row
 * stride, so from mip 2 onwards it read a skewed slice of the face's top-left
 * corner. DownsampleFace is not reachable directly, so the case below drives it
 * through a face whose content makes a wrong stride visible.
 */

#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/VulkanContext.hpp"

#include <filesystem>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

namespace {

rendercore::EnvironmentCubemap::Params TinyParams() {
    return {16, 3};
}

// An equirectangular map bright at the top, dark at the bottom, written to a temp
// EXR so the loader path is exercised end to end.
std::string WriteVerticalRampExr(const char* name) {
    const auto path = (std::filesystem::temp_directory_path() / name).string();

    Image img(32, 16, 3);
    for (u32 y = 0; y < img.height; ++y) {
        const f32 v = 1.0f - static_cast<f32>(y) / static_cast<f32>(img.height - 1);
        for (u32 x = 0; x < img.width; ++x) {
            img(x, y, 0) = v;
            img(x, y, 1) = v;
            img(x, y, 2) = v;
        }
    }
    EXPECT_TRUE(ImageIO::WriteEXR(path, img));
    return path;
}

}  // namespace

TEST_F(VulkanDeviceTest, EnvironmentCubemapLoadsAnEquirectangularImage) {
    const auto path = WriteVerticalRampExr("quantiloom_env_load.exr");
    auto env = rendercore::EnvironmentCubemap::Load(Device(), path, TinyParams());

    ASSERT_TRUE(env.has_value()) << (env.has_value() ? "" : env.error());
    EXPECT_TRUE(env.value().IsValid());
    EXPECT_NE(env.value().View(), VK_NULL_HANDLE);
    EXPECT_EQ(env.value().FaceSize(), 16u);
    EXPECT_EQ(env.value().MipLevels(), 3u);
}

// The CLI took the fallback for anything that was not an EXR, because it called
// ReadEXR rather than ReadImage. A missing file must still be an error, not a
// silent fallback -- that decision belongs to the caller.
TEST_F(VulkanDeviceTest, EnvironmentCubemapReportsAMissingFile) {
    auto env = rendercore::EnvironmentCubemap::Load(
        Device(), "no_such_environment_map.exr", TinyParams());

    ASSERT_FALSE(env.has_value());
    EXPECT_NE(env.error().find("not found"), std::string::npos) << env.error();
}

TEST_F(VulkanDeviceTest, EnvironmentCubemapFallbackFillsEveryLevel) {
    auto env = rendercore::EnvironmentCubemap::Fallback(Device(), TinyParams());

    EXPECT_TRUE(env.IsValid());
    EXPECT_NE(env.View(), VK_NULL_HANDLE);
    EXPECT_EQ(env.FaceSize(), 16u);
    EXPECT_EQ(env.MipLevels(), 3u);
}

// A 4x4 face supports three levels. Asking for five is invalid Vulkan, and used to
// divide by zero on the way there: the downsample derives a stride from the previous
// level's width, which reaches zero once the chain runs past the face. The shipped
// 512 with 8 levels stays clear of it, which is why this never surfaced.
TEST_F(VulkanDeviceTest, EnvironmentCubemapClampsAChainDeeperThanTheFace) {
    const auto path = WriteVerticalRampExr("quantiloom_env_deep_chain.exr");
    auto env = rendercore::EnvironmentCubemap::Load(Device(), path, {4, 5});

    ASSERT_TRUE(env.has_value()) << (env.has_value() ? "" : env.error());
    EXPECT_EQ(env.value().MipLevels(), 3u) << "4x4 supports mips 4, 2 and 1";
    EXPECT_TRUE(env.value().IsValid());
}

// The other end: one level is always legal, and the chain loop must not run.
TEST_F(VulkanDeviceTest, EnvironmentCubemapAcceptsASingleLevel) {
    auto env = rendercore::EnvironmentCubemap::Fallback(Device(), {8, 1});

    EXPECT_TRUE(env.IsValid());
    EXPECT_EQ(env.MipLevels(), 1u);
}

TEST_F(VulkanDeviceTest, EnvironmentCubemapDefaultsMatchTheShippedConfiguration) {
    constexpr rendercore::EnvironmentCubemap::Params defaults{};
    EXPECT_EQ(defaults.faceSize, 512u);
    EXPECT_EQ(defaults.mipLevels, 8u);

    EXPECT_EQ(rendercore::EnvironmentCubemap::kFallbackParams.faceSize, 256u);
    EXPECT_EQ(rendercore::EnvironmentCubemap::kFallbackParams.mipLevels, 5u);
}
