// ============================================================================
// Quantiloom - Unit Tests for the primary-hit depth AOV and BlitDepthTo
// ============================================================================
// The depth AOV feeds the GUI's occluded grid overlay: hit distance along the
// normalized primary ray, -1 where it missed. These cover the whole exported
// path -- render a frame, BlitDepthTo into a caller-owned image, read it back
// -- with structural asserts (sentinel vs. plausible distance), not pixels.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/CommandHelper.hpp"
#include "renderer/ExternalRenderContext.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class DepthAovTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_depth_aov";
        std::filesystem::create_directories(testDir);

        auto* shared = quantiloom::testing::SharedVulkanDevice();
        ASSERT_NE(shared, nullptr);

        ExternalRenderContext::InitParams params{};
        params.instance = shared->GetInstance();
        params.physicalDevice = shared->GetPhysicalDevice();
        params.device = shared->GetDevice();
        params.graphicsQueue = shared->GetGraphicsQueue();
        params.graphicsQueueFamily = shared->GetGraphicsQueueFamily();
        params.targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;
        params.width = kSize;
        params.height = kSize;
        params.pipelineCacheDir = testDir.string();

        auto created = ExternalRenderContext::Create(params);
        ASSERT_TRUE(created.has_value()) << created.error();
        context = std::move(created.value());
    }

    void TearDown() override {
        context.reset();
        if (!testDir.empty() && std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
        VulkanDeviceTest::TearDown();
    }

    /// Same in-repo cornell box fixture as test_apply_config.cpp, with the
    /// camera injectable so a case can aim at the box or at empty sky.
    void ApplyScene(const std::string& cameraKeys) {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = (root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");

        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\n" << cameraKeys
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        ASSERT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
    }

    /// Render one frame and pull the whole depth AOV back to the CPU through
    /// the exported BlitDepthTo -- the exact sequence the GUI overlay records.
    std::vector<f32> RenderAndReadDepth() {
        auto& ctx = Device();

        GpuImage colorTarget(ctx.GetAllocator(), ctx.GetDevice(), kSize, kSize,
                             VK_FORMAT_B8G8R8A8_SRGB,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        GpuImage depthTarget(ctx.GetAllocator(), ctx.GetDevice(), kSize, kSize,
                             VK_FORMAT_R32_SFLOAT,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT);
        GpuBuffer readback(ctx.GetAllocator(), kSize * kSize * sizeof(f32),
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

        CommandHelper::ExecuteImmediate(ctx, [&](VkCommandBuffer cmd) {
            context->RenderFrame(cmd, colorTarget.GetImage(),
                                 VK_IMAGE_LAYOUT_UNDEFINED, kSize, kSize);
            context->BlitDepthTo(cmd, depthTarget.GetImage(),
                                 VK_IMAGE_LAYOUT_UNDEFINED, kSize, kSize);

            // BlitDepthTo leaves the target SHADER_READ_ONLY; copy it out.
            // Manual barrier: CommandHelper does not know this transition.
            VkImageMemoryBarrier toSrc{};
            toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image = depthTarget.GetImage();
            toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toSrc.subresourceRange.levelCount = 1;
            toSrc.subresourceRange.layerCount = 1;
            toSrc.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &toSrc);
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {kSize, kSize, 1};
            vkCmdCopyImageToBuffer(cmd, depthTarget.GetImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readback.GetHandle(), 1, &region);
        });

        std::vector<f32> depths(kSize * kSize);
        void* mapped = readback.Map();
        std::memcpy(depths.data(), mapped, depths.size() * sizeof(f32));
        readback.Unmap();
        return depths;
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};

/// The scene the fixture names has to exist for any of this to mean anything.
bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_F(DepthAovTest, HitPixelsCarryAPlausibleDistance) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // The classic cornell view. The box is in its original units -- roughly
    // 556 x 549 x 559 spanning from the origin -- so the center ray from
    // z = -800 crosses several hundred units before any wall.
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    const auto depths = RenderAndReadDepth();

    const f32 center = depths[(kSize / 2) * kSize + (kSize / 2)];
    EXPECT_GT(center, 100.0f);
    EXPECT_LT(center, 5000.0f);
}

TEST_F(DepthAovTest, MissPixelsCarryTheSentinel) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // Aim horizontally from far above the box (top is at y ~ 549): with a
    // 45-degree FOV no ray in the frustum can dip low enough to reach it, so
    // every primary ray misses. (Not straight up -- forward parallel to the
    // up vector degenerates the camera basis.)
    ApplyScene("position = [0.0, 2000.0, 0.0]\nlook_at = [100.0, 2000.0, 0.0]\n");
    const auto depths = RenderAndReadDepth();

    for (const f32 d : depths) {
        ASSERT_FLOAT_EQ(d, -1.0f);
    }
}
