// ============================================================================
// Quantiloom - Unit Tests for ExternalRenderContext::Pick
// ============================================================================
// Pick drives click-to-select in the GUI: a 1x1 inline ray-query dispatch
// whose ray must agree with the raygen shader's ray for the same pixel.
// Structural asserts: hits map to a valid node, sky misses, and the reported
// distance is a plausible camera-to-wall length -- not pixels.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class PickTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_pick";
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

    /// Same cornell fixture as test_depth_aov.cpp: the box spans roughly
    /// 556 x 549 x 559 from the origin, in its original units.
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

    void RenderRawFrame() {
        GpuImage target(Device().GetAllocator(), Device().GetDevice(), kSize, kSize,
            VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            context->RenderFrame(cmd, target.GetImage(), VK_IMAGE_LAYOUT_UNDEFINED, kSize, kSize);
        });
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_F(PickTest, CenterPixelHitsAValidNode) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");

    const auto result = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(result.has_value()) << result.error();
    ASSERT_TRUE(result.value().hit);

    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    EXPECT_LT(result.value().nodeIndex, scene->nodes.size());

    // Camera is 800 units in front of the box; any wall is hundreds away
    EXPECT_GT(result.value().hitT, 100.0f);
    EXPECT_LT(result.value().hitT, 5000.0f);

    // worldPosition must be origin + direction * hitT, i.e. its distance from
    // the camera equals hitT
    const glm::vec3 camPos(278.0f, 274.0f, -800.0f);
    EXPECT_NEAR(glm::length(result.value().worldPosition - camPos),
                result.value().hitT, 0.5f);
}

TEST_F(PickTest, SkyPixelMisses) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // High above the box aimed horizontally: nothing in the frustum
    ApplyScene("position = [0.0, 2000.0, 0.0]\nlook_at = [100.0, 2000.0, 0.0]\n");

    const auto result = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_FALSE(result.value().hit);
}

TEST_F(PickTest, OutOfBoundsIsAnError) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    EXPECT_FALSE(context->Pick(kSize, 0).has_value());
    EXPECT_FALSE(context->Pick(0, kSize).has_value());
}

TEST_F(PickTest, AsynchronousPixelMatchesSynchronousReadingAndCarriesItsCoordinates) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    context->SetRenderScale(0.5f);
    context->SetDebugMode(DebugVisualizationMode::ShadedNormal);
    RenderRawFrame();
    const auto expected = context->ReadPixelValue(kSize - 1, kSize / 2);
    ASSERT_TRUE(expected.has_value());
    const auto requested = context->RequestPixelValue(kSize - 1, kSize / 2, 73);
    ASSERT_TRUE(requested.has_value()) << requested.error();
    ASSERT_TRUE(requested.value());
    // Waiting belongs to this deterministic test, never to Request/Poll.
    ASSERT_EQ(vkQueueWaitIdle(Device().GetGraphicsQueue()), VK_SUCCESS);
    const auto completed = context->PollPixelValue();
    ASSERT_TRUE(completed.has_value()) << completed.error();
    ASSERT_TRUE(completed.value().has_value());
    const auto& reading = *completed.value();
    EXPECT_EQ(reading.requestId, 73u);
    EXPECT_EQ(reading.x, kSize - 1);
    EXPECT_EQ(reading.y, kSize / 2);
    EXPECT_EQ(reading.accumulatedSamples, 1u);
    for (u32 c = 0; c < 4; ++c) EXPECT_EQ(reading.value[c], expected.value()[c]);
    EXPECT_FALSE(context->PollPixelValue().value().has_value());
}

TEST_F(PickTest, AsynchronousPixelRingCoalescesCompletedCopiesAndReusesSlots) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    RenderRawFrame();
    for (u32 i = 0; i < 3; ++i) {
        const auto requested = context->RequestPixelValue(i, i + 1, 100 + i);
        ASSERT_TRUE(requested.has_value());
        ASSERT_TRUE(requested.value());
    }
    const auto busy = context->RequestPixelValue(10, 11, 103);
    ASSERT_TRUE(busy.has_value());
    EXPECT_FALSE(busy.value());
    ASSERT_EQ(vkQueueWaitIdle(Device().GetGraphicsQueue()), VK_SUCCESS);
    const auto completed = context->PollPixelValue();
    ASSERT_TRUE(completed.has_value());
    ASSERT_TRUE(completed.value().has_value());
    EXPECT_EQ(completed.value()->requestId, 102u);
    EXPECT_EQ(completed.value()->x, 2u);
    EXPECT_EQ(completed.value()->y, 3u);
    const auto reused = context->RequestPixelValue(10, 11, 103);
    ASSERT_TRUE(reused.has_value());
    EXPECT_TRUE(reused.value());
}

TEST_F(PickTest, AsynchronousPixelDropsResultsWhenTheAccumulationChanges) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf unavailable";
    ApplyScene("position = [278.0, 274.0, -800.0]\nlook_at = [278.0, 274.0, 0.0]\n");
    EXPECT_FALSE(context->RequestPixelValue(0, 0, 1).has_value());
    RenderRawFrame();
    ASSERT_TRUE(context->RequestPixelValue(0, 0, 1).value());
    context->ResetAccumulation();
    ASSERT_EQ(vkQueueWaitIdle(Device().GetGraphicsQueue()), VK_SUCCESS);
    const auto stale = context->PollPixelValue();
    ASSERT_TRUE(stale.has_value());
    EXPECT_FALSE(stale.value().has_value());
    EXPECT_FALSE(context->RequestPixelValue(0, 0, 2).has_value());
    RenderRawFrame();
    ASSERT_TRUE(context->RequestPixelValue(0, 0, 2).value());
    ASSERT_EQ(vkQueueWaitIdle(Device().GetGraphicsQueue()), VK_SUCCESS);
    const auto fresh = context->PollPixelValue();
    ASSERT_TRUE(fresh.has_value());
    ASSERT_TRUE(fresh.value().has_value());
    EXPECT_EQ(fresh.value()->requestId, 2u);
    EXPECT_FALSE(context->RequestPixelValue(kSize, 0, 3).has_value());
}
