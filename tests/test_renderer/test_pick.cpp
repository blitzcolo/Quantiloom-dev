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
