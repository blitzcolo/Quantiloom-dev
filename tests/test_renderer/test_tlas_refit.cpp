// ============================================================================
// Quantiloom - Unit Tests for RefitAccelerationStructure
// ============================================================================
// The interactive TLAS path: SetNodeTransform + RefitAccelerationStructure
// per mouse-move during a gizmo drag, full rebuild only on release. Pick
// (itself tested in test_pick.cpp) is the probe: geometry that really moved
// stops answering under the old pixel and answers again when moved back.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class TlasRefitTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_tlas_refit";
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

        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = (root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\nposition = [278.0, 274.0, -800.0]\n"
                    "look_at = [278.0, 274.0, 0.0]\n"
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        ASSERT_TRUE(loaded.has_value());
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
    }

    void TearDown() override {
        context.reset();
        if (!testDir.empty() && std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
        VulkanDeviceTest::TearDown();
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_F(TlasRefitTest, RefitMovesGeometryAndMovesItBack) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // Baseline: the center pixel sees the box
    auto before = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(before.has_value()) << before.error();
    ASSERT_TRUE(before.value().hit);

    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    std::vector<glm::mat4> original;
    original.reserve(scene->nodes.size());
    for (const auto& node : scene->nodes) {
        original.push_back(node.transform);
    }

    // Shove every node far off-camera and refit
    const glm::mat4 shove = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 100000.0f, 0.0f));
    for (u32 i = 0; i < scene->nodes.size(); ++i) {
        context->SetNodeTransform(i, shove * original[i]);
    }
    context->RefitAccelerationStructure();

    auto moved = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(moved.has_value()) << moved.error();
    EXPECT_FALSE(moved.value().hit) << "geometry did not move after refit";

    // Move everything back: the same pixel must see the box again, at the
    // same distance (the refit is exact for rigid translations)
    for (u32 i = 0; i < scene->nodes.size(); ++i) {
        context->SetNodeTransform(i, original[i]);
    }
    context->RefitAccelerationStructure();

    auto restored = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(restored.has_value()) << restored.error();
    ASSERT_TRUE(restored.value().hit);
    EXPECT_NEAR(restored.value().hitT, before.value().hitT, 1.0f);
}
