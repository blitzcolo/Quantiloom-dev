// ============================================================================
// Quantiloom - Unit Tests for runtime scene topology edits
// ============================================================================
// DuplicateNode / RemoveNode / RestoreNode drive copy-paste-delete in the
// GUI. The contract under test: a duplicate is a real instance (Pick can hit
// it, nearest-first), removal is a tombstone (indices never shift, the node
// stops tracing), and restore is its exact undo. Pick doubles as the
// verification probe because it exercises the same instance->node mapping
// the GUI relies on after every rebuild.
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

using namespace quantiloom;

namespace {

constexpr u32 kSize = 64;

class SceneEditTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_scene_edit";
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

    /// Same cornell fixture as test_pick.cpp: the box spans roughly
    /// 556 x 549 x 559 from the origin, camera 800 units in front.
    void ApplyScene() {
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
        ASSERT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
    }

    /// Pick the center pixel and require a hit.
    u32 CenterPickNode() {
        const auto result = context->Pick(kSize / 2, kSize / 2);
        EXPECT_TRUE(result.has_value()) << (result.has_value() ? "" : result.error());
        EXPECT_TRUE(result.value().hit);
        return result.value().nodeIndex;
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_F(SceneEditTest, DuplicateRemoveRestoreRoundTrip) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene();
    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    const size_t originalNodeCount = scene->nodes.size();

    // Whatever the camera sees at the center is the node we instance
    const u32 originalIndex = CenterPickNode();

    // The copy sits 500 units nearer the camera, so the same center ray must
    // now hit the copy first
    auto duplicated = context->DuplicateNode(originalIndex, "scene_edit_copy");
    ASSERT_TRUE(duplicated.has_value()) << duplicated.error();
    const u32 copyIndex = duplicated.value();
    EXPECT_EQ(copyIndex, static_cast<u32>(originalNodeCount));
    EXPECT_EQ(scene->nodes[copyIndex].name, "scene_edit_copy");
    EXPECT_EQ(scene->nodes[copyIndex].meshIndex, scene->nodes[originalIndex].meshIndex);

    const glm::mat4 nearer = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -500.0f)) *
                             scene->nodes[originalIndex].transform;
    context->SetNodeTransform(copyIndex, nearer);
    context->RebuildAccelerationStructure();

    EXPECT_EQ(CenterPickNode(), copyIndex);

    // Tombstone the copy: the original is what the ray hits again, and no
    // index shifted to make that happen
    EXPECT_TRUE(context->RemoveNode(copyIndex));
    EXPECT_FALSE(context->RemoveNode(copyIndex)) << "double remove must report false";
    EXPECT_EQ(scene->nodes.size(), originalNodeCount + 1) << "tombstone must not erase";
    context->RebuildAccelerationStructure();

    EXPECT_EQ(CenterPickNode(), originalIndex);

    // Restore is the undo of remove. Going through the refit entry point on
    // purpose: it must detect the topology change and fall back to a rebuild.
    EXPECT_TRUE(context->RestoreNode(copyIndex));
    EXPECT_FALSE(context->RestoreNode(copyIndex)) << "double restore must report false";
    context->RefitAccelerationStructure();

    EXPECT_EQ(CenterPickNode(), copyIndex);
}

TEST_F(SceneEditTest, InvalidIndicesAreRejected) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    ApplyScene();
    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    const u32 outOfRange = static_cast<u32>(scene->nodes.size());

    EXPECT_FALSE(context->DuplicateNode(outOfRange, "nope").has_value());
    EXPECT_FALSE(context->RemoveNode(outOfRange));
    EXPECT_FALSE(context->RestoreNode(outOfRange));
    EXPECT_FALSE(context->RestoreNode(0)) << "restoring an active node must report false";
}
