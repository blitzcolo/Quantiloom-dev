// ============================================================================
// Quantiloom - Unit Tests for [[duplicates]] config entries
// ============================================================================
// [[duplicates]] is how a copy-paste made in Studio survives into the
// document: a named shallow copy of a node the scene file placed. Under
// test: the copy exists after ApplyConfig with the right mesh, name and
// transform; it really traces (Pick hits it, mapped to the right node
// index); a duplicate can chain off another duplicate and be addressed by
// [[nodes]]; and a bad source warns without failing the apply.
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

class ConfigDuplicatesTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_config_dup";
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

    /// Same cornell fixture as test_pick.cpp; `extra` is appended verbatim,
    /// which is where [[duplicates]] / [[nodes]] entries go.
    ConfigApplyReport ApplyScene(const std::string& extra) {
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
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n"
                 << extra;
        }
        auto loaded = Config::Load(path.string());
        EXPECT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        return context->ApplyConfig(loaded.value());
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
};

bool CornellBoxAvailable() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return std::filesystem::exists(root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");
}

}  // namespace

TEST_F(ConfigDuplicatesTest, DuplicateExistsAndTraces) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // First pass, no duplicates: learn which node the center ray hits and
    // how many nodes the file itself places
    auto report = ApplyScene("");
    ASSERT_TRUE(report.ok()) << report.FirstError();
    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    const size_t baseCount = scene->nodes.size();

    auto pick = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(pick.has_value()) << pick.error();
    ASSERT_TRUE(pick.value().hit);
    const String sourceName = scene->nodes[pick.value().nodeIndex].name;
    ASSERT_FALSE(sourceName.empty()) << "loader should synthesize names for unnamed nodes";

    // Second pass: paste a copy of that node 500 units nearer the camera.
    // The same center ray must now hit the duplicate, mapped to its index.
    report = ApplyScene("[[duplicates]]\nsource = \"" + sourceName +
                        "\"\nname = \"pasted\"\ntranslation = [0.0, 0.0, -500.0]\n");
    ASSERT_TRUE(report.ok()) << report.FirstError();
    EXPECT_EQ(report.nodesDuplicated, 1u);

    scene = context->GetScene();
    ASSERT_EQ(scene->nodes.size(), baseCount + 1);
    const auto& copy = scene->nodes.back();
    EXPECT_EQ(copy.name, "pasted");
    EXPECT_TRUE(copy.active);
    EXPECT_EQ(glm::vec3(copy.transform[3]), glm::vec3(0.0f, 0.0f, -500.0f));

    pick = context->Pick(kSize / 2, kSize / 2);
    ASSERT_TRUE(pick.has_value()) << pick.error();
    ASSERT_TRUE(pick.value().hit);
    EXPECT_EQ(pick.value().nodeIndex, static_cast<u32>(baseCount));
}

TEST_F(ConfigDuplicatesTest, ChainsAndNodesOverridesCompose) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    // copyB duplicates copyA (file order), and [[nodes]] -- resolved after
    // [[duplicates]] -- may address a duplicate by name
    const auto report = ApplyScene(
        "[[duplicates]]\nsource = \"Node_0\"\nname = \"copyA\"\n"
        "translation = [0.0, 0.0, 100.0]\n"
        "[[duplicates]]\nsource = \"copyA\"\nname = \"copyB\"\n"
        "[[nodes]]\nname = \"copyB\"\ntranslation = [123.0, 0.0, 0.0]\n");
    ASSERT_TRUE(report.ok()) << report.FirstError();
    EXPECT_EQ(report.nodesDuplicated, 2u);
    EXPECT_GE(report.nodesTransformed, 1u);

    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    const auto& copyB = scene->nodes.back();
    EXPECT_EQ(copyB.name, "copyB");
    EXPECT_EQ(glm::vec3(copyB.transform[3]), glm::vec3(123.0f, 0.0f, 0.0f));
}

TEST_F(ConfigDuplicatesTest, UnknownSourceWarnsWithoutFailing) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    const auto report = ApplyScene(
        "[[duplicates]]\nsource = \"no_such_node\"\nname = \"orphan\"\n");
    EXPECT_TRUE(report.ok()) << report.FirstError();
    EXPECT_EQ(report.nodesDuplicated, 0u);

    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);
    for (const auto& node : scene->nodes) {
        EXPECT_NE(node.name, "orphan");
    }
}
