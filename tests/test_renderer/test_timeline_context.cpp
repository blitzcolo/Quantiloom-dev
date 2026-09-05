// ============================================================================
// Quantiloom - Unit Tests for the interactive timeline
// ============================================================================
// SetTimelineTime is the scrub path: it must move the nodes a trajectory
// names, leave everything else alone, and reconcile with a gizmo drag so that
// an edit made at one instant survives every other. That a scrub costs no
// exchange precompute is checked where it is observable -- in
// test_thermal_preview.cpp, against exchangeRunCount.
//
// Skipped rather than failed on a machine with no ray-tracing GPU.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>

using namespace quantiloom;

namespace {

constexpr u32 kSize = 32;

std::filesystem::path CubePath() {
    const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
    return root / "assets" / "models" / "cornell_box" / "cornell_box.gltf";
}

/// Two instances of the same file: one still, one driven along +X at 3 m/s.
/// The trajectory is a linear engine because it is the one form with an
/// answer that can be written down exactly.
class TimelineContextTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        if (!std::filesystem::exists(CubePath())) {
            GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";
        }

        testDir = std::filesystem::temp_directory_path() / "quantiloom_timeline_context";
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

        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [32, 32]\n"
                 << "[camera]\nposition = [278.0, 274.0, -800.0]\n"
                    "look_at = [278.0, 274.0, 0.0]\n"
                 << "[scene]\n"
                 << "[spectral]\n"
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [1.0, 1.0, 1.0]\nsky_radiance = [0.1, 0.1, 0.1]\n"
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[timeline]\nend_s = 4\nticks_per_second = 20\ntime_s = 0\n"
                 << "\n[[models]]\nfile = \"" << CubePath().generic_string() << "\"\n"
                    "name = \"still\"\n"
                 << "\n[[models]]\nfile = \"" << CubePath().generic_string() << "\"\n"
                    "name = \"mover\"\n"
                 << "[models.motion.location]\ntype = \"linear\"\nvelocity = [3.0, 0.0, 0.0]\n";
        }

        auto loaded = Config::Load(path.string());
        ASSERT_TRUE(loaded.has_value());
        const auto report = context->ApplyConfig(loaded.value());
        ASSERT_TRUE(report.ok()) << report.FirstError();
        applyReport = report;
    }

    void TearDown() override {
        context.reset();
        if (!testDir.empty() && std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
        VulkanDeviceTest::TearDown();
    }

    /// The first node the "mover" model brought. Node names carry the model
    /// name and a slash, which is what makes this findable at all.
    u32 MoverNode() const {
        const Scene* scene = context->GetScene();
        for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i) {
            if (scene->nodes[i].name.rfind("mover/", 0) == 0) return i;
        }
        return 0;
    }

    u32 StillNode() const {
        const Scene* scene = context->GetScene();
        for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i) {
            if (scene->nodes[i].name.rfind("still/", 0) == 0) return i;
        }
        return 0;
    }

    std::filesystem::path testDir;
    std::unique_ptr<ExternalRenderContext> context;
    ConfigApplyReport applyReport;
};

}  // namespace

TEST_F(TimelineContextTest, TheReportSaysWhatTheConfigBrought) {
    EXPECT_TRUE(applyReport.timelinePresent);
    EXPECT_EQ(applyReport.modelsLoaded, 2u);
    EXPECT_EQ(applyReport.motionTracks, 1u);
}

TEST_F(TimelineContextTest, BothModelsAreInTheSameScene) {
    const Scene* scene = context->GetScene();
    ASSERT_NE(scene, nullptr);

    const bool hasStill = std::any_of(scene->nodes.begin(), scene->nodes.end(),
                                      [](const SceneNode& n) {
                                          return n.name.rfind("still/", 0) == 0;
                                      });
    const bool hasMover = std::any_of(scene->nodes.begin(), scene->nodes.end(),
                                      [](const SceneNode& n) {
                                          return n.name.rfind("mover/", 0) == 0;
                                      });
    EXPECT_TRUE(hasStill);
    EXPECT_TRUE(hasMover);
}

TEST_F(TimelineContextTest, TheInfoDescribesTheClockTheConfigDeclared) {
    const TimelineInfo info = context->GetTimelineInfo();
    EXPECT_TRUE(info.present);
    EXPECT_DOUBLE_EQ(info.start_s, 0.0);
    EXPECT_DOUBLE_EQ(info.end_s, 4.0);
    EXPECT_DOUBLE_EQ(info.ticksPerSecond, 20.0);
    EXPECT_EQ(info.modelCount, 2u);
    EXPECT_GT(info.animatedNodeCount, 0u);
    EXPECT_EQ(info.TickCount(), 81);  // 0 s to 4 s inclusive, twenty a second
}

TEST_F(TimelineContextTest, ScrubbingMovesOnlyTheAnimatedModel) {
    const Scene* scene = context->GetScene();
    const u32 mover = MoverNode();
    const u32 still = StillNode();

    const glm::mat4 moverAtZero = scene->nodes[mover].transform;
    const glm::mat4 stillAtZero = scene->nodes[still].transform;

    ASSERT_TRUE(context->SetTimelineTime(1.0).has_value());

    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][0], moverAtZero[3][0] + 3.0f);
    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][1], moverAtZero[3][1]);
    EXPECT_EQ(scene->nodes[still].transform, stillAtZero);
}

TEST_F(TimelineContextTest, ScrubbingBackReturnsToTheSamePose) {
    const Scene* scene = context->GetScene();
    const u32 mover = MoverNode();
    const glm::mat4 atZero = scene->nodes[mover].transform;

    ASSERT_TRUE(context->SetTimelineTime(2.5).has_value());
    ASSERT_TRUE(context->SetTimelineTime(0.0).has_value());
    EXPECT_EQ(scene->nodes[mover].transform, atZero);
}

TEST_F(TimelineContextTest, ScrubbingToTheSameInstantIsIdempotent) {
    const Scene* scene = context->GetScene();
    const u32 mover = MoverNode();

    ASSERT_TRUE(context->SetTimelineTime(1.25).has_value());
    const glm::mat4 once = scene->nodes[mover].transform;
    ASSERT_TRUE(context->SetTimelineTime(1.25).has_value());
    EXPECT_EQ(scene->nodes[mover].transform, once);
}

TEST_F(TimelineContextTest, AGizmoEditAtOneInstantSurvivesEveryOther) {
    const Scene* scene = context->GetScene();
    const u32 mover = MoverNode();

    ASSERT_TRUE(context->SetTimelineTime(2.0).has_value());
    const glm::mat4 atTwo = scene->nodes[mover].transform;

    // Lift it by five metres, the way a drag would.
    const glm::mat4 lifted = glm::translate(glm::mat4(1.0f), glm::vec3(0, 5, 0)) * atTwo;
    context->SetNodeTransform(mover, lifted);
    context->RefitAccelerationStructure();

    // The offset is part of the rest pose now, so it is there at every tick.
    ASSERT_TRUE(context->SetTimelineTime(0.0).has_value());
    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][1], atTwo[3][1] + 5.0f);
    // ...and the trajectory still runs from wherever the rest pose is.
    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][0], atTwo[3][0] - 6.0f);

    ASSERT_TRUE(context->SetTimelineTime(2.0).has_value());
    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][1], atTwo[3][1] + 5.0f);
    EXPECT_FLOAT_EQ(scene->nodes[mover].transform[3][0], atTwo[3][0]);
}

TEST_F(TimelineContextTest, TheRestPoseIsWhatADocumentWouldRecord) {
    const Scene* scene = context->GetScene();
    const u32 mover = MoverNode();
    const u32 still = StillNode();

    const glm::mat4 rest = context->GetNodeRestTransform(mover);

    ASSERT_TRUE(context->SetTimelineTime(3.0).has_value());
    // The rest pose does not follow the clock...
    EXPECT_EQ(context->GetNodeRestTransform(mover), rest);
    // ...but the node does.
    EXPECT_NE(scene->nodes[mover].transform, rest);

    // A node with no trajectory has one pose, and it is both.
    EXPECT_EQ(context->GetNodeRestTransform(still), scene->nodes[still].transform);
}
