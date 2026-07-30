// ============================================================================
// Quantiloom - Unit Tests for ExternalRenderContext::ApplyConfig
// ============================================================================
// ConfigResolve's tests cover what a config *means*. These cover that applying
// it actually reaches the context: the end of the path that the CLI and the GUI
// now share, on a real device.
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

class ApplyConfigTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        testDir = std::filesystem::temp_directory_path() / "quantiloom_apply_config";
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
        params.width = 64;
        params.height = 64;
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

    /// A config naming a scene that is committed in-repo, so this needs no
    /// submodule.
    ///
    /// Extra keys go inside the sections the fixture already writes rather than
    /// being appended as text: TOML will not let a table be opened twice.
    Config WriteConfig(const std::string& spectralKeys = "",
                       const std::string& lightingKeys = "") {
        const std::filesystem::path root(QUANTILOOM_SOURCE_ROOT);
        const auto gltf = (root / "assets" / "models" / "cornell_box" / "cornell_box.gltf");

        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << "[renderer]\nresolution = [64, 64]\n"
                 << "[camera]\nposition = [0.0, 1.0, 3.0]\nlook_at = [0.0, 1.0, 0.0]\n"
                 << "[spectral]\n" << spectralKeys
                 << "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = [0.0, 0.0, 0.0]\nsky_radiance = [0.0, 0.0, 0.0]\n"
                 << lightingKeys
                 << "[material]\nalbedo = [0.8, 0.8, 0.8]\n"
                 << "[scene]\ngltf = \"" << gltf.generic_string() << "\"\n";
        }
        auto loaded = Config::Load(path.string());
        EXPECT_TRUE(loaded.has_value()) << "fixture TOML did not parse";
        return loaded.value();
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

TEST_F(ApplyConfigTest, AppliesSpectralStateAndReportsWhatItLoaded) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("mode = \"lwir_fused\"\n");

    ConfigApplyOptions options;
    options.missingRequired = ConfigApplyOptions::MissingKeyPolicy::WarnAndDefault;
    const auto report = context->ApplyConfig(config, options);

    ASSERT_TRUE(report.ok()) << report.FirstError();
    EXPECT_TRUE(report.sceneLoaded);
    EXPECT_TRUE(context->HasScene());

    // The getters are how a host reads back what was applied -- the report
    // deliberately does not carry a second copy.
    EXPECT_EQ(context->GetSpectralMode(), SpectralMode::LWIR_Fused);
    EXPECT_FLOAT_EQ(context->GetWavelength(), 10000.0f);  // band centre, not 550
}

TEST_F(ApplyConfigTest, ShadowRaysReachTheLightingParams) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("");
    const auto report = context->ApplyConfig(config);
    ASSERT_TRUE(report.ok()) << report.FirstError();

    // Default on, the same as the CLI gets from a config without the key.
    EXPECT_EQ(context->GetLightingParams().enableShadowRays, 1u);
}

TEST_F(ApplyConfigTest, ResolutionIsEchoedRatherThanApplied) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("");
    const auto report = context->ApplyConfig(config);
    ASSERT_TRUE(report.ok()) << report.FirstError();

    // A viewport renders at its own size; the config's is reported so the host
    // can show it rather than silently disagree with the file.
    EXPECT_EQ(report.configWidth, 64u);
    EXPECT_EQ(report.configHeight, 64u);
    // The camera keeps the viewport's aspect ratio, not the config's.
    EXPECT_FLOAT_EQ(context->GetCamera().GetAspectRatio(), 1.0f);
}

TEST_F(ApplyConfigTest, TheConfigCameraOverridesTheSceneFilesOwn) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("");
    const auto report = context->ApplyConfig(config);
    ASSERT_TRUE(report.ok()) << report.FirstError();

    // AdoptScene takes the scene's camera; the config is the document of record.
    EXPECT_FLOAT_EQ(context->GetCamera().GetPosition().z, 3.0f);
    EXPECT_FLOAT_EQ(context->GetCamera().GetLookAt().y, 1.0f);
}

TEST_F(ApplyConfigTest, MultispectralWarnsAndPreviewsInRgb) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("mode = \"multispectral\"\n");
    const auto report = context->ApplyConfig(config);
    ASSERT_TRUE(report.ok()) << report.FirstError();

    // A cube render is a separate, non-progressive renderer. Saying so beats
    // silently resolving the mode to RGB, which is what used to happen.
    EXPECT_EQ(context->GetSpectralMode(), SpectralMode::RGB);
    bool warned = false;
    for (const auto& m : report.messages) {
        if (m.key == "spectral.mode" &&
            m.severity == ConfigApplyMessage::Severity::Warning) {
            warned = true;
        }
    }
    EXPECT_TRUE(warned);
}

TEST_F(ApplyConfigTest, ApplyingTwiceIsHowADocumentIsReplayed) {
    if (!CornellBoxAvailable()) GTEST_SKIP() << "cornell_box.gltf not in assets/models/cornell_box";

    auto config = WriteConfig("mode = \"vis_fused\"\n", "solar_lut = \"equal_energy\"\n");
    ASSERT_TRUE(context->ApplyConfig(config).ok());

    // A lost device is rebuilt by applying the same config again; nothing
    // should accumulate across the two.
    const auto second = context->ApplyConfig(config);
    ASSERT_TRUE(second.ok()) << second.FirstError();
    EXPECT_EQ(context->GetSpectralMode(), SpectralMode::VIS_Fused);
    EXPECT_TRUE(second.solarLutLoaded);
    EXPECT_EQ(context->GetAccumulatedSamples(), 0u);
}
