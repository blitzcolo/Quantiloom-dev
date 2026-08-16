// ============================================================================
// Quantiloom - Unit Tests for renderer/ThermalPreview via the ERC facade
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <filesystem>

using namespace quantiloom;

namespace {

#ifndef QUANTILOOM_SOURCE_ROOT
#define QUANTILOOM_SOURCE_ROOT "."
#endif

class ThermalPreviewTest : public quantiloom::testing::VulkanDeviceTest {
protected:
    void SetUp() override {
        VulkanDeviceTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        auto* shared = quantiloom::testing::SharedVulkanDevice();
        ASSERT_NE(shared, nullptr);

        testDir = std::filesystem::temp_directory_path() / "quantiloom_thermal_preview";
        std::filesystem::create_directories(testDir);

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

    Config MakeThermalConfig() const {
        const String toml = String(R"(
[renderer]
resolution = [64, 64]
spp = 1

[spectral]
mode = "lwir_fused"
band = "LWIR"

[scene]
gltf = ")") + QUANTILOOM_SOURCE_ROOT + R"(/assets/models/endmember_checker.glb"
default_temperature_k = 288.15

[camera]
position = [0.0, 14.0, 6.0]
look_at  = [0.0, 0.0, 0.0]
fov_y    = 50.0

[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance  = [0.0, 0.0, 0.0]
sky_radiance  = [0.0, 0.0, 0.0]

[atmosphere]
preset = "disabled"
sky_model = "clear_sky"
air_temperature_k = 293.15
relative_humidity = 50.0

[thermal]
enabled = true
time_h = 12.0
start_time_h = 0.0
timestep_s = 60.0
layers = 10
initial = "steady"
sun_irradiance_w_m2 = 900.0
exchange_rays = 64
exchange_top_k = 8

[material]
albedo = [0.5, 0.5, 0.5]

[[materials]]
name = "CheckerGround"
ir_emissivity = 0.92
thermal_conductivity_w_mk = 1.4
density_kg_m3 = 2300.0
specific_heat_j_kgk = 880.0
thickness_m = 0.2
convection_h_w_m2k = 10.0
shortwave_absorptivity = 0.6
interior_bc = "adiabatic"

[thermography]
enabled = true
emissivity = 0.92
reflected_temperature_k = 273.0

[quality]
fail_on_srgb_upsample = false

[sensor]
enabled = false
)";
        auto result = Config::Parse(toml);
        EXPECT_TRUE(result.has_value()) << result.error();
        return std::move(result.value());
    }

    std::unique_ptr<ExternalRenderContext> context;
    std::filesystem::path testDir;
};

}  // namespace

TEST_F(ThermalPreviewTest, ApplyConfigReportsTheThermalMaterials) {
    const Config config = MakeThermalConfig();
    const auto report = context->ApplyConfig(config);
    EXPECT_TRUE(report.ok()) << report.FirstError();
    EXPECT_GT(report.thermalMaterialsApplied, 0u);
}

TEST_F(ThermalPreviewTest, DisableRestoresTheDummyAndTheContextStaysReady) {
    context->ApplyConfig(MakeThermalConfig());
    context->SetThermalSolveEnabled(false);
    EXPECT_TRUE(context->IsReady());
}

TEST_F(ThermalPreviewTest, SetThermalTimeReturnsAnErrorWhenNoMaterialsAreSet) {
    auto result = context->SetThermalTime(12.0);
    EXPECT_FALSE(result.has_value());
}
