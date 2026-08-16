// ============================================================================
// Quantiloom - Unit Tests for renderer/ThermalPreview via the ERC facade
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/VulkanContext.hpp"
#include "support/VulkanTestDevice.hpp"

#include <filesystem>
#include <fstream>

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

TEST_F(ThermalPreviewTest, TheStatusSaysWhyTheSolveDidNotRun) {
    // A panel that can only say "no result" leaves the user guessing between a
    // scene with no geometry and a scene with no thermal properties.
    auto result = context->SetThermalTime(12.0);
    ASSERT_FALSE(result.has_value());
    EXPECT_FALSE(context->GetThermalSolveStatus().error.empty());
}

TEST_F(ThermalPreviewTest, AForcingFileGivesTheTrajectoryOneSunColumnPerRow) {
    // Constant forcing is one column -- one sun position for the whole run,
    // which is right when the sun does not move. A forcing file has to give
    // the interactive path what the offline one gets: a column per row, so the
    // shadows track the day instead of sitting where they were at hour zero.
    // And swapping the file must not re-run the hemisphere precompute, which
    // is the expensive half and does not depend on where the sun is.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    const ThermalSolveStatus constant = context->GetThermalSolveStatus();
    EXPECT_EQ(constant.sunSampleCount, 1u);
    ASSERT_GT(constant.exchangeRunCount, 0u);

    const auto forcingPath = testDir / "forcing.csv";
    {
        std::ofstream out(forcingPath);
        out << "# time_h, air_k, dni, azimuth, elevation, sky_k, diffuse, rh\n"
               "6,  288.0, 200, 90,  15, 268.0, 90,  70\n"
               "9,  292.0, 700, 120, 45, 271.0, 140, 55\n"
               "12, 297.0, 900, 180, 70, 274.0, 160, 40\n"
               "15, 296.0, 650, 240, 45, 273.0, 130, 45\n"
               "18, 291.0, 150, 270, 10, 269.0, 70,  60\n";
    }

    ThermalSolveParams params;
    params.startTime_h = 6.0;
    params.timestep_s = 300.0;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.forcingFile = forcingPath.string();
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    const ThermalSolveStatus diurnal = context->GetThermalSolveStatus();
    EXPECT_EQ(diurnal.sunSampleCount, 5u);
    EXPECT_EQ(diurnal.exchangeRunCount, constant.exchangeRunCount)
        << "changing the forcing file must not rebuild the view factors";
    EXPECT_DOUBLE_EQ(diurnal.sliderStartTime_h, 6.0);
    EXPECT_DOUBLE_EQ(diurnal.sliderEndTime_h, 18.0);
}

TEST_F(ThermalPreviewTest, AWetMaterialCoolsTheInteractiveSolve) {
    // The whole point of the wetness factor reaching the facade: a material
    // edit has to move the temperatures the viewport is showing.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 dry = context->GetThermalSolveStatus().meanTemperature_K;
    ASSERT_GT(dry, 0.0);

    ThermalMaterialParams wet;
    wet.conductivity_W_mK = 1.4f;
    wet.density_kg_m3 = 2300.0f;
    wet.specificHeat_J_kgK = 880.0f;
    wet.thickness_m = 0.2f;
    wet.convection_W_m2K = 10.0f;
    wet.shortwaveAbsorptivity = 0.6f;
    wet.wetnessFactor = 0.8f;
    context->SetThermalMaterial("CheckerGround", wet);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    EXPECT_LT(context->GetThermalSolveStatus().meanTemperature_K, dry - 5.0);
}

TEST_F(ThermalPreviewTest, DiffuseIrradianceWarmsASceneWithNoDirectSun) {
    // Overcast: no disc at all. Before the diffuse term this scene had no
    // solar input and settled on the air temperature.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 0.0;
    params.skyTemperature_K = 274.0;
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 overcastWithoutDiffuse =
        context->GetThermalSolveStatus().meanTemperature_K;

    params.diffuseIrradiance_W_m2 = 400.0;
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    EXPECT_GT(context->GetThermalSolveStatus().meanTemperature_K,
              overcastWithoutDiffuse + 5.0);
}
