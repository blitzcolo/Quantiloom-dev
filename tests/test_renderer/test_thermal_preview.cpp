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

TEST_F(ThermalPreviewTest, AskingForAMaterialTangentChangesNoTemperature) {
    // A derivative is carried beside the trajectory, not inside it. If asking
    // for one moved the temperatures, every number the viewport showed would
    // depend on what the panel happened to have selected -- and the fit these
    // exist for would be fitting against a perturbed solve.
    //
    // One tangent against two, rather than none against one, because asking
    // for any of them moves the solve onto the CPU stepper: the GPU one does
    // not integrate them, and the two steppers differ in their last bits by
    // construction. So the comparison here holds the stepper fixed and varies
    // only what it is carrying, which is the claim being made.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 800.0;
    params.parameterSensitivities = {ThermalSensitivityParameter::Convection};
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 one = context->GetThermalSolveStatus().meanTemperature_K;
    ASSERT_GT(one, 0.0);

    params.parameterSensitivities = {ThermalSensitivityParameter::Convection,
                                     ThermalSensitivityParameter::Conductivity};
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    EXPECT_NEAR(context->GetThermalSolveStatus().meanTemperature_K, one, 1e-9)
        << "carrying a second tangent perturbed the trajectory both are tangents of";
}

TEST_F(ThermalPreviewTest, ASolveThatCannotCarryATangentDoesNotRun) {
    // The bug this pins: the state is sized by the descriptor, so a stepper
    // that does not integrate the material tangents still receives the vectors
    // and hands them back at zero. A zero derivative reads as "no parameter
    // changes anything", which is an answer rather than an absence, and every
    // check above it passes while it is wrong.
    //
    // What makes it observable without reaching into the stepper choice: the
    // tangent has to come out nonzero for a scene under a sun.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 800.0;
    params.parameterSensitivities = {ThermalSensitivityParameter::Convection};
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    auto field =
        context->GetThermalParameterSensitivity(ThermalSensitivityParameter::Convection);
    ASSERT_TRUE(field.has_value()) << field.error();
    ASSERT_FALSE(field.value().empty());
    EXPECT_LT(field.value()[0], 0.0)
        << "more convection cools a surface the sun is heating, so dT/dh is negative";
}

TEST_F(ThermalPreviewTest, AnElementTrajectoryReplaysWithoutMovingTheViewport) {
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 800.0;
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 shown = context->GetThermalSolveStatus().meanTemperature_K;
    ASSERT_GT(shown, 0.0);

    auto probe = context->GetElementTrajectory(0, 6.0, 18.0, 13);
    ASSERT_TRUE(probe.has_value()) << probe.error();
    const auto& trajectory = probe.value();

    EXPECT_EQ(trajectory.time_h.size(), 13u);
    EXPECT_EQ(trajectory.surfaceTemperature_K.size(), 13u);
    EXPECT_EQ(trajectory.backTemperature_K.size(), 13u);
    EXPECT_DOUBLE_EQ(trajectory.time_h.front(), 6.0);
    EXPECT_DOUBLE_EQ(trajectory.time_h.back(), 18.0);
    for (const f64 T : trajectory.surfaceTemperature_K) {
        EXPECT_GT(T, 100.0) << "a replayed sample is not a temperature";
    }

    // A probe is a question about the past, not a request to move. Replaying
    // walks the timeline to 18:00 and the viewport is still showing noon, so
    // the status it reports has to be noon's.
    EXPECT_DOUBLE_EQ(context->GetThermalSolveStatus().meanTemperature_K, shown);
}

TEST_F(ThermalPreviewTest, TheSixFluxesSumToWhatTheSurfaceIsStoring) {
    // The claim that makes the decomposition readable rather than six
    // unrelated numbers: they are the whole balance, so their sum is the rate
    // the surface half-cell is storing heat. Checked against a finite
    // difference of the temperature the same replay reports, which is the only
    // reading of it that does not just restate the code.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.timestep_s = 30.0;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 800.0;
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    // Half a minute apart, which is the timestep: closer and the difference is
    // rounding, further and the balance has moved between the two samples.
    const f64 dt_h = 30.0 / 3600.0;
    auto probe = context->GetElementTrajectory(0, 12.0, 12.0 + dt_h, 2);
    ASSERT_TRUE(probe.has_value()) << probe.error();
    const auto& t = probe.value();
    ASSERT_EQ(t.fluxes.size(), 2u) << "the CPU stepper decomposes its own balance";

    const ThermalSurfaceFluxes& f = t.fluxes.front();
    const f64 sum = f.shortwave_W_m2 + f.longwave_W_m2 + f.convection_W_m2 +
                    f.latent_W_m2 + f.conduction_W_m2 + f.lateral_W_m2;

    // rho c dx/2 dT/dt, the half-cell's storage. The material is the config's
    // CheckerGround: 1.4 W/mK, 2300 kg/m^3, 880 J/kgK, 0.2 m over 10 nodes.
    const f64 rhoC = 2300.0 * 880.0;
    const f64 dx = 0.2 / 9.0;
    const f64 dTdt =
        (t.surfaceTemperature_K[1] - t.surfaceTemperature_K[0]) / (dt_h * 3600.0);
    const f64 storing = rhoC * dx * 0.5 * dTdt;

    // Loose, and honestly so: the fluxes are read at the start of the step and
    // the difference spans it, so the two agree to the scheme's order rather
    // than exactly. What this catches is a term with the wrong sign or a term
    // left out, which is a factor rather than a percent.
    EXPECT_NEAR(sum, storing, 0.25 * std::abs(storing) + 5.0)
        << "fluxes " << f.shortwave_W_m2 << " sw, " << f.longwave_W_m2 << " lw, "
        << f.convection_W_m2 << " conv, " << f.latent_W_m2 << " lat, "
        << f.conduction_W_m2 << " cond, " << f.lateral_W_m2 << " lat.cond";

    // In full sun at noon the two that carry the surface are the absorbed
    // shortwave, which is positive, and the conduction into the slab, which is
    // not: a sunlit surface is where the heat arrives and the slab is where it
    // goes.
    EXPECT_GT(f.shortwave_W_m2, 0.0);
    EXPECT_LT(f.conduction_W_m2, 0.0);
}

TEST_F(ThermalPreviewTest, AProbeSaysWhenItCannotAnswer) {
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    EXPECT_FALSE(context->GetElementTrajectory(1u << 30, 0.0, 24.0, 8).has_value())
        << "an element past the end of the mesh is an error, not an empty answer";
    EXPECT_FALSE(context->GetElementTrajectory(0, 0.0, 24.0, 1).has_value())
        << "one sample is not a trajectory";

    // Given backwards, which a slider dragged the other way will do.
    auto reversed = context->GetElementTrajectory(0, 18.0, 6.0, 5);
    ASSERT_TRUE(reversed.has_value()) << reversed.error();
    EXPECT_DOUBLE_EQ(reversed.value().time_h.front(), 6.0);
    EXPECT_DOUBLE_EQ(reversed.value().time_h.back(), 18.0);
}

TEST_F(ThermalPreviewTest, APickNamesTheElementAProbeAsksAbout) {
    // The path from a click in the viewport to a chart of that surface's day:
    // a pick reports an instance and a triangle within it, the solve indexes
    // its elements flat, and without this map a host holding a PickResult has
    // no way to name what it just clicked on.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    quantiloom::PickResult pick;
    pick.hit = true;
    pick.instanceIndex = 0;
    pick.primitiveIndex = 0;
    auto element = context->ThermalElementAt(pick);
    ASSERT_TRUE(element.has_value()) << element.error();
    EXPECT_TRUE(context->GetElementTrajectory(element.value(), 11.0, 13.0, 3).has_value());

    // A ray that reached the sky named nothing, and saying so beats naming
    // element zero.
    quantiloom::PickResult missed;
    missed.hit = false;
    EXPECT_FALSE(context->ThermalElementAt(missed).has_value());

    quantiloom::PickResult offMesh;
    offMesh.hit = true;
    offMesh.instanceIndex = 1u << 20;
    EXPECT_FALSE(context->ThermalElementAt(offMesh).has_value());
}

TEST_F(ThermalPreviewTest, TheWhatIfPreviewMatchesASolveToFirstOrder) {
    // What the preview claims: T + dT/dp * step is what a re-solve at p + step
    // would produce, in the limit of a small step. The claim is checkable
    // without the GPU at all -- the trajectory carries the tangent and the
    // solve carries the answer -- so this compares the two on the mean.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.airTemperature_K = 293.15;
    params.sunIrradiance_W_m2 = 800.0;
    params.parameterSensitivities = {ThermalSensitivityParameter::Convection};
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 base = context->GetThermalSolveStatus().meanTemperature_K;
    ASSERT_GT(base, 0.0);

    auto before = context->GetElementTrajectory(0, 12.0, 12.0 + 1.0 / 60.0, 2);
    ASSERT_TRUE(before.has_value()) << before.error();
    const f64 baseElement = before.value().surfaceTemperature_K.front();

    auto tangent =
        context->GetThermalParameterSensitivity(ThermalSensitivityParameter::Convection);
    ASSERT_TRUE(tangent.has_value()) << tangent.error();
    ASSERT_FALSE(tangent.value().empty());
    // Nonzero, and this is not a formality: a stepper that does not integrate
    // the tangent still receives the vector the descriptor sized and hands it
    // back at zero, which reads as "no slider changes anything" rather than as
    // "nobody integrated this". That is the failure this whole test exists to
    // catch, and it was the state of the GPU path until the host was taught to
    // ask whether a stepper carries what it is being given.
    ASSERT_NE(tangent.value()[0], 0.0f)
        << "the solve carried the parameter and produced a derivative of nothing";

    // A step small enough for the linearisation to hold: h is 10 W/m^2K in the
    // config's material, and half a unit of it is a five percent change.
    constexpr f64 kStep = 0.5;
    ASSERT_TRUE(context->SetThermalWhatIf(ThermalSensitivityParameter::Convection, kStep)
                    .has_value());

    // What the solve says for the same change, by making it for real.
    ThermalMaterialParams warmer;
    warmer.conductivity_W_mK = 1.4f;
    warmer.density_kg_m3 = 2300.0f;
    warmer.specificHeat_J_kgK = 880.0f;
    warmer.thickness_m = 0.2f;
    warmer.convection_W_m2K = 10.0f + static_cast<f32>(kStep);
    warmer.shortwaveAbsorptivity = 0.6f;
    context->SetThermalMaterial("CheckerGround", warmer);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());
    const f64 solved = context->GetThermalSolveStatus().meanTemperature_K;

    // The solve moved, so there is something for the preview to predict.
    ASSERT_GT(std::abs(solved - base), 0.01)
        << "half a unit of h changed nothing, so this proves nothing";

    // And the tangent predicts that move. Per element, against the element the
    // probe can also read, so this is the derivative compared with a finite
    // difference of the thing it is the derivative of.
    auto after = context->GetElementTrajectory(0, 12.0, 12.0 + 1.0 / 60.0, 2);
    ASSERT_TRUE(after.has_value()) << after.error();
    const f64 solvedElement = after.value().surfaceTemperature_K.front();

    EXPECT_NEAR(solvedElement, baseElement + tangent.value()[0] * kStep,
                0.1 * std::abs(solvedElement - baseElement) + 0.02)
        << "dT/dh = " << tangent.value()[0] << " K per W/m2K predicted "
        << (baseElement + tangent.value()[0] * kStep) << ", the solve gave "
        << solvedElement << " from " << baseElement;
}

TEST_F(ThermalPreviewTest, AWhatIfForAParameterNobodyAskedForIsRefused) {
    // The tangent is sized into the state, so previewing a parameter the solve
    // does not carry cannot be answered by looking harder -- it needs a
    // different solve. Said, rather than shown as a preview of zero.
    ASSERT_TRUE(context->ApplyConfig(MakeThermalConfig()).ok());

    ThermalSolveParams params;
    params.exchangeRays = 64;
    params.exchangeTopK = 8;
    params.parameterSensitivities = {ThermalSensitivityParameter::Convection};
    context->SetThermalSolveParams(params);
    ASSERT_TRUE(context->SetThermalTime(12.0).has_value());

    auto refused = context->SetThermalWhatIf(ThermalSensitivityParameter::Conductivity, 0.1);
    EXPECT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().find("parameterSensitivities"), String::npos)
        << "the message should say what to do about it: " << refused.error();

    // Zero is always accepted: it is how a host turns the preview off, and a
    // host that has just switched parameters should not have to know whether
    // the old one was carried.
    EXPECT_TRUE(context->SetThermalWhatIf(ThermalSensitivityParameter::Conductivity, 0.0)
                    .has_value());
}
