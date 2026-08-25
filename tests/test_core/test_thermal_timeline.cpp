// ============================================================================
// Quantiloom - Unit Tests for thermal/ThermalTimeline.hpp
// ============================================================================
// The timeline is what makes scrubbing the time slider interactive rather
// than a fresh solve from t=0 every time. Its correctness hinges on two
// invariants:
//
//   1. A query at hour H produces the same state regardless of what was
//      queried before it (path-independence).
//
//   2. An off-grid query does not contaminate the grid: StateAt(12.3)
//      followed by StateAt(18) must give the same answer as a fresh
//      timeline queried at 18.
//
// Both are tested here with the CPU stepper so the results are f64-exact.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalTimeline.hpp"

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> OneElement() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

ThermalMaterial ConcreteMaterial() {
    ThermalMaterial material;
    material.conductivity_W_mK = 1.4f;
    material.density_kg_m3 = 2300.0f;
    material.specificHeat_J_kgK = 880.0f;
    material.thickness_m = 0.2f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.6f;
    material.longwaveEmissivity = 0.92f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

ThermalForcing ConstantForcing() {
    ThermalForcing f;
    f.airTemperature_K = 293.15;
    f.sunIrradiance_W_m2 = 900.0;
    f.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    f.skyTemperature_K = 270.0;
    return f;
}

ThermalTimeline::Desc DefaultDesc() {
    ThermalTimeline::Desc desc;
    desc.startTime_h = 0.0;
    desc.timestep_s = 60.0;
    desc.checkpointStride_h = 1.0;
    desc.nodeCount = 10;
    desc.initial = InitialCondition::Steady;
    desc.initialTemperature_K = 293.15;
    return desc;
}

SunVisibilityTable OneColumnTable(usize n, f64 time_h) {
    SunVisibilityTable table;
    table.sampleTime_h = {time_h};
    table.visibility.assign(n, 1.0f);
    return table;
}

}  // namespace

TEST(ThermalTimelineTest, AQueryMatchesAnUninterruptedRun) {
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{ConcreteMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);
    const auto sunTable = OneColumnTable(1, 0.0);
    const auto forcing = ConstantForcing();
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;
    auto desc = DefaultDesc();

    // Fresh timeline, queried at several hours then at 18
    ThermalTimeline scrubbed(desc, elements, materials, exchange, sunTable,
                             noSeries, forcing, stepper);
    scrubbed.StateAt(6.0);
    scrubbed.StateAt(12.0);
    scrubbed.StateAt(3.0);  // backwards
    const ThermalState& s1 = scrubbed.StateAt(18.0);

    // Fresh timeline, queried only at 18
    ThermalTimeline fresh(desc, elements, materials, exchange, sunTable,
                          noSeries, forcing, stepper);
    const ThermalState& s2 = fresh.StateAt(18.0);

    ASSERT_EQ(s1.temperature_K.size(), s2.temperature_K.size());
    for (usize i = 0; i < s1.temperature_K.size(); ++i) {
        EXPECT_DOUBLE_EQ(s1.temperature_K[i], s2.temperature_K[i])
            << "node " << i;
    }
}

TEST(ThermalTimelineTest, AnOffGridQueryLeavesTheGridUntouched) {
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{ConcreteMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);
    const auto sunTable = OneColumnTable(1, 0.0);
    const auto forcing = ConstantForcing();
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;
    auto desc = DefaultDesc();

    // Query at off-grid time, then at on-grid time
    ThermalTimeline tl(desc, elements, materials, exchange, sunTable,
                       noSeries, forcing, stepper);
    tl.StateAt(12.3);  // off-grid
    const ThermalState& s1 = tl.StateAt(18.0);  // on-grid

    // Fresh timeline, queried only at 18
    ThermalTimeline fresh(desc, elements, materials, exchange, sunTable,
                          noSeries, forcing, stepper);
    const ThermalState& s2 = fresh.StateAt(18.0);

    ASSERT_EQ(s1.temperature_K.size(), s2.temperature_K.size());
    for (usize i = 0; i < s1.temperature_K.size(); ++i) {
        EXPECT_DOUBLE_EQ(s1.temperature_K[i], s2.temperature_K[i])
            << "off-grid query contaminated the grid at node " << i;
    }
}

TEST(ThermalTimelineTest, ScrubbingBackwardsReusesACheckpoint) {
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{ConcreteMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);
    const auto sunTable = OneColumnTable(1, 0.0);
    const auto forcing = ConstantForcing();
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;
    auto desc = DefaultDesc();
    desc.checkpointStride_h = 1.0;

    ThermalTimeline tl(desc, elements, materials, exchange, sunTable,
                       noSeries, forcing, stepper);

    tl.StateAt(20.0);  // creates checkpoints at 0,1,...,20
    EXPECT_GE(tl.CheckpointCount(), 20u);

    // Scrub backwards to 19.5 — should cost at most stride/dt + 1 = 61 steps
    tl.StateAt(19.5);
    const u32 cost = tl.LastStepCount();
    const u32 maxCost = static_cast<u32>(desc.checkpointStride_h * 3600.0 / desc.timestep_s) + 2;
    EXPECT_LE(cost, maxCost) << "backwards scrub cost " << cost
                             << " steps, expected at most " << maxCost;
}

TEST(ThermalTimelineTest, TheSteadyInitialStateIsComputedOnce) {
    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{ConcreteMaterial()};
    const auto exchange = MakeOpenSkyExchange(1);
    const auto sunTable = OneColumnTable(1, 0.0);
    const auto forcing = ConstantForcing();
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;
    auto desc = DefaultDesc();

    ThermalTimeline tl(desc, elements, materials, exchange, sunTable,
                       noSeries, forcing, stepper);

    // The initial checkpoint is already stored
    EXPECT_EQ(tl.CheckpointCount(), 1u);

    // Querying at startTime returns the initial state (zero additional steps)
    tl.StateAt(0.0);
    EXPECT_EQ(tl.LastStepCount(), 0u);
}

TEST(ThermalTimelineTest, ThePartialStepUsesTheRemainderDt) {
    const auto elements = OneElement();

    // A lumped material (high conductivity, thin) so we can compare
    // against the exponential closed form
    ThermalMaterial material;
    material.conductivity_W_mK = 400.0f;
    material.density_kg_m3 = 8000.0f;
    material.specificHeat_J_kgK = 400.0f;
    material.thickness_m = 0.01f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    const Vector<ThermalMaterial> materials{material};

    const auto exchange = MakeOpenSkyExchange(1);
    const auto sunTable = OneColumnTable(1, 0.0);
    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;
    forcing.sunIrradiance_W_m2 = 0.0;
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;

    ThermalTimeline::Desc desc;
    desc.startTime_h = 0.0;
    desc.timestep_s = 60.0;
    desc.checkpointStride_h = 1.0;
    desc.nodeCount = 12;
    desc.initial = InitialCondition::Uniform;
    desc.initialTemperature_K = 350.0;

    ThermalTimeline tl(desc, elements, materials, exchange, sunTable,
                       noSeries, forcing, stepper);

    // Query at a non-grid time: 0.5 hours = 30 minutes = 1800 s
    // dt=60, so 30 full steps to k=30, then no remainder (30*60=1800)
    // Try an actual off-grid query: 0.5 + 30s = 0.50833... hours
    const f64 queryTime = 0.5 + 30.0 / 3600.0;  // 30.5 minutes
    const ThermalState& state = tl.StateAt(queryTime);

    const f64 tau = 8000.0 * 400.0 * 0.01 / 10.0;  // 3200 s
    const f64 elapsed = queryTime * 3600.0;
    const f64 expected = 280.0 + (350.0 - 280.0) * std::exp(-elapsed / tau);
    EXPECT_NEAR(state.Surface(0), expected, 0.5);
}

TEST(ThermalTimelineTest, RunThermalSolveStepsOnTheFixedGrid) {
    // Verify that RunThermalSolve (now a thin wrapper) still produces
    // a valid result with the fixed-grid trajectory.
    Scene scene;
    scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.5f), "Concrete"));
    scene.materials[0].irEmissivityCurve = {{3000.0f, 0.92f}, {15000.0f, 0.92f}};

    GeometryPrimitive prim;
    prim.materialId = 0;
    prim.positions = {
        glm::vec3(-1, 0, -1), glm::vec3(1, 0, -1),
        glm::vec3(1, 0, 1),   glm::vec3(-1, 0, 1),
    };
    prim.normals.assign(4, glm::vec3(0, 1, 0));
    prim.indices = {0, 2, 1, 0, 3, 2};
    Mesh mesh;
    mesh.primitives.push_back(std::move(prim));
    scene.meshes.push_back(std::move(mesh));
    SceneNode node;
    node.meshIndex = 0;
    node.transform = glm::mat4(1.0f);
    node.active = true;
    scene.nodes.push_back(node);

    ThermalConfig config;
    config.enabled = true;
    config.time_h = 6.0;
    config.startTime_h = 0.0;
    config.timestep_s = 60.0;
    config.nodeCount = 10;
    config.initial = InitialCondition::Steady;
    config.airTemperature_K = 293.15;
    config.sunIrradiance_W_m2 = 900.0;
    config.skyTemperature_K = 270.0;

    ThermalMaterial tm;
    tm.conductivity_W_mK = 1.4f;
    tm.density_kg_m3 = 2300.0f;
    tm.specificHeat_J_kgK = 880.0f;
    tm.thickness_m = 0.2f;
    tm.convection_W_m2K = 10.0f;
    tm.shortwaveAbsorptivity = 0.6f;
    config.materials["Concrete"] = tm;

    const auto exchange = MakeOpenSkyExchange(2);
    const ThermalResult result = RunThermalSolve(scene, config, exchange);

    EXPECT_TRUE(result.error.empty()) << result.error;
    EXPECT_GT(result.participatingElements, 0u);
    EXPECT_GT(result.stepsTaken, 0u);
    EXPECT_GT(result.meanTemperature_K, 250.0);
    EXPECT_LT(result.meanTemperature_K, 400.0);

    // RunThermalSolve now takes a stepper, defaulted to null so callers
    // without a Vulkan device -- this test among them -- keep working. A null
    // stepper must be the CPU one exactly, not merely something like it: the
    // solve cache keys on which stepper ran, so a default that quietly became
    // a different implementation would serve entries across the two.
    const ThermalResult explicitDefault =
        RunThermalSolve(scene, config, exchange, SunVisibilityTable{}, nullptr);
    ASSERT_TRUE(explicitDefault.error.empty()) << explicitDefault.error;
    EXPECT_EQ(explicitDefault.surfaceTemperature_K, result.surfaceTemperature_K);
    EXPECT_EQ(explicitDefault.sunSensitivity_K, result.sunSensitivity_K);
    EXPECT_EQ(explicitDefault.stepsTaken, result.stepsTaken);
    EXPECT_DOUBLE_EQ(explicitDefault.meanTemperature_K, result.meanTemperature_K);

    CpuCrankNicolsonStepper cpu;
    const ThermalResult explicitCpu =
        RunThermalSolve(scene, config, exchange, SunVisibilityTable{}, &cpu);
    ASSERT_TRUE(explicitCpu.error.empty()) << explicitCpu.error;
    EXPECT_EQ(explicitCpu.surfaceTemperature_K, result.surfaceTemperature_K);
}

TEST(ThermalTimelineTest, TheBlendedSunTableScalesTheAbsorbedFlux) {
    const auto elements = OneElement();
    ThermalMaterial material;
    material.conductivity_W_mK = 400.0f;
    material.density_kg_m3 = 8000.0f;
    material.specificHeat_J_kgK = 400.0f;
    material.thickness_m = 0.01f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 1.0f;
    material.longwaveEmissivity = 0.0f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    // Two-column sun table: fully lit at t=0, fully shadowed at t=1
    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0, 1.0};
    sunTable.visibility = {1.0f, 0.0f};

    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;
    forcing.sunIrradiance_W_m2 = 900.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 280.0;
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    CpuCrankNicolsonStepper stepper;

    ThermalTimeline::Desc desc;
    desc.startTime_h = 0.0;
    desc.timestep_s = 60.0;
    desc.checkpointStride_h = 1.0;
    desc.nodeCount = 10;
    desc.initial = InitialCondition::Uniform;
    desc.initialTemperature_K = 280.0;

    // Full sun: element heats above air
    ThermalTimeline fullSun(desc, elements, materials, exchange,
                            sunTable, noSeries, forcing, stepper);
    const f64 T_fullSun = fullSun.StateAt(0.5).Surface(0);

    // Replace with a table that's fully shadowed
    SunVisibilityTable darkTable;
    darkTable.sampleTime_h = {0.0, 1.0};
    darkTable.visibility = {0.0f, 0.0f};

    ThermalTimeline noSun(desc, elements, materials, exchange,
                          darkTable, noSeries, forcing, stepper);
    const f64 T_noSun = noSun.StateAt(0.5).Surface(0);

    // With sun the element should be warmer than without
    EXPECT_GT(T_fullSun, T_noSun + 1.0)
        << "full sun " << T_fullSun << ", no sun " << T_noSun;
}
