// ============================================================================
// Quantiloom - Unit Tests for renderer/GpuThermalStepper
// ============================================================================
// The GPU stepper mirrors the CPU math in f32. What is tested here is that the
// mirror is accurate: closed forms at f32 tolerance, and a full diurnal run
// that matches the CPU stepper to within the f32 noise floor.
//
// Skipped on a machine with no GPU, like every other GPU test here.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/GpuThermalStepper.hpp"
#include "support/VulkanTestDevice.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalSolver.hpp"

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;
using namespace quantiloom::rendercore;

namespace {

Vector<ThermalElement> OneElement() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

ThermalState MakeState(usize elements, u32 nodes, f64 temperature_K) {
    ThermalState state;
    state.nodeCount = nodes;
    state.temperature_K.assign(elements * nodes, temperature_K);
    return state;
}

class ThermalStepGpuTest : public quantiloom::testing::VulkanDeviceTest {};

}  // namespace

TEST_F(ThermalStepGpuTest, ALumpedBodyCoolsExponentially) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    const auto elements = OneElement();
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

    ThermalForcing forcing;
    forcing.airTemperature_K = 280.0;
    forcing.sunIrradiance_W_m2 = 0.0;

    const f64 T0 = 350.0;
    const f64 tau = 8000.0 * 400.0 * 0.01 / 10.0;  // 3200 s
    const f64 dt = tau / 200.0;
    const i32 steps = 400;

    ThermalState state = MakeState(1, 12, T0);

    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = {1.0f};

    Vector<ThermalBatchStep> batch(steps);
    for (i32 i = 0; i < steps; ++i) {
        batch[i].forcing = forcing;
        batch[i].dt_s = dt;
    }

    gpuStepper.StepMany(state, elements, materials, exchange, sunTable, batch);

    const f64 elapsed = steps * dt;
    const f64 expected = 280.0 + (T0 - 280.0) * std::exp(-elapsed / tau);
    EXPECT_NEAR(state.Surface(0), expected, 0.5);
}

TEST_F(ThermalStepGpuTest, AHeldBackFaceProducesTheLinearSteadyProfile) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    ThermalMaterial material;
    material.conductivity_W_mK = 1.0f;
    material.density_kg_m3 = 2000.0f;
    material.specificHeat_J_kgK = 900.0f;
    material.thickness_m = 0.2f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.0f;
    material.interiorBoundary = InteriorBoundary::FixedTemperature;
    material.interiorTemperature_K = 293.15f;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 273.15;

    ThermalState state = MakeState(1, 21, 283.15);

    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = {1.0f};

    Vector<ThermalBatchStep> batch(4000);
    for (auto& b : batch) {
        b.forcing = forcing;
        b.dt_s = 60.0;
    }
    gpuStepper.StepMany(state, elements, materials, exchange, sunTable, batch);

    const f64 rCond = 0.2 / 1.0;
    const f64 rConv = 1.0 / 10.0;
    const f64 expected = 273.15 + (293.15 - 273.15) * rConv / (rConv + rCond);
    EXPECT_NEAR(state.Surface(0), expected, 0.05);
}

TEST_F(ThermalStepGpuTest, ASurfaceUnderAColdSkySettlesBelowTheAir) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    ThermalMaterial material;
    material.conductivity_W_mK = 400.0f;
    material.density_kg_m3 = 8000.0f;
    material.specificHeat_J_kgK = 400.0f;
    material.thickness_m = 0.01f;
    material.convection_W_m2K = 10.0f;
    material.shortwaveAbsorptivity = 0.0f;
    material.longwaveEmissivity = 0.95f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;

    const auto elements = OneElement();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 288.15;
    forcing.skyTemperature_K = 265.0;
    forcing.sunIrradiance_W_m2 = 0.0;

    ThermalState state = MakeState(1, 8, 288.15);
    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = {0.0f};

    Vector<ThermalBatchStep> batch(5000);
    for (auto& b : batch) {
        b.forcing = forcing;
        b.dt_s = 5.0;
    }
    gpuStepper.StepMany(state, elements, materials, exchange, sunTable, batch);

    EXPECT_LT(state.Surface(0), forcing.airTemperature_K);
}

TEST_F(ThermalStepGpuTest, ADiurnalRunMatchesTheCpuStepper) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    // Two elements with exchange so the radiative coupling path is exercised.
    Vector<ThermalElement> elements(2);
    for (auto& el : elements) {
        el.area_m2 = 1.0f;
        el.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        el.materialId = 0;
    }
    // Also add a non-participating element
    ThermalElement inert;
    inert.area_m2 = 1.0f;
    inert.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    inert.materialId = 1;
    elements.push_back(inert);

    ThermalMaterial concrete;
    concrete.conductivity_W_mK = 1.4f;
    concrete.density_kg_m3 = 2300.0f;
    concrete.specificHeat_J_kgK = 880.0f;
    concrete.thickness_m = 0.2f;
    concrete.convection_W_m2K = 10.0f;
    concrete.shortwaveAbsorptivity = 0.6f;
    concrete.longwaveEmissivity = 0.92f;

    ThermalMaterial inertMat;
    inertMat.conductivity_W_mK = 0.0f;

    const Vector<ThermalMaterial> materials{concrete, inertMat};

    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart = {0, 1, 2, 2};
    exchange.viewFactors.column = {1, 0};
    exchange.viewFactors.value = {0.3f, 0.3f};
    exchange.skyFraction = {0.7f, 0.7f, 1.0f};
    exchange.sunVisibility = {1.0f, 0.5f, 1.0f};

    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = exchange.sunVisibility;

    const u32 nodeCount = 10;
    const f64 dt = 60.0;
    const i32 steps = 1440;  // 24 hours
    const f64 T0 = 290.0;

    // CPU run
    ThermalState cpuState = MakeState(3, nodeCount, T0);
    CpuCrankNicolsonStepper cpuStepper;
    ThermalForcing forcing;
    forcing.airTemperature_K = 293.15;
    forcing.sunIrradiance_W_m2 = 900.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 268.0;

    for (i32 i = 0; i < steps; ++i) {
        cpuStepper.Step(cpuState, elements, materials, exchange, forcing, dt,
                        {exchange.sunVisibility});
    }

    // GPU run
    ThermalState gpuState = MakeState(3, nodeCount, T0);
    Vector<ThermalBatchStep> batch(steps);
    for (auto& b : batch) {
        b.forcing = forcing;
        b.dt_s = dt;
    }
    gpuStepper.StepMany(gpuState, elements, materials, exchange, sunTable, batch);

    // Compare participating elements
    f64 maxDiff = 0.0;
    for (usize e = 0; e < 2; ++e) {
        const f64 diff = std::abs(gpuState.Surface(e) - cpuState.Surface(e));
        maxDiff = std::max(maxDiff, diff);
    }
    EXPECT_LT(maxDiff, 0.05)
        << "GPU-CPU surface temperature mismatch after " << steps << " steps: "
        << maxDiff << " K";

    // Non-participating element must not have been touched by the GPU
    EXPECT_DOUBLE_EQ(gpuState.Surface(2), T0);
}

TEST_F(ThermalStepGpuTest, MaterialsWithNoConductivityAreNeverWritten) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    ThermalMaterial inert;
    inert.conductivity_W_mK = 0.0f;

    ThermalElement element;
    element.area_m2 = 1.0f;
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.materialId = 0;

    const Vector<ThermalElement> elements{element};
    const Vector<ThermalMaterial> materials{inert};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalForcing forcing;
    forcing.airTemperature_K = 400.0;

    ThermalState state = MakeState(1, 8, 300.0);

    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = {1.0f};

    Vector<ThermalBatchStep> batch(100);
    for (auto& b : batch) {
        b.forcing = forcing;
        b.dt_s = 60.0;
    }
    gpuStepper.StepMany(state, elements, materials, exchange, sunTable, batch);

    EXPECT_DOUBLE_EQ(state.Surface(0), 300.0);
}

TEST_F(ThermalStepGpuTest, MoreThanMaxNodesIsRefused) {
    GpuThermalStepper gpuStepper(Device());
    if (!gpuStepper.IsValid()) {
        GTEST_SKIP() << "thermal_step.spv unavailable";
    }

    const auto elements = OneElement();
    ThermalMaterial material;
    material.conductivity_W_mK = 1.0f;
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);

    ThermalState state = MakeState(1, GpuThermalStepper::kMaxNodes + 1, 300.0);
    SunVisibilityTable sunTable;
    sunTable.sampleTime_h = {0.0};
    sunTable.visibility = {1.0f};

    ThermalBatchStep bs;
    bs.forcing.airTemperature_K = 280.0;
    bs.dt_s = 60.0;

    // Should be a no-op rather than a crash
    gpuStepper.StepMany(state, elements, materials, exchange, sunTable, {&bs, 1});
    EXPECT_DOUBLE_EQ(state.Surface(0), 300.0);
}
