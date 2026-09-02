// ============================================================================
// Quantiloom - the trajectory's tangents in the material's own parameters
// ============================================================================
// A surface temperature is a function of the material it is computed for, and
// two things want to know how strongly. A fit does: h against a station's
// record is a one-parameter Gauss-Newton, and a Gauss-Newton without a
// derivative is a parameter sweep. An uncertainty budget does: what a 0.02
// uncertainty in emissivity is worth in kelvin is dT/deps times 0.02, and
// nothing else answers it.
//
// The derivative has to be the trajectory's, not a formula's, for the same
// reason dT/dv does -- the steady response and the single-step response are
// both wrong by a factor for anything with thermal inertia. So each of these
// is checked against a centred finite difference of two whole runs, which is
// the definition evaluated the expensive way.
//
// Two of the five move the matrix rather than the flux: k sets the conduction
// between nodes and rho c sets the capacity, so their rows carry
// -(dA/dp) T^{n+1} and can only be built after the temperature is solved.
// Those two are the ones this file would catch an error in.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> OneElementFacingUp() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

ThermalMaterial Sand() {
    ThermalMaterial material;
    material.conductivity_W_mK = 0.3f;
    material.density_kg_m3 = 1600.0f;
    material.specificHeat_J_kgK = 800.0f;
    material.thickness_m = 0.15f;
    material.convection_W_m2K = 14.0f;
    material.shortwaveAbsorptivity = 0.72f;
    material.longwaveEmissivity = 0.9f;
    material.interiorBoundary = InteriorBoundary::Adiabatic;
    return material;
}

ThermalForcing SunAtNoon() {
    ThermalForcing forcing;
    forcing.airTemperature_K = 312.0;
    forcing.sunIrradiance_W_m2 = 980.0;
    forcing.diffuseIrradiance_W_m2 = 90.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 288.0;
    return forcing;
}

constexpr usize kSteps = 180;   // three hours at a minute a step
constexpr u32 kNodes = 8;

/// The surface after the run, and the carried tangent in @p parameter when one
/// was asked for.
struct Run {
    f64 surface_K = 0.0;
    f64 sensitivity = 0.0;
};

Run Trajectory(const ThermalMaterial& material,
               const Vector<ThermalParameter>& parameters = {}) {
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{material};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> visibility{0.6f};

    ThermalState state;
    state.nodeCount = kNodes;
    state.temperature_K.assign(kNodes, 300.0);
    if (!parameters.empty()) {
        state.parameters = parameters;
        state.parameterSensitivity.assign(kNodes * parameters.size(), 0.0);
    }

    CpuCrankNicolsonStepper stepper;
    for (usize i = 0; i < kSteps; ++i) {
        stepper.Step(state, elements, materials, exchange, SunAtNoon(), 60.0, {visibility});
    }

    Run run;
    run.surface_K = state.Surface(0);
    if (!parameters.empty()) run.sensitivity = state.SurfaceParameterSensitivity(0, 0);
    return run;
}

/// The parameter's value on a material, and a copy with it moved.
struct Knob {
    f64 (*read)(const ThermalMaterial&);
    void (*write)(ThermalMaterial&, f64);
};

const Knob& KnobFor(const ThermalParameter parameter) {
    static const Knob convection{
        [](const ThermalMaterial& m) { return static_cast<f64>(m.convection_W_m2K); },
        [](ThermalMaterial& m, const f64 v) { m.convection_W_m2K = static_cast<f32>(v); }};
    static const Knob emissivity{
        [](const ThermalMaterial& m) { return static_cast<f64>(m.longwaveEmissivity); },
        [](ThermalMaterial& m, const f64 v) { m.longwaveEmissivity = static_cast<f32>(v); }};
    static const Knob absorptivity{
        [](const ThermalMaterial& m) { return static_cast<f64>(m.shortwaveAbsorptivity); },
        [](ThermalMaterial& m, const f64 v) { m.shortwaveAbsorptivity = static_cast<f32>(v); }};
    static const Knob conductivity{
        [](const ThermalMaterial& m) { return static_cast<f64>(m.conductivity_W_mK); },
        [](ThermalMaterial& m, const f64 v) { m.conductivity_W_mK = static_cast<f32>(v); }};
    // rho c as the balance uses it, moved through the density so the pair is
    // one number here as it is there.
    static const Knob capacity{
        [](const ThermalMaterial& m) {
            return static_cast<f64>(m.density_kg_m3) * m.specificHeat_J_kgK;
        },
        [](ThermalMaterial& m, const f64 v) {
            m.density_kg_m3 = static_cast<f32>(v / m.specificHeat_J_kgK);
        }};

    switch (parameter) {
        case ThermalParameter::Convection:   return convection;
        case ThermalParameter::Emissivity:   return emissivity;
        case ThermalParameter::Absorptivity: return absorptivity;
        case ThermalParameter::Conductivity: return conductivity;
        default:                             return capacity;
    }
}

}  // namespace

TEST(ThermalParameterSensitivityTest, EachIsAFiniteDifferenceOfTwoRuns) {
    constexpr f64 kRelative = 1e-3;

    for (u8 i = 0; i < static_cast<u8>(ThermalParameter::Count); ++i) {
        const auto parameter = static_cast<ThermalParameter>(i);
        const Knob& knob = KnobFor(parameter);
        const ThermalMaterial base = Sand();
        const f64 value = knob.read(base);

        ThermalMaterial plus = base;
        ThermalMaterial minus = base;
        knob.write(plus, value * (1.0 + kRelative));
        knob.write(minus, value * (1.0 - kRelative));

        // The step actually taken, read back off the material: these are f32
        // fields, and rho c is moved through a division.
        const f64 step = 0.5 * (knob.read(plus) - knob.read(minus));
        ASSERT_GT(step, 0.0) << ThermalParameterName(parameter);

        const f64 difference =
            (Trajectory(plus).surface_K - Trajectory(minus).surface_K) / (2.0 * step);
        const f64 carried = Trajectory(base, {parameter}).sensitivity;

        EXPECT_NEAR(carried, difference, std::max(std::abs(difference) * 5e-3, 1e-9))
            << ThermalParameterName(parameter) << ": carried " << carried
            << ", finite difference " << difference;
    }
}

TEST(ThermalParameterSensitivityTest, TheyHaveTheSignsThePhysicsAsksFor) {
    // Cheap, and it catches a sign error that a finite difference would
    // reproduce faithfully if the two runs were swapped.
    const ThermalMaterial sand = Sand();

    // A sunlit surface above the air: more exchange with the air cools it,
    // more emissivity cools it under a cold sky, more absorbed sun warms it.
    EXPECT_LT(Trajectory(sand, {ThermalParameter::Convection}).sensitivity, 0.0);
    EXPECT_LT(Trajectory(sand, {ThermalParameter::Emissivity}).sensitivity, 0.0);
    EXPECT_GT(Trajectory(sand, {ThermalParameter::Absorptivity}).sensitivity, 0.0);
    // A heavier slab warms more slowly, so three hours in it is cooler.
    EXPECT_LT(Trajectory(sand, {ThermalParameter::HeatCapacity}).sensitivity, 0.0);
}

TEST(ThermalParameterSensitivityTest, CarryingThemChangesNoTemperature) {
    const ThermalMaterial sand = Sand();
    const f64 bare = Trajectory(sand).surface_K;
    EXPECT_DOUBLE_EQ(Trajectory(sand, {ThermalParameter::Convection}).surface_K, bare);
    EXPECT_DOUBLE_EQ(
        Trajectory(sand, {ThermalParameter::Conductivity, ThermalParameter::HeatCapacity,
                          ThermalParameter::Emissivity})
            .surface_K,
        bare);
}

TEST(ThermalParameterSensitivityTest, TheConvectionKnobIsInertWhereSomethingElseSetsH) {
    // A forcing column or a wind law is what h IS then, and the material's own
    // number never reaches the balance -- so its derivative is zero rather
    // than the derivative of a number nobody used.
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Sand()};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> visibility{0.6f};

    ThermalForcing stated = SunAtNoon();
    stated.convection_W_m2K = 25.0;

    ThermalState state;
    state.nodeCount = kNodes;
    state.temperature_K.assign(kNodes, 300.0);
    state.parameters = {ThermalParameter::Convection};
    state.parameterSensitivity.assign(kNodes, 0.0);

    CpuCrankNicolsonStepper stepper;
    for (usize i = 0; i < kSteps; ++i) {
        stepper.Step(state, elements, materials, exchange, stated, 60.0, {visibility});
    }
    EXPECT_DOUBLE_EQ(state.SurfaceParameterSensitivity(0, 0), 0.0);
}
