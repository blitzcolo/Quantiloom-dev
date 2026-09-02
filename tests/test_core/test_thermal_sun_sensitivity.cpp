// ============================================================================
// Quantiloom - Unit Tests for the trajectory's sun tangent
// ============================================================================
// The solver runs on one element per triangle, so the shadow it computes can
// only have edges where the mesh has them. The fix is not to refine the mesh
// but to ship, beside each temperature, how far that temperature would move
// per unit of the element's own sun visibility -- and let the shading pass
// trace the shadow at its own resolution.
//
// That number has to be the derivative of the trajectory, not of a formula.
// The steady response overshoots badly for anything with thermal inertia and
// the one-step response undershoots just as badly, so what these tests check
// is that the carried tangent agrees with a finite difference of two whole
// runs -- which is the definition, evaluated the expensive way.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"

#include "renderer/ThermalSunResponse.hpp"
#include "thermal/ThermalSolver.hpp"  // MakeOpenSkyExchange

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

constexpr f64 kStefanBoltzmann = 5.670374419e-8;

Vector<ThermalElement> OneElementFacingUp() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

/// Dry desert sand, as `env_desert.toml` describes it: the case the shadow
/// artefact was found in, and a slab thick enough that its response to the sun
/// takes hours rather than minutes.
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
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 288.0;
    return forcing;
}

ThermalState MakeState(const usize elements, const u32 nodes, const f64 temperature_K,
                       const bool carryTangent) {
    ThermalState state;
    state.nodeCount = nodes;
    state.temperature_K.assign(elements * nodes, temperature_K);
    if (carryTangent) {
        state.sunSensitivity_K.assign(elements * nodes, 0.0);
    }
    return state;
}

/// Run @p steps of @p dt with a fixed sun visibility, returning the final
/// surface temperature and, if one was carried, the tangent beside it.
struct RunResult {
    f64 surface_K = 0.0;
    f64 sensitivity_K = 0.0;
};

RunResult RunTrajectory(const f32 visibility, const usize steps, const f64 dt_s,
              const bool carryTangent, const u32 nodes = 8) {
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Sand()};
    const auto exchange = MakeOpenSkyExchange(1);
    const ThermalForcing forcing = SunAtNoon();
    const Vector<f32> sunVisibility{visibility};

    CpuCrankNicolsonStepper stepper;
    ThermalState state = MakeState(1, nodes, 300.0, carryTangent);
    for (usize i = 0; i < steps; ++i) {
        stepper.Step(state, elements, materials, exchange, forcing, dt_s, {sunVisibility});
    }

    RunResult result;
    result.surface_K = state.Surface(0);
    if (carryTangent) result.sensitivity_K = state.SurfaceSensitivity(0);
    return result;
}

}  // namespace

// ============================================================================
// It is the derivative of the trajectory
// ============================================================================

TEST(ThermalSunSensitivityTest, MatchesAFiniteDifferenceOfTwoRuns) {
    // dT/dv against (T(v + delta) - T(v - delta)) / 2 delta, centred so the
    // comparison is second-order in delta and the residual is the balance's
    // own curvature rather than the difference stencil's.
    constexpr usize kSteps = 180;   // three hours at a minute a step
    constexpr f64 kDelta = 1e-3;

    const RunResult carried = RunTrajectory(0.5f, kSteps, 60.0, true);
    const f64 plus = RunTrajectory(0.5f + static_cast<f32>(kDelta), kSteps, 60.0, false).surface_K;
    const f64 minus = RunTrajectory(0.5f - static_cast<f32>(kDelta), kSteps, 60.0, false).surface_K;

    const f64 finiteDifference = (plus - minus) / (2.0 * kDelta);
    EXPECT_NEAR(carried.sensitivity_K, finiteDifference,
                std::abs(finiteDifference) * 2e-3)
        << "carried " << carried.sensitivity_K << " K, finite difference "
        << finiteDifference << " K";
}

TEST(ThermalSunSensitivityTest, PredictsAFullShadowToWithinAKelvin) {
    // The use the renderer puts it to: a pixel the shadow ray finds occluded
    // gets T_element + (0 - v) dT/dv. Against the trajectory that ran fully
    // shadowed, that extrapolation has only the balance's curvature to be
    // wrong by -- worth checking at the amplitude it is actually used at
    // rather than at a differential one.
    constexpr usize kSteps = 180;

    const RunResult lit = RunTrajectory(1.0f, kSteps, 60.0, true);
    const f64 shadowed = RunTrajectory(0.0f, kSteps, 60.0, false).surface_K;

    const f64 extrapolated = lit.surface_K + (0.0 - 1.0) * lit.sensitivity_K;
    // 1.0 K on a 28.9 K contrast, and this is the worst case the renderer can
    // ask for: v(x) - v_element is at most 1, and over that span the radiative
    // admittance 4 eps sigma T^3 changes by a third, which is exactly what a
    // first-order extrapolation cannot follow. Against the alternative -- a
    // 0.6 m triangle either fully lit or fully dark -- it is not close.
    EXPECT_NEAR(extrapolated, shadowed, 1.5)
        << "extrapolated " << extrapolated << " K against " << shadowed << " K";
    // And it is a real thermal contrast, not a rounding error -- otherwise the
    // agreement above would be trivially satisfied.
    EXPECT_GT(lit.surface_K - shadowed, 10.0);
}

// ============================================================================
// It is not either of the two formulas it is easy to mistake it for
// ============================================================================

TEST(ThermalSunSensitivityTest, LiesBetweenTheSteadyAndTheSingleStepResponse) {
    // The steady response, alpha E cos / (h + 4 eps sigma T^3), is what a
    // surface with no heat capacity would do; the single-step response is what
    // one with infinite capacity would do in one step. Sand at three hours is
    // strictly between them, and by enough that neither would have served.
    constexpr usize kSteps = 180;
    constexpr f64 dt = 60.0;

    const RunResult carried = RunTrajectory(1.0f, kSteps, dt, true);
    const ThermalMaterial sand = Sand();

    const f64 drive = static_cast<f64>(sand.shortwaveAbsorptivity) * 980.0;  // cos = 1
    const f64 T = carried.surface_K;
    const f64 steady = drive / (sand.convection_W_m2K +
                                4.0 * sand.longwaveEmissivity * kStefanBoltzmann * T * T * T);

    const f64 dx = static_cast<f64>(sand.thickness_m) / 7.0;  // nodes - 1
    const f64 rhoC = static_cast<f64>(sand.density_kg_m3) * sand.specificHeat_J_kgK;
    const f64 oneStep = drive / (rhoC * dx / (2.0 * dt) +
                                 0.5 * (sand.conductivity_W_mK / dx + sand.convection_W_m2K));

    EXPECT_LT(carried.sensitivity_K, steady * 0.9)
        << "a steady-state coefficient would have overstated the shadow";
    EXPECT_GT(carried.sensitivity_K, oneStep * 1.5)
        << "a single-step coefficient would have understated it";
}

TEST(ThermalSunSensitivityTest, GrowsWithHowLongTheSunHasBeenUp) {
    // The property that makes it a state: the same forcing gives a larger
    // response the longer it has been applied, up to the steady value.
    // Monotone, and it takes most of a day to arrive: 2.9 K after a minute,
    // 25.1 K after an hour, 30.3 K after twelve. A coefficient computed from
    // the instantaneous balance would have to pick one of those and be wrong
    // at every other hour.
    f64 previous = 0.0;
    for (const usize steps : {1u, 5u, 15u, 30u, 60u, 120u, 360u, 720u}) {
        const f64 sensitivity = RunTrajectory(1.0f, steps, 60.0, true).sensitivity_K;
        EXPECT_GT(sensitivity, previous) << "at " << steps << " steps";
        previous = sensitivity;
    }
    EXPECT_GT(previous, 30.0);
    EXPECT_LT(RunTrajectory(1.0f, 1, 60.0, true).sensitivity_K, 0.2 * previous);
}

// ============================================================================
// The contract around it
// ============================================================================

TEST(ThermalSunSensitivityTest, CarryingItChangesNoTemperature) {
    // The tangent shares the matrix but never feeds back into the state. A
    // caller that turns it on must get the same trajectory to the bit, or
    // every physics gate in the suite would be reading a different solver.
    const RunResult without = RunTrajectory(0.5f, 200, 60.0, false);
    const RunResult with = RunTrajectory(0.5f, 200, 60.0, true);
    EXPECT_DOUBLE_EQ(with.surface_K, without.surface_K);
}

TEST(ThermalSunSensitivityTest, AnUnsizedTangentIsLeftAlone) {
    // Empty means "nobody asked", and a stepper must not resize the state into
    // wanting one -- the timeline's checkpoints are copies of it.
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Sand()};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> sunVisibility{1.0f};

    CpuCrankNicolsonStepper stepper;
    ThermalState state = MakeState(1, 8, 300.0, false);
    stepper.Step(state, elements, materials, exchange, SunAtNoon(), 60.0, {sunVisibility});

    EXPECT_TRUE(state.sunSensitivity_K.empty());
    EXPECT_FALSE(state.HasSensitivity());
}

TEST(ThermalSunSensitivityTest, DecaysBackTowardZeroAfterSunset) {
    // Nothing pumps it once the sun is down, so it relaxes on the surface's
    // own time constant -- which is what keeps a night render from carrying a
    // correction the geometry can no longer justify.
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Sand()};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> sunVisibility{1.0f};

    CpuCrankNicolsonStepper stepper;
    ThermalState state = MakeState(1, 8, 300.0, true);

    ThermalForcing day = SunAtNoon();
    for (usize i = 0; i < 360; ++i) {
        stepper.Step(state, elements, materials, exchange, day, 60.0, {sunVisibility});
    }
    const f64 atDusk = state.SurfaceSensitivity(0);
    ASSERT_GT(atDusk, 5.0);

    ThermalForcing night = day;
    night.sunIrradiance_W_m2 = 0.0;
    night.airTemperature_K = 293.0;
    for (usize i = 0; i < 720; ++i) {
        stepper.Step(state, elements, materials, exchange, night, 60.0, {sunVisibility});
    }
    EXPECT_LT(state.SurfaceSensitivity(0), atDusk * 0.5);
    EXPECT_GT(state.SurfaceSensitivity(0), 0.0) << "it decays, it does not flip sign";
}

TEST(ThermalSunSensitivityTest, AHeldInteriorDoesNotRespondThroughTheWall) {
    // A Dirichlet back face is a room at its own temperature, and a room does
    // not warm because the sun came out. The tangent's last node must stay at
    // zero even while its surface node climbs.
    const auto elements = OneElementFacingUp();
    ThermalMaterial wall = Sand();
    wall.interiorBoundary = InteriorBoundary::FixedTemperature;
    wall.interiorTemperature_K = 293.15f;
    const Vector<ThermalMaterial> materials{wall};
    const auto exchange = MakeOpenSkyExchange(1);
    const Vector<f32> sunVisibility{1.0f};

    CpuCrankNicolsonStepper stepper;
    ThermalState state = MakeState(1, 8, 300.0, true);
    for (usize i = 0; i < 300; ++i) {
        stepper.Step(state, elements, materials, exchange, SunAtNoon(), 60.0, {sunVisibility});
    }

    EXPECT_GT(state.SurfaceSensitivity(0), 1.0);
    EXPECT_DOUBLE_EQ(state.sunSensitivity_K[7], 0.0);
}

// ============================================================================
// The buffer the shader reads
// ============================================================================

TEST(ThermalSunResponseTest, PacksAHeaderThenOneRecordPerElement) {
    const Vector<f32> sensitivity{12.0f, -3.0f, 0.0f};
    const Vector<f32> visibility{1.0f, 0.25f, 0.0f};

    const auto records = rendercore::MakeThermalSunResponse(sensitivity, visibility,
                                                            glm::vec3(0.0f, 3.0f, 0.0f));
    ASSERT_EQ(records.size(), 4u);

    // Header: the direction normalised, and the flag that switches the shader's
    // whole correction on.
    EXPECT_FLOAT_EQ(records[0].a.y, 1.0f);
    EXPECT_FLOAT_EQ(records[0].b, 1.0f);

    // Elements start at 1, so the shader's index is 1 + base + PrimitiveIndex().
    EXPECT_FLOAT_EQ(records[1].a.x, 12.0f);
    EXPECT_FLOAT_EQ(records[1].a.y, 1.0f);
    EXPECT_FLOAT_EQ(records[2].a.x, -3.0f);
    EXPECT_FLOAT_EQ(records[2].a.y, 0.25f);
}

TEST(ThermalSunResponseTest, AnInconsistentSolveBindsOneInertRecord) {
    // The descriptor has to be valid whether or not a solve ran, and the w = 0
    // header is what tells the shader to leave the temperature alone.
    for (const auto& records : {
             rendercore::MakeThermalSunResponse({}, {}, glm::vec3(0.0f, 1.0f, 0.0f)),
             rendercore::MakeThermalSunResponse({1.0f, 2.0f}, {1.0f},
                                                glm::vec3(0.0f, 1.0f, 0.0f)),
             rendercore::MakeThermalSunResponse({1.0f}, {1.0f}, glm::vec3(0.0f)),
         }) {
        ASSERT_EQ(records.size(), 1u);
        EXPECT_FLOAT_EQ(records[0].b, 0.0f);
    }
}

// ============================================================================
// One tangent per hour of the shadow's history
// ============================================================================
// The tangent above answers "what if this element had seen more sun, all day",
// and a shading pass applies it with the shadow it traces NOW. Under a still
// sun that is exact. Under a moving one it is an assumption: a pixel shaded at
// noon may have been lit at ten, and the ground under it is still warm.
//
// So the state can also carry a tangent per recent sun column. What these
// check is that each is the derivative of the trajectory with respect to THAT
// column's visibility, that together they partition the whole-day tangent
// rather than adding to it, and that a window too small to hold every column
// loses resolution rather than energy.
// ============================================================================

namespace {

struct LagRun {
    f64 surface_K = 0.0;
    f64 total_K = 0.0;
    Vector<f64> lag_K;
    Vector<u32> lagColumn;
};

/// A trajectory over @p visibility.size() sun columns an hour apart, stepped a
/// minute at a time through the whole span, with @p slots per-column tangents.
LagRun RunColumns(const Vector<f32>& visibility, const u32 slots,
                  const bool carryTangent = true) {
    const usize columns = visibility.size();
    const auto elements = OneElementFacingUp();
    const Vector<ThermalMaterial> materials{Sand()};
    const auto exchange = MakeOpenSkyExchange(1);

    SunVisibilityTable table;
    table.sampleTime_h.resize(columns);
    for (usize k = 0; k < columns; ++k) table.sampleTime_h[k] = static_cast<f64>(k);
    table.visibility = visibility;  // one element, so sample-major is column order
    table.sampleDirection.assign(columns, glm::vec3(0.0f, 1.0f, 0.0f));

    ThermalState state = MakeState(1, 8, 300.0, carryTangent);
    if (carryTangent && slots > 0) {
        state.lagColumn.assign(slots, ThermalState::kNoLagColumn);
        state.lagSensitivity_K.assign(state.temperature_K.size() * slots, 0.0);
    }

    const f64 span_h = static_cast<f64>(columns - 1);
    const usize steps = static_cast<usize>(span_h * 60.0);
    Vector<ThermalBatchStep> batch;
    batch.reserve(steps);
    for (usize i = 0; i < steps; ++i) {
        const f64 t_mid = (static_cast<f64>(i) + 0.5) / 60.0;
        ThermalBatchStep step;
        step.forcing = SunAtNoon();
        step.dt_s = 60.0;
        table.SampleIndices(t_mid, step.sunSampleA, step.sunSampleB, step.sunBlend);
        batch.push_back(step);
    }

    CpuCrankNicolsonStepper stepper;
    stepper.StepMany(state, elements, materials, exchange, table, batch);

    LagRun run;
    run.surface_K = state.Surface(0);
    if (carryTangent) run.total_K = state.SurfaceSensitivity(0);
    for (u32 s = 0; s < state.LagSlots(); ++s) {
        run.lag_K.push_back(state.SurfaceLagSensitivity(s, 0));
        run.lagColumn.push_back(state.lagColumn[s]);
    }
    return run;
}

}  // namespace

TEST(ThermalSunMemoryTest, EachColumnsTangentIsAFiniteDifferenceInThatColumn) {
    // Perturb ONE column's visibility and nothing else. What moves is that
    // column's share of the day, and the tangent carried for it is what says
    // by how much.
    constexpr f64 kDelta = 1e-3;
    const Vector<f32> base{0.5f, 0.5f, 0.5f, 0.5f};

    const LagRun carried = RunColumns(base, 4);
    ASSERT_EQ(carried.lag_K.size(), 4u);

    for (u32 column = 0; column < base.size(); ++column) {
        Vector<f32> plus = base;
        Vector<f32> minus = base;
        plus[column] += static_cast<f32>(kDelta);
        minus[column] -= static_cast<f32>(kDelta);

        const f64 difference =
            (RunColumns(plus, 0, false).surface_K - RunColumns(minus, 0, false).surface_K) /
            (2.0 * kDelta);

        // Which slot holds this column: the window is large enough here that
        // every one of them is in it.
        i32 slot = -1;
        for (u32 s = 0; s < carried.lagColumn.size(); ++s) {
            if (carried.lagColumn[s] == column) slot = static_cast<i32>(s);
        }
        ASSERT_GE(slot, 0) << "column " << column << " is not tracked";

        EXPECT_NEAR(carried.lag_K[static_cast<usize>(slot)], difference,
                    std::max(std::abs(difference) * 5e-3, 1e-4))
            << "column " << column << ": carried "
            << carried.lag_K[static_cast<usize>(slot)] << " K, finite difference "
            << difference << " K";
    }
}

TEST(ThermalSunMemoryTest, TheColumnsPartitionTheWholeDayTangent) {
    // They are a decomposition of the same answer, not an addition to it: with
    // a slot for every column, what the slots hold sums to the whole-day
    // tangent and the shading pass has nothing left over to apply against the
    // present sun.
    const LagRun run = RunColumns({0.4f, 0.9f, 0.2f, 0.7f}, 4);
    ASSERT_EQ(run.lag_K.size(), 4u);

    f64 sum = 0.0;
    for (const f64 lag : run.lag_K) sum += lag;
    EXPECT_NEAR(sum, run.total_K, std::abs(run.total_K) * 1e-9)
        << "sum of columns " << sum << " K against the whole day's "
        << run.total_K << " K";
}

TEST(ThermalSunMemoryTest, AWindowTooSmallLosesResolutionRatherThanEnergy) {
    // One slot can only track the newest column. The rest of the day is still
    // in the whole-day tangent, which is what the shading pass applies against
    // the present sun -- so nothing is lost except the ability to trace those
    // hours where they actually were.
    const Vector<f32> visibility{0.4f, 0.9f, 0.2f, 0.7f};
    const LagRun narrow = RunColumns(visibility, 1);
    const LagRun wide = RunColumns(visibility, 4);

    ASSERT_EQ(narrow.lag_K.size(), 1u);
    EXPECT_EQ(narrow.lagColumn[0], 3u) << "the newest column is the one worth tracing";
    EXPECT_NEAR(narrow.total_K, wide.total_K, std::abs(wide.total_K) * 1e-9)
        << "the whole-day tangent does not depend on how it was decomposed";
    EXPECT_LT(std::abs(narrow.lag_K[0]), std::abs(narrow.total_K))
        << "one column is a part of the day, not the whole of it";
}

TEST(ThermalSunMemoryTest, CarryingThemChangesNoTemperature) {
    const Vector<f32> visibility{0.4f, 0.9f, 0.2f, 0.7f};
    const f64 bare = RunColumns(visibility, 0).surface_K;
    for (const u32 slots : {1u, 2u, 4u}) {
        EXPECT_DOUBLE_EQ(RunColumns(visibility, slots).surface_K, bare)
            << slots << " slots";
    }
}

TEST(ThermalSunResponseTest, ColumnsRideBesideThePresentSunInOneBuffer) {
    // Stride 1 + M, the past sun directions first, then 1 + M records per
    // element with the present sun's remainder at the front.
    const Vector<f32> sensitivity{12.0f, -3.0f};
    const Vector<f32> visibility{1.0f, 0.25f};
    const Vector<f32> lagSensitivity{4.0f, -1.0f, 2.0f, 0.5f};  // slot-major
    const Vector<f32> lagVisibility{0.8f, 0.1f, 0.6f, 0.2f};
    const Vector<glm::vec3> lagDirection{glm::vec3(1.0f, 0.0f, 0.0f),
                                         glm::vec3(0.0f, 0.0f, 2.0f)};

    const auto records = rendercore::MakeThermalSunResponse(
        sensitivity, visibility, glm::vec3(0.0f, 1.0f, 0.0f), lagSensitivity,
        lagVisibility, lagDirection);

    ASSERT_EQ(records.size(), 3u * 3u);
    EXPECT_FLOAT_EQ(records[0].b, 3.0f) << "1 + two columns";
    EXPECT_FLOAT_EQ(records[1].a.x, 1.0f);
    EXPECT_FLOAT_EQ(records[2].a.z, 1.0f) << "directions are normalised";

    // Element 0 at stride * 1: the remainder, then each column.
    EXPECT_FLOAT_EQ(records[3].a.x, 12.0f - 4.0f - 2.0f);
    EXPECT_FLOAT_EQ(records[3].a.y, 1.0f);
    EXPECT_FLOAT_EQ(records[4].a.x, 4.0f);
    EXPECT_FLOAT_EQ(records[4].a.y, 0.8f);
    EXPECT_FLOAT_EQ(records[5].a.x, 2.0f);
    EXPECT_FLOAT_EQ(records[5].a.y, 0.6f);

    // Element 1 at stride * 2.
    EXPECT_FLOAT_EQ(records[6].a.x, -3.0f - (-1.0f) - 0.5f);
    EXPECT_FLOAT_EQ(records[7].a.x, -1.0f);
    EXPECT_FLOAT_EQ(records[8].a.x, 0.5f);
}

TEST(ThermalSunResponseTest, ASlotWithNoSunPositionIsDropped) {
    // Nothing for the shader to trace toward, so its share stays where it was:
    // in the present-sun record, which is where it lived before columns did.
    const auto records = rendercore::MakeThermalSunResponse(
        {10.0f}, {0.5f}, glm::vec3(0.0f, 1.0f, 0.0f), {4.0f}, {0.8f},
        {glm::vec3(0.0f)});

    ASSERT_EQ(records.size(), 2u);
    EXPECT_FLOAT_EQ(records[0].b, 1.0f);
    EXPECT_FLOAT_EQ(records[1].a.x, 10.0f) << "the whole of it, undivided";
}
