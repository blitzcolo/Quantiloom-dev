// ============================================================================
// Quantiloom - Unit Tests for piecewise-static thermal geometry
// ============================================================================
// The claim an epoch schedule makes is that ONE trajectory can cross a change
// of geometry: the ground a truck parked on cools while the ground it left
// warms back up, with the transient in between rather than a step. That claim
// has three parts, and each is a test here:
//
//   1. Before the boundary, an epoch schedule is bit-for-bit the single-epoch
//      answer. A scene that has not changed yet must not be affected by a
//      change that has not happened.
//   2. After it, the temperature leaves the first steady state and approaches
//      the second one -- monotonically, and eventually to within a fifth of a
//      kelvin of a solve that ran in the second geometry from the start. It
//      takes days: the transient is the point.
//   3. A batch is never stepped across a boundary, whatever the checkpoint
//      stride does.
//
// Plus the planner, which decides where the boundaries go, and which is the
// part with no physics in it at all.
// ============================================================================

#include <gtest/gtest.h>

#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalEpochs.hpp"
#include "thermal/ThermalSolver.hpp"
#include "thermal/ThermalTimeline.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

using namespace quantiloom;
using namespace quantiloom::thermal;

namespace {

Vector<ThermalElement> OnePlate() {
    ThermalElement element;
    element.centroid = glm::vec3(0.0f);
    element.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    element.area_m2 = 1.0f;
    element.materialId = 0;
    return {element};
}

ThermalMaterial Concrete() {
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

ThermalForcing SunnyForcing() {
    ThermalForcing forcing;
    forcing.airTemperature_K = 293.15;
    forcing.sunIrradiance_W_m2 = 900.0;
    forcing.sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    forcing.skyTemperature_K = 270.0;
    return forcing;
}

ThermalTimeline::Desc PlateDesc() {
    ThermalTimeline::Desc desc;
    desc.startTime_h = 0.0;
    desc.timestep_s = 60.0;
    desc.checkpointStride_h = 1.0;
    desc.nodeCount = 10;
    desc.initial = InitialCondition::Steady;
    desc.initialTemperature_K = 293.15;
    desc.carrySunSensitivity = false;
    return desc;
}

/// Open sky and full sun: a plate on a plain.
ThermalGeometryEpoch OpenEpoch(f64 from_h) {
    ThermalGeometryEpoch epoch;
    epoch.from_h = from_h;
    epoch.elements = OnePlate();
    epoch.exchange = MakeOpenSkyExchange(1);
    epoch.sunTable.sampleTime_h = {0.0};
    epoch.sunTable.visibility = {1.0f};
    epoch.sunTable.sampleDirection = {glm::vec3(0.0f, 1.0f, 0.0f)};
    return epoch;
}

/// Half the sky and no sun: a truck parked over it.
ThermalGeometryEpoch ShadedEpoch(f64 from_h) {
    ThermalGeometryEpoch epoch = OpenEpoch(from_h);
    epoch.exchange.skyFraction = {0.5f};
    epoch.exchange.sunVisibility = {0.0f};
    epoch.sunTable.visibility = {0.0f};
    return epoch;
}

/// Records which exchange each batch was stepped against, so a test can see
/// whether a batch ever spanned a boundary.
class CountingStepper : public IThermalStepper {
public:
    void Step(ThermalState& state, const Vector<ThermalElement>& elements,
              const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
              const ThermalForcing& forcing, f64 dt_s,
              const ShortwaveSample& shortwave) override {
        m_inner.Step(state, elements, materials, exchange, forcing, dt_s, shortwave);
    }

    void StepMany(ThermalState& state, const Vector<ThermalElement>& elements,
                  const Vector<ThermalMaterial>& materials, const ExchangeGeometry& exchange,
                  const SunVisibilityTable& sunTable,
                  std::span<const ThermalBatchStep> steps) override {
        batchExchange.push_back(&exchange);
        batchSize.push_back(steps.size());
        m_inner.StepMany(state, elements, materials, exchange, sunTable, steps);
    }

    [[nodiscard]] const char* Name() const override { return "counting"; }
    [[nodiscard]] bool CarriesShells() const override { return m_inner.CarriesShells(); }

    Vector<const ExchangeGeometry*> batchExchange;
    Vector<usize> batchSize;

private:
    CpuCrankNicolsonStepper m_inner;
};

}  // namespace

// ============================================================================
// The planner
// ============================================================================

TEST(ThermalEpochPlanTest, NothingMovingIsOneEpoch) {
    EpochPlanInput input;
    input.start_s = 0.0;
    input.end_s = 100.0;
    input.stride_s = 10.0;

    const Vector<f64> times = PlanEpochTimes(input);
    ASSERT_EQ(times.size(), 1u);
    EXPECT_DOUBLE_EQ(times[0], 0.0);
}

TEST(ThermalEpochPlanTest, AStandingObjectCostsNoBoundaries) {
    EpochPlanInput input;
    input.start_s = 0.0;
    input.end_s = 100.0;
    input.stride_s = 10.0;
    input.minMove_m = 0.05f;

    EpochPlanInput::Node node;
    node.boundRadius_m = 1.0f;
    node.poseAt = [](f64) { return glm::mat4(1.0f); };
    input.nodes.push_back(std::move(node));

    const Vector<f64> times = PlanEpochTimes(input);
    EXPECT_EQ(times.size(), 1u);
}

TEST(ThermalEpochPlanTest, TheStrideSamplesSomethingThatKeepsMoving) {
    EpochPlanInput input;
    input.start_s = 0.0;
    input.end_s = 10.0;
    input.stride_s = 1.0;
    input.minMove_m = 0.5f;

    EpochPlanInput::Node node;
    node.boundRadius_m = 1.0f;
    // One metre a second, so every stride is a metre and clears the threshold.
    node.poseAt = [](f64 t) {
        return glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<f32>(t), 0, 0));
    };
    input.nodes.push_back(std::move(node));

    const Vector<f64> times = PlanEpochTimes(input);
    EXPECT_EQ(times.size(), 11u);  // 0 s through 10 s
}

TEST(ThermalEpochPlanTest, TheMinimumMoveDropsCandidatesNothingHappenedAt) {
    EpochPlanInput input;
    input.start_s = 0.0;
    input.end_s = 10.0;
    input.stride_s = 1.0;
    input.minMove_m = 2.5f;

    EpochPlanInput::Node node;
    node.boundRadius_m = 1.0f;
    node.poseAt = [](f64 t) {
        return glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<f32>(t), 0, 0));
    };
    input.nodes.push_back(std::move(node));

    // Every 2.5 m, so at 3, 6 and 9 seconds -- the candidates in between moved
    // less than the threshold since the last kept one.
    const Vector<f64> times = PlanEpochTimes(input);
    ASSERT_EQ(times.size(), 4u);
    EXPECT_DOUBLE_EQ(times[1], 3.0);
    EXPECT_DOUBLE_EQ(times[2], 6.0);
    EXPECT_DOUBLE_EQ(times[3], 9.0);
}

TEST(ThermalEpochPlanTest, AKeyframeTimeIsACandidateWhateverTheStrideSays) {
    EpochPlanInput input;
    input.start_s = 0.0;
    input.end_s = 10.0;
    input.stride_s = 0.0;  // no periodic candidates at all
    input.minMove_m = 0.1f;

    EpochPlanInput::Node node;
    node.boundRadius_m = 1.0f;
    node.changeTimes = {4.0};
    node.poseAt = [](f64 t) {
        return glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<f32>(t), 0, 0));
    };
    input.nodes.push_back(std::move(node));

    const Vector<f64> times = PlanEpochTimes(input);
    ASSERT_EQ(times.size(), 2u);
    EXPECT_DOUBLE_EQ(times[1], 4.0);
}

// ============================================================================
// A trajectory across a change of geometry
// ============================================================================

TEST(ThermalEpochTest, BeforeTheBoundaryTheAnswerIsTheSingleEpochOne) {
    const Vector<ThermalMaterial> materials{Concrete()};
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    const ThermalForcing forcing = SunnyForcing();

    ThermalGeometrySchedule single;
    single.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));

    ThermalGeometrySchedule split;
    split.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));
    split.epochs.push_back(ShadedEpoch(6.0));

    CpuCrankNicolsonStepper stepperA;
    CpuCrankNicolsonStepper stepperB;
    ThermalTimeline one(PlateDesc(), single, materials, noSeries, forcing, stepperA);
    ThermalTimeline two(PlateDesc(), split, materials, noSeries, forcing, stepperB);

    for (const f64 hour : {0.5, 2.0, 4.0, 5.5}) {
        EXPECT_DOUBLE_EQ(one.StateAt(hour).Surface(0), two.StateAt(hour).Surface(0))
            << "at " << hour << " h";
    }
}

TEST(ThermalEpochTest, AfterTheBoundaryItApproachesTheOtherGeometrysAnswer) {
    const Vector<ThermalMaterial> materials{Concrete()};
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    const ThermalForcing forcing = SunnyForcing();

    ThermalGeometrySchedule split;
    split.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));
    split.epochs.push_back(ShadedEpoch(6.0));

    // The same shaded geometry, but from the very beginning: this is what the
    // split trajectory has to converge to once the transient has run out.
    ThermalGeometrySchedule shadedThroughout;
    shadedThroughout.epochs.push_back(ShadedEpoch(-std::numeric_limits<f64>::infinity()));

    CpuCrankNicolsonStepper stepperA;
    CpuCrankNicolsonStepper stepperB;
    ThermalTimeline moving(PlateDesc(), split, materials, noSeries, forcing, stepperA);
    ThermalTimeline shaded(PlateDesc(), shadedThroughout, materials, noSeries, forcing,
                           stepperB);

    const f64 sunny = moving.StateAt(5.9).Surface(0);
    const f64 justAfter = moving.StateAt(6.5).Surface(0);
    const f64 aDayLater = moving.StateAt(30.0).Surface(0);
    const f64 muchLater = moving.StateAt(246.0).Surface(0);
    const f64 settled = shaded.StateAt(246.0).Surface(0);

    // The shade is colder than the sun, and the plate leaves the first answer
    // for the second rather than jumping to it.
    EXPECT_LT(justAfter, sunny);
    EXPECT_LT(aDayLater, justAfter);
    EXPECT_LT(muchLater, aDayLater);

    // This is the whole point of the design, so it is worth saying what the
    // numbers are. A 0.2 m adiabatic-backed concrete slab has a time constant
    // of many hours: a day after the truck parked it is still some 4 K above
    // where it will end up, which is exactly the transient a frozen geometry
    // cannot produce and a per-frame re-solve throws away.
    EXPECT_GT(std::abs(aDayLater - settled), 1.0);
    EXPECT_NEAR(muchLater, settled, 0.2);
}

TEST(ThermalEpochTest, ABatchNeverSpansABoundary) {
    const Vector<ThermalMaterial> materials{Concrete()};
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    const ThermalForcing forcing = SunnyForcing();

    ThermalGeometrySchedule split;
    split.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));
    // Half past, so it lands inside a checkpoint stride rather than on one.
    split.epochs.push_back(ShadedEpoch(0.5));

    auto desc = PlateDesc();
    desc.checkpointStride_h = 1.0;
    desc.initial = InitialCondition::Uniform;  // no relaxation to confuse the count

    CountingStepper stepper;
    ThermalTimeline timeline(desc, split, materials, noSeries, forcing, stepper);
    timeline.StateAt(2.0);

    ASSERT_FALSE(stepper.batchExchange.empty());
    const ExchangeGeometry* first = &split.epochs[0].exchange;
    const ExchangeGeometry* second = &split.epochs[1].exchange;

    // Every batch is one epoch's, and the first geometry is used before the
    // second -- never after it.
    bool seenSecond = false;
    for (const ExchangeGeometry* used : stepper.batchExchange) {
        ASSERT_TRUE(used == first || used == second);
        if (used == second) seenSecond = true;
        EXPECT_FALSE(seenSecond && used == first) << "an epoch came back after it ended";
    }
    EXPECT_TRUE(seenSecond);

    // The first epoch is exactly 30 steps of 60 s: half an hour.
    usize firstEpochSteps = 0;
    for (usize b = 0; b < stepper.batchExchange.size(); ++b) {
        if (stepper.batchExchange[b] == first) firstEpochSteps += stepper.batchSize[b];
    }
    EXPECT_EQ(firstEpochSteps, 30u);
}

TEST(ThermalEpochTest, TheScheduleFindsTheEpochAnHourBelongsTo) {
    ThermalGeometrySchedule schedule;
    schedule.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));
    schedule.epochs.push_back(ShadedEpoch(6.0));
    schedule.epochs.push_back(OpenEpoch(12.0));

    EXPECT_EQ(schedule.EpochAt(-100.0), 0u);
    EXPECT_EQ(schedule.EpochAt(0.0), 0u);
    EXPECT_EQ(schedule.EpochAt(5.999), 0u);
    EXPECT_EQ(schedule.EpochAt(6.0), 1u);
    EXPECT_EQ(schedule.EpochAt(11.9), 1u);
    EXPECT_EQ(schedule.EpochAt(12.0), 2u);
    EXPECT_EQ(schedule.EpochAt(1e6), 2u);
}

TEST(ThermalEpochTest, TheSingleGeometryConstructorStillWorks) {
    // The overload every existing caller uses, over an owned single-epoch
    // schedule. It has to give the same trajectory as the schedule form.
    const auto elements = OnePlate();
    const Vector<ThermalMaterial> materials{Concrete()};
    const Vector<std::pair<f64, ThermalForcing>> noSeries;
    const ThermalForcing forcing = SunnyForcing();
    const ExchangeGeometry exchange = MakeOpenSkyExchange(1);

    SunVisibilityTable table;
    table.sampleTime_h = {0.0};
    table.visibility = {1.0f};
    table.sampleDirection = {glm::vec3(0.0f, 1.0f, 0.0f)};

    CpuCrankNicolsonStepper stepperA;
    CpuCrankNicolsonStepper stepperB;
    ThermalTimeline legacy(PlateDesc(), elements, materials, exchange, table, noSeries,
                           forcing, stepperA);

    ThermalGeometrySchedule schedule;
    schedule.epochs.push_back(OpenEpoch(-std::numeric_limits<f64>::infinity()));
    ThermalTimeline modern(PlateDesc(), schedule, materials, noSeries, forcing, stepperB);

    EXPECT_EQ(legacy.EpochCount(), 1u);
    EXPECT_DOUBLE_EQ(legacy.StateAt(8.0).Surface(0), modern.StateAt(8.0).Surface(0));
}
