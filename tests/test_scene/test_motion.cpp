// ============================================================================
// Quantiloom - Unit Tests for scene/Motion.hpp and scene/MotionSpec.hpp
// ============================================================================
// Tests cover:
// - Keyframe interpolation: step, linear, cubic, and the loop wrap
// - Piecewise expressions, including the segment-local variable `s`
// - The parametric engines: straight line, circle, spin, along_velocity,
//   look_at
// - ChangeTimes and PoseDisplacement, which the thermal epoch planner reads
// - Duration strings and the keyframe CSV
// - What a bad expression does: warn, drop the segment, keep the track
// ============================================================================

#include <gtest/gtest.h>

#include "core/Config.hpp"
#include "scene/Motion.hpp"
#include "scene/MotionSpec.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace quantiloom;
using namespace quantiloom::scene;

namespace {

glm::dvec3 TranslationOf(const glm::mat4& m) {
    return glm::dvec3(m[3][0], m[3][1], m[3][2]);
}

bool NearVec(const glm::dvec3& a, const glm::dvec3& b, f64 epsilon = 1e-6) {
    return glm::length(a - b) < epsilon;
}

MotionKey PositionKey(f64 t, const glm::dvec3& p) {
    MotionKey key;
    key.t_s = t;
    key.hasPosition = true;
    key.position = p;
    return key;
}

std::unique_ptr<MotionTrack> CompileOrDie(const MotionSpec& spec) {
    Vector<String> warnings;
    auto compiled = MotionTrack::Compile(spec, &warnings);
    EXPECT_TRUE(compiled.has_value()) << (compiled.has_value() ? "" : compiled.error());
    return compiled.has_value() ? std::move(*compiled) : nullptr;
}

MotionPathResolver Identity() {
    return [](const String& path) { return path; };
}

}  // namespace

// ============================================================================
// Keyframes
// ============================================================================

TEST(MotionTest, LinearKeysAreExactAtTheMidpoint) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(4.0, {10, 0, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(2.0)), {5, 0, 0}));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(0.0)), {0, 0, 0}));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(4.0)), {10, 0, 0}));
}

TEST(MotionTest, HoldClampsOutsideTheAuthoredRange) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.extrapolate = MotionExtrap::Hold;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(4.0, {10, 0, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(-100.0)), {0, 0, 0}));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(1000.0)), {10, 0, 0}));
}

TEST(MotionTest, StepHoldsThePreviousKey) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.interpolation = MotionInterp::Step;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(4.0, {10, 0, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(3.999)), {0, 0, 0}));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(4.0)), {10, 0, 0}));
}

TEST(MotionTest, CubicPassesThroughEveryKey) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.interpolation = MotionInterp::Cubic;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(1.0, {1, 2, 0}),
                 PositionKey(2.0, {4, 1, 0}), PositionKey(3.0, {9, 3, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    for (const MotionKey& key : spec.keys) {
        EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(key.t_s)), key.position, 1e-5))
            << "at t = " << key.t_s;
    }
}

TEST(MotionTest, LoopWrapsOverTheWholeSpan) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.extrapolate = MotionExtrap::Loop;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(4.0, {8, 0, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    // 5 s is one second into the second lap.
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(5.0)), {2, 0, 0}));
    // ...and so is -3 s, one lap back.
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(-3.0)), {2, 0, 0}));
}

TEST(MotionTest, RotationKeysSlerpTheShortWay) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;

    MotionKey a;
    a.t_s = 0.0;
    a.hasRotation = true;
    a.rotation = glm::angleAxis(0.0, glm::dvec3(0, 1, 0));
    MotionKey b;
    b.t_s = 2.0;
    b.hasRotation = true;
    b.rotation = glm::angleAxis(glm::radians(90.0), glm::dvec3(0, 1, 0));
    spec.keys = {a, b};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    // 45 degrees about +Y takes +X to (cos45, 0, -sin45).
    const glm::vec3 turned = glm::vec3(track->Evaluate(1.0) * glm::vec4(1, 0, 0, 0));
    EXPECT_NEAR(turned.x, std::cos(glm::radians(45.0f)), 1e-5f);
    EXPECT_NEAR(turned.z, -std::sin(glm::radians(45.0f)), 1e-5f);
}

// ============================================================================
// Piecewise expressions
// ============================================================================

TEST(MotionTest, SegmentExpressionsSeeGlobalTAndLocalS) {
    MotionSpec spec;
    spec.form = MotionForm::Segments;

    MotionSegment first;
    first.from_s = 0.0;
    first.to_s = 2.0;
    first.hasPosition = true;
    first.position[0] = "2*t";
    first.position[1] = "0";
    first.position[2] = "sin(t)";

    MotionSegment second;
    second.from_s = 2.0;
    second.to_s = 6.0;
    second.hasPosition = true;
    second.position[0] = "4 + s";
    second.position[1] = "0";
    second.position[2] = "0";

    spec.segments = {first, second};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(1.5)), {3.0, 0.0, std::sin(1.5)}, 1e-5));
    // s restarts at every segment: 3.5 s global is 1.5 s into the second one.
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(3.5)), {5.5, 0.0, 0.0}, 1e-5));
}

TEST(MotionTest, ABadExpressionDropsItsSegmentAndWarns) {
    MotionSpec spec;
    spec.form = MotionForm::Segments;

    MotionSegment good;
    good.from_s = 0.0;
    good.to_s = 2.0;
    good.hasPosition = true;
    good.position[0] = "s";
    good.position[1] = "0";
    good.position[2] = "0";

    MotionSegment bad;
    bad.from_s = 2.0;
    bad.to_s = 4.0;
    bad.hasPosition = true;
    bad.position[0] = "2 * * unclosed(";
    bad.position[1] = "0";
    bad.position[2] = "0";

    spec.segments = {good, bad};

    Vector<String> warnings;
    auto compiled = MotionTrack::Compile(spec, &warnings);
    ASSERT_TRUE(compiled.has_value());
    EXPECT_FALSE(warnings.empty());
    // The good segment survived; past its end the track holds.
    EXPECT_TRUE(NearVec(TranslationOf((*compiled)->Evaluate(1.0)), {1, 0, 0}, 1e-5));
    EXPECT_TRUE(NearVec(TranslationOf((*compiled)->Evaluate(3.0)), {2, 0, 0}, 1e-5));
}

TEST(MotionTest, ATrackWithNoUsableSegmentFailsToCompile) {
    MotionSpec spec;
    spec.form = MotionForm::Segments;

    MotionSegment bad;
    bad.from_s = 0.0;
    bad.to_s = 1.0;
    bad.hasPosition = true;
    bad.position[0] = "))(";
    bad.position[1] = "0";
    bad.position[2] = "0";
    spec.segments = {bad};

    Vector<String> warnings;
    auto compiled = MotionTrack::Compile(spec, &warnings);
    EXPECT_FALSE(compiled.has_value());
}

// ============================================================================
// Engines
// ============================================================================

TEST(MotionTest, LinearEngineIsPositionPlusVelocityTimesTime) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.origin_s = 10.0;
    spec.location.type = MotionLocationType::Linear;
    spec.location.start = {1, 2, 3};
    spec.location.velocity = {3, 0, 0};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(10.0)), {1, 2, 3}));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(12.0)), {7, 2, 3}));
    EXPECT_TRUE(NearVec(track->VelocityAt(11.0), {3, 0, 0}));
}

TEST(MotionTest, CircleEngineReturnsAfterOneTurn) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.location.type = MotionLocationType::Circle;
    spec.location.center = {0, 0, 0};
    spec.location.radius = 5.0;
    spec.location.angularSpeedDeg_s = 90.0;  // a full turn in four seconds

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    // Zero degrees is +Z, and the swing is toward +X.
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(0.0)), {0, 0, 5}, 1e-9));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(1.0)), {5, 0, 0}, 1e-9));
    EXPECT_TRUE(NearVec(TranslationOf(track->Evaluate(4.0)), {0, 0, 5}, 1e-9));
}

TEST(MotionTest, AlongVelocityPointsForwardAtTheVelocity) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.location.type = MotionLocationType::Linear;
    spec.location.velocity = {0, 0, -4};  // heading -Z
    spec.orientation.type = MotionOrientationType::AlongVelocity;
    spec.orientation.forward = {0, 0, 1};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    const glm::vec3 nose = glm::vec3(track->Evaluate(1.0) * glm::vec4(0, 0, 1, 0));
    EXPECT_NEAR(nose.x, 0.0f, 1e-5f);
    EXPECT_NEAR(nose.y, 0.0f, 1e-5f);
    EXPECT_NEAR(nose.z, -1.0f, 1e-5f);
}

TEST(MotionTest, LookAtPointsForwardAtTheTarget) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.location.type = MotionLocationType::Linear;
    spec.location.start = {0, 0, 0};
    spec.orientation.type = MotionOrientationType::LookAt;
    spec.orientation.target = {10, 0, 0};
    spec.orientation.forward = {0, 0, 1};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    const glm::vec3 nose = glm::vec3(track->Evaluate(0.0) * glm::vec4(0, 0, 1, 0));
    EXPECT_NEAR(nose.x, 1.0f, 1e-5f);
    EXPECT_NEAR(nose.y, 0.0f, 1e-5f);
    EXPECT_NEAR(nose.z, 0.0f, 1e-5f);
}

TEST(MotionTest, SpinTurnsAtItsStatedRate) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.orientation.type = MotionOrientationType::Spin;
    spec.orientation.axis = {0, 1, 0};
    spec.orientation.rateDeg_s = 90.0;

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    const glm::vec3 turned = glm::vec3(track->Evaluate(1.0) * glm::vec4(1, 0, 0, 0));
    EXPECT_NEAR(turned.x, 0.0f, 1e-5f);
    EXPECT_NEAR(turned.z, -1.0f, 1e-5f);
}

// ============================================================================
// What the epoch planner reads
// ============================================================================

TEST(MotionTest, ChangeTimesAreTheKeysInsideTheWindow) {
    MotionSpec spec;
    spec.form = MotionForm::Keys;
    spec.keys = {PositionKey(0.0, {0, 0, 0}), PositionKey(2.0, {1, 0, 0}),
                 PositionKey(5.0, {2, 0, 0}), PositionKey(9.0, {3, 0, 0})};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);

    const Vector<f64> times = track->ChangeTimes(1.0, 6.0);
    ASSERT_EQ(times.size(), 2u);
    EXPECT_DOUBLE_EQ(times[0], 2.0);
    EXPECT_DOUBLE_EQ(times[1], 5.0);
}

TEST(MotionTest, EnginesAreSmoothAndHaveNoChangeTimes) {
    MotionSpec spec;
    spec.form = MotionForm::Engine;
    spec.location.type = MotionLocationType::Linear;
    spec.location.velocity = {1, 0, 0};

    auto track = CompileOrDie(spec);
    ASSERT_NE(track, nullptr);
    EXPECT_TRUE(track->ChangeTimes(-1e6, 1e6).empty());
}

TEST(MotionTest, PoseDisplacementCountsRotationAsArcLength) {
    const glm::mat4 identity(1.0f);
    const glm::mat4 quarterTurn = glm::rotate(identity, glm::radians(90.0f), glm::vec3(0, 1, 0));
    EXPECT_NEAR(PoseDisplacement(identity, quarterTurn, 1.0f), glm::half_pi<f32>(), 1e-4f);

    const glm::mat4 shifted = glm::translate(identity, glm::vec3(3, 4, 0));
    EXPECT_NEAR(PoseDisplacement(identity, shifted, 1.0f), 5.0f, 1e-5f);
}

// ============================================================================
// Reading a motion table
// ============================================================================

TEST(MotionSpecTest, DurationStringsCarryTheirUnits) {
    EXPECT_DOUBLE_EQ(*ParseDurationString("4"), 4.0);
    EXPECT_DOUBLE_EQ(*ParseDurationString("15s"), 15.0);
    EXPECT_DOUBLE_EQ(*ParseDurationString("90min"), 5400.0);
    EXPECT_DOUBLE_EQ(*ParseDurationString("36h"), 129600.0);
    EXPECT_DOUBLE_EQ(*ParseDurationString("2.5d"), 216000.0);
    EXPECT_FALSE(ParseDurationString("2 furlongs").has_value());
    EXPECT_FALSE(ParseDurationString("").has_value());
}

TEST(MotionSpecTest, InlineKeysAreRead) {
    const auto config = Config::Parse(R"(
[[keys]]
t = 0
position = [0, 0, 0]
[[keys]]
t = "4s"
position = [10, 0, 0]
rotation_euler_degrees = [0, 90, 0]
)");
    ASSERT_TRUE(config.has_value());

    Vector<String> warnings;
    auto spec = ParseMotionSpec(*config, 0.0, "models.car", Identity(), warnings);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->form, MotionForm::Keys);
    ASSERT_EQ(spec->keys.size(), 2u);
    EXPECT_DOUBLE_EQ(spec->keys[1].t_s, 4.0);
    EXPECT_TRUE(spec->keys[1].hasRotation);
    EXPECT_TRUE(warnings.empty());
}

TEST(MotionSpecTest, KeysWinOverSegmentsAndSaySo) {
    const auto config = Config::Parse(R"(
[[keys]]
t = 0
position = [0, 0, 0]
[[segments]]
from = 0
to = 1
position = ["s", "0", "0"]
)");
    ASSERT_TRUE(config.has_value());

    Vector<String> warnings;
    auto spec = ParseMotionSpec(*config, 0.0, "models.car", Identity(), warnings);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->form, MotionForm::Keys);
    EXPECT_TRUE(spec->segments.empty());
    EXPECT_FALSE(warnings.empty());
}

TEST(MotionSpecTest, TheHeadingFormOfALinearEngine) {
    const auto config = Config::Parse(R"(
[location]
type = "linear"
heading_deg = 90
speed_m_s = 3
[orientation]
type = "along_velocity"
)");
    ASSERT_TRUE(config.has_value());

    Vector<String> warnings;
    auto spec = ParseMotionSpec(*config, 0.0, "models.car", Identity(), warnings);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->form, MotionForm::Engine);
    // 90 degrees clockwise from +Z is +X.
    EXPECT_NEAR(spec->location.velocity.x, 3.0, 1e-9);
    EXPECT_NEAR(spec->location.velocity.z, 0.0, 1e-9);
    EXPECT_EQ(spec->orientation.type, MotionOrientationType::AlongVelocity);
}

TEST(MotionSpecTest, AnUnknownEngineTypeIsIgnoredWithAWarning) {
    const auto config = Config::Parse(R"(
[location]
type = "helix"
)");
    ASSERT_TRUE(config.has_value());

    Vector<String> warnings;
    auto spec = ParseMotionSpec(*config, 0.0, "models.car", Identity(), warnings);
    EXPECT_FALSE(spec.has_value());
    EXPECT_FALSE(warnings.empty());
}

TEST(MotionSpecTest, KeyframesLoadFromACsv) {
    const auto path = std::filesystem::temp_directory_path() / "quantiloom_motion_keys.csv";
    {
        std::ofstream file(path);
        file << "# a comment\n";
        file << "t,x,y,z\n";
        file << "0, 0, 0, 0\n";
        file << "\n";
        file << "2, 4, 0, 0\n";
        file << "4, 8, 0, 0, 0, 0, 0, 1\n";
    }

    auto keys = LoadMotionKeysCsv(path.string());
    ASSERT_TRUE(keys.has_value()) << (keys.has_value() ? "" : keys.error());
    const Vector<MotionKey>& rows = keys.value();
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_DOUBLE_EQ(rows[1].t_s, 2.0);
    EXPECT_TRUE(NearVec(rows[1].position, {4, 0, 0}));
    EXPECT_FALSE(rows[1].hasRotation);
    EXPECT_TRUE(rows[2].hasRotation);

    std::filesystem::remove(path);
}

TEST(MotionSpecTest, AMissingCsvIsReportedNotThrown) {
    auto keys = LoadMotionKeysCsv("this-file-does-not-exist-12345.csv");
    EXPECT_FALSE(keys.has_value());
}
