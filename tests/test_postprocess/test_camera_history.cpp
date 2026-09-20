// ============================================================================
// Quantiloom - Unit tests for the camera acquisition history (M4-2):
// checkpoints, deterministic replay, product-free advances and warmup.
//
// Everything here is CPU-side and synthetic: the spectral frame samplers are
// closed-form radiance fields, so no GPU, device or asset path is involved.
// The RNG streams are keyed on acquisitionIndex, which is what makes a
// checkpoint/restore replay bit-identical and what these tests pin down.
// ============================================================================

#include <gtest/gtest.h>

#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

constexpr double kPi = 3.14159265358979323846;

CameraConfig PhotonConfig(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
    config.device.id = "history_cmos";
    config.device.detector = DetectorKind::Photon;
    config.device.cfa = CfaPattern::Mono;
    ResponseCurve qe;
    qe.kind = ResponseKind::AbsoluteQE;
    qe.wavelengthNm = {500.0, 600.0};
    qe.value = {0.5, 0.5};
    ResponseStack response;
    response.quantumEfficiency = qe;
    config.device.channels.push_back({"Mono", response});
    config.optics.sensorWidthPx = width;
    config.optics.sensorHeightPx = height;
    config.optics.fNumber = 2.0;
    config.optics.pixelPitchUm = 5.0;
    config.readout.exposureSeconds = 0.01;
    config.readout.framePeriodSeconds = 0.1;
    config.readout.analogGain = 1.0;
    config.readout.electronsPerDn = 1.0;
    config.readout.adcBits = 16;
    config.readout.outputBits = 16;
    config.photon.fullWellElectrons = 60000.0;
    config.quality.wavelengthSamples = 2;
    config.quality.timeSamples = 1;
    config.quality.pixelSamples = 1;
    config.quality.noiseFree = true;
    config.products.bandMeasurement = true;
    config.products.rawDn = true;
    config.products.correctedDeviceSignal = true;
    config.products.display = true;
    return config;
}

CameraConfig ThermalConfig(u32 width, u32 height) {
    auto config = PhotonConfig(width, height);
    config.device.id = "history_thermal";
    config.device.detector = DetectorKind::Thermal;
    config.device.channels.clear();
    ResponseCurve absorptance;
    absorptance.kind = ResponseKind::ThermalAbsorptance;
    absorptance.wavelengthNm = {8000.0, 14000.0};
    absorptance.value = {1.0, 1.0};
    ResponseStack response;
    response.thermalAbsorptance = absorptance;
    config.device.channels.push_back({"Mono", response});
    config.optics.pixelPitchUm = 10.0;
    config.thermal.timeConstantSeconds = 0.008;
    config.thermal.responsivityDnPerWatt = 1e13;
    config.products.apparentTemperature = false;
    return config;
}

// Absorbed power per pixel for a uniform radiance field on the thermal
// config above: flat absorptance over 8000-14000 nm at f/2, 10 um pitch.
// Mirrors the closed form in test_camera_cpu.cpp.
double ThermalPowerW(double radiance) {
    return radiance * 6000.0 * (kPi / 17.0) * 1e-10;
}

SpectralFrameSampler UniformRadiance(u32 width, u32 height, double radiance) {
    return [=](double, double) -> Result<Image, String> {
        Image sample(width, height, 1);
        std::fill(sample.data.begin(), sample.data.end(), static_cast<f32>(radiance));
        return sample;
    };
}

// Dark before switchTimeSeconds, lit at and after it.
SpectralFrameSampler StepRadiance(u32 width, u32 height, double radiance,
                                  double switchTimeSeconds) {
    return [=](double time, double) -> Result<Image, String> {
        Image sample(width, height, 1);
        const double value = time < switchTimeSeconds ? 0.0 : radiance;
        std::fill(sample.data.begin(), sample.data.end(), static_cast<f32>(value));
        return sample;
    };
}

// A product-free acquisition step, the pipeline-level twin of what
// OfflineRenderer::AdvanceCameraState runs for a skipped sequence tick.
CameraAdvanceFn ProductFreeAdvance(const CpuCameraPipeline& pipeline,
                                   const SpectralFrameSampler& sampler) {
    return [&](CaptureState& state, f64 timeSeconds) -> Result<void, String> {
        auto output = pipeline.Capture(state, timeSeconds, sampler);
        if (!output)
            return Result<void, String>::Err(output.error());
        const CameraOutput& got = output.value();
        if (got.tracedRadiance || got.cieLinearSrgb || got.bandMeasurement ||
            got.rawDn || got.correctedDeviceSignal || got.apparentTemperature ||
            got.display)
            return Result<void, String>::Err(
                "product-free advance produced a product");
        return Result<void, String>::Ok();
    };
}

void ExpectStateFieldsEqual(const CaptureState& a, const CaptureState& b) {
    EXPECT_EQ(a.acquisitionIndex, b.acquisitionIndex);
    EXPECT_DOUBLE_EQ(a.frameTimeSeconds, b.frameTimeSeconds);
    EXPECT_EQ(a.thermalPixelStateW, b.thermalPixelStateW);
    EXPECT_DOUBLE_EQ(a.nextExposureSeconds, b.nextExposureSeconds);
    EXPECT_DOUBLE_EQ(a.nextAnalogGain, b.nextAnalogGain);
    EXPECT_EQ(a.nextWhiteBalance, b.nextWhiteBalance);
}

} // namespace

TEST(CameraHistoryTest, CheckpointRestoreReplaysAcquisitionsBitExactly) {
    auto config = PhotonConfig(8, 8);
    config.quality.noiseFree = false; // exercise the temporal noise streams
    CpuCameraPipeline pipeline(config);
    const auto sampler = UniformRadiance(8, 8, 1e-5);

    CaptureState state;
    ASSERT_TRUE(pipeline.Capture(state, 0.0, sampler).has_value());
    ASSERT_TRUE(pipeline.Capture(state, 0.1, sampler).has_value());
    const auto checkpoint = CheckpointCamera(state);
    ASSERT_TRUE(checkpoint.has_value());
    EXPECT_EQ(checkpoint.value().acquisitionIndex, 2u);

    const std::array<f64, 3> replayTimes = {0.2, 0.3, 0.4};
    auto runOut = [&](CaptureState& working, std::vector<std::vector<f32>>& raws) {
        for (const f64 t : replayTimes) {
            auto output = pipeline.Capture(working, t, sampler);
            ASSERT_TRUE(output.has_value());
            ASSERT_TRUE(output.value().rawDn.has_value());
            raws.push_back(output.value().rawDn->image.data);
        }
    };

    CaptureState original = state;
    std::vector<std::vector<f32>> originalRaws;
    runOut(original, originalRaws);

    ASSERT_TRUE(RestoreCamera(state, checkpoint.value()).has_value());
    EXPECT_EQ(state.acquisitionIndex, 2u);
    EXPECT_EQ(state.historyEpoch, 1u);
    std::vector<std::vector<f32>> replayedRaws;
    runOut(state, replayedRaws);

    ASSERT_EQ(originalRaws.size(), replayedRaws.size());
    for (size_t frame = 0; frame < originalRaws.size(); ++frame)
        EXPECT_EQ(originalRaws[frame], replayedRaws[frame])
            << "replayed frame " << frame << " diverged";
    ExpectStateFieldsEqual(state, original);
}

TEST(CameraHistoryTest, EveryRestoreStartsANewHistoryEpoch) {
    CaptureState state;
    state.acquisitionIndex = 7;
    state.frameTimeSeconds = 1.25;
    state.historyEpoch = 3;
    state.thermalPixelStateW = {1.0, 2.0};
    const auto checkpoint = CheckpointCamera(state);
    ASSERT_TRUE(checkpoint.has_value());

    ASSERT_TRUE(RestoreCamera(state, checkpoint.value()).has_value());
    EXPECT_EQ(state.historyEpoch, 4u);
    EXPECT_EQ(state.acquisitionIndex, 7u);
    EXPECT_EQ(state.thermalPixelStateW, checkpoint.value().thermalPixelStateW);

    // A later restore -- after history ran on from the checkpoint -- bumps
    // the epoch again. Restoring one checkpoint twice lands on the identical
    // state, so it carries the identical epoch.
    state.acquisitionIndex = 9;
    const auto later = CheckpointCamera(state);
    ASSERT_TRUE(later.has_value());
    ASSERT_TRUE(RestoreCamera(state, later.value()).has_value());
    EXPECT_EQ(state.historyEpoch, 5u);
    EXPECT_EQ(state.acquisitionIndex, 9u);
}

TEST(CameraHistoryTest, ProductFreeAdvanceStepsFirstOrderThermalResponse) {
    constexpr u32 kSize = 4;
    auto config = ThermalConfig(kSize, kSize);
    config.products.tracedRadiance = false;
    config.products.cieLinearSrgb = false;
    config.products.bandMeasurement = false;
    config.products.rawDn = false;
    config.products.correctedDeviceSignal = false;
    config.products.display = false;
    CpuCameraPipeline pipeline(config);
    const double tau = config.thermal.timeConstantSeconds;
    // Power steps up at tau/2, so the establishing dark frame at t=0 (whose
    // single time sample lands at the exposure midpoint) sees pure dark,
    // while every later frame is fully lit.
    const auto sampler = StepRadiance(kSize, kSize, 0.01, 0.5 * tau);
    const double power = ThermalPowerW(0.01);
    const CameraAdvanceFn advance = ProductFreeAdvance(pipeline, sampler);

    CaptureState state;
    // Establish the thermal vector at zero (avoids the first-frame
    // steady-state initialisation that a cold state would get).
    ASSERT_TRUE(pipeline.Capture(state, 0.0, sampler).has_value());
    ASSERT_EQ(state.thermalPixelStateW.size(),
              static_cast<size_t>(kSize) * kSize);
    for (const double value : state.thermalPixelStateW) EXPECT_DOUBLE_EQ(value, 0.0);

    const std::array<std::pair<f64, f64>, 3> expectations = {
        std::pair{1.0, 1.0 - std::exp(-1.0)},
        std::pair{2.0, 1.0 - std::exp(-2.0)},
        std::pair{3.0, 1.0 - std::exp(-3.0)},
    };
    // The radiance field is f32 end to end, so the closed form agrees with
    // the pipeline to float precision, not to the last double.
    for (const auto& [tauMultiple, fraction] : expectations) {
        ASSERT_TRUE(AdvanceCameraState(state, tauMultiple * tau, advance).has_value());
        for (const double value : state.thermalPixelStateW)
            EXPECT_NEAR(value, power * fraction, power * 1e-6)
                << "at " << tauMultiple << " tau";
    }
    EXPECT_EQ(state.acquisitionIndex, 4u);
    EXPECT_DOUBLE_EQ(state.frameTimeSeconds, 3.0 * tau);
}

TEST(CameraHistoryTest, WarmUpMatchesOneDirectLongIntegration) {
    constexpr u32 kSize = 2;
    auto config = ThermalConfig(kSize, kSize);
    config.products.bandMeasurement = false;
    config.products.rawDn = false;
    config.products.correctedDeviceSignal = false;
    config.products.display = false;
    CpuCameraPipeline pipeline(config);
    const double tau = config.thermal.timeConstantSeconds;
    const double period = tau;
    constexpr u64 kSteps = 4;
    const double now = 10.0; // far clear of the clock-origin clamp
    const double power = ThermalPowerW(0.01);
    const auto dark = UniformRadiance(kSize, kSize, 0.0);
    const auto lit = UniformRadiance(kSize, kSize, 0.01);

    // A: dark frame to establish the vector, then the warmup pre-roll. The
    // grid ends one period before `now`, so the first real acquisition after
    // the warmup continues the same cadence.
    CaptureState warmed;
    ASSERT_TRUE(pipeline.Capture(warmed, 0.0, dark).has_value());
    warmed.frameTimeSeconds = now;
    ASSERT_TRUE(WarmUpCamera(warmed, kSteps * period, period,
                             ProductFreeAdvance(pipeline, lit))
                    .has_value());
    EXPECT_DOUBLE_EQ(warmed.frameTimeSeconds, now - period);
    EXPECT_EQ(warmed.acquisitionIndex, 1u + kSteps);

    // B: the same span integrated in one acquisition.
    CaptureState direct;
    ASSERT_TRUE(pipeline.Capture(direct, 0.0, dark).has_value());
    direct.frameTimeSeconds = now - (kSteps + 1) * period; // warmup's anchor
    ASSERT_TRUE(pipeline.Capture(direct, now - period, lit).has_value());

    ASSERT_EQ(warmed.thermalPixelStateW.size(), direct.thermalPixelStateW.size());
    for (size_t i = 0; i < warmed.thermalPixelStateW.size(); ++i)
        EXPECT_NEAR(warmed.thermalPixelStateW[i], direct.thermalPixelStateW[i],
                    power * 1e-6);
    // And both agree with the closed form of the constant-forcing response.
    EXPECT_NEAR(warmed.thermalPixelStateW[0], power * (1.0 - std::exp(-4.0)),
                power * 1e-6);
}

TEST(CameraHistoryTest, WarmUpClampedStepsLeaveThermalStateUntouched) {
    constexpr u32 kSize = 1;
    auto config = ThermalConfig(kSize, kSize);
    config.products.bandMeasurement = false;
    config.products.rawDn = false;
    config.products.correctedDeviceSignal = false;
    config.products.display = false;
    CpuCameraPipeline pipeline(config);
    const double tau = config.thermal.timeConstantSeconds;
    const double period = tau;
    const double power = ThermalPowerW(0.01);
    const auto dark = UniformRadiance(kSize, kSize, 0.0);
    const auto lit = UniformRadiance(kSize, kSize, 0.01);

    // Warmup from `now = period` reaching back 3 periods: the nominal grid
    // starts at -2*period and clamps onto the clock origin, so all three
    // advances land on t=0: the first still measures one full period from the
    // rewound anchor, and the two clamped steps freeze the thermal state.
    CaptureState warmed;
    ASSERT_TRUE(pipeline.Capture(warmed, 0.0, dark).has_value());
    warmed.frameTimeSeconds = period;
    ASSERT_TRUE(WarmUpCamera(warmed, 3.0 * period, period,
                             ProductFreeAdvance(pipeline, lit))
                    .has_value());

    // The same sequence advanced by hand.
    CaptureState manual;
    ASSERT_TRUE(pipeline.Capture(manual, 0.0, dark).has_value());
    const CameraAdvanceFn advance = ProductFreeAdvance(pipeline, lit);
    manual.frameTimeSeconds = -period; // warmup's rewound anchor
    ASSERT_TRUE(AdvanceCameraState(manual, 0.0, advance).has_value());
    const double afterFirst = manual.thermalPixelStateW[0];
    EXPECT_NEAR(afterFirst, power * (1.0 - std::exp(-1.0)), power * 1e-6);
    ASSERT_TRUE(AdvanceCameraState(manual, 0.0, advance).has_value());
    EXPECT_DOUBLE_EQ(manual.thermalPixelStateW[0], afterFirst)
        << "a clamped step must not move the thermal state";
    ASSERT_TRUE(AdvanceCameraState(manual, 0.0, advance).has_value());
    EXPECT_DOUBLE_EQ(manual.thermalPixelStateW[0], afterFirst)
        << "a clamped step must not move the thermal state";

    ExpectStateFieldsEqual(warmed, manual);
}

TEST(CameraHistoryTest, CaptureReprocessIsDeterministicAndDoesNotMutate) {
    auto config = PhotonConfig(8, 8);
    config.quality.noiseFree = false;
    CpuCameraPipeline pipeline(config);
    const auto sampler = UniformRadiance(8, 8, 1e-5);

    CaptureState state;
    const CaptureState entry = state; // the true pre-commit state
    const auto original = pipeline.Capture(state, 1.0, sampler);
    ASSERT_TRUE(original.has_value());
    // The acquisition above committed: this is the state a host would hold.
    const CaptureState committed = state;

    const auto first = pipeline.CaptureReprocess(entry, 1.0, sampler);
    const auto second = pipeline.CaptureReprocess(entry, 1.0, sampler);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    // The state the reprocess ran against is untouched.
    ExpectStateFieldsEqual(entry, CaptureState{});
    EXPECT_EQ(entry.historyEpoch, 0u);

    // Same acquisitionIndex, same counter-seeded noise: bit-identical to the
    // original acquisition from the same state.
    ASSERT_TRUE(first.value().rawDn.has_value());
    ASSERT_TRUE(second.value().rawDn.has_value());
    ASSERT_TRUE(original.value().rawDn.has_value());
    EXPECT_EQ(first.value().rawDn->image.data, original.value().rawDn->image.data);
    EXPECT_EQ(second.value().rawDn->image.data, original.value().rawDn->image.data);
    EXPECT_EQ(first.value().rawDn->signal.acquisitionIndex,
              original.value().rawDn->signal.acquisitionIndex);
}

TEST(CameraHistoryTest, ReprocessUsesEntryStateNotCommittedState) {
    // CaptureReprocess runs against the state it is handed: replaying an
    // earlier tick from a saved copy must reproduce that tick's output even
    // after the committed state has moved on.
    auto config = ThermalConfig(1, 1);
    config.products.bandMeasurement = true;
    config.products.rawDn = true;
    config.products.correctedDeviceSignal = false;
    config.products.display = false;
    CpuCameraPipeline pipeline(config);
    const double tau = config.thermal.timeConstantSeconds;
    const auto dark = UniformRadiance(1, 1, 0.0);
    const auto lit = UniformRadiance(1, 1, 0.01);

    CaptureState state;
    ASSERT_TRUE(pipeline.Capture(state, 0.0, dark).has_value());
    const CaptureState preLit = state; // index 1, thermal state still zero
    // Commit one lit frame at tau: the committed state now holds 1-exp(-1).
    ASSERT_TRUE(pipeline.Capture(state, tau, lit).has_value());
    const CaptureState committed = state;
    ASSERT_GT(committed.thermalPixelStateW[0], 0.0);

    // Reprocess the dark tick from the pre-lit copy: the working state holds
    // a zero thermal vector at a zero frame time, so a dark sample leaves the
    // pixel at zero DN -- the committed lit state must not leak in.
    const auto replay = pipeline.CaptureReprocess(preLit, 0.5 * tau, dark);
    ASSERT_TRUE(replay.has_value());
    ASSERT_TRUE(replay.value().rawDn.has_value());
    EXPECT_FLOAT_EQ(replay.value().rawDn->image.data[0], 0.0f);
    EXPECT_EQ(replay.value().rawDn->signal.acquisitionIndex, 1u);
    ExpectStateFieldsEqual(state, committed);
}
