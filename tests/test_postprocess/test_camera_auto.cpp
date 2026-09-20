#include <gtest/gtest.h>

#include "postprocess/CameraAutoControl.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <array>
#include <cmath>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

// ---------------------------------------------------------------------------
// Config builders (mirroring test_camera_isp.cpp)
// ---------------------------------------------------------------------------

CameraConfig MonoPhotonConfig(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
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
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    config.photon.fullWellElectrons = 16383.0;
    config.quality.noiseFree = true;
    config.products.rawDn = true;
    config.products.correctedDeviceSignal = true;
    config.products.display = true;
    return config;
}

CameraConfig BayerPhotonConfig(u32 width, u32 height) {
    CameraConfig config = MonoPhotonConfig(width, height);
    config.device.cfa = CfaPattern::RGGB;
    config.device.channels.clear();
    ResponseCurve qe;
    qe.kind = ResponseKind::AbsoluteQE;
    qe.wavelengthNm = {500.0, 600.0};
    qe.value = {0.5, 0.5};
    for (const char* name : {"R", "G", "B"}) {
        ResponseStack response;
        response.quantumEfficiency = qe;
        config.device.channels.push_back({name, response});
    }
    return config;
}

CameraConfig ThermalMonoConfig(u32 width, u32 height) {
    CameraConfig config = MonoPhotonConfig(width, height);
    config.device.detector = DetectorKind::Thermal;
    ResponseCurve absorptance;
    absorptance.kind = ResponseKind::ThermalAbsorptance;
    absorptance.wavelengthNm = {8000.0, 14000.0};
    absorptance.value = {1.0, 1.0};
    config.device.channels.clear();
    ResponseStack response;
    response.thermalAbsorptance = absorptance;
    config.device.channels.push_back({"Mono", response});
    config.thermal.timeConstantSeconds = 0.008;
    config.thermal.responsivityDnPerWatt = 1e13;
    return config;
}

// One flat measured-rate frame in e-/s (or W for a thermal detector).
Image FlatRate(const CameraConfig& config, f64 rate) {
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    const u32 channels = config.device.cfa == CfaPattern::MultiChannel ?
        static_cast<u32>(config.device.channels.size()) : 1u;
    Image image(width, height, channels);
    for (auto& value : image.data) value = static_cast<f32>(rate);
    return image;
}

} // namespace

// ---------------------------------------------------------------------------
// StepAutoControl unit behaviour
// ---------------------------------------------------------------------------

TEST(CameraAutoControlTest, DisabledLoopsReturnPreviousExactly) {
    const IspConfig isp;
    const AutoControlState previous{0.02, 3.0, {1.5, 1.0, 0.75}};
    const AutoControlInput input{0.01, 0.0, {0.3, 0.6, 0.9}};
    const AutoControlState aeOnly = StepAutoControl(isp, previous, input,
                                                    /*enableAe=*/false,
                                                    /*enableAwb=*/false);
    EXPECT_DOUBLE_EQ(aeOnly.exposure, previous.exposure);
    EXPECT_DOUBLE_EQ(aeOnly.analogGain, previous.analogGain);
    EXPECT_EQ(aeOnly.whiteBalance, previous.whiteBalance);
    const AutoControlState awbOff = StepAutoControl(isp, previous, input,
                                                    /*enableAe=*/true,
                                                    /*enableAwb=*/false);
    EXPECT_EQ(awbOff.whiteBalance, previous.whiteBalance);
    const AutoControlState aeOff = StepAutoControl(isp, previous, input,
                                                   /*enableAe=*/false,
                                                   /*enableAwb=*/true);
    EXPECT_DOUBLE_EQ(aeOff.exposure, previous.exposure);
    EXPECT_DOUBLE_EQ(aeOff.analogGain, previous.analogGain);
}

TEST(CameraAutoControlTest, ClosedLoopConvergesToTargetLuminance) {
    // Simulated feedback: the measured mean is inversely proportional to the
    // total optical scale (exposure*gain), so the loop closes and the mean
    // lands on the target. (In the real photon chain the corrected statistic
    // is analog-gain-invariant -- see CaptureLoopConverges below -- which is
    // exactly why the gain rail only carries corrections the exposure rail
    // cannot absorb; the controller math is what this case pins down.)
    IspConfig isp;
    isp.autoExposure = true;
    isp.autoControl.smoothing = 0.5; // faster than default for a short test
    AutoControlState state{0.01, 1.0, {1.0, 1.0, 1.0}};
    const f64 initialScale = state.exposure * state.analogGain;
    for (int i = 0; i < 40; ++i) {
        const f64 scale = state.exposure * state.analogGain;
        // A dim scene: the measured mean is proportional to the optical
        // scale and starts at target/4, so the loop must open up 4x.
        const AutoControlInput input{
            isp.autoControl.targetLuminance * scale / (4.0 * initialScale),
            0.0, {1.0, 1.0, 1.0}};
        state = StepAutoControl(isp, state, input, true, false);
    }
    // The scene needed 4x more light: total scale must approach that, with
    // exposure absorbing everything (well below the default 1 s ceiling).
    EXPECT_NEAR(state.exposure * state.analogGain, 4.0 * initialScale, 1e-3);
    EXPECT_NEAR(state.exposure, 4.0 * initialScale, 1e-3);
    EXPECT_DOUBLE_EQ(state.analogGain, 1.0);
}

TEST(CameraAutoControlTest, GainOnlyMovesAfterExposureHitsItsLimit) {
    IspConfig isp;
    isp.autoExposure = true;
    isp.autoControl.smoothing = 1.0; // step straight to the rails
    isp.autoControl.maxExposureSeconds = 0.02; // tiny: clamps at once
    AutoControlState state{0.01, 1.0, {1.0, 1.0, 1.0}};
    const AutoControlInput dark{1e-5, 0.0, {1.0, 1.0, 1.0}}; // ratio 18000
    state = StepAutoControl(isp, state, dark, true, false);
    EXPECT_DOUBLE_EQ(state.exposure, 0.02); // clamped
    // The gain rail took up the residual the exposure rail could not absorb
    // and slammed into its own ceiling.
    EXPECT_DOUBLE_EQ(state.analogGain, isp.autoControl.maxGain);
    // Exposure at the ceiling from here on: repeated steps move nothing.
    for (int i = 0; i < 10; ++i)
        state = StepAutoControl(isp, state, dark, true, false);
    EXPECT_DOUBLE_EQ(state.exposure, 0.02);
    EXPECT_DOUBLE_EQ(state.analogGain, isp.autoControl.maxGain);
}

TEST(CameraAutoControlTest, RailsClampTheOutput) {
    IspConfig isp;
    isp.autoExposure = true;
    isp.autoControl.smoothing = 1.0; // step straight to the rails
    AutoControlState state{0.01, 1.0, {1.0, 1.0, 1.0}};
    const AutoControlInput bright{10.0, 0.0, {1.0, 1.0, 1.0}}; // ratio 0.018
    for (int i = 0; i < 10; ++i)
        state = StepAutoControl(isp, state, bright, true, false);
    EXPECT_DOUBLE_EQ(state.exposure, isp.autoControl.minExposureSeconds);
    EXPECT_DOUBLE_EQ(state.analogGain, 1.0); // gain never drops below 1
    const AutoControlInput dark{1e-6, 0.0, {1.0, 1.0, 1.0}};
    for (int i = 0; i < 10; ++i)
        state = StepAutoControl(isp, state, dark, true, false);
    EXPECT_DOUBLE_EQ(state.exposure, isp.autoControl.maxExposureSeconds);
    EXPECT_DOUBLE_EQ(state.analogGain, isp.autoControl.maxGain);
}

TEST(CameraAutoControlTest, IirStepIsBoundedBySmoothingFraction) {
    IspConfig isp;
    isp.autoExposure = true;
    isp.autoControl.smoothing = 0.2;
    const AutoControlState previous{0.01, 1.0, {1.0, 1.0, 1.0}};
    const AutoControlInput input{isp.autoControl.targetLuminance / 10.0, 0.0,
                                 {1.0, 1.0, 1.0}};
    const AutoControlState next = StepAutoControl(isp, previous, input,
                                                  true, false);
    // The unclamped correction target is exposure*10 = 0.1; a 0.2 IIR moves
    // at most 20% of the remaining gap per acquisition.
    EXPECT_NEAR(next.exposure, 0.01 + 0.2 * (0.1 - 0.01), 1e-12);
    EXPECT_DOUBLE_EQ(next.analogGain, 1.0);
}

TEST(CameraAutoControlTest, AwbConvergesToGreyWorldGains) {
    IspConfig isp;
    isp.autoWhiteBalance = true;
    isp.autoControl.smoothing = 0.5;
    AutoControlState state{0.01, 1.0, {1.0, 1.0, 1.0}};
    // R reads half of G, B a quarter: grey-world gains {2, 1, 4}.
    const AutoControlInput input{0.1, 0.0, {0.25, 0.5, 0.125}};
    for (int i = 0; i < 40; ++i)
        state = StepAutoControl(isp, state, input, false, true);
    EXPECT_NEAR(state.whiteBalance[0], 2.0, 1e-3);
    EXPECT_NEAR(state.whiteBalance[1], 1.0, 1e-12);
    EXPECT_NEAR(state.whiteBalance[2], 4.0, 1e-3);
}

TEST(CameraAutoControlTest, NoMeasurementHoldsPrevious) {
    IspConfig isp;
    isp.autoExposure = true;
    isp.autoWhiteBalance = true;
    const AutoControlState previous{0.02, 2.0, {1.2, 1.0, 0.9}};
    // Zero mean (every pixel saturated or black): no information, hold.
    const AutoControlInput empty{0.0, 1.0, {0.0, 0.0, 0.0}};
    const AutoControlState next = StepAutoControl(isp, previous, empty, true, true);
    EXPECT_DOUBLE_EQ(next.exposure, previous.exposure);
    EXPECT_DOUBLE_EQ(next.analogGain, previous.analogGain);
    EXPECT_EQ(next.whiteBalance, previous.whiteBalance);
}

// ---------------------------------------------------------------------------
// Pipeline integration: the loop closes through CaptureState
// ---------------------------------------------------------------------------

TEST(CameraAutoControlTest, CaptureLoopConvergesExposureOnTheMeasuredPath) {
    CameraConfig config = MonoPhotonConfig(8, 8);
    config.isp.autoExposure = true;
    config.isp.autoControl.smoothing = 0.5;
    const CpuCameraPipeline pipeline(config);
    // 4000 e-/s needs 0.737 s to fill 18% of the well: reachable by the
    // exposure rail alone (max 1 s), so the loop must settle there with the
    // gain rail untouched.
    const Image rate = FlatRate(config, 4000.0); // e-/s
    CaptureState state;
    for (int i = 0; i < 24; ++i) {
        auto output = pipeline.CaptureMeasured(state, 0.1 * (i + 1), rate);
        ASSERT_TRUE(output.has_value());
    }
    const f64 requiredExposure = config.isp.autoControl.targetLuminance *
                                     config.photon.fullWellElectrons / 4000.0;
    EXPECT_NEAR(state.nextExposureSeconds, requiredExposure, 1e-3);
    // The corrected statistic the loop measures is analog-gain-invariant (the
    // readout divides gain back out), so the gain rail never moves while the
    // exposure rail can absorb the correction: exposure-first, gain at the
    // boundary -- the designed division of labour.
    EXPECT_DOUBLE_EQ(state.nextAnalogGain, 1.0);
}

TEST(CameraAutoControlTest, GainRailCarriesWhatExposureCannotReach) {
    CameraConfig config = MonoPhotonConfig(8, 8);
    config.isp.autoExposure = true;
    config.isp.autoControl.smoothing = 0.5;
    const CpuCameraPipeline pipeline(config);
    // 1000 e-/s would need 2.95 s (target 0.18 * 16383 / 1000): beyond the
    // 1 s exposure ceiling. The exposure rail saturates and the gain rail
    // carries the unreachable residual to its own ceiling, then the loop
    // settles there -- the statistics are gain-invariant, so the measured
    // mean keeps asking for more and the rails hold at their limits.
    const Image rate = FlatRate(config, 1000.0); // e-/s
    CaptureState state;
    for (int i = 0; i < 24; ++i) {
        auto output = pipeline.CaptureMeasured(state, 0.1 * (i + 1), rate);
        ASSERT_TRUE(output.has_value());
    }
    EXPECT_NEAR(state.nextExposureSeconds,
                config.isp.autoControl.maxExposureSeconds, 1e-6);
    EXPECT_NEAR(state.nextAnalogGain, config.isp.autoControl.maxGain, 1e-4);
}

TEST(CameraAutoControlTest, SecondCaptureAppliesTheFeedback) {
    CameraConfig config = MonoPhotonConfig(8, 8);
    config.isp.autoExposure = true;
    config.isp.autoControl.smoothing = 1.0; // jump straight to the target
    const CpuCameraPipeline pipeline(config);
    const Image rate = FlatRate(config, 1000.0);
    CaptureState state;
    auto output1 = pipeline.CaptureMeasured(state, 0.1, rate);
    ASSERT_TRUE(output1.has_value());
    // Smoothing 1.0 with the dim scene drives exposure to its rail and the
    // gain rail takes the residual (ratio ~295, exposure rail tops at 100x).
    EXPECT_DOUBLE_EQ(state.nextExposureSeconds,
                     config.isp.autoControl.maxExposureSeconds);
    const f64 gainAfterFirst = state.nextAnalogGain;
    // The next acquisition must actually run with that feedback: the RAW DN
    // grows by the exposure*gain ratio (noise-free chain, 1 e-/DN).
    auto output2 = pipeline.CaptureMeasured(state, 0.2, rate);
    ASSERT_TRUE(output2.has_value());
    ASSERT_TRUE(output1.value().rawDn && output2.value().rawDn);
    const f64 dn1 = output1.value().rawDn->image.data[0];
    const f64 dn2 = output2.value().rawDn->image.data[0];
    const f64 expectedScale = config.isp.autoControl.maxExposureSeconds *
                                  gainAfterFirst /
                              (config.readout.exposureSeconds *
                                  config.readout.analogGain);
    // One DN of quantization on dn2 is 0.1 in this ratio (dn1 is 10 DN), so
    // the bound is a quantization step, not a floating-point tolerance.
    EXPECT_NEAR(dn2 / dn1, expectedScale, 0.06);
}

TEST(CameraAutoControlTest, AwbFeedbackFollowsChannelImbalance) {
    CameraConfig config = BayerPhotonConfig(8, 8);
    config.isp.autoWhiteBalance = true;
    config.isp.autoControl.smoothing = 0.5;
    const CpuCameraPipeline pipeline(config);
    // Per-pixel CFA rates: R at 1000, G at 2000, B at 500 e-/s.
    Image rate(8, 8, 1);
    for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            const u32 channel = (y % 2 == 0) ? (x % 2 == 0 ? 0u : 1u)
                                             : (x % 2 == 0 ? 1u : 2u);
            const f64 value = channel == 0 ? 1000.0 : channel == 1 ? 2000.0 : 500.0;
            rate(x, y, 0) = static_cast<f32>(value);
        }
    CaptureState state;
    auto output = pipeline.CaptureMeasured(state, 0.1, rate);
    ASSERT_TRUE(output.has_value());
    // One 0.5-smoothed step from unity toward grey-world {2, 1, 4}.
    EXPECT_NEAR(state.nextWhiteBalance[0], 1.5, 1e-9);
    EXPECT_DOUBLE_EQ(state.nextWhiteBalance[1], 1.0);
    EXPECT_NEAR(state.nextWhiteBalance[2], 2.5, 1e-9);
    // The feedback is applied to the NEXT acquisition's display only: the RAW
    // product never sees white balance. Replay the first capture against a
    // fresh state (authored WB) for comparison.
    CaptureState replay; // fresh state: no feedback, authored config
    auto outputReplay = pipeline.CaptureMeasured(replay, 0.1, rate);
    ASSERT_TRUE(outputReplay.has_value());
    ASSERT_TRUE(output.value().rawDn && outputReplay.value().rawDn);
    EXPECT_EQ(output.value().rawDn->image.data, outputReplay.value().rawDn->image.data);
    ASSERT_TRUE(output.value().display && outputReplay.value().display);
    // WB {1.5, 1, 2.5} on the second committed capture visibly moves the
    // display while RAW stays identical.
    auto output2 = pipeline.CaptureMeasured(state, 0.2, rate);
    ASSERT_TRUE(output2.has_value());
    ASSERT_TRUE(output2.value().rawDn);
    EXPECT_EQ(output2.value().rawDn->image.data,
              outputReplay.value().rawDn->image.data);
    EXPECT_NE(output2.value().display->image.data,
              outputReplay.value().display->image.data);
}

TEST(CameraAutoControlTest, ThermalChainSkipsTheController) {
    CameraConfig config = ThermalMonoConfig(8, 8);
    config.isp.autoExposure = true;
    config.isp.autoWhiteBalance = true;
    const CpuCameraPipeline pipeline(config);
    const Image rate = FlatRate(config, 1e-9); // W
    CaptureState state;
    for (int i = 0; i < 3; ++i) {
        auto output = pipeline.CaptureMeasured(state, 0.1 * (i + 1), rate);
        ASSERT_TRUE(output.has_value());
        // AE has no physical meaning on a power-responding detector: the
        // feedback fields keep the authored config values forever.
        EXPECT_DOUBLE_EQ(state.nextExposureSeconds,
                         config.readout.exposureSeconds);
        EXPECT_DOUBLE_EQ(state.nextAnalogGain, config.readout.analogGain);
        EXPECT_EQ(state.nextWhiteBalance,
                  (std::array<f64, 3>{1.0, 1.0, 1.0}));
    }
}

TEST(CameraAutoControlTest, ReprocessDoesNotAdvanceTheLoop) {
    CameraConfig config = MonoPhotonConfig(8, 8);
    config.isp.autoExposure = true;
    const CpuCameraPipeline pipeline(config);
    const Image rate = FlatRate(config, 1000.0);
    CaptureState state;
    auto committed = pipeline.CaptureMeasured(state, 0.1, rate);
    ASSERT_TRUE(committed.has_value());
    const f64 exposureAfterCommit = state.nextExposureSeconds;
    // A reprocess runs the full measurement against a copy but commits
    // nothing: the feedback must be exactly where the committed capture left
    // it, and re-running it changes nothing either.
    for (int i = 0; i < 3; ++i) {
        const SpectralFrameSampler sampler =
            [&rate](f64, f64) { return Result<Image, String>(rate); };
        auto reprocessed = pipeline.CaptureReprocess(state, 0.1, sampler);
        ASSERT_TRUE(reprocessed.has_value());
        EXPECT_DOUBLE_EQ(state.nextExposureSeconds, exposureAfterCommit);
        EXPECT_EQ(state.acquisitionIndex, 1u);
    }
}
