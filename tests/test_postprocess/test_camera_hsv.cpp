#include <gtest/gtest.h>

#include "postprocess/CameraIsp.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <array>
#include <cmath>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

CameraConfig BaseConfig() {
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
    config.optics.sensorWidthPx = 4;
    config.optics.sensorHeightPx = 4;
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
    config.products.apparentTemperature = true;
    config.products.display = true;
    return config;
}

// One 3-channel encoded-sRGB image, every pixel the same colour.
Image SolidDisplay(u32 width, u32 height, const std::array<f64, 3>& rgb) {
    Image image(width, height, 3);
    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x)
            for (u32 c = 0; c < 3; ++c)
                image(x, y, c) = static_cast<f32>(rgb[c]);
    return image;
}

std::array<f64, 3> Pixel(const Image& image, u32 x, u32 y) {
    return {image(x, y, 0), image(x, y, 1), image(x, y, 2)};
}

// Reference hue->rgb for S=V=1, mirroring the sector formula both sides use.
std::array<f64, 3> PureHue(f64 degrees) {
    const f64 h = degrees / 360.0;
    const f64 sector = h * 6.0;
    const u32 i = static_cast<u32>(sector) % 6u;
    const f64 f = sector - std::floor(sector);
    switch (i) {
    case 0: return {1.0, f, 0.0};
    case 1: return {1.0 - f, 1.0, 0.0};
    case 2: return {0.0, 1.0, f};
    case 3: return {0.0, 1.0 - f, 1.0};
    case 4: return {f, 0.0, 1.0};
    default: return {1.0, 0.0, 1.0 - f};
    }
}

Image FlatRate(const CameraConfig& config, f64 rate) {
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    Image image(width, height, 1);
    for (auto& value : image.data) value = static_cast<f32>(rate);
    return image;
}

} // namespace

// ---------------------------------------------------------------------------
// ApplyHsv unit behaviour
// ---------------------------------------------------------------------------

TEST(CameraHsvTest, DefaultConfigLeavesDisplayBitIdenticalThroughRunIsp) {
    const CameraConfig base = BaseConfig();
    CameraConfig explicitIdentity = base;
    explicitIdentity.isp.hsv.hueOffsetDegrees = 0.0;
    explicitIdentity.isp.hsv.saturationScale = 1.0;
    explicitIdentity.isp.hsv.valueGamma = 1.0;
    explicitIdentity.isp.hsv.empiricalNoise = false;
    explicitIdentity.isp.hsv.temporalDrift = false;
    // A frame with a spread of DN values so every sRGB segment is exercised.
    const CpuCameraPipeline pipeline(base);
    const Image rate = FlatRate(base, 3000.0);
    CaptureState stateA, stateB;
    auto outA = pipeline.CaptureMeasured(stateA, 0.1, rate);
    CameraConfig explicitConfig = explicitIdentity;
    const CpuCameraPipeline pipelineB(explicitConfig);
    auto outB = pipelineB.CaptureMeasured(stateB, 0.1, rate);
    ASSERT_TRUE(outA.has_value());
    ASSERT_TRUE(outB.has_value());
    // Identical configs (all defaults) must be bit-identical, i.e. the HSV
    // stage contributed nothing even though it exists in the chain.
    EXPECT_EQ(outA.value().display->image.data, outB.value().display->image.data);
}

TEST(CameraHsvTest, AllDisabledApplyHsvIsNumericallyIdentity) {
    const CameraConfig config = BaseConfig();
    const Image input = SolidDisplay(4, 4, {0.25, 0.5, 0.9});
    auto output = ApplyHsv(config, input, /*acquisitionIndex=*/7);
    ASSERT_TRUE(output.has_value());
    for (u32 y = 0; y < 4; ++y)
        for (u32 x = 0; x < 4; ++x)
            for (u32 c = 0; c < 3; ++c)
                EXPECT_NEAR(output.value().data[(y * 4 + x) * 3 + c],
                            input(x, y, c), 1e-6);
}

TEST(CameraHsvTest, HueOffsetWrapsAroundTheCircle) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.hueOffsetDegrees = 20.0;
    // A pure hue at 350 degrees, S=V=1.
    const Image input = SolidDisplay(4, 4, PureHue(350.0));
    auto output = ApplyHsv(config, input, 0);
    ASSERT_TRUE(output.has_value());
    const std::array<f64, 3> pixel = Pixel(*output, 0, 0);
    const std::array<f64, 3> expected = PureHue(10.0);
    for (u32 c = 0; c < 3; ++c)
        EXPECT_NEAR(pixel[c], expected[c], 1e-6);
}

TEST(CameraHsvTest, LowSaturationSuppressesTheHueOffset) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.hueOffsetDegrees = 60.0;
    // Grey: S=0, hue is meaningless and must not move.
    const Image grey = SolidDisplay(4, 4, {0.5, 0.5, 0.5});
    auto greyOut = ApplyHsv(config, grey, 0);
    ASSERT_TRUE(greyOut.has_value());
    for (u32 c = 0; c < 3; ++c)
        EXPECT_NEAR(Pixel(*greyOut, 0, 0)[c], 0.5, 1e-9);
    // Deep in the chromatic region the same offset must apply at full weight:
    // pure 350-degree hue + 60 degrees wraps to 50 degrees.
    const Image vivid = SolidDisplay(4, 4, PureHue(350.0));
    auto vividOut = ApplyHsv(config, vivid, 0);
    ASSERT_TRUE(vividOut.has_value());
    const std::array<f64, 3> expected = PureHue(50.0);
    for (u32 c = 0; c < 3; ++c)
        EXPECT_NEAR(Pixel(*vividOut, 0, 0)[c], expected[c], 1e-6);
}

TEST(CameraHsvTest, SaturationScaleClampsToOne) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.saturationScale = 5.0;
    // HSV(210 degrees, 0.4, 0.8): sector 3, so (p, q, v).
    const f64 h = 210.0 / 360.0;
    const f64 sector = h * 6.0;
    const u32 i = static_cast<u32>(sector) % 6u;
    const f64 f = sector - std::floor(sector);
    ASSERT_EQ(i, 3u); // blue-cyan sector
    const std::array<f64, 3> rgb{0.8 * (1.0 - 0.4), 0.8 * (1.0 - 0.4 * f), 0.8};
    const Image input = SolidDisplay(4, 4, rgb);
    auto output = ApplyHsv(config, input, 0);
    ASSERT_TRUE(output.has_value());
    // S'=clamp(0.4*5)=1, V unchanged: max-min must equal max.
    const std::array<f64, 3> pixel = Pixel(*output, 0, 0);
    const f64 maxc = std::max({pixel[0], pixel[1], pixel[2]});
    const f64 minc = std::min({pixel[0], pixel[1], pixel[2]});
    EXPECT_NEAR(maxc, 0.8, 1e-6);
    EXPECT_NEAR(maxc - minc, maxc, 1e-6);
}

TEST(CameraHsvTest, ValueGammaBrightensMidtones) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.valueGamma = 2.0;
    const Image input = SolidDisplay(4, 4, {0.25, 0.25, 0.25});
    auto output = ApplyHsv(config, input, 0);
    ASSERT_TRUE(output.has_value());
    // V' = pow(0.25, 1/2) = 0.5; grey stays grey.
    for (u32 c = 0; c < 3; ++c) {
        EXPECT_NEAR(Pixel(*output, 0, 0)[c], 0.5, 1e-6);
        EXPECT_NEAR(Pixel(*output, 3, 3)[c], 0.5, 1e-6);
    }
}

TEST(CameraHsvTest, EmpiricalEffectsAreDeterministicPerAcquisition) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.empiricalNoise = true;
    config.isp.hsv.temporalDrift = true;
    const Image input = SolidDisplay(4, 4, {0.5, 0.5, 0.5});
    auto first = ApplyHsv(config, input, /*acquisitionIndex=*/3);
    auto repeat = ApplyHsv(config, input, /*acquisitionIndex=*/3);
    auto other = ApplyHsv(config, input, /*acquisitionIndex=*/4);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(repeat.has_value());
    ASSERT_TRUE(other.has_value());
    // Same acquisition index: bit-identical (a same-tick reprocess reproduces
    // the same pattern).
    EXPECT_EQ(first.value().data, repeat.value().data);
    // A different acquisition draws a different sample of both streams.
    EXPECT_NE(first.value().data, other.value().data);
    // The pattern is per-pixel, not one scalar: not every pixel moved the
    // same way, but every value must stay in display range.
    bool anyChange = false;
    for (size_t i = 0; i < first.value().data.size(); ++i) {
        EXPECT_GE(first.value().data[i], 0.0f);
        EXPECT_LE(first.value().data[i], 1.0f);
        if (first.value().data[i] != input.data[i]) anyChange = true;
    }
    EXPECT_TRUE(anyChange);
}

TEST(CameraHsvTest, EffectsOffByDefaultEvenWithNonzeroSigmas) {
    CameraConfig config = BaseConfig();
    // Sigmas are inert while the flags are off (the defaults).
    config.isp.hsv.empiricalNoiseSigma = 0.5;
    config.isp.hsv.temporalDriftSigma = 0.5;
    const Image input = SolidDisplay(4, 4, {0.3, 0.6, 0.9});
    auto output = ApplyHsv(config, input, 9);
    ASSERT_TRUE(output.has_value());
    for (size_t i = 0; i < input.data.size(); ++i)
        EXPECT_NEAR(output.value().data[i], input.data[i], 1e-6);
}

// ---------------------------------------------------------------------------
// End-to-end invariance: HSV touches display only
// ---------------------------------------------------------------------------

TEST(CameraHsvTest, HsvChangesDisplayOnlyAndIsDeterministic) {
    const CameraConfig base = BaseConfig();
    CameraConfig graded = base;
    graded.isp.hsv.hueOffsetDegrees = 30.0;
    graded.isp.hsv.saturationScale = 1.4;
    graded.isp.hsv.valueGamma = 1.1;
    graded.isp.hsv.empiricalNoise = true;
    graded.isp.hsv.temporalDrift = true;
    const CpuCameraPipeline offPipeline(base);
    const CpuCameraPipeline onPipeline(graded);
    const Image rate = FlatRate(base, 3000.0);
    CaptureState offState, onState, onStateReplay;
    auto off = offPipeline.CaptureMeasured(offState, 0.1, rate);
    auto on = onPipeline.CaptureMeasured(onState, 0.1, rate);
    auto onReplay = onPipeline.CaptureMeasured(onStateReplay, 0.1, rate);
    ASSERT_TRUE(off.has_value());
    ASSERT_TRUE(on.has_value());
    ASSERT_TRUE(onReplay.has_value());
    // Display chain: graded, and the empirical streams key on the acquisition
    // index, so two fresh first acquisitions are bit-identical.
    ASSERT_TRUE(off.value().display && on.value().display);
    EXPECT_EQ(on.value().display->image.data, onReplay.value().display->image.data);
    EXPECT_NE(off.value().display->image.data, on.value().display->image.data);
    // Measurement products never pass through the HSV stage.
    ASSERT_TRUE(off.value().rawDn && on.value().rawDn);
    EXPECT_EQ(off.value().rawDn->image.data, on.value().rawDn->image.data);
    ASSERT_TRUE(off.value().correctedDeviceSignal && on.value().correctedDeviceSignal);
    EXPECT_EQ(off.value().correctedDeviceSignal->image.data,
              on.value().correctedDeviceSignal->image.data);
    ASSERT_TRUE(off.value().apparentTemperature && on.value().apparentTemperature);
    EXPECT_EQ(off.value().apparentTemperature->image.data,
              on.value().apparentTemperature->image.data);
}

TEST(CameraHsvTest, DifferentAcquisitionIndexDrawsADifferentPattern) {
    CameraConfig config = BaseConfig();
    config.isp.hsv.empiricalNoise = true;
    const CpuCameraPipeline pipeline(config);
    const Image rate = FlatRate(config, 3000.0);
    CaptureState stateA, stateB;
    auto a = pipeline.CaptureMeasured(stateA, 0.1, rate);   // acquisition 0
    auto b = pipeline.CaptureMeasured(stateB, 0.2, rate);   // acquisition 0
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(a.value().display->image.data, b.value().display->image.data);
    // Two sequential acquisitions of one committed state: the second frame
    // carries the next index and must show a different empirical pattern even
    // though the scene is static and noise-free up to the display stage.
    CaptureState sequence;
    auto frame1 = pipeline.CaptureMeasured(sequence, 0.1, rate);
    ASSERT_TRUE(frame1.has_value());
    auto frame2 = pipeline.CaptureMeasured(sequence, 0.2, rate);
    ASSERT_TRUE(frame2.has_value());
    EXPECT_NE(frame1.value().display->image.data, frame2.value().display->image.data);
}
