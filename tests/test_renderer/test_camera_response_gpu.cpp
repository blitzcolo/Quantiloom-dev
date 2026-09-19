#include <gtest/gtest.h>

#include "renderer/CameraResponseGpu.hpp"

#include <bit>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

constexpr double kPi = 3.14159265358979323846;

f32 FloatAt(const Vector<u32>& words, size_t index) {
    return std::bit_cast<f32>(words.at(index));
}

ResponseCurve Curve(ResponseKind kind, std::vector<f64> nm,
                    std::vector<f64> value) {
    ResponseCurve curve;
    curve.kind = kind;
    curve.wavelengthNm = std::move(nm);
    curve.value = std::move(value);
    return curve;
}

CameraConfig ConfigWithPhotonChannels(u32 channels, CfaPattern cfa) {
    CameraConfig config;
    config.enabled = true;
    config.device.detector = DetectorKind::Photon;
    config.device.cfa = cfa;
    config.optics.fNumber = 2.0;
    config.optics.pixelPitchUm = 5.0;
    config.optics.sensorWidthPx = 3;
    config.optics.sensorHeightPx = 2;
    for (u32 i = 0; i < channels; ++i) {
        ResponseStack stack;
        stack.quantumEfficiency = Curve(
            ResponseKind::AbsoluteQE, {500.0, 600.0}, {0.5, 0.5});
        config.device.channels.push_back({"channel" + std::to_string(i), stack});
    }
    return config;
}

void ExpectOriginalKnots(const Vector<u32>& words, size_t info,
                         size_t field, const ResponseCurve& source) {
    const u32 byteOffset = words.at(info + field);
    const u32 count = words.at(info + field + 1);
    ASSERT_EQ(count, source.wavelengthNm.size());
    ASSERT_EQ(byteOffset % sizeof(u32), 0u);
    const size_t firstWord = byteOffset / sizeof(u32);
    ASSERT_LE(firstWord + 2u * count, words.size());
    for (u32 i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(FloatAt(words, firstWord + 2u * i),
                        static_cast<f32>(source.wavelengthNm[i]));
        EXPECT_FLOAT_EQ(FloatAt(words, firstWord + 2u * i + 1u),
                        static_cast<f32>(source.amplitude * source.value[i]));
    }
}

void ExpectNormalizedCdf(const Vector<u32>& words, size_t info) {
    const u32 byteOffset = words.at(info + 6);
    const u32 bins = words.at(info + 7);
    ASSERT_EQ(bins, kCameraResponsePdfBins);
    ASSERT_EQ(byteOffset % sizeof(u32), 0u);
    const size_t firstWord = byteOffset / sizeof(u32);
    ASSERT_LE(firstWord + 2u * bins, words.size());
    double massSum = 0.0;
    f32 previousHigh = 0.0f;
    for (u32 bin = 0; bin < bins; ++bin) {
        const f32 mass = FloatAt(words, firstWord + 2u * bin);
        const f32 high = FloatAt(words, firstWord + 2u * bin + 1u);
        EXPECT_GE(mass, 0.0f);
        EXPECT_GE(high, previousHigh);
        massSum += mass;
        previousHigh = high;
    }
    EXPECT_NEAR(massSum, 1.0, 2e-5);
    EXPECT_FLOAT_EQ(previousHigh, 1.0f);
}

} // namespace

TEST(CameraResponseGpuTest, PreservesSeparateUnevenCurveKnotsAndNormalizedCdfs) {
    auto config = ConfigWithPhotonChannels(2, CfaPattern::MultiChannel);
    config.optics.fillFactor = 0.75;
    config.optics.cosFourthVignetting = true;
    auto& first = config.device.channels[0].response;
    first.quantumEfficiency = Curve(
        ResponseKind::RelativeQE,
        {500.0, 500.1, 501.0, 510.0}, {0.0, 1.0, 0.0, 0.0});
    first.quantumEfficiency->normalization = RelativeNormalization::PeakOne;
    first.quantumEfficiency->amplitude = 0.5;
    first.quantumEfficiency->amplitudeSource = "measured peak QE";
    first.lensTransmission = Curve(
        ResponseKind::LensTransmission,
        {500.0, 500.2, 510.0}, {1.0, 0.8, 0.9});
    first.filterTransmission = Curve(
        ResponseKind::FilterTransmission,
        {500.0, 500.7, 510.0}, {0.7, 1.0, 0.6});
    auto& second = config.device.channels[1].response;
    second.quantumEfficiency = Curve(
        ResponseKind::AbsoluteQE,
        {600.0, 600.3, 610.0}, {0.0, 0.8, 0.0});
    second.lensTransmission = Curve(
        ResponseKind::LensTransmission,
        {600.0, 605.0, 610.0}, {0.8, 0.9, 0.8});
    second.filterTransmission = Curve(
        ResponseKind::FilterTransmission,
        {600.0, 600.6, 610.0}, {0.5, 0.9, 0.5});

    const auto encoded = EncodeCameraResponseGpu(config, 3000.0f, 5.0f, 129u);
    ASSERT_TRUE(encoded.has_value());
    const auto& words = encoded.value();
    ASSERT_GE(words.size(),
              kCameraResponseHeaderWords +
                  2u * kCameraResponseChannelWords);
    EXPECT_EQ(words[0], kCameraResponseMagic);
    EXPECT_EQ(words[1], kCameraResponseGpuVersion);
    EXPECT_EQ(words[2], static_cast<u32>(DetectorKind::Photon));
    EXPECT_EQ(words[3], static_cast<u32>(CfaPattern::MultiChannel));
    EXPECT_EQ(words[4], 2u);
    EXPECT_EQ(words[5], 128u);
    EXPECT_EQ(words[6], 1u);
    EXPECT_EQ(words[7], 3u);
    EXPECT_FLOAT_EQ(FloatAt(words, 8), static_cast<f32>(kPi / 17.0));
    EXPECT_FLOAT_EQ(FloatAt(words, 9), static_cast<f32>(25e-12 * 0.75));
    EXPECT_FLOAT_EQ(FloatAt(words, 10), 0.05f);
    EXPECT_EQ(words[11], 2u);
    EXPECT_FLOAT_EQ(FloatAt(words, 12), 3000.0f);
    EXPECT_FLOAT_EQ(FloatAt(words, 13), 5.0f);
    EXPECT_EQ(words[14], 129u);

    for (u32 channel = 0; channel < 2; ++channel) {
        const size_t info = kCameraResponseHeaderWords +
                            channel * kCameraResponseChannelWords;
        const auto& stack = config.device.channels[channel].response;
        ExpectOriginalKnots(words, info, 0, *stack.quantumEfficiency);
        ExpectOriginalKnots(words, info, 2, *stack.lensTransmission);
        ExpectOriginalKnots(words, info, 4, *stack.filterTransmission);
        EXPECT_LT(words[info + 0], words[info + 2]);
        EXPECT_LT(words[info + 2], words[info + 4]);
        EXPECT_LT(words[info + 4], words[info + 6]);
        EXPECT_FLOAT_EQ(FloatAt(words, info + 8),
                        static_cast<f32>(stack.quantumEfficiency->MinNm()));
        EXPECT_FLOAT_EQ(FloatAt(words, info + 9),
                        static_cast<f32>(stack.quantumEfficiency->MaxNm()));
        ExpectNormalizedCdf(words, info);
    }
    const size_t firstInfo = kCameraResponseHeaderWords;
    const size_t cdfWord = words[firstInfo + 6] / sizeof(u32);
    const f32 emptyBinMass = FloatAt(words, cdfWord + 2u * 100u);
    EXPECT_FLOAT_EQ(emptyBinMass, 0.0f);
    // Even an empty CDF bin remains sampleable by the declared 5% uniform
    // proposal component; a narrow measured line cannot disappear.
    const f32 supportedPdf = FloatAt(words, 10) /
                             (FloatAt(words, firstInfo + 9) -
                              FloatAt(words, firstInfo + 8));
    EXPECT_GT(supportedPdf, 0.0f);
}

TEST(CameraResponseGpuTest, ProposalWeightsPhotonsByWavelengthButThermalByPower) {
    auto photon = ConfigWithPhotonChannels(1, CfaPattern::Mono);
    photon.device.channels[0].response.quantumEfficiency =
        Curve(ResponseKind::AbsoluteQE, {500.0, 600.0}, {1.0, 1.0});
    const auto encodedPhoton = EncodeCameraResponseGpu(photon);
    ASSERT_TRUE(encodedPhoton.has_value());
    const auto& p = encodedPhoton.value();
    const size_t pInfo = kCameraResponseHeaderWords;
    ExpectNormalizedCdf(p, pInfo);
    const size_t pCdf = p[pInfo + 6] / sizeof(u32);
    const double firstLo = 500.0;
    const double firstHi = 500.0 + 100.0 / 128.0;
    const double expectedFirst = (firstHi * firstHi - firstLo * firstLo) /
                                 (600.0 * 600.0 - 500.0 * 500.0);
    EXPECT_NEAR(FloatAt(p, pCdf), expectedFirst, 1e-7);
    EXPECT_LT(FloatAt(p, pCdf),
              FloatAt(p, pCdf + 2u * 127u));

    auto thermal = photon;
    thermal.device.detector = DetectorKind::Thermal;
    thermal.device.channels[0].response.quantumEfficiency.reset();
    thermal.device.channels[0].response.thermalAbsorptance =
        Curve(ResponseKind::ThermalAbsorptance,
              {8000.0, 14000.0}, {1.0, 1.0});
    const auto encodedThermal = EncodeCameraResponseGpu(thermal);
    ASSERT_TRUE(encodedThermal.has_value());
    const auto& t = encodedThermal.value();
    const size_t tInfo = kCameraResponseHeaderWords;
    ExpectNormalizedCdf(t, tInfo);
    const size_t tCdf = t[tInfo + 6] / sizeof(u32);
    EXPECT_NEAR(FloatAt(t, tCdf), 1.0 / 128.0, 1e-7);
    EXPECT_NEAR(FloatAt(t, tCdf + 2u * 127u), 1.0 / 128.0, 1e-7);
}

TEST(CameraResponseGpuTest, SystemResponseIsNotMultipliedByOptionalComponents) {
    auto config = ConfigWithPhotonChannels(1, CfaPattern::Mono);
    auto& stack = config.device.channels[0].response;
    stack.quantumEfficiency.reset();
    stack.systemResponse = Curve(
        ResponseKind::SystemPhotonQE, {500.0, 600.0}, {0.3, 0.5});
    const auto encoded = EncodeCameraResponseGpu(config);
    ASSERT_TRUE(encoded.has_value());
    const auto& words = encoded.value();
    const size_t info = kCameraResponseHeaderWords;
    ExpectOriginalKnots(words, info, 0, *stack.systemResponse);
    EXPECT_EQ(words[info + 3], 0u); // No separate lens curve.
    EXPECT_EQ(words[info + 5], 0u); // No separate filter curve.
    stack.lensTransmission = Curve(
        ResponseKind::LensTransmission, {500.0, 600.0}, {0.8, 0.8});
    EXPECT_FALSE(EncodeCameraResponseGpu(config).has_value());
}

TEST(CameraResponseGpuTest, RejectsMoreThanThreeChannelsAndInvalidAtmosphereGrid) {
    EXPECT_TRUE(EncodeCameraResponseGpu(
        ConfigWithPhotonChannels(1, CfaPattern::Mono)).has_value());
    EXPECT_TRUE(EncodeCameraResponseGpu(
        ConfigWithPhotonChannels(3, CfaPattern::RGGB)).has_value());
    EXPECT_TRUE(EncodeCameraResponseGpu(
        ConfigWithPhotonChannels(2, CfaPattern::MultiChannel)).has_value());
    EXPECT_FALSE(EncodeCameraResponseGpu(
        ConfigWithPhotonChannels(4, CfaPattern::MultiChannel)).has_value());
    EXPECT_FALSE(EncodeCameraResponseGpu(
        ConfigWithPhotonChannels(1, CfaPattern::Mono),
        3000.0f, 0.0f, 10u).has_value());
}

TEST(CameraResponseGpuTest, PreviewPixelCenterMapsToPhysicalCfaPixel) {
    const auto physicalX = [](u32 previewX, u32 previewWidth, u32 physicalWidth) {
        return ((2u * previewX + 1u) * physicalWidth) /
               (2u * previewWidth);
    };
    EXPECT_EQ(physicalX(0, 3, 2), 0u);
    EXPECT_EQ(physicalX(1, 3, 2), 1u);
    EXPECT_EQ(physicalX(2, 3, 2), 1u);
    // The companion CameraGpuTest reads NUC output from a real 3->2 compute
    // dispatch; this closed form pins its coordinate convention.
}
