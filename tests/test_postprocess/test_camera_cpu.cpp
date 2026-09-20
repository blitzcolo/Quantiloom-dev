#include <gtest/gtest.h>

#include "io/ImageIO.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

constexpr double kH = 6.62607015e-34;
constexpr double kC = 299792458.0;
constexpr double kPi = 3.14159265358979323846;

CameraConfig PhotonConfig(u32 width, u32 height) {
    CameraConfig config;
    config.enabled = true;
    config.device.id = "closed_form_cmos";
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
    config.device.id = "closed_form_thermal";
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
    config.quality.wavelengthSamples = 2;
    return config;
}

double PhotonRatePerUnitRadiance(const CameraConfig& config, double qe = 0.5) {
    const double area = std::pow(config.optics.pixelPitchUm * 1e-6, 2);
    const double omega = kPi / (1.0 + 4.0 * config.optics.fNumber * config.optics.fNumber);
    return omega * area * qe * 1e-9 *
           (600.0 * 600.0 - 500.0 * 500.0) / (2.0 * kH * kC);
}

double RadianceForElectrons(const CameraConfig& config, double expectedElectrons) {
    return expectedElectrons /
           (PhotonRatePerUnitRadiance(config) * config.readout.exposureSeconds);
}

SpectralFrameSampler UniformRadiance(u32 width, u32 height, double radiance) {
    return [=](double, double) -> Result<Image, String> {
        Image sample(width, height, 1);
        std::fill(sample.data.begin(), sample.data.end(), static_cast<f32>(radiance));
        return sample;
    };
}

SpectralFrameSampler BlackbodyRadiance(u32 width, u32 height, double kelvin) {
    return [=](double, double nm) -> Result<Image, String> {
        const auto radiance = PlanckRadianceWm2SrNm(nm, kelvin);
        if (!radiance) return Result<Image, String>::Err(radiance.error());
        Image sample(width, height, 1);
        std::fill(sample.data.begin(), sample.data.end(),
                  static_cast<f32>(radiance.value()));
        return sample;
    };
}

double Mean(const Image& image) {
    return std::accumulate(image.data.begin(), image.data.end(), 0.0) /
           static_cast<double>(image.data.size());
}

double PopulationVariance(const Image& image) {
    const double mean = Mean(image);
    double sum = 0.0;
    for (const f32 value : image.data) sum += (value - mean) * (value - mean);
    return sum / static_cast<double>(image.data.size());
}

std::filesystem::path UniqueTemporaryDirectory(const char* name) {
    std::random_device random;
    for (u32 attempt = 0; attempt < 32; ++attempt) {
        const auto path = std::filesystem::temp_directory_path() /
            (String(name) + "_" + std::to_string(random()) + "_" +
             std::to_string(attempt));
        if (std::filesystem::create_directory(path)) return path;
    }
    throw std::runtime_error("could not create a unique camera test directory");
}

} // namespace

TEST(CpuCameraPipelineTest, UniformSpectrumMatchesPhotonIntegralAndExposure) {
    const auto config = PhotonConfig(1, 1);
    const double radiance = 1e-5;
    CpuCameraPipeline pipeline(config);
    CaptureState state;
    const auto captured = pipeline.Capture(
        state, 1.0, UniformRadiance(1, 1, radiance));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().bandMeasurement.has_value());
    ASSERT_TRUE(captured.value().rawDn.has_value());
    ASSERT_TRUE(captured.value().correctedDeviceSignal.has_value());
    const double rate = radiance * PhotonRatePerUnitRadiance(config);
    EXPECT_NEAR(captured.value().bandMeasurement->image.data[0], rate, rate * 1e-4);
    EXPECT_EQ(captured.value().bandMeasurement->signal.unit, "e-/s");
    EXPECT_EQ(captured.value().rawDn->signal.unit, "DN");
    const double expectedDn = std::floor(rate * config.readout.exposureSeconds + 0.5);
    EXPECT_DOUBLE_EQ(captured.value().rawDn->image.data[0], expectedDn);
    EXPECT_DOUBLE_EQ(captured.value().correctedDeviceSignal->image.data[0], expectedDn);
    EXPECT_EQ(state.acquisitionIndex, 1u);
    EXPECT_NEAR(captured.value().rawDn->signal.exposureStartSeconds, 0.995, 1e-12);
    EXPECT_NEAR(captured.value().rawDn->signal.exposureEndSeconds, 1.005, 1e-12);
}

TEST(CpuCameraPipelineTest, AreaChangesRateAndExposureChangesCharge) {
    const double radiance = 1e-4;
    auto base = PhotonConfig(1, 1);
    auto widePixel = base;
    widePixel.optics.pixelPitchUm = 10.0;
    widePixel.readout.exposureSeconds = 0.02;
    CaptureState baseState, wideState;
    const auto first = CpuCameraPipeline(base).Capture(
        baseState, 0.0, UniformRadiance(1, 1, radiance));
    const auto second = CpuCameraPipeline(widePixel).Capture(
        wideState, 0.0, UniformRadiance(1, 1, radiance));
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    const double rateA = first.value().bandMeasurement->image.data[0];
    const double rateB = second.value().bandMeasurement->image.data[0];
    EXPECT_NEAR(rateB / rateA, 4.0, 1e-4);
    const double dnA = first.value().rawDn->image.data[0];
    const double dnB = second.value().rawDn->image.data[0];
    EXPECT_NEAR(dnB / dnA, 8.0, 0.02);
}

TEST(CpuCameraPipelineTest, FullWellGainBlackLevelAndAdcHaveDeclaredOrder) {
    auto config = PhotonConfig(1, 1);
    config.photon.fullWellElectrons = 100.0;
    config.readout.analogGain = 1.5;
    config.readout.electronsPerDn = 2.0;
    config.readout.blackLevelDn = 100.0;
    config.readout.adcBits = 8;
    config.readout.outputBits = 8;
    const double radiance = RadianceForElectrons(config, 123.4);
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(1, 1, radiance));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().rawDn.has_value());
    // Clamp to 100 e- before analog gain; floor(1.5*100/2 + 100 + 0.5) = 175.
    EXPECT_FLOAT_EQ(captured.value().rawDn->image.data[0], 175.0f);
    EXPECT_NEAR(captured.value().correctedDeviceSignal->image.data[0], 100.0, 1e-5);

    config.photon.fullWellElectrons = 10000.0;
    const auto saturated = CpuCameraPipeline(config).Capture(
        state, 0.1, UniformRadiance(1, 1, RadianceForElectrons(config, 1000.0)));
    ASSERT_TRUE(saturated.has_value());
    EXPECT_FLOAT_EQ(saturated.value().rawDn->image.data[0], 255.0f);
}

TEST(CpuCameraPipelineTest, BayerProducesOneRawSamplePerPixel) {
    auto config = PhotonConfig(2, 2);
    config.device.cfa = CfaPattern::RGGB;
    config.device.channels.clear();
    for (const auto& [name, qeValue] :
         std::array<std::pair<const char*, double>, 3>{{{"R", 1.0}, {"G", 0.5}, {"B", 0.25}}}) {
        ResponseStack response;
        ResponseCurve qe;
        qe.kind = ResponseKind::AbsoluteQE;
        qe.wavelengthNm = {500.0, 600.0};
        qe.value = {qeValue, qeValue};
        response.quantumEfficiency = qe;
        config.device.channels.push_back({name, response});
    }
    const double radiance = RadianceForElectrons(config, 100.0);
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(2, 2, radiance));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().rawDn.has_value());
    const auto& raw = captured.value().rawDn->image;
    EXPECT_EQ(raw.channels, 1u);
    EXPECT_EQ(raw.data.size(), 4u);
    EXPECT_EQ(captured.value().rawDn->signal.cfa, CfaPattern::RGGB);
    EXPECT_NEAR(raw(0, 0, 0), 200.0, 1.0);
    EXPECT_NEAR(raw(1, 0, 0), 100.0, 1.0);
    EXPECT_NEAR(raw(0, 1, 0), 100.0, 1.0);
    EXPECT_NEAR(raw(1, 1, 0), 50.0, 1.0);
}

TEST(CpuCameraPipelineTest, FailedSceneSampleDoesNotAdvanceAcquisition) {
    const auto config = PhotonConfig(1, 1);
    CaptureState state;
    state.acquisitionIndex = 17;
    const SpectralFrameSampler failing = [](double, double) -> Result<Image, String> {
        return Result<Image, String>::Err("scene sample failed");
    };
    const auto captured = CpuCameraPipeline(config).Capture(state, 0.0, failing);
    EXPECT_FALSE(captured.has_value());
    EXPECT_EQ(state.acquisitionIndex, 17u);
}

TEST(CpuCameraPipelineTest, PhotonShotNoiseHasPoissonMeanVarianceAndDarkFrames) {
    auto config = PhotonConfig(64, 64);
    config.quality.noiseFree = false;
    const double radiance = RadianceForElectrons(config, 100.0);
    CaptureState state;
    const auto bright = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(64, 64, radiance));
    ASSERT_TRUE(bright.has_value());
    const auto& raw = bright.value().rawDn->image;
    EXPECT_NEAR(Mean(raw), 100.0, 1.0);
    EXPECT_NEAR(PopulationVariance(raw), 100.0, 10.0);

    const auto lowCount = CpuCameraPipeline(config).Capture(
        state, 0.1, UniformRadiance(
            64, 64, RadianceForElectrons(config, 1.0)));
    ASSERT_TRUE(lowCount.has_value());
    const auto& lowRaw = lowCount.value().rawDn->image;
    const double zeroFraction = static_cast<double>(std::count(
        lowRaw.data.begin(), lowRaw.data.end(), 0.0f)) / lowRaw.data.size();
    // For true Poisson(1), P(N=0)=exp(-1). A clipped Gaussian shot
    // approximation has a different zero mass.
    EXPECT_NEAR(zeroFraction, std::exp(-1.0), 0.03);
}

TEST(CpuCameraPipelineTest, DarkChargeMeanAndVarianceGrowWithExposure) {
    auto shortConfig = PhotonConfig(64, 64);
    shortConfig.quality.noiseFree = false;
    shortConfig.photon.darkCurrentElectronsPerSecond = 100.0;
    auto longConfig = shortConfig;
    longConfig.readout.exposureSeconds = 0.02;
    CaptureState shortState, longState;
    const auto shortFrame = CpuCameraPipeline(shortConfig).Capture(
        shortState, 0.0, UniformRadiance(64, 64, 0.0));
    const auto longFrame = CpuCameraPipeline(longConfig).Capture(
        longState, 0.0, UniformRadiance(64, 64, 0.0));
    ASSERT_TRUE(shortFrame.has_value());
    ASSERT_TRUE(longFrame.has_value());
    const auto& shortRaw = shortFrame.value().rawDn->image;
    const auto& longRaw = longFrame.value().rawDn->image;
    EXPECT_NEAR(Mean(shortRaw), 1.0, 0.10);
    EXPECT_NEAR(Mean(longRaw), 2.0, 0.12);
    EXPECT_NEAR(PopulationVariance(shortRaw), 1.0, 0.16);
    EXPECT_NEAR(PopulationVariance(longRaw), 2.0, 0.24);
}

TEST(CpuCameraPipelineTest, FixedPrnuPersistsAcrossAcquisitions) {
    auto config = PhotonConfig(64, 64);
    config.quality.noiseFree = false;
    config.photon.prnuSigma = 0.1;
    config.photon.enableFpn = true;
    const auto sampler = UniformRadiance(
        64, 64, RadianceForElectrons(config, 1000.0));
    CpuCameraPipeline pipeline(config);
    CaptureState state;
    const auto first = pipeline.Capture(state, 0.0, sampler);
    const auto second = pipeline.Capture(state, 0.1, sampler);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    const auto& a = first.value().rawDn->image;
    const auto& b = second.value().rawDn->image;
    ASSERT_EQ(a.data.size(), b.data.size());
    const double meanA = Mean(a), meanB = Mean(b);
    double covariance = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        covariance += (a.data[i] - meanA) * (b.data[i] - meanB);
    covariance /= a.data.size();
    const double correlation = covariance /
        std::sqrt(PopulationVariance(a) * PopulationVariance(b));
    // The 100 e- fixed-pattern spread dominates ~32 e- photon noise.
    EXPECT_GT(correlation, 0.83);
    EXPECT_EQ(state.acquisitionIndex, 2u);
}

TEST(CpuCameraPipelineTest, FixedBiasDoesNotShrinkTemporalReadNoise) {
    auto config = PhotonConfig(64, 64);
    config.quality.noiseFree = false;
    config.readout.blackLevelDn = 100.0;
    config.photon.readNoiseElectronsRms = 4.0;
    config.photon.biasDnRms = 3.0;
    config.photon.enableFpn = true;
    CpuCameraPipeline pipeline(config);
    CaptureState state;
    const auto first = pipeline.Capture(state, 0.0, UniformRadiance(64, 64, 0.0));
    const auto second = pipeline.Capture(state, 0.1, UniformRadiance(64, 64, 0.0));
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    const auto& a = first.value().rawDn->image;
    const auto& b = second.value().rawDn->image;
    Image difference(64, 64, 1);
    for (size_t i = 0; i < difference.data.size(); ++i)
        difference.data[i] = a.data[i] - b.data[i];
    // Differencing cancels the fixed bias; Var(A-B)/2 is temporal read
    // variance (16 e-^2), with a small contribution from quantization.
    EXPECT_NEAR(PopulationVariance(difference) / 2.0, 16.0, 2.5);
    EXPECT_NEAR(PopulationVariance(a), 25.0, 4.0);
}

TEST(CpuCameraPipelineTest, IndependentNucMapsModifyCorrectionButNotRaw) {
    auto config = PhotonConfig(2, 1);
    config.photon.applyNuc = true;
    config.photon.nucGainMap = {1.0, 0.5};
    config.photon.nucOffsetElectronsMap = {0.0, 10.0};
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(
            2, 1, RadianceForElectrons(config, 100.0)));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().rawDn.has_value());
    ASSERT_TRUE(captured.value().correctedDeviceSignal.has_value());
    const auto& raw = captured.value().rawDn->image;
    const auto& corrected = captured.value().correctedDeviceSignal->image;
    EXPECT_FLOAT_EQ(raw.data[0], 100.0f);
    EXPECT_FLOAT_EQ(raw.data[1], 100.0f);
    EXPECT_FLOAT_EQ(corrected.data[0], 100.0f);
    EXPECT_FLOAT_EQ(corrected.data[1], 60.0f);
}

TEST(CpuCameraPipelineTest, DisplayIsReprocessedFromQuantizedRaw) {
    auto config = PhotonConfig(2, 1);
    config.photon.fullWellElectrons = 1000.0;
    // One frame carrying both quantization levels: the dark pixel lands on DN
    // 0 and the bright one on DN 1, and the two must come out of the tone
    // stage as different display values.
    const auto twoLevel = [](double left, double right) -> SpectralFrameSampler {
        return [=](double, double) -> Result<Image, String> {
            Image sample(2, 1, 1);
            sample.data = {static_cast<f32>(left), static_cast<f32>(right)};
            return sample;
        };
    };
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, twoLevel(
            RadianceForElectrons(config, 0.4), RadianceForElectrons(config, 0.6)));
    ASSERT_TRUE(captured.has_value());
    EXPECT_FLOAT_EQ(captured.value().rawDn->image.data[0], 0.0f);
    EXPECT_FLOAT_EQ(captured.value().rawDn->image.data[1], 1.0f);
    EXPECT_EQ(captured.value().display->signal.kind, SignalKind::DisplaySrgb);
    EXPECT_EQ(captured.value().display->signal.unit, "encoded sRGB");
    // The display branch consumes the corrected signal: one DN is one
    // electron here, normalized by the 1000 e- well. That is small, but
    // distinctly nonzero. The display is 2x1x3; the bright pixel is the
    // second pixel, i.e. channels 3..5.
    EXPECT_FLOAT_EQ(captured.value().display->image.data[0], 0.0f);
    for (u32 c = 0; c < 3; ++c)
        EXPECT_NEAR(captured.value().display->image.data[3 + c],
                    static_cast<f32>(12.92 / 1000.0), 1e-7f);
}

TEST(CpuCameraPipelineTest, RejectsShortOrNonfiniteNucCalibrationMaps) {
    auto config = PhotonConfig(2, 1);
    config.photon.applyNuc = true;
    config.photon.nucGainMap = {1.0};
    CaptureState state;
    const auto shortMap = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(2, 1, 0.0));
    EXPECT_FALSE(shortMap.has_value());
    EXPECT_EQ(state.acquisitionIndex, 0u);
    config.photon.nucGainMap = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto nanMap = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(2, 1, 0.0));
    EXPECT_FALSE(nanMap.has_value());
    EXPECT_EQ(state.acquisitionIndex, 0u);
}

TEST(CpuCameraPipelineTest, NativeChannelsDoNotReuseOnePixelNoiseStream) {
    auto config = PhotonConfig(64, 64);
    config.device.cfa = CfaPattern::MultiChannel;
    const auto baseChannel = config.device.channels.front();
    config.device.channels = {baseChannel, baseChannel, baseChannel};
    config.device.channels[0].name = "R";
    config.device.channels[1].name = "G";
    config.device.channels[2].name = "B";
    config.quality.noiseFree = false;
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, UniformRadiance(
            64, 64, RadianceForElectrons(config, 100.0)));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().rawDn.has_value());
    const auto& raw = captured.value().rawDn->image;
    ASSERT_EQ(raw.channels, 3u);
    size_t identical = 0;
    for (u32 y = 0; y < raw.height; ++y)
        for (u32 x = 0; x < raw.width; ++x)
            if (raw(x, y, 0) == raw(x, y, 1)) ++identical;
    const double identicalFraction = static_cast<double>(identical) /
                                     (raw.width * raw.height);
    // Independent Poisson(100) draws match occasionally, not at every pixel.
    EXPECT_LT(identicalFraction, 0.08);
}

TEST(CpuCameraPipelineTest, ThermalPixelStateFollowsFirstOrderStepInDetectorCoordinates) {
    const auto config = ThermalConfig(1, 1);
    CpuCameraPipeline pipeline(config);
    CaptureState state;
    const auto dark = pipeline.Capture(state, 0.0, UniformRadiance(1, 1, 0.0));
    ASSERT_TRUE(dark.has_value());
    ASSERT_EQ(state.thermalPixelStateW.size(), 1u);
    EXPECT_DOUBLE_EQ(state.thermalPixelStateW[0], 0.0);
    const auto lit = pipeline.Capture(state, 0.008, UniformRadiance(1, 1, 0.01));
    ASSERT_TRUE(lit.has_value());
    ASSERT_TRUE(lit.value().bandMeasurement.has_value());
    const double power = 0.01 * 6000.0 * (kPi / 17.0) * 1e-10;
    EXPECT_NEAR(lit.value().bandMeasurement->image.data[0], power, power * 1e-4);
    EXPECT_EQ(lit.value().bandMeasurement->signal.unit, "W");
    EXPECT_NEAR(state.thermalPixelStateW[0], power * (1.0 - std::exp(-1.0)),
                power * 1e-4);
    EXPECT_EQ(state.acquisitionIndex, 2u);
}

TEST(CpuCameraPipelineTest, ThermalNetdUsesResponseDerivativeAndReadoutBandwidth) {
    auto config = ThermalConfig(64, 64);
    config.quality.noiseFree = false;
    config.thermal.netdKelvin = 0.04;
    config.thermal.netdReferenceTemperatureK = 300.0;
    config.thermal.netdNoiseBandwidthHz = 100.0;
    config.thermal.netdOpticalCondition = "f/2 reference lens";
    config.thermal.readoutWindowSeconds = 0.005; // B=1/(2T)=100 Hz.
    CaptureState state;
    const auto reference = CpuCameraPipeline(config).Capture(
        state, 0.0, BlackbodyRadiance(64, 64, 300.0));
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(reference.value().rawDn.has_value());
    const auto& raw = reference.value().rawDn->image;
    // Independent CODATA Planck integration over 8-14 um at f/2 and 10 um
    // pitch gives dP/dT=1.54829012e-11 W/K. NETD=0.04 K at 1e13 DN/W
    // therefore sets sigma=6.19316 DN, with a small ADC variance.
    EXPECT_NEAR(PopulationVariance(raw), 6.19316049 * 6.19316049, 5.0);
    EXPECT_EQ(raw.metadata.at("camera_netd_reference_k"), "300.000000");
    EXPECT_EQ(raw.metadata.at("camera_netd_reference_bandwidth_hz"), "100.000000");

    config.thermal.readoutWindowSeconds = 0.02; // B=25 Hz, sigma halves.
    CaptureState slowState;
    const auto slow = CpuCameraPipeline(config).Capture(
        slowState, 0.0, BlackbodyRadiance(64, 64, 300.0));
    ASSERT_TRUE(slow.has_value());
    EXPECT_NEAR(PopulationVariance(slow.value().rawDn->image) /
                    PopulationVariance(raw),
                0.25, 0.08);
    config.thermal.readNoiseDnRms = 1.0;
    CaptureState invalidState;
    EXPECT_FALSE(CpuCameraPipeline(config).Capture(
        invalidState, 0.0, BlackbodyRadiance(64, 64, 300.0)).has_value());
}

TEST(CpuCameraPipelineTest, ApparentTemperatureAndCorrectedPowerCanBeRequestedTogether) {
    auto config = ThermalConfig(1, 1);
    config.quality.wavelengthSamples = 64;
    config.products.apparentTemperature = true;
    config.products.correctedDeviceSignal = true;
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, BlackbodyRadiance(1, 1, 300.0));
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().apparentTemperature.has_value());
    ASSERT_TRUE(captured.value().correctedDeviceSignal.has_value());
    EXPECT_NEAR(captured.value().apparentTemperature->image.data[0], 300.0, 0.5);
    EXPECT_EQ(captured.value().apparentTemperature->signal.unit, "K");
    EXPECT_EQ(captured.value().correctedDeviceSignal->signal.unit, "W");
    EXPECT_GT(captured.value().correctedDeviceSignal->image.data[0], 0.0f);
}

TEST(CpuCameraPipelineTest, PngWriterDoesNotEncodeDisplaySrgbTwice) {
    Image encoded(1, 1, 3);
    encoded.data = {0.5f, 0.5f, 0.5f};
    encoded.metadata["camera_signal_kind"] = "display_srgb";
    Image preview = encoded;
    preview.metadata["camera_signal_kind"] = "device_preview_srgb";
    Image linear = encoded;
    linear.metadata.clear();
    const auto temp = UniqueTemporaryDirectory("quantiloom_camera_png_test");
    const auto encodedPath = temp / "quantiloom_camera_encoded_midgray.png";
    const auto previewPath = temp / "quantiloom_camera_preview_midgray.png";
    const auto linearPath = temp / "quantiloom_camera_linear_midgray.png";
    struct RemoveTemporaryDirectory {
        std::filesystem::path path;
        ~RemoveTemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{temp};
    ASSERT_TRUE(ImageIO::WritePNG(encodedPath.string(), encoded));
    ASSERT_TRUE(ImageIO::WritePNG(previewPath.string(), preview));
    ASSERT_TRUE(ImageIO::WritePNG(linearPath.string(), linear));
    const auto encodedRead = ImageIO::ReadImage(encodedPath.string());
    const auto previewRead = ImageIO::ReadImage(previewPath.string());
    const auto linearRead = ImageIO::ReadImage(linearPath.string());
    ASSERT_TRUE(encodedRead.has_value());
    ASSERT_TRUE(previewRead.has_value());
    ASSERT_TRUE(linearRead.has_value());
    const int encodedCode = static_cast<int>(std::lround(encodedRead->data[0] * 255.0f));
    const int previewCode = static_cast<int>(std::lround(previewRead->data[0] * 255.0f));
    const int linearCode = static_cast<int>(std::lround(linearRead->data[0] * 255.0f));
    EXPECT_EQ(encodedCode, 128);
    EXPECT_EQ(previewCode, 128);
    EXPECT_EQ(linearCode, 188);
}

TEST(CpuCameraPipelineTest, AiryBlurSpreadsImpulseWhileConservingInteriorEnergy) {
    const auto config = PhotonConfig(9, 9);
    const SpectralFrameSampler impulse = [](double, double) -> Result<Image, String> {
        Image image(9, 9, 1);
        image(4, 4, 0) = 1e-5f;
        return image;
    };
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(state, 0.0, impulse);
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().bandMeasurement.has_value());
    const auto& band = captured.value().bandMeasurement->image;
    const double expectedTotal = 1e-5 * PhotonRatePerUnitRadiance(config);
    const double total = std::accumulate(band.data.begin(), band.data.end(), 0.0);
    EXPECT_NEAR(total, expectedTotal, expectedTotal * 1e-4);
    EXPECT_GT(band(4, 4, 0), band(4, 3, 0));
    EXPECT_GT(band(4, 3, 0), 0.0f);
    EXPECT_NEAR(band(3, 4, 0), band(5, 4, 0), expectedTotal * 1e-6);
}

TEST(CpuCameraPipelineTest, RollingRowsIntegrateTheirOwnExposureWindows) {
    auto config = PhotonConfig(1, 2);
    config.readout.shutter = ShutterKind::Rolling;
    config.readout.exposureSeconds = 0.02;
    config.readout.rowDelaySeconds = 0.01;
    config.quality.timeSamples = 4;
    const SpectralFrameSampler linearInTime = [](double time, double) ->
        Result<Image, String> {
        Image image(1, 2, 1);
        std::fill(image.data.begin(), image.data.end(), static_cast<f32>(time * 1e-5));
        return image;
    };
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 1.0, linearInTime);
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().bandMeasurement.has_value());
    const auto& band = captured.value().bandMeasurement->image;
    EXPECT_NEAR(band(0, 0, 0), 1e-5 * PhotonRatePerUnitRadiance(config), 0.002);
    EXPECT_NEAR(band(0, 1, 0), 1.01e-5 * PhotonRatePerUnitRadiance(config), 0.002);
    EXPECT_NEAR(captured.value().rawDn->signal.exposureStartSeconds, 0.99, 1e-12);
    EXPECT_NEAR(captured.value().rawDn->signal.exposureEndSeconds, 1.02, 1e-12);
}

TEST(CpuCameraPipelineTest, SignedMonteCarloSpectrumCancelsBeforePhysicalClamp) {
    const auto config = PhotonConfig(1, 1);
    // L(lambda) is linear: +2e-5 at 500 nm, -1e-5 at 600 nm.
    // Integral L(lambda)*lambda d(lambda) = +0.25, so the final device
    // measurement is positive even though one wavelength sample is negative.
    const SpectralFrameSampler signedSpectrum = [](double, double nm) ->
        Result<Image, String> {
        Image image(1, 1, 1);
        image.data[0] = static_cast<f32>(
            2e-5 - 3e-7 * (nm - 500.0));
        return image;
    };
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, signedSpectrum);
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().bandMeasurement.has_value());
    const double omega = kPi / 17.0;
    const double area = 25e-12;
    const double expected = omega * area * 0.5 * 1e-9 * 0.25 / (kH * kC);
    EXPECT_NEAR(captured.value().bandMeasurement->image.data[0],
                expected, expected * 1e-4);
    EXPECT_EQ(captured.value().bandMeasurement->image.metadata.count(
                  "camera_negative_mc_measurement_elements"), 0u);
    EXPECT_EQ(state.acquisitionIndex, 1u);
}

TEST(CpuCameraPipelineTest, NegativeMonteCarloMeasurementIsClampedAndRecorded) {
    const auto config = PhotonConfig(1, 1);
    // Here the same spectral slope begins at +1e-5 and ends at -2e-5;
    // integral L(lambda)*lambda d(lambda) = -0.30, so only the final
    // measurement is clamped to zero.
    const SpectralFrameSampler signedSpectrum = [](double, double nm) ->
        Result<Image, String> {
        Image image(1, 1, 1);
        image.data[0] = static_cast<f32>(
            1e-5 - 3e-7 * (nm - 500.0));
        return image;
    };
    CaptureState state;
    const auto captured = CpuCameraPipeline(config).Capture(
        state, 0.0, signedSpectrum);
    ASSERT_TRUE(captured.has_value());
    ASSERT_TRUE(captured.value().bandMeasurement.has_value());
    const auto& band = *captured.value().bandMeasurement;
    EXPECT_FLOAT_EQ(band.image.data[0], 0.0f);
    EXPECT_EQ(band.image.metadata.at("camera_negative_mc_measurement_elements"), "1");
    EXPECT_GT(std::stod(band.image.metadata.at("camera_negative_mc_rate_sum")), 0.0);
    EXPECT_LT(std::stod(band.image.metadata.at("camera_negative_mc_rate_minimum")), 0.0);
    EXPECT_EQ(state.acquisitionIndex, 1u);

    Image nonphysicalMeasured(1, 1, 1);
    nonphysicalMeasured.data[0] = -1.0f;
    CaptureState directState;
    const auto rejected = CpuCameraPipeline(config).CaptureMeasured(
        directState, 0.0, nonphysicalMeasured);
    EXPECT_FALSE(rejected.has_value());
    EXPECT_EQ(directState.acquisitionIndex, 0u);
}
