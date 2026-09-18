#include <gtest/gtest.h>

#include "io/ImageIO.hpp"
#include "postprocess/CameraPhysics.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

constexpr double kH = 6.62607015e-34;
constexpr double kC = 299792458.0;
constexpr double kPi = 3.14159265358979323846;

ResponseCurve Flat(ResponseKind kind, double firstNm, double lastNm, double value) {
    ResponseCurve curve;
    curve.kind = kind;
    curve.wavelengthNm = {firstNm, lastNm};
    curve.value = {value, value};
    return curve;
}

ResponseStack PhotonStack(double qe = 0.5) {
    ResponseStack stack;
    stack.quantumEfficiency = Flat(ResponseKind::AbsoluteQE, 500.0, 600.0, qe);
    return stack;
}

ResponseStack ThermalStack(double absorptance = 0.5) {
    ResponseStack stack;
    stack.thermalAbsorptance =
        Flat(ResponseKind::ThermalAbsorptance, 8000.0, 14000.0, absorptance);
    return stack;
}

} // namespace

TEST(CameraPhysicsTest, FlatSpectrumHasClosedFormPhotonRate) {
    // E_lambda = 2 W/m^2/nm, QE = 1/2, A = 1e-12 m^2.
    // Photon energy is hc/lambda, so the wavelength integral is quadratic.
    const std::array<SpectralIrradianceSample, 2> source = {{{500.0, 2.0}, {600.0, 2.0}}};
    const auto measured = IntegratePhoton(source, PhotonStack(), 1e-12);
    ASSERT_TRUE(measured.has_value());
    const double expected = 2.0 * 0.5 * 1e-12 * 1e-9 *
                            (600.0 * 600.0 - 500.0 * 500.0) /
                            (2.0 * kH * kC);
    EXPECT_NEAR(measured.value().electronRatePerSecond, expected, expected * 1e-10);
    EXPECT_NEAR(measured.value().electronRatePerSecond, 276876411.21484905, 0.03);
}

TEST(CameraPhysicsTest, PhotonRateScalesWithPixelArea) {
    const std::array<SpectralIrradianceSample, 2> source = {{{500.0, 2.0}, {600.0, 2.0}}};
    const auto small = IntegratePhoton(source, PhotonStack(), 1e-12);
    const auto large = IntegratePhoton(source, PhotonStack(), 4e-12);
    ASSERT_TRUE(small.has_value());
    ASSERT_TRUE(large.has_value());
    EXPECT_NEAR(large.value().electronRatePerSecond,
                4.0 * small.value().electronRatePerSecond,
                small.value().electronRatePerSecond * 1e-10);
}

TEST(CameraPhysicsTest, SeparateOpticalFactorsMultiplyOnce) {
    const std::array<SpectralIrradianceSample, 2> source = {{{500.0, 2.0}, {600.0, 2.0}}};
    ResponseStack stack = PhotonStack(0.25);
    stack.lensTransmission = Flat(ResponseKind::LensTransmission, 500.0, 600.0, 0.8);
    stack.filterTransmission = Flat(ResponseKind::FilterTransmission, 500.0, 600.0, 0.5);
    const auto measured = IntegratePhoton(source, stack, 1e-12);
    ASSERT_TRUE(measured.has_value());
    // 0.25 * 0.8 * 0.5 = 0.1 total response, versus 0.5 in the flat reference.
    EXPECT_NEAR(measured.value().electronRatePerSecond, 276876411.21484905 / 5.0, 0.01);
}

TEST(CameraPhysicsTest, IntegratesResponseAndSourceBreakpoints) {
    ResponseStack stack = PhotonStack(1.0);
    stack.quantumEfficiency->wavelengthNm = {500.0, 550.0, 600.0};
    stack.quantumEfficiency->value = {0.0, 1.0, 0.0};
    const std::array<SpectralIrradianceSample, 3> source =
        {{{500.0, 2.0}, {525.0, 2.0}, {600.0, 2.0}}};
    const auto measured = IntegratePhoton(source, stack, 1e-12);
    ASSERT_TRUE(measured.has_value());
    // Symmetric triangular QE has area 50 nm and mean wavelength 550 nm.
    const double expected = 2.0 * 50.0 * 1e-12 * 550e-9 / (kH * kC);
    EXPECT_NEAR(measured.value().electronRatePerSecond, expected, expected * 1e-10);
}

TEST(CameraPhysicsTest, RejectsIncompleteSourceAndOpticalCoverage) {
    const std::array<SpectralIrradianceSample, 2> shortSource = {{{500.0, 2.0}, {590.0, 2.0}}};
    EXPECT_FALSE(IntegratePhoton(shortSource, PhotonStack(), 1e-12).has_value());
    const std::array<SpectralIrradianceSample, 2> complete = {{{500.0, 2.0}, {600.0, 2.0}}};
    auto stack = PhotonStack();
    stack.filterTransmission = Flat(ResponseKind::FilterTransmission, 510.0, 600.0, 1.0);
    EXPECT_FALSE(IntegratePhoton(complete, stack, 1e-12).has_value());
}

TEST(CameraPhysicsTest, RejectsNonfiniteNegativeAndUnorderedSpectralInput) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<SpectralIrradianceSample, 2> nanSource = {{{500.0, nan}, {600.0, 2.0}}};
    const std::array<SpectralIrradianceSample, 2> infSource = {{{500.0, 2.0}, {600.0, inf}}};
    const std::array<SpectralIrradianceSample, 2> negative = {{{500.0, -1.0}, {600.0, 2.0}}};
    const std::array<SpectralIrradianceSample, 3> duplicate =
        {{{500.0, 2.0}, {500.0, 2.0}, {600.0, 2.0}}};
    const std::array<SpectralIrradianceSample, 2> valid = {{{500.0, 2.0}, {600.0, 2.0}}};
    EXPECT_FALSE(IntegratePhoton(nanSource, PhotonStack(), 1e-12).has_value());
    EXPECT_FALSE(IntegratePhoton(infSource, PhotonStack(), 1e-12).has_value());
    EXPECT_FALSE(IntegratePhoton(negative, PhotonStack(), 1e-12).has_value());
    EXPECT_FALSE(IntegratePhoton(duplicate, PhotonStack(), 1e-12).has_value());
    EXPECT_FALSE(IntegratePhoton(valid, PhotonStack(), nan).has_value());
    EXPECT_FALSE(IntegratePhoton(valid, PhotonStack(), inf).has_value());
    EXPECT_FALSE(IntegratePhoton(valid, PhotonStack(), 0.0).has_value());
}

TEST(CameraPhysicsTest, RejectsMalformedResponseAndDoubleCountedSystemResponse) {
    auto stack = PhotonStack();
    stack.quantumEfficiency->wavelengthNm = {500.0, 500.0};
    EXPECT_FALSE(ValidateResponseStack(stack, DetectorKind::Photon).has_value());
    stack = PhotonStack();
    stack.quantumEfficiency->value[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ValidateResponseStack(stack, DetectorKind::Photon).has_value());
    stack = PhotonStack();
    stack.quantumEfficiency->value[0] = -0.1;
    EXPECT_FALSE(ValidateResponseStack(stack, DetectorKind::Photon).has_value());
    stack = PhotonStack();
    stack.quantumEfficiency->value[0] = 1.1;
    EXPECT_FALSE(ValidateResponseStack(stack, DetectorKind::Photon).has_value());
    stack = PhotonStack();
    stack.systemResponse = Flat(ResponseKind::SystemPhotonQE, 500.0, 600.0, 0.5);
    EXPECT_FALSE(ValidateResponseStack(stack, DetectorKind::Photon).has_value());
}

TEST(CameraPhysicsTest, RelativeShapeRequiresDocumentedAmplitudeAndNormalization) {
    ResponseCurve curve;
    curve.kind = ResponseKind::RelativeQE;
    curve.normalization = RelativeNormalization::PeakOne;
    curve.wavelengthNm = {500.0, 550.0, 600.0};
    curve.value = {0.0, 1.0, 0.0};
    curve.amplitude = 0.4;
    EXPECT_FALSE(ValidateResponse(curve).has_value());
    curve.amplitudeSource = "calibrated peak QE";
    EXPECT_TRUE(ValidateResponse(curve).has_value());
    curve.value = {0.0, 0.9, 0.0};
    EXPECT_FALSE(ValidateResponse(curve).has_value());
    curve.value = {0.0, 1.0, 0.0};
    curve.amplitude = 1.1;
    EXPECT_FALSE(ValidateResponse(curve).has_value());
    curve.normalization = RelativeNormalization::AreaOne;
    curve.value = {0.0, 0.02, 0.0}; // Triangle area is 1 nm.
    curve.amplitude = 40.0;        // Integrated QE 40 nm; peak QE 0.8.
    EXPECT_TRUE(ValidateResponse(curve).has_value());
}

TEST(CameraPhysicsTest, AbsoluteResponseCannotBeRenormalized) {
    auto curve = Flat(ResponseKind::AbsoluteQE, 500.0, 600.0, 0.5);
    curve.amplitude = 2.0;
    EXPECT_FALSE(ValidateResponse(curve).has_value());
    curve = Flat(ResponseKind::FilterTransmission, 500.0, 600.0, 0.5);
    curve.normalization = RelativeNormalization::PeakOne;
    EXPECT_FALSE(ValidateResponse(curve).has_value());
}

TEST(CameraPhysicsTest, SystemResponseIsValidOnlyForItsDetectorType) {
    ResponseStack photon;
    photon.systemResponse = Flat(ResponseKind::SystemPhotonQE, 500.0, 600.0, 0.4);
    EXPECT_TRUE(ValidateResponseStack(photon, DetectorKind::Photon).has_value());
    EXPECT_FALSE(ValidateResponseStack(photon, DetectorKind::Thermal).has_value());
    const std::array<SpectralIrradianceSample, 2> visible =
        {{{500.0, 2.0}, {600.0, 2.0}}};
    EXPECT_TRUE(IntegratePhoton(visible, photon, 1e-12).has_value());

    ResponseStack thermal;
    thermal.systemResponse =
        Flat(ResponseKind::SystemThermalAbsorptance, 8000.0, 14000.0, 0.6);
    EXPECT_TRUE(ValidateResponseStack(thermal, DetectorKind::Thermal).has_value());
    EXPECT_FALSE(ValidateResponseStack(thermal, DetectorKind::Photon).has_value());
    thermal.filterTransmission =
        Flat(ResponseKind::FilterTransmission, 8000.0, 14000.0, 0.5);
    EXPECT_FALSE(ValidateResponseStack(thermal, DetectorKind::Thermal).has_value());
}

TEST(CameraPhysicsTest, RadianceConversionUsesFiniteApertureAndCosFourthLaw) {
    const std::array<SpectralRadianceSample, 2> radiance = {{{500.0, 1.0}, {600.0, 2.0}}};
    const auto onAxis = RadianceToIrradiance(radiance, 2.0, 0.0);
    ASSERT_TRUE(onAxis.has_value());
    const double throughput = kPi / 17.0; // pi/(1 + 4*N^2), N=2.
    ASSERT_EQ(onAxis.value().size(), 2u);
    EXPECT_NEAR(onAxis.value()[0].irradianceWm2Nm, throughput, 1e-14);
    EXPECT_NEAR(onAxis.value()[1].irradianceWm2Nm, 2.0 * throughput, 1e-14);
    const auto offAxis = RadianceToIrradiance(radiance, 2.0, kPi / 6.0);
    ASSERT_TRUE(offAxis.has_value());
    EXPECT_NEAR(offAxis.value()[0].irradianceWm2Nm, throughput * 0.5625, 1e-14);
    const auto noVignette = RadianceToIrradiance(radiance, 2.0, kPi / 6.0, false);
    ASSERT_TRUE(noVignette.has_value());
    EXPECT_NEAR(noVignette.value()[0].irradianceWm2Nm, throughput, 1e-14);
}

TEST(CameraPhysicsTest, RadianceConversionRejectsInvalidGeometryAndSpectrum) {
    const std::array<SpectralRadianceSample, 2> valid = {{{500.0, 1.0}, {600.0, 1.0}}};
    const std::array<SpectralRadianceSample, 2> negative = {{{500.0, -1.0}, {600.0, 1.0}}};
    EXPECT_FALSE(RadianceToIrradiance(valid, 0.0, 0.0).has_value());
    EXPECT_FALSE(RadianceToIrradiance(valid, std::numeric_limits<double>::infinity(), 0.0).has_value());
    EXPECT_FALSE(RadianceToIrradiance(valid, 2.0, std::numeric_limits<double>::quiet_NaN()).has_value());
    EXPECT_FALSE(RadianceToIrradiance(negative, 2.0, 0.0).has_value());
}

TEST(CameraPhysicsTest, ThermalPowerUsesAbsorptanceWithoutPhotonEnergy) {
    const std::array<SpectralIrradianceSample, 2> source =
        {{{8000.0, 0.01}, {14000.0, 0.01}}};
    const auto measured = IntegrateThermal(source, ThermalStack(), 1e-10);
    ASSERT_TRUE(measured.has_value());
    EXPECT_NEAR(measured.value().absorbedPowerW, 0.01 * 6000.0 * 0.5 * 1e-10, 1e-20);
    EXPECT_FALSE(IntegratePhoton(source, ThermalStack(), 1e-10).has_value());
}

TEST(CameraPhysicsTest, ThermalStepMatchesFirstOrderStepAndDecay) {
    const auto zeroStep = StepThermalResponse(0.0, 1.0, 0.0, 0.008);
    ASSERT_TRUE(zeroStep.has_value());
    EXPECT_DOUBLE_EQ(zeroStep.value(), 0.0);
    const auto rise = StepThermalResponse(0.0, 1.0, 0.008, 0.008);
    ASSERT_TRUE(rise.has_value());
    EXPECT_NEAR(rise.value(), 1.0 - std::exp(-1.0), 1e-14);
    const auto decay = StepThermalResponse(1.0, 0.0, 0.008, 0.008);
    ASSERT_TRUE(decay.has_value());
    EXPECT_NEAR(decay.value(), std::exp(-1.0), 1e-14);
    EXPECT_FALSE(StepThermalResponse(0.0, 1.0, -0.1, 0.008).has_value());
    EXPECT_FALSE(StepThermalResponse(0.0, 1.0, 0.1, -0.008).has_value());
}

TEST(CameraPhysicsTest, PhysicalFovUsesArrayWidthAndPixelPitch) {
    const auto fov = HorizontalFovRadians(50.0, 20.0, 640);
    ASSERT_TRUE(fov.has_value());
    EXPECT_NEAR(fov.value(), 2.0 * std::atan(0.0128 / 0.1), 1e-14);
    const auto inverse = EffectiveFocalLengthMm(fov.value(), 20.0, 640);
    ASSERT_TRUE(inverse.has_value());
    EXPECT_NEAR(inverse.value(), 50.0, 1e-12);
    EXPECT_FALSE(HorizontalFovRadians(50.0, 20.0, 0).has_value());
    EXPECT_FALSE(HorizontalFovRadians(0.0, 20.0, 640).has_value());
    EXPECT_FALSE(HorizontalFovRadians(50.0, -20.0, 640).has_value());
}

TEST(CameraPhysicsTest, CollectionAreaUsesPhysicalPixelAndFillFactorOnce) {
    OpticsConfig optics;
    optics.pixelPitchUm = 5.0;
    optics.fillFactor = 0.8;
    const auto area = PixelCollectionAreaM2(optics);
    ASSERT_TRUE(area.has_value());
    EXPECT_NEAR(area.value(), 2e-11, 1e-25);
    optics.fillFactor = 0.0;
    EXPECT_FALSE(PixelCollectionAreaM2(optics).has_value());

    CameraConfig config;
    config.optics.sensorWidthPx = 640;
    config.optics.sensorHeightPx = 512;
    config.optics.fillFactor = 0.8;
    config.device.channels.push_back({"mono", PhotonStack()});
    ASSERT_TRUE(ValidateCameraConfig(config).has_value());
    config.device.channels[0].response.quantumEfficiency->includesPixelFillFactor = true;
    EXPECT_FALSE(ValidateCameraConfig(config).has_value());
    config.device.channels[0].response = {};
    config.device.channels[0].response.systemResponse =
        Flat(ResponseKind::SystemPhotonQE, 500.0, 600.0, 0.5);
    EXPECT_FALSE(ValidateCameraConfig(config).has_value());
}

TEST(CameraPhysicsTest, AiryPeakAndFirstDarkRingAreDimensionless) {
    const auto peak = AiryIntensityNormalized(0.0, 550.0, 2.8);
    ASSERT_TRUE(peak.has_value());
    EXPECT_DOUBLE_EQ(peak.value(), 1.0);
    const double firstZero = 3.8317059702075125 * 550e-9 * 2.8 / kPi;
    const auto ring = AiryIntensityNormalized(firstZero, 550.0, 2.8);
    ASSERT_TRUE(ring.has_value());
    EXPECT_LT(std::abs(ring.value()), 1e-12);
    EXPECT_FALSE(AiryIntensityNormalized(0.0, 0.0, 2.8).has_value());
    EXPECT_FALSE(AiryIntensityNormalized(-1.0, 550.0, 2.8).has_value());
}

TEST(CameraPhysicsTest, AiryPsfHasUnitAreaNormalizationAtItsPeak) {
    // Integrating [2 J1(x)/x]^2 over the image plane gives
    // 4*(lambda*N)^2/pi, so its inverse is the unit-area density scale.
    const double scale = 550e-9 * 2.8;
    const auto peakDensity = AiryPsfPerSquareMeter(0.0, 550.0, 2.8);
    ASSERT_TRUE(peakDensity.has_value());
    EXPECT_NEAR(peakDensity.value(), kPi / (4.0 * scale * scale),
                peakDensity.value() * 1e-12);
    const auto tinyPixel = AiryPixelFraction(0.0, 0.0, 1e-8, 550.0, 2.8, 4);
    ASSERT_TRUE(tinyPixel.has_value());
    EXPECT_NEAR(tinyPixel.value(), peakDensity.value() * 1e-16,
                peakDensity.value() * 1e-16 * 1e-4);
}

TEST(CameraPhysicsTest, PlanckValueAndDerivativeHaveIndependentClosedForms) {
    // CODATA 2018 constants, T=300 K, lambda=10 um. Units are per nm.
    const auto value = PlanckRadianceWm2SrNm(10000.0, 300.0);
    const auto derivative = PlanckDerivativeWm2SrNmPerK(10000.0, 300.0);
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(derivative.has_value());
    EXPECT_NEAR(value.value(), 0.009924033330070694, 5e-14);
    EXPECT_NEAR(derivative.value(), 0.00015997156725132191, 5e-15);
    EXPECT_FALSE(PlanckRadianceWm2SrNm(10000.0, 0.0).has_value());
}

TEST(CameraPhysicsTest, BlackbodyResponseDerivativeMatchesSymmetricDifference) {
    const auto low = IntegrateBlackbodyThermal(299.9, ThermalStack(), 2.0, 1e-10);
    const auto high = IntegrateBlackbodyThermal(300.1, ThermalStack(), 2.0, 1e-10);
    const auto derivative = BlackbodyThermalDerivativeWPerK(300.0, ThermalStack(), 2.0, 1e-10);
    ASSERT_TRUE(low.has_value());
    ASSERT_TRUE(high.has_value());
    ASSERT_TRUE(derivative.has_value());
    const double symmetricDifference = (high.value().absorbedPowerW -
                                        low.value().absorbedPowerW) / 0.2;
    EXPECT_NEAR(derivative.value(), symmetricDifference,
                std::abs(symmetricDifference) * 2e-6);
    EXPECT_GT(derivative.value(), 0.0);
}

TEST(CameraPhysicsTest, NarrowFilterBlackbodyIntegralResolvesItsOwnBreakpoints) {
    ResponseStack stack;
    stack.thermalAbsorptance =
        Flat(ResponseKind::ThermalAbsorptance, 8000.0, 14000.0, 1.0);
    ResponseCurve filter;
    filter.kind = ResponseKind::FilterTransmission;
    filter.wavelengthNm = {8000.0, 10000.0, 10001.0, 10002.0, 14000.0};
    filter.value = {0.0, 0.0, 1.0, 0.0, 0.0};
    stack.filterTransmission = filter;
    const auto measured = IntegrateBlackbodyThermal(300.0, stack, 2.0, 1e-10);
    ASSERT_TRUE(measured.has_value());
    // A 2 nm wide triangular filter has exactly 1 nm area. Planck radiance
    // changes negligibly over that span, so the center value is an independent
    // narrowband reference within 0.1 percent.
    const double centerRadiance = 0.009923870239569406;
    const double expected = centerRadiance * (kPi / 17.0) * 1e-10;
    EXPECT_NEAR(measured.value().absorbedPowerW, expected, expected * 0.001);
}

TEST(CameraPhysicsTest, CameraMotionRejectsDegenerateViewDirection) {
    CameraConfig config;
    config.optics.sensorWidthPx = 640;
    config.optics.sensorHeightPx = 512;
    config.device.channels.push_back({"mono", PhotonStack()});
    config.motion.keys.push_back({0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}});
    ASSERT_TRUE(ValidateCameraConfig(config).has_value());
    config.motion.keys[0].lookAt = config.motion.keys[0].position;
    EXPECT_FALSE(ValidateCameraConfig(config).has_value());
    config.motion.keys[0].lookAt = {0.0, 0.0, -1.0};
    config.motion.keys.push_back({1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}});
    // Each endpoint is valid, but linear interpolation makes the view vector
    // exactly zero at the midpoint.
    EXPECT_FALSE(ValidateCameraConfig(config).has_value());
}

TEST(CameraPhysicsTest, ProductMetadataRetainsWavelengthAxisAndSignalUnits) {
    CameraProduct spectral;
    spectral.image = Image(2, 2, 2);
    spectral.signal.kind = SignalKind::SpectralRadiance;
    spectral.signal.unit = "W/m^2/sr/nm";
    spectral.signal.channelWavelengthNm = {550.0};
    EXPECT_FALSE(AnnotateProductMetadata(spectral).has_value());
    spectral.signal.channelWavelengthNm = {550.0, 650.0};
    ASSERT_TRUE(AnnotateProductMetadata(spectral).has_value());
    EXPECT_EQ(spectral.image.metadata.at("camera_signal_kind"), "spectral_radiance");
    EXPECT_EQ(spectral.image.metadata.at("camera_channel_0_wavelength_nm"), "550.000000");
    EXPECT_EQ(spectral.image.metadata.at("camera_channel_1_wavelength_nm"), "650.000000");

    CameraProduct band;
    band.image = Image(2, 2, 1);
    band.signal.kind = SignalKind::BandMeasurement;
    band.signal.unit = "e-/s";
    band.signal.responseProfileId = "generic_cmos";
    band.signal.responseMinNm = 500.0;
    band.signal.responseMaxNm = 600.0;
    ASSERT_TRUE(AnnotateProductMetadata(band).has_value());
    EXPECT_EQ(band.image.metadata.at("camera_unit"), "e-/s");
    band.signal.unit = "W/m^2/sr/nm";
    EXPECT_FALSE(AnnotateProductMetadata(band).has_value());

    CameraProduct nativeRgb;
    nativeRgb.image = Image(1, 1, 3);
    nativeRgb.signal.kind = SignalKind::BandMeasurement;
    nativeRgb.signal.unit = "e-/s";
    nativeRgb.signal.responseProfileId = "rgb_cfa";
    nativeRgb.signal.responseMinNm = 400.0;
    nativeRgb.signal.responseMaxNm = 700.0;
    nativeRgb.signal.channelsPerPixel = 3;
    nativeRgb.signal.channelResponseIds = {"R", "G", "B"};
    nativeRgb.signal.channelResponseSpanNm = {{580.0, 700.0},
                                               {480.0, 630.0},
                                               {400.0, 520.0}};
    ASSERT_TRUE(AnnotateProductMetadata(nativeRgb).has_value());
    EXPECT_EQ(nativeRgb.image.metadata.at("camera_channel_0_response_id"), "R");
    EXPECT_EQ(nativeRgb.image.metadata.at("camera_channel_2_response_id"), "B");
}

TEST(CameraPhysicsTest, ExrChannelSortingKeepsWavelengthWithNamedData) {
    CameraProduct spectral;
    spectral.image = Image(1, 1, 2);
    spectral.image.channelNames = {"Z_500", "A_600"};
    spectral.image(0, 0, 0) = 5.0f;
    spectral.image(0, 0, 1) = 6.0f;
    spectral.signal.kind = SignalKind::SpectralRadiance;
    spectral.signal.unit = "W/m^2/sr/nm";
    spectral.signal.channelWavelengthNm = {500.0, 600.0};
    ASSERT_TRUE(AnnotateProductMetadata(spectral).has_value());

    const auto path = std::filesystem::temp_directory_path() /
                      "quantiloom_camera_named_channels.exr";
    struct RemoveTemporaryFile {
        std::filesystem::path path;
        ~RemoveTemporaryFile() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{path};
    ASSERT_TRUE(ImageIO::WriteEXR(path.string(), spectral.image));
    const auto loaded = ImageIO::ReadEXR(path.string());
    ASSERT_TRUE(loaded.has_value());
    ASSERT_EQ(loaded->channels, 2u);
    for (u32 i = 0; i < 2; ++i) {
        const String prefix = "camera_channel_" + std::to_string(i) + "_";
        const String name = loaded->metadata.at(prefix + "name");
        const u32 channel = loaded->ChannelIndex(name, loaded->channels);
        ASSERT_LT(channel, loaded->channels);
        EXPECT_DOUBLE_EQ(std::stod(loaded->metadata.at(prefix + "wavelength_nm")),
                         i == 0 ? 500.0 : 600.0);
        EXPECT_FLOAT_EQ((*loaded)(0, 0, channel), i == 0 ? 5.0f : 6.0f);
    }
}

TEST(CameraPhysicsTest, CounterRandomIsRepeatableAndFixedPatternIgnoresCapture) {
    const auto first = CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::PhotonShot, 0u);
    // Golden values computed independently with Python uint32 modular arithmetic.
    EXPECT_EQ(first, 0xf3472877u);
    EXPECT_EQ(CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::FixedPrnu, 0u),
              0x3c64784eu);
    EXPECT_EQ(CounterRandomU32(0x548Cu, 1234u, 0x100000007ull,
                               NoiseClass::PhotonShot, 0u), 0x0877f273u);
    EXPECT_EQ(CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::PhotonShot, 1u),
              0x7b3e1d5fu);
    EXPECT_EQ(first, CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::PhotonShot, 0u));
    EXPECT_NE(first, CounterRandomU32(0x548Cu, 1234u, 8u, NoiseClass::PhotonShot, 0u));
    EXPECT_NE(first, CounterRandomU32(0x548Cu, 1235u, 7u, NoiseClass::PhotonShot, 0u));
    EXPECT_EQ(CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::FixedPrnu, 0u),
              CounterRandomU32(0x548Cu, 1234u, 999u, NoiseClass::FixedPrnu, 0u));
    EXPECT_EQ(CounterRandomU32(0x548Cu, 1234u, 7u, NoiseClass::FixedDsnu, 0u),
              CounterRandomU32(0x548Cu, 1234u, 999u, NoiseClass::FixedDsnu, 0u));
    const double uniform = CounterUniform01(0x548Cu, 1234u, 7u, NoiseClass::Read, 0u);
    EXPECT_GE(uniform, 0.0);
    EXPECT_LT(uniform, 1.0);
}
