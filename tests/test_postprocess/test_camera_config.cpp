#include <gtest/gtest.h>

#include "postprocess/CameraConfigIO.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <system_error>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

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

TEST(CameraConfigIOTest, LegacyLwirRemainsPhotonAndKeepsItsBandAssumption) {
    const auto document = Config::Parse(R"(
[renderer]
resolution = [640, 512]
[camera]
fov_y = 50.0
[sensor]
enabled = true
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 20.0
quantum_efficiency = 0.65
well_capacity_e = 300000.0
read_noise_e_rms = 500.0
dark_current_e_s = 130.0
integration_time_s = 0.02
bit_depth = 16
gain = 5.0
)");
    ASSERT_TRUE(document.has_value());
    const auto migrated = ParseCameraConfig(document.value(), SpectralMode::LWIR_Fused);
    ASSERT_TRUE(migrated.has_value());
    const auto& camera = migrated.value();
    EXPECT_TRUE(camera.enabled);
    EXPECT_EQ(camera.device.detector, DetectorKind::Photon);
    EXPECT_EQ(camera.device.calibration, CalibrationStatus::GenericAssumption);
    ASSERT_EQ(camera.device.channels.size(), 1u);
    ASSERT_TRUE(camera.device.channels[0].response.quantumEfficiency.has_value());
    const auto& qe = *camera.device.channels[0].response.quantumEfficiency;
    EXPECT_EQ(qe.kind, ResponseKind::AbsoluteQE);
    EXPECT_DOUBLE_EQ(qe.MinNm(), 8000.0);
    EXPECT_DOUBLE_EQ(qe.MaxNm(), 12000.0);
    EXPECT_NEAR(qe.value.front(), 0.65, 1e-7);
    EXPECT_NEAR(qe.value.back(), 0.65, 1e-7);
    EXPECT_NEAR(camera.optics.focalLengthMm,
                512.0 * 20e-3 / (2.0 * std::tan(25.0 * 3.14159265358979323846 / 180.0)),
                1e-5);
    EXPECT_DOUBLE_EQ(camera.photon.fullWellElectrons, 300000.0);
    EXPECT_DOUBLE_EQ(camera.photon.readNoiseElectronsRms, 500.0);
    EXPECT_DOUBLE_EQ(camera.photon.darkCurrentElectronsPerSecond, 130.0);
    EXPECT_NEAR(camera.readout.exposureSeconds, 0.02, 1e-8);
    EXPECT_DOUBLE_EQ(camera.readout.electronsPerDn, 5.0);
    EXPECT_DOUBLE_EQ(camera.readout.analogGain, 1.0);
}

TEST(CameraConfigIOTest, LegacyVisibleInputIsMarkedAsUncalibratedApproximation) {
    const auto document = Config::Parse(R"(
[renderer]
resolution = [320, 240]
[camera]
fov_y = 45.0
[sensor]
enabled = true
quantum_efficiency = 0.8
)");
    ASSERT_TRUE(document.has_value());
    const auto migrated = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(migrated.has_value());
    const auto& camera = migrated.value();
    EXPECT_EQ(camera.device.detector, DetectorKind::Photon);
    EXPECT_FALSE(camera.calibratedFastRgbInput);
    EXPECT_DOUBLE_EQ(camera.fastRgbRadianceScale, 0.0);
    EXPECT_EQ(camera.device.calibration, CalibrationStatus::GenericAssumption);
}

TEST(CameraConfigIOTest, VersionedConfigRoundTripKeepsPhysicalAndDisplaySettings) {
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "measured_cmos"
cfa = "mono"

[sensor.optics]
focal_length_mm = 35.0
f_number = 4.0
pixel_pitch_um = 3.45
sensor_width_px = 640
sensor_height_px = 480

[sensor.exposure]
time_s = 0.02

[sensor.readout]
shutter = "global"
frame_period_s = 0.05
electrons_per_dn = 2.0
adc_bits = 12
output_bits = 16

[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
wavelength_nm = [450.0, 900.0]
value = [0.4, 0.5]
provenance = "manufacturer"
document = "QE datasheet"
document_version = "2026-01"

[isp]
auto_exposure = false
auto_white_balance = false

[isp.hsv]
hue_offset_deg = 15.0
saturation_scale = 0.8
value_gamma = 1.2

[effects.hsv]
empirical_noise = true
temporal_drift = false

[camera.motion]
interpolation = "linear"
extrapolate = "hold"

[[camera.motion.keys]]
t = 0.0
position = [0.0, 0.0, 0.0]
look_at = [0.0, 0.0, -1.0]

[[camera.motion.keys]]
t = 1.5
position = [1.0, 0.0, 0.0]
look_at = [1.0, 0.0, -1.0]
)");
    ASSERT_TRUE(document.has_value());
    const auto first = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(first.has_value());
    ASSERT_EQ(first.value().device.channels.size(), 1u);
    ASSERT_TRUE(first.value().device.channels[0].response.quantumEfficiency.has_value());
    const auto& curve = *first.value().device.channels[0].response.quantumEfficiency;
    // The real device span wins even when the renderer's generic VIS band ends
    // at 780 nm. A later scene sampling stage must report missing coverage.
    EXPECT_DOUBLE_EQ(curve.MinNm(), 450.0);
    EXPECT_DOUBLE_EQ(curve.MaxNm(), 900.0);
    EXPECT_EQ(curve.source.provenance, ValueProvenance::Manufacturer);
    EXPECT_EQ(curve.source.documentVersion, "2026-01");
    EXPECT_DOUBLE_EQ(first.value().isp.hsv.hueOffsetDegrees, 15.0);
    EXPECT_DOUBLE_EQ(first.value().isp.hsv.saturationScale, 0.8);
    EXPECT_DOUBLE_EQ(first.value().isp.hsv.valueGamma, 1.2);
    EXPECT_TRUE(first.value().isp.hsv.empiricalNoise);
    ASSERT_EQ(first.value().motion.keys.size(), 2u);
    EXPECT_DOUBLE_EQ(first.value().motion.keys[1].timeSeconds, 1.5);

    const String saved = CameraConfigToToml(first.value());
    const auto reloadedDocument = Config::Parse(saved);
    ASSERT_TRUE(reloadedDocument.has_value());
    const auto second = ParseCameraConfig(reloadedDocument.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(second.value().device.channels.size(), 1u);
    ASSERT_TRUE(second.value().device.channels[0].response.quantumEfficiency.has_value());
    EXPECT_EQ(second.value().version, first.value().version);
    EXPECT_EQ(second.value().device.id, "measured_cmos");
    EXPECT_DOUBLE_EQ(second.value().optics.focalLengthMm, 35.0);
    EXPECT_DOUBLE_EQ(second.value().optics.pixelPitchUm, 3.45);
    EXPECT_DOUBLE_EQ(second.value().readout.exposureSeconds, 0.02);
    EXPECT_DOUBLE_EQ(second.value().readout.electronsPerDn, 2.0);
    EXPECT_EQ(second.value().readout.adcBits, 12u);
    EXPECT_EQ(second.value().readout.outputBits, 16u);
    EXPECT_EQ(second.value().device.channels[0].response.quantumEfficiency->wavelengthNm,
              curve.wavelengthNm);
    EXPECT_EQ(second.value().device.channels[0].response.quantumEfficiency->value,
              curve.value);
    EXPECT_EQ(second.value().device.channels[0].response.quantumEfficiency->source.document,
              "QE datasheet");
    EXPECT_DOUBLE_EQ(second.value().isp.hsv.hueOffsetDegrees, 15.0);
    EXPECT_TRUE(second.value().isp.hsv.empiricalNoise);
    EXPECT_EQ(second.value().motion.keys, first.value().motion.keys);
}

TEST(CameraConfigIOTest, PreviewResolutionDoesNotChangeVersionedPhysicalSensor) {
    const String sensor = R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "physical_array"
cfa = "mono"
[sensor.optics]
focal_length_mm = 35.0
f_number = 4.0
pixel_pitch_um = 3.45
sensor_width_px = 640
sensor_height_px = 480
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
wavelength_nm = [500.0, 600.0]
value = [0.5, 0.5]
)";
    const auto small = Config::Parse(
        String("[renderer]\nresolution = [320, 240]\n") + sensor);
    const auto large = Config::Parse(
        String("[renderer]\nresolution = [1920, 1080]\n") + sensor);
    ASSERT_TRUE(small.has_value());
    ASSERT_TRUE(large.has_value());
    const auto a = ParseCameraConfig(small.value(), SpectralMode::VIS_Hero);
    const auto b = ParseCameraConfig(large.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(a.value().optics.sensorWidthPx, 640u);
    EXPECT_EQ(b.value().optics.sensorWidthPx, 640u);
    EXPECT_EQ(a.value().optics.sensorHeightPx, 480u);
    EXPECT_EQ(b.value().optics.sensorHeightPx, 480u);
    EXPECT_DOUBLE_EQ(a.value().optics.focalLengthMm, b.value().optics.focalLengthMm);
    EXPECT_DOUBLE_EQ(a.value().optics.pixelPitchUm, b.value().optics.pixelPitchUm);
    EXPECT_DOUBLE_EQ(a.value().optics.fillFactor, b.value().optics.fillFactor);
}

TEST(CameraConfigIOTest, VersionedConfigRejectsDuplicateResponseWavelength) {
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "bad_curve"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 16
sensor_height_px = 16
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
wavelength_nm = [500.0, 500.0]
value = [0.5, 0.5]
)");
    ASSERT_TRUE(document.has_value());
    EXPECT_FALSE(ParseCameraConfig(document.value(), SpectralMode::VIS_Hero).has_value());
}

TEST(CameraConfigIOTest, UnknownNewerCameraSchemaIsExplicitError) {
    const auto document = Config::Parse(R"(
[sensor]
version = 999
enabled = true
)");
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("version"), String::npos);
}

TEST(CameraConfigIOTest, RelativeResponseNeedsAmplitudeSource) {
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "unscaled_shape"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 16
sensor_height_px = 16
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "relative_qe"
normalization = "peak_one"
amplitude = 0.6
wavelength_nm = [500.0, 550.0, 600.0]
value = [0.0, 1.0, 0.0]
)");
    ASSERT_TRUE(document.has_value());
    EXPECT_FALSE(ParseCameraConfig(document.value(), SpectralMode::VIS_Hero).has_value());
}

TEST(CameraConfigIOTest, DuplicateCameraMotionTimesAreRejected) {
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "bad_motion"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 16
sensor_height_px = 16
[camera.motion]
interpolation = "linear"
extrapolate = "hold"
[[camera.motion.keys]]
t = 0.5
position = [0.0, 0.0, 0.0]
look_at = [0.0, 0.0, -1.0]
[[camera.motion.keys]]
t = 0.5
position = [1.0, 0.0, 0.0]
look_at = [1.0, 0.0, -1.0]
)");
    ASSERT_TRUE(document.has_value());
    EXPECT_FALSE(ParseCameraConfig(document.value(), SpectralMode::VIS_Hero).has_value());
}

TEST(CameraConfigIOTest, ResponseFilePathColumnAndSourceSurviveRoundTrip) {
    const auto dir = UniqueTemporaryDirectory("quantiloom_camera_response_test");
    struct RemoveTemporaryDirectory {
        std::filesystem::path path;
        ~RemoveTemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{dir};
    {
        std::ofstream file(dir / "qe.csv");
        file << "wavelength_nm,unused,qe\n"
                "500,9,0.2\n"
                "600,8,0.3\n";
    }
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "file_qe"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 16
sensor_height_px = 16
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
path = "qe.csv"
column = 3
provenance = "digitized"
document = "maker plot"
document_version = "2026"
condition = "25 C"
)");
    ASSERT_TRUE(document.has_value());
    const auto first = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero,
                                         dir.string());
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(first.value().device.channels[0].response.quantumEfficiency.has_value());
    const auto& curve = *first.value().device.channels[0].response.quantumEfficiency;
    EXPECT_EQ(curve.dataPath, "qe.csv");
    EXPECT_EQ(curve.dataColumn, 3u);
    EXPECT_EQ(curve.wavelengthNm, (std::vector<f64>{500.0, 600.0}));
    EXPECT_EQ(curve.value, (std::vector<f64>{0.2, 0.3}));
    EXPECT_EQ(curve.source.provenance, ValueProvenance::Digitized);
    EXPECT_EQ(curve.source.document, "maker plot");
    EXPECT_EQ(curve.source.condition, "25 C");

    const String saved = CameraConfigToToml(first.value());
    const auto reloadedDocument = Config::Parse(saved);
    ASSERT_TRUE(reloadedDocument.has_value());
    const auto second = ParseCameraConfig(reloadedDocument.value(),
                                          SpectralMode::VIS_Hero, dir.string());
    ASSERT_TRUE(second.has_value());
    const auto& restored = *second.value().device.channels[0].response.quantumEfficiency;
    EXPECT_EQ(restored.dataPath, "qe.csv");
    EXPECT_EQ(restored.dataColumn, 3u);
    EXPECT_EQ(restored.wavelengthNm, curve.wavelengthNm);
    EXPECT_EQ(restored.value, curve.value);
    EXPECT_EQ(restored.source.provenance, ValueProvenance::Digitized);
    EXPECT_EQ(restored.source.documentVersion, "2026");
}

TEST(CameraConfigIOTest, StrictResponseFileRejectsMalformedRows) {
    const auto dir = UniqueTemporaryDirectory("quantiloom_camera_bad_response_test");
    struct RemoveTemporaryDirectory {
        std::filesystem::path path;
        ~RemoveTemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{dir};
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "bad_file"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 16
sensor_height_px = 16
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
path = "qe.csv"
column = 2
)");
    ASSERT_TRUE(document.has_value());
    for (const char* body : {
             "500,0.2\nbad\n600,0.3\n",
             "500,0.2\n480,0.3\n",
             "500,0.2\n600,-0.3\n"}) {
        {
            std::ofstream file(dir / "qe.csv", std::ios::trunc);
            file << body;
        }
        EXPECT_FALSE(ParseCameraConfig(document.value(), SpectralMode::VIS_Hero,
                                        dir.string()).has_value());
    }
}

TEST(CameraConfigIOTest, CalibrationMapFilesRemainBoundToTheirPixelValues) {
    const auto dir = UniqueTemporaryDirectory("quantiloom_camera_nuc_map_test");
    struct RemoveTemporaryDirectory {
        std::filesystem::path path;
        ~RemoveTemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{dir};
    {
        std::ofstream gain(dir / "gain.txt");
        gain << "1.0,0.9\n";
        std::ofstream offset(dir / "offset.txt");
        offset << "0.0 2.0\n";
    }
    const auto document = Config::Parse(R"(
[sensor]
version = 1
enabled = true
detector = "photon"
device_id = "nuc_map"
cfa = "mono"
[sensor.optics]
focal_length_mm = 50.0
f_number = 2.8
pixel_pitch_um = 5.0
sensor_width_px = 2
sensor_height_px = 1
[sensor.photon]
apply_nuc = true
nuc_gain_map_path = "gain.txt"
nuc_offset_electrons_map_path = "offset.txt"
)");
    ASSERT_TRUE(document.has_value());
    const auto first = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero,
                                         dir.string());
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first.value().photon.nucGainMapPath, "gain.txt");
    EXPECT_EQ(first.value().photon.nucOffsetElectronsMapPath, "offset.txt");
    EXPECT_EQ(first.value().photon.nucGainMap, (std::vector<f64>{1.0, 0.9}));
    EXPECT_EQ(first.value().photon.nucOffsetElectronsMap, (std::vector<f64>{0.0, 2.0}));
    const auto saved = CameraConfigToToml(first.value());
    const auto reloaded = Config::Parse(saved);
    ASSERT_TRUE(reloaded.has_value());
    const auto second = ParseCameraConfig(reloaded.value(), SpectralMode::VIS_Hero,
                                          dir.string());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second.value().photon.nucGainMap, first.value().photon.nucGainMap);
    EXPECT_EQ(second.value().photon.nucOffsetElectronsMap,
              first.value().photon.nucOffsetElectronsMap);
    EXPECT_EQ(second.value().photon.nucGainMapPath, "gain.txt");

    {
        std::ofstream gain(dir / "gain.txt", std::ios::trunc);
        gain << "1.0\n";
    }
    EXPECT_FALSE(ParseCameraConfig(document.value(), SpectralMode::VIS_Hero,
                                    dir.string()).has_value());
}

TEST(CameraConfigIOTest, NondefaultCameraSettingsSurviveTypedRoundTrip) {
    CameraConfig authored;
    authored.enabled = true;
    authored.device.id = "reference_visible";
    authored.device.displayName = "Reference visible";
    authored.device.documentVersion = "2026.1";
    authored.device.documentUrl = "https://example.invalid/datasheet";
    authored.device.readoutMode = "slow_12bit";
    authored.device.calibration = CalibrationStatus::HardwareReference;
    authored.device.detector = DetectorKind::Photon;
    authored.device.cfa = CfaPattern::Mono;
    authored.device.effectiveMinNm = 500.0;
    authored.device.effectiveMaxNm = 600.0;
    authored.device.parameterSources.push_back({
        "readout.adc_bits", ValueProvenance::Derived,
        "readout drawing", "rev B", "25 C"});
    ResponseCurve qe;
    qe.kind = ResponseKind::RelativeQE;
    qe.normalization = RelativeNormalization::PeakOne;
    qe.wavelengthNm = {500.0, 550.0, 600.0};
    qe.value = {0.0, 1.0, 0.0};
    qe.amplitude = 0.65;
    qe.amplitudeSource = "calibrated peak QE";
    qe.source = {"qe", ValueProvenance::Manufacturer,
                 "QE graph", "rev 2", "25 C"};
    ResponseCurve lens;
    lens.kind = ResponseKind::LensTransmission;
    lens.wavelengthNm = {500.0, 600.0};
    lens.value = {0.8, 0.9};
    lens.source = {"lens", ValueProvenance::Digitized,
                   "lens graph", "rev A", "f/4"};
    ResponseCurve filter;
    filter.kind = ResponseKind::FilterTransmission;
    filter.wavelengthNm = {500.0, 600.0};
    filter.value = {0.7, 0.6};
    ResponseStack stack;
    stack.quantumEfficiency = qe;
    stack.lensTransmission = lens;
    stack.filterTransmission = filter;
    authored.device.channels.push_back({"Mono", stack});

    authored.optics.focalLengthMm = 35.0;
    authored.optics.fNumber = 4.0;
    authored.optics.pixelPitchUm = 3.45;
    authored.optics.fillFactor = 0.75;
    authored.optics.sensorWidthPx = 2;
    authored.optics.sensorHeightPx = 2;
    authored.optics.focusDistanceM = 2.5;
    authored.optics.cosFourthVignetting = true;
    authored.optics.psfSigmaPixelsOverride = 1.5;
    authored.optics.knownPsfPath = "reference_psf.exr";
    authored.readout.shutter = ShutterKind::Rolling;
    authored.readout.exposureSeconds = 0.0075;
    authored.readout.rowDelaySeconds = 0.0005;
    authored.readout.framePeriodSeconds = 0.05;
    authored.readout.analogGain = 1.7;
    authored.readout.electronsPerDn = 2.3;
    authored.readout.blackLevelDn = 32.0;
    authored.readout.adcBits = 12;
    authored.readout.outputBits = 16;
    authored.readout.effectiveBits = 11.2;
    authored.photon.fullWellElectrons = 10000.0;
    authored.photon.darkCurrentElectronsPerSecond = 12.0;
    authored.photon.readNoiseElectronsRms = 2.1;
    authored.photon.prnuSigma = 0.01;
    authored.photon.dsnuElectronsRms = 5.0;
    authored.photon.dsnuReferenceExposureSeconds = 0.0075;
    authored.photon.biasDnRms = 0.5;
    authored.photon.applyNuc = true;
    authored.photon.nucResidualFraction = 0.03;
    authored.photon.enableShotNoise = false;
    authored.photon.enableDarkCurrent = true;
    authored.photon.enableDarkShotNoise = false;
    authored.photon.enableReadNoise = true;
    authored.photon.enableFpn = true;
    authored.photon.nucGainMap = {1.0, 0.9, 1.1, 1.2};
    authored.photon.nucOffsetElectronsMap = {0.0, 1.0, -1.0, 0.5};
    authored.thermal.timeConstantSeconds = 0.015;
    authored.thermal.responsivityDnPerWatt = 1e12;
    authored.thermal.driftDnPerSecond = 0.2;
    authored.thermal.readoutWindowSeconds = 0.005;
    authored.thermal.netdKelvin = 0.04;
    authored.thermal.netdReferenceTemperatureK = 300.0;
    authored.thermal.netdNoiseBandwidthHz = 100.0;
    authored.thermal.netdOpticalCondition = "f/4";
    authored.thermal.nucGainMap = {1.0, 1.1, 0.9, 1.0};
    authored.thermal.nucOffsetDnMap = {0.0, 1.0, -1.0, 0.5};
    authored.quality.backend = ProcessingBackend::GpuPreview;
    authored.quality.noiseFree = true;
    authored.quality.wavelengthSamples = 64;
    authored.quality.timeSamples = 4;
    authored.quality.pixelSamples = 3;
    authored.quality.gpuTimePositions = 12;
    authored.outputColorSpace = OutputColorSpace::DeviceNative;
    authored.products.tracedRadiance = true;
    authored.products.cieLinearSrgb = true;
    authored.products.bandMeasurement = true;
    authored.products.rawDn = true;
    authored.products.correctedDeviceSignal = true;
    authored.products.apparentTemperature = false;
    authored.products.display = false;
    authored.randomSeed = 123456789u;
    authored.calibratedFastRgbInput = true;
    authored.fastRgbRadianceScale = 1.2e-6;
    authored.fastRgbToDevice = {0.8, 0.1, 0.1,
                                0.1, 0.9, 0.0,
                                0.0, 0.2, 0.8};
    authored.isp.autoExposure = true;
    authored.isp.autoWhiteBalance = true;
    authored.isp.whiteBalance = {1.1, 0.9, 1.05};
    authored.isp.deviceToLinearSrgb = {1.2, -0.1, -0.1,
                                        0.0, 1.0, 0.0,
                                        -0.1, 0.0, 1.1};
    authored.isp.denoise = true;
    authored.isp.denoiseStrength = 0.2;
    authored.isp.sharpen = true;
    authored.isp.sharpenStrength = 0.3;
    authored.isp.toneGamma = 1.3;
    authored.isp.clipOutOfGamut = false;
    authored.isp.infraredTone = DisplayToneMode::Clahe;
    authored.isp.infraredPalette = DisplayPalette::Viridis;
    authored.isp.contrastLowPercentile = 2.0;
    authored.isp.contrastHighPercentile = 98.0;
    authored.isp.hsv.hueOffsetDegrees = 17.0;
    authored.isp.hsv.saturationScale = 0.85;
    authored.isp.hsv.valueGamma = 1.15;
    authored.isp.hsv.empiricalNoise = true;
    authored.isp.hsv.temporalDrift = true;
    authored.motion.keys = {{0.013, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}},
                            {1.75, {1.0, 0.0, 0.0}, {1.0, 0.0, -1.0}}};

    const String saved = CameraConfigToToml(authored);
    const auto document = Config::Parse(saved);
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(parsed.has_value());
    const auto& got = parsed.value();
    EXPECT_TRUE(got.enabled);
    EXPECT_EQ(got.device.id, authored.device.id);
    EXPECT_EQ(got.device.displayName, authored.device.displayName);
    EXPECT_EQ(got.device.documentVersion, authored.device.documentVersion);
    EXPECT_EQ(got.device.documentUrl, authored.device.documentUrl);
    EXPECT_EQ(got.device.readoutMode, authored.device.readoutMode);
    EXPECT_EQ(got.device.calibration, authored.device.calibration);
    EXPECT_EQ(got.device.cfa, authored.device.cfa);
    EXPECT_DOUBLE_EQ(got.device.effectiveMinNm, 500.0);
    EXPECT_DOUBLE_EQ(got.device.effectiveMaxNm, 600.0);
    ASSERT_EQ(got.device.parameterSources.size(), 1u);
    EXPECT_EQ(got.device.parameterSources[0].provenance, ValueProvenance::Derived);
    EXPECT_EQ(got.device.parameterSources[0].condition, "25 C");
    ASSERT_EQ(got.device.channels.size(), 1u);
    const auto& gotResponse = got.device.channels[0].response;
    ASSERT_TRUE(gotResponse.quantumEfficiency.has_value());
    ASSERT_TRUE(gotResponse.lensTransmission.has_value());
    ASSERT_TRUE(gotResponse.filterTransmission.has_value());
    EXPECT_EQ(gotResponse.quantumEfficiency->normalization, RelativeNormalization::PeakOne);
    EXPECT_DOUBLE_EQ(gotResponse.quantumEfficiency->amplitude, 0.65);
    EXPECT_EQ(gotResponse.quantumEfficiency->amplitudeSource, "calibrated peak QE");
    EXPECT_EQ(gotResponse.quantumEfficiency->wavelengthNm, qe.wavelengthNm);
    EXPECT_EQ(gotResponse.quantumEfficiency->value, qe.value);
    EXPECT_EQ(gotResponse.quantumEfficiency->source.document, "QE graph");
    EXPECT_EQ(gotResponse.quantumEfficiency->source.parameterPath, "qe");
    EXPECT_EQ(gotResponse.lensTransmission->value, lens.value);
    EXPECT_EQ(gotResponse.filterTransmission->value, filter.value);
    EXPECT_EQ(gotResponse.lensTransmission->source.provenance, ValueProvenance::Digitized);
    EXPECT_DOUBLE_EQ(got.optics.focalLengthMm, authored.optics.focalLengthMm);
    EXPECT_DOUBLE_EQ(got.optics.fNumber, authored.optics.fNumber);
    EXPECT_DOUBLE_EQ(got.optics.pixelPitchUm, authored.optics.pixelPitchUm);
    EXPECT_DOUBLE_EQ(got.optics.fillFactor, authored.optics.fillFactor);
    EXPECT_EQ(got.optics.sensorWidthPx, 2u);
    EXPECT_EQ(got.optics.sensorHeightPx, 2u);
    EXPECT_DOUBLE_EQ(got.optics.focusDistanceM, 2.5);
    EXPECT_TRUE(got.optics.cosFourthVignetting);
    EXPECT_DOUBLE_EQ(got.optics.psfSigmaPixelsOverride, 1.5);
    EXPECT_EQ(got.optics.knownPsfPath, "reference_psf.exr");
    EXPECT_EQ(got.readout.shutter, ShutterKind::Rolling);
    EXPECT_DOUBLE_EQ(got.readout.exposureSeconds, authored.readout.exposureSeconds);
    EXPECT_DOUBLE_EQ(got.readout.rowDelaySeconds, authored.readout.rowDelaySeconds);
    EXPECT_DOUBLE_EQ(got.readout.framePeriodSeconds, authored.readout.framePeriodSeconds);
    EXPECT_DOUBLE_EQ(got.readout.analogGain, authored.readout.analogGain);
    EXPECT_DOUBLE_EQ(got.readout.electronsPerDn, authored.readout.electronsPerDn);
    EXPECT_DOUBLE_EQ(got.readout.blackLevelDn, 32.0);
    EXPECT_EQ(got.readout.adcBits, 12u);
    EXPECT_EQ(got.readout.outputBits, 16u);
    EXPECT_DOUBLE_EQ(got.readout.effectiveBits, 11.2);
    EXPECT_DOUBLE_EQ(got.photon.fullWellElectrons, authored.photon.fullWellElectrons);
    EXPECT_DOUBLE_EQ(got.photon.darkCurrentElectronsPerSecond,
                     authored.photon.darkCurrentElectronsPerSecond);
    EXPECT_DOUBLE_EQ(got.photon.readNoiseElectronsRms, authored.photon.readNoiseElectronsRms);
    EXPECT_DOUBLE_EQ(got.photon.prnuSigma, authored.photon.prnuSigma);
    EXPECT_DOUBLE_EQ(got.photon.dsnuElectronsRms, authored.photon.dsnuElectronsRms);
    EXPECT_DOUBLE_EQ(got.photon.dsnuReferenceExposureSeconds,
                     authored.photon.dsnuReferenceExposureSeconds);
    EXPECT_DOUBLE_EQ(got.photon.biasDnRms, authored.photon.biasDnRms);
    EXPECT_TRUE(got.photon.applyNuc);
    EXPECT_DOUBLE_EQ(got.photon.nucResidualFraction, authored.photon.nucResidualFraction);
    EXPECT_FALSE(got.photon.enableShotNoise);
    EXPECT_TRUE(got.photon.enableDarkCurrent);
    EXPECT_FALSE(got.photon.enableDarkShotNoise);
    EXPECT_TRUE(got.photon.enableReadNoise);
    EXPECT_TRUE(got.photon.enableFpn);
    EXPECT_EQ(got.photon.nucGainMap, authored.photon.nucGainMap);
    EXPECT_EQ(got.photon.nucOffsetElectronsMap, authored.photon.nucOffsetElectronsMap);
    EXPECT_DOUBLE_EQ(got.thermal.timeConstantSeconds, authored.thermal.timeConstantSeconds);
    EXPECT_DOUBLE_EQ(got.thermal.responsivityDnPerWatt,
                     authored.thermal.responsivityDnPerWatt);
    EXPECT_DOUBLE_EQ(got.thermal.driftDnPerSecond, authored.thermal.driftDnPerSecond);
    EXPECT_DOUBLE_EQ(got.thermal.netdKelvin, authored.thermal.netdKelvin);
    EXPECT_DOUBLE_EQ(got.thermal.netdReferenceTemperatureK,
                     authored.thermal.netdReferenceTemperatureK);
    EXPECT_DOUBLE_EQ(got.thermal.netdNoiseBandwidthHz,
                     authored.thermal.netdNoiseBandwidthHz);
    EXPECT_EQ(got.thermal.netdOpticalCondition, authored.thermal.netdOpticalCondition);
    EXPECT_EQ(got.thermal.nucGainMap, authored.thermal.nucGainMap);
    EXPECT_EQ(got.thermal.nucOffsetDnMap, authored.thermal.nucOffsetDnMap);
    EXPECT_EQ(got.quality.backend, ProcessingBackend::GpuPreview);
    EXPECT_TRUE(got.quality.noiseFree);
    EXPECT_EQ(got.quality.wavelengthSamples, 64u);
    EXPECT_EQ(got.quality.timeSamples, 4u);
    EXPECT_EQ(got.quality.pixelSamples, 3u);
    EXPECT_EQ(got.quality.gpuTimePositions, 12u);
    EXPECT_EQ(got.outputColorSpace, OutputColorSpace::DeviceNative);
    EXPECT_TRUE(got.products.tracedRadiance);
    EXPECT_TRUE(got.products.cieLinearSrgb);
    EXPECT_TRUE(got.products.bandMeasurement);
    EXPECT_TRUE(got.products.rawDn);
    EXPECT_TRUE(got.products.correctedDeviceSignal);
    EXPECT_FALSE(got.products.apparentTemperature);
    EXPECT_FALSE(got.products.display);
    EXPECT_EQ(got.randomSeed, 123456789u);
    EXPECT_TRUE(got.calibratedFastRgbInput);
    EXPECT_DOUBLE_EQ(got.fastRgbRadianceScale, authored.fastRgbRadianceScale);
    EXPECT_EQ(got.fastRgbToDevice, authored.fastRgbToDevice);
    EXPECT_TRUE(got.isp.autoExposure);
    EXPECT_TRUE(got.isp.autoWhiteBalance);
    EXPECT_EQ(got.isp.whiteBalance, authored.isp.whiteBalance);
    EXPECT_EQ(got.isp.deviceToLinearSrgb, authored.isp.deviceToLinearSrgb);
    EXPECT_TRUE(got.isp.denoise);
    EXPECT_DOUBLE_EQ(got.isp.denoiseStrength, authored.isp.denoiseStrength);
    EXPECT_TRUE(got.isp.sharpen);
    EXPECT_DOUBLE_EQ(got.isp.sharpenStrength, authored.isp.sharpenStrength);
    EXPECT_DOUBLE_EQ(got.isp.toneGamma, authored.isp.toneGamma);
    EXPECT_FALSE(got.isp.clipOutOfGamut);
    EXPECT_EQ(got.isp.infraredTone, DisplayToneMode::Clahe);
    EXPECT_EQ(got.isp.infraredPalette, DisplayPalette::Viridis);
    EXPECT_DOUBLE_EQ(got.isp.contrastLowPercentile, 2.0);
    EXPECT_DOUBLE_EQ(got.isp.contrastHighPercentile, 98.0);
    EXPECT_DOUBLE_EQ(got.isp.hsv.hueOffsetDegrees, 17.0);
    EXPECT_DOUBLE_EQ(got.isp.hsv.saturationScale, 0.85);
    EXPECT_DOUBLE_EQ(got.isp.hsv.valueGamma, 1.15);
    EXPECT_TRUE(got.isp.hsv.empiricalNoise);
    EXPECT_TRUE(got.isp.hsv.temporalDrift);
    EXPECT_EQ(got.motion.keys, authored.motion.keys);
}
