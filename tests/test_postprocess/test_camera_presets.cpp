#include <gtest/gtest.h>

#include "postprocess/CameraConfigIO.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CameraPresets.hpp"
#include "postprocess/CpuCameraPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::camera;

namespace {

SpectralFrameSampler UniformRadiance(u32 width, u32 height, f64 radiance) {
    return [=](f64, f64) -> Result<Image, String> {
        Image sample(width, height, 1);
        std::fill(sample.data.begin(), sample.data.end(),
                  static_cast<f32>(radiance));
        return sample;
    };
}

bool NotesContain(const std::vector<PresetParameterNote>& notes,
                  const String& needle) {
    for (const auto& note : notes) {
        if (note.provenance.find(needle) != String::npos ||
            note.source.find(needle) != String::npos ||
            note.value.find(needle) != String::npos)
            return true;
    }
    return false;
}

const ResponseCurve& BaseCurveOf(const CameraConfig& config) {
    return *config.device.channels.front().response.quantumEfficiency;
}

} // namespace

TEST(CameraPresetTest, EveryPresetBuildsAValidConfig) {
    for (const CameraPresetKind kind : AllCameraPresets()) {
        const auto config = MakePresetCameraConfig(kind);
        ASSERT_TRUE(config.has_value())
            << CameraPresetToken(kind) << ": " << config.error();
        const auto valid = ValidateCameraConfig(config.value());
        EXPECT_TRUE(valid.has_value())
            << CameraPresetToken(kind) << ": " << valid.error();
        EXPECT_TRUE(config.value().enabled);
        EXPECT_EQ(config.value().version, kCameraConfigVersion);
        EXPECT_FALSE(CameraPresetDisplayName(kind).empty());
        EXPECT_EQ(CameraPresetFromToken(CameraPresetToken(kind)).value(), kind);
    }
}

TEST(CameraPresetTest, GenericCmosIsPubliclyDefined) {
    const auto config = MakePresetCameraConfig(CameraPresetKind::GenericCmos);
    ASSERT_TRUE(config.has_value());
    const auto& camera = config.value();
    EXPECT_EQ(camera.device.detector, DetectorKind::Photon);
    EXPECT_EQ(camera.device.cfa, CfaPattern::RGGB);
    EXPECT_EQ(camera.device.calibration, CalibrationStatus::GenericAssumption);
    ASSERT_EQ(camera.device.channels.size(), 3u);
    EXPECT_EQ(camera.optics.sensorWidthPx, 320u);
    EXPECT_EQ(camera.optics.sensorHeightPx, 240u);
    EXPECT_EQ(camera.readout.shutter, ShutterKind::Global);
    EXPECT_DOUBLE_EQ(BaseCurveOf(camera).value.front(), 0.5);
}

TEST(CameraPresetTest, AlviumMonoMatchesDatasheetNumbers) {
    const auto config =
        MakePresetCameraConfig(CameraPresetKind::Alvium1800U507Mono);
    ASSERT_TRUE(config.has_value());
    const auto& camera = config.value();
    EXPECT_EQ(camera.optics.sensorWidthPx, 2464u);
    EXPECT_EQ(camera.optics.sensorHeightPx, 2056u);
    EXPECT_DOUBLE_EQ(camera.optics.pixelPitchUm, 3.45);
    EXPECT_EQ(camera.readout.shutter, ShutterKind::Global);
    EXPECT_EQ(camera.readout.outputBits, 12u);
    EXPECT_EQ(camera.readout.adcBits, 12u);
    EXPECT_DOUBLE_EQ(camera.photon.fullWellElectrons, 10400.0);
    EXPECT_DOUBLE_EQ(camera.photon.readNoiseElectronsRms, 2.1);
    // The published QE anchor: 64% at 529 nm.
    const auto& qe = BaseCurveOf(camera);
    ASSERT_EQ(qe.wavelengthNm.size(), 10u);
    const auto it = std::find(qe.wavelengthNm.begin(), qe.wavelengthNm.end(),
                              529.0);
    ASSERT_NE(it, qe.wavelengthNm.end());
    EXPECT_DOUBLE_EQ(qe.value[static_cast<size_t>(it - qe.wavelengthNm.begin())],
                     0.64);
}

TEST(CameraPresetTest, AlviumColorBorrowsMonoNumbersOnlyAsLabelledAssumptions) {
    const auto config =
        MakePresetCameraConfig(CameraPresetKind::Alvium1800U507Color);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config.value().device.cfa, CfaPattern::RGGB);
    ASSERT_EQ(config.value().device.channels.size(), 3u);
    // The borrowed numbers must be flagged, never presented as colour
    // measurements.
    const auto notes = CameraPresetProvenanceNotes(
        CameraPresetKind::Alvium1800U507Color);
    EXPECT_TRUE(NotesContain(notes, "borrowed assumption"));
    EXPECT_TRUE(NotesContain(notes, "not a colour measurement"));
}

TEST(CameraPresetTest, HamamatsuCoolingConditionsSetOnlyDarkCurrent) {
    const auto cold60 =
        MakePresetCameraConfig(CameraPresetKind::HamamatsuC12741_11Minus60C);
    const auto cold70 =
        MakePresetCameraConfig(CameraPresetKind::HamamatsuC12741_11Minus70C);
    ASSERT_TRUE(cold60.has_value());
    ASSERT_TRUE(cold70.has_value());
    for (const auto* camera : {&cold60.value(), &cold70.value()}) {
        EXPECT_EQ(camera->optics.sensorWidthPx, 640u);
        EXPECT_EQ(camera->optics.sensorHeightPx, 512u);
        EXPECT_DOUBLE_EQ(camera->optics.pixelPitchUm, 20.0);
        EXPECT_DOUBLE_EQ(camera->photon.fullWellElectrons, 300000.0);
        EXPECT_DOUBLE_EQ(camera->photon.readNoiseElectronsRms, 500.0);
        EXPECT_EQ(camera->readout.outputBits, 16u);
        // 16-bit is the transport format; the ADC width stays an explicit
        // assumption, never the output width.
        EXPECT_NE(camera->readout.adcBits, camera->readout.outputBits);
    }
    EXPECT_DOUBLE_EQ(cold60.value().photon.darkCurrentElectronsPerSecond,
                     300.0);
    EXPECT_DOUBLE_EQ(cold70.value().photon.darkCurrentElectronsPerSecond,
                     130.0);
}

TEST(CameraPresetTest, FlirA6751StaysOnTheThreeToFiveMicronModel) {
    const auto config = MakePresetCameraConfig(CameraPresetKind::FlirA6751);
    ASSERT_TRUE(config.has_value());
    const auto& camera = config.value();
    EXPECT_EQ(camera.device.detector, DetectorKind::Photon);
    EXPECT_DOUBLE_EQ(camera.optics.pixelPitchUm, 15.0);
    EXPECT_DOUBLE_EQ(camera.optics.fNumber, 2.5);
    EXPECT_EQ(camera.readout.outputBits, 14u);
    // The A6751-specific 3-5 um band: the A6750's 1-5 um broadband must not
    // leak in.
    EXPECT_DOUBLE_EQ(camera.device.effectiveMinNm, 3000.0);
    EXPECT_DOUBLE_EQ(camera.device.effectiveMaxNm, 5000.0);
    const auto& qe = BaseCurveOf(camera);
    EXPECT_DOUBLE_EQ(qe.MinNm(), 3000.0);
    EXPECT_DOUBLE_EQ(qe.MaxNm(), 5000.0);
}

TEST(CameraPresetTest, BosonNetdIsALimitAndTimeConstantAnEstimate) {
    const auto config =
        MakePresetCameraConfig(CameraPresetKind::Boson640Industrial);
    ASSERT_TRUE(config.has_value());
    const auto& camera = config.value();
    EXPECT_EQ(camera.device.detector, DetectorKind::Thermal);
    EXPECT_EQ(camera.optics.sensorWidthPx, 640u);
    EXPECT_EQ(camera.optics.sensorHeightPx, 512u);
    EXPECT_DOUBLE_EQ(camera.optics.pixelPitchUm, 12.0);
    EXPECT_DOUBLE_EQ(camera.device.effectiveMinNm, 8000.0);
    EXPECT_DOUBLE_EQ(camera.device.effectiveMaxNm, 14000.0);
    EXPECT_DOUBLE_EQ(camera.thermal.netdKelvin, 0.040);
    EXPECT_DOUBLE_EQ(camera.thermal.timeConstantSeconds, 0.008);
    // NETD constrains the output noise: independent read noise would
    // double-count it.
    EXPECT_DOUBLE_EQ(camera.thermal.readNoiseDnRms, 0.0);
    const auto notes =
        CameraPresetProvenanceNotes(CameraPresetKind::Boson640Industrial);
    EXPECT_TRUE(NotesContain(notes, "limit"));
    EXPECT_TRUE(NotesContain(notes, "estimated"));
}

TEST(CameraPresetTest, GenericPresetsRunTheCpuReferenceChain) {
    struct Case {
        CameraPresetKind kind;
        f64 radiance;
    };
    const Case cases[] = {
        {CameraPresetKind::GenericCmos, 1e-5},
        {CameraPresetKind::GenericIngaas, 1e-5},
        {CameraPresetKind::GenericCooledPhotonIr, 1e-4},
        {CameraPresetKind::GenericUncooledThermalIr, 10.0},
    };
    for (const auto& test : cases) {
        auto built = MakePresetCameraConfig(test.kind);
        ASSERT_TRUE(built.has_value());
        CameraConfig config = std::move(built.value());
        config.quality.wavelengthSamples = 2;
        config.quality.timeSamples = 1;
        config.quality.pixelSamples = 1;
        config.quality.noiseFree = true;
        const CpuCameraPipeline pipeline(config);
        CaptureState state;
        const auto output = pipeline.Capture(
            state, 0.0,
            UniformRadiance(config.optics.sensorWidthPx,
                            config.optics.sensorHeightPx, test.radiance));
        ASSERT_TRUE(output.has_value())
            << CameraPresetToken(test.kind) << ": " << output.error();
        EXPECT_EQ(state.acquisitionIndex, 1u);
        ASSERT_TRUE(output.value().rawDn.has_value());
        EXPECT_EQ(output.value().rawDn->image.width,
                  config.optics.sensorWidthPx);
        ASSERT_TRUE(output.value().display.has_value());
        // A second acquisition advances the state again.
        const auto second = pipeline.Capture(
            state, config.readout.framePeriodSeconds,
            UniformRadiance(config.optics.sensorWidthPx,
                            config.optics.sensorHeightPx, test.radiance));
        ASSERT_TRUE(second.has_value());
        EXPECT_EQ(state.acquisitionIndex, 2u);
    }
}

TEST(CameraPresetTomlTest, PresetKeyBuildsTheNamedConfig) {
    const auto document = Config::Parse(R"(
[sensor]
preset = "alvium_1800_u507_mono"
)");
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(parsed.has_value());
    const auto& camera = parsed.value();
    EXPECT_TRUE(camera.enabled);
    EXPECT_EQ(camera.device.calibration, CalibrationStatus::HardwareReference);
    EXPECT_EQ(camera.optics.sensorWidthPx, 2464u);
    EXPECT_DOUBLE_EQ(camera.photon.readNoiseElectronsRms, 2.1);
    EXPECT_EQ(camera.device.cfa, CfaPattern::Mono);
}

TEST(CameraPresetTomlTest, ExplicitKeysOverridePresetValues) {
    const auto document = Config::Parse(R"(
[sensor]
preset = "alvium_1800_u507_mono"
[sensor.optics]
focal_length_mm = 35.0
[sensor.photon]
read_noise_e_rms = 5.0
[sensor.exposure]
time_s = 0.02
)");
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_TRUE(parsed.has_value());
    const auto& camera = parsed.value();
    // Overridden values.
    EXPECT_DOUBLE_EQ(camera.optics.focalLengthMm, 35.0);
    EXPECT_DOUBLE_EQ(camera.photon.readNoiseElectronsRms, 5.0);
    EXPECT_DOUBLE_EQ(camera.readout.exposureSeconds, 0.02);
    // Preset values where no explicit key exists.
    EXPECT_DOUBLE_EQ(camera.optics.pixelPitchUm, 3.45);
    EXPECT_EQ(camera.optics.sensorWidthPx, 2464u);
    EXPECT_DOUBLE_EQ(camera.photon.fullWellElectrons, 10400.0);
    EXPECT_EQ(camera.readout.outputBits, 12u);
}

TEST(CameraPresetTomlTest, ExplicitChannelsReplacePresetChannelsWholesale) {
    const auto document = Config::Parse(R"(
[sensor]
preset = "boson_640_industrial"
detector = "photon"
effective_min_nm = 3000.0
effective_max_nm = 5000.0
[[sensor.channels]]
name = "Mono"
[sensor.channels.qe]
kind = "absolute_qe"
wavelength_nm = [3000.0, 5000.0]
value = [0.6, 0.6]
[sensor.optics]
sensor_width_px = 64
sensor_height_px = 64
)");
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::MWIR_Fused);
    ASSERT_TRUE(parsed.has_value());
    const auto& camera = parsed.value();
    ASSERT_EQ(camera.device.channels.size(), 1u);
    EXPECT_DOUBLE_EQ(
        camera.device.channels[0].response.quantumEfficiency->MinNm(), 3000.0);
}

TEST(CameraPresetTomlTest, UnknownPresetNameIsRejected) {
    const auto document = Config::Parse(R"(
[sensor]
preset = "not_a_camera"
)");
    ASSERT_TRUE(document.has_value());
    const auto parsed = ParseCameraConfig(document.value(), SpectralMode::VIS_Hero);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_NE(parsed.error().find("preset"), String::npos);
}

TEST(CameraPresetTomlTest, ExpandedRoundTripIsLossless) {
    const auto document = Config::Parse(R"(
[sensor]
preset = "hamamatsu_c12741_11_m70c"
[sensor.photon]
dark_current_e_s = 131.0
)");
    ASSERT_TRUE(document.has_value());
    const auto first = ParseCameraConfig(document.value(), SpectralMode::SWIR_Fused);
    ASSERT_TRUE(first.has_value());

    const String saved = CameraConfigToToml(first.value());
    // The preset key itself is never written; the config is expanded.
    EXPECT_EQ(saved.find("preset"), String::npos);
    const auto reloaded = Config::Parse(saved);
    ASSERT_TRUE(reloaded.has_value());
    const auto second = ParseCameraConfig(reloaded.value(), SpectralMode::SWIR_Fused);
    ASSERT_TRUE(second.has_value());
    const auto& a = first.value();
    const auto& b = second.value();
    EXPECT_EQ(a.device.id, b.device.id);
    EXPECT_EQ(a.device.calibration, b.device.calibration);
    EXPECT_EQ(a.optics.sensorWidthPx, b.optics.sensorWidthPx);
    EXPECT_EQ(a.optics.sensorHeightPx, b.optics.sensorHeightPx);
    EXPECT_DOUBLE_EQ(a.optics.pixelPitchUm, b.optics.pixelPitchUm);
    EXPECT_DOUBLE_EQ(a.photon.fullWellElectrons, b.photon.fullWellElectrons);
    EXPECT_DOUBLE_EQ(a.photon.readNoiseElectronsRms,
                     b.photon.readNoiseElectronsRms);
    EXPECT_DOUBLE_EQ(a.photon.darkCurrentElectronsPerSecond,
                     b.photon.darkCurrentElectronsPerSecond);
    EXPECT_EQ(a.readout.outputBits, b.readout.outputBits);
    EXPECT_EQ(a.readout.adcBits, b.readout.adcBits);
    EXPECT_DOUBLE_EQ(a.readout.electronsPerDn, b.readout.electronsPerDn);
    ASSERT_EQ(a.device.channels.size(), b.device.channels.size());
    const auto& curveA = BaseCurveOf(a);
    const auto& curveB = BaseCurveOf(b);
    EXPECT_EQ(curveA.wavelengthNm, curveB.wavelengthNm);
    EXPECT_EQ(curveA.value, curveB.value);
    EXPECT_EQ(a.device.parameterSources.size(), b.device.parameterSources.size());
}
