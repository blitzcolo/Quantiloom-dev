#include "postprocess/CameraPresets.hpp"

#include <array>
#include <utility>
#include <vector>

namespace quantiloom::camera {
namespace {

// Source documents, checked 2026-09 against the maker pages named in
// plan-0919 section 3. The per-parameter grades live in
// CameraPresetProvenanceNotes().
constexpr const char* kAlviumDatasheet =
    "https://www.alliedvision.com/assets/support/Camera-Documentation/"
    "Allied-Vision/Cameras/Alvium/Alvium-USB/Data-Sheets/"
    "Alvium_1800_U-507_DataSheet_en.pdf";
constexpr const char* kHamamatsuPage =
    "https://www.hamamatsu.com/jp/en/product/cameras/ingaas-cameras/"
    "C12741-11.html";
constexpr const char* kFlirA6751Page = "https://www.flir.com/products/a6750-mwir";
constexpr const char* kFlirBosonPage = "https://oem.flir.com/products/boson";
constexpr const char* kFlirBosonFaq =
    "https://flir.custhelp.com/app/answers/print/a_id/3128";

using Note = PresetParameterNote;

Note NoteOf(const String& parameter, const String& value,
            const String& provenance, const String& source) {
    return Note{parameter, value, provenance, source};
}

ParameterSource Source(const char* path, ValueProvenance provenance,
                       const char* document, const char* version,
                       const char* condition) {
    return ParameterSource{path, provenance, document, version, condition};
}

ResponseCurve Curve(ResponseKind kind, std::initializer_list<f64> wavelengthNm,
                    std::initializer_list<f64> value, ParameterSource source) {
    ResponseCurve curve;
    curve.kind = kind;
    curve.wavelengthNm.assign(wavelengthNm.begin(), wavelengthNm.end());
    curve.value.assign(value.begin(), value.end());
    curve.source = std::move(source);
    return curve;
}

ChannelProfile PhotonChannel(const char* name, ResponseCurve qe) {
    ChannelProfile channel;
    channel.name = name;
    channel.response.quantumEfficiency = std::move(qe);
    return channel;
}

ChannelProfile ThermalChannel(const char* name, ResponseCurve absorptance) {
    ChannelProfile channel;
    channel.name = name;
    channel.response.thermalAbsorptance = std::move(absorptance);
    return channel;
}

void SetMonoPhotonChannel(CameraConfig& config, ResponseCurve qe) {
    config.device.channels.clear();
    config.device.channels.push_back(PhotonChannel("Mono", std::move(qe)));
}

void SetMonoThermalChannel(CameraConfig& config, ResponseCurve absorptance) {
    config.device.channels.clear();
    config.device.channels.push_back(
        ThermalChannel("Mono", std::move(absorptance)));
}

// Shared versioned skeleton for every preset: enabled, spectral input and
// identity ISP. Curves are defined inclusive of the pixel, so the optics fill
// factor stays at one.
CameraConfig Base(const char* id, const char* displayName,
                  CalibrationStatus calibration, DetectorKind detector,
                  CfaPattern cfa) {
    CameraConfig config;
    config.enabled = true;
    config.version = kCameraConfigVersion;
    config.device.id = id;
    config.device.displayName = displayName;
    config.device.calibration = calibration;
    config.device.detector = detector;
    config.device.cfa = cfa;
    return config;
}

// ---------------------------------------------------------------------------
// Generic closed-form families. Every value here is a modelling definition
// published by this library, not a device measurement; the response curves
// are flat on purpose so closed-form tests can integrate them by hand.
// ---------------------------------------------------------------------------

CameraConfig GenericCmosConfig() {
    auto config = Base("generic_cmos", "Generic CMOS (visible, Bayer, global)",
                       CalibrationStatus::GenericAssumption,
                       DetectorKind::Photon, CfaPattern::RGGB);
    config.device.documentVersion = "quantiloom-generic-1";
    config.device.readoutMode = "global_shutter";
    const auto qeSource = Source("device.channels.response",
                                 ValueProvenance::Assumed,
                                 "Quantiloom generic CMOS definition", "1",
                                 "flat response over 400-700 nm");
    for (const char* name : {"R", "G", "B"}) {
        config.device.channels.push_back(PhotonChannel(
            name, Curve(ResponseKind::AbsoluteQE, {400.0, 700.0}, {0.5, 0.5},
                        qeSource)));
    }
    config.optics.focalLengthMm = 8.0;
    config.optics.fNumber = 4.0;
    config.optics.pixelPitchUm = 5.0;
    config.optics.sensorWidthPx = 320;
    config.optics.sensorHeightPx = 240;
    config.readout.exposureSeconds = 0.01;
    config.readout.framePeriodSeconds = 1.0 / 30.0;
    config.readout.adcBits = 12;
    config.readout.outputBits = 12;
    config.readout.electronsPerDn = 5.0; // 20000 e- over 4095 DN.
    config.photon.fullWellElectrons = 20000.0;
    config.photon.darkCurrentElectronsPerSecond = 10.0;
    config.photon.readNoiseElectronsRms = 3.0;
    config.device.effectiveMinNm = 400.0;
    config.device.effectiveMaxNm = 700.0;
    return config;
}

CameraConfig GenericIngaasConfig() {
    auto config = Base("generic_ingaas", "Generic InGaAs (SWIR, mono)",
                       CalibrationStatus::GenericAssumption,
                       DetectorKind::Photon, CfaPattern::Mono);
    config.device.documentVersion = "quantiloom-generic-1";
    config.device.readoutMode = "global_shutter";
    SetMonoPhotonChannel(config, Curve(
        ResponseKind::AbsoluteQE, {950.0, 1700.0}, {0.7, 0.7},
        Source("device.channels.response", ValueProvenance::Assumed,
               "Quantiloom generic InGaAs definition", "1",
               "flat response over 950-1700 nm")));
    config.optics.focalLengthMm = 25.0;
    config.optics.fNumber = 2.0;
    config.optics.pixelPitchUm = 20.0;
    config.optics.sensorWidthPx = 320;
    config.optics.sensorHeightPx = 240;
    config.readout.exposureSeconds = 0.005;
    config.readout.framePeriodSeconds = 1.0 / 30.0;
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    config.readout.electronsPerDn = 12.5; // 100000 e- over 8191 DN.
    config.photon.fullWellElectrons = 100000.0;
    config.photon.darkCurrentElectronsPerSecond = 500.0;
    config.photon.readNoiseElectronsRms = 200.0;
    config.device.effectiveMinNm = 950.0;
    config.device.effectiveMaxNm = 1700.0;
    return config;
}

CameraConfig GenericCooledPhotonIrConfig() {
    auto config = Base("generic_cooled_photon_ir",
                       "Generic cooled photon IR (MWIR, mono)",
                       CalibrationStatus::GenericAssumption,
                       DetectorKind::Photon, CfaPattern::Mono);
    config.device.documentVersion = "quantiloom-generic-1";
    config.device.readoutMode = "snapshot_iwr";
    SetMonoPhotonChannel(config, Curve(
        ResponseKind::AbsoluteQE, {3000.0, 5000.0}, {0.6, 0.6},
        Source("device.channels.response", ValueProvenance::Assumed,
               "Quantiloom generic cooled IR definition", "1",
               "flat response over 3-5 um")));
    config.optics.focalLengthMm = 50.0;
    config.optics.fNumber = 2.5;
    config.optics.pixelPitchUm = 15.0;
    config.optics.sensorWidthPx = 320;
    config.optics.sensorHeightPx = 240;
    config.readout.exposureSeconds = 0.002;
    config.readout.framePeriodSeconds = 1.0 / 60.0;
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    config.readout.electronsPerDn = 122.0; // 1e6 e- over 8191 DN.
    config.photon.fullWellElectrons = 1000000.0;
    config.photon.darkCurrentElectronsPerSecond = 20000.0;
    config.photon.readNoiseElectronsRms = 500.0;
    config.device.effectiveMinNm = 3000.0;
    config.device.effectiveMaxNm = 5000.0;
    return config;
}

CameraConfig GenericUncooledThermalIrConfig() {
    auto config = Base("generic_uncooled_thermal_ir",
                       "Generic uncooled thermal IR (LWIR VOx, mono)",
                       CalibrationStatus::GenericAssumption,
                       DetectorKind::Thermal, CfaPattern::Mono);
    config.device.documentVersion = "quantiloom-generic-1";
    config.device.readoutMode = "microbolometer_integrate_then_read";
    SetMonoThermalChannel(config, Curve(
        ResponseKind::ThermalAbsorptance, {8000.0, 14000.0}, {0.8, 0.8},
        Source("device.channels.response", ValueProvenance::Assumed,
               "Quantiloom generic uncooled thermal definition", "1",
               "flat absorptance over 8-14 um")));
    config.optics.focalLengthMm = 19.0;
    config.optics.fNumber = 1.4;
    config.optics.pixelPitchUm = 12.0;
    config.optics.sensorWidthPx = 320;
    config.optics.sensorHeightPx = 240;
    config.readout.exposureSeconds = 1.0 / 30.0;
    config.readout.framePeriodSeconds = 1.0 / 30.0;
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    config.thermal.timeConstantSeconds = 0.012;
    config.thermal.responsivityDnPerWatt = 1.0e13;
    config.thermal.readoutWindowSeconds = 1.0 / 30.0;
    config.thermal.netdKelvin = 0.05;
    config.thermal.netdReferenceTemperatureK = 300.0;
    config.thermal.netdNoiseBandwidthHz = 30.0;
    config.thermal.netdOpticalCondition = "f/1.4, reference conditions assumed";
    config.device.effectiveMinNm = 8000.0;
    config.device.effectiveMaxNm = 14000.0;
    return config;
}

// ---------------------------------------------------------------------------
// Alvium 1800 U-507 (Sony IMX264). The datasheet's EMVA 1288 imaging
// performance (QE 64 % @ 529 nm, temporal dark noise 2.1 e-, saturation
// capacity 10400 e-) is measured on the MONOCHROME model without optical
// filter; the colour preset reuses those numbers only as labelled borrowed
// assumptions, never as colour measurements.
// ---------------------------------------------------------------------------

ResponseCurve Imx264Qe(ValueProvenance provenance, const char* document,
                       const char* condition) {
    // Shape digitized from the datasheet QE plot, anchored to the published
    // 64 % at 529 nm.
    return Curve(ResponseKind::AbsoluteQE,
                 {400.0, 450.0, 500.0, 529.0, 570.0, 600.0, 650.0, 700.0,
                  750.0, 800.0},
                 {0.30, 0.48, 0.60, 0.64, 0.66, 0.64, 0.58, 0.45, 0.28, 0.12},
                 Source("device.channels.response", provenance, document,
                        "Alvium 1800 U-507 datasheet", condition));
}

CameraConfig AlviumConfig(bool colour) {
    auto config = Base(colour ? "alliedvision_alvium_1800_u507_color"
                              : "alliedvision_alvium_1800_u507_mono",
                       colour ? "Allied Vision Alvium 1800 U-507 (colour)"
                              : "Allied Vision Alvium 1800 U-507 (mono)",
                       CalibrationStatus::HardwareReference,
                       DetectorKind::Photon,
                       colour ? CfaPattern::RGGB : CfaPattern::Mono);
    config.device.documentUrl = kAlviumDatasheet;
    config.device.documentVersion = "Alvium 1800 U-507 datasheet (EMVA 1288 R3.1 typicals)";
    config.device.readoutMode = "global_shutter_12bit";
    const char* noiseGrade = colour ? "borrowed assumption" : "manufacturer";
    const char* noiseCondition = colour
        ? "mono model measurement borrowed for the colour variant; not a colour measurement"
        : "typical, EMVA 1288, monochrome model, no optical filter";

    if (colour) {
        // The QE curve is borrowed unchanged: the Bayer dye transmission is
        // not published, so scaling by invented filter spectra would add
        // false precision. The provenance notes say so explicitly.
        const auto qe = Imx264Qe(ValueProvenance::Assumed, kAlviumDatasheet,
                                 noiseCondition);
        for (const char* name : {"R", "G", "B"})
            config.device.channels.push_back(PhotonChannel(name, qe));
    } else {
        SetMonoPhotonChannel(config,
                             Imx264Qe(ValueProvenance::Digitized,
                                      kAlviumDatasheet,
                                      "typical QE, monochrome model"));
    }

    config.optics.focalLengthMm = 25.0; // Body-only camera; lens is a choice.
    config.optics.fNumber = 2.8;
    config.optics.pixelPitchUm = 3.45;
    config.optics.sensorWidthPx = 2464;
    config.optics.sensorHeightPx = 2056;
    config.readout.shutter = ShutterKind::Global;
    config.readout.exposureSeconds = 0.01;
    config.readout.framePeriodSeconds = 1.0 / 30.0;
    config.readout.adcBits = 12;
    config.readout.outputBits = 12;
    config.readout.electronsPerDn = 10400.0 / 4095.0; // Saturation over DN range.
    config.photon.fullWellElectrons = 10400.0;        // Saturation capacity.
    config.photon.darkCurrentElectronsPerSecond = 50.0; // Not published; assumed.
    config.photon.readNoiseElectronsRms = 2.1;
    config.photon.prnuSigma = 0.005; // Not published; assumed small.
    config.device.effectiveMinNm = 400.0;
    config.device.effectiveMaxNm = 800.0;
    config.device.parameterSources = {
        Source("optics.sensor_width_px", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "IMX264"),
        Source("optics.sensor_height_px", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "IMX264"),
        Source("optics.pixel_pitch_um", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "IMX264"),
        Source("readout.shutter", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "global shutter"),
        Source("readout.output_bits", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "12-bit output"),
        Source("photon.full_well_e", ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", "saturation capacity, mono"),
        Source("photon.read_noise_e_rms", colour ? ValueProvenance::Assumed
                                                 : ValueProvenance::Manufacturer,
               kAlviumDatasheet, "datasheet", noiseCondition),
        Source("photon.dark_current_e_s", ValueProvenance::Assumed,
               kAlviumDatasheet, "datasheet",
               "not published for U-507; small room-temperature value assumed"),
        Source("optics.focal_length_mm", ValueProvenance::Assumed,
               kAlviumDatasheet, "datasheet",
               "body-only camera; 25 mm C-mount lens assumed"),
        Source("readout.electrons_per_dn", ValueProvenance::Derived,
               kAlviumDatasheet, "datasheet",
               "saturation capacity divided by 12-bit DN range"),
    };
    return config;
}

// ---------------------------------------------------------------------------
// Hamamatsu C12741-11, two cooling conditions of the same head. The 16-bit
// figure on the product page is the digital output (transport) format; the
// ADC width is not published, so it is set to 14 bits and labelled assumed
// rather than silently treating the transport width as the ADC width.
// ---------------------------------------------------------------------------

CameraConfig C12741Config(bool minus70C) {
    auto config = Base(minus70C ? "hamamatsu_c12741_11_m70c"
                                : "hamamatsu_c12741_11_m60c",
                       minus70C ? "Hamamatsu C12741-11 (-70 C water cooled)"
                                : "Hamamatsu C12741-11 (-60 C forced-air)",
                       CalibrationStatus::HardwareReference,
                       DetectorKind::Photon, CfaPattern::Mono);
    config.device.documentUrl = kHamamatsuPage;
    config.device.documentVersion = "Hamamatsu C12741-11 product page";
    config.device.readoutMode = "global_shutter";

    // Typical Hamamatsu InGaAs absolute-response shape over 0.9-1.75 um. The
    // spectral-response graph publishes the SHAPE; the absolute QE peak is
    // not given for this model, so 0.75 at 1.5 um is an explicit assumption.
    SetMonoPhotonChannel(config, Curve(
        ResponseKind::AbsoluteQE,
        {900.0, 950.0, 1000.0, 1100.0, 1300.0, 1500.0, 1600.0, 1700.0,
         1750.0, 1800.0},
        {0.02, 0.35, 0.55, 0.68, 0.75, 0.75, 0.72, 0.55, 0.25, 0.03},
        Source("device.channels.response", ValueProvenance::Digitized,
               kHamamatsuPage, "product page",
               "shape digitized from typical InGaAs response curve; absolute "
               "peak 0.75 assumed (not published for C12741-11)")));

    config.optics.focalLengthMm = 25.0; // C-mount; lens is a choice.
    config.optics.fNumber = 2.0;
    config.optics.pixelPitchUm = 20.0;
    config.optics.sensorWidthPx = 640;
    config.optics.sensorHeightPx = 512;
    config.readout.shutter = ShutterKind::Global;
    config.readout.exposureSeconds = 0.05;
    config.readout.framePeriodSeconds = 1.0 / 7.2; // Full-frame rate.
    config.readout.adcBits = 14;   // Assumed; 16-bit is the transport format.
    config.readout.outputBits = 16;
    config.readout.electronsPerDn = 300000.0 / 65535.0;
    config.photon.fullWellElectrons = 300000.0;
    config.photon.darkCurrentElectronsPerSecond = minus70C ? 130.0 : 300.0;
    config.photon.readNoiseElectronsRms = 500.0;
    config.photon.prnuSigma = 0.01; // Not published; assumed.
    config.device.effectiveMinNm = 900.0;
    config.device.effectiveMaxNm = 1800.0;
    config.device.parameterSources = {
        Source("optics.sensor_width_px", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "640 (H)"),
        Source("optics.sensor_height_px", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "512 (V)"),
        Source("optics.pixel_pitch_um", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "20 um"),
        Source("photon.full_well_e", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "typical"),
        Source("photon.read_noise_e_rms", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "typical"),
        Source("photon.dark_current_e_s", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page",
               minus70C ? "typical, -70 C, water cooled"
                        : "typical, -60 C, forced-air"),
        Source("readout.output_bits", ValueProvenance::Manufacturer,
               kHamamatsuPage, "product page", "digital output 16 bit"),
        Source("readout.adc_bits", ValueProvenance::Assumed,
               kHamamatsuPage, "product page",
               "ADC width not published; 16-bit is the transport format, not "
               "the ADC width; 14-bit ADC assumed"),
        Source("readout.frame_period_s", ValueProvenance::Derived,
               kHamamatsuPage, "product page", "7.2 frames/s full frame"),
        Source("readout.electrons_per_dn", ValueProvenance::Derived,
               kHamamatsuPage, "product page",
               "full well divided by 16-bit DN range"),
        Source("optics.focal_length_mm", ValueProvenance::Assumed,
               kHamamatsuPage, "product page",
               "C-mount body; 25 mm lens assumed"),
        Source("photon.prnu_sigma", ValueProvenance::Assumed,
               kHamamatsuPage, "product page", "not published; assumed"),
    };
    return config;
}

// ---------------------------------------------------------------------------
// FLIR A6751 (the 3-5 um member of the A6750 series). QE, full well, dark
// current and read noise are NOT published for this model; the values below
// are explicit assumptions chosen to be plausible for a cooled InSb FPA and
// are deliberately NOT derived from the published NETD -- the plan forbids
// back-filling detector parameters from a system-level sensitivity figure.
// ---------------------------------------------------------------------------

CameraConfig A6751Config() {
    auto config = Base("flir_a6751", "FLIR A6751 MWIR (InSb, 3-5 um)",
                       CalibrationStatus::HardwareReference,
                       DetectorKind::Photon, CfaPattern::Mono);
    config.device.documentUrl = kFlirA6751Page;
    config.device.documentVersion = "FLIR A6750 series product page (A6751 row)";
    config.device.readoutMode = "snapshot_integrate_while_read";

    SetMonoPhotonChannel(config, Curve(
        ResponseKind::AbsoluteQE, {3000.0, 5000.0}, {0.70, 0.70},
        Source("device.channels.response", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "QE not published for A6751; flat 0.7 over the 3-5 um band "
               "assumed (typical for cooled InSb)")));

    config.optics.focalLengthMm = 25.0; // 17-200 mm lenses exist; choice assumed.
    config.optics.fNumber = 2.5;
    config.optics.pixelPitchUm = 15.0;
    config.optics.sensorWidthPx = 640;
    config.optics.sensorHeightPx = 512;
    config.readout.shutter = ShutterKind::Global;
    config.readout.exposureSeconds = 0.002;
    config.readout.framePeriodSeconds = 1.0 / 125.0; // Max full-window rate.
    config.readout.adcBits = 14;
    config.readout.outputBits = 14;
    config.readout.electronsPerDn = 7.2e6 / 16383.0;
    config.photon.fullWellElectrons = 7.2e6;
    config.photon.darkCurrentElectronsPerSecond = 5.0e4;
    config.photon.readNoiseElectronsRms = 1500.0;
    config.device.effectiveMinNm = 3000.0;
    config.device.effectiveMaxNm = 5000.0;
    config.device.parameterSources = {
        Source("device.effective_min_nm", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page", "A6751/A6753: 3.0-5.0 um"),
        Source("device.effective_max_nm", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page",
               "not the A6750's 1.0-5.0 um broadband"),
        Source("optics.f_number", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page", "A6750/A6751: f/2.5"),
        Source("optics.pixel_pitch_um", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page", "15 um"),
        Source("readout.frame_period_s", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page",
               "0.0015 Hz to 125 Hz full window"),
        Source("readout.output_bits", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page", "14-bit dynamic range"),
        // The 22 mK typical NETD is a SYSTEM sensitivity figure; it is
        // recorded here for the panel but never used to derive the detector
        // parameters below.
        Source("netd.typical_k", ValueProvenance::Manufacturer,
               kFlirA6751Page, "product page",
               "A6751/A6753: 22 mK typical; system figure, not a detector "
               "parameter; not used to derive QE/well/noise"),
        Source("photon.full_well_e", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "not published for A6751; family science datasheet lists "
               "7.2 Me-, adopted unverified"),
        Source("photon.dark_current_e_s", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "not published; plausible cooled-InSb value assumed"),
        Source("photon.read_noise_e_rms", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "not published; plausible cooled-InSb value assumed"),
        Source("device.channels.response", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "QE not published; flat 0.7 over 3-5 um assumed"),
        Source("optics.focal_length_mm", ValueProvenance::Assumed,
               kFlirA6751Page, "product page",
               "lens choice; 25 mm from the available range assumed"),
    };
    return config;
}

// ---------------------------------------------------------------------------
// FLIR Boson 640 Industrial. VOx microbolometer. The <=40 mK Industrial
// figure is a specification LIMIT, not a measured typical, and the ~8 ms
// thermal time constant is an FAQ estimate; neither may be mixed with Boson+
// (<=20 mK) figures. Responsivity, noise-equivalent bandwidth and the ADC
// width are not published and stay explicit assumptions.
// ---------------------------------------------------------------------------

CameraConfig BosonConfig() {
    auto config = Base("flir_boson_640_industrial",
                       "FLIR Boson 640 Industrial (VOx LWIR)",
                       CalibrationStatus::HardwareReference,
                       DetectorKind::Thermal, CfaPattern::Mono);
    config.device.documentUrl = kFlirBosonPage;
    config.device.documentVersion = "Teledyne FLIR Boson OEM series page";
    config.device.readoutMode = "microbolometer_60hz";

    SetMonoThermalChannel(config, Curve(
        ResponseKind::ThermalAbsorptance, {8000.0, 14000.0}, {0.80, 0.80},
        Source("device.channels.response", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "absorptance not published; flat 0.8 over the 8-14 um window "
               "assumed (VOx with germanium window, typical)")));

    config.optics.focalLengthMm = 9.2; // 50 deg HFOV radiometric variant.
    config.optics.fNumber = 1.4;       // Fast integrated lens class.
    config.optics.pixelPitchUm = 12.0;
    config.optics.sensorWidthPx = 640;
    config.optics.sensorHeightPx = 512;
    config.readout.exposureSeconds = 1.0 / 60.0;
    config.readout.framePeriodSeconds = 1.0 / 60.0; // 60 Hz baseline rate.
    config.readout.adcBits = 14;    // Assumed; output width not published.
    config.readout.outputBits = 14;
    config.thermal.timeConstantSeconds = 0.008;
    config.thermal.responsivityDnPerWatt = 1.0e13; // Assumed absolute scale.
    config.thermal.readoutWindowSeconds = 1.0 / 60.0;
    config.thermal.netdKelvin = 0.040; // Industrial-grade limit.
    config.thermal.netdReferenceTemperatureK = 298.15;
    config.thermal.netdNoiseBandwidthHz = 30.0;
    config.thermal.netdOpticalCondition =
        "f/1.4 assumed; NETD is the Industrial-grade limit at reference scene";
    config.device.effectiveMinNm = 8000.0;
    config.device.effectiveMaxNm = 14000.0;
    config.device.parameterSources = {
        Source("optics.sensor_width_px", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page", "VGA 640 x 512"),
        Source("optics.pixel_pitch_um", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page", "12 um"),
        Source("device.effective_min_nm", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page", "8-14 um"),
        Source("device.effective_max_nm", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page", "8-14 um"),
        Source("thermal.netd_k", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page",
               "Industrial grade limit <=40 mK; a limit, not a measured "
               "typical; Boson+ figures must not be mixed in"),
        Source("thermal.time_constant_s", ValueProvenance::Assumed,
               kFlirBosonFaq, "FLIR FAQ a_id/3128",
               "official FAQ estimate of about 8 ms; marked estimated"),
        Source("readout.frame_period_s", ValueProvenance::Manufacturer,
               kFlirBosonPage, "series page", "60 Hz baseline rate"),
        Source("thermal.responsivity_dn_w", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "absolute responsivity not published; assumed scale, not "
               "derived from the NETD limit"),
        Source("thermal.netd_noise_bandwidth_hz", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "measurement bandwidth not published; assumed"),
        Source("readout.adc_bits", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "ADC width not published; assumed"),
        Source("optics.focal_length_mm", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "9.2 mm lens of the 50 deg HFOV 640 variant assumed"),
        Source("device.channels.response", ValueProvenance::Assumed,
               kFlirBosonPage, "series page",
               "absorptance not published; flat 0.8 assumed"),
    };
    return config;
}

std::vector<Note> GenericNotes(const char* family) {
    return {NoteOf("all", "see config", "assumed",
                   String(family) +
                       ": every parameter is a Quantiloom modelling "
                       "definition published for closed-form testing, not a "
                       "device measurement")};
}

} // namespace

Result<CameraConfig, String> MakePresetCameraConfig(CameraPresetKind kind) {
    switch (kind) {
        case CameraPresetKind::GenericCmos: return GenericCmosConfig();
        case CameraPresetKind::GenericIngaas: return GenericIngaasConfig();
        case CameraPresetKind::GenericCooledPhotonIr:
            return GenericCooledPhotonIrConfig();
        case CameraPresetKind::GenericUncooledThermalIr:
            return GenericUncooledThermalIrConfig();
        case CameraPresetKind::Alvium1800U507Mono: return AlviumConfig(false);
        case CameraPresetKind::Alvium1800U507Color: return AlviumConfig(true);
        case CameraPresetKind::HamamatsuC12741_11Minus60C:
            return C12741Config(false);
        case CameraPresetKind::HamamatsuC12741_11Minus70C:
            return C12741Config(true);
        case CameraPresetKind::FlirA6751: return A6751Config();
        case CameraPresetKind::Boson640Industrial: return BosonConfig();
    }
    return Result<CameraConfig, String>::Err("unknown camera preset kind");
}

String CameraPresetDisplayName(CameraPresetKind kind) {
    switch (kind) {
        case CameraPresetKind::GenericCmos:
            return "Generic CMOS (visible, Bayer, global)";
        case CameraPresetKind::GenericIngaas:
            return "Generic InGaAs (SWIR, mono)";
        case CameraPresetKind::GenericCooledPhotonIr:
            return "Generic cooled photon IR (MWIR, mono)";
        case CameraPresetKind::GenericUncooledThermalIr:
            return "Generic uncooled thermal IR (LWIR, mono)";
        case CameraPresetKind::Alvium1800U507Mono:
            return "Allied Vision Alvium 1800 U-507 (mono)";
        case CameraPresetKind::Alvium1800U507Color:
            return "Allied Vision Alvium 1800 U-507 (colour)";
        case CameraPresetKind::HamamatsuC12741_11Minus60C:
            return "Hamamatsu C12741-11 (-60 C)";
        case CameraPresetKind::HamamatsuC12741_11Minus70C:
            return "Hamamatsu C12741-11 (-70 C)";
        case CameraPresetKind::FlirA6751: return "FLIR A6751 MWIR";
        case CameraPresetKind::Boson640Industrial:
            return "FLIR Boson 640 Industrial";
    }
    return "Unknown camera preset";
}

String CameraPresetToken(CameraPresetKind kind) {
    switch (kind) {
        case CameraPresetKind::GenericCmos: return "generic_cmos";
        case CameraPresetKind::GenericIngaas: return "generic_ingaas";
        case CameraPresetKind::GenericCooledPhotonIr:
            return "generic_cooled_photon_ir";
        case CameraPresetKind::GenericUncooledThermalIr:
            return "generic_uncooled_thermal_ir";
        case CameraPresetKind::Alvium1800U507Mono:
            return "alvium_1800_u507_mono";
        case CameraPresetKind::Alvium1800U507Color:
            return "alvium_1800_u507_color";
        case CameraPresetKind::HamamatsuC12741_11Minus60C:
            return "hamamatsu_c12741_11_m60c";
        case CameraPresetKind::HamamatsuC12741_11Minus70C:
            return "hamamatsu_c12741_11_m70c";
        case CameraPresetKind::FlirA6751: return "flir_a6751";
        case CameraPresetKind::Boson640Industrial: return "boson_640_industrial";
    }
    return "unknown";
}

Result<CameraPresetKind, String> CameraPresetFromToken(const String& token) {
    for (const CameraPresetKind kind : AllCameraPresets()) {
        if (CameraPresetToken(kind) == token) return kind;
    }
    return Result<CameraPresetKind, String>::Err(
        "unknown camera preset '" + token + "'");
}

std::vector<CameraPresetKind> AllCameraPresets() {
    return {CameraPresetKind::GenericCmos,
            CameraPresetKind::GenericIngaas,
            CameraPresetKind::GenericCooledPhotonIr,
            CameraPresetKind::GenericUncooledThermalIr,
            CameraPresetKind::Alvium1800U507Mono,
            CameraPresetKind::Alvium1800U507Color,
            CameraPresetKind::HamamatsuC12741_11Minus60C,
            CameraPresetKind::HamamatsuC12741_11Minus70C,
            CameraPresetKind::FlirA6751,
            CameraPresetKind::Boson640Industrial};
}

std::vector<Note> CameraPresetProvenanceNotes(CameraPresetKind kind) {
    switch (kind) {
        case CameraPresetKind::GenericCmos:
            return GenericNotes("GenericCmos");
        case CameraPresetKind::GenericIngaas:
            return GenericNotes("GenericIngaas");
        case CameraPresetKind::GenericCooledPhotonIr:
            return GenericNotes("GenericCooledPhotonIr");
        case CameraPresetKind::GenericUncooledThermalIr:
            return GenericNotes("GenericUncooledThermalIr");
        case CameraPresetKind::Alvium1800U507Mono:
            return {
                NoteOf("optics.sensor_width_px", "2464", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("optics.sensor_height_px", "2056", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("optics.pixel_pitch_um", "3.45", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("readout.shutter", "global", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("readout.adc_bits", "12", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("readout.output_bits", "12", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("photon.full_well_e", "10400 e-", "manufacturer",
                       String(kAlviumDatasheet) +
                           " saturation capacity, EMVA 1288 typical, mono"),
                NoteOf("photon.read_noise_e_rms", "2.1 e-", "manufacturer",
                       String(kAlviumDatasheet) +
                           " temporal dark noise, EMVA 1288 typical, mono"),
                NoteOf("photon.dark_current_e_s", "50 e-/s", "assumed",
                       "not published for the U-507; room-temperature value "
                       "assumed"),
                NoteOf("device.channels.response", "QE curve 0.12-0.66",
                       "digitized", String(kAlviumDatasheet) +
                           " QE plot, anchored to the published 64% at 529 nm"),
                NoteOf("readout.electrons_per_dn", "2.54", "derived",
                       "saturation capacity divided by the 12-bit DN range"),
                NoteOf("optics.focal_length_mm", "25", "assumed",
                       "body-only camera; C-mount lens choice"),
                NoteOf("readout.frame_period_s", "1/30", "assumed",
                       "nominal 30 fps operating point"),
                NoteOf("photon.prnu_sigma", "0.005", "assumed",
                       "not published"),
            };
        case CameraPresetKind::Alvium1800U507Color:
            return {
                NoteOf("optics.sensor_width_px", "2464", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("optics.sensor_height_px", "2056", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("optics.pixel_pitch_um", "3.45", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("device.cfa", "rggb", "manufacturer",
                       kAlviumDatasheet),
                NoteOf("photon.full_well_e", "10400 e-", "borrowed assumption",
                       String(kAlviumDatasheet) +
                       " saturation capacity measured on the MONOCHROME "
                       "model; borrowed for the colour variant, not a "
                       "colour measurement"),
                NoteOf("photon.read_noise_e_rms", "2.1 e-",
                       "borrowed assumption",
                       String(kAlviumDatasheet) +
                       " temporal dark noise measured on the MONOCHROME "
                       "model; borrowed for the colour variant, not a "
                       "colour measurement"),
                NoteOf("device.channels.response", "QE curve 0.12-0.66",
                       "borrowed assumption",
                       String(kAlviumDatasheet) +
                       " monochrome QE plot borrowed unchanged; the Bayer "
                       "dye transmission is not published, so no invented "
                       "CFA scaling is applied"),
                NoteOf("photon.dark_current_e_s", "50 e-/s", "assumed",
                       "not published for the U-507; room-temperature value "
                       "assumed"),
                NoteOf("optics.focal_length_mm", "25", "assumed",
                       "body-only camera; C-mount lens choice"),
                NoteOf("photon.prnu_sigma", "0.005", "assumed",
                       "not published"),
            };
        case CameraPresetKind::HamamatsuC12741_11Minus60C:
        case CameraPresetKind::HamamatsuC12741_11Minus70C: {
            const bool minus70 =
                kind == CameraPresetKind::HamamatsuC12741_11Minus70C;
            std::vector<Note> notes = {
                NoteOf("optics.sensor_width_px", "640", "manufacturer",
                       kHamamatsuPage),
                NoteOf("optics.sensor_height_px", "512", "manufacturer",
                       kHamamatsuPage),
                NoteOf("optics.pixel_pitch_um", "20", "manufacturer",
                       kHamamatsuPage),
                NoteOf("photon.full_well_e", "300000 e-", "manufacturer",
                       String(kHamamatsuPage) + " typical"),
                NoteOf("photon.read_noise_e_rms", "500 e-", "manufacturer",
                       String(kHamamatsuPage) + " typical"),
                NoteOf("photon.dark_current_e_s", minus70 ? "130" : "300",
                       "manufacturer",
                       String(kHamamatsuPage) +
                       (minus70 ? " typical, -70 C, water cooled"
                                : " typical, -60 C, forced-air")),
                NoteOf("readout.output_bits", "16", "manufacturer",
                       String(kHamamatsuPage) +
                       " digital output 16 bit; this is the TRANSPORT "
                       "format, not the ADC width"),
                NoteOf("readout.adc_bits", "14", "assumed",
                       "ADC width not published; the 16-bit figure is a "
                       "transport format and must not be read as ADC width"),
                NoteOf("readout.frame_period_s", "1/7.2", "derived",
                       "7.2 frames/s full frame"),
                NoteOf("readout.electrons_per_dn", "4.58", "derived",
                       "full well divided by the 16-bit DN range"),
                NoteOf("device.channels.response", "InGaAs 0.9-1.8 um",
                       "digitized", String(kHamamatsuPage) +
                       " typical InGaAs response shape; absolute QE peak "
                       "0.75 assumed (not published for this model)"),
                NoteOf("optics.focal_length_mm", "25", "assumed",
                       "C-mount body; lens choice"),
                NoteOf("photon.prnu_sigma", "0.01", "assumed",
                       "not published"),
            };
            return notes;
        }
        case CameraPresetKind::FlirA6751:
            return {
                NoteOf("device.spectral_range", "3.0-5.0 um", "manufacturer",
                       String(kFlirA6751Page) +
                       " A6751/A6753 row; the A6750's 1.0-5.0 um broadband "
                       "is NOT used"),
                NoteOf("optics.pixel_pitch_um", "15", "manufacturer",
                       kFlirA6751Page),
                NoteOf("optics.f_number", "2.5", "manufacturer",
                       kFlirA6751Page),
                NoteOf("readout.frame_period_s", "1/125", "manufacturer",
                       String(kFlirA6751Page) +
                       " 0.0015 Hz to 125 Hz full window; the period is "
                       "taken at the maximum rate"),
                NoteOf("readout.output_bits", "14", "manufacturer",
                       kFlirA6751Page),
                NoteOf("netd.typical", "22 mK", "manufacturer",
                       String(kFlirA6751Page) +
                       " A6751/A6753: 22 mK typical; a SYSTEM figure used "
                       "as published and NOT to derive QE, well or noise"),
                NoteOf("photon.full_well_e", "7.2e6 e-", "assumed",
                       "not published for A6751; family science datasheet "
                       "lists 7.2 Me-; adopted unverified"),
                NoteOf("photon.dark_current_e_s", "5e4 e-/s", "assumed",
                       "not published; plausible cooled-InSb value"),
                NoteOf("photon.read_noise_e_rms", "1500 e-", "assumed",
                       "not published; plausible cooled-InSb value"),
                NoteOf("device.channels.response", "flat QE 0.7", "assumed",
                       "QE not published for A6751; flat 0.7 over 3-5 um "
                       "assumed (typical cooled InSb)"),
                NoteOf("optics.focal_length_mm", "25", "assumed",
                       "17-200 mm lens range; 25 mm assumed"),
            };
        case CameraPresetKind::Boson640Industrial:
            return {
                NoteOf("optics.sensor_width_px", "640", "manufacturer",
                       kFlirBosonPage),
                NoteOf("optics.sensor_height_px", "512", "manufacturer",
                       kFlirBosonPage),
                NoteOf("optics.pixel_pitch_um", "12", "manufacturer",
                       kFlirBosonPage),
                NoteOf("device.spectral_range", "8-14 um", "manufacturer",
                       kFlirBosonPage),
                NoteOf("thermal.netd_k", "0.040", "limit",
                       String(kFlirBosonPage) +
                       " Industrial grade: NETD <= 40 mK is a "
                       "specification LIMIT, not a measured typical; "
                       "Boson+ (<=20 mK) figures must not be mixed in"),
                NoteOf("thermal.time_constant_s", "0.008", "estimated",
                       String(kFlirBosonFaq) +
                       " official FAQ estimate of about 8 ms"),
                NoteOf("readout.frame_period_s", "1/60", "manufacturer",
                       String(kFlirBosonPage) + " 60 Hz baseline rate"),
                NoteOf("thermal.responsivity_dn_w", "1e13", "assumed",
                       "absolute responsivity not published; assumed scale, "
                       "NOT derived from the NETD limit"),
                NoteOf("thermal.netd_noise_bandwidth_hz", "30", "assumed",
                       "measurement bandwidth not published"),
                NoteOf("readout.adc_bits", "14", "assumed",
                       "ADC width not published"),
                NoteOf("optics.focal_length_mm", "9.2", "assumed",
                       "9.2 mm lens of the 50 deg HFOV 640 variant"),
                NoteOf("device.channels.response", "flat absorptance 0.8",
                       "assumed",
                       "absorptance not published; VOx-with-window typical"),
            };
    }
    return {};
}

} // namespace quantiloom::camera
