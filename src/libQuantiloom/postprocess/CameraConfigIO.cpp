#include "postprocess/CameraConfigIO.hpp"
#include "postprocess/CameraConfigIOInternal.hpp"

#include "postprocess/CameraPhysics.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "scene/MotionSpec.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <tuple>

namespace quantiloom {
namespace {

using namespace camera;

template<class T>
Result<T, String> Error(String message) {
    return typename Result<T, String>::Err(std::move(message));
}

String Quoted(const String& value) {
    String out = "\"";
    for (const char c : value) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        if (c == '\r') { out += "\\r"; continue; }
        out += c;
    }
    return out + '"';
}

template<size_t N>
void WriteArray(std::ostream& out, const std::array<f64, N>& values) {
    out << '[';
    for (size_t i = 0; i < N; ++i) {
        if (i) out << ", ";
        out << values[i];
    }
    out << ']';
}

void WriteArray(std::ostream& out, const std::vector<f64>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << values[i];
    }
    out << ']';
}

template<size_t N>
Result<std::array<f64, N>, String> ReadArray(const Config& config,
                                             StringView key,
                                             const std::array<f64, N>& fallback) {
    if (!config.Has(key)) return fallback;
    const auto values = config.GetDoubleArray(key);
    if (values.size() != N)
        return Error<std::array<f64, N>>(String(key) + " needs " +
                                         std::to_string(N) + " numbers");
    std::array<f64, N> out{};
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

String Lower(String value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

template<class Enum>
Result<Enum, String> TokenError(StringView field, const String& value) {
    return Error<Enum>("unknown " + String(field) + " '" + value + "'");
}

Result<DetectorKind, String> Detector(const String& raw) {
    const String text = Lower(raw);
    if (text == "photon") return DetectorKind::Photon;
    if (text == "thermal") return DetectorKind::Thermal;
    return TokenError<DetectorKind>("detector", raw);
}
Result<CalibrationStatus, String> Calibration(const String& raw) {
    const String text = Lower(raw);
    if (text == "generic_assumption") return CalibrationStatus::GenericAssumption;
    if (text == "hardware_reference") return CalibrationStatus::HardwareReference;
    if (text == "calibrated") return CalibrationStatus::Calibrated;
    return TokenError<CalibrationStatus>("calibration_status", raw);
}
Result<CfaPattern, String> Cfa(const String& raw) {
    const String text = Lower(raw);
    if (text == "mono") return CfaPattern::Mono;
    if (text == "rggb") return CfaPattern::RGGB;
    if (text == "grbg") return CfaPattern::GRBG;
    if (text == "gbrg") return CfaPattern::GBRG;
    if (text == "bggr") return CfaPattern::BGGR;
    if (text == "multi_channel") return CfaPattern::MultiChannel;
    return TokenError<CfaPattern>("cfa", raw);
}
Result<ShutterKind, String> Shutter(const String& raw) {
    const String text = Lower(raw);
    if (text == "global") return ShutterKind::Global;
    if (text == "rolling") return ShutterKind::Rolling;
    return TokenError<ShutterKind>("shutter", raw);
}
Result<ProcessingBackend, String> Backend(const String& raw) {
    const String text = Lower(raw);
    if (text == "cpu_reference") return ProcessingBackend::CpuReference;
    if (text == "gpu_preview") return ProcessingBackend::GpuPreview;
    return TokenError<ProcessingBackend>("processing_backend", raw);
}
Result<OutputColorSpace, String> ColorSpace(const String& raw) {
    const String text = Lower(raw);
    if (text == "device_native") return OutputColorSpace::DeviceNative;
    if (text == "cie_linear_srgb") return OutputColorSpace::CieLinearSrgb;
    if (text == "display_srgb") return OutputColorSpace::DisplaySrgb;
    return TokenError<OutputColorSpace>("output_color_space", raw);
}
Result<ValueProvenance, String> Provenance(const String& raw) {
    const String text = Lower(raw);
    if (text == "manufacturer") return ValueProvenance::Manufacturer;
    if (text == "digitized") return ValueProvenance::Digitized;
    if (text == "derived") return ValueProvenance::Derived;
    if (text == "assumed") return ValueProvenance::Assumed;
    return TokenError<ValueProvenance>("provenance", raw);
}
Result<ResponseKind, String> ResponseType(const String& raw) {
    const String text = Lower(raw);
    if (text == "absolute_qe") return ResponseKind::AbsoluteQE;
    if (text == "relative_qe") return ResponseKind::RelativeQE;
    if (text == "lens_transmission") return ResponseKind::LensTransmission;
    if (text == "filter_transmission") return ResponseKind::FilterTransmission;
    if (text == "thermal_absorptance") return ResponseKind::ThermalAbsorptance;
    if (text == "system_photon_qe") return ResponseKind::SystemPhotonQE;
    if (text == "system_thermal_absorptance") return ResponseKind::SystemThermalAbsorptance;
    return TokenError<ResponseKind>("response kind", raw);
}
Result<RelativeNormalization, String> Normalization(const String& raw) {
    const String text = Lower(raw);
    if (text == "none") return RelativeNormalization::None;
    if (text == "peak_one") return RelativeNormalization::PeakOne;
    if (text == "area_one") return RelativeNormalization::AreaOne;
    return TokenError<RelativeNormalization>("response normalization", raw);
}
Result<DisplayToneMode, String> Tone(const String& raw) {
    const String text = Lower(raw);
    if (text == "linear") return DisplayToneMode::Linear;
    if (text == "equalize") return DisplayToneMode::Equalize;
    if (text == "clahe") return DisplayToneMode::Clahe;
    return TokenError<DisplayToneMode>("infrared tone", raw);
}
Result<DisplayPalette, String> Palette(const String& raw) {
    const String text = Lower(raw);
    if (text == "grey") return DisplayPalette::Grey;
    if (text == "grey_inverted") return DisplayPalette::GreyInverted;
    if (text == "ironbow") return DisplayPalette::Ironbow;
    if (text == "rainbow") return DisplayPalette::Rainbow;
    if (text == "viridis") return DisplayPalette::Viridis;
    return TokenError<DisplayPalette>("infrared palette", raw);
}

ParameterSource ReadSource(const Config& config, String parameterPath = {}) {
    ParameterSource source;
    source.parameterPath = config.GetString("parameter_path", parameterPath);
    source.document = config.GetString("document", "");
    source.documentVersion = config.GetString("document_version", "");
    source.condition = config.GetString("condition", "");
    if (auto parsed = Provenance(config.GetString("provenance", "assumed")); parsed)
        source.provenance = *parsed;
    return source;
}

std::filesystem::path ResolvePath(const String& path, const String& baseDir) {
    const std::filesystem::path asWritten(path);
    if (asWritten.is_absolute() || baseDir.empty()) return asWritten;
    const auto besideConfig = std::filesystem::path(baseDir) / asWritten;
    if (std::filesystem::exists(besideConfig)) return besideConfig;
    return asWritten;
}

String Trim(String text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == String::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

Vector<String> SplitFields(const String& text) {
    Vector<String> fields;
    if (text.find(',') != String::npos) {
        std::istringstream input(text);
        String field;
        while (std::getline(input, field, ','))
            fields.push_back(Trim(field));
        if (!text.empty() && text.back() == ',') fields.push_back({});
    } else {
        std::istringstream input(text);
        String field;
        while (input >> field) fields.push_back(field);
    }
    return fields;
}

Result<f64, String> StrictNumber(const String& text, const String& context) {
    if (text.empty()) return Error<f64>(context + ": missing numeric cell");
    try {
        size_t consumed = 0;
        const f64 value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value))
            return Error<f64>(context + ": expected finite decimal number");
        return value;
    } catch (const std::exception&) {
        return Error<f64>(context + ": expected finite decimal number");
    }
}

Result<Vector<std::pair<f64, f64>>, String> ReadStrictResponseFile(
    const std::filesystem::path& path, u32 valueColumn) {
    std::ifstream input(path);
    if (!input) return Error<Vector<std::pair<f64, f64>>>(
        "cannot open response file: " + path.string());
    Vector<std::pair<f64, f64>> samples;
    String line;
    size_t lineNumber = 0;
    bool firstDataOrHeader = true;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (const auto comment = line.find('#'); comment != String::npos)
            line.resize(comment);
        line = Trim(line);
        if (line.empty()) continue;
        const auto cells = SplitFields(line);
        const String context = path.string() + ":" + std::to_string(lineNumber);
        if (firstDataOrHeader && !cells.empty() &&
            Lower(cells.front()) == "wavelength_nm") {
            if (cells.size() < valueColumn)
                return Error<Vector<std::pair<f64, f64>>>(
                    context + ": header has fewer columns than requested");
            firstDataOrHeader = false;
            continue;
        }
        firstDataOrHeader = false;
        if (cells.size() < valueColumn)
            return Error<Vector<std::pair<f64, f64>>>(
                context + ": missing value column " + std::to_string(valueColumn));
        const auto wavelength = StrictNumber(cells[0], context + " wavelength_nm");
        const auto value = StrictNumber(cells[valueColumn - 1], context + " response");
        if (!wavelength) return Error<Vector<std::pair<f64, f64>>>(wavelength.error());
        if (!value) return Error<Vector<std::pair<f64, f64>>>(value.error());
        if (*wavelength <= 0.0 || *value < 0.0 ||
            (!samples.empty() && *wavelength <= samples.back().first))
            return Error<Vector<std::pair<f64, f64>>>(
                context + ": wavelength must be positive and strictly increasing; "
                "response must be nonnegative");
        samples.emplace_back(*wavelength, *value);
    }
    if (input.bad())
        return Error<Vector<std::pair<f64, f64>>>(
            "I/O error reading response file: " + path.string());
    if (samples.size() < 2)
        return Error<Vector<std::pair<f64, f64>>>(
            "response file needs at least two numeric rows: " + path.string());
    return samples;
}

Result<std::vector<f64>, String> ReadCalibrationFile(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return Error<std::vector<f64>>(
        "cannot open calibration file: " + path.string());
    std::vector<f64> values;
    String line;
    size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (const auto comment = line.find('#'); comment != String::npos)
            line.resize(comment);
        line = Trim(line);
        if (line.empty()) continue;
        const auto cells = SplitFields(line);
        for (const auto& cell : cells) {
            auto value = StrictNumber(
                cell, path.string() + ":" + std::to_string(lineNumber));
            if (!value) return Error<std::vector<f64>>(value.error());
            values.push_back(*value);
        }
    }
    if (input.bad())
        return Error<std::vector<f64>>(
            "I/O error reading calibration file: " + path.string());
    if (values.empty())
        return Error<std::vector<f64>>(
            "calibration file has no values: " + path.string());
    return values;
}

Result<void, String> ReadCalibration(
    const Config& document, StringView inlineKey, StringView pathKey,
    const String& baseDir, std::vector<f64>& values, String& dataPath) {
    const String path = document.GetString(pathKey, "");
    if (!path.empty() && document.Has(inlineKey))
        return Result<void, String>::Err(
            String(pathKey) + " and " + String(inlineKey) +
            " are mutually exclusive");
    if (!path.empty()) {
        auto loaded = ReadCalibrationFile(ResolvePath(path, baseDir));
        if (!loaded) return Result<void, String>::Err(loaded.error());
        values = std::move(*loaded);
        dataPath = path;
    } else {
        values = document.GetDoubleArray(inlineKey);
    }
    return Result<void, String>::Ok();
}

Result<ResponseCurve, String> ReadCurve(const Config& table, ResponseKind defaultKind,
                                       const String& baseDir, const String& label) {
    ResponseCurve curve;
    auto kind = ResponseType(table.GetString("kind", ""));
    if (table.Has("kind")) {
        if (!kind) return Error<ResponseCurve>(label + ": " + kind.error());
        curve.kind = *kind;
    } else {
        curve.kind = defaultKind;
    }
    const auto normalization = Normalization(table.GetString("normalization", "none"));
    if (!normalization) return Error<ResponseCurve>(label + ": " + normalization.error());
    curve.normalization = *normalization;
    curve.amplitude = table.GetDouble("amplitude", 1.0);
    curve.amplitudeSource = table.GetString("amplitude_source", "");
    curve.includesPixelFillFactor = table.GetBool("includes_pixel_fill_factor", false);
    curve.source = ReadSource(table, label);
    if (table.Has("provenance")) {
        const auto provenance = Provenance(table.GetString("provenance"));
        if (!provenance) return Error<ResponseCurve>(label + ": " + provenance.error());
        curve.source.provenance = *provenance;
    }
    const String path = table.GetString("path", "");
    if (!path.empty()) {
        if (table.Has("wavelength_nm") || table.Has("value"))
            return Error<ResponseCurve>(label + ": use either path or inline samples");
        const u32 column = table.Get<u32>("column", 2);
        if (column < 2) return Error<ResponseCurve>(label + ": column is 1-based, at least 2");
        const auto resolved = ResolvePath(path, baseDir);
        if (!std::filesystem::exists(resolved))
            return Error<ResponseCurve>(label + ": response file does not exist: " + path);
        auto loaded = ReadStrictResponseFile(resolved, column);
        if (!loaded) return Error<ResponseCurve>(label + ": " + loaded.error());
        for (const auto& [lambda, value] : *loaded) {
            curve.wavelengthNm.push_back(lambda);
            curve.value.push_back(value);
        }
        curve.dataPath = path;
        curve.dataColumn = column;
    } else {
        curve.wavelengthNm = table.GetDoubleArray("wavelength_nm");
        curve.value = table.GetDoubleArray("value");
    }
    const auto valid = ValidateResponse(curve);
    if (!valid) return Error<ResponseCurve>(label + ": " + valid.error());
    return curve;
}

ResponseCurve FlatCurve(f64 lo, f64 hi, f64 value, ResponseKind kind,
                        const String& sourceText) {
    ResponseCurve curve;
    curve.kind = kind;
    curve.wavelengthNm = {lo, hi};
    curve.value = {value, value};
    curve.source = {"sensor.channels.response", ValueProvenance::Assumed,
                    sourceText, "migration-v1", "flat response over legacy render mode"};
    return curve;
}

std::pair<f64, f64> BandFor(const Config& document, SpectralMode mode) {
    if (const auto band = GetFusedBandInfo(mode))
        return {band->lambdaMinNm, band->lambdaMaxNm};
    if (mode == SpectralMode::Single) {
        const f64 wavelength = document.GetDouble("spectral.wavelength_nm", 550.0);
        return {wavelength - 0.5, wavelength + 0.5};
    }
    return {380.0, 780.0}; // A labelled assumption for fast RGB / legacy cubes.
}

void SetExtent(const Config& document, OpticsConfig& optics) {
    const auto resolution = document.GetIntArray("renderer.resolution");
    optics.sensorWidthPx = resolution.size() == 2 && resolution[0] > 0
                                ? static_cast<u32>(resolution[0]) : 1920;
    optics.sensorHeightPx = resolution.size() == 2 && resolution[1] > 0
                                 ? static_cast<u32>(resolution[1]) : 1080;
}

ParameterSource Assumption(String path, String description) {
    return {std::move(path), ValueProvenance::Assumed, std::move(description),
            "migration-v1", "legacy [sensor] had no device response or calibration"};
}

CameraConfig MigrateLegacy(const Config& document, SpectralMode mode) {
    CameraConfig camera;
    camera.enabled = PostprocessConfig::IsSensorEnabled(document);
    camera.products.apparentTemperature =
        camera.enabled && PostprocessConfig::IsThermographyEnabled(document);
    camera.device.id = "generic_legacy_photon";
    camera.device.displayName = "Legacy generic photon detector";
    camera.device.documentVersion = "migration-v1";
    camera.device.calibration = CalibrationStatus::GenericAssumption;
    camera.device.detector = DetectorKind::Photon; // Even a legacy LWIR scene.
    if (mode == SpectralMode::RGB)
        camera.inputKind = CameraInputKind::FastRgbApproximation;
    const bool legacyColor = mode == SpectralMode::RGB ||
                             mode == SpectralMode::VIS_Fused ||
                             mode == SpectralMode::VIS_Hero;
    camera.device.cfa = legacyColor ? CfaPattern::RGGB : CfaPattern::Mono;

    const SensorParams legacy = PostprocessConfig::ParseSensorParams(document);
    camera.optics.focalLengthMm = legacy.focalLength_mm;
    camera.optics.fNumber = legacy.fNumber;
    camera.optics.pixelPitchUm = legacy.pixelPitch_um;
    camera.optics.psfSigmaPixelsOverride = legacy.psfSigma_px;
    SetExtent(document, camera.optics);
    // The old pinhole render was framed by camera.fov_y. Its independent
    // focal_length_mm did not affect the view. Keep that composition on load.
    if (document.GetString("camera.projection", "perspective") != "orthographic") {
        const f64 fovY = document.GetDouble("camera.fov_y", 60.0);
        if (fovY > 0.0 && fovY < 180.0) {
            const f64 halfAngle = fovY * std::numbers::pi / 360.0;
            camera.optics.focalLengthMm =
                camera.optics.sensorHeightPx * camera.optics.pixelPitchUm * 0.001 /
                (2.0 * std::tan(halfAngle));
            camera.device.parameterSources.push_back({
                "optics.focal_length_mm", ValueProvenance::Derived,
                "camera.fov_y and renderer.resolution", "migration-v1",
                "effective focal length preserves legacy vertical field of view"});
        }
    } else {
        camera.device.parameterSources.push_back(Assumption(
            "optics.focal_length_mm",
            "orthographic legacy view has no physical focal length; retained authored value"));
    }
    camera.readout.exposureSeconds = legacy.integrationTime_s;
    camera.readout.electronsPerDn = legacy.gain;
    camera.readout.adcBits = legacy.bitDepth;
    camera.readout.outputBits = legacy.bitDepth;
    camera.photon.fullWellElectrons = legacy.wellCapacity_e;
    camera.photon.darkCurrentElectronsPerSecond = legacy.darkCurrent_e_s;
    camera.photon.readNoiseElectronsRms = legacy.readNoise_e_rms;
    camera.photon.prnuSigma = legacy.prnuSigma;
    camera.photon.dsnuElectronsRms = legacy.dsnuSigma_e;
    camera.photon.dsnuReferenceExposureSeconds = legacy.integrationTime_s;
    camera.photon.applyNuc = legacy.enableNUC;
    camera.photon.nucResidualFraction = 1.0 - legacy.nucEfficiency;
    camera.photon.enableShotNoise = legacy.enablePoissonNoise;
    camera.photon.enableDarkCurrent = legacy.enableDarkCurrent;
    camera.photon.enableDarkShotNoise = legacy.enableDarkCurrent;
    camera.photon.enableReadNoise = legacy.enableReadNoise;
    camera.photon.enableFpn = legacy.enableFPN;
    camera.randomSeed = legacy.noiseSeed;

    const auto [lo, hi] = BandFor(document, mode);
    camera.device.effectiveMinNm = lo;
    camera.device.effectiveMaxNm = hi;
    for (const char* name : legacyColor
             ? std::vector<const char*>{"R", "G", "B"}
             : std::vector<const char*>{"Mono"}) {
        ChannelProfile channel;
        channel.name = name;
        channel.response.quantumEfficiency =
            FlatCurve(lo, hi, legacy.quantumEfficiency, ResponseKind::AbsoluteQE,
                      "legacy scalar sensor.quantum_efficiency");
        camera.device.channels.push_back(std::move(channel));
    }
    camera.device.parameterSources.push_back(Assumption(
        "device.channels.response",
        "flat generic response on the rendered spectral band; not a measured device QE"));
    camera.device.parameterSources.push_back(Assumption(
        "readout.adc_bits", "legacy bit_depth treated as both ADC and output width"));
    if (legacyColor) {
        camera.calibratedFastRgbInput = false;
        camera.fastRgbRadianceScale = 0.0;
        camera.device.parameterSources.push_back(Assumption(
            "fast_rgb_input",
            "legacy RGB/CIE color maps approximately to generic Bayer R/G/B; "
            "it cannot recover device spectral response"));
    }
    return camera;
}

Result<void, String> ReadMotion(const Config& document, CameraMotionConfig& motion) {
    if (!document.HasSection("camera.motion")) return Result<void, String>::Ok();
    const String interpolation = Lower(document.GetString("camera.motion.interpolation", "linear"));
    const String extrapolation = Lower(document.GetString("camera.motion.extrapolate", "hold"));
    if (interpolation != "linear" || extrapolation != "hold")
        return Result<void, String>::Err(
            "camera.motion supports interpolation=linear and extrapolate=hold");
    for (const Config& table : document.GetTableArray("camera.motion.keys")) {
        CameraPoseKey key;
        Vector<String> warnings;
        const auto time = scene::ReadDuration(table, "t", "camera.motion.keys", warnings);
        if (!warnings.empty() || !time)
            return Result<void, String>::Err(
                warnings.empty() ? "camera.motion key needs t" : warnings.front());
        key.timeSeconds = *time;
        const auto position = ReadArray<3>(table, "position", key.position);
        const auto target = ReadArray<3>(table, "look_at", key.lookAt);
        if (!position || !target || !table.Has("position") || !table.Has("look_at"))
            return Result<void, String>::Err(
                "camera.motion key needs position and look_at arrays of 3 numbers");
        key.position = *position;
        key.lookAt = *target;
        motion.keys.push_back(key);
    }
    if (motion.keys.empty())
        return Result<void, String>::Err("camera.motion needs at least one key");
    return Result<void, String>::Ok();
}

Result<void, String> ReadResponseSlot(const Config& channel, StringView slot,
                                     ResponseKind kind, const String& baseDir,
                                     std::optional<ResponseCurve>& destination) {
    if (!channel.HasSection(slot)) return Result<void, String>::Ok();
    auto table = channel.GetTable(slot);
    if (!table) return Result<void, String>::Err(String(slot) + ": expected a table");
    auto curve = ReadCurve(*table, kind, baseDir, "sensor.channels." + String(slot));
    if (!curve) return Result<void, String>::Err(curve.error());
    destination = std::move(*curve);
    return Result<void, String>::Ok();
}

Result<ChannelProfile, String> ReadChannel(const Config& table, DetectorKind detector,
                                          const String& baseDir) {
    ChannelProfile channel;
    channel.name = table.GetString("name", "Mono");
    auto& response = channel.response;
    using Slot = std::tuple<StringView, ResponseKind, std::optional<ResponseCurve>*>;
    const std::array<Slot, 5> slots = {{
        {"lens", ResponseKind::LensTransmission, &response.lensTransmission},
        {"filter", ResponseKind::FilterTransmission, &response.filterTransmission},
        {"qe", ResponseKind::AbsoluteQE, &response.quantumEfficiency},
        {"absorptance", ResponseKind::ThermalAbsorptance, &response.thermalAbsorptance},
        {"system", detector == DetectorKind::Photon
                       ? ResponseKind::SystemPhotonQE
                       : ResponseKind::SystemThermalAbsorptance,
         &response.systemResponse}
    }};
    for (const auto& [slot, kind, destination] : slots) {
        auto read = ReadResponseSlot(table, slot, kind, baseDir, *destination);
        if (!read) return Error<ChannelProfile>(channel.name + ": " + read.error());
    }
    const auto valid = ValidateResponseStack(channel.response, detector);
    if (!valid) return Error<ChannelProfile>(channel.name + ": " + valid.error());
    return channel;
}

Result<void, String> ReadSources(const Config& document,
                                std::vector<ParameterSource>& sources) {
    for (const Config& table : document.GetTableArray("sensor.parameter_sources")) {
        const auto provenance = Provenance(table.GetString("provenance", "assumed"));
        if (!provenance) return Result<void, String>::Err(provenance.error());
        ParameterSource source = ReadSource(table);
        source.provenance = *provenance;
        if (source.parameterPath.empty())
            return Result<void, String>::Err("sensor.parameter_sources needs parameter_path");
        sources.push_back(std::move(source));
    }
    return Result<void, String>::Ok();
}

String Token(DetectorKind value) {
    return value == DetectorKind::Photon ? "photon" : "thermal";
}
String Token(CalibrationStatus value) {
    switch (value) {
        case CalibrationStatus::GenericAssumption: return "generic_assumption";
        case CalibrationStatus::HardwareReference: return "hardware_reference";
        case CalibrationStatus::Calibrated: return "calibrated";
    }
    return "generic_assumption";
}
String Token(CfaPattern value) {
    switch (value) {
        case CfaPattern::Mono: return "mono";
        case CfaPattern::RGGB: return "rggb";
        case CfaPattern::GRBG: return "grbg";
        case CfaPattern::GBRG: return "gbrg";
        case CfaPattern::BGGR: return "bggr";
        case CfaPattern::MultiChannel: return "multi_channel";
    }
    return "mono";
}
String Token(ValueProvenance value) {
    switch (value) {
        case ValueProvenance::Manufacturer: return "manufacturer";
        case ValueProvenance::Digitized: return "digitized";
        case ValueProvenance::Derived: return "derived";
        case ValueProvenance::Assumed: return "assumed";
    }
    return "assumed";
}
String Token(ResponseKind value) {
    switch (value) {
        case ResponseKind::AbsoluteQE: return "absolute_qe";
        case ResponseKind::RelativeQE: return "relative_qe";
        case ResponseKind::LensTransmission: return "lens_transmission";
        case ResponseKind::FilterTransmission: return "filter_transmission";
        case ResponseKind::ThermalAbsorptance: return "thermal_absorptance";
        case ResponseKind::SystemPhotonQE: return "system_photon_qe";
        case ResponseKind::SystemThermalAbsorptance: return "system_thermal_absorptance";
    }
    return "absolute_qe";
}
String Token(RelativeNormalization value) {
    switch (value) {
        case RelativeNormalization::None: return "none";
        case RelativeNormalization::PeakOne: return "peak_one";
        case RelativeNormalization::AreaOne: return "area_one";
    }
    return "none";
}
String Token(DisplayToneMode value) {
    switch (value) {
        case DisplayToneMode::Linear: return "linear";
        case DisplayToneMode::Equalize: return "equalize";
        case DisplayToneMode::Clahe: return "clahe";
    }
    return "linear";
}
String Token(DisplayPalette value) {
    switch (value) {
        case DisplayPalette::Grey: return "grey";
        case DisplayPalette::GreyInverted: return "grey_inverted";
        case DisplayPalette::Ironbow: return "ironbow";
        case DisplayPalette::Rainbow: return "rainbow";
        case DisplayPalette::Viridis: return "viridis";
    }
    return "grey";
}
String Token(ProcessingBackend value) {
    return value == ProcessingBackend::GpuPreview ? "gpu_preview" : "cpu_reference";
}
String Token(OutputColorSpace value) {
    switch (value) {
        case OutputColorSpace::DeviceNative: return "device_native";
        case OutputColorSpace::CieLinearSrgb: return "cie_linear_srgb";
        case OutputColorSpace::DisplaySrgb: return "display_srgb";
    }
    return "display_srgb";
}

void WriteSource(std::ostream& out, const ParameterSource& source) {
    out << "parameter_path = " << Quoted(source.parameterPath) << '\n';
    out << "provenance = " << Quoted(Token(source.provenance)) << '\n';
    if (!source.document.empty())
        out << "document = " << Quoted(source.document) << '\n';
    if (!source.documentVersion.empty())
        out << "document_version = " << Quoted(source.documentVersion) << '\n';
    if (!source.condition.empty())
        out << "condition = " << Quoted(source.condition) << '\n';
}

void WriteCurve(std::ostream& out, StringView slot, const ResponseCurve& curve) {
    out << "[sensor.channels." << slot << "]\n";
    out << "kind = " << Quoted(Token(curve.kind)) << '\n';
    out << "normalization = " << Quoted(Token(curve.normalization)) << '\n';
    out << "amplitude = " << curve.amplitude << '\n';
    if (!curve.amplitudeSource.empty())
        out << "amplitude_source = " << Quoted(curve.amplitudeSource) << '\n';
    out << "includes_pixel_fill_factor = "
        << (curve.includesPixelFillFactor ? "true" : "false") << '\n';
    if (!curve.dataPath.empty()) {
        out << "path = " << Quoted(curve.dataPath) << '\n';
        out << "column = " << curve.dataColumn << '\n';
    } else {
        out << "wavelength_nm = ";
        WriteArray(out, curve.wavelengthNm);
        out << "\nvalue = ";
        WriteArray(out, curve.value);
        out << '\n';
    }
    if (!curve.source.parameterPath.empty())
        out << "parameter_path = " << Quoted(curve.source.parameterPath) << '\n';
    out << "provenance = " << Quoted(Token(curve.source.provenance)) << '\n';
    if (!curve.source.document.empty())
        out << "document = " << Quoted(curve.source.document) << '\n';
    if (!curve.source.documentVersion.empty())
        out << "document_version = " << Quoted(curve.source.documentVersion) << '\n';
    if (!curve.source.condition.empty())
        out << "condition = " << Quoted(curve.source.condition) << '\n';
    out << '\n';
}

} // namespace

Result<camera::CameraConfig, String> ParseCameraConfig(
    const Config& document, SpectralMode mode, const String& baseDir) {
    using namespace camera;
    CameraConfig camera;
    if (!document.Has("sensor.version")) {
        camera = MigrateLegacy(document, mode);
    } else {
        const u32 version = document.Get<u32>("sensor.version", 0);
        if (version != kCameraConfigVersion)
            return Error<CameraConfig>("unsupported sensor.version " +
                                       std::to_string(version));
        camera.version = version;
        camera.enabled = document.GetBool("sensor.enabled", false);
        camera.device.id = document.GetString("sensor.device_id", "generic_photon");
        camera.device.displayName = document.GetString("sensor.display_name", camera.device.id);
        camera.device.documentVersion = document.GetString("sensor.document_version", "");
        camera.device.documentUrl = document.GetString("sensor.document_url", "");
        camera.device.readoutMode = document.GetString("sensor.readout_mode", "");
        camera.device.effectiveMinNm = document.GetDouble("sensor.effective_min_nm", 0.0);
        camera.device.effectiveMaxNm = document.GetDouble("sensor.effective_max_nm", 0.0);
        auto detector = Detector(document.GetString("sensor.detector", "photon"));
        auto calibration = Calibration(
            document.GetString("sensor.calibration_status", "generic_assumption"));
        auto cfa = Cfa(document.GetString("sensor.cfa", "mono"));
        if (!detector) return Error<CameraConfig>(detector.error());
        if (!calibration) return Error<CameraConfig>(calibration.error());
        if (!cfa) return Error<CameraConfig>(cfa.error());
        camera.device.detector = *detector;
        camera.device.calibration = *calibration;
        camera.device.cfa = *cfa;
        const String inputKind = Lower(
            document.GetString("sensor.input_kind", "spectral"));
        if (inputKind == "spectral")
            camera.inputKind = CameraInputKind::SpectralMeasurement;
        else if (inputKind == "fast_rgb")
            camera.inputKind = CameraInputKind::FastRgbApproximation;
        else
            return Error<CameraConfig>(
                "sensor.input_kind must be spectral or fast_rgb");
        auto sources = ReadSources(document, camera.device.parameterSources);
        if (!sources) return Error<CameraConfig>(sources.error());

        // Versioned physical geometry must be authored by the device profile
        // or explicitly by the scene. The viewport/output resolution is not
        // the sensor array and must never change pixel area or field of view.
        camera.optics.focalLengthMm =
            document.GetDouble("sensor.optics.focal_length_mm", camera.optics.focalLengthMm);
        camera.optics.fNumber =
            document.GetDouble("sensor.optics.f_number", camera.optics.fNumber);
        camera.optics.pixelPitchUm =
            document.GetDouble("sensor.optics.pixel_pitch_um", camera.optics.pixelPitchUm);
        camera.optics.psfSigmaPixelsOverride =
            document.GetDouble("sensor.optics.psf_sigma_px",
                               camera.optics.psfSigmaPixelsOverride);
        camera.optics.fillFactor =
            document.GetDouble("sensor.optics.fill_factor", camera.optics.fillFactor);
        camera.optics.sensorWidthPx =
            document.Get<u32>("sensor.optics.sensor_width_px", camera.optics.sensorWidthPx);
        camera.optics.sensorHeightPx =
            document.Get<u32>("sensor.optics.sensor_height_px", camera.optics.sensorHeightPx);
        camera.optics.focusDistanceM =
            document.GetDouble("sensor.optics.focus_distance_m", camera.optics.focusDistanceM);
        camera.optics.cosFourthVignetting =
            document.GetBool("sensor.optics.cos_fourth_vignetting", false);
        camera.optics.knownPsfSourcePath =
            document.GetString("sensor.optics.known_psf_path", "");
        camera.optics.knownPsfPath =
            camera.optics.knownPsfSourcePath.empty()
                ? String{}
                : ResolvePath(camera.optics.knownPsfSourcePath, baseDir).string();
        camera.readout.exposureSeconds =
            document.GetDouble("sensor.exposure.time_s", camera.readout.exposureSeconds);
        auto shutter = Shutter(document.GetString("sensor.readout.shutter", "global"));
        if (!shutter) return Error<CameraConfig>(shutter.error());
        camera.readout.shutter = *shutter;
        camera.readout.rowDelaySeconds =
            document.GetDouble("sensor.readout.row_delay_s", camera.readout.rowDelaySeconds);
        camera.readout.framePeriodSeconds =
            document.GetDouble("sensor.readout.frame_period_s",
                               camera.readout.framePeriodSeconds);
        camera.readout.analogGain =
            document.GetDouble("sensor.readout.analog_gain", camera.readout.analogGain);
        camera.readout.electronsPerDn =
            document.GetDouble("sensor.readout.electrons_per_dn", camera.readout.electronsPerDn);
        camera.readout.blackLevelDn =
            document.GetDouble("sensor.readout.black_level_dn", camera.readout.blackLevelDn);
        camera.readout.adcBits =
            document.Get<u32>("sensor.readout.adc_bits", camera.readout.adcBits);
        camera.readout.outputBits =
            document.Get<u32>("sensor.readout.output_bits", camera.readout.outputBits);
        camera.readout.effectiveBits =
            document.GetDouble("sensor.readout.effective_bits", camera.readout.effectiveBits);

        auto& photon = camera.photon;
        photon.fullWellElectrons =
            document.GetDouble("sensor.photon.full_well_e", photon.fullWellElectrons);
        photon.darkCurrentElectronsPerSecond =
            document.GetDouble("sensor.photon.dark_current_e_s",
                               photon.darkCurrentElectronsPerSecond);
        photon.readNoiseElectronsRms =
            document.GetDouble("sensor.photon.read_noise_e_rms", photon.readNoiseElectronsRms);
        photon.prnuSigma = document.GetDouble("sensor.photon.prnu_sigma", photon.prnuSigma);
        photon.dsnuElectronsRms =
            document.GetDouble("sensor.photon.dsnu_e_rms", photon.dsnuElectronsRms);
        photon.dsnuReferenceExposureSeconds =
            document.GetDouble("sensor.photon.dsnu_reference_exposure_s",
                               photon.dsnuReferenceExposureSeconds);
        photon.biasDnRms =
            document.GetDouble("sensor.photon.bias_dn_rms", photon.biasDnRms);
        photon.applyNuc = document.GetBool("sensor.photon.apply_nuc", photon.applyNuc);
        photon.nucResidualFraction =
            document.GetDouble("sensor.photon.nuc_residual_fraction",
                               photon.nucResidualFraction);
        photon.enableShotNoise =
            document.GetBool("sensor.photon.enable_shot_noise", photon.enableShotNoise);
        photon.enableDarkCurrent =
            document.GetBool("sensor.photon.enable_dark_current", photon.enableDarkCurrent);
        photon.enableDarkShotNoise =
            document.GetBool("sensor.photon.enable_dark_shot_noise",
                             photon.enableDarkShotNoise);
        photon.enableReadNoise =
            document.GetBool("sensor.photon.enable_read_noise", photon.enableReadNoise);
        photon.enableFpn =
            document.GetBool("sensor.photon.enable_fpn", photon.enableFpn);
        auto photonGain = ReadCalibration(
            document, "sensor.photon.nuc_gain_map",
            "sensor.photon.nuc_gain_map_path", baseDir,
            photon.nucGainMap, photon.nucGainMapPath);
        auto photonOffset = ReadCalibration(
            document, "sensor.photon.nuc_offset_electrons_map",
            "sensor.photon.nuc_offset_electrons_map_path", baseDir,
            photon.nucOffsetElectronsMap, photon.nucOffsetElectronsMapPath);
        if (!photonGain) return Error<CameraConfig>(photonGain.error());
        if (!photonOffset) return Error<CameraConfig>(photonOffset.error());

        auto& thermal = camera.thermal;
        thermal.timeConstantSeconds =
            document.GetDouble("sensor.thermal.time_constant_s", thermal.timeConstantSeconds);
        thermal.responsivityDnPerWatt =
            document.GetDouble("sensor.thermal.responsivity_dn_w",
                               thermal.responsivityDnPerWatt);
        thermal.readNoiseDnRms =
            document.GetDouble("sensor.thermal.read_noise_dn_rms", thermal.readNoiseDnRms);
        thermal.driftDnPerSecond =
            document.GetDouble("sensor.thermal.drift_dn_s", thermal.driftDnPerSecond);
        thermal.readoutWindowSeconds =
            document.GetDouble("sensor.thermal.readout_window_s",
                               thermal.readoutWindowSeconds);
        thermal.netdKelvin = document.GetDouble("sensor.thermal.netd_k", thermal.netdKelvin);
        thermal.netdReferenceTemperatureK =
            document.GetDouble("sensor.thermal.netd_reference_temperature_k",
                               thermal.netdReferenceTemperatureK);
        thermal.netdNoiseBandwidthHz =
            document.GetDouble("sensor.thermal.netd_noise_bandwidth_hz",
                               thermal.netdNoiseBandwidthHz);
        thermal.netdOpticalCondition =
            document.GetString("sensor.thermal.netd_optical_condition", "");
        auto thermalGain = ReadCalibration(
            document, "sensor.thermal.nuc_gain_map",
            "sensor.thermal.nuc_gain_map_path", baseDir,
            thermal.nucGainMap, thermal.nucGainMapPath);
        auto thermalOffset = ReadCalibration(
            document, "sensor.thermal.nuc_offset_dn_map",
            "sensor.thermal.nuc_offset_dn_map_path", baseDir,
            thermal.nucOffsetDnMap, thermal.nucOffsetDnMapPath);
        if (!thermalGain) return Error<CameraConfig>(thermalGain.error());
        if (!thermalOffset) return Error<CameraConfig>(thermalOffset.error());

        const auto channels = document.GetTableArray("sensor.channels");
        if (!channels.empty()) {
            for (const Config& table : channels) {
                auto channel = ReadChannel(table, camera.device.detector, baseDir);
                if (!channel) return Error<CameraConfig>(channel.error());
                camera.device.channels.push_back(std::move(*channel));
            }
        } else {
            if (camera.device.calibration != CalibrationStatus::GenericAssumption)
                return Error<CameraConfig>(
                    "hardware reference/calibrated sensor needs explicit response channels");
            if (camera.device.cfa == CfaPattern::MultiChannel)
                return Error<CameraConfig>(
                    "multi_channel sensor needs at least two explicit response channels");
            const auto [lo, hi] = BandFor(document, mode);
            for (const char* name : camera.device.cfa == CfaPattern::Mono
                     ? std::vector<const char*>{"Mono"}
                     : std::vector<const char*>{"R", "G", "B"}) {
                ChannelProfile channel;
                channel.name = name;
                if (camera.device.detector == DetectorKind::Photon)
                    channel.response.quantumEfficiency =
                        FlatCurve(lo, hi, 1.0, ResponseKind::AbsoluteQE,
                                  "generic detector response assumption");
                else
                    channel.response.thermalAbsorptance =
                        FlatCurve(lo, hi, 1.0, ResponseKind::ThermalAbsorptance,
                                  "generic detector response assumption");
                camera.device.channels.push_back(std::move(channel));
            }
            camera.device.parameterSources.push_back(Assumption(
                "device.channels.response", "generic flat detector response"));
        }
        if (camera.device.effectiveMinNm == 0.0 &&
            camera.device.effectiveMaxNm == 0.0) {
            f64 lo = std::numeric_limits<f64>::infinity(), hi = 0.0;
            for (const auto& channel : camera.device.channels) {
                const auto& response = channel.response;
                const ResponseCurve* base = response.systemResponse
                    ? &*response.systemResponse
                    : camera.device.detector == DetectorKind::Photon
                          ? &*response.quantumEfficiency : &*response.thermalAbsorptance;
                lo = std::min(lo, base->MinNm());
                hi = std::max(hi, base->MaxNm());
            }
            camera.device.effectiveMinNm = lo;
            camera.device.effectiveMaxNm = hi;
        }

        camera.randomSeed = document.Get<u32>("sensor.noise_seed", camera.randomSeed);
        camera.calibratedFastRgbInput =
            document.GetBool("sensor.fast_rgb.calibrated", false);
        camera.fastRgbRadianceScale =
            document.GetDouble("sensor.fast_rgb.radiance_scale",
                               camera.fastRgbRadianceScale);
        auto rgbMatrix = ReadArray<9>(document, "sensor.fast_rgb.to_device_matrix",
                                      camera.fastRgbToDevice);
        if (!rgbMatrix) return Error<CameraConfig>(rgbMatrix.error());
        camera.fastRgbToDevice = *rgbMatrix;
    }

    auto& isp = camera.isp;
    isp.autoExposure = document.GetBool("isp.auto_exposure", isp.autoExposure);
    isp.autoWhiteBalance = document.GetBool("isp.auto_white_balance", isp.autoWhiteBalance);
    auto whiteBalance = ReadArray<3>(document, "isp.white_balance", isp.whiteBalance);
    auto colorMatrix = ReadArray<9>(document, "isp.device_to_linear_srgb",
                                    isp.deviceToLinearSrgb);
    if (!whiteBalance) return Error<CameraConfig>(whiteBalance.error());
    if (!colorMatrix) return Error<CameraConfig>(colorMatrix.error());
    isp.whiteBalance = *whiteBalance;
    isp.deviceToLinearSrgb = *colorMatrix;
    isp.denoise = document.GetBool("isp.denoise", isp.denoise);
    isp.denoiseStrength = document.GetDouble("isp.denoise_strength", isp.denoiseStrength);
    isp.sharpen = document.GetBool("isp.sharpen", isp.sharpen);
    isp.sharpenStrength = document.GetDouble("isp.sharpen_strength", isp.sharpenStrength);
    isp.toneGamma = document.GetDouble("isp.tone_gamma", isp.toneGamma);
    isp.clipOutOfGamut = document.GetBool("isp.clip_out_of_gamut", isp.clipOutOfGamut);
    auto tone = Tone(document.GetString("isp.infrared_tone", "linear"));
    auto palette = Palette(document.GetString("isp.infrared_palette", "grey"));
    if (!tone) return Error<CameraConfig>(tone.error());
    if (!palette) return Error<CameraConfig>(palette.error());
    isp.infraredTone = *tone;
    isp.infraredPalette = *palette;
    isp.contrastLowPercentile =
        document.GetDouble("isp.contrast_low_percentile", isp.contrastLowPercentile);
    isp.contrastHighPercentile =
        document.GetDouble("isp.contrast_high_percentile", isp.contrastHighPercentile);
    isp.hsv.hueOffsetDegrees =
        document.GetDouble("isp.hsv.hue_offset_deg", isp.hsv.hueOffsetDegrees);
    isp.hsv.saturationScale =
        document.GetDouble("isp.hsv.saturation_scale", isp.hsv.saturationScale);
    isp.hsv.valueGamma =
        document.GetDouble("isp.hsv.value_gamma", isp.hsv.valueGamma);
    isp.hsv.empiricalNoise =
        document.GetBool("effects.hsv.empirical_noise", isp.hsv.empiricalNoise);
    isp.hsv.temporalDrift =
        document.GetBool("effects.hsv.temporal_drift", isp.hsv.temporalDrift);

    auto backend = Backend(document.GetString("sensor.quality.backend", "cpu_reference"));
    if (!backend) return Error<CameraConfig>(backend.error());
    camera.quality.backend = *backend;
    camera.quality.wavelengthSamples =
        document.Get<u32>("sensor.quality.wavelength_samples",
                          camera.quality.wavelengthSamples);
    camera.quality.timeSamples =
        document.Get<u32>("sensor.quality.time_samples", camera.quality.timeSamples);
    camera.quality.pixelSamples =
        document.Get<u32>("sensor.quality.pixel_samples", camera.quality.pixelSamples);
    camera.quality.gpuTimePositions =
        document.Get<u32>("sensor.quality.gpu_time_positions",
                          camera.quality.gpuTimePositions);
    camera.quality.noiseFree =
        document.GetBool("sensor.quality.noise_free", camera.quality.noiseFree);
    auto colorSpace = ColorSpace(
        document.GetString("sensor.output_color_space", "display_srgb"));
    if (!colorSpace) return Error<CameraConfig>(colorSpace.error());
    camera.outputColorSpace = *colorSpace;
    camera.products.tracedRadiance =
        document.GetBool("sensor.products.traced_radiance", camera.products.tracedRadiance);
    camera.products.cieLinearSrgb =
        document.GetBool("sensor.products.cie_linear_srgb", camera.products.cieLinearSrgb);
    camera.products.bandMeasurement =
        document.GetBool("sensor.products.band_measurement", camera.products.bandMeasurement);
    camera.products.rawDn =
        document.GetBool("sensor.products.raw_dn", camera.products.rawDn);
    camera.products.correctedDeviceSignal =
        document.GetBool("sensor.products.corrected_device_signal",
                         camera.products.correctedDeviceSignal);
    camera.products.apparentTemperature =
        document.GetBool("sensor.products.apparent_temperature",
                         camera.products.apparentTemperature);
    camera.products.display =
        document.GetBool("sensor.products.display", camera.products.display);

    const auto motion = ReadMotion(document, camera.motion);
    if (!motion) return Error<CameraConfig>(motion.error());
    const auto valid = ValidateCameraConfig(camera);
    if (!valid) return Error<CameraConfig>("camera config: " + valid.error());
    return camera;
}

String CameraConfigToToml(const camera::CameraConfig& camera) {
    using namespace camera;
    std::ostringstream out;
    out << std::setprecision(17);
    out << "[sensor]\n";
    out << "version = " << camera.version << '\n';
    out << "enabled = " << (camera.enabled ? "true" : "false") << '\n';
    out << "device_id = " << Quoted(camera.device.id) << '\n';
    out << "display_name = " << Quoted(camera.device.displayName) << '\n';
    out << "detector = " << Quoted(Token(camera.device.detector)) << '\n';
    out << "input_kind = " <<
        Quoted(camera.inputKind == CameraInputKind::FastRgbApproximation
                   ? "fast_rgb" : "spectral") << '\n';
    out << "cfa = " << Quoted(Token(camera.device.cfa)) << '\n';
    out << "calibration_status = " << Quoted(Token(camera.device.calibration)) << '\n';
    out << "effective_min_nm = " << camera.device.effectiveMinNm << '\n';
    out << "effective_max_nm = " << camera.device.effectiveMaxNm << '\n';
    out << "output_color_space = " << Quoted(Token(camera.outputColorSpace)) << '\n';
    out << "noise_seed = " << camera.randomSeed << '\n';
    if (!camera.device.documentVersion.empty())
        out << "document_version = " << Quoted(camera.device.documentVersion) << '\n';
    if (!camera.device.documentUrl.empty())
        out << "document_url = " << Quoted(camera.device.documentUrl) << '\n';
    if (!camera.device.readoutMode.empty())
        out << "readout_mode = " << Quoted(camera.device.readoutMode) << '\n';
    out << '\n';

    const auto& optics = camera.optics;
    out << "[sensor.optics]\n";
    out << "focal_length_mm = " << optics.focalLengthMm << '\n';
    out << "f_number = " << optics.fNumber << '\n';
    out << "pixel_pitch_um = " << optics.pixelPitchUm << '\n';
    out << "psf_sigma_px = " << optics.psfSigmaPixelsOverride << '\n';
    out << "fill_factor = " << optics.fillFactor << '\n';
    out << "sensor_width_px = " << optics.sensorWidthPx << '\n';
    out << "sensor_height_px = " << optics.sensorHeightPx << '\n';
    if (std::isfinite(optics.focusDistanceM))
        out << "focus_distance_m = " << optics.focusDistanceM << '\n';
    out << "cos_fourth_vignetting = "
        << (optics.cosFourthVignetting ? "true" : "false") << '\n';
    if (!optics.knownPsfSourcePath.empty() || !optics.knownPsfPath.empty())
        out << "known_psf_path = " <<
            Quoted(optics.knownPsfSourcePath.empty()
                       ? optics.knownPsfPath : optics.knownPsfSourcePath) << '\n';
    out << "\n[sensor.exposure]\ntime_s = " << camera.readout.exposureSeconds << '\n';

    const auto& readout = camera.readout;
    out << "\n[sensor.readout]\n";
    out << "shutter = " << Quoted(readout.shutter == ShutterKind::Global
                                    ? "global" : "rolling") << '\n';
    out << "row_delay_s = " << readout.rowDelaySeconds << '\n';
    out << "frame_period_s = " << readout.framePeriodSeconds << '\n';
    out << "analog_gain = " << readout.analogGain << '\n';
    out << "electrons_per_dn = " << readout.electronsPerDn << '\n';
    out << "black_level_dn = " << readout.blackLevelDn << '\n';
    out << "adc_bits = " << readout.adcBits << '\n';
    out << "output_bits = " << readout.outputBits << '\n';
    out << "effective_bits = " << readout.effectiveBits << '\n';

    const auto& photon = camera.photon;
    out << "\n[sensor.photon]\n";
    out << "full_well_e = " << photon.fullWellElectrons << '\n';
    out << "dark_current_e_s = " << photon.darkCurrentElectronsPerSecond << '\n';
    out << "read_noise_e_rms = " << photon.readNoiseElectronsRms << '\n';
    out << "prnu_sigma = " << photon.prnuSigma << '\n';
    out << "dsnu_e_rms = " << photon.dsnuElectronsRms << '\n';
    out << "dsnu_reference_exposure_s = "
        << photon.dsnuReferenceExposureSeconds << '\n';
    out << "bias_dn_rms = " << photon.biasDnRms << '\n';
    out << "apply_nuc = " << (photon.applyNuc ? "true" : "false") << '\n';
    out << "nuc_residual_fraction = " << photon.nucResidualFraction << '\n';
    out << "enable_shot_noise = " << (photon.enableShotNoise ? "true" : "false") << '\n';
    out << "enable_dark_current = " << (photon.enableDarkCurrent ? "true" : "false") << '\n';
    out << "enable_dark_shot_noise = "
        << (photon.enableDarkShotNoise ? "true" : "false") << '\n';
    out << "enable_read_noise = " << (photon.enableReadNoise ? "true" : "false") << '\n';
    out << "enable_fpn = " << (photon.enableFpn ? "true" : "false") << '\n';
    if (!photon.nucGainMapPath.empty()) {
        out << "nuc_gain_map_path = " << Quoted(photon.nucGainMapPath) << '\n';
    } else if (!photon.nucGainMap.empty()) {
        out << "nuc_gain_map = ";
        WriteArray(out, photon.nucGainMap);
        out << '\n';
    }
    if (!photon.nucOffsetElectronsMapPath.empty()) {
        out << "nuc_offset_electrons_map_path = "
            << Quoted(photon.nucOffsetElectronsMapPath) << '\n';
    } else if (!photon.nucOffsetElectronsMap.empty()) {
        out << "nuc_offset_electrons_map = ";
        WriteArray(out, photon.nucOffsetElectronsMap);
        out << '\n';
    }

    const auto& thermal = camera.thermal;
    out << "\n[sensor.thermal]\n";
    out << "time_constant_s = " << thermal.timeConstantSeconds << '\n';
    out << "responsivity_dn_w = " << thermal.responsivityDnPerWatt << '\n';
    out << "read_noise_dn_rms = " << thermal.readNoiseDnRms << '\n';
    out << "drift_dn_s = " << thermal.driftDnPerSecond << '\n';
    out << "readout_window_s = " << thermal.readoutWindowSeconds << '\n';
    out << "netd_k = " << thermal.netdKelvin << '\n';
    out << "netd_reference_temperature_k = "
        << thermal.netdReferenceTemperatureK << '\n';
    out << "netd_noise_bandwidth_hz = " << thermal.netdNoiseBandwidthHz << '\n';
    if (!thermal.netdOpticalCondition.empty())
        out << "netd_optical_condition = "
            << Quoted(thermal.netdOpticalCondition) << '\n';
    if (!thermal.nucGainMapPath.empty()) {
        out << "nuc_gain_map_path = " << Quoted(thermal.nucGainMapPath) << '\n';
    } else if (!thermal.nucGainMap.empty()) {
        out << "nuc_gain_map = ";
        WriteArray(out, thermal.nucGainMap);
        out << '\n';
    }
    if (!thermal.nucOffsetDnMapPath.empty()) {
        out << "nuc_offset_dn_map_path = " << Quoted(thermal.nucOffsetDnMapPath) << '\n';
    } else if (!thermal.nucOffsetDnMap.empty()) {
        out << "nuc_offset_dn_map = ";
        WriteArray(out, thermal.nucOffsetDnMap);
        out << '\n';
    }
    out << '\n';

    for (const auto& source : camera.device.parameterSources) {
        out << "[[sensor.parameter_sources]]\n";
        WriteSource(out, source);
        out << '\n';
    }
    for (const auto& channel : camera.device.channels) {
        out << "[[sensor.channels]]\n";
        out << "name = " << Quoted(channel.name) << "\n\n";
        const auto& response = channel.response;
        if (response.lensTransmission)
            WriteCurve(out, "lens", *response.lensTransmission);
        if (response.filterTransmission)
            WriteCurve(out, "filter", *response.filterTransmission);
        if (response.quantumEfficiency)
            WriteCurve(out, "qe", *response.quantumEfficiency);
        if (response.thermalAbsorptance)
            WriteCurve(out, "absorptance", *response.thermalAbsorptance);
        if (response.systemResponse)
            WriteCurve(out, "system", *response.systemResponse);
    }

    const auto& quality = camera.quality;
    out << "[sensor.quality]\n";
    out << "backend = " << Quoted(Token(quality.backend)) << '\n';
    out << "wavelength_samples = " << quality.wavelengthSamples << '\n';
    out << "time_samples = " << quality.timeSamples << '\n';
    out << "pixel_samples = " << quality.pixelSamples << '\n';
    out << "gpu_time_positions = " << quality.gpuTimePositions << '\n';
    out << "noise_free = " << (quality.noiseFree ? "true" : "false") << "\n\n";
    const auto& products = camera.products;
    out << "[sensor.products]\n";
    out << "traced_radiance = " << (products.tracedRadiance ? "true" : "false") << '\n';
    out << "cie_linear_srgb = " << (products.cieLinearSrgb ? "true" : "false") << '\n';
    out << "band_measurement = " << (products.bandMeasurement ? "true" : "false") << '\n';
    out << "raw_dn = " << (products.rawDn ? "true" : "false") << '\n';
    out << "corrected_device_signal = "
        << (products.correctedDeviceSignal ? "true" : "false") << '\n';
    out << "apparent_temperature = "
        << (products.apparentTemperature ? "true" : "false") << '\n';
    out << "display = " << (products.display ? "true" : "false") << "\n\n";
    out << "[sensor.fast_rgb]\n";
    out << "calibrated = " << (camera.calibratedFastRgbInput ? "true" : "false") << '\n';
    out << "radiance_scale = " << camera.fastRgbRadianceScale << '\n';
    out << "to_device_matrix = ";
    WriteArray(out, camera.fastRgbToDevice);
    out << "\n\n";

    const auto& isp = camera.isp;
    out << "[isp]\n";
    out << "auto_exposure = " << (isp.autoExposure ? "true" : "false") << '\n';
    out << "auto_white_balance = " << (isp.autoWhiteBalance ? "true" : "false") << '\n';
    out << "white_balance = ";
    WriteArray(out, isp.whiteBalance);
    out << "\ndevice_to_linear_srgb = ";
    WriteArray(out, isp.deviceToLinearSrgb);
    out << '\n';
    out << "denoise = " << (isp.denoise ? "true" : "false") << '\n';
    out << "denoise_strength = " << isp.denoiseStrength << '\n';
    out << "sharpen = " << (isp.sharpen ? "true" : "false") << '\n';
    out << "sharpen_strength = " << isp.sharpenStrength << '\n';
    out << "tone_gamma = " << isp.toneGamma << '\n';
    out << "clip_out_of_gamut = " << (isp.clipOutOfGamut ? "true" : "false") << '\n';
    out << "infrared_tone = " << Quoted(Token(isp.infraredTone)) << '\n';
    out << "infrared_palette = " << Quoted(Token(isp.infraredPalette)) << '\n';
    out << "contrast_low_percentile = " << isp.contrastLowPercentile << '\n';
    out << "contrast_high_percentile = " << isp.contrastHighPercentile << "\n\n";
    out << "[isp.hsv]\n";
    out << "hue_offset_deg = " << isp.hsv.hueOffsetDegrees << '\n';
    out << "saturation_scale = " << isp.hsv.saturationScale << '\n';
    out << "value_gamma = " << isp.hsv.valueGamma << "\n\n";
    out << "[effects.hsv]\n";
    out << "empirical_noise = "
        << (isp.hsv.empiricalNoise ? "true" : "false") << '\n';
    out << "temporal_drift = "
        << (isp.hsv.temporalDrift ? "true" : "false") << "\n\n";

    if (!camera.motion.keys.empty()) {
        out << "[camera.motion]\n";
        out << "interpolation = \"linear\"\n";
        out << "extrapolate = \"hold\"\n\n";
        for (const auto& key : camera.motion.keys) {
            out << "[[camera.motion.keys]]\n";
            out << "t = " << key.timeSeconds << '\n';
            out << "position = ";
            WriteArray(out, key.position);
            out << "\nlook_at = ";
            WriteArray(out, key.lookAt);
            out << "\n\n";
        }
    }
    return out.str();
}

Result<camera::CameraConfig, String> CameraConfigFromSensorParams(
    const SensorParams& sensor, SpectralMode mode, u32 physicalWidth,
    u32 physicalHeight, f64 verticalFovDegrees, f64 wavelengthNm) {
    if (physicalWidth == 0 || physicalHeight == 0 ||
        !std::isfinite(verticalFovDegrees) ||
        verticalFovDegrees <= 0.0 || verticalFovDegrees >= 180.0)
        return Error<camera::CameraConfig>(
            "legacy sensor conversion needs physical width, height and vertical FOV");
    // The old API carries a SensorParams value, while ParseCameraConfig owns
    // every legacy-to-versioned rule. Feed those exact legacy keys through
    // that one parser rather than implementing a second host-side mapping.
    std::ostringstream toml;
    toml << std::setprecision(std::numeric_limits<f32>::max_digits10);
    toml << "[renderer]\nresolution = [" << physicalWidth << ", "
         << physicalHeight << "]\n";
    toml << "[camera]\nfov_y = " << verticalFovDegrees << "\n";
    toml << "[spectral]\nwavelength_nm = " << wavelengthNm << "\n";
    toml << "[sensor]\nenabled = true\n";
    toml << "focal_length_mm = " << sensor.focalLength_mm << '\n';
    toml << "f_number = " << sensor.fNumber << '\n';
    toml << "pixel_pitch_um = " << sensor.pixelPitch_um << '\n';
    toml << "psf_sigma_px = " << sensor.psfSigma_px << '\n';
    toml << "quantum_efficiency = " << sensor.quantumEfficiency << '\n';
    toml << "well_capacity_e = " << sensor.wellCapacity_e << '\n';
    toml << "read_noise_e_rms = " << sensor.readNoise_e_rms << '\n';
    toml << "dark_current_e_s = " << sensor.darkCurrent_e_s << '\n';
    toml << "integration_time_s = " << sensor.integrationTime_s << '\n';
    toml << "bit_depth = " << sensor.bitDepth << '\n';
    toml << "gain = " << sensor.gain << '\n';
    toml << "enable_poisson_noise = "
         << (sensor.enablePoissonNoise ? "true" : "false") << '\n';
    toml << "enable_read_noise = "
         << (sensor.enableReadNoise ? "true" : "false") << '\n';
    toml << "enable_dark_current = "
         << (sensor.enableDarkCurrent ? "true" : "false") << '\n';
    toml << "enable_fpn = " << (sensor.enableFPN ? "true" : "false") << '\n';
    toml << "noise_seed = " << sensor.noiseSeed << '\n';
    toml << "detector_temperature_k = " << sensor.detectorTemperature_K << '\n';
    toml << "[sensor.fpn]\n";
    toml << "prnu_sigma = " << sensor.prnuSigma << '\n';
    toml << "dsnu_sigma_e = " << sensor.dsnuSigma_e << '\n';
    toml << "enable_nuc = " << (sensor.enableNUC ? "true" : "false") << '\n';
    toml << "nuc_efficiency = " << sensor.nucEfficiency << '\n';
    auto document = Config::Parse(toml.str());
    if (!document) return Error<camera::CameraConfig>(document.error());
    auto migrated = ParseCameraConfig(*document, mode);
    if (!migrated) return migrated;
    migrated.value().optics.psfSigmaPixelsOverride = sensor.psfSigma_px;
    migrated.value().optics.cosFourthVignetting =
        sensor.enableVignetting && !sensor.isTelecentric;
    return migrated;
}

} // namespace quantiloom
