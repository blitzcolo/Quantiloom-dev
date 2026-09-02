/**
 * @file SpectralCubeIO.cpp
 * @brief Implementation of SpectralCube I/O utilities
 *
 * @author blitzcolo
 */

#include "io/SpectralCubeIO.hpp"
#include "io/ImageIO.hpp"
#include "core/Log.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

// OpenEXR is reached through ImageIO, which writes a single part. The multipart
// API is what brought a shutdown crash here, through a static initialisation
// order it shares with VMA
// (https://github.com/AcademySoftwareFoundation/openexr/issues/1234), and the
// spectral layout below wants one part anyway.

namespace quantiloom {

// ============================================================================
// ENVIInterleave Helpers
// ============================================================================

const char* ENVIInterleaveToString(ENVIInterleave interleave) {
    switch (interleave) {
        case ENVIInterleave::BSQ: return "bsq";
        case ENVIInterleave::BIL: return "bil";
        case ENVIInterleave::BIP: return "bip";
        default: return "bsq";
    }
}

static ENVIInterleave StringToENVIInterleave(const String& str) {
    String lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "bsq") return ENVIInterleave::BSQ;
    if (lower == "bil") return ENVIInterleave::BIL;
    if (lower == "bip") return ENVIInterleave::BIP;

    return ENVIInterleave::BSQ;  // Default
}

// ============================================================================
// ENVI Format Implementation
// ============================================================================

bool SpectralCubeIO::WriteENVIHeader(
    const String& headerPath,
    const SpectralCube& cube,
    ENVIInterleave interleave
) {
    std::ofstream hdr(headerPath);
    if (!hdr.is_open()) {
        LOG_ERROR("Failed to create ENVI header: {}", headerPath);
        return false;
    }

    hdr << "ENVI\n";
    hdr << "description = {Quantiloom Hyperspectral Render Output}\n";
    hdr << "samples = " << cube.width << "\n";
    hdr << "lines = " << cube.height << "\n";
    hdr << "bands = " << cube.nbands << "\n";
    hdr << "header offset = 0\n";
    hdr << "file type = ENVI Standard\n";
    hdr << "data type = 4\n";  // float32
    hdr << "interleave = " << ENVIInterleaveToString(interleave) << "\n";
    hdr << "byte order = 0\n";  // Little endian

    // Write wavelength list
    hdr << "wavelength = {\n";
    for (u32 i = 0; i < cube.nbands; ++i) {
        hdr << std::fixed << std::setprecision(2) << cube.wavelengths[i];
        if (i < cube.nbands - 1) {
            hdr << ", ";
            if ((i + 1) % 10 == 0) hdr << "\n";
        }
    }
    hdr << "}\n";

    hdr << "wavelength units = Nanometers\n";

    // Write custom metadata
    for (const auto& [key, value] : cube.metadata) {
        hdr << key << " = " << value << "\n";
    }

    hdr.close();
    return true;
}

Vector<f32> SpectralCubeIO::ConvertInterleave(
    const SpectralCube& cube,
    ENVIInterleave targetInterleave
) {
    const u32 W = cube.width;
    const u32 H = cube.height;
    const u32 B = cube.nbands;

    Vector<f32> output(W * H * B);

    switch (targetInterleave) {
        case ENVIInterleave::BSQ:
            // SpectralCube is already BSQ, direct copy
            std::memcpy(output.data(), cube.data.data(), W * H * B * sizeof(f32));
            break;

        case ENVIInterleave::BIL:
            // BIL: [lines][bands][samples]
            for (u32 y = 0; y < H; ++y) {
                for (u32 b = 0; b < B; ++b) {
                    for (u32 x = 0; x < W; ++x) {
                        u32 bilIdx = (y * B + b) * W + x;
                        output[bilIdx] = cube(x, y, b);
                    }
                }
            }
            break;

        case ENVIInterleave::BIP:
            // BIP: [lines][samples][bands]
            for (u32 y = 0; y < H; ++y) {
                for (u32 x = 0; x < W; ++x) {
                    for (u32 b = 0; b < B; ++b) {
                        u32 bipIdx = (y * W + x) * B + b;
                        output[bipIdx] = cube(x, y, b);
                    }
                }
            }
            break;
    }

    return output;
}

void SpectralCubeIO::ConvertToBSQ(
    const Vector<f32>& data,
    SpectralCube& cube,
    ENVIInterleave sourceInterleave
) {
    const u32 W = cube.width;
    const u32 H = cube.height;
    const u32 B = cube.nbands;

    switch (sourceInterleave) {
        case ENVIInterleave::BSQ:
            // Direct copy
            std::memcpy(cube.data.data(), data.data(), W * H * B * sizeof(f32));
            break;

        case ENVIInterleave::BIL:
            // BIL: [lines][bands][samples] -> BSQ
            for (u32 y = 0; y < H; ++y) {
                for (u32 b = 0; b < B; ++b) {
                    for (u32 x = 0; x < W; ++x) {
                        u32 bilIdx = (y * B + b) * W + x;
                        cube(x, y, b) = data[bilIdx];
                    }
                }
            }
            break;

        case ENVIInterleave::BIP:
            // BIP: [lines][samples][bands] -> BSQ
            for (u32 y = 0; y < H; ++y) {
                for (u32 x = 0; x < W; ++x) {
                    for (u32 b = 0; b < B; ++b) {
                        u32 bipIdx = (y * W + x) * B + b;
                        cube(x, y, b) = data[bipIdx];
                    }
                }
            }
            break;
    }
}

bool SpectralCubeIO::WriteENVI(
    const SpectralCube& cube,
    const String& basePath,
    ENVIInterleave interleave
) {
    if (!cube.IsValid()) {
        LOG_ERROR("Cannot write invalid SpectralCube");
        return false;
    }

    String headerPath = basePath + ".hdr";
    String dataPath = basePath + ".dat";

    // Write header
    if (!WriteENVIHeader(headerPath, cube, interleave)) {
        return false;
    }

    // Convert and write data
    Vector<f32> outputData = ConvertInterleave(cube, interleave);

    std::ofstream dat(dataPath, std::ios::binary);
    if (!dat.is_open()) {
        LOG_ERROR("Failed to create ENVI data file: {}", dataPath);
        return false;
    }

    dat.write(reinterpret_cast<const char*>(outputData.data()),
              outputData.size() * sizeof(f32));

    if (!dat.good()) {
        LOG_ERROR("Error writing ENVI data file");
        return false;
    }

    dat.close();

    LOG_INFO("Wrote ENVI hyperspectral cube: {} ({} x {} x {} bands)",
             basePath, cube.width, cube.height, cube.nbands);

    return true;
}

Result<std::map<String, String>, String> SpectralCubeIO::ParseENVIHeader(
    const String& headerPath
) {
    std::ifstream hdr(headerPath);
    if (!hdr.is_open()) {
        return Result<std::map<String, String>, String>::Err("Failed to open ENVI header: " + headerPath);
    }

    std::map<String, String> params;
    String line;
    String currentKey;
    String currentValue;
    bool inBraces = false;

    // Check for ENVI signature
    std::getline(hdr, line);
    if (line.find("ENVI") == String::npos) {
        return Result<std::map<String, String>, String>::Err("Not a valid ENVI header (missing ENVI signature)");
    }

    while (std::getline(hdr, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        if (inBraces) {
            // Continue collecting multi-line value
            currentValue += " " + line;
            if (line.find('}') != String::npos) {
                inBraces = false;
                // Remove braces
                size_t start = currentValue.find('{');
                size_t end = currentValue.rfind('}');
                if (start != String::npos && end != String::npos) {
                    currentValue = currentValue.substr(start + 1, end - start - 1);
                }
                params[currentKey] = currentValue;
            }
            continue;
        }

        // Parse key = value
        size_t eqPos = line.find('=');
        if (eqPos == String::npos) continue;

        currentKey = line.substr(0, eqPos);
        currentValue = line.substr(eqPos + 1);

        // Trim whitespace
        auto trim = [](String& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(currentKey);
        trim(currentValue);

        // Check for multi-line value
        if (currentValue.find('{') != String::npos &&
            currentValue.find('}') == String::npos) {
            inBraces = true;
            continue;
        }

        // Remove braces if present
        if (!currentValue.empty() && currentValue.front() == '{' && currentValue.back() == '}') {
            currentValue = currentValue.substr(1, currentValue.size() - 2);
        }

        params[currentKey] = currentValue;
    }

    return params;
}

Result<SpectralCube, String> SpectralCubeIO::ReadENVI(const String& basePath) {
    // Determine header and data paths
    String headerPath = basePath;
    if (headerPath.size() < 4 || headerPath.substr(headerPath.size() - 4) != ".hdr") {
        headerPath += ".hdr";
    }

    // Parse header
    auto headerResult = ParseENVIHeader(headerPath);
    if (!headerResult.has_value()) {
        return Result<SpectralCube, String>::Err(headerResult.error());
    }

    const auto& params = headerResult.value();

    // Extract required parameters using std::optional
    auto getParam = [&params](const String& key) -> std::optional<String> {
        auto it = params.find(key);
        if (it == params.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    auto samplesResult = getParam("samples");
    auto linesResult = getParam("lines");
    auto bandsResult = getParam("bands");
    auto dataTypeResult = getParam("data type");
    auto interleaveResult = getParam("interleave");

    if (!samplesResult) return Result<SpectralCube, String>::Err("Missing required parameter: samples");
    if (!linesResult) return Result<SpectralCube, String>::Err("Missing required parameter: lines");
    if (!bandsResult) return Result<SpectralCube, String>::Err("Missing required parameter: bands");

    u32 width = static_cast<u32>(std::stoul(*samplesResult));
    u32 height = static_cast<u32>(std::stoul(*linesResult));
    u32 nbands = static_cast<u32>(std::stoul(*bandsResult));

    // Get data type (must be float32 = 4)
    u32 dataType = dataTypeResult ? static_cast<u32>(std::stoul(*dataTypeResult)) : 4;
    if (dataType != 4) {
        return Result<SpectralCube, String>::Err("Unsupported ENVI data type: " + std::to_string(dataType) + " (only float32 = 4 supported)");
    }

    // Get interleave
    ENVIInterleave interleave = ENVIInterleave::BSQ;
    if (interleaveResult.has_value()) {
        interleave = StringToENVIInterleave(interleaveResult.value());
    }

    // Parse wavelengths
    f32 lambdaMin = 0.0f;
    f32 lambdaMax = 0.0f;
    Vector<f32> wavelengths;

    auto wavelengthResult = getParam("wavelength");
    if (wavelengthResult.has_value()) {
        // Parse comma-separated wavelengths
        std::istringstream ss(wavelengthResult.value());
        String token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n"));
            token.erase(token.find_last_not_of(" \t\n") + 1);
            if (!token.empty()) {
                wavelengths.push_back(std::stof(token));
            }
        }

        if (!wavelengths.empty()) {
            lambdaMin = wavelengths.front();
            lambdaMax = wavelengths.back();
        }
    }

    // Fallback wavelength range if not specified
    if (wavelengths.empty()) {
        lambdaMin = 400.0f;
        lambdaMax = 400.0f + (nbands - 1) * 10.0f;
    }

    // Create cube
    SpectralCube cube(width, height, nbands, lambdaMin, lambdaMax);

    // Override wavelengths if custom ones were provided
    if (!wavelengths.empty() && wavelengths.size() == nbands) {
        cube.wavelengths = wavelengths;
    }

    // Read data file
    String dataPath = basePath;
    if (dataPath.size() >= 4 && dataPath.substr(dataPath.size() - 4) == ".hdr") {
        dataPath = dataPath.substr(0, dataPath.size() - 4);
    }

    // Try .dat, then .raw
    std::ifstream dat(dataPath + ".dat", std::ios::binary);
    if (!dat.is_open()) {
        dat.open(dataPath + ".raw", std::ios::binary);
    }
    if (!dat.is_open()) {
        return Result<SpectralCube, String>::Err("Failed to open ENVI data file: " + dataPath);
    }

    // Check header offset
    u32 headerOffset = 0;
    auto offsetResult = getParam("header offset");
    if (offsetResult.has_value()) {
        headerOffset = static_cast<u32>(std::stoul(offsetResult.value()));
    }
    dat.seekg(headerOffset);

    // Read data
    Vector<f32> rawData(width * height * nbands);
    dat.read(reinterpret_cast<char*>(rawData.data()), rawData.size() * sizeof(f32));

    if (!dat.good()) {
        return Result<SpectralCube, String>::Err("Error reading ENVI data file");
    }

    // Convert to BSQ (internal format)
    ConvertToBSQ(rawData, cube, interleave);

    LOG_INFO("Read ENVI hyperspectral cube: {} x {} x {} bands, {}-{} nm",
             width, height, nbands, lambdaMin, lambdaMax);

    return std::move(cube);
}

// ============================================================================
// GeoTIFF Format Implementation
// ============================================================================

bool SpectralCubeIO::WriteGeoTIFF(
    const SpectralCube& cube,
    const String& path
) {
    // GeoTIFF requires libtiff - for now, emit a simple multi-band raw format
    // that can be imported by GDAL/QGIS with the right settings

    LOG_WARN("GeoTIFF output not yet implemented - falling back to raw binary");

    // Write as raw binary with a simple sidecar text file describing the format
    String rawPath = path;
    if (rawPath.size() >= 4 && rawPath.substr(rawPath.size() - 4) == ".tif") {
        rawPath = rawPath.substr(0, rawPath.size() - 4);
    }

    // Write data
    std::ofstream raw(rawPath + ".raw", std::ios::binary);
    if (!raw.is_open()) {
        LOG_ERROR("Failed to create raw file: {}", rawPath);
        return false;
    }

    raw.write(reinterpret_cast<const char*>(cube.data.data()),
              cube.data.size() * sizeof(f32));
    raw.close();

    // Write metadata file for GDAL import
    std::ofstream meta(rawPath + ".hdr");
    meta << "ENVI\n";  // GDAL can read ENVI-style headers
    meta << "samples = " << cube.width << "\n";
    meta << "lines = " << cube.height << "\n";
    meta << "bands = " << cube.nbands << "\n";
    meta << "data type = 4\n";
    meta << "interleave = bsq\n";
    meta << "byte order = 0\n";
    meta.close();

    LOG_INFO("Wrote raw hyperspectral data (GDAL-compatible): {}.raw", rawPath);
    return true;
}

Result<SpectralCube, String> SpectralCubeIO::ReadGeoTIFF(const String& path) {
    // TODO: Implement GeoTIFF reading with libtiff
    return Result<SpectralCube, String>::Err("GeoTIFF reading not yet implemented");
}

// ============================================================================
// OpenEXR Format Implementation
// ============================================================================

// The layout is Fichet, Pacanowski and Wilkie 2021 (JCGT 10(3)): one channel
// per band in a single part, the wavelength spelled into the channel name, and
// string attributes saying which layout and which units. ART, Mitsuba and the
// spectral-exr tools read it, which is the whole reason to prefer it over a
// private convention -- a cube leaves here as something another renderer can
// open, rather than as an image with sixty-four channels named by us.
//
// A cube this renderer writes is emissive and unpolarised, so the Stokes
// component is always 0 and the prefix is always "S0.".

namespace {

constexpr const char* kSpectralLayoutVersion = "1.0";

// Per band, per steradian, per square metre, per nanometre: the cube holds a
// spectral radiance density, not a band integral, and the layout's usual
// "W.m^-2.sr^-1" would be the wrong statement by a factor of the bandwidth.
constexpr const char* kDefaultEmissiveUnits = "W.m^-2.sr^-1.nm^-1";

// The wavelength as a channel name spells it: the shortest decimal that reads
// back as the same f32, with a comma for the decimal separator because a dot is
// OpenEXR's layer separator.
String WavelengthToChannelText(const f32 wavelength_nm) {
    char buf[32];
    String text;
    if (auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), wavelength_nm);
        ec == std::errc{}) {
        text.assign(buf, end);
    } else {
        text = std::to_string(wavelength_nm);  // unreachable for an f32 in 32 bytes
    }
    std::replace(text.begin(), text.end(), '.', ',');
    return text;
}

String EmissiveChannelName(const f32 wavelength_nm) {
    return "S0." + WavelengthToChannelText(wavelength_nm) + "nm";
}

// Nanometres per unit of the suffix a channel name ends in, or 0 for a suffix
// this reader does not know. Only the metre family: the layout also allows a
// frequency axis, which this cube has no way to hold.
f32 ChannelUnitToNanometres(const std::string_view unit) {
    if (unit == "nm") return 1.0f;
    if (unit == "pm") return 1.0e-3f;
    if (unit == "um") return 1.0e3f;
    if (unit == "mm") return 1.0e6f;
    if (unit == "cm") return 1.0e7f;
    if (unit == "dm") return 1.0e8f;
    if (unit == "m")  return 1.0e9f;
    return 0.0f;
}

// The wavelength a spectral channel name carries, in nm, or nothing for a
// channel that is not one under this prefix -- an RGB proxy, an alpha, a Stokes
// component beyond the first.
std::optional<f32> ChannelNameToWavelength(
    const String& name,
    const std::string_view prefix
) {
    if (name.size() <= prefix.size() ||
        std::string_view(name).substr(0, prefix.size()) != prefix) {
        return std::nullopt;
    }

    const String body = name.substr(prefix.size());

    // The unit is the trailing run of letters. It cannot be ambiguous: a number
    // never ends in one, and an exponent's 'e' is always followed by a digit.
    size_t unitStart = body.size();
    while (unitStart > 0 &&
           std::isalpha(static_cast<unsigned char>(body[unitStart - 1])) != 0) {
        --unitStart;
    }

    const f32 scale = ChannelUnitToNanometres(std::string_view(body).substr(unitStart));
    if (scale <= 0.0f) return std::nullopt;

    String number = body.substr(0, unitStart);
    std::replace(number.begin(), number.end(), ',', '.');

    f32 value = 0.0f;
    const char* first = number.data();
    const char* last = first + number.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr != last) return std::nullopt;

    return value * scale;
}

}  // namespace

bool SpectralCubeIO::WriteEXR(
    const SpectralCube& cube,
    const String& path
) {
    if (!cube.IsValid()) {
        LOG_ERROR("Cannot write invalid SpectralCube");
        return false;
    }

    Image image(cube.width, cube.height, cube.nbands);

    // Two bands at one wavelength would collide in the channel list, which is a
    // name-keyed map: the file would come back with fewer bands than it was
    // given and no error anywhere.
    std::set<String> seen;
    for (u32 b = 0; b < cube.nbands; ++b) {
        String name = EmissiveChannelName(cube.wavelengths[b]);
        if (!seen.insert(name).second) {
            LOG_ERROR("Two bands share the wavelength {} nm; EXR channel names would collide",
                      cube.wavelengths[b]);
            return false;
        }
        image.channelNames[b] = std::move(name);
    }

    // BSQ to channel-last.
    for (u32 b = 0; b < cube.nbands; ++b) {
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                image(x, y, b) = cube(x, y, b);
            }
        }
    }

    image.metadata = cube.metadata;
    image.metadata["spectralLayoutVersion"] = kSpectralLayoutVersion;
    if (!image.metadata.contains("emissiveUnits")) {
        image.metadata["emissiveUnits"] = kDefaultEmissiveUnits;
    }

    if (!ImageIO::WriteEXR(path, image)) {
        return false;
    }

    LOG_INFO("Wrote spectral EXR cube: {} ({} x {} x {} bands, {}-{} nm)",
             path, cube.width, cube.height, cube.nbands,
             cube.wavelengths.front(), cube.wavelengths.back());
    return true;
}

Result<SpectralCube, String> SpectralCubeIO::ReadEXR(const String& path) {
    std::optional<Image> image = ImageIO::ReadEXR(path);
    if (!image.has_value()) {
        return Result<SpectralCube, String>::Err("Failed to read EXR file: " + path);
    }

    // A channel list is a name-keyed map, so the order a file hands back is
    // alphabetical -- "S0.1000nm" arrives before "S0.400nm" -- and the band
    // order has to come from the wavelengths themselves.
    Vector<std::pair<f32, u32>> bands;
    for (const std::string_view prefix : {"S0.", "T."}) {
        for (u32 c = 0; c < image->channels && c < image->channelNames.size(); ++c) {
            if (auto wavelength = ChannelNameToWavelength(image->channelNames[c], prefix)) {
                bands.emplace_back(*wavelength, c);
            }
        }
        if (!bands.empty()) break;  // emissive or reflective, never a mixture
    }

    if (bands.empty()) {
        return Result<SpectralCube, String>::Err(
            "No spectral channels (S0.<wavelength>nm) in EXR file: " + path);
    }

    std::ranges::sort(bands);

    const u32 nbands = static_cast<u32>(bands.size());
    SpectralCube cube(image->width, image->height, nbands,
                      bands.front().first, bands.back().first);

    for (u32 b = 0; b < nbands; ++b) {
        cube.wavelengths[b] = bands[b].first;
    }
    if (nbands == 1) {
        cube.delta_lambda = 0.0f;  // the constructor divides by nbands - 1
    }

    for (u32 b = 0; b < nbands; ++b) {
        const u32 c = bands[b].second;
        for (u32 y = 0; y < cube.height; ++y) {
            for (u32 x = 0; x < cube.width; ++x) {
                cube(x, y, b) = (*image)(x, y, c);
            }
        }
    }

    cube.metadata = image->metadata;

    LOG_INFO("Read spectral EXR cube: {} x {} x {} bands, {}-{} nm",
             cube.width, cube.height, cube.nbands, cube.lambda_min, cube.lambda_max);

    return std::move(cube);
}

// ============================================================================
// Utility Methods
// ============================================================================

String SpectralCubeIO::DetectFormat(const String& path) {
    std::filesystem::path p(path);
    String ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".hdr" || ext == ".dat" || ext == ".raw") return "envi";
    if (ext == ".tif" || ext == ".tiff") return "geotiff";
    if (ext == ".exr") return "exr";

    return "unknown";
}

Result<SpectralCube, String> SpectralCubeIO::Read(const String& path) {
    String format = DetectFormat(path);

    if (format == "envi") {
        return ReadENVI(path);
    } else if (format == "geotiff") {
        return ReadGeoTIFF(path);
    } else if (format == "exr") {
        return ReadEXR(path);
    }

    return Result<SpectralCube, String>::Err("Unknown file format: " + path);
}

} // namespace quantiloom
