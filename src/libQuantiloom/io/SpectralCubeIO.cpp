/**
 * @file SpectralCubeIO.cpp
 * @brief Implementation of SpectralCube I/O utilities
 *
 * @author blitzcolo
 */

#include "io/SpectralCubeIO.hpp"
#include "core/Log.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>

// OpenEXR headers - DISABLED due to static initialization conflict with VMA
// See: https://github.com/AcademySoftwareFoundation/openexr/issues/1234
// The multipart OpenEXR API causes shutdown crashes when used alongside
// Vulkan/VMA in the same process. Using ENVI format for SpectralCube I/O instead.
// Single-image EXR I/O via ImageIO is unaffected.

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

// NOTE: Full OpenEXR multipart implementation causes shutdown crash due to
// static initialization order issues with VMA/Vulkan. Using stub implementations
// for now. SpectralCube EXR I/O will be implemented via ImageIO utilities.

bool SpectralCubeIO::WriteEXR(
    const SpectralCube& cube,
    const String& path
) {
    LOG_WARN("SpectralCubeIO::WriteEXR not yet implemented - use ENVI format instead");
    return false;
}

Result<SpectralCube, String> SpectralCubeIO::ReadEXR(const String& path) {
    return Result<SpectralCube, String>::Err("SpectralCubeIO::ReadEXR not yet implemented - use ENVI format instead");
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
