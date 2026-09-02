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
//
// Written by hand, and the reason is what "GeoTIFF" has to mean for a cube
// that came out of a renderer. libtiff and GDAL are both large dependencies
// whose value here would be the georeferencing -- and a rendered scene has no
// coordinate reference system, no tie point and no pixel scale, so a file
// claiming one would be claiming something false. What is actually wanted is a
// TIFF that QGIS, GDAL, ENVI and Python can open with one band per wavelength
// and the wavelengths attached, and that is a baseline TIFF plus GDAL's own
// metadata tag.
//
// So: classic little-endian TIFF, one IFD, 32-bit IEEE float samples, no
// compression, PlanarConfiguration = 2 (one plane per band) because the cube is
// already band-sequential and that makes the strips a memcpy. Band descriptions
// and the per-band wavelength go in tag 42112, GDAL_METADATA, which is the tag
// gdalinfo reads and which nothing else has to understand.
//
// Classic TIFF addresses with 32 bits, so the whole file must stay under 4 GB.
// A cube that would not fit is refused rather than truncated -- BigTIFF is the
// answer to that and it is a different format, not a larger offset.

namespace {

constexpr u32 kTiffTagImageWidth        = 256;
constexpr u32 kTiffTagImageLength       = 257;
constexpr u32 kTiffTagBitsPerSample     = 258;
constexpr u32 kTiffTagCompression       = 259;
constexpr u32 kTiffTagPhotometric       = 262;
constexpr u32 kTiffTagStripOffsets      = 273;
constexpr u32 kTiffTagSamplesPerPixel   = 277;
constexpr u32 kTiffTagRowsPerStrip      = 278;
constexpr u32 kTiffTagStripByteCounts   = 279;
constexpr u32 kTiffTagPlanarConfig      = 284;
constexpr u32 kTiffTagSampleFormat      = 339;
constexpr u32 kTiffTagGdalMetadata      = 42112;

constexpr u16 kTiffTypeShort = 3;
constexpr u16 kTiffTypeLong  = 4;
constexpr u16 kTiffTypeAscii = 2;

constexpr u16 kSampleFormatIeeeFloat = 3;
constexpr u16 kPlanarSeparate = 2;
constexpr u16 kPlanarChunky = 1;

/// One directory entry, built in memory so the whole IFD can be sized before
/// any of it is written -- an entry's value lives inline when it fits in four
/// bytes and out of line when it does not, and the out-of-line offsets are only
/// knowable once the count is.
struct TiffEntry {
    u16 tag = 0;
    u16 type = 0;
    u32 count = 0;
    u32 valueOrOffset = 0;
    /// Bytes to append after the IFD, empty when the value went inline.
    std::vector<u8> payload;
};

void PutU16(std::vector<u8>& out, const u16 v) {
    out.push_back(static_cast<u8>(v & 0xFF));
    out.push_back(static_cast<u8>((v >> 8) & 0xFF));
}

void PutU32(std::vector<u8>& out, const u32 v) {
    out.push_back(static_cast<u8>(v & 0xFF));
    out.push_back(static_cast<u8>((v >> 8) & 0xFF));
    out.push_back(static_cast<u8>((v >> 16) & 0xFF));
    out.push_back(static_cast<u8>((v >> 24) & 0xFF));
}

u16 GetU16(const std::vector<u8>& in, const usize at, const bool little) {
    if (at + 2 > in.size()) return 0;
    return little ? static_cast<u16>(in[at] | (in[at + 1] << 8))
                  : static_cast<u16>((in[at] << 8) | in[at + 1]);
}

u32 GetU32(const std::vector<u8>& in, const usize at, const bool little) {
    if (at + 4 > in.size()) return 0;
    if (little) {
        return static_cast<u32>(in[at]) | (static_cast<u32>(in[at + 1]) << 8) |
               (static_cast<u32>(in[at + 2]) << 16) | (static_cast<u32>(in[at + 3]) << 24);
    }
    return (static_cast<u32>(in[at]) << 24) | (static_cast<u32>(in[at + 1]) << 16) |
           (static_cast<u32>(in[at + 2]) << 8) | static_cast<u32>(in[at + 3]);
}

/// The GDAL metadata document. Band descriptions carry the wavelength as text
/// so `gdalinfo` prints it, and a WAVELENGTH item carries it as a number so a
/// reader -- ours included -- gets it back without parsing a sentence.
String GdalMetadataXml(const SpectralCube& cube) {
    std::ostringstream xml;
    xml << "<GDALMetadata>\n";
    for (u32 b = 0; b < cube.nbands; ++b) {
        const f32 lambda = b < cube.wavelengths.size()
                               ? cube.wavelengths[b]
                               : cube.lambda_min + static_cast<f32>(b) * cube.delta_lambda;
        xml << "  <Item name=\"DESCRIPTION\" sample=\"" << b
            << "\" role=\"description\">" << std::fixed << std::setprecision(3) << lambda
            << " nm</Item>\n";
        xml << "  <Item name=\"WAVELENGTH\" sample=\"" << b << "\">" << std::fixed
            << std::setprecision(6) << lambda << "</Item>\n";
    }
    xml << "  <Item name=\"WAVELENGTH_UNITS\">nm</Item>\n";
    // The same string the spectral EXR carries, so a cube is identifiable as
    // emissive radiance in either container rather than in one of them.
    xml << "  <Item name=\"EMISSIVE_UNITS\">W.m^-2.sr^-1.nm^-1</Item>\n";
    for (const auto& [key, value] : cube.metadata) {
        xml << "  <Item name=\"" << key << "\">" << value << "</Item>\n";
    }
    xml << "</GDALMetadata>";
    return xml.str();
}

/// The value of one Item, by name and optional sample index. A hand parse
/// rather than an XML library: the document is one this file wrote, the shape
/// is fixed, and a dependency for it would be larger than the format.
bool GdalMetadataItem(const String& xml, const String& name, const int sample,
                      String& out) {
    String needle = "name=\"" + name + "\"";
    usize at = 0;
    while ((at = xml.find(needle, at)) != String::npos) {
        const usize itemStart = xml.rfind("<Item", at);
        const usize close = xml.find('>', at);
        if (itemStart == String::npos || close == String::npos) return false;
        const String attributes = xml.substr(itemStart, close - itemStart);

        bool matches = true;
        if (sample >= 0) {
            const String wanted = "sample=\"" + std::to_string(sample) + "\"";
            matches = attributes.find(wanted) != String::npos;
        }
        if (matches) {
            const usize end = xml.find("</Item>", close);
            if (end == String::npos) return false;
            out = xml.substr(close + 1, end - close - 1);
            return true;
        }
        at = close;
    }
    return false;
}

}  // namespace

bool SpectralCubeIO::WriteGeoTIFF(
    const SpectralCube& cube,
    const String& path
) {
    if (!cube.IsValid()) {
        LOG_ERROR("Cannot write invalid SpectralCube");
        return false;
    }

    const u64 planeBytes =
        static_cast<u64>(cube.width) * cube.height * sizeof(f32);
    const u64 pixelBytes = planeBytes * cube.nbands;

    // Classic TIFF offsets are 32 bits, and the header plus the IFD sit in
    // front of the data, so the ceiling is a little under 4 GB. Refused rather
    // than truncated: a file that says it has sixty-four bands and carries
    // thirty is worse than no file.
    if (pixelBytes + (1u << 20) > 0xFFFFFFFFull) {
        LOG_ERROR("SpectralCube is {} MB, past what a classic TIFF can address. "
                  "Write ENVI or spectral EXR for a cube this size.",
                  pixelBytes / (1024 * 1024));
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create TIFF file: {}", path);
        return false;
    }

    const String metadata = GdalMetadataXml(cube);

    // Entries in ascending tag order, which the spec requires and readers rely
    // on. Values of five bytes or more are appended after the IFD.
    std::vector<TiffEntry> entries;
    const auto addInline = [&entries](const u32 tag, const u16 type, const u32 value) {
        TiffEntry e;
        e.tag = static_cast<u16>(tag);
        e.type = type;
        e.count = 1;
        // A SHORT in a four-byte field sits in the low half on a little-endian
        // file, which is what "left justified within the 4-byte field" means
        // once the byte order is applied.
        e.valueOrOffset = value;
        entries.push_back(std::move(e));
    };
    const auto addArray = [&entries](const u32 tag, const u16 type, const u32 count,
                                     std::vector<u8> payload) {
        TiffEntry e;
        e.tag = static_cast<u16>(tag);
        e.type = type;
        e.count = count;
        e.payload = std::move(payload);
        entries.push_back(std::move(e));
    };

    addInline(kTiffTagImageWidth, kTiffTypeLong, cube.width);
    addInline(kTiffTagImageLength, kTiffTypeLong, cube.height);

    std::vector<u8> bitsPerSample;
    for (u32 b = 0; b < cube.nbands; ++b) PutU16(bitsPerSample, 32);
    if (cube.nbands == 1) {
        addInline(kTiffTagBitsPerSample, kTiffTypeShort, 32);
    } else {
        addArray(kTiffTagBitsPerSample, kTiffTypeShort, cube.nbands, std::move(bitsPerSample));
    }

    addInline(kTiffTagCompression, kTiffTypeShort, 1);      // none
    addInline(kTiffTagPhotometric, kTiffTypeShort, 1);      // BlackIsZero

    // One strip per band. Legal, and it makes each strip exactly one of the
    // cube's own planes, so writing is a single write per band and reading is a
    // single read.
    std::vector<u8> stripOffsets(cube.nbands * 4, 0);       // patched below
    std::vector<u8> stripByteCounts;
    for (u32 b = 0; b < cube.nbands; ++b) {
        PutU32(stripByteCounts, static_cast<u32>(planeBytes));
    }
    const usize stripOffsetsEntry = entries.size();
    if (cube.nbands == 1) {
        addInline(kTiffTagStripOffsets, kTiffTypeLong, 0);  // patched below
    } else {
        addArray(kTiffTagStripOffsets, kTiffTypeLong, cube.nbands, std::move(stripOffsets));
    }

    addInline(kTiffTagSamplesPerPixel, kTiffTypeShort, cube.nbands);
    addInline(kTiffTagRowsPerStrip, kTiffTypeLong, cube.height);

    if (cube.nbands == 1) {
        addInline(kTiffTagStripByteCounts, kTiffTypeLong, static_cast<u32>(planeBytes));
    } else {
        addArray(kTiffTagStripByteCounts, kTiffTypeLong, cube.nbands,
                 std::move(stripByteCounts));
    }

    addInline(kTiffTagPlanarConfig, kTiffTypeShort, kPlanarSeparate);

    std::vector<u8> sampleFormat;
    for (u32 b = 0; b < cube.nbands; ++b) PutU16(sampleFormat, kSampleFormatIeeeFloat);
    if (cube.nbands == 1) {
        addInline(kTiffTagSampleFormat, kTiffTypeShort, kSampleFormatIeeeFloat);
    } else {
        addArray(kTiffTagSampleFormat, kTiffTypeShort, cube.nbands, std::move(sampleFormat));
    }

    std::vector<u8> metadataBytes(metadata.begin(), metadata.end());
    metadataBytes.push_back(0);  // ASCII values are NUL-terminated
    // The count read into its own variable first. As arguments to one call the
    // size() and the move are unsequenced, and the compiler is free to move the
    // vector out before reading its length -- which it did, writing a
    // GDAL_METADATA tag of count zero that every reader then skipped, including
    // this file's own.
    const auto metadataCount = static_cast<u32>(metadataBytes.size());
    addArray(kTiffTagGdalMetadata, kTiffTypeAscii, metadataCount,
             std::move(metadataBytes));

    // Layout: header (8) | IFD (2 + 12n + 4) | out-of-line values | pixels.
    const u32 ifdOffset = 8;
    const u32 ifdBytes = 2 + 12 * static_cast<u32>(entries.size()) + 4;
    u32 cursor = ifdOffset + ifdBytes;
    for (TiffEntry& e : entries) {
        if (e.payload.empty()) continue;
        // Values are word aligned, which costs at most one byte and keeps a
        // reader that maps the file from doing unaligned loads.
        if (cursor & 1u) ++cursor;
        e.valueOrOffset = cursor;
        cursor += static_cast<u32>(e.payload.size());
    }
    if (cursor & 1u) ++cursor;
    const u32 pixelOffset = cursor;

    // Now the strip offsets are knowable.
    if (cube.nbands == 1) {
        entries[stripOffsetsEntry].valueOrOffset = pixelOffset;
    } else {
        std::vector<u8> offsets;
        for (u32 b = 0; b < cube.nbands; ++b) {
            PutU32(offsets, pixelOffset + static_cast<u32>(b * planeBytes));
        }
        entries[stripOffsetsEntry].payload = std::move(offsets);
    }

    std::vector<u8> header;
    header.push_back('I');
    header.push_back('I');
    PutU16(header, 42);
    PutU32(header, ifdOffset);

    std::vector<u8> ifd;
    PutU16(ifd, static_cast<u16>(entries.size()));
    for (const TiffEntry& e : entries) {
        PutU16(ifd, e.tag);
        PutU16(ifd, e.type);
        PutU32(ifd, e.count);
        PutU32(ifd, e.valueOrOffset);
    }
    PutU32(ifd, 0);  // no next IFD

    file.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(ifd.data()),
               static_cast<std::streamsize>(ifd.size()));

    u32 written = ifdOffset + ifdBytes;
    const auto pad = [&file, &written](const u32 to) {
        while (written < to) {
            file.put('\0');
            ++written;
        }
    };
    for (const TiffEntry& e : entries) {
        if (e.payload.empty()) continue;
        pad(e.valueOrOffset);
        file.write(reinterpret_cast<const char*>(e.payload.data()),
                   static_cast<std::streamsize>(e.payload.size()));
        written += static_cast<u32>(e.payload.size());
    }
    pad(pixelOffset);

    // The cube is already band-sequential, so a plane is a strip is a write.
    file.write(reinterpret_cast<const char*>(cube.data.data()),
               static_cast<std::streamsize>(pixelBytes));
    file.close();

    LOG_INFO("Wrote TIFF cube: {} ({} x {} x {} bands, {:.1f}-{:.1f} nm, "
             "float32, band descriptions in GDAL_METADATA)",
             path, cube.width, cube.height, cube.nbands,
             cube.lambda_min, cube.lambda_max);
    return true;
}

Result<SpectralCube, String> SpectralCubeIO::ReadGeoTIFF(const String& path) {
    using CubeResult = Result<SpectralCube, String>;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return CubeResult::Err("File not found: " + path);
    }
    const auto size = static_cast<usize>(file.tellg());
    file.seekg(0);
    std::vector<u8> bytes(size);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    file.close();

    if (bytes.size() < 8) {
        return CubeResult::Err("Not a TIFF: " + path + " is " +
                               std::to_string(bytes.size()) + " bytes");
    }
    const bool little = bytes[0] == 'I' && bytes[1] == 'I';
    const bool big = bytes[0] == 'M' && bytes[1] == 'M';
    if (!little && !big) {
        return CubeResult::Err("Not a TIFF: no byte-order mark in " + path);
    }
    const u16 magic = GetU16(bytes, 2, little);
    if (magic == 43) {
        return CubeResult::Err(
            path + " is BigTIFF, which this reader does not do. It is a different "
            "format rather than a larger offset, and nothing here writes it.");
    }
    if (magic != 42) {
        return CubeResult::Err("Not a TIFF: bad magic in " + path);
    }

    const u32 ifdOffset = GetU32(bytes, 4, little);
    if (ifdOffset + 2 > bytes.size()) {
        return CubeResult::Err("Truncated TIFF: the directory is past the end of " + path);
    }
    const u16 entryCount = GetU16(bytes, ifdOffset, little);

    // What a spectral cube needs out of the directory. Everything else a TIFF
    // may carry is ignored on purpose: this reads the files this writer makes
    // and any other baseline float TIFF, not the format's whole surface.
    u32 width = 0, height = 0, nbands = 1, rowsPerStrip = 0;
    u16 compression = 1, planar = kPlanarChunky, sampleFormat = 0, bitsPerSample = 0;
    std::vector<u32> stripOffsets, stripByteCounts;
    String metadata;

    const auto readArray = [&](const u32 tagType, const u32 count, const u32 valueOrOffset,
                               std::vector<u32>& out) {
        const u32 elementSize = tagType == kTiffTypeShort ? 2u : 4u;
        out.clear();
        out.reserve(count);
        if (count * elementSize <= 4) {
            // Inline: the value sits in the entry itself.
            for (u32 i = 0; i < count; ++i) {
                out.push_back(tagType == kTiffTypeShort
                                  ? ((valueOrOffset >> (16 * i)) & 0xFFFF)
                                  : valueOrOffset);
            }
            return;
        }
        for (u32 i = 0; i < count; ++i) {
            const usize at = valueOrOffset + static_cast<usize>(i) * elementSize;
            out.push_back(tagType == kTiffTypeShort
                              ? static_cast<u32>(GetU16(bytes, at, little))
                              : GetU32(bytes, at, little));
        }
    };

    for (u16 i = 0; i < entryCount; ++i) {
        const usize at = ifdOffset + 2 + static_cast<usize>(i) * 12;
        if (at + 12 > bytes.size()) break;
        const u16 tag = GetU16(bytes, at, little);
        const u16 type = GetU16(bytes, at + 2, little);
        const u32 count = GetU32(bytes, at + 4, little);
        const u32 value = GetU32(bytes, at + 8, little);
        // A SHORT that fits inline sits in the first two bytes of the field.
        const u16 shortValue = (type == kTiffTypeShort && count == 1)
                                   ? GetU16(bytes, at + 8, little)
                                   : 0;

        switch (tag) {
            case kTiffTagImageWidth:
                width = type == kTiffTypeShort ? shortValue : value;
                break;
            case kTiffTagImageLength:
                height = type == kTiffTypeShort ? shortValue : value;
                break;
            case kTiffTagBitsPerSample: {
                std::vector<u32> bits;
                readArray(type, count, value, bits);
                bitsPerSample = bits.empty() ? 0 : static_cast<u16>(bits.front());
                break;
            }
            case kTiffTagCompression:
                compression = shortValue;
                break;
            case kTiffTagSamplesPerPixel:
                nbands = type == kTiffTypeShort ? shortValue : value;
                break;
            case kTiffTagRowsPerStrip:
                rowsPerStrip = type == kTiffTypeShort ? shortValue : value;
                break;
            case kTiffTagPlanarConfig:
                planar = shortValue;
                break;
            case kTiffTagSampleFormat: {
                std::vector<u32> formats;
                readArray(type, count, value, formats);
                sampleFormat = formats.empty() ? 0 : static_cast<u16>(formats.front());
                break;
            }
            case kTiffTagStripOffsets:
                readArray(type, count, value, stripOffsets);
                break;
            case kTiffTagStripByteCounts:
                readArray(type, count, value, stripByteCounts);
                break;
            case kTiffTagGdalMetadata: {
                if (count <= 4) break;  // no room for a document
                const usize end = std::min<usize>(value + count, bytes.size());
                metadata.assign(bytes.begin() + static_cast<isize>(value),
                                bytes.begin() + static_cast<isize>(end));
                if (!metadata.empty() && metadata.back() == '\0') metadata.pop_back();
                break;
            }
            default:
                break;
        }
    }

    if (width == 0 || height == 0 || nbands == 0) {
        return CubeResult::Err("TIFF has no dimensions: " + path);
    }
    if (compression != 1) {
        return CubeResult::Err(
            "TIFF " + path + " is compressed (scheme " + std::to_string(compression) +
            "). This reader takes uncompressed strips only, which is what the "
            "writer here produces.");
    }
    if (bitsPerSample != 32 || sampleFormat != kSampleFormatIeeeFloat) {
        return CubeResult::Err(
            "TIFF " + path + " is not 32-bit float (" + std::to_string(bitsPerSample) +
            " bits, sample format " + std::to_string(sampleFormat) +
            "). A spectral cube is radiance, and an integer image is not one.");
    }
    if (stripOffsets.empty()) {
        return CubeResult::Err("TIFF " + path + " names no strips");
    }
    if (rowsPerStrip == 0) rowsPerStrip = height;

    // Wavelengths from the GDAL metadata, or a uniform grid when the file
    // carries none -- which is what a TIFF from another tool looks like, and is
    // a cube whose band centres nobody wrote down rather than an error.
    SpectralCube cube;
    cube.width = width;
    cube.height = height;
    cube.nbands = nbands;
    cube.wavelengths.resize(nbands);
    bool haveWavelengths = !metadata.empty();
    for (u32 b = 0; b < nbands && haveWavelengths; ++b) {
        String item;
        if (!GdalMetadataItem(metadata, "WAVELENGTH", static_cast<int>(b), item)) {
            haveWavelengths = false;
            break;
        }
        try {
            cube.wavelengths[b] = std::stof(item);
        } catch (const std::exception&) {
            haveWavelengths = false;
        }
    }
    if (!haveWavelengths) {
        for (u32 b = 0; b < nbands; ++b) cube.wavelengths[b] = static_cast<f32>(b);
        LOG_WARN("TIFF {} carries no band wavelengths; bands are numbered instead", path);
    }
    cube.lambda_min = cube.wavelengths.front();
    cube.lambda_max = cube.wavelengths.back();
    cube.delta_lambda = nbands > 1
        ? (cube.lambda_max - cube.lambda_min) / static_cast<f32>(nbands - 1)
        : 0.0f;

    cube.data.assign(static_cast<usize>(width) * height * nbands, 0.0f);

    const usize rowFloats = width;
    const usize planeFloats = static_cast<usize>(width) * height;
    if (planar == kPlanarSeparate) {
        // One plane per band, which is what this writer makes: strip s belongs
        // to band s / stripsPerPlane.
        const u32 stripsPerPlane =
            std::max(1u, (height + rowsPerStrip - 1) / rowsPerStrip);
        for (usize s = 0; s < stripOffsets.size(); ++s) {
            const u32 band = static_cast<u32>(s / stripsPerPlane);
            const u32 stripInPlane = static_cast<u32>(s % stripsPerPlane);
            if (band >= nbands) break;
            const u32 firstRow = stripInPlane * rowsPerStrip;
            const u32 rows = std::min(rowsPerStrip, height - firstRow);
            const usize src = stripOffsets[s];
            const usize wanted = static_cast<usize>(rows) * rowFloats * sizeof(f32);
            if (src + wanted > bytes.size()) {
                return CubeResult::Err("Truncated TIFF: strip " + std::to_string(s) +
                                       " runs past the end of " + path);
            }
            std::memcpy(cube.data.data() + band * planeFloats + firstRow * rowFloats,
                        bytes.data() + src, wanted);
        }
    } else {
        // Chunky: samples interleaved per pixel, which is what most other
        // tools write. Deinterleaved into the cube's band-sequential order.
        for (usize s = 0; s < stripOffsets.size(); ++s) {
            const u32 firstRow = static_cast<u32>(s) * rowsPerStrip;
            if (firstRow >= height) break;
            const u32 rows = std::min(rowsPerStrip, height - firstRow);
            const usize src = stripOffsets[s];
            const usize wanted =
                static_cast<usize>(rows) * rowFloats * nbands * sizeof(f32);
            if (src + wanted > bytes.size()) {
                return CubeResult::Err("Truncated TIFF: strip " + std::to_string(s) +
                                       " runs past the end of " + path);
            }
            const f32* in = reinterpret_cast<const f32*>(bytes.data() + src);
            for (u32 y = 0; y < rows; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    for (u32 b = 0; b < nbands; ++b) {
                        cube.data[b * planeFloats + (firstRow + y) * rowFloats + x] =
                            in[(static_cast<usize>(y) * width + x) * nbands + b];
                    }
                }
            }
        }
    }

    // Whatever else the document carried, back onto the cube.
    if (!metadata.empty()) {
        String units;
        if (GdalMetadataItem(metadata, "WAVELENGTH_UNITS", -1, units)) {
            cube.metadata["wavelength_units"] = units;
        }
        if (GdalMetadataItem(metadata, "EMISSIVE_UNITS", -1, units)) {
            cube.metadata["emissive_units"] = units;
        }
    }

    LOG_INFO("Read TIFF cube: {} ({} x {} x {} bands, {:.1f}-{:.1f} nm)",
             path, width, height, nbands, cube.lambda_min, cube.lambda_max);
    return CubeResult(std::move(cube));
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
