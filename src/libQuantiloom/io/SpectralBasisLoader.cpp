#include "SpectralBasisLoader.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>

// JSON parsing (use nlohmann/json if available, otherwise simple manual parsing)
// For simplicity, we use a simple JSON parser here
#include <ranges>
#include <sstream>

namespace quantiloom {

// ============================================================================
// Binary Constants
// ============================================================================
static constexpr char MAGIC[4] = {'Q', 'B', 'A', 'S'};
static constexpr u32 SUPPORTED_VERSION = 3;
static constexpr size_t HEADER_SIZE = 64;
static constexpr size_t BAND_HEADER_SIZE = 16;

// ============================================================================
// Helper: Read little-endian values from byte buffer
// ============================================================================
template<typename T>
static T ReadLE(const std::vector<u8>& data, size_t& offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

// ============================================================================
// LoadBasis - Parse binary basis file
// ============================================================================
bool SpectralBasisLoader::LoadBasis(const std::filesystem::path& basisFilePath) {
    QL_LOG_INFO("Loading spectral basis from: {}", basisFilePath.string());

    // Read entire file
    std::ifstream file(basisFilePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        QL_LOG_ERROR("  Failed to open basis file: {}", basisFilePath.string());
        return false;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<u8> data(fileSize);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<long long>(fileSize))) {
        QL_LOG_ERROR("  Failed to read basis file");
        return false;
    }
    file.close();

    // Parse header
    size_t offset = 0;

    // Check magic
    if (fileSize < HEADER_SIZE) {
        QL_LOG_ERROR("  File too small for header");
        return false;
    }

    if (std::memcmp(data.data(), MAGIC, 4) != 0) {
        QL_LOG_ERROR("  Invalid magic number (expected 'QBAS')");
        return false;
    }
    offset += 4;

    // Version
    m_basisVersion = ReadLE<u32>(data, offset);
    if (m_basisVersion != SUPPORTED_VERSION) {
        QL_LOG_ERROR("  Unsupported version {} (expected {})", m_basisVersion, SUPPORTED_VERSION);
        return false;
    }

    // Number of bands
    u32 numBands = ReadLE<u32>(data, offset);
    QL_LOG_INFO("  Version: {}, Bands: {}", m_basisVersion, numBands);

    // Skip reserved bytes
    offset = HEADER_SIZE;

    // Parse each band
    m_basisFunctions.clear();

    static const char* bandNames[] = {"VIS", "NIR", "SWIR", "MWIR", "LWIR"};

    for (u32 bandIdx = 0; bandIdx < numBands; ++bandIdx) {
        if (offset + BAND_HEADER_SIZE > fileSize) {
            QL_LOG_ERROR("  Unexpected end of file at band {}", bandIdx);
            return false;
        }

        BasisFunctions basis;
        basis.name = (bandIdx < 5) ? bandNames[bandIdx] : ("Band" + std::to_string(bandIdx));

        // Band header (16 bytes)
        basis.wavelengthStart_um = ReadLE<f32>(data, offset);
        basis.wavelengthEnd_um = ReadLE<f32>(data, offset);
        basis.numSamples = ReadLE<u32>(data, offset);
        basis.numBasis = ReadLE<u32>(data, offset);

        // Read basis data
        size_t dataSize = basis.numBasis * basis.numSamples * sizeof(f32);
        if (offset + dataSize > fileSize) {
            QL_LOG_ERROR("  Unexpected end of file reading band {} data", bandIdx);
            return false;
        }

        basis.data.resize(basis.numBasis * basis.numSamples);
        std::memcpy(basis.data.data(), data.data() + offset, dataSize);
        offset += dataSize;

        QL_LOG_INFO("  {} band: {:.3f}-{:.3f} um, {} samples, {} basis functions",
                    basis.name, basis.wavelengthStart_um, basis.wavelengthEnd_um,
                    basis.numSamples, basis.numBasis);

        m_basisFunctions[basis.name] = std::move(basis);
    }

    QL_LOG_INFO("  Loaded {} bands total", m_basisFunctions.size());
    return true;
}

// ============================================================================
// Simple JSON Parser Helpers
// ============================================================================
// Note: This is a minimal JSON parser sufficient for our needs.
// For production, consider using nlohmann/json or similar.

static void SkipWhitespace(const String& json, size_t& pos) {
    while (pos < json.size() && std::isspace(json[pos])) ++pos;
}

static String ParseString(const String& json, size_t& pos) {
    if (json[pos] != '"') return "";
    ++pos;

    String result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            switch (json[pos]) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    if (pos < json.size()) ++pos;  // Skip closing quote
    return result;
}

static f32 ParseNumber(const String& json, size_t& pos) {
    size_t start = pos;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' ||
           json[pos] == '-' || json[pos] == '+' || json[pos] == 'e' || json[pos] == 'E')) {
        ++pos;
    }
    return std::stof(json.substr(start, pos - start));
}

static std::vector<f32> ParseNumberArray(const String& json, size_t& pos) {
    std::vector<f32> result;

    SkipWhitespace(json, pos);
    if (json[pos] != '[') return result;
    ++pos;

    while (pos < json.size()) {
        SkipWhitespace(json, pos);
        if (json[pos] == ']') { ++pos; break; }
        if (json[pos] == ',') { ++pos; continue; }

        result.push_back(ParseNumber(json, pos));
    }

    return result;
}

// Forward to next key or end of object
static void SkipValue(const String& json, size_t& pos) {
    SkipWhitespace(json, pos);
    if (pos >= json.size()) return;

    if (json[pos] == '"') {
        ParseString(json, pos);
    } else if (json[pos] == '[') {
        int depth = 1;
        ++pos;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == '[') ++depth;
            else if (json[pos] == ']') --depth;
            ++pos;
        }
    } else if (json[pos] == '{') {
        int depth = 1;
        ++pos;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == '{') ++depth;
            else if (json[pos] == '}') --depth;
            ++pos;
        }
    } else {
        // Number or literal
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}' && json[pos] != ']') {
            ++pos;
        }
    }
}

// ============================================================================
// LoadMaterials - Parse JSON materials file
// ============================================================================
bool SpectralBasisLoader::LoadMaterials(const std::filesystem::path& jsonFilePath) {
    QL_LOG_INFO("Loading spectral materials from: {}", jsonFilePath.string());

    // Read entire file
    std::ifstream file(jsonFilePath);
    if (!file.is_open()) {
        QL_LOG_ERROR("  Failed to open materials file: {}", jsonFilePath.string());
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    String json = buffer.str();
    file.close();

    size_t pos = 0;
    m_materials.clear();

    // Find metadata section
    size_t metadataPos = json.find("\"metadata\"");
    if (metadataPos != String::npos) {
        pos = json.find('{', metadataPos);
        if (pos != String::npos) {
            ++pos;
            while (pos < json.size()) {
                SkipWhitespace(json, pos);
                if (json[pos] == '}') { ++pos; break; }
                if (json[pos] == ',') { ++pos; continue; }

                String key = ParseString(json, pos);
                SkipWhitespace(json, pos);
                if (json[pos] == ':') ++pos;
                SkipWhitespace(json, pos);

                if (key == "generator") m_generator = ParseString(json, pos);
                else if (key == "source_library") m_sourceLibrary = ParseString(json, pos);
                else if (key == "date_generated") m_dateGenerated = ParseString(json, pos);
                else SkipValue(json, pos);
            }
        }
    }

    QL_LOG_INFO("  Generator: {}", m_generator);
    QL_LOG_INFO("  Source: {}", m_sourceLibrary);

    // Find materials section
    size_t materialsPos = json.find("\"materials\"");
    if (materialsPos == String::npos) {
        QL_LOG_ERROR("  No 'materials' section found in JSON");
        return false;
    }

    pos = json.find('{', materialsPos + 11);
    if (pos == String::npos) {
        QL_LOG_ERROR("  Malformed materials section");
        return false;
    }
    ++pos;

    // Parse each material
    while (pos < json.size()) {
        SkipWhitespace(json, pos);
        if (json[pos] == '}') break;
        if (json[pos] == ',') { ++pos; continue; }

        // Material name (key)
        String materialName = ParseString(json, pos);
        if (materialName.empty()) break;

        SkipWhitespace(json, pos);
        if (json[pos] == ':') ++pos;
        SkipWhitespace(json, pos);

        if (json[pos] != '{') { SkipValue(json, pos); continue; }
        ++pos;

        MaterialSpectralData material;
        material.name = materialName;

        // Parse material object
        while (pos < json.size()) {
            SkipWhitespace(json, pos);
            if (json[pos] == '}') { ++pos; break; }
            if (json[pos] == ',') { ++pos; continue; }

            String key = ParseString(json, pos);
            SkipWhitespace(json, pos);
            if (json[pos] == ':') ++pos;
            SkipWhitespace(json, pos);

            if (key == "source") {
                // Parse source object
                if (json[pos] == '{') {
                    ++pos;
                    while (pos < json.size()) {
                        SkipWhitespace(json, pos);
                        if (json[pos] == '}') { ++pos; break; }
                        if (json[pos] == ',') { ++pos; continue; }

                        String srcKey = ParseString(json, pos);
                        SkipWhitespace(json, pos);
                        if (json[pos] == ':') ++pos;
                        SkipWhitespace(json, pos);

                        if (srcKey == "filename") material.filename = ParseString(json, pos);
                        else if (srcKey == "record_id") material.recordId = ParseString(json, pos);
                        else if (srcKey == "instrument") material.instrument = ParseString(json, pos);
                        else if (srcKey == "chapter") material.chapter = ParseString(json, pos);
                        else SkipValue(json, pos);
                    }
                }
            } else if (key == "bands") {
                // Parse bands object
                if (json[pos] == '{') {
                    ++pos;
                    while (pos < json.size()) {
                        SkipWhitespace(json, pos);
                        if (json[pos] == '}') { ++pos; break; }
                        if (json[pos] == ',') { ++pos; continue; }

                        String bandName = ParseString(json, pos);
                        SkipWhitespace(json, pos);
                        if (json[pos] == ':') ++pos;
                        SkipWhitespace(json, pos);

                        MaterialSpectralData::BandData bandData;

                        if (json[pos] == '{') {
                            ++pos;
                            while (pos < json.size()) {
                                SkipWhitespace(json, pos);
                                if (json[pos] == '}') { ++pos; break; }
                                if (json[pos] == ',') { ++pos; continue; }

                                String bandKey = ParseString(json, pos);
                                SkipWhitespace(json, pos);
                                if (json[pos] == ':') ++pos;
                                SkipWhitespace(json, pos);

                                if (bandKey == "basis_weights") {
                                    bandData.weights = ParseNumberArray(json, pos);
                                } else if (bandKey == "rmse") {
                                    bandData.rmse = ParseNumber(json, pos);
                                } else if (bandKey == "explained_variance") {
                                    bandData.explainedVariance = ParseNumber(json, pos);
                                } else {
                                    SkipValue(json, pos);
                                }
                            }
                        }

                        if (!bandData.weights.empty()) {
                            material.bands[bandName] = std::move(bandData);
                        }
                    }
                }
            } else {
                SkipValue(json, pos);
            }
        }

        if (!material.bands.empty()) {
            m_materials[materialName] = std::move(material);
        }
    }

    QL_LOG_INFO("  Loaded {} materials", m_materials.size());
    return true;
}

// ============================================================================
// Query Methods
// ============================================================================

std::vector<String> SpectralBasisLoader::GetMaterialNames() const {
    std::vector<String> names;
    names.reserve(m_materials.size());
    for (const auto &name: m_materials | std::views::keys) {
        names.push_back(name);
    }
    return names;
}

const MaterialSpectralData* SpectralBasisLoader::FindMaterial(const String& name) const {
    auto it = m_materials.find(name);
    return (it != m_materials.end()) ? &it->second : nullptr;
}

const MaterialSpectralData* SpectralBasisLoader::FindMaterialPartial(const String& pattern) const {
    // Convert pattern to lowercase for case-insensitive search
    String lowerPattern = pattern;
    std::ranges::transform(lowerPattern, lowerPattern.begin(),
                           [](const unsigned char c) { return std::tolower(c); });

    for (const auto& [name, material] : m_materials) {
        String lowerName = name;
        std::ranges::transform(lowerName, lowerName.begin(),
                               [](const unsigned char c) { return std::tolower(c); });

        if (lowerName.find(lowerPattern) != String::npos) {
            return &material;
        }
    }
    return nullptr;
}

const BasisFunctions* SpectralBasisLoader::GetBasis(const String& bandName) const {
    auto it = m_basisFunctions.find(bandName);
    return (it != m_basisFunctions.end()) ? &it->second : nullptr;
}

// ============================================================================
// Spectral Reconstruction
// ============================================================================

SpectralCurve SpectralBasisLoader::ReconstructCurve(const String& materialName,
                                                     const String& bandName) const {
    SpectralCurve curve;

    // Find material
    const MaterialSpectralData* material = FindMaterial(materialName);
    if (!material) {
        QL_LOG_WARN("Material not found: {}", materialName);
        return curve;
    }

    // Find band weights
    auto bandIt = material->bands.find(bandName);
    if (bandIt == material->bands.end()) {
        QL_LOG_WARN("Band {} not found for material {}", bandName, materialName);
        return curve;
    }
    const auto& bandData = bandIt->second;

    // Find basis functions
    const BasisFunctions* basis = GetBasis(bandName);
    if (!basis || !basis->IsValid()) {
        QL_LOG_WARN("Basis functions not found for band {}", bandName);
        return curve;
    }

    // Validate weight count matches basis count
    if (bandData.weights.size() != basis->numBasis) {
        QL_LOG_WARN("Weight count mismatch: {} weights vs {} basis functions",
                    bandData.weights.size(), basis->numBasis);
        return curve;
    }

    // Reconstruct: spectrum[i] = sum(weights[j] * basis[j][i])
    curve.samples.reserve(basis->numSamples);

    for (u32 i = 0; i < basis->numSamples; ++i) {
        f32 value = 0.0f;
        for (u32 j = 0; j < basis->numBasis; ++j) {
            value += bandData.weights[j] * basis->Get(j, i);
        }
        // Clamp to [0, 1] (reflectance)
        value = std::clamp(value, 0.0f, 1.0f);

        f32 wavelength_nm = basis->GetWavelength_nm(i);
        curve.samples.emplace_back(wavelength_nm, value);
    }

    return curve;
}

SpectralCurveGPU SpectralBasisLoader::ReconstructCurveGPU(const String& materialName,
                                                          const String& bandName) const {
    SpectralCurve curve = ReconstructCurve(materialName, bandName);
    if (curve.samples.empty()) {
        return SpectralCurveGPU{};
    }
    return SpectralCurveGPU::FromCPU(curve);
}

SpectralCurve SpectralBasisLoader::ReconstructFullSpectrum(const String& materialName) const {
    SpectralCurve fullCurve;

    // Reconstruct each band
    SpectralCurve visCurve = ReconstructCurve(materialName, "VIS");
    SpectralCurve nirCurve = ReconstructCurve(materialName, "NIR");
    SpectralCurve swirCurve = ReconstructCurve(materialName, "SWIR");
    SpectralCurve mwirCurve = ReconstructCurve(materialName, "MWIR");
    SpectralCurve lwirCurve = ReconstructCurve(materialName, "LWIR");

    // Combine in order: VIS -> NIR -> SWIR -> MWIR -> LWIR
    // Note: There may be overlap between bands, so we take the value from the
    // band that is "primary" for that wavelength range

    // VIS: 350-780nm
    for (const auto& [wl, val] : visCurve.samples) {
        if (wl < 780.0f) {  // Exclude overlap region
            fullCurve.samples.emplace_back(wl, val);
        }
    }

    // NIR: 780-1100nm
    for (const auto& [wl, val] : nirCurve.samples) {
        if (wl >= 780.0f && wl < 1100.0f) {
            fullCurve.samples.emplace_back(wl, val);
        }
    }

    // SWIR: 1100-2500nm
    for (const auto& [wl, val] : swirCurve.samples) {
        if (wl >= 1100.0f && wl < 2500.0f) {
            fullCurve.samples.emplace_back(wl, val);
        }
    }

    // MWIR: 2500-6500nm
    for (const auto& [wl, val] : mwirCurve.samples) {
        if (wl >= 2500.0f && wl < 6500.0f) {
            fullCurve.samples.emplace_back(wl, val);
        }
    }

    // LWIR: 6500-15000nm
    for (const auto& [wl, val] : lwirCurve.samples) {
        if (wl >= 6500.0f) {
            fullCurve.samples.emplace_back(wl, val);
        }
    }

    return fullCurve;
}

} // namespace quantiloom
