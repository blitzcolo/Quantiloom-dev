#include "io/SpectralIO.hpp"

#include "core/Blackbody.hpp"
#include "core/D65Illuminant.hpp"
#include "core/FluorescentIlluminants.hpp"
#include "io/SpectralBasisLoader.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// Public API: LoadSpectralCurveCSV
// ============================================================================

Result<std::vector<std::pair<f32, f32>>, String>
SpectralIO::LoadSpectralCurveCSV(const std::filesystem::path& csvPath) {
    // Check if file exists
    if (!std::filesystem::exists(csvPath)) {
        return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
            "File not found: " + csvPath.string()
        });
    }

    // Open CSV file
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
            "Failed to open file: " + csvPath.string()
        });
    }

    std::vector<std::pair<f32, f32>> curve;
    std::string line;
    u32 lineNumber = 0;
    f32 lastWavelength = -std::numeric_limits<f32>::infinity();

    while (std::getline(file, line)) {
        ++lineNumber;

        // Trim leading whitespace
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;  // Empty line
        }

        // Skip comments
        if (line[start] == '#') {
            continue;
        }

        // Parse two floats: wavelength_nm, value
        f32 wavelength = 0.0f;
        f32 value = 0.0f;

        // Use sscanf for robust parsing
        int parsed = std::sscanf(line.c_str(), "%f , %f", &wavelength, &value);
        if (parsed != 2) {
            // Try without spaces around comma
            parsed = std::sscanf(line.c_str(), "%f, %f", &wavelength, &value);
        }
        if (parsed != 2) {
            // Try with only space after comma
            parsed = std::sscanf(line.c_str(), "%f ,%f", &wavelength, &value);
        }

        if (parsed != 2) {
            return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
                "Parse error at line " + std::to_string(lineNumber) + ": '" + line + "'"
            });
        }

        // Validate monotonicity
        if (wavelength <= lastWavelength) {
            return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
                "Wavelengths not monotonically increasing at line " + std::to_string(lineNumber) +
                ": " + std::to_string(wavelength) + " <= " + std::to_string(lastWavelength)
            });
        }

        // Validate wavelength is positive
        if (wavelength <= 0.0f) {
            return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
                "Invalid wavelength at line " + std::to_string(lineNumber) + ": " + std::to_string(wavelength)
            });
        }

        // Validate value is in [0, 1] for material properties (emissivity, reflectance, transmittance)
        if (value < 0.0f || value > 1.0f) {
            QL_LOG_WARN("SpectralIO::LoadSpectralCurveCSV: Value {} out of [0, 1] range at line {} (clamping)",
                        value, lineNumber);
            value = std::clamp(value, 0.0f, 1.0f);
        }

        curve.emplace_back(wavelength, value);
        lastWavelength = wavelength;
    }

    // Validate that we loaded at least 2 points for interpolation
    if (curve.size() < 2) {
        return Result<std::vector<std::pair<f32, f32> > >(Result<std::vector<std::pair<f32, f32> > >::Err{
            "Spectral curve must have at least 2 data points, got " + std::to_string(curve.size())
        });
    }

    QL_LOG_INFO("SpectralIO::LoadSpectralCurveCSV: Loaded {} points from {} (λ: {:.1f}-{:.1f} nm)",
                curve.size(), csvPath.filename().string(), curve.front().first, curve.back().first);

    return Result(std::move(curve));
}

// ============================================================================
// USGS Spectral Library Loading
// ============================================================================

// USGS invalid data marker (used to indicate bad measurements)
static constexpr f32 USGS_INVALID_VALUE = -1.23e+034f;
static constexpr f32 USGS_INVALID_THRESHOLD = -1.0e+030f;

Result<SpectralCurve, String>
SpectralIO::LoadUSGS(const std::filesystem::path& reflectanceFile,
                     const std::filesystem::path& wavelengthFile) {
    // Check files exist
    if (!std::filesystem::exists(reflectanceFile)) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "USGS reflectance file not found: " + reflectanceFile.string()
        });
    }
    if (!std::filesystem::exists(wavelengthFile)) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "USGS wavelength file not found: " + wavelengthFile.string()
        });
    }

    // Read wavelength file (micrometers -> nanometers)
    std::vector<f32> wavelengths_nm;
    {
        std::ifstream waveFile(wavelengthFile);
        if (!waveFile.is_open()) {
            return Result<SpectralCurve>(Result<SpectralCurve>::Err{
                "Failed to open wavelength file: " + wavelengthFile.string()
            });
        }

        std::string line;
        u32 lineNum = 0;
        while (std::getline(waveFile, line)) {
            ++lineNum;

            // Skip first line (header)
            if (lineNum == 1) continue;

            // Trim whitespace
            const size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;

            // Parse wavelength (micrometers)
            f32 wavelength_um = 0.0f;
            if (std::sscanf(line.c_str(), "%f", &wavelength_um) != 1) {
                QL_LOG_WARN("USGS: Failed to parse wavelength at line {}", lineNum);
                continue;
            }

            // Convert µm to nm
            wavelengths_nm.push_back(wavelength_um * 1000.0f);
        }
    }

    if (wavelengths_nm.empty()) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "No wavelengths parsed from: " + wavelengthFile.string()
        });
    }

    // Read reflectance file
    std::vector<f32> reflectances;
    std::string materialName;
    {
        std::ifstream refFile(reflectanceFile);
        if (!refFile.is_open()) {
            return Result<SpectralCurve>(Result<SpectralCurve>::Err{
                "Failed to open reflectance file: " + reflectanceFile.string()
            });
        }

        std::string line;
        u32 lineNum = 0;
        while (std::getline(refFile, line)) {
            ++lineNum;

            // First line is metadata (extract material name)
            if (lineNum == 1) {
                // Format: " splib07a Record=13456: Hematite_Coatd_Qtz BR93-25B   BECKa AREF"
                const size_t colonPos = line.find(':');
                if (colonPos != std::string::npos && colonPos + 2 < line.size()) {
                    materialName = line.substr(colonPos + 2);
                    // Trim trailing whitespace
                    const size_t end = materialName.find_last_not_of(" \t\r\n");
                    if (end != std::string::npos) {
                        materialName = materialName.substr(0, end + 1);
                    }
                }
                continue;
            }

            // Trim whitespace
            const size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;

            // Parse reflectance value
            f32 reflectance = 0.0f;
            if (std::sscanf(line.c_str(), "%f", &reflectance) != 1) {
                QL_LOG_WARN("USGS: Failed to parse reflectance at line {}", lineNum);
                reflectances.push_back(USGS_INVALID_VALUE);  // Mark as invalid
                continue;
            }

            reflectances.push_back(reflectance);
        }
    }

    // Validate counts match
    if (wavelengths_nm.size() != reflectances.size()) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "USGS data mismatch: " + std::to_string(wavelengths_nm.size()) +
            " wavelengths but " + std::to_string(reflectances.size()) + " reflectances"
        });
    }

    // Build SpectralCurve, skipping invalid data points
    SpectralCurve curve;
    curve.samples.reserve(wavelengths_nm.size());

    u32 validCount = 0;
    u32 invalidCount = 0;

    for (size_t i = 0; i < wavelengths_nm.size(); ++i) {
        const f32 wavelength = wavelengths_nm[i];
        const f32 reflectance = reflectances[i];

        // Skip invalid data points (USGS uses -1.23e+034 for bad data)
        if (reflectance < USGS_INVALID_THRESHOLD) {
            ++invalidCount;
            continue;
        }

        // Clamp reflectance to valid [0, 1] range
        const f32 clampedRef = std::clamp(reflectance, 0.0f, 1.0f);

        curve.samples.emplace_back(wavelength, clampedRef);
        ++validCount;
    }

    if (curve.samples.size() < 2) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "USGS: Not enough valid data points (" + std::to_string(validCount) +
            " valid, " + std::to_string(invalidCount) + " invalid)"
        });
    }

    QL_LOG_INFO("SpectralIO::LoadUSGS: Loaded '{}' - {} valid points, {} invalid (λ: {:.1f}-{:.1f} nm)",
                materialName,
                validCount, invalidCount,
                curve.samples.front().first, curve.samples.back().first);

    return Result(std::move(curve));
}

Result<SpectralCurve, String>
SpectralIO::LoadUSGSAuto(const std::filesystem::path& reflectanceFile) {
    // Extract spectrometer code from filename
    // Format: splib07a_{Material}_{SampleID}_{Spectrometer}_AREF.txt
    // E.g., splib07a_Hematite_Coatd_Qtz_BR93-25B_BECKa_AREF.txt -> BECKa
    const std::string filename = reflectanceFile.stem().string();

    // Find spectrometer code (before _AREF suffix)
    const size_t arefPos = filename.rfind("_AREF");
    if (arefPos == std::string::npos) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "USGS filename doesn't match expected pattern (*_AREF.txt): " + filename
        });
    }

    // Find spectrometer code (last underscore before _AREF)
    const size_t spectroStart = filename.rfind('_', arefPos - 1);
    if (spectroStart == std::string::npos) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "Cannot extract spectrometer code from filename: " + filename
        });
    }

    const std::string spectroCode = filename.substr(spectroStart + 1, arefPos - spectroStart - 1);

    // Map spectrometer code to wavelength file pattern
    std::string wavelengthPattern;
    if (spectroCode.find("BECK") != std::string::npos) {
        wavelengthPattern = "splib07a_Wavelengths_BECK_Beckman";
    } else if (spectroCode.find("ASD") != std::string::npos) {
        wavelengthPattern = "splib07a_Wavelengths_ASD";
    } else if (spectroCode.find("NIC4") != std::string::npos) {
        wavelengthPattern = "splib07a_Wavelengths_NIC4_Nicolet";
    } else if (spectroCode.find("AVIRIS") != std::string::npos) {
        wavelengthPattern = "splib07a_Wavelengths_AVIRIS";
    } else {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "Unknown USGS spectrometer code: " + spectroCode
        });
    }

    // Search for wavelength file in parent directory
    const std::filesystem::path parentDir = reflectanceFile.parent_path().parent_path();

    std::filesystem::path wavelengthFile;
    for (const auto& entry : std::filesystem::directory_iterator(parentDir)) {
        if (entry.is_regular_file()) {
            const std::string entryFilename = entry.path().filename().string();
            if (entryFilename.find(wavelengthPattern) != std::string::npos &&
                entryFilename.find(".txt") != std::string::npos) {
                wavelengthFile = entry.path();
                break;
            }
        }
    }

    if (wavelengthFile.empty()) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "Cannot find wavelength file matching pattern '" + wavelengthPattern +
            "' in " + parentDir.string()
        });
    }

    QL_LOG_DEBUG("SpectralIO::LoadUSGSAuto: Auto-detected wavelength file: {}",
                 wavelengthFile.filename().string());

    return LoadUSGS(reflectanceFile, wavelengthFile);
}

// ============================================================================
// RefractiveIndex.INFO Loading (YAML format)
// ============================================================================

Result<ComplexRefractiveIndex, String>
SpectralIO::LoadRefractiveIndexYAML(const std::filesystem::path& yamlFile) {
    // Check file exists
    if (!std::filesystem::exists(yamlFile)) {
        return Result<ComplexRefractiveIndex>(Result<ComplexRefractiveIndex>::Err{
            "YAML file not found: " + yamlFile.string()
        });
    }

    // Open file
    std::ifstream file(yamlFile);
    if (!file.is_open()) {
        return Result<ComplexRefractiveIndex>(Result<ComplexRefractiveIndex>::Err{
            "Failed to open YAML file: " + yamlFile.string()
        });
    }

    ComplexRefractiveIndex cri;
    bool inDataBlock = false;
    bool isNKType = false;
    std::string currentType;

    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;

        const std::string trimmed = line.substr(start);

        // Skip comments
        if (trimmed[0] == '#') continue;

        // Detect data type
        if (trimmed.find("type:") != std::string::npos) {
            if (trimmed.find("tabulated nk") != std::string::npos) {
                isNKType = true;
                currentType = "nk";
            } else if (trimmed.find("tabulated n") != std::string::npos) {
                // Pure n data (no k), treat k as 0
                isNKType = true;
                currentType = "n";
            } else {
                // Other types (formula, etc.) - not supported yet
                isNKType = false;
            }
            continue;
        }

        // Detect data block start
        if (trimmed.find("data: |") != std::string::npos) {
            inDataBlock = true;
            continue;
        }

        // End data block on non-indented line (new section)
        if (inDataBlock && line[0] != ' ' && line[0] != '\t') {
            inDataBlock = false;
            continue;
        }

        // Parse data inside block
        if (inDataBlock && isNKType) {
            f32 wavelength_um = 0.0f;
            f32 n_val = 0.0f;
            f32 k_val = 0.0f;

            int parsed = 0;
            if (currentType == "nk") {
                parsed = std::sscanf(trimmed.c_str(), "%f %f %f",
                                      &wavelength_um, &n_val, &k_val);
                if (parsed != 3) continue;
            } else if (currentType == "n") {
                parsed = std::sscanf(trimmed.c_str(), "%f %f",
                                      &wavelength_um, &n_val);
                if (parsed != 2) continue;
                k_val = 0.0f;  // Transparent material
            }

            // Skip invalid data
            if (wavelength_um <= 0.0f) continue;

            // Convert µm to nm
            const f32 wavelength_nm = wavelength_um * 1000.0f;

            cri.wavelengths_nm.push_back(wavelength_nm);
            cri.n.push_back(n_val);
            cri.k.push_back(k_val);
        }
    }

    if (cri.wavelengths_nm.size() < 2) {
        return Result<ComplexRefractiveIndex>(Result<ComplexRefractiveIndex>::Err{
            "Not enough data points in YAML file (found " +
            std::to_string(cri.wavelengths_nm.size()) + ", need at least 2)"
        });
    }

    // Validate monotonicity
    for (size_t i = 1; i < cri.wavelengths_nm.size(); ++i) {
        if (cri.wavelengths_nm[i] <= cri.wavelengths_nm[i - 1]) {
            return Result<ComplexRefractiveIndex>(Result<ComplexRefractiveIndex>::Err{
                "Wavelengths not monotonically increasing at index " + std::to_string(i)
            });
        }
    }

    QL_LOG_INFO("SpectralIO::LoadRefractiveIndexYAML: Loaded {} points from {} (λ: {:.1f}-{:.1f} nm)",
                cri.wavelengths_nm.size(),
                yamlFile.filename().string(),
                cri.wavelengths_nm.front(), cri.wavelengths_nm.back());

    return Result(std::move(cri));
}

// ============================================================================
// ASTM G-173 Solar Spectrum Loading
// ============================================================================

Result<SpectralCurve, String>
SpectralIO::LoadASTMG173(const std::filesystem::path& csvPath, u32 column) {
    // Validate column (2=ETR, 3=Global, 4=Direct+circumsolar)
    if (column < 2 || column > 4) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "ASTM G-173: Invalid column " + std::to_string(column) +
            " (valid: 2=ETR, 3=Global, 4=Direct+circumsolar)"
        });
    }

    // Check file exists
    if (!std::filesystem::exists(csvPath)) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "ASTM G-173 file not found: " + csvPath.string()
        });
    }

    // Open CSV file
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "Failed to open ASTM G-173 file: " + csvPath.string()
        });
    }

    SpectralCurve curve;
    std::string line;
    u32 lineNumber = 0;
    f32 lastWavelength = -std::numeric_limits<f32>::infinity();

    while (std::getline(file, line)) {
        ++lineNumber;

        // Trim BOM and leading whitespace
        size_t start = 0;
        // Skip UTF-8 BOM if present (EF BB BF)
        if (line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            start = 3;
        }
        start = line.find_first_not_of(" \t\r\n", start);
        if (start == std::string::npos) continue;

        // Skip header line (contains "Wvlgth" or non-numeric first char)
        const char firstChar = line[start];
        if (!std::isdigit(firstChar) && firstChar != '-' && firstChar != '.') {
            continue;  // Header or comment line
        }

        // Parse 4 columns: wavelength, ETR, Global, Direct+circumsolar
        f32 wavelength = 0.0f;
        f32 col2 = 0.0f, col3 = 0.0f, col4 = 0.0f;

        // ASTM G-173 uses comma separator, scientific notation allowed
        int parsed = std::sscanf(line.c_str() + start,
                                  "%f,%f,%f,%f",
                                  &wavelength, &col2, &col3, &col4);

        if (parsed != 4) {
            // Try with spaces after commas
            parsed = std::sscanf(line.c_str() + start,
                                  "%f, %f, %f, %f",
                                  &wavelength, &col2, &col3, &col4);
        }

        if (parsed != 4) {
            QL_LOG_WARN("ASTM G-173: Parse error at line {} (got {} values): '{}'",
                        lineNumber, parsed, line.substr(start, 60));
            continue;
        }

        // Validate monotonicity
        if (wavelength <= lastWavelength) {
            QL_LOG_WARN("ASTM G-173: Non-monotonic wavelength at line {}: {} <= {}",
                        lineNumber, wavelength, lastWavelength);
            continue;
        }

        // Select target column
        f32 value = 0.0f;
        switch (column) {
            case 2: value = col2; break;  // ETR
            case 3: value = col3; break;  // Global tilt
            case 4: value = col4; break;  // Direct+circumsolar
        }

        // Validate irradiance value (can be very small but not negative)
        if (value < 0.0f) {
            value = 0.0f;  // Clamp negative to zero
        }

        curve.samples.emplace_back(wavelength, value);
        lastWavelength = wavelength;
    }

    // Validate minimum samples
    if (curve.samples.size() < 2) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "ASTM G-173: Not enough valid data points (found " +
            std::to_string(curve.samples.size()) + ")"
        });
    }

    const char* columnName = (column == 2) ? "ETR" :
                             (column == 3) ? "Global" : "Direct+circumsolar";

    QL_LOG_INFO("SpectralIO::LoadASTMG173: Loaded {} points [{}] from {} (λ: {:.1f}-{:.1f} nm)",
                curve.samples.size(), columnName,
                csvPath.filename().string(),
                curve.samples.front().first, curve.samples.back().first);

    return Result(std::move(curve));
}

Result<std::pair<SpectralCurve, SpectralCurve>, String>
SpectralIO::LoadASTMG173SunAndSky(const std::filesystem::path& csvPath) {
    // Check file exists
    if (!std::filesystem::exists(csvPath)) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "ASTM G-173 file not found: " + csvPath.string()
            });
    }

    // Open CSV file
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "Failed to open ASTM G-173 file: " + csvPath.string()
            });
    }

    SpectralCurve sunCurve;   // Direct+circumsolar
    SpectralCurve skyCurve;   // Diffuse = Global - Direct

    std::string line;
    u32 lineNumber = 0;
    f32 lastWavelength = -std::numeric_limits<f32>::infinity();

    while (std::getline(file, line)) {
        ++lineNumber;

        // Trim BOM and whitespace
        size_t start = 0;
        if (line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            start = 3;
        }
        start = line.find_first_not_of(" \t\r\n", start);
        if (start == std::string::npos) continue;

        // Skip header
        const char firstChar = line[start];
        if (!std::isdigit(firstChar) && firstChar != '-' && firstChar != '.') {
            continue;
        }

        // Parse columns
        f32 wavelength = 0.0f;
        f32 col2 = 0.0f, col3 = 0.0f, col4 = 0.0f;

        int parsed = std::sscanf(line.c_str() + start,
                                  "%f,%f,%f,%f",
                                  &wavelength, &col2, &col3, &col4);

        if (parsed != 4) {
            parsed = std::sscanf(line.c_str() + start,
                                  "%f, %f, %f, %f",
                                  &wavelength, &col2, &col3, &col4);
        }

        if (parsed != 4 || wavelength <= lastWavelength) {
            continue;
        }

        // Direct sun = column 4
        const f32 directSun = std::max(0.0f, col4);

        // Diffuse sky = Global - Direct (can be zero or small at some wavelengths)
        const f32 diffuseSky = std::max(0.0f, col3 - col4);

        sunCurve.samples.emplace_back(wavelength, directSun);
        skyCurve.samples.emplace_back(wavelength, diffuseSky);

        lastWavelength = wavelength;
    }

    if (sunCurve.samples.size() < 2) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "ASTM G-173: Not enough valid data points"
            });
    }

    QL_LOG_INFO("SpectralIO::LoadASTMG173SunAndSky: Loaded {} points for sun/sky (λ: {:.1f}-{:.1f} nm)",
                sunCurve.samples.size(),
                sunCurve.samples.front().first, sunCurve.samples.back().first);

    return Result(std::make_pair(std::move(sunCurve), std::move(skyCurve)));
}

// ============================================================================
// libRadtran uvspec Output Loading
// ============================================================================

// Helper: Convert wavelength to nm based on unit
static f32 ConvertWavelengthToNm(f32 value, const String& unit) {
    if (unit == "nm") {
        return value;
    } else if (unit == "um") {
        return value * 1000.0f;  // µm to nm
    } else if (unit == "cm-1") {
        // Wavenumber to wavelength: λ(nm) = 1e7 / ν(cm⁻¹)
        if (value <= 0.0f) return 0.0f;
        return 1e7f / value;
    }
    // Default: assume nm
    return value;
}

// Helper: Parse a line of libRadtran uvspec output
// Returns: (wavelength, column_values[]) or empty if parse fails
static std::optional<std::pair<f32, std::vector<f32>>>
ParseLibRadtranLine(const std::string& line) {
    // Trim leading whitespace
    const size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return std::nullopt;

    // Skip comments
    if (line[start] == '#') return std::nullopt;

    // Tokenize by whitespace, with commas and semicolons counting as
    // whitespace so a CSV parses too. libRadtran writes columns; ASTM G-173 and
    // most published solar spectra ship as CSV, and refusing those meant the
    // only solar spectrum in this repository could not be loaded by the reader
    // that exists to load solar spectra.
    std::string body = line.substr(start);
    std::replace_if(body.begin(), body.end(),
                    [](char ch) { return ch == ',' || ch == ';'; }, ' ');

    std::vector<f32> values;
    std::istringstream iss(body);
    f32 val;
    while (iss >> val) {
        values.push_back(val);
    }

    // Need at least wavelength + one data column
    if (values.size() < 2) return std::nullopt;

    const f32 wavelength = values[0];
    values.erase(values.begin());  // Remove wavelength from data columns

    return std::make_pair(wavelength, std::move(values));
}

Result<SpectralCurve, String>
SpectralIO::LoadLibRadtranUvspec(const std::filesystem::path& uvspecFile,
                                  u32 column,
                                  const String& wavelengthUnit) {
    // Validate column index (2-5 for standard uvspec output; wider tables are
    // legitimate -- CIE_illum_FLs.csv is 28 columns, one per fluorescent lamp --
    // so the cap only exists to turn a nonsense index into a message).
    if (column < 2 || column > 64) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "libRadtran: Invalid column " + std::to_string(column) +
            " (valid: 2-64; 2=edir, 3=edn, 4=eup, 5=uavg for uvspec output)"
        });
    }

    // Validate wavelength unit
    if (wavelengthUnit != "nm" && wavelengthUnit != "um" && wavelengthUnit != "cm-1") {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "libRadtran: Unsupported wavelength unit '" + wavelengthUnit +
            "' (supported: nm, um, cm-1)"
        });
    }

    // Check file exists
    if (!std::filesystem::exists(uvspecFile)) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "libRadtran file not found: " + uvspecFile.string()
        });
    }

    // Open file
    std::ifstream file(uvspecFile);
    if (!file.is_open()) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "Failed to open libRadtran file: " + uvspecFile.string()
        });
    }

    SpectralCurve curve;
    std::string line;
    u32 lineNumber = 0;
    f32 lastWavelength_nm = -std::numeric_limits<f32>::infinity();
    bool isWavenumber = (wavelengthUnit == "cm-1");

    while (std::getline(file, line)) {
        ++lineNumber;

        auto parsed = ParseLibRadtranLine(line);
        if (!parsed) continue;

        auto& [wavelength_raw, columns] = *parsed;

        // Check if requested column exists
        const u32 dataIndex = column - 2;  // column 2 -> index 0
        if (dataIndex >= columns.size()) {
            QL_LOG_WARN("libRadtran: Line {} has only {} data columns, need column {}",
                        lineNumber, columns.size() + 1, column);
            continue;
        }

        // Convert wavelength to nm
        const f32 wavelength_nm = ConvertWavelengthToNm(wavelength_raw, wavelengthUnit);

        if (wavelength_nm <= 0.0f) {
            QL_LOG_WARN("libRadtran: Invalid wavelength at line {}: {}", lineNumber, wavelength_raw);
            continue;
        }

        // Get requested column value
        const f32 value = std::max(0.0f, columns[dataIndex]);  // Clamp negative to zero

        // For wavenumber input, data comes in reverse order (high λ to low λ)
        // We'll sort later if needed
        curve.samples.emplace_back(wavelength_nm, value);

        // Track for monotonicity check (after potential reversal)
        if (!isWavenumber) {
            if (wavelength_nm <= lastWavelength_nm) {
                QL_LOG_WARN("libRadtran: Non-monotonic wavelength at line {}: {} <= {}",
                            lineNumber, wavelength_nm, lastWavelength_nm);
            }
            lastWavelength_nm = wavelength_nm;
        }
    }

    // Validate minimum samples
    if (curve.samples.size() < 2) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "libRadtran: Not enough valid data points (found " +
            std::to_string(curve.samples.size()) + ")"
        });
    }

    // Sort by wavelength (needed for wavenumber input which is reverse-ordered)
    std::ranges::sort(curve.samples, [](const auto& a, const auto& b) { return a.first < b.first; });

    // Validate monotonicity after sort
    for (size_t i = 1; i < curve.samples.size(); ++i) {
        if (curve.samples[i].first <= curve.samples[i - 1].first) {
            return Result<SpectralCurve>(Result<SpectralCurve>::Err{
                "libRadtran: Duplicate wavelength at " + std::to_string(curve.samples[i].first) + " nm"
            });
        }
    }

    const char* columnName = (column == 2) ? "edir (direct)" :
                             (column == 3) ? "edn (diffuse down)" :
                             (column == 4) ? "eup (diffuse up)" :
                             (column == 5) ? "uavg (mean)" : "custom";

    QL_LOG_INFO("SpectralIO::LoadLibRadtranUvspec: Loaded {} points [{}] from {} (λ: {:.1f}-{:.1f} nm)",
                curve.samples.size(), columnName,
                uvspecFile.filename().string(),
                curve.samples.front().first, curve.samples.back().first);

    return Result(std::move(curve));
}

// ============================================================================
// Public API: emission spectra
// ============================================================================

namespace {

/// Every built-in that is not a blackbody spans exactly the range its source
/// standardises, and stops there. Illuminant A's defining equation would happily
/// evaluate at 10 um -- CIE 015:2018 tabulates it to 830 nm, and continuing it
/// past that would be the renderer inventing the part of a lamp nobody measured.
/// A caller who genuinely wants a 2856 K Planckian across the infrared should
/// say `blackbody_2856k`, which is that claim written down.
constexpr f32 kEqualEnergyMinNm = 300.0f;
constexpr f32 kEqualEnergyMaxNm = 20000.0f;
constexpr f32 kBlackbodyMinNm = 300.0f;
constexpr f32 kBlackbodyMaxNm = 20000.0f;
constexpr f32 kBlackbodyStepNm = 10.0f;
constexpr f32 kIlluminantAMinNm = 300.0f;
constexpr f32 kIlluminantAMaxNm = 830.0f;
constexpr f32 kIlluminantAStepNm = 1.0f;
constexpr f32 kHalogenTemperatureK = 3000.0f;

String ToLowerAscii(const String& s) {
    String out = s;
    std::ranges::transform(out, out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

/// CIE 015:2018 equation 4.1. The 1.435e7 nm*K is the 1968 value of c2 and the
/// 2848 K is the equation's own constant, not the lamp's temperature -- together
/// they put illuminant A at a distribution temperature of 2856 K on ITS-90.
/// Checked against the CIE's published 1 nm table to 5e-6 relative on all 531
/// rows; assets/luts/README.md records that check.
f64 IlluminantARelative(f64 lambdaNm) {
    constexpr f64 c2NmK = 1.435e7;
    constexpr f64 tK = 2848.0;
    const f64 shape = std::pow(560.0 / lambdaNm, 5.0);
    const f64 num = std::exp(c2NmK / (tK * 560.0)) - 1.0;
    const f64 den = std::exp(c2NmK / (tK * lambdaNm)) - 1.0;
    return 100.0 * shape * (num / den);
}

/// Parse "blackbody_3000k" into 3000. Returns nullopt when the token is not of
/// that shape, which is how the caller falls through to treating it as a path.
std::optional<f32> ParseBlackbodyToken(const String& lower) {
    constexpr StringView kPrefix = "blackbody_";
    if (!lower.starts_with(kPrefix) || !lower.ends_with('k')) return std::nullopt;
    const auto digits = lower.substr(kPrefix.size(), lower.size() - kPrefix.size() - 1);
    if (digits.empty()) return std::nullopt;
    // strtof rather than from_chars: libc++ only grew floating-point from_chars
    // very recently, and this has to build under Clang as well as MSVC and GCC.
    const char* first = digits.c_str();
    char* end = nullptr;
    const f32 t = std::strtof(first, &end);
    if (end != first + digits.size()) return std::nullopt;
    if (!(t > 0.0f) || t > 100000.0f) return std::nullopt;
    return t;
}

SpectralCurve MakeUniform(f32 minNm, f32 maxNm, f32 stepNm, const auto& fn) {
    SpectralCurve curve;
    const auto n = static_cast<u32>((maxNm - minNm) / stepNm) + 1u;
    curve.samples.reserve(n);
    for (u32 i = 0; i < n; ++i) {
        const f32 lambda = minNm + static_cast<f32>(i) * stepNm;
        curve.samples.emplace_back(lambda, static_cast<f32>(fn(lambda)));
    }
    return curve;
}

}  // namespace

const Vector<SpectralIO::EmissionSpectrumInfo>& SpectralIO::BuiltinEmissionSpectra() {
    static const Vector<EmissionSpectrumInfo> kTable = [] {
        Vector<EmissionSpectrumInfo> t;
        t.push_back({"equal_energy",
                     "CIE illuminant E -- flat, favours no wavelength",
                     kEqualEnergyMinNm, kEqualEnergyMaxNm});
        t.push_back({"d65",
                     "CIE standard illuminant D65 -- average daylight, the sRGB white point",
                     D65_LAMBDA_MIN, D65_LAMBDA_MAX});
        t.push_back({"illuminant_a",
                     "CIE standard illuminant A -- 2856 K tungsten, the incandescent standard",
                     kIlluminantAMinNm, kIlluminantAMaxNm});
        t.push_back({"halogen",
                     "Tungsten halogen -- alias for blackbody_3000k",
                     kBlackbodyMinNm, kBlackbodyMaxNm});
        for (u32 i = 0; i < FL_LAMP_COUNT; ++i) {
            const String token = FL_LAMP_TOKENS[i];
            String what = "CIE fluorescent lamp " + token.substr(4);
            if (token == "cie_f2" || token == "cie_f7" || token == "cie_f11") {
                // CIE 015:2018 singles these three out as the ones to use when
                // only one fluorescent lamp is being tested against.
                what += " (CIE-preferred representative)";
            }
            t.push_back({token, what, FL_LAMBDA_MIN, FL_LAMBDA_MAX});
        }
        t.push_back({"blackbody_<T>k",
                     "Planck's law at T kelvin, absolute -- e.g. blackbody_3000k",
                     kBlackbodyMinNm, kBlackbodyMaxNm});
        return t;
    }();
    return kTable;
}

Result<SpectralCurve, String> SpectralIO::LoadEmissionSpectrum(
    const String& nameOrPath, const std::filesystem::path& baseDir, u32 column) {
    using Res = Result<SpectralCurve, String>;

    const String lower = ToLowerAscii(nameOrPath);

    // Tokens are matched before paths, so a file literally named "d65" in the
    // working directory cannot shadow the built-in. That ordering is deliberate:
    // a config naming a built-in must always get the same lamp.
    if (lower == "equal_energy" || lower == "illuminant_e") {
        SpectralCurve curve;
        curve.samples.emplace_back(kEqualEnergyMinNm, 1.0f);
        curve.samples.emplace_back(kEqualEnergyMaxNm, 1.0f);
        return Res(std::move(curve));
    }

    if (lower == "d65") {
        return Res(MakeUniform(D65_LAMBDA_MIN, D65_LAMBDA_MAX, D65_LAMBDA_STEP,
                               [](f32 lambda) { return D65Relative(lambda); }));
    }

    if (lower == "illuminant_a") {
        return Res(MakeUniform(kIlluminantAMinNm, kIlluminantAMaxNm, kIlluminantAStepNm,
                               [](f32 lambda) { return IlluminantARelative(lambda); }));
    }

    const f32 blackbodyK =
        (lower == "halogen") ? kHalogenTemperatureK
                             : ParseBlackbodyToken(lower).value_or(0.0f);
    if (blackbodyK > 0.0f) {
        return Res(MakeUniform(kBlackbodyMinNm, kBlackbodyMaxNm, kBlackbodyStepNm,
                               [blackbodyK](f32 lambda) {
                                   return blackbody::SpectralRadiancePerNm(lambda, blackbodyK);
                               }));
    }

    for (u32 i = 0; i < FL_LAMP_COUNT; ++i) {
        if (lower != FL_LAMP_TOKENS[i]) continue;
        const f32* row = CIE_FL[i];
        return Res(MakeUniform(FL_LAMBDA_MIN, FL_LAMBDA_MAX, FL_LAMBDA_STEP,
                               [row](f32 lambda) {
                                   const f32 pos = (lambda - FL_LAMBDA_MIN) / FL_LAMBDA_STEP;
                                   const auto i0 = static_cast<u32>(pos);
                                   return row[std::min(i0, FL_LUT_SIZE - 1u)];
                               }));
    }

    // Not a token, so it is a file. A misspelt token lands here and fails as a
    // missing path, which reads badly, so say both things.
    std::filesystem::path path(nameOrPath);
    if (path.is_relative() && !baseDir.empty()) path = baseDir / path;
    if (!std::filesystem::exists(path)) {
        return Res(Res::Err{
            "Emission spectrum '" + nameOrPath + "' is neither a built-in nor a file "
            "that exists (looked for " + path.string() +
            "). Built-in tokens: equal_energy, d65, illuminant_a, halogen, "
            "cie_f1..cie_f12, cie_f3.1..cie_f3.15, blackbody_<T>k."});
    }

    auto loaded = LoadLibRadtranUvspec(path, column, "nm");
    if (!loaded) {
        return Res(Res::Err{"Emission spectrum '" + path.string() + "': " + loaded.error()});
    }
    return Res(std::move(loaded.value()));
}

Result<std::pair<SpectralCurve, SpectralCurve>, String>
SpectralIO::LoadLibRadtranSunAndSky(const std::filesystem::path& uvspecFile,
                                     const String& wavelengthUnit,
                                     u32 directColumn,
                                     u32 diffuseColumn,
                                     bool diffuseIsGlobal) {
    // Validate wavelength unit
    if (wavelengthUnit != "nm" && wavelengthUnit != "um" && wavelengthUnit != "cm-1") {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "libRadtran: Unsupported wavelength unit '" + wavelengthUnit +
                "' (supported: nm, um, cm-1)"
            });
    }

    // Check file exists
    if (!std::filesystem::exists(uvspecFile)) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "libRadtran file not found: " + uvspecFile.string()
            });
    }

    // Open file
    std::ifstream file(uvspecFile);
    if (!file.is_open()) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "Failed to open libRadtran file: " + uvspecFile.string()
            });
    }

    SpectralCurve sunCurve;   // edir (column 2)
    SpectralCurve skyCurve;   // edn (column 3)

    std::string line;
    u32 lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;

        auto parsed = ParseLibRadtranLine(line);
        if (!parsed) continue;

        auto& [wavelength_raw, columns] = *parsed;

        // Columns are 1-based with the wavelength as column 1, so the data
        // vector -- which has the wavelength removed -- is indexed from 2.
        // diffuseColumn = 0 means the file has no diffuse column at all, which
        // is the shape of a reference illuminant: CIE D65 is one spectrum, not
        // a sky and a sun. The diffuse curve then comes back empty.
        const bool haveDiffuse = diffuseColumn >= 2;
        const u32 directIndex  = directColumn >= 2 ? directColumn - 2 : 0;
        const u32 diffuseIndex = haveDiffuse ? diffuseColumn - 2 : 0;
        const size_t needed =
            (haveDiffuse ? std::max(directIndex, diffuseIndex) : directIndex) + 1;
        if (columns.size() < needed) {
            QL_LOG_WARN("Solar spectrum: line {} has {} data columns, need {} "
                        "for direct=col{} diffuse=col{}",
                        lineNumber, columns.size(), needed,
                        directColumn, diffuseColumn);
            continue;
        }

        // Convert wavelength to nm
        const f32 wavelength_nm = ConvertWavelengthToNm(wavelength_raw, wavelengthUnit);

        if (wavelength_nm <= 0.0f) continue;

        const f32 directSun = std::max(0.0f, columns[directIndex]);
        // ASTM G-173 and friends publish a global column rather than a diffuse
        // one. Using it as the sky counts the direct beam a second time, so
        // subtract it; clamped because measured global and direct columns can
        // cross by a hair in the noise floor.
        f32 diffuseSky = 0.0f;
        if (haveDiffuse) {
            const f32 rawDiffuse = columns[diffuseIndex];
            diffuseSky = diffuseIsGlobal ? std::max(0.0f, rawDiffuse - directSun)
                                         : std::max(0.0f, rawDiffuse);
        }

        sunCurve.samples.emplace_back(wavelength_nm, directSun);
        // Zeros rather than no samples when the file has no diffuse column: an
        // empty curve is not the same as a dark one, and SolarSpectralLUT
        // rejects it, which silently left the whole illuminant off the GPU
        // while the derived RGB colour looked perfectly correct.
        skyCurve.samples.emplace_back(wavelength_nm, diffuseSky);
    }

    // Validate minimum samples
    if (sunCurve.samples.size() < 2) {
        return Result<std::pair<SpectralCurve, SpectralCurve>>(
            Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                "libRadtran: Not enough valid data points (found " +
                std::to_string(sunCurve.samples.size()) + ")"
            });
    }

    // Sort by wavelength (needed for wavenumber input)
    std::ranges::sort(sunCurve.samples, [](const auto& a, const auto& b) { return a.first < b.first; });
    std::ranges::sort(skyCurve.samples, [](const auto& a, const auto& b) { return a.first < b.first; });

    // Validate monotonicity
    for (size_t i = 1; i < sunCurve.samples.size(); ++i) {
        if (sunCurve.samples[i].first <= sunCurve.samples[i - 1].first) {
            return Result<std::pair<SpectralCurve, SpectralCurve>>(
                Result<std::pair<SpectralCurve, SpectralCurve>>::Err{
                    "libRadtran: Duplicate wavelength at " +
                    std::to_string(sunCurve.samples[i].first) + " nm"
                });
        }
    }

    QL_LOG_INFO("SpectralIO::LoadLibRadtranSunAndSky: Loaded {} points for sun/sky from {} (λ: {:.1f}-{:.1f} nm)",
                sunCurve.samples.size(),
                uvspecFile.filename().string(),
                sunCurve.samples.front().first, sunCurve.samples.back().first);

    return Result(std::make_pair(std::move(sunCurve), std::move(skyCurve)));
}

// ============================================================================
// Public API: ReconstructBasisCurve
// ============================================================================

Result<SpectralCurve, String>
SpectralIO::ReconstructBasisCurve(const std::filesystem::path& basisFile,
                                  const std::filesystem::path& materialsJson,
                                  const String& materialName,
                                  const String& band) {
    using Err = Result<SpectralCurve, String>::Err;

    if (materialName.empty()) {
        return Result<SpectralCurve, String>(Err{"No material name given"});
    }

    // One loader per (basis, materials) pair, kept alive for the process. The
    // JSON is megabytes and a browser reconstructs a curve per selection, so
    // re-parsing per call would make selecting a row visibly slow. Keyed on
    // both paths because a scene can name a different database than the one
    // the previous scene did.
    //
    // Deliberately not thread-safe beyond the mutex: reconstruction is pure
    // once the loader is built, and the mutex only guards the cache itself.
    struct CacheKey {
        std::string basis;
        std::string materials;
        bool operator==(const CacheKey& other) const {
            return basis == other.basis && materials == other.materials;
        }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& key) const {
            return std::hash<std::string>{}(key.basis) ^
                   (std::hash<std::string>{}(key.materials) << 1);
        }
    };

    static std::mutex cacheMutex;
    static std::unordered_map<CacheKey, std::shared_ptr<SpectralBasisLoader>, CacheKeyHash> cache;

    const CacheKey key{basisFile.string(), materialsJson.string()};

    std::shared_ptr<SpectralBasisLoader> loader;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (auto it = cache.find(key); it != cache.end()) {
            loader = it->second;
        } else {
            if (!std::filesystem::exists(basisFile)) {
                return Result<SpectralCurve, String>(
                    Err{"Basis file not found: " + basisFile.string()});
            }
            if (!std::filesystem::exists(materialsJson)) {
                return Result<SpectralCurve, String>(
                    Err{"Materials JSON not found: " + materialsJson.string()});
            }
            auto fresh = std::make_shared<SpectralBasisLoader>();
            if (!fresh->Load(basisFile, materialsJson)) {
                return Result<SpectralCurve, String>(
                    Err{"Failed to load NMF database from " + basisFile.string() +
                        " and " + materialsJson.string()});
            }
            QL_LOG_INFO("SpectralIO::ReconstructBasisCurve: loaded {} materials, {} bands from {}",
                        fresh->GetMaterialCount(), fresh->GetNumBands(),
                        basisFile.filename().string());
            loader = fresh;
            cache.emplace(key, loader);
        }
    }

    // Exact first, then substring -- the same order and meaning ConfigResolve
    // applies to quantiloom_material_ref, so a name that renders also browses.
    const MaterialSpectralData* data = loader->FindMaterial(materialName);
    if (!data) {
        data = loader->FindMaterialPartial(materialName);
    }
    if (!data) {
        return Result<SpectralCurve, String>(
            Err{"Material '" + materialName + "' is not in this database"});
    }

    SpectralCurve curve = loader->ReconstructCurve(data->name, band);
    if (curve.samples.empty()) {
        return Result<SpectralCurve, String>(
            Err{"Material '" + data->name + "' has no data for band '" + band + "'"});
    }
    return Result<SpectralCurve, String>(std::move(curve));
}

} // namespace quantiloom
