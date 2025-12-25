#include "SpectralIO.hpp"

#include <H5Cpp.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

namespace quantiloom {

// ============================================================================
// Helper: Write metadata as HDF5 attributes
// ============================================================================

static void WriteMetadata(const H5::H5File& file, const SpectralCube& cube) {
    // Create metadata group
    const H5::Group metaGroup = file.createGroup("/metadata");

    // Write scalar attributes
    {
        const H5::DataSpace scalar(H5S_SCALAR);

        // lambda_min
        const H5::Attribute attr_lmin = metaGroup.createAttribute(
            "lambda_min", H5::PredType::NATIVE_FLOAT, scalar);
        attr_lmin.write(H5::PredType::NATIVE_FLOAT, &cube.lambda_min);

        // lambda_max
        const H5::Attribute attr_lmax = metaGroup.createAttribute(
            "lambda_max", H5::PredType::NATIVE_FLOAT, scalar);
        attr_lmax.write(H5::PredType::NATIVE_FLOAT, &cube.lambda_max);

        // delta_lambda
        const H5::Attribute attr_delta = metaGroup.createAttribute(
            "delta_lambda", H5::PredType::NATIVE_FLOAT, scalar);
        attr_delta.write(H5::PredType::NATIVE_FLOAT, &cube.delta_lambda);
    }

    // Write string attributes from metadata map
    const H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    const H5::DataSpace scalar(H5S_SCALAR);

    for (const auto& [key, value] : cube.metadata) {
        H5::Attribute attr = metaGroup.createAttribute(key, strType, scalar);
        attr.write(strType, value);
    }
}

// ============================================================================
// Helper: Read metadata from HDF5 attributes
// ============================================================================

static void ReadMetadata(const H5::H5File& file, SpectralCube& cube) {
    try {
        const H5::Group metaGroup = file.openGroup("/metadata");

        // Read scalar attributes
        {
            const H5::Attribute attr_lmin = metaGroup.openAttribute("lambda_min");
            attr_lmin.read(H5::PredType::NATIVE_FLOAT, &cube.lambda_min);

            const H5::Attribute attr_lmax = metaGroup.openAttribute("lambda_max");
            attr_lmax.read(H5::PredType::NATIVE_FLOAT, &cube.lambda_max);

            const H5::Attribute attr_delta = metaGroup.openAttribute("delta_lambda");
            attr_delta.read(H5::PredType::NATIVE_FLOAT, &cube.delta_lambda);
        }

        // Read string attributes
        const H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
        for (hsize_t i = 0; i < metaGroup.getNumAttrs(); ++i) {
            H5::Attribute attr = metaGroup.openAttribute(i);
            std::string name = attr.getName();

            // Skip scalar numeric attributes
            if (name == "lambda_min" || name == "lambda_max" || name == "delta_lambda") {
                continue;
            }

            // Read string attribute
            std::string value;
            attr.read(strType, value);
            cube.metadata[name] = value;
        }

    } catch (const H5::Exception& e) {
        QL_LOG_WARN("SpectralIO::ReadMetadata: Failed to read metadata: {}", e.getDetailMsg());
    }
}

// ============================================================================
// Public API: WriteHDF5
// ============================================================================

bool SpectralIO::WriteHDF5(const std::string& filepath, const SpectralCube& cube) {
    if (!cube.IsValid()) {
        QL_LOG_ERROR("SpectralIO::WriteHDF5: Invalid spectral cube");
        return false;
    }

    try {
        // Create HDF5 file (overwrite if exists)
        const H5::H5File file(filepath, H5F_ACC_TRUNC);

        // ====================================================================
        // Write main data cube: /data [nbands, height, width]
        // ====================================================================
        {
            hsize_t dims[3] = {cube.nbands, cube.height, cube.width};
            const H5::DataSpace dataspace(3, dims);

            const H5::DataSet dataset = file.createDataSet(
                "/data", H5::PredType::NATIVE_FLOAT, dataspace);

            dataset.write(cube.data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // ====================================================================
        // Write wavelength array: /wavelengths [nbands]
        // ====================================================================
        {
            hsize_t dims[1] = {cube.nbands};
            const H5::DataSpace dataspace(1, dims);

            const H5::DataSet dataset = file.createDataSet(
                "/wavelengths", H5::PredType::NATIVE_FLOAT, dataspace);

            dataset.write(cube.wavelengths.data(), H5::PredType::NATIVE_FLOAT);
        }

        // ====================================================================
        // Write metadata as attributes
        // ====================================================================
        WriteMetadata(file, cube);

        QL_LOG_INFO("SpectralIO::WriteHDF5: Wrote {}x{}x{} cube to {}",
                    cube.width, cube.height, cube.nbands, filepath);
        return true;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("SpectralIO::WriteHDF5: Failed to write {}: {}",
                     filepath, e.getDetailMsg());
        return false;
    }
}

// ============================================================================
// Public API: ReadHDF5
// ============================================================================

std::optional<SpectralCube> SpectralIO::ReadHDF5(const std::string& filepath) {
    if (!FileExists(filepath)) {
        QL_LOG_ERROR("SpectralIO::ReadHDF5: File not found: {}", filepath);
        return std::nullopt;
    }

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        // ====================================================================
        // Read /data dimensions
        // ====================================================================
        const H5::DataSet dataset = file.openDataSet("/data");
        const H5::DataSpace dataspace = dataset.getSpace();

        if (int rank = dataspace.getSimpleExtentNdims(); rank != 3) {
            QL_LOG_ERROR("SpectralIO::ReadHDF5: Expected 3D dataset, got rank {}", rank);
            return std::nullopt;
        }

        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims);

        u32 nbands = static_cast<u32>(dims[0]);
        u32 height = static_cast<u32>(dims[1]);
        u32 width = static_cast<u32>(dims[2]);

        // ====================================================================
        // Read metadata to get wavelength range
        // ====================================================================
        SpectralCube cube;
        cube.width = width;
        cube.height = height;
        cube.nbands = nbands;

        ReadMetadata(file, cube);

        // ====================================================================
        // Allocate and read data
        // ====================================================================
        cube.data.resize(width * height * nbands);
        dataset.read(cube.data.data(), H5::PredType::NATIVE_FLOAT);

        // ====================================================================
        // Read wavelength array
        // ====================================================================
        {
            const H5::DataSet waveDataset = file.openDataSet("/wavelengths");
            const H5::DataSpace waveSpace = waveDataset.getSpace();

            hsize_t waveDims[1];
            waveSpace.getSimpleExtentDims(waveDims);

            if (waveDims[0] != nbands) {
                QL_LOG_ERROR("SpectralIO::ReadHDF5: Wavelength array size mismatch");
                return std::nullopt;
            }

            cube.wavelengths.resize(nbands);
            waveDataset.read(cube.wavelengths.data(), H5::PredType::NATIVE_FLOAT);
        }

        // Validate cube
        if (!cube.IsValid()) {
            QL_LOG_ERROR("SpectralIO::ReadHDF5: Loaded cube failed validation");
            return std::nullopt;
        }

        QL_LOG_INFO("SpectralIO::ReadHDF5: Read {}x{}x{} cube from {}",
                    width, height, nbands, filepath);
        return cube;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("SpectralIO::ReadHDF5: Failed to read {}: {}",
                     filepath, e.getDetailMsg());
        return std::nullopt;
    }
}

// ============================================================================
// Public API: FileExists
// ============================================================================

bool SpectralIO::FileExists(const std::string& filepath) {
    return std::filesystem::exists(filepath) &&
           std::filesystem::is_regular_file(filepath);
}

// ============================================================================
// Public API: GetDimensions
// ============================================================================

std::optional<std::tuple<u32, u32, u32>> SpectralIO::GetDimensions(
    const std::string& filepath)
{
    if (!FileExists(filepath)) {
        return std::nullopt;
    }

    try {
        const H5::H5File file(filepath, H5F_ACC_RDONLY);
        const H5::DataSet dataset = file.openDataSet("/data");
        const H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims);

        return std::make_tuple(
            static_cast<u32>(dims[2]),  // width
            static_cast<u32>(dims[1]),  // height
            static_cast<u32>(dims[0])   // nbands
        );

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("SpectralIO::GetDimensions: Failed: {}", e.getDetailMsg());
        return std::nullopt;
    }
}

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
//
// libRadtran uvspec produces space-separated output with configurable columns.
// Standard output format (output_quantity irradiance):
//   wavelength  edir  edn  eup  uavg
//
// Column indices (1-based, column 1 = wavelength):
//   2 = edir: Direct solar irradiance
//   3 = edn:  Downward diffuse irradiance
//   4 = eup:  Upward diffuse irradiance
//   5 = uavg: Mean irradiance
//
// Wavelength units vary based on libRadtran configuration:
//   - nm (default): nanometers
//   - um: micrometers (multiply by 1000 to get nm)
//   - cm-1: wavenumber (convert: λ_nm = 1e7 / wavenumber)
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

    // Tokenize by whitespace
    std::vector<f32> values;
    std::istringstream iss(line.substr(start));
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
    // Validate column index (2-5 for standard output)
    if (column < 2 || column > 10) {
        return Result<SpectralCurve>(Result<SpectralCurve>::Err{
            "libRadtran: Invalid column " + std::to_string(column) +
            " (valid: 2=edir, 3=edn, 4=eup, 5=uavg, or higher for custom output)"
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

Result<std::pair<SpectralCurve, SpectralCurve>, String>
SpectralIO::LoadLibRadtranSunAndSky(const std::filesystem::path& uvspecFile,
                                     const String& wavelengthUnit) {
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
    bool isWavenumber = (wavelengthUnit == "cm-1");

    while (std::getline(file, line)) {
        ++lineNumber;

        auto parsed = ParseLibRadtranLine(line);
        if (!parsed) continue;

        auto& [wavelength_raw, columns] = *parsed;

        // Need at least edir (col 2) and edn (col 3), i.e., 2 data columns
        if (columns.size() < 2) {
            QL_LOG_WARN("libRadtran: Line {} has only {} data columns, need at least 2 (edir, edn)",
                        lineNumber, columns.size());
            continue;
        }

        // Convert wavelength to nm
        const f32 wavelength_nm = ConvertWavelengthToNm(wavelength_raw, wavelengthUnit);

        if (wavelength_nm <= 0.0f) continue;

        // edir = column index 0 (column 2 in 1-based)
        // edn = column index 1 (column 3 in 1-based)
        const f32 directSun = std::max(0.0f, columns[0]);
        const f32 diffuseSky = std::max(0.0f, columns[1]);

        sunCurve.samples.emplace_back(wavelength_nm, directSun);
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

} // namespace quantiloom
