#include "LUTLoader.hpp"

#include <H5Cpp.h>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// Helper: Read 1D float array from HDF5 dataset
// ============================================================================

static bool Read1DArray(
    H5::H5File& file,
    const std::string& datasetName,
    std::vector<f32>& outArray)
{
    try {
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 1) {
            QL_LOG_ERROR("LUTLoader: Expected 1D dataset for {}, got rank {}", datasetName, rank);
            return false;
        }

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);

        outArray.resize(dims[0]);
        dataset.read(outArray.data(), H5::PredType::NATIVE_FLOAT);

        return true;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("LUTLoader: Failed to read {}: {}", datasetName, e.getDetailMsg());
        return false;
    }
}

// ============================================================================
// Helper: Write 1D float array to HDF5 dataset
// ============================================================================

static bool Write1DArray(
    H5::H5File& file,
    const std::string& datasetName,
    const std::vector<f32>& data)
{
    try {
        hsize_t dims[1] = {data.size()};
        H5::DataSpace dataspace(1, dims);

        H5::DataSet dataset = file.createDataSet(
            datasetName, H5::PredType::NATIVE_FLOAT, dataspace);

        dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
        return true;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("LUTLoader: Failed to write {}: {}", datasetName, e.getDetailMsg());
        return false;
    }
}

// ============================================================================
// Helper: Read metadata from /metadata group
// ============================================================================

static void ReadMetadata(H5::H5File& file, AtmosphereLUT& lut) {
    try {
        H5::Group metaGroup = file.openGroup("/metadata");

        for (hsize_t i = 0; i < metaGroup.getNumAttrs(); ++i) {
            H5::Attribute attr = metaGroup.openAttribute(i);
            std::string name = attr.getName();

            // Query the actual datatype instead of assuming
            H5::DataType dtype = attr.getDataType();

            if (dtype.getClass() == H5T_STRING) {
                H5::StrType strType = attr.getStrType();
                std::string value;

                if (strType.isVariableStr()) {
                    // Variable-length string: use char* buffer
                    char* c_str = nullptr;
                    attr.read(strType, &c_str);
                    if (c_str != nullptr) {
                        value = std::string(c_str);
                        // CRITICAL: free memory allocated by HDF5 for variable-length strings
                        H5free_memory(c_str);
                    }
                } else {
                    // Fixed-length string: allocate buffer of exact size
                    size_t str_len = strType.getSize();
                    std::vector<char> buffer(str_len + 1, '\0');
                    attr.read(strType, buffer.data());
                    value = std::string(buffer.data());
                }

                lut.metadata[name] = value;
            }
        }

    } catch (const H5::Exception& e) {
        QL_LOG_WARN("LUTLoader: Failed to read metadata: {}", e.getDetailMsg());
    }
}

// ============================================================================
// Helper: Write metadata to /metadata group
// ============================================================================

static void WriteMetadata(H5::H5File& file, const AtmosphereLUT& lut) {
    try {
        H5::Group metaGroup = file.createGroup("/metadata");
        H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
        H5::DataSpace scalar(H5S_SCALAR);

        for (const auto& [key, value] : lut.metadata) {
            H5::Attribute attr = metaGroup.createAttribute(key, strType, scalar);
            attr.write(strType, value);
        }

    } catch (const H5::Exception& e) {
        QL_LOG_WARN("LUTLoader: Failed to write metadata: {}", e.getDetailMsg());
    }
}

// ============================================================================
// Public API: LoadHDF5
// ============================================================================

std::optional<AtmosphereLUT> LUTLoader::LoadHDF5(const std::string& filepath) {
    if (!FileExists(filepath)) {
        QL_LOG_ERROR("LUTLoader::LoadHDF5: File not found: {}", filepath);
        return std::nullopt;
    }

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        AtmosphereLUT lut;

        // Read datasets
        if (!Read1DArray(file, "/wavelengths", lut.wavelengths)) {
            return std::nullopt;
        }

        if (!Read1DArray(file, "/solar_irradiance", lut.solar_irradiance)) {
            return std::nullopt;
        }

        if (!Read1DArray(file, "/sky_radiance", lut.sky_radiance)) {
            return std::nullopt;
        }

        if (!Read1DArray(file, "/transmittance", lut.transmittance)) {
            return std::nullopt;
        }

        // Read metadata
        ReadMetadata(file, lut);

        // Validate
        if (!lut.IsValid()) {
            QL_LOG_ERROR("LUTLoader::LoadHDF5: Loaded LUT failed validation");
            return std::nullopt;
        }

        QL_LOG_INFO("LUTLoader::LoadHDF5: Loaded LUT with {} wavelength samples from {}",
                    lut.Size(), filepath);
        return lut;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("LUTLoader::LoadHDF5: Failed to load {}: {}",
                     filepath, e.getDetailMsg());
        return std::nullopt;
    }
}

// ============================================================================
// Public API: SaveHDF5
// ============================================================================

bool LUTLoader::SaveHDF5(const std::string& filepath, const AtmosphereLUT& lut) {
    if (!lut.IsValid()) {
        QL_LOG_ERROR("LUTLoader::SaveHDF5: Invalid LUT");
        return false;
    }

    try {
        H5::H5File file(filepath, H5F_ACC_TRUNC);

        // Write datasets
        if (!Write1DArray(file, "/wavelengths", lut.wavelengths)) {
            return false;
        }

        if (!Write1DArray(file, "/solar_irradiance", lut.solar_irradiance)) {
            return false;
        }

        if (!Write1DArray(file, "/sky_radiance", lut.sky_radiance)) {
            return false;
        }

        if (!Write1DArray(file, "/transmittance", lut.transmittance)) {
            return false;
        }

        // Write metadata
        WriteMetadata(file, lut);

        QL_LOG_INFO("LUTLoader::SaveHDF5: Saved LUT with {} wavelength samples to {}",
                    lut.Size(), filepath);
        return true;

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("LUTLoader::SaveHDF5: Failed to save {}: {}",
                     filepath, e.getDetailMsg());
        return false;
    }
}

// ============================================================================
// Public API: FileExists
// ============================================================================

bool LUTLoader::FileExists(const std::string& filepath) {
    return std::filesystem::exists(filepath) &&
           std::filesystem::is_regular_file(filepath);
}

// ============================================================================
// Public API: GetWavelengthRange
// ============================================================================

std::optional<std::pair<f32, f32>> LUTLoader::GetWavelengthRange(
    const std::string& filepath)
{
    if (!FileExists(filepath)) {
        return std::nullopt;
    }

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("/wavelengths");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);

        if (dims[0] < 2) {
            return std::nullopt;
        }

        std::vector<f32> wavelengths(dims[0]);
        dataset.read(wavelengths.data(), H5::PredType::NATIVE_FLOAT);

        return std::make_pair(wavelengths.front(), wavelengths.back());

    } catch (const H5::Exception& e) {
        QL_LOG_ERROR("LUTLoader::GetWavelengthRange: Failed: {}", e.getDetailMsg());
        return std::nullopt;
    }
}

} // namespace quantiloom
