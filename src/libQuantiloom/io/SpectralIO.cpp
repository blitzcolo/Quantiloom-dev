#include "SpectralIO.hpp"

#include <H5Cpp.h>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// Helper: Write metadata as HDF5 attributes
// ============================================================================

static void WriteMetadata(H5::H5File& file, const SpectralCube& cube) {
    // Create metadata group
    H5::Group metaGroup = file.createGroup("/metadata");

    // Write scalar attributes
    {
        H5::DataSpace scalar(H5S_SCALAR);

        // lambda_min
        H5::Attribute attr_lmin = metaGroup.createAttribute(
            "lambda_min", H5::PredType::NATIVE_FLOAT, scalar);
        attr_lmin.write(H5::PredType::NATIVE_FLOAT, &cube.lambda_min);

        // lambda_max
        H5::Attribute attr_lmax = metaGroup.createAttribute(
            "lambda_max", H5::PredType::NATIVE_FLOAT, scalar);
        attr_lmax.write(H5::PredType::NATIVE_FLOAT, &cube.lambda_max);

        // delta_lambda
        H5::Attribute attr_delta = metaGroup.createAttribute(
            "delta_lambda", H5::PredType::NATIVE_FLOAT, scalar);
        attr_delta.write(H5::PredType::NATIVE_FLOAT, &cube.delta_lambda);
    }

    // Write string attributes from metadata map
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);

    for (const auto& [key, value] : cube.metadata) {
        H5::Attribute attr = metaGroup.createAttribute(key, strType, scalar);
        attr.write(strType, value);
    }
}

// ============================================================================
// Helper: Read metadata from HDF5 attributes
// ============================================================================

static void ReadMetadata(H5::H5File& file, SpectralCube& cube) {
    try {
        H5::Group metaGroup = file.openGroup("/metadata");

        // Read scalar attributes
        {
            H5::Attribute attr_lmin = metaGroup.openAttribute("lambda_min");
            attr_lmin.read(H5::PredType::NATIVE_FLOAT, &cube.lambda_min);

            H5::Attribute attr_lmax = metaGroup.openAttribute("lambda_max");
            attr_lmax.read(H5::PredType::NATIVE_FLOAT, &cube.lambda_max);

            H5::Attribute attr_delta = metaGroup.openAttribute("delta_lambda");
            attr_delta.read(H5::PredType::NATIVE_FLOAT, &cube.delta_lambda);
        }

        // Read string attributes
        H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
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
        H5::H5File file(filepath, H5F_ACC_TRUNC);

        // ====================================================================
        // Write main data cube: /data [nbands, height, width]
        // ====================================================================
        {
            hsize_t dims[3] = {cube.nbands, cube.height, cube.width};
            H5::DataSpace dataspace(3, dims);

            H5::DataSet dataset = file.createDataSet(
                "/data", H5::PredType::NATIVE_FLOAT, dataspace);

            dataset.write(cube.data.data(), H5::PredType::NATIVE_FLOAT);
        }

        // ====================================================================
        // Write wavelength array: /wavelengths [nbands]
        // ====================================================================
        {
            hsize_t dims[1] = {cube.nbands};
            H5::DataSpace dataspace(1, dims);

            H5::DataSet dataset = file.createDataSet(
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
        H5::DataSet dataset = file.openDataSet("/data");
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 3) {
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
            H5::DataSet waveDataset = file.openDataSet("/wavelengths");
            H5::DataSpace waveSpace = waveDataset.getSpace();

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
        H5::H5File file(filepath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("/data");
        H5::DataSpace dataspace = dataset.getSpace();

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

} // namespace quantiloom
