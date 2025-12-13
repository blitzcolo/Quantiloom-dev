#pragma once

#include "core/SpectralCube.hpp"
#include "core/SpectralData.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// SpectralIO - HDF5 hyperspectral cube reading/writing + Material databases
// ============================================================================
// HDF5 structure:
//   /data              - 3D dataset [nbands, height, width], float32
//   /wavelengths       - 1D dataset [nbands], float32
//   /metadata          - Group containing string attributes
//
// Material Database Support:
//   - USGS Spectral Library (DHR reflectance)
//   - RefractiveIndex.INFO (complex refractive index n,k)
//   - Custom CSV format
// ============================================================================

class QL_API SpectralIO {
public:
    // ========================================================================
    // HDF5 Writing
    // ========================================================================

    // Write spectral cube to HDF5 file
    static bool WriteHDF5(const std::string& filepath, const SpectralCube& cube);

    // ========================================================================
    // HDF5 Reading
    // ========================================================================

    // Read spectral cube from HDF5 file
    static std::optional<SpectralCube> ReadHDF5(const std::string& filepath);

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if file exists and is valid HDF5
    static bool FileExists(const std::string& filepath);

    // Get cube dimensions without loading data (fast peek)
    static std::optional<std::tuple<u32, u32, u32>> GetDimensions(const std::string& filepath);

    // ========================================================================
    // Spectral Curve Loading (for material properties)
    // ========================================================================

    // Load spectral curve from CSV file
    // Format: wavelength_nm, value (e.g., reflectance, emissivity, transmittance)
    // Example CSV:
    //   400.0, 0.12
    //   410.0, 0.15
    //   ...
    //
    // Returns: Vector of (wavelength_nm, value) pairs
    // NOTE: Wavelengths must be monotonically increasing
    static Result<std::vector<std::pair<f32, f32>>, String>
    LoadSpectralCurveCSV(const std::filesystem::path& csvPath);

    // ========================================================================
    // USGS Spectral Library Support
    // ========================================================================
    // USGS splib07a format: separate wavelength file + reflectance file
    //
    // Wavelength file naming:
    //   splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt
    //   splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt
    //
    // Reflectance file naming convention:
    //   splib07a_{Material}_{SampleID}_{Spectrometer}_AREF.txt
    //   e.g., splib07a_Hematite_Coatd_Qtz_BR93-25B_BECKa_AREF.txt
    //
    // Data format:
    //   - Line 1: metadata (record number, material name, etc.)
    //   - Lines 2+: reflectance values (one per line)
    //   - Invalid data: -1.23e+034
    //
    // Spectrometer codes (from filename):
    //   BECKa/BECKb -> Wavelengths_BECK_Beckman_0.2-3.0_microns.txt
    //   ASDFR/ASDHR -> Wavelengths_ASD_*.txt
    //   NIC4        -> Wavelengths_NIC4_Nicolet_*.txt
    // ========================================================================

    // Load USGS spectral curve
    // @param reflectanceFile: Path to USGS reflectance file (*_AREF.txt)
    // @param wavelengthFile:  Path to corresponding wavelength file
    // @return: SpectralCurve with wavelength (nm) and reflectance pairs
    static Result<SpectralCurve, String>
    LoadUSGS(const std::filesystem::path& reflectanceFile,
             const std::filesystem::path& wavelengthFile);

    // Auto-detect wavelength file from reflectance filename
    // Searches for wavelength file in same directory or parent directory
    static Result<SpectralCurve, String>
    LoadUSGSAuto(const std::filesystem::path& reflectanceFile);

    // ========================================================================
    // RefractiveIndex.INFO Database Support
    // ========================================================================
    // RefractiveIndex.INFO YAML format (tabulated nk):
    //
    // DATA:
    //   - type: tabulated nk
    //     data: |
    //         wavelength_um n k
    //         5.0000 1.3995 0.0369
    //         5.0050 1.3994 0.0365
    //         ...
    //
    // Note: Wavelengths are in micrometers (µm), converted to nm internally
    // ========================================================================

    // Load complex refractive index from RefractiveIndex.INFO YAML file
    // @param yamlFile: Path to .yml file (e.g., Lane.yml, Johnson.yml)
    // @return: ComplexRefractiveIndex with wavelength (nm), n, k
    static Result<ComplexRefractiveIndex, String>
    LoadRefractiveIndexYAML(const std::filesystem::path& yamlFile);
};

} // namespace quantiloom
