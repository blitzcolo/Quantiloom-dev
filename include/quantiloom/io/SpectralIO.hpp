#pragma once

#include "core/SpectralCube.hpp"
#include "core/SpectralData.hpp"
#include "core/Log.hpp"
#include <string>
#include <optional>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// SpectralIO - Spectral data loading for materials and illumination
// ============================================================================
// Supported formats:
//   - CSV: Generic wavelength,value pairs
//   - USGS Spectral Library: DHR reflectance measurements
//   - RefractiveIndex.INFO: Complex refractive index (n,k) YAML
//   - ASTM G-173: Standard solar irradiance spectra
//   - libRadtran uvspec: Atmospheric radiative transfer output
// ============================================================================

class QL_API SpectralIO {
public:
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
    // Emission spectra (what a surface emits, not what it reflects)
    // ========================================================================
    //
    // A glTF material's self-emission is an RGB triple, and an RGB triple is
    // not a lamp. The renderer's only way to get a spectrum out of one is to
    // upsample it and multiply by D65 -- a computer-graphics construct, and
    // exactly the kind of invention the band-aware convention forbids for
    // reflectance. `emissive_curve` is the way out: it binds a measured or
    // standard spectral radiance to a material, the same way `solar_lut` binds
    // one to the sun.
    //
    // A token names a built-in; anything else is a path to a table.

    struct EmissionSpectrumInfo {
        String token;        ///< what a config writes, e.g. "cie_f7"
        String description;  ///< one line, for a UI listing
        f32 lambdaMinNm = 0.0f;
        f32 lambdaMaxNm = 0.0f;
    };

    // Every built-in emission spectrum, in a stable order suitable for a UI.
    // The blackbody family is parametric and appears once, as "blackbody_<T>k".
    [[nodiscard]] static const Vector<EmissionSpectrumInfo>& BuiltinEmissionSpectra();

    // Resolve a built-in token or a file path to a spectral radiance curve.
    //
    // Tokens (case-insensitive):
    //   equal_energy / illuminant_e   flat, CIE illuminant E
    //   d65                           CIE standard illuminant D65
    //   illuminant_a                  CIE standard illuminant A, 2856 K tungsten
    //   halogen                       alias for blackbody_3000k
    //   cie_f1 .. cie_f12             CIE fluorescent lamps
    //   cie_f3.1 .. cie_f3.15         CIE fluorescent lamps, FL3 series
    //   blackbody_<T>k                Planck at T kelvin, e.g. blackbody_3000k
    //
    // Everything except the blackbody family is a RELATIVE distribution, as
    // published -- normalised to 100 at 560 nm, with no absolute level. The
    // blackbody family is absolute, in W m^-2 sr^-1 nm^-1. Which is which is
    // why the caller must decide on a scale rather than inherit one; see
    // EmissiveScale in ConfigResolve.hpp.
    //
    // @param nameOrPath  A token above, or a path to a whitespace- or
    //                    comma-separated table with wavelength in column 1
    // @param baseDir     Directory a relative path resolves against
    // @param column      1-based column holding the radiance, for a file
    [[nodiscard]] static Result<SpectralCurve, String> LoadEmissionSpectrum(
        const String& nameOrPath, const std::filesystem::path& baseDir, u32 column = 2);

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
    // ASTM G-173 Solar Spectrum Loading
    // ========================================================================
    // ASTM G-173-03 Reference Solar Spectral Irradiances
    // Standard tables for solar spectrum at Air Mass 1.5
    //
    // File format (4-column CSV with header):
    //   Wvlgth nm, Etr W*m-2*nm-1, Global tilt W*m-2*nm-1, Direct+circumsolar W*m-2*nm-1
    //
    // Columns:
    //   1. Wavelength (nm) - 280 to 4000nm
    //   2. ETR: Extraterrestrial radiation (top of atmosphere)
    //   3. Global tilt: Hemispherical on 37° tilted surface
    //   4. Direct+circumsolar: Direct normal + circumsolar (used for sun)
    //
    // For direct sun illumination, use column 4 (Direct+circumsolar)
    // For diffuse sky, subtract direct from global (column 3 - column 4)
    //
    // Units: Spectral irradiance W·m⁻²·nm⁻¹
    // ========================================================================

    // Load ASTM G-173 solar spectrum CSV
    // @param csvPath: Path to astmg173.csv
    // @param column: Which column to load (2=ETR, 3=Global, 4=Direct+circumsolar)
    // @return: SpectralCurve with (wavelength_nm, irradiance) pairs
    static Result<SpectralCurve, String>
    LoadASTMG173(const std::filesystem::path& csvPath, u32 column = 4);

    // Load both direct sun and diffuse sky spectra from ASTM G-173
    // @param csvPath: Path to astmg173.csv
    // @return: Pair of (direct_sun, diffuse_sky) spectral curves
    static Result<std::pair<SpectralCurve, SpectralCurve>, String>
    LoadASTMG173SunAndSky(const std::filesystem::path& csvPath);

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

    // ========================================================================
    // libRadtran uvspec Output Loading
    // ========================================================================
    // libRadtran is a radiative transfer model for atmospheric science.
    // The uvspec tool outputs spectral irradiance data in a text format.
    //
    // Typical uvspec output format (space-separated columns):
    //   # comment lines start with #
    //   wavelength(nm)  edir  edn  eup  uavg
    //   280.000  0.000e+00  4.731e-23  0.000e+00  0.000e+00
    //   300.000  5.140e-01  1.023e-04  2.498e-06  1.234e-05
    //   ...
    //
    // Columns (1-indexed, column 1 is wavelength):
    //   edir: Direct solar irradiance (W·m⁻²·nm⁻¹)
    //   edn:  Downward diffuse irradiance (W·m⁻²·nm⁻¹)
    //   eup:  Upward diffuse irradiance (W·m⁻²·nm⁻¹)
    //   uavg: Mean irradiance (W·m⁻²·nm⁻¹)
    //
    // For rendering:
    //   - Direct sun illumination: use edir (column 2)
    //   - Diffuse sky illumination: use edn (column 3)
    //
    // NOTE: libRadtran output format is configurable. This loader assumes
    // the standard "edir edn eup uavg" output format from:
    //   uvspec -i input.inp -o output.txt
    //
    // UNITS: Input wavelength can be in nm (default) or other units
    // specified by wavelengthUnit parameter. Output is always in nm.
    // ========================================================================

    // Load single spectral column from libRadtran uvspec output
    // @param uvspecFile: Path to uvspec output file
    // @param column: Which data column to load (2=edir, 3=edn, 4=eup, 5=uavg)
    // @param wavelengthUnit: Wavelength unit in input file ("nm", "um", "cm-1")
    // @return: SpectralCurve with (wavelength_nm, irradiance) pairs
    static Result<SpectralCurve, String>
    LoadLibRadtranUvspec(const std::filesystem::path& uvspecFile,
                         u32 column = 2,
                         const String& wavelengthUnit = "nm");

    // Load both direct sun and diffuse sky spectra from a tabulated file
    //
    // The sun's spectrum is user-supplied data, the same as a material's
    // reflectance curve. Whitespace- or comma-separated, one row per
    // wavelength; column numbers are 1-based and count the wavelength as
    // column 1. The defaults are libRadtran uvspec's layout.
    //
    // ASTM G-173, the reference terrestrial spectrum this repository ships,
    // needs directColumn = 4 and diffuseColumn = 3 with diffuseIsGlobal set,
    // because its column 3 is global irradiance -- direct included -- and
    // using it as the diffuse sky would count the sun twice.
    //
    // @param file: Path to the spectrum
    // @param wavelengthUnit: Wavelength unit in the file ("nm", "um", "cm-1")
    // @param directColumn: 1-based column holding direct solar irradiance
    // @param diffuseColumn: 1-based column holding diffuse (or global) sky,
    //        or 0 when the file has none -- a reference illuminant such as
    //        CIE D65 is one spectrum, not a sun and a sky. The diffuse curve
    //        comes back empty in that case.
    // @param diffuseIsGlobal: subtract direct from diffuseColumn, clamped at 0
    // @return: Pair of (direct_sun, diffuse_sky) spectral curves
    static Result<std::pair<SpectralCurve, SpectralCurve>, String>
    LoadLibRadtranSunAndSky(const std::filesystem::path& file,
                            const String& wavelengthUnit = "nm",
                            u32 directColumn = 2,
                            u32 diffuseColumn = 3,
                            bool diffuseIsGlobal = false);

    // ========================================================================
    // NMF measured-material database
    // ========================================================================

    // Reconstruct one material's measured reflectance from an NMF basis.
    //
    // The databases baked into assets/spectral/ (USGS, ECOSTRESS, RII) store a
    // shared basis plus per-material weights; a curve is the weighted sum. The
    // renderer does this internally when a scene names
    // quantiloom_material_ref, but a host that wants to *offer* the database --
    // a material browser, a preview plot -- needs the same reconstruction
    // without loading a scene, which is what this is for.
    //
    // The name is matched exactly first, then as a case-insensitive substring,
    // the same order and meaning the config path uses. Both files are parsed on
    // first use and cached, because the materials JSON is megabytes and a
    // browser reconstructs a curve per selection.
    //
    // @param basisFile: quantiloom_basis_v3_*.qlbin
    // @param materialsJson: quantiloom_materials_*.json
    // @param materialName: database entry, exact or substring
    // @param band: "VIS", "NIR", "SWIR", "MWIR" or "LWIR"
    // @return: the reconstructed curve, or why it could not be produced
    static Result<SpectralCurve, String>
    ReconstructBasisCurve(const std::filesystem::path& basisFile,
                          const std::filesystem::path& materialsJson,
                          const String& materialName,
                          const String& band);
};

} // namespace quantiloom
