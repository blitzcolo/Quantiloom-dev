/**
 * @file SpectralCubeIO.hpp
 * @brief I/O utilities for SpectralCube (ENVI, GeoTIFF, EXR formats)
 *
 * Supports reading and writing hyperspectral data cubes in common formats:
 * - ENVI (Environmental Data Analysis): Industry standard for remote sensing
 * - GeoTIFF: Multi-band TIFF with metadata
 * - OpenEXR: HDR format, spectral layout of Fichet et al. 2021 (one channel
 *   per band, wavelength in the channel name)
 *
 * ENVI Format Details:
 * - Header file (.hdr): ASCII metadata describing data cube
 * - Data file (.dat/.raw): Binary raster data
 * - Interleave options: BSQ, BIL, BIP
 * - Data types: float32 (type=4), float64 (type=5)
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralCube.hpp"
#include <string>
#include <map>

namespace quantiloom {

// ============================================================================
// ENVI Interleave Types
// ============================================================================

/**
 * @enum ENVIInterleave
 * @brief ENVI data interleave formats
 *
 * Memory layout options for hyperspectral data:
 * - BSQ: Band Sequential - all pixels of band 0, then all pixels of band 1, etc.
 * - BIL: Band Interleaved by Line - all bands of line 0, then all bands of line 1
 * - BIP: Band Interleaved by Pixel - all bands of pixel (0,0), then (0,1), etc.
 *
 * Performance characteristics:
 * - BSQ: Best for band-by-band processing (spectral analysis per pixel is slow)
 * - BIL: Balanced for both spatial and spectral access
 * - BIP: Best for per-pixel spectral analysis (spatial processing is slow)
 */
enum class ENVIInterleave : u32 {
    BSQ = 0,  ///< Band Sequential (default, matches SpectralCube internal layout)
    BIL,      ///< Band Interleaved by Line
    BIP       ///< Band Interleaved by Pixel
};

/**
 * @brief Convert ENVIInterleave to string for header
 */
const char* ENVIInterleaveToString(ENVIInterleave interleave);

// ============================================================================
// SpectralCubeIO - Static I/O Utilities
// ============================================================================

/**
 * @class SpectralCubeIO
 * @brief Static utility class for reading/writing SpectralCube
 *
 * Provides static methods for I/O operations on SpectralCube.
 * All methods are self-contained and do not require instantiation.
 *
 * Supported formats:
 * - ENVI (.hdr + .dat): Industry standard hyperspectral format
 * - GeoTIFF (.tif): Multi-band raster with metadata
 * - OpenEXR (.exr): HDR format, one channel per band (Fichet et al. 2021)
 *
 * Example usage:
 * @code
 * // Write to ENVI
 * SpectralCube cube(1024, 768, 128, 400.0f, 2500.0f);
 * // ... fill cube with data ...
 * SpectralCubeIO::WriteENVI(cube, "output/hyperspectral", ENVIInterleave::BSQ);
 *
 * // Read from ENVI
 * auto result = SpectralCubeIO::ReadENVI("input/hyperspectral");
 * if (result.IsOk()) {
 *     SpectralCube& loadedCube = result.Value();
 * }
 * @endcode
 */
class SpectralCubeIO {
public:
    // ========================================================================
    // ENVI Format
    // ========================================================================

    /**
     * @brief Write SpectralCube to ENVI format
     *
     * Creates two files:
     * - basePath.hdr: ASCII header with metadata
     * - basePath.dat: Binary data (float32, native byte order)
     *
     * @param cube SpectralCube to write
     * @param basePath Output path without extension
     * @param interleave Data interleave format (default: BSQ)
     * @return true on success
     */
    static bool WriteENVI(
        const SpectralCube& cube,
        const String& basePath,
        ENVIInterleave interleave = ENVIInterleave::BSQ
    );

    /**
     * @brief Read SpectralCube from ENVI format
     *
     * Reads header (.hdr) and data (.dat or .raw) files.
     * Automatically handles different interleave formats and byte orders.
     *
     * @param basePath Input path without extension (or with .hdr)
     * @return Result containing SpectralCube or error message
     */
    static Result<SpectralCube, String> ReadENVI(const String& basePath);

    // ========================================================================
    // GeoTIFF Format
    // ========================================================================

    /**
     * @brief Write SpectralCube to GeoTIFF format
     *
     * Creates a multi-band TIFF file with:
     * - Float32 samples
     * - One band per wavelength
     * - Metadata in TIFF tags
     *
     * @param cube SpectralCube to write
     * @param path Output path (should end with .tif)
     * @return true on success
     *
     * @note Requires libtiff support (compile-time option)
     */
    static bool WriteGeoTIFF(
        const SpectralCube& cube,
        const String& path
    );

    /**
     * @brief Read SpectralCube from GeoTIFF format
     *
     * @param path Input path
     * @return Result containing SpectralCube or error message
     */
    static Result<SpectralCube, String> ReadGeoTIFF(const String& path);

    // ========================================================================
    // OpenEXR Format
    // ========================================================================

    /**
     * @brief Write SpectralCube to a spectral OpenEXR file
     *
     * Single part, one f32 channel per band, in the layout of Fichet,
     * Pacanowski and Wilkie 2021 (JCGT 10(3)): channels are named
     * "S0.<wavelength>nm" with a comma for the decimal separator, and the
     * header carries `spectralLayoutVersion` and `emissiveUnits`. The cube's
     * own metadata is written alongside as string attributes.
     *
     * @param cube SpectralCube to write
     * @param path Output path (should end with .exr)
     * @return true on success; false for an invalid cube or two bands at one
     *         wavelength, whose channel names would collide
     */
    static bool WriteEXR(
        const SpectralCube& cube,
        const String& path
    );

    /**
     * @brief Read SpectralCube from a spectral OpenEXR file
     *
     * Bands are ordered by the wavelength parsed out of each channel name, not
     * by channel order, which OpenEXR sorts alphabetically. Channels that name
     * no wavelength (an RGB proxy, an alpha) are skipped.
     *
     * @param path Input path
     * @return Result containing SpectralCube or error message
     */
    static Result<SpectralCube, String> ReadEXR(const String& path);

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * @brief Detect format from file extension
     * @param path File path
     * @return Format string ("envi", "geotiff", "exr", or "unknown")
     */
    static String DetectFormat(const String& path);

    /**
     * @brief Auto-detect format and read SpectralCube
     * @param path Input path
     * @return Result containing SpectralCube or error message
     */
    static Result<SpectralCube, String> Read(const String& path);

private:
    // Internal helpers

    /**
     * @brief Write ENVI header file
     */
    static bool WriteENVIHeader(
        const String& headerPath,
        const SpectralCube& cube,
        ENVIInterleave interleave
    );

    /**
     * @brief Parse ENVI header file
     */
    static Result<std::map<String, String>, String> ParseENVIHeader(
        const String& headerPath
    );

    /**
     * @brief Convert cube data from BSQ to specified interleave
     */
    static Vector<f32> ConvertInterleave(
        const SpectralCube& cube,
        ENVIInterleave targetInterleave
    );

    /**
     * @brief Convert data from specified interleave to BSQ
     */
    static void ConvertToBSQ(
        const Vector<f32>& data,
        SpectralCube& cube,
        ENVIInterleave sourceInterleave
    );
};

} // namespace quantiloom
