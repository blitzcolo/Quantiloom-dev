/**
 * @file SpectralCube.hpp
 * @brief Hyperspectral data cube container for multi-wavelength rendering output
 *
 * Provides SpectralCube struct for storing 3D hyperspectral image data:
 * - Spatial dimensions: width × height (pixels)
 * - Spectral dimension: nbands (wavelength channels)
 * - Wavelength metadata: range and spacing
 *
 * Memory layout: C-order (band-major, BSQ format)
 * @code
 * data[band * height * width + y * width + x]
 * @endcode
 *
 * This layout choice provides:
 * 1. Per-band contiguity: Each wavelength band is contiguous in memory
 * 2. HDF5 compatibility: Matches HDF5 C-order datasets
 * 3. MODTRAN comparison: Atmospheric models use band-major output
 * 4. Efficient band extraction: Single memcpy per band
 *
 * Alternative layouts (BIP/BIL) can be handled via HDF5 chunking and transpose.
 *
 * Usage example:
 * @code
 * // Create hyperspectral cube: 512x512 spatial, 64 bands, 400-800nm
 * SpectralCube cube(512, 512, 64, 400.0f, 800.0f);
 *
 * // Set pixel value
 * cube(x, y, band) = radiance;
 *
 * // Query wavelength for band
 * f32 wavelength = cube.wavelengths[band];
 *
 * // Find band closest to target wavelength
 * u32 band850 = cube.FindBand(850.0f);
 *
 * // Save to HDF5
 * SpectralIO::WriteHDF5("output.h5", cube);
 * @endcode
 *
 * @note Memory layout: data[b][y][x] (C-order, band-major)
 * @note Always uses f32 storage (physical radiance units)
 * @note Wavelength array auto-generated from lambda_min/max/nbands
 *
 * @see SpectralIO::WriteHDF5 for saving hyperspectral data
 * @see SpectralIO::ReadHDF5 for loading hyperspectral data
 *
 * @author wtflmao
 */

#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// SpectralCube - Hyperspectral data cube (multi-wavelength output)
// ============================================================================
/**
 * @struct SpectralCube
 * @brief 3D hyperspectral data cube with band-major memory layout (BSQ format)
 *
 * Stores multi-wavelength rendering output as a 3D array: [band][y][x].
 * Each band represents a different wavelength channel.
 *
 * Coordinate system:
 * - X: Horizontal axis (0 to width-1)
 * - Y: Vertical axis (0 to height-1)
 * - Band: Spectral axis (0 to nbands-1, maps to wavelength via wavelengths[])
 *
 * Memory layout (C-order / Band Sequential):
 * @code
 * // Band 0: data[0 ... width*height-1]
 * // Band 1: data[width*height ... 2*width*height-1]
 * // ...
 * // Band N: data[N*width*height ... (N+1)*width*height-1]
 * @endcode
 *
 * @note Data stored as f32 (physical radiance: W·sr⁻¹·m⁻²·nm⁻¹)
 * @note Wavelengths auto-generated: λ[b] = lambda_min + b × delta_lambda
 * @note Metadata stored as key-value strings (preserved in HDF5 attributes)
 *
 * @see SpectralIO for HDF5 I/O operations
 */
struct SpectralCube {
    // Spatial dimensions
    u32 width = 0;
    u32 height = 0;

    // Spectral dimension
    u32 nbands = 0;

    // Wavelength range (nm)
    f32 lambda_min = 0.0f;
    f32 lambda_max = 0.0f;
    f32 delta_lambda = 0.0f;

    // Pixel data (C-order: [band][y][x])
    // Always stored as f32
    std::vector<f32> data;

    // Wavelength array (nbands elements, in nm)
    // wavelengths[b] = lambda_min + b * delta_lambda
    std::vector<f32> wavelengths;

    // Generic metadata
    std::unordered_map<std::string, std::string> metadata;

    // ========================================================================
    // Constructors
    // ========================================================================

    SpectralCube() = default;

    SpectralCube(const u32 w, const u32 h, const u32 nb, const f32 lmin, const f32 lmax)
        : width(w), height(h), nbands(nb),
          lambda_min(lmin), lambda_max(lmax) {

        // Compute delta_lambda
        delta_lambda = (lmax - lmin) / static_cast<f32>(nb - 1);

        // Allocate data
        data.resize(w * h * nb, 0.0f);

        // Generate wavelength array
        wavelengths.resize(nb);
        for (u32 b = 0; b < nb; ++b) {
            wavelengths[b] = lmin + static_cast<float>(b) * delta_lambda;
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get pixel value at (x, y, band)
    inline f32& operator()(const u32 x, const u32 y, const u32 b) {
        return data[b * height * width + y * width + x];
    }

    inline const f32& operator()(const u32 x, const u32 y, const u32 b) const {
        return data[b * height * width + y * width + x];
    }

    // Get pointer to entire band (useful for per-band processing)
    inline f32* BandPtr(const u32 b) {
        return &data[b * height * width];
    }

    [[nodiscard]] inline const f32* BandPtr(const u32 b) const {
        return &data[b * height * width];
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    // Total number of pixels per band
    [[nodiscard]] inline u32 PixelsPerBand() const { return width * height; }

    // Total number of elements
    [[nodiscard]] inline u32 TotalElements() const { return width * height * nbands; }

    // Check if cube is valid
    [[nodiscard]] inline bool IsValid() const {
        return width > 0 && height > 0 && nbands > 0 &&
               data.size() == TotalElements() &&
               wavelengths.size() == nbands &&
               lambda_min < lambda_max &&
               delta_lambda > 0.0f;
    }

    // Clear cube data
    void Clear() { std::fill(data.begin(), data.end(), 0.0f); }

    // Get wavelength for band index
    [[nodiscard]] inline f32 GetWavelength(const u32 b) const {
        return wavelengths[b];
    }

    // Find band index closest to given wavelength (nm)
    [[nodiscard]] u32 FindClosestBand(const f32 target_nm) const {
        if (nbands == 0) return 0;

        u32 closest = 0;
        f32 minDist = std::abs(wavelengths[0] - target_nm);

        for (u32 b = 1; b < nbands; ++b) {
            if (const f32 dist = std::abs(wavelengths[b] - target_nm); dist < minDist) {
                minDist = dist;
                closest = b;
            }
        }

        return closest;
    }
};

} // namespace quantiloom
