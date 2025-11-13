#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// SpectralCube - Hyperspectral data cube (HS-OFF output)
// ============================================================================
// Memory layout: C-order (band-major)
//   data[b * height * width + y * width + x]
//
// This layout is chosen for:
// 1. Per-band rendering: each band is contiguous in memory
// 2. HDF5 compatibility: HDF5 datasets use C-order by default
// 3. MODTRAN comparison: most atmospheric models output band-major data
//
// Alternative layouts (e.g., BIP/BSQ/BIL) can be handled via HDF5 chunking.
// ============================================================================

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

    SpectralCube(u32 w, u32 h, u32 nb, f32 lmin, f32 lmax)
        : width(w), height(h), nbands(nb),
          lambda_min(lmin), lambda_max(lmax) {

        // Compute delta_lambda
        delta_lambda = (lmax - lmin) / static_cast<f32>(nb - 1);

        // Allocate data
        data.resize(w * h * nb, 0.0f);

        // Generate wavelength array
        wavelengths.resize(nb);
        for (u32 b = 0; b < nb; ++b) {
            wavelengths[b] = lmin + b * delta_lambda;
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    // Get pixel value at (x, y, band)
    inline f32& operator()(u32 x, u32 y, u32 b) {
        return data[b * height * width + y * width + x];
    }

    inline const f32& operator()(u32 x, u32 y, u32 b) const {
        return data[b * height * width + y * width + x];
    }

    // Get pointer to entire band (useful for per-band processing)
    inline f32* BandPtr(u32 b) {
        return &data[b * height * width];
    }

    inline const f32* BandPtr(u32 b) const {
        return &data[b * height * width];
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    // Total number of pixels per band
    inline u32 PixelsPerBand() const { return width * height; }

    // Total number of elements
    inline u32 TotalElements() const { return width * height * nbands; }

    // Check if cube is valid
    inline bool IsValid() const {
        return width > 0 && height > 0 && nbands > 0 &&
               data.size() == TotalElements() &&
               wavelengths.size() == nbands &&
               lambda_min < lambda_max &&
               delta_lambda > 0.0f;
    }

    // Clear cube data
    void Clear() { std::fill(data.begin(), data.end(), 0.0f); }

    // Get wavelength for band index
    inline f32 GetWavelength(u32 b) const {
        return wavelengths[b];
    }

    // Find band index closest to given wavelength (nm)
    u32 FindClosestBand(f32 target_nm) const {
        if (nbands == 0) return 0;

        u32 closest = 0;
        f32 minDist = std::abs(wavelengths[0] - target_nm);

        for (u32 b = 1; b < nbands; ++b) {
            f32 dist = std::abs(wavelengths[b] - target_nm);
            if (dist < minDist) {
                minDist = dist;
                closest = b;
            }
        }

        return closest;
    }
};

} // namespace quantiloom
