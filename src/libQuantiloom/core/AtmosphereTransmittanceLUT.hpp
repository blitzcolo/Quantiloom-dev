/**
 * @file AtmosphereTransmittanceLUT.hpp
 * @brief 3D atmospheric transmittance lookup table for spectral rendering
 *
 * Provides AtmosphereTransmittanceLUT struct for wavelength-dependent atmospheric effects:
 * - Transmittance: τ(λ,h,θ) ∈ [0,1], fraction of light transmitted through atmosphere
 * - Path radiance: L_path(λ,h,θ), atmospheric self-emission (W·sr⁻¹·m⁻²·nm⁻¹)
 *
 * Dimensions:
 * 1. Wavelength (λ): 300-14000nm typical (UV-Vis-SWIR-MWIR-LWIR)
 * 2. Altitude (h): 0-30000m typical (observer height above sea level)
 * 3. Zenith angle (θ): 0-85° typical (angle from vertical)
 *
 * Interpolation:
 * - Trilinear interpolation across all three dimensions
 * - Uniform axes (λ, h): O(1) index computation
 * - Non-uniform axis (θ): O(N) linear search (typically only 5-10 values)
 *
 * Data sources:
 * - MODTRAN: Industry-standard atmospheric radiative transfer code
 * - libRadtran: Open-source UV/Vis/IR atmospheric model
 * - Custom LUT generation scripts (see scripts/atmosphere-qlut-gen/)
 *
 * File format (.qlut):
 * - TOML header (2048 bytes): Metadata (wavelength range, altitude, angles, etc.)
 * - Binary data (float32 arrays): Transmittance and path radiance in C-order
 *
 * @note Use AtmosphereTransmittanceLUTLoader to load .qlut files
 * @note Transmittance includes both absorption and scattering (Beer-Lambert law)
 * @note Path radiance includes atmospheric scattering into line-of-sight
 *
 * @see AtmosphereTransmittanceLUTLoader for .qlut file I/O
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Log.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace quantiloom {
struct AtmosphereTransmittanceLUT;

// ============================================================================
// Uniform Grid Axis
// ============================================================================
/**
 * @struct UniformAxis
 * @brief Uniformly-spaced 1D axis for O(1) index computation
 *
 * Represents a regularly-sampled dimension (wavelength, altitude).
 * Index computation: idx = (value - start) / step
 *
 * Example:
 * @code
 * UniformAxis wavelengthAxis;
 * wavelengthAxis.start = 400.0f;  // nm
 * wavelengthAxis.stop = 800.0f;
 * wavelengthAxis.count = 81;
 * wavelengthAxis.step = (800.0f - 400.0f) / 80.0f;  // 5nm spacing
 *
 * f32 idx = wavelengthAxis.GetFractionalIndex(550.0f);  // Returns 30.0
 * @endcode
 *
 * @note GetFractionalIndex() returns fractional indices for linear interpolation
 * @note Values outside [start, stop] are clamped to boundaries
 */
struct UniformAxis {
    f32 start = 0.0f;    // First value (inclusive)
    f32 stop  = 0.0f;    // Last value (inclusive)
    f32 step  = 1.0f;    // Spacing between samples
    u32 count = 0;       // Number of samples

    // Compute fractional index for interpolation
    // Returns value in [0, count-1] range, clamped at boundaries
    [[nodiscard]] inline f32 GetFractionalIndex(f32 value) const {
        if (count == 0) return 0.0f;
        if (step <= 0.0f) return 0.0f;

        f32 idx = (value - start) / step;
        return std::clamp(idx, 0.0f, static_cast<f32>(count - 1));
    }

    // Get value at integer index
    [[nodiscard]] inline f32 GetValue(u32 index) const {
        return start + static_cast<f32>(index) * step;
    }

    // Check if axis is valid
    [[nodiscard]] inline bool IsValid() const {
        if (count == 0) return false;
        if (step <= 0.0f) return false;
        // Verify stop matches start + (count-1) * step (with tolerance)
        f32 expected_stop = start + static_cast<f32>(count - 1) * step;
        return std::abs(stop - expected_stop) < step * 0.1f;
    }
};

// ============================================================================
// Non-uniform Axis (for zenith angles)
// ============================================================================
// For non-uniformly spaced values like zenith angles [0, 15, 30, 45, 60, 75, 85]
// Requires linear search but typically only 5-10 values
// ============================================================================

struct NonUniformAxis {
    Vector<f32> values;  // Sorted values (ascending)

    // Compute fractional index for interpolation
    // Uses linear search (axis is small)
    [[nodiscard]] inline f32 GetFractionalIndex(f32 value) const {
        if (values.empty()) return 0.0f;

        // Clamp to boundaries
        if (value <= values.front()) return 0.0f;
        if (value >= values.back()) return static_cast<f32>(values.size() - 1);

        // Linear search (small axis, typically 5-10 elements)
        for (usize i = 0; i < values.size() - 1; ++i) {
            if (value >= values[i] && value < values[i + 1]) {
                f32 t = (value - values[i]) / (values[i + 1] - values[i]);
                return static_cast<f32>(i) + t;
            }
        }

        return static_cast<f32>(values.size() - 1);
    }

    [[nodiscard]] inline u32 Count() const {
        return static_cast<u32>(values.size());
    }

    [[nodiscard]] inline bool IsValid() const {
        if (values.empty()) return false;
        // Check monotonically increasing
        for (usize i = 1; i < values.size(); ++i) {
            if (values[i] <= values[i - 1]) return false;
        }
        return true;
    }
};

// ============================================================================
// AtmosphereTransmittanceLUT
// ============================================================================

struct AtmosphereTransmittanceLUT {
    // ========================================================================
    // Metadata
    // ========================================================================
    String name;           // Human-readable name (e.g., "Midlatitude Summer")
    String source;         // Generation source (e.g., "libRadtran 2.0.4")
    String created;        // Creation date (ISO 8601)
    String atmospheric_model;  // Atmospheric model (e.g., "US_Standard_1976")
    i32 ihaze   = 4;       // MODTRAN IHAZE aerosol model (0-6)
    i32 weather = 0;       // Weather condition (0=clear)

    // ========================================================================
    // Axes
    // ========================================================================
    UniformAxis    wavelength;   // Wavelength in nm (uniform sampling)
    UniformAxis    altitude;     // Observer altitude in meters (uniform sampling)
    NonUniformAxis zenith;       // Solar zenith angle in degrees (non-uniform)

    // ========================================================================
    // Data Arrays
    // ========================================================================
    // Layout: C-order (row-major), indexed as [wavelength][altitude][zenith]
    // Total size: wavelength.count * altitude.count * zenith.Count()
    //
    // Index formula:
    //   idx = i_wave * (n_alt * n_zen) + i_alt * n_zen + i_zen
    // ========================================================================

    Vector<f32> transmittance;   // τ(λ,h,θ) ∈ [0, 1]
    Vector<f32> path_radiance;   // L_path(λ,h,θ) in W·sr⁻¹·m⁻²·nm⁻¹

    // ========================================================================
    // Validation
    // ========================================================================

    [[nodiscard]] bool IsValid() const {
        // Check axes
        if (!wavelength.IsValid()) {
            Log::Error("AtmosphereTransmittanceLUT: Invalid wavelength axis");
            return false;
        }
        if (!altitude.IsValid()) {
            Log::Error("AtmosphereTransmittanceLUT: Invalid altitude axis");
            return false;
        }
        if (!zenith.IsValid()) {
            Log::Error("AtmosphereTransmittanceLUT: Invalid zenith axis");
            return false;
        }

        // Check data array sizes
        const usize expected_size = static_cast<usize>(wavelength.count) *
                                    static_cast<usize>(altitude.count) *
                                    static_cast<usize>(zenith.Count());

        if (transmittance.size() != expected_size) {
            Log::Error("AtmosphereTransmittanceLUT: Transmittance array size mismatch "
                       "(expected {}, got {})", expected_size, transmittance.size());
            return false;
        }

        // Path radiance is optional (can be empty for simplified LUTs)
        if (!path_radiance.empty() && path_radiance.size() != expected_size) {
            Log::Error("AtmosphereTransmittanceLUT: Path radiance array size mismatch "
                       "(expected {}, got {})", expected_size, path_radiance.size());
            return false;
        }

        return true;
    }

    // ========================================================================
    // Indexing Helpers
    // ========================================================================

    [[nodiscard]] inline usize GetDataIndex(u32 i_wave, u32 i_alt, u32 i_zen) const {
        const u32 n_alt = altitude.count;
        const u32 n_zen = zenith.Count();
        return static_cast<usize>(i_wave) * (n_alt * n_zen) +
               static_cast<usize>(i_alt) * n_zen +
               static_cast<usize>(i_zen);
    }

    [[nodiscard]] inline usize TotalDataSize() const {
        return static_cast<usize>(wavelength.count) *
               static_cast<usize>(altitude.count) *
               static_cast<usize>(zenith.Count());
    }

    // ========================================================================
    // Trilinear Interpolation Query
    // ========================================================================

    // Query transmittance at arbitrary (wavelength, altitude, zenith) point
    // Uses trilinear interpolation for smooth results
    //
    // Parameters:
    //   wavelength_nm: Wavelength in nanometers
    //   altitude_m: Observer altitude in meters above sea level
    //   zenith_deg: Zenith angle in degrees (0 = looking straight up)
    //
    // Returns:
    //   Interpolated transmittance τ ∈ [0, 1]
    [[nodiscard]] f32 QueryTransmittance(f32 wavelength_nm, f32 altitude_m, f32 zenith_deg) const {
        return TrilinearSample(transmittance, wavelength_nm, altitude_m, zenith_deg);
    }

    // Query path radiance at arbitrary point
    // Returns 0 if path_radiance data is not available
    [[nodiscard]] f32 QueryPathRadiance(f32 wavelength_nm, f32 altitude_m, f32 zenith_deg) const {
        if (path_radiance.empty()) return 0.0f;
        return TrilinearSample(path_radiance, wavelength_nm, altitude_m, zenith_deg);
    }

private:
    // Trilinear interpolation implementation
    [[nodiscard]] f32 TrilinearSample(const Vector<f32>& data,
                                       f32 wavelength_nm,
                                       f32 altitude_m,
                                       f32 zenith_deg) const {
        if (data.empty()) return 0.0f;

        // Get fractional indices
        f32 fw = wavelength.GetFractionalIndex(wavelength_nm);
        f32 fa = altitude.GetFractionalIndex(altitude_m);
        f32 fz = zenith.GetFractionalIndex(zenith_deg);

        // Integer indices and fractions
        u32 iw0 = static_cast<u32>(fw);
        u32 ia0 = static_cast<u32>(fa);
        u32 iz0 = static_cast<u32>(fz);

        u32 iw1 = std::min(iw0 + 1, wavelength.count - 1);
        u32 ia1 = std::min(ia0 + 1, altitude.count - 1);
        u32 iz1 = std::min(iz0 + 1, zenith.Count() - 1);

        f32 tw = fw - static_cast<f32>(iw0);
        f32 ta = fa - static_cast<f32>(ia0);
        f32 tz = fz - static_cast<f32>(iz0);

        // Sample 8 corners of the cube
        f32 c000 = data[GetDataIndex(iw0, ia0, iz0)];
        f32 c001 = data[GetDataIndex(iw0, ia0, iz1)];
        f32 c010 = data[GetDataIndex(iw0, ia1, iz0)];
        f32 c011 = data[GetDataIndex(iw0, ia1, iz1)];
        f32 c100 = data[GetDataIndex(iw1, ia0, iz0)];
        f32 c101 = data[GetDataIndex(iw1, ia0, iz1)];
        f32 c110 = data[GetDataIndex(iw1, ia1, iz0)];
        f32 c111 = data[GetDataIndex(iw1, ia1, iz1)];

        // Trilinear interpolation
        // Interpolate along z (zenith)
        f32 c00 = c000 * (1.0f - tz) + c001 * tz;
        f32 c01 = c010 * (1.0f - tz) + c011 * tz;
        f32 c10 = c100 * (1.0f - tz) + c101 * tz;
        f32 c11 = c110 * (1.0f - tz) + c111 * tz;

        // Interpolate along a (altitude)
        f32 c0 = c00 * (1.0f - ta) + c01 * ta;
        f32 c1 = c10 * (1.0f - ta) + c11 * ta;

        // Interpolate along w (wavelength)
        return c0 * (1.0f - tw) + c1 * tw;
    }
};

// ============================================================================
// Convenience type alias
// ============================================================================
using AtmosphereLUT3D = AtmosphereTransmittanceLUT;

} // namespace quantiloom
