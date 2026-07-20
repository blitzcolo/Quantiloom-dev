/**
 * @file LUT.hpp
 * @brief MODTRAN atmospheric lookup table for LUT-fast rendering mode
 *
 * Provides AtmosphereLUT struct for pre-computed atmospheric data:
 * - Solar irradiance at top-of-atmosphere (W·m⁻²·nm⁻¹)
 * - Sky radiance at zenith (W·m⁻²·sr⁻¹·nm⁻¹)
 * - Direct solar transmittance τ(λ) = exp(-optical_depth)
 *
 * Used in LUT-fast mode to avoid full volumetric atmospheric path tracing.
 * Data sourced from MODTRAN, libRadtran, or other atmospheric radiative transfer codes.
 *
 * Wavelength range:
 * - Typical: 300-2500nm (UV-Vis-NIR-SWIR)
 * - Sampling: 5-10nm intervals
 *
 * Query interface:
 * - Linear interpolation for arbitrary wavelengths
 * - GetWavelengthRange() for supported spectral domain
 *
 * @note Transmittance is direct solar only (no diffuse/path radiance)
 * @note Sky radiance assumes zenith view (no angular dependence)
 * @note For multi-angle data, use AtmosphereTransmittanceLUT instead
 *
 * @see AtmosphereTransmittanceLUT for 3D (λ,h,θ) lookup tables
 * @see LUTLoader for TOML file I/O
 *
 * @author blitzcolo
 */

#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// AtmosphereLUT - MODTRAN lookup table for LUT-fast mode
// ============================================================================
/**
 * @struct AtmosphereLUT
 * @brief 1D atmospheric lookup table with wavelength-dependent solar/sky data
 *
 * Stores pre-computed atmospheric illumination for a specific viewing geometry:
 * - Solar irradiance: Direct + circumsolar radiation at top-of-atmosphere
 * - Sky radiance: Diffuse skylight (zenith view)
 * - Transmittance: Atmospheric attenuation factor (Beer-Lambert law)
 *
 * Data generation:
 * - MODTRAN: tape5 input → tape7 output (extract relevant channels)
 * - libRadtran: uvspec input → wavelength/irradiance/radiance columns
 *
 * Metadata examples:
 * - "solar_zenith_deg": "30" (sun angle)
 * - "visibility_km": "23" (atmospheric clarity)
 * - "model": "US_Standard" (atmospheric profile)
 *
 * @note All arrays must have same length (wavelengths.size())
 * @note Wavelengths must be monotonically increasing
 * @note Query methods perform linear interpolation
 */
struct AtmosphereLUT {
    // Wavelength axis (nm), must be monotonically increasing
    std::vector<f32> wavelengths;

    // Solar irradiance at top-of-atmosphere (W/m^2/nm)
    // Same length as wavelengths
    std::vector<f32> solar_irradiance;

    // Sky radiance at zenith (W/m^2/sr/nm)
    // Same length as wavelengths
    std::vector<f32> sky_radiance;

    // Direct solar transmittance (atmosphere only, no clouds)
    // Same length as wavelengths
    // transmittance[i] = exp(-tau[i]) where tau is optical depth
    std::vector<f32> transmittance;

    // Metadata (optional)
    // e.g., {"solar_zenith_deg": "30", "visibility_km": "23", "model": "US_Standard"}
    std::unordered_map<std::string, std::string> metadata;

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if LUT is valid
    [[nodiscard]] inline bool IsValid() const {
        const usize n = wavelengths.size();
        if (n == 0) return false;

        // All arrays must have same length
        if (solar_irradiance.size() != n ||
            sky_radiance.size() != n ||
            transmittance.size() != n) {
            return false;
        }

        // Wavelengths must be monotonically increasing
        for (usize i = 1; i < n; ++i) {
            if (wavelengths[i] <= wavelengths[i - 1]) {
                return false;
            }
        }

        return true;
    }

    // Get number of wavelength samples
    [[nodiscard]] inline usize Size() const { return wavelengths.size(); }

    // Linear interpolation helper
    // Returns interpolated value at target_nm
    // If target_nm is out of range, clamps to boundary values
    [[nodiscard]] f32 Interpolate(const std::vector<f32>& values, const f32 target_nm) const {
        if (wavelengths.empty()) return 0.0f;

        // Clamp to boundaries
        if (target_nm <= wavelengths.front()) {
            return values.front();
        }
        if (target_nm >= wavelengths.back()) {
            return values.back();
        }

        // Binary search for surrounding wavelengths
        usize left = 0;
        usize right = wavelengths.size() - 1;

        while (right - left > 1) {
            if (const usize mid = (left + right) / 2; wavelengths[mid] < target_nm) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation
        const f32 lambda0 = wavelengths[left];
        const f32 lambda1 = wavelengths[right];
        const f32 t = (target_nm - lambda0) / (lambda1 - lambda0);

        return values[left] * (1.0f - t) + values[right] * t;
    }

    // Get solar irradiance at specific wavelength (nm)
    [[nodiscard]] inline f32 GetSolarIrradiance(const f32 lambda_nm) const {
        return Interpolate(solar_irradiance, lambda_nm);
    }

    // Get sky radiance at specific wavelength (nm)
    [[nodiscard]] inline f32 GetSkyRadiance(const f32 lambda_nm) const {
        return Interpolate(sky_radiance, lambda_nm);
    }

    // Get transmittance at specific wavelength (nm)
    [[nodiscard]] inline f32 GetTransmittance(const f32 lambda_nm) const {
        return Interpolate(transmittance, lambda_nm);
    }
};

} // namespace quantiloom
