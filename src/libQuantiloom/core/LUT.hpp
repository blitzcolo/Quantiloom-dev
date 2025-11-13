#pragma once

#include "Types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace quantiloom {

// ============================================================================
// AtmosphereLUT - MODTRAN lookup table for LUT-fast mode
// ============================================================================
// This structure holds pre-computed atmospheric data from MODTRAN.
// Used in MS-RT's LUT-fast mode to avoid full volumetric path tracing.
//
// Units:
// - wavelengths: nm
// - solar_irradiance: W/m^2/nm (at top-of-atmosphere)
// - sky_radiance: W/m^2/sr/nm (zenith, for simplicity)
// - transmittance: dimensionless [0, 1]
//
// Future extensions (M4):
// - Multi-angle sky radiance
// - Path radiance for different view geometries
// - Absorption/scattering coefficients for volume rendering
// ============================================================================

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
    inline bool IsValid() const {
        usize n = wavelengths.size();
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
    inline usize Size() const { return wavelengths.size(); }

    // Linear interpolation helper
    // Returns interpolated value at target_nm
    // If target_nm is out of range, clamps to boundary values
    f32 Interpolate(const std::vector<f32>& values, f32 target_nm) const {
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
            usize mid = (left + right) / 2;
            if (wavelengths[mid] < target_nm) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation
        f32 lambda0 = wavelengths[left];
        f32 lambda1 = wavelengths[right];
        f32 t = (target_nm - lambda0) / (lambda1 - lambda0);

        return values[left] * (1.0f - t) + values[right] * t;
    }

    // Get solar irradiance at specific wavelength (nm)
    inline f32 GetSolarIrradiance(f32 lambda_nm) const {
        return Interpolate(solar_irradiance, lambda_nm);
    }

    // Get sky radiance at specific wavelength (nm)
    inline f32 GetSkyRadiance(f32 lambda_nm) const {
        return Interpolate(sky_radiance, lambda_nm);
    }

    // Get transmittance at specific wavelength (nm)
    inline f32 GetTransmittance(f32 lambda_nm) const {
        return Interpolate(transmittance, lambda_nm);
    }
};

} // namespace quantiloom
