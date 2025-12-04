#pragma once

#include "core/Types.hpp"
#include <string>
#include <vector>

#include "core/Log.hpp"

// ============================================================================
// MODTRAN LUT Loader Interface
// ============================================================================
// Provides interface for loading MODTRAN atmospheric lookup tables (LUTs)
// These LUTs contain wavelength-dependent atmospheric parameters:
//   - Solar radiance at top of atmosphere: L_sun(λ)
//   - Sky radiance (Rayleigh + Mie scattering): L_sky(λ)
//   - Atmospheric transmittance (vertical path): τ(λ)
//
// FILE FORMAT: HDF5 with following datasets:
//   /wavelengths_nm       - 1D array of wavelengths (nm)
//   /sun_radiance         - Solar spectral radiance (W·sr⁻¹·m⁻²·nm⁻¹)
//   /sky_radiance         - Sky spectral radiance (W·sr⁻¹·m⁻²·nm⁻¹)
//   /transmittance        - Atmospheric transmittance [0, 1]
//   /metadata             - MODTRAN run parameters (altitude, visibility, etc.)
//
// IMPLEMENTATION STATUS:
//   - INTERFACE ONLY - Empty implementation placeholder
//   - User must implement actual HDF5 loading logic
//   - See MODTRAN documentation for LUT generation procedure
//
// USAGE:
//   MODTRANLoader loader;
//   MODTRANLUT lut = loader.Load("path/to/modtran_lut.h5");
//   float sunRadiance = lut.QuerySunRadiance(550.0f);  // Query at 550nm
// ============================================================================

namespace quantiloom {

// ============================================================================
// MODTRAN LUT Data Structure
// ============================================================================
// Contains wavelength-dependent atmospheric parameters from MODTRAN
// Supports linear interpolation for continuous wavelength queries
// ============================================================================

struct MODTRANLUT {
    // Wavelength grid (nm)
    Vector<f32> wavelengths_nm;

    // Spectral radiance arrays (W·sr⁻¹·m⁻²·nm⁻¹)
    Vector<f32> sun_radiance;   // Solar irradiance at top of atmosphere
    Vector<f32> sky_radiance;   // Sky radiance (scattered light)

    // Atmospheric transmittance [0, 1] (vertical path)
    Vector<f32> transmittance;

    // Metadata (MODTRAN run parameters)
    struct Metadata {
        f32 altitude_m = 0.0f;       // Observer altitude (m)
        f32 visibility_km = 23.0f;   // Horizontal visibility (km)
        f32 solar_zenith_deg = 0.0f; // Solar zenith angle (degrees)
        String model = "MODTRAN6";   // MODTRAN version
        String atmosphere = "MLS";   // Atmospheric profile (MLS, TRP, SAW, etc.)
    } metadata;

    // Default constructor: empty LUT
    MODTRANLUT() = default;

    // Check if LUT is valid (non-empty, consistent array sizes)
    bool IsValid() const {
        if (wavelengths_nm.empty()) return false;

        size_t n = wavelengths_nm.size();
        if (sun_radiance.size() != n) return false;
        if (sky_radiance.size() != n) return false;
        if (transmittance.size() != n) return false;

        // Check wavelengths are monotonic increasing
        for (size_t i = 1; i < n; ++i) {
            if (wavelengths_nm[i] <= wavelengths_nm[i - 1]) {
                return false;
            }
        }

        return true;
    }

    // Query sun radiance at specific wavelength (linear interpolation)
    // Returns 0.0 if wavelength is out of range or LUT is empty
    f32 QuerySunRadiance(const f32 lambda_nm) const {
        return InterpolateSpectrum(wavelengths_nm, sun_radiance, lambda_nm);
    }

    // Query sky radiance at specific wavelength (linear interpolation)
    // Returns 0.0 if wavelength is out of range or LUT is empty
    f32 QuerySkyRadiance(const f32 lambda_nm) const {
        return InterpolateSpectrum(wavelengths_nm, sky_radiance, lambda_nm);
    }

    // Query transmittance at specific wavelength (linear interpolation)
    // Returns 1.0 (no attenuation) if wavelength is out of range or LUT is empty
    f32 QueryTransmittance(const f32 lambda_nm) const {
        if (wavelengths_nm.empty()) return 1.0f;
        const f32 result = InterpolateSpectrum(wavelengths_nm, transmittance, lambda_nm);
        return (result > 0.0f) ? result : 1.0f;  // Fallback to no attenuation
    }

    // Get wavelength range
    std::pair<f32, f32> GetWavelengthRange() const {
        if (wavelengths_nm.empty()) return {0.0f, 0.0f};
        return {wavelengths_nm.front(), wavelengths_nm.back()};
    }

private:
    // Helper: Linear interpolation of spectrum
    static f32 InterpolateSpectrum(const Vector<f32>& wavelengths,
                                   const Vector<f32>& values,
                                   const f32 lambda_nm) {
        if (wavelengths.empty()) return 0.0f;

        // Out of range - return edge values
        if (lambda_nm <= wavelengths.front()) return values.front();
        if (lambda_nm >= wavelengths.back()) return values.back();

        // Binary search for surrounding samples (efficient for large LUTs)
        for (size_t i = 0; i < wavelengths.size() - 1; ++i) {
            const f32 lambda0 = wavelengths[i];
            const f32 lambda1 = wavelengths[i + 1];

            if (lambda_nm >= lambda0 && lambda_nm <= lambda1) {
                const f32 value0 = values[i];
                const f32 value1 = values[i + 1];

                // Linear interpolation
                const f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
                return value0 * (1.0f - t) + value1 * t;
            }
        }

        return 0.0f;  // Should never reach here
    }
};

// ============================================================================
// MODTRAN LUT Loader
// ============================================================================
// Loads MODTRAN LUTs from HDF5 files
// This is a PLACEHOLDER implementation - user must fill in HDF5 loading logic
// ============================================================================

class MODTRANLoader {
public:
    // Load MODTRAN LUT from HDF5 file
    //
    // IMPLEMENTATION REQUIRED:
    // - Open HDF5 file
    // - Read datasets: /wavelengths_nm, /sun_radiance, /sky_radiance, /transmittance
    // - Read metadata attributes
    // - Validate data consistency
    //
    // PLACEHOLDER: Returns empty LUT with warning message
    static MODTRANLUT Load(const String& filepath) {
        MODTRANLUT lut;

        // ====================================================================
        // TODO: IMPLEMENT HDF5 LOADING LOGIC HERE
        // ====================================================================
        // Example pseudo-code:
        //
        // 1. Open HDF5 file:
        //    H5::H5File file(filepath, H5F_ACC_RDONLY);
        //
        // 2. Read wavelength dataset:
        //    H5::DataSet ds_wavelengths = file.openDataSet("/wavelengths_nm");
        //    ds_wavelengths.read(lut.wavelengths_nm.data(), ...);
        //
        // 3. Read radiance datasets:
        //    H5::DataSet ds_sun = file.openDataSet("/sun_radiance");
        //    ds_sun.read(lut.sun_radiance.data(), ...);
        //    ... (similarly for sky_radiance, transmittance)
        //
        // 4. Read metadata attributes:
        //    H5::Group metadata = file.openGroup("/metadata");
        //    metadata.openAttribute("altitude_m").read(..., &lut.metadata.altitude_m);
        //    ... (similarly for other metadata fields)
        //
        // 5. Validate LUT:
        //    if (!lut.IsValid()) {
        //        throw std::runtime_error("Invalid MODTRAN LUT data");
        //    }
        //
        // ====================================================================

        // PLACEHOLDER WARNING
        QL_LOG_WARN("MODTRANLoader::Load() is a PLACEHOLDER - no implementation yet!");
        QL_LOG_WARN("  Requested file: {}", filepath);
        QL_LOG_WARN("  Returning EMPTY LUT. User must implement HDF5 loading logic.");
        QL_LOG_WARN("  See src/libQuantiloom/io/MODTRANLoader.hpp for interface details.");

        return lut;  // Returns empty LUT
    }

    // Generate dummy LUT for testing (placeholder)
    // Creates a simple LUT with constant radiance values
    // NOT physically accurate - for M1 testing only
    static MODTRANLUT CreateDummyLUT() {
        MODTRANLUT lut;

        // Simple visible spectrum (380-780nm, 5nm spacing)
        for (f32 lambda = 380.0f; lambda <= 780.0f; lambda += 5.0f) {
            lut.wavelengths_nm.push_back(lambda);
        }

        const size_t n = lut.wavelengths_nm.size();

        // Dummy sun radiance (constant 1000 W·sr⁻¹·m⁻²·nm⁻¹)
        lut.sun_radiance.resize(n, 1000.0f);

        // Dummy sky radiance (constant 100 W·sr⁻¹·m⁻²·nm⁻¹)
        lut.sky_radiance.resize(n, 100.0f);

        // Dummy transmittance (constant 0.8 = 20% atmospheric absorption)
        lut.transmittance.resize(n, 0.8f);

        // Dummy metadata
        lut.metadata.altitude_m = 0.0f;
        lut.metadata.visibility_km = 23.0f;
        lut.metadata.solar_zenith_deg = 30.0f;
        lut.metadata.model = "DUMMY";
        lut.metadata.atmosphere = "MLS";

        return lut;
    }
};

} // namespace quantiloom
