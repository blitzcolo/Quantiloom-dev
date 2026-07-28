/**
 * @file SpectralData.hpp
 * @brief Wavelength-dependent material property data structures for physically-based spectral rendering
 *
 * Provides CPU/GPU data structures for:
 * - SpectralCurve: Variable-length spectral reflectance/emissivity curves (CPU-side)
 * - SpectralCurveGPU: Fixed-size uniform-sampled curves for GPU (64 samples max)
 * - ComplexRefractiveIndex: n,k curves for physical Fresnel calculations (metals, dielectrics)
 * - ComplexRefractiveIndexGPU: Fixed-size n,k data for GPU
 * - SolarSpectralLUT: Sun and sky irradiance curves (ASTM G-173, libRadtran data)
 *
 * Design philosophy:
 * - CPU: Flexible std::vector-based, arbitrary-length curves
 * - GPU: Fixed-size arrays for fast upload and O(1) query
 * - Uniform sampling: Wavelengths computed as λ[i] = start + i * step (no storage overhead)
 * - Resampling: CPU irregular data → GPU uniform grid (linear interpolation)
 *
 * GPU structs MUST match shader definitions in common.hlsli exactly (verified by static_assert).
 * All GPU structs use std430 layout for efficient SSBO binding.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include <vector>
#include <utility>

// ============================================================================
// Spectral Curve Data Structures
// ============================================================================
/**
 * @defgroup SpectralData Spectral Data Structures
 * @brief Wavelength-dependent material properties for physically-based spectral path tracing
 *
 * Supports both sparse (measured) and dense (computed) spectral representations.
 * Enables quantitative hyperspectral rendering with full wavelength fidelity.
 */

namespace quantiloom {

// ============================================================================
// SpectralCurve - CPU-side variable-length spectral curve
// ============================================================================
/**
 * @struct SpectralCurve
 * @brief CPU-side variable-length spectral reflectance/emissivity curve
 *
 * Stores wavelength-value pairs for arbitrary spectral data (reflectance, emissivity, transmittance).
 * Supports linear interpolation for continuous wavelength queries.
 *
 * Typical uses:
 * - Measured spectral reflectance from spectrometers (e.g., ASD FieldSpec, USB4000)
 * - Emissivity curves from material databases (e.g., ASTER, ECOSTRESS)
 * - Complex refractive index data (n,k) from RefractiveIndex.INFO
 *
 * Data source examples:
 * - USGS spectral library: 400-2500nm, 5nm spacing
 * - RefractiveIndex.INFO: Arbitrary wavelength sampling
 * - Custom CSV files: User-defined sampling
 *
 * @note Wavelengths must be monotonically increasing
 * @note Evaluate() performs linear interpolation (O(N) search)
 * @note For GPU use, convert to SpectralCurveGPU for O(1) query
 *
 * @see SpectralCurveGPU for fixed-size GPU representation
 * @see SpectralIO::LoadSpectralCurveCSV() for loading from CSV files
 */
struct SpectralCurve {
    Vector<std::pair<f32, f32>> samples;  // (wavelength_nm, value)

    // Default constructor: empty curve
    SpectralCurve() = default;

    // Construct from wavelength and value arrays
    SpectralCurve(const Vector<f32>& wavelengths, const Vector<f32>& values) {
        if (wavelengths.size() != values.size()) {
            // Log warning, but don't throw - just create empty curve
            return;
        }
        samples.reserve(wavelengths.size());
        for (size_t i = 0; i < wavelengths.size(); ++i) {
            samples.emplace_back(wavelengths[i], values[i]);
        }
    }

    // Evaluate curve at specific wavelength using linear interpolation
    // Returns 0.0 if wavelength is out of range or curve is empty
    [[nodiscard]] f32 Evaluate(const f32 lambda_nm) const {
        if (samples.empty()) return 0.0f;

        // Out of range - return edge values
        if (lambda_nm <= samples.front().first) return samples.front().second;
        if (lambda_nm >= samples.back().first) return samples.back().second;

        // Binary search for surrounding samples
        usize left = 0;
        usize right = samples.size() - 1;
        while (right - left > 1) {
            usize mid = (left + right) / 2;
            if (samples[mid].first < lambda_nm) {
                left = mid;
            } else {
                right = mid;
            }
        }

        const f32 lambda0 = samples[left].first;
        const f32 lambda1 = samples[right].first;
        const f32 value0 = samples[left].second;
        const f32 value1 = samples[right].second;

        // Linear interpolation
        const f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
        return value0 * (1.0f - t) + value1 * t;
    }

    // Check if curve is valid (non-empty, monotonic wavelengths)
    [[nodiscard]] bool IsValid() const {
        if (samples.empty()) return false;

        for (size_t i = 1; i < samples.size(); ++i) {
            if (samples[i].first <= samples[i - 1].first) {
                return false;  // Wavelengths must be strictly increasing
            }
        }
        return true;
    }

    // Get wavelength range
    [[nodiscard]] std::pair<f32, f32> GetWavelengthRange() const {
        if (samples.empty()) return {0.0f, 0.0f};
        return {samples.front().first, samples.back().first};
    }
};

// ============================================================================
// SpectralCurveGPU - GPU-side fixed-size spectral curve (UNIFORM SAMPLING)
// ============================================================================
/**
 * @struct SpectralCurveGPU
 * @brief GPU-side fixed-size spectral curve with uniform wavelength sampling
 *
 * Fixed-size (64 samples max) representation for efficient GPU upload and O(1) query.
 * Uses uniform wavelength grid: λ[i] = startWavelength_nm + i × stepSize_nm
 *
 * CRITICAL DESIGN DECISIONS:
 * - Wavelengths NOT stored explicitly (saves 256 bytes per curve)
 * - O(1) lookup instead of O(log N) binary search
 * - CPU must resample irregular measured data before upload (see FromCPU())
 * - GPU shader computes wavelength on-the-fly (no memory reads)
 *
 * Example configurations:
 * @code
 * // Visible spectrum: 380-786nm (64 samples)
 * gpu.startWavelength_nm = 380.0f;
 * gpu.stepSize_nm = 6.35f;
 * gpu.numSamples = 64;
 *
 * // Full IR range: 3000-11960nm (64 samples)
 * gpu.startWavelength_nm = 3000.0f;
 * gpu.stepSize_nm = 140.0f;
 * gpu.numSamples = 64;
 * @endcode
 *
 * @note SIZE: 64×4 + 4 + 4 + 4 + 4 = 272 bytes per curve
 * @note MUST match GPU SpectralCurveGPU in common.hlsli (verified by static_assert)
 * @note Use FromCPU() to convert variable-length SpectralCurve to uniform grid
 *
 * @see SpectralCurve for CPU-side representation
 * @see common.hlsli SampleSpectralCurve() for GPU-side evaluation
 */

 static constexpr u32 MAX_SPECTRAL_SAMPLES = 64;
struct SpectralCurveGPU {
    f32 values[MAX_SPECTRAL_SAMPLES]{};  // Spectral values at uniform wavelength grid
    f32 startWavelength_nm = 0.0f;       // First wavelength in grid (nm)
    f32 stepSize_nm = 0.0f;              // Wavelength step size (nm)
    u32 numSamples = 0;                  // Number of valid samples (0 to MAX_SPECTRAL_SAMPLES)
    u32 _padding = 0;                    // Padding for 16-byte alignment (std430)

    // Default constructor: empty curve
    SpectralCurveGPU() = default;

    // Get wavelength at sample index (O(1) computation)
    [[nodiscard]] f32 GetWavelength(u32 index) const {
        return startWavelength_nm + static_cast<f32>(index) * stepSize_nm;
    }

    // Get wavelength range
    [[nodiscard]] std::pair<f32, f32> GetWavelengthRange() const {
        if (numSamples == 0) return {0.0f, 0.0f};
        return {startWavelength_nm, GetWavelength(numSamples - 1)};
    }

    // Evaluate curve at specific wavelength using O(1) direct indexing
    // Matches GPU-side SampleSpectralCurve() in common.hlsli
    [[nodiscard]] f32 Evaluate(f32 lambda_nm) const {
        if (numSamples == 0 || stepSize_nm <= 0.0f) return 0.0f;

        // Compute fractional index: (λ - λ₀) / Δλ
        const f32 index_f = (lambda_nm - startWavelength_nm) / stepSize_nm;

        // Clamp to valid range
        if (index_f < 0.0f) return values[0];
        if (index_f >= static_cast<f32>(numSamples - 1)) return values[numSamples - 1];

        // Linear interpolation between adjacent samples
        const u32 index0 = static_cast<u32>(index_f);
        const u32 index1 = index0 + 1;
        const f32 t = index_f - static_cast<f32>(index0);

        return values[index0] * (1.0f - t) + values[index1] * t;
    }

    // ========================================================================
    // Convert from CPU SpectralCurve (resample to uniform grid)
    // ========================================================================
    // IMPORTANT: This resamples arbitrary non-uniform measured data to a
    // uniform wavelength grid for efficient GPU query.
    //
    // Parameters:
    //   curve: Source SpectralCurve with arbitrary wavelength samples
    //   targetSamples: Number of uniform samples (default: MAX_SPECTRAL_SAMPLES)
    //
    // The resulting uniform grid spans the full wavelength range of the input.
    // ========================================================================
    static SpectralCurveGPU FromCPU(const SpectralCurve& curve,
                                     u32 targetSamples = MAX_SPECTRAL_SAMPLES) {
        SpectralCurveGPU gpu;

        if (curve.samples.empty()) {
            return gpu;  // Empty curve
        }

        // Clamp target samples to valid range
        targetSamples = std::min(targetSamples, MAX_SPECTRAL_SAMPLES);
        if (targetSamples < 2) targetSamples = 2;

        // Compute uniform sampling parameters from source curve range
        const f32 lambda_min = curve.samples.front().first;
        const f32 lambda_max = curve.samples.back().first;

        gpu.startWavelength_nm = lambda_min;
        gpu.stepSize_nm = (lambda_max - lambda_min) / static_cast<f32>(targetSamples - 1);
        gpu.numSamples = targetSamples;

        // Resample to uniform grid using source curve's interpolation
        for (u32 i = 0; i < targetSamples; ++i) {
            const f32 lambda = lambda_min + static_cast<f32>(i) * gpu.stepSize_nm;
            gpu.values[i] = curve.Evaluate(lambda);
        }

        return gpu;
    }

    // ========================================================================
    // Create uniform spectral curve directly (for procedural generation)
    // ========================================================================
    static SpectralCurveGPU CreateUniform(f32 startWavelength, f32 endWavelength,
                                           u32 numSamples, f32 constantValue) {
        SpectralCurveGPU gpu;

        numSamples = std::min(numSamples, MAX_SPECTRAL_SAMPLES);
        if (numSamples < 2) return gpu;

        gpu.startWavelength_nm = startWavelength;
        gpu.stepSize_nm = (endWavelength - startWavelength) / static_cast<f32>(numSamples - 1);
        gpu.numSamples = numSamples;

        for (u32 i = 0; i < numSamples; ++i) {
            gpu.values[i] = constantValue;
        }

        return gpu;
    }
};

// Verify GPU struct size: 64×4 + 4 + 4 + 4 + 4 = 272 bytes
// CRITICAL: Must match GPU-side SpectralCurveGPU in common.hlsli!
static_assert(sizeof(SpectralCurveGPU) == 272, "SpectralCurveGPU size mismatch! Expected 272 bytes (must match GPU)");

// ============================================================================
// ComplexRefractiveIndex - CPU-side complex refractive index (n, k)
// ============================================================================
// Stores wavelength-dependent complex refractive index for Fresnel calculations.
// Data source: RefractiveIndex.INFO database (n, k tabulated data)
//
// PHYSICS:
// - N = n + ik where n = refractive index, k = extinction coefficient
// - n determines phase velocity: v = c/n
// - k determines absorption: intensity decays as exp(-4πkd/λ)
//
// USAGE:
// - Metal surfaces: high k values (gold, silver, copper, aluminum)
// - Dielectrics: k ≈ 0 in transparent regions (glass, water)
// - Semiconductors: varies with wavelength (silicon, germanium)
// ============================================================================

struct ComplexRefractiveIndex {
    Vector<f32> wavelengths_nm;  // Wavelength samples (nm)
    Vector<f32> n;               // Real part (refractive index)
    Vector<f32> k;               // Imaginary part (extinction coefficient)

    // Default constructor: empty data
    ComplexRefractiveIndex() = default;

    // Evaluate n and k at specific wavelength using linear interpolation
    [[nodiscard]] std::pair<f32, f32> Evaluate(f32 lambda_nm) const {
        if (wavelengths_nm.empty()) return {1.0f, 0.0f};  // Default: air

        // Out of range - return edge values
        if (lambda_nm <= wavelengths_nm.front()) {
            return {n.front(), k.front()};
        }
        if (lambda_nm >= wavelengths_nm.back()) {
            return {n.back(), k.back()};
        }

        // Linear search and interpolation
        for (size_t i = 0; i < wavelengths_nm.size() - 1; ++i) {
            const f32 lambda0 = wavelengths_nm[i];
            const f32 lambda1 = wavelengths_nm[i + 1];

            if (lambda_nm >= lambda0 && lambda_nm <= lambda1) {
                const f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
                const f32 n_val = n[i] * (1.0f - t) + n[i + 1] * t;
                const f32 k_val = k[i] * (1.0f - t) + k[i + 1] * t;
                return {n_val, k_val};
            }
        }

        return {1.0f, 0.0f};  // Should never reach here
    }

    // Get wavelength range
    [[nodiscard]] std::pair<f32, f32> GetWavelengthRange() const {
        if (wavelengths_nm.empty()) return {0.0f, 0.0f};
        return {wavelengths_nm.front(), wavelengths_nm.back()};
    }

    // Check if valid
    [[nodiscard]] bool IsValid() const {
        if (wavelengths_nm.empty()) return false;
        if (wavelengths_nm.size() != n.size() || wavelengths_nm.size() != k.size()) return false;
        return true;
    }

    // Calculate Fresnel reflectance at normal incidence: R = |(n-1+ik)/(n+1+ik)|²
    // This is the specular reflectance F0 for PBR rendering
    [[nodiscard]] f32 FresnelR0(f32 lambda_nm) const {
        auto [n_val, k_val] = Evaluate(lambda_nm);
        // R = [(n-1)² + k²] / [(n+1)² + k²]
        const f32 numerator = (n_val - 1.0f) * (n_val - 1.0f) + k_val * k_val;
        const f32 denominator = (n_val + 1.0f) * (n_val + 1.0f) + k_val * k_val;
        return numerator / denominator;
    }
};

// ============================================================================
// ComplexRefractiveIndexGPU - GPU-side fixed-size complex refractive index
// ============================================================================
// Uniform sampling for O(1) GPU query, stores both n and k curves.
//
// SIZE: 64×4 (n) + 64×4 (k) + 4 + 4 + 4 + 4 = 528 bytes per curve
// ============================================================================

struct ComplexRefractiveIndexGPU {
    f32 n[MAX_SPECTRAL_SAMPLES]{};       // Refractive index at uniform grid
    f32 k[MAX_SPECTRAL_SAMPLES]{};       // Extinction coefficient at uniform grid
    f32 startWavelength_nm = 0.0f;       // First wavelength (nm)
    f32 stepSize_nm = 0.0f;              // Step size (nm)
    u32 numSamples = 0;                  // Valid sample count
    u32 _padding = 0;                    // 16-byte alignment

    // Default constructor
    ComplexRefractiveIndexGPU() = default;

    // Get wavelength at index
    [[nodiscard]] f32 GetWavelength(u32 index) const {
        return startWavelength_nm + static_cast<f32>(index) * stepSize_nm;
    }

    // Evaluate n,k at wavelength (O(1) with interpolation)
    [[nodiscard]] std::pair<f32, f32> Evaluate(f32 lambda_nm) const {
        if (numSamples == 0 || stepSize_nm <= 0.0f) return {1.0f, 0.0f};

        const f32 index_f = (lambda_nm - startWavelength_nm) / stepSize_nm;

        if (index_f < 0.0f) return {n[0], k[0]};
        if (index_f >= static_cast<f32>(numSamples - 1)) {
            return {n[numSamples - 1], k[numSamples - 1]};
        }

        const u32 i0 = static_cast<u32>(index_f);
        const u32 i1 = i0 + 1;
        const f32 t = index_f - static_cast<f32>(i0);

        return {
            n[i0] * (1.0f - t) + n[i1] * t,
            k[i0] * (1.0f - t) + k[i1] * t
        };
    }

    // Calculate Fresnel R0 at wavelength
    [[nodiscard]] f32 FresnelR0(f32 lambda_nm) const {
        auto [n_val, k_val] = Evaluate(lambda_nm);
        const f32 numerator = (n_val - 1.0f) * (n_val - 1.0f) + k_val * k_val;
        const f32 denominator = (n_val + 1.0f) * (n_val + 1.0f) + k_val * k_val;
        return numerator / denominator;
    }

    // Convert from CPU ComplexRefractiveIndex
    static ComplexRefractiveIndexGPU FromCPU(const ComplexRefractiveIndex& cri,
                                              u32 targetSamples = MAX_SPECTRAL_SAMPLES) {
        ComplexRefractiveIndexGPU gpu;

        if (cri.wavelengths_nm.empty()) return gpu;

        targetSamples = std::min(targetSamples, MAX_SPECTRAL_SAMPLES);
        if (targetSamples < 2) targetSamples = 2;

        const f32 lambda_min = cri.wavelengths_nm.front();
        const f32 lambda_max = cri.wavelengths_nm.back();

        gpu.startWavelength_nm = lambda_min;
        gpu.stepSize_nm = (lambda_max - lambda_min) / static_cast<f32>(targetSamples - 1);
        gpu.numSamples = targetSamples;

        for (u32 i = 0; i < targetSamples; ++i) {
            const f32 lambda = lambda_min + static_cast<f32>(i) * gpu.stepSize_nm;
            auto [n_val, k_val] = cri.Evaluate(lambda);
            gpu.n[i] = n_val;
            gpu.k[i] = k_val;
        }

        return gpu;
    }
};

// Verify GPU struct size: 64×4 + 64×4 + 4 + 4 + 4 + 4 = 528 bytes
static_assert(sizeof(ComplexRefractiveIndexGPU) == 528, "ComplexRefractiveIndexGPU size mismatch!");

// ============================================================================
// SolarSpectralLUT - GPU-side solar illumination spectral curves
// ============================================================================
// Stores full spectral irradiance curves for sun and sky illumination.
// Enables physically-accurate spectral rendering with ASTM G-173 or libRadtran data.
//
// DATA SOURCES:
// - ASTM G-173-03 Reference Air Mass 1.5 Spectra (terrestrial solar irradiance)
//   - Direct sun: Column 4 (Direct+circumsolar) - W·m⁻²·nm⁻¹
//   - Diffuse sky: Global - Direct - W·m⁻²·nm⁻¹
// - libRadtran: High-accuracy atmospheric radiative transfer (for custom atmospheres)
//
// WAVELENGTH RANGE:
// - ASTM G-173: 280-4000nm (covers UV through near-IR)
// - Typical visible rendering: 380-780nm
// - Thermal IR rendering: 3000-14000nm (requires separate data)
//
// USAGE:
// - Upload once at scene initialization (static solar conditions)
// - Query in shader: SampleSpectralCurve(sunIrradiance, wavelength_nm)
// - Convert irradiance to radiance using sun solid angle: L = E / Ω_sun
//   where Ω_sun ≈ 6.8e-5 sr (angular diameter ~0.53°)
//
// SIZE: 272 + 272 = 544 bytes
// ============================================================================

struct SolarSpectralLUT {
    SpectralCurveGPU sunIrradiance;   // Direct sun spectral irradiance (W·m⁻²·nm⁻¹)
    SpectralCurveGPU skyIrradiance;   // Diffuse sky spectral irradiance (W·m⁻²·nm⁻¹)

    // Default constructor: empty curves
    SolarSpectralLUT() = default;

    // Construct from CPU-side SpectralCurves (resamples to uniform grid)
    static SolarSpectralLUT FromCPU(const SpectralCurve& sunCurve,
                                     const SpectralCurve& skyCurve,
                                     u32 targetSamples = MAX_SPECTRAL_SAMPLES) {
        SolarSpectralLUT lut;
        lut.sunIrradiance = SpectralCurveGPU::FromCPU(sunCurve, targetSamples);
        lut.skyIrradiance = SpectralCurveGPU::FromCPU(skyCurve, targetSamples);
        return lut;
    }

    // Check if valid (both curves have data)
    [[nodiscard]] bool IsValid() const {
        return sunIrradiance.numSamples > 0 && skyIrradiance.numSamples > 0;
    }

    // Get wavelength range (intersection of sun and sky ranges)
    [[nodiscard]] std::pair<f32, f32> GetWavelengthRange() const {
        auto [sunMin, sunMax] = sunIrradiance.GetWavelengthRange();
        auto [skyMin, skyMax] = skyIrradiance.GetWavelengthRange();
        return {std::max(sunMin, skyMin), std::min(sunMax, skyMax)};
    }
};

// Verify SolarSpectralLUT size: 272 + 272 = 544 bytes
static_assert(sizeof(SolarSpectralLUT) == 544, "SolarSpectralLUT size mismatch! Expected 544 bytes");

} // namespace quantiloom
