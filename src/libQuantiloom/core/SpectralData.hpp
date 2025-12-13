#pragma once

#include "core/Types.hpp"
#include <vector>
#include <utility>

// ============================================================================
// Spectral Curve Data Structures
// ============================================================================
// Represents wavelength-dependent material properties for physically-based
// spectral path tracing. Supports both sparse (measured) and dense (computed)
// spectral representations.
//
// DESIGN PHILOSOPHY:
// - CPU side: Flexible representation (std::vector for variable-length curves)
// - GPU side: Fixed-size representation (arrays for fast upload/query)
//
// USAGE:
// - Store measured spectral reflectance/emissivity curves
// - Replace scalar spectralAlbedo with full spectral fidelity
// - Enable quantitative HS-OFF mode with physical validation
// ============================================================================

namespace quantiloom {

// ============================================================================
// SpectralCurve - CPU-side variable-length spectral curve
// ============================================================================
// Stores wavelength-value pairs for arbitrary spectral data
// Supports linear interpolation for continuous wavelength queries
// ============================================================================

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
        for (size_t i = 0; i < samples.size() - 1; ++i) {
            const f32 lambda0 = samples[i].first;

            if (const f32 lambda1 = samples[i + 1].first; lambda_nm >= lambda0 && lambda_nm <= lambda1) {
                const f32 value0 = samples[i].second;
                const f32 value1 = samples[i + 1].second;

                // Linear interpolation
                const f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
                return value0 * (1.0f - t) + value1 * t;
            }
        }

        return 0.0f;  // Should never reach here
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
// Fixed-size representation for efficient GPU upload and O(1) query.
//
// CRITICAL: This struct MUST match the GPU definition in common.hlsli!
//
// UNIFORM SAMPLING DESIGN:
// - Wavelengths are NOT stored explicitly (saves 256 bytes per curve)
// - Wavelength computed as: λ[i] = startWavelength_nm + i × stepSize_nm
// - Enables O(1) lookup instead of O(log N) binary search
// - CPU must resample irregular measured data to uniform grid before upload
//
// EXAMPLE CONFIGURATIONS:
// - Visible spectrum: start=380nm, step=6.35nm, samples=64 → covers 380-786nm
// - UV-Vis-NIR: start=360nm, step=10nm, samples=64 → covers 360-990nm
// - Full IR range: start=3000nm, step=140nm, samples=64 → covers 3000-11960nm
//
// SIZE: 64×4 + 4 + 4 + 4 + 4 = 272 bytes per curve (was 528 bytes)
// ============================================================================

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

} // namespace quantiloom
