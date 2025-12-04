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
// SpectralCurveGPU - GPU-side fixed-size spectral curve
// ============================================================================
// Fixed-size representation for efficient GPU upload and query
// Trades flexibility for performance (no dynamic allocation on GPU)
//
// MAX_SPECTRAL_SAMPLES:
// - 64 samples covers most measured curves with 5-10nm resolution
// - For 380-780nm visible range: 5nm spacing = 80 samples (fits in 64 with downsampling)
// - For 3000-12000nm IR range: 150nm spacing = 60 samples
// ============================================================================

static constexpr u32 MAX_SPECTRAL_SAMPLES = 64;

struct SpectralCurveGPU {
    f32 wavelengths[MAX_SPECTRAL_SAMPLES]{};  // Wavelength in nm (must be monotonic increasing)
    f32 values[MAX_SPECTRAL_SAMPLES]{};       // Spectral values (dimensionless or W/sr/m²/nm)
    u32 numSamples;                         // Actual number of valid samples (0 to MAX_SPECTRAL_SAMPLES)
    u32 _padding[3]{};                        // Align to 16 bytes for std430 layout

    // Default constructor: empty curve
    SpectralCurveGPU() : numSamples(0) {
        for (u32 i = 0; i < MAX_SPECTRAL_SAMPLES; ++i) {
            wavelengths[i] = 0.0f;
            values[i] = 0.0f;
        }
        _padding[0] = _padding[1] = _padding[2] = 0;
    }

    // Convert from CPU SpectralCurve (with downsampling if needed)
    static SpectralCurveGPU FromCPU(const SpectralCurve& curve) {
        SpectralCurveGPU gpu;

        if (curve.samples.empty()) {
            gpu.numSamples = 0;
            return gpu;
        }

        // If curve fits in MAX_SPECTRAL_SAMPLES, copy directly
        if (curve.samples.size() <= MAX_SPECTRAL_SAMPLES) {
            gpu.numSamples = static_cast<u32>(curve.samples.size());
            for (u32 i = 0; i < gpu.numSamples; ++i) {
                gpu.wavelengths[i] = curve.samples[i].first;
                gpu.values[i] = curve.samples[i].second;
            }
        } else {
            // Downsample uniformly to fit in MAX_SPECTRAL_SAMPLES
            gpu.numSamples = MAX_SPECTRAL_SAMPLES;
            const f32 lambda_min = curve.samples.front().first;
            const f32 lambda_max = curve.samples.back().first;

            for (u32 i = 0; i < MAX_SPECTRAL_SAMPLES; ++i) {
                const f32 t = static_cast<f32>(i) / static_cast<f32>(MAX_SPECTRAL_SAMPLES - 1);
                const f32 lambda = lambda_min + t * (lambda_max - lambda_min);

                gpu.wavelengths[i] = lambda;
                gpu.values[i] = curve.Evaluate(lambda);
            }
        }

        return gpu;
    }
};

// Verify GPU struct size (should be 64*4 + 64*4 + 4 + 12 = 528 bytes)
static_assert(sizeof(SpectralCurveGPU) == 528, "SpectralCurveGPU size mismatch! Expected 528 bytes");

} // namespace quantiloom
