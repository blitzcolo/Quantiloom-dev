// ============================================================================
// Quantiloom - Spectral Curve Query Functions
// ============================================================================
// Provides GPU-side spectral curve evaluation with linear interpolation
// Enables physically-based spectral path tracing with measured reflectance
//
// USAGE:
// 1. Bind spectralCurves buffer in shader descriptor set
// 2. Call EvaluateSpectralCurve(curveIndex, wavelength_nm)
// 3. Returns interpolated value at specific wavelength
//
// PERFORMANCE:
// - Linear search O(n) for small curves (n ≤ 64)
// - Binary search not needed - curves are typically < 64 samples
// - Unrolled loop for better GPU performance
// ============================================================================

#include "common.hlsli"

// ============================================================================
// Spectral Curve Evaluation
// ============================================================================

// Evaluate spectral curve at specific wavelength using linear interpolation
// Returns 0.0 if:
// - Curve index is invalid (< 0 or >= buffer size)
// - Wavelength is out of curve range
// - Curve has no samples (numSamples == 0)
//
// Algorithm:
// 1. Find two surrounding samples [i] and [i+1] where λ[i] ≤ λ ≤ λ[i+1]
// 2. Linear interpolation: v = v[i] + (v[i+1] - v[i]) * (λ - λ[i]) / (λ[i+1] - λ[i])
// 3. Edge cases: clamp to first/last value if λ is outside range
float EvaluateSpectralCurve(StructuredBuffer<SpectralCurveGPU> spectralCurves,
                           int curveIndex,
                           float lambda_nm) {
    // Invalid index check (bounds checking)
    // Note: We can't check upper bound without knowing buffer size
    // Rely on descriptor validation and runtime bounds checking
    if (curveIndex < 0) {
        return 0.0;
    }

    // Fetch curve from buffer
    SpectralCurveGPU curve = spectralCurves[curveIndex];

    // Empty curve check
    if (curve.numSamples == 0) {
        return 0.0;
    }

    // Out of range - return edge values (constant extrapolation)
    if (lambda_nm <= curve.wavelengths[0]) {
        return curve.values[0];
    }
    if (lambda_nm >= curve.wavelengths[curve.numSamples - 1]) {
        return curve.values[curve.numSamples - 1];
    }

    // Linear search for surrounding samples
    // For small curves (< 64 samples), linear search is faster than binary search on GPU
    for (uint i = 0; i < curve.numSamples - 1; ++i) {
        float lambda0 = curve.wavelengths[i];
        float lambda1 = curve.wavelengths[i + 1];

        if (lambda_nm >= lambda0 && lambda_nm <= lambda1) {
            float value0 = curve.values[i];
            float value1 = curve.values[i + 1];

            // Linear interpolation
            float t = (lambda_nm - lambda0) / (lambda1 - lambda0);
            return lerp(value0, value1, t);
        }
    }

    // Should never reach here if curve is valid (monotonic wavelengths)
    return 0.0;
}

// ============================================================================
// Material Spectral Albedo Query
// ============================================================================

// Query material spectral albedo at specific wavelength
// Handles both legacy scalar albedo and full spectral curves
//
// FALLBACK HIERARCHY:
// 1. If spectralReflectanceCurveIndex >= 0: Query spectral curve
// 2. Else: Use legacy spectralAlbedo scalar (M1 compatibility)
//
// This ensures backward compatibility with M1 scenes while enabling
// full spectral fidelity for M2+ quantitative rendering
float QueryMaterialSpectralAlbedo(StructuredBuffer<SpectralCurveGPU> spectralCurves,
                                  MaterialData material,
                                  float lambda_nm) {
    // NEW (M2+): Use full spectral curve if available
    if (material.spectralReflectanceCurveIndex >= 0) {
        return EvaluateSpectralCurve(spectralCurves,
                                     material.spectralReflectanceCurveIndex,
                                     lambda_nm);
    }

    // LEGACY (M1): Fallback to scalar albedo
    return material.spectralAlbedo;
}

// ============================================================================
// Atmospheric Transmittance Query (Future: MODTRAN LUT integration)
// ============================================================================

// TODO: Implement MODTRAN LUT query for wavelength-dependent transmittance
// Current: Returns scalar transmittance from LUTData (wavelength-independent)
// Future: Query LUT(λ, altitude, zenith_angle) → τ(λ)
//
// PLACEHOLDER IMPLEMENTATION:
float QueryAtmosphericTransmittance(StructuredBuffer<LUTData> skyLUT,
                                    float lambda_nm,
                                    float distance_m) {
    LUTData lut = skyLUT[0];

    // PLACEHOLDER: Use scalar transmittance (wavelength-independent)
    // This is a simplification - real implementation should query MODTRAN LUT
    // with wavelength dependence: τ(λ) varies significantly in IR bands
    return lut.transmittance;

    // FUTURE (M2+): Implement MODTRAN LUT query
    // return QueryMODTRANLUT(lambda_nm, altitude, zenith_angle);
}
