#include "Material.hpp"
#include "core/Log.hpp"
#include <algorithm>
#include <cmath>

namespace quantiloom {

// ============================================================================
// Helper: Linear interpolation for spectral curves
// ============================================================================
// Generic interpolation function, reused by all IR property getters
// Returns 0.0 if curve is empty or wavelength out of range
// ============================================================================

static f32 InterpolateSpectralCurve(
    const Vector<std::pair<f32, f32>>& curve,
    f32 lambda_nm,
    f32 fallback = 0.0f)
{
    if (curve.empty()) {
        return fallback;
    }

    // Clamp to boundaries
    if (lambda_nm <= curve.front().first) {
        return curve.front().second;
    }
    if (lambda_nm >= curve.back().first) {
        return curve.back().second;
    }

    // Binary search for surrounding wavelengths
    usize left = 0;
    usize right = curve.size() - 1;

    while (right - left > 1) {
        usize mid = (left + right) / 2;
        if (curve[mid].first < lambda_nm) {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Linear interpolation
    f32 lambda0 = curve[left].first;
    f32 lambda1 = curve[right].first;
    f32 value0 = curve[left].second;
    f32 value1 = curve[right].second;

    f32 t = (lambda_nm - lambda0) / (lambda1 - lambda0);
    return value0 * (1.0f - t) + value1 * t;
}

// ============================================================================
// IR Property Getters (implementations)
// ============================================================================

f32 Material::GetIREmissivity(f32 lambda_nm) const {
    return InterpolateSpectralCurve(irEmissivityCurve, lambda_nm, 0.0f);
}

f32 Material::GetIRReflectance(f32 lambda_nm) const {
    // Fallback to spectralAlbedo if no IR curve available
    return InterpolateSpectralCurve(irReflectanceCurve, lambda_nm, spectralAlbedo);
}

f32 Material::GetIRTransmittance(f32 lambda_nm) const {
    return InterpolateSpectralCurve(irTransmittanceCurve, lambda_nm, 0.0f);
}

// ============================================================================
// Kirchhoff's Law Validation
// ============================================================================

bool Material::ValidateIRKirchhoffLaw() const {
    // If no IR data, skip validation
    if (!HasIRData()) {
        return true;
    }

    // Collect all wavelengths from all curves
    Vector<f32> allWavelengths;

    for (const auto& [lambda, value] : irEmissivityCurve) {
        allWavelengths.push_back(lambda);
    }
    for (const auto& [lambda, value] : irReflectanceCurve) {
        allWavelengths.push_back(lambda);
    }
    for (const auto& [lambda, value] : irTransmittanceCurve) {
        allWavelengths.push_back(lambda);
    }

    // Remove duplicates and sort
    std::sort(allWavelengths.begin(), allWavelengths.end());
    allWavelengths.erase(
        std::unique(allWavelengths.begin(), allWavelengths.end()),
        allWavelengths.end()
    );

    // Check Kirchhoff's law at each wavelength: ε + ρ + τ ≤ 1
    const f32 TOLERANCE = 1e-3f;  // Allow small numerical errors

    for (f32 lambda : allWavelengths) {
        f32 epsilon = GetIREmissivity(lambda);
        f32 rho = GetIRReflectance(lambda);
        f32 tau = GetIRTransmittance(lambda);

        f32 sum = epsilon + rho + tau;

        if (sum > 1.0f + TOLERANCE) {
            QL_LOG_WARN("Material '{}': Kirchhoff's law violated at {:.1f} nm: ε+ρ+τ = {:.3f} > 1.0",
                        name, lambda, sum);
            return false;
        }
    }

    return true;
}

} // namespace quantiloom
