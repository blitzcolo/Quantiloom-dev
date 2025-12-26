#include "Material.hpp"
#include "core/Log.hpp"
#include <algorithm>
#include <cmath>
#include <ranges>

namespace quantiloom {

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

    for (const auto &lambda: irEmissivityCurve | std::views::keys) {
        allWavelengths.push_back(lambda);
    }
    for (const auto &lambda: irReflectanceCurve | std::views::keys) {
        allWavelengths.push_back(lambda);
    }
    for (const auto &lambda: irTransmittanceCurve | std::views::keys) {
        allWavelengths.push_back(lambda);
    }

    // Remove duplicates and sort
    std::ranges::sort(allWavelengths);
    allWavelengths.erase(
        std::ranges::unique(allWavelengths).begin(),
        allWavelengths.end()
    );

    // Check Kirchhoff's law at each wavelength: ε + ρ + τ ≤ 1

    for (f32 lambda : allWavelengths) {
        const f32 epsilon = GetIREmissivity(lambda);
        const f32 rho = GetIRReflectance(lambda);
        const f32 tau = GetIRTransmittance(lambda);

        f32 sum = epsilon + rho + tau;

        if (constexpr f32 TOLERANCE = 1e-3f; sum > 1.0f + TOLERANCE) {
            QL_LOG_WARN("Material '{}': Kirchhoff's law violated at {:.1f} nm: ε+ρ+τ = {:.3f} > 1.0",
                        name, lambda, sum);
            return false;
        }
    }

    return true;
}

} // namespace quantiloom
