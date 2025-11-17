#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <string>

// ============================================================================
// Material - Surface material properties
// ============================================================================
// M1 Phase: Simple Lambertian BRDF only
// - albedo: diffuse reflectance (RGB approximation, will be replaced by
//           spectral reflectance curves in M2+)
//
// Future (M2+):
// - Spectral reflectance curves (wavelength-dependent)
// - Emissive properties (self-emission, temperature)
// - BRDF models (GGX, microfacet, etc.)
// ============================================================================

namespace quantiloom {

struct Material {
    // M1: Simple Lambertian parameters
    glm::vec3 albedo{0.8f, 0.8f, 0.8f};  // Diffuse reflectance [0, 1]

    // Metadata
    String name;  // Material name (for debugging)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if material is valid
    bool IsValid() const {
        // Albedo must be in [0, 1] range
        if (albedo.r < 0.0f || albedo.r > 1.0f ||
            albedo.g < 0.0f || albedo.g > 1.0f ||
            albedo.b < 0.0f || albedo.b > 1.0f) {
            return false;
        }
        return true;
    }

    // Get average albedo (for diagnostics)
    f32 GetAverageAlbedo() const {
        return (albedo.r + albedo.g + albedo.b) / 3.0f;
    }
};

} // namespace quantiloom
