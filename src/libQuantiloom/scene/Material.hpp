#pragma once

#include "core/Types.hpp"
#include <glm/glm.hpp>
#include <string>

// ============================================================================
// Material - PBR material properties (glTF 2.0 metallic-roughness model)
// ============================================================================
// Implements full glTF 2.0 material specification:
// - Base color (RGBA factor + optional texture)
// - Metallic-Roughness workflow
// - Normal mapping
// - Emissive properties
// - Alpha blending modes
//
// For spectral rendering (M1 compatibility):
// - spectralAlbedo: Scalar reflectance computed from baseColorFactor
//
// Texture binding:
// - Texture indices refer to Scene::textures array
// - Index -1 means no texture (use factor value directly)
//
// Shader mapping:
// - All parameters uploaded to GPU via MaterialData buffer
// - Textures accessed via bindless descriptor array
// ============================================================================

namespace quantiloom {

struct Material {
    // ========================================================================
    // PBR Base Color
    // ========================================================================
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};  // RGBA [0, 1]
    i32 baseColorTextureIndex = -1;  // -1 = no texture

    // ========================================================================
    // Metallic-Roughness
    // ========================================================================
    f32 metallicFactor = 0.0f;   // [0, 1] (0 = dielectric, 1 = metal)
    f32 roughnessFactor = 1.0f;  // [0, 1] (0 = smooth, 1 = rough)
    i32 metallicRoughnessTextureIndex = -1;  // -1 = no texture
    // NOTE: In glTF, this is a combined texture (R=unused, G=roughness, B=metallic)

    // ========================================================================
    // Normal Mapping
    // ========================================================================
    i32 normalTextureIndex = -1;  // -1 = no normal map
    f32 normalScale = 1.0f;       // Normal map intensity [0, inf]

    // ========================================================================
    // Emissive
    // ========================================================================
    glm::vec3 emissiveFactor{0.0f, 0.0f, 0.0f};  // RGB [0, inf] (HDR allowed)
    i32 emissiveTextureIndex = -1;  // -1 = no texture

    // ========================================================================
    // Alpha Mode
    // ========================================================================
    enum class AlphaMode : u32 {
        Opaque = 0,  // Alpha channel ignored
        Mask = 1,    // Binary alpha test (alphaCutoff threshold)
        Blend = 2    // Alpha blending (requires sorted rendering)
    };

    AlphaMode alphaMode = AlphaMode::Opaque;
    f32 alphaCutoff = 0.5f;  // Threshold for AlphaMode::Mask

    // ========================================================================
    // Spectral Mode (M1 compatibility and M2+ full spectral)
    // ========================================================================
    // LEGACY (M1): Scalar spectral reflectance for single-wavelength rendering
    // Computed from baseColorFactor during scene loading:
    //   spectralAlbedo = (R + G + B) / 3.0
    // This field is kept for backward compatibility with M1 test scenes
    f32 spectralAlbedo = 0.8f;

    // NEW (M2+): Index into Scene::spectralReflectanceCurves array
    // -1 = no spectral curve (fallback to spectralAlbedo scalar)
    // >=0 = index into spectral curve buffer for physically-based spectral rendering
    // This enables full spectral fidelity for HS-OFF quantitative mode
    i32 spectralReflectanceCurveIndex = -1;

    // ========================================================================
    // Spectral Data Source Tracking (for HS-OFF validation)
    // ========================================================================
    // Tracks the origin of spectral data to enforce quality gates
    enum class SpectralSource : u32 {
        Unknown = 0,           // Default: unspecified
        Measured = 1,          // Physically measured spectral data (quantitative)
        RGBUpsampled = 2,      // Upsampled from RGB (sRGB or linear) - NOT quantitative
        Procedural = 3         // Procedurally generated (e.g., metal Fresnel)
    };
    SpectralSource spectralSource = SpectralSource::Unknown;

    // ========================================================================
    // Infrared Material Properties (for MWIR/LWIR modes)
    // ========================================================================
    // Spectral curves for infrared rendering (3-12μm wavelength range)
    // Used for quantitative thermal imaging simulation

    // Emissivity curve ε(λ): fraction of blackbody radiation emitted [0, 1]
    // By Kirchhoff's law: ε(λ) = α(λ) = 1 - ρ(λ) - τ(λ) in thermal equilibrium
    Vector<std::pair<f32, f32>> irEmissivityCurve;  // (wavelength_nm, emissivity)

    // Reflectance curve ρ(λ): fraction of incident radiation reflected [0, 1]
    Vector<std::pair<f32, f32>> irReflectanceCurve;  // (wavelength_nm, reflectance)

    // Transmittance curve τ(λ): fraction of incident radiation transmitted [0, 1]
    Vector<std::pair<f32, f32>> irTransmittanceCurve;  // (wavelength_nm, transmittance)

    // Surface temperature (K) for self-emission calculation
    // If <= 0, no thermal emission (or use scene ambient temperature)
    f32 irTemperature_K = 0.0f;

    // ========================================================================
    // Quantiloom Spectral Material Reference (from glTF extras)
    // ========================================================================
    // When set, this material uses pre-computed spectral data from SpectralBaker.
    // The name references an entry in the quantiloom_materials.json database.
    // Example glTF extras: { "quantiloom_material": "Gold_HS111.3B" }
    String quantiloomMaterialRef;  // Empty = no reference

    // ========================================================================
    // Metadata
    // ========================================================================
    String name;  // Material name (for debugging)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if material is valid
    [[nodiscard]] bool IsValid() const {
        // Base color must be in valid range
        if (baseColorFactor.r < 0.0f || baseColorFactor.g < 0.0f ||
            baseColorFactor.b < 0.0f || baseColorFactor.a < 0.0f) {
            return false;
        }

        // Metallic and roughness must be in [0, 1]
        if (metallicFactor < 0.0f || metallicFactor > 1.0f ||
            roughnessFactor < 0.0f || roughnessFactor > 1.0f) {
            return false;
        }

        // Alpha cutoff must be in [0, 1]
        if (alphaCutoff < 0.0f || alphaCutoff > 1.0f) {
            return false;
        }

        return true;
    }

    // Compute spectral albedo from base color (for M1 mode)
    void ComputeSpectralAlbedo() {
        spectralAlbedo = (baseColorFactor.r + baseColorFactor.g + baseColorFactor.b) / 3.0f;
    }

    // Check if material has any textures
    [[nodiscard]] bool HasTextures() const {
        return baseColorTextureIndex != -1 ||
               metallicRoughnessTextureIndex != -1 ||
               normalTextureIndex != -1 ||
               emissiveTextureIndex != -1;
    }

    // ========================================================================
    // Infrared Material Property Helpers
    // ========================================================================

    // Get IR emissivity at specific wavelength (linear interpolation)
    // Returns 0.0 if curve is empty or wavelength out of range
    [[nodiscard]] f32 GetIREmissivity(f32 lambda_nm) const;

    // Get IR reflectance at specific wavelength (linear interpolation)
    // Returns spectralAlbedo fallback if curve is empty
    [[nodiscard]] f32 GetIRReflectance(f32 lambda_nm) const;

    // Get IR transmittance at specific wavelength (linear interpolation)
    // Returns 0.0 if curve is empty (opaque)
    [[nodiscard]] f32 GetIRTransmittance(f32 lambda_nm) const;

    // Validate Kirchhoff's law: ε + ρ + τ ≤ 1 at all wavelengths
    // Returns true if valid, false if energy conservation violated
    [[nodiscard]] bool ValidateIRKirchhoffLaw() const;

    // Check if material has IR data
    [[nodiscard]] bool HasIRData() const {
        return !irEmissivityCurve.empty() ||
               !irReflectanceCurve.empty() ||
               !irTransmittanceCurve.empty();
    }

    // Create simple Lambertian material (for procedural geometry)
    static Material CreateLambertian(const glm::vec3& albedo, const String& name = "Lambertian") {
        Material mat;
        mat.name = name;
        mat.baseColorFactor = glm::vec4(albedo, 1.0f);
        mat.metallicFactor = 0.0f;   // Non-metal
        mat.roughnessFactor = 1.0f;  // Fully rough (Lambertian limit)
        mat.ComputeSpectralAlbedo();
        mat.spectralSource = SpectralSource::RGBUpsampled;  // Mark as upsampled
        return mat;
    }
};

} // namespace quantiloom
