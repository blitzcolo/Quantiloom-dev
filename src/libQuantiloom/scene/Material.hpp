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
    // Spectral Mode (M1 compatibility)
    // ========================================================================
    // Scalar spectral reflectance for single-wavelength rendering
    // Computed from baseColorFactor during scene loading:
    //   spectralAlbedo = (R + G + B) / 3.0
    f32 spectralAlbedo = 0.8f;

    // ========================================================================
    // Metadata
    // ========================================================================
    String name;  // Material name (for debugging)

    // ========================================================================
    // Utilities
    // ========================================================================

    // Check if material is valid
    bool IsValid() const {
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
    bool HasTextures() const {
        return baseColorTextureIndex != -1 ||
               metallicRoughnessTextureIndex != -1 ||
               normalTextureIndex != -1 ||
               emissiveTextureIndex != -1;
    }

    // Create simple Lambertian material (for procedural geometry)
    static Material CreateLambertian(const glm::vec3& albedo, const String& name = "Lambertian") {
        Material mat;
        mat.name = name;
        mat.baseColorFactor = glm::vec4(albedo, 1.0f);
        mat.metallicFactor = 0.0f;   // Non-metal
        mat.roughnessFactor = 1.0f;  // Fully rough (Lambertian limit)
        mat.ComputeSpectralAlbedo();
        return mat;
    }
};

} // namespace quantiloom
