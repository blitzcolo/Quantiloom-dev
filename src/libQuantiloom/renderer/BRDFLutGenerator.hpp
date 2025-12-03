#pragma once

#include "core/Types.hpp"
#include "core/Image.hpp"
#include <glm/glm.hpp>
#include <vector>

// ============================================================================
// BRDF LUT Generator - For Image-Based Lighting (IBL)
// ============================================================================
// Generates a 2D lookup table for split-sum approximation of IBL specular
//
// LUT Format:
// - Size: 512x512 (NdotV × roughness)
// - Channels: RG16F or RG32F
//   - R: Scale term (for F·G integral)
//   - G: Bias term (for F·G integral)
//
// Usage:
//   float2 brdf = brdfLUT.Sample(NdotV, roughness);
//   float3 ibl = prefilteredColor * (F0 * brdf.x + brdf.y);
//
// References:
// - Epic Games, "Real Shading in Unreal Engine 4" (2013)
// - Karis, "Specular BRDF Reference" (2014)
// ============================================================================

namespace quantiloom {

class BRDFLutGenerator {
public:
    // ========================================================================
    // Configuration
    // ========================================================================

    struct Config {
        u32 resolution = 512;          // LUT texture size (NxN)
        u32 sampleCount = 1024;        // Monte Carlo samples per pixel
        bool highPrecision = false;    // Use float32 (true) vs float16 (false)
    };

    // ========================================================================
    // Generation
    // ========================================================================

    // Generate BRDF integration LUT (CPU-side Monte Carlo)
    // Returns Image with 2 channels (R=scale, G=bias)
    static Image Generate(const Config& config);
    static Image Generate();  // Uses default Config

    // Generate and save to file (EXR format)
    static bool GenerateAndSave(const String& filepath, const Config& config);
    static bool GenerateAndSave(const String& filepath);  // Uses default Config

private:
    // ========================================================================
    // Internal Computation
    // ========================================================================

    // Integrate BRDF for given NdotV and roughness
    // Returns (scale, bias) for split-sum approximation
    static glm::vec2 IntegrateBRDF(f32 NdotV, f32 roughness, u32 sampleCount);

    // GGX importance sampling (for specular lobe)
    static glm::vec3 ImportanceSampleGGX(glm::vec2 Xi, glm::vec3 N, f32 roughness);

    // GGX geometry function (Smith)
    static f32 GeometrySmith_GGX_IBL(f32 NdotV, f32 NdotL, f32 roughness);

    // Hammersley 2D sequence (low-discrepancy sampling)
    static glm::vec2 Hammersley(u32 i, u32 N);

    // Radical inverse (for Hammersley)
    static f32 RadicalInverse_VdC(u32 bits);
};

} // namespace quantiloom
