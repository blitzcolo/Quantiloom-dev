#pragma once

#include "core/Types.hpp"
#include "core/Image.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <optional>

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
// Caching:
//   The LUT is scene-independent and can be cached to disk as a binary file.
//   Use LoadFromBinary() to skip expensive Monte Carlo integration.
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

    // ========================================================================
    // Binary Cache I/O (Fast Load/Save)
    // ========================================================================
    // Binary format for fast caching without EXR overhead.
    // File structure:
    //   - Header (16 bytes): magic(4) + version(4) + resolution(4) + sampleCount(4)
    //   - Data: raw float32 array (resolution × resolution × 2 channels)
    // ========================================================================

    // Save generated LUT to binary cache file
    // Returns true on success, false on failure
    static bool SaveToBinary(const String& filepath, const Image& lut, const Config& config);

    // Load LUT from binary cache file
    // Returns Image if file exists and is valid, std::nullopt otherwise
    // expectedConfig: If provided, validates that cached file matches expected parameters
    static std::optional<Image> LoadFromBinary(const String& filepath, const Config* expectedConfig = nullptr);

    // Check if binary cache file exists and is valid
    static bool IsCacheValid(const String& filepath, const Config& expectedConfig);

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

    // ========================================================================
    // Binary Format Constants
    // ========================================================================
    static constexpr u32 CACHE_MAGIC = 0x4C444642;   // "BFDL" (BRDF LUT)
    static constexpr u32 CACHE_VERSION = 1;
};

} // namespace quantiloom
