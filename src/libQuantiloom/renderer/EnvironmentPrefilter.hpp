// ============================================================================
// Quantiloom - Environment Prefilter for Image-Based Lighting
// ============================================================================
// Generates prefiltered environment cubemaps and BRDF integration LUT
// for physically-based IBL specular reflections
//
// Implementation based on:
// - "Real Shading in Unreal Engine 4" (Brian Karis, Epic Games, 2013)
// - "Moving Frostbite to PBR" (Sébastien Lagarde, Frostbite, 2014)
// - glTF 2.0 IBL specification
//
// CROSS-PLATFORM: Pure Vulkan compute shaders, no vendor-specific extensions
// ============================================================================

#pragma once

#include "core/Types.hpp"
#include "core/Log.hpp"
#include "core/Image.hpp"
#include "GpuImage.hpp"
#include "GpuBuffer.hpp"
#include "VulkanContext.hpp"

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

namespace quantiloom {

// ============================================================================
// EnvironmentPrefilter Class
// ============================================================================
// Generates IBL assets from HDR environment maps
// ============================================================================

class EnvironmentPrefilter {
public:
    // ========================================================================
    // Configuration
    // ========================================================================

    struct Config {
        String hdrPath;                  // HDR equirectangular map (e.g., "forest.hdr")
        u32 cubemapSize = 512;           // Cubemap resolution (per face)
        u32 irradianceSize = 64;         // Irradiance map resolution (for diffuse IBL)
        u32 prefilteredMipLevels = 5;    // Number of roughness levels (mip chain)
        u32 brdfLUTSize = 512;           // BRDF integration LUT resolution
        u32 numSamples = 1024;           // Monte Carlo samples for prefiltering
    };

    // ========================================================================
    // Public Interface
    // ========================================================================

    explicit EnvironmentPrefilter(VulkanContext& ctx);
    ~EnvironmentPrefilter();

    // Initialize compute pipelines
    bool Initialize();

    // Generate all IBL assets from HDR environment map
    // Returns: {prefilteredCubemap, irradianceMap, brdfLUT}
    struct IBLAssets {
        GpuImage prefilteredCubemap;  // Prefiltered specular cubemap (with mipmaps)
        GpuImage irradianceMap;       // Diffuse irradiance cubemap (single mip)
        GpuImage brdfLUT;             // BRDF integration lookup table (2D texture)
    };

    Result<IBLAssets, String> GenerateFromHDR(const Config& config);

    // Generate BRDF LUT only (independent of environment map)
    Result<GpuImage, String> GenerateBRDF_LUT(u32 resolution = 512);

private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    // Load HDR image and convert to cubemap
    Result<GpuImage, String> LoadHDREquirectangular(const String& path);
    Result<GpuImage, String> EquirectangularToCubemap(const GpuImage& equirect, u32 size);

    // Prefilter cubemap for different roughness levels
    Result<GpuImage, String> PrefilterCubemap(const GpuImage& envCubemap, u32 mipLevels, u32 numSamples);

    // Generate diffuse irradiance map (convolution with cosine-weighted hemisphere)
    Result<GpuImage, String> GenerateIrradianceMap(const GpuImage& envCubemap, u32 size);

    // Create compute pipelines
    bool CreatePipelines();
    void DestroyPipelines();

    // ========================================================================
    // Vulkan Resources
    // ========================================================================

    VulkanContext& m_context;

    // Compute pipelines
    VkPipeline m_equirectToCubePipeline = VK_NULL_HANDLE;
    VkPipeline m_prefilterPipeline = VK_NULL_HANDLE;
    VkPipeline m_irradiancePipeline = VK_NULL_HANDLE;
    VkPipeline m_brdfLUTPipeline = VK_NULL_HANDLE;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    // Samplers
    VkSampler m_linearSampler = VK_NULL_HANDLE;
};

} // namespace quantiloom
