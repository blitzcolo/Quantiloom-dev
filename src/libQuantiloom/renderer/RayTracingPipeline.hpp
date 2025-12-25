/**
 * @file RayTracingPipeline.hpp
 * @brief Vulkan ray tracing pipeline management and shader binding
 *
 * Provides RayTracingPipeline class for managing:
 * - Ray tracing shader loading and compilation (SPIR-V)
 * - VkRayTracingPipelineKHR creation with shader groups
 * - Shader Binding Table (SBT) construction with proper alignment
 * - Descriptor set management for resource binding
 * - TraceRays() execution interface
 *
 * Shader Groups:
 * - Ray Generation: Primary ray generation from camera
 * - Closest Hit: Surface shading and material evaluation
 * - Miss: Sky/environment sampling
 *
 * Resource Bindings (Fixed Descriptor Layout):
 * - Binding 0: Output image (storage image)
 * - Binding 1: TLAS (acceleration structure)
 * - Binding 2: Lighting parameters (uniform buffer)
 * - Binding 3-4: Geometry buffers (vertex, index)
 * - Binding 5: Material buffer (storage buffer)
 * - Binding 6-7: Texture arrays (bindless, sampled images + samplers)
 * - Binding 8: UV buffer (optional)
 * - Binding 9: Tangent buffer (optional)
 * - Binding 10-12: IBL resources (prefiltered envmap, BRDF LUT, sampler)
 * - Binding 13: Spectral reflectance curves (storage buffer)
 * - Binding 14: Complex refractive index (n,k) data (storage buffer)
 * - Binding 15: Solar spectral LUT (storage buffer)
 * - Binding 16: Normal buffer (required)
 * - Binding 17: Atmospheric parameters (storage buffer)
 *
 * Usage example:
 * @code
 * VulkanContext context;
 * RayTracingPipeline pipeline(context, "raygen.spv", "closesthit.spv", "miss.spv");
 *
 * // Bind resources
 * pipeline.BindOutputImage(outputImage);
 * pipeline.BindAccelerationStructure(tlas.GetHandle());
 * pipeline.BindGeometryBuffers(vertexBuffer, indexBuffer, &uvBuffer);
 * pipeline.BindMaterialBuffer(materialBuffer);
 * pipeline.BindTextures(imageViews, samplers);
 *
 * // Set camera and render
 * pipeline.SetCameraData(cameraData);
 * pipeline.SetSamplingParams(frameIndex, sampleIndex, spp, randomSeed);
 *
 * CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
 *     pipeline.TraceRays(cmd, width, height);
 * });
 * @endcode
 *
 * @note Pipeline must outlive all bound resources
 * @note All shaders must be compiled to SPIR-V before loading
 * @note Descriptor binding indices MUST match shader layout qualifiers
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "VulkanContext.hpp"
#include "GpuBuffer.hpp"
#include "GpuImage.hpp"
#include "scene/Camera.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>

// ============================================================================
// RayTracingPipeline - Manages Vulkan Ray Tracing pipeline and SBT
// ============================================================================

namespace quantiloom {

/**
 * @class RayTracingPipeline
 * @brief Manages Vulkan ray tracing pipeline, shader binding table, and resource bindings
 *
 * Central class for ray tracing rendering in Quantiloom. Handles all aspects of
 * Vulkan ray tracing pipeline creation, resource binding, and ray dispatch.
 *
 * Key responsibilities:
 * - Shader module loading from SPIR-V files
 * - Pipeline creation with ray generation, closest hit, and miss shaders
 * - Shader Binding Table (SBT) construction with correct memory alignment
 * - Descriptor set layout creation and management
 * - Resource binding interface (buffers, images, acceleration structures)
 * - Push constants for camera and sampling parameters
 * - vkCmdTraceRaysKHR dispatch
 *
 * Resource Binding Architecture:
 * All resources are bound through descriptor sets using fixed binding indices.
 * The binding layout is defined in shaders and MUST match exactly:
 * @code
 * // Shader layout (common.hlsli)
 * [[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
 * [[vk::binding(1, 0)]] RaytracingAccelerationStructure scene;
 * [[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;
 * // ... etc
 * @endcode
 *
 * Shader Binding Table Layout:
 * @code
 * | Raygen | Closest Hit | Miss |
 * | ------ | ----------- | ---- |
 * | 1 shader | 1 shader  | 1 shader |
 * @endcode
 *
 * @note Non-copyable, non-movable (owns Vulkan resources)
 * @note Must be created AFTER VulkanContext
 * @note Must be destroyed BEFORE VulkanContext
 *
 * @see VulkanContext for Vulkan initialization
 * @see GpuBuffer for buffer resource management
 * @see GpuImage for image resource management
 * @see CommandHelper for command buffer utilities
 */
class QL_API RayTracingPipeline {
public:
    // ========================================================================
    // Shader stage descriptors
    // ========================================================================

    struct ShaderStage {
        std::string spirvPath;  // Path to compiled SPIR-V file
        VkShaderStageFlagBits stage;  // RAYGEN, CLOSEST_HIT, MISS, etc.
    };

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Create pipeline with minimal shader set (Raygen + ClosestHit + Miss)
    RayTracingPipeline(
        VulkanContext& context,
        const std::string& raygenPath,
        const std::string& closestHitPath,
        const std::string& missPath
    );

    ~RayTracingPipeline();

    // Non-copyable, non-movable
    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;
    RayTracingPipeline(RayTracingPipeline&&) = delete;
    RayTracingPipeline& operator=(RayTracingPipeline&&) = delete;

    // ========================================================================
    // Descriptor binding (resources)
    // ========================================================================

    // Bind output image (binding 0)
    void BindOutputImage(const GpuImage& image) const;

    // Bind TLAS (binding 1)
    void BindAccelerationStructure(VkAccelerationStructureKHR tlas) const;

    // Bind LUT buffer (binding 2)
    void BindLUTBuffer(const GpuBuffer& buffer) const;

    // Bind geometry buffers (binding 3: vertex, binding 4: index, binding 8: UVs [optional])
    void BindGeometryBuffers(const GpuBuffer& vertexBuffer, const GpuBuffer& indexBuffer, const GpuBuffer* uvBuffer = nullptr) const;

    // Bind material buffer (binding 5)
    void BindMaterialBuffer(const GpuBuffer& buffer) const;

    // Bind tangent buffer (binding 9) - Optional
    void BindTangentBuffer(const GpuBuffer& buffer) const;

    // Bind normal buffer (binding 16) - Required for smooth shading
    void BindNormalBuffer(const GpuBuffer& buffer) const;

    // Bind texture arrays (binding 6: textures, binding 7: samplers)
    // Uses bindless descriptor indexing (VK_EXT_descriptor_indexing)
    // If imageViews is empty, binds a single dummy white texture
    void BindTextures(const std::vector<VkImageView>& imageViews,
                      const std::vector<VkSampler>& samplers) const;

    // ========================================================================
    // IBL (Image-Based Lighting) Bindings
    // ========================================================================
    // Bind IBL textures for physically-based specular reflections
    // - Binding 10: Prefiltered environment cubemap (with mipmaps for roughness)
    // - Binding 11: BRDF integration LUT (2D texture)
    // - Binding 12: IBL sampler (shared)
    // ========================================================================

    // Bind prefiltered environment cubemap (binding 10)
    void BindPrefilteredEnvMap(VkImageView imageView) const;

    // Bind BRDF integration LUT for IBL (binding 11: texture, binding 12: sampler)
    void BindBRDFLut(VkImageView imageView, VkSampler sampler) const;

    // ========================================================================
    // Spectral Curves Buffer (Binding 13)
    // ========================================================================
    // Bind measured spectral reflectance curves for quantitative HS-OFF mode
    // Each curve is a SpectralCurveGPU struct (272 bytes) with uniform sampling
    // ========================================================================

    // Bind spectral curves buffer (binding 13)
    // Pass nullptr or empty buffer to disable spectral curve lookup (uses RGB fallback)
    void BindSpectralCurvesBuffer(const GpuBuffer* buffer) const;

    // ========================================================================
    // Complex Refractive Index Buffer (Binding 14)
    // ========================================================================
    // Bind measured complex refractive index (n, k) for physical Fresnel
    // Each entry is a ComplexRefractiveIndexGPU struct (528 bytes)
    // Used for accurate specular reflection on metals (gold, silver, etc.)
    // ========================================================================

    // Bind complex refractive index buffer (binding 14)
    // Pass nullptr to disable physical Fresnel (uses PBR F0 approximation)
    void BindComplexRefractiveIndexBuffer(const GpuBuffer* buffer) const;

    // ========================================================================
    // Solar Spectral LUT Buffer (Binding 15)
    // ========================================================================
    // Bind ASTM G-173 solar irradiance curves for spectral rendering
    // Structure: SolarSpectralLUT (544 bytes = 2 × SpectralCurveGPU)
    // - sunIrradiance: Direct+circumsolar irradiance (W·m⁻²·nm⁻¹)
    // - skyIrradiance: Diffuse sky irradiance (W·m⁻²·nm⁻¹)
    // ========================================================================

    // Bind solar spectral LUT buffer (binding 15)
    // Pass nullptr to use LightingParams RGB fallback
    void BindSolarSpectralLUT(const GpuBuffer* buffer) const;

    // ========================================================================
    // Atmospheric Parameters (Binding 17)
    // ========================================================================
    // Bind atmospheric parameters buffer for Delta-Tracking volumetric rendering
    // Provides Rayleigh + Mie scattering coefficients for wavelength-dependent extinction
    // Pass nullptr to disable atmospheric scattering (use LUT fallback)
    // ========================================================================

    // Bind atmospheric parameters buffer (binding 17)
    // Pass nullptr to disable Delta-Tracking
    void BindAtmosphericParams(const GpuBuffer* buffer) const;

    // Update all bindings (call after all Bind* calls)
    static void UpdateDescriptorSets();

    // ========================================================================
    // Rendering
    // ========================================================================

    // Set camera parameters (call before TraceRays)
    void SetCameraData(const struct CameraData& cameraData);

    // Set accumulation sampling parameters (call before TraceRays)
    void SetSamplingParams(u32 frameIndex, u32 sampleIndex, u32 totalSamples, u32 randomSeed);

    // Record trace rays command into provided command buffer
    void TraceRays(VkCommandBuffer cmd, u32 width, u32 height) const;

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] VkPipeline GetPipeline() const { return m_pipeline; }
    [[nodiscard]] VkPipelineLayout GetPipelineLayout() const { return m_pipelineLayout; }

private:
    // ========================================================================
    // Initialization steps
    // ========================================================================

    void CreateDescriptorSetLayout();
    void CreatePipelineLayout();
    void LoadShaders();
    void CreatePipeline();
    void CreateShaderBindingTable();

    // ========================================================================
    // Helpers
    // ========================================================================

    // Load SPIR-V shader from file
    static std::vector<u32> LoadSPIRV(const std::string& path);

    // Create shader module from SPIR-V
    [[nodiscard]] VkShaderModule CreateShaderModule(const std::vector<u32>& spirv) const;

    // Get SBT aligned size
    static u32 AlignedSize(u32 size, u32 alignment);

    // ========================================================================
    // Vulkan handles
    // ========================================================================

    VulkanContext& m_context;

    // Shader paths
    std::string m_raygenPath;
    std::string m_closestHitPath;
    std::string m_missPath;

    // Pipeline objects
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    // Descriptor pool and sets
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    // Shader Binding Table (SBT)
    std::unique_ptr<GpuBuffer> m_sbtBuffer;
    VkStridedDeviceAddressRegionKHR m_raygenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};

    // Shader modules (temporary, destroyed after pipeline creation)
    std::vector<VkShaderModule> m_shaderModules;

    // Ray Tracing properties (cached from context)
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{};

    // Push constants (camera + sampling parameters)
    PushConstantsRayGen m_pushConstants{};
};

} // namespace quantiloom
