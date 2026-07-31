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
 * - Binding 17: NN atmosphere LUT header (storage buffer)
 * - Binding 18: Instance geometry info (storage buffer)
 * - Binding 19: CIE CMF LUT (storage buffer)
 * - Binding 20: NN atmosphere LUT data blob (storage buffer)
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
 * @author blitzcolo
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
#include <unordered_map>

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
class RayTracingPipeline {
public:
    // ========================================================================
    // Specialization constants
    // ========================================================================

    struct SpecConstants {
        u32 spectralMode = 7;  // SPECTRAL_MODE_RGB
        u32 debugEnabled = 0;
    };

    // ========================================================================
    // Shader stage descriptors
    // ========================================================================

    struct ShaderStage {
        std::string spirvPath;  // Path to compiled SPIR-V file
        VkShaderStageFlagBits stage;  // RAYGEN, CLOSEST_HIT, MISS, etc.
    };

    // ========================================================================
    // Pipeline Cache (for faster startup)
    // ========================================================================

    /**
     * @brief Load pipeline cache from disk
     * @param context VulkanContext to create cache for
     * @param cachePath Path to cache file (e.g., "pipeline_cache.bin")
     * @return VkPipelineCache handle (VK_NULL_HANDLE if failed or file doesn't exist)
     *
     * Call once at application startup before creating any pipelines.
     * The cache accelerates shader compilation on subsequent runs.
     */
    static VkPipelineCache LoadPipelineCache(VulkanContext& context, const std::string& cachePath);

    /**
     * @brief Save pipeline cache to disk
     * @param context VulkanContext that owns the cache
     * @param cache Pipeline cache to save
     * @param cachePath Path to save cache file
     * @return true if saved successfully
     *
     * Call at application shutdown after all pipelines are destroyed.
     */
    static bool SavePipelineCache(VulkanContext& context, VkPipelineCache cache, const std::string& cachePath);

    /**
     * @brief Destroy pipeline cache
     * @param context VulkanContext that owns the cache
     * @param cache Pipeline cache to destroy
     */
    static void DestroyPipelineCache(VulkanContext& context, VkPipelineCache cache);

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Create pipeline with minimal shader set (Raygen + ClosestHit + Miss)
    // Optional pipelineCache accelerates creation (use LoadPipelineCache() to obtain)
    RayTracingPipeline(
        VulkanContext& context,
        const std::string& raygenPath,
        const std::string& closestHitPath,
        const std::string& missPath,
        VkPipelineCache pipelineCache = VK_NULL_HANDLE
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

    // Bind primary-hit depth AOV image (binding 22)
    void BindDepthImage(const GpuImage& image) const;

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
    void BindPrefilteredEnvMap(VkImageView imageView, VkSampler sampler) const;

    // Bind BRDF integration LUT for IBL (binding 11: texture, binding 12: sampler)
    void BindBRDFLut(VkImageView imageView, VkSampler sampler) const;

    // ========================================================================
    // Spectral Curves Buffer (Binding 13)
    // ========================================================================
    // Bind measured spectral reflectance curves for quantitative spectral rendering
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
    // NN Atmosphere (Bindings 17 + 20)
    // ========================================================================
    // Bind the baked NN atmosphere LUT: header (AtmosNNHeaderGPU, binding 17)
    // and the flat float data blob (binding 20). When the atmosphere is
    // disabled, bind a header with enabled = 0 and a small dummy data buffer;
    // shaders must not sample the LUT when header.enabled == 0.
    // ========================================================================

    void BindAtmosphereNN(const GpuBuffer* header, const GpuBuffer* data) const;

    // ========================================================================
    // Instance Geometry Info Buffer (Binding 18)
    // ========================================================================
    // Bind per-instance geometry offset information for multi-BLAS support
    // Each TLAS instance has its own offset into the merged global geometry buffers
    // Structure: InstanceGeometryInfo (32 bytes) with vertex/index/normal/UV/tangent offsets
    // ========================================================================

    // Bind instance geometry info buffer (binding 18)
    void BindInstanceGeometryBuffer(const GpuBuffer& buffer) const;

    // ========================================================================
    // CIE 1931 Color Matching Functions LUT (Binding 19)
    // ========================================================================
    // High-precision CIE XYZ CMFs for VIS_FUSED mode spectral integration
    // 401 samples covering 380-780nm at 1nm resolution
    // Compiled in from core/CIE_CMF_Data.hpp, not read from assets/luts/
    // Each sample is float3(x_bar, y_bar, z_bar)
    // ========================================================================

    // Bind CIE CMF LUT buffer (binding 19)
    void BindCIE_CMF_LUT(const GpuBuffer& buffer) const;

    // Update all bindings (call after all Bind* calls)
    static void UpdateDescriptorSets();

    // ========================================================================
    // Rendering
    // ========================================================================

    // Set camera parameters (call before TraceRays)
    void SetCameraData(const struct CameraData& cameraData);

    // Set accumulation sampling parameters (call before TraceRays)
    void SetSamplingParams(u32 frameIndex, u32 sampleIndex, u32 totalSamples, u32 randomSeed);

    // Record trace rays command into provided command buffer.
    // finalDispatch=true inserts a RT->TRANSFER barrier (for readback).
    // finalDispatch=false inserts a RT->RT barrier (for next sample accumulation).
    void TraceRays(VkCommandBuffer cmd, u32 width, u32 height, bool finalDispatch = true) const;

    // Select pipeline variant for the given specialization constants.
    // Lazily creates and caches the pipeline + SBT on first use per combination.
    void SetSpecConstants(u32 spectralMode, bool debugEnabled);

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
    std::string m_shadowMissPath;  // Auto-derived from missPath or explicitly set

    // Pipeline objects
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;  // External, not owned

    // Descriptor pool and sets
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    // Shader Binding Table (SBT)
    std::unique_ptr<GpuBuffer> m_sbtBuffer;
    VkStridedDeviceAddressRegionKHR m_raygenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};

    // SPIR-V bytecode (kept for lazy pipeline variant creation)
    std::vector<std::vector<u32>> m_spirvData;  // [raygen, chit, miss, shadow_miss]

    // Ray Tracing properties (cached from context)
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{};

    // Pipeline variant cache (keyed by packed spec constants)
    struct PipelineVariant {
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::unique_ptr<GpuBuffer> sbtBuffer;
        VkStridedDeviceAddressRegionKHR raygenRegion{};
        VkStridedDeviceAddressRegionKHR missRegion{};
        VkStridedDeviceAddressRegionKHR hitRegion{};
        VkStridedDeviceAddressRegionKHR callableRegion{};
    };

    std::unordered_map<u64, PipelineVariant> m_variantCache;
    PipelineVariant* m_activeVariant = nullptr;

    PipelineVariant CreatePipelineVariant(const SpecConstants& spec);
    static u64 PackSpecKey(u32 spectralMode, u32 debugEnabled) {
        return (static_cast<u64>(debugEnabled) << 32) | spectralMode;
    }

    // Dynamic texture limit (based on device capabilities)
    // 1024 if descriptor indexing available, 32 otherwise
    u32 m_maxTextures = 1024;

    // Push constants (camera + sampling parameters)
    PushConstantsRayGen m_pushConstants{};
};

} // namespace quantiloom
