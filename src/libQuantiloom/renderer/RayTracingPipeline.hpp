#pragma once

#include "core/Types.hpp"
#include "VulkanContext.hpp"
#include "GpuBuffer.hpp"
#include "GpuImage.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>

// ============================================================================
// RayTracingPipeline - Manages Vulkan Ray Tracing pipeline and SBT
// ============================================================================
// Responsibilities:
// - Load and compile ray tracing shaders (SPIR-V)
// - Create VkRayTracingPipelineKHR with shader groups
// - Build Shader Binding Table (SBT) with correct alignment
// - Manage descriptor set layouts and descriptor sets
// - Provide TraceRays() interface for rendering
//
// Lifetime:
// - Must be created AFTER VulkanContext
// - Must be destroyed BEFORE VulkanContext
// - Non-copyable, non-movable (owns Vulkan resources)
// ============================================================================

namespace quantiloom {

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
    void BindOutputImage(const GpuImage& image);

    // Bind TLAS (binding 1)
    void BindAccelerationStructure(VkAccelerationStructureKHR tlas);

    // Bind LUT buffer (binding 2)
    void BindLUTBuffer(const GpuBuffer& buffer);

    // Update all bindings (call after all Bind* calls)
    void UpdateDescriptorSets();

    // ========================================================================
    // Rendering
    // ========================================================================

    // Record trace rays command into provided command buffer
    void TraceRays(VkCommandBuffer cmd, u32 width, u32 height);

    // ========================================================================
    // Accessors
    // ========================================================================

    VkPipeline GetPipeline() const { return m_pipeline; }
    VkPipelineLayout GetPipelineLayout() const { return m_pipelineLayout; }

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
    std::vector<u32> LoadSPIRV(const std::string& path);

    // Create shader module from SPIR-V
    VkShaderModule CreateShaderModule(const std::vector<u32>& spirv);

    // Get SBT aligned size
    u32 AlignedSize(u32 size, u32 alignment) const;

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
};

} // namespace quantiloom
