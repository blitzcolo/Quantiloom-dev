#pragma once

#include "gpu/Trace.hpp"
#include "gpu/vulkan/VulkanRt.hpp"
#include "renderer/GpuBuffer.hpp"

#include <vulkan/vulkan.h>

#include <memory>
#include <span>

namespace quantiloom::gpu::vulkan {

class VulkanPipelineCache final : public PipelineCache {
public:
    [[nodiscard]] static Result<std::unique_ptr<PipelineCache>, String> Create(VkDevice device,
                                                                               std::span<const std::byte> initialData);
    ~VulkanPipelineCache() override;
    VulkanPipelineCache(const VulkanPipelineCache&) = delete;
    VulkanPipelineCache& operator=(const VulkanPipelineCache&) = delete;

    [[nodiscard]] Result<Vector<std::byte>, String> Serialize() const override;
    [[nodiscard]] VkPipelineCache Native() const { return cache_; }

private:
    VulkanPipelineCache() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
};

/// A ray tracing pipeline and the shader binding table that goes with it.
class VulkanTraceKernel final : public TraceKernel {
public:
    [[nodiscard]] static Result<std::unique_ptr<TraceKernel>, String> Create(
        VmaAllocator allocator, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& properties,
        const RtFunctions& rt, const TraceKernelDesc& desc);
    ~VulkanTraceKernel() override;
    VulkanTraceKernel(const VulkanTraceKernel&) = delete;
    VulkanTraceKernel& operator=(const VulkanTraceKernel&) = delete;

    [[nodiscard]] const KernelLayout& Layout() const override { return *layout_; }
    [[nodiscard]] StringView Name() const override { return name_; }
    [[nodiscard]] VkPipeline Native() const { return pipeline_; }

    /// vkCmdTraceRaysKHR over this kernel's SBT regions.
    void TraceRays(VkCommandBuffer cmd, u32 width, u32 height, u32 depth) const;

private:
    VulkanTraceKernel() = default;

    std::unique_ptr<GpuBuffer> sbt_;  // declared first: destroyed after the pipeline
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    PFN_vkCmdTraceRaysKHR traceRays_ = nullptr;
    const KernelLayout* layout_ = nullptr;
    String name_;
    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};
};

}  // namespace quantiloom::gpu::vulkan
