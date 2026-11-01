#pragma once

#include "gpu/RtScene.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "renderer/GpuBuffer.hpp"

#include <vulkan/vulkan.h>

#include <memory>
#include <span>

namespace quantiloom::gpu::vulkan {

/// The ray tracing entry points. The loader exports none of them, so each VulkanDevice fetches
/// them once with vkGetDeviceProcAddr and copies the table into every command list it hands
/// out. A null member means the device lacks the extension.
struct RtFunctions {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR createStructure = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getStructureAddress = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyStructure = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildStructures = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getGroupHandles = nullptr;
    PFN_vkGetRayTracingShaderGroupStackSizeKHR getGroupStackSize = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;
};

[[nodiscard]] RtFunctions LoadRtFunctions(VkDevice device);

/// A geometry and the build info that points at it, filled in place (the info holds the
/// geometry's address, so the pair is never copied).
struct NativeRtGeometry {
    VkAccelerationStructureGeometryKHR geometry{};
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    u32 primitiveCount = 0;

    NativeRtGeometry() = default;
    NativeRtGeometry(const NativeRtGeometry&) = delete;
    NativeRtGeometry& operator=(const NativeRtGeometry&) = delete;
};

/// The fields AccelerationStructure.cpp fills: the geometry with its input addresses, and the
/// info in Build mode with no structures and no scratch, which is the build sizes query's form.
void FillRtGeometry(VkDevice device, const RtGeometry& geometry, NativeRtGeometry& out);

/// vkGetBufferDeviceAddress plus `offset`; 0 for a null buffer.
[[nodiscard]] VkDeviceAddress BufferAddress(VkDevice device, BufferRef buffer, u64 offset = 0);

/// Device::GetRtBuildSizes.
[[nodiscard]] RtBuildSizes QueryRtBuildSizes(const RtFunctions& rt, const RtGeometry& geometry);

inline constexpr u32 kRtInstanceStride = static_cast<u32>(sizeof(VkAccelerationStructureInstanceKHR));

/// Device::PackRtInstances: one VkAccelerationStructureInstanceKHR per instance.
void WriteRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out);

class VulkanRtAccelerationStructure final : public RtAccelerationStructure {
public:
    [[nodiscard]] static Result<std::unique_ptr<RtAccelerationStructure>, String> Create(VmaAllocator allocator,
                                                                                       const RtFunctions& rt,
                                                                                       RtLevel level, u64 bytes);
    ~VulkanRtAccelerationStructure() override;
    VulkanRtAccelerationStructure(const VulkanRtAccelerationStructure&) = delete;
    VulkanRtAccelerationStructure& operator=(const VulkanRtAccelerationStructure&) = delete;

    [[nodiscard]] RtSceneRef Ref() const override { return RtSceneRef{ToHandle(structure_), address_}; }
    [[nodiscard]] RtLevel Level() const override { return level_; }

private:
    VulkanRtAccelerationStructure(GpuBuffer&& storage, RtLevel level)
        : storage_(std::move(storage)), level_(level) {}

    GpuBuffer storage_;  // declared first: destroyed after the structure built on it
    VkDevice device_ = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR destroy_ = nullptr;
    VkAccelerationStructureKHR structure_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    RtLevel level_;
};

}  // namespace quantiloom::gpu::vulkan
