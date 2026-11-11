// The Vulkan backend's entry points of gpu/DeviceFactory.hpp.

#include "gpu/DeviceFactory.hpp"

#include "core/ResultFail.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "gpu/vulkan/VulkanContextAdapter.hpp"
#include "gpu/vulkan/VulkanDevice.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "renderer/GpuInterop.hpp"

#include <exception>
#include <memory>

namespace quantiloom::gpu::vulkan {

bool IsCompiled() {
    return true;
}

Result<std::unique_ptr<Device>, String> CreateSystemDevice(const SystemDeviceOptions&) {
    // VulkanContext's constructor throws on every failure; its text becomes the Err.
    try {
        return std::unique_ptr<Device>(std::make_unique<VulkanDevice>(std::make_unique<VulkanContext>()));
    } catch (const std::exception& e) {
        return Fail<std::unique_ptr<Device>>(e.what());
    }
}

Result<std::unique_ptr<Device>, String> CreateDeviceFromHandles(const GpuDeviceHandles& handles) {
    VulkanContext::ExternalHandles native{};
    native.instance = FromHandle<VkInstance>(handles.instance);
    native.physicalDevice = FromHandle<VkPhysicalDevice>(handles.physicalDevice);
    native.device = FromHandle<VkDevice>(handles.device);
    native.graphicsQueue = FromHandle<VkQueue>(handles.queue);
    native.graphicsQueueFamily = handles.queueFamily;
    native.allocator = FromHandle<VmaAllocator>(handles.vmaAllocator);
    try {
        return std::unique_ptr<Device>(
            std::make_unique<VulkanDevice>(std::make_unique<VulkanContextAdapter>(native, true)));
    } catch (const std::exception& e) {
        return Fail<std::unique_ptr<Device>>(e.what());
    }
}

}  // namespace quantiloom::gpu::vulkan
