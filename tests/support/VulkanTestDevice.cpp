#include "support/VulkanTestDevice.hpp"

#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "gpu/vulkan/VulkanDevice.hpp"
#include "renderer/interop/VulkanInterop.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/GpuTestDevice.hpp"

namespace quantiloom::testing {

static_assert(kHostLayoutUndefined == static_cast<u32>(VK_IMAGE_LAYOUT_UNDEFINED));
static_assert(kHostLayoutGeneral == static_cast<u32>(VK_IMAGE_LAYOUT_GENERAL));

VulkanContext* SharedVulkanDevice() {
    gpu::Device* device = SharedGpuDevice(gpu::Backend::Vulkan);
    return device != nullptr ? &static_cast<gpu::vulkan::VulkanDevice*>(device)->Context() : nullptr;
}

const std::string& VulkanDeviceUnavailableReason() {
    return GpuDeviceUnavailableReason(gpu::Backend::Vulkan);
}

GpuDeviceHandles VulkanHostHandles(gpu::Device& device) {
    const VulkanContext& context = static_cast<gpu::vulkan::VulkanDevice&>(device).Context();
    vk::Device native;
    native.physicalDevice = context.GetPhysicalDevice();
    native.device = context.GetDevice();
    native.queueFamily = context.GetGraphicsQueueFamily();
    native.queueIndex = 0;
    native.queue = context.GetGraphicsQueue();
    native.instance = context.GetInstance();
    auto handles = vk::Handles(native, context.GetInstance(), nullptr);
    handles.vmaAllocator = reinterpret_cast<GpuHandle>(context.GetAllocator());
    return handles;
}

GpuCommandRecording VulkanHostRecording(gpu::CommandList& list) {
    return vk::Commands(static_cast<gpu::vulkan::VulkanCommandList&>(list).Native());
}

void VulkanDeviceTest::SetUp() {
    if (SharedVulkanDevice() == nullptr) {
        GTEST_SKIP() << VulkanDeviceUnavailableReason();
    }
}

VulkanContext& VulkanDeviceTest::Device() const {
    return *SharedVulkanDevice();
}

gpu::Device& VulkanDeviceTest::GpuDevice() const {
    return *SharedGpuDevice(gpu::Backend::Vulkan);
}

} // namespace quantiloom::testing
