#include "support/VulkanTestDevice.hpp"

#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "gpu/vulkan/VulkanDevice.hpp"
#include "gpu/vulkan/VulkanResources.hpp"
#include "renderer/interop/VulkanInterop.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/GpuTestDevice.hpp"

namespace quantiloom::testing {

static_assert(kHostLayoutUndefined == static_cast<u32>(VK_IMAGE_LAYOUT_UNDEFINED));
static_assert(kHostLayoutGeneral == static_cast<u32>(VK_IMAGE_LAYOUT_GENERAL));
static_assert(kHostLayoutTransferDst == static_cast<u32>(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));

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

u32 VulkanTextureMipLevels(const gpu::Texture& texture) {
    return gpu::vulkan::AsGpuImage(texture).GetMipLevels();
}

f32 VulkanPackedInstanceTranslationX(const void* packed, u32 index) {
    return static_cast<const VkAccelerationStructureInstanceKHR*>(packed)[index].transform.matrix[0][3];
}

void VulkanDeviceTest::SetUp() {
    if (SharedVulkanDevice() == nullptr) {
        GTEST_SKIP() << VulkanDeviceUnavailableReason();
    }
}

VulkanContext& VulkanDeviceTest::Device() const {
    return *SharedVulkanDevice();
}

} // namespace quantiloom::testing
