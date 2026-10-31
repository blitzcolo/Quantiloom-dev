/**
 * @file GpuInteropTestUtils.hpp
 * @brief Fill the neutral GPU PODs from the suite's shared Vulkan device
 *
 * The public context API no longer takes Vulkan types, so the tests go
 * through the same builders a host uses (renderer/interop/VulkanInterop.hpp)
 * rather than restating the handle packing in every file.
 */

#pragma once

#include "renderer/VulkanContext.hpp"
#include "renderer/interop/VulkanInterop.hpp"

namespace quantiloom::testing {

inline GpuDeviceHandles GpuHandles(const VulkanContext& context) {
    vk::Device device;
    device.physicalDevice = context.GetPhysicalDevice();
    device.device = context.GetDevice();
    device.queueFamily = context.GetGraphicsQueueFamily();
    device.queueIndex = 0;
    device.queue = context.GetGraphicsQueue();
    device.instance = context.GetInstance();
    auto handles = vk::Handles(device, context.GetInstance(), nullptr);
    handles.vmaAllocator = reinterpret_cast<GpuHandle>(context.GetAllocator());
    return handles;
}

}  // namespace quantiloom::testing
