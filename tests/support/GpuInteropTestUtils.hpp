/**
 * @file GpuInteropTestUtils.hpp
 * @brief Fill the neutral GPU PODs from a test device, as a host would
 *
 * The public context API takes no backend types, so a test hands ExternalRenderContext
 * the same PODs a host builds (renderer/GpuInterop.hpp): its device's handles, the
 * command buffer a gpu::CommandList records into, and a test-owned texture as a
 * present target. The builders dispatch on the backend (GpuTestDevice.cpp).
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/GpuTypes.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "renderer/GpuInterop.hpp"
#include "renderer/interop/VulkanInterop.hpp"

namespace quantiloom::gpu {
class CommandList;
class Device;
}  // namespace quantiloom::gpu

namespace quantiloom::testing {

/// Host layout tokens (GpuImageTarget::currentLayout/finalLayout) for the images the
/// tests own: the Vulkan VkImageLayout values, pinned in VulkanTestDevice.cpp. Metal
/// ignores layouts.
inline constexpr u32 kHostLayoutUndefined = 0;
inline constexpr u32 kHostLayoutGeneral = 1;

/// The handles a host would give ExternalRenderContext::Create for @p device.
GpuDeviceHandles GpuHandles(gpu::Device& device);

/// The command buffer @p list records into, as a host's recording.
GpuCommandRecording GpuRecording(gpu::CommandList& list);

/// A test-owned texture of @p backend as a host's target, its extent from the ref.
GpuImageTarget GpuTarget(gpu::Backend backend, gpu::TextureRef texture, GpuTargetFormat format,
                         u32 currentLayout, u32 finalLayout);

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
