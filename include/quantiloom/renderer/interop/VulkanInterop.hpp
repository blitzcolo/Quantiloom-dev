/**
 * @file VulkanInterop.hpp
 * @brief Vulkan-typed entry points and builders for the neutral GPU API
 *
 * Only consumers of the Quantiloom::VulkanInterop CMake component include
 * this header: it is the one public header that includes <vulkan/vulkan.h>.
 * renderer/GpuInterop.hpp and renderer/ExternalRenderContext.hpp stay
 * backend-neutral.
 *
 * The device helpers exist so that one requirement list serves the CLI and a
 * windowed host: vk::CreateDevice() and the SDK's own VulkanContext are built
 * from the same internal VulkanDeviceRequirements (the extension list, the
 * feature chain and the suitability test), so the two can no longer drift.
 *
 * Synchronization contract of the recording calls
 * ----------------------------------------------
 * A host compositor (QRhi) tracks each texture's layout, access and stage,
 * and generates its barriers from that record. It cannot see what the SDK
 * records, so the SDK's barriers are deliberately conservative:
 *
 *   - The FIRST barrier on the target waits on everything submitted before
 *     it: srcStage = ALL_COMMANDS, srcAccess = MEMORY_WRITE, from the
 *     caller-declared currentLayout. This also covers the write-after-read
 *     case of a compositor still sampling the previous frame's texture.
 *   - The LAST barrier moves the target to the caller-chosen finalLayout
 *     with dstStage = ALL_COMMANDS and dstAccess = MEMORY_READ | MEMORY_WRITE.
 *     The host's next use of the texture must request exactly that layout, so
 *     its first barrier is a same-layout barrier rather than a transition
 *     ordered from a scope it never saw. On QRhi the host then publishes the
 *     layout with QRhiTexture::setNativeLayout().
 *
 * BlitDepthTo keeps SHADER_READ_ONLY_OPTIMAL as its default final layout,
 * with the same conservative destination scope.
 */

#pragma once

#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "renderer/GpuInterop.hpp"

#include <vulkan/vulkan.h>

namespace quantiloom::vk {

struct DeviceRequest {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    /// VK_NULL_HANDLE: the SDK picks (first discrete device that passes, else first that passes).
    VkPhysicalDevice preferredPhysicalDevice = VK_NULL_HANDLE;
    /// Host-side present check for the graphics family (QVulkanInstance::supportsPresent).
    bool (*supportsPresent)(void* user, VkPhysicalDevice, uint32_t queueFamily) = nullptr;
    void* supportsPresentUser = nullptr;
    /// Enabled when supported, ignored otherwise (QRhi's preferred-for-import list).
    const char* const* optionalDeviceExtensions = nullptr;
    uint32_t optionalDeviceExtensionCount = 0;
    /// Always enabled; failure if unsupported (VK_KHR_swapchain for a presenting host).
    const char* const* requiredHostDeviceExtensions = nullptr;
    uint32_t requiredHostDeviceExtensionCount = 0;
};

struct Device {                     // host-owned; release with DestroyDevice()
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    uint32_t queueIndex = 0;
    VkQueue queue = VK_NULL_HANDLE;
    /// Kept so DestroyDevice can resolve vkDestroyDevice through the host's
    /// loader; not used by Handles().
    VkInstance instance = VK_NULL_HANDLE;
};

/// Create a device carrying every extension and feature the SDK needs, plus
/// the host's optional and required lists. The host owns the result.
QL_API Result<Device, String> CreateDevice(const DeviceRequest& request);
QL_API void DestroyDevice(const Device& device, PFN_vkGetInstanceProcAddr getInstanceProcAddr);

// Header-only builders: no exports, no behaviour.
inline GpuDeviceHandles Handles(const Device& d, VkInstance instance,
                                PFN_vkGetInstanceProcAddr gipa) {
    GpuDeviceHandles h;
    h.backend = GpuBackend::Vulkan;
    h.instance = reinterpret_cast<GpuHandle>(instance);
    h.physicalDevice = reinterpret_cast<GpuHandle>(d.physicalDevice);
    h.device = reinterpret_cast<GpuHandle>(d.device);
    h.queue = reinterpret_cast<GpuHandle>(d.queue);
    h.queueFamily = d.queueFamily;
    h.queueIndex = d.queueIndex;
    h.getProcAddr = reinterpret_cast<GpuHandle>(gipa);
    return h;
}
inline GpuCommandRecording Commands(VkCommandBuffer cb) {
    return {GpuBackend::Vulkan, kGpuInteropVersion, reinterpret_cast<GpuHandle>(cb)};
}
inline GpuImageTarget Target(VkImage image, GpuTargetFormat format, uint32_t w, uint32_t h,
                             VkImageLayout current, VkImageLayout final) {
    return {GpuBackend::Vulkan, kGpuInteropVersion, (GpuHandle)image, format, w, h,
            static_cast<u32>(current), static_cast<u32>(final)};
}

}  // namespace quantiloom::vk
