/**
 * @file VulkanDeviceRequirements.hpp
 * @brief The one list of what a Quantiloom Vulkan device must carry
 *
 * Both device creators read this: the SDK's own VulkanContext (CLI, offline)
 * and the public vk::CreateDevice() helper a windowed host uses. The two used
 * to keep separate lists, and they drifted -- Studio asked for features the
 * SDK never needed while missing ray query and descriptor indexing.
 *
 * The entry points are resolved through the host's vkGetInstanceProcAddr when
 * one is given, so an embedded host and the SDK go through the same loader;
 * the default is the SDK's own linked loader.
 */

#pragma once

#include "core/Types.hpp"

#include <vulkan/vulkan.h>

#include <vector>

namespace quantiloom::vulkan {

/// The subset of Vulkan entry points device selection and creation needs.
/// Defaults are the SDK's linked loader.
struct DeviceFunctions {
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = &vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = &vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = &vkGetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties =
        &vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties =
        &vkEnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice CreateDevice = &vkCreateDevice;
};

/// Resolve through @p getInstanceProcAddr, keeping the linked loader for
/// anything it does not return. A null @p getInstanceProcAddr means the
/// defaults throughout.
DeviceFunctions LoadDeviceFunctions(VkInstance instance,
                                    PFN_vkGetInstanceProcAddr getInstanceProcAddr);

/// Extensions every Quantiloom device is created with.
const std::vector<const char*>& RequiredDeviceExtensions();

struct DeviceCapabilities {
    bool hasVulkan13 = false;
    bool hasSynchronization2 = false;
    bool hasDynamicRendering = false;
    bool hasDescriptorIndexing = false;
};

DeviceCapabilities QueryDeviceCapabilities(VkPhysicalDevice device, const DeviceFunctions& functions);

/// The pNext chain for VkDeviceCreateInfo. Owns every struct it links, so the
/// chain stays valid for as long as the value does; pass &features2 as
/// VkDeviceCreateInfo::pNext.
struct FeatureChain {
    VkPhysicalDeviceFeatures2 features2{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
    VkPhysicalDeviceVulkan12Features vulkan12{};
    VkPhysicalDeviceVulkan13Features vulkan13{};
};

/// Fill @p chain in place: its pNext pointers reference its own members, so
/// it must be built where it will be used, never returned by value.
/// @p coreFeatures are the 1.0 features to enable; a zeroed value enables
/// none, which is what the SDK's own device has always done. A host whose
/// compositor reads its capabilities from the *supported* features (QRhi)
/// passes every supported bit except robustBufferAccess instead.
void BuildFeatureChain(FeatureChain& chain, const DeviceCapabilities& capabilities,
                       const VkPhysicalDeviceFeatures& coreFeatures);

bool IsDeviceSuitable(VkPhysicalDevice device, const DeviceFunctions& functions = {});

/// Graphics + compute family, the one queue the SDK records and submits on.
Optional<u32> FindGraphicsQueueFamily(VkPhysicalDevice device, const DeviceFunctions& functions = {});

}  // namespace quantiloom::vulkan
