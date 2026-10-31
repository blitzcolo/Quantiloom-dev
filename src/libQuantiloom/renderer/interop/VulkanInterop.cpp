#include "renderer/interop/VulkanInterop.hpp"

#include "core/Log.hpp"
#include "renderer/VulkanDeviceRequirements.hpp"

#include <set>
#include <string>
#include <vector>

namespace quantiloom::vk {
namespace {

using CreateResult = Result<Device, String>;

struct Candidate {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    u32 queueFamily = 0;
    bool discrete = false;
};

/// Suitability is the SDK's own list; the host adds only its present check,
/// which the SDK cannot make for it (it has no surface).
Optional<Candidate> Select(const DeviceRequest& request,
                           const vulkan::DeviceFunctions& functions,
                           VkPhysicalDevice physicalDevice) {
    if (!vulkan::IsDeviceSuitable(physicalDevice, functions)) return std::nullopt;
    const auto family = vulkan::FindGraphicsQueueFamily(physicalDevice, functions);
    if (!family.has_value()) return std::nullopt;
    if (request.supportsPresent &&
        !request.supportsPresent(request.supportsPresentUser, physicalDevice, *family)) {
        QL_LOG_INFO("  Skipping a device the host cannot present from (queue family {})", *family);
        return std::nullopt;
    }
    VkPhysicalDeviceProperties properties{};
    functions.GetPhysicalDeviceProperties(physicalDevice, &properties);
    return Candidate{physicalDevice, *family,
                     properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU};
}

CreateResult CheckHostExtensions(VkPhysicalDevice physicalDevice,
                                 const vulkan::DeviceFunctions& functions,
                                 const DeviceRequest& request,
                                 std::vector<const char*>& enabled) {
    u32 count = 0;
    functions.EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    functions.EnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, available.data());

    std::set<std::string> supported;
    for (const auto& extension : available) supported.insert(extension.extensionName);
    std::set<std::string> already(enabled.begin(), enabled.end());

    for (u32 i = 0; i < request.requiredHostDeviceExtensionCount; ++i) {
        const char* name = request.requiredHostDeviceExtensions[i];
        if (!supported.contains(name)) {
            return CreateResult::Err(String("required host device extension is not supported: ") +
                                     name);
        }
        if (already.insert(name).second) enabled.push_back(name);
    }
    for (u32 i = 0; i < request.optionalDeviceExtensionCount; ++i) {
        const char* name = request.optionalDeviceExtensions[i];
        if (supported.contains(name) && already.insert(name).second) enabled.push_back(name);
    }
    return CreateResult(Device{});
}

}  // namespace

Result<Device, String> CreateDevice(const DeviceRequest& request) {
    if (request.instance == VK_NULL_HANDLE) {
        return CreateResult::Err("vk::CreateDevice: request.instance is VK_NULL_HANDLE");
    }
    const vulkan::DeviceFunctions functions =
        vulkan::LoadDeviceFunctions(request.instance, request.getInstanceProcAddr);

    u32 deviceCount = 0;
    if (functions.EnumeratePhysicalDevices(request.instance, &deviceCount, nullptr) != VK_SUCCESS ||
        deviceCount == 0) {
        return CreateResult::Err("vk::CreateDevice: no Vulkan physical devices");
    }
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    functions.EnumeratePhysicalDevices(request.instance, &deviceCount, physicalDevices.data());

    Optional<Candidate> chosen;
    if (request.preferredPhysicalDevice != VK_NULL_HANDLE) {
        chosen = Select(request, functions, request.preferredPhysicalDevice);
        if (!chosen.has_value()) {
            return CreateResult::Err(
                "vk::CreateDevice: the preferred physical device does not meet the SDK's "
                "requirements (ray tracing extensions and features, a graphics+compute family, "
                "and the host's present check)");
        }
    } else {
        Optional<Candidate> firstPassing;
        for (const VkPhysicalDevice physicalDevice : physicalDevices) {
            auto candidate = Select(request, functions, physicalDevice);
            if (!candidate.has_value()) continue;
            if (candidate->discrete) {
                chosen = candidate;
                break;
            }
            if (!firstPassing.has_value()) firstPassing = candidate;
        }
        if (!chosen.has_value()) chosen = firstPassing;
        if (!chosen.has_value()) {
            return CreateResult::Err(
                "vk::CreateDevice: no physical device meets the SDK's requirements "
                "(ray tracing extensions and features, a graphics+compute family, "
                "and the host's present check)");
        }
    }

    std::vector<const char*> extensions = vulkan::RequiredDeviceExtensions();
    if (auto checked = CheckHostExtensions(chosen->physicalDevice, functions, request, extensions);
        !checked.has_value()) {
        return CreateResult::Err(checked.error());
    }

    const vulkan::DeviceCapabilities capabilities =
        vulkan::QueryDeviceCapabilities(chosen->physicalDevice, functions);

    // Every supported 1.0 feature except robustBufferAccess: a compositor
    // that imports the device (QRhi) reads its capabilities from the
    // *supported* features and would otherwise act on one that was never
    // enabled.
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    functions.GetPhysicalDeviceFeatures2(chosen->physicalDevice, &supportedFeatures);
    supportedFeatures.features.robustBufferAccess = VK_FALSE;

    vulkan::FeatureChain chain;
    vulkan::BuildFeatureChain(chain, capabilities, supportedFeatures.features);

    const f32 queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = chosen->queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &chain.features2;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledExtensionCount = static_cast<u32>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    Device result;
    result.instance = request.instance;
    result.physicalDevice = chosen->physicalDevice;
    result.queueFamily = chosen->queueFamily;
    result.queueIndex = 0;
    const VkResult created =
        functions.CreateDevice(chosen->physicalDevice, &createInfo, nullptr, &result.device);
    if (created != VK_SUCCESS) {
        return CreateResult::Err("vk::CreateDevice: vkCreateDevice failed with VkResult " +
                                 std::to_string(static_cast<int>(created)));
    }
    vkGetDeviceQueue(result.device, result.queueFamily, result.queueIndex, &result.queue);

    VkPhysicalDeviceProperties properties{};
    functions.GetPhysicalDeviceProperties(chosen->physicalDevice, &properties);
    QL_LOG_INFO("vk::CreateDevice: {} (queue family {}, {} device extensions)",
                properties.deviceName, result.queueFamily, extensions.size());
    return CreateResult(result);
}

void DestroyDevice(const Device& device, PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
    if (device.device == VK_NULL_HANDLE) return;
    PFN_vkDestroyDevice destroy = &vkDestroyDevice;
    if (getInstanceProcAddr) {
        if (const PFN_vkVoidFunction resolved =
                getInstanceProcAddr(device.instance, "vkDestroyDevice")) {
            destroy = reinterpret_cast<PFN_vkDestroyDevice>(resolved);
        }
    }
    destroy(device.device, nullptr);
}

}  // namespace quantiloom::vk
