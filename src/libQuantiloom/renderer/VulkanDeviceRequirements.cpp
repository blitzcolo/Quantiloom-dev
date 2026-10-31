#include "renderer/VulkanDeviceRequirements.hpp"

#include "core/Log.hpp"

#include <set>
#include <string>

namespace quantiloom::vulkan {

namespace {

template <typename T>
void Resolve(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr,
             const char* name, T& target) {
    if (!getInstanceProcAddr) return;
    if (const PFN_vkVoidFunction resolved = getInstanceProcAddr(instance, name)) {
        target = reinterpret_cast<T>(resolved);
    }
}

}  // namespace

DeviceFunctions LoadDeviceFunctions(VkInstance instance,
                                    PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
    DeviceFunctions functions;
    Resolve(instance, getInstanceProcAddr, "vkEnumeratePhysicalDevices",
            functions.EnumeratePhysicalDevices);
    Resolve(instance, getInstanceProcAddr, "vkGetPhysicalDeviceProperties",
            functions.GetPhysicalDeviceProperties);
    Resolve(instance, getInstanceProcAddr, "vkGetPhysicalDeviceFeatures2",
            functions.GetPhysicalDeviceFeatures2);
    Resolve(instance, getInstanceProcAddr, "vkGetPhysicalDeviceQueueFamilyProperties",
            functions.GetPhysicalDeviceQueueFamilyProperties);
    Resolve(instance, getInstanceProcAddr, "vkEnumerateDeviceExtensionProperties",
            functions.EnumerateDeviceExtensionProperties);
    Resolve(instance, getInstanceProcAddr, "vkCreateDevice", functions.CreateDevice);
    return functions;
}

const std::vector<const char*>& RequiredDeviceExtensions() {
    static const std::vector<const char*> extensions = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,  // Required if shaders use RayQuery capability
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    };
    return extensions;
}

DeviceCapabilities QueryDeviceCapabilities(VkPhysicalDevice device,
                                           const DeviceFunctions& functions) {
    DeviceCapabilities capabilities;

    VkPhysicalDeviceProperties properties{};
    functions.GetPhysicalDeviceProperties(device, &properties);
    capabilities.hasVulkan13 = properties.apiVersion >= VK_API_VERSION_1_3;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    functions.GetPhysicalDeviceFeatures2(device, &features2);

    if (capabilities.hasVulkan13) {
        capabilities.hasSynchronization2 = features13.synchronization2 == VK_TRUE;
        capabilities.hasDynamicRendering = features13.dynamicRendering == VK_TRUE;
    }
    capabilities.hasDescriptorIndexing =
        features12.descriptorIndexing == VK_TRUE &&
        features12.runtimeDescriptorArray == VK_TRUE &&
        features12.descriptorBindingPartiallyBound == VK_TRUE &&
        features12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;
    return capabilities;
}

void BuildFeatureChain(FeatureChain& chain, const DeviceCapabilities& capabilities,
                       const VkPhysicalDeviceFeatures& coreFeatures) {
    chain = FeatureChain{};

    chain.vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    // Conditional, and not currently used by the SDK ("ghost" features): a
    // host may want them, so they are enabled where the device offers them.
    if (capabilities.hasVulkan13) {
        chain.vulkan13.synchronization2 = capabilities.hasSynchronization2 ? VK_TRUE : VK_FALSE;
        chain.vulkan13.dynamicRendering = capabilities.hasDynamicRendering ? VK_TRUE : VK_FALSE;
    }

    chain.vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    chain.vulkan12.bufferDeviceAddress = VK_TRUE;  // MANDATORY for RT
    chain.vulkan12.descriptorIndexing = VK_TRUE;
    chain.vulkan12.runtimeDescriptorArray = VK_TRUE;
    chain.vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
    chain.vulkan12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    chain.vulkan12.pNext = &chain.vulkan13;

    chain.rayTracingPipeline.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    chain.rayTracingPipeline.rayTracingPipeline = VK_TRUE;
    chain.rayTracingPipeline.pNext = &chain.vulkan12;

    chain.rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    chain.rayQuery.rayQuery = VK_TRUE;
    chain.rayQuery.pNext = &chain.rayTracingPipeline;

    chain.accelerationStructure.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    chain.accelerationStructure.accelerationStructure = VK_TRUE;
    chain.accelerationStructure.pNext = &chain.rayQuery;

    chain.features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    chain.features2.pNext = &chain.accelerationStructure;
    chain.features2.features = coreFeatures;
}

bool IsDeviceSuitable(VkPhysicalDevice device, const DeviceFunctions& functions) {
    VkPhysicalDeviceProperties deviceProperties{};
    functions.GetPhysicalDeviceProperties(device, &deviceProperties);

    // Query Vulkan 1.2 features for validation
    VkPhysicalDeviceVulkan12Features features12Query{};
    features12Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2Query{};
    features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2Query.pNext = &features12Query;

    functions.GetPhysicalDeviceFeatures2(device, &features2Query);

    // bufferDeviceAddress is MANDATORY for Ray Tracing (BLAS/TLAS addressing)
    if (!features12Query.bufferDeviceAddress) {
        QL_LOG_WARN("  Skipping {}: Missing bufferDeviceAddress (required for Ray Tracing)", deviceProperties.deviceName);
        return false;
    }

    // Warn about integrated GPU (not blocking, just informational)
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        QL_LOG_WARN("  Integrated GPU detected: {} (may have reduced performance)", deviceProperties.deviceName);
    }

    // Check for graphics queue
    auto queueFamily = FindGraphicsQueueFamily(device, functions);
    if (!queueFamily.has_value()) {
        QL_LOG_WARN("  Skipping {}: No suitable queue family", deviceProperties.deviceName);
        return false;
    }

    // Check for required Ray Tracing extensions
    u32 extensionCount = 0;
    functions.EnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    functions.EnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions;
    for (const char* name : RequiredDeviceExtensions()) requiredExtensions.insert(name);

    for (const auto&[extensionName, specVersion] : availableExtensions) {
        requiredExtensions.erase(extensionName);
    }

    if (!requiredExtensions.empty()) {
        QL_LOG_WARN("  Skipping {}: Missing Ray Tracing extensions:", deviceProperties.deviceName);
        for (const auto& missing : requiredExtensions) {
            QL_LOG_WARN("    - {}", missing);
        }
        return false;
    }

    // Check for Ray Tracing features
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rtPipelineFeatures.pNext = &rayQueryFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtPipelineFeatures;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &asFeatures;

    functions.GetPhysicalDeviceFeatures2(device, &features2);

    if (!asFeatures.accelerationStructure || !rtPipelineFeatures.rayTracingPipeline) {
        QL_LOG_WARN("  Skipping {}: Ray Tracing features not supported", deviceProperties.deviceName);
        QL_LOG_WARN("    - Acceleration Structure: {}", asFeatures.accelerationStructure ? "YES" : "NO");
        QL_LOG_WARN("    - Ray Tracing Pipeline: {}", rtPipelineFeatures.rayTracingPipeline ? "YES" : "NO");
        return false;
    }

    // The extension being present does not imply the feature is supported;
    // enabling rayQuery unconditionally here used to fail vkCreateDevice on
    // devices that expose VK_KHR_ray_query without the feature.
    if (!rayQueryFeatures.rayQuery) {
        QL_LOG_WARN("  Skipping {}: rayQuery feature not supported", deviceProperties.deviceName);
        return false;
    }

    QL_LOG_INFO("  Checking {}: All requirements met", deviceProperties.deviceName);
    return true;
}

Optional<u32> FindGraphicsQueueFamily(VkPhysicalDevice device, const DeviceFunctions& functions) {
    u32 queueFamilyCount = 0;
    functions.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    functions.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (u32 i = 0; i < queueFamilyCount; ++i) {
        // Look for graphics + compute + transfer
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
            queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return std::nullopt;
}

}  // namespace quantiloom::vulkan
