// VMA must be included FIRST to define VMA_H
#include <vk_mem_alloc.h>

#include "VulkanContext.hpp"
#include "core/Log.hpp"
#include <stdexcept>
#include <set>

namespace quantiloom {

// ============================================================================
// Debug Messenger Callback
// ============================================================================

#ifdef QUANTILOOM_ENABLE_VALIDATION
static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    (void)messageType;
    (void)pUserData;

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        QL_LOG_WARN("[Vulkan Validation] {}", pCallbackData->pMessage);
    } else {
        QL_LOG_INFO("[Vulkan Validation] {}", pCallbackData->pMessage);
    }

    return VK_FALSE;  // Do not abort
}
#endif

// ============================================================================
// Constructor / Destructor
// ============================================================================

VulkanContext::VulkanContext() {
    QL_LOG_INFO("Initializing Vulkan context...");

    CreateInstance();
    SetupDebugMessenger();
    SelectPhysicalDevice();
    CreateDevice();
    CreateAllocator();

    QL_LOG_INFO("Vulkan context initialized successfully");
}

// ============================================================================
// Protected Constructor (External Handles - for Adapter Pattern)
// ============================================================================

VulkanContext::VulkanContext(const ExternalHandles& handles, bool createAllocatorIfNull) {
    QL_LOG_INFO("Initializing VulkanContext from external handles...");

    // External handles - we do NOT own VkInstance/VkDevice
    m_ownsVulkanHandles = false;

    // Directly assign external handles (not owned by us)
    m_instance = handles.instance;
    m_physicalDevice = handles.physicalDevice;
    m_device = handles.device;
    m_graphicsQueue = handles.graphicsQueue;
    m_graphicsQueueFamily = handles.graphicsQueueFamily;
    m_debugMessenger = VK_NULL_HANDLE;  // External context manages debug

    // Query device properties
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);

    // Query Ray Tracing properties
    m_rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    m_asProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    m_asProperties.pNext = &m_rtPipelineProperties;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &m_asProperties;

    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

    // Clear pNext pointers after query
    m_asProperties.pNext = nullptr;
    m_rtPipelineProperties.pNext = nullptr;

    m_rayTracingSupported = true;  // Assume external context has RT support

    // Handle allocator
    if (handles.allocator != VK_NULL_HANDLE) {
        m_allocator = handles.allocator;
        m_ownsAllocator = false;
        QL_LOG_INFO("  Using external VMA allocator");
    } else if (createAllocatorIfNull) {
        // Create internal VMA allocator
        VmaVulkanFunctions vulkanFunctions{};
        vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorInfo{};
        // Use VK 1.3 if available, otherwise VK 1.2
        u32 apiVersion = m_capabilities.hasVulkan13 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;
        allocatorInfo.vulkanApiVersion = apiVersion;
        allocatorInfo.instance = m_instance;
        allocatorInfo.physicalDevice = m_physicalDevice;
        allocatorInfo.device = m_device;
        allocatorInfo.pVulkanFunctions = &vulkanFunctions;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        VkResult result = vmaCreateAllocator(&allocatorInfo, &m_allocator);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VMA allocator for external context");
        }
        m_ownsAllocator = true;
        QL_LOG_INFO("  Created internal VMA allocator for external context");
    } else {
        m_allocator = VK_NULL_HANDLE;
        m_ownsAllocator = false;
    }

    QL_LOG_INFO("VulkanContext initialized from external handles:");
    QL_LOG_INFO("  Device: {}", m_deviceProperties.deviceName);
    QL_LOG_INFO("  Max RT recursion: {}", m_rtPipelineProperties.maxRayRecursionDepth);
}

VulkanContext::~VulkanContext() {
    QL_LOG_INFO("Destroying Vulkan context...");

    // Wait for all GPU operations to complete before destroying resources
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    // Destruction order: reverse of construction
    // VMA allocator must be destroyed BEFORE device (only if we own it)
    if (m_ownsAllocator && m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }

    // Only destroy VkDevice/VkInstance if we own them (not for external handles)
    if (m_ownsVulkanHandles) {
        if (m_device != VK_NULL_HANDLE) {
            vkDestroyDevice(m_device, nullptr);
            m_device = VK_NULL_HANDLE;
        }

        // Destroy debug messenger if it was created
        if (m_debugMessenger != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
            auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
                m_instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (func != nullptr) {
                func(m_instance, m_debugMessenger, nullptr);
            }
            m_debugMessenger = VK_NULL_HANDLE;
        }

        if (m_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_instance, nullptr);
            m_instance = VK_NULL_HANDLE;
        }
    }

    QL_LOG_INFO("Vulkan context destroyed");
}

// ============================================================================
// Instance Creation
// ============================================================================

void VulkanContext::CreateInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Quantiloom";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    appInfo.pEngineName = "Quantiloom HS-core";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    appInfo.apiVersion = VK_API_VERSION_1_2;  // Vulkan 1.2 minimum (RT extensions available)

    const auto extensions = GetRequiredInstanceExtensions();
    const auto layers = GetRequiredValidationLayers();

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<u32>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<u32>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    if (const VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance); result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    QL_LOG_INFO("Vulkan instance created (API version 1.2+)");
}

// ============================================================================
// Debug Messenger Setup
// ============================================================================

void VulkanContext::SetupDebugMessenger() {
#ifdef QUANTILOOM_ENABLE_VALIDATION
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));

    if (func != nullptr) {
        VkResult result = func(m_instance, &createInfo, nullptr, &m_debugMessenger);
        if (result == VK_SUCCESS) {
            QL_LOG_INFO("Vulkan validation layers enabled");
        }
    }
#endif
}

// ============================================================================
// Physical Device Selection
// ============================================================================

void VulkanContext::SelectPhysicalDevice() {
    u32 deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error(
            "No Vulkan-compatible GPUs found. Please ensure:\n"
            "  1. Latest GPU drivers are installed\n"
            "  2. Vulkan SDK is properly configured\n"
            "  3. GPU supports Vulkan 1.3+"
        );
    }

    QL_LOG_INFO("Found {} Vulkan device(s), checking compatibility...", deviceCount);

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    // Select first suitable device
    for (const auto& device : devices) {
        if (IsDeviceSuitable(device)) {
            m_physicalDevice = device;
            vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);

            // Query Ray Tracing properties
            m_rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            m_asProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
            m_asProperties.pNext = &m_rtPipelineProperties;

            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &m_asProperties;

            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

            // CRITICAL: Clear pNext pointers after query to prevent stack corruption detection
            // The pNext chain is only needed during the query; keeping it alive can confuse
            // MSVC's stack guard in debug builds, as it detects cross-member pointers
            m_asProperties.pNext = nullptr;
            m_rtPipelineProperties.pNext = nullptr;

            // ========================================================================
            // Detect Device Capabilities (Runtime Feature Detection)
            // ========================================================================

            m_capabilities.hasRayTracing = true;  // Already validated in IsDeviceSuitable()

            // Check Vulkan 1.3 support
            if (m_deviceProperties.apiVersion >= VK_API_VERSION_1_3) {
                m_capabilities.hasVulkan13 = true;

                // Query VK 1.3 features availability
                VkPhysicalDeviceVulkan13Features features13{};
                features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

                VkPhysicalDeviceFeatures2 features2{};
                features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                features2.pNext = &features13;

                vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);

                m_capabilities.hasSynchronization2 = features13.synchronization2;
                m_capabilities.hasDynamicRendering = features13.dynamicRendering;
            }

            // Query VK 1.2 descriptor indexing features
            VkPhysicalDeviceVulkan12Features features12{};
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &features12;

            vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);

            // Descriptor indexing requires ALL 4 sub-features for full bindless support
            m_capabilities.hasDescriptorIndexing =
                features12.descriptorIndexing &&
                features12.runtimeDescriptorArray &&
                features12.descriptorBindingPartiallyBound &&
                features12.shaderSampledImageArrayNonUniformIndexing;

            // Log detailed info
            QL_LOG_INFO("========================================");
            QL_LOG_INFO("Selected GPU: {}", m_deviceProperties.deviceName);
            QL_LOG_INFO("  API version: {}.{}.{}",
                VK_VERSION_MAJOR(m_deviceProperties.apiVersion),
                VK_VERSION_MINOR(m_deviceProperties.apiVersion),
                VK_VERSION_PATCH(m_deviceProperties.apiVersion));
            QL_LOG_INFO("  Driver version: {}.{}.{}",
                VK_VERSION_MAJOR(m_deviceProperties.driverVersion),
                VK_VERSION_MINOR(m_deviceProperties.driverVersion),
                VK_VERSION_PATCH(m_deviceProperties.driverVersion));
            QL_LOG_INFO("Ray Tracing Capabilities:");
            QL_LOG_INFO("  Max recursion depth: {}", m_rtPipelineProperties.maxRayRecursionDepth);
            QL_LOG_INFO("  Shader group handle size: {}", m_rtPipelineProperties.shaderGroupHandleSize);
            QL_LOG_INFO("  Max geometry count: {}", m_asProperties.maxGeometryCount);
            QL_LOG_INFO("Device Capabilities:");
            QL_LOG_INFO("  Vulkan 1.3: {}", m_capabilities.hasVulkan13 ? "YES" : "NO");
            QL_LOG_INFO("  Descriptor Indexing: {}", m_capabilities.hasDescriptorIndexing ? "YES" : "NO");
            QL_LOG_INFO("  Synchronization2: {}", m_capabilities.hasSynchronization2 ? "YES" : "NO");
            QL_LOG_INFO("  Dynamic Rendering: {}", m_capabilities.hasDynamicRendering ? "YES" : "NO");
            QL_LOG_INFO("========================================");

            return;
        }
    }

    // No suitable device found - provide helpful error message
    std::string errorMsg =
        "No GPU with Ray Tracing support found.\n\n"
        "Quantiloom requires:\n"
        "  - Vulkan 1.2 or newer (1.3+ recommended)\n"
        "  - VK_KHR_ray_tracing_pipeline extension\n"
        "  - VK_KHR_acceleration_structure extension\n"
        "  - VK_KHR_buffer_device_address (VK 1.2 core feature)\n\n"
        "Supported GPUs:\n"
        "  - NVIDIA RTX 20xx series or newer (driver 450+)\n"
        "  - AMD RX 6000 series or newer (driver 21.10+)\n"
        "  - Intel Arc A-series (driver 30.0.100+)\n\n"
        "Note: Integrated GPUs (Intel Iris Xe, AMD Vega) typically do NOT support hardware RT.";

    throw std::runtime_error(errorMsg);
}

// ============================================================================
// Device Creation
// ============================================================================

void VulkanContext::CreateDevice() {
    auto queueFamilyOpt = FindGraphicsQueueFamily(m_physicalDevice);
    if (!queueFamilyOpt.has_value()) {
        throw std::runtime_error("No graphics queue family found");
    }

    m_graphicsQueueFamily = *queueFamilyOpt;

    // Queue creation
    f32 queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Required extensions for ray tracing
    std::vector<const char*> deviceExtensions = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,  // Required if shaders use RayQuery capability
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    };

    // Enable Vulkan 1.3 features (conditional - only if device supports)
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    // Conditionally enable VK 1.3 features if available
    // NOTE: These are NOT currently used (ghost features), but enabled for future-proofing
    if (m_capabilities.hasVulkan13) {
        if (m_capabilities.hasSynchronization2) {
            features13.synchronization2 = VK_TRUE;
            QL_LOG_INFO("  Enabling synchronization2 (for future use)");
        }
        if (m_capabilities.hasDynamicRendering) {
            features13.dynamicRendering = VK_TRUE;
            QL_LOG_INFO("  Enabling dynamicRendering (for future use)");
        }
    }

    // Enable Vulkan 1.2 features (MANDATORY)
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;  // MANDATORY for RT
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;  // Required for PARTIALLY_BOUND_BIT
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;  // Required for NonUniformResourceIndex
    features12.pNext = &features13;

    // Log warning if descriptor indexing not fully supported
    if (!m_capabilities.hasDescriptorIndexing) {
        QL_LOG_WARN("Descriptor indexing NOT fully supported - bindless texture limit reduced to 32");
    }

    // Enable Ray Tracing features
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;
    rtPipelineFeatures.pNext = &features12;

    // Enable Ray Query features (required if shaders declare RayQueryKHR capability)
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;
    rayQueryFeatures.pNext = &rtPipelineFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;
    asFeatures.pNext = &rayQueryFeatures;

    // Device features
    VkPhysicalDeviceFeatures2 deviceFeatures{};
    deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures.pNext = &asFeatures;

    // Device creation
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &deviceFeatures;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledExtensionCount = static_cast<u32>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device");
    }

    // Get queue handle
    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);

    m_rayTracingSupported = true;
    QL_LOG_INFO("Vulkan device created with Ray Tracing support");
}

// ============================================================================
// VMA Allocator Creation
// ============================================================================

void VulkanContext::CreateAllocator() {
    // VMA requires function pointers (dynamic loading)
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    // Use VK 1.3 if available, otherwise VK 1.2
    u32 apiVersion = m_capabilities.hasVulkan13 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;
    allocatorInfo.vulkanApiVersion = apiVersion;
    allocatorInfo.instance = m_instance;
    allocatorInfo.physicalDevice = m_physicalDevice;
    allocatorInfo.device = m_device;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;  // Required for ray tracing

    if (const VkResult result = vmaCreateAllocator(&allocatorInfo, &m_allocator); result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }

    QL_LOG_INFO("VMA allocator created");
}

// ============================================================================
// Helpers
// ============================================================================

std::vector<const char*> VulkanContext::GetRequiredInstanceExtensions() {
    std::vector<const char*> extensions;

#ifdef QUANTILOOM_ENABLE_VALIDATION
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    // Ray tracing requires additional instance extensions
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    return extensions;
}

std::vector<const char*> VulkanContext::GetRequiredValidationLayers() {
    std::vector<const char*> layers;

#ifdef QUANTILOOM_ENABLE_VALIDATION
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    return layers;
}

bool VulkanContext::IsDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    // Query Vulkan 1.2 features for validation
    VkPhysicalDeviceVulkan12Features features12Query{};
    features12Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2Query{};
    features2Query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2Query.pNext = &features12Query;

    vkGetPhysicalDeviceFeatures2(device, &features2Query);

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
    auto queueFamily = FindGraphicsQueueFamily(device);
    if (!queueFamily.has_value()) {
        QL_LOG_WARN("  Skipping {}: No suitable queue family", deviceProperties.deviceName);
        return false;
    }

    // Check for required Ray Tracing extensions
    u32 extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    };

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

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtPipelineFeatures;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &asFeatures;

    vkGetPhysicalDeviceFeatures2(device, &features2);

    if (!asFeatures.accelerationStructure || !rtPipelineFeatures.rayTracingPipeline) {
        QL_LOG_WARN("  Skipping {}: Ray Tracing features not supported", deviceProperties.deviceName);
        QL_LOG_WARN("    - Acceleration Structure: {}", asFeatures.accelerationStructure ? "YES" : "NO");
        QL_LOG_WARN("    - Ray Tracing Pipeline: {}", rtPipelineFeatures.rayTracingPipeline ? "YES" : "NO");
        return false;
    }

    QL_LOG_INFO("  Checking {}: All requirements met", deviceProperties.deviceName);
    return true;
}

Optional<u32> VulkanContext::FindGraphicsQueueFamily(VkPhysicalDevice device) {
    u32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (u32 i = 0; i < queueFamilyCount; ++i) {
        // Look for graphics + compute + transfer
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
            queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return std::nullopt;
}

} // namespace quantiloom
