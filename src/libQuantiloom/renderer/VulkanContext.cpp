// VMA must be included FIRST to define VMA_H
#include <vk_mem_alloc.h>

#include "VulkanContext.hpp"
#include "renderer/VulkanDeviceRequirements.hpp"
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
    m_capabilities.hasRayTracing = true;

    // ========================================================================
    // Query device capabilities (CRITICAL for descriptor indexing support)
    // ========================================================================
    // This was missing before, causing hasDescriptorIndexing to remain false
    // and texture limit to be reduced to 32 instead of 1024.
    // ========================================================================

    // Check Vulkan 1.3 support
    if (m_deviceProperties.apiVersion >= VK_API_VERSION_1_3) {
        m_capabilities.hasVulkan13 = true;

        // Query VK 1.3 features availability
        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceFeatures2 features2_13{};
        features2_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2_13.pNext = &features13;

        vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2_13);

        m_capabilities.hasSynchronization2 = features13.synchronization2;
        m_capabilities.hasDynamicRendering = features13.dynamicRendering;
    }

    // Query VK 1.2 descriptor indexing features (required for bindless textures)
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2_12{};
    features2_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2_12.pNext = &features12;

    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2_12);

    // Descriptor indexing requires ALL 4 sub-features for full bindless support
    m_capabilities.hasDescriptorIndexing =
        features12.descriptorIndexing &&
        features12.runtimeDescriptorArray &&
        features12.descriptorBindingPartiallyBound &&
        features12.shaderSampledImageArrayNonUniformIndexing;

    QL_LOG_INFO("  Vulkan 1.3: {}", m_capabilities.hasVulkan13 ? "YES" : "NO");
    QL_LOG_INFO("  Descriptor Indexing: {}", m_capabilities.hasDescriptorIndexing ? "YES" : "NO");
    QL_LOG_INFO("  Synchronization2: {}", m_capabilities.hasSynchronization2 ? "YES" : "NO");
    QL_LOG_INFO("  Dynamic Rendering: {}", m_capabilities.hasDynamicRendering ? "YES" : "NO");

    if (!m_capabilities.hasDescriptorIndexing) {
        QL_LOG_WARN("Descriptor indexing NOT fully supported - bindless texture limit reduced to 32");
    }

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

    // Immediate-submission state, lazily created by CommandHelper::ExecuteImmediate
    if (m_device != VK_NULL_HANDLE) {
        if (m_immediateFence != VK_NULL_HANDLE) {
            vkDestroyFence(m_device, m_immediateFence, nullptr);
            m_immediateFence = VK_NULL_HANDLE;
        }
        if (m_immediateCommandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, m_immediateCommandPool, nullptr);
            m_immediateCommandPool = VK_NULL_HANDLE;
        }
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

    // The extension list and the feature chain are shared with the public
    // vk::CreateDevice() helper (renderer/VulkanDeviceRequirements.hpp), so a
    // windowed host and the CLI cannot drift into asking for different devices.
    const auto& deviceExtensions = vulkan::RequiredDeviceExtensions();
    if (m_capabilities.hasVulkan13) {
        if (m_capabilities.hasSynchronization2) {
            QL_LOG_INFO("  Enabling synchronization2 (for future use)");
        }
        if (m_capabilities.hasDynamicRendering) {
            QL_LOG_INFO("  Enabling dynamicRendering (for future use)");
        }
    }
    if (!m_capabilities.hasDescriptorIndexing) {
        QL_LOG_WARN("Descriptor indexing NOT fully supported - bindless texture limit reduced to 32");
    }

    // The SDK's own device enables no optional 1.0 feature, as it always has.
    vulkan::DeviceCapabilities capabilities{};
    capabilities.hasVulkan13 = m_capabilities.hasVulkan13;
    capabilities.hasSynchronization2 = m_capabilities.hasSynchronization2;
    capabilities.hasDynamicRendering = m_capabilities.hasDynamicRendering;
    capabilities.hasDescriptorIndexing = m_capabilities.hasDescriptorIndexing;
    vulkan::FeatureChain chain;
    vulkan::BuildFeatureChain(chain, capabilities, VkPhysicalDeviceFeatures{});

    // Device creation
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &chain.features2;
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
    return vulkan::IsDeviceSuitable(device);
}

Optional<u32> VulkanContext::FindGraphicsQueueFamily(VkPhysicalDevice device) {
    return vulkan::FindGraphicsQueueFamily(device);
}

} // namespace quantiloom
