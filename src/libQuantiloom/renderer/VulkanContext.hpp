#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <vulkan/vulkan.h>

// VMA configuration: use dynamic Vulkan functions
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <vector>

// ============================================================================
// VulkanContext - Centralized Vulkan lifecycle management
// ============================================================================
// Responsibilities:
// - Create and destroy VkInstance (with validation layers if enabled)
// - Select and create VkDevice (with required extensions/features)
// - Create and destroy VmaAllocator (for memory management)
// - Provide access to core Vulkan handles
//
// Lifetime:
// - Must outlive ALL resources (buffers, images, acceleration structures)
// - Typically created at application startup, destroyed at shutdown
// - Singleton-like usage (non-copyable, non-movable)
// ============================================================================

namespace quantiloom {

class QL_API VulkanContext {
public:
    // ========================================================================
    // Lifecycle
    // ========================================================================

    VulkanContext();
    ~VulkanContext();

    // Non-copyable, non-movable (singleton-like)
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    // ========================================================================
    // Accessors
    // ========================================================================

    VkInstance GetInstance() const { return m_instance; }
    VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    VkDevice GetDevice() const { return m_device; }
    VkQueue GetGraphicsQueue() const { return m_graphicsQueue; }
    u32 GetGraphicsQueueFamily() const { return m_graphicsQueueFamily; }

    VmaAllocator GetAllocator() const { return m_allocator; }

    // ========================================================================
    // Utility
    // ========================================================================

    // Get physical device properties
    const VkPhysicalDeviceProperties& GetDeviceProperties() const { return m_deviceProperties; }

    // Check if Ray Tracing is supported
    bool IsRayTracingSupported() const { return m_rayTracingSupported; }

    // Get Ray Tracing properties (only valid if IsRayTracingSupported() == true)
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& GetRayTracingProperties() const {
        return m_rtPipelineProperties;
    }

private:
    // ========================================================================
    // Initialization Steps
    // ========================================================================

    void CreateInstance();
    void SetupDebugMessenger();
    void SelectPhysicalDevice();
    void CreateDevice();
    void CreateAllocator();

    // ========================================================================
    // Helpers
    // ========================================================================

    // Get required instance extensions
    std::vector<const char*> GetRequiredInstanceExtensions() const;

    // Get required validation layers
    std::vector<const char*> GetRequiredValidationLayers() const;

    // Check if physical device is suitable (has required features)
    bool IsDeviceSuitable(VkPhysicalDevice device) const;

    // Find queue family index (graphics + compute + transfer)
    Optional<u32> FindGraphicsQueueFamily(VkPhysicalDevice device) const;

    // ========================================================================
    // Vulkan Handles (destruction order: reverse of declaration)
    // ========================================================================

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;

    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    u32 m_graphicsQueueFamily = 0;

    VkPhysicalDeviceProperties m_deviceProperties{};
    bool m_rayTracingSupported = false;

    // Ray Tracing properties (if supported)
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{};

    VmaAllocator m_allocator = VK_NULL_HANDLE;  // Last created, first destroyed

#ifdef QUANTILOOM_ENABLE_VALIDATION
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
#endif
};

} // namespace quantiloom
