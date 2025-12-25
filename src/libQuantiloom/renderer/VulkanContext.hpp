/**
 * @file VulkanContext.hpp
 * @brief Centralized Vulkan lifecycle management and device initialization
 *
 * Provides VulkanContext class for managing all Vulkan resources:
 * - VkInstance creation with validation layers (debug builds)
 * - Physical device selection (GPU with ray tracing support)
 * - Logical device creation with required extensions/features
 * - VmaAllocator for efficient memory management
 * - Queue management (graphics/compute/transfer)
 *
 * VulkanContext acts as the root of the Vulkan resource hierarchy:
 * - MUST outlive ALL Vulkan resources (buffers, images, acceleration structures, pipelines)
 * - Singleton-like usage (non-copyable, non-movable)
 * - Created once at application startup, destroyed at shutdown
 *
 * Ray tracing requirements:
 * - VK_KHR_ray_tracing_pipeline extension
 * - VK_KHR_acceleration_structure extension
 * - VK_KHR_deferred_host_operations (for async AS builds)
 * - Ray query features (ray queries in compute/graphics shaders)
 *
 * @note This class is Windows/Linux cross-platform compatible
 * @note Requires Vulkan 1.3 or higher with ray tracing extensions
 * @note Uses VMA (Vulkan Memory Allocator) for automatic memory management
 *
 * @author wtflmao
 */

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

namespace quantiloom {

/**
 * @class VulkanContext
 * @brief Manages Vulkan instance, device, and memory allocator lifecycle
 *
 * Central initialization and cleanup for all Vulkan resources.
 * Provides access to core Vulkan handles for creating buffers, images, pipelines, etc.
 *
 * Initialization sequence:
 * 1. CreateInstance() - VkInstance with validation layers (debug)
 * 2. SetupDebugMessenger() - Validation error reporting
 * 3. SelectPhysicalDevice() - Choose GPU with ray tracing support
 * 4. CreateDevice() - VkDevice with required extensions/features
 * 5. CreateAllocator() - VmaAllocator for memory management
 *
 * Resource destruction order (reverse of creation):
 * 1. VmaAllocator destroyed (frees all allocations)
 * 2. VkDevice destroyed (releases GPU resources)
 * 3. VkDebugUtilsMessenger destroyed (stops validation)
 * 4. VkInstance destroyed (releases Vulkan driver)
 *
 * Usage example:
 * @code
 * // Create context (initializes Vulkan)
 * VulkanContext context;
 *
 * if (!context.IsRayTracingSupported()) {
 *     QL_LOG_ERROR("Ray tracing not supported!");
 *     return;
 * }
 *
 * // Create resources using context
 * GpuBuffer vertexBuffer(context.GetAllocator(), size, usage, memUsage);
 * GpuImage outputImage(context.GetAllocator(), context.GetDevice(), width, height, format, usage, memUsage);
 *
 * // Resources MUST be destroyed before context goes out of scope
 * // (automatic via RAII destructors)
 * @endcode
 *
 * @note Non-copyable, non-movable (singleton-like pattern)
 * @note Must outlive ALL Vulkan resources created with it
 * @note Validation layers enabled in debug builds only
 * @note Aborts if ray tracing not supported (hard requirement)
 *
 * @see GpuBuffer for vertex/index/uniform buffer management
 * @see GpuImage for texture and render target management
 * @see AccelerationStructure for BLAS/TLAS ray tracing structures
 */
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

    [[nodiscard]] VkInstance GetInstance() const { return m_instance; }
    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    [[nodiscard]] VkDevice GetDevice() const { return m_device; }
    [[nodiscard]] VkQueue GetGraphicsQueue() const { return m_graphicsQueue; }
    [[nodiscard]] u32 GetGraphicsQueueFamily() const { return m_graphicsQueueFamily; }

    [[nodiscard]] VmaAllocator GetAllocator() const { return m_allocator; }

    // ========================================================================
    // Utility
    // ========================================================================

    // Get physical device properties
    [[nodiscard]] const VkPhysicalDeviceProperties& GetDeviceProperties() const { return m_deviceProperties; }

    // Check if Ray Tracing is supported
    [[nodiscard]] bool IsRayTracingSupported() const { return m_rayTracingSupported; }

    // Get Ray Tracing properties (only valid if IsRayTracingSupported() == true)
    [[nodiscard]] const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& GetRayTracingProperties() const {
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
    static std::vector<const char*> GetRequiredInstanceExtensions();

    // Get required validation layers
    static std::vector<const char*> GetRequiredValidationLayers();

    // Check if physical device is suitable (has required features)
    bool IsDeviceSuitable(VkPhysicalDevice device) const;

    // Find queue family index (graphics + compute + transfer)
    static Optional<u32> FindGraphicsQueueFamily(VkPhysicalDevice device);

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

    // Debug messenger - CRITICAL: Do NOT wrap in #ifdef to avoid ODR violation!
    // Different translation units may have different QUANTILOOM_ENABLE_VALIDATION
    // definitions, causing class layout mismatch and memory corruption.
    // Instead, keep the member always present and guard only the usage code.
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
};

} // namespace quantiloom
