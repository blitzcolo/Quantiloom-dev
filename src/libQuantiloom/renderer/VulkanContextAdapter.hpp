/**
 * @file VulkanContextAdapter.hpp
 * @brief Adapter class that creates VulkanContext from external Vulkan handles
 *
 * Provides VulkanContextAdapter for creating a VulkanContext that uses
 * externally-managed Vulkan resources (e.g., from Qt's QVulkanWindow).
 *
 * This adapter allows ExternalRenderContext to use existing RayTracingPipeline,
 * BLAS, and TLAS classes without modification by providing a VulkanContext
 * backed by externally-provided Vulkan handles.
 *
 * Key differences from VulkanContext:
 * - Does NOT create/destroy VkInstance, VkDevice, VkPhysicalDevice, VkQueue
 * - Can optionally create/destroy VmaAllocator if not provided externally
 * - All Vulkan handles are externally managed and must outlive this adapter
 *
 * Usage pattern:
 * @code
 * // In ExternalRenderContext::Initialize()
 * VulkanContext::ExternalHandles handles{};
 * handles.instance = params.instance;
 * handles.physicalDevice = params.physicalDevice;
 * handles.device = params.device;
 * handles.graphicsQueue = params.graphicsQueue;
 * handles.graphicsQueueFamily = params.graphicsQueueFamily;
 * handles.allocator = params.externalAllocator;
 *
 * m_contextAdapter = std::make_unique<VulkanContextAdapter>(handles);
 *
 * // Now use m_contextAdapter anywhere VulkanContext& is required
 * RayTracingPipeline pipeline(*m_contextAdapter, ...);
 * BLAS blas(*m_contextAdapter, primitive);
 * @endcode
 *
 * @note External handles MUST remain valid for lifetime of adapter
 * @note Thread safety: same as external context (typically not thread-safe)
 *
 * @author wtflmao
 */

#pragma once

#include "VulkanContext.hpp"

namespace quantiloom {

/**
 * @class VulkanContextAdapter
 * @brief Creates VulkanContext from external Vulkan handles
 *
 * This adapter derives from VulkanContext and uses the protected constructor
 * to create a context backed by externally-managed Vulkan resources.
 *
 * @note This is an internal implementation detail - external users should
 *       interact through ExternalRenderContext API
 */
class QL_API VulkanContextAdapter : public VulkanContext {
public:
    /**
     * @brief Construct adapter from external Vulkan handles
     * @param handles External Vulkan handles (must remain valid)
     * @param createAllocatorIfNull If true and handles.allocator is null, creates internal allocator
     * @throws std::runtime_error if required handles are null or invalid
     */
    explicit VulkanContextAdapter(const ExternalHandles& handles, bool createAllocatorIfNull = true);

    /**
     * @brief Destructor - only destroys internally-created allocator
     */
    ~VulkanContextAdapter() override = default;

    // Non-copyable, non-movable (references external handles)
    VulkanContextAdapter(const VulkanContextAdapter&) = delete;
    VulkanContextAdapter& operator=(const VulkanContextAdapter&) = delete;
    VulkanContextAdapter(VulkanContextAdapter&&) = delete;
    VulkanContextAdapter& operator=(VulkanContextAdapter&&) = delete;
};

} // namespace quantiloom
