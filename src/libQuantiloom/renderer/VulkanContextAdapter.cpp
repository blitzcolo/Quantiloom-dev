/**
 * @file VulkanContextAdapter.cpp
 * @brief Implementation of VulkanContextAdapter for external Vulkan handle injection
 *
 * @author wtflmao
 */

#include "VulkanContextAdapter.hpp"
#include "core/Log.hpp"
#include <stdexcept>

namespace quantiloom {

VulkanContextAdapter::VulkanContextAdapter(const ExternalHandles& handles, bool createAllocatorIfNull)
    : VulkanContext(handles, createAllocatorIfNull) {
    // Validate handles
    if (handles.instance == VK_NULL_HANDLE) {
        throw std::runtime_error("VulkanContextAdapter: VkInstance is null");
    }
    if (handles.physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("VulkanContextAdapter: VkPhysicalDevice is null");
    }
    if (handles.device == VK_NULL_HANDLE) {
        throw std::runtime_error("VulkanContextAdapter: VkDevice is null");
    }
    if (handles.graphicsQueue == VK_NULL_HANDLE) {
        throw std::runtime_error("VulkanContextAdapter: VkQueue is null");
    }

    QL_LOG_INFO("VulkanContextAdapter created successfully");
}

} // namespace quantiloom
