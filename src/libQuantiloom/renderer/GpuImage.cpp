#include "GpuImage.hpp"
#include "core/Log.hpp"
#include <stdexcept>

namespace quantiloom {

// ============================================================================
// Construction / Destruction
// ============================================================================

GpuImage::GpuImage(VmaAllocator allocator, VkDevice device,
                   u32 width, u32 height,
                   VkFormat format,
                   VkImageUsageFlags usage,
                   VmaMemoryUsage memUsage,
                   u32 mipLevels)
    : m_allocator(allocator),
      m_device(device),
      m_format(format),
      m_extent{width, height},
      m_mipLevels(mipLevels) {

    // Create VkImage
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;

    VkResult result = vmaCreateImage(m_allocator, &imageInfo, &allocInfo,
                                     &m_image, &m_allocation, nullptr);

    if (result != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to create VkImage via VMA: error code {}", static_cast<int>(result));
        throw std::runtime_error("GpuImage creation failed");
    }

    // Create VkImageView
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    result = vkCreateImageView(m_device, &viewInfo, nullptr, &m_view);

    if (result != VK_SUCCESS) {
        // Clean up image if view creation fails
        vmaDestroyImage(m_allocator, m_image, m_allocation);
        QL_LOG_ERROR("Failed to create VkImageView: error code {}", static_cast<int>(result));
        throw std::runtime_error("GpuImage view creation failed");
    }
}

GpuImage::~GpuImage() {
    // Destruction order: view before image
    if (m_view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_view, nullptr);
    }

    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
    }
}

// ============================================================================
// Move Semantics
// ============================================================================

GpuImage::GpuImage(GpuImage&& other) noexcept
    : m_allocator(other.m_allocator),
      m_device(other.m_device),
      m_image(other.m_image),
      m_view(other.m_view),
      m_allocation(other.m_allocation),
      m_format(other.m_format),
      m_extent(other.m_extent),
      m_mipLevels(other.m_mipLevels) {

    // Nullify source to prevent double-free
    other.m_image = VK_NULL_HANDLE;
    other.m_view = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
}

GpuImage& GpuImage::operator=(GpuImage&& other) noexcept {
    if (this != &other) {
        // Destroy current resources
        if (m_view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_view, nullptr);
        }
        if (m_image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, m_image, m_allocation);
        }

        // Move from other
        m_allocator = other.m_allocator;
        m_device = other.m_device;
        m_image = other.m_image;
        m_view = other.m_view;
        m_allocation = other.m_allocation;
        m_format = other.m_format;
        m_extent = other.m_extent;
        m_mipLevels = other.m_mipLevels;

        // Nullify source
        other.m_image = VK_NULL_HANDLE;
        other.m_view = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace quantiloom
