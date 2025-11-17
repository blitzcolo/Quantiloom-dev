#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

// ============================================================================
// GpuImage - RAII wrapper for VkImage with VMA allocation
// ============================================================================
// Responsibilities:
// - Create VkImage with VMA memory allocation
// - Create VkImageView for shader access
// - Automatically destroy on destruction (RAII)
// - Movable but non-copyable (strict ownership)
//
// Usage:
//   GpuImage renderTarget(allocator, device, width, height,
//       VK_FORMAT_R32G32B32A32_SFLOAT,
//       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
//
// Note:
// - Unlike GpuBuffer, images cannot be directly mapped
// - Data transfer requires staging buffers and vkCmdCopyBufferToImage
// - Layout transitions are managed externally via pipeline barriers
// ============================================================================

namespace quantiloom {

class QL_API GpuImage {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    // Create 2D image with VMA
    GpuImage(VmaAllocator allocator, VkDevice device,
             u32 width, u32 height,
             VkFormat format,
             VkImageUsageFlags usage,
             VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_GPU_ONLY,
             u32 mipLevels = 1);

    // Destructor: automatically destroys VkImage, VkImageView, and VmaAllocation
    ~GpuImage();

    // ========================================================================
    // Move Semantics (non-copyable)
    // ========================================================================

    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;

    GpuImage(GpuImage&& other) noexcept;
    GpuImage& operator=(GpuImage&& other) noexcept;

    // ========================================================================
    // Accessors
    // ========================================================================

    VkImage GetImage() const { return m_image; }
    VkImageView GetView() const { return m_view; }
    VkFormat GetFormat() const { return m_format; }
    VkExtent2D GetExtent() const { return m_extent; }
    u32 GetMipLevels() const { return m_mipLevels; }
    bool IsValid() const { return m_image != VK_NULL_HANDLE; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;

    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;

    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent{};
    u32 m_mipLevels = 1;
};

} // namespace quantiloom
