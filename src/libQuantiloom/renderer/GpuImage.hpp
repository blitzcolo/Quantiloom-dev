/**
 * @file GpuImage.hpp
 * @brief RAII wrapper for VkImage with VMA memory allocation and automatic view creation
 *
 * Provides GpuImage class for managing Vulkan image resources:
 * - Automatic VkImage creation with VMA memory allocation
 * - Automatic VkImageView creation for shader access
 * - RAII lifecycle (automatic destruction on scope exit)
 * - Mipmap support (for environment maps, texture filtering)
 *
 * Image types supported:
 * - 2D images (textures, render targets)
 * - Storage images (compute shader output)
 * - Sampled images (texture sampling in shaders)
 * - Transfer src/dst (for GPU-CPU data transfer)
 *
 * Common formats:
 * - VK_FORMAT_R32G32B32A32_SFLOAT: HDR render targets (128 bpp)
 * - VK_FORMAT_R8G8B8A8_UNORM: LDR textures (32 bpp)
 * - VK_FORMAT_R8G8B8A8_SRGB: sRGB textures with gamma (32 bpp)
 * - VK_FORMAT_R32G32_SFLOAT: BRDF LUT (64 bpp)
 *
 * Data transfer:
 * Unlike GpuBuffer, images cannot be directly mapped to CPU memory.
 * Use staging buffers with vkCmdCopyBufferToImage/vkCmdCopyImageToBuffer.
 *
 * @note Movable but non-copyable (strict ownership semantics)
 * @note Image layout transitions managed externally via pipeline barriers
 * @note View is created automatically (can't be changed after construction)
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <vulkan/vulkan.h>

// Forward declarations for VMA types (avoid including heavy vk_mem_alloc.h)
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

// VMA memory usage enum - only define if VMA header not already included
// Note: This enum is also declared in GpuBuffer.hpp - both are needed
//       since headers may be included independently
// The real VMA header defines AMD_VULKAN_MEMORY_ALLOCATOR_H
#if !defined(AMD_VULKAN_MEMORY_ALLOCATOR_H) && !defined(QUANTILOOM_VMA_MEMORY_USAGE_DEFINED)
#define QUANTILOOM_VMA_MEMORY_USAGE_DEFINED
typedef enum VmaMemoryUsage {
    VMA_MEMORY_USAGE_UNKNOWN = 0,
    VMA_MEMORY_USAGE_GPU_ONLY = 1,
    VMA_MEMORY_USAGE_CPU_ONLY = 2,
    VMA_MEMORY_USAGE_CPU_TO_GPU = 3,
    VMA_MEMORY_USAGE_GPU_TO_CPU = 4,
    VMA_MEMORY_USAGE_CPU_COPY = 5,
    VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6,
    VMA_MEMORY_USAGE_AUTO = 7,
    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8,
    VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9,
} VmaMemoryUsage;
#endif

// ============================================================================
// GpuImage - RAII wrapper for VkImage with VMA allocation
// ============================================================================

namespace quantiloom {

/**
 * @class GpuImage
 * @brief RAII-managed VkImage with automatic VMA allocation, view creation, and cleanup
 *
 * Provides safe management of Vulkan 2D image resources.
 * Images and image views are destroyed automatically when GpuImage goes out of scope.
 *
 * Memory management:
 * - Uses VMA for efficient allocation (selects appropriate heap automatically)
 * - VkImage and VkImageView created in constructor
 * - Both destroyed in destructor (no manual vkDestroyImage required)
 *
 * Image layouts:
 * Images are created in VK_IMAGE_LAYOUT_UNDEFINED and must be transitioned
 * to appropriate layouts before use:
 * - VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: For texture sampling
 * - VK_IMAGE_LAYOUT_GENERAL: For storage image read/write
 * - VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: For GPU-to-CPU copy
 * - VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: For CPU-to-GPU copy
 *
 * Usage example:
 * @code
 * // Create HDR render target
 * GpuImage renderTarget(
 *     allocator,
 *     device,
 *     1920, 1080,  // width, height
 *     VK_FORMAT_R32G32B32A32_SFLOAT,
 *     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
 *     VMA_MEMORY_USAGE_GPU_ONLY
 * );
 *
 * // Transition to GENERAL layout for compute shader write
 * CommandHelper::TransitionImageLayoutImmediate(
 *     context, renderTarget.GetImage(), renderTarget.GetFormat(),
 *     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL
 * );
 *
 * // Create texture with mipmaps
 * GpuImage envMap(
 *     allocator, device,
 *     512, 512,  // width, height
 *     VK_FORMAT_R8G8B8A8_SRGB,
 *     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
 *     VMA_MEMORY_USAGE_GPU_ONLY,
 *     5  // mip levels
 * );
 * @endcode
 *
 * @note Non-copyable (prevents double-free of VkImage)
 * @note Movable (allows std::vector<GpuImage> and return values)
 * @note Cannot be mapped directly (use staging buffers for data transfer)
 * @note Image layout transitions require pipeline barriers (see CommandHelper)
 *
 * @see GpuBuffer for buffer resource management
 * @see CommandHelper::TransitionImageLayout for layout transitions
 * @see TextureManager for texture upload from CPU Image data
 */
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

    // Create cubemap image (6 array layers, cube-compatible view)
    // Note: arrayLayers and flags parameters allow flexibility
    GpuImage(VmaAllocator allocator, VkDevice device,
             u32 width, u32 height,
             VkFormat format,
             VkImageUsageFlags usage,
             VmaMemoryUsage memUsage,
             u32 mipLevels,
             u32 arrayLayers,           // 6 for cubemap
             VkImageCreateFlags flags,  // VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT for cubemap
             VkImageViewType viewType); // VK_IMAGE_VIEW_TYPE_CUBE for cubemap

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

    [[nodiscard]] VkImage GetImage() const { return m_image; }
    [[nodiscard]] VkImageView GetView() const { return m_view; }
    [[nodiscard]] VkFormat GetFormat() const { return m_format; }
    [[nodiscard]] VkExtent2D GetExtent() const { return m_extent; }
    [[nodiscard]] u32 GetMipLevels() const { return m_mipLevels; }
    [[nodiscard]] bool IsValid() const { return m_image != VK_NULL_HANDLE; }

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
