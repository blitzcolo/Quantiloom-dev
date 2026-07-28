/**
 * @file GpuBuffer.hpp
 * @brief RAII wrapper for VkBuffer with VMA memory allocation
 *
 * Provides GpuBuffer class for managing Vulkan buffer resources:
 * - Automatic VkBuffer creation with VMA memory allocation
 * - RAII lifecycle (automatic destruction on scope exit)
 * - Map/Unmap interface for CPU-visible buffers
 * - Upload helper for data transfer
 * - Device address query for ray tracing shader access
 *
 * Memory types supported:
 * - VMA_MEMORY_USAGE_GPU_ONLY: Device-local (fastest GPU access, no CPU access)
 * - VMA_MEMORY_USAGE_CPU_TO_GPU: Host-visible staging (CPU write, GPU read)
 * - VMA_MEMORY_USAGE_GPU_TO_CPU: Host-visible readback (GPU write, CPU read)
 *
 * Common use cases:
 * - Vertex/index buffers (GPU_ONLY, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
 * - Uniform buffers (CPU_TO_GPU, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
 * - Storage buffers (CPU_TO_GPU, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
 * - Staging buffers (CPU_TO_GPU, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
 * - Readback buffers (GPU_TO_CPU, VK_BUFFER_USAGE_TRANSFER_DST_BIT)
 *
 * @note Movable but non-copyable (strict ownership semantics)
 * @note Map/Upload only work on HOST_VISIBLE buffers
 * @note GetDeviceAddress requires VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
 *
 * @author blitzcolo
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
// Note: This enum is also declared in GpuImage.hpp - both are needed
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
// GpuBuffer - RAII wrapper for VkBuffer with VMA allocation
// ============================================================================

namespace quantiloom {

/**
 * @class GpuBuffer
 * @brief RAII-managed VkBuffer with automatic VMA memory allocation and cleanup
 *
 * Provides safe, automatic management of Vulkan buffer resources.
 * Buffers are destroyed automatically when GpuBuffer goes out of scope.
 *
 * Memory management:
 * - Uses VMA (Vulkan Memory Allocator) for efficient allocation
 * - Automatically selects appropriate memory heap based on usage flags
 * - No manual vkFreeMemory required (RAII handles cleanup)
 *
 * Usage patterns:
 * @code
 * // Vertex buffer (GPU-only, no CPU access)
 * GpuBuffer vertexBuffer(
 *     allocator,
 *     sizeof(vertices),
 *     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
 *     VMA_MEMORY_USAGE_GPU_ONLY
 * );
 *
 * // Staging buffer for upload (CPU-to-GPU)
 * GpuBuffer staging(
 *     allocator,
 *     dataSize,
 *     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
 *     VMA_MEMORY_USAGE_CPU_TO_GPU
 * );
 * staging.Upload(data, dataSize);  // Automatic Map/Unmap
 *
 * // Uniform buffer (CPU writes every frame)
 * GpuBuffer uniformBuffer(
 *     allocator,
 *     sizeof(UniformData),
 *     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
 *     VMA_MEMORY_USAGE_CPU_TO_GPU
 * );
 * void* mapped = uniformBuffer.Map();
 * memcpy(mapped, &uniformData, sizeof(UniformData));
 * uniformBuffer.Unmap();
 * @endcode
 *
 * @note Non-copyable (prevents accidental double-free)
 * @note Movable (allows std::vector<GpuBuffer> and return values)
 * @note Map/Upload only work on VMA_MEMORY_USAGE_CPU_TO_GPU or GPU_TO_CPU buffers
 * @note Device address requires VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT usage flag
 *
 * @see GpuImage for image resource management
 * @see VulkanContext for VmaAllocator access
 */
class GpuBuffer {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    // Create buffer with VMA.
    //
    // minAlignment is the alignment the *caller* needs from the buffer's device
    // address, over and above whatever the driver reports for the usage flags.
    // Leave it 0 unless a spec rule demands more: acceleration-structure scratch
    // buffers must meet minAccelerationStructureScratchOffsetAlignment (128 on
    // NVIDIA) and TLAS instance data must be 16-byte aligned, neither of which
    // is implied by the usage bits, so VMA would otherwise be free to hand back
    // a 16-byte-aligned suballocation and the build would silently corrupt.
    GpuBuffer(VmaAllocator allocator, VkDeviceSize size,
              VkBufferUsageFlags usage, VmaMemoryUsage memUsage,
              VkDeviceSize minAlignment = 0);

    // Destructor: automatically destroys VkBuffer and VmaAllocation
    ~GpuBuffer();

    // ========================================================================
    // Move Semantics (non-copyable)
    // ========================================================================

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] VkBuffer GetHandle() const { return m_buffer; }
    [[nodiscard]] VkDeviceSize GetSize() const { return m_size; }
    [[nodiscard]] bool IsValid() const { return m_buffer != VK_NULL_HANDLE; }

    // ========================================================================
    // Memory Access (only for HOST_VISIBLE buffers)
    // ========================================================================

    // Map buffer memory (returns nullptr on failure)
    void* Map();

    // Unmap buffer memory
    void Unmap();

    // Upload data to buffer (auto Map/Unmap)
    // Only works for HOST_VISIBLE buffers
    void Upload(const void* data, VkDeviceSize uploadSize, VkDeviceSize offset = 0);

    // ========================================================================
    // Device Address (for ray tracing)
    // ========================================================================

    // Get device address (requires VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    VkDeviceAddress GetDeviceAddress(VkDevice device) const;

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    void* m_mappedData = nullptr;
};

} // namespace quantiloom
