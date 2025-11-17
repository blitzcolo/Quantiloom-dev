#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

// ============================================================================
// GpuBuffer - RAII wrapper for VkBuffer with VMA allocation
// ============================================================================
// Responsibilities:
// - Create VkBuffer with VMA memory allocation
// - Automatically destroy on destruction (RAII)
// - Provide Map/Unmap interface for CPU-accessible buffers
// - Movable but non-copyable (strict ownership)
//
// Usage:
//   GpuBuffer vertexBuffer(allocator, size,
//       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
//       VMA_MEMORY_USAGE_GPU_ONLY);
//
// Memory types:
// - VMA_MEMORY_USAGE_GPU_ONLY: Device-local (fastest for GPU)
// - VMA_MEMORY_USAGE_CPU_TO_GPU: Staging buffer (host-visible)
// - VMA_MEMORY_USAGE_GPU_TO_CPU: Readback buffer
// ============================================================================

namespace quantiloom {

class QL_API GpuBuffer {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    // Create buffer with VMA
    GpuBuffer(VmaAllocator allocator, VkDeviceSize size,
              VkBufferUsageFlags usage, VmaMemoryUsage memUsage);

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

    VkBuffer GetHandle() const { return m_buffer; }
    VkDeviceSize GetSize() const { return m_size; }
    bool IsValid() const { return m_buffer != VK_NULL_HANDLE; }

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
