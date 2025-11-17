#include "GpuBuffer.hpp"
#include "core/Log.hpp"
#include <cstring>
#include <stdexcept>

namespace quantiloom {

// ============================================================================
// Construction / Destruction
// ============================================================================

GpuBuffer::GpuBuffer(VmaAllocator allocator, VkDeviceSize size,
                     VkBufferUsageFlags usage, VmaMemoryUsage memUsage)
    : m_allocator(allocator), m_size(size) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;

    VkResult result = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
                                      &m_buffer, &m_allocation, nullptr);

    if (result != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to create VkBuffer via VMA: error code {}", static_cast<int>(result));
        throw std::runtime_error("GpuBuffer creation failed");
    }
}

GpuBuffer::~GpuBuffer() {
    if (m_buffer != VK_NULL_HANDLE) {
        // Ensure buffer is unmapped before destruction
        if (m_mappedData != nullptr) {
            Unmap();
        }
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }
}

// ============================================================================
// Move Semantics
// ============================================================================

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : m_allocator(other.m_allocator),
      m_buffer(other.m_buffer),
      m_allocation(other.m_allocation),
      m_size(other.m_size),
      m_mappedData(other.m_mappedData) {

    // Critical: nullify source to prevent double-free
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_mappedData = nullptr;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
        // Destroy current resource
        if (m_buffer != VK_NULL_HANDLE) {
            if (m_mappedData != nullptr) {
                Unmap();
            }
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        }

        // Move from other
        m_allocator = other.m_allocator;
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_size = other.m_size;
        m_mappedData = other.m_mappedData;

        // Nullify source
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_mappedData = nullptr;
    }
    return *this;
}

// ============================================================================
// Memory Access
// ============================================================================

void* GpuBuffer::Map() {
    if (m_mappedData != nullptr) {
        // Already mapped
        return m_mappedData;
    }

    VkResult result = vmaMapMemory(m_allocator, m_allocation, &m_mappedData);
    if (result != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to map buffer memory: error code {}", static_cast<int>(result));
        return nullptr;
    }

    return m_mappedData;
}

void GpuBuffer::Unmap() {
    if (m_mappedData != nullptr) {
        vmaUnmapMemory(m_allocator, m_allocation);
        m_mappedData = nullptr;
    }
}

void GpuBuffer::Upload(const void* data, VkDeviceSize uploadSize, VkDeviceSize offset) {
    if (offset + uploadSize > m_size) {
        QL_LOG_ERROR("Upload size ({} bytes) exceeds buffer size ({} bytes)",
                     offset + uploadSize, m_size);
        return;
    }

    void* mapped = Map();
    if (mapped != nullptr) {
        std::memcpy(static_cast<char*>(mapped) + offset, data, uploadSize);
        Unmap();
    }
}

// ============================================================================
// Device Address (for ray tracing)
// ============================================================================

VkDeviceAddress GpuBuffer::GetDeviceAddress(VkDevice device) const {
    VkBufferDeviceAddressInfo addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = m_buffer;

    return vkGetBufferDeviceAddress(device, &addressInfo);
}

} // namespace quantiloom
