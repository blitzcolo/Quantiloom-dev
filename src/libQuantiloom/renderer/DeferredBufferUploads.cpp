#include "renderer/DeferredBufferUploads.hpp"

#include "renderer/GpuBuffer.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace quantiloom {

namespace {

constexpr VkDeviceSize kMaxCmdUpdateBytes = 65536;
constexpr VkPipelineStageFlags kReaderStages =
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkAccessFlags kReaderAccess =
    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

} // namespace

void DeferredBufferUploads::Enqueue(const GpuBuffer& dst, VkDeviceSize offset,
                                    const void* data, VkDeviceSize bytes) {
    assert(offset % 4 == 0);
    assert(bytes % 4 == 0);
    const VkBuffer handle = dst.GetHandle();
    for (auto& pending : m_pending) {
        if (pending.dst == handle && pending.offset == offset &&
            pending.bytes.size() == bytes) {
            pending.bytes.assign(static_cast<const u8*>(data),
                                 static_cast<const u8*>(data) + bytes);
            return;
        }
    }
    Pending pending;
    pending.dst = handle;
    pending.offset = offset;
    pending.bytes.assign(static_cast<const u8*>(data),
                         static_cast<const u8*>(data) + bytes);
    m_pending.push_back(std::move(pending));
}

void DeferredBufferUploads::Record(VkCommandBuffer cmd) {
    if (m_pending.empty()) return;

    // Earlier submissions may still be reading these buffers when this one
    // runs; the barrier pair is what orders the write between them.
    VkMemoryBarrier writable{};
    writable.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    writable.srcAccessMask = kReaderAccess;
    writable.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, kReaderStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &writable, 0, nullptr, 0, nullptr);

    for (const auto& pending : m_pending) {
        for (VkDeviceSize offset = 0; offset < pending.bytes.size();
             offset += kMaxCmdUpdateBytes) {
            const VkDeviceSize chunk = std::min<VkDeviceSize>(
                kMaxCmdUpdateBytes, pending.bytes.size() - offset);
            vkCmdUpdateBuffer(cmd, pending.dst, pending.offset + offset, chunk,
                              pending.bytes.data() + offset);
        }
    }

    VkMemoryBarrier ready{};
    ready.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = kReaderAccess;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, kReaderStages, 0,
                         1, &ready, 0, nullptr, 0, nullptr);

    m_pending.clear();
}

} // namespace quantiloom
