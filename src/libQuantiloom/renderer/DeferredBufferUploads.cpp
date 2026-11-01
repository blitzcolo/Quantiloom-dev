#include "renderer/DeferredBufferUploads.hpp"

#include "gpu/vulkan/VulkanBridge.hpp"
#include "renderer/GpuBuffer.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>

namespace quantiloom {

namespace {

constexpr VkDeviceSize kMaxCmdUpdateBytes = 65536;
// Ray tracing and compute shader reads, uniform reads included.
constexpr gpu::Access kReaders =
    gpu::Access::RayTraceRead | gpu::Access::ComputeRead | gpu::Access::UniformRead;

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

    auto list = gpu::vulkan::Wrap(cmd);

    // Earlier submissions may still be reading these buffers when this one
    // runs; the barrier pair is what orders the write between them.
    list.Barrier(gpu::GlobalBarrier{kReaders, gpu::Access::TransferWrite});

    for (const auto& pending : m_pending) {
        // A pending entry keeps the handle only; the update reads nothing else.
        const gpu::BufferRef dst = gpu::vulkan::Ref(pending.dst, gpu::kWholeSize);
        for (VkDeviceSize offset = 0; offset < pending.bytes.size();
             offset += kMaxCmdUpdateBytes) {
            const VkDeviceSize chunk = std::min<VkDeviceSize>(
                kMaxCmdUpdateBytes, pending.bytes.size() - offset);
            list.UpdateBuffer(dst, pending.offset + offset,
                              std::as_bytes(std::span<const u8>(
                                  pending.bytes.data() + offset, static_cast<std::size_t>(chunk))));
        }
    }

    list.Barrier(gpu::GlobalBarrier{gpu::Access::TransferWrite, kReaders});

    m_pending.clear();
}

} // namespace quantiloom
