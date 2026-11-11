#include "renderer/DeferredBufferUploads.hpp"

#include "gpu/CommandList.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>

namespace quantiloom {

namespace {

constexpr u64 kMaxCmdUpdateBytes = 65536;
// Ray tracing and compute shader reads, uniform reads included.
constexpr gpu::Access kReaders =
    gpu::Access::RayTraceRead | gpu::Access::ComputeRead | gpu::Access::UniformRead;

} // namespace

void DeferredBufferUploads::Enqueue(gpu::BufferRef dst, u64 offset,
                                    const void* data, u64 bytes) {
    assert(offset % 4 == 0);
    assert(bytes % 4 == 0);
    for (auto& pending : m_pending) {
        if (pending.dst.handle == dst.handle && pending.offset == offset &&
            pending.bytes.size() == bytes) {
            pending.bytes.assign(static_cast<const u8*>(data),
                                 static_cast<const u8*>(data) + bytes);
            return;
        }
    }
    Pending pending;
    pending.dst = dst;
    pending.offset = offset;
    pending.bytes.assign(static_cast<const u8*>(data),
                         static_cast<const u8*>(data) + bytes);
    m_pending.push_back(std::move(pending));
}

void DeferredBufferUploads::Record(gpu::CommandList& list) {
    if (m_pending.empty()) return;

    // Earlier submissions may still be reading these buffers when this one
    // runs; the barrier pair is what orders the write between them.
    list.Barrier(gpu::GlobalBarrier{kReaders, gpu::Access::TransferWrite});

    for (const auto& pending : m_pending) {
        // The update reads the handle only.
        const gpu::BufferRef dst = pending.dst;
        for (u64 offset = 0; offset < pending.bytes.size();
             offset += kMaxCmdUpdateBytes) {
            const u64 chunk = std::min<u64>(
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
