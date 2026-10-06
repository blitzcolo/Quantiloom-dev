/**
 * @file DeferredBufferUploads.hpp
 * @brief Small host writes recorded into the next submitted command buffer
 *
 * A direct write through a persistent CPU_TO_GPU mapping races any submitted
 * frame still reading the buffer, so callers used to drain the queue before
 * touching it -- which put a vkQueueWaitIdle behind every sun-direction
 * setter. Enqueue() copies the payload; Record() then replays the pending
 * writes into a command buffer the caller is already about to submit,
 * bracketed by barriers, so the GPU itself orders the write after the
 * earlier readers in submission order. No host wait, and a setter that runs
 * twice between two frames records only the latest bytes.
 *
 * vkCmdUpdateBuffer carries at most 65536 bytes per call, so payloads are
 * chunked; payloads too large to sit in a command buffer at all (megabyte
 * thermal fields, the atmosphere LUT) keep a synchronous upload path and
 * never reach this queue -- see kMaxPayloadBytes.
 *
 * Entries name their destination by VkBuffer handle, so a pending write
 * must be flushed before the GpuBuffer it targets is destroyed; callers
 * replacing a buffer drain the queue first.
 */

#pragma once

#include "core/Types.hpp"
#include <vulkan/vulkan.h>
#include <vector>

namespace quantiloom {

class GpuBuffer;

class DeferredBufferUploads {
public:
    /// Largest payload recorded into a command buffer. Bigger writes keep a
    /// synchronous path in the caller.
    static constexpr VkDeviceSize kMaxPayloadBytes = 1u << 20;

    /// Copy `bytes` of `data` for a deferred write of `dst` at `offset`.
    /// A pending write to the same (dst, offset, bytes) is replaced rather
    /// than appended. Offset and size must be multiples of 4, as
    /// vkCmdUpdateBuffer requires.
    void Enqueue(const GpuBuffer& dst, VkDeviceSize offset, const void* data,
                 VkDeviceSize bytes);

    /// Record all pending uploads into `cmd`: one barrier making the target
    /// ranges writable by the transfer stage (after earlier shader readers),
    /// the vkCmdUpdateBuffer writes, and one barrier making them visible to
    /// ray tracing and compute shader reads. Clears the queue afterwards.
    void Record(VkCommandBuffer cmd);

    [[nodiscard]] bool Empty() const { return m_pending.empty(); }
    void Clear() { m_pending.clear(); }

private:
    struct Pending {
        VkBuffer dst;
        VkDeviceSize offset;
        std::vector<u8> bytes;
    };
    std::vector<Pending> m_pending;
};

} // namespace quantiloom
