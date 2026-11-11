#include "renderer/AsyncPixelReadback.hpp"

#include "gpu/Device.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>

namespace quantiloom::rendercore {

namespace {

// The Err of a failed gpu/Sync.hpp call is the backend's result code as a
// decimal, so the message still names the code.
[[noreturn]] void ThrowReadbackError(const String& code) {
    throw std::runtime_error("asynchronous pixel readback Vulkan error " + code);
}

void Check(const Result<void, String>& result) {
    if (!result) ThrowReadbackError(result.error());
}

template <class T>
T Take(Result<T, String>&& result) {
    if (!result) ThrowReadbackError(result.error());
    return std::move(result).value();
}

} // namespace

AsyncPixelReadback::AsyncPixelReadback(gpu::Device& device) : m_device(device) {
    try {
        gpu::CommandPoolDesc pool;
        pool.resettable = true;
        m_pool = Take(device.CreateCommandPool(pool));
        for (auto& slot : m_slots) {
            gpu::BufferDesc buffer;
            buffer.size = sizeof(glm::vec4);
            buffer.usage = gpu::BufferUsage::TransferDst;
            buffer.memory = gpu::MemoryClass::Readback;
            auto created = device.CreateBuffer(buffer);
            if (!created) throw std::runtime_error(created.error());
            slot.buffer = std::move(created).value();
            slot.fence = Take(device.CreateFence());
            slot.command = Take(m_pool->Allocate());
        }
    } catch (...) {
        Cleanup();
        throw;
    }
}

AsyncPixelReadback::~AsyncPixelReadback() { Cleanup(); }

bool AsyncPixelReadback::Submit(gpu::TextureRef image, u32 renderX, u32 renderY,
                                const PixelReading& reading) {
    // The accumulation image: GENERAL, one mip, one layer.
    const gpu::TextureRef target = image;
    for (auto& slot : m_slots) {
        if (slot.pending) continue;
        Check(slot.command->Reset());
        gpu::CommandList& list = *Take(slot.command->Begin());

        // Keep GENERAL so the copy does not change the accumulation's
        // layout contract. Both prior shader writes and subsequent
        // shader accesses are ordered against this read.
        const gpu::TextureBarrier toCopy{target, gpu::Access::ComputeWrite, gpu::Access::TransferRead};
        gpu::BarrierBatch before;
        before.textures = std::span<const gpu::TextureBarrier>(&toCopy, 1);
        before.srcStages = gpu::PipelineStage::AllCommands;
        list.Barrier(before);

        gpu::BufferTextureCopy copy;
        copy.x = static_cast<i32>(renderX);
        copy.y = static_cast<i32>(renderY);
        copy.width = 1;
        copy.height = 1;
        list.CopyTextureToBuffer(target, slot.buffer->Ref(), copy);

        const gpu::BufferBarrier host{slot.buffer->Ref(), gpu::Access::TransferWrite, gpu::Access::HostRead};
        const gpu::TextureBarrier fromCopy{target, gpu::Access::TransferRead,
                                           gpu::Access::ComputeRead | gpu::Access::ComputeWrite};
        gpu::BarrierBatch after;
        after.buffers = std::span<const gpu::BufferBarrier>(&host, 1);
        after.textures = std::span<const gpu::TextureBarrier>(&fromCopy, 1);
        after.dstStages = gpu::PipelineStage::AllCommands | gpu::PipelineStage::Host;
        list.Barrier(after);
        Check(slot.command->End());
        Check(slot.fence->Reset());
        Check(m_device.Submit(*slot.command, slot.fence.get()));
        slot.reading = reading;
        slot.serial = ++m_serial;
        slot.pending = true;
        return true;
    }
    return false;
}

Optional<PixelReading> AsyncPixelReadback::Poll(u64 generation, u64 acquisition) {
    Optional<PixelReading> result;
    u64 newest = 0;
    for (auto& slot : m_slots) {
        if (!slot.pending) continue;
        if (Take(slot.fence->Status()) == gpu::FenceStatus::NotReady) continue;
        slot.pending = false;
        if (slot.reading.imageGeneration != generation ||
            slot.reading.acquisitionIndex != acquisition || slot.serial <= newest) continue;
        const void* mapped = slot.buffer->MapRead();
        if (!mapped) throw std::runtime_error("cannot map asynchronous pixel readback");
        std::memcpy(&slot.reading.value, mapped, sizeof(glm::vec4));
        slot.buffer->Unmap();
        newest = slot.serial;
        result = slot.reading;
    }
    return result;
}

void AsyncPixelReadback::Cleanup() noexcept {
    for (auto& slot : m_slots) {
        if (slot.pending) static_cast<void>(slot.fence->Wait());
        slot.fence.reset();
        slot.pending = false;
        slot.buffer.reset();
    }
    for (auto& slot : m_slots) slot.command.reset();
    m_pool.reset();
}

} // namespace quantiloom::rendercore
