/**
 * @file CommandList.hpp
 * @brief Barriers and recording interface of the compute RHI (internal)
 *
 * Every recording call emits exactly one native command (Vulkan), so a command-stream
 * diff before and after a migration step is meaningful.
 */

#pragma once

#include "gpu/Kernel.hpp"

#include <span>
#include <type_traits>

namespace quantiloom::gpu {

struct GlobalBarrier {
    Access from = Access::None;
    Access to = Access::None;
};

struct BufferBarrier {
    BufferRef buffer;
    Access from = Access::None;
    Access to = Access::None;
    u64 offset = 0;
    u64 size = kWholeSize;
};

struct TextureBarrier {
    TextureRef texture;
    Access from = Access::None;
    Access to = Access::None;
    bool discard = false;  // old contents undefined (UNDEFINED layout)
};

/// One native barrier call: stages are the OR over every entry's scopes.
struct BarrierBatch {
    std::span<const GlobalBarrier> memory = {};
    std::span<const BufferBarrier> buffers = {};
    std::span<const TextureBarrier> textures = {};
};

struct BufferCopyRegion {
    u64 srcOffset = 0;
    u64 dstOffset = 0;
    u64 size = 0;
};

struct TextureCopyRegion {  // layer 0, mip 0, zero offsets
    u32 width = 0;
    u32 height = 0;
};

class CommandList {
public:
    virtual ~CommandList() = default;
    [[nodiscard]] virtual Backend GetBackend() const = 0;

    virtual void SetKernel(const ComputeKernel& kernel) = 0;
    /// Takes the layout explicitly so the call may precede SetKernel.
    virtual void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group) = 0;
    virtual void PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) = 0;
    template <class T>
    void PushConstants(const KernelLayout& layout, const T& block) {
        static_assert(std::is_trivially_copyable_v<T>);
        PushConstants(layout, std::span<const std::byte>(reinterpret_cast<const std::byte*>(&block), sizeof(T)));
    }
    virtual void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) = 0;

    virtual void Barrier(const BarrierBatch& batch) = 0;
    void Barrier(const GlobalBarrier& b) {
        Barrier(BarrierBatch{std::span<const GlobalBarrier>{&b, 1}, {}, {}});
    }
    void Barrier(const BufferBarrier& b) {
        Barrier(BarrierBatch{{}, std::span<const BufferBarrier>{&b, 1}, {}});
    }
    void Barrier(const TextureBarrier& b) {
        Barrier(BarrierBatch{{}, {}, std::span<const TextureBarrier>{&b, 1}});
    }

    virtual void FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) = 0;
    /// At most 65536 bytes, a multiple of 4 (vkCmdUpdateBuffer's limits).
    virtual void UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) = 0;
    virtual void CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) = 0;
    /// Both textures in their copy layouts per policy (GENERAL for storage).
    virtual void CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) = 0;

    virtual void ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) = 0;
    virtual void WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) = 0;
};

}  // namespace quantiloom::gpu
