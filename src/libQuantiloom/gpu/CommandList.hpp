/**
 * @file CommandList.hpp
 * @brief Barriers and recording interface of the compute RHI (internal)
 *
 * Every recording call emits exactly one native command (Vulkan), so a command-stream
 * diff before and after a migration step is meaningful.
 */

#pragma once

#include "gpu/Kernel.hpp"

#include <optional>
#include <span>
#include <type_traits>

namespace quantiloom::gpu {

inline constexpr u32 kRemaining = ~u32{0};  // VK_REMAINING_MIP_LEVELS / VK_REMAINING_ARRAY_LAYERS

class TraceKernel;  // Trace.hpp
struct RtBuild;     // RtScene.hpp

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

/// The host layout token of a host image left for sampling, for a caller that must name one
/// itself (pinned in VulkanMapping.cpp). Metal ignores it.
inline constexpr u32 kHostLayoutShaderReadOnly = 5;  // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL

struct TextureBarrier {
    TextureRef texture;
    Access from = Access::None;
    Access to = Access::None;
    bool discard = false;  // old contents undefined (UNDEFINED layout)
    u32 baseMip = 0;
    u32 mipCount = 1;      // kRemaining allowed
    u32 layerCount = 0;    // 0 = texture.layers; kRemaining allowed
    /// A host-owned image's layout token (GpuImageTarget::currentLayout / finalLayout),
    /// used verbatim instead of the policy's layout. Metal ignores both.
    std::optional<u32> hostLayoutBefore;
    std::optional<u32> hostLayoutAfter;
};

/// One native barrier call: stages are the OR over every entry's scopes, unless an
/// override is set (for the barriers whose stages are not the OR of their accesses).
struct BarrierBatch {
    std::span<const GlobalBarrier> memory = {};
    std::span<const BufferBarrier> buffers = {};
    std::span<const TextureBarrier> textures = {};
    PipelineStage srcStages = PipelineStage::None;  // None = the OR of the entries' source scopes
    PipelineStage dstStages = PipelineStage::None;  // None = the OR of the entries' destination scopes
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

/// One buffer<->texture copy region (VkBufferImageCopy). Depth 1, z offset 0.
struct BufferTextureCopy {
    u64 bufferOffset = 0;
    u32 bufferRowLength = 0;    // 0 = tightly packed
    u32 bufferImageHeight = 0;  // 0 = tightly packed
    u32 mipLevel = 0;
    u32 baseLayer = 0;
    u32 layerCount = 1;
    i32 x = 0;
    i32 y = 0;
    u32 width = 0;
    u32 height = 0;
};

/// One blit region (VkImageBlit): layer 0 of each mip, z from 0 to 1.
struct BlitRegion {
    u32 srcMip = 0;
    u32 dstMip = 0;
    i32 srcX0 = 0;
    i32 srcY0 = 0;
    i32 srcX1 = 0;
    i32 srcY1 = 0;
    i32 dstX0 = 0;
    i32 dstY0 = 0;
    i32 dstX1 = 0;
    i32 dstY1 = 0;
};

/// Which pipeline a bind group is bound for.
enum class PipelineKind : u8 { Compute, Trace };

class CommandList {
public:
    virtual ~CommandList() = default;
    [[nodiscard]] virtual Backend GetBackend() const = 0;

    virtual void SetKernel(const ComputeKernel& kernel) = 0;
    /// Takes the layout explicitly so the call may precede SetKernel.
    virtual void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group, PipelineKind kind) = 0;
    void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group) {
        SetBindGroup(layout, space, group, PipelineKind::Compute);
    }
    /// Stages from the layout's PushConstantVisibility().
    virtual void PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) = 0;
    template <class T>
    void PushConstants(const KernelLayout& layout, const T& block) {
        static_assert(std::is_trivially_copyable_v<T>);
        PushConstants(layout, std::span<const std::byte>(reinterpret_cast<const std::byte*>(&block), sizeof(T)));
    }
    virtual void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) = 0;

    /// Binds the kernel at the ray tracing bind point (vkCmdBindPipeline). Its bind groups are
    /// set with PipelineKind::Trace. Metal: the converted ray generation kernel's compute
    /// pipeline state.
    virtual void SetTraceKernel(const TraceKernel& kernel) = 0;
    /// width x height x depth rays through the kernel's own SBT (vkCmdTraceRaysKHR).
    /// Metal: dispatchThreads over the same grid.
    virtual void Trace(const TraceKernel& kernel, u32 width, u32 height, u32 depth) = 0;
    /// One build or refit with one geometry and one range (vkCmdBuildAccelerationStructuresKHR).
    /// Vulkan: throws std::runtime_error when the list has no device functions to call (a
    /// device without acceleration structures). Metal: an acceleration structure encoder's
    /// build or refit.
    virtual void BuildRtAccelerationStructure(const RtBuild& build) = 0;

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
    /// The texture in its copy layout per policy: TRANSFER_DST_OPTIMAL under Optimal,
    /// GENERAL under General.
    virtual void CopyBufferToTexture(BufferRef src, TextureRef dst, const BufferTextureCopy& region) = 0;
    /// The texture in its copy layout per policy: TRANSFER_SRC_OPTIMAL under Optimal,
    /// GENERAL under General.
    virtual void CopyTextureToBuffer(TextureRef src, BufferRef dst, const BufferTextureCopy& region) = 0;
    /// Source and destination in their transfer layouts per policy, as for the copies.
    /// On Metal a scaling or converting blit is a compute pass, not one command.
    virtual void BlitTexture(TextureRef src, TextureRef dst, const BlitRegion& region, Filter filter) = 0;

    virtual void ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) = 0;
    virtual void WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) = 0;

    virtual void SetEvent(EventRef event, Stage after) = 0;
    virtual void ResetEvent(EventRef event, Stage after) = 0;
};

}  // namespace quantiloom::gpu
