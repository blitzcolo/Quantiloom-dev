#pragma once

#include "gpu/CommandList.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "gpu/vulkan/VulkanRt.hpp"

#include <vulkan/vulkan.h>

namespace quantiloom::gpu::vulkan {

class VulkanCommandList final : public CommandList {
public:
    explicit VulkanCommandList(VkCommandBuffer cmd) : cmd_(cmd) {}
    /// What a VulkanDevice hands out: the list can also record acceleration structure builds.
    VulkanCommandList(VkCommandBuffer cmd, const RtFunctions& rt) : cmd_(cmd), rt_(rt) {}

    [[nodiscard]] Backend GetBackend() const override { return Backend::Vulkan; }
    [[nodiscard]] VkCommandBuffer Native() const { return cmd_; }

    void SetKernel(const ComputeKernel& kernel) override;
    void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group, PipelineKind kind) override;
    using CommandList::SetBindGroup;
    void PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) override;
    using CommandList::PushConstants;
    void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) override;

    void SetTraceKernel(const TraceKernel& kernel) override;
    void Trace(const TraceKernel& kernel, u32 width, u32 height, u32 depth) override;
    void BuildRtAccelerationStructure(const RtBuild& build) override;

    void Barrier(const BarrierBatch& batch) override;
    using CommandList::Barrier;

    void FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) override;
    void UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) override;
    void CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) override;
    void CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) override;
    void CopyBufferToTexture(BufferRef src, TextureRef dst, const BufferTextureCopy& region) override;
    void CopyTextureToBuffer(TextureRef src, BufferRef dst, const BufferTextureCopy& region) override;
    void BlitTexture(TextureRef src, TextureRef dst, const BlitRegion& region, Filter filter) override;

    void ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) override;
    void WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) override;

    void SetEvent(EventRef event, Stage after) override;
    void ResetEvent(EventRef event, Stage after) override;

private:
    VkCommandBuffer cmd_;
    RtFunctions rt_{};
};

/// A list over a raw command buffer, for Vulkan-only code that records on one (the Vulkan-only
/// tests). It has no ray tracing entry points.
[[nodiscard]] inline VulkanCommandList Wrap(VkCommandBuffer cmd) {
    return VulkanCommandList(cmd);
}

}  // namespace quantiloom::gpu::vulkan
