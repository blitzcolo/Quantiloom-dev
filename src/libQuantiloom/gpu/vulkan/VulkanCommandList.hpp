#pragma once

#include "gpu/CommandList.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"

#include <vulkan/vulkan.h>

namespace quantiloom::gpu::vulkan {

class VulkanCommandList final : public CommandList {
public:
    explicit VulkanCommandList(VkCommandBuffer cmd) : cmd_(cmd) {}

    [[nodiscard]] Backend GetBackend() const override { return Backend::Vulkan; }
    [[nodiscard]] VkCommandBuffer Native() const { return cmd_; }

    void SetKernel(const ComputeKernel& kernel) override;
    void SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group) override;
    void PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) override;
    using CommandList::PushConstants;
    void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) override;

    void Barrier(const BarrierBatch& batch) override;
    using CommandList::Barrier;

    void FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) override;
    void UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) override;
    void CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) override;
    void CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) override;

    void ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) override;
    void WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) override;

private:
    VkCommandBuffer cmd_;
};

}  // namespace quantiloom::gpu::vulkan
