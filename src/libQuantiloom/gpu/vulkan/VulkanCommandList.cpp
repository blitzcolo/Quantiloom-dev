#include "gpu/vulkan/VulkanCommandList.hpp"

#include "core/Platform.hpp"
#include "gpu/vulkan/VulkanResources.hpp"

namespace quantiloom::gpu::vulkan {

void VulkanCommandList::SetKernel(const ComputeKernel& kernel) {
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<const VulkanComputeKernel&>(kernel).Native());
}

void VulkanCommandList::SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group) {
    const VkDescriptorSet set = FromHandle<VkDescriptorSet>(group.handle);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<const VulkanKernelLayout&>(layout).Native(),
                            space, 1, &set, 0, nullptr);
}

void VulkanCommandList::PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) {
    QL_ASSERT(bytes.size() == layout.PushConstantBytes(), "push constant size differs from the kernel layout");
    vkCmdPushConstants(cmd_, static_cast<const VulkanKernelLayout&>(layout).Native(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<u32>(bytes.size()), bytes.data());
}

void VulkanCommandList::Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) {
    vkCmdDispatch(cmd_, groupsX, groupsY, groupsZ);
}

void VulkanCommandList::Barrier(const BarrierBatch& batch) {
    if (batch.memory.empty() && batch.buffers.empty() && batch.textures.empty()) return;

    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;

    Vector<VkMemoryBarrier> memory;
    for (const GlobalBarrier& b : batch.memory) {
        const AccessScope src = ScopeOf(b.from, true);
        const AccessScope dst = ScopeOf(b.to, false);
        srcStages |= src.stages;
        dstStages |= dst.stages;
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = src.access;
        barrier.dstAccessMask = dst.access;
        memory.push_back(barrier);
    }

    Vector<VkBufferMemoryBarrier> buffers;
    for (const BufferBarrier& b : batch.buffers) {
        const AccessScope src = ScopeOf(b.from, true);
        const AccessScope dst = ScopeOf(b.to, false);
        srcStages |= src.stages;
        dstStages |= dst.stages;
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = src.access;
        barrier.dstAccessMask = dst.access;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = FromHandle<VkBuffer>(b.buffer.handle);
        barrier.offset = b.offset;
        barrier.size = b.size;
        buffers.push_back(barrier);
    }

    Vector<VkImageMemoryBarrier> images;
    for (const TextureBarrier& b : batch.textures) {
        const AccessScope src = ScopeOf(b.from, true);
        const AccessScope dst = ScopeOf(b.to, false);
        srcStages |= src.stages;
        dstStages |= dst.stages;
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = src.access;
        barrier.dstAccessMask = dst.access;
        barrier.oldLayout = LayoutOf(b.texture.layout, b.from, b.discard);
        barrier.newLayout = LayoutOf(b.texture.layout, b.to, false);
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = FromHandle<VkImage>(b.texture.image);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, b.texture.layers};
        images.push_back(barrier);
    }

    vkCmdPipelineBarrier(cmd_, srcStages, dstStages, 0, static_cast<u32>(memory.size()), memory.data(),
                         static_cast<u32>(buffers.size()), buffers.data(), static_cast<u32>(images.size()),
                         images.data());
}

void VulkanCommandList::FillBuffer(BufferRef dst, u64 offset, u64 size, u32 value) {
    vkCmdFillBuffer(cmd_, FromHandle<VkBuffer>(dst.handle), offset, size, value);
}

void VulkanCommandList::UpdateBuffer(BufferRef dst, u64 offset, std::span<const std::byte> bytes) {
    QL_ASSERT(bytes.size() <= 65536 && bytes.size() % 4 == 0 && offset % 4 == 0,
              "vkCmdUpdateBuffer needs <= 65536 bytes, a multiple of 4");
    vkCmdUpdateBuffer(cmd_, FromHandle<VkBuffer>(dst.handle), offset, bytes.size(), bytes.data());
}

void VulkanCommandList::CopyBuffer(BufferRef src, BufferRef dst, std::span<const BufferCopyRegion> regions) {
    Vector<VkBufferCopy> native;
    native.reserve(regions.size());
    for (const BufferCopyRegion& r : regions) native.push_back({r.srcOffset, r.dstOffset, r.size});
    vkCmdCopyBuffer(cmd_, FromHandle<VkBuffer>(src.handle), FromHandle<VkBuffer>(dst.handle),
                    static_cast<u32>(native.size()), native.data());
}

void VulkanCommandList::CopyTexture(TextureRef src, TextureRef dst, const TextureCopyRegion& region) {
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {region.width, region.height, 1};
    vkCmdCopyImage(cmd_, FromHandle<VkImage>(src.image), VK_IMAGE_LAYOUT_GENERAL, FromHandle<VkImage>(dst.image),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

void VulkanCommandList::ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) {
    vkCmdResetQueryPool(cmd_, FromHandle<VkQueryPool>(pool.handle), first, count);
}

void VulkanCommandList::WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) {
    vkCmdWriteTimestamp(cmd_, TimestampStage(after), FromHandle<VkQueryPool>(pool.handle), index);
}

}  // namespace quantiloom::gpu::vulkan
