#include "gpu/vulkan/VulkanCommandList.hpp"

#include "core/Platform.hpp"
#include "gpu/vulkan/VulkanResources.hpp"
#include "gpu/vulkan/VulkanTrace.hpp"

#include <stdexcept>

namespace quantiloom::gpu::vulkan {

namespace {

/// The layout a copy or blit names: the transfer layout under Optimal, GENERAL under General.
VkImageLayout TransferLayout(const TextureRef& texture, VkImageLayout optimal) {
    return texture.layout == TextureLayoutPolicy::Optimal ? optimal : VK_IMAGE_LAYOUT_GENERAL;
}

VkBufferImageCopy NativeRegion(const BufferTextureCopy& region) {
    VkBufferImageCopy copy{};
    copy.bufferOffset = region.bufferOffset;
    copy.bufferRowLength = region.bufferRowLength;
    copy.bufferImageHeight = region.bufferImageHeight;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, region.mipLevel, region.baseLayer, region.layerCount};
    copy.imageOffset = {region.x, region.y, 0};
    copy.imageExtent = {region.width, region.height, 1};
    return copy;
}

}  // namespace

void VulkanCommandList::SetKernel(const ComputeKernel& kernel) {
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<const VulkanComputeKernel&>(kernel).Native());
}

void VulkanCommandList::SetBindGroup(const KernelLayout& layout, u32 space, BindGroup group, PipelineKind kind) {
    const VkDescriptorSet set = FromHandle<VkDescriptorSet>(group.handle);
    const VkPipelineBindPoint bindPoint =
        kind == PipelineKind::Trace ? VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR : VK_PIPELINE_BIND_POINT_COMPUTE;
    vkCmdBindDescriptorSets(cmd_, bindPoint, static_cast<const VulkanKernelLayout&>(layout).Native(), space, 1, &set,
                            0, nullptr);
}

void VulkanCommandList::PushConstants(const KernelLayout& layout, std::span<const std::byte> bytes) {
    QL_ASSERT(bytes.size() == layout.PushConstantBytes(), "push constant size differs from the kernel layout");
    vkCmdPushConstants(cmd_, static_cast<const VulkanKernelLayout&>(layout).Native(),
                       ToVk(layout.PushConstantVisibility()), 0, static_cast<u32>(bytes.size()), bytes.data());
}

void VulkanCommandList::Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) {
    vkCmdDispatch(cmd_, groupsX, groupsY, groupsZ);
}

void VulkanCommandList::SetTraceKernel(const TraceKernel& kernel) {
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      static_cast<const VulkanTraceKernel&>(kernel).Native());
}

void VulkanCommandList::Trace(const TraceKernel& kernel, u32 width, u32 height, u32 depth) {
    static_cast<const VulkanTraceKernel&>(kernel).TraceRays(cmd_, width, height, depth);
}

void VulkanCommandList::BuildRtAccelerationStructure(const RtBuild& build) {
    if (rt_.cmdBuildStructures == nullptr) {
        // A device without the extension hands out lists with no device functions. The same
        // message AccelerationStructure.cpp throws for a missing function.
        throw std::runtime_error("Failed to load acceleration structure functions");
    }
    NativeRtGeometry native;
    FillRtGeometry(rt_.device, build.geometry, native);
    if (build.mode == RtBuildMode::Update) {
        native.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        native.info.srcAccelerationStructure = FromHandle<VkAccelerationStructureKHR>(build.src.handle);
    }
    native.info.dstAccelerationStructure = FromHandle<VkAccelerationStructureKHR>(build.dst.handle);
    native.info.scratchData.deviceAddress = BufferAddress(rt_.device, build.scratch);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = native.primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    rt_.cmdBuildStructures(cmd_, 1, &native.info, &ranges);
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
        barrier.oldLayout = b.hostLayoutBefore ? static_cast<VkImageLayout>(*b.hostLayoutBefore)
                                               : LayoutOf(b.texture.layout, b.from, b.discard);
        barrier.newLayout = b.hostLayoutAfter ? static_cast<VkImageLayout>(*b.hostLayoutAfter)
                                              : LayoutOf(b.texture.layout, b.to, false);
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = FromHandle<VkImage>(b.texture.image);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, b.baseMip, b.mipCount, 0,
                                    b.layerCount == 0 ? b.texture.layers : b.layerCount};
        images.push_back(barrier);
    }

    if (batch.srcStages != PipelineStage::None) srcStages = ToVk(batch.srcStages);
    if (batch.dstStages != PipelineStage::None) dstStages = ToVk(batch.dstStages);

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

void VulkanCommandList::CopyBufferToTexture(BufferRef src, TextureRef dst, const BufferTextureCopy& region) {
    const VkBufferImageCopy copy = NativeRegion(region);
    vkCmdCopyBufferToImage(cmd_, FromHandle<VkBuffer>(src.handle), FromHandle<VkImage>(dst.image),
                           TransferLayout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), 1, &copy);
}

void VulkanCommandList::CopyTextureToBuffer(TextureRef src, BufferRef dst, const BufferTextureCopy& region) {
    const VkBufferImageCopy copy = NativeRegion(region);
    vkCmdCopyImageToBuffer(cmd_, FromHandle<VkImage>(src.image), TransferLayout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                           FromHandle<VkBuffer>(dst.handle), 1, &copy);
}

void VulkanCommandList::BlitTexture(TextureRef src, TextureRef dst, const BlitRegion& region, Filter filter) {
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, region.srcMip, 0, 1};
    blit.srcOffsets[0] = {region.srcX0, region.srcY0, 0};
    blit.srcOffsets[1] = {region.srcX1, region.srcY1, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, region.dstMip, 0, 1};
    blit.dstOffsets[0] = {region.dstX0, region.dstY0, 0};
    blit.dstOffsets[1] = {region.dstX1, region.dstY1, 1};
    vkCmdBlitImage(cmd_, FromHandle<VkImage>(src.image), TransferLayout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                   FromHandle<VkImage>(dst.image), TransferLayout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), 1, &blit,
                   ToVk(filter));
}

void VulkanCommandList::ResetTimestamps(TimestampPoolRef pool, u32 first, u32 count) {
    vkCmdResetQueryPool(cmd_, FromHandle<VkQueryPool>(pool.handle), first, count);
}

void VulkanCommandList::WriteTimestamp(TimestampPoolRef pool, u32 index, Stage after) {
    vkCmdWriteTimestamp(cmd_, TimestampStage(after), FromHandle<VkQueryPool>(pool.handle), index);
}

void VulkanCommandList::SetEvent(EventRef event, Stage after) {
    vkCmdSetEvent(cmd_, FromHandle<VkEvent>(event.handle), TimestampStage(after));
}

void VulkanCommandList::ResetEvent(EventRef event, Stage after) {
    vkCmdResetEvent(cmd_, FromHandle<VkEvent>(event.handle), TimestampStage(after));
}

}  // namespace quantiloom::gpu::vulkan
