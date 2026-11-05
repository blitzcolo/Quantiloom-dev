#include "support/GpuTestHelpers.hpp"

#include "gpu/DeviceFactory.hpp"
#include "renderer/GpuAlloc.hpp"
#include "renderer/TextureTransitions.hpp"
#include "support/GpuInteropTestUtils.hpp"
#include "support/MetalTestDevice.hpp"
#include "support/VulkanTestDevice.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <span>

namespace quantiloom::testing {

namespace {

std::unique_ptr<gpu::Buffer> FilledStaging(gpu::Device& device, const void* data, u64 bytes) {
    auto staging = CreateBufferOrThrow(
        device, gpu::BufferDesc{bytes, gpu::BufferUsage::TransferSrc, gpu::MemoryClass::Upload, 0});
    staging->Upload(data, bytes);
    return staging;
}

// TRANSFER_DST_OPTIMAL under the Optimal policy; GENERAL after, which no access implies
// there, so it travels as a host layout.
void RecordComputeHandoff(gpu::CommandList& list, gpu::TextureRef target, gpu::Access to) {
    gpu::TextureBarrier ready;
    ready.texture = target;
    ready.from = gpu::Access::TransferWrite;
    ready.to = to;
    ready.layerCount = target.layers;
    ready.hostLayoutAfter = kHostLayoutGeneral;
    list.Barrier(ready);
}

}  // namespace

void UploadTexture(gpu::Device& device, gpu::TextureRef texture, const void* data, u64 bytes,
                   UploadHandoff handoff) {
    const auto staging = FilledStaging(device, data, bytes);
    const gpu::TextureRef target = texture.WithLayout(gpu::TextureLayoutPolicy::Optimal);
    const u32 layers = texture.layers;
    device.ExecuteImmediate([&](gpu::CommandList& list) {
        rendercore::LegacyTransition(list, target, rendercore::Layout::Undefined,
                                     rendercore::Layout::TransferDst, 1, layers);
        for (u32 layer = 0; layer < layers; ++layer) {
            gpu::BufferTextureCopy region;
            region.bufferOffset = bytes / layers * layer;
            region.baseLayer = layer;
            region.width = texture.width;
            region.height = texture.height;
            list.CopyBufferToTexture(staging->Ref(), target, region);
        }
        switch (handoff) {
            case UploadHandoff::ComputeRead:
                RecordComputeHandoff(list, target, gpu::Access::ComputeRead);
                break;
            case UploadHandoff::ComputeReadWrite:
                RecordComputeHandoff(list, target, gpu::Access::ComputeRead | gpu::Access::ComputeWrite);
                break;
            case UploadHandoff::AllCommands: {
                // CommandHelper's fallback transition: no access to name, so both layouts
                // travel as host layouts and both scopes are all commands.
                gpu::TextureBarrier ready;
                ready.texture = target;
                ready.layerCount = layers;
                ready.hostLayoutBefore = kHostLayoutTransferDst;
                ready.hostLayoutAfter = kHostLayoutGeneral;
                gpu::BarrierBatch batch;
                batch.textures = std::span<const gpu::TextureBarrier>(&ready, 1);
                batch.srcStages = gpu::PipelineStage::AllCommands;
                batch.dstStages = gpu::PipelineStage::AllCommands;
                list.Barrier(batch);
                break;
            }
        }
    });
}

void ReuploadTexture(gpu::Device& device, gpu::TextureRef texture, const void* data, u64 bytes) {
    const auto staging = FilledStaging(device, data, bytes);
    const gpu::TextureRef target = texture.WithLayout(gpu::TextureLayoutPolicy::Optimal);
    device.ExecuteImmediate([&](gpu::CommandList& list) {
        gpu::TextureBarrier toDst;
        toDst.texture = target;
        toDst.from = gpu::Access::ComputeRead;
        toDst.to = gpu::Access::TransferWrite;
        toDst.layerCount = 1;
        toDst.hostLayoutBefore = kHostLayoutGeneral;
        list.Barrier(toDst);
        gpu::BufferTextureCopy region;
        region.width = texture.width;
        region.height = texture.height;
        list.CopyBufferToTexture(staging->Ref(), target, region);
        RecordComputeHandoff(list, target, gpu::Access::ComputeRead);
    });
}

std::vector<u32> ReadTextureWords(gpu::Device& device, gpu::TextureRef texture, u32 width, u32 height) {
    const u64 bytes = static_cast<u64>(width) * height * 4 * sizeof(u32);
    const auto staging = CreateBufferOrThrow(
        device, gpu::BufferDesc{bytes, gpu::BufferUsage::TransferDst, gpu::MemoryClass::Readback, 0});
    const gpu::TextureRef source = texture.WithLayout(gpu::TextureLayoutPolicy::Optimal);
    device.ExecuteImmediate([&](gpu::CommandList& list) {
        rendercore::LegacyTransition(list, source, rendercore::Layout::General, rendercore::Layout::TransferSrc);
        gpu::BufferTextureCopy region;
        region.width = width;
        region.height = height;
        list.CopyTextureToBuffer(source, staging->Ref(), region);
        rendercore::LegacyTransition(list, source, rendercore::Layout::TransferSrc, rendercore::Layout::General);
    });
    std::vector<u32> out(static_cast<size_t>(width) * height * 4);
    const void* mapped = staging->Map();
    if (!mapped) return {};
    std::memcpy(out.data(), mapped, bytes);
    staging->Unmap();
    return out;
}

u32 TextureMipLevels(gpu::Backend backend, const gpu::Texture& texture) {
    switch (backend) {
        case gpu::Backend::Vulkan: return VulkanTextureMipLevels(texture);
        case gpu::Backend::Metal: return MetalTextureMipLevels(texture);
    }
    ADD_FAILURE() << "no mip inspection for the " << gpu::BackendName(backend) << " backend";
    return 0;
}

f32 PackedInstanceTranslationX(gpu::Backend backend, const void* packed, u32 index) {
    switch (backend) {
        case gpu::Backend::Vulkan: return VulkanPackedInstanceTranslationX(packed, index);
        case gpu::Backend::Metal: return MetalPackedInstanceTranslationX(packed, index);
    }
    ADD_FAILURE() << "no instance inspection for the " << gpu::BackendName(backend) << " backend";
    return 0.0f;
}

}  // namespace quantiloom::testing
