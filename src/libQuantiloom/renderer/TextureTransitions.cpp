#include "renderer/TextureTransitions.hpp"

#include "core/Log.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Device.hpp"
#include "renderer/GpuAlloc.hpp"

#include <cstring>
#include <span>
#include <stdexcept>

namespace quantiloom::rendercore {

namespace {

const char* LayoutName(Layout layout) {
    switch (layout) {
        case Layout::Undefined: return "Undefined";
        case Layout::General: return "General";
        case Layout::TransferDst: return "TransferDst";
        case Layout::TransferSrc: return "TransferSrc";
        case Layout::ShaderReadOnly: return "ShaderReadOnly";
    }
    return "?";
}

}  // namespace

void LegacyTransition(gpu::CommandList& list, gpu::TextureRef texture, Layout from, Layout to,
                      u32 mips, u32 layers) {
    using gpu::Access;
    constexpr Access kShaderWrite = Access::RayTraceWrite | Access::ComputeWrite;

    // Under Optimal a pure transfer access names its transfer layout, a pure shader read
    // names SHADER_READ_ONLY, a discard names UNDEFINED, and a shader write names GENERAL.
    gpu::TextureBarrier barrier;
    barrier.texture = texture.WithLayout(gpu::TextureLayoutPolicy::Optimal);
    barrier.baseMip = 0;
    barrier.mipCount = mips;
    barrier.layerCount = layers;
    gpu::BarrierBatch batch;

    if (from == Layout::Undefined && to == Layout::General) {
        barrier.discard = true;
        barrier.to = kShaderWrite;
    } else if (from == Layout::Undefined && to == Layout::TransferDst) {
        barrier.discard = true;
        barrier.to = Access::TransferWrite;
    } else if (from == Layout::TransferDst && to == Layout::ShaderReadOnly) {
        barrier.from = Access::TransferWrite;
        barrier.to = Access::RayTraceRead | Access::FragmentRead;
    } else if (from == Layout::General && to == Layout::TransferSrc) {
        barrier.from = kShaderWrite;
        barrier.to = Access::TransferRead;
    } else if (from == Layout::TransferSrc && to == Layout::General) {
        barrier.from = Access::TransferRead;
        barrier.to = kShaderWrite;
    } else {
        // No access on either side, all commands against all commands. With no access a
        // layout cannot be named, so the old layout is UNDEFINED for Undefined and GENERAL
        // otherwise, and the new one is GENERAL.
        QL_LOG_WARN("Unsupported layout transition: {} -> {}", LayoutName(from), LayoutName(to));
        barrier.discard = from == Layout::Undefined;
        batch.srcStages = gpu::PipelineStage::AllCommands;
        batch.dstStages = gpu::PipelineStage::AllCommands;
    }

    batch.textures = std::span<const gpu::TextureBarrier>(&barrier, 1);
    list.Barrier(batch);
}

void LegacyTransitionImmediate(gpu::Device& device, gpu::TextureRef texture, Layout from,
                               Layout to, u32 mips, u32 layers) {
    device.ExecuteImmediate([&](gpu::CommandList& list) {
        LegacyTransition(list, texture, from, to, mips, layers);
    });

    QL_LOG_INFO("Image layout transition: {} -> {} (immediate, mips={}, layers={})",
                LayoutName(from), LayoutName(to), mips, layers);
}

std::vector<f32> ReadbackTexture(gpu::Device& device, gpu::TextureRef texture, u32 width,
                                 u32 height) {
    if (texture.format != gpu::Format::RGBA32Float) {
        throw std::runtime_error(
            "ReadbackImage: Only RGBA32Float (R32G32B32A32 float) is supported in M1");
    }

    constexpr u32 bytesPerPixel = 4 * sizeof(f32);
    const u64 bufferSize = static_cast<u64>(width) * height * bytesPerPixel;

    const std::unique_ptr<gpu::Buffer> staging = CreateBufferOrThrow(
        device, gpu::BufferDesc{bufferSize, gpu::BufferUsage::TransferDst, gpu::MemoryClass::Readback, 0});

    const gpu::TextureRef source = texture.WithLayout(gpu::TextureLayoutPolicy::Optimal);
    device.ExecuteImmediate([&](gpu::CommandList& list) {
        LegacyTransition(list, source, Layout::General, Layout::TransferSrc);

        gpu::BufferTextureCopy region;
        region.width = width;
        region.height = height;
        list.CopyTextureToBuffer(source, staging->Ref(), region);

        LegacyTransition(list, source, Layout::TransferSrc, Layout::General);
    });

    std::vector<f32> pixels(static_cast<size_t>(width) * height * 4);
    const void* mappedData = staging->MapRead();
    if (mappedData == nullptr) {
        throw std::runtime_error("Failed to map staging buffer for readback");
    }

    std::memcpy(pixels.data(), mappedData, bufferSize);
    staging->Unmap();

    return pixels;
}

}  // namespace quantiloom::rendercore
