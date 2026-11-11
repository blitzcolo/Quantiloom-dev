/**
 * @file GpuTestHelpers.hpp
 * @brief The uploads, readbacks and native inspections the GPU tests share, on gpu::
 *
 * Each helper records what the hand-written Vulkan it replaced recorded -- the same
 * objects, transitions and copies, in the same order -- so a suite's api_dump command
 * stream does not move. Allocation failures throw, as GpuBuffer and GpuImage did
 * (renderer/GpuAlloc.hpp), so a case that fails to allocate still fails.
 *
 * Float readbacks need no helper here: rendercore::ReadbackTexture is the one
 * CommandHelper::ReadbackImage became.
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/Device.hpp"

#include <vector>

namespace quantiloom::testing {

/// How UploadTexture hands the texture over after the copy. On Vulkan each leaves it in
/// GENERAL.
enum class UploadHandoff : u8 {
    ComputeRead,       // transfer write -> compute read
    ComputeReadWrite,  // transfer write -> compute read and write
    AllCommands,       // no access on either side, all commands -> all commands
};

/**
 * @brief Fill mip 0 of every layer of @p texture, discarding what it held
 *
 * A TransferSrc staging buffer in Upload memory holding @p bytes, then one submission:
 * the legacy Undefined -> TransferDst transition over the texture's layers, one copy
 * per layer (layer i from byte i * bytes / layers), and the @p handoff barrier.
 */
void UploadTexture(gpu::Device& device, gpu::TextureRef texture, const void* data, u64 bytes,
                   UploadHandoff handoff = UploadHandoff::ComputeRead);

/// Rewrite mip 0, layer 0 of a texture compute reads in GENERAL: compute read -> transfer
/// write, the copy, transfer write -> compute read, in one submission.
void ReuploadTexture(gpu::Device& device, gpu::TextureRef texture, const void* data, u64 bytes);

/// Mip 0, layer 0 of a 16-byte-a-texel texture compute writes in GENERAL, as words: the
/// legacy General <-> TransferSrc transitions around the copy, then Map (no invalidate).
/// Empty when the readback buffer cannot be mapped.
[[nodiscard]] std::vector<u32> ReadTextureWords(gpu::Device& device, gpu::TextureRef texture,
                                                u32 width, u32 height);

/// The mip count @p backend gave @p texture.
[[nodiscard]] u32 TextureMipLevels(gpu::Backend backend, const gpu::Texture& texture);

/// The x translation of instance @p index in a buffer gpu::Device::PackRtInstances wrote.
[[nodiscard]] f32 PackedInstanceTranslationX(gpu::Backend backend, const void* packed, u32 index);

}  // namespace quantiloom::testing
