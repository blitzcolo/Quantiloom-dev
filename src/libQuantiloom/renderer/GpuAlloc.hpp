/**
 * @file GpuAlloc.hpp
 * @brief The renderer's throwing allocation contract, on top of gpu::Device (internal)
 *
 * GpuBuffer and GpuImage threw std::runtime_error on a failed allocation, and the callers
 * of the renderer modules catch it. These rethrow gpu::Device's Err, which is that text, so
 * every existing catch still sees the failure (the p03 compute modules translate instead).
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/Device.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace quantiloom {

template <class T>
[[nodiscard]] T ValueOrThrow(Result<T, String>&& result) {
    if (!result) throw std::runtime_error(result.error());
    return std::move(result).value();
}

[[nodiscard]] inline std::unique_ptr<gpu::Buffer> CreateBufferOrThrow(gpu::Device& device,
                                                                    const gpu::BufferDesc& desc) {
    return ValueOrThrow(device.CreateBuffer(desc));
}

[[nodiscard]] inline std::unique_ptr<gpu::Texture> CreateTextureOrThrow(gpu::Device& device,
                                                                      const gpu::TextureDesc& desc) {
    return ValueOrThrow(device.CreateTexture(desc));
}

}  // namespace quantiloom
