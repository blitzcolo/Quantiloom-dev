/**
 * @file Resources.hpp
 * @brief Owning buffer and texture objects of the compute RHI (internal)
 */

#pragma once

#include "gpu/GpuTypes.hpp"

namespace quantiloom::gpu {

struct BufferDesc {
    u64 size = 0;
    BufferUsage usage = BufferUsage::None;
    MemoryClass memory = MemoryClass::DeviceLocal;
    u64 minAlignment = 0;  // GpuBuffer's minAlignment (acceleration-structure scratch later)
};

class Buffer {
public:
    virtual ~Buffer() = default;
    [[nodiscard]] virtual BufferRef Ref() const = 0;
    [[nodiscard]] u64 Size() const { return Ref().size; }
    /// Host-visible classes only. nullptr on failure. Repeated calls return the same
    /// pointer until Unmap. No cache maintenance (GpuBuffer::Map).
    virtual void* Map() = 0;
    /// Map, then invalidate non-coherent host caches. Call after the GPU work that wrote
    /// the buffer has completed (GpuBuffer::MapRead).
    virtual const void* MapRead() = 0;
    virtual void Unmap() = 0;
    /// Bounds-checked map, copy, flush, unmap. Logs and drops on overflow (GpuBuffer::Upload).
    virtual void Upload(const void* data, u64 bytes, u64 offset = 0) = 0;
};

struct TextureDesc {
    u32 width = 0;
    u32 height = 0;
    u32 layers = 1;
    u32 mipLevels = 1;
    Format format = Format::Undefined;
    TextureUsage usage = TextureUsage::None;
    TextureDimension dimension = TextureDimension::Tex2D;
    MemoryClass memory = MemoryClass::DeviceLocal;
};

class Texture {
public:
    virtual ~Texture() = default;
    /// Layout policy is General iff the usage has Storage.
    [[nodiscard]] virtual TextureRef Ref() const = 0;
};

}  // namespace quantiloom::gpu
