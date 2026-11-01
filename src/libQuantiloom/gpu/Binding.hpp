/**
 * @file Binding.hpp
 * @brief Descriptor layouts, groups and writes of the compute RHI (internal)
 *
 * A BindingSlot mirrors one HLSL declaration `register(<class><slot>, space<n>)`
 * (src/shaders/ql_bind.hlsli): slot is the register number and the Vulkan binding, the
 * layout's space is the HLSL space and the Vulkan set.
 */

#pragma once

#include "gpu/GpuTypes.hpp"

#include <span>

namespace quantiloom::gpu {

/// One per HLSL resource class (the register letter in parentheses).
enum class BindingKind : u8 {
    ReadOnlyBuffer,         // (t) StructuredBuffer / ByteAddressBuffer     -> STORAGE_BUFFER
    ReadWriteBuffer,        // (u) RWStructuredBuffer / RWByteAddressBuffer -> STORAGE_BUFFER
    StorageTexture,         // (u) RWTexture2D / RWTexture2DArray           -> STORAGE_IMAGE
    SampledTexture,         // (t) Texture2D / TextureCube                  -> SAMPLED_IMAGE
    Sampler,                // (s) SamplerState                             -> SAMPLER
    ConstantBuffer,         // (b) ConstantBuffer<T>                        -> UNIFORM_BUFFER
    AccelerationStructure,  // (t) reserved                                 -> ACCELERATION_STRUCTURE_KHR
};

inline constexpr u32 kUnboundedCount = ~u32{0};  // reserved: bindless arrays in space 1

struct BindingSlot {
    u32 slot = 0;
    BindingKind kind{};
    u32 count = 1;
    bool partiallyBound = false;
};

struct BindingLayoutDesc {
    u32 space = 0;
    std::span<const BindingSlot> slots;  // creation fails if a slot number repeats
};

class BindingLayout {
public:
    virtual ~BindingLayout() = default;
    [[nodiscard]] virtual u32 Space() const = 0;
    [[nodiscard]] virtual std::span<const BindingSlot> Slots() const = 0;
};

/// A descriptor set (Vulkan) or a descriptor-table region (Metal). Valid until its pool is
/// reset or destroyed. Not owned.
struct BindGroup {
    NativeHandle handle = 0;
    [[nodiscard]] bool IsValid() const { return handle != 0; }
};

class BindGroupPool {
public:
    virtual ~BindGroupPool() = default;
    /// `count` groups of the pool's layout in one native allocation.
    [[nodiscard]] virtual Result<Vector<BindGroup>, String> Allocate(u32 count) = 0;
    /// Invalidates every group the pool handed out.
    virtual void Reset() = 0;
};

struct BindingWrite {
    u32 slot = 0;
    BindingKind kind{};
    BufferRef buffer{};
    u64 offset = 0;
    u64 range = kWholeSize;
    TextureRef texture{};

    static BindingWrite Buffer(u32 slot, BindingKind kind, BufferRef b, u64 offset = 0,
                               u64 range = kWholeSize) {
        BindingWrite w;
        w.slot = slot;
        w.kind = kind;
        w.buffer = b;
        w.offset = offset;
        w.range = range;
        return w;
    }
    static BindingWrite StorageTexture(u32 slot, TextureRef t) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::StorageTexture;
        w.texture = t;
        return w;
    }
};

}  // namespace quantiloom::gpu
