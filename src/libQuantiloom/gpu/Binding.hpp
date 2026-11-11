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
    AccelerationStructure,  // (t) RaytracingAccelerationStructure          -> ACCELERATION_STRUCTURE_KHR
};

inline constexpr u32 kUnboundedCount = ~u32{0};  // reserved: bindless arrays in space 1

struct BindingSlot {
    u32 slot = 0;
    BindingKind kind{};
    u32 count = 1;
    bool partiallyBound = false;
    /// Vulkan stageFlags; Metal ignores it. Trailing, so `{slot, kind, count, partial}` still
    /// means a compute-only binding.
    ShaderStages visibility = ShaderStages::Compute;
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

/// One descriptor write. A SampledTexture or Sampler write with a non-empty span is one
/// array write of span.size() descriptors starting at arrayElement; the span must outlive
/// the WriteBindGroup call.
struct BindingWrite {
    u32 slot = 0;
    BindingKind kind{};
    BufferRef buffer{};
    u64 offset = 0;
    u64 range = kWholeSize;
    TextureRef texture{};
    u32 arrayElement = 0;
    SamplerRef sampler{};
    RtSceneRef scene{};
    std::span<const TextureRef> textures{};
    std::span<const SamplerRef> samplers{};

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
    /// Vulkan layout from the policy: GENERAL under General, SHADER_READ_ONLY_OPTIMAL under Optimal.
    static BindingWrite SampledTexture(u32 slot, TextureRef t) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::SampledTexture;
        w.texture = t;
        return w;
    }
    static BindingWrite Sampler(u32 slot, SamplerRef s) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::Sampler;
        w.sampler = s;
        return w;
    }
    static BindingWrite Scene(u32 slot, RtSceneRef s) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::AccelerationStructure;
        w.scene = s;
        return w;
    }
    static BindingWrite SampledTextures(u32 slot, std::span<const TextureRef> t, u32 first = 0) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::SampledTexture;
        w.textures = t;
        w.arrayElement = first;
        return w;
    }
    static BindingWrite Samplers(u32 slot, std::span<const SamplerRef> s, u32 first = 0) {
        BindingWrite w;
        w.slot = slot;
        w.kind = BindingKind::Sampler;
        w.samplers = s;
        w.arrayElement = first;
        return w;
    }
};

}  // namespace quantiloom::gpu
