/**
 * @file GpuTypes.hpp
 * @brief Backend-neutral enums and non-owning resource references of the compute RHI (internal)
 *
 * No backend header is included here or in any other gpu/*.hpp; native handles travel as
 * u64. See gpu/AGENTS.md.
 */

#pragma once

#include "core/Types.hpp"

#include <type_traits>

namespace quantiloom::gpu {

enum class Backend : u8 { Vulkan = 1, Metal = 2 };

/// A native object as an integer: VkImage/VkImageView/VkBuffer/VkDescriptorSet/VkQueryPool
/// on Vulkan, an MTL:: pointer on Metal. Non-dispatchable Vulkan handles are 64-bit on
/// every platform.
using NativeHandle = u64;

inline constexpr u64 kWholeSize = ~u64{0};  // VK_WHOLE_SIZE on Vulkan

enum class Format : u16 { Undefined, R32Float, RGBA32Float, RGBA32Uint, RGBA8Unorm, RGBA8Srgb, BGRA8Srgb };

enum class MemoryClass : u8 {
    DeviceLocal,  // VMA_MEMORY_USAGE_GPU_ONLY   | MTL::StorageModePrivate
    Upload,       // VMA_MEMORY_USAGE_CPU_TO_GPU | Shared, write-combined
    Readback,     // VMA_MEMORY_USAGE_GPU_TO_CPU | Shared
    HostOnly,     // VMA_MEMORY_USAGE_CPU_ONLY   | Shared
};

enum class BufferUsage : u32 {
    None = 0,
    Storage = 1u << 0,
    Uniform = 1u << 1,
    TransferSrc = 1u << 2,
    TransferDst = 1u << 3,
    DeviceAddress = 1u << 4,               // reserved (ray tracing)
    AccelerationStructureInput = 1u << 5,  // reserved (ray tracing)
};

enum class TextureUsage : u32 {
    None = 0,
    Storage = 1u << 0,
    Sampled = 1u << 1,
    TransferSrc = 1u << 2,
    TransferDst = 1u << 3,
};

enum class TextureDimension : u8 { Tex2D, Tex2DArray, Cube };

/// Coarse access classes. OR them to say "any of these". The Vulkan backend maps each bit
/// to one (pipeline stage, access mask) pair and ORs the pairs (VulkanMapping.cpp).
enum class Access : u32 {
    None = 0,
    ComputeRead = 1u << 0,
    ComputeWrite = 1u << 1,
    RayTraceRead = 1u << 2,
    RayTraceWrite = 1u << 3,
    TransferRead = 1u << 4,
    TransferWrite = 1u << 5,
    HostRead = 1u << 6,
    HostWrite = 1u << 7,
    AnyRead = 1u << 8,
    AnyWrite = 1u << 9,
    AccelerationStructureRead = 1u << 10,   // reserved (ray tracing)
    AccelerationStructureWrite = 1u << 11,  // reserved (ray tracing)
};

/// Where a timestamp is taken.
enum class Stage : u8 { Compute, RayTracing, Transfer, AllCommands };

/// How a backend with layouts lays a texture out. General: every state is
/// VK_IMAGE_LAYOUT_GENERAL (storage images). Optimal: per-state layouts (reserved).
enum class TextureLayoutPolicy : u8 { General, Optimal };

#define QL_GPU_FLAG_OPERATORS(E)                                                              \
    constexpr E operator|(E a, E b) {                                                         \
        using U = std::underlying_type_t<E>;                                                  \
        return static_cast<E>(static_cast<U>(a) | static_cast<U>(b));                         \
    }                                                                                         \
    constexpr E operator&(E a, E b) {                                                         \
        using U = std::underlying_type_t<E>;                                                  \
        return static_cast<E>(static_cast<U>(a) & static_cast<U>(b));                         \
    }                                                                                         \
    constexpr E& operator|=(E& a, E b) { return a = a | b; }                                  \
    constexpr bool HasAny(E value, E bits) {                                                  \
        using U = std::underlying_type_t<E>;                                                  \
        return (static_cast<U>(value) & static_cast<U>(bits)) != 0;                           \
    }

QL_GPU_FLAG_OPERATORS(BufferUsage)
QL_GPU_FLAG_OPERATORS(TextureUsage)
QL_GPU_FLAG_OPERATORS(Access)

#undef QL_GPU_FLAG_OPERATORS

struct BufferRef {
    NativeHandle handle = 0;
    u64 size = 0;
    [[nodiscard]] bool IsValid() const { return handle != 0; }
    friend bool operator==(const BufferRef&, const BufferRef&) = default;
};

struct TextureRef {
    NativeHandle image = 0;  // VkImage | MTL::Texture*
    NativeHandle view = 0;   // VkImageView | 0
    Format format = Format::Undefined;
    u32 width = 0;
    u32 height = 0;
    u32 layers = 1;
    TextureLayoutPolicy layout = TextureLayoutPolicy::General;
    [[nodiscard]] bool IsValid() const { return image != 0; }
    /// What a descriptor names: the view on Vulkan, the texture on Metal. The modules'
    /// "the caller's image changed" guards compare this.
    [[nodiscard]] NativeHandle Identity() const { return view != 0 ? view : image; }
    friend bool operator==(const TextureRef&, const TextureRef&) = default;
};

struct TimestampPoolRef {
    NativeHandle handle = 0;
    u32 count = 0;
};

struct DeviceInfo {
    Backend backend{};
    String name;
    u32 vendorId = 0;
    u32 deviceId = 0;
    u32 driverVersion = 0;
    u64 maxStorageBufferRange = 0;
    u32 maxComputeSharedMemoryBytes = 0;
    f64 timestampPeriodNs = 0.0;
    bool hardwareRayTracing = false;
};

}  // namespace quantiloom::gpu
