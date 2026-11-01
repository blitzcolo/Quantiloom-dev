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

/// No RGB32Float, ever: Metal has no 3-channel 2D texture format. A 3-channel image is
/// padded to RGBA32Float with alpha 1 (gpu/AGENTS.md).
enum class Format : u16 {
    Undefined,
    R32Float,
    RGBA32Float,
    RGBA32Uint,
    RGBA8Unorm,
    RGBA8Srgb,
    BGRA8Srgb,
    RG32Float,    // the BRDF LUT
    RGBA16Float,  // GpuTargetFormat::Rgba16Float host targets
    BC7Unorm,     // compressed scene textures (TextureManager)
    BC7Srgb,
};

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
    DeviceAddress = 1u << 4,               // ray tracing: build inputs, scratch, SBT
    AccelerationStructureInput = 1u << 5,  // ray tracing: vertex, index and instance inputs
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
    AccelerationStructureRead = 1u << 10,   // an acceleration structure build reading one
    AccelerationStructureWrite = 1u << 11,  // an acceleration structure build writing one
    RayTraceSceneRead = 1u << 12,           // a trace reading the TLAS: ray tracing stage, AS read
    ComputeSceneRead = 1u << 13,            // an inline ray query in compute: compute stage, AS read
    FragmentRead = 1u << 14,                // fragment stage, shader read (legacy sampled-image masks)
    UniformRead = 1u << 15,                 // uniform read with no stage of its own: OR it with a
                                            // shader read, never use it alone
};

/// Stage masks for the barriers whose stages are not the OR of their accesses
/// (BarrierBatch::srcStages/dstStages). Vulkan maps each bit to one pipeline stage bit;
/// Metal ignores them (encoder boundaries carry the order).
enum class PipelineStage : u32 {
    None = 0,
    TopOfPipe = 1u << 0,
    Compute = 1u << 1,
    RayTracing = 1u << 2,
    Transfer = 1u << 3,
    Host = 1u << 4,
    AccelerationStructureBuild = 1u << 5,
    Fragment = 1u << 6,
    BottomOfPipe = 1u << 7,
    AllCommands = 1u << 8,
};

/// Where a timestamp is taken, or after which stage an event is set or reset.
enum class Stage : u8 { Compute, RayTracing, Transfer, AllCommands, TopOfPipe, BottomOfPipe };

/// Vulkan descriptor and push-constant visibility. Metal ignores it.
enum class ShaderStages : u32 {
    None = 0,
    Compute = 1u << 0,
    RayGen = 1u << 1,
    ClosestHit = 1u << 2,
    AnyHit = 1u << 3,
    Miss = 1u << 4,
};

/// Sampler and blit filtering. Zero is Vulkan's zero value, as for every sampler enum below.
enum class Filter : u8 { Nearest, Linear };
enum class AddressMode : u8 { Repeat, MirroredRepeat, ClampToEdge, ClampToBorder };
enum class CompareOp : u8 { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always };
/// Metal has only transparent black, opaque black and opaque white; the int/float split
/// is Vulkan's.
enum class BorderColor : u8 {
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite,
};

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
QL_GPU_FLAG_OPERATORS(PipelineStage)
QL_GPU_FLAG_OPERATORS(ShaderStages)

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
    /// The same texture under another layout policy for one use: a storage image (General)
    /// used as a blit or copy source is passed WithLayout(Optimal), which is what moving it
    /// to TRANSFER_SRC_OPTIMAL and back does today.
    [[nodiscard]] TextureRef WithLayout(TextureLayoutPolicy policy) const {
        TextureRef ref = *this;
        ref.layout = policy;
        return ref;
    }
    friend bool operator==(const TextureRef&, const TextureRef&) = default;
};

struct TimestampPoolRef {
    NativeHandle handle = 0;
    u32 count = 0;
};

struct SamplerRef {
    NativeHandle handle = 0;  // VkSampler | MTL::SamplerState*
    [[nodiscard]] bool IsValid() const { return handle != 0; }
    friend bool operator==(const SamplerRef&, const SamplerRef&) = default;
};

struct EventRef {
    NativeHandle handle = 0;  // VkEvent | MTL::SharedEvent*
    [[nodiscard]] bool IsValid() const { return handle != 0; }
};

/// An acceleration structure as a descriptor and a TLAS instance name it.
struct RtSceneRef {
    NativeHandle handle = 0;  // VkAccelerationStructureKHR | MTL::AccelerationStructure*
    u64 address = 0;          // device address | gpuResourceID
    [[nodiscard]] bool IsValid() const { return handle != 0; }
    friend bool operator==(const RtSceneRef&, const RtSceneRef&) = default;
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
    u32 maxBindlessTextures = 0;           // 1024 with descriptor indexing, else 32
    u32 maxTraceRecursionDepth = 0;        // maxRayRecursionDepth; 0 without ray tracing
    u64 rtScratchAlignment = 0;            // minAccelerationStructureScratchOffsetAlignment, as reported
    bool timestampsOnAllQueues = false;    // limits.timestampComputeAndGraphics
    bool linearFilterRGBA32Float = false;  // RGBA32Float supports linear filtering (optimal tiling)
};

}  // namespace quantiloom::gpu
