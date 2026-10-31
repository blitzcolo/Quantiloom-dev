/**
 * @file GpuInterop.hpp
 * @brief Backend-neutral handles between a host's GPU objects and the SDK
 *
 * The SDK's public surface carries no Vulkan (or Metal) types. A host that
 * owns a device, a command buffer and a frame texture describes them with the
 * PODs here, and the typed builders in renderer/interop/ fill those PODs in.
 *
 * Handles are integers, like QRhiTexture::NativeTexture{quint64 object}:
 * u64 rather than void* because VkImage/VkBuffer are non-dispatchable
 * handles on some platforms. Nothing here is owned by the SDK -- every handle
 * must stay valid for as long as the SDK may use it.
 *
 * Every struct carries a version. ExternalRenderContext::Create() rejects a
 * version it was not built against, so a host compiled against a different
 * struct generation gets an error rather than undefined behaviour.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom {

enum class GpuBackend : u32 { Vulkan = 1, Metal = 2 };

/// Bumped only together with an SDK major version; Create() rejects a mismatch.
inline constexpr u32 kGpuInteropVersion = 1;

/// Native handles as integers, like QRhiTexture::NativeTexture{quint64 object}.
/// u64 rather than void* because VkImage/VkBuffer are non-dispatchable handles.
using GpuHandle = u64;

/// What ExternalRenderContext::Create() runs on.
/// Vulkan: instance=VkInstance, physicalDevice=VkPhysicalDevice, device=VkDevice,
///         queue=VkQueue, queueFamily/queueIndex as created, getProcAddr=
///         PFN_vkGetInstanceProcAddr, vmaAllocator optional (VmaAllocator).
/// Metal:  device=id<MTLDevice>, queue=id<MTLCommandQueue>; the rest zero.
struct GpuDeviceHandles {
    GpuBackend backend{};
    u32 version = kGpuInteropVersion;
    GpuHandle instance = 0;
    GpuHandle physicalDevice = 0;
    GpuHandle device = 0;
    GpuHandle queue = 0;
    u32 queueFamily = 0;
    u32 queueIndex = 0;
    GpuHandle getProcAddr = 0;
    GpuHandle vmaAllocator = 0;   // former InitParams::externalAllocator
};

/// The command buffer a recording call appends to.
/// Vulkan: a VkCommandBuffer in the recording state, outside any render pass.
/// Metal:  an id<MTLCommandBuffer> with no encoder open.
struct GpuCommandRecording {
    GpuBackend backend{};
    u32 version = kGpuInteropVersion;
    GpuHandle commandBuffer = 0;
};

enum class GpuTargetFormat : u32 { Rgba8Srgb = 1, Bgra8Srgb = 2, Rgba16Float = 3, R32Float = 4 };

/// A host-owned image a call writes in full.
/// Vulkan: image=VkImage; currentLayout/finalLayout are VkImageLayout values.
/// Metal:  image=id<MTLTexture>; layouts ignored.
struct GpuImageTarget {
    GpuBackend backend{};
    u32 version = kGpuInteropVersion;
    GpuHandle image = 0;
    GpuTargetFormat format{};
    u32 width = 0;
    u32 height = 0;
    u32 currentLayout = 0;
    u32 finalLayout = 0;
};

}  // namespace quantiloom
