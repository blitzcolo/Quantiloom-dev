/**
 * @file MetalInterop.hpp
 * @brief Metal builders for the neutral GPU API (header-only)
 *
 * Handles are bridged void* rather than Objective-C types, so this header
 * needs no Objective-C and exports nothing. The Metal backend is implemented
 * behind the same exported signatures as Vulkan's; a build without it answers
 * Create() with "this SDK build has no Metal backend".
 */

#pragma once

#include "renderer/GpuInterop.hpp"

namespace quantiloom::mtl {

/// @param mtlDevice      id<MTLDevice>, not retained
/// @param mtlCommandQueue id<MTLCommandQueue>, not retained
inline GpuDeviceHandles Handles(void* mtlDevice, void* mtlCommandQueue) {
    GpuDeviceHandles h;
    h.backend = GpuBackend::Metal;
    h.device = reinterpret_cast<GpuHandle>(mtlDevice);
    h.queue = reinterpret_cast<GpuHandle>(mtlCommandQueue);
    return h;
}

/// @param mtlCommandBuffer id<MTLCommandBuffer> with no encoder open
inline GpuCommandRecording Commands(void* mtlCommandBuffer) {
    return {GpuBackend::Metal, kGpuInteropVersion, reinterpret_cast<GpuHandle>(mtlCommandBuffer)};
}

/// @param mtlTexture id<MTLTexture> the call writes in full; layouts do not apply
inline GpuImageTarget Target(void* mtlTexture, GpuTargetFormat format, uint32_t w, uint32_t h) {
    GpuImageTarget t;
    t.backend = GpuBackend::Metal;
    t.image = reinterpret_cast<GpuHandle>(mtlTexture);
    t.format = format;
    t.width = w;
    t.height = h;
    return t;
}

}  // namespace quantiloom::mtl
