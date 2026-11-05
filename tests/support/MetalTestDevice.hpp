/**
 * @file MetalTestDevice.hpp
 * @brief The Metal side of the host-interop builders and native inspections
 *
 * What VulkanTestDevice.hpp declares for Vulkan, for a Metal device: the host PODs
 * GpuHandles/GpuRecording build (GpuInteropTestUtils.hpp) and the two native
 * inspections GpuTestHelpers.hpp dispatches. Implemented in support/metal/ on a build
 * with the Metal backend and by support/NoMetalTestDevice.cpp everywhere else.
 */

#pragma once

#include "core/Types.hpp"
#include "renderer/GpuInterop.hpp"

namespace quantiloom::gpu {
class CommandList;
class Device;
class Texture;
}  // namespace quantiloom::gpu

namespace quantiloom::testing {

GpuDeviceHandles MetalHostHandles(gpu::Device& device);
GpuCommandRecording MetalHostRecording(gpu::CommandList& list);
u32 MetalTextureMipLevels(const gpu::Texture& texture);
f32 MetalPackedInstanceTranslationX(const void* packed, u32 index);

}  // namespace quantiloom::testing
