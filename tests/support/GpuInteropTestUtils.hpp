/**
 * @file GpuInteropTestUtils.hpp
 * @brief Fill the neutral GPU PODs from a test device, as a host would
 *
 * The public context API takes no backend types, so a test hands ExternalRenderContext
 * the same PODs a host builds (renderer/GpuInterop.hpp): its device's handles, the
 * command buffer a gpu::CommandList records into, and a test-owned texture as a
 * present target. The builders dispatch on the backend (GpuTestDevice.cpp).
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/GpuTypes.hpp"
#include "renderer/GpuInterop.hpp"

namespace quantiloom::gpu {
class CommandList;
class Device;
}  // namespace quantiloom::gpu

namespace quantiloom::testing {

/// Host layout tokens (GpuImageTarget::currentLayout/finalLayout, and
/// gpu::TextureBarrier::hostLayoutBefore/After for a layout no access implies) for the
/// images the tests own: the Vulkan VkImageLayout values, pinned in VulkanTestDevice.cpp.
/// gpu::kHostLayoutShaderReadOnly is the fourth. Metal ignores layouts.
inline constexpr u32 kHostLayoutUndefined = 0;
inline constexpr u32 kHostLayoutGeneral = 1;
inline constexpr u32 kHostLayoutTransferDst = 7;

/// The handles a host would give ExternalRenderContext::Create for @p device.
GpuDeviceHandles GpuHandles(gpu::Device& device);

/// The command buffer @p list records into, as a host's recording.
GpuCommandRecording GpuRecording(gpu::CommandList& list);

/// A test-owned texture of @p backend as a host's target, its extent from the ref.
GpuImageTarget GpuTarget(gpu::Backend backend, gpu::TextureRef texture, GpuTargetFormat format,
                         u32 currentLayout, u32 finalLayout);

}  // namespace quantiloom::testing
