/**
 * @file TextureTransitions.hpp
 * @brief The legacy image layout transitions and the RGBA32F readback, on gpu:: (internal)
 *
 * CommandHelper's transition table and readback, re-expressed as gpu::CommandList calls.
 * Each supported transition is one gpu::TextureBarrier whose accesses give, on Vulkan,
 * exactly the barrier CommandHelper built by hand. Callers move here from CommandHelper
 * one migration step at a time.
 */

#pragma once

#include "core/Types.hpp"
#include "gpu/GpuTypes.hpp"

#include <vector>

namespace quantiloom::gpu {
class CommandList;
class Device;
}  // namespace quantiloom::gpu

namespace quantiloom::rendercore {

/// The layouts the legacy transitions name. Only Vulkan has layouts; on Metal each
/// transition below orders the accesses and nothing else.
enum class Layout : u8 { Undefined, General, TransferDst, TransferSrc, ShaderReadOnly };

/**
 * @brief One barrier over mips [0, mips) and layers [0, layers) of @p texture
 *
 * The texture's layout policy does not matter; the barrier names it under Optimal. The
 * supported pairs, and what each orders:
 *
 * | from -> to                  | before                       | after                        |
 * |-----------------------------|------------------------------|------------------------------|
 * | Undefined -> General        | nothing (contents discarded) | ray tracing and compute writes |
 * | Undefined -> TransferDst    | nothing (contents discarded) | transfer writes              |
 * | TransferDst -> ShaderReadOnly | transfer writes            | ray tracing and fragment reads |
 * | General -> TransferSrc      | ray tracing and compute writes | transfer reads             |
 * | TransferSrc -> General      | transfer reads               | ray tracing and compute writes |
 *
 * Any other pair logs a warning and orders all commands before against all commands
 * after with no memory access, as CommandHelper's fallback did. No caller uses one.
 */
void LegacyTransition(gpu::CommandList& list, gpu::TextureRef texture, Layout from, Layout to,
                      u32 mips = 1, u32 layers = 1);

/// LegacyTransition in its own Device::ExecuteImmediate submission, logged at info level as
/// CommandHelper::TransitionImageLayoutImmediate was.
void LegacyTransitionImmediate(gpu::Device& device, gpu::TextureRef texture, Layout from,
                               Layout to, u32 mips = 1, u32 layers = 1);

/**
 * @brief Read mip 0, layer 0 of an RGBA32Float image back to the host (synchronous)
 *
 * The image is in the General layout before and after: it moves to TransferSrc for the
 * copy and back. Row-major [R,G,B,A, R,G,B,A, ...].
 *
 * @throws std::runtime_error for any other format, when the staging buffer cannot be
 *         created or mapped, or when the submission fails
 */
[[nodiscard]] std::vector<f32> ReadbackTexture(gpu::Device& device, gpu::TextureRef texture,
                                               u32 width, u32 height);

}  // namespace quantiloom::rendercore
