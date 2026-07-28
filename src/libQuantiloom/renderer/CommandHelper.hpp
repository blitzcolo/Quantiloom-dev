/**
 * @file CommandHelper.hpp
 * @brief One-time command buffer utilities for immediate GPU operations
 *
 * Provides CommandHelper class with static utility functions for:
 * - One-shot command buffer execution (synchronous GPU operations)
 * - Image layout transitions (pipeline barriers)
 * - Image readback (GPU-to-CPU transfer)
 *
 * All operations are synchronous (block until GPU completes).
 * Command buffers are allocated from a transient pool, submitted, and freed automatically.
 *
 * Common use cases:
 * - Acceleration structure builds (BLAS/TLAS)
 * - Image layout transitions (UNDEFINED → GENERAL, TRANSFER_DST → SHADER_READ_ONLY)
 * - Texture uploads (staging buffer → GPU image)
 * - Render target readback (GPU → CPU for saving to disk)
 *
 * @note All functions are synchronous (vkQueueWaitIdle)
 * @note Not suitable for per-frame operations (use persistent command buffers instead)
 * @note Designed for initialization and cleanup only
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "VulkanContext.hpp"
#include <vulkan/vulkan.h>
#include <functional>

// ============================================================================
// CommandHelper - One-time command buffer submission utilities
// ============================================================================

namespace quantiloom {

/**
 * @class CommandHelper
 * @brief Static utility class for immediate GPU command execution
 *
 * Provides convenience functions for common one-time GPU operations.
 * All commands execute synchronously (blocking CPU until GPU completes).
 *
 * ExecuteImmediate workflow:
 * 1. Allocate temporary command buffer from graphics queue
 * 2. Begin command buffer recording
 * 3. Execute user-provided lambda function (records commands)
 * 4. End command buffer recording
 * 5. Submit to graphics queue
 * 6. Wait for completion (vkQueueWaitIdle)
 * 7. Free command buffer
 *
 * Usage patterns:
 * @code
 * // Acceleration structure builds
 * CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
 *     for (auto& blas : blasList) {
 *         blas.Build(cmd);
 *     }
 *     tlas.Build(cmd);
 * });
 *
 * // Image layout transition (immediate)
 * CommandHelper::TransitionImageLayoutImmediate(
 *     context, image, format,
 *     VK_IMAGE_LAYOUT_UNDEFINED,
 *     VK_IMAGE_LAYOUT_GENERAL
 * );
 *
 * // Readback render output to CPU
 * std::vector<f32> pixels = CommandHelper::ReadbackImage(
 *     context, outputImage.GetImage(), outputImage.GetFormat(),
 *     width, height
 * );
 * @endcode
 *
 * @note All methods block CPU until GPU completes
 * @note Use persistent command buffers for per-frame operations
 * @note TransitionImageLayout requires source/dest access masks to be inferred from layouts
 *
 * @see VulkanContext for queue access
 * @see GpuImage for image resource management
 */
class CommandHelper {
public:
    // ========================================================================
    // One-Time Command Execution
    // ========================================================================

    // Execute commands immediately (synchronous)
    // Creates temporary command buffer, records, submits, and waits for completion
    static void ExecuteImmediate(
        const VulkanContext& context,
        const std::function<void(VkCommandBuffer)> &recordFunc
    );

    // ========================================================================
    // Image Layout Transitions
    // ========================================================================

    // Transition image layout with full pipeline barrier
    static void TransitionImageLayout(
        VkCommandBuffer cmd,
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        u32 mipLevels = 1,
        u32 arrayLayers = 1  // Added for cubemap support (6 layers)
    );

    // Immediate layout transition (creates and submits command buffer)
    static void TransitionImageLayoutImmediate(
        const VulkanContext& context,
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        u32 mipLevels = 1,
        u32 arrayLayers = 1  // Added for cubemap support (6 layers)
    );

    // ========================================================================
    // Image Readback
    // ========================================================================

    // Read back image from GPU to CPU memory (synchronous)
    // Returns pixel data in row-major order: [R,G,B,A, R,G,B,A, ...]
    // Only supports VK_FORMAT_R32G32B32A32_SFLOAT for M1
    // Image must be in GENERAL or TRANSFER_SRC_OPTIMAL layout
    static std::vector<f32> ReadbackImage(
        const VulkanContext& context,
        VkImage image,
        VkFormat format,
        u32 width,
        u32 height
    );
};

} // namespace quantiloom
