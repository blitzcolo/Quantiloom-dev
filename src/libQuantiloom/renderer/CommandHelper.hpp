#pragma once

#include "core/Types.hpp"
#include "VulkanContext.hpp"
#include <vulkan/vulkan.h>
#include <functional>

// ============================================================================
// CommandHelper - One-time command buffer submission utilities
// ============================================================================
// Provides helper functions for:
// - One-shot command buffer execution (e.g., AS builds, layout transitions)
// - Image layout transitions
// - Buffer/image copies
//
// Usage pattern:
//   CommandHelper::ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
//       // Record commands here
//       blas.Build(cmd);
//       tlas.Build(cmd);
//   });
// ============================================================================

namespace quantiloom {

class QL_API CommandHelper {
public:
    // ========================================================================
    // One-Time Command Execution
    // ========================================================================

    // Execute commands immediately (synchronous)
    // Creates temporary command buffer, records, submits, and waits for completion
    static void ExecuteImmediate(
        VulkanContext& context,
        std::function<void(VkCommandBuffer)> recordFunc
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
        u32 mipLevels = 1
    );

    // Immediate layout transition (creates and submits command buffer)
    static void TransitionImageLayoutImmediate(
        VulkanContext& context,
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        u32 mipLevels = 1
    );

    // ========================================================================
    // Image Readback
    // ========================================================================

    // Read back image from GPU to CPU memory (synchronous)
    // Returns pixel data in row-major order: [R,G,B,A, R,G,B,A, ...]
    // Only supports VK_FORMAT_R32G32B32A32_SFLOAT for M1
    // Image must be in GENERAL or TRANSFER_SRC_OPTIMAL layout
    static std::vector<f32> ReadbackImage(
        VulkanContext& context,
        VkImage image,
        VkFormat format,
        u32 width,
        u32 height
    );
};

} // namespace quantiloom
