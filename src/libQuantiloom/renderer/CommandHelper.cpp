#include "CommandHelper.hpp"
#include "core/Log.hpp"
#include <stdexcept>

namespace quantiloom {

// ============================================================================
// One-Time Command Execution
// ============================================================================

void CommandHelper::ExecuteImmediate(
    VulkanContext& context,
    std::function<void(VkCommandBuffer)> recordFunc)
{
    VkDevice device = context.GetDevice();
    VkQueue queue = context.GetGraphicsQueue();
    u32 queueFamily = context.GetGraphicsQueueFamily();

    // Create temporary command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;  // Short-lived buffers

    VkCommandPool commandPool;
    VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create temporary command pool");
    }

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    result = vkAllocateCommandBuffers(device, &allocInfo, &cmd);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        throw std::runtime_error("Failed to allocate command buffer");
    }

    // Begin recording
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &beginInfo);

    // Execute user-provided recording function
    recordFunc(cmd);

    // End recording
    vkEndCommandBuffer(cmd);

    // Submit and wait for completion
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        throw std::runtime_error("Failed to submit command buffer");
    }

    // Wait for completion (synchronous)
    vkQueueWaitIdle(queue);

    // Cleanup
    vkDestroyCommandPool(device, commandPool, nullptr);
}

// ============================================================================
// Image Layout Transitions
// ============================================================================

void CommandHelper::TransitionImageLayout(
    VkCommandBuffer cmd,
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    u32 mipLevels)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // Determine access masks and pipeline stages based on layouts
    VkPipelineStageFlags srcStage = 0;
    VkPipelineStageFlags dstStage = 0;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        // Common case: Initialize storage image for ray tracing output
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        // Prepare for readback (e.g., copying to staging buffer)
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        // Return to GENERAL after transfer
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    }
    else {
        // Fallback for unsupported transitions
        QL_LOG_WARN("Unsupported layout transition: {} -> {}",
                    static_cast<int>(oldLayout), static_cast<int>(newLayout));
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(
        cmd,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void CommandHelper::TransitionImageLayoutImmediate(
    VulkanContext& context,
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    u32 mipLevels)
{
    ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        TransitionImageLayout(cmd, image, format, oldLayout, newLayout, mipLevels);
    });

    QL_LOG_INFO("Image layout transition: {} -> {} (immediate)",
                static_cast<int>(oldLayout), static_cast<int>(newLayout));
}

} // namespace quantiloom
