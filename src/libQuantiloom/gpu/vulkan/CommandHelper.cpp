#include "CommandHelper.hpp"
#include "GpuBuffer.hpp"
#include "core/Log.hpp"
#include <limits>
#include <mutex>
#include <stdexcept>
#include <cstring>

namespace quantiloom {

// ============================================================================
// One-Time Command Execution
// ============================================================================

void CommandHelper::ExecuteImmediate(
    const VulkanContext& context,
    const std::function<void(VkCommandBuffer)> &recordFunc)
{
    VkDevice device = context.GetDevice();
    VkQueue queue = context.GetGraphicsQueue();
    constexpr u64 kWaitTimeoutNs = std::numeric_limits<u64>::max();

    // Recursive: a record lambda may itself allocate resources that submit an
    // immediate command buffer (e.g. a camera image allocation inside a
    // RenderFrame recorded under ExecuteImmediate). Other threads still block
    // here, so cross-thread callers serialize. The depth counter is only
    // touched while the lock is held.
    std::lock_guard<std::recursive_mutex> lock(context.m_immediateSubmitMutex);
    const bool nested = context.m_immediateDepth > 0;
    struct DepthGuard {
        int& depth;
        explicit DepthGuard(int& d) : depth(d) { ++depth; }
        ~DepthGuard() { --depth; }
    } depthGuard(context.m_immediateDepth);

    // The outermost call uses the shared pool and fence. A nested call runs
    // while the outer command buffer is still being recorded, so it gets its
    // own pool (the outer pool is mid-recording) and its own fence (the outer
    // one is not yet submitted); both are destroyed on exit.
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool poolReclaimable = true;  // false once a wait failed: cb may be pending
    struct TempGuard {
        VkDevice device;
        VkCommandPool& pool;
        VkFence& fence;
        bool active;
        bool& poolReclaimable;
        ~TempGuard() {
            if (!active) return;
            if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
            if (pool != VK_NULL_HANDLE && poolReclaimable)
                vkDestroyCommandPool(device, pool, nullptr);
        }
    } tempGuard{device, pool, fence, nested, poolReclaimable};

    if (nested) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = context.GetGraphicsQueueFamily();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
            throw std::runtime_error("Failed to create nested immediate command pool");
        }
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            fence = VK_NULL_HANDLE;
            throw std::runtime_error("Failed to create nested immediate fence");
        }
    } else {
        // Lazily create the shared pool and fence once per context
        if (context.m_immediateCommandPool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = context.GetGraphicsQueueFamily();
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            if (vkCreateCommandPool(device, &poolInfo, nullptr,
                                    &context.m_immediateCommandPool) != VK_SUCCESS) {
                context.m_immediateCommandPool = VK_NULL_HANDLE;
                throw std::runtime_error("Failed to create immediate command pool");
            }
        }
        if (context.m_immediateFence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(device, &fenceInfo, nullptr,
                              &context.m_immediateFence) != VK_SUCCESS) {
                context.m_immediateFence = VK_NULL_HANDLE;
                throw std::runtime_error("Failed to create immediate submission fence");
            }
        }
        pool = context.m_immediateCommandPool;
        fence = context.m_immediateFence;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &cmd);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }

    auto freeCommandBuffer = [&]() {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
    };

    // Drop a shared fence whose state is unknown so the next call recreates it
    auto discardFence = [&]() {
        if (nested) return;  // TempGuard destroys the nested fence
        vkDestroyFence(device, context.m_immediateFence, nullptr);
        context.m_immediateFence = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    };

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(cmd, &beginInfo);
    if (result != VK_SUCCESS) {
        freeCommandBuffer();
        throw std::runtime_error("Failed to begin immediate command buffer");
    }

    try {
        recordFunc(cmd);
    } catch (...) {
        freeCommandBuffer();
        throw;
    }

    result = vkEndCommandBuffer(cmd);
    if (result != VK_SUCCESS) {
        freeCommandBuffer();
        throw std::runtime_error("Failed to end immediate command buffer");
    }

    // Fence lifecycle: unsignaled -> submitted -> waited -> reset
    // (VUID-vkQueueSubmit-fence-00063 forbids submitting a signaled fence).
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    result = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        freeCommandBuffer();
        QL_LOG_ERROR("  [DEBUG CommandHelper] vkQueueSubmit FAILED with VkResult: {}", static_cast<int>(result));
        throw std::runtime_error("Failed to submit command buffer (VkResult: " + std::to_string(result) + ")");
    }

    result = vkWaitForFences(device, 1, &fence, VK_TRUE, kWaitTimeoutNs);
    if (result != VK_SUCCESS) {
        // Fence state unknown; the command buffer may still be pending and is
        // not freed (reclaimed with the pool at ~VulkanContext).
        poolReclaimable = false;
        discardFence();
        QL_LOG_ERROR("  [DEBUG CommandHelper] vkWaitForFences FAILED with VkResult: {} (GPU CRASHED)", static_cast<int>(result));
        throw std::runtime_error("Failed to wait for immediate submission (VkResult: " + std::to_string(result) + ")");
    }

    if (!nested) {
        result = vkResetFences(device, 1, &fence);
        if (result != VK_SUCCESS) {
            freeCommandBuffer();
            discardFence();
            throw std::runtime_error("Failed to reset immediate fence (VkResult: " +
                                     std::to_string(result) + ")");
        }
    }

    freeCommandBuffer();
}

// ============================================================================
// Image Layout Transitions
// ============================================================================

void CommandHelper::TransitionImageLayout(
    VkCommandBuffer cmd,
    VkImage image,
    const VkFormat format,
    const VkImageLayout oldLayout,
    const VkImageLayout newLayout,
    const u32 mipLevels,
    const u32 arrayLayers)
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
    barrier.subresourceRange.layerCount = arrayLayers;  // Support cubemaps (6 layers)

    // Determine access masks and pipeline stages based on layouts
    VkPipelineStageFlags srcStage = 0;
    VkPipelineStageFlags dstStage = 0;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        // Common case: Initialize storage image for ray tracing output
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        // Prepare for texture upload (buffer -> image)
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        // After upload, prepare for shader reads
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        // Prepare for readback (e.g., copying to staging buffer)
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        // Return to GENERAL after transfer
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
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
    const VulkanContext& context,
    VkImage image,
    const VkFormat format,
    const VkImageLayout oldLayout,
    const VkImageLayout newLayout,
    u32 mipLevels,
    u32 arrayLayers)
{
    ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        TransitionImageLayout(cmd, image, format, oldLayout, newLayout, mipLevels, arrayLayers);
    });

    QL_LOG_INFO("Image layout transition: {} -> {} (immediate, mips={}, layers={})",
                static_cast<int>(oldLayout), static_cast<int>(newLayout), mipLevels, arrayLayers);
}

// ============================================================================
// Image Readback
// ============================================================================

std::vector<f32> CommandHelper::ReadbackImage(
    const VulkanContext& context,
    VkImage image,
    const VkFormat format,
    u32 width,
    u32 height)
{
    // Validate format (M1: only support RGBA32F)
    if (format != VK_FORMAT_R32G32B32A32_SFLOAT) {
        throw std::runtime_error("ReadbackImage: Only VK_FORMAT_R32G32B32A32_SFLOAT is supported in M1");
    }

    // Calculate buffer size (4 channels * 4 bytes per float)
    constexpr u32 bytesPerPixel = 4 * sizeof(f32);
    const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * height * bytesPerPixel;

    //QL_LOG_DEBUG("Reading back image ({}x{}, {} bytes)...", width, height, bufferSize);

    // Create staging buffer (GPU -> CPU)
    GpuBuffer stagingBuffer(
        context.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_TO_CPU
    );

    // Execute copy command
    ExecuteImmediate(context, [&](VkCommandBuffer cmd) {
        // Transition image to TRANSFER_SRC_OPTIMAL (if currently in GENERAL)
        TransitionImageLayout(cmd, image, format,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        // Define copy region (entire image, mip level 0)
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;   // Tightly packed
        region.bufferImageHeight = 0; // Tightly packed
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        // Copy image to buffer
        vkCmdCopyImageToBuffer(
            cmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            stagingBuffer.GetHandle(),
            1,
            &region
        );

        // Transition image back to GENERAL (restore original layout)
        TransitionImageLayout(cmd, image, format,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_IMAGE_LAYOUT_GENERAL);
    });

    // Map staging buffer and read pixel data
    std::vector<f32> pixels(width * height * 4);
    const void* mappedData = stagingBuffer.MapRead();
    if (mappedData == nullptr) {
        throw std::runtime_error("Failed to map staging buffer for readback");
    }

    std::memcpy(pixels.data(), mappedData, bufferSize);
    stagingBuffer.Unmap();

    //QL_LOG_DEBUG("  Readback complete ({} pixels)", width * height);

    return pixels;
}

} // namespace quantiloom
