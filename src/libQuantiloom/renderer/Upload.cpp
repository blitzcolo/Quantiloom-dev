#include "Upload.hpp"
#include "core/Log.hpp"
#include <stdexcept>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Helper: Execute one-time command buffer
// ============================================================================

namespace {

// Execute a single-use command buffer (synchronous)
// Returns VkCommandBuffer that caller must execute
VkCommandBuffer BeginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void EndSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue queue, VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);  // Synchronous, blocking

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Create a temporary command pool (for upload operations)
VkCommandPool CreateTransferCommandPool(VkDevice device, u32 queueFamily) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;  // Short-lived commands

    VkCommandPool commandPool;
    VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create transfer command pool");
    }

    return commandPool;
}

} // anonymous namespace

// ============================================================================
// Vertex Buffer Upload
// ============================================================================

std::unique_ptr<GpuBuffer> UploadVertexBuffer(
    const VulkanContext& ctx,
    const Mesh& mesh) {

    if (mesh.positions.empty()) {
        QL_LOG_WARN("Mesh has no vertices, skipping vertex buffer upload");
        return nullptr;
    }

    VkDeviceSize bufferSize = sizeof(glm::vec3) * mesh.positions.size();

    // Create staging buffer (HOST_VISIBLE for CPU write)
    GpuBuffer stagingBuffer(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Upload data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, mesh.positions.data(), bufferSize);
    stagingBuffer.Unmap();

    // Create device-local buffer (GPU_ONLY for optimal performance)
    auto deviceBuffer = std::make_unique<GpuBuffer>(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |  // For ray tracing
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Copy staging -> device buffer via command buffer
    VkCommandPool commandPool = CreateTransferCommandPool(ctx.GetDevice(), ctx.GetGraphicsQueueFamily());
    VkCommandBuffer cmd = BeginSingleTimeCommands(ctx.GetDevice(), commandPool);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = bufferSize;

    vkCmdCopyBuffer(cmd, stagingBuffer.GetHandle(), deviceBuffer->GetHandle(), 1, &copyRegion);

    EndSingleTimeCommands(ctx.GetDevice(), commandPool, ctx.GetGraphicsQueue(), cmd);
    vkDestroyCommandPool(ctx.GetDevice(), commandPool, nullptr);

    QL_LOG_INFO("Uploaded vertex buffer: {} vertices ({} bytes)",
                mesh.positions.size(), bufferSize);

    return deviceBuffer;
}

// ============================================================================
// Index Buffer Upload
// ============================================================================

std::unique_ptr<GpuBuffer> UploadIndexBuffer(
    const VulkanContext& ctx,
    const Mesh& mesh) {

    if (mesh.indices.empty()) {
        QL_LOG_WARN("Mesh has no indices, skipping index buffer upload");
        return nullptr;
    }

    VkDeviceSize bufferSize = sizeof(u32) * mesh.indices.size();

    // Create staging buffer
    GpuBuffer stagingBuffer(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Upload data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, mesh.indices.data(), bufferSize);
    stagingBuffer.Unmap();

    // Create device-local buffer
    auto deviceBuffer = std::make_unique<GpuBuffer>(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |  // For ray tracing
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Copy staging -> device buffer
    VkCommandPool commandPool = CreateTransferCommandPool(ctx.GetDevice(), ctx.GetGraphicsQueueFamily());
    VkCommandBuffer cmd = BeginSingleTimeCommands(ctx.GetDevice(), commandPool);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = bufferSize;

    vkCmdCopyBuffer(cmd, stagingBuffer.GetHandle(), deviceBuffer->GetHandle(), 1, &copyRegion);

    EndSingleTimeCommands(ctx.GetDevice(), commandPool, ctx.GetGraphicsQueue(), cmd);
    vkDestroyCommandPool(ctx.GetDevice(), commandPool, nullptr);

    QL_LOG_INFO("Uploaded index buffer: {} indices ({} bytes)",
                mesh.indices.size(), bufferSize);

    return deviceBuffer;
}

// ============================================================================
// Image Upload (placeholder for M1+)
// ============================================================================

std::unique_ptr<GpuImage> UploadImage(
    const VulkanContext& ctx,
    const Image& cpuImage) {

    (void)ctx;
    (void)cpuImage;

    QL_LOG_WARN("UploadImage not yet implemented (TODO: M1)");
    return nullptr;
}

} // namespace quantiloom
