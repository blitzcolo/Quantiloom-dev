#include "Upload.hpp"
#include "CommandHelper.hpp"
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
    const GeometryPrimitive& primitive) {

    if (primitive.positions.empty()) {
        QL_LOG_WARN("GeometryPrimitive has no vertices, skipping vertex buffer upload");
        return nullptr;
    }

    VkDeviceSize bufferSize = sizeof(glm::vec3) * primitive.positions.size();

    // Create staging buffer (HOST_VISIBLE for CPU write)
    GpuBuffer stagingBuffer(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Upload data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, primitive.positions.data(), bufferSize);
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
                primitive.positions.size(), bufferSize);

    return deviceBuffer;
}

// ============================================================================
// Index Buffer Upload
// ============================================================================

std::unique_ptr<GpuBuffer> UploadIndexBuffer(
    const VulkanContext& ctx,
    const GeometryPrimitive& primitive) {

    if (primitive.indices.empty()) {
        QL_LOG_WARN("GeometryPrimitive has no indices, skipping index buffer upload");
        return nullptr;
    }

    VkDeviceSize bufferSize = sizeof(u32) * primitive.indices.size();

    // Create staging buffer
    GpuBuffer stagingBuffer(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Upload data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, primitive.indices.data(), bufferSize);
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
                primitive.indices.size(), bufferSize);

    return deviceBuffer;
}

// ============================================================================
// Image Upload
// ============================================================================

std::unique_ptr<GpuImage> UploadImage(
    const VulkanContext& ctx,
    const Image& cpuImage) {

    if (!cpuImage.IsValid()) {
        QL_LOG_WARN("Image is invalid, skipping upload");
        return nullptr;
    }

    // Determine VkFormat based on channel count
    VkFormat format = VK_FORMAT_UNDEFINED;
    u32 bytesPerPixel = 0;

    switch (cpuImage.channels) {
        case 1:
            format = VK_FORMAT_R32_SFLOAT;
            bytesPerPixel = sizeof(f32) * 1;
            break;
        case 2:
            format = VK_FORMAT_R32G32_SFLOAT;
            bytesPerPixel = sizeof(f32) * 2;
            break;
        case 3:
            format = VK_FORMAT_R32G32B32_SFLOAT;
            bytesPerPixel = sizeof(f32) * 3;
            break;
        case 4:
            format = VK_FORMAT_R32G32B32A32_SFLOAT;
            bytesPerPixel = sizeof(f32) * 4;
            break;
        default:
            QL_LOG_ERROR("Unsupported channel count: {}", cpuImage.channels);
            return nullptr;
    }

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(cpuImage.width) * cpuImage.height * bytesPerPixel;

    QL_LOG_INFO("Uploading image: {}x{} ({} channels, {} bytes)",
                cpuImage.width, cpuImage.height, cpuImage.channels, bufferSize);

    // Step 1: Create staging buffer (CPU-accessible)
    GpuBuffer stagingBuffer(
        ctx.GetAllocator(),
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );

    // Step 2: Upload CPU data to staging buffer
    void* data = stagingBuffer.Map();
    std::memcpy(data, cpuImage.data.data(), bufferSize);
    stagingBuffer.Unmap();

    // Step 3: Create device-local GPU image
    auto gpuImage = std::make_unique<GpuImage>(
        ctx.GetAllocator(),
        ctx.GetDevice(),
        cpuImage.width,
        cpuImage.height,
        format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        1  // mipLevels
    );

    // Step 4: Execute upload via command buffer
    VkCommandPool commandPool = CreateTransferCommandPool(ctx.GetDevice(), ctx.GetGraphicsQueueFamily());
    VkCommandBuffer cmd = BeginSingleTimeCommands(ctx.GetDevice(), commandPool);

    // Transition: UNDEFINED -> TRANSFER_DST_OPTIMAL
    CommandHelper::TransitionImageLayout(
        cmd,
        gpuImage->GetImage(),
        format,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1  // mipLevels
    );

    // Define copy region (buffer -> image)
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;   // Tightly packed
    region.bufferImageHeight = 0; // Tightly packed
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {cpuImage.width, cpuImage.height, 1};

    // Copy staging buffer to GPU image
    vkCmdCopyBufferToImage(
        cmd,
        stagingBuffer.GetHandle(),
        gpuImage->GetImage(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    // Transition: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    CommandHelper::TransitionImageLayout(
        cmd,
        gpuImage->GetImage(),
        format,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        1  // mipLevels
    );

    EndSingleTimeCommands(ctx.GetDevice(), commandPool, ctx.GetGraphicsQueue(), cmd);
    vkDestroyCommandPool(ctx.GetDevice(), commandPool, nullptr);

    QL_LOG_INFO("  Image uploaded successfully ({}x{}, format: {})",
                cpuImage.width, cpuImage.height, static_cast<int>(format));

    return gpuImage;
}

} // namespace quantiloom
