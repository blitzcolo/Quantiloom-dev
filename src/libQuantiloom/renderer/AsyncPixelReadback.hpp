#pragma once

#include "renderer/GpuBuffer.hpp"
#include "renderer/PixelReading.hpp"
#include "renderer/VulkanContext.hpp"
#include <cstring>
#include <stdexcept>

namespace quantiloom::rendercore {

/// Queue-ordered pixel copies with reusable resources. Poll never waits;
/// destruction is the only path that waits for outstanding copies. The host
/// must submit its recorded render before requesting a copy on the same queue.
class AsyncPixelReadback {
public:
    explicit AsyncPixelReadback(const VulkanContext& context) : m_context(context) {
        try {
            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.queueFamilyIndex = context.GetGraphicsQueueFamily();
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            Check(vkCreateCommandPool(context.GetDevice(), &pool, nullptr, &m_pool));
            for (auto& slot : m_slots) {
                slot.buffer = std::make_unique<GpuBuffer>(context.GetAllocator(), sizeof(glm::vec4),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
                VkFenceCreateInfo fence{};
                fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                Check(vkCreateFence(context.GetDevice(), &fence, nullptr, &slot.fence));
                VkCommandBufferAllocateInfo command{};
                command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                command.commandPool = m_pool;
                command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                command.commandBufferCount = 1;
                Check(vkAllocateCommandBuffers(context.GetDevice(), &command, &slot.command));
            }
        } catch (...) {
            Cleanup();
            throw;
        }
    }

    ~AsyncPixelReadback() { Cleanup(); }
    AsyncPixelReadback(const AsyncPixelReadback&) = delete;
    AsyncPixelReadback& operator=(const AsyncPixelReadback&) = delete;

    bool Submit(VkImage image, u32 renderX, u32 renderY, const PixelReading& reading) {
        for (auto& slot : m_slots) {
            if (slot.pending) continue;
            Check(vkResetCommandBuffer(slot.command, 0));
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Check(vkBeginCommandBuffer(slot.command, &begin));

            // Keep GENERAL so the copy does not change the accumulation's
            // layout contract. Both prior shader writes and subsequent
            // shader accesses are ordered against this read.
            VkImageMemoryBarrier imageBarrier{};
            imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.image = image;
            imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageBarrier.subresourceRange.levelCount = 1;
            imageBarrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(slot.command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);

            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.layerCount = 1;
            copy.imageOffset = {static_cast<i32>(renderX), static_cast<i32>(renderY), 0};
            copy.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(slot.command, image, VK_IMAGE_LAYOUT_GENERAL,
                                  slot.buffer->GetHandle(), 1, &copy);

            imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            VkBufferMemoryBarrier host{};
            host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host.buffer = slot.buffer->GetHandle();
            host.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(slot.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &host, 1, &imageBarrier);
            Check(vkEndCommandBuffer(slot.command));
            Check(vkResetFences(m_context.GetDevice(), 1, &slot.fence));
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &slot.command;
            Check(vkQueueSubmit(m_context.GetGraphicsQueue(), 1, &submit, slot.fence));
            slot.reading = reading;
            slot.serial = ++m_serial;
            slot.pending = true;
            return true;
        }
        return false;
    }

    Optional<PixelReading> Poll(u64 generation, u64 acquisition) {
        Optional<PixelReading> result;
        u64 newest = 0;
        for (auto& slot : m_slots) {
            if (!slot.pending) continue;
            const auto status = vkGetFenceStatus(m_context.GetDevice(), slot.fence);
            if (status == VK_NOT_READY) continue;
            Check(status);
            slot.pending = false;
            if (slot.reading.imageGeneration != generation ||
                slot.reading.acquisitionIndex != acquisition || slot.serial <= newest) continue;
            const void* mapped = slot.buffer->MapRead();
            if (!mapped) throw std::runtime_error("cannot map asynchronous pixel readback");
            std::memcpy(&slot.reading.value, mapped, sizeof(glm::vec4));
            slot.buffer->Unmap();
            newest = slot.serial;
            result = slot.reading;
        }
        return result;
    }

private:
    static void Check(VkResult result) {
        if (result != VK_SUCCESS)
            throw std::runtime_error("asynchronous pixel readback Vulkan error " + std::to_string(result));
    }
    void Cleanup() noexcept {
        for (auto& slot : m_slots) {
            if (slot.pending) vkWaitForFences(m_context.GetDevice(), 1, &slot.fence, VK_TRUE, UINT64_MAX);
            if (slot.fence) vkDestroyFence(m_context.GetDevice(), slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
            slot.pending = false;
            slot.buffer.reset();
        }
        if (m_pool) vkDestroyCommandPool(m_context.GetDevice(), m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    struct Slot {
        std::unique_ptr<GpuBuffer> buffer;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool pending = false;
        u64 serial = 0;
        PixelReading reading;
    };
    const VulkanContext& m_context;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    std::array<Slot, 3> m_slots;
    u64 m_serial = 0;
};

} // namespace quantiloom::rendercore
