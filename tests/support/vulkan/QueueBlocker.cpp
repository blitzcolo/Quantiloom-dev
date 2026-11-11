// The Vulkan implementation of QueueBlocker: the event gate test_pick.cpp used to build
// by hand, call for call.

#include "support/vulkan/QueueBlocker.hpp"

#include "core/ResultFail.hpp"
#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanContext.hpp"
#include "gpu/vulkan/VulkanDevice.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "renderer/interop/VulkanInterop.hpp"

namespace quantiloom::testing {

namespace {

class VulkanQueueBlocker final : public QueueBlocker {
public:
    VulkanQueueBlocker(VkDevice device, VkQueue queue) : device_(device), queue_(queue) {}

    ~VulkanQueueBlocker() override {
        if (submitted_) vkQueueWaitIdle(queue_);
        if (event_) vkDestroyEvent(device_, event_, nullptr);
        if (fence_) vkDestroyFence(device_, fence_, nullptr);
        if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
    }

    VulkanQueueBlocker(const VulkanQueueBlocker&) = delete;
    VulkanQueueBlocker& operator=(const VulkanQueueBlocker&) = delete;

    Result<void, String> CreateObjects(u32 queueFamily) {
        VkEventCreateInfo eventInfo{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        VkResult result = vkCreateEvent(device_, &eventInfo, nullptr, &event_);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        result = vkCreateFence(device_, &fenceInfo, nullptr, &fence_);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = queueFamily;
        result = vkCreateCommandPool(device_, &poolInfo, nullptr, &pool_);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        return Result<void, String>::Ok();
    }

    Result<GpuCommandRecording, String> Block() override {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkResult result = vkAllocateCommandBuffers(device_, &allocation, &cmd_);
        if (result != VK_SUCCESS) return Fail<GpuCommandRecording>(gpu::vulkan::ResultCode(result));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        result = vkBeginCommandBuffer(cmd_, &begin);
        if (result != VK_SUCCESS) return Fail<GpuCommandRecording>(gpu::vulkan::ResultCode(result));
        vkCmdWaitEvents(cmd_, 1, &event_, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        nullptr, 0, nullptr, 0, nullptr);
        return vk::Commands(cmd_);
    }

    Result<void, String> Submit() override {
        VkResult result = vkEndCommandBuffer(cmd_);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd_;
        result = vkQueueSubmit(queue_, 1, &submit, fence_);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        submitted_ = true;
        return Result<void, String>::Ok();
    }

    void Release() override { vkSetEvent(device_, event_); }

    Result<gpu::FenceStatus, String> Status() override {
        const VkResult result = vkGetFenceStatus(device_, fence_);
        if (result == VK_SUCCESS) return gpu::FenceStatus::Signaled;
        if (result == VK_NOT_READY) return gpu::FenceStatus::NotReady;
        return Fail<gpu::FenceStatus>(gpu::vulkan::ResultCode(result));
    }

    Result<void, String> Wait(u64 timeoutNs) override {
        const VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeoutNs);
        if (result != VK_SUCCESS) return Fail<void>(gpu::vulkan::ResultCode(result));
        return Result<void, String>::Ok();
    }

private:
    VkDevice device_;
    VkQueue queue_;
    VkEvent event_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool submitted_ = false;
};

}  // namespace

bool QueueBlocker::IsSupported(gpu::Backend backend) {
    return backend == gpu::Backend::Vulkan;
}

Result<std::unique_ptr<QueueBlocker>, String> QueueBlocker::Create(gpu::Device& device) {
    if (!IsSupported(device.Info().backend)) {
        return Fail<std::unique_ptr<QueueBlocker>>("no queue blocker for this backend");
    }
    const VulkanContext& context = static_cast<gpu::vulkan::VulkanDevice&>(device).Context();
    auto blocker = std::make_unique<VulkanQueueBlocker>(context.GetDevice(), context.GetGraphicsQueue());
    auto created = blocker->CreateObjects(context.GetGraphicsQueueFamily());
    if (!created.has_value()) return Fail<std::unique_ptr<QueueBlocker>>(created.error());
    return std::unique_ptr<QueueBlocker>(std::move(blocker));
}

}  // namespace quantiloom::testing
