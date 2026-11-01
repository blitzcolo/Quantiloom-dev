#include "gpu/vulkan/VulkanSync.hpp"

#include "core/ResultFail.hpp"

namespace quantiloom::gpu::vulkan {

// ----------------------------------------------------------------------------
// VulkanCommandPool
// ----------------------------------------------------------------------------

Result<std::unique_ptr<CommandPool>, String> VulkanCommandPool::Create(VkDevice device, u32 queueFamily,
                                                                       const CommandPoolDesc& desc) {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    if (desc.transient) info.flags |= VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (desc.resettable) info.flags |= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = queueFamily;

    std::unique_ptr<VulkanCommandPool> pool(new VulkanCommandPool());
    const VkResult result = vkCreateCommandPool(device, &info, nullptr, &pool->pool_);
    if (result != VK_SUCCESS) {
        pool->pool_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<CommandPool>>(ResultCode(result));
    }
    pool->device_ = device;
    return std::unique_ptr<CommandPool>(std::move(pool));
}

VulkanCommandPool::~VulkanCommandPool() {
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool_, nullptr);
}

Result<std::unique_ptr<CommandBuffer>, String> VulkanCommandPool::Allocate() {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const VkResult result = vkAllocateCommandBuffers(device_, &info, &cmd);
    if (result != VK_SUCCESS) {
        return Fail<std::unique_ptr<CommandBuffer>>(ResultCode(result));
    }
    return std::unique_ptr<CommandBuffer>(std::make_unique<VulkanCommandBuffer>(device_, pool_, cmd));
}

// ----------------------------------------------------------------------------
// VulkanCommandBuffer
// ----------------------------------------------------------------------------

VulkanCommandBuffer::~VulkanCommandBuffer() {
    const VkCommandBuffer cmd = list_.Native();
    if (cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(device_, pool_, 1, &cmd);
}

Result<void, String> VulkanCommandBuffer::Reset() {
    const VkResult result = vkResetCommandBuffer(list_.Native(), 0);
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

Result<CommandList*, String> VulkanCommandBuffer::Begin() {
    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult result = vkBeginCommandBuffer(list_.Native(), &info);
    if (result != VK_SUCCESS) return Fail<CommandList*>(ResultCode(result));
    CommandList* list = &list_;
    return list;
}

Result<void, String> VulkanCommandBuffer::End() {
    const VkResult result = vkEndCommandBuffer(list_.Native());
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

// ----------------------------------------------------------------------------
// VulkanFence
// ----------------------------------------------------------------------------

Result<std::unique_ptr<Fence>, String> VulkanFence::Create(VkDevice device) {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    std::unique_ptr<VulkanFence> fence(new VulkanFence());
    const VkResult result = vkCreateFence(device, &info, nullptr, &fence->fence_);
    if (result != VK_SUCCESS) {
        fence->fence_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<Fence>>(ResultCode(result));
    }
    fence->device_ = device;
    return std::unique_ptr<Fence>(std::move(fence));
}

VulkanFence::~VulkanFence() {
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
}

Result<FenceStatus, String> VulkanFence::Status() {
    const VkResult result = vkGetFenceStatus(device_, fence_);
    if (result == VK_SUCCESS) return FenceStatus::Signaled;
    if (result == VK_NOT_READY) return FenceStatus::NotReady;
    return Fail<FenceStatus>(ResultCode(result));
}

Result<void, String> VulkanFence::Wait(u64 timeoutNs) {
    const VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeoutNs);
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

Result<void, String> VulkanFence::Reset() {
    const VkResult result = vkResetFences(device_, 1, &fence_);
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

// ----------------------------------------------------------------------------
// VulkanEvent
// ----------------------------------------------------------------------------

Result<std::unique_ptr<Event>, String> VulkanEvent::Create(VkDevice device) {
    VkEventCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;

    std::unique_ptr<VulkanEvent> event(new VulkanEvent());
    const VkResult result = vkCreateEvent(device, &info, nullptr, &event->event_);
    if (result != VK_SUCCESS) {
        event->event_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<Event>>(ResultCode(result));
    }
    event->device_ = device;
    return std::unique_ptr<Event>(std::move(event));
}

VulkanEvent::~VulkanEvent() {
    if (event_ != VK_NULL_HANDLE) vkDestroyEvent(device_, event_, nullptr);
}

bool VulkanEvent::IsSet() const {
    return vkGetEventStatus(device_, event_) == VK_EVENT_SET;
}

// ----------------------------------------------------------------------------
// VulkanTimestampPool
// ----------------------------------------------------------------------------

Result<std::unique_ptr<TimestampPool>, String> VulkanTimestampPool::Create(VkDevice device, u32 count) {
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = count;

    std::unique_ptr<VulkanTimestampPool> pool(new VulkanTimestampPool());
    const VkResult result = vkCreateQueryPool(device, &info, nullptr, &pool->pool_);
    if (result != VK_SUCCESS) {
        pool->pool_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<TimestampPool>>(ResultCode(result));
    }
    pool->device_ = device;
    pool->count_ = count;
    return std::unique_ptr<TimestampPool>(std::move(pool));
}

VulkanTimestampPool::~VulkanTimestampPool() {
    if (pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, pool_, nullptr);
}

Result<ResolveStatus, String> VulkanTimestampPool::Resolve(u32 first, u32 count, TimestampResolve mode,
                                                           std::span<u64> out) {
    const bool availability = mode == TimestampResolve::WithAvailability;
    const size_t values = availability ? 2 * static_cast<size_t>(count) : static_cast<size_t>(count);
    if (out.size() < values) {
        return Fail<ResolveStatus>("timestamp resolve needs " + std::to_string(values) + " values, got " +
                                   std::to_string(out.size()));
    }
    const VkDeviceSize stride = availability ? 2 * sizeof(u64) : sizeof(u64);
    VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;
    flags |= availability ? VK_QUERY_RESULT_WITH_AVAILABILITY_BIT : VK_QUERY_RESULT_WAIT_BIT;
    const VkResult result =
        vkGetQueryPoolResults(device_, pool_, first, count, values * sizeof(u64), out.data(), stride, flags);
    if (result == VK_SUCCESS) return ResolveStatus::Ready;
    if (result == VK_NOT_READY) return ResolveStatus::NotReady;
    return Fail<ResolveStatus>(ResultCode(result));
}

}  // namespace quantiloom::gpu::vulkan
