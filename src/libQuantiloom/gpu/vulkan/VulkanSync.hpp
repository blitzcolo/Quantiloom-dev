#pragma once

#include "gpu/Sync.hpp"
#include "gpu/vulkan/VulkanCommandList.hpp"

#include <vulkan/vulkan.h>

#include <memory>

namespace quantiloom::gpu::vulkan {

class VulkanCommandPool final : public CommandPool {
public:
    [[nodiscard]] static Result<std::unique_ptr<CommandPool>, String> Create(VkDevice device, u32 queueFamily,
                                                                             const CommandPoolDesc& desc,
                                                                             const RtFunctions& rt);
    ~VulkanCommandPool() override;
    VulkanCommandPool(const VulkanCommandPool&) = delete;
    VulkanCommandPool& operator=(const VulkanCommandPool&) = delete;

    [[nodiscard]] Result<std::unique_ptr<CommandBuffer>, String> Allocate() override;
    [[nodiscard]] VkCommandPool Native() const { return pool_; }

private:
    VulkanCommandPool() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    RtFunctions rt_{};  // copied into every buffer's list
};

class VulkanCommandBuffer final : public CommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBuffer cmd, const RtFunctions& rt)
        : device_(device), pool_(pool), list_(cmd, rt) {}
    ~VulkanCommandBuffer() override;
    VulkanCommandBuffer(const VulkanCommandBuffer&) = delete;
    VulkanCommandBuffer& operator=(const VulkanCommandBuffer&) = delete;

    [[nodiscard]] Result<void, String> Reset() override;
    [[nodiscard]] Result<CommandList*, String> Begin() override;
    [[nodiscard]] Result<void, String> End() override;
    [[nodiscard]] VkCommandBuffer Native() const { return list_.Native(); }

private:
    VkDevice device_;
    VkCommandPool pool_;
    VulkanCommandList list_;
};

class VulkanFence final : public Fence {
public:
    [[nodiscard]] static Result<std::unique_ptr<Fence>, String> Create(VkDevice device);
    ~VulkanFence() override;
    VulkanFence(const VulkanFence&) = delete;
    VulkanFence& operator=(const VulkanFence&) = delete;

    [[nodiscard]] Result<FenceStatus, String> Status() override;
    [[nodiscard]] Result<void, String> Wait(u64 timeoutNs = ~u64{0}) override;
    [[nodiscard]] Result<void, String> Reset() override;
    [[nodiscard]] VkFence Native() const { return fence_; }

private:
    VulkanFence() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

class VulkanEvent final : public Event {
public:
    [[nodiscard]] static Result<std::unique_ptr<Event>, String> Create(VkDevice device);
    ~VulkanEvent() override;
    VulkanEvent(const VulkanEvent&) = delete;
    VulkanEvent& operator=(const VulkanEvent&) = delete;

    [[nodiscard]] EventRef Ref() const override { return EventRef{ToHandle(event_)}; }
    [[nodiscard]] bool IsSet() const override;

private:
    VulkanEvent() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkEvent event_ = VK_NULL_HANDLE;
};

class VulkanTimestampPool final : public TimestampPool {
public:
    [[nodiscard]] static Result<std::unique_ptr<TimestampPool>, String> Create(VkDevice device, u32 count);
    ~VulkanTimestampPool() override;
    VulkanTimestampPool(const VulkanTimestampPool&) = delete;
    VulkanTimestampPool& operator=(const VulkanTimestampPool&) = delete;

    [[nodiscard]] TimestampPoolRef Ref() const override { return TimestampPoolRef{ToHandle(pool_), count_}; }
    [[nodiscard]] Result<ResolveStatus, String> Resolve(u32 first, u32 count, TimestampResolve mode,
                                                      std::span<u64> out) override;

private:
    VulkanTimestampPool() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    u32 count_ = 0;
};

}  // namespace quantiloom::gpu::vulkan
