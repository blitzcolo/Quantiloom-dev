/**
 * @file Sync.hpp
 * @brief Command pools, command buffers, fences, events and timestamp pools of the RHI (internal)
 *
 * The objects a private submit loop owns: Device::CreateCommandPool -> CommandPool::Allocate
 * -> CommandBuffer::Begin/End -> Device::Submit, with a Fence to wait on. Each call maps onto
 * one Vulkan call. On Vulkan, the Err of a Vulkan call that failed is its VkResult as a
 * decimal ("-4"), so a caller can keep a message that prints the code.
 */

#pragma once

#include "gpu/CommandList.hpp"

#include <memory>
#include <span>

namespace quantiloom::gpu {

/// The flag bits of the pools the renderer creates.
struct CommandPoolDesc {
    bool transient = false;   // short-lived buffers (VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)
    bool resettable = false;  // buffers reset one at a time (..._RESET_COMMAND_BUFFER_BIT)
};

/// One primary command buffer. Destroy it before the pool it came from.
class CommandBuffer {
public:
    virtual ~CommandBuffer() = default;  // vkFreeCommandBuffers
    /// vkResetCommandBuffer with no flags. Needs a resettable pool.
    [[nodiscard]] virtual Result<void, String> Reset() = 0;
    /// vkBeginCommandBuffer with ONE_TIME_SUBMIT. The list records into this buffer and lives
    /// as long as the buffer does.
    [[nodiscard]] virtual Result<CommandList*, String> Begin() = 0;
    [[nodiscard]] virtual Result<void, String> End() = 0;  // vkEndCommandBuffer
};

class CommandPool {
public:
    virtual ~CommandPool() = default;  // vkDestroyCommandPool
    /// vkAllocateCommandBuffers of one primary buffer.
    [[nodiscard]] virtual Result<std::unique_ptr<CommandBuffer>, String> Allocate() = 0;
};

enum class FenceStatus : u8 { Signaled, NotReady };

/// Created unsignaled.
class Fence {
public:
    virtual ~Fence() = default;                                       // vkDestroyFence
    [[nodiscard]] virtual Result<FenceStatus, String> Status() = 0;   // vkGetFenceStatus
    [[nodiscard]] virtual Result<void, String> Wait(u64 timeoutNs = ~u64{0}) = 0;  // vkWaitForFences
    [[nodiscard]] virtual Result<void, String> Reset() = 0;           // vkResetFences
};

/// A GPU-to-host signal set and reset from a command list (CommandList::SetEvent/ResetEvent).
class Event {
public:
    virtual ~Event() = default;  // vkDestroyEvent
    [[nodiscard]] virtual EventRef Ref() const = 0;
    [[nodiscard]] virtual bool IsSet() const = 0;  // vkGetEventStatus == VK_EVENT_SET
};

/// Wait: block until every value is written, one u64 per query.
/// WithAvailability: never block, [value, availability] per query.
enum class TimestampResolve : u8 { Wait, WithAvailability };
enum class ResolveStatus : u8 { Ready, NotReady };

class TimestampPool {
public:
    virtual ~TimestampPool() = default;  // vkDestroyQueryPool
    [[nodiscard]] virtual TimestampPoolRef Ref() const = 0;
    /// vkGetQueryPoolResults of queries [first, first + count) as 64-bit values. `out` must
    /// hold count values (Wait) or 2 * count (WithAvailability); exactly that many are
    /// written. NotReady when the driver reports that not every query was available.
    [[nodiscard]] virtual Result<ResolveStatus, String> Resolve(u32 first, u32 count, TimestampResolve mode,
                                                              std::span<u64> out) = 0;
};

}  // namespace quantiloom::gpu
