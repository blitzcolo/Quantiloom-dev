/**
 * @file AsyncPixelReadback.hpp
 * @brief Queue-ordered pixel copies with reusable resources
 *
 * Poll never waits; destruction is the only path that waits for outstanding
 * copies. The host must submit its recorded render before requesting a copy
 * on the same queue.
 */

#pragma once

#include "renderer/GpuBuffer.hpp"
#include "renderer/PixelReading.hpp"
#include "renderer/VulkanContext.hpp"
#include <array>
#include <memory>

namespace quantiloom::rendercore {

class AsyncPixelReadback {
public:
    explicit AsyncPixelReadback(const VulkanContext& context);
    ~AsyncPixelReadback();
    AsyncPixelReadback(const AsyncPixelReadback&) = delete;
    AsyncPixelReadback& operator=(const AsyncPixelReadback&) = delete;

    /// Record and submit a one-pixel copy of `image` into a free slot. Returns
    /// false only when every slot has an outstanding copy.
    bool Submit(VkImage image, u32 renderX, u32 renderY, const PixelReading& reading);

    /// Return the newest completed copy that belongs to `generation` /
    /// `acquisition`, without waiting for any fence.
    Optional<PixelReading> Poll(u64 generation, u64 acquisition);

private:
    static void Check(VkResult result);
    void Cleanup() noexcept;
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
