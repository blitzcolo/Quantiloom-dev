/**
 * @file test_vulkan_device.cpp
 * @brief Cover for the shared test device itself
 *
 * The fixture is infrastructure the render-path tests are about to depend on, so it
 * gets the same treatment as the code it will cover. These cases assert the two
 * properties everything downstream assumes -- the device is ray-tracing capable, and
 * there is exactly one of it -- plus a VMA round trip, which is the smallest thing
 * that proves the allocator handed out by Device() is actually usable.
 */

#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/VulkanContext.hpp"

#include <cstring>
#include <numeric>
#include <vector>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

TEST_F(VulkanDeviceTest, ProvidesARayTracingCapableDevice) {
    const VulkanContext& ctx = Device();

    EXPECT_NE(ctx.GetInstance(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.GetPhysicalDevice(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.GetDevice(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.GetGraphicsQueue(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.GetAllocator(), nullptr);

    ASSERT_TRUE(ctx.IsRayTracingSupported());
    EXPECT_GE(ctx.GetRayTracingProperties().maxRayRecursionDepth, 1u);
}

// Creating a device per case would dominate a suite that runs in two seconds. This
// is the property that keeps it cheap, so it is worth stating rather than assuming.
TEST_F(VulkanDeviceTest, SharesOneDeviceAcrossCases) {
    EXPECT_EQ(::quantiloom::testing::SharedVulkanDevice(),
              ::quantiloom::testing::SharedVulkanDevice());
    EXPECT_EQ(&Device(), ::quantiloom::testing::SharedVulkanDevice());
}

// Every render stage still to be covered allocates through this allocator, so a
// round trip through it is the first thing worth knowing works.
TEST_F(VulkanDeviceTest, RoundTripsHostVisibleMemory) {
    std::vector<u32> written(256);
    std::iota(written.begin(), written.end(), 1u);
    const VkDeviceSize bytes = written.size() * sizeof(u32);

    GpuBuffer buffer(Device().GetAllocator(), bytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    ASSERT_TRUE(buffer.IsValid());
    EXPECT_EQ(buffer.GetSize(), bytes);

    buffer.Upload(written.data(), bytes);

    const void* mapped = buffer.Map();
    ASSERT_NE(mapped, nullptr);

    std::vector<u32> readBack(written.size());
    std::memcpy(readBack.data(), mapped, bytes);
    buffer.Unmap();

    EXPECT_EQ(readBack, written);
}

// A readback may stay mapped across GPU writes. Every read still needs cache
// invalidation after GPU completion, including when Map() returns an old pointer.
TEST_F(VulkanDeviceTest, ReadsRepeatedGpuWritesThroughAnExistingMapping) {
    constexpr VkDeviceSize bytes = 256 * sizeof(u32);
    GpuBuffer readback(Device().GetAllocator(), bytes,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    for (u32 pattern : {0x12345678u, 0x89abcdefu, 0u}) {
        CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, readback.GetHandle(), 0, bytes, pattern);
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = readback.GetHandle();
            barrier.size = bytes;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 0, nullptr, 1, &barrier, 0, nullptr);
        });
        const auto* data = static_cast<const u32*>(readback.MapRead());
        ASSERT_NE(data, nullptr);
        for (usize i = 0; i < bytes / sizeof(u32); ++i) EXPECT_EQ(data[i], pattern);
    }
    readback.Unmap();
}

TEST_F(VulkanDeviceTest, AlignsSbtDeviceAddressesAtTheAllocationBase) {
    const auto alignment = Device().GetRayTracingProperties().shaderGroupBaseAlignment;
    ASSERT_GT(alignment, 0u);
    // Keep several small allocations alive so this exercises suballocation
    // offsets, rather than only the naturally aligned start of a VMA block.
    std::vector<GpuBuffer> buffers;
    for (u32 i = 0; i < 9; ++i) {
        buffers.emplace_back(Device().GetAllocator(), (i + 1) * alignment,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, alignment);
        EXPECT_EQ(buffers.back().GetDeviceAddress(Device().GetDevice()) % alignment, 0u);
    }
}
