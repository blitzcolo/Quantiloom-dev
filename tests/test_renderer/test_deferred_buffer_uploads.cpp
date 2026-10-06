/**
 * @file test_deferred_buffer_uploads.cpp
 * @brief Cover for renderer/DeferredBufferUploads
 *
 * The class exists so host writers stop draining the queue before touching a
 * mapped buffer a submitted frame may still read: pending writes are replayed
 * into the next command buffer, bracketed by barriers. These cases cover the
 * mechanics -- coalescing by (buffer, offset, size), the 64 KiB
 * vkCmdUpdateBuffer chunk limit, and non-zero offsets -- not the barriers,
 * which nothing short of a race can observe.
 */

#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "renderer/CommandHelper.hpp"
#include "renderer/DeferredBufferUploads.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/VulkanContext.hpp"

#include <cstring>
#include <numeric>
#include <vector>

using namespace quantiloom;
using quantiloom::testing::VulkanDeviceTest;

namespace {

// The queue writes through vkCmdUpdateBuffer, so a destination has to carry
// TRANSFER_DST; GPU_TO_CPU keeps the buffer host-readable for verification.
GpuBuffer MakeTarget(VulkanContext& device, VkDeviceSize bytes) {
    GpuBuffer buffer(device.GetAllocator(), bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VMA_MEMORY_USAGE_GPU_TO_CPU);
    return buffer;
}

std::vector<u8> ReadBytes(GpuBuffer& buffer, VkDeviceSize offset,
                          VkDeviceSize bytes) {
    std::vector<u8> out(static_cast<size_t>(bytes));
    const void* mapped = buffer.MapRead();
    EXPECT_NE(mapped, nullptr);
    if (mapped) {
        std::memcpy(out.data(),
                    static_cast<const u8*>(mapped) + offset,
                    static_cast<size_t>(bytes));
        buffer.Unmap();
    }
    return out;
}

} // namespace

TEST_F(VulkanDeviceTest, DeferredUploadWritesRecordedBytes) {
    GpuBuffer target = MakeTarget(Device(), 64);
    ASSERT_TRUE(target.IsValid());

    std::vector<u8> payload(64);
    std::iota(payload.begin(), payload.end(), u8{7});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target, 0, payload.data(), payload.size());
    EXPECT_FALSE(uploads.Empty());
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        uploads.Record(cmd);
    });
    EXPECT_TRUE(uploads.Empty());
    EXPECT_EQ(ReadBytes(target, 0, 64), payload);
}

TEST_F(VulkanDeviceTest, DeferredUploadCoalescesSameRange) {
    GpuBuffer target = MakeTarget(Device(), 16);
    ASSERT_TRUE(target.IsValid());

    const std::vector<u8> first(16, u8{0xAA});
    const std::vector<u8> second(16, u8{0x5C});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target, 0, first.data(), first.size());
    uploads.Enqueue(target, 0, second.data(), second.size());
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        uploads.Record(cmd);
    });
    EXPECT_EQ(ReadBytes(target, 0, 16), second);
}

TEST_F(VulkanDeviceTest, DeferredUploadHonoursOffset) {
    GpuBuffer target = MakeTarget(Device(), 32);
    ASSERT_TRUE(target.IsValid());
    const std::vector<u8> zeros(32, u8{0});
    target.Upload(zeros.data(), zeros.size());

    const std::vector<u8> payload{1, 2, 3, 4};

    DeferredBufferUploads uploads;
    uploads.Enqueue(target, 8, payload.data(), payload.size());
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        uploads.Record(cmd);
    });

    const std::vector<u8> contents = ReadBytes(target, 0, 32);
    std::vector<u8> expected(32, u8{0});
    std::copy(payload.begin(), payload.end(), expected.begin() + 8);
    EXPECT_EQ(contents, expected);
}

TEST_F(VulkanDeviceTest, DeferredUploadChunksPast64KiB) {
    // vkCmdUpdateBuffer carries 65536 bytes per call, so this payload forces
    // the chunked loop; a dropped chunk shows up as a zero run in the tail.
    const size_t bytes = 3 * 65536 + 64;
    GpuBuffer target = MakeTarget(Device(), bytes);
    ASSERT_TRUE(target.IsValid());

    std::vector<u8> payload(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        payload[i] = static_cast<u8>((i * 131u + 7u) & 0xFF);
    }

    DeferredBufferUploads uploads;
    uploads.Enqueue(target, 0, payload.data(), payload.size());
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        uploads.Record(cmd);
    });
    EXPECT_EQ(ReadBytes(target, 0, bytes), payload);
}

TEST_F(VulkanDeviceTest, DeferredUploadDistinctRangesAllLand) {
    GpuBuffer target = MakeTarget(Device(), 64);
    ASSERT_TRUE(target.IsValid());

    // Same offset but different sizes are two writes, not one coalesced
    // write: the larger one is recorded in full.
    const std::vector<u8> narrow(16, u8{0x11});
    const std::vector<u8> wide(64, u8{0x22});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target, 0, narrow.data(), narrow.size());
    uploads.Enqueue(target, 0, wide.data(), wide.size());
    CommandHelper::ExecuteImmediate(Device(), [&](VkCommandBuffer cmd) {
        uploads.Record(cmd);
    });
    EXPECT_EQ(ReadBytes(target, 0, 64), wide);
}
