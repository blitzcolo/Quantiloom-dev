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

#include "support/GpuTestDevice.hpp"

#include "gpu/Device.hpp"
#include "renderer/DeferredBufferUploads.hpp"

#include <cstring>
#include <numeric>
#include <vector>

using namespace quantiloom;
using quantiloom::testing::GpuDeviceTest;

namespace {

// The queue writes through vkCmdUpdateBuffer, so a destination has to carry
// TransferDst; Readback keeps the buffer host-readable for verification.
std::unique_ptr<gpu::Buffer> MakeTarget(gpu::Device& device, u64 bytes) {
    auto created = device.CreateBuffer(gpu::BufferDesc{
        bytes, gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst, gpu::MemoryClass::Readback, 0});
    EXPECT_TRUE(created.has_value()) << (created.has_value() ? "" : created.error());
    return created.has_value() ? std::move(created.value()) : nullptr;
}

std::vector<u8> ReadBytes(gpu::Buffer& buffer, u64 offset,
                          u64 bytes) {
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

TEST_P(GpuDeviceTest, DeferredUploadWritesRecordedBytes) {
    std::unique_ptr<gpu::Buffer> target = MakeTarget(Gpu(), 64);
    ASSERT_NE(target, nullptr);

    std::vector<u8> payload(64);
    std::iota(payload.begin(), payload.end(), u8{7});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target->Ref(), 0, payload.data(), payload.size());
    EXPECT_FALSE(uploads.Empty());
    Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
        uploads.Record(list);
    });
    EXPECT_TRUE(uploads.Empty());
    EXPECT_EQ(ReadBytes(*target, 0, 64), payload);
}

TEST_P(GpuDeviceTest, DeferredUploadCoalescesSameRange) {
    std::unique_ptr<gpu::Buffer> target = MakeTarget(Gpu(), 16);
    ASSERT_NE(target, nullptr);

    const std::vector<u8> first(16, u8{0xAA});
    const std::vector<u8> second(16, u8{0x5C});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target->Ref(), 0, first.data(), first.size());
    uploads.Enqueue(target->Ref(), 0, second.data(), second.size());
    Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
        uploads.Record(list);
    });
    EXPECT_EQ(ReadBytes(*target, 0, 16), second);
}

TEST_P(GpuDeviceTest, DeferredUploadHonoursOffset) {
    std::unique_ptr<gpu::Buffer> target = MakeTarget(Gpu(), 32);
    ASSERT_NE(target, nullptr);
    const std::vector<u8> zeros(32, u8{0});
    target->Upload(zeros.data(), zeros.size());

    const std::vector<u8> payload{1, 2, 3, 4};

    DeferredBufferUploads uploads;
    uploads.Enqueue(target->Ref(), 8, payload.data(), payload.size());
    Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
        uploads.Record(list);
    });

    const std::vector<u8> contents = ReadBytes(*target, 0, 32);
    std::vector<u8> expected(32, u8{0});
    std::copy(payload.begin(), payload.end(), expected.begin() + 8);
    EXPECT_EQ(contents, expected);
}

TEST_P(GpuDeviceTest, DeferredUploadChunksPast64KiB) {
    // vkCmdUpdateBuffer carries 65536 bytes per call, so this payload forces
    // the chunked loop; a dropped chunk shows up as a zero run in the tail.
    const size_t bytes = 3 * 65536 + 64;
    std::unique_ptr<gpu::Buffer> target = MakeTarget(Gpu(), bytes);
    ASSERT_NE(target, nullptr);

    std::vector<u8> payload(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        payload[i] = static_cast<u8>((i * 131u + 7u) & 0xFF);
    }

    DeferredBufferUploads uploads;
    uploads.Enqueue(target->Ref(), 0, payload.data(), payload.size());
    Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
        uploads.Record(list);
    });
    EXPECT_EQ(ReadBytes(*target, 0, bytes), payload);
}

TEST_P(GpuDeviceTest, DeferredUploadDistinctRangesAllLand) {
    std::unique_ptr<gpu::Buffer> target = MakeTarget(Gpu(), 64);
    ASSERT_NE(target, nullptr);

    // Same offset but different sizes are two writes, not one coalesced
    // write: the larger one is recorded in full.
    const std::vector<u8> narrow(16, u8{0x11});
    const std::vector<u8> wide(64, u8{0x22});

    DeferredBufferUploads uploads;
    uploads.Enqueue(target->Ref(), 0, narrow.data(), narrow.size());
    uploads.Enqueue(target->Ref(), 0, wide.data(), wide.size());
    Gpu().ExecuteImmediate([&](gpu::CommandList& list) {
        uploads.Record(list);
    });
    EXPECT_EQ(ReadBytes(*target, 0, 64), wide);
}
