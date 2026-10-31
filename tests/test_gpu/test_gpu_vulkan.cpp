#include <gtest/gtest.h>

#include "support/VulkanTestDevice.hpp"

#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanBridge.hpp"

#include <array>
#include <cstring>
#include <vector>

using namespace quantiloom;
using namespace quantiloom::gpu;
using quantiloom::testing::VulkanDeviceTest;

namespace {

class GpuVulkanTest : public VulkanDeviceTest {};

std::unique_ptr<Buffer> MakeBuffer(gpu::Device& device, u64 size, BufferUsage usage, MemoryClass memory) {
    auto created = device.CreateBuffer(BufferDesc{size, usage, memory, 0});
    EXPECT_TRUE(created.has_value()) << (created.has_value() ? "" : created.error());
    return created.has_value() ? std::move(created.value()) : nullptr;
}

}  // namespace

TEST_F(GpuVulkanTest, InfoReportsTheVulkanDevice) {
    const DeviceInfo& info = GpuDevice().Info();
    EXPECT_EQ(info.backend, Backend::Vulkan);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.maxStorageBufferRange, 0u);
}

TEST_F(GpuVulkanTest, CopyBufferRoundTripsThroughReadback) {
    auto& device = GpuDevice();
    std::array<u32, 64> data{};
    for (u32 i = 0; i < data.size(); ++i) data[i] = i * 2654435761u;

    auto src = MakeBuffer(device, sizeof(data), BufferUsage::TransferSrc, MemoryClass::Upload);
    auto dst = MakeBuffer(device, sizeof(data), BufferUsage::TransferDst, MemoryClass::Readback);
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    src->Upload(data.data(), sizeof(data));

    const BufferCopyRegion region{0, 0, sizeof(data)};
    device.ExecuteImmediate([&](CommandList& cmd) {
        cmd.CopyBuffer(src->Ref(), dst->Ref(), std::span<const BufferCopyRegion>(&region, 1));
        cmd.Barrier(MemoryBarrier{Access::TransferWrite, Access::HostRead});
    });

    const auto* mapped = static_cast<const u32*>(dst->MapRead());
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(std::memcmp(mapped, data.data(), sizeof(data)), 0);
    dst->Unmap();
}

TEST_F(GpuVulkanTest, FillAndUpdateWriteTheBuffer) {
    auto& device = GpuDevice();
    auto dst = MakeBuffer(device, 64, BufferUsage::TransferDst, MemoryClass::Readback);
    ASSERT_NE(dst, nullptr);
    const std::array<u32, 4> patch{1u, 2u, 3u, 4u};

    device.ExecuteImmediate([&](CommandList& cmd) {
        cmd.FillBuffer(dst->Ref(), 0, kWholeSize, 0xAAu);
        cmd.Barrier(BufferBarrier{dst->Ref(), Access::TransferWrite, Access::TransferWrite});
        cmd.UpdateBuffer(dst->Ref(), 16, std::as_bytes(std::span<const u32>(patch)));
        cmd.Barrier(MemoryBarrier{Access::TransferWrite, Access::HostRead});
    });

    const auto* words = static_cast<const u32*>(dst->MapRead());
    ASSERT_NE(words, nullptr);
    EXPECT_EQ(words[0], 0xAAu);
    EXPECT_EQ(words[4], 1u);
    EXPECT_EQ(words[7], 4u);
    EXPECT_EQ(words[8], 0xAAu);
    dst->Unmap();
}

TEST_F(GpuVulkanTest, NestedExecuteImmediateDoesNotDeadlock) {
    auto& device = GpuDevice();
    bool inner = false;
    device.ExecuteImmediate([&](CommandList&) { device.ExecuteImmediate([&](CommandList&) { inner = true; }); });
    EXPECT_TRUE(inner);
}

TEST_F(GpuVulkanTest, TextureRefCarriesTheDescriptorIdentity) {
    auto& device = GpuDevice();
    TextureDesc desc;
    desc.width = 16;
    desc.height = 8;
    desc.format = Format::RGBA32Float;
    desc.usage = TextureUsage::Storage | TextureUsage::TransferSrc | TextureUsage::TransferDst;
    auto created = device.CreateTexture(desc);
    ASSERT_TRUE(created.has_value()) << created.error();
    const TextureRef ref = created.value()->Ref();
    EXPECT_TRUE(ref.IsValid());
    EXPECT_EQ(ref.width, 16u);
    EXPECT_EQ(ref.layout, TextureLayoutPolicy::General);
    EXPECT_EQ(ref.Identity(), ref.view);
    EXPECT_EQ(vulkan::Ref(vulkan::AsGpuImage(*created.value())), ref);
}

TEST_F(GpuVulkanTest, RepeatedBindingSlotsAreRejected) {
    const std::array<BindingSlot, 2> slots{BindingSlot{3, BindingKind::ReadOnlyBuffer, 1, false},
                                           BindingSlot{3, BindingKind::ReadWriteBuffer, 1, false}};
    const auto layout = GpuDevice().CreateBindingLayout(BindingLayoutDesc{0, slots});
    EXPECT_FALSE(layout.has_value());
}

TEST_F(GpuVulkanTest, AKernelBindsAndWritesItsGroup) {
    auto& device = GpuDevice();
    const StringView names[] = {"spectral_reconstruct"};
    const auto code = device.Shaders().Load(names);
    if (!code.has_value()) GTEST_SKIP() << code.error();

    const std::array<BindingSlot, 5> slots{BindingSlot{0, BindingKind::ReadOnlyBuffer, 1, false},
                                           BindingSlot{1, BindingKind::ReadWriteBuffer, 1, false},
                                           BindingSlot{2, BindingKind::ReadOnlyBuffer, 1, false},
                                           BindingSlot{3, BindingKind::ReadOnlyBuffer, 1, false},
                                           BindingSlot{4, BindingKind::ReadOnlyBuffer, 1, false}};
    auto layout = device.CreateBindingLayout(BindingLayoutDesc{0, slots});
    ASSERT_TRUE(layout.has_value()) << layout.error();
    const BindingLayout* spaces[] = {layout.value().get()};
    auto kernelLayout = device.CreateKernelLayout(KernelLayoutDesc{spaces, 32});
    ASSERT_TRUE(kernelLayout.has_value()) << kernelLayout.error();
    auto kernel = device.CreateComputeKernel(ComputeKernelDesc{&code.value()[0], kernelLayout.value().get(), {}});
    ASSERT_TRUE(kernel.has_value()) << kernel.error();
    EXPECT_EQ(kernel.value()->Name(), "spectral_reconstruct");

    auto pool = device.CreateBindGroupPool(*layout.value(), 2);
    ASSERT_TRUE(pool.has_value()) << pool.error();
    auto groups = pool.value()->Allocate(2);
    ASSERT_TRUE(groups.has_value()) << groups.error();
    ASSERT_EQ(groups.value().size(), 2u);

    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<BindingWrite> writes;
    for (u32 slot = 0; slot < 5; ++slot) {
        buffers.push_back(MakeBuffer(device, 256, BufferUsage::Storage, MemoryClass::DeviceLocal));
        ASSERT_NE(buffers.back(), nullptr);
        writes.push_back(BindingWrite::Buffer(slot, slots[slot].kind, buffers.back()->Ref()));
    }
    device.WriteBindGroup(groups.value()[0], writes);

    pool.value()->Reset();
    EXPECT_TRUE(pool.value()->Allocate(1).has_value());
}
