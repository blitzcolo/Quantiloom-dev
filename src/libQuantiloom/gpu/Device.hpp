/**
 * @file Device.hpp
 * @brief The factory and synchronous submission surface of the compute RHI (internal)
 */

#pragma once

#include "gpu/Binding.hpp"
#include "gpu/CommandList.hpp"
#include "gpu/Kernel.hpp"
#include "gpu/Resources.hpp"
#include "gpu/RtScene.hpp"
#include "gpu/ShaderLibrary.hpp"
#include "gpu/Sync.hpp"
#include "gpu/Trace.hpp"

#include <functional>
#include <memory>
#include <span>

namespace quantiloom::gpu {

class Device {
public:
    virtual ~Device() = default;
    [[nodiscard]] virtual const DeviceInfo& Info() const = 0;
    [[nodiscard]] virtual const ShaderLibrary& Shaders() const = 0;

    [[nodiscard]] virtual Result<std::unique_ptr<Buffer>, String> CreateBuffer(const BufferDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<Texture>, String> CreateTexture(const TextureDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<BindingLayout>, String> CreateBindingLayout(
        const BindingLayoutDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<KernelLayout>, String> CreateKernelLayout(
        const KernelLayoutDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<ComputeKernel>, String> CreateComputeKernel(
        const ComputeKernelDesc&) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<BindGroupPool>, String> CreateBindGroupPool(
        const BindingLayout&, u32 maxGroups) = 0;

    /// One native update, writes applied in order. Same rule as vkUpdateDescriptorSets:
    /// the group must not be in use by a pending submission.
    virtual void WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) = 0;

    /// Record, submit, and wait. Re-entrant: a nested call gets its own pool and fence.
    /// Throws std::runtime_error when the submit or the wait fails (CommandHelper's contract).
    virtual void ExecuteImmediate(const std::function<void(CommandList&)>& record) = 0;

    /// vkDeviceWaitIdle. Err on device loss.
    [[nodiscard]] virtual Result<void, String> WaitIdle() = 0;

    [[nodiscard]] virtual Result<std::unique_ptr<Sampler>, String> CreateSampler(const SamplerDesc& desc) = 0;
    /// Not named CreateEvent: <windows.h> defines that as a macro, and a translation unit that
    /// saw it would declare and call a differently named member than VulkanDevice.cpp defines.
    [[nodiscard]] virtual Result<std::unique_ptr<Event>, String> CreateGpuEvent() = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<TimestampPool>, String> CreateTimestampPool(u32 count) = 0;
    /// On the device's one queue family.
    [[nodiscard]] virtual Result<std::unique_ptr<CommandPool>, String> CreateCommandPool(
        const CommandPoolDesc& desc) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<Fence>, String> CreateFence() = 0;
    /// vkQueueSubmit of one ended command buffer, signalling `signal` when it is not null.
    /// Does not wait: pair it with Fence::Wait or WaitQueueIdle.
    [[nodiscard]] virtual Result<void, String> Submit(CommandBuffer& commandBuffer, Fence* signal) = 0;
    /// vkQueueWaitIdle on the device's queue.
    [[nodiscard]] virtual Result<void, String> WaitQueueIdle() = 0;
    /// A host's command buffer in the recording state (GpuCommandRecording::commandBuffer).
    [[nodiscard]] virtual std::unique_ptr<CommandList> WrapRecording(NativeHandle commandBuffer) = 0;
    /// A host's image (GpuImageTarget::image): no view, layer 1, policy Optimal. Its layouts
    /// travel in TextureBarrier::hostLayoutBefore/After.
    [[nodiscard]] virtual TextureRef ImportHostTexture(NativeHandle image, Format format, u32 width,
                                                       u32 height) = 0;

    /// From a previous run's PipelineCache::Serialize bytes. When the driver rejects them
    /// (stale or foreign) it warns and creates an empty cache instead; Err only when that fails too.
    [[nodiscard]] virtual Result<std::unique_ptr<PipelineCache>, String> CreatePipelineCache(
        std::span<const std::byte> initialData) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<TraceKernel>, String> CreateTraceKernel(
        const TraceKernelDesc& desc) = 0;
    /// What a build of `geometry` needs (vkGetAccelerationStructureBuildSizesKHR); all zero on a
    /// device without acceleration structures. Metal: accelerationStructureSizesWithDescriptor.
    [[nodiscard]] virtual RtBuildSizes GetRtBuildSizes(const RtGeometry& geometry) const = 0;
    /// A storage buffer of `bytes`, the structure on it and, for a Bottom level, its device
    /// address, in that order. Metal: newAccelerationStructureWithSize.
    [[nodiscard]] virtual Result<std::unique_ptr<RtAccelerationStructure>, String>
    CreateRtAccelerationStructure(RtLevel level, u64 bytes) = 0;
    /// Bytes per packed instance record: 64 on Vulkan.
    [[nodiscard]] virtual u32 RtInstanceStride() const = 0;
    /// Writes one native record per instance into `out`, which holds at least
    /// instances.size() * RtInstanceStride() bytes.
    virtual void PackRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out) const = 0;
    /// The device address of a buffer created with BufferUsage::DeviceAddress
    /// (vkGetBufferDeviceAddress); 0 for a null buffer. Metal: MTL::Buffer::gpuAddress.
    [[nodiscard]] virtual u64 BufferDeviceAddress(BufferRef buffer) const = 0;
};

}  // namespace quantiloom::gpu
