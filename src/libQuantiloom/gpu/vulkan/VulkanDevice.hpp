#pragma once

#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanRt.hpp"

namespace quantiloom {
class VulkanContext;
}

namespace quantiloom::gpu::vulkan {

class VulkanDevice final : public Device {
public:
    explicit VulkanDevice(VulkanContext& context);

    [[nodiscard]] const DeviceInfo& Info() const override { return info_; }
    [[nodiscard]] const ShaderLibrary& Shaders() const override { return shaders_; }

    [[nodiscard]] Result<std::unique_ptr<Buffer>, String> CreateBuffer(const BufferDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Texture>, String> CreateTexture(const TextureDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<BindingLayout>, String> CreateBindingLayout(
        const BindingLayoutDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<KernelLayout>, String> CreateKernelLayout(
        const KernelLayoutDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<ComputeKernel>, String> CreateComputeKernel(
        const ComputeKernelDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<BindGroupPool>, String> CreateBindGroupPool(const BindingLayout& layout,
                                                                                     u32 maxGroups) override;

    void WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) override;
    void ExecuteImmediate(const std::function<void(CommandList&)>& record) override;
    [[nodiscard]] Result<void, String> WaitIdle() override;

    [[nodiscard]] Result<std::unique_ptr<Sampler>, String> CreateSampler(const SamplerDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Event>, String> CreateGpuEvent() override;
    [[nodiscard]] Result<std::unique_ptr<TimestampPool>, String> CreateTimestampPool(u32 count) override;
    [[nodiscard]] Result<std::unique_ptr<CommandPool>, String> CreateCommandPool(
        const CommandPoolDesc& desc) override;
    [[nodiscard]] Result<std::unique_ptr<Fence>, String> CreateFence() override;
    [[nodiscard]] Result<void, String> Submit(CommandBuffer& commandBuffer, Fence* signal) override;
    [[nodiscard]] Result<void, String> WaitQueueIdle() override;
    [[nodiscard]] std::unique_ptr<CommandList> WrapRecording(NativeHandle commandBuffer) override;
    [[nodiscard]] TextureRef ImportHostTexture(NativeHandle image, Format format, u32 width, u32 height) override;

    [[nodiscard]] Result<std::unique_ptr<PipelineCache>, String> CreatePipelineCache(
        std::span<const std::byte> initialData) override;
    [[nodiscard]] Result<std::unique_ptr<TraceKernel>, String> CreateTraceKernel(
        const TraceKernelDesc& desc) override;
    [[nodiscard]] RtBuildSizes GetRtBuildSizes(const RtGeometry& geometry) const override;
    [[nodiscard]] Result<std::unique_ptr<RtAccelerationStructure>, String> CreateRtAccelerationStructure(
        RtLevel level, u64 bytes) override;
    [[nodiscard]] u32 RtInstanceStride() const override;
    void PackRtInstances(std::span<const RtInstance> instances, std::span<std::byte> out) const override;

    [[nodiscard]] VulkanContext& Context() const { return context_; }

private:
    VulkanContext& context_;
    DeviceInfo info_;
    ShaderLibrary shaders_{ShaderFormat::SpirV};
    RtFunctions rt_;  // loaded once; copied into every command list this device hands out
};

}  // namespace quantiloom::gpu::vulkan
