#pragma once

#include "gpu/Device.hpp"

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

    [[nodiscard]] VulkanContext& Context() const { return context_; }

private:
    VulkanContext& context_;
    DeviceInfo info_;
    ShaderLibrary shaders_{ShaderFormat::SpirV};
};

}  // namespace quantiloom::gpu::vulkan
