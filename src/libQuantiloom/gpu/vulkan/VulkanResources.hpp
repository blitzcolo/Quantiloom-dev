#pragma once

#include "gpu/Binding.hpp"
#include "gpu/Kernel.hpp"
#include "gpu/Resources.hpp"
#include "gpu/ShaderLibrary.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"
#include "gpu/vulkan/GpuImage.hpp"

#include <vulkan/vulkan.h>

#include <memory>

namespace quantiloom::gpu::vulkan {

class VulkanBuffer final : public Buffer {
public:
    explicit VulkanBuffer(GpuBuffer&& buffer) : buffer_(std::move(buffer)) {}
    [[nodiscard]] BufferRef Ref() const override {
        return BufferRef{ToHandle(buffer_.GetHandle()), static_cast<u64>(buffer_.GetSize())};
    }
    void* Map() override { return buffer_.Map(); }
    const void* MapRead() override { return buffer_.MapRead(); }
    void Unmap() override { buffer_.Unmap(); }
    void Upload(const void* data, u64 bytes, u64 offset = 0) override { buffer_.Upload(data, bytes, offset); }
    [[nodiscard]] GpuBuffer& Native() { return buffer_; }
    [[nodiscard]] const GpuBuffer& Native() const { return buffer_; }

private:
    GpuBuffer buffer_;
};

class VulkanTexture final : public Texture {
public:
    VulkanTexture(GpuImage&& image, const TextureDesc& desc) : image_(std::move(image)), desc_(desc) {}
    [[nodiscard]] TextureRef Ref() const override {
        TextureRef ref;
        ref.image = ToHandle(image_.GetImage());
        ref.view = ToHandle(image_.GetView());
        ref.format = desc_.format;
        ref.width = desc_.width;
        ref.height = desc_.height;
        ref.layers = desc_.layers;
        ref.layout = HasAny(desc_.usage, TextureUsage::Storage) ? TextureLayoutPolicy::General
                                                                : TextureLayoutPolicy::Optimal;
        return ref;
    }
    [[nodiscard]] const GpuImage& Native() const { return image_; }

private:
    GpuImage image_;
    TextureDesc desc_;
};

// A raw GpuImage or GpuBuffer as a ref, and the GpuImage behind a texture, for Vulkan-only code
// that holds raw objects (the Vulkan-only tests). The image ref's policy is General.
[[nodiscard]] inline TextureRef Ref(const GpuImage& image, u32 layers = 1) {
    TextureRef ref;
    ref.image = ToHandle(image.GetImage());
    ref.view = ToHandle(image.GetView());
    ref.format = FromVk(image.GetFormat());
    ref.width = image.GetExtent().width;
    ref.height = image.GetExtent().height;
    ref.layers = layers;
    ref.layout = TextureLayoutPolicy::General;
    return ref;
}

[[nodiscard]] inline BufferRef Ref(const GpuBuffer& buffer) {
    return BufferRef{ToHandle(buffer.GetHandle()), static_cast<u64>(buffer.GetSize())};
}

[[nodiscard]] inline const GpuImage& AsGpuImage(const Texture& texture) {
    return static_cast<const VulkanTexture&>(texture).Native();
}

class VulkanBindingLayout final : public BindingLayout {
public:
    [[nodiscard]] static Result<std::unique_ptr<BindingLayout>, String> Create(VkDevice device,
                                                                               const BindingLayoutDesc& desc);
    ~VulkanBindingLayout() override;
    VulkanBindingLayout(const VulkanBindingLayout&) = delete;
    VulkanBindingLayout& operator=(const VulkanBindingLayout&) = delete;

    [[nodiscard]] u32 Space() const override { return space_; }
    [[nodiscard]] std::span<const BindingSlot> Slots() const override { return slots_; }
    [[nodiscard]] VkDescriptorSetLayout Native() const { return layout_; }

private:
    VulkanBindingLayout() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    u32 space_ = 0;
    Vector<BindingSlot> slots_;
};

class VulkanKernelLayout final : public KernelLayout {
public:
    [[nodiscard]] static Result<std::unique_ptr<KernelLayout>, String> Create(VkDevice device,
                                                                              const KernelLayoutDesc& desc);
    ~VulkanKernelLayout() override;
    VulkanKernelLayout(const VulkanKernelLayout&) = delete;
    VulkanKernelLayout& operator=(const VulkanKernelLayout&) = delete;

    [[nodiscard]] u32 PushConstantBytes() const override { return pushBytes_; }
    [[nodiscard]] ShaderStages PushConstantVisibility() const override { return pushVisibility_; }
    [[nodiscard]] VkPipelineLayout Native() const { return layout_; }

private:
    VulkanKernelLayout() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    u32 pushBytes_ = 0;
    ShaderStages pushVisibility_ = ShaderStages::Compute;
};

class VulkanComputeKernel final : public ComputeKernel {
public:
    [[nodiscard]] static Result<std::unique_ptr<ComputeKernel>, String> Create(VkDevice device,
                                                                               const ComputeKernelDesc& desc);
    ~VulkanComputeKernel() override;
    VulkanComputeKernel(const VulkanComputeKernel&) = delete;
    VulkanComputeKernel& operator=(const VulkanComputeKernel&) = delete;

    [[nodiscard]] const KernelLayout& Layout() const override { return *layout_; }
    [[nodiscard]] StringView Name() const override { return name_; }
    [[nodiscard]] VkPipeline Native() const { return pipeline_; }

private:
    VulkanComputeKernel() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    const KernelLayout* layout_ = nullptr;
    String name_;
};

class VulkanBindGroupPool final : public BindGroupPool {
public:
    [[nodiscard]] static Result<std::unique_ptr<BindGroupPool>, String> Create(VkDevice device,
                                                                               const BindingLayout& layout,
                                                                               u32 maxGroups);
    ~VulkanBindGroupPool() override;
    VulkanBindGroupPool(const VulkanBindGroupPool&) = delete;
    VulkanBindGroupPool& operator=(const VulkanBindGroupPool&) = delete;

    [[nodiscard]] Result<Vector<BindGroup>, String> Allocate(u32 count) override;
    void Reset() override;

private:
    VulkanBindGroupPool() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
};

class VulkanSampler final : public Sampler {
public:
    [[nodiscard]] static Result<std::unique_ptr<Sampler>, String> Create(VkDevice device, const SamplerDesc& desc);
    ~VulkanSampler() override;
    VulkanSampler(const VulkanSampler&) = delete;
    VulkanSampler& operator=(const VulkanSampler&) = delete;

    [[nodiscard]] SamplerRef Ref() const override { return SamplerRef{ToHandle(sampler_)}; }
    [[nodiscard]] VkSampler Native() const { return sampler_; }

private:
    VulkanSampler() = default;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace quantiloom::gpu::vulkan
