#include "gpu/vulkan/VulkanBridge.hpp"

#include "core/Platform.hpp"
#include "gpu/vulkan/VulkanDevice.hpp"
#include "gpu/vulkan/VulkanResources.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"

namespace quantiloom::gpu::vulkan {

std::unique_ptr<gpu::Device> AdoptContext(VulkanContext& context) {
    return std::make_unique<VulkanDevice>(context);
}

VulkanCommandList Wrap(VkCommandBuffer cmd) {
    return VulkanCommandList(cmd);
}

VkCommandBuffer NativeCommandBuffer(gpu::CommandList& list) {
    QL_ASSERT(list.GetBackend() == Backend::Vulkan, "not a Vulkan command list");
    return static_cast<VulkanCommandList&>(list).Native();
}

gpu::TextureRef Ref(const GpuImage& image, u32 layers) {
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

gpu::BufferRef Ref(const GpuBuffer& buffer) {
    return BufferRef{ToHandle(buffer.GetHandle()), static_cast<u64>(buffer.GetSize())};
}

gpu::BufferRef Ref(VkBuffer buffer, VkDeviceSize size) {
    return BufferRef{ToHandle(buffer), static_cast<u64>(size)};
}

gpu::TimestampPoolRef Ref(VkQueryPool pool, u32 count) {
    return TimestampPoolRef{ToHandle(pool), count};
}

VkImage NativeImage(gpu::TextureRef ref) {
    return FromHandle<VkImage>(ref.image);
}

VkImageView NativeView(gpu::TextureRef ref) {
    return FromHandle<VkImageView>(ref.view);
}

VkBuffer NativeBuffer(gpu::BufferRef ref) {
    return FromHandle<VkBuffer>(ref.handle);
}

VulkanContext& Context(gpu::Device& device) {
    return static_cast<VulkanDevice&>(device).Context();
}

const GpuImage& AsGpuImage(const gpu::Texture& texture) {
    return static_cast<const VulkanTexture&>(texture).Native();
}

GpuBuffer& AsGpuBuffer(gpu::Buffer& buffer) {
    return static_cast<VulkanBuffer&>(buffer).Native();
}

VkPipeline NativePipeline(const gpu::ComputeKernel& kernel) {
    return static_cast<const VulkanComputeKernel&>(kernel).Native();
}

VkPipelineLayout NativeLayout(const gpu::KernelLayout& layout) {
    return static_cast<const VulkanKernelLayout&>(layout).Native();
}

VkDescriptorSet NativeSet(gpu::BindGroup group) {
    return FromHandle<VkDescriptorSet>(group.handle);
}

}  // namespace quantiloom::gpu::vulkan
