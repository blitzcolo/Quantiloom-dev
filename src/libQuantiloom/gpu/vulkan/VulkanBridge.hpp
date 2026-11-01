#pragma once

#include "gpu/Device.hpp"
#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanMapping.hpp"

#include <vulkan/vulkan.h>

namespace quantiloom {
class VulkanContext;
class GpuImage;
class GpuBuffer;
}

namespace quantiloom::gpu::vulkan {

[[nodiscard]] std::unique_ptr<gpu::Device> AdoptContext(VulkanContext& context);

[[nodiscard]] VulkanCommandList Wrap(VkCommandBuffer cmd);
[[nodiscard]] VkCommandBuffer NativeCommandBuffer(gpu::CommandList& list);

[[nodiscard]] gpu::TextureRef Ref(const GpuImage& image, u32 layers = 1);
[[nodiscard]] gpu::BufferRef Ref(const GpuBuffer& buffer);
[[nodiscard]] gpu::BufferRef Ref(VkBuffer buffer, VkDeviceSize size);
[[nodiscard]] gpu::TimestampPoolRef Ref(VkQueryPool pool, u32 count);

[[nodiscard]] VkImage NativeImage(gpu::TextureRef ref);
[[nodiscard]] VkImageView NativeView(gpu::TextureRef ref);
[[nodiscard]] VkBuffer NativeBuffer(gpu::BufferRef ref);

// Transitional: deleted when GpuBuffer/GpuImage/CommandHelper move into gpu/vulkan/.
[[nodiscard]] VulkanContext& Context(gpu::Device& device);
[[nodiscard]] const GpuImage& AsGpuImage(const gpu::Texture& texture);
[[nodiscard]] GpuBuffer& AsGpuBuffer(gpu::Buffer& buffer);
[[nodiscard]] VkPipeline NativePipeline(const gpu::ComputeKernel& kernel);
[[nodiscard]] VkPipelineLayout NativeLayout(const gpu::KernelLayout& layout);
[[nodiscard]] VkDescriptorSet NativeSet(gpu::BindGroup group);

// The ray tracing modules' raw samplers, views and acceleration structures until they hold gpu::
// objects. A view-only SampledRef has no image and policy Optimal: it is written
// SHADER_READ_ONLY_OPTIMAL, as RayTracingPipeline's texture, BRDF LUT and environment writes are.
[[nodiscard]] gpu::SamplerRef Ref(VkSampler sampler);                    // Transitional; deleted in S19
[[nodiscard]] gpu::RtSceneRef Ref(VkAccelerationStructureKHR structure,
                                  VkDeviceAddress address = 0);           // Transitional; deleted in S19
[[nodiscard]] gpu::TextureRef SampledRef(VkImageView view);               // Transitional; deleted in S19
[[nodiscard]] VkSampler NativeSampler(gpu::SamplerRef ref);               // Transitional; deleted in S19
[[nodiscard]] VkAccelerationStructureKHR NativeScene(gpu::RtSceneRef ref);  // Transitional; deleted in S19

}  // namespace quantiloom::gpu::vulkan
