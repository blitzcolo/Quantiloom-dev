#include "gpu/vulkan/VulkanDevice.hpp"

#include "core/ResultFail.hpp"
#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanResources.hpp"
#include "gpu/vulkan/VulkanSync.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/VulkanContext.hpp"

#include <exception>

namespace quantiloom::gpu::vulkan {

namespace {

VkImageLayout SampledLayout(const TextureRef& texture) {
    return texture.layout == TextureLayoutPolicy::Optimal ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                          : VK_IMAGE_LAYOUT_GENERAL;
}

VkDescriptorImageInfo SampledInfo(const TextureRef& texture) {
    return VkDescriptorImageInfo{VK_NULL_HANDLE, FromHandle<VkImageView>(texture.view), SampledLayout(texture)};
}

VkDescriptorImageInfo SamplerInfo(const SamplerRef& sampler) {
    return VkDescriptorImageInfo{FromHandle<VkSampler>(sampler.handle), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
}

}  // namespace

VulkanDevice::VulkanDevice(VulkanContext& context) : context_(context) {
    const VkPhysicalDeviceProperties& props = context.GetDeviceProperties();
    info_.backend = Backend::Vulkan;
    info_.name = props.deviceName;
    info_.vendorId = props.vendorID;
    info_.deviceId = props.deviceID;
    info_.driverVersion = props.driverVersion;
    info_.maxStorageBufferRange = props.limits.maxStorageBufferRange;
    info_.maxComputeSharedMemoryBytes = props.limits.maxComputeSharedMemorySize;
    info_.timestampPeriodNs = static_cast<f64>(props.limits.timestampPeriod);
    info_.hardwareRayTracing = context.IsRayTracingSupported();
    info_.maxBindlessTextures = context.GetCapabilities().hasDescriptorIndexing ? 1024u : 32u;
    info_.maxTraceRecursionDepth = context.GetRayTracingProperties().maxRayRecursionDepth;
    info_.rtScratchAlignment = context.GetAccelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment;
    info_.timestampsOnAllQueues = props.limits.timestampComputeAndGraphics == VK_TRUE;
    VkFormatProperties accumulation{};
    vkGetPhysicalDeviceFormatProperties(context.GetPhysicalDevice(), VK_FORMAT_R32G32B32A32_SFLOAT, &accumulation);
    info_.linearFilterRGBA32Float =
        (accumulation.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0U;
}

Result<std::unique_ptr<Buffer>, String> VulkanDevice::CreateBuffer(const BufferDesc& desc) {
    try {
        return std::unique_ptr<Buffer>(std::make_unique<VulkanBuffer>(
            GpuBuffer(context_.GetAllocator(), desc.size, ToVk(desc.usage), ToVma(desc.memory), desc.minAlignment)));
    } catch (const std::exception& e) {
        return Fail<std::unique_ptr<Buffer>>(String("GpuBuffer creation failed: ") + e.what());
    }
}

Result<std::unique_ptr<Texture>, String> VulkanDevice::CreateTexture(const TextureDesc& desc) {
    try {
        const VkFormat format = ToVk(desc.format);
        const VkImageUsageFlags usage = ToVk(desc.usage);
        const VmaMemoryUsage memory = ToVma(desc.memory);
        if (desc.layers == 1 && desc.dimension == TextureDimension::Tex2D) {
            return std::unique_ptr<Texture>(std::make_unique<VulkanTexture>(
                GpuImage(context_.GetAllocator(), context_.GetDevice(), desc.width, desc.height, format, usage, memory,
                         desc.mipLevels),
                desc));
        }
        const bool cube = desc.dimension == TextureDimension::Cube;
        return std::unique_ptr<Texture>(std::make_unique<VulkanTexture>(
            GpuImage(context_.GetAllocator(), context_.GetDevice(), desc.width, desc.height, format, usage, memory,
                     desc.mipLevels, desc.layers, static_cast<VkImageCreateFlags>(cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0),
                     cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D_ARRAY),
            desc));
    } catch (const std::exception& e) {
        return Fail<std::unique_ptr<Texture>>(String("GpuImage creation failed: ") + e.what());
    }
}

Result<std::unique_ptr<BindingLayout>, String> VulkanDevice::CreateBindingLayout(const BindingLayoutDesc& desc) {
    return VulkanBindingLayout::Create(context_.GetDevice(), desc);
}

Result<std::unique_ptr<KernelLayout>, String> VulkanDevice::CreateKernelLayout(const KernelLayoutDesc& desc) {
    return VulkanKernelLayout::Create(context_.GetDevice(), desc);
}

Result<std::unique_ptr<ComputeKernel>, String> VulkanDevice::CreateComputeKernel(const ComputeKernelDesc& desc) {
    return VulkanComputeKernel::Create(context_.GetDevice(), desc);
}

Result<std::unique_ptr<BindGroupPool>, String> VulkanDevice::CreateBindGroupPool(const BindingLayout& layout,
                                                                                 u32 maxGroups) {
    return VulkanBindGroupPool::Create(context_.GetDevice(), layout, maxGroups);
}

void VulkanDevice::WriteBindGroup(BindGroup group, std::span<const BindingWrite> writes) {
    const VkDescriptorSet set = FromHandle<VkDescriptorSet>(group.handle);

    // The native writes point into these arrays, so each is reserved to its final size
    // before the first push_back and never reallocates.
    size_t imageCount = 0;
    for (const BindingWrite& w : writes) {
        if (w.kind == BindingKind::SampledTexture && !w.textures.empty()) {
            imageCount += w.textures.size();
        } else if (w.kind == BindingKind::Sampler && !w.samplers.empty()) {
            imageCount += w.samplers.size();
        } else {
            imageCount += 1;
        }
    }
    Vector<VkDescriptorBufferInfo> bufferInfos;
    Vector<VkDescriptorImageInfo> imageInfos;
    Vector<VkAccelerationStructureKHR> scenes;
    Vector<VkWriteDescriptorSetAccelerationStructureKHR> sceneInfos;
    bufferInfos.reserve(writes.size());
    imageInfos.reserve(imageCount);
    scenes.reserve(writes.size());
    sceneInfos.reserve(writes.size());
    Vector<VkWriteDescriptorSet> native;
    native.reserve(writes.size());

    for (const BindingWrite& w : writes) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = w.slot;
        write.dstArrayElement = w.arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = ToDescriptorType(w.kind);
        switch (w.kind) {
            case BindingKind::StorageTexture:
                imageInfos.push_back(
                    {VK_NULL_HANDLE, FromHandle<VkImageView>(w.texture.view), VK_IMAGE_LAYOUT_GENERAL});
                write.pImageInfo = &imageInfos.back();
                break;
            case BindingKind::SampledTexture: {
                const size_t first = imageInfos.size();
                if (w.textures.empty()) {
                    imageInfos.push_back(SampledInfo(w.texture));
                } else {
                    for (const TextureRef& t : w.textures) imageInfos.push_back(SampledInfo(t));
                    write.descriptorCount = static_cast<u32>(w.textures.size());
                }
                write.pImageInfo = &imageInfos[first];
                break;
            }
            case BindingKind::Sampler: {
                const size_t first = imageInfos.size();
                if (w.samplers.empty()) {
                    imageInfos.push_back(SamplerInfo(w.sampler));
                } else {
                    for (const SamplerRef& s : w.samplers) imageInfos.push_back(SamplerInfo(s));
                    write.descriptorCount = static_cast<u32>(w.samplers.size());
                }
                write.pImageInfo = &imageInfos[first];
                break;
            }
            case BindingKind::AccelerationStructure: {
                scenes.push_back(FromHandle<VkAccelerationStructureKHR>(w.scene.handle));
                VkWriteDescriptorSetAccelerationStructureKHR sceneInfo{};
                sceneInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                sceneInfo.accelerationStructureCount = 1;
                sceneInfo.pAccelerationStructures = &scenes.back();
                sceneInfos.push_back(sceneInfo);
                write.pNext = &sceneInfos.back();
                break;
            }
            case BindingKind::ReadOnlyBuffer:
            case BindingKind::ReadWriteBuffer:
            case BindingKind::ConstantBuffer:
                bufferInfos.push_back({FromHandle<VkBuffer>(w.buffer.handle), w.offset, w.range});
                write.pBufferInfo = &bufferInfos.back();
                break;
        }
        native.push_back(write);
    }
    vkUpdateDescriptorSets(context_.GetDevice(), static_cast<u32>(native.size()), native.data(), 0, nullptr);
}

void VulkanDevice::ExecuteImmediate(const std::function<void(CommandList&)>& record) {
    CommandHelper::ExecuteImmediate(context_, [&](VkCommandBuffer cmd) {
        VulkanCommandList list(cmd);
        record(list);
    });
}

Result<void, String> VulkanDevice::WaitIdle() {
    if (vkDeviceWaitIdle(context_.GetDevice()) != VK_SUCCESS) {
        return Result<void, String>::Err("vkDeviceWaitIdle failed");
    }
    return Result<void, String>::Ok();
}

Result<std::unique_ptr<Sampler>, String> VulkanDevice::CreateSampler(const SamplerDesc& desc) {
    return VulkanSampler::Create(context_.GetDevice(), desc);
}

Result<std::unique_ptr<Event>, String> VulkanDevice::CreateGpuEvent() {
    return VulkanEvent::Create(context_.GetDevice());
}

Result<std::unique_ptr<TimestampPool>, String> VulkanDevice::CreateTimestampPool(u32 count) {
    return VulkanTimestampPool::Create(context_.GetDevice(), count);
}

Result<std::unique_ptr<CommandPool>, String> VulkanDevice::CreateCommandPool(const CommandPoolDesc& desc) {
    return VulkanCommandPool::Create(context_.GetDevice(), context_.GetGraphicsQueueFamily(), desc);
}

Result<std::unique_ptr<Fence>, String> VulkanDevice::CreateFence() {
    return VulkanFence::Create(context_.GetDevice());
}

Result<void, String> VulkanDevice::Submit(CommandBuffer& commandBuffer, Fence* signal) {
    const VkCommandBuffer cmd = static_cast<VulkanCommandBuffer&>(commandBuffer).Native();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    const VkFence fence = signal != nullptr ? static_cast<VulkanFence*>(signal)->Native() : VK_NULL_HANDLE;
    const VkResult result = vkQueueSubmit(context_.GetGraphicsQueue(), 1, &submit, fence);
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

Result<void, String> VulkanDevice::WaitQueueIdle() {
    const VkResult result = vkQueueWaitIdle(context_.GetGraphicsQueue());
    if (result != VK_SUCCESS) return Fail<void>(ResultCode(result));
    return Result<void, String>::Ok();
}

std::unique_ptr<CommandList> VulkanDevice::WrapRecording(NativeHandle commandBuffer) {
    return std::make_unique<VulkanCommandList>(FromHandle<VkCommandBuffer>(commandBuffer));
}

TextureRef VulkanDevice::ImportHostTexture(NativeHandle image, Format format, u32 width, u32 height) {
    TextureRef ref;
    ref.image = image;
    ref.view = 0;
    ref.format = format;
    ref.width = width;
    ref.height = height;
    ref.layers = 1;
    ref.layout = TextureLayoutPolicy::Optimal;
    return ref;
}

}  // namespace quantiloom::gpu::vulkan
