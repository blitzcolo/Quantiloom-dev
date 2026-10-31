#include "gpu/vulkan/VulkanDevice.hpp"

#include "core/ResultFail.hpp"
#include "gpu/vulkan/VulkanCommandList.hpp"
#include "gpu/vulkan/VulkanResources.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/VulkanContext.hpp"

#include <exception>

namespace quantiloom::gpu::vulkan {

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
    Vector<VkDescriptorBufferInfo> bufferInfos;
    Vector<VkDescriptorImageInfo> imageInfos;
    bufferInfos.reserve(writes.size());
    imageInfos.reserve(writes.size());
    Vector<VkWriteDescriptorSet> native;
    native.reserve(writes.size());

    for (const BindingWrite& w : writes) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = w.slot;
        write.descriptorCount = 1;
        write.descriptorType = ToDescriptorType(w.kind);
        if (w.kind == BindingKind::StorageTexture || w.kind == BindingKind::SampledTexture) {
            imageInfos.push_back({VK_NULL_HANDLE, FromHandle<VkImageView>(w.texture.view), VK_IMAGE_LAYOUT_GENERAL});
            write.pImageInfo = &imageInfos.back();
        } else {
            bufferInfos.push_back({FromHandle<VkBuffer>(w.buffer.handle), w.offset, w.range});
            write.pBufferInfo = &bufferInfos.back();
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

}  // namespace quantiloom::gpu::vulkan
