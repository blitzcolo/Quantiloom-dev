#include "gpu/vulkan/VulkanResources.hpp"

#include "core/ResultFail.hpp"

#include <algorithm>
#include <cstring>
#include <set>

namespace quantiloom::gpu::vulkan {

// ----------------------------------------------------------------------------
// VulkanBindingLayout
// ----------------------------------------------------------------------------

Result<std::unique_ptr<BindingLayout>, String> VulkanBindingLayout::Create(VkDevice device,
                                                                           const BindingLayoutDesc& desc) {
    std::set<u32> seen;
    for (const BindingSlot& slot : desc.slots) {
        if (!seen.insert(slot.slot).second) {
            return Fail<std::unique_ptr<BindingLayout>>(
                "binding slot " + std::to_string(slot.slot) + " repeats in space " + std::to_string(desc.space));
        }
    }

    Vector<VkDescriptorSetLayoutBinding> bindings;
    Vector<VkDescriptorBindingFlags> flags;
    bool anyPartial = false;
    for (const BindingSlot& slot : desc.slots) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = slot.slot;
        binding.descriptorType = ToDescriptorType(slot.kind);
        binding.descriptorCount = slot.count;
        binding.stageFlags = ToVk(slot.visibility);
        bindings.push_back(binding);
        flags.push_back(static_cast<VkDescriptorBindingFlags>(slot.partiallyBound ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0));
        anyPartial = anyPartial || slot.partiallyBound;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = static_cast<u32>(flags.size());
    flagsInfo.pBindingFlags = flags.data();

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = anyPartial ? &flagsInfo : nullptr;
    info.bindingCount = static_cast<u32>(bindings.size());
    info.pBindings = bindings.data();

    std::unique_ptr<VulkanBindingLayout> layout(new VulkanBindingLayout());
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout->layout_) != VK_SUCCESS) {
        layout->layout_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<BindingLayout>>("vkCreateDescriptorSetLayout failed");
    }
    layout->device_ = device;
    layout->space_ = desc.space;
    layout->slots_.assign(desc.slots.begin(), desc.slots.end());
    return std::unique_ptr<BindingLayout>(std::move(layout));
}

VulkanBindingLayout::~VulkanBindingLayout() {
    if (layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
}

// ----------------------------------------------------------------------------
// VulkanKernelLayout
// ----------------------------------------------------------------------------

Result<std::unique_ptr<KernelLayout>, String> VulkanKernelLayout::Create(VkDevice device,
                                                                         const KernelLayoutDesc& desc) {
    Vector<VkDescriptorSetLayout> sets;
    for (u32 i = 0; i < desc.spaces.size(); ++i) {
        const auto* space = static_cast<const VulkanBindingLayout*>(desc.spaces[i]);
        if (space == nullptr || space->Space() != i) {
            return Fail<std::unique_ptr<KernelLayout>>("kernel layout spaces must be contiguous from 0");
        }
        sets.push_back(space->Native());
    }

    VkPushConstantRange range{ToVk(desc.pushConstantVisibility), 0, desc.pushConstantBytes};
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = static_cast<u32>(sets.size());
    info.pSetLayouts = sets.data();
    info.pushConstantRangeCount = desc.pushConstantBytes > 0 ? 1u : 0u;
    info.pPushConstantRanges = desc.pushConstantBytes > 0 ? &range : nullptr;

    std::unique_ptr<VulkanKernelLayout> layout(new VulkanKernelLayout());
    if (vkCreatePipelineLayout(device, &info, nullptr, &layout->layout_) != VK_SUCCESS) {
        layout->layout_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<KernelLayout>>("vkCreatePipelineLayout failed");
    }
    layout->device_ = device;
    layout->pushBytes_ = desc.pushConstantBytes;
    layout->pushVisibility_ = desc.pushConstantVisibility;
    return std::unique_ptr<KernelLayout>(std::move(layout));
}

VulkanKernelLayout::~VulkanKernelLayout() {
    if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
}

// ----------------------------------------------------------------------------
// VulkanComputeKernel
// ----------------------------------------------------------------------------

Result<std::unique_ptr<ComputeKernel>, String> VulkanComputeKernel::Create(VkDevice device,
                                                                           const ComputeKernelDesc& desc) {
    if (desc.code == nullptr || desc.layout == nullptr) {
        return Fail<std::unique_ptr<ComputeKernel>>("compute kernel needs code and a layout");
    }
    const ShaderCode& code = *desc.code;
    if (code.bytes.empty() || code.bytes.size() % sizeof(u32) != 0) {
        return Fail<std::unique_ptr<ComputeKernel>>("shader " + code.name + " is not a whole number of SPIR-V words");
    }

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.bytes.size();
    moduleInfo.pCode = reinterpret_cast<const u32*>(code.bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
        return Fail<std::unique_ptr<ComputeKernel>>("vkCreateShaderModule failed for " + code.name);
    }

    Vector<VkSpecializationMapEntry> entries;
    Vector<u32> values;
    for (const SpecializationConstant& constant : desc.specialization) {
        entries.push_back({constant.id, static_cast<u32>(values.size() * sizeof(u32)), sizeof(u32)});
        values.push_back(constant.value);
    }
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = static_cast<u32>(entries.size());
    specialization.pMapEntries = entries.data();
    specialization.dataSize = values.size() * sizeof(u32);
    specialization.pData = values.data();

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.stage.pSpecializationInfo = entries.empty() ? nullptr : &specialization;
    pipelineInfo.layout = static_cast<const VulkanKernelLayout*>(desc.layout)->Native();

    std::unique_ptr<VulkanComputeKernel> kernel(new VulkanComputeKernel());
    const VkResult result =
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &kernel->pipeline_);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS) {
        kernel->pipeline_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<ComputeKernel>>("vkCreateComputePipelines failed for " + code.name);
    }
    kernel->device_ = device;
    kernel->layout_ = desc.layout;
    kernel->name_ = code.name;
    return std::unique_ptr<ComputeKernel>(std::move(kernel));
}

VulkanComputeKernel::~VulkanComputeKernel() {
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
}

// ----------------------------------------------------------------------------
// VulkanBindGroupPool
// ----------------------------------------------------------------------------

Result<std::unique_ptr<BindGroupPool>, String> VulkanBindGroupPool::Create(VkDevice device,
                                                                           const BindingLayout& layout,
                                                                           u32 maxGroups) {
    Vector<VkDescriptorPoolSize> sizes;
    for (const BindingSlot& slot : layout.Slots()) {
        const VkDescriptorType type = ToDescriptorType(slot.kind);
        auto it = std::find_if(sizes.begin(), sizes.end(),
                               [type](const VkDescriptorPoolSize& s) { return s.type == type; });
        if (it == sizes.end()) {
            sizes.push_back({type, slot.count * maxGroups});
        } else {
            it->descriptorCount += slot.count * maxGroups;
        }
    }

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = maxGroups;
    info.poolSizeCount = static_cast<u32>(sizes.size());
    info.pPoolSizes = sizes.data();

    std::unique_ptr<VulkanBindGroupPool> pool(new VulkanBindGroupPool());
    if (vkCreateDescriptorPool(device, &info, nullptr, &pool->pool_) != VK_SUCCESS) {
        pool->pool_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<BindGroupPool>>("vkCreateDescriptorPool failed");
    }
    pool->device_ = device;
    pool->setLayout_ = static_cast<const VulkanBindingLayout&>(layout).Native();
    return std::unique_ptr<BindGroupPool>(std::move(pool));
}

VulkanBindGroupPool::~VulkanBindGroupPool() {
    if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
}

Result<Vector<BindGroup>, String> VulkanBindGroupPool::Allocate(u32 count) {
    Vector<VkDescriptorSetLayout> layouts(count, setLayout_);
    Vector<VkDescriptorSet> sets(count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool_;
    info.descriptorSetCount = count;
    info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &info, sets.data()) != VK_SUCCESS) {
        return Result<Vector<BindGroup>, String>::Err("vkAllocateDescriptorSets failed");
    }
    Vector<BindGroup> groups;
    groups.reserve(count);
    for (VkDescriptorSet set : sets) groups.push_back(BindGroup{ToHandle(set)});
    return groups;
}

void VulkanBindGroupPool::Reset() {
    vkResetDescriptorPool(device_, pool_, 0);
}

// ----------------------------------------------------------------------------
// VulkanSampler
// ----------------------------------------------------------------------------

Result<std::unique_ptr<Sampler>, String> VulkanSampler::Create(VkDevice device, const SamplerDesc& desc) {
    const VkSamplerCreateInfo info = ToVk(desc);
    std::unique_ptr<VulkanSampler> sampler(new VulkanSampler());
    const VkResult result = vkCreateSampler(device, &info, nullptr, &sampler->sampler_);
    if (result != VK_SUCCESS) {
        sampler->sampler_ = VK_NULL_HANDLE;
        return Fail<std::unique_ptr<Sampler>>(ResultCode(result));
    }
    sampler->device_ = device;
    return std::unique_ptr<Sampler>(std::move(sampler));
}

VulkanSampler::~VulkanSampler() {
    if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, sampler_, nullptr);
}

}  // namespace quantiloom::gpu::vulkan
