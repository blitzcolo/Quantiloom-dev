#include "gpu/vulkan/VulkanTrace.hpp"

#include "core/Log.hpp"
#include "core/ResultFail.hpp"
#include "gpu/ShaderLibrary.hpp"
#include "gpu/vulkan/VulkanResources.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>

namespace quantiloom::gpu::vulkan {

namespace {

u32 AlignedSize(const u32 size, const u32 alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool CreateShaderModule(VkDevice device, const ShaderCode& code, VkShaderModule& module) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.bytes.size();
    createInfo.pCode = reinterpret_cast<const u32*>(code.bytes.data());
    return vkCreateShaderModule(device, &createInfo, nullptr, &module) == VK_SUCCESS;
}

// What the driver says each group needs on the ray tracing stack, and what the
// pipeline is therefore given when nothing calls
// vkCmdSetRayTracingPipelineStackSizeKHR (nothing here does). The default is
// the specification's own formula over the recursion depth the pipeline was
// created with, so a deeper recursion multiplies the closest-hit's share and a
// heavier raygen adds once. Logged per kernel because the numbers depend on
// the specialisation constants the kernel was compiled with.
void LogPipelineStackSizes(const RtFunctions& rt, VkPipeline pipeline, StringView name, u32 recursionDepth) {
    if (rt.getGroupStackSize == nullptr) {
        QL_LOG_WARN("  Ray tracing stack sizes: vkGetRayTracingShaderGroupStackSizeKHR unavailable");
        return;
    }
    const auto size = [&](u32 group, VkShaderGroupShaderKHR shader) {
        return static_cast<u64>(rt.getGroupStackSize(rt.device, pipeline, group, shader));
    };
    const u64 rgen = size(0, VK_SHADER_GROUP_SHADER_GENERAL_KHR);
    const u64 chit = size(1, VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR);
    const u64 ahit = size(1, VK_SHADER_GROUP_SHADER_ANY_HIT_KHR);
    const u64 miss = std::max(size(2, VK_SHADER_GROUP_SHADER_GENERAL_KHR),
                              size(3, VK_SHADER_GROUP_SHADER_GENERAL_KHR));
    const u64 depth = recursionDepth;
    const u64 deepest = std::max(chit, miss);
    const u64 pipelineDefault =
        rgen + std::min<u64>(1, depth) * std::max(deepest, ahit) +
        (depth > 0 ? depth - 1 : 0) * deepest;
    QL_LOG_INFO("  Ray tracing stack sizes ({}): raygen {} B, closest-hit {} B, any-hit {} B, "
                "miss {} B; pipeline default at recursion depth {}: {} B",
                name, rgen, chit, ahit, miss, recursionDepth, pipelineDefault);
}

}  // namespace

// ----------------------------------------------------------------------------
// VulkanPipelineCache
// ----------------------------------------------------------------------------

Result<std::unique_ptr<PipelineCache>, String> VulkanPipelineCache::Create(VkDevice device,
                                                                           std::span<const std::byte> initialData) {
    VkPipelineCacheCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    createInfo.initialDataSize = initialData.size();
    createInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();

    std::unique_ptr<VulkanPipelineCache> cache(new VulkanPipelineCache());
    VkResult result = vkCreatePipelineCache(device, &createInfo, nullptr, &cache->cache_);
    if (result != VK_SUCCESS) {
        QL_LOG_WARN("Failed to create pipeline cache (VkResult={}), creating empty cache", static_cast<int>(result));

        // Try again without initial data (cache might be corrupted or incompatible)
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        result = vkCreatePipelineCache(device, &createInfo, nullptr, &cache->cache_);
        if (result != VK_SUCCESS) {
            cache->cache_ = VK_NULL_HANDLE;
            return Fail<std::unique_ptr<PipelineCache>>("Failed to create empty pipeline cache");
        }
    }
    cache->device_ = device;
    return std::unique_ptr<PipelineCache>(std::move(cache));
}

VulkanPipelineCache::~VulkanPipelineCache() {
    if (cache_ != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, cache_, nullptr);
}

Result<Vector<std::byte>, String> VulkanPipelineCache::Serialize() const {
    size_t cacheSize = 0;
    VkResult result = vkGetPipelineCacheData(device_, cache_, &cacheSize, nullptr);
    if (result != VK_SUCCESS || cacheSize == 0) {
        return Fail<Vector<std::byte>>("Failed to get pipeline cache size");
    }
    Vector<std::byte> cacheData(cacheSize);
    result = vkGetPipelineCacheData(device_, cache_, &cacheSize, cacheData.data());
    if (result != VK_SUCCESS) {
        return Fail<Vector<std::byte>>("Failed to get pipeline cache data");
    }
    cacheData.resize(cacheSize);
    return cacheData;
}

// ----------------------------------------------------------------------------
// VulkanTraceKernel
// ----------------------------------------------------------------------------

Result<std::unique_ptr<TraceKernel>, String> VulkanTraceKernel::Create(
    VmaAllocator allocator, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& properties,
    const RtFunctions& rt, const TraceKernelDesc& desc) {
    using Out = std::unique_ptr<TraceKernel>;

    // The stage order RayTracingPipeline stores its SPIR-V in.
    const std::array<const ShaderCode*, 5> codes = {desc.rayGen, desc.closestHit, desc.miss, desc.shadowMiss,
                                                    desc.anyHit};
    if (desc.layout == nullptr || std::find(codes.begin(), codes.end(), nullptr) != codes.end()) {
        return Fail<Out>("trace kernel needs ray generation, closest-hit, any-hit, miss and shadow-miss code "
                         "and a layout");
    }
    for (const ShaderCode* code : codes) {
        if (code->bytes.empty() || code->bytes.size() % sizeof(u32) != 0) {
            return Fail<Out>("shader " + code->name + " is not a whole number of SPIR-V words");
        }
    }
    if (rt.createPipelines == nullptr) return Fail<Out>("Failed to load vkCreateRayTracingPipelinesKHR");
    if (rt.getGroupHandles == nullptr) return Fail<Out>("Failed to load vkGetRayTracingShaderGroupHandlesKHR");
    if (rt.cmdTraceRays == nullptr) return Fail<Out>("Failed to load vkCmdTraceRaysKHR");

    const VkDevice device = rt.device;

    // Create temporary shader modules
    std::array<VkShaderModule, 5> modules{};
    for (size_t i = 0; i < modules.size(); ++i) {
        if (!CreateShaderModule(device, *codes[i], modules[i])) {
            for (size_t j = 0; j < i; ++j) vkDestroyShaderModule(device, modules[j], nullptr);
            return Fail<Out>("Failed to create shader module");
        }
    }

    // One map entry per constant, 4 bytes each in order: for RayTracingPipeline's
    // {spectralMode, debugEnabled} that is {id 0, offset 0}, {id 1, offset 4}, 8 bytes.
    Vector<VkSpecializationMapEntry> specEntries;
    Vector<u32> specValues;
    for (const SpecializationConstant& constant : desc.specialization) {
        specEntries.push_back({constant.id, static_cast<u32>(specValues.size() * sizeof(u32)), sizeof(u32)});
        specValues.push_back(constant.value);
    }
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = static_cast<u32>(specEntries.size());
    specInfo.pMapEntries = specEntries.data();
    specInfo.dataSize = specValues.size() * sizeof(u32);
    specInfo.pData = specValues.data();
    const auto specializationFor = [&](ShaderStages stage) -> const VkSpecializationInfo* {
        if (specEntries.empty() || !HasAny(desc.specializedStages, stage)) return nullptr;
        return &specInfo;
    };

    std::array<VkPipelineShaderStageCreateInfo, 5> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = modules[0];
    stages[0].pName = "main";
    stages[0].pSpecializationInfo = specializationFor(ShaderStages::RayGen);

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[1].module = modules[1];
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = specializationFor(ShaderStages::ClosestHit);

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = modules[2];
    stages[2].pName = "main";
    stages[2].pSpecializationInfo = specializationFor(ShaderStages::Miss);

    // The shadow miss is never specialized.
    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[3].module = modules[3];
    stages[3].pName = "main";

    stages[4].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[4].module = modules[4];
    stages[4].pName = "main";
    stages[4].pSpecializationInfo = specializationFor(ShaderStages::AnyHit);

    // Four groups. The any-hit joins the one triangles hit group rather than
    // forming its own, so the shader binding table has one hit record and two
    // miss records, and every TraceRay's sbtRecordOffset and missIndex keep
    // their meaning.
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[1].generalShader = VK_SHADER_UNUSED_KHR;
    groups[1].closestHitShader = 1;
    groups[1].anyHitShader = 4;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[3].generalShader = 3;
    groups[3].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = static_cast<u32>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<u32>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    pipelineInfo.layout = static_cast<const VulkanKernelLayout*>(desc.layout)->Native();

    const VkPipelineCache cache =
        desc.cache != nullptr ? static_cast<const VulkanPipelineCache*>(desc.cache)->Native() : VK_NULL_HANDLE;

    QL_LOG_INFO("  Calling vkCreateRayTracingPipelinesKHR...");
    QL_LOG_INFO("  Using pipeline cache: {}", cache != VK_NULL_HANDLE ? "YES" : "NO");
    Log::Flush();

    const auto startTime = std::chrono::high_resolution_clock::now();

    std::unique_ptr<VulkanTraceKernel> kernel(new VulkanTraceKernel());
    kernel->name_ = desc.name.empty() ? desc.rayGen->name : String(desc.name);
    const VkResult result =
        rt.createPipelines(device, VK_NULL_HANDLE, cache, 1, &pipelineInfo, nullptr, &kernel->pipeline_);

    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    QL_LOG_INFO("  vkCreateRayTracingPipelinesKHR returned: {} (took {} ms)", static_cast<int>(result),
                duration.count());

    for (VkShaderModule module : modules) vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS) {
        kernel->pipeline_ = VK_NULL_HANDLE;
        return Fail<Out>("Failed to create ray tracing pipeline variant");
    }
    kernel->device_ = device;  // from here on the destructor destroys the pipeline

    // Build the SBT
    const u32 handleSize = properties.shaderGroupHandleSize;
    const u32 handleAlignment = properties.shaderGroupHandleAlignment;
    const u32 baseAlignment = properties.shaderGroupBaseAlignment;
    const u32 handleSizeAligned = AlignedSize(handleSize, handleAlignment);

    const u32 raygenSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missStride = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missCount = 2;
    const u32 missSize = missCount * missStride;
    const u32 hitSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 sbtSize = raygenSize + missSize + hitSize;

    constexpr u32 groupCount = 4;
    Vector<u8> handleData(groupCount * handleSize);
    if (const VkResult handlesResult = rt.getGroupHandles(device, kernel->pipeline_, 0, groupCount,
                                                          handleData.size(), handleData.data());
        handlesResult != VK_SUCCESS) {
        // Carrying on would build the SBT from the zero-initialised handleData; the
        // kernel's destructor destroys the pipeline.
        return Fail<Out>("Failed to get ray tracing shader group handles (VkResult " +
                         std::to_string(static_cast<int>(handlesResult)) + ")");
    }
    LogPipelineStackSizes(rt, kernel->pipeline_, kernel->name_, desc.maxRecursionDepth);

    Vector<u8> sbtData(sbtSize, 0);
    std::memcpy(sbtData.data(), handleData.data() + handleSize * 0, handleSize);
    std::memcpy(sbtData.data() + raygenSize, handleData.data() + handleSize * 2, handleSize);
    std::memcpy(sbtData.data() + raygenSize + missStride, handleData.data() + handleSize * 3, handleSize);
    std::memcpy(sbtData.data() + raygenSize + missSize, handleData.data() + handleSize * 1, handleSize);

    try {
        kernel->sbt_ = std::make_unique<GpuBuffer>(
            allocator, sbtSize,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, baseAlignment);
    } catch (const std::exception& e) {
        return Fail<Out>(e.what());
    }
    kernel->sbt_->Upload(sbtData.data(), sbtSize);

    const VkDeviceAddress sbtAddress = kernel->sbt_->GetDeviceAddress(device);
    kernel->raygenRegion_.deviceAddress = sbtAddress;
    kernel->raygenRegion_.stride = raygenSize;
    kernel->raygenRegion_.size = raygenSize;
    kernel->missRegion_.deviceAddress = sbtAddress + raygenSize;
    kernel->missRegion_.stride = missStride;
    kernel->missRegion_.size = missSize;
    kernel->hitRegion_.deviceAddress = sbtAddress + raygenSize + missSize;
    kernel->hitRegion_.stride = hitSize;
    kernel->hitRegion_.size = hitSize;
    kernel->callableRegion_ = {};

    kernel->traceRays_ = rt.cmdTraceRays;
    kernel->layout_ = desc.layout;

    QL_LOG_INFO("  Pipeline variant created (SBT: {} bytes)", sbtSize);
    return Out(std::move(kernel));
}

VulkanTraceKernel::~VulkanTraceKernel() {
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
}

void VulkanTraceKernel::TraceRays(VkCommandBuffer cmd, u32 width, u32 height, u32 depth) const {
    traceRays_(cmd, &raygenRegion_, &missRegion_, &hitRegion_, &callableRegion_, width, height, depth);
}

}  // namespace quantiloom::gpu::vulkan
