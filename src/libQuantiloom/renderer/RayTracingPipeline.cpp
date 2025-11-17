#include "RayTracingPipeline.hpp"
#include "core/Log.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Constructor / Destructor
// ============================================================================

RayTracingPipeline::RayTracingPipeline(
    VulkanContext& context,
    const std::string& raygenPath,
    const std::string& closestHitPath,
    const std::string& missPath)
    : m_context(context)
    , m_raygenPath(raygenPath)
    , m_closestHitPath(closestHitPath)
    , m_missPath(missPath)
{
    QL_LOG_INFO("Creating Ray Tracing pipeline...");

    // Cache RT properties
    m_rtProperties = m_context.GetRayTracingProperties();

    // Create pipeline in order with exception safety
    try {
        CreateDescriptorSetLayout();
        CreatePipelineLayout();
        LoadShaders();
        CreatePipeline();
        CreateShaderBindingTable();

        QL_LOG_INFO("Ray Tracing pipeline created successfully");
    }
    catch (const std::exception& e) {
        // Clean up partially created resources before rethrowing
        VkDevice device = m_context.GetDevice();

        // Destroy shader modules if they were created
        for (auto module : m_shaderModules) {
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, module, nullptr);
            }
        }

        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, m_pipeline, nullptr);
        }

        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        }

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        }

        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
        }

        QL_LOG_ERROR("Failed to create Ray Tracing pipeline: {}", e.what());
        throw;  // Rethrow the exception
    }
}

RayTracingPipeline::~RayTracingPipeline() {
    VkDevice device = m_context.GetDevice();

    // Destroy in reverse order
    m_sbtBuffer.reset();

    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    }

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
    }

    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    }

    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
    }

    QL_LOG_INFO("Ray Tracing pipeline destroyed");
}

// ============================================================================
// Descriptor Set Layout
// ============================================================================

void RayTracingPipeline::CreateDescriptorSetLayout() {
    VkDevice device = m_context.GetDevice();

    // Define bindings (matches shader layout)
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    // Binding 0: Output image (RWTexture2D)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 1: Acceleration structure (TLAS)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 2: LUT buffer (StructuredBuffer)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes(3);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    result = vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    QL_LOG_INFO("  Descriptor set layout created");
}

// ============================================================================
// Pipeline Layout
// ============================================================================

void RayTracingPipeline::CreatePipelineLayout() {
    VkDevice device = m_context.GetDevice();

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 0;  // No push constants for M1

    VkResult result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    QL_LOG_INFO("  Pipeline layout created");
}

// ============================================================================
// Shader Loading
// ============================================================================

void RayTracingPipeline::LoadShaders() {
    VkDevice device = m_context.GetDevice();

    // Load all shaders
    auto raygenSpirv = LoadSPIRV(m_raygenPath);
    auto chitSpirv = LoadSPIRV(m_closestHitPath);
    auto missSpirv = LoadSPIRV(m_missPath);

    // Create shader modules (will be destroyed after pipeline creation)
    m_shaderModules.resize(3);
    m_shaderModules[0] = CreateShaderModule(raygenSpirv);
    m_shaderModules[1] = CreateShaderModule(chitSpirv);
    m_shaderModules[2] = CreateShaderModule(missSpirv);

    QL_LOG_INFO("  Shaders loaded: {} / {} / {}", m_raygenPath, m_closestHitPath, m_missPath);
}

std::vector<u32> RayTracingPipeline::LoadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<u32> buffer(fileSize / sizeof(u32));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
    file.close();

    return buffer;
}

VkShaderModule RayTracingPipeline::CreateShaderModule(const std::vector<u32>& spirv) {
    VkDevice device = m_context.GetDevice();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(u32);
    createInfo.pCode = spirv.data();

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

// ============================================================================
// Pipeline Creation
// ============================================================================

void RayTracingPipeline::CreatePipeline() {
    VkDevice device = m_context.GetDevice();

    // Define shader stages
    std::vector<VkPipelineShaderStageCreateInfo> stages(3);

    // Raygen
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = m_shaderModules[0];
    stages[0].pName = "main";

    // Closest Hit
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[1].module = m_shaderModules[1];
    stages[1].pName = "main";

    // Miss
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = m_shaderModules[2];
    stages[2].pName = "main";

    // Define shader groups
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(3);

    // Group 0: Raygen
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 1: Hit group (closest hit only)
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[1].generalShader = VK_SHADER_UNUSED_KHR;
    groups[1].closestHitShader = 1;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 2: Miss
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Create pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = static_cast<u32>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<u32>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;  // No recursion for M1
    pipelineInfo.layout = m_pipelineLayout;

    // Get function pointer for vkCreateRayTracingPipelinesKHR
    auto vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");

    if (!vkCreateRayTracingPipelinesKHR) {
        throw std::runtime_error("Failed to load vkCreateRayTracingPipelinesKHR");
    }

    VkResult result = vkCreateRayTracingPipelinesKHR(
        device,
        VK_NULL_HANDLE,  // No pipeline cache for M1
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &m_pipeline
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline");
    }

    // Destroy shader modules (no longer needed)
    for (auto module : m_shaderModules) {
        vkDestroyShaderModule(device, module, nullptr);
    }
    m_shaderModules.clear();

    QL_LOG_INFO("  Ray Tracing pipeline created");
}

// ============================================================================
// Shader Binding Table (SBT)
// ============================================================================

void RayTracingPipeline::CreateShaderBindingTable() {
    VkDevice device = m_context.GetDevice();

    // Get function pointer for vkGetRayTracingShaderGroupHandlesKHR
    auto vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
        vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");

    if (!vkGetRayTracingShaderGroupHandlesKHR) {
        throw std::runtime_error("Failed to load vkGetRayTracingShaderGroupHandlesKHR");
    }

    // Calculate sizes and offsets
    const u32 handleSize = m_rtProperties.shaderGroupHandleSize;
    const u32 handleAlignment = m_rtProperties.shaderGroupHandleAlignment;
    const u32 baseAlignment = m_rtProperties.shaderGroupBaseAlignment;

    const u32 handleSizeAligned = AlignedSize(handleSize, handleAlignment);

    // SBT layout: [Raygen] [Miss] [Hit]
    const u32 raygenSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 hitSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 sbtSize = raygenSize + missSize + hitSize;

    // Get shader group handles
    const u32 groupCount = 3;
    std::vector<u8> handleData(groupCount * handleSize);

    VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
        device,
        m_pipeline,
        0,
        groupCount,
        handleData.size(),
        handleData.data()
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get shader group handles");
    }

    // Build SBT buffer data
    std::vector<u8> sbtData(sbtSize, 0);

    // Copy handles with alignment
    std::memcpy(sbtData.data(), handleData.data(), handleSize);  // Raygen
    std::memcpy(sbtData.data() + raygenSize, handleData.data() + handleSize * 2, handleSize);  // Miss
    std::memcpy(sbtData.data() + raygenSize + missSize, handleData.data() + handleSize * 1, handleSize);  // Hit

    // Create SBT buffer
    m_sbtBuffer = std::make_unique<GpuBuffer>(
        m_context.GetAllocator(),
        sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    m_sbtBuffer->Upload(sbtData.data(), sbtSize);

    // Define SBT regions
    VkDeviceAddress sbtAddress = m_sbtBuffer->GetDeviceAddress(device);

    m_raygenRegion.deviceAddress = sbtAddress;
    m_raygenRegion.stride = raygenSize;
    m_raygenRegion.size = raygenSize;

    m_missRegion.deviceAddress = sbtAddress + raygenSize;
    m_missRegion.stride = missSize;
    m_missRegion.size = missSize;

    m_hitRegion.deviceAddress = sbtAddress + raygenSize + missSize;
    m_hitRegion.stride = hitSize;
    m_hitRegion.size = hitSize;

    m_callableRegion = {};  // No callable shaders

    QL_LOG_INFO("  Shader Binding Table created (size: {} bytes)", sbtSize);
}

u32 RayTracingPipeline::AlignedSize(u32 size, u32 alignment) const {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// Descriptor Binding
// ============================================================================

void RayTracingPipeline::BindOutputImage(const GpuImage& image) {
    VkDevice device = m_context.GetDevice();

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = image.GetView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindAccelerationStructure(VkAccelerationStructureKHR tlas) {
    VkDevice device = m_context.GetDevice();

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = &asInfo;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 1;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    write.descriptorCount = 1;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindLUTBuffer(const GpuBuffer& buffer) {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 2;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::UpdateDescriptorSets() {
    // Bindings are updated immediately in Bind* methods
    // This is a no-op for M1, but kept for API consistency
}

// ============================================================================
// Rendering
// ============================================================================

void RayTracingPipeline::TraceRays(VkCommandBuffer cmd, u32 width, u32 height) {
    // Get function pointer for vkCmdTraceRaysKHR
    auto vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)
        vkGetDeviceProcAddr(m_context.GetDevice(), "vkCmdTraceRaysKHR");

    if (!vkCmdTraceRaysKHR) {
        throw std::runtime_error("Failed to load vkCmdTraceRaysKHR");
    }

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        m_pipelineLayout,
        0,
        1,
        &m_descriptorSet,
        0,
        nullptr
    );

    // Trace rays
    vkCmdTraceRaysKHR(
        cmd,
        &m_raygenRegion,
        &m_missRegion,
        &m_hitRegion,
        &m_callableRegion,
        width,
        height,
        1  // depth
    );
}

} // namespace quantiloom
