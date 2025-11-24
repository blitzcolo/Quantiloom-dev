#include "RayTracingPipeline.hpp"
#include "core/Log.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <filesystem>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <limits.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif

namespace quantiloom {

// ============================================================================
// Helper: Get executable directory
// ============================================================================

static std::filesystem::path GetExecutableDirectory() {
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path();
#elif defined(__linux__)
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#else
    return std::filesystem::current_path();
#endif
}

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
    // NOTE: Using fixed array size for textures (max 1024) for simplicity
    // Can be made dynamic via VkDescriptorSetVariableDescriptorCountAllocateInfo in M2+
    constexpr u32 MAX_TEXTURES = 1024;

    std::vector<VkDescriptorSetLayoutBinding> bindings(8);

    // Binding 0: Output image (RWTexture2D)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: Acceleration structure (TLAS)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[1].pImmutableSamplers = nullptr;

    // Binding 2: LUT buffer (StructuredBuffer)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    bindings[2].pImmutableSamplers = nullptr;

    // Binding 3: Vertex buffer (StructuredBuffer<float3>)
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[3].pImmutableSamplers = nullptr;

    // Binding 4: Index buffer (StructuredBuffer<uint>)
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[4].pImmutableSamplers = nullptr;

    // Binding 5: Material buffer (StructuredBuffer<MaterialData>)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[5].pImmutableSamplers = nullptr;

    // Binding 6: Texture array (Texture2D[])
    // Uses VK_EXT_descriptor_indexing for runtime array indexing
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[6].descriptorCount = MAX_TEXTURES;
    bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[6].pImmutableSamplers = nullptr;

    // Binding 7: Sampler array (SamplerState[])
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[7].descriptorCount = MAX_TEXTURES;
    bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[7].pImmutableSamplers = nullptr;

    // Enable descriptor indexing flags for texture arrays
    // This allows runtime indexing and partially bound descriptors
    std::vector<VkDescriptorBindingFlags> bindingFlags(8, 0);
    bindingFlags[6] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;  // Not all textures need to be bound
    bindingFlags[7] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;  // Not all samplers need to be bound

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<u32>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes(5);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 4;  // LUT + vertex + index + material buffers
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[3].descriptorCount = MAX_TEXTURES;  // Texture array
    poolSizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[4].descriptorCount = MAX_TEXTURES;  // Sampler array

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

    // Push constant range for camera data
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(CameraData);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    VkResult result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    QL_LOG_INFO("  Pipeline layout created with push constants (camera data)");
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
    // Try multiple search paths
    std::vector<std::filesystem::path> searchPaths = {
        path,  // Original path (relative to CWD or absolute)
        GetExecutableDirectory() / path,  // Relative to executable directory
    };

    std::ifstream file;
    std::filesystem::path foundPath;

    for (const auto& tryPath : searchPaths) {
        file.open(tryPath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            foundPath = tryPath;
            break;
        }
    }

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path +
            " (searched in CWD and executable directory)");
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

void RayTracingPipeline::BindGeometryBuffers(const GpuBuffer& vertexBuffer, const GpuBuffer& indexBuffer) {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo vertexInfo{};
    vertexInfo.buffer = vertexBuffer.GetHandle();
    vertexInfo.offset = 0;
    vertexInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo indexInfo{};
    indexInfo.buffer = indexBuffer.GetHandle();
    indexInfo.offset = 0;
    indexInfo.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> writes(2);

    // Binding 3: Vertex buffer
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 3;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &vertexInfo;

    // Binding 4: Index buffer
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 4;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &indexInfo;

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
}

void RayTracingPipeline::BindMaterialBuffer(const GpuBuffer& buffer) {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 5;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindTextures(const std::vector<VkImageView>& imageViews,
                                       const std::vector<VkSampler>& samplers) {
    VkDevice device = m_context.GetDevice();

    if (imageViews.size() != samplers.size()) {
        QL_LOG_ERROR("Texture binding failed: imageViews.size() ({}) != samplers.size() ({})",
                     imageViews.size(), samplers.size());
        throw std::runtime_error("Mismatched texture and sampler array sizes");
    }

    if (imageViews.empty()) {
        QL_LOG_WARN("No textures to bind (TextureManager should provide at least a dummy texture)");
        return;
    }

    u32 textureCount = static_cast<u32>(imageViews.size());
    QL_LOG_INFO("Binding {} textures to descriptor set", textureCount);

    // Build descriptor image info array for textures
    std::vector<VkDescriptorImageInfo> imageInfos(textureCount);
    for (u32 i = 0; i < textureCount; ++i) {
        imageInfos[i].imageView = imageViews[i];
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[i].sampler = VK_NULL_HANDLE;  // Sampler is separate
    }

    // Build descriptor image info array for samplers
    std::vector<VkDescriptorImageInfo> samplerInfos(textureCount);
    for (u32 i = 0; i < textureCount; ++i) {
        samplerInfos[i].sampler = samplers[i];
        samplerInfos[i].imageView = VK_NULL_HANDLE;  // Image view is separate
        samplerInfos[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Not used for samplers
    }

    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes(2);

    // Binding 6: Texture array (SAMPLED_IMAGE)
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 6;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].descriptorCount = textureCount;
    writes[0].pImageInfo = imageInfos.data();

    // Binding 7: Sampler array
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 7;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].descriptorCount = textureCount;
    writes[1].pImageInfo = samplerInfos.data();

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
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

    // Push camera constants
    vkCmdPushConstants(
        cmd,
        m_pipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        0,
        sizeof(CameraData),
        &m_cameraData
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

    // CRITICAL: Add memory barrier after ray tracing to ensure output image writes are visible
    // Without this, subsequent readback may read stale/incomplete data, or GPU may hang
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );
}

void RayTracingPipeline::SetCameraData(const CameraData& cameraData) {
    m_cameraData = cameraData;
}

} // namespace quantiloom
