/**
 * @file GpuSpectralReconstructor.cpp
 * @brief Implementation of GPU-accelerated spectral cube reconstruction
 *
 * @author blitzcolo
 */

#include "hs_core/GpuSpectralReconstructor.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/CommandHelper.hpp"
#include "core/Log.hpp"

#include <chrono>
#include <fstream>
#include <vector>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Status String Conversion
// ============================================================================

const char* GpuReconstructorStatusToString(GpuReconstructorStatus status) {
    switch (status) {
        case GpuReconstructorStatus::Success:            return "Success";
        case GpuReconstructorStatus::NotInitialized:     return "Not initialized";
        case GpuReconstructorStatus::InvalidInput:       return "Invalid input";
        case GpuReconstructorStatus::ShaderCompileFailed: return "Shader compile failed";
        case GpuReconstructorStatus::PipelineCreateFailed: return "Pipeline create failed";
        case GpuReconstructorStatus::BufferAllocFailed:  return "Buffer alloc failed";
        case GpuReconstructorStatus::DispatchFailed:     return "Dispatch failed";
        case GpuReconstructorStatus::ReadbackFailed:     return "Readback failed";
        case GpuReconstructorStatus::OutOfMemory:        return "Out of memory";
        case GpuReconstructorStatus::Timeout:            return "Timeout";
        default:                                         return "Unknown status";
    }
}

// ============================================================================
// Push Constants Structure (must match shader)
// ============================================================================

struct ReconstructPushConstants {
    u32 width;
    u32 height;
    u32 srcBands;
    u32 dstBands;
    u32 interpolationMode;
    u32 _pad0;
    u32 _pad1;
    u32 _pad2;
};

// ============================================================================
// GpuSpectralReconstructor::Impl
// ============================================================================

struct GpuSpectralReconstructor::Impl {
    VulkanContext& context;

    // Compute pipeline resources
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    bool initialized = false;

    // Statistics
    f64 lastReconstructionTime = 0.0;
    u64 lastMemoryUsage = 0;
    u64 lastPixelCount = 0;

    explicit Impl(VulkanContext& ctx) : context(ctx) {}

    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        VkDevice device = context.GetDevice();

        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
            shaderModule = VK_NULL_HANDLE;
        }

        initialized = false;
    }

    /**
     * @brief Load SPIR-V shader from file
     */
    Vector<u32> LoadShaderFile(const String& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to open shader: {}", path);
            return {};
        }

        usize fileSize = static_cast<usize>(file.tellg());
        if (fileSize == 0 || fileSize % 4 != 0) {
            LOG_ERROR("GpuSpectralReconstructor: Invalid SPIR-V file size: {}", fileSize);
            return {};
        }

        Vector<u32> code(fileSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), fileSize);

        return code;
    }

    /**
     * @brief Create shader module from SPIR-V code
     */
    VkShaderModule CreateShaderModule(const Vector<u32>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(u32);
        createInfo.pCode = code.data();

        VkShaderModule module;
        if (vkCreateShaderModule(context.GetDevice(), &createInfo, nullptr, &module) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return module;
    }

    /**
     * @brief Create descriptor set layout for compute shader bindings
     */
    bool CreateDescriptorSetLayout() {
        // Bindings:
        // 0: sparseCube (storage buffer, readonly)
        // 1: fullCube (storage buffer, writeonly)
        // 2: srcWavelengths (storage buffer, readonly)
        // 3: dstWavelengths (storage buffer, readonly)
        // 4: mapping (storage buffer, readonly) - optional

        Vector<VkDescriptorSetLayoutBinding> bindings(5);

        for (u32 i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(context.GetDevice(), &layoutInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to create descriptor set layout");
            return false;
        }

        return true;
    }

    /**
     * @brief Create pipeline layout with push constants
     */
    bool CreatePipelineLayout() {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ReconstructPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(context.GetDevice(), &layoutInfo, nullptr,
                                   &pipelineLayout) != VK_SUCCESS) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to create pipeline layout");
            return false;
        }

        return true;
    }

    /**
     * @brief Create descriptor pool for allocating descriptor sets
     */
    bool CreateDescriptorPool() {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 5;  // 5 bindings

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(context.GetDevice(), &poolInfo, nullptr,
                                   &descriptorPool) != VK_SUCCESS) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to create descriptor pool");
            return false;
        }

        return true;
    }

    /**
     * @brief Create compute pipeline
     */
    bool CreateComputePipeline() {
        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;

        if (vkCreateComputePipelines(context.GetDevice(), VK_NULL_HANDLE, 1,
                                     &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to create compute pipeline");
            return false;
        }

        return true;
    }

    /**
     * @brief Convert InterpolationMethod to shader mode value
     */
    u32 GetInterpolationMode(InterpolationMethod method) {
        switch (method) {
            case InterpolationMethod::Linear:     return 0;
            case InterpolationMethod::CatmullRom: return 1;
            case InterpolationMethod::Akima:      return 2;
            default:                              return 1;  // Default to Catmull-Rom
        }
    }
};

// ============================================================================
// GpuSpectralReconstructor - Public Interface
// ============================================================================

GpuSpectralReconstructor::GpuSpectralReconstructor(VulkanContext& context)
    : m_impl(std::make_unique<Impl>(context)) {
    LOG_DEBUG("GpuSpectralReconstructor created");
}

GpuSpectralReconstructor::~GpuSpectralReconstructor() = default;

GpuSpectralReconstructor::GpuSpectralReconstructor(GpuSpectralReconstructor&&) noexcept = default;
GpuSpectralReconstructor& GpuSpectralReconstructor::operator=(GpuSpectralReconstructor&&) noexcept = default;

GpuReconstructorStatus GpuSpectralReconstructor::Initialize() {
    if (m_impl->initialized) {
        return GpuReconstructorStatus::Success;
    }

    LOG_INFO("GpuSpectralReconstructor: Initializing...");

    // Load shader SPIR-V
    // Try multiple paths for shader location
    Vector<String> shaderPaths = {
        "shaders/spectral_reconstruct.spv",
        "../shaders/spectral_reconstruct.spv",
        "src/shaders/spectral_reconstruct.spv",
        "../src/shaders/spectral_reconstruct.spv"
    };

    Vector<u32> shaderCode;
    for (const auto& path : shaderPaths) {
        shaderCode = m_impl->LoadShaderFile(path);
        if (!shaderCode.empty()) {
            LOG_DEBUG("GpuSpectralReconstructor: Loaded shader from {}", path);
            break;
        }
    }

    if (shaderCode.empty()) {
        LOG_ERROR("GpuSpectralReconstructor: Could not find spectral_reconstruct.spv");
        return GpuReconstructorStatus::ShaderCompileFailed;
    }

    // Create shader module
    m_impl->shaderModule = m_impl->CreateShaderModule(shaderCode);
    if (m_impl->shaderModule == VK_NULL_HANDLE) {
        return GpuReconstructorStatus::ShaderCompileFailed;
    }

    // Create descriptor set layout
    if (!m_impl->CreateDescriptorSetLayout()) {
        m_impl->Cleanup();
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    // Create pipeline layout
    if (!m_impl->CreatePipelineLayout()) {
        m_impl->Cleanup();
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    // Create descriptor pool
    if (!m_impl->CreateDescriptorPool()) {
        m_impl->Cleanup();
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    // Create compute pipeline
    if (!m_impl->CreateComputePipeline()) {
        m_impl->Cleanup();
        return GpuReconstructorStatus::PipelineCreateFailed;
    }

    m_impl->initialized = true;
    LOG_INFO("GpuSpectralReconstructor: Initialization complete");

    return GpuReconstructorStatus::Success;
}

bool GpuSpectralReconstructor::IsReady() const {
    return m_impl->initialized;
}

void GpuSpectralReconstructor::Shutdown() {
    m_impl->Cleanup();
}

std::pair<GpuReconstructorStatus, SpectralCube> GpuSpectralReconstructor::Reconstruct(
    const SpectralCube& sparseCube,
    const AdaptiveGridInfo& gridInfo,
    const HyperspectralConfig& targetConfig,
    const GpuReconstructorConfig& config
) {
    // Generate target wavelength list from config
    Vector<f32> targetWavelengths;
    u32 numBands = targetConfig.GetNumBands();
    targetWavelengths.reserve(numBands);

    for (u32 i = 0; i < numBands; ++i) {
        targetWavelengths.push_back(targetConfig.GetWavelength(i));
    }

    return ReconstructCustom(sparseCube, targetWavelengths, config);
}

std::pair<GpuReconstructorStatus, SpectralCube> GpuSpectralReconstructor::ReconstructCustom(
    const SpectralCube& sparseCube,
    const Vector<f32>& targetWavelengths,
    const GpuReconstructorConfig& config
) {
    using Clock = std::chrono::high_resolution_clock;
    auto startTime = Clock::now();

    // Validate input
    if (!sparseCube.IsValid()) {
        LOG_ERROR("GpuSpectralReconstructor: Invalid sparse cube");
        return {GpuReconstructorStatus::InvalidInput, SpectralCube{}};
    }

    if (targetWavelengths.empty()) {
        LOG_ERROR("GpuSpectralReconstructor: Empty target wavelengths");
        return {GpuReconstructorStatus::InvalidInput, SpectralCube{}};
    }

    if (!m_impl->initialized) {
        auto status = Initialize();
        if (status != GpuReconstructorStatus::Success) {
            return {status, SpectralCube{}};
        }
    }

    u32 width = sparseCube.width;
    u32 height = sparseCube.height;
    u32 srcBands = sparseCube.nbands;
    u32 dstBands = static_cast<u32>(targetWavelengths.size());

    if (config.verbose) {
        LOG_INFO("GpuSpectralReconstructor: Reconstructing {}x{}x{} from {} source bands",
                 width, height, dstBands, srcBands);
    }

    // Calculate buffer sizes
    VkDeviceSize srcCubeSize = width * height * srcBands * sizeof(f32);
    VkDeviceSize dstCubeSize = width * height * dstBands * sizeof(f32);
    VkDeviceSize srcWlSize = srcBands * sizeof(f32);
    VkDeviceSize dstWlSize = dstBands * sizeof(f32);
    VkDeviceSize mappingSize = dstBands * 6 * sizeof(u32);  // Not used in current shader

    m_impl->lastMemoryUsage = srcCubeSize + dstCubeSize + srcWlSize + dstWlSize + mappingSize;

    try {
        // Create GPU buffers
        GpuBuffer srcCubeBuffer(m_impl->context.GetAllocator(), srcCubeSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        GpuBuffer dstCubeBuffer(m_impl->context.GetAllocator(), dstCubeSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        GpuBuffer srcWlBuffer(m_impl->context.GetAllocator(), srcWlSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        GpuBuffer dstWlBuffer(m_impl->context.GetAllocator(), dstWlSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Dummy mapping buffer (required by shader but not used in simple mode)
        GpuBuffer mappingBuffer(m_impl->context.GetAllocator(), sizeof(u32) * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Staging buffers for upload
        GpuBuffer srcCubeStaging(m_impl->context.GetAllocator(), srcCubeSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY);

        GpuBuffer srcWlStaging(m_impl->context.GetAllocator(), srcWlSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY);

        GpuBuffer dstWlStaging(m_impl->context.GetAllocator(), dstWlSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY);

        // Upload source cube data
        {
            f32* mapped = static_cast<f32*>(srcCubeStaging.Map());
            std::memcpy(mapped, sparseCube.data.data(), srcCubeSize);
            srcCubeStaging.Unmap();
        }

        // Upload source wavelengths
        {
            f32* mapped = static_cast<f32*>(srcWlStaging.Map());
            std::memcpy(mapped, sparseCube.wavelengths.data(), srcWlSize);
            srcWlStaging.Unmap();
        }

        // Upload target wavelengths
        {
            f32* mapped = static_cast<f32*>(dstWlStaging.Map());
            std::memcpy(mapped, targetWavelengths.data(), dstWlSize);
            dstWlStaging.Unmap();
        }

        // Copy staging to GPU buffers
        CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};

            copyRegion.size = srcCubeSize;
            vkCmdCopyBuffer(cmd, srcCubeStaging.GetHandle(), srcCubeBuffer.GetHandle(), 1, &copyRegion);

            copyRegion.size = srcWlSize;
            vkCmdCopyBuffer(cmd, srcWlStaging.GetHandle(), srcWlBuffer.GetHandle(), 1, &copyRegion);

            copyRegion.size = dstWlSize;
            vkCmdCopyBuffer(cmd, dstWlStaging.GetHandle(), dstWlBuffer.GetHandle(), 1, &copyRegion);

            // Memory barrier before compute
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        });

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_impl->descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_impl->descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(m_impl->context.GetDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
            LOG_ERROR("GpuSpectralReconstructor: Failed to allocate descriptor set");
            return {GpuReconstructorStatus::BufferAllocFailed, SpectralCube{}};
        }

        // Update descriptor set
        Vector<VkDescriptorBufferInfo> bufferInfos(5);
        bufferInfos[0] = {srcCubeBuffer.GetHandle(), 0, srcCubeSize};
        bufferInfos[1] = {dstCubeBuffer.GetHandle(), 0, dstCubeSize};
        bufferInfos[2] = {srcWlBuffer.GetHandle(), 0, srcWlSize};
        bufferInfos[3] = {dstWlBuffer.GetHandle(), 0, dstWlSize};
        bufferInfos[4] = {mappingBuffer.GetHandle(), 0, sizeof(u32) * 4};

        Vector<VkWriteDescriptorSet> writes(5);
        for (u32 i = 0; i < 5; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bufferInfos[i];
        }

        vkUpdateDescriptorSets(m_impl->context.GetDevice(),
            static_cast<u32>(writes.size()), writes.data(), 0, nullptr);

        // Setup push constants
        ReconstructPushConstants pushConstants{};
        pushConstants.width = width;
        pushConstants.height = height;
        pushConstants.srcBands = srcBands;
        pushConstants.dstBands = dstBands;
        pushConstants.interpolationMode = m_impl->GetInterpolationMode(config.method);

        // Dispatch compute shader
        CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                m_impl->pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_impl->pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

            // Calculate workgroup count
            u32 groupCountX = (width + config.workgroupSizeX - 1) / config.workgroupSizeX;
            u32 groupCountY = (height + config.workgroupSizeY - 1) / config.workgroupSizeY;

            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

            // Memory barrier after compute
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        });

        // Read back results
        GpuBuffer dstCubeStaging(m_impl->context.GetAllocator(), dstCubeSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY);

        CommandHelper::ExecuteImmediate(m_impl->context, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = dstCubeSize;
            vkCmdCopyBuffer(cmd, dstCubeBuffer.GetHandle(), dstCubeStaging.GetHandle(), 1, &copyRegion);
        });

        // Create result cube
        SpectralCube result(width, height, dstBands,
            targetWavelengths.front(), targetWavelengths.back());
        result.wavelengths = targetWavelengths;

        // Copy data from staging buffer
        {
            f32* mapped = static_cast<f32*>(dstCubeStaging.Map());
            std::memcpy(result.data.data(), mapped, dstCubeSize);
            dstCubeStaging.Unmap();
        }

        // Add metadata
        result.metadata = sparseCube.metadata;
        result.metadata["reconstruction_method"] =
            (config.method == InterpolationMethod::Linear) ? "gpu_linear" :
            (config.method == InterpolationMethod::CatmullRom) ? "gpu_catmull_rom" : "gpu_akima";

        // Reset descriptor pool for next use
        vkResetDescriptorPool(m_impl->context.GetDevice(), m_impl->descriptorPool, 0);

        // Calculate statistics
        auto endTime = Clock::now();
        m_impl->lastReconstructionTime = std::chrono::duration<f64>(endTime - startTime).count();
        m_impl->lastPixelCount = width * height;

        if (config.verbose) {
            LOG_INFO("GpuSpectralReconstructor: Complete in {:.3f}s ({:.2f} Mpixels/s)",
                     m_impl->lastReconstructionTime,
                     (m_impl->lastPixelCount / 1e6) / m_impl->lastReconstructionTime);
        }

        return {GpuReconstructorStatus::Success, std::move(result)};
    }
    catch (const std::exception& e) {
        LOG_ERROR("GpuSpectralReconstructor: Exception: {}", e.what());
        return {GpuReconstructorStatus::OutOfMemory, SpectralCube{}};
    }
}

f64 GpuSpectralReconstructor::GetLastReconstructionTime() const {
    return m_impl->lastReconstructionTime;
}

u64 GpuSpectralReconstructor::GetLastMemoryUsage() const {
    return m_impl->lastMemoryUsage;
}

f64 GpuSpectralReconstructor::GetPixelsPerSecond() const {
    if (m_impl->lastReconstructionTime > 0) {
        return static_cast<f64>(m_impl->lastPixelCount) / m_impl->lastReconstructionTime;
    }
    return 0.0;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool IsGpuReconstructionSupported(VulkanContext& context) {
    // Check for compute shader support (always available in Vulkan)
    // Check for sufficient buffer size limits
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(context.GetPhysicalDevice(), &props);

    // Need at least 256MB for typical hyperspectral cubes
    const VkDeviceSize minBufferSize = 256 * 1024 * 1024;

    return props.limits.maxStorageBufferRange >= minBufferSize;
}

u64 EstimateReconstructionMemory(const SpectralCube& sparseCube, u32 targetBands) {
    u64 srcSize = sparseCube.width * sparseCube.height * sparseCube.nbands * sizeof(f32);
    u64 dstSize = sparseCube.width * sparseCube.height * targetBands * sizeof(f32);
    u64 wlSize = (sparseCube.nbands + targetBands) * sizeof(f32);
    u64 stagingSize = srcSize + dstSize;  // For upload/download

    return srcSize + dstSize + wlSize + stagingSize;
}

} // namespace quantiloom
