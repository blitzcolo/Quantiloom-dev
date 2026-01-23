#include "RayTracingPipeline.hpp"
#include "core/Log.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <filesystem>
#include <chrono>

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
// Helper: Get Maximum Texture Count
// ============================================================================
// Dynamic texture limit based on device capabilities
// Returns 1024 if descriptor indexing available, 32 otherwise
// ============================================================================

static u32 GetMaxTexturesForDevice(const VulkanContext& ctx) {
    return ctx.GetCapabilities().hasDescriptorIndexing ? 1024 : 32;
}

// ============================================================================
// Pipeline Cache Static Methods
// ============================================================================

VkPipelineCache RayTracingPipeline::LoadPipelineCache(VulkanContext& context, const std::string& cachePath) {
    VkDevice device = context.GetDevice();

    // Try to load existing cache from disk
    std::vector<char> cacheData;
    std::ifstream file(cachePath, std::ios::binary | std::ios::ate);

    if (file.is_open()) {
        size_t fileSize = static_cast<size_t>(file.tellg());
        cacheData.resize(fileSize);
        file.seekg(0);
        file.read(cacheData.data(), static_cast<std::streamsize>(fileSize));
        file.close();
        QL_LOG_INFO("Loaded pipeline cache from disk: {} ({} bytes)", cachePath, fileSize);
    } else {
        QL_LOG_INFO("No existing pipeline cache found at: {}", cachePath);
    }

    VkPipelineCacheCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    createInfo.initialDataSize = cacheData.size();
    createInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    VkPipelineCache cache = VK_NULL_HANDLE;
    VkResult result = vkCreatePipelineCache(device, &createInfo, nullptr, &cache);

    if (result != VK_SUCCESS) {
        QL_LOG_WARN("Failed to create pipeline cache (VkResult={}), creating empty cache", static_cast<int>(result));

        // Try again without initial data (cache might be corrupted or incompatible)
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        result = vkCreatePipelineCache(device, &createInfo, nullptr, &cache);

        if (result != VK_SUCCESS) {
            QL_LOG_ERROR("Failed to create empty pipeline cache");
            return VK_NULL_HANDLE;
        }
    }

    return cache;
}

bool RayTracingPipeline::SavePipelineCache(VulkanContext& context, VkPipelineCache cache, const std::string& cachePath) {
    if (cache == VK_NULL_HANDLE) {
        return false;
    }

    VkDevice device = context.GetDevice();

    // Get cache data size
    size_t cacheSize = 0;
    VkResult result = vkGetPipelineCacheData(device, cache, &cacheSize, nullptr);
    if (result != VK_SUCCESS || cacheSize == 0) {
        QL_LOG_WARN("Failed to get pipeline cache size");
        return false;
    }

    // Get cache data
    std::vector<char> cacheData(cacheSize);
    result = vkGetPipelineCacheData(device, cache, &cacheSize, cacheData.data());
    if (result != VK_SUCCESS) {
        QL_LOG_WARN("Failed to get pipeline cache data");
        return false;
    }

    // Write to disk
    std::ofstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        QL_LOG_WARN("Failed to open pipeline cache file for writing: {}", cachePath);
        return false;
    }

    file.write(cacheData.data(), static_cast<std::streamsize>(cacheSize));
    file.close();

    QL_LOG_INFO("Saved pipeline cache to disk: {} ({} bytes)", cachePath, cacheSize);
    return true;
}

void RayTracingPipeline::DestroyPipelineCache(VulkanContext& context, VkPipelineCache cache) {
    if (cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(context.GetDevice(), cache, nullptr);
    }
}

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
    const std::string& missPath,
    VkPipelineCache pipelineCache)
    : m_context(context)
    , m_raygenPath(raygenPath)
    , m_closestHitPath(closestHitPath)
    , m_missPath(missPath)
    , m_pipelineCache(pipelineCache)
{
    QL_LOG_INFO("Creating Ray Tracing pipeline...");

    // Derive shadow miss shader path from miss shader path
    // e.g., "miss.rmiss.spv" -> "shadow_miss.rmiss.spv"
    // or "shaders/miss.spv" -> "shaders/shadow_miss.spv"
    std::filesystem::path missFsPath(missPath);
    std::filesystem::path shadowMissName = "shadow_" + missFsPath.filename().string();
    m_shadowMissPath = (missFsPath.parent_path() / shadowMissName).string();
    if (m_shadowMissPath.empty()) {
        m_shadowMissPath = shadowMissName.string();
    }

    // Cache RT properties
    m_rtProperties = m_context.GetRayTracingProperties();

    // Initialize dynamic texture limit based on device capabilities
    m_maxTextures = GetMaxTexturesForDevice(m_context);
    QL_LOG_INFO("Texture limit: {} textures", m_maxTextures);

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
        for (const auto module : m_shaderModules) {
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
    // NOTE: Texture array size dynamically adjusted based on device capabilities
    // 1024 if descriptor indexing available, 32 otherwise
    std::vector<VkDescriptorSetLayoutBinding> bindings(20);  // Added CIE CMF LUT (binding 19)

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
    // Array size dynamic: 1024 if descriptor indexing available, 32 otherwise
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[6].descriptorCount = m_maxTextures;
    bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[6].pImmutableSamplers = nullptr;

    // Binding 7: Sampler array (SamplerState[])
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[7].descriptorCount = m_maxTextures;
    bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[7].pImmutableSamplers = nullptr;

    // Binding 8: UV buffer (StructuredBuffer<float2>) - Optional
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[8].pImmutableSamplers = nullptr;

    // Binding 9: Tangent buffer (StructuredBuffer<float4>) - Optional
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[9].pImmutableSamplers = nullptr;

    // ========================================================================
    // IBL (Image-Based Lighting) Bindings
    // ========================================================================
    // These bindings provide environment map data for physically-based
    // specular reflections on metallic and rough surfaces
    // ========================================================================

    // Binding 10: Prefiltered environment cubemap (TextureCube<float4>)
    // Contains environment map convolved with GGX for different roughness levels (mipmap chain)
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[10].pImmutableSamplers = nullptr;

    // Binding 11: BRDF integration LUT texture (Texture2D<float2>)
    // 2D lookup table: (NdotV, roughness) -> (scale, bias) for split-sum approximation
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[11].pImmutableSamplers = nullptr;

    // Binding 12: IBL sampler (SamplerState)
    // Linear sampler for IBL texture sampling (shared by prefiltered env and BRDF LUT)
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[12].pImmutableSamplers = nullptr;

    // ========================================================================
    // Spectral Curves Buffer (Binding 13)
    // ========================================================================
    // Contains measured spectral reflectance curves for quantitative rendering
    // Each curve is a SpectralCurveGPU struct with uniform wavelength sampling
    // ========================================================================

    // Binding 13: Spectral curves buffer (StructuredBuffer<SpectralCurveGPU>)
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[13].pImmutableSamplers = nullptr;

    // ========================================================================
    // Complex Refractive Index Buffer (Binding 14)
    // ========================================================================
    // Contains measured complex refractive index (n, k) for physical Fresnel
    // Each entry is a ComplexRefractiveIndexGPU struct (528 bytes)
    // ========================================================================

    // Binding 14: Complex refractive index buffer (StructuredBuffer<ComplexRefractiveIndexGPU>)
    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[14].pImmutableSamplers = nullptr;

    // ========================================================================
    // Solar Spectral LUT Buffer (Binding 15)
    // ========================================================================
    // Contains ASTM G-173 solar irradiance curves for spectral rendering
    // Structure: SolarSpectralLUT (544 bytes = 2 × SpectralCurveGPU)
    // - sunIrradiance: Direct+circumsolar irradiance (W·m⁻²·nm⁻¹)
    // - skyIrradiance: Diffuse sky irradiance (W·m⁻²·nm⁻¹)
    // ========================================================================

    // Binding 15: Solar spectral LUT buffer (SolarSpectralLUT)
    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    bindings[15].pImmutableSamplers = nullptr;

    // ========================================================================
    // Normal Buffer (Binding 16)
    // ========================================================================
    // Contains per-vertex normal vectors for smooth shading interpolation
    // CRITICAL: Required for smooth shading. Without this, shaders compute
    // geometric normals (flat shading), causing visible triangle boundaries.
    // ========================================================================

    // Binding 16: Normal buffer (StructuredBuffer<float3>)
    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[16].pImmutableSamplers = nullptr;

    // ========================================================================
    // Binding 17: Atmospheric Parameters (StructuredBuffer<AtmosphericParams>)
    // ========================================================================
    // Physical parameters for Delta-Tracking volumetric atmospheric rendering
    // Provides Rayleigh + Mie scattering coefficients
    // Used in both miss shader (Delta-Tracking) and closest hit (transmittance)
    // ========================================================================

    // Binding 17: Atmospheric params buffer (StructuredBuffer<AtmosphericParams>)
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[17].descriptorCount = 1;
    bindings[17].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[17].pImmutableSamplers = nullptr;

    // ========================================================================
    // Binding 18: Instance Geometry Info (StructuredBuffer<InstanceGeometryInfo>)
    // ========================================================================
    // Per-TLAS-instance geometry offset information for multi-BLAS support
    // Each entry contains offsets into merged global geometry buffers
    // Indexed by InstanceIndex() in shader
    // ========================================================================

    // Binding 18: Instance geometry info buffer (StructuredBuffer<InstanceGeometryInfo>)
    bindings[18].binding = 18;
    bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[18].descriptorCount = 1;
    bindings[18].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[18].pImmutableSamplers = nullptr;

    // ========================================================================
    // Binding 19: CIE 1931 Color Matching Functions LUT
    // ========================================================================
    // High-precision CIE XYZ CMFs for VIS_FUSED mode spectral integration
    // 401 samples (380-780nm @ 1nm), each sample is float3(x_bar, y_bar, z_bar)
    // Provides <0.1% error vs analytical approximation's 10-20% at edges
    // ========================================================================

    // Binding 19: CIE CMF LUT buffer (StructuredBuffer<float3>)
    bindings[19].binding = 19;
    bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[19].descriptorCount = 1;
    bindings[19].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;  // Allow both hit and miss shaders to access CIE LUT
    bindings[19].pImmutableSamplers = nullptr;

    // Enable descriptor indexing flags for texture arrays
    // This allows runtime indexing and partially bound descriptors
    std::vector<VkDescriptorBindingFlags> bindingFlags(20, 0);  // Updated for CIE CMF LUT (binding 19)
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
    poolSizes[2].descriptorCount = 13;  // LUT + vertex + index + material + UV + tangent + normal + spectral curves + CRI + solar LUT + atmospheric params + instance geometry info + CIE CMF LUT
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[3].descriptorCount = m_maxTextures + 2;  // Texture array + prefiltered env + BRDF LUT
    poolSizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[4].descriptorCount = m_maxTextures + 1;  // Sampler array + IBL sampler

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

    // Push constant range for camera data + sampling parameters
    // IMPORTANT: Include all shader stages that access push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                    VK_SHADER_STAGE_MISS_BIT_KHR;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstantsRayGen);

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
    QL_LOG_INFO("  [LoadShaders] Loading SPIR-V shaders...");

    // Load all shaders
    const auto raygenSpirv = LoadSPIRV(m_raygenPath);
    QL_LOG_INFO("  [LoadShaders] Raygen loaded: {} words", raygenSpirv.size());

    const auto chitSpirv = LoadSPIRV(m_closestHitPath);
    QL_LOG_INFO("  [LoadShaders] ClosestHit loaded: {} words", chitSpirv.size());

    const auto missSpirv = LoadSPIRV(m_missPath);
    QL_LOG_INFO("  [LoadShaders] Miss loaded: {} words", missSpirv.size());

    const auto shadowMissSpirv = LoadSPIRV(m_shadowMissPath);
    QL_LOG_INFO("  [LoadShaders] ShadowMiss loaded: {} words", shadowMissSpirv.size());

    // Create shader modules (will be destroyed after pipeline creation)
    m_shaderModules.resize(4);
    m_shaderModules[0] = CreateShaderModule(raygenSpirv);
    m_shaderModules[1] = CreateShaderModule(chitSpirv);
    m_shaderModules[2] = CreateShaderModule(missSpirv);
    m_shaderModules[3] = CreateShaderModule(shadowMissSpirv);

    QL_LOG_INFO("  Shaders loaded: {} / {} / {} / {}", m_raygenPath, m_closestHitPath, m_missPath, m_shadowMissPath);
    QL_LOG_INFO("  [LoadShaders] All shader modules created successfully");
    Log::Flush();  // Ensure logs are visible before potential hang
}

std::vector<u32> RayTracingPipeline::LoadSPIRV(const std::string& path) {
    // Try multiple search paths
    const std::vector<std::filesystem::path> searchPaths = {
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

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<u32> buffer(fileSize / sizeof(u32));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
    file.close();

    QL_LOG_DEBUG("  Loaded shader from: {}", foundPath.string());

    return buffer;
}

VkShaderModule RayTracingPipeline::CreateShaderModule(const std::vector<u32>& spirv) const {
    VkDevice device = m_context.GetDevice();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(u32);
    createInfo.pCode = spirv.data();

    VkShaderModule shaderModule;
    if (const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

// ============================================================================
// Pipeline Creation
// ============================================================================

void RayTracingPipeline::CreatePipeline() {
    QL_LOG_INFO("  [CreatePipeline] Starting...");
    Log::Flush();
    VkDevice device = m_context.GetDevice();
    QL_LOG_INFO("  [CreatePipeline] Got VkDevice: {}", (void*)device);
    Log::Flush();

    // Define shader stages (4 stages: raygen, closesthit, miss, shadow_miss)
    std::vector<VkPipelineShaderStageCreateInfo> stages(4);

    // Stage 0: Raygen
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = m_shaderModules[0];
    stages[0].pName = "main";

    // Stage 1: Closest Hit
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[1].module = m_shaderModules[1];
    stages[1].pName = "main";

    // Stage 2: Miss (primary rays - sky background)
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = m_shaderModules[2];
    stages[2].pName = "main";

    // Stage 3: Shadow Miss (shadow rays - not occluded)
    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[3].module = m_shaderModules[3];
    stages[3].pName = "main";

    // Define shader groups (4 groups: raygen, hit, miss, shadow_miss)
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(4);

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

    // Group 2: Miss (primary rays)
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 3: Shadow Miss
    groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[3].generalShader = 3;
    groups[3].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Create pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = static_cast<u32>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<u32>(groups.size());
    pipelineInfo.pGroups = groups.data();
    // Recursion depth 10: supports 8 transmission bounces + shadow rays
    // Required for glass/water refraction with multiple internal reflections
    pipelineInfo.maxPipelineRayRecursionDepth = 10;
    pipelineInfo.layout = m_pipelineLayout;

    QL_LOG_INFO("  [CreatePipeline] Pipeline info prepared, getting function pointer...");
    Log::Flush();

    // Get function pointer for vkCreateRayTracingPipelinesKHR
    const auto vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));

    QL_LOG_INFO("  [CreatePipeline] vkGetDeviceProcAddr returned: {}", (void*)vkCreateRayTracingPipelinesKHR);
    Log::Flush();

    if (!vkCreateRayTracingPipelinesKHR) {
        throw std::runtime_error("Failed to load vkCreateRayTracingPipelinesKHR");
    }

    QL_LOG_INFO("  Calling vkCreateRayTracingPipelinesKHR...");
    QL_LOG_INFO("  (If this is the last log you see, the driver is hanging in pipeline creation)");
    QL_LOG_INFO("  Using pipeline cache: {}", m_pipelineCache != VK_NULL_HANDLE ? "YES" : "NO");
    Log::Flush();  // CRITICAL: Flush before potential driver hang

    auto startTime = std::chrono::high_resolution_clock::now();

    const VkResult result = vkCreateRayTracingPipelinesKHR(
        device,
        VK_NULL_HANDLE,  // No deferred operation
        m_pipelineCache,  // Pipeline cache for faster compilation
        1,
        &pipelineInfo,
        nullptr,
        &m_pipeline
    );

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    QL_LOG_INFO("  vkCreateRayTracingPipelinesKHR returned: {} (took {} ms)", static_cast<int>(result), duration.count());

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline");
    }

    // Destroy shader modules (no longer needed)
    for (const auto module : m_shaderModules) {
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
    const auto vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        vkGetDeviceProcAddr(
        device, "vkGetRayTracingShaderGroupHandlesKHR"));

    if (!vkGetRayTracingShaderGroupHandlesKHR) {
        throw std::runtime_error("Failed to load vkGetRayTracingShaderGroupHandlesKHR");
    }

    // Calculate sizes and offsets
    const u32 handleSize = m_rtProperties.shaderGroupHandleSize;
    const u32 handleAlignment = m_rtProperties.shaderGroupHandleAlignment;
    const u32 baseAlignment = m_rtProperties.shaderGroupBaseAlignment;

    const u32 handleSizeAligned = AlignedSize(handleSize, handleAlignment);

    // SBT layout: [Raygen] [Miss (primary + shadow)] [Hit]
    // Group order in pipeline: 0=Raygen, 1=Hit, 2=Miss(primary), 3=Miss(shadow)
    // SBT order must be: Raygen, Miss, Hit (miss before hit)
    const u32 raygenSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missStride = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missCount = 2;  // Primary miss + Shadow miss
    const u32 missSize = missCount * missStride;
    const u32 hitSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 sbtSize = raygenSize + missSize + hitSize;

    // Get shader group handles (4 groups now)
    constexpr u32 groupCount = 4;
    std::vector<u8> handleData(groupCount * handleSize);

    const VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
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
    // Group indices: 0=Raygen, 1=Hit, 2=Miss(primary), 3=Miss(shadow)
    // Raygen at offset 0
    std::memcpy(sbtData.data(), handleData.data() + handleSize * 0, handleSize);
    // Miss (primary) at offset raygenSize
    std::memcpy(sbtData.data() + raygenSize, handleData.data() + handleSize * 2, handleSize);
    // Miss (shadow) at offset raygenSize + missStride
    std::memcpy(sbtData.data() + raygenSize + missStride, handleData.data() + handleSize * 3, handleSize);
    // Hit at offset raygenSize + missSize
    std::memcpy(sbtData.data() + raygenSize + missSize, handleData.data() + handleSize * 1, handleSize);

    // Create SBT buffer
    m_sbtBuffer = std::make_unique<GpuBuffer>(
        m_context.GetAllocator(),
        sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    m_sbtBuffer->Upload(sbtData.data(), sbtSize);

    // Define SBT regions
    const VkDeviceAddress sbtAddress = m_sbtBuffer->GetDeviceAddress(device);

    m_raygenRegion.deviceAddress = sbtAddress;
    m_raygenRegion.stride = raygenSize;
    m_raygenRegion.size = raygenSize;

    // Miss region contains 2 miss shaders:
    // - missIndex=0: Primary miss (sky background)
    // - missIndex=1: Shadow miss (not occluded)
    m_missRegion.deviceAddress = sbtAddress + raygenSize;
    m_missRegion.stride = missStride;
    m_missRegion.size = missSize;

    m_hitRegion.deviceAddress = sbtAddress + raygenSize + missSize;
    m_hitRegion.stride = hitSize;
    m_hitRegion.size = hitSize;

    m_callableRegion = {};  // No callable shaders

    QL_LOG_INFO("  Shader Binding Table created (size: {} bytes, {} miss entries)", sbtSize, missCount);
}

u32 RayTracingPipeline::AlignedSize(const u32 size, const u32 alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// Descriptor Binding
// ============================================================================

void RayTracingPipeline::BindOutputImage(const GpuImage& image) const {
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

void RayTracingPipeline::BindAccelerationStructure(VkAccelerationStructureKHR tlas) const {
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

void RayTracingPipeline::BindLUTBuffer(const GpuBuffer& buffer) const {
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

void RayTracingPipeline::BindGeometryBuffers(const GpuBuffer &vertexBuffer, const GpuBuffer &indexBuffer,
                                             const GpuBuffer *uvBuffer) const {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo vertexInfo{};
    vertexInfo.buffer = vertexBuffer.GetHandle();
    vertexInfo.offset = 0;
    vertexInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo indexInfo{};
    indexInfo.buffer = indexBuffer.GetHandle();
    indexInfo.offset = 0;
    indexInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo uvInfo{};
    if (uvBuffer) {
        uvInfo.buffer = uvBuffer->GetHandle();
        uvInfo.offset = 0;
        uvInfo.range = VK_WHOLE_SIZE;
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(3);

    // Binding 3: Vertex buffer
    VkWriteDescriptorSet vertexWrite{};
    vertexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vertexWrite.dstSet = m_descriptorSet;
    vertexWrite.dstBinding = 3;
    vertexWrite.dstArrayElement = 0;
    vertexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexWrite.descriptorCount = 1;
    vertexWrite.pBufferInfo = &vertexInfo;
    writes.push_back(vertexWrite);

    // Binding 4: Index buffer
    VkWriteDescriptorSet indexWrite{};
    indexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    indexWrite.dstSet = m_descriptorSet;
    indexWrite.dstBinding = 4;
    indexWrite.dstArrayElement = 0;
    indexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    indexWrite.descriptorCount = 1;
    indexWrite.pBufferInfo = &indexInfo;
    writes.push_back(indexWrite);

    // Binding 8: UV buffer (optional)
    if (uvBuffer) {
        VkWriteDescriptorSet uvWrite{};
        uvWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        uvWrite.dstSet = m_descriptorSet;
        uvWrite.dstBinding = 8;
        uvWrite.dstArrayElement = 0;
        uvWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        uvWrite.descriptorCount = 1;
        uvWrite.pBufferInfo = &uvInfo;
        writes.push_back(uvWrite);
        QL_LOG_DEBUG("  [DEBUG] Bound UV buffer to binding 8");
    } else {
        QL_LOG_DEBUG("  [DEBUG] No UV buffer provided, skipping binding 8");
    }

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
}

void RayTracingPipeline::BindMaterialBuffer(const GpuBuffer& buffer) const {
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

void RayTracingPipeline::BindTangentBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 9;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    QL_LOG_DEBUG("  [DEBUG] Bound tangent buffer to binding 9");
}

void RayTracingPipeline::BindNormalBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 16;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    QL_LOG_DEBUG("  [DEBUG] Bound normal buffer to binding 16");
}

void RayTracingPipeline::BindTextures(const std::vector<VkImageView>& imageViews,
                                       const std::vector<VkSampler>& samplers) const {
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

    // Validate texture count against device capability limit
    if (textureCount > m_maxTextures) {
        QL_LOG_ERROR("Scene has {} textures, but device limit is {} - cannot render",
                     textureCount, m_maxTextures);
        throw std::runtime_error("Scene texture count exceeds device capability");
    }

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

void RayTracingPipeline::BindPrefilteredEnvMap(VkImageView imageView) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_INFO("Binding prefiltered environment map to descriptor set (binding 10)");

    // Build descriptor info for prefiltered environment cubemap
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.sampler = VK_NULL_HANDLE;  // Sampler is separate (binding 12)

    // Update descriptor set
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 10;  // Binding 10: prefilteredEnvMap (TextureCube)
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindBRDFLut(VkImageView imageView, VkSampler sampler) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_INFO("Binding BRDF integration LUT to descriptor set (bindings 11, 12)");

    // Build descriptor info for BRDF LUT texture
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.sampler = VK_NULL_HANDLE;  // Sampler is separate

    // Build descriptor info for IBL sampler (shared by prefiltered env and BRDF LUT)
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = sampler;
    samplerInfo.imageView = VK_NULL_HANDLE;  // Image view is separate
    samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Not used for samplers

    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes(2);

    // Binding 11: BRDF LUT texture (SAMPLED_IMAGE)
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 11;  // Updated from 10 to 11
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;

    // Binding 12: IBL sampler (shared by prefiltered env and BRDF LUT)
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 12;  // Updated from 11 to 12
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &samplerInfo;

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
}

// ============================================================================
// Spectral Curves Buffer Binding
// ============================================================================

void RayTracingPipeline::BindSpectralCurvesBuffer(const GpuBuffer* buffer) const {
    VkDevice device = m_context.GetDevice();

    if (buffer == nullptr) {
        QL_LOG_INFO("Spectral curves buffer is null - spectral curve lookup will use RGB fallback");
        // Note: Shader must handle spectralReflectanceCurveIndex < 0 for fallback
        return;
    }

    QL_LOG_INFO("Binding spectral curves buffer to descriptor set (binding 13)");
    QL_LOG_INFO("  Buffer size: {} bytes", buffer->GetSize());
    QL_LOG_INFO("  Expected curves: ~{}", buffer->GetSize() / 272);  // 272 bytes per SpectralCurveGPU

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 13;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ============================================================================
// Complex Refractive Index Buffer Binding
// ============================================================================

void RayTracingPipeline::BindComplexRefractiveIndexBuffer(const GpuBuffer* buffer) const {
    VkDevice device = m_context.GetDevice();

    if (buffer == nullptr) {
        QL_LOG_INFO("Complex refractive index buffer is null - will use PBR F0 approximation");
        // Note: Shader must handle complexRefractiveIndexIndex < 0 for fallback
        return;
    }

    QL_LOG_INFO("Binding complex refractive index buffer to descriptor set (binding 14)");
    QL_LOG_INFO("  Buffer size: {} bytes", buffer->GetSize());
    QL_LOG_INFO("  Expected entries: ~{}", buffer->GetSize() / 528);  // 528 bytes per ComplexRefractiveIndexGPU

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 14;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ============================================================================
// Solar Spectral LUT Buffer Binding
// ============================================================================

void RayTracingPipeline::BindSolarSpectralLUT(const GpuBuffer* buffer) const {
    VkDevice device = m_context.GetDevice();

    if (buffer == nullptr) {
        QL_LOG_INFO("Solar spectral LUT buffer is null - will use LightingParams RGB fallback");
        // Note: Shader must check if SolarSpectralLUT is valid before use
        return;
    }

    QL_LOG_INFO("Binding solar spectral LUT buffer to descriptor set (binding 15)");
    QL_LOG_INFO("  Buffer size: {} bytes (expected: 544)", buffer->GetSize());

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 15;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ============================================================================
// Bind Atmospheric Parameters Buffer (Binding 17)
// ============================================================================

void RayTracingPipeline::BindAtmosphericParams(const GpuBuffer* buffer) const {
    VkDevice device = m_context.GetDevice();

    if (buffer == nullptr) {
        QL_LOG_INFO("Atmospheric params buffer is null - Delta-Tracking disabled, using LUT fallback");
        // Note: Shader must check if atmospheric params are valid before use
        return;
    }

    QL_LOG_INFO("Binding atmospheric params buffer to descriptor set (binding 17)");
    QL_LOG_INFO("  Buffer size: {} bytes (expected: 64)", buffer->GetSize());

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 17;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindInstanceGeometryBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding instance geometry buffer to descriptor set (binding 18)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 18;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindCIE_CMF_LUT(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding CIE CMF LUT buffer to descriptor set (binding 19)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 19;
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

void RayTracingPipeline::TraceRays(VkCommandBuffer cmd, const u32 width, const u32 height) const {
    // Get function pointer for vkCmdTraceRaysKHR
    const auto vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(
        m_context.GetDevice(), "vkCmdTraceRaysKHR"));

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

    // Push constants (camera + sampling parameters)
    // IMPORTANT: stageFlags must match the pipeline layout push constant range
    vkCmdPushConstants(
        cmd,
        m_pipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
        0,
        sizeof(PushConstantsRayGen),
        &m_pushConstants
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
    m_pushConstants.camera = cameraData;
}

void RayTracingPipeline::SetSamplingParams(const u32 frameIndex, const u32 sampleIndex, const u32 totalSamples,
                                           const u32 randomSeed) {
    m_pushConstants.frameIndex = frameIndex;
    m_pushConstants.sampleIndex = sampleIndex;
    m_pushConstants.totalSamples = totalSamples;
    m_pushConstants.randomSeed = randomSeed;
}

} // namespace quantiloom
