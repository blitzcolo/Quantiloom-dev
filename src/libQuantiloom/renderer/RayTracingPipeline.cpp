#include "RayTracingPipeline.hpp"
#include "core/Log.hpp"
#include <algorithm>
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
// Ray Recursion Depth
// ============================================================================
// The deepest TraceRay chain the closest-hit shader can build. Its budget is
// spelled out at MAX_PATH_DEPTH in src/shaders/closesthit.rchit: eight child
// rays sharing one payload.depth counter, plus the primary ray and one shadow
// ray at the bottom of the chain.
//
// Validated against the device below rather than clamped. Clamping would keep
// the pipeline creating and let chains run past the limit, which is undefined
// behaviour that surfaces as a device loss somewhere unrelated; and silently
// shortening paths would change the physics without saying so.
// ============================================================================

static constexpr u32 kMaxPipelineRayRecursionDepth = 10;

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
    const std::string& anyHitPath,
    VkPipelineCache pipelineCache)
    : m_context(context)
    , m_raygenPath(raygenPath)
    , m_closestHitPath(closestHitPath)
    , m_missPath(missPath)
    , m_anyHitPath(anyHitPath)
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

    // The shaders assume the full recursion chain is available. A device that
    // offers less cannot run them at any quality setting, and the failure it
    // would produce -- chains overrunning the pipeline's declared depth -- is
    // undefined behaviour that surfaces as a device loss somewhere unrelated.
    // Checked once at construction so it names itself instead.
    if (m_rtProperties.maxRayRecursionDepth < kMaxPipelineRayRecursionDepth) {
        throw std::runtime_error(
            "Ray tracing device supports a recursion depth of only " +
            std::to_string(m_rtProperties.maxRayRecursionDepth) +
            "; Quantiloom's path depth needs " +
            std::to_string(kMaxPipelineRayRecursionDepth) +
            " (see MAX_PATH_DEPTH in src/shaders/closesthit.rchit)");
    }

    // Initialize dynamic texture limit based on device capabilities
    m_maxTextures = GetMaxTexturesForDevice(m_context);
    QL_LOG_INFO("Texture limit: {} textures", m_maxTextures);

    // Create pipeline in order with exception safety
    try {
        CreateDescriptorSetLayout();
        CreatePipelineLayout();
        LoadShaders();
        SetSpecConstants(7, false);  // default: RGB, debug off

        QL_LOG_INFO("Ray Tracing pipeline created successfully");
    }
    catch (const std::exception& e) {
        VkDevice device = m_context.GetDevice();

        for (auto& [key, variant] : m_variantCache) {
            if (variant.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, variant.pipeline, nullptr);
            }
        }
        m_variantCache.clear();

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

    m_activeVariant = nullptr;
    m_pipeline = VK_NULL_HANDLE;  // just a copied handle, owned by variant cache
    for (auto& [key, variant] : m_variantCache) {
        variant.sbtBuffer.reset();
        if (variant.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, variant.pipeline, nullptr);
        }
    }
    m_variantCache.clear();

    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
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
    std::vector<VkDescriptorSetLayoutBinding> bindings(28);  // ..23 emissive triangles, 24 thermal temperatures, 25 RGB->spectrum coefficients, 26 thermal sun response, 27 thermal parameter tangent

    // Binding 0: Output image (RWTexture2D)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: Acceleration structure (TLAS)
    // CLOSEST_HIT visibility is required for the recursive rays traced from
    // closesthit.rchit (thermal-IR hemisphere sampling, shadow rays);
    // accessing the TLAS from a stage without visibility is undefined
    // behavior and crashes the device the moment those rays are traced.
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
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
    bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    bindings[4].pImmutableSamplers = nullptr;

    // Binding 5: Material buffer (StructuredBuffer<MaterialData>)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    bindings[5].pImmutableSamplers = nullptr;

    // Binding 6: Texture array (Texture2D[])
    // Uses VK_EXT_descriptor_indexing for runtime array indexing
    // Array size dynamic: 1024 if descriptor indexing available, 32 otherwise
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[6].descriptorCount = m_maxTextures;
    bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    bindings[6].pImmutableSamplers = nullptr;

    // Binding 7: Sampler array (SamplerState[])
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[7].descriptorCount = m_maxTextures;
    bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    bindings[7].pImmutableSamplers = nullptr;

    // Binding 8: UV buffer (StructuredBuffer<float2>) - Optional
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
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
    // NN atmosphere LUT header (baked from MODTRAN surrogate networks)
    // Used in both miss shader (sky) and closest hit (tau / path radiance)
    // ========================================================================

    // Binding 17: NN atmosphere header (StructuredBuffer<AtmosNNHeader>)
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
    bindings[18].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
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

    // ========================================================================
    // Binding 20: NN atmosphere LUT data blob (StructuredBuffer<float>)
    // ========================================================================
    // Flat float array holding the baked tau / lpath / ldown spectral grids;
    // indexed via offsets in the binding-17 header.
    // ========================================================================

    bindings[20].binding = 20;
    bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[20].descriptorCount = 1;
    bindings[20].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[20].pImmutableSamplers = nullptr;

    // ========================================================================
    // Environment Map Sampler (Binding 21)
    // ========================================================================
    // Separate from the IBL sampler at binding 12, which clamps maxLod to 0. That
    // is right for the single-level BRDF LUT and wrong for the environment map: it
    // pinned every lookup to mip 0, so the prefiltered chain was never sampled.
    bindings[21].binding = 21;
    bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[21].descriptorCount = 1;
    bindings[21].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[21].pImmutableSamplers = nullptr;

    // Binding 22: Primary-hit depth AOV (raygen writes hit distance, -1 = miss)
    bindings[22].binding = 22;
    bindings[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[22].descriptorCount = 1;
    bindings[22].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[22].pImmutableSamplers = nullptr;

    // Binding 23: Emissive triangles for next-event estimation
    // (StructuredBuffer<EmissiveTriangleGPU>). Always bound, possibly with a
    // single zero entry; LightingParams::emissiveTriangleCount says whether the
    // shader may read it.
    bindings[23].binding = 23;
    bindings[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[23].descriptorCount = 1;
    bindings[23].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[23].pImmutableSamplers = nullptr;

    // Binding 24: Per-element surface temperatures from the thermal solver
    // (StructuredBuffer<float>). Always bound, with a single zero entry when
    // no solve ran; InstanceGeometryInfo::thermalElementBase says whether the
    // shader may read it, so a scene without a solver reads the material's own
    // temperature exactly as before.
    bindings[24].binding = 24;
    bindings[24].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[24].descriptorCount = 1;
    bindings[24].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[24].pImmutableSamplers = nullptr;

    // Binding 25: Jakob-Hanika RGB -> spectrum coefficients
    // (StructuredBuffer<float4>, xyz = c0,c1,c2, w unused). Three sub-tables of
    // res^3 nodes selected by which RGB channel is largest; see
    // core/RgbToSpectrum.hpp for the layout and the axis warp.
    //
    // Visible to the miss shader as well as closest hit, because the sky's
    // fallback RGB radiance is upsampled there by the same rule a surface
    // reflectance is.
    bindings[25].binding = 25;
    bindings[25].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[25].descriptorCount = 1;
    bindings[25].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    bindings[25].pImmutableSamplers = nullptr;

    // Binding 26: how each element's temperature responds to the sun
    // (StructuredBuffer<float4>). Binding 24 says what temperature the balance
    // gave a triangle; this says what to do about the fact that a triangle is
    // coarser than the shadow crossing it.
    //
    // Record 0 is a header -- xyz is the sun direction the solve used, w is 1
    // when the rest of the buffer is real -- and record 1 + thermalElementBase
    // + PrimitiveIndex() is the element's (dT/dv, v_element). The header lives
    // here rather than in LightingParams because the forcing file owns that
    // sun direction and it need not be [lighting] sun_direction; and at index 0
    // rather than at the end because that needs no element count. A scene with
    // no solve binds one zeroed record, which the w = 0 gate turns off.
    bindings[26].binding = 26;
    bindings[26].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[26].descriptorCount = 1;
    bindings[26].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[26].pImmutableSamplers = nullptr;

    // Binding 27: how each element's temperature responds to one material
    // parameter (StructuredBuffer<float>), and by how much to move it.
    //
    // Element 0 is the step: dp, the amount the parameter is being asked
    // "what if" about, and zero when nothing is. Element 1 + thermalElementBase
    // + PrimitiveIndex() is dT/dp for that triangle. So the shader adds
    // tangent[1 + e] * tangent[0] to the temperature and a scene that is not
    // previewing anything binds a single zero, which the step of zero turns
    // off without a branch of its own.
    //
    // The step lives here rather than in LightingParams because that struct
    // has no room left -- both its padding floats are spoken for, and growing
    // it changes the SDK/Studio pairing hash. Which is also why binding 26
    // carries its own header.
    bindings[27].binding = 27;
    bindings[27].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[27].descriptorCount = 1;
    bindings[27].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[27].pImmutableSamplers = nullptr;

    // Enable descriptor indexing flags for texture arrays
    // This allows runtime indexing and partially bound descriptors
    std::vector<VkDescriptorBindingFlags> bindingFlags(28, 0);
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
    poolSizes[0].descriptorCount = 2;  // output image + depth AOV
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // Bindings 2,3,4,5,8,9,13,14,15,16,17,18,19,20,23,24,25,26. Counted from the
    // layout rather than from the list below, which had drifted: it omitted the
    // atmosphere data blob (binding 20) and so asked the pool for one fewer
    // descriptor than the set declares.
    poolSizes[2].descriptorCount = 19;  // lighting params + vertex + index + material + UV + tangent + spectral curves + CRI + solar LUT + normal + atmosphere header + instance geometry info + CIE CMF LUT + atmosphere data + emissive triangles + thermal temperatures + RGB->spectrum table + thermal sun response + thermal parameter tangent
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[3].descriptorCount = m_maxTextures + 2;  // Texture array + prefiltered env + BRDF LUT
    poolSizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[4].descriptorCount = m_maxTextures + 2;  // Sampler array + IBL sampler + environment sampler

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
                                    VK_SHADER_STAGE_MISS_BIT_KHR |
                                    VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
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

    m_spirvData.resize(5);
    m_spirvData[0] = LoadSPIRV(m_raygenPath);
    QL_LOG_INFO("  [LoadShaders] Raygen loaded: {} words", m_spirvData[0].size());

    m_spirvData[1] = LoadSPIRV(m_closestHitPath);
    QL_LOG_INFO("  [LoadShaders] ClosestHit loaded: {} words", m_spirvData[1].size());

    m_spirvData[2] = LoadSPIRV(m_missPath);
    QL_LOG_INFO("  [LoadShaders] Miss loaded: {} words", m_spirvData[2].size());

    m_spirvData[3] = LoadSPIRV(m_shadowMissPath);
    QL_LOG_INFO("  [LoadShaders] ShadowMiss loaded: {} words", m_spirvData[3].size());

    m_spirvData[4] = LoadSPIRV(m_anyHitPath);
    QL_LOG_INFO("  [LoadShaders] AnyHit loaded: {} words", m_spirvData[4].size());

    QL_LOG_INFO("  Shaders loaded: {} / {} / {} / {}", m_raygenPath, m_closestHitPath, m_missPath, m_shadowMissPath);
    Log::Flush();
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

RayTracingPipeline::PipelineVariant RayTracingPipeline::CreatePipelineVariant(const SpecConstants& spec) {
    VkDevice device = m_context.GetDevice();

    // Create temporary shader modules from stored SPIR-V
    std::vector<VkShaderModule> modules(5);
    for (int i = 0; i < 5; ++i) {
        modules[i] = CreateShaderModule(m_spirvData[i]);
    }

    // Specialization data: applied to closesthit (stage 1) and miss (stage 2)
    VkSpecializationMapEntry specEntries[2] = {};
    specEntries[0].constantID = 0;
    specEntries[0].offset = offsetof(SpecConstants, spectralMode);
    specEntries[0].size = sizeof(u32);
    specEntries[1].constantID = 1;
    specEntries[1].offset = offsetof(SpecConstants, debugEnabled);
    specEntries[1].size = sizeof(u32);

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 2;
    specInfo.pMapEntries = specEntries;
    specInfo.dataSize = sizeof(SpecConstants);
    specInfo.pData = &spec;

    std::vector<VkPipelineShaderStageCreateInfo> stages(5, VkPipelineShaderStageCreateInfo{});
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = modules[0];
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[1].module = modules[1];
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = &specInfo;

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = modules[2];
    stages[2].pName = "main";
    stages[2].pSpecializationInfo = &specInfo;

    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[3].module = modules[3];
    stages[3].pName = "main";

    // No specialization info, as the shadow miss above has none: an any-hit
    // decides whether a surface is present, which is a coverage question with
    // no wavelength in it. It has no use for the spectral mode.
    stages[4].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[4].module = modules[4];
    stages[4].pName = "main";

    // Still four groups. The any-hit joins the one triangles hit group rather
    // than forming its own, so the shader binding table below is unchanged --
    // one hit record, two miss records, and every TraceRay's sbtRecordOffset
    // and missIndex keep the meaning they had.
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(4, VkRayTracingShaderGroupCreateInfoKHR{});
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
    pipelineInfo.maxPipelineRayRecursionDepth = kMaxPipelineRayRecursionDepth;
    pipelineInfo.layout = m_pipelineLayout;

    QL_LOG_INFO("  [CreatePipelineVariant] spectralMode={}, debugEnabled={}", spec.spectralMode, spec.debugEnabled);
    Log::Flush();

    const auto vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
    if (!vkCreateRayTracingPipelinesKHR) {
        throw std::runtime_error("Failed to load vkCreateRayTracingPipelinesKHR");
    }

    QL_LOG_INFO("  Calling vkCreateRayTracingPipelinesKHR...");
    QL_LOG_INFO("  Using pipeline cache: {}", m_pipelineCache != VK_NULL_HANDLE ? "YES" : "NO");
    Log::Flush();

    auto startTime = std::chrono::high_resolution_clock::now();

    PipelineVariant variant;
    const VkResult result = vkCreateRayTracingPipelinesKHR(
        device, VK_NULL_HANDLE, m_pipelineCache,
        1, &pipelineInfo, nullptr, &variant.pipeline);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    QL_LOG_INFO("  vkCreateRayTracingPipelinesKHR returned: {} (took {} ms)", static_cast<int>(result), duration.count());

    if (result != VK_SUCCESS) {
        for (auto m : modules) vkDestroyShaderModule(device, m, nullptr);
        throw std::runtime_error("Failed to create ray tracing pipeline variant");
    }

    for (auto m : modules) vkDestroyShaderModule(device, m, nullptr);

    // Build SBT for this variant
    const auto vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
    if (!vkGetRayTracingShaderGroupHandlesKHR) {
        throw std::runtime_error("Failed to load vkGetRayTracingShaderGroupHandlesKHR");
    }

    const u32 handleSize = m_rtProperties.shaderGroupHandleSize;
    const u32 handleAlignment = m_rtProperties.shaderGroupHandleAlignment;
    const u32 baseAlignment = m_rtProperties.shaderGroupBaseAlignment;
    const u32 handleSizeAligned = AlignedSize(handleSize, handleAlignment);

    const u32 raygenSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missStride = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 missCount = 2;
    const u32 missSize = missCount * missStride;
    const u32 hitSize = AlignedSize(handleSizeAligned, baseAlignment);
    const u32 sbtSize = raygenSize + missSize + hitSize;

    constexpr u32 groupCount = 4;
    std::vector<u8> handleData(groupCount * handleSize);
    vkGetRayTracingShaderGroupHandlesKHR(device, variant.pipeline, 0, groupCount, handleData.size(), handleData.data());

    std::vector<u8> sbtData(sbtSize, 0);
    std::memcpy(sbtData.data(), handleData.data() + handleSize * 0, handleSize);
    std::memcpy(sbtData.data() + raygenSize, handleData.data() + handleSize * 2, handleSize);
    std::memcpy(sbtData.data() + raygenSize + missStride, handleData.data() + handleSize * 3, handleSize);
    std::memcpy(sbtData.data() + raygenSize + missSize, handleData.data() + handleSize * 1, handleSize);

    variant.sbtBuffer = std::make_unique<GpuBuffer>(
        m_context.GetAllocator(), sbtSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    variant.sbtBuffer->Upload(sbtData.data(), sbtSize);

    const VkDeviceAddress sbtAddress = variant.sbtBuffer->GetDeviceAddress(device);
    variant.raygenRegion.deviceAddress = sbtAddress;
    variant.raygenRegion.stride = raygenSize;
    variant.raygenRegion.size = raygenSize;
    variant.missRegion.deviceAddress = sbtAddress + raygenSize;
    variant.missRegion.stride = missStride;
    variant.missRegion.size = missSize;
    variant.hitRegion.deviceAddress = sbtAddress + raygenSize + missSize;
    variant.hitRegion.stride = hitSize;
    variant.hitRegion.size = hitSize;
    variant.callableRegion = {};

    QL_LOG_INFO("  Pipeline variant created (SBT: {} bytes)", sbtSize);
    return variant;
}

void RayTracingPipeline::SetSpecConstants(u32 spectralMode, bool debugEnabled) {
    const u32 dbg = debugEnabled ? 1u : 0u;
    const u64 key = PackSpecKey(spectralMode, dbg);

    auto it = m_variantCache.find(key);
    if (it == m_variantCache.end()) {
        SpecConstants spec{spectralMode, dbg};
        auto [inserted, ok] = m_variantCache.emplace(key, CreatePipelineVariant(spec));
        it = inserted;
    }

    m_activeVariant = &it->second;

    // Keep legacy fields in sync for GetPipeline() backward compat
    m_pipeline = m_activeVariant->pipeline;
    m_raygenRegion = m_activeVariant->raygenRegion;
    m_missRegion = m_activeVariant->missRegion;
    m_hitRegion = m_activeVariant->hitRegion;
    m_callableRegion = m_activeVariant->callableRegion;
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

void RayTracingPipeline::BindDepthImage(const GpuImage& image) const {
    VkDevice device = m_context.GetDevice();

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = image.GetView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 22;
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

void RayTracingPipeline::BindPrefilteredEnvMap(VkImageView imageView, VkSampler sampler) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_INFO("Binding prefiltered environment map to descriptor set (bindings 10, 21)");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.sampler = VK_NULL_HANDLE;  // Sampler is separate (binding 21)

    // Its own sampler rather than the one at binding 12: that one clamps maxLod to
    // 0, which pinned every environment lookup to mip 0 and left the prefiltered
    // chain unreachable.
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = sampler;
    samplerInfo.imageView = VK_NULL_HANDLE;
    samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    std::array<VkWriteDescriptorSet, 2> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 10;  // prefilteredEnvMap (TextureCube)
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 21;  // envSampler
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &samplerInfo;

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
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
// Bind NN Atmosphere Buffers (Bindings 17 + 20)
// ============================================================================

void RayTracingPipeline::BindAtmosphereNN(const GpuBuffer* header,
                                          const GpuBuffer* data) const {
    VkDevice device = m_context.GetDevice();

    if (header == nullptr || data == nullptr) {
        QL_LOG_INFO("NN atmosphere buffers are null - atmosphere disabled");
        return;
    }

    QL_LOG_INFO("Binding NN atmosphere buffers (binding 17 header {} B, "
                "binding 20 data {} B)", header->GetSize(), data->GetSize());

    VkDescriptorBufferInfo headerInfo{};
    headerInfo.buffer = header->GetHandle();
    headerInfo.offset = 0;
    headerInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo dataInfo{};
    dataInfo.buffer = data->GetHandle();
    dataInfo.offset = 0;
    dataInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 17;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &headerInfo;
    writes[1] = writes[0];
    writes[1].dstBinding = 20;
    writes[1].pBufferInfo = &dataInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
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

void RayTracingPipeline::BindEmissiveTriangleBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding emissive triangle buffer to descriptor set (binding 23)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 23;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindThermalSunResponseBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding thermal sun response buffer to descriptor set (binding 26)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 26;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindThermalTangentBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding thermal parameter tangent buffer to descriptor set (binding 27)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 27;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RayTracingPipeline::BindThermalTemperatureBuffer(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding thermal temperature buffer to descriptor set (binding 24)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 24;
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

void RayTracingPipeline::BindRgbToSpectrumTable(const GpuBuffer& buffer) const {
    VkDevice device = m_context.GetDevice();

    QL_LOG_DEBUG("Binding RGB->spectrum coefficient table to descriptor set (binding 25)");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 25;
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

void RayTracingPipeline::TraceRays(VkCommandBuffer cmd, const u32 width, const u32 height, const bool finalDispatch) const {
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
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
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

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    if (finalDispatch) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    } else {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
}

void RayTracingPipeline::SetCameraData(const CameraData& cameraData) {
    m_pushConstants.camera = cameraData;
}

void RayTracingPipeline::SetSamplingParams(const u32 frameIndex, const u32 sampleIndex, const u32 totalSamples,
                                           const u32 randomSeed, const u32 sequenceSeed) {
    m_pushConstants.frameIndex = frameIndex;
    m_pushConstants.sampleIndex = sampleIndex;
    m_pushConstants.totalSamples = totalSamples;
    m_pushConstants.randomSeed = randomSeed;
    m_pushConstants.sequenceSeed = sequenceSeed;
}

} // namespace quantiloom
