/**
 * @file ExternalRenderContext.cpp
 * @brief Implementation of external Vulkan context injection API
 *
 * @author wtflmao
 */

// VMA must be included BEFORE GpuBuffer.hpp/GpuImage.hpp to avoid enum redefinition
#include <vk_mem_alloc.h>

#include "ExternalRenderContext.hpp"
#include "VulkanContextAdapter.hpp"
#include "RayTracingPipeline.hpp"
#include "AccelerationStructure.hpp"
#include "GpuBuffer.hpp"
#include "GpuImage.hpp"
#include "TextureManager.hpp"
#include "CommandHelper.hpp"
#include "BRDFLutGenerator.hpp"
#include "LightingParams.hpp"
#include "AtmosphericConfig.hpp"

#include "core/Log.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "io/GltfLoader.hpp"
#include "io/UsdLoader.hpp"
#include "io/ImageIO.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <chrono>
#include <filesystem>
#include <random>

// Platform-specific includes for cache directory
#if defined(_WIN32)
    #include <shlobj.h>
    #include <windows.h>
#elif defined(__APPLE__)
    #include <pwd.h>
    #include <unistd.h>
#else  // Linux
    #include <pwd.h>
    #include <unistd.h>
#endif

namespace quantiloom {

// ============================================================================
// Platform-specific cache directory helper
// ============================================================================

/**
 * @brief Get the default pipeline cache directory for the current platform
 * @return Path to cache directory (creates if doesn't exist)
 *
 * Platform-specific locations:
 *   Windows: %LOCALAPPDATA%/Quantiloom/cache/
 *   Linux:   ~/.cache/Quantiloom/
 *   macOS:   ~/Library/Caches/Quantiloom/
 */
static std::string GetDefaultCacheDirectory() {
    std::filesystem::path cacheDir;

#if defined(_WIN32)
    // Windows: Use LOCALAPPDATA
    wchar_t* localAppData = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &localAppData))) {
        cacheDir = std::filesystem::path(localAppData) / "Quantiloom" / "cache";
        CoTaskMemFree(localAppData);
    } else {
        // Fallback to temp directory
        cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
    }

#elif defined(__APPLE__)
    // macOS: Use ~/Library/Caches/
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (home) {
        cacheDir = std::filesystem::path(home) / "Library" / "Caches" / "Quantiloom";
    } else {
        cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
    }

#else  // Linux
    // Linux: Use XDG_CACHE_HOME or ~/.cache/
    const char* xdgCache = getenv("XDG_CACHE_HOME");
    if (xdgCache && xdgCache[0] != '\0') {
        cacheDir = std::filesystem::path(xdgCache) / "Quantiloom";
    } else {
        const char* home = getenv("HOME");
        if (!home) {
            struct passwd* pw = getpwuid(getuid());
            if (pw) home = pw->pw_dir;
        }
        if (home) {
            cacheDir = std::filesystem::path(home) / ".cache" / "Quantiloom";
        } else {
            cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
        }
    }
#endif

    // Create directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    if (ec) {
        QL_LOG_WARN("Failed to create cache directory {}: {}", cacheDir.string(), ec.message());
        // Fall back to current directory
        return ".";
    }

    return cacheDir.string();
}

// ============================================================================
// MaterialDataCPU - GPU upload structure (must match shader)
// ============================================================================

struct MaterialDataCPU {
    glm::vec4 baseColorFactor;
    i32 baseColorTextureIndex;
    f32 metallicFactor;
    f32 roughnessFactor;
    i32 metallicRoughnessTextureIndex;
    i32 normalTextureIndex;
    f32 normalScale;
    u32 doubleSided;
    f32 _padding0;
    glm::vec3 emissiveFactor;
    i32 emissiveTextureIndex;
    u32 alphaMode;
    f32 alphaCutoff;
    f32 spectralAlbedo;
    i32 spectralReflectanceCurveIndex;
    f32 irEmissivity;
    f32 irTransmittance;
    f32 irTemperature_K;
    i32 complexRefractiveIndexIndex;

    // ========================================================================
    // Transmission Properties (KHR_materials_transmission + KHR_materials_volume)
    // ========================================================================
    f32 ior;                          // Index of refraction (1.0=air, 1.33=water, 1.5=glass)
    f32 transmission;                 // Transmission strength [0,1]
    i32 transmissionTextureIndex;     // Transmission texture (-1 = no texture)
    f32 _padding1;                    // Padding for alignment

    glm::vec3 attenuationColor;       // Color at attenuation distance
    f32 attenuationDistance;          // Distance for attenuation (m, 0 = no attenuation)

    f32 thicknessFactor;              // Thickness for thin-walled approximation
    i32 thicknessTextureIndex;        // Thickness texture (-1 = no texture)
    f32 dispersion;                   // Abbe number reciprocal (0 = no dispersion)
    f32 _padding2;                    // Padding for alignment

    // ========================================================================
    // Participating Media Properties (fog, smoke, SSS)
    // ========================================================================
    f32 volumeDensity;                // Medium density multiplier (0 = no volume)
    f32 scatteringCoeff;              // Scattering coefficient σ_s (m⁻¹)
    f32 absorptionCoeff;              // Absorption coefficient σ_a (m⁻¹)
    f32 phaseG;                       // Henyey-Greenstein g parameter [-1,1]
};

static_assert(sizeof(MaterialDataCPU) == 160, "MaterialDataCPU size mismatch");

// ============================================================================
// InstanceGeometryInfo - Per-instance geometry offset info (must match shader)
// ============================================================================
// When multiple BLAS exist, shader needs to know where each instance's geometry
// data starts in the merged global buffers. This structure provides those offsets.
//
// Shader usage:
//   uint instanceIdx = InstanceIndex();
//   InstanceGeometryInfo geo = instanceGeometryInfo[instanceIdx];
//   uint globalIdx = geo.indexOffset + PrimitiveIndex() * 3 + localVertexIdx;
//   float3 v = vertexBuffer[geo.vertexOffset + indexBuffer[globalIdx]];
// ============================================================================

struct InstanceGeometryInfo {
    u32 vertexOffset;   // Offset into global vertex buffer (in vertex count)
    u32 indexOffset;    // Offset into global index buffer (in index count)
    u32 normalOffset;   // Offset into global normal buffer (in normal count)
    u32 uvOffset;       // Offset into global UV buffer (in UV count)
    u32 tangentOffset;  // Offset into global tangent buffer (in tangent count)
    u32 materialId;     // Material index (replaces instanceCustomIndex usage)
    u32 pad[2];         // Padding for 32-byte alignment
};

static_assert(sizeof(InstanceGeometryInfo) == 32, "InstanceGeometryInfo size mismatch");

// ============================================================================
// ExternalRenderContext::Impl - PIMPL implementation
// ============================================================================

struct ExternalRenderContext::Impl {
    // Context adapter (provides VulkanContext interface for external handles)
    std::unique_ptr<VulkanContextAdapter> contextAdapter;

    // External Vulkan handles (cached for reference)
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    u32 graphicsQueueFamily = 0;
    VkFormat targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;

    // Current dimensions
    u32 width = 1280;
    u32 height = 720;

    // Scene data
    std::unique_ptr<Scene> scene;
    Camera camera;

    // Acceleration structures
    std::vector<std::unique_ptr<BLAS>> blasList;
    std::unique_ptr<TLAS> tlas;

    // GPU resources
    std::unique_ptr<GpuImage> outputImage;
    std::unique_ptr<GpuBuffer> lightingParamsBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    std::unique_ptr<GpuBuffer> spectralCurvesBuffer;
    std::unique_ptr<GpuBuffer> criBuffer;
    std::unique_ptr<GpuBuffer> solarLutBuffer;
    std::unique_ptr<GpuBuffer> atmosphericBuffer;
    std::unique_ptr<GpuBuffer> cieCmfBuffer;  // CIE 1931 CMF LUT for VIS_Fused mode (binding 19)

    // Merged global geometry buffers (for multi-BLAS support)
    // All BLAS geometry data is merged into single global buffers
    // Shader uses InstanceGeometryInfo offsets to index correctly
    std::unique_ptr<GpuBuffer> globalVertexBuffer;
    std::unique_ptr<GpuBuffer> globalIndexBuffer;
    std::unique_ptr<GpuBuffer> globalNormalBuffer;
    std::unique_ptr<GpuBuffer> globalUVBuffer;
    std::unique_ptr<GpuBuffer> globalTangentBuffer;
    std::unique_ptr<GpuBuffer> instanceGeometryBuffer;  // Per-instance geometry offsets

    // IBL resources
    std::unique_ptr<GpuImage> brdfLutTexture;
    std::unique_ptr<GpuImage> envMapImage;
    VkSampler brdfLutSampler = VK_NULL_HANDLE;

    // Texture manager
    std::unique_ptr<TextureManager> textureManager;

    // Ray tracing pipeline
    std::unique_ptr<RayTracingPipeline> pipeline;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::string pipelineCachePath;  // Set in Create() based on InitParams or platform default

    // Command pool for internal operations
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // Rendering state
    SpectralMode spectralMode = SpectralMode::RGB;  // Default: Fast RGB mode
    DebugVisualizationMode debugMode = DebugVisualizationMode::None;  // Debug visualization mode
    f32 wavelength_nm = 550.0f;
    u32 spp = 1;
    LightingParams lightingParams;

    // Atmospheric configuration
    AtmosphericConfig atmosphericConfig;  // CPU-side config (default: disabled)
    bool atmosphericDirty = true;         // Need upload to GPU

    // Environment map state
    bool hasCustomEnvMap = false;         // True if LoadEnvironmentMap succeeded

    // Accumulation
    u32 accumulatedSamples = 0;
    u32 frameIndex = 0;

    // Random number generator for better sample distribution
    // Uses Mersenne Twister for high-quality randomness (matches CLI app)
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<u32> randDist{0, std::numeric_limits<u32>::max()};

    // Statistics
    f32 lastFrameTimeMs = 0.0f;
    std::chrono::steady_clock::time_point frameStartTime;

    // Pixel readback buffer (for debug hover display)
    std::unique_ptr<GpuBuffer> pixelReadbackBuffer;

    // Ready flag
    bool isReady = false;

    // Destructor
    ~Impl() {
        Cleanup();
    }

    void Cleanup() {
        // Wait for GPU to finish
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }

        // Destroy resources in reverse order
        pipeline.reset();

        // Save and destroy pipeline cache
        if (pipelineCache != VK_NULL_HANDLE && contextAdapter) {
            RayTracingPipeline::SavePipelineCache(*contextAdapter, pipelineCache, pipelineCachePath);
            RayTracingPipeline::DestroyPipelineCache(*contextAdapter, pipelineCache);
            pipelineCache = VK_NULL_HANDLE;
        }

        textureManager.reset();

        if (brdfLutSampler != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroySampler(device, brdfLutSampler, nullptr);
            brdfLutSampler = VK_NULL_HANDLE;
        }

        envMapImage.reset();
        brdfLutTexture.reset();

        atmosphericBuffer.reset();
        solarLutBuffer.reset();
        criBuffer.reset();
        spectralCurvesBuffer.reset();
        materialBuffer.reset();
        lightingParamsBuffer.reset();
        outputImage.reset();
        pixelReadbackBuffer.reset();

        // Reset merged global geometry buffers
        globalVertexBuffer.reset();
        globalIndexBuffer.reset();
        globalNormalBuffer.reset();
        globalUVBuffer.reset();
        globalTangentBuffer.reset();
        instanceGeometryBuffer.reset();

        tlas.reset();
        blasList.clear();

        if (commandPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }

        scene.reset();

        // Destroy context adapter last (may own VMA allocator)
        contextAdapter.reset();

        isReady = false;
    }
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

ExternalRenderContext::ExternalRenderContext()
    : m_impl(std::make_unique<Impl>()) {
}

ExternalRenderContext::~ExternalRenderContext() = default;

ExternalRenderContext::ExternalRenderContext(ExternalRenderContext&&) noexcept = default;
ExternalRenderContext& ExternalRenderContext::operator=(ExternalRenderContext&&) noexcept = default;

// ============================================================================
// Factory Method
// ============================================================================

Result<std::unique_ptr<ExternalRenderContext>, String> ExternalRenderContext::Create(const InitParams& params) {
    // Validate parameters
    if (params.instance == VK_NULL_HANDLE) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err("VkInstance is null");
    }
    if (params.physicalDevice == VK_NULL_HANDLE) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err("VkPhysicalDevice is null");
    }
    if (params.device == VK_NULL_HANDLE) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err("VkDevice is null");
    }
    if (params.graphicsQueue == VK_NULL_HANDLE) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err("VkQueue is null");
    }
    if (params.width == 0 || params.height == 0) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err("Invalid dimensions");
    }

    // Use new directly since constructor is private (make_unique can't access it)
    std::unique_ptr<ExternalRenderContext> context(new ExternalRenderContext());
    auto initResult = context->Initialize(params);
    if (!initResult.has_value()) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err(initResult.error());
    }

    return Result<std::unique_ptr<ExternalRenderContext>, String>(std::move(context));
}

// ============================================================================
// Initialization
// ============================================================================

Result<void, String> ExternalRenderContext::Initialize(const InitParams& params) {
    QL_LOG_INFO("Initializing ExternalRenderContext...");

    // Store external handles
    m_impl->instance = params.instance;
    m_impl->physicalDevice = params.physicalDevice;
    m_impl->device = params.device;
    m_impl->graphicsQueue = params.graphicsQueue;
    m_impl->graphicsQueueFamily = params.graphicsQueueFamily;
    m_impl->targetColorFormat = params.targetColorFormat;
    m_impl->width = params.width;
    m_impl->height = params.height;

    // Set pipeline cache path (use provided path or platform-specific default)
    if (!params.pipelineCacheDir.empty()) {
        m_impl->pipelineCachePath = (std::filesystem::path(params.pipelineCacheDir) / "pipeline_cache.bin").string();
        // Ensure directory exists
        std::error_code ec;
        std::filesystem::create_directories(params.pipelineCacheDir, ec);
        if (ec) {
            QL_LOG_WARN("Failed to create cache directory {}: {}", params.pipelineCacheDir, ec.message());
        }
    } else {
        m_impl->pipelineCachePath = (std::filesystem::path(GetDefaultCacheDirectory()) / "pipeline_cache.bin").string();
    }
    QL_LOG_INFO("Pipeline cache path: {}", m_impl->pipelineCachePath);

    // Create VulkanContextAdapter from external handles
    VulkanContext::ExternalHandles adapterHandles{};
    adapterHandles.instance = params.instance;
    adapterHandles.physicalDevice = params.physicalDevice;
    adapterHandles.device = params.device;
    adapterHandles.graphicsQueue = params.graphicsQueue;
    adapterHandles.graphicsQueueFamily = params.graphicsQueueFamily;
    adapterHandles.allocator = params.externalAllocator;

    try {
        m_impl->contextAdapter = std::make_unique<VulkanContextAdapter>(adapterHandles, true);
    } catch (const std::exception& e) {
        return Result<void, String>::Err(String("Failed to create VulkanContextAdapter: ") + e.what());
    }

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = params.graphicsQueueFamily;

    VkResult result = vkCreateCommandPool(params.device, &poolInfo, nullptr, &m_impl->commandPool);
    if (result != VK_SUCCESS) {
        return Result<void, String>::Err("Failed to create command pool");
    }

    // Initialize default lighting params
    m_impl->lightingParams = CreateDefaultLightingParams();

    // Create output image
    m_impl->outputImage = std::make_unique<GpuImage>(
        m_impl->contextAdapter->GetAllocator(),
        m_impl->contextAdapter->GetDevice(),
        params.width, params.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Transition output image to GENERAL layout
    TransitionImageLayoutImmediate(
        m_impl->outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Create lighting params buffer
    m_impl->lightingParamsBuffer = std::make_unique<GpuBuffer>(
        m_impl->contextAdapter->GetAllocator(),
        sizeof(LightingParams),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->lightingParamsBuffer->Upload(&m_impl->lightingParams, sizeof(LightingParams));

    // Create dummy buffers for optional bindings
    CreateDummyBuffers();

    // Create BRDF LUT for IBL
    CreateBRDFLut();

    // Create fallback environment map
    CreateFallbackEnvMap();

    // Create texture manager
    m_impl->textureManager = std::make_unique<TextureManager>(*m_impl->contextAdapter);

    QL_LOG_INFO("ExternalRenderContext initialized ({}x{})", params.width, params.height);
    return Result<void, String>::Ok();
}

// ============================================================================
// Scene Loading
// ============================================================================

Result<void, String> ExternalRenderContext::LoadSceneFromConfig(const String& configPath) {
    auto configResult = Config::Load(configPath);
    if (!configResult.has_value()) {
        return Result<void, String>::Err("Failed to load config: " + configResult.error());
    }
    return LoadScene(configResult.value());
}

Result<void, String> ExternalRenderContext::LoadScene(const Config& config) {
    // Check for glTF file
    if (config.Has("scene.gltf")) {
        auto gltfPath = config.Get<String>("scene.gltf");
        return LoadSceneFromGltf(gltfPath);
    }

    // Check for USD file
    if (config.Has("scene.usd")) {
        auto usdPath = config.Get<String>("scene.usd");
        return LoadSceneFromUsd(usdPath);
    }

    return Result<void, String>::Err("No scene.gltf or scene.usd specified in config");
}

Result<void, String> ExternalRenderContext::LoadSceneFromGltf(const String& gltfPath) {
    QL_LOG_INFO("Loading glTF scene: {}", gltfPath);

    auto result = GltfLoader::LoadFromFile(gltfPath);
    if (!result.has_value()) {
        return Result<void, String>::Err("Failed to load glTF: " + result.error());
    }

    m_impl->scene = std::make_unique<Scene>(std::move(result.value()));

    // Setup camera from scene
    m_impl->camera = m_impl->scene->camera;
    m_impl->camera.SetAspectRatio(static_cast<f32>(m_impl->width) / static_cast<f32>(m_impl->height));

    // Upload textures
    m_impl->textureManager->UploadTextures(m_impl->scene->textures);

    // Build acceleration structures and GPU resources
    BuildAccelerationStructures();
    UpdateGpuResources();

    // Create ray tracing pipeline
    CreatePipeline();

    m_impl->isReady = true;
    m_impl->accumulatedSamples = 0;

    QL_LOG_INFO("Scene loaded: {} meshes, {} materials, {} textures",
                m_impl->scene->meshes.size(),
                m_impl->scene->materials.size(),
                m_impl->scene->textures.size());

    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::LoadSceneFromUsd(const String& usdPath) {
    QL_LOG_INFO("Loading USD scene: {}", usdPath);

    auto result = UsdLoader::LoadFromFile(usdPath);
    if (!result.has_value()) {
        return Result<void, String>::Err("Failed to load USD: " + result.error());
    }

    m_impl->scene = std::make_unique<Scene>(std::move(result.value()));

    // Setup camera from scene
    m_impl->camera = m_impl->scene->camera;
    m_impl->camera.SetAspectRatio(static_cast<f32>(m_impl->width) / static_cast<f32>(m_impl->height));

    // Upload textures
    m_impl->textureManager->UploadTextures(m_impl->scene->textures);

    // Build acceleration structures and GPU resources
    BuildAccelerationStructures();
    UpdateGpuResources();

    // Create ray tracing pipeline
    CreatePipeline();

    m_impl->isReady = true;
    m_impl->accumulatedSamples = 0;

    QL_LOG_INFO("USD scene loaded: {} meshes, {} materials, {} textures",
                m_impl->scene->meshes.size(),
                m_impl->scene->materials.size(),
                m_impl->scene->textures.size());

    return Result<void, String>::Ok();
}

bool ExternalRenderContext::HasScene() const {
    return m_impl->scene != nullptr;
}

const Scene* ExternalRenderContext::GetScene() const {
    return m_impl->scene.get();
}

// ============================================================================
// Rendering
// ============================================================================

void ExternalRenderContext::RenderFrame(
    VkCommandBuffer cmd,
    VkImage targetImage,
    VkImageLayout targetLayout,
    u32 width,
    u32 height) {

    if (!m_impl->isReady || !m_impl->pipeline) {
        QL_LOG_WARN("ExternalRenderContext::RenderFrame called but not ready");
        return;
    }

    m_impl->frameStartTime = std::chrono::steady_clock::now();

    // Handle resize
    if (width != m_impl->width || height != m_impl->height) {
        Resize(width, height);
    }

    // Update camera data with current state
    CameraData cameraData = m_impl->camera.GetCameraData();
    cameraData.wavelength_nm = m_impl->wavelength_nm;
    cameraData.spectral_mode = static_cast<u32>(m_impl->spectralMode);
    cameraData.debug_mode = static_cast<u32>(m_impl->debugMode);
    m_impl->pipeline->SetCameraData(cameraData);

    // Set sampling parameters
    // Use Mersenne Twister RNG for better sample distribution (reduces fireflies)
    u32 randomSeed = m_impl->randDist(m_impl->rng) ^ (m_impl->frameIndex * 997 + m_impl->accumulatedSamples * 1009);
    m_impl->pipeline->SetSamplingParams(
        m_impl->frameIndex,
        m_impl->accumulatedSamples,
        m_impl->spp,
        randomSeed
    );

    // Execute ray tracing (writes to internal outputImage in GENERAL layout)
    m_impl->pipeline->TraceRays(cmd, width, height);

    // ========================================================================
    // Blit outputImage to target swapchain image
    // ========================================================================

    // Step 1: Transition outputImage from GENERAL to TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier outputBarrier{};
    outputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.image = m_impl->outputImage->GetImage();
    outputBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputBarrier.subresourceRange.baseMipLevel = 0;
    outputBarrier.subresourceRange.levelCount = 1;
    outputBarrier.subresourceRange.baseArrayLayer = 0;
    outputBarrier.subresourceRange.layerCount = 1;
    outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // Step 2: Transition target image to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier targetBarrier{};
    targetBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    targetBarrier.oldLayout = targetLayout;
    targetBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    targetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    targetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    targetBarrier.image = targetImage;
    targetBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    targetBarrier.subresourceRange.baseMipLevel = 0;
    targetBarrier.subresourceRange.levelCount = 1;
    targetBarrier.subresourceRange.baseArrayLayer = 0;
    targetBarrier.subresourceRange.layerCount = 1;
    targetBarrier.srcAccessMask = 0;  // Previous access unknown
    targetBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkImageMemoryBarrier barriers[2] = {outputBarrier, targetBarrier};
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        2, barriers
    );

    // Step 3: Blit (with format conversion: R32G32B32A32_SFLOAT -> B8G8R8A8_SRGB)
    // vkCmdBlitImage handles HDR->SDR clamping automatically (values > 1.0 become 1.0)
    VkImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.mipLevel = 0;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0] = {0, 0, 0};
    blitRegion.srcOffsets[1] = {static_cast<i32>(width), static_cast<i32>(height), 1};

    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.mipLevel = 0;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0] = {0, 0, 0};
    blitRegion.dstOffsets[1] = {static_cast<i32>(width), static_cast<i32>(height), 1};

    vkCmdBlitImage(
        cmd,
        m_impl->outputImage->GetImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        targetImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blitRegion,
        VK_FILTER_NEAREST  // No filtering needed for same-size blit
    );

    // Step 4: Transition outputImage back to GENERAL for next frame
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    // Step 5: Transition target to PRESENT_SRC_KHR for presentation
    targetBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    targetBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    targetBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    targetBarrier.dstAccessMask = 0;

    barriers[0] = outputBarrier;
    barriers[1] = targetBarrier;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        2, barriers
    );

    m_impl->accumulatedSamples++;
    m_impl->frameIndex++;

    // Calculate frame time
    auto frameEndTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(frameEndTime - m_impl->frameStartTime);
    m_impl->lastFrameTimeMs = duration.count() / 1000.0f;
}

void ExternalRenderContext::Resize(u32 width, u32 height) {
    if (width == m_impl->width && height == m_impl->height) {
        return;
    }

    QL_LOG_INFO("ExternalRenderContext::Resize {}x{} -> {}x{}",
                m_impl->width, m_impl->height, width, height);

    m_impl->width = width;
    m_impl->height = height;

    // Wait for GPU
    vkDeviceWaitIdle(m_impl->device);

    // Recreate output image
    m_impl->outputImage = std::make_unique<GpuImage>(
        m_impl->contextAdapter->GetAllocator(),
        m_impl->contextAdapter->GetDevice(),
        width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    TransitionImageLayoutImmediate(
        m_impl->outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Re-bind output image
    if (m_impl->pipeline) {
        m_impl->pipeline->BindOutputImage(*m_impl->outputImage);
    }

    // Update camera aspect ratio
    m_impl->camera.SetAspectRatio(static_cast<f32>(width) / static_cast<f32>(height));

    // Reset accumulation
    ResetAccumulation();
}

void ExternalRenderContext::ResetAccumulation() {
    m_impl->accumulatedSamples = 0;
}

// ============================================================================
// Camera Control
// ============================================================================

void ExternalRenderContext::SetCameraViewMatrix(const glm::mat4& viewMatrix) {
    // Extract camera position from view matrix inverse
    // View matrix transforms world -> camera, so inverse gives camera position
    glm::mat4 invView = glm::inverse(viewMatrix);
    glm::vec3 position = glm::vec3(invView[3]);
    glm::vec3 forward = -glm::vec3(invView[2]);  // Camera looks along -Z
    glm::vec3 target = position + forward;

    m_impl->camera.SetPosition(position);
    m_impl->camera.SetLookAt(target);
    ResetAccumulation();
}

void ExternalRenderContext::SetCameraLookAt(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) {
    m_impl->camera.SetPosition(position);
    m_impl->camera.SetLookAt(target);
    m_impl->camera.SetUp(up);
    ResetAccumulation();
}

void ExternalRenderContext::SetCameraFOV(f32 fovYDegrees) {
    m_impl->camera.SetFovY(fovYDegrees);
    ResetAccumulation();
}

const Camera& ExternalRenderContext::GetCamera() const {
    return m_impl->camera;
}

// ============================================================================
// Spectral Rendering Parameters
// ============================================================================

void ExternalRenderContext::SetSpectralMode(SpectralMode mode) {
    if (m_impl->spectralMode != mode) {
        m_impl->spectralMode = mode;
        ResetAccumulation();
    }
}

void ExternalRenderContext::SetWavelength(f32 wavelength_nm) {
    if (m_impl->wavelength_nm != wavelength_nm) {
        m_impl->wavelength_nm = wavelength_nm;
        ResetAccumulation();
    }
}

void ExternalRenderContext::SetSPP(u32 spp) {
    m_impl->spp = spp;
}

SpectralMode ExternalRenderContext::GetSpectralMode() const {
    return m_impl->spectralMode;
}

f32 ExternalRenderContext::GetWavelength() const {
    return m_impl->wavelength_nm;
}

u32 ExternalRenderContext::GetSPP() const {
    return m_impl->spp;
}

// ============================================================================
// Debug Visualization
// ============================================================================

void ExternalRenderContext::SetDebugMode(DebugVisualizationMode mode) {
    if (m_impl->debugMode != mode) {
        m_impl->debugMode = mode;
        ResetAccumulation();  // Reset accumulation when debug mode changes
    }
}

DebugVisualizationMode ExternalRenderContext::GetDebugMode() const {
    return m_impl->debugMode;
}

// ============================================================================
// Lighting Parameters
// ============================================================================

void ExternalRenderContext::SetLightingParams(const LightingParams& params) {
    m_impl->lightingParams = params;
    m_impl->lightingParamsBuffer->Upload(&m_impl->lightingParams, sizeof(LightingParams));
    ResetAccumulation();
}

void ExternalRenderContext::SetSunDirection(const glm::vec3& direction) {
    m_impl->lightingParams.sunDirection = glm::normalize(direction);
    m_impl->lightingParamsBuffer->Upload(&m_impl->lightingParams, sizeof(LightingParams));
    ResetAccumulation();
}

void ExternalRenderContext::SetSunRadiance(const glm::vec3& radiance) {
    m_impl->lightingParams.sunRadiance_rgb = radiance;
    m_impl->lightingParams.sunRadiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0f;
    m_impl->lightingParamsBuffer->Upload(&m_impl->lightingParams, sizeof(LightingParams));
    ResetAccumulation();
}

void ExternalRenderContext::SetSkyRadiance(const glm::vec3& radiance) {
    m_impl->lightingParams.skyRadiance_rgb = radiance;
    m_impl->lightingParams.skyRadiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0f;
    m_impl->lightingParamsBuffer->Upload(&m_impl->lightingParams, sizeof(LightingParams));
    ResetAccumulation();
}

const LightingParams& ExternalRenderContext::GetLightingParams() const {
    return m_impl->lightingParams;
}

// ============================================================================
// Scene Editing (Stubs for Phase 2)
// ============================================================================

u32 ExternalRenderContext::AddMesh(const Mesh& mesh, const glm::mat4& transform) {
    // TODO: Implement in Phase 2
    (void)mesh;
    (void)transform;
    QL_LOG_WARN("ExternalRenderContext::AddMesh not implemented yet");
    return 0;
}

bool ExternalRenderContext::RemoveNode(u32 nodeIndex) {
    // TODO: Implement in Phase 2
    (void)nodeIndex;
    QL_LOG_WARN("ExternalRenderContext::RemoveNode not implemented yet");
    return false;
}

void ExternalRenderContext::SetNodeTransform(u32 nodeIndex, const glm::mat4& transform) {
    if (!m_impl->scene) {
        QL_LOG_WARN("SetNodeTransform: No scene loaded");
        return;
    }

    if (nodeIndex >= m_impl->scene->nodes.size()) {
        QL_LOG_WARN("SetNodeTransform: Invalid node index {}", nodeIndex);
        return;
    }

    // Update the node's transform in the scene
    m_impl->scene->nodes[nodeIndex].transform = transform;

    QL_LOG_DEBUG("SetNodeTransform: Updated node {} transform", nodeIndex);
}

void ExternalRenderContext::UpdateMaterial(u32 materialIndex, const Material& material) {
    // TODO: Implement in Phase 2
    (void)materialIndex;
    (void)material;
    QL_LOG_WARN("ExternalRenderContext::UpdateMaterial not implemented yet");
}

void ExternalRenderContext::RebuildAccelerationStructure() {
    if (!m_impl->scene || m_impl->blasList.empty()) {
        QL_LOG_WARN("RebuildAccelerationStructure: No scene or BLAS available");
        return;
    }

    QL_LOG_DEBUG("Rebuilding TLAS with updated transforms...");

    // Wait for GPU to finish any pending work
    vkDeviceWaitIdle(m_impl->device);

    // Reset TLAS
    m_impl->tlas.reset();
    m_impl->tlas = std::make_unique<TLAS>(*m_impl->contextAdapter);

    // Execute TLAS build on GPU
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
        // Build mapping from mesh index to BLAS starting index
        std::vector<size_t> meshToBlasStart;
        meshToBlasStart.reserve(m_impl->scene->meshes.size());
        size_t blasStart = 0;
        for (const auto& mesh : m_impl->scene->meshes) {
            meshToBlasStart.push_back(blasStart);
            blasStart += mesh.primitives.size();
        }

        // Add instances to TLAS with current transforms
        for (const auto& node : m_impl->scene->nodes) {
            const Mesh& mesh = m_impl->scene->meshes[node.meshIndex];
            size_t blasBase = meshToBlasStart[node.meshIndex];

            for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
                const auto& primitive = mesh.primitives[primIdx];
                // Get material's doubleSided property for hardware backface culling control
                bool doubleSided = true;  // Default: disable culling (backward compatible)
                if (primitive.materialId >= 0 &&
                    static_cast<size_t>(primitive.materialId) < m_impl->scene->materials.size()) {
                    doubleSided = m_impl->scene->materials[primitive.materialId].doubleSided;
                }

                m_impl->tlas->AddInstance(
                    *m_impl->blasList[blasBase + primIdx],
                    primitive.materialId,
                    node.transform,
                    doubleSided
                );
            }
        }

        m_impl->tlas->Build(cmd);
    });

    // Re-bind TLAS to pipeline
    if (m_impl->pipeline) {
        m_impl->pipeline->BindAccelerationStructure(m_impl->tlas->GetHandle());
    }

    QL_LOG_DEBUG("TLAS rebuilt successfully");
}

// ============================================================================
// Status and Statistics
// ============================================================================

u32 ExternalRenderContext::GetAccumulatedSamples() const {
    return m_impl->accumulatedSamples;
}

f32 ExternalRenderContext::GetLastFrameTimeMs() const {
    return m_impl->lastFrameTimeMs;
}

bool ExternalRenderContext::IsReady() const {
    return m_impl->isReady;
}

Result<glm::vec4, String> ExternalRenderContext::ReadPixelValue(u32 x, u32 y) {
    // Validate bounds
    if (x >= m_impl->width || y >= m_impl->height) {
        return Result<glm::vec4, String>::Err("Pixel coordinates out of bounds");
    }

    if (!m_impl->isReady || !m_impl->outputImage) {
        return Result<glm::vec4, String>::Err("Render context not ready");
    }

    // Create staging buffer if not exists (16 bytes = sizeof(float4))
    if (!m_impl->pixelReadbackBuffer) {
        m_impl->pixelReadbackBuffer = std::make_unique<GpuBuffer>(
            m_impl->contextAdapter->GetAllocator(),
            16,  // sizeof(float) * 4
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_TO_CPU  // CPU-readable staging buffer
        );
    }

    // Allocate command buffer for copy operation
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_impl->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(m_impl->device, &allocInfo, &cmd) != VK_SUCCESS) {
        return Result<glm::vec4, String>::Err("Failed to allocate command buffer");
    }

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition outputImage to TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;  // outputImage is kept in GENERAL
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_impl->outputImage->GetImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    // Copy single pixel to staging buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;    // Tightly packed
    region.bufferImageHeight = 0;  // Tightly packed
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {static_cast<i32>(x), static_cast<i32>(y), 0};
    region.imageExtent = {1, 1, 1};  // Single pixel

    vkCmdCopyImageToBuffer(
        cmd,
        m_impl->outputImage->GetImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_impl->pixelReadbackBuffer->GetHandle(),
        1, &region
    );

    // Transition outputImage back to GENERAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    // End and submit command buffer
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(m_impl->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_impl->graphicsQueue);  // Wait for copy to complete

    // Free command buffer
    vkFreeCommandBuffers(m_impl->device, m_impl->commandPool, 1, &cmd);

    // Map staging buffer and read pixel value
    void* mappedData = m_impl->pixelReadbackBuffer->Map();
    if (mappedData == nullptr) {
        return Result<glm::vec4, String>::Err("Failed to map pixel readback buffer");
    }

    f32* pixelData = static_cast<f32*>(mappedData);
    glm::vec4 result(pixelData[0], pixelData[1], pixelData[2], pixelData[3]);

    m_impl->pixelReadbackBuffer->Unmap();

    return result;
}

Result<Image, String> ExternalRenderContext::CaptureScreenshot() {
    if (!m_impl->isReady || !m_impl->outputImage) {
        return Result<Image, String>::Err("Render context not ready");
    }

    // Read back entire outputImage using CommandHelper
    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        *m_impl->contextAdapter,
        m_impl->outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        m_impl->width,
        m_impl->height
    );

    // Create Image from pixel data
    Image screenshot(m_impl->width, m_impl->height, 4);  // RGBA
    screenshot.data = std::move(pixels);
    screenshot.channelNames = {"R", "G", "B", "A"};

    // Add metadata
    screenshot.metadata["spectral_mode"] = std::to_string(static_cast<int>(m_impl->spectralMode));
    screenshot.metadata["wavelength_nm"] = std::to_string(m_impl->wavelength_nm);
    screenshot.metadata["accumulated_samples"] = std::to_string(m_impl->accumulatedSamples);
    screenshot.metadata["spp_target"] = std::to_string(m_impl->spp);

    return std::move(screenshot);
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void ExternalRenderContext::BuildAccelerationStructures() {
    if (!m_impl->scene) return;

    QL_LOG_INFO("Building acceleration structures...");

    // Clear existing
    m_impl->blasList.clear();
    m_impl->tlas.reset();
    m_impl->globalVertexBuffer.reset();
    m_impl->globalIndexBuffer.reset();
    m_impl->globalNormalBuffer.reset();
    m_impl->globalUVBuffer.reset();
    m_impl->globalTangentBuffer.reset();
    m_impl->instanceGeometryBuffer.reset();

    // ========================================================================
    // Phase 1: Calculate total sizes and offsets for merged geometry buffers
    // ========================================================================

    struct BlasGeometryOffset {
        u32 vertexOffset;
        u32 indexOffset;
        u32 normalOffset;
        u32 uvOffset;
        u32 tangentOffset;
        u32 materialId;
    };

    std::vector<BlasGeometryOffset> blasOffsets;
    u32 totalVertices = 0;
    u32 totalIndices = 0;
    u32 totalNormals = 0;
    u32 totalUVs = 0;
    u32 totalTangents = 0;

    // First pass: calculate offsets for each primitive
    for (const auto& mesh : m_impl->scene->meshes) {
        for (const auto& primitive : mesh.primitives) {
            BlasGeometryOffset offset;
            offset.vertexOffset = totalVertices;
            offset.indexOffset = totalIndices;
            offset.normalOffset = totalNormals;
            offset.uvOffset = totalUVs;
            offset.tangentOffset = totalTangents;
            offset.materialId = primitive.materialId;
            blasOffsets.push_back(offset);

            totalVertices += static_cast<u32>(primitive.positions.size());
            totalIndices += static_cast<u32>(primitive.indices.size());
            totalNormals += static_cast<u32>(primitive.normals.empty() ?
                primitive.positions.size() : primitive.normals.size());
            totalUVs += static_cast<u32>(primitive.uvs.empty() ?
                primitive.positions.size() : primitive.uvs.size());
            totalTangents += static_cast<u32>(primitive.tangents.empty() ?
                primitive.positions.size() : primitive.tangents.size());
        }
    }

    QL_LOG_INFO("  Merged geometry: {} vertices, {} indices, {} normals, {} UVs, {} tangents",
                totalVertices, totalIndices, totalNormals, totalUVs, totalTangents);

    // ========================================================================
    // Phase 2: Create merged CPU-side data arrays
    // ========================================================================

    std::vector<glm::vec3> mergedVertices(totalVertices);
    std::vector<u32> mergedIndices(totalIndices);
    std::vector<glm::vec3> mergedNormals(totalNormals);
    std::vector<glm::vec2> mergedUVs(totalUVs);
    std::vector<glm::vec4> mergedTangents(totalTangents);

    size_t blasIdx = 0;
    for (const auto& mesh : m_impl->scene->meshes) {
        for (const auto& primitive : mesh.primitives) {
            const BlasGeometryOffset& offset = blasOffsets[blasIdx];

            // Copy vertices
            std::copy(primitive.positions.begin(), primitive.positions.end(),
                     mergedVertices.begin() + offset.vertexOffset);

            // Copy indices (adjust by vertex offset for global indexing)
            for (size_t i = 0; i < primitive.indices.size(); ++i) {
                mergedIndices[offset.indexOffset + i] = primitive.indices[i];
            }

            // Copy normals (or generate fallback)
            if (!primitive.normals.empty()) {
                std::copy(primitive.normals.begin(), primitive.normals.end(),
                         mergedNormals.begin() + offset.normalOffset);
            } else {
                // Loaders should have generated normals - this is a fallback for edge cases
                QL_LOG_WARN("ExternalRenderContext: Primitive has no normals after loading - using fallback up vector");
                std::fill(mergedNormals.begin() + offset.normalOffset,
                         mergedNormals.begin() + offset.normalOffset + primitive.positions.size(),
                         glm::vec3(0.0f, 1.0f, 0.0f));
            }

            // Copy UVs (or generate fallback)
            if (!primitive.uvs.empty()) {
                std::copy(primitive.uvs.begin(), primitive.uvs.end(),
                         mergedUVs.begin() + offset.uvOffset);
            } else {
                std::fill(mergedUVs.begin() + offset.uvOffset,
                         mergedUVs.begin() + offset.uvOffset + primitive.positions.size(),
                         glm::vec2(0.0f, 0.0f));
            }

            // Copy tangents (or generate fallback)
            if (!primitive.tangents.empty()) {
                std::copy(primitive.tangents.begin(), primitive.tangents.end(),
                         mergedTangents.begin() + offset.tangentOffset);
            } else {
                std::fill(mergedTangents.begin() + offset.tangentOffset,
                         mergedTangents.begin() + offset.tangentOffset + primitive.positions.size(),
                         glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
            }

            ++blasIdx;
        }
    }

    // ========================================================================
    // Phase 2.5: Validate merged buffer data (diagnostic logging)
    // ========================================================================

    QL_LOG_DEBUG("=== Buffer Merge Validation ===");
    QL_LOG_DEBUG("  Total vertices: {}, Total indices: {}", totalVertices, totalIndices);

    // Validate triangles and compute geometric normals
    size_t numTriangles = totalIndices / 3;
    QL_LOG_DEBUG("  Total triangles: {}", numTriangles);

    for (size_t triIdx = 0; triIdx < std::min(numTriangles, size_t(12)); ++triIdx) {
        u32 idx0 = mergedIndices[triIdx * 3 + 0];
        u32 idx1 = mergedIndices[triIdx * 3 + 1];
        u32 idx2 = mergedIndices[triIdx * 3 + 2];

        // Validate index bounds
        if (idx0 >= totalVertices || idx1 >= totalVertices || idx2 >= totalVertices) {
            QL_LOG_ERROR("  Triangle {}: INVALID INDICES [{}, {}, {}] (max={})",
                        triIdx, idx0, idx1, idx2, totalVertices - 1);
            continue;
        }

        glm::vec3 v0 = mergedVertices[idx0];
        glm::vec3 v1 = mergedVertices[idx1];
        glm::vec3 v2 = mergedVertices[idx2];

        // Compute geometric normal
        glm::vec3 e0 = v1 - v0;
        glm::vec3 e1 = v2 - v0;
        glm::vec3 geoNormal = glm::normalize(glm::cross(e0, e1));

        // Check if axis-aligned (for cube validation)
        bool axisAligned = (std::abs(std::abs(geoNormal.x) - 1.0f) < 0.01f &&
                           std::abs(geoNormal.y) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                          (std::abs(geoNormal.x) < 0.01f &&
                           std::abs(std::abs(geoNormal.y) - 1.0f) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                          (std::abs(geoNormal.x) < 0.01f && std::abs(geoNormal.y) < 0.01f &&
                           std::abs(std::abs(geoNormal.z) - 1.0f) < 0.01f);

        QL_LOG_DEBUG("  Triangle {}: indices=[{}, {}, {}], geoNormal=({:.3f}, {:.3f}, {:.3f}) {}",
                    triIdx, idx0, idx1, idx2,
                    geoNormal.x, geoNormal.y, geoNormal.z,
                    axisAligned ? "FINE" : "SKEWED");
    }

    // Also log stored normals for comparison
    if (!mergedNormals.empty()) {
        QL_LOG_DEBUG("  --- Stored vertex normals (first 8) ---");
        for (size_t i = 0; i < std::min(size_t(8), mergedNormals.size()); ++i) {
            QL_LOG_DEBUG("    normal[{}] = ({:.3f}, {:.3f}, {:.3f})",
                        i, mergedNormals[i].x, mergedNormals[i].y, mergedNormals[i].z);
        }
    }

    QL_LOG_DEBUG("=== End Buffer Merge Validation ===");

    // ========================================================================
    // Phase 3: Create merged GPU buffers and upload data
    // ========================================================================

    VmaAllocator allocator = m_impl->contextAdapter->GetAllocator();

    m_impl->globalVertexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        totalVertices * sizeof(glm::vec3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->globalVertexBuffer->Upload(mergedVertices.data(), mergedVertices.size() * sizeof(glm::vec3));

    m_impl->globalIndexBuffer = std::make_unique<GpuBuffer>(
        allocator,
        totalIndices * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->globalIndexBuffer->Upload(mergedIndices.data(), mergedIndices.size() * sizeof(u32));

    m_impl->globalNormalBuffer = std::make_unique<GpuBuffer>(
        allocator,
        totalNormals * sizeof(glm::vec3),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->globalNormalBuffer->Upload(mergedNormals.data(), mergedNormals.size() * sizeof(glm::vec3));

    m_impl->globalUVBuffer = std::make_unique<GpuBuffer>(
        allocator,
        totalUVs * sizeof(glm::vec2),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->globalUVBuffer->Upload(mergedUVs.data(), mergedUVs.size() * sizeof(glm::vec2));

    m_impl->globalTangentBuffer = std::make_unique<GpuBuffer>(
        allocator,
        totalTangents * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->globalTangentBuffer->Upload(mergedTangents.data(), mergedTangents.size() * sizeof(glm::vec4));

    QL_LOG_INFO("  Created merged geometry buffers");

    // ========================================================================
    // Phase 4: Build BLAS for each primitive (using original per-BLAS buffers)
    // ========================================================================

    for (const auto& mesh : m_impl->scene->meshes) {
        for (const auto& primitive : mesh.primitives) {
            m_impl->blasList.emplace_back(
                std::make_unique<BLAS>(*m_impl->contextAdapter, primitive)
            );
        }
    }

    // Build TLAS
    m_impl->tlas = std::make_unique<TLAS>(*m_impl->contextAdapter);

    // ========================================================================
    // Phase 5: Build instance geometry info and add TLAS instances
    // ========================================================================

    // Build mapping from mesh index to BLAS starting index
    std::vector<size_t> meshToBlasStart;
    meshToBlasStart.reserve(m_impl->scene->meshes.size());
    size_t blasStart = 0;
    for (const auto& mesh : m_impl->scene->meshes) {
        meshToBlasStart.push_back(blasStart);
        blasStart += mesh.primitives.size();
    }

    // Collect instance geometry info (one per TLAS instance)
    std::vector<InstanceGeometryInfo> instanceInfos;

    // Execute builds on GPU and add TLAS instances
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
        // Build all BLAS
        for (auto& blas : m_impl->blasList) {
            blas->Build(cmd);
        }

        // Add instances to TLAS and record geometry info
        for (const auto& node : m_impl->scene->nodes) {
            const Mesh& mesh = m_impl->scene->meshes[node.meshIndex];
            size_t blasBase = meshToBlasStart[node.meshIndex];

            for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
                size_t globalBlasIdx = blasBase + primIdx;
                const BlasGeometryOffset& geoOffset = blasOffsets[globalBlasIdx];

                // Create instance geometry info
                InstanceGeometryInfo info{};
                info.vertexOffset = geoOffset.vertexOffset;
                info.indexOffset = geoOffset.indexOffset;
                info.normalOffset = geoOffset.normalOffset;
                info.uvOffset = geoOffset.uvOffset;
                info.tangentOffset = geoOffset.tangentOffset;
                info.materialId = geoOffset.materialId;
                info.pad[0] = 0;
                info.pad[1] = 0;
                instanceInfos.push_back(info);

                // Log instance geometry info for debugging
                QL_LOG_DEBUG("  Instance {}: vertexOff={}, indexOff={}, normalOff={}, uvOff={}, tangentOff={}, matId={}",
                            instanceInfos.size() - 1,
                            info.vertexOffset, info.indexOffset, info.normalOffset,
                            info.uvOffset, info.tangentOffset, info.materialId);

                // Get material's doubleSided property for hardware backface culling control
                bool doubleSided = true;  // Default: disable culling (backward compatible)
                if (static_cast<size_t>(geoOffset.materialId) < m_impl->scene->materials.size()) {
                    doubleSided = m_impl->scene->materials[geoOffset.materialId].doubleSided;
                }

                // Add TLAS instance
                // Note: instanceCustomIndex is now the TLAS instance index (for InstanceGeometryInfo lookup)
                // Material ID is stored in InstanceGeometryInfo instead
                m_impl->tlas->AddInstance(
                    *m_impl->blasList[globalBlasIdx],
                    static_cast<u32>(instanceInfos.size() - 1),  // Instance index for geometry info lookup
                    node.transform,
                    doubleSided
                );
            }
        }

        m_impl->tlas->Build(cmd);
    });

    // ========================================================================
    // Phase 6: Upload instance geometry info buffer
    // ========================================================================

    if (!instanceInfos.empty()) {
        m_impl->instanceGeometryBuffer = std::make_unique<GpuBuffer>(
            allocator,
            instanceInfos.size() * sizeof(InstanceGeometryInfo),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        m_impl->instanceGeometryBuffer->Upload(instanceInfos.data(),
            instanceInfos.size() * sizeof(InstanceGeometryInfo));

        QL_LOG_INFO("  Created instance geometry info buffer ({} instances)", instanceInfos.size());
    }

    QL_LOG_INFO("  Built {} BLAS, 1 TLAS", m_impl->blasList.size());
}

void ExternalRenderContext::UpdateGpuResources() {
    if (!m_impl->scene) return;

    QL_LOG_INFO("Updating GPU resources...");

    // Upload materials
    std::vector<MaterialDataCPU> materialData;
    materialData.reserve(m_impl->scene->materials.size());

    for (const auto& mat : m_impl->scene->materials) {
        MaterialDataCPU cpuMat{};
        cpuMat.baseColorFactor = mat.baseColorFactor;
        cpuMat.baseColorTextureIndex = mat.baseColorTextureIndex;
        cpuMat.metallicFactor = mat.metallicFactor;
        cpuMat.roughnessFactor = mat.roughnessFactor;
        cpuMat.metallicRoughnessTextureIndex = mat.metallicRoughnessTextureIndex;
        cpuMat.normalTextureIndex = mat.normalTextureIndex;
        cpuMat.normalScale = mat.normalScale;
        cpuMat.doubleSided = mat.doubleSided ? 1u : 0u;
        cpuMat.emissiveFactor = mat.emissiveFactor;
        cpuMat.emissiveTextureIndex = mat.emissiveTextureIndex;
        cpuMat.alphaMode = static_cast<u32>(mat.alphaMode);
        cpuMat.alphaCutoff = mat.alphaCutoff;
        cpuMat.spectralAlbedo = mat.spectralAlbedo;
        cpuMat.spectralReflectanceCurveIndex = -1;
        cpuMat.irEmissivity = 0.0f;
        cpuMat.irTransmittance = 0.0f;
        cpuMat.irTemperature_K = mat.irTemperature_K;
        cpuMat.complexRefractiveIndexIndex = -1;

        // Transmission properties (KHR_materials_transmission + KHR_materials_volume)
        cpuMat.ior = mat.ior;
        cpuMat.transmission = mat.transmission;
        cpuMat.transmissionTextureIndex = mat.transmissionTextureIndex;
        cpuMat._padding1 = 0.0f;
        cpuMat.attenuationColor = mat.attenuationColor;
        cpuMat.attenuationDistance = mat.attenuationDistance;
        cpuMat.thicknessFactor = mat.thicknessFactor;
        cpuMat.thicknessTextureIndex = mat.thicknessTextureIndex;
        cpuMat.dispersion = mat.dispersion;
        cpuMat._padding2 = 0.0f;

        // Volume properties (fog, smoke, SSS)
        cpuMat.volumeDensity = mat.volumeDensity;
        cpuMat.scatteringCoeff = mat.scatteringCoeff;
        cpuMat.absorptionCoeff = mat.absorptionCoeff;
        cpuMat.phaseG = mat.phaseG;

        materialData.push_back(cpuMat);
    }

    if (!materialData.empty()) {
        m_impl->materialBuffer = std::make_unique<GpuBuffer>(
            m_impl->contextAdapter->GetAllocator(),
            materialData.size() * sizeof(MaterialDataCPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        m_impl->materialBuffer->Upload(materialData.data(), materialData.size() * sizeof(MaterialDataCPU));
    }

    QL_LOG_INFO("  GPU resources updated");
}

void ExternalRenderContext::CreateDummyBuffers() {
    auto allocator = m_impl->contextAdapter->GetAllocator();

    // Create dummy spectral curves buffer
    struct DummySpectralCurve {
        f32 data[272 / sizeof(f32)];
    } dummyCurve{};

    m_impl->spectralCurvesBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummySpectralCurve),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->spectralCurvesBuffer->Upload(&dummyCurve, sizeof(DummySpectralCurve));

    // Create dummy CRI buffer
    struct DummyCRI {
        f32 data[528 / sizeof(f32)];
    } dummyCri{};

    m_impl->criBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummyCRI),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->criBuffer->Upload(&dummyCri, sizeof(DummyCRI));

    // Create dummy solar LUT buffer
    struct DummySolarLUT {
        f32 data[544 / sizeof(f32)];
    } dummySolar{};

    m_impl->solarLutBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummySolarLUT),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->solarLutBuffer->Upload(&dummySolar, sizeof(DummySolarLUT));

    // Create dummy atmospheric buffer
    struct DummyAtmospheric {
        f32 data[64 / sizeof(f32)];
    } dummyAtmo{};

    m_impl->atmosphericBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummyAtmospheric),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->atmosphericBuffer->Upload(&dummyAtmo, sizeof(DummyAtmospheric));

    // Create CIE 1931 CMF LUT buffer (required for VIS_Fused mode)
    // Use hardcoded CIE data from CIE_CMF_Data.hpp (401 samples, 380-780nm at 1nm)
    std::vector<glm::vec4> cieCmfData;
    cieCmfData.reserve(CIE_CMF_LUT_SIZE);
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        cieCmfData.emplace_back(
            CIE_1931_2DEG[i][0],  // x_bar
            CIE_1931_2DEG[i][1],  // y_bar
            CIE_1931_2DEG[i][2],  // z_bar
            0.0f                   // padding for 16-byte alignment
        );
    }

    m_impl->cieCmfBuffer = std::make_unique<GpuBuffer>(
        allocator,
        cieCmfData.size() * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->cieCmfBuffer->Upload(cieCmfData.data(), cieCmfData.size() * sizeof(glm::vec4));

    QL_LOG_DEBUG("  CIE CMF LUT created ({} samples)", CIE_CMF_LUT_SIZE);
}

void ExternalRenderContext::CreateBRDFLut() {
    QL_LOG_INFO("Creating BRDF LUT...");

    auto allocator = m_impl->contextAdapter->GetAllocator();
    auto device = m_impl->contextAdapter->GetDevice();

    // Try to load from cache first
    const String brdfCachePath = "assets/luts/brdf_lut_512_ggx.bin";
    BRDFLutGenerator::Config brdfConfig;
    brdfConfig.resolution = 512;
    brdfConfig.sampleCount = 1024;

    Image brdfLutImage;
    auto cachedLut = BRDFLutGenerator::LoadFromBinary(brdfCachePath, &brdfConfig);
    if (cachedLut.has_value()) {
        brdfLutImage = std::move(cachedLut.value());
    } else {
        brdfLutImage = BRDFLutGenerator::Generate(brdfConfig);
        BRDFLutGenerator::SaveToBinary(brdfCachePath, brdfLutImage, brdfConfig);
    }

    // Create GPU texture
    m_impl->brdfLutTexture = std::make_unique<GpuImage>(
        allocator,
        device,
        512, 512,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Transition to TRANSFER_DST
    TransitionImageLayoutImmediate(
        m_impl->brdfLutTexture->GetImage(),
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );

    // Upload BRDF LUT data
    std::vector<f32> lutData(512 * 512 * 2);
    for (u32 y = 0; y < 512; ++y) {
        for (u32 x = 0; x < 512; ++x) {
            u32 idx = (y * 512 + x) * 2;
            lutData[idx + 0] = brdfLutImage(x, y, 0);
            lutData[idx + 1] = brdfLutImage(x, y, 1);
        }
    }

    GpuBuffer stagingBuffer(
        allocator,
        lutData.size() * sizeof(f32),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    stagingBuffer.Upload(lutData.data(), lutData.size() * sizeof(f32));

    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {512, 512, 1};

        vkCmdCopyBufferToImage(
            cmd,
            stagingBuffer.GetHandle(),
            m_impl->brdfLutTexture->GetImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );
    });

    // Transition to SHADER_READ_ONLY
    TransitionImageLayoutImmediate(
        m_impl->brdfLutTexture->GetImage(),
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    // Create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    vkCreateSampler(device, &samplerInfo, nullptr, &m_impl->brdfLutSampler);
}

void ExternalRenderContext::CreateFallbackEnvMap() {
    QL_LOG_INFO("Creating fallback environment map...");

    auto allocator = m_impl->contextAdapter->GetAllocator();
    auto device = m_impl->contextAdapter->GetDevice();

    constexpr u32 envMapSize = 256;
    constexpr u32 envMapMips = 5;

    m_impl->envMapImage = std::make_unique<GpuImage>(
        allocator,
        device,
        envMapSize, envMapSize,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        envMapMips,
        6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        VK_IMAGE_VIEW_TYPE_CUBE
    );

    // Transition to TRANSFER_DST
    CommandHelper::TransitionImageLayoutImmediate(
        *m_impl->contextAdapter,
        m_impl->envMapImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        envMapMips,
        6
    );

    // Create sky-blue fallback data and upload for all faces/mips
    constexpr f32 skyColor[4] = {0.5f, 0.7f, 1.0f, 1.0f};

    for (u32 mip = 0; mip < envMapMips; ++mip) {
        u32 mipSize = envMapSize >> mip;
        if (mipSize == 0) mipSize = 1;

        std::vector<f32> pixelData(mipSize * mipSize * 4);
        for (u32 i = 0; i < mipSize * mipSize; ++i) {
            pixelData[i * 4 + 0] = skyColor[0];
            pixelData[i * 4 + 1] = skyColor[1];
            pixelData[i * 4 + 2] = skyColor[2];
            pixelData[i * 4 + 3] = skyColor[3];
        }

        GpuBuffer stagingBuffer(
            allocator,
            pixelData.size() * sizeof(f32),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        stagingBuffer.Upload(pixelData.data(), pixelData.size() * sizeof(f32));

        for (u32 face = 0; face < 6; ++face) {
            CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = mip;
                region.imageSubresource.baseArrayLayer = face;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {mipSize, mipSize, 1};

                vkCmdCopyBufferToImage(
                    cmd,
                    stagingBuffer.GetHandle(),
                    m_impl->envMapImage->GetImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region
                );
            });
        }
    }

    // Transition to SHADER_READ_ONLY
    CommandHelper::TransitionImageLayoutImmediate(
        *m_impl->contextAdapter,
        m_impl->envMapImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        envMapMips,
        6
    );
}

void ExternalRenderContext::CreatePipeline() {
    QL_LOG_INFO("Creating ray tracing pipeline...");

    // Load or create pipeline cache for faster shader compilation
    if (m_impl->pipelineCache == VK_NULL_HANDLE) {
        m_impl->pipelineCache = RayTracingPipeline::LoadPipelineCache(
            *m_impl->contextAdapter,
            m_impl->pipelineCachePath
        );
    }

    // Create pipeline using context adapter with cache
    m_impl->pipeline = std::make_unique<RayTracingPipeline>(
        *m_impl->contextAdapter,
        "raygen.spv",
        "closesthit.spv",
        "miss.spv",
        m_impl->pipelineCache
    );

    // Bind resources
    m_impl->pipeline->BindOutputImage(*m_impl->outputImage);
    m_impl->pipeline->BindAccelerationStructure(m_impl->tlas->GetHandle());
    m_impl->pipeline->BindLUTBuffer(*m_impl->lightingParamsBuffer);

    // Bind merged global geometry buffers (instead of per-BLAS buffers)
    if (m_impl->globalVertexBuffer && m_impl->globalIndexBuffer) {
        m_impl->pipeline->BindGeometryBuffers(
            *m_impl->globalVertexBuffer,
            *m_impl->globalIndexBuffer,
            m_impl->globalUVBuffer.get()
        );
        if (m_impl->globalTangentBuffer) {
            m_impl->pipeline->BindTangentBuffer(*m_impl->globalTangentBuffer);
        }
        if (m_impl->globalNormalBuffer) {
            m_impl->pipeline->BindNormalBuffer(*m_impl->globalNormalBuffer);
        }
        if (m_impl->instanceGeometryBuffer) {
            m_impl->pipeline->BindInstanceGeometryBuffer(*m_impl->instanceGeometryBuffer);
        }
    }

    // Bind materials
    if (m_impl->materialBuffer) {
        m_impl->pipeline->BindMaterialBuffer(*m_impl->materialBuffer);
    }

    // Bind textures
    m_impl->pipeline->BindTextures(
        m_impl->textureManager->GetImageViews(),
        m_impl->textureManager->GetSamplers()
    );

    // Bind IBL
    m_impl->pipeline->BindPrefilteredEnvMap(m_impl->envMapImage->GetView());
    m_impl->pipeline->BindBRDFLut(m_impl->brdfLutTexture->GetView(), m_impl->brdfLutSampler);

    // Bind optional buffers
    m_impl->pipeline->BindSpectralCurvesBuffer(m_impl->spectralCurvesBuffer.get());
    m_impl->pipeline->BindComplexRefractiveIndexBuffer(m_impl->criBuffer.get());
    m_impl->pipeline->BindSolarSpectralLUT(m_impl->solarLutBuffer.get());
    m_impl->pipeline->BindAtmosphericParams(m_impl->atmosphericBuffer.get());

    // Bind CIE CMF LUT (required for VIS_Fused spectral mode)
    if (m_impl->cieCmfBuffer) {
        m_impl->pipeline->BindCIE_CMF_LUT(*m_impl->cieCmfBuffer);
    }

    QL_LOG_INFO("  Ray tracing pipeline created and bound");
}

void ExternalRenderContext::TransitionImageLayoutImmediate(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout) {

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_impl->commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_impl->device, &allocInfo, &cmd);

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Image memory barrier
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // End and submit
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(m_impl->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_impl->graphicsQueue);

    vkFreeCommandBuffers(m_impl->device, m_impl->commandPool, 1, &cmd);

    (void)format;  // Format used for barrier determination in more complex cases
}

// ============================================================================
// Atmospheric Configuration
// ============================================================================

void ExternalRenderContext::SetAtmosphericConfig(const AtmosphericConfig& config) {
    m_impl->atmosphericConfig = config;
    m_impl->atmosphericDirty = true;

    // Upload to GPU immediately if buffer exists
    if (m_impl->atmosphericBuffer) {
        AtmosphericParamsGPU gpuParams = config.ToGPU();
        m_impl->atmosphericBuffer->Upload(&gpuParams, sizeof(AtmosphericParamsGPU));
        m_impl->atmosphericDirty = false;
    }

    ResetAccumulation();
    QL_LOG_DEBUG("Atmospheric config updated: {}",
                 config.IsEnabled() ? "enabled" : "disabled");
}

void ExternalRenderContext::SetAtmosphericPreset(const String& preset) {
    AtmosphericConfig config;

    if (preset == "clear_day") {
        config = AtmosphericConfig::ClearDay();
    } else if (preset == "hazy") {
        config = AtmosphericConfig::Hazy();
    } else if (preset == "polluted_urban") {
        config = AtmosphericConfig::PollutedUrban();
    } else if (preset == "mountain_top") {
        config = AtmosphericConfig::MountainTop();
    } else if (preset == "mars") {
        config = AtmosphericConfig::Mars();
    } else {
        config = AtmosphericConfig::Disabled();
    }

    SetAtmosphericConfig(config);
    QL_LOG_INFO("Atmospheric preset set to: {}", preset);
}

const AtmosphericConfig& ExternalRenderContext::GetAtmosphericConfig() const {
    return m_impl->atmosphericConfig;
}

// ============================================================================
// Environment Map (IBL)
// ============================================================================

Result<void, String> ExternalRenderContext::LoadEnvironmentMap(const String& hdrPath) {
    QL_LOG_INFO("Loading environment map: {}", hdrPath);

    // Check if file exists
    if (!ImageIO::FileExists(hdrPath)) {
        return Result<void, String>::Err("Environment map file not found: " + hdrPath);
    }

    // Load HDR image
    auto equirectOpt = ImageIO::ReadImage(hdrPath);
    if (!equirectOpt.has_value()) {
        return Result<void, String>::Err("Failed to load HDR image: " + hdrPath);
    }

    Image& equirect = equirectOpt.value();
    QL_LOG_INFO("  HDR image loaded: {}x{}, {} channels",
                equirect.width, equirect.height, equirect.channels);

    // Convert equirectangular to cubemap using CPU-based conversion
    constexpr u32 envMapSize = 512;
    constexpr u32 envMapMips = 8;

    // Helper: Get cubemap face direction
    auto cubemapFaceDirection = [](u32 face, f32 u, f32 v) -> glm::vec3 {
        f32 uc = 2.0f * u - 1.0f;
        f32 vc = 2.0f * v - 1.0f;

        switch (face) {
            case 0: return glm::normalize(glm::vec3( 1.0f,   -vc,   -uc));  // +X
            case 1: return glm::normalize(glm::vec3(-1.0f,   -vc,    uc));  // -X
            case 2: return glm::normalize(glm::vec3(   uc,  1.0f,    vc));  // +Y
            case 3: return glm::normalize(glm::vec3(   uc, -1.0f,   -vc));  // -Y
            case 4: return glm::normalize(glm::vec3(   uc,   -vc,  1.0f));  // +Z
            case 5: return glm::normalize(glm::vec3(  -uc,   -vc, -1.0f));  // -Z
            default: return glm::vec3(0.0f);
        }
    };

    // Helper: Sample equirectangular map
    auto sampleEquirect = [&equirect](const glm::vec3& dir) -> glm::vec3 {
        f32 theta = std::atan2(dir.z, dir.x);
        f32 phi = std::asin(glm::clamp(dir.y, -1.0f, 1.0f));

        f32 u = (theta + glm::pi<f32>()) / (2.0f * glm::pi<f32>());
        f32 v = (phi + glm::pi<f32>() / 2.0f) / glm::pi<f32>();

        u32 width = equirect.width;
        u32 height = equirect.height;

        f32 fx = u * static_cast<f32>(width - 1);
        f32 fy = v * static_cast<f32>(height - 1);

        u32 x0 = static_cast<u32>(fx) % width;
        u32 y0 = static_cast<u32>(fy) % height;
        u32 x1 = (x0 + 1) % width;
        u32 y1 = std::min(y0 + 1, height - 1);

        f32 wx = fx - std::floor(fx);
        f32 wy = fy - std::floor(fy);

        glm::vec3 c00(equirect(x0, y0, 0), equirect(x0, y0, 1), equirect(x0, y0, 2));
        glm::vec3 c10(equirect(x1, y0, 0), equirect(x1, y0, 1), equirect(x1, y0, 2));
        glm::vec3 c01(equirect(x0, y1, 0), equirect(x0, y1, 1), equirect(x0, y1, 2));
        glm::vec3 c11(equirect(x1, y1, 0), equirect(x1, y1, 1), equirect(x1, y1, 2));

        glm::vec3 c0 = c00 * (1.0f - wx) + c10 * wx;
        glm::vec3 c1 = c01 * (1.0f - wx) + c11 * wx;

        return c0 * (1.0f - wy) + c1 * wy;
    };

    // Convert equirectangular to cubemap faces
    QL_LOG_INFO("  Converting equirectangular to cubemap ({}x{} per face)...", envMapSize, envMapSize);
    std::vector<Image> cubemapFaces(6);
    for (u32 face = 0; face < 6; ++face) {
        cubemapFaces[face] = Image(envMapSize, envMapSize, 3);
        for (u32 y = 0; y < envMapSize; ++y) {
            for (u32 x = 0; x < envMapSize; ++x) {
                f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(envMapSize);
                f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(envMapSize);
                glm::vec3 dir = cubemapFaceDirection(face, u, v);
                glm::vec3 color = sampleEquirect(dir);
                cubemapFaces[face](x, y, 0) = color.r;
                cubemapFaces[face](x, y, 1) = color.g;
                cubemapFaces[face](x, y, 2) = color.b;
            }
        }
    }

    // Wait for GPU
    vkDeviceWaitIdle(m_impl->device);

    auto allocator = m_impl->contextAdapter->GetAllocator();

    // Recreate GPU environment map
    m_impl->envMapImage.reset();
    m_impl->envMapImage = std::make_unique<GpuImage>(
        allocator,
        m_impl->device,
        envMapSize, envMapSize,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        envMapMips,
        6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        VK_IMAGE_VIEW_TYPE_CUBE
    );

    // Transition to TRANSFER_DST
    CommandHelper::TransitionImageLayoutImmediate(
        *m_impl->contextAdapter,
        m_impl->envMapImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        envMapMips,
        6
    );

    // Upload base mip level (mip 0)
    QL_LOG_INFO("  Uploading cubemap to GPU...");
    for (u32 face = 0; face < 6; ++face) {
        const Image& faceImage = cubemapFaces[face];

        std::vector<f32> pixelData(envMapSize * envMapSize * 4);
        for (u32 y = 0; y < envMapSize; ++y) {
            for (u32 x = 0; x < envMapSize; ++x) {
                u32 idx = (y * envMapSize + x) * 4;
                pixelData[idx + 0] = faceImage(x, y, 0);
                pixelData[idx + 1] = faceImage(x, y, 1);
                pixelData[idx + 2] = faceImage(x, y, 2);
                pixelData[idx + 3] = 1.0f;
            }
        }

        GpuBuffer stagingBuffer(
            allocator,
            pixelData.size() * sizeof(f32),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        stagingBuffer.Upload(pixelData.data(), pixelData.size() * sizeof(f32));

        CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = face;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {envMapSize, envMapSize, 1};

            vkCmdCopyBufferToImage(
                cmd,
                stagingBuffer.GetHandle(),
                m_impl->envMapImage->GetImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region
            );
        });
    }

    // Generate simple mipmaps (box filter)
    QL_LOG_INFO("  Generating mipmap chain...");
    for (u32 mip = 1; mip < envMapMips; ++mip) {
        u32 mipSize = envMapSize >> mip;
        if (mipSize == 0) mipSize = 1;
        u32 prevMipSize = envMapSize >> (mip - 1);

        for (u32 face = 0; face < 6; ++face) {
            std::vector<f32> mipData(mipSize * mipSize * 4);
            const Image& baseFace = cubemapFaces[face];

            for (u32 y = 0; y < mipSize; ++y) {
                for (u32 x = 0; x < mipSize; ++x) {
                    glm::vec4 sum(0.0f);
                    u32 srcX = x * 2;
                    u32 srcY = y * 2;
                    u32 count = 0;
                    for (u32 dy = 0; dy < 2 && (srcY + dy) < prevMipSize; ++dy) {
                        for (u32 dx = 0; dx < 2 && (srcX + dx) < prevMipSize; ++dx) {
                            // Sample from base face (approximation)
                            u32 sx = std::min((srcX + dx) * (envMapSize / prevMipSize), envMapSize - 1);
                            u32 sy = std::min((srcY + dy) * (envMapSize / prevMipSize), envMapSize - 1);
                            sum.r += baseFace(sx, sy, 0);
                            sum.g += baseFace(sx, sy, 1);
                            sum.b += baseFace(sx, sy, 2);
                            sum.a += 1.0f;
                            count++;
                        }
                    }
                    if (count > 0) {
                        sum /= static_cast<f32>(count);
                    }
                    u32 idx = (y * mipSize + x) * 4;
                    mipData[idx + 0] = sum.r;
                    mipData[idx + 1] = sum.g;
                    mipData[idx + 2] = sum.b;
                    mipData[idx + 3] = 1.0f;
                }
            }

            GpuBuffer stagingBuffer(
                allocator,
                mipData.size() * sizeof(f32),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU
            );
            stagingBuffer.Upload(mipData.data(), mipData.size() * sizeof(f32));

            CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = mip;
                region.imageSubresource.baseArrayLayer = face;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {mipSize, mipSize, 1};

                vkCmdCopyBufferToImage(
                    cmd,
                    stagingBuffer.GetHandle(),
                    m_impl->envMapImage->GetImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region
                );
            });
        }
    }

    // Transition to SHADER_READ_ONLY
    CommandHelper::TransitionImageLayoutImmediate(
        *m_impl->contextAdapter,
        m_impl->envMapImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        envMapMips,
        6
    );

    // Re-bind to pipeline if exists
    if (m_impl->pipeline) {
        m_impl->pipeline->BindPrefilteredEnvMap(m_impl->envMapImage->GetView());
    }

    m_impl->hasCustomEnvMap = true;
    ResetAccumulation();

    QL_LOG_INFO("Environment map loaded successfully: {}", hdrPath);
    return Result<void, String>::Ok();
}

bool ExternalRenderContext::HasEnvironmentMap() const {
    return m_impl->hasCustomEnvMap;
}

} // namespace quantiloom
