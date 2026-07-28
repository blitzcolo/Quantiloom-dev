/**
 * @file ExternalRenderContext.cpp
 * @brief Implementation of external Vulkan context injection API
 *
 * @author blitzcolo
 */

// VMA must be included BEFORE GpuBuffer.hpp/GpuImage.hpp to avoid enum redefinition
#include <vk_mem_alloc.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/RenderCore.hpp"
#include "VulkanContextAdapter.hpp"
#include "RayTracingPipeline.hpp"
#include "AccelerationStructure.hpp"
#include "GpuBuffer.hpp"
#include "GpuImage.hpp"
#include "TextureManager.hpp"
#include "CommandHelper.hpp"
#include "BRDFLutGenerator.hpp"
#include "renderer/LightingParams.hpp"
#include "MaterialGpuData.hpp"
#include "atmos/AtmosphereBaker.hpp"

#include "core/Log.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "core/SpectralData.hpp"
#include "io/GltfLoader.hpp"
#include "io/UsdLoader.hpp"
#include "io/ImageIO.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
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

    // GPU resources
    std::unique_ptr<GpuImage> outputImage;
    std::unique_ptr<GpuBuffer> lightingParamsBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    std::unique_ptr<GpuBuffer> spectralCurvesBuffer;
    std::unique_ptr<GpuBuffer> criBuffer;
    std::unique_ptr<GpuBuffer> solarLutBuffer;
    std::unique_ptr<GpuBuffer> atmosHeaderBuffer;  // AtmosNNHeaderGPU (binding 17)
    std::unique_ptr<GpuBuffer> atmosDataBuffer;    // Baked LUT blob (binding 20)
    std::unique_ptr<GpuBuffer> cieCmfBuffer;  // CIE 1931 CMF LUT for VIS_Fused mode (binding 19)

    // CRI management (CPU-side copy for rebuild when new entries are added)
    std::vector<ComplexRefractiveIndexGPU> criEntries;

    // Merged global geometry buffers (for multi-BLAS support)
    // All BLAS geometry data is merged into single global buffers
    // Shader uses InstanceGeometryInfo offsets to index correctly

    // IBL resources
    rendercore::BrdfLut brdfLut;
    rendercore::EnvironmentCubemap envMap;
    rendercore::SceneGeometry geometry;

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

    // NN atmosphere state (baked lazily before rendering when the key changes)
    AtmosphereNNConfig atmosphereConfig;              // CPU-side config (default: disabled)
    std::unique_ptr<AtmosModelPack> atmosModelPack;   // Loaded network packs
    uint64_t atmosBakeKey = 0;                        // 0 = nothing baked yet

    // Rebakes/uploads the NN atmosphere LUT when the bake key changed
    void UpdateAtmosphereNN();

    // Environment map state
    bool hasCustomEnvMap = false;         // True if LoadEnvironmentMap succeeded

    // Accumulation
    u32 accumulatedSamples = 0;
    u32 frameIndex = 0;

    // Random number generator for better sample distribution
    // Uses Mersenne Twister for high-quality randomness (matches CLI app)
    //
    // Seeded deterministically by default, and re-seeded whenever accumulation
    // restarts. It used to be seeded once from std::random_device, which made
    // every interactive render unreproducible: the same scene and camera drew
    // from wherever the generator had got to, so the image depended on session
    // history and could never be compared against the CLI's output.
    u32 samplingSeed = constants::DEFAULT_SAMPLING_SEED;  // 0 = nondeterministic
    std::mt19937 rng{constants::DEFAULT_SAMPLING_SEED};
    std::uniform_int_distribution<u32> randDist{0, std::numeric_limits<u32>::max()};

    // Restart the sampling sequence. Called from ResetAccumulation() so the
    // sequence and the accumulation it feeds always begin together.
    void ReseedRng() {
        rng.seed(samplingSeed != 0U ? samplingSeed : std::random_device{}());
    }

    // Statistics
    f32 lastFrameTimeMs = 0.0f;
    std::chrono::steady_clock::time_point frameStartTime;

    // Pixel readback buffer (for debug hover display)
    std::unique_ptr<GpuBuffer> pixelReadbackBuffer;

    // CLAHE display enhancement resources
    ExternalRenderContext::CLAHEParams claheParams;
    std::unique_ptr<GpuImage> displayImage;           // CLAHE-processed output for display
    std::unique_ptr<GpuBuffer> claheHistogramBuffer;  // Per-tile histograms
    std::unique_ptr<GpuBuffer> claheCdfBuffer;        // Per-tile CDFs
    std::unique_ptr<GpuBuffer> claheMinMaxBuffer;     // Per-tile min/max for normalization
    VkDescriptorSetLayout claheDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout clahePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool claheDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet claheDescriptorSet = VK_NULL_HANDLE;
    VkPipeline claheHistogramPipeline = VK_NULL_HANDLE;
    VkPipeline claheCdfPipeline = VK_NULL_HANDLE;
    VkPipeline claheApplyPipeline = VK_NULL_HANDLE;
    VkShaderModule claheHistogramShader = VK_NULL_HANDLE;
    VkShaderModule claheCdfShader = VK_NULL_HANDLE;
    VkShaderModule claheApplyShader = VK_NULL_HANDLE;
    bool claheInitialized = false;

    // Cached min/max values for CLAHE (computed after frame completion)
    f32 cachedImageMin = 0.0f;
    f32 cachedImageMax = 1.0f;
    bool hasCachedMinMax = false;

    // GPU Sensor simulation resources
    bool gpuSensorEnabled = false;
    SensorParams gpuSensorParams;
    std::unique_ptr<GpuImage> sensorImage;              // Sensor-processed output (noisy radiance)
    std::unique_ptr<GpuImage> sensorTempImage;          // Temporary image for multi-pass processing
    VkDescriptorSetLayout sensorDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout sensorPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool sensorDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet sensorDescriptorSet = VK_NULL_HANDLE;
    VkPipeline sensorRadianceToElectronsPipeline = VK_NULL_HANDLE;
    VkPipeline sensorPoissonNoisePipeline = VK_NULL_HANDLE;
    VkPipeline sensorPsfBlurHorizontalPipeline = VK_NULL_HANDLE;
    VkPipeline sensorPsfBlurVerticalPipeline = VK_NULL_HANDLE;
    VkPipeline sensorQuantizeToRadiancePipeline = VK_NULL_HANDLE;
    VkShaderModule sensorRadianceToElectronsShader = VK_NULL_HANDLE;
    VkShaderModule sensorPoissonNoiseShader = VK_NULL_HANDLE;
    VkShaderModule sensorPsfBlurHorizontalShader = VK_NULL_HANDLE;
    VkShaderModule sensorPsfBlurVerticalShader = VK_NULL_HANDLE;
    VkShaderModule sensorQuantizeToRadianceShader = VK_NULL_HANDLE;
    VkPipeline sensorFpnPipeline = VK_NULL_HANDLE;
    VkShaderModule sensorFpnShader = VK_NULL_HANDLE;
    std::unique_ptr<GpuImage> fpnPrnuMap;              // PRNU map (width x height, R32_SFLOAT)
    std::unique_ptr<GpuImage> fpnDsnuMap;              // DSNU map (width x height, R32_SFLOAT)
    VkDescriptorSetLayout sensorFpnDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout sensorFpnPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet sensorFpnDescriptorSet = VK_NULL_HANDLE;
    bool fpnMapsGenerated = false;
    bool sensorInitialized = false;

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

        // BrdfLut owns its sampler; releasing it here keeps the destruction order
        // with the rest of the GPU resources rather than deferring to Impl's own.
        brdfLut = {};

        envMap = {};

        atmosHeaderBuffer.reset();
        atmosDataBuffer.reset();
        solarLutBuffer.reset();
        criBuffer.reset();
        spectralCurvesBuffer.reset();
        materialBuffer.reset();
        lightingParamsBuffer.reset();
        outputImage.reset();
        pixelReadbackBuffer.reset();

        // Cleanup CLAHE resources
        displayImage.reset();
        claheHistogramBuffer.reset();
        claheCdfBuffer.reset();
        claheMinMaxBuffer.reset();
        if (device != VK_NULL_HANDLE) {
            if (claheHistogramPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, claheHistogramPipeline, nullptr);
                claheHistogramPipeline = VK_NULL_HANDLE;
            }
            if (claheCdfPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, claheCdfPipeline, nullptr);
                claheCdfPipeline = VK_NULL_HANDLE;
            }
            if (claheApplyPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, claheApplyPipeline, nullptr);
                claheApplyPipeline = VK_NULL_HANDLE;
            }
            if (claheHistogramShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, claheHistogramShader, nullptr);
                claheHistogramShader = VK_NULL_HANDLE;
            }
            if (claheCdfShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, claheCdfShader, nullptr);
                claheCdfShader = VK_NULL_HANDLE;
            }
            if (claheApplyShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, claheApplyShader, nullptr);
                claheApplyShader = VK_NULL_HANDLE;
            }
            if (claheDescriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, claheDescriptorPool, nullptr);
                claheDescriptorPool = VK_NULL_HANDLE;
            }
            if (clahePipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, clahePipelineLayout, nullptr);
                clahePipelineLayout = VK_NULL_HANDLE;
            }
            if (claheDescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, claheDescriptorSetLayout, nullptr);
                claheDescriptorSetLayout = VK_NULL_HANDLE;
            }
        }
        claheInitialized = false;

        // Destroy GPU sensor resources
        sensorImage.reset();
        sensorTempImage.reset();
        if (device != VK_NULL_HANDLE) {
            if (sensorRadianceToElectronsPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorRadianceToElectronsPipeline, nullptr);
                sensorRadianceToElectronsPipeline = VK_NULL_HANDLE;
            }
            if (sensorPoissonNoisePipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorPoissonNoisePipeline, nullptr);
                sensorPoissonNoisePipeline = VK_NULL_HANDLE;
            }
            if (sensorPsfBlurHorizontalPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorPsfBlurHorizontalPipeline, nullptr);
                sensorPsfBlurHorizontalPipeline = VK_NULL_HANDLE;
            }
            if (sensorPsfBlurVerticalPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorPsfBlurVerticalPipeline, nullptr);
                sensorPsfBlurVerticalPipeline = VK_NULL_HANDLE;
            }
            if (sensorQuantizeToRadiancePipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorQuantizeToRadiancePipeline, nullptr);
                sensorQuantizeToRadiancePipeline = VK_NULL_HANDLE;
            }
            if (sensorRadianceToElectronsShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorRadianceToElectronsShader, nullptr);
                sensorRadianceToElectronsShader = VK_NULL_HANDLE;
            }
            if (sensorPoissonNoiseShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorPoissonNoiseShader, nullptr);
                sensorPoissonNoiseShader = VK_NULL_HANDLE;
            }
            if (sensorPsfBlurHorizontalShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorPsfBlurHorizontalShader, nullptr);
                sensorPsfBlurHorizontalShader = VK_NULL_HANDLE;
            }
            if (sensorPsfBlurVerticalShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorPsfBlurVerticalShader, nullptr);
                sensorPsfBlurVerticalShader = VK_NULL_HANDLE;
            }
            if (sensorQuantizeToRadianceShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorQuantizeToRadianceShader, nullptr);
                sensorQuantizeToRadianceShader = VK_NULL_HANDLE;
            }
            if (sensorDescriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, sensorDescriptorPool, nullptr);
                sensorDescriptorPool = VK_NULL_HANDLE;
            }
            if (sensorPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, sensorPipelineLayout, nullptr);
                sensorPipelineLayout = VK_NULL_HANDLE;
            }
            if (sensorDescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, sensorDescriptorSetLayout, nullptr);
                sensorDescriptorSetLayout = VK_NULL_HANDLE;
            }
            // Destroy FPN resources
            if (sensorFpnPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, sensorFpnPipeline, nullptr);
                sensorFpnPipeline = VK_NULL_HANDLE;
            }
            if (sensorFpnShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, sensorFpnShader, nullptr);
                sensorFpnShader = VK_NULL_HANDLE;
            }
            if (sensorFpnPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, sensorFpnPipelineLayout, nullptr);
                sensorFpnPipelineLayout = VK_NULL_HANDLE;
            }
            if (sensorFpnDescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, sensorFpnDescriptorSetLayout, nullptr);
                sensorFpnDescriptorSetLayout = VK_NULL_HANDLE;
            }
        }
        fpnPrnuMap.reset();
        fpnDsnuMap.reset();
        fpnMapsGenerated = false;
        sensorInitialized = false;

        // Reset merged global geometry buffers
        geometry = {};


        if (commandPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }

        scene.reset();

        // Destroy context adapter last (may own VMA allocator)
        contextAdapter.reset();

        isReady = false;
    }

    // ------------------------------------------------------------------
    // Pipeline steps
    // ------------------------------------------------------------------
    // These were private members of ExternalRenderContext. They are internal
    // build/record steps, and declaring them in the public header both exported
    // them from the DLL and advertised the render pipeline's structure to
    // anyone reading it. They belong to the state they operate on.

    Result<void, String> Initialize(const InitParams& params);

    // Replace every GPU resource derived from the scene: textures, acceleration
    // structures, geometry buffers, materials, pipeline. Waits for the device
    // first -- see the comment on the definition. Both scene loaders go through
    // this rather than repeating the sequence.
    void AdoptScene(Scene&& loaded);

    void RebuildSceneGpuResources();

    void BuildAccelerationStructures();
    void UpdateGpuResources();
    void CreateDummyBuffers();
    void CreateBRDFLut();
    void CreateFallbackEnvMap();
    void CreatePipeline();

    void CreateCLAHEPipeline();
    void ExecuteCLAHE(VkCommandBuffer cmd, u32 width, u32 height);
    void ComputeImageMinMax(f32& outMin, f32& outMax);

    void CreateGPUSensorPipeline();
    void GenerateAndUploadFPNMaps();
    void ExecuteGPUSensorChain(VkCommandBuffer cmd, u32 width, u32 height);

    void TransitionImageLayoutImmediate(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout
    );

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
    auto initResult = context->m_impl->Initialize(params);
    if (!initResult.has_value()) {
        return Result<std::unique_ptr<ExternalRenderContext>, String>::Err(initResult.error());
    }

    return Result<std::unique_ptr<ExternalRenderContext>, String>(std::move(context));
}

// ============================================================================
// Initialization
// ============================================================================

Result<void, String> ExternalRenderContext::Impl::Initialize(const InitParams& params) {
    QL_LOG_INFO("Initializing ExternalRenderContext...");

    // Store external handles
    instance = params.instance;
    physicalDevice = params.physicalDevice;
    device = params.device;
    graphicsQueue = params.graphicsQueue;
    graphicsQueueFamily = params.graphicsQueueFamily;
    targetColorFormat = params.targetColorFormat;
    width = params.width;
    height = params.height;

    // Set pipeline cache path (use provided path or platform-specific default)
    if (!params.pipelineCacheDir.empty()) {
        pipelineCachePath = (std::filesystem::path(params.pipelineCacheDir) / "pipeline_cache.bin").string();
        // Ensure directory exists
        std::error_code ec;
        std::filesystem::create_directories(params.pipelineCacheDir, ec);
        if (ec) {
            QL_LOG_WARN("Failed to create cache directory {}: {}", params.pipelineCacheDir, ec.message());
        }
    } else {
        pipelineCachePath = (std::filesystem::path(GetDefaultCacheDirectory()) / "pipeline_cache.bin").string();
    }
    QL_LOG_INFO("Pipeline cache path: {}", pipelineCachePath);

    // Create VulkanContextAdapter from external handles
    VulkanContext::ExternalHandles adapterHandles{};
    adapterHandles.instance = params.instance;
    adapterHandles.physicalDevice = params.physicalDevice;
    adapterHandles.device = params.device;
    adapterHandles.graphicsQueue = params.graphicsQueue;
    adapterHandles.graphicsQueueFamily = params.graphicsQueueFamily;
    adapterHandles.allocator = params.externalAllocator;

    try {
        contextAdapter = std::make_unique<VulkanContextAdapter>(adapterHandles, true);
    } catch (const std::exception& e) {
        return Result<void, String>::Err(String("Failed to create VulkanContextAdapter: ") + e.what());
    }

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = params.graphicsQueueFamily;

    VkResult result = vkCreateCommandPool(params.device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        return Result<void, String>::Err("Failed to create command pool");
    }

    // Initialize default lighting params
    lightingParams = CreateDefaultLightingParams();

    // Create output image
    outputImage = std::make_unique<GpuImage>(
        contextAdapter->GetAllocator(),
        contextAdapter->GetDevice(),
        params.width, params.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Transition output image to GENERAL layout
    TransitionImageLayoutImmediate(
        outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Create lighting params buffer
    lightingParamsBuffer = std::make_unique<GpuBuffer>(
        contextAdapter->GetAllocator(),
        sizeof(LightingParams),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    lightingParamsBuffer->Upload(&lightingParams, sizeof(LightingParams));

    // Create dummy buffers for optional bindings
    CreateDummyBuffers();

    // Create BRDF LUT for IBL
    CreateBRDFLut();

    // Create fallback environment map
    CreateFallbackEnvMap();

    // Create texture manager
    textureManager = std::make_unique<TextureManager>(*contextAdapter);

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
    auto scene = rendercore::LoadSceneFromConfig(config);
    if (!scene.has_value()) {
        return Result<void, String>::Err(scene.error());
    }
    m_impl->AdoptScene(std::move(scene.value()));
    ResetAccumulation();
    return Result<void, String>::Ok();
}

// Everything here frees GPU memory the previous scene owned before allocating
// the replacement: UploadTextures destroys the old images and samplers,
// BuildAccelerationStructures resets the BLAS list, the TLAS and every global
// geometry buffer, and CreatePipeline destroys the old pipeline, descriptor
// pool and SBT. A frame submitted by the host and still executing holds
// references to all of it, and freeing underneath it faults the GPU.
//
// Resize() and RebuildAccelerationStructure() already wait for exactly this
// reason; the full scene swap, which frees the most, was the one path that did
// not. Loading a scene is not a hot path, so a full device wait is the right
// instrument -- there is no per-frame cost to protect here.
// Take ownership of a freshly loaded scene and rebuild everything derived from it.
// Shared by the three loaders, which had identical tails.
void ExternalRenderContext::Impl::AdoptScene(Scene&& loaded) {
    scene = std::make_unique<Scene>(std::move(loaded));

    // Setup camera from scene
    camera = scene->camera;
    camera.SetAspectRatio(static_cast<f32>(width) / static_cast<f32>(height));

    RebuildSceneGpuResources();

    isReady = true;

    QL_LOG_INFO("Scene loaded: {} meshes, {} materials, {} textures",
                scene->meshes.size(), scene->materials.size(), scene->textures.size());
}

void ExternalRenderContext::Impl::RebuildSceneGpuResources() {
    vkDeviceWaitIdle(device);

    textureManager->UploadTextures(scene->textures);
    BuildAccelerationStructures();
    UpdateGpuResources();
    CreatePipeline();
}

Result<void, String> ExternalRenderContext::LoadSceneFromGltf(const String& gltfPath) {
    QL_LOG_INFO("Loading glTF scene: {}", gltfPath);

    auto result = GltfLoader::LoadFromFile(gltfPath);
    if (!result.has_value()) {
        return Result<void, String>::Err("Failed to load glTF: " + result.error());
    }

    m_impl->AdoptScene(std::move(result.value()));
    // Via ResetAccumulation() rather than clearing the counter directly, so the
    // sampling sequence restarts with it.
    ResetAccumulation();

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

    m_impl->AdoptScene(std::move(result.value()));
    // Via ResetAccumulation() rather than clearing the counter directly, so the
    // sampling sequence restarts with it.
    ResetAccumulation();

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

// Lazily (re)bakes the NN atmosphere LUT when the bake key changed and
// uploads header + data. On bake failure (missing network files etc.) the
// atmosphere is disabled with a critical log -- no analytic fallback exists.
void ExternalRenderContext::Impl::UpdateAtmosphereNN() {
    constexpr uint64_t kDisabledKey = 1;  // 0 = dirty, 1 = disabled uploaded

    auto uploadDisabled = [this]() {
        AtmosNNHeaderGPU disabledHeader{};
        atmosHeaderBuffer->Upload(&disabledHeader, sizeof(disabledHeader));
        atmosBakeKey = kDisabledKey;
    };

    if (!atmosphereConfig.enabled || !atmosModelPack) {
        if (atmosBakeKey != kDisabledKey) uploadDisabled();
        return;
    }

    AtmosLambdaGrid grid = RenderBandLambdaGrid(
        spectralMode, static_cast<double>(wavelength_nm));
    if (grid.band.empty()) {
        if (atmosBakeKey != kDisabledKey) {
            if (!grid.error.empty())
                QL_LOG_CRITICAL("NN atmosphere: {}", grid.error);
            else
                QL_LOG_WARN("NN atmosphere: spectral mode has no NN coverage, "
                            "atmosphere disabled for this mode");
            uploadDisabled();
        }
        return;
    }

    // Resolve geometry defaults from the live renderer state
    AtmosphereNNConfig resolved = atmosphereConfig;
    const glm::vec3 sunDir = lightingParams.sunDirection;
    if (resolved.sunFromLighting && glm::length(sunDir) > 1e-6f) {
        const glm::vec3 s = glm::normalize(sunDir);
        resolved.sunZenithDeg = glm::degrees(std::acos(std::clamp(s.y, -1.0f, 1.0f)));
        resolved.sunAzimuthDeg = glm::degrees(std::atan2(s.x, s.z));
    }
    if (resolved.h1FromCamera) {
        const f32 wu = lightingParams.worldUnitsToMeters > 0.0f
                           ? lightingParams.worldUnitsToMeters : 1.0f;
        resolved.h1Km = std::max(
            static_cast<double>(camera.GetPosition().y * wu) / 1000.0, 0.0);
    }

    const uint64_t key =
        AtmosphereBaker::BakeKey(resolved, grid.band, grid.lambdasNm);
    if (key == atmosBakeKey) return;

    try {
        AtmosphereBaker baker(*atmosModelPack);
        AtmosBakeResult baked =
            baker.Bake(resolved, grid.band, grid.lambdasNm, grid.windowHalfWidthNm);
        if (baked.data.size() > kAtmosMaxDataFloats) {
            QL_LOG_CRITICAL("NN atmosphere: baked LUT ({} floats) exceeds GPU "
                            "buffer capacity ({})", baked.data.size(),
                            kAtmosMaxDataFloats);
            uploadDisabled();
            return;
        }
        const glm::vec3 s = glm::length(sunDir) > 1e-6f
                                ? glm::normalize(sunDir) : glm::vec3(0, 1, 0);
        baked.header.sunDirWorld[0] = s.x;
        baked.header.sunDirWorld[1] = s.y;
        baked.header.sunDirWorld[2] = s.z;
        baked.header.worldUnitsToMeters =
            lightingParams.worldUnitsToMeters > 0.0f
                ? lightingParams.worldUnitsToMeters : 1.0f;
        atmosDataBuffer->Upload(baked.data.data(),
                                baked.data.size() * sizeof(f32));
        atmosHeaderBuffer->Upload(&baked.header, sizeof(baked.header));
        atmosBakeKey = key;
    } catch (const std::exception& e) {
        QL_LOG_CRITICAL("NN atmosphere bake failed, atmosphere disabled: {}",
                        e.what());
        uploadDisabled();
    }
}

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

    // Update CLAHE min/max cache from previous frame's output image
    // This is safe here because QVulkanWindow ensures the previous frame's
    // GPU work is complete before calling startNextFrame/RenderFrame again
    // Only update every N frames to reduce readback overhead, but always update
    // on frame 1 (after first render) and whenever cache is invalid
    constexpr u32 minMaxUpdateInterval = 10; // Update every 10 frames
    if (m_impl->claheParams.enabled && m_impl->claheInitialized &&
        m_impl->accumulatedSamples > 0 &&
        (!m_impl->hasCachedMinMax ||
         m_impl->accumulatedSamples == 1 ||  // Always update after first frame
         m_impl->frameIndex % minMaxUpdateInterval == 0)) {
        m_impl->ComputeImageMinMax(m_impl->cachedImageMin, m_impl->cachedImageMax);
        m_impl->hasCachedMinMax = true;
        //QL_LOG_DEBUG("CLAHE: Updated min/max cache: [{}, {}]",
        //             m_impl->cachedImageMin, m_impl->cachedImageMax);
    }

    // Handle resize
    if (width != m_impl->width || height != m_impl->height) {
        Resize(width, height);
    }

    // Rebake the NN atmosphere LUT if the bake key changed (band, weather,
    // quantized altitude / sun geometry)
    m_impl->UpdateAtmosphereNN();

    // Update camera data with current state
    CameraData cameraData = m_impl->camera.GetCameraData();
    cameraData.wavelength_nm = m_impl->wavelength_nm;
    cameraData.spectral_mode = static_cast<u32>(m_impl->spectralMode);
    cameraData.debug_mode = static_cast<u32>(m_impl->debugMode);
    m_impl->pipeline->SetCameraData(cameraData);

    // The shader branches on the SPEC_SPECTRAL_MODE specialization constant,
    // not the push-constant copy above. Sync the pipeline variant here at the
    // point of use: setters may run before the pipeline exists (scene not yet
    // loaded), and scene reload recreates the pipeline with the default
    // variant. Cached variants make this a hash lookup.
    m_impl->pipeline->SetSpecConstants(
        static_cast<u32>(m_impl->spectralMode),
        m_impl->debugMode != DebugVisualizationMode::None);

    // Set sampling parameters
    // Use Mersenne Twister RNG for better sample distribution (reduces fireflies)
    //
    // Mixed with accumulatedSamples only. frameIndex used to be part of this,
    // but it counts every frame ever drawn and is never reset, so it made the
    // seed depend on session history rather than on the accumulation -- the one
    // thing that had to be reproducible. Dropping it makes this identical to
    // the CLI's `dist(rng) ^ (frameIndex * 997 + sampleIndex * 1009)`, where
    // frameIndex is fixed at 0 and sampleIndex is the accumulation index.
    u32 randomSeed = m_impl->randDist(m_impl->rng) ^ (m_impl->accumulatedSamples * 1009);

    // frameIndex is passed as 0, exactly as the CLI passes it. raygen.rgen
    // folds this push constant into the per-pixel seed
    // (`seed ^= pushConsts.frameIndex * 26699`) and uses it for nothing else,
    // so feeding it the free-running frame counter made the image depend on how
    // many frames the session had drawn before this accumulation began. The
    // sample-to-sample variation it was there to provide already comes from
    // sampleIndex below.
    constexpr u32 kShaderFrameIndex = 0;
    m_impl->pipeline->SetSamplingParams(
        kShaderFrameIndex,
        m_impl->accumulatedSamples,
        m_impl->spp,
        randomSeed
    );

    // Execute ray tracing (writes to internal outputImage in GENERAL layout)
    m_impl->pipeline->TraceRays(cmd, width, height);

    // Determine which image to blit to the swapchain
    // Priority: GPU Sensor → CLAHE → Raw output
    VkImage blitSourceImage = m_impl->outputImage->GetImage();

    // Apply GPU sensor simulation if enabled
    if (m_impl->gpuSensorEnabled && m_impl->sensorInitialized && m_impl->sensorImage) {
        // Execute GPU sensor chain: outputImage -> sensorImage
        m_impl->ExecuteGPUSensorChain(cmd, width, height);
        blitSourceImage = m_impl->sensorImage->GetImage();
    }

    // Apply CLAHE to sensor output (or raw output if sensor disabled)
    if (m_impl->claheParams.enabled && m_impl->claheInitialized && m_impl->displayImage) {
        // If sensor is enabled, CLAHE processes sensorImage
        // If sensor is disabled, CLAHE processes outputImage
        // Note: ExecuteCLAHE reads from outputImage by default, need to update descriptor
        // For now, CLAHE always reads from outputImage (TODO: make it read from current source)
        m_impl->ExecuteCLAHE(cmd, width, height);
        blitSourceImage = m_impl->displayImage->GetImage();
    }

    // ========================================================================
    // Blit source image to target swapchain image
    // ========================================================================

    // Step 1: Transition source image from GENERAL to TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier outputBarrier{};
    outputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.image = blitSourceImage;
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
    // Use appropriate source stage based on which processing was applied
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    if (m_impl->claheParams.enabled && m_impl->claheInitialized) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;  // CLAHE was last
    } else if (m_impl->gpuSensorEnabled && m_impl->sensorInitialized) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;  // GPU sensor was last
    }
    vkCmdPipelineBarrier(
        cmd,
        srcStage,
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
        blitSourceImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        targetImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blitRegion,
        VK_FILTER_NEAREST  // No filtering needed for same-size blit
    );

    // Step 4: Transition source image back to GENERAL for next frame
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
    // Use appropriate destination stage for the source image
    VkPipelineStageFlags dstStage = (blitSourceImage == m_impl->outputImage->GetImage())
        ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dstStage | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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

    m_impl->TransitionImageLayoutImmediate(
        m_impl->outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Re-bind output image
    if (m_impl->pipeline) {
        m_impl->pipeline->BindOutputImage(*m_impl->outputImage);
    }

    // Recreate CLAHE display image if initialized
    if (m_impl->claheInitialized && m_impl->displayImage) {
        m_impl->displayImage = std::make_unique<GpuImage>(
            m_impl->contextAdapter->GetAllocator(),
            m_impl->contextAdapter->GetDevice(),
            width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        m_impl->TransitionImageLayoutImmediate(
            m_impl->displayImage->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        // Update CLAHE descriptor set with new images
        VkDescriptorImageInfo inputImageInfo{};
        inputImageInfo.imageView = m_impl->outputImage->GetView();
        inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputImageInfo{};
        outputImageInfo.imageView = m_impl->displayImage->GetView();
        outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::vector<VkWriteDescriptorSet> writes(2);

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_impl->claheDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputImageInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_impl->claheDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputImageInfo;

        vkUpdateDescriptorSets(m_impl->device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    }

    // Recreate sensor images if initialized
    if (m_impl->sensorInitialized && m_impl->sensorImage) {
        auto sensorAllocator = m_impl->contextAdapter->GetAllocator();
        auto sensorDevice = m_impl->contextAdapter->GetDevice();

        m_impl->sensorImage = std::make_unique<GpuImage>(
            sensorAllocator, sensorDevice,
            width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        m_impl->sensorTempImage = std::make_unique<GpuImage>(
            sensorAllocator, sensorDevice,
            width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        m_impl->TransitionImageLayoutImmediate(
            m_impl->sensorImage->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        m_impl->TransitionImageLayoutImmediate(
            m_impl->sensorTempImage->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        // Recreate FPN map images for new dimensions
        if (m_impl->fpnPrnuMap) {
            m_impl->fpnPrnuMap = std::make_unique<GpuImage>(
                sensorAllocator, sensorDevice,
                width, height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );
            m_impl->fpnDsnuMap = std::make_unique<GpuImage>(
                sensorAllocator, sensorDevice,
                width, height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );

            m_impl->TransitionImageLayoutImmediate(
                m_impl->fpnPrnuMap->GetImage(),
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL
            );
            m_impl->TransitionImageLayoutImmediate(
                m_impl->fpnDsnuMap->GetImage(),
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL
            );
        }

        // Invalidate FPN maps so they get regenerated for new dimensions
        m_impl->fpnMapsGenerated = false;
    }

    // Update camera aspect ratio
    m_impl->camera.SetAspectRatio(static_cast<f32>(width) / static_cast<f32>(height));

    // Reset accumulation
    ResetAccumulation();
}

void ExternalRenderContext::ResetAccumulation() {
    m_impl->accumulatedSamples = 0;
    // Restart the sampling sequence with the accumulation it feeds, so an
    // accumulation pass always draws the same samples for the same seed
    // regardless of what was rendered before it.
    m_impl->ReseedRng();
}

void ExternalRenderContext::SetSamplingSeed(u32 seed) {
    m_impl->samplingSeed = seed;
    if (seed == 0U) {
        QL_LOG_INFO("Sampling seed: nondeterministic (seed = 0)");
    } else {
        QL_LOG_INFO("Sampling seed: {}", seed);
    }
    ResetAccumulation();
}

u32 ExternalRenderContext::GetSamplingSeed() const {
    return m_impl->samplingSeed;
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
        if (m_impl->pipeline) {
            m_impl->pipeline->SetSpecConstants(
                static_cast<u32>(mode),
                m_impl->debugMode != DebugVisualizationMode::None);
        }
        ResetAccumulation();
    }
}

void ExternalRenderContext::SetWavelength(f32 wavelength_nm) {
    if (m_impl->wavelength_nm == wavelength_nm) {
        return;
    }
    m_impl->wavelength_nm = wavelength_nm;

    // The material buffer holds IR emissivity and transmittance sampled at the old
    // wavelength, so it is now stale. Re-uploading in place keeps the descriptor
    // binding valid -- the material count has not changed -- and follows what
    // UpdateMaterial already does for a single entry.
    if (m_impl->scene && m_impl->materialBuffer) {
        m_impl->UpdateGpuResources();
        if (m_impl->pipeline && m_impl->materialBuffer) {
            m_impl->pipeline->BindMaterialBuffer(*m_impl->materialBuffer);
        }
    }

    ResetAccumulation();
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
        if (m_impl->pipeline) {
            m_impl->pipeline->SetSpecConstants(
                static_cast<u32>(m_impl->spectralMode),
                mode != DebugVisualizationMode::None);
        }
        ResetAccumulation();
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
// Material CPU↔GPU conversion helper
// ============================================================================

// Compute a single representative scalar from a sparse (wavelength_nm, value) curve
// by trapezoidal integration over the curve's wavelength domain.
// Returns fallback if the curve has fewer than two samples.

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
    if (!m_impl->scene) {
        QL_LOG_WARN("UpdateMaterial: No scene loaded");
        return;
    }
    if (materialIndex >= m_impl->scene->materials.size()) {
        QL_LOG_WARN("UpdateMaterial: Invalid material index {}", materialIndex);
        return;
    }
    if (!m_impl->materialBuffer) {
        QL_LOG_WARN("UpdateMaterial: No material buffer");
        return;
    }

    // 1. Update CPU-side scene data
    m_impl->scene->materials[materialIndex] = material;

    // 2. Convert to GPU format
    const rendercore::MaterialGpuIndices indices{material.spectralReflectanceCurveIndex,
                                                 material.complexRefractiveIndexIndex};
    MaterialDataCPU cpuMat =
        rendercore::ConvertMaterial(material, m_impl->wavelength_nm, indices);

    // 3. Partial upload at offset
    VkDeviceSize offset = materialIndex * sizeof(MaterialDataCPU);
    m_impl->materialBuffer->Upload(&cpuMat, sizeof(MaterialDataCPU), offset);

    // 4. Reset accumulation (visual feedback)
    ResetAccumulation();

    QL_LOG_DEBUG("UpdateMaterial: Updated material {} ('{}')", materialIndex, material.name);
}

i32 ExternalRenderContext::AddComplexRefractiveIndex(const ComplexRefractiveIndex& cri) {
    if (!cri.IsValid()) {
        QL_LOG_WARN("AddComplexRefractiveIndex: Invalid CRI data");
        return -1;
    }

    // Convert CPU → GPU format (resample to uniform 64-sample grid)
    ComplexRefractiveIndexGPU gpuCRI = ComplexRefractiveIndexGPU::FromCPU(cri);

    // Append to entries
    i32 index = static_cast<i32>(m_impl->criEntries.size());
    m_impl->criEntries.push_back(gpuCRI);

    // Rebuild GPU buffer
    auto allocator = m_impl->contextAdapter->GetAllocator();
    m_impl->criBuffer = std::make_unique<GpuBuffer>(
        allocator,
        m_impl->criEntries.size() * sizeof(ComplexRefractiveIndexGPU),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    m_impl->criBuffer->Upload(
        m_impl->criEntries.data(),
        m_impl->criEntries.size() * sizeof(ComplexRefractiveIndexGPU));

    // Rebind descriptor
    if (m_impl->pipeline) {
        m_impl->pipeline->BindComplexRefractiveIndexBuffer(m_impl->criBuffer.get());
    }

    QL_LOG_INFO("AddComplexRefractiveIndex: Added CRI at index {} "
                "(wavelength range: {:.0f}-{:.0f} nm, {} samples)",
                index, cri.wavelengths_nm.front(), cri.wavelengths_nm.back(),
                cri.wavelengths_nm.size());

    return index;
}

void ExternalRenderContext::RebuildAccelerationStructure() {
    if (!m_impl->scene || !m_impl->geometry.IsValid()) {
        QL_LOG_WARN("RebuildAccelerationStructure: No scene or geometry available");
        return;
    }

    // A frame the host submitted may still be tracing the TLAS being replaced.
    vkDeviceWaitIdle(m_impl->device);

    m_impl->geometry.RebuildTlas(*m_impl->contextAdapter, *m_impl->scene);

    if (m_impl->pipeline) {
        m_impl->pipeline->BindAccelerationStructure(m_impl->geometry.Tlas().GetHandle());
    }
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

const String& ExternalRenderContext::GetPipelineCachePath() const {
    return m_impl->pipelineCachePath;
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
// CLAHE Display Enhancement
// ============================================================================

void ExternalRenderContext::SetCLAHEParams(const CLAHEParams& params) {
    bool wasEnabled = m_impl->claheParams.enabled;
    m_impl->claheParams = params;

    // Initialize CLAHE resources if enabling for the first time
    if (params.enabled && !wasEnabled && !m_impl->claheInitialized) {
        m_impl->CreateCLAHEPipeline();
    }

    QL_LOG_DEBUG("CLAHE params: enabled={}, clipLimit={}, tileSize={}, luminanceOnly={}",
                 params.enabled, params.clipLimit, params.tileSize, params.luminanceOnly);
}

const ExternalRenderContext::CLAHEParams& ExternalRenderContext::GetCLAHEParams() const {
    return m_impl->claheParams;
}

// ============================================================================
// GPU Sensor Simulation API
// ============================================================================

void ExternalRenderContext::SetGPUSensorEnabled(bool enabled) {
    bool wasEnabled = m_impl->gpuSensorEnabled;
    m_impl->gpuSensorEnabled = enabled;

    // Initialize GPU sensor resources if enabling for the first time
    if (enabled && !wasEnabled && !m_impl->sensorInitialized) {
        m_impl->CreateGPUSensorPipeline();
    }

    QL_LOG_DEBUG("GPU Sensor: enabled={}", enabled);
}

void ExternalRenderContext::SetGPUSensorParams(const SensorParams& params) {
    m_impl->gpuSensorParams = params;

    // Initialize GPU sensor resources if not already done
    if (m_impl->gpuSensorEnabled && !m_impl->sensorInitialized) {
        m_impl->CreateGPUSensorPipeline();
    }

    QL_LOG_DEBUG("GPU Sensor params updated: QE={}, f#={}, gain={}, bitDepth={}",
                 params.quantumEfficiency, params.fNumber, params.gain, params.bitDepth);
}

bool ExternalRenderContext::IsGPUSensorEnabled() const {
    return m_impl->gpuSensorEnabled;
}

const SensorParams& ExternalRenderContext::GetGPUSensorParams() const {
    return m_impl->gpuSensorParams;
}

// ============================================================================
// CLAHE Display Image Capture
// ============================================================================

Result<Image, String> ExternalRenderContext::CaptureDisplayImage() {
    if (!m_impl->isReady) {
        return Result<Image, String>::Err("Render context not ready");
    }

    // Determine source image: priority displayImage → sensorImage → outputImage
    VkImage sourceImage = m_impl->outputImage->GetImage();

    // Priority 1: CLAHE output (includes all effects)
    if (m_impl->claheParams.enabled && m_impl->claheInitialized && m_impl->displayImage) {
        sourceImage = m_impl->displayImage->GetImage();
        QL_LOG_DEBUG("CaptureDisplayImage: Using displayImage (CLAHE enabled)");
    }
    // Priority 2: Sensor output (includes sensor effects)
    else if (m_impl->gpuSensorEnabled && m_impl->sensorInitialized && m_impl->sensorImage) {
        sourceImage = m_impl->sensorImage->GetImage();
        QL_LOG_DEBUG("CaptureDisplayImage: Using sensorImage (GPU sensor enabled)");
    }
    // Priority 3: Raw output
    else {
        QL_LOG_DEBUG("CaptureDisplayImage: Using outputImage (no post-processing)");
    }

    // Read back the appropriate image using CommandHelper
    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        *m_impl->contextAdapter,
        sourceImage,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        m_impl->width,
        m_impl->height
    );

    // Create Image from pixel data
    Image displayImage(m_impl->width, m_impl->height, 4);  // RGBA
    displayImage.data = std::move(pixels);
    displayImage.channelNames = {"R", "G", "B", "A"};

    // Add metadata
    displayImage.metadata["spectral_mode"] = std::to_string(static_cast<int>(m_impl->spectralMode));
    displayImage.metadata["wavelength_nm"] = std::to_string(m_impl->wavelength_nm);
    displayImage.metadata["accumulated_samples"] = std::to_string(m_impl->accumulatedSamples);
    displayImage.metadata["spp_target"] = std::to_string(m_impl->spp);
    displayImage.metadata["clahe_applied"] = m_impl->claheParams.enabled ? "true" : "false";
    if (m_impl->claheParams.enabled) {
        displayImage.metadata["clahe_clip_limit"] = std::to_string(m_impl->claheParams.clipLimit);
        displayImage.metadata["clahe_tile_size"] = std::to_string(m_impl->claheParams.tileSize);
    }
    displayImage.metadata["gpu_sensor_applied"] = m_impl->gpuSensorEnabled ? "true" : "false";
    if (m_impl->gpuSensorEnabled) {
        displayImage.metadata["sensor_f_number"] = std::to_string(m_impl->gpuSensorParams.fNumber);
        displayImage.metadata["sensor_quantum_efficiency"] = std::to_string(m_impl->gpuSensorParams.quantumEfficiency);
    }

    return std::move(displayImage);
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void ExternalRenderContext::Impl::BuildAccelerationStructures() {
    if (!scene) return;
    geometry = rendercore::SceneGeometry::Build(*contextAdapter, *scene);
}

void ExternalRenderContext::Impl::UpdateGpuResources() {
    if (!scene) return;

    QL_LOG_INFO("Updating GPU resources...");
    materialBuffer = rendercore::BuildMaterialBuffer(*contextAdapter, *scene, wavelength_nm);
    QL_LOG_INFO("  GPU resources updated");
}

void ExternalRenderContext::Impl::CreateDummyBuffers() {
    auto allocator = contextAdapter->GetAllocator();

    // Create dummy spectral curves buffer
    struct DummySpectralCurve {
        f32 data[272 / sizeof(f32)];
    } dummyCurve{};

    spectralCurvesBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummySpectralCurve),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    spectralCurvesBuffer->Upload(&dummyCurve, sizeof(DummySpectralCurve));

    // Create dummy CRI buffer
    struct DummyCRI {
        f32 data[528 / sizeof(f32)];
    } dummyCri{};

    criBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummyCRI),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    criBuffer->Upload(&dummyCri, sizeof(DummyCRI));

    // Create dummy solar LUT buffer
    struct DummySolarLUT {
        f32 data[544 / sizeof(f32)];
    } dummySolar{};

    solarLutBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(DummySolarLUT),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    solarLutBuffer->Upload(&dummySolar, sizeof(DummySolarLUT));

    // Create NN atmosphere buffers. The header starts zeroed (enabled = 0,
    // atmosphere off); the data blob is preallocated at its maximum size so
    // rebakes are pure uploads with no descriptor rebinding.
    AtmosNNHeaderGPU disabledHeader{};
    atmosHeaderBuffer = std::make_unique<GpuBuffer>(
        allocator,
        sizeof(AtmosNNHeaderGPU),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    atmosHeaderBuffer->Upload(&disabledHeader, sizeof(AtmosNNHeaderGPU));

    atmosDataBuffer = std::make_unique<GpuBuffer>(
        allocator,
        kAtmosMaxDataFloats * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    {
        std::vector<f32> zeroData(kAtmosMaxDataFloats, 0.0f);
        atmosDataBuffer->Upload(zeroData.data(), zeroData.size() * sizeof(f32));
    }

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

    cieCmfBuffer = std::make_unique<GpuBuffer>(
        allocator,
        cieCmfData.size() * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    cieCmfBuffer->Upload(cieCmfData.data(), cieCmfData.size() * sizeof(glm::vec4));

    QL_LOG_DEBUG("  CIE CMF LUT created ({} samples)", CIE_CMF_LUT_SIZE);
}

void ExternalRenderContext::Impl::CreateBRDFLut() {
    brdfLut = rendercore::BrdfLut::Create(*contextAdapter);
}

void ExternalRenderContext::Impl::CreateFallbackEnvMap() {
    envMap = rendercore::EnvironmentCubemap::Fallback(*contextAdapter);
}

void ExternalRenderContext::Impl::CreatePipeline() {
    QL_LOG_INFO("Creating ray tracing pipeline...");

    // Load or create pipeline cache for faster shader compilation
    if (pipelineCache == VK_NULL_HANDLE) {
        pipelineCache = RayTracingPipeline::LoadPipelineCache(
            *contextAdapter,
            pipelineCachePath
        );
    }

    // Create pipeline using context adapter with cache
    pipeline = std::make_unique<RayTracingPipeline>(
        *contextAdapter,
        "raygen.spv",
        "closesthit.spv",
        "miss.spv",
        pipelineCache
    );

    // Bind resources
    pipeline->BindOutputImage(*outputImage);
    pipeline->BindAccelerationStructure(geometry.Tlas().GetHandle());
    pipeline->BindLUTBuffer(*lightingParamsBuffer);

    // Bind merged global geometry buffers (instead of per-BLAS buffers)
    if (geometry.IsValid()) {
        pipeline->BindGeometryBuffers(geometry.Vertices(), geometry.Indices(),
                                      &geometry.UVs());
        pipeline->BindTangentBuffer(geometry.Tangents());
        pipeline->BindNormalBuffer(geometry.Normals());
        if (geometry.InstanceCount() > 0) {
            pipeline->BindInstanceGeometryBuffer(geometry.InstanceInfo());
        }
    }

    // Bind materials
    if (materialBuffer) {
        pipeline->BindMaterialBuffer(*materialBuffer);
    }

    // Bind textures
    pipeline->BindTextures(
        textureManager->GetImageViews(),
        textureManager->GetSamplers()
    );

    // Bind IBL
    pipeline->BindPrefilteredEnvMap(envMap.View(), envMap.Sampler());
    pipeline->BindBRDFLut(brdfLut.View(), brdfLut.Sampler());

    // Bind optional buffers
    pipeline->BindSpectralCurvesBuffer(spectralCurvesBuffer.get());
    pipeline->BindComplexRefractiveIndexBuffer(criBuffer.get());
    pipeline->BindSolarSpectralLUT(solarLutBuffer.get());
    pipeline->BindAtmosphereNN(atmosHeaderBuffer.get(),
                                       atmosDataBuffer.get());

    // Bind CIE CMF LUT (required for VIS_Fused spectral mode)
    if (cieCmfBuffer) {
        pipeline->BindCIE_CMF_LUT(*cieCmfBuffer);
    }

    QL_LOG_INFO("  Ray tracing pipeline created and bound");
}

// ============================================================================
// Helper: Get executable directory for shader loading
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

void ExternalRenderContext::Impl::CreateCLAHEPipeline() {
    if (claheInitialized) return;

    QL_LOG_INFO("Creating CLAHE compute pipeline...");

    auto device = this->device;
    auto allocator = contextAdapter->GetAllocator();

    // Helper function to load shader file
    auto loadShaderFile = [](const String& path) -> std::vector<u32> {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return {};
        }
        size_t fileSize = static_cast<size_t>(file.tellg());
        if (fileSize == 0 || fileSize % 4 != 0) {
            return {};
        }
        std::vector<u32> code(fileSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), fileSize);
        return code;
    };

    // Try multiple paths for shader location
    // Include executable directory for SDK installations
    auto exeDir = GetExecutableDirectory();
    std::vector<std::filesystem::path> shaderPaths = {
        "clahe_histogram.spv",
        exeDir / "clahe_histogram.spv",  // SDK installation directory
        "shaders/clahe_histogram.spv",
        exeDir / "shaders" / "clahe_histogram.spv",
        "../shaders/clahe_histogram.spv",
        "src/shaders/clahe_histogram.spv"
    };

    std::vector<u32> histogramCode, cdfCode, applyCode;
    for (const auto& basePath : shaderPaths) {
        // Convert path to string and replace "histogram" with cdf/apply
        String histPath = basePath.string();
        String cdfPath = histPath;
        String applyPath = histPath;
        size_t pos = histPath.find("histogram");
        if (pos != String::npos) {
            cdfPath.replace(pos, 9, "cdf");
            applyPath.replace(pos, 9, "apply");
        }

        histogramCode = loadShaderFile(histPath);
        if (!histogramCode.empty()) {
            cdfCode = loadShaderFile(cdfPath);
            applyCode = loadShaderFile(applyPath);
            if (!cdfCode.empty() && !applyCode.empty()) {
                QL_LOG_DEBUG("CLAHE: Loaded shaders from {}", histPath);
                break;
            }
        }
    }

    if (histogramCode.empty() || cdfCode.empty() || applyCode.empty()) {
        QL_LOG_WARN("CLAHE: Could not load shader files, CLAHE disabled");
        return;
    }

    // Create shader modules
    auto createShaderModule = [device](const std::vector<u32>& code) -> VkShaderModule {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(u32);
        createInfo.pCode = code.data();
        VkShaderModule module;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return module;
    };

    claheHistogramShader = createShaderModule(histogramCode);
    claheCdfShader = createShaderModule(cdfCode);
    claheApplyShader = createShaderModule(applyCode);

    if (claheHistogramShader == VK_NULL_HANDLE ||
        claheCdfShader == VK_NULL_HANDLE ||
        claheApplyShader == VK_NULL_HANDLE) {
        QL_LOG_WARN("CLAHE: Failed to create shader modules");
        return;
    }

    // Create descriptor set layout
    // binding 0: inputImage (storage image)
    // binding 1: outputImage (storage image)
    // binding 2: histogramBuffer (storage buffer)
    // binding 3: cdfBuffer (storage buffer)
    // binding 4: minMaxBuffer (storage buffer)
    std::vector<VkDescriptorSetLayoutBinding> bindings(5);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &claheDescriptorSetLayout) != VK_SUCCESS) {
        QL_LOG_WARN("CLAHE: Failed to create descriptor set layout");
        return;
    }

    // Create pipeline layout with push constants
    // Push constants match CLAHEPushConstants in shader
    struct CLAHEPushConstants {
        u32 imageWidth;
        u32 imageHeight;
        u32 tileCountX;
        u32 tileCountY;
        f32 clipLimit;
        u32 luminanceOnly;
        f32 inputMin;
        f32 inputMax;
        u32 passIndex;
        u32 padding[3];
    };

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(CLAHEPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &claheDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &clahePipelineLayout) != VK_SUCCESS) {
        QL_LOG_WARN("CLAHE: Failed to create pipeline layout");
        return;
    }

    // Create compute pipelines for each pass
    auto createComputePipeline = [device, this](VkShaderModule shader) -> VkPipeline {
        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shader;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = clahePipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return pipeline;
    };

    claheHistogramPipeline = createComputePipeline(claheHistogramShader);
    claheCdfPipeline = createComputePipeline(claheCdfShader);
    claheApplyPipeline = createComputePipeline(claheApplyShader);

    if (claheHistogramPipeline == VK_NULL_HANDLE ||
        claheCdfPipeline == VK_NULL_HANDLE ||
        claheApplyPipeline == VK_NULL_HANDLE) {
        QL_LOG_WARN("CLAHE: Failed to create compute pipelines");
        return;
    }

    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &claheDescriptorPool) != VK_SUCCESS) {
        QL_LOG_WARN("CLAHE: Failed to create descriptor pool");
        return;
    }

    // Create display image (same format as outputImage)
    displayImage = std::make_unique<GpuImage>(
        allocator,
        device,
        width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Transition display image to GENERAL layout
    TransitionImageLayoutImmediate(
        displayImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Create buffers for CLAHE
    // Max 64x64 tiles, 256 bins per tile
    constexpr u32 maxTiles = 64 * 64;
    constexpr u32 histogramBins = 256;

    claheHistogramBuffer = std::make_unique<GpuBuffer>(
        allocator,
        maxTiles * histogramBins * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    claheCdfBuffer = std::make_unique<GpuBuffer>(
        allocator,
        maxTiles * histogramBins * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    claheMinMaxBuffer = std::make_unique<GpuBuffer>(
        allocator,
        maxTiles * 2 * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = claheDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &claheDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &claheDescriptorSet) != VK_SUCCESS) {
        QL_LOG_WARN("CLAHE: Failed to allocate descriptor set");
        return;
    }

    // Update descriptor set
    VkDescriptorImageInfo inputImageInfo{};
    inputImageInfo.imageView = outputImage->GetView();
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = displayImage->GetView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo histogramBufferInfo{};
    histogramBufferInfo.buffer = claheHistogramBuffer->GetHandle();
    histogramBufferInfo.offset = 0;
    histogramBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo cdfBufferInfo{};
    cdfBufferInfo.buffer = claheCdfBuffer->GetHandle();
    cdfBufferInfo.offset = 0;
    cdfBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo minMaxBufferInfo{};
    minMaxBufferInfo.buffer = claheMinMaxBuffer->GetHandle();
    minMaxBufferInfo.offset = 0;
    minMaxBufferInfo.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> writes(5);

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = claheDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &inputImageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = claheDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &outputImageInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = claheDescriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &histogramBufferInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = claheDescriptorSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &cdfBufferInfo;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = claheDescriptorSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &minMaxBufferInfo;

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);

    claheInitialized = true;
    QL_LOG_INFO("CLAHE: Compute pipeline created successfully");
}

// ============================================================================
// GPU Sensor Pipeline Creation
// ============================================================================

void ExternalRenderContext::Impl::CreateGPUSensorPipeline() {
    if (sensorInitialized) return;

    QL_LOG_INFO("Creating GPU sensor compute pipeline...");

    auto device = this->device;
    auto allocator = contextAdapter->GetAllocator();

    // Helper function to load shader file
    auto loadShaderFile = [](const String& path) -> std::vector<u32> {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return {};
        }
        size_t fileSize = static_cast<size_t>(file.tellg());
        if (fileSize == 0 || fileSize % 4 != 0) {
            return {};
        }
        std::vector<u32> code(fileSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), fileSize);
        return code;
    };

    // Try multiple paths for shader location
    auto exeDir = GetExecutableDirectory();
    std::vector<std::filesystem::path> shaderPaths = {
        "sensor_radiance_to_electrons.spv",
        exeDir / "sensor_radiance_to_electrons.spv",
        "shaders/sensor_radiance_to_electrons.spv",
        exeDir / "shaders" / "sensor_radiance_to_electrons.spv",
        "../shaders/sensor_radiance_to_electrons.spv",
        "src/shaders/sensor_radiance_to_electrons.spv"
    };

    std::vector<u32> radianceToElectronsCode, poissonNoiseCode, psfBlurHorizontalCode, psfBlurVerticalCode, quantizeToRadianceCode, fpnCode;

    for (const auto& basePath : shaderPaths) {
        String radiancePath = basePath.string();
        String poissonPath = radiancePath;
        String blurHPath = radiancePath;
        String blurVPath = radiancePath;
        String quantizePath = radiancePath;
        String fpnPath = radiancePath;

        size_t pos = radiancePath.find("radiance_to_electrons");
        if (pos != String::npos) {
            poissonPath.replace(pos, 21, "poisson_noise");
            blurHPath.replace(pos, 21, "psf_blur_horizontal");
            blurVPath.replace(pos, 21, "psf_blur_vertical");
            quantizePath.replace(pos, 21, "quantize_to_radiance");
            fpnPath.replace(fpnPath.find("radiance_to_electrons"), 21, "fpn");
        }

        radianceToElectronsCode = loadShaderFile(radiancePath);
        if (!radianceToElectronsCode.empty()) {
            poissonNoiseCode = loadShaderFile(poissonPath);
            psfBlurHorizontalCode = loadShaderFile(blurHPath);
            psfBlurVerticalCode = loadShaderFile(blurVPath);
            quantizeToRadianceCode = loadShaderFile(quantizePath);
            fpnCode = loadShaderFile(fpnPath);

            if (!poissonNoiseCode.empty() && !psfBlurHorizontalCode.empty() &&
                !psfBlurVerticalCode.empty() && !quantizeToRadianceCode.empty()) {
                QL_LOG_DEBUG("GPU Sensor: Loaded shaders from {}", radiancePath);
                if (fpnCode.empty()) {
                    QL_LOG_WARN("GPU Sensor: FPN shader not found at {}, FPN will be disabled", fpnPath);
                }
                break;
            }
        }
    }

    if (radianceToElectronsCode.empty() || poissonNoiseCode.empty() ||
        psfBlurHorizontalCode.empty() || psfBlurVerticalCode.empty() ||
        quantizeToRadianceCode.empty()) {
        QL_LOG_WARN("GPU Sensor: Could not load shader files, GPU sensor disabled");
        return;
    }

    // Create shader modules
    auto createShaderModule = [device](const std::vector<u32>& code) -> VkShaderModule {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(u32);
        createInfo.pCode = code.data();
        VkShaderModule module;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return module;
    };

    sensorRadianceToElectronsShader = createShaderModule(radianceToElectronsCode);
    sensorPoissonNoiseShader = createShaderModule(poissonNoiseCode);
    sensorPsfBlurHorizontalShader = createShaderModule(psfBlurHorizontalCode);
    sensorPsfBlurVerticalShader = createShaderModule(psfBlurVerticalCode);
    sensorQuantizeToRadianceShader = createShaderModule(quantizeToRadianceCode);
    if (!fpnCode.empty()) {
        sensorFpnShader = createShaderModule(fpnCode);
    }

    if (sensorRadianceToElectronsShader == VK_NULL_HANDLE ||
        sensorPoissonNoiseShader == VK_NULL_HANDLE ||
        sensorPsfBlurHorizontalShader == VK_NULL_HANDLE ||
        sensorPsfBlurVerticalShader == VK_NULL_HANDLE ||
        sensorQuantizeToRadianceShader == VK_NULL_HANDLE) {
        QL_LOG_WARN("GPU Sensor: Failed to create shader modules");
        return;
    }

    // Create descriptor set layout
    // binding 0: inputImage (sampled image or storage image)
    // binding 1: outputImage (storage image)
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &sensorDescriptorSetLayout) != VK_SUCCESS) {
        QL_LOG_WARN("GPU Sensor: Failed to create descriptor set layout");
        return;
    }

    // Create pipeline layout with push constants (max size for all passes)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128;  // Large enough for all pass structures

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &sensorDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &sensorPipelineLayout) != VK_SUCCESS) {
        QL_LOG_WARN("GPU Sensor: Failed to create pipeline layout");
        return;
    }

    // Create compute pipelines for each pass
    auto createComputePipeline = [device, this](VkShaderModule shader) -> VkPipeline {
        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shader;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = sensorPipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return pipeline;
    };

    sensorRadianceToElectronsPipeline = createComputePipeline(sensorRadianceToElectronsShader);
    sensorPoissonNoisePipeline = createComputePipeline(sensorPoissonNoiseShader);
    sensorPsfBlurHorizontalPipeline = createComputePipeline(sensorPsfBlurHorizontalShader);
    sensorPsfBlurVerticalPipeline = createComputePipeline(sensorPsfBlurVerticalShader);
    sensorQuantizeToRadiancePipeline = createComputePipeline(sensorQuantizeToRadianceShader);

    if (sensorRadianceToElectronsPipeline == VK_NULL_HANDLE ||
        sensorPoissonNoisePipeline == VK_NULL_HANDLE ||
        sensorPsfBlurHorizontalPipeline == VK_NULL_HANDLE ||
        sensorPsfBlurVerticalPipeline == VK_NULL_HANDLE ||
        sensorQuantizeToRadiancePipeline == VK_NULL_HANDLE) {
        QL_LOG_WARN("GPU Sensor: Failed to create compute pipelines");
        return;
    }

    // Create descriptor pool
    // 5 passes × 2 bindings = 10, plus FPN pass with 4 bindings = 14 total
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 14}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 6;  // 5 existing + 1 FPN

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &sensorDescriptorPool) != VK_SUCCESS) {
        QL_LOG_WARN("GPU Sensor: Failed to create descriptor pool");
        return;
    }

    // Create sensor images (same format as outputImage)
    sensorImage = std::make_unique<GpuImage>(
        allocator,
        device,
        width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    sensorTempImage = std::make_unique<GpuImage>(
        allocator,
        device,
        width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    // Transition sensor images to GENERAL layout
    TransitionImageLayoutImmediate(
        sensorImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    TransitionImageLayoutImmediate(
        sensorTempImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = sensorDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &sensorDescriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &sensorDescriptorSet) != VK_SUCCESS) {
        QL_LOG_WARN("GPU Sensor: Failed to allocate descriptor set");
        return;
    }

    // ========================================================================
    // FPN: Create descriptor set layout (4 bindings), pipeline, and FPN maps
    // ========================================================================
    if (sensorFpnShader != VK_NULL_HANDLE) {
        // FPN descriptor set layout: 4 storage images
        // binding 0: inputImage (electron image, in)
        // binding 1: outputImage (electron image, out)
        // binding 2: prnuMap (PRNU texture)
        // binding 3: dsnuMap (DSNU texture)
        std::vector<VkDescriptorSetLayoutBinding> fpnBindings(4);
        for (u32 i = 0; i < 4; ++i) {
            fpnBindings[i].binding = i;
            fpnBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            fpnBindings[i].descriptorCount = 1;
            fpnBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            fpnBindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo fpnLayoutInfo{};
        fpnLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        fpnLayoutInfo.bindingCount = static_cast<u32>(fpnBindings.size());
        fpnLayoutInfo.pBindings = fpnBindings.data();

        if (vkCreateDescriptorSetLayout(device, &fpnLayoutInfo, nullptr, &sensorFpnDescriptorSetLayout) != VK_SUCCESS) {
            QL_LOG_WARN("GPU Sensor: Failed to create FPN descriptor set layout");
        } else {
            // FPN pipeline layout
            VkPushConstantRange fpnPushRange{};
            fpnPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            fpnPushRange.offset = 0;
            fpnPushRange.size = 128;

            VkPipelineLayoutCreateInfo fpnPipeLayoutInfo{};
            fpnPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            fpnPipeLayoutInfo.setLayoutCount = 1;
            fpnPipeLayoutInfo.pSetLayouts = &sensorFpnDescriptorSetLayout;
            fpnPipeLayoutInfo.pushConstantRangeCount = 1;
            fpnPipeLayoutInfo.pPushConstantRanges = &fpnPushRange;

            if (vkCreatePipelineLayout(device, &fpnPipeLayoutInfo, nullptr, &sensorFpnPipelineLayout) != VK_SUCCESS) {
                QL_LOG_WARN("GPU Sensor: Failed to create FPN pipeline layout");
            } else {
                // FPN compute pipeline
                VkPipelineShaderStageCreateInfo fpnStageInfo{};
                fpnStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                fpnStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                fpnStageInfo.module = sensorFpnShader;
                fpnStageInfo.pName = "main";

                VkComputePipelineCreateInfo fpnPipeInfo{};
                fpnPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                fpnPipeInfo.stage = fpnStageInfo;
                fpnPipeInfo.layout = sensorFpnPipelineLayout;

                if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &fpnPipeInfo, nullptr, &sensorFpnPipeline) != VK_SUCCESS) {
                    QL_LOG_WARN("GPU Sensor: Failed to create FPN compute pipeline");
                }
            }
        }

        // Create FPN map images and allocate FPN descriptor set
        if (sensorFpnPipeline != VK_NULL_HANDLE) {
            // PRNU map: width x height, R32G32B32A32_SFLOAT (shader reads .r channel)
            fpnPrnuMap = std::make_unique<GpuImage>(
                allocator, device,
                width, height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );

            // DSNU map: width x height, R32G32B32A32_SFLOAT (shader reads .r channel)
            fpnDsnuMap = std::make_unique<GpuImage>(
                allocator, device,
                width, height,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );

            // Transition FPN maps to GENERAL layout
            TransitionImageLayoutImmediate(
                fpnPrnuMap->GetImage(),
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL
            );
            TransitionImageLayoutImmediate(
                fpnDsnuMap->GetImage(),
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL
            );

            // Allocate FPN descriptor set
            VkDescriptorSetAllocateInfo fpnAllocInfo{};
            fpnAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            fpnAllocInfo.descriptorPool = sensorDescriptorPool;
            fpnAllocInfo.descriptorSetCount = 1;
            fpnAllocInfo.pSetLayouts = &sensorFpnDescriptorSetLayout;

            if (vkAllocateDescriptorSets(device, &fpnAllocInfo, &sensorFpnDescriptorSet) != VK_SUCCESS) {
                QL_LOG_WARN("GPU Sensor: Failed to allocate FPN descriptor set");
                sensorFpnPipeline = VK_NULL_HANDLE;  // Disable FPN
            } else {
                QL_LOG_INFO("GPU Sensor: FPN pipeline and maps created successfully");
            }
        }
    }

    sensorInitialized = true;
    QL_LOG_INFO("GPU Sensor: Compute pipeline created successfully");
}

// ============================================================================
// GPU Sensor: Generate and Upload FPN Maps
// ============================================================================

void ExternalRenderContext::Impl::GenerateAndUploadFPNMaps() {
    if (!sensorInitialized || sensorFpnPipeline == VK_NULL_HANDLE) return;
    if (fpnMapsGenerated) return;

    const auto& params = gpuSensorParams;
    const u32 width = this->width;
    const u32 height = this->height;
    auto allocator = contextAdapter->GetAllocator();

    QL_LOG_INFO("GPU Sensor: Generating FPN maps {}x{} (PRNU sigma={:.2f}%, DSNU sigma={:.1f} e-)",
                width, height, params.prnuSigma * 100.0f, params.dsnuSigma_e);

    std::mt19937 rng(42);  // Fixed seed for reproducible FPN pattern

    // Helper: 1D Gaussian kernel
    auto makeGaussianKernel = [](f32 sigma) -> std::vector<f32> {
        i32 radius = static_cast<i32>(std::ceil(3.0f * sigma));
        std::vector<f32> kernel(2 * radius + 1);
        f32 sum = 0.0f;
        for (i32 i = -radius; i <= radius; ++i) {
            kernel[i + radius] = std::exp(-0.5f * (i * i) / (sigma * sigma));
            sum += kernel[i + radius];
        }
        for (auto& v : kernel) v /= sum;
        return kernel;
    };

    // ---- Generate PRNU map (vertical stripes = per-column) ----
    // RGBA float data for the full image
    std::vector<f32> prnuData(width * height * 4, 0.0f);

    if (params.prnuSigma > 1e-6f) {
        std::normal_distribution<f32> prnuDist(0.0f, params.prnuSigma);

        // Per-column random values
        std::vector<f32> columnNoise(width);
        for (u32 x = 0; x < width; ++x) {
            columnNoise[x] = prnuDist(rng);
        }

        // Smooth with sigma=8 for wider stripes
        auto kernel = makeGaussianKernel(8.0f);
        i32 radius = static_cast<i32>(kernel.size()) / 2;
        std::vector<f32> smoothed(width);
        for (u32 x = 0; x < width; ++x) {
            f32 sum = 0.0f;
            for (i32 k = -radius; k <= radius; ++k) {
                i32 xk = std::clamp(static_cast<i32>(x) + k, 0, static_cast<i32>(width) - 1);
                sum += columnNoise[xk] * kernel[k + radius];
            }
            smoothed[x] = sum;
        }

        // Expand to 2D with 10% pixel-level variation
        std::normal_distribution<f32> pixelNoise(0.0f, params.prnuSigma * 0.1f);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                prnuData[(y * width + x) * 4 + 0] = smoothed[x] + pixelNoise(rng);
            }
        }

        // Renormalize to target sigma
        f32 mean = 0.0f;
        for (u32 i = 0; i < width * height; ++i) mean += prnuData[i * 4];
        mean /= static_cast<f32>(width * height);

        f32 variance = 0.0f;
        for (u32 i = 0; i < width * height; ++i) {
            f32 diff = prnuData[i * 4] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<f32>(width * height);
        f32 currentSigma = std::sqrt(variance);

        if (currentSigma > 1e-6f) {
            f32 scale = params.prnuSigma / currentSigma;
            for (u32 i = 0; i < width * height; ++i) {
                prnuData[i * 4] = (prnuData[i * 4] - mean) * scale;
            }
        }
    }

    // ---- Generate DSNU map (horizontal stripes = per-row) ----
    std::vector<f32> dsnuData(width * height * 4, 0.0f);

    if (params.dsnuSigma_e > 1e-6f) {
        std::normal_distribution<f32> dsnuDist(0.0f, params.dsnuSigma_e);

        // Per-row random values
        std::vector<f32> rowNoise(height);
        for (u32 y = 0; y < height; ++y) {
            rowNoise[y] = dsnuDist(rng);
        }

        // Smooth with sigma=5 for wider stripes
        auto kernel = makeGaussianKernel(5.0f);
        i32 radius = static_cast<i32>(kernel.size()) / 2;
        std::vector<f32> smoothed(height);
        for (u32 y = 0; y < height; ++y) {
            f32 sum = 0.0f;
            for (i32 k = -radius; k <= radius; ++k) {
                i32 yk = std::clamp(static_cast<i32>(y) + k, 0, static_cast<i32>(height) - 1);
                sum += rowNoise[yk] * kernel[k + radius];
            }
            smoothed[y] = sum;
        }

        // Expand to 2D with 10% pixel-level variation
        std::normal_distribution<f32> pixelNoise(0.0f, params.dsnuSigma_e * 0.1f);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                dsnuData[(y * width + x) * 4 + 0] = smoothed[y] + pixelNoise(rng);
            }
        }

        // Renormalize to target sigma
        f32 mean = 0.0f;
        for (u32 i = 0; i < width * height; ++i) mean += dsnuData[i * 4];
        mean /= static_cast<f32>(width * height);

        f32 variance = 0.0f;
        for (u32 i = 0; i < width * height; ++i) {
            f32 diff = dsnuData[i * 4] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<f32>(width * height);
        f32 currentSigma = std::sqrt(variance);

        if (currentSigma > 1e-6f) {
            f32 scale = params.dsnuSigma_e / currentSigma;
            for (u32 i = 0; i < width * height; ++i) {
                dsnuData[i * 4] = (dsnuData[i * 4] - mean) * scale;
            }
        }
    }

    // ---- Upload PRNU and DSNU maps to GPU ----
    const VkDeviceSize mapSize = width * height * 4 * sizeof(f32);

    // Upload PRNU map
    {
        GpuBuffer stagingBuffer(allocator, mapSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        stagingBuffer.Upload(prnuData.data(), mapSize);

        TransitionImageLayoutImmediate(
            fpnPrnuMap->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );

        CommandHelper::ExecuteImmediate(*contextAdapter, [&](VkCommandBuffer cmd) {
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {width, height, 1};

            vkCmdCopyBufferToImage(cmd, stagingBuffer.GetHandle(),
                fpnPrnuMap->GetImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        });

        TransitionImageLayoutImmediate(
            fpnPrnuMap->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL
        );
    }

    // Upload DSNU map
    {
        GpuBuffer stagingBuffer(allocator, mapSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        stagingBuffer.Upload(dsnuData.data(), mapSize);

        TransitionImageLayoutImmediate(
            fpnDsnuMap->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );

        CommandHelper::ExecuteImmediate(*contextAdapter, [&](VkCommandBuffer cmd) {
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {width, height, 1};

            vkCmdCopyBufferToImage(cmd, stagingBuffer.GetHandle(),
                fpnDsnuMap->GetImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        });

        TransitionImageLayoutImmediate(
            fpnDsnuMap->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL
        );
    }

    // Update FPN descriptor set with map images
    VkDescriptorImageInfo fpnPrnuInfo{};
    fpnPrnuInfo.imageView = fpnPrnuMap->GetView();
    fpnPrnuInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo fpnDsnuInfo{};
    fpnDsnuInfo.imageView = fpnDsnuMap->GetView();
    fpnDsnuInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet fpnWrites[2] = {};
    fpnWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    fpnWrites[0].dstSet = sensorFpnDescriptorSet;
    fpnWrites[0].dstBinding = 2;
    fpnWrites[0].descriptorCount = 1;
    fpnWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    fpnWrites[0].pImageInfo = &fpnPrnuInfo;

    fpnWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    fpnWrites[1].dstSet = sensorFpnDescriptorSet;
    fpnWrites[1].dstBinding = 3;
    fpnWrites[1].descriptorCount = 1;
    fpnWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    fpnWrites[1].pImageInfo = &fpnDsnuInfo;

    vkUpdateDescriptorSets(device, 2, fpnWrites, 0, nullptr);

    fpnMapsGenerated = true;
    QL_LOG_INFO("GPU Sensor: FPN maps generated and uploaded ({}x{})", width, height);
}

// ============================================================================
// GPU Sensor Chain Execution
// ============================================================================

void ExternalRenderContext::Impl::ExecuteGPUSensorChain(VkCommandBuffer cmd, u32 width, u32 height) {
    if (!sensorInitialized) return;

    // Generate FPN maps on first use (lazy initialization)
    if (!fpnMapsGenerated && sensorFpnPipeline != VK_NULL_HANDLE) {
        GenerateAndUploadFPNMaps();
    }

    const auto& params = gpuSensorParams;

    // Helper: Insert pipeline barrier between compute passes
    auto insertBarrier = [cmd]() {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    };

    // Initial barrier: wait for ray tracing to finish
    VkMemoryBarrier initialBarrier{};
    initialBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    initialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    initialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &initialBarrier, 0, nullptr, 0, nullptr);

    // Calculate PSF sigma from f-number and wavelength
    // σ_psf ≈ 1.22 × λ × f# / pixel_pitch (result is in pixels)
    // wavelength_nm * 1e-9 = wavelength in meters
    // pixelPitch_um * 1e-6 = pixel pitch in meters
    // Division gives result directly in pixels
    f32 psfSigma = 1.22f * (params.wavelength_nm * 1e-9f) * params.fNumber / (params.pixelPitch_um * 1e-6f);
    // Clamp to reasonable range
    psfSigma = std::max(0.1f, std::min(psfSigma, 10.0f));
    u32 kernelRadius = static_cast<u32>(std::ceil(3.0f * psfSigma));

    // ========================================================================
    // Pass 1: PSF Blur Horizontal (on radiance, matching CPU order)
    // ========================================================================
    {
        struct PushConstants {
            f32 sigma;
            u32 kernelRadius;
            u32 imageWidth;
            u32 imageHeight;
            u32 passIndex;
            u32 padding[3];
        } pushConstants;

        pushConstants.sigma = psfSigma;
        pushConstants.kernelRadius = kernelRadius;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;
        pushConstants.passIndex = 0;  // Horizontal

        // Update descriptor set: outputImage → sensorTempImage
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView = outputImage->GetView();
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = sensorTempImage->GetView();
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sensorDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPsfBlurHorizontalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPipelineLayout, 0, 1, &sensorDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    // ========================================================================
    // Pass 2: PSF Blur Vertical (on radiance, matching CPU order)
    // ========================================================================
    {
        struct PushConstants {
            f32 sigma;
            u32 kernelRadius;
            u32 imageWidth;
            u32 imageHeight;
            u32 passIndex;
            u32 padding[3];
        } pushConstants;

        pushConstants.sigma = psfSigma;
        pushConstants.kernelRadius = kernelRadius;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;
        pushConstants.passIndex = 1;  // Vertical

        // Update descriptor set: sensorTempImage → sensorImage
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView = sensorTempImage->GetView();
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = sensorImage->GetView();
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sensorDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPsfBlurVerticalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPipelineLayout, 0, 1, &sensorDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    // ========================================================================
    // Pass 3: Radiance → Photo-electrons (after PSF blur on clean radiance)
    // ========================================================================
    {
        struct PushConstants {
            f32 quantumEfficiency;
            f32 pixelPitch_um;
            f32 focalLength_mm;
            f32 fNumber;
            f32 integrationTime_s;
            f32 wellCapacity_e;
            f32 wavelength_nm;
            f32 darkCurrent_e_s;
            u32 enableDarkCurrent;
            u32 enableVignetting;
            f32 fov_deg;
            u32 isTelecentric;
            u32 imageWidth;
            u32 imageHeight;
            u32 padding[2];
        } pushConstants;

        pushConstants.quantumEfficiency = params.quantumEfficiency;
        pushConstants.pixelPitch_um = params.pixelPitch_um;
        pushConstants.focalLength_mm = params.focalLength_mm;
        pushConstants.fNumber = params.fNumber;
        pushConstants.integrationTime_s = params.integrationTime_s;
        pushConstants.wellCapacity_e = params.wellCapacity_e;
        pushConstants.wavelength_nm = params.wavelength_nm;
        pushConstants.darkCurrent_e_s = params.darkCurrent_e_s;
        pushConstants.enableDarkCurrent = params.enableDarkCurrent ? 1u : 0u;
        pushConstants.enableVignetting = params.enableVignetting ? 1u : 0u;
        pushConstants.fov_deg = params.fov_deg;
        pushConstants.isTelecentric = params.isTelecentric ? 1u : 0u;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;

        // Update descriptor set: sensorImage → sensorTempImage
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView = sensorImage->GetView();
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = sensorTempImage->GetView();
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sensorDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorRadianceToElectronsPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPipelineLayout, 0, 1, &sensorDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    // ========================================================================
    // Pass 4: Poisson + Read Noise
    // ========================================================================
    {
        struct PushConstants {
            u32 frameIndex;
            u32 enablePoissonNoise;
            f32 readNoise_e_rms;
            u32 enableReadNoise;
            f32 wellCapacity_e;
            u32 imageWidth;
            u32 imageHeight;
            u32 padding;
        } pushConstants;

        pushConstants.frameIndex = frameIndex;
        pushConstants.enablePoissonNoise = params.enablePoissonNoise ? 1u : 0u;
        pushConstants.readNoise_e_rms = params.readNoise_e_rms;
        pushConstants.enableReadNoise = params.enableReadNoise ? 1u : 0u;
        pushConstants.wellCapacity_e = params.wellCapacity_e;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;

        // Update descriptor set: sensorTempImage (in/out)
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = sensorTempImage->GetView();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &imageInfo;

        writes[1] = writes[0];
        writes[1].dstBinding = 1;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPoissonNoisePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPipelineLayout, 0, 1, &sensorDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    // ========================================================================
    // Pass 5: FPN (PRNU + DSNU) - in electron domain
    // ========================================================================
    if (sensorFpnPipeline != VK_NULL_HANDLE && fpnMapsGenerated &&
        params.enableFPN) {
        struct PushConstants {
            u32 enableFPN;
            u32 enableNUC;
            f32 nucEfficiency;
            u32 imageWidth;
            u32 imageHeight;
            u32 padding[3];
        } pushConstants;

        pushConstants.enableFPN = 1u;
        pushConstants.enableNUC = params.enableNUC ? 1u : 0u;
        pushConstants.nucEfficiency = params.nucEfficiency;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;

        // Update FPN descriptor set: bindings 0,1 = sensorTempImage (in/out)
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = sensorTempImage->GetView();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorFpnDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &imageInfo;

        writes[1] = writes[0];
        writes[1].dstBinding = 1;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorFpnPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            sensorFpnPipelineLayout, 0, 1,
            &sensorFpnDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorFpnPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    // ========================================================================
    // Pass 6: Quantize → Radiance
    // ========================================================================
    {
        struct PushConstants {
            f32 gain;
            u32 bitDepth;
            f32 quantumEfficiency;
            f32 pixelPitch_um;
            f32 focalLength_mm;
            f32 fNumber;
            f32 integrationTime_s;
            f32 wavelength_nm;
            f32 darkCurrent_e_s;
            u32 enableDarkCurrent;
            u32 imageWidth;
            u32 imageHeight;
        } pushConstants;

        pushConstants.gain = params.gain;
        pushConstants.bitDepth = params.bitDepth;
        pushConstants.quantumEfficiency = params.quantumEfficiency;
        pushConstants.pixelPitch_um = params.pixelPitch_um;
        pushConstants.focalLength_mm = params.focalLength_mm;
        pushConstants.fNumber = params.fNumber;
        pushConstants.integrationTime_s = params.integrationTime_s;
        pushConstants.wavelength_nm = params.wavelength_nm;
        pushConstants.darkCurrent_e_s = params.darkCurrent_e_s;
        pushConstants.enableDarkCurrent = params.enableDarkCurrent ? 1u : 0u;
        pushConstants.imageWidth = width;
        pushConstants.imageHeight = height;

        // Update descriptor set: sensorTempImage → sensorImage (final output)
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView = sensorTempImage->GetView();
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = sensorImage->GetView();
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sensorDescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &inputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sensorDescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorQuantizeToRadiancePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sensorPipelineLayout, 0, 1, &sensorDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, sensorPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        insertBarrier();
    }

    //QL_LOG_DEBUG("GPU Sensor: Executed 5-pass sensor chain (PSF sigma={:.2f} pixels)", psfSigma);
}

void ExternalRenderContext::Impl::ComputeImageMinMax(f32& outMin, f32& outMax) {
    // Read back image pixels
    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        *contextAdapter,
        outputImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        width,
        height
    );

    // Collect valid luminance values and find absolute range
    std::vector<f32> luminances;
    luminances.reserve(pixels.size() / 4);

    f32 absMin = std::numeric_limits<f32>::max();
    f32 absMax = std::numeric_limits<f32>::lowest();

    for (size_t i = 0; i < pixels.size(); i += 4) {
        f32 r = pixels[i];
        f32 g = pixels[i + 1];
        f32 b = pixels[i + 2];

        // Skip invalid pixels
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
            continue;

        // BT.709 luminance
        f32 lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        luminances.push_back(lum);
        absMin = std::min(absMin, lum);
        absMax = std::max(absMax, lum);
    }

    // Fallback if no valid pixels
    if (luminances.empty() || absMin >= absMax) {
        outMin = 0.0f;
        outMax = 1.0f;
        return;
    }

    // Check if range is "narrow" enough to not need percentile clipping
    // If max/min ratio < 100, use absolute min/max (like SWIR/MWIR)
    f32 ratio = (absMin > 1e-10f) ? (absMax / absMin) : (absMax - absMin + 1.0f);
    if (ratio < 100.0f) {
        outMin = absMin;
        outMax = absMax;
        //QL_LOG_DEBUG("CLAHE: Using absolute min/max (ratio {:.1f}x)", ratio);
        return;
    }

    // Wide range detected - use percentile-based normalization
    // Build histogram for percentile calculation (more efficient than sorting)
    constexpr size_t histBins = 65536;
    std::vector<u32> histogram(histBins, 0);

    // Map luminance to histogram bins
    f32 scale = (histBins - 1) / (absMax - absMin);
    for (f32 lum : luminances) {
        size_t bin = static_cast<size_t>((lum - absMin) * scale);
        bin = std::min(bin, histBins - 1);
        histogram[bin]++;
    }

    // Find 1st and 99th percentile bins
    size_t totalPixels = luminances.size();
    size_t target01 = static_cast<size_t>(totalPixels * 0.01f);
    size_t target99 = static_cast<size_t>(totalPixels * 0.99f);

    size_t cumulative = 0;
    size_t bin01 = 0, bin99 = histBins - 1;

    for (size_t i = 0; i < histBins; ++i) {
        cumulative += histogram[i];
        if (cumulative >= target01 && bin01 == 0) {
            bin01 = i;
        }
        if (cumulative >= target99) {
            bin99 = i;
            break;
        }
    }

    // Convert bins back to luminance values
    f32 invScale = (absMax - absMin) / (histBins - 1);
    outMin = absMin + bin01 * invScale;
    outMax = absMin + bin99 * invScale;

    // Ensure valid range
    if (outMin >= outMax) {
        outMin = absMin;
        outMax = absMax;
    }

    //QL_LOG_DEBUG("CLAHE: Using 1st/99th percentile (ratio {:.1f}x, clipped [{:.6g}, {:.6g}] -> [{:.6g}, {:.6g}])",
    //             ratio, absMin, absMax, outMin, outMax);
}

void ExternalRenderContext::Impl::ExecuteCLAHE(VkCommandBuffer cmd, u32 width, u32 height) {
    if (!claheInitialized) return;

    // Dynamically update input image binding based on sensor state
    VkImageView inputView = outputImage->GetView();

    // If sensor is enabled, CLAHE should process sensor output
    if (gpuSensorEnabled && sensorInitialized && sensorImage) {
        inputView = sensorImage->GetView();
        //QL_LOG_DEBUG("CLAHE: Processing sensor output");
    } else {
        //QL_LOG_DEBUG("CLAHE: Processing raw output");
    }

    // Update descriptor set binding 0 (input image)
    VkDescriptorImageInfo inputImageInfo{};
    inputImageInfo.imageView = inputView;
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inputImageInfo.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet inputWrite{};
    inputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    inputWrite.dstSet = claheDescriptorSet;
    inputWrite.dstBinding = 0;
    inputWrite.dstArrayElement = 0;
    inputWrite.descriptorCount = 1;
    inputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    inputWrite.pImageInfo = &inputImageInfo;

    vkUpdateDescriptorSets(device, 1, &inputWrite, 0, nullptr);

    // tileSize from UI represents tile grid dimension (e.g., 8 = 8x8 grid)
    // NOT pixels per tile
    u32 tileCountX = static_cast<u32>(claheParams.tileSize);
    u32 tileCountY = static_cast<u32>(claheParams.tileSize);

    // Clamp to max tiles (matching buffer allocation)
    tileCountX = std::min(tileCountX, 64u);
    tileCountY = std::min(tileCountY, 64u);

    // Ensure at least 1 tile
    tileCountX = std::max(tileCountX, 1u);
    tileCountY = std::max(tileCountY, 1u);

    //QL_LOG_DEBUG("CLAHE: Executing with {}x{} tiles on {}x{} image",
    //             tileCountX, tileCountY, width, height);

    // Use cached min/max values from previous frame
    // This avoids GPU sync issues during command buffer recording
    // First frame uses default values until cache is populated
    f32 inputMin = cachedImageMin;
    f32 inputMax = cachedImageMax;

    // Push constants structure (must match shader)
    struct CLAHEPushConstants {
        u32 imageWidth;
        u32 imageHeight;
        u32 tileCountX;
        u32 tileCountY;
        f32 clipLimit;
        u32 luminanceOnly;
        f32 inputMin;
        f32 inputMax;
        u32 passIndex;
        u32 padding[3];
    };

    CLAHEPushConstants pushConstants{};
    pushConstants.imageWidth = width;
    pushConstants.imageHeight = height;
    pushConstants.tileCountX = tileCountX;
    pushConstants.tileCountY = tileCountY;
    pushConstants.clipLimit = claheParams.clipLimit;
    pushConstants.luminanceOnly = claheParams.luminanceOnly ? 1 : 0;
    pushConstants.inputMin = inputMin;
    pushConstants.inputMax = inputMax;

    // Memory barrier: wait for ray tracing to finish
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );

    // Clear histogram buffer to zero before Pass 1
    // Without this, garbage data from uninitialized GPU memory causes
    // corrupted histograms and eventual TDR timeout (GPU crash)
    vkCmdFillBuffer(cmd, claheHistogramBuffer->GetHandle(), 0, VK_WHOLE_SIZE, 0);

    // Buffer barrier: fill write → compute shader read/write
    VkBufferMemoryBarrier fillBarrier{};
    fillBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    fillBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fillBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fillBarrier.buffer = claheHistogramBuffer->GetHandle();
    fillBarrier.offset = 0;
    fillBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &fillBarrier,
        0, nullptr
    );

    // Pass 1: Build histograms
    // Dispatch one workgroup per tile
    pushConstants.passIndex = 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheHistogramPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            clahePipelineLayout, 0, 1,
                            &claheDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, clahePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, tileCountX, tileCountY, 1);

    // Barrier between passes
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );

    // Pass 2: Clip, redistribute, compute CDF
    pushConstants.passIndex = 1;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheCdfPipeline);
    vkCmdPushConstants(cmd, clahePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, tileCountX, tileCountY, 1);

    // Barrier between passes
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );

    // Pass 3: Apply interpolated mapping
    pushConstants.passIndex = 2;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheApplyPipeline);
    vkCmdPushConstants(cmd, clahePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushConstants), &pushConstants);
    // Dispatch one thread per pixel
    u32 groupCountX = (width + 15) / 16;
    u32 groupCountY = (height + 15) / 16;
    vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

    // Final barrier: CLAHE output ready for blit
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );
}

void ExternalRenderContext::Impl::TransitionImageLayoutImmediate(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout) {

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

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

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);

    (void)format;  // Format used for barrier determination in more complex cases
}

// ============================================================================
// NN Atmosphere Configuration
// ============================================================================

void ExternalRenderContext::SetAtmosphere(const AtmosphereNNConfig& config) {
    m_impl->atmosphereConfig = config;
    m_impl->atmosModelPack.reset();
    m_impl->atmosBakeKey = 0;  // Force rebake (or disable-upload) next frame

    if (config.enabled && !config.modelPackDir.empty()) {
        // Throws if the directory does not exist -- hard error, no fallback
        m_impl->atmosModelPack =
            std::make_unique<AtmosModelPack>(config.modelPackDir);
    }

    ResetAccumulation();
    QL_LOG_INFO("NN atmosphere config updated: {} (preset '{}')",
                config.enabled ? "enabled" : "disabled", config.preset);
}

const AtmosphereNNConfig& ExternalRenderContext::GetAtmosphere() const {
    return m_impl->atmosphereConfig;
}

// ============================================================================
// Environment Map (IBL)
// ============================================================================

Result<void, String> ExternalRenderContext::LoadEnvironmentMap(const String& hdrPath) {
    // The previous scene's cubemap is freed below; a frame the host submitted may
    // still be reading it.
    vkDeviceWaitIdle(m_impl->device);

    auto loaded = rendercore::EnvironmentCubemap::Load(*m_impl->contextAdapter, hdrPath);
    if (!loaded.has_value()) {
        return Result<void, String>::Err(loaded.error());
    }
    m_impl->envMap = std::move(loaded.value());

    if (m_impl->pipeline) {
        m_impl->pipeline->BindPrefilteredEnvMap(m_impl->envMap.View(), m_impl->envMap.Sampler());
    }
    m_impl->hasCustomEnvMap = true;
    ResetAccumulation();
    return Result<void, String>::Ok();
}

bool ExternalRenderContext::HasEnvironmentMap() const {
    return m_impl->hasCustomEnvMap;
}

} // namespace quantiloom
