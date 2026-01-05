/**
 * @file ExternalRenderContext.cpp
 * @brief Implementation of external Vulkan context injection API
 *
 * @author wtflmao
 */

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

#include "core/Log.hpp"
#include "io/GltfLoader.hpp"
#include "io/ImageIO.hpp"

#include <vk_mem_alloc.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <chrono>

namespace quantiloom {

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
    glm::vec2 _padding0;
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
};

static_assert(sizeof(MaterialDataCPU) == 96, "MaterialDataCPU size mismatch");

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

    // IBL resources
    std::unique_ptr<GpuImage> brdfLutTexture;
    std::unique_ptr<GpuImage> envMapImage;
    VkSampler brdfLutSampler = VK_NULL_HANDLE;

    // Texture manager
    std::unique_ptr<TextureManager> textureManager;

    // Ray tracing pipeline
    std::unique_ptr<RayTracingPipeline> pipeline;

    // Command pool for internal operations
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // Rendering state
    SpectralMode spectralMode = SpectralMode::RGB_Fused;
    f32 wavelength_nm = 550.0f;
    u32 spp = 1;
    LightingParams lightingParams;

    // Accumulation
    u32 accumulatedSamples = 0;
    u32 frameIndex = 0;

    // Statistics
    f32 lastFrameTimeMs = 0.0f;
    std::chrono::steady_clock::time_point frameStartTime;

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

    return Result<void, String>::Err("No scene.gltf specified in config");
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
    VkImageView targetImageView,
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
    m_impl->pipeline->SetCameraData(cameraData);

    // Set sampling parameters
    u32 randomSeed = m_impl->frameIndex * 997 + m_impl->accumulatedSamples * 1009;
    m_impl->pipeline->SetSamplingParams(
        m_impl->frameIndex,
        m_impl->accumulatedSamples,
        m_impl->spp,
        randomSeed
    );

    // Execute ray tracing
    m_impl->pipeline->TraceRays(cmd, width, height);

    // TODO: Copy from outputImage to target swapchain image
    // This requires image blit/copy with format conversion
    (void)targetImageView;  // Unused for now

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
    // TODO: Implement in Phase 2
    (void)nodeIndex;
    (void)transform;
    QL_LOG_WARN("ExternalRenderContext::SetNodeTransform not implemented yet");
}

void ExternalRenderContext::UpdateMaterial(u32 materialIndex, const Material& material) {
    // TODO: Implement in Phase 2
    (void)materialIndex;
    (void)material;
    QL_LOG_WARN("ExternalRenderContext::UpdateMaterial not implemented yet");
}

void ExternalRenderContext::RebuildAccelerationStructure() {
    // TODO: Implement in Phase 2
    QL_LOG_WARN("ExternalRenderContext::RebuildAccelerationStructure not implemented yet");
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

// ============================================================================
// Private Helper Methods
// ============================================================================

void ExternalRenderContext::BuildAccelerationStructures() {
    if (!m_impl->scene) return;

    QL_LOG_INFO("Building acceleration structures...");

    // Clear existing
    m_impl->blasList.clear();
    m_impl->tlas.reset();

    // Build BLAS for each primitive
    for (const auto& mesh : m_impl->scene->meshes) {
        for (const auto& primitive : mesh.primitives) {
            m_impl->blasList.emplace_back(
                std::make_unique<BLAS>(*m_impl->contextAdapter, primitive)
            );
        }
    }

    // Build TLAS
    m_impl->tlas = std::make_unique<TLAS>(*m_impl->contextAdapter);

    // Execute builds on GPU
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
        // Build all BLAS
        for (auto& blas : m_impl->blasList) {
            blas->Build(cmd);
        }

        // Add instances to TLAS
        size_t blasIndex = 0;
        for (const auto& node : m_impl->scene->nodes) {
            const Mesh& mesh = m_impl->scene->meshes[node.meshIndex];

            for (const auto& primitive : mesh.primitives) {
                m_impl->tlas->AddInstance(
                    *m_impl->blasList[blasIndex],
                    primitive.materialId,
                    node.transform
                );
                ++blasIndex;
            }
        }

        m_impl->tlas->Build(cmd);
    });

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
        cpuMat.emissiveFactor = mat.emissiveFactor;
        cpuMat.emissiveTextureIndex = mat.emissiveTextureIndex;
        cpuMat.alphaMode = static_cast<u32>(mat.alphaMode);
        cpuMat.alphaCutoff = mat.alphaCutoff;
        cpuMat.spectralAlbedo = mat.spectralAlbedo;
        cpuMat.spectralReflectanceCurveIndex = -1;
        cpuMat.irEmissivity = 0.0f;
        cpuMat.irTransmittance = 0.0f;
        cpuMat.irTemperature_K = 0.0f;
        cpuMat.complexRefractiveIndexIndex = -1;
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

    // Create pipeline using context adapter
    m_impl->pipeline = std::make_unique<RayTracingPipeline>(
        *m_impl->contextAdapter,
        "raygen.spv",
        "closesthit.spv",
        "miss.spv"
    );

    // Bind resources
    m_impl->pipeline->BindOutputImage(*m_impl->outputImage);
    m_impl->pipeline->BindAccelerationStructure(m_impl->tlas->GetHandle());
    m_impl->pipeline->BindLUTBuffer(*m_impl->lightingParamsBuffer);

    // Bind geometry buffers from first BLAS
    if (!m_impl->blasList.empty()) {
        const auto& firstBlas = *m_impl->blasList[0];
        const GpuBuffer* uvBuffer = firstBlas.HasUVs() ? &firstBlas.GetUVBuffer() : nullptr;
        m_impl->pipeline->BindGeometryBuffers(
            firstBlas.GetVertexBuffer(),
            firstBlas.GetIndexBuffer(),
            uvBuffer
        );
        m_impl->pipeline->BindTangentBuffer(firstBlas.GetTangentBuffer());
        m_impl->pipeline->BindNormalBuffer(firstBlas.GetNormalBuffer());
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

} // namespace quantiloom
