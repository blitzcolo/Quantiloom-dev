#include "RayTracingPipeline.hpp"
#include "DeferredBufferUploads.hpp"
#include "RtBindings.hpp"
#include "ShaderBinary.hpp"
#include "renderer/GpuAlloc.hpp"
#include "renderer/TextureTransitions.hpp"
#include "core/ResourceLocator.hpp"
#include "renderer/FusionRecordLayout.hpp"
#include "core/Log.hpp"
#include "core/SpectralData.hpp"
#include "core/Image.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <span>
#include <stdexcept>
#include <cstring>
#include <filesystem>

namespace quantiloom {

// ============================================================================
// Ray Recursion Depth
// ============================================================================
// Every ray is traced from raygen: src/shaders/path_driver.hlsli walks each
// path with a bounded frame stack, and the closest-hit and miss shaders only
// report where a ray landed, tracing nothing. So no shader nests a TraceRay
// inside another and the depth is 1, the minimum the specification lets a
// device report. Path length is MAX_PATH_DEPTH in src/shaders/shade_vertex.hlsli
// and costs frames, not recursion.
//
// Still validated against the device below rather than clamped: a stage that
// traced again would need more, and running past the declared depth is
// undefined behaviour that surfaces as a device loss somewhere unrelated.
// ============================================================================

static constexpr u32 kMaxPipelineRayRecursionDepth = 1;

// A shader as the gpu:: kernel factories take it: the words LoadSPIRV read, as
// bytes, named after the file.
static gpu::ShaderCode ToShaderCode(const std::string& path, const std::vector<u32>& words) {
    gpu::ShaderCode code;
    code.name = std::filesystem::path(path).filename().string();
    code.path = path;
    code.bytes.resize(words.size() * sizeof(u32));
    std::memcpy(code.bytes.data(), words.data(), code.bytes.size());
    return code;
}

// A whole-buffer write of a storage buffer binding, with the kind its row in
// RtBindings.hpp declares (read-only or read-write; both are storage buffers).
static gpu::BindingWrite RtBufferWrite(u32 binding, gpu::BufferRef buffer) {
    return gpu::BindingWrite::Buffer(binding, rendercore::kRtBindingTemplate[binding].kind, buffer);
}

static void WriteStorageImage(gpu::Device& device, gpu::BindGroup set, u32 binding,
                              gpu::TextureRef image) {
    const gpu::BindingWrite writes[] = {gpu::BindingWrite::StorageTexture(binding, image)};
    device.WriteBindGroup(set, writes);
}

static void WriteStorageBuffer(gpu::Device& device, gpu::BindGroup set, u32 binding,
                               gpu::BufferRef buffer) {
    const gpu::BindingWrite writes[] = {RtBufferWrite(binding, buffer)};
    device.WriteBindGroup(set, writes);
}

// A one-texel storage image, the shape of every fallback the pipeline binds.
static gpu::TextureDesc FallbackStorageImage(gpu::Format format) {
    gpu::TextureDesc desc;
    desc.width = 1;
    desc.height = 1;
    desc.format = format;
    desc.usage = gpu::TextureUsage::Storage | gpu::TextureUsage::TransferDst;
    desc.memory = gpu::MemoryClass::DeviceLocal;
    return desc;
}

// A host-written storage buffer, which is what every fallback and staging block here is.
static gpu::BufferDesc HostStorage(u64 bytes, gpu::BufferUsage extra = gpu::BufferUsage::None) {
    return gpu::BufferDesc{bytes, gpu::BufferUsage::Storage | extra, gpu::MemoryClass::Upload, 0};
}

// ============================================================================
// Pipeline Cache Static Methods
// ============================================================================

std::unique_ptr<gpu::PipelineCache> RayTracingPipeline::LoadPipelineCache(gpu::Device& device,
                                                                       const std::string& cachePath) {
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

    // Data the driver rejects (corrupted or incompatible) gets a warning and an
    // empty cache instead; the error is only that the empty one failed too.
    auto cache = device.CreatePipelineCache(
        std::as_bytes(std::span<const char>(cacheData)));
    if (!cache.has_value()) {
        QL_LOG_ERROR("{}", cache.error());
        return nullptr;
    }

    return std::move(cache.value());
}

bool RayTracingPipeline::SavePipelineCache(const gpu::PipelineCache* cache, const std::string& cachePath) {
    if (cache == nullptr) {
        return false;
    }

    // Get cache data: its size, then the data
    auto cacheData = cache->Serialize();
    if (!cacheData.has_value()) {
        QL_LOG_WARN("{}", cacheData.error());
        return false;
    }
    const Vector<std::byte>& bytes = cacheData.value();

    // Write to disk
    std::ofstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        QL_LOG_WARN("Failed to open pipeline cache file for writing: {}", cachePath);
        return false;
    }

    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();

    QL_LOG_INFO("Saved pipeline cache to disk: {} ({} bytes)", cachePath, bytes.size());
    return true;
}

void RayTracingPipeline::DestroyPipelineCache(std::unique_ptr<gpu::PipelineCache>& cache) {
    cache.reset();
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

RayTracingPipeline::RayTracingPipeline(
    gpu::Device& device,
    const std::string& raygenPath,
    const std::string& closestHitPath,
    const std::string& missPath,
    const std::string& anyHitPath,
    const gpu::PipelineCache* pipelineCache)
    : m_device(device)
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

    // The pipeline is created with kMaxPipelineRayRecursionDepth. Every
    // conforming ray tracing device offers at least 1, which is all the
    // raygen-driven integrator needs, so this only trips on a device that
    // reports less than the specification allows -- and then names itself here
    // rather than as a device loss somewhere unrelated.
    const u32 maxRecursionDepth = m_device.Info().maxTraceRecursionDepth;
    if (maxRecursionDepth < kMaxPipelineRayRecursionDepth) {
        throw std::runtime_error(
            "Ray tracing device supports a recursion depth of only " +
            std::to_string(maxRecursionDepth) +
            "; Quantiloom traces every ray from raygen and needs " +
            std::to_string(kMaxPipelineRayRecursionDepth) +
            " (see src/shaders/path_driver.hlsli)");
    }

    // Initialize dynamic texture limit based on device capabilities:
    // 1024 with descriptor indexing, 32 otherwise
    m_maxTextures = m_device.Info().maxBindlessTextures;
    QL_LOG_INFO("Texture limit: {} textures", m_maxTextures);

    // Create pipeline in order with exception safety
    try {
        CreateDescriptorSetLayout();
        CreatePipelineLayout();
        LoadShaders();
        SetSpecConstants(7, false);  // default: RGB, debug off

        // Bindings 28/29 have descriptorCount 1 without PARTIALLY_BOUND. Even
        // an old RGB scene therefore needs real descriptors until camera mode
        // supplies its full-resolution image and encoded response table.
        m_cameraFallbackImage =
            CreateTextureOrThrow(m_device, FallbackStorageImage(gpu::Format::RGBA32Float));
        rendercore::LegacyTransitionImmediate(m_device, m_cameraFallbackImage->Ref(),
                                              rendercore::Layout::Undefined,
                                              rendercore::Layout::General);
        m_cameraFallbackDepth =
            CreateTextureOrThrow(m_device, FallbackStorageImage(gpu::Format::R32Float));
        rendercore::LegacyTransitionImmediate(m_device, m_cameraFallbackDepth->Ref(),
                                              rendercore::Layout::Undefined,
                                              rendercore::Layout::General);
        m_cameraFallbackResponse = CreateBufferOrThrow(m_device, HostStorage(64u));
        const SpectralCurveGPU emptyCurve{};
        m_spectralFallback = CreateBufferOrThrow(m_device, HostStorage(sizeof(emptyCurve)));
        m_spectralFallback->Upload(&emptyCurve, sizeof(emptyCurve));
        const ComplexRefractiveIndexGPU emptyCri{};
        m_criFallback = CreateBufferOrThrow(m_device, HostStorage(sizeof(emptyCri)));
        m_criFallback->Upload(&emptyCri, sizeof(emptyCri));
        const u32 disabledHeader[16] = {};
        m_cameraFallbackResponse->Upload(disabledHeader, sizeof(disabledHeader));
        for (u32 i = 0; i < kProjectionSetCount; ++i) {
            m_projectionBuffers[i] =
                CreateBufferOrThrow(m_device, HostStorage(64u, gpu::BufferUsage::TransferDst));
            SetCameraProjection(nullptr, i);
        }
        SetFusionTransport({rendercore::FusionTransportGpu{}});
        BeginFusionRecording(0, 0, 0, 0);
        SetFusionProbes({});
        m_cameraFallbackAtmosHeader = CreateBufferOrThrow(m_device, HostStorage(80u));
        const u32 disabledAtmosHeader[20] = {};
        m_cameraFallbackAtmosHeader->Upload(disabledAtmosHeader, sizeof(disabledAtmosHeader));
        m_cameraFallbackAtmosData = CreateBufferOrThrow(m_device, HostStorage(2048u * sizeof(f32)));
        const std::vector<f32> disabledAtmosData(2048u, 0.0f);
        m_cameraFallbackAtmosData->Upload(disabledAtmosData.data(),
                                          disabledAtmosData.size() * sizeof(f32));
        for (gpu::BindGroup set :
             {m_descriptorSet, m_cameraDescriptorSet, m_observerDescriptorSet}) {
            WriteStorageImage(m_device, set, 28u, m_cameraFallbackImage->Ref());
            WriteStorageBuffer(m_device, set, 29u, m_cameraFallbackResponse->Ref());
            WriteStorageBuffer(m_device, set, 17u, m_cameraFallbackAtmosHeader->Ref());
            WriteStorageBuffer(m_device, set, 20u, m_cameraFallbackAtmosData->Ref());
        }
        for (gpu::BindGroup set : {m_cameraDescriptorSet, m_observerDescriptorSet}) {
            WriteStorageImage(m_device, set, 0u, m_cameraFallbackImage->Ref());
            WriteStorageImage(m_device, set, 22u, m_cameraFallbackDepth->Ref());
        }
        WriteStorageImage(m_device, m_descriptorSet, 22u, m_cameraFallbackDepth->Ref());

        QL_LOG_INFO("Ray Tracing pipeline created successfully");
    }
    catch (const std::exception& e) {
        m_variantCache.clear();  // each kernel: its pipeline, then its shader binding table

        m_kernelLayout.reset();
        m_bindGroupPool.reset();
        m_bindingLayout.reset();

        QL_LOG_ERROR("Failed to create Ray Tracing pipeline: {}", e.what());
        throw;  // Rethrow the exception
    }
}

RayTracingPipeline::~RayTracingPipeline() {
    m_geometryKernel.reset();

    m_activeVariant = nullptr;
    m_variantCache.clear();  // each kernel: its pipeline, then its shader binding table

    m_bindGroupPool.reset();
    m_kernelLayout.reset();
    m_bindingLayout.reset();

    QL_LOG_INFO("Ray Tracing pipeline destroyed");
}

// ============================================================================
// Descriptor Set Layout
// ============================================================================

void RayTracingPipeline::CreateDescriptorSetLayout() {
    // One row per binding in RtBindings.hpp, with the final stage sets; only the
    // bindless count is per device (1024 with descriptor indexing, 32 otherwise).
    auto slots = std::to_array(rendercore::kRtBindingTemplate);
    for (gpu::BindingSlot& slot : slots) {
        if (slot.count == rendercore::QL_BINDLESS) slot.count = m_maxTextures;
    }

    auto layout = m_device.CreateBindingLayout(gpu::BindingLayoutDesc{0, slots});
    if (!layout.has_value()) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    m_bindingLayout = std::move(layout.value());

    // Pool sizes follow from the layout, per descriptor type in order of first
    // appearance, three sets' worth: storage images 12 (bindings 0, 22, 28, 31),
    // acceleration structures 3, storage buffers 81 (27 per set), sampled images
    // and samplers 3 * (m_maxTextures + 2) each.
    auto pool = m_device.CreateBindGroupPool(*m_bindingLayout, 3);
    if (!pool.has_value()) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
    m_bindGroupPool = std::move(pool.value());

    auto sets = m_bindGroupPool->Allocate(3);
    if (!sets.has_value()) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    m_descriptorSet = sets.value()[0];
    m_cameraDescriptorSet = sets.value()[1];
    m_observerDescriptorSet = sets.value()[2];

    QL_LOG_INFO("  Descriptor set layout created");
}

// ============================================================================
// Pipeline Layout
// ============================================================================

void RayTracingPipeline::CreatePipelineLayout() {
    // Push constant range for camera data + sampling parameters
    // IMPORTANT: Include all shader stages that access push constants
    const gpu::BindingLayout* const spaces[] = {m_bindingLayout.get()};
    gpu::KernelLayoutDesc desc;
    desc.spaces = spaces;
    desc.pushConstantBytes = static_cast<u32>(sizeof(TracePushConstants));
    desc.pushConstantVisibility = gpu::ShaderStages::RayGen | gpu::ShaderStages::ClosestHit |
                                  gpu::ShaderStages::Miss | gpu::ShaderStages::AnyHit |
                                  gpu::ShaderStages::Compute;

    auto layout = m_device.CreateKernelLayout(desc);
    if (!layout.has_value()) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    m_kernelLayout = std::move(layout.value());

    QL_LOG_INFO("  Pipeline layout created with push constants (camera data)");
}

// ============================================================================
// Shader Loading
// ============================================================================

void RayTracingPipeline::LoadShaders() {
    QL_LOG_INFO("  [LoadShaders] Loading SPIR-V shaders...");

    m_spirvData.resize(5);
    m_spirvData[0] = ToShaderCode(m_raygenPath, LoadSPIRV(m_raygenPath));
    QL_LOG_INFO("  [LoadShaders] Raygen loaded: {} words", m_spirvData[0].bytes.size() / sizeof(u32));

    m_spirvData[1] = ToShaderCode(m_closestHitPath, LoadSPIRV(m_closestHitPath));
    QL_LOG_INFO("  [LoadShaders] ClosestHit loaded: {} words", m_spirvData[1].bytes.size() / sizeof(u32));

    m_spirvData[2] = ToShaderCode(m_missPath, LoadSPIRV(m_missPath));
    QL_LOG_INFO("  [LoadShaders] Miss loaded: {} words", m_spirvData[2].bytes.size() / sizeof(u32));

    m_spirvData[3] = ToShaderCode(m_shadowMissPath, LoadSPIRV(m_shadowMissPath));
    QL_LOG_INFO("  [LoadShaders] ShadowMiss loaded: {} words", m_spirvData[3].bytes.size() / sizeof(u32));

    m_spirvData[4] = ToShaderCode(m_anyHitPath, LoadSPIRV(m_anyHitPath));
    QL_LOG_INFO("  [LoadShaders] AnyHit loaded: {} words", m_spirvData[4].bytes.size() / sizeof(u32));

    QL_LOG_INFO("  Shaders loaded: {} / {} / {} / {}", m_raygenPath, m_closestHitPath, m_missPath, m_shadowMissPath);
    Log::Flush();
}

std::vector<u32> RayTracingPipeline::LoadSPIRV(const std::string& path) {
    // One search list for every shader the SDK loads (core/ResourceLocator.hpp):
    // an explicitly named file is honoured as given, anything else is resolved
    // by name against the accepted resource root.
    const std::filesystem::path given(path);
    if (given.is_absolute() || given.has_parent_path()) {
        std::vector<u32> direct = rendercore::ReadSpirvFile(given);
        if (!direct.empty()) {
            QL_LOG_DEBUG("  Loaded shader from: {}", given.string());
            return direct;
        }
    }

    std::vector<u32> buffer = rendercore::LoadSpirv(given.filename().string());
    if (buffer.empty()) {
        const String why = core::ResourceNotFoundMessage();
        throw std::runtime_error("Failed to open shader file: " + path +
                                 (why.empty() ? " (not in the resolved resource root)" : " (" + why + ")"));
    }
    QL_LOG_DEBUG("  Loaded shader from: {}", path);
    return buffer;
}

// ============================================================================
// Pipeline Creation
// ============================================================================

std::unique_ptr<gpu::TraceKernel> RayTracingPipeline::CreatePipelineVariant(const SpecConstants& spec) {
    // The raygen now selects CameraMeasurement before the CIE colour path, so
    // all three tracing stages must see the same spectral-mode specialization:
    // {id 0, offset 0} spectralMode and {id 1, offset 4} debugEnabled, 8 bytes,
    // on raygen, closest hit and miss. The shadow miss and the any-hit take
    // none: an any-hit decides whether a surface is present, which is a
    // coverage question with no wavelength in it.
    const gpu::SpecializationConstant constants[] = {{0, spec.spectralMode}, {1, spec.debugEnabled}};

    // Five stages in four groups: the any-hit joins the one triangles hit group
    // rather than forming its own, so the shader binding table has one hit
    // record and two miss records, and every TraceRay's sbtRecordOffset and
    // missIndex keep the meaning they had.
    gpu::TraceKernelDesc desc;
    desc.rayGen = &m_spirvData[0];
    desc.closestHit = &m_spirvData[1];
    desc.miss = &m_spirvData[2];
    desc.shadowMiss = &m_spirvData[3];
    desc.anyHit = &m_spirvData[4];
    desc.layout = m_kernelLayout.get();
    desc.specialization = constants;
    desc.specializedStages = gpu::ShaderStages::RayGen | gpu::ShaderStages::ClosestHit | gpu::ShaderStages::Miss;
    desc.maxRecursionDepth = kMaxPipelineRayRecursionDepth;
    desc.cache = m_pipelineCache;
    const String name = "mode " + std::to_string(spec.spectralMode);  // the stack-size log's label
    desc.name = name;

    QL_LOG_INFO("  [CreatePipelineVariant] spectralMode={}, debugEnabled={}", spec.spectralMode, spec.debugEnabled);
    Log::Flush();

    auto kernel = m_device.CreateTraceKernel(desc);
    if (!kernel.has_value()) {
        throw std::runtime_error(kernel.error());
    }
    return std::move(kernel.value());
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

    m_activeVariant = it->second.get();
    m_activeSpectralMode = spectralMode;
}

// ============================================================================
// Descriptor Binding
// ============================================================================

void RayTracingPipeline::UpdateSharedDescriptorSets(std::span<const gpu::BindingWrite> writes) const {
    for (gpu::BindGroup target : {m_descriptorSet, m_cameraDescriptorSet, m_observerDescriptorSet}) {
        m_device.WriteBindGroup(target, writes);
    }
}

void RayTracingPipeline::BindOutputImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_descriptorSet, 0u, image);
}

void RayTracingPipeline::BindDepthImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_descriptorSet, 22u, image);
}

void RayTracingPipeline::BindCameraMeasurementImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_cameraDescriptorSet, 28u, image);
}

void RayTracingPipeline::BindCameraResponseBuffer(gpu::BufferRef buffer) const {
    WriteStorageBuffer(m_device, m_cameraDescriptorSet, 29u, buffer);
}
void RayTracingPipeline::BindCameraLightingBuffer(gpu::BufferRef buffer) const {
    WriteStorageBuffer(m_device, m_cameraDescriptorSet, 2u, buffer);
    WriteStorageBuffer(m_device, m_observerDescriptorSet, 2u, buffer);
}

void RayTracingPipeline::SetCameraProjection(const camera::CameraProjection* p, u32 set,
                                             bool scalarVignetting) {
    if (set >= kProjectionSetCount)
        throw std::invalid_argument("invalid projection descriptor set");
    std::array<f32, 16> values{};
    if (p) {
        values[0] = 1;
        values[1] = static_cast<f32>(p->model);
        values[2] = static_cast<f32>(p->maxThetaRadians);
        values[4] = static_cast<f32>(p->fx);
        values[5] = static_cast<f32>(p->fy);
        values[6] = static_cast<f32>(p->cx);
        values[7] = static_cast<f32>(p->cy);
        for (size_t i = 0; i < 5; ++i)
            values[8 + i] = static_cast<f32>(p->coefficients[i]);
        values[13] = scalarVignetting ? 1.0f : 0.0f;
    }
    // A submitted frame may still be reading this buffer; with an upload
    // queue attached the write is recorded into the next submission instead.
    if (m_uploadQueue)
        m_uploadQueue->Enqueue(m_projectionBuffers[set]->Ref(), 0, values.data(), sizeof(values));
    else
        m_projectionBuffers[set]->Upload(values.data(), sizeof(values));
    const gpu::BindGroup sets[] = {m_descriptorSet, m_cameraDescriptorSet,
                                   m_observerDescriptorSet};
    WriteStorageBuffer(m_device, sets[set], 32u, m_projectionBuffers[set]->Ref());
}

void RayTracingPipeline::SetFusionTransport(const Vector<rendercore::FusionTransportGpu>& records) {
    const auto size = std::max<size_t>(1, records.size()) * sizeof(rendercore::FusionTransportGpu);
    auto replacement = CreateBufferOrThrow(m_device, HostStorage(size));
    const rendercore::FusionTransportGpu disabled{};
    replacement->Upload(records.empty() ? &disabled : records.data(), size);
    for (auto set : {m_descriptorSet, m_cameraDescriptorSet, m_observerDescriptorSet})
        WriteStorageBuffer(m_device, set, 35u, replacement->Ref());
    m_fusionTransportBuffer = std::move(replacement);
}

void RayTracingPipeline::BeginFusionRecording(u32 width, u32 height, u32 spp, u32 maxRays,
                                              bool aggregate, bool quantitative) {
    const u64 total = static_cast<u64>(width) * height * spp;
    if (total > std::numeric_limits<u32>::max())
        throw std::runtime_error("fusion ray indexing exceeds UINT range");
    const u32 stride = total && maxRays ? static_cast<u32>((total + maxRays - 1) / maxRays) : 1;
    const u32 count = total && maxRays != std::numeric_limits<u32>::max()
                          ? static_cast<u32>((total + stride - 1) / stride)
                          : 0;
    constexpr u32 slots = rendercore::kFusionMaxSlotsPerRay;
    const size_t pathSize =
        rendercore::kFusionHeaderBytes +
        static_cast<size_t>(count) *
            (rendercore::kFusionRayRecordBytes + slots * rendercore::kFusionVertexRecordBytes);
    const size_t coordinateOffset =
        pathSize +
        (aggregate ? static_cast<size_t>(width) * height * rendercore::kFusionPixelRecordBytes : 0);
    const size_t termsOffset =
        coordinateOffset + (aggregate ? static_cast<size_t>(count) * rendercore::kFusionCoordColumnBytes : 0);
    const size_t size =
        termsOffset + (aggregate ? static_cast<size_t>(count) * slots * rendercore::kFusionTermColumnBytes
                                 : 0);
    if (size > m_device.Info().maxStorageBufferRange)
        throw std::runtime_error(
            "fusion record buffer exceeds device storage range; lower fusionMaxRecordedRays");
    if (m_initialFusionMedia.size() > rendercore::kFusionMaxInitialMedia)
        throw std::runtime_error("invalid initial medium count");
    if (!m_fusionRecordBuffer || m_fusionRecordBuffer->Size() < size) {
        // Grow-only: a wavelength sweep that shrinks the footprint (fewer
        // recorded rays, no aggregate image) reuses the allocation instead
        // of freeing and remapping ~100 MB per band.
        auto replacement = CreateBufferOrThrow(
            m_device, gpu::BufferDesc{size, gpu::BufferUsage::Storage | gpu::BufferUsage::TransferDst,
                                      gpu::MemoryClass::Readback, 0});
        for (auto set : {m_descriptorSet, m_cameraDescriptorSet, m_observerDescriptorSet})
            WriteStorageBuffer(m_device, set, 36u, replacement->Ref());
        m_fusionRecordBuffer = std::move(replacement);
    }
    m_fusionRecordSize = size;
    u32 header[rendercore::kFusionHeaderWords] = {count || aggregate ? 1u : 0u, stride, count,
                                                  slots, width, height, spp, 0};
    if (aggregate) {
        header[rendercore::kFusionPixelSectionWord] = static_cast<u32>(pathSize);
        header[rendercore::kFusionTermsSectionWord] = static_cast<u32>(termsOffset);
        header[rendercore::kFusionCoordSectionWord] = static_cast<u32>(coordinateOffset);
    }
    header[rendercore::kFusionQuantitativeWord] = aggregate || quantitative ? 1u : 0u;
    header[rendercore::kFusionInitialMediumCountWord] =
        static_cast<u32>(m_initialFusionMedia.size());
    for (size_t i = 0; i < m_initialFusionMedia.size(); ++i)
        header[rendercore::kFusionInitialMediaWords + i] = m_initialFusionMedia[i];
    std::copy(std::begin(header), std::end(header), m_fusionHeader.begin());
    // The wipe is recorded ahead of the next trace rather than uploaded now:
    // the header lands by vkCmdUpdateBuffer and the body by vkCmdFillBuffer,
    // so arming a band costs two recorded commands, not a host memset.
    m_pendingFusionZero = size > sizeof(m_fusionHeader) ? size - sizeof(m_fusionHeader) : 0;
    m_fusionInitPending = true;
}

void RayTracingPipeline::RecordFusionInit(gpu::CommandList& list) const {
    if (!m_fusionInitPending)
        return;
    m_fusionInitPending = false;
    if (!m_fusionRecordBuffer) {
        m_pendingFusionZero = 0;
        return;
    }
    // Ray tracing and compute shaders read and write the records.
    constexpr gpu::Access kShaderAccess = gpu::Access::RayTraceRead | gpu::Access::RayTraceWrite |
                                          gpu::Access::ComputeRead | gpu::Access::ComputeWrite;
    const gpu::BufferRef dst = m_fusionRecordBuffer->Ref();
    list.Barrier(gpu::BufferBarrier{dst, kShaderAccess, gpu::Access::TransferWrite});
    list.UpdateBuffer(dst, 0, std::as_bytes(std::span(m_fusionHeader)));
    if (m_pendingFusionZero)
        list.FillBuffer(dst, sizeof(m_fusionHeader), m_pendingFusionZero, 0);
    m_pendingFusionZero = 0;
    list.Barrier(gpu::BufferBarrier{dst, gpu::Access::TransferWrite, kShaderAccess});
}

void RayTracingPipeline::SetFusionInitialMedia(const Vector<u32>& media) {
    if (media.size() > rendercore::kFusionMaxInitialMedia)
        throw std::invalid_argument("too many initial fusion media");
    if (m_initialFusionMedia == media)
        return;
    m_initialFusionMedia = media;
    if (m_fusionRecordBuffer) {
        // Header words 8 and 16..23 only; the rest of the buffer stands.
        // The write lands with the next trace, so no device idle here.
        std::array<u32, rendercore::kFusionMaxInitialMedia> values{};
        std::copy(media.begin(), media.end(), values.begin());
        m_fusionHeader[rendercore::kFusionInitialMediumCountWord] =
            static_cast<u32>(media.size());
        std::copy(values.begin(), values.end(),
                  m_fusionHeader.begin() + rendercore::kFusionInitialMediaWords);
        m_fusionInitPending = true;
    }
}

Vector<u8> RayTracingPipeline::ReadFusionRecording() {
    if (!m_fusionRecordBuffer || m_fusionRecordSize == 0)
        return {};
    m_device.ExecuteImmediate([&](gpu::CommandList& list) {
        list.Barrier(gpu::BufferBarrier{m_fusionRecordBuffer->Ref(),
                                        gpu::Access::RayTraceWrite | gpu::Access::TransferWrite,
                                        gpu::Access::HostRead});
    });
    Vector<u8> bytes(static_cast<size_t>(m_fusionRecordSize));
    const void* data = m_fusionRecordBuffer->MapRead();
    if (!data)
        throw std::runtime_error("cannot read fusion path records");
    std::memcpy(bytes.data(), data, bytes.size());
    m_fusionRecordBuffer->Unmap();
    return bytes;
}

void RayTracingPipeline::SetFusionProbes(const Vector<ProbeRay>& probes) {
    const size_t size = std::max<size_t>(1, probes.size()) * sizeof(ProbeRay);
    auto buffer = CreateBufferOrThrow(m_device, HostStorage(size));
    const ProbeRay disabled{};
    buffer->Upload(probes.empty() ? &disabled : probes.data(), size);
    for (auto set : {m_descriptorSet, m_cameraDescriptorSet, m_observerDescriptorSet})
        WriteStorageBuffer(m_device, set, 37u, buffer->Ref());
    m_fusionProbeBuffer = std::move(buffer);
    // Header word 10 carries the probe count; it lands with the next trace.
    m_fusionHeader[rendercore::kFusionProbeCountWord] = static_cast<u32>(probes.size());
    m_fusionInitPending = true;
}

Vector<RayTracingPipeline::GeometryHit>
RayTracingPipeline::CapturePrimaryGeometry(u32 width, u32 height,
                                           const Vector<glm::vec2>* queryPixels) {
    if (!width || !height)
        throw std::invalid_argument("invalid geometry grid");
    if (!m_geometryKernel) {
        // A compute pipeline on the ray tracing layout: same set, same push range.
        const std::string path =
            (std::filesystem::path(m_raygenPath).parent_path() / "fusion_geometry.spv").string();
        const gpu::ShaderCode code = ToShaderCode(path, LoadSPIRV(path));
        gpu::ComputeKernelDesc desc;
        desc.code = &code;
        desc.layout = m_kernelLayout.get();
        auto kernel = m_device.CreateComputeKernel(desc);
        if (!kernel.has_value())
            throw std::runtime_error("cannot create geometry capture pipeline");
        m_geometryKernel = std::move(kernel.value());
    }
    const size_t count = Image::CheckedElementCount(width, height, 1);
    if (queryPixels && queryPixels->size() != count)
        throw std::invalid_argument("geometry query count mismatch");
    std::unique_ptr<gpu::Buffer> pixels;
    if (queryPixels) {
        pixels = CreateBufferOrThrow(m_device, HostStorage(count * sizeof(glm::vec2)));
        pixels->Upload(queryPixels->data(), count * sizeof(glm::vec2));
        WriteStorageBuffer(m_device, m_descriptorSet, 34u, pixels->Ref());
    }
    const std::unique_ptr<gpu::Buffer> output = CreateBufferOrThrow(
        m_device, gpu::BufferDesc{count * sizeof(GeometryHit), gpu::BufferUsage::Storage,
                                  gpu::MemoryClass::Readback, 0});
    WriteStorageBuffer(m_device, m_descriptorSet, 33u, output->Ref());
    auto pc = m_pushConstants;
    pc.cameraAndSampling.sampleIndex = width;
    pc.cameraAndSampling.totalSamples = height;
    pc.cameraAndSampling.frameIndex = queryPixels ? 1u : 0u;
    m_device.ExecuteImmediate([&](gpu::CommandList& list) {
        list.SetKernel(*m_geometryKernel);
        list.SetBindGroup(*m_kernelLayout, 0, m_descriptorSet);
        list.PushConstants(*m_kernelLayout, pc);
        list.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        list.Barrier(gpu::BufferBarrier{output->Ref(), gpu::Access::ComputeWrite,
                                        gpu::Access::HostRead});
    });
    Vector<GeometryHit> hits(count);
    const auto* data = output->MapRead();
    if (!data)
        throw std::runtime_error("cannot read geometry capture buffer");
    std::memcpy(hits.data(), data, count * sizeof(GeometryHit));
    output->Unmap();
    return hits;
}

void RayTracingPipeline::BindCameraDynamicCounterBuffer(gpu::BufferRef buffer) const {
    WriteStorageBuffer(m_device, m_cameraDescriptorSet, 30u, buffer);
}

void RayTracingPipeline::BindCameraMeasurementDepthImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_cameraDescriptorSet, 31u, image);
}

void RayTracingPipeline::BindCameraDepthImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_cameraDescriptorSet, 22u, image);
}

void RayTracingPipeline::BindCameraObserverOutputImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_observerDescriptorSet, 0u, image);
}

void RayTracingPipeline::BindCameraObserverDepthImage(gpu::TextureRef image) const {
    WriteStorageImage(m_device, m_observerDescriptorSet, 22u, image);
}

void RayTracingPipeline::BindAccelerationStructure(gpu::RtSceneRef tlas) const {
    const gpu::BindingWrite writes[] = {gpu::BindingWrite::Scene(1u, tlas)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindLUTBuffer(gpu::BufferRef buffer) const {
    const gpu::BindingWrite writes[] = {RtBufferWrite(2u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindGeometryBuffers(gpu::BufferRef vertexBuffer, gpu::BufferRef indexBuffer,
                                             gpu::BufferRef uvBuffer) const {
    std::vector<gpu::BindingWrite> writes;
    writes.reserve(3);

    writes.push_back(RtBufferWrite(3u, vertexBuffer));  // Binding 3: Vertex buffer
    writes.push_back(RtBufferWrite(4u, indexBuffer));   // Binding 4: Index buffer

    // Binding 8: UV buffer (optional)
    if (uvBuffer.IsValid()) {
        writes.push_back(RtBufferWrite(8u, uvBuffer));
        QL_LOG_DEBUG("  [DEBUG] Bound UV buffer to binding 8");
    } else {
        QL_LOG_DEBUG("  [DEBUG] No UV buffer provided, skipping binding 8");
    }

    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindMaterialBuffer(gpu::BufferRef buffer) const {
    const gpu::BindingWrite writes[] = {RtBufferWrite(5u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindTangentBuffer(gpu::BufferRef buffer) const {
    const gpu::BindingWrite writes[] = {RtBufferWrite(9u, buffer)};
    UpdateSharedDescriptorSets(writes);
    QL_LOG_DEBUG("  [DEBUG] Bound tangent buffer to binding 9");
}

void RayTracingPipeline::BindNormalBuffer(gpu::BufferRef buffer) const {
    const gpu::BindingWrite writes[] = {RtBufferWrite(16u, buffer)};
    UpdateSharedDescriptorSets(writes);
    QL_LOG_DEBUG("  [DEBUG] Bound normal buffer to binding 16");
}

void RayTracingPipeline::BindTextures(std::span<const gpu::TextureRef> textures,
                                       std::span<const gpu::SamplerRef> samplers) const {
    if (textures.size() != samplers.size()) {
        QL_LOG_ERROR("Texture binding failed: imageViews.size() ({}) != samplers.size() ({})",
                     textures.size(), samplers.size());
        throw std::runtime_error("Mismatched texture and sampler array sizes");
    }

    if (textures.empty()) {
        m_pushConstants.textureCount = 0;
        QL_LOG_WARN("No textures to bind (TextureManager should provide at least a dummy texture)");
        return;
    }

    u32 textureCount = static_cast<u32>(textures.size());

    // Validate texture count against device capability limit
    if (textureCount > m_maxTextures) {
        QL_LOG_ERROR("Scene has {} textures, but device limit is {} - cannot render",
                     textureCount, m_maxTextures);
        throw std::runtime_error("Scene texture count exceeds device capability");
    }

    m_pushConstants.textureCount = textureCount;
    QL_LOG_INFO("Binding {} textures to descriptor set", textureCount);

    // Sampled images in the layout their refs name (shader-read-only for
    // TextureManager's), samplers separate.
    // Binding 6: texture array, binding 7: sampler array, from element 0
    const gpu::BindingWrite writes[] = {gpu::BindingWrite::SampledTextures(6u, textures),
                                        gpu::BindingWrite::Samplers(7u, samplers)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindPrefilteredEnvMap(gpu::TextureRef texture, gpu::SamplerRef sampler) const {
    QL_LOG_INFO("Binding prefiltered environment map to descriptor set (bindings 10, 21)");

    // Its own sampler rather than the one at binding 12: that one clamps maxLod to
    // 0, which pinned every environment lookup to mip 0 and left the prefiltered
    // chain unreachable.
    const gpu::BindingWrite writes[] = {
        gpu::BindingWrite::SampledTexture(10u, texture),  // prefilteredEnvMap (TextureCube)
        gpu::BindingWrite::Sampler(21u, sampler),         // envSampler
    };
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindBRDFLut(gpu::TextureRef texture, gpu::SamplerRef sampler) const {
    QL_LOG_INFO("Binding BRDF integration LUT to descriptor set (bindings 11, 12)");

    // Binding 11: BRDF LUT texture; binding 12: IBL sampler (shared by prefiltered env and BRDF LUT)
    const gpu::BindingWrite writes[] = {
        gpu::BindingWrite::SampledTexture(11u, texture),
        gpu::BindingWrite::Sampler(12u, sampler),
    };
    UpdateSharedDescriptorSets(writes);
}

// ============================================================================
// Spectral Curves Buffer Binding
// ============================================================================

void RayTracingPipeline::BindSpectralCurvesBuffer(gpu::BufferRef buffer) const {
    if (!buffer.IsValid())
        buffer = m_spectralFallback->Ref();

    QL_LOG_INFO("Binding spectral curves buffer to descriptor set (binding 13)");
    QL_LOG_INFO("  Buffer size: {} bytes", buffer.size);
    QL_LOG_INFO("  Expected curves: ~{}", buffer.size / 272);  // 272 bytes per SpectralCurveGPU

    const gpu::BindingWrite writes[] = {RtBufferWrite(13u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

// ============================================================================
// Complex Refractive Index Buffer Binding
// ============================================================================

void RayTracingPipeline::BindComplexRefractiveIndexBuffer(gpu::BufferRef buffer) const {
    if (!buffer.IsValid())
        buffer = m_criFallback->Ref();

    QL_LOG_INFO("Binding complex refractive index buffer to descriptor set (binding 14)");
    QL_LOG_INFO("  Buffer size: {} bytes", buffer.size);
    QL_LOG_INFO("  Expected entries: ~{}", buffer.size / 528);  // 528 bytes per ComplexRefractiveIndexGPU

    const gpu::BindingWrite writes[] = {RtBufferWrite(14u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

// ============================================================================
// Solar Spectral LUT Buffer Binding
// ============================================================================

void RayTracingPipeline::BindSolarSpectralLUT(gpu::BufferRef buffer) const {
    if (!buffer.IsValid()) {
        QL_LOG_INFO("Solar spectral LUT buffer is null - will use LightingParams RGB fallback");
        // Note: Shader must check if SolarSpectralLUT is valid before use
        return;
    }

    QL_LOG_INFO("Binding solar spectral LUT buffer to descriptor set (binding 15)");
    QL_LOG_INFO("  Buffer size: {} bytes (expected: 544)", buffer.size);

    const gpu::BindingWrite writes[] = {RtBufferWrite(15u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

// ============================================================================
// Bind NN Atmosphere Buffers (Bindings 17 + 20)
// ============================================================================

void RayTracingPipeline::BindAtmosphereNN(gpu::BufferRef header,
                                          gpu::BufferRef data) const {
    // The viewport LUT may change while camera and observer traces already
    // recorded in a command buffer still use their own fixed descriptors.
    WriteStorageBuffer(m_device, m_descriptorSet, 17u,
                       header.IsValid() ? header : m_cameraFallbackAtmosHeader->Ref());
    WriteStorageBuffer(m_device, m_descriptorSet, 20u,
                       data.IsValid() ? data : m_cameraFallbackAtmosData->Ref());
}

void RayTracingPipeline::BindCameraAtmosphereNN(gpu::BufferRef header,
                                                gpu::BufferRef data) const {
    WriteStorageBuffer(m_device, m_cameraDescriptorSet, 17u,
                       header.IsValid() ? header : m_cameraFallbackAtmosHeader->Ref());
    WriteStorageBuffer(m_device, m_cameraDescriptorSet, 20u,
                       data.IsValid() ? data : m_cameraFallbackAtmosData->Ref());
}

void RayTracingPipeline::BindCameraObserverAtmosphereNN(gpu::BufferRef header,
                                                        gpu::BufferRef data) const {
    WriteStorageBuffer(m_device, m_observerDescriptorSet, 17u,
                       header.IsValid() ? header : m_cameraFallbackAtmosHeader->Ref());
    WriteStorageBuffer(m_device, m_observerDescriptorSet, 20u,
                       data.IsValid() ? data : m_cameraFallbackAtmosData->Ref());
}

void RayTracingPipeline::BindInstanceGeometryBuffer(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding instance geometry buffer to descriptor set (binding 18)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(18u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindEmissiveTriangleBuffer(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding emissive triangle buffer to descriptor set (binding 23)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(23u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindThermalSunResponseBuffer(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding thermal sun response buffer to descriptor set (binding 26)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(26u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindThermalTangentBuffer(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding thermal parameter tangent buffer to descriptor set (binding 27)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(27u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindThermalTemperatureBuffer(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding thermal temperature buffer to descriptor set (binding 24)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(24u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindCIE_CMF_LUT(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding CIE CMF LUT buffer to descriptor set (binding 19)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(19u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::BindRgbToSpectrumTable(gpu::BufferRef buffer) const {
    QL_LOG_DEBUG("Binding RGB->spectrum coefficient table to descriptor set (binding 25)");

    const gpu::BindingWrite writes[] = {RtBufferWrite(25u, buffer)};
    UpdateSharedDescriptorSets(writes);
}

void RayTracingPipeline::UpdateDescriptorSets() {
    // Bindings are updated immediately in Bind* methods
    // This is a no-op for M1, but kept for API consistency
}

// ============================================================================
// Rendering
// ============================================================================

void RayTracingPipeline::TraceRays(gpu::CommandList& list, const u32 width, const u32 height, const bool finalDispatch) const {
    // The trace goes through the kernel's own entry point and SBT regions, so any
    // list records it, a plain wrapped one included.

    // An armed record init -- header update plus zero fill -- lands in this
    // submission before the trace that reads it.
    RecordFusionInit(list);

    // Bind pipeline and descriptor set
    list.SetTraceKernel(*m_activeVariant);

    const gpu::BindGroup activeSet = m_activeSpectralMode ==
        static_cast<u32>(SpectralMode::CameraMeasurement)
        ? m_cameraDescriptorSet
        : (m_useObserverDescriptorSet ? m_observerDescriptorSet : m_descriptorSet);
    list.SetBindGroup(*m_kernelLayout, 0, activeSet, gpu::PipelineKind::Trace);

    // Push constants (camera + sampling parameters), with the stages of the
    // pipeline layout's push constant range
    list.PushConstants(*m_kernelLayout, m_pushConstants);

    // Trace rays
    list.Trace(*m_activeVariant, width, height, 1);

    if (finalDispatch) {
        list.Barrier(gpu::GlobalBarrier{gpu::Access::RayTraceWrite, gpu::Access::TransferRead});
    } else {
        list.Barrier(gpu::GlobalBarrier{gpu::Access::RayTraceWrite,
                                        gpu::Access::RayTraceRead | gpu::Access::RayTraceWrite});
    }
}

void RayTracingPipeline::SetCameraData(const CameraData& cameraData) {
    m_pushConstants.cameraAndSampling.camera = cameraData;
}

void RayTracingPipeline::SetSamplingParams(const u32 frameIndex, const u32 sampleIndex, const u32 totalSamples,
                                           const u32 randomSeed, const u32 sequenceSeed) {
    m_pushConstants.cameraAndSampling.frameIndex = frameIndex;
    m_pushConstants.cameraAndSampling.sampleIndex = sampleIndex;
    m_pushConstants.cameraAndSampling.totalSamples = totalSamples;
    m_pushConstants.cameraAndSampling.randomSeed = randomSeed;
    m_pushConstants.cameraAndSampling.sequenceSeed = sequenceSeed;
}

void RayTracingPipeline::SetTimeStratum(const u32 stratum, const u32 count) {
    m_pushConstants.cameraAndSampling.timeStratum = stratum;
    m_pushConstants.cameraAndSampling.timeStratumCount = count;
}

} // namespace quantiloom
