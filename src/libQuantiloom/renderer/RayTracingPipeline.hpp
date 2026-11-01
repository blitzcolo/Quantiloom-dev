/**
 * @file RayTracingPipeline.hpp
 * @brief Ray tracing pipeline management and shader binding
 *
 * Provides RayTracingPipeline class for managing:
 * - Ray tracing shader loading and compilation (SPIR-V)
 * - gpu::TraceKernel variants, one per specialization (each owns its shader binding table)
 * - Descriptor set management for resource binding
 * - TraceRays() execution interface
 *
 * Shader Groups:
 * - Ray Generation: Primary ray generation from camera
 * - Closest Hit: Surface shading and material evaluation
 * - Miss: Sky/environment sampling
 *
 * Resource Bindings (Fixed Descriptor Layout):
 * - Binding 0: Output image (storage image)
 * - Binding 1: TLAS (acceleration structure)
 * - Binding 2: Lighting parameters (uniform buffer)
 * - Binding 3-4: Geometry buffers (vertex, index)
 * - Binding 5: Material buffer (storage buffer)
 * - Binding 6-7: Texture arrays (bindless, sampled images + samplers)
 * - Binding 8: UV buffer (optional)
 * - Binding 9: Tangent buffer (optional)
 * - Binding 10-12: IBL resources (prefiltered envmap, BRDF LUT, sampler)
 * - Binding 13: Spectral reflectance curves (storage buffer)
 * - Binding 14: Complex refractive index (n,k) data (storage buffer)
 * - Binding 15: Solar spectral LUT (storage buffer)
 * - Binding 16: Normal buffer (required)
 * - Binding 17: NN atmosphere LUT header (storage buffer)
 * - Binding 18: Instance geometry info (storage buffer)
 * - Binding 19: CIE CMF LUT (storage buffer)
 * - Binding 20: NN atmosphere LUT data blob (storage buffer)
 * - Binding 23: Emissive triangles for next-event estimation (storage buffer)
 *
 * Usage example:
 * @code
 * RayTracingPipeline pipeline(device, "raygen.spv", "closesthit.spv", "miss.spv", "anyhit.spv");
 *
 * // Bind resources
 * pipeline.BindOutputImage(outputImage->Ref());
 * pipeline.BindAccelerationStructure(tlas.Ref());
 * pipeline.BindGeometryBuffers(vertices->Ref(), indices->Ref(), uvs->Ref());
 * pipeline.BindMaterialBuffer(materials->Ref());
 * pipeline.BindTextures(textureManager.TextureRefs(), textureManager.SamplerRefs());
 *
 * // Set camera and render
 * pipeline.SetCameraData(cameraData);
 * pipeline.SetSamplingParams(frameIndex, sampleIndex, spp, randomSeed);
 *
 * device.ExecuteImmediate([&](gpu::CommandList& list) {
 *     pipeline.TraceRays(list, width, height);
 * });
 * @endcode
 *
 * @note Pipeline must outlive all bound resources
 * @note All shaders must be compiled to SPIR-V before loading
 * @note Descriptor binding indices MUST match shader layout qualifiers
 *
 * @author blitzcolo
 */

#pragma once
#include "scene/CameraProjection.hpp"
#include "renderer/FusionRecordLayout.hpp"
#include "renderer/FusionTransport.hpp"
// GeometryHit and FusionProbeRay, defined once for the fusion kernels and the
// host. By a path relative to this file, as RtBindings.hpp includes its manifest.
#include "../../shaders/ql_shared_fusion.hlsli"
#include <array>

#include "core/Types.hpp"
#include "scene/Camera.hpp"
#include "gpu/Device.hpp"
#include <vector>
#include <cstddef>
#include <span>
#include <string>
#include <memory>
#include <unordered_map>

// ============================================================================
// RayTracingPipeline - Manages Vulkan Ray Tracing pipeline and SBT
// ============================================================================

namespace quantiloom {

class DeferredBufferUploads;

/**
 * @class RayTracingPipeline
 * @brief Manages the ray tracing pipeline variants and resource bindings
 *
 * Central class for ray tracing rendering in Quantiloom. Handles pipeline
 * creation through gpu::Device, resource binding, and ray dispatch.
 *
 * Key responsibilities:
 * - Shader module loading from SPIR-V files
 * - Pipeline creation with ray generation, closest hit, and miss shaders
 * - Shader binding tables, owned by each gpu::TraceKernel
 * - Descriptor set layout creation and management
 * - Resource binding interface (buffer, texture, sampler and scene refs)
 * - Push constants for camera and sampling parameters
 * - Trace dispatch on the caller's gpu::CommandList
 *
 * Resource Binding Architecture:
 * All resources are bound through descriptor sets using fixed binding indices.
 * The binding layout is defined in shaders and MUST match exactly:
 * @code
 * // Shader layout (common.hlsli)
 * [[vk::binding(0, 0)]] RWTexture2D<float4> outputImage;
 * [[vk::binding(1, 0)]] RaytracingAccelerationStructure scene;
 * [[vk::binding(2, 0)]] StructuredBuffer<LightingParams> lightingParams;
 * // ... etc
 * @endcode
 *
 * Shader Binding Table Layout:
 * @code
 * | Raygen | Closest Hit | Miss |
 * | ------ | ----------- | ---- |
 * | 1 shader | 1 shader  | 1 shader |
 * @endcode
 *
 * @note Non-copyable, non-movable (owns GPU resources)
 * @note Must be created AFTER the gpu::Device it is given
 * @note Must be destroyed BEFORE that device
 *
 * @see gpu::Device for the objects it creates
 * @see gpu::Buffer and gpu::Texture for the resources it binds
 */
class RayTracingPipeline {
public:
    // ========================================================================
    // Specialization constants
    // ========================================================================

    struct SpecConstants {
        u32 spectralMode = 7;  // SPECTRAL_MODE_RGB
        u32 debugEnabled = 0;
    };

    // ========================================================================
    // Pipeline Cache (for faster startup)
    // ========================================================================

    /**
     * @brief Load pipeline cache from disk
     * @param device Device to create the cache on
     * @param cachePath Path to cache file (e.g., "pipeline_cache.bin")
     * @return The cache, empty when the file is missing or rejected; null only if
     *         not even an empty cache could be created
     *
     * Call once at application startup before creating any pipelines.
     * The cache accelerates shader compilation on subsequent runs.
     */
    static std::unique_ptr<gpu::PipelineCache> LoadPipelineCache(gpu::Device& device,
                                                                 const std::string& cachePath);

    /**
     * @brief Save pipeline cache to disk
     * @param cache Pipeline cache to save (null: nothing to save)
     * @param cachePath Path to save cache file
     * @return true if saved successfully
     *
     * Call at application shutdown after all pipelines are destroyed.
     */
    static bool SavePipelineCache(const gpu::PipelineCache* cache, const std::string& cachePath);

    /**
     * @brief Destroy pipeline cache
     * @param cache Pipeline cache to destroy; null afterwards
     */
    static void DestroyPipelineCache(std::unique_ptr<gpu::PipelineCache>& cache);

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Create pipeline with minimal shader set (Raygen + ClosestHit + Miss)
    // Optional pipelineCache accelerates creation (use LoadPipelineCache() to obtain).
    // `device` must outlive the pipeline.
    RayTracingPipeline(
        gpu::Device& device,
        const std::string& raygenPath,
        const std::string& closestHitPath,
        const std::string& missPath,
        const std::string& anyHitPath,
        const gpu::PipelineCache* pipelineCache = nullptr
    );

    ~RayTracingPipeline();

    // Non-copyable, non-movable
    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;
    RayTracingPipeline(RayTracingPipeline&&) = delete;
    RayTracingPipeline& operator=(RayTracingPipeline&&) = delete;

    // ========================================================================
    // Descriptor binding (resources)
    // ========================================================================
    // Storage images are written GENERAL. A sampled texture is written in the
    // layout its ref's policy names (shader-read-only for an Optimal texture).

    // Bind output image (binding 0)
    void BindOutputImage(gpu::TextureRef image) const;

    // Bind primary-hit depth AOV image (binding 22)
    void BindDepthImage(gpu::TextureRef image) const;

    // Bind TLAS (binding 1)
    void BindAccelerationStructure(gpu::RtSceneRef tlas) const;

    // Bind LUT buffer (binding 2)
    void BindLUTBuffer(gpu::BufferRef buffer) const;

    // Bind geometry buffers (binding 3: vertex, binding 4: index, binding 8: UVs [optional,
    // an empty ref skips it])
    void BindGeometryBuffers(gpu::BufferRef vertexBuffer, gpu::BufferRef indexBuffer,
                             gpu::BufferRef uvBuffer = {}) const;

    // Bind material buffer (binding 5)
    void BindMaterialBuffer(gpu::BufferRef buffer) const;

    // Bind tangent buffer (binding 9) - Optional
    void BindTangentBuffer(gpu::BufferRef buffer) const;

    // Bind normal buffer (binding 16) - Required for smooth shading
    void BindNormalBuffer(gpu::BufferRef buffer) const;

    // Bind texture arrays (binding 6: textures, binding 7: samplers)
    // Bindless: up to DeviceInfo::maxBindlessTextures entries
    // If textures is empty, nothing is written and the texture count is 0
    void BindTextures(std::span<const gpu::TextureRef> textures,
                      std::span<const gpu::SamplerRef> samplers) const;

    // ========================================================================
    // IBL (Image-Based Lighting) Bindings
    // ========================================================================
    // Bind IBL textures for physically-based specular reflections
    // - Binding 10: Prefiltered environment cubemap (with mipmaps for roughness)
    // - Binding 11: BRDF integration LUT (2D texture)
    // - Binding 12: IBL sampler (shared)
    // ========================================================================

    // Bind prefiltered environment cubemap (binding 10)
    void BindPrefilteredEnvMap(gpu::TextureRef texture, gpu::SamplerRef sampler) const;

    // Bind BRDF integration LUT for IBL (binding 11: texture, binding 12: sampler)
    void BindBRDFLut(gpu::TextureRef texture, gpu::SamplerRef sampler) const;

    // ========================================================================
    // Spectral Curves Buffer (Binding 13)
    // ========================================================================
    // Bind measured spectral reflectance curves for quantitative spectral rendering
    // Each curve is a SpectralCurveGPU struct (272 bytes) with uniform sampling
    // ========================================================================

    // Bind spectral curves buffer (binding 13)
    // Pass an empty ref to disable spectral curve lookup (binds the one-entry fallback)
    void BindSpectralCurvesBuffer(gpu::BufferRef buffer) const;

    // ========================================================================
    // Complex Refractive Index Buffer (Binding 14)
    // ========================================================================
    // Bind measured complex refractive index (n, k) for physical Fresnel
    // Each entry is a ComplexRefractiveIndexGPU struct (528 bytes)
    // Used for accurate specular reflection on metals (gold, silver, etc.)
    // ========================================================================

    // Bind complex refractive index buffer (binding 14)
    // Pass an empty ref to disable physical Fresnel (uses PBR F0 approximation)
    void BindComplexRefractiveIndexBuffer(gpu::BufferRef buffer) const;

    // ========================================================================
    // Solar Spectral LUT Buffer (Binding 15)
    // ========================================================================
    // Bind ASTM G-173 solar irradiance curves for spectral rendering
    // Structure: SolarSpectralLUT (544 bytes = 2 × SpectralCurveGPU)
    // - sunIrradiance: Direct+circumsolar irradiance (W·m⁻²·nm⁻¹)
    // - skyIrradiance: Diffuse sky irradiance (W·m⁻²·nm⁻¹)
    // ========================================================================

    // Bind solar spectral LUT buffer (binding 15)
    // Pass an empty ref to use LightingParams RGB fallback
    void BindSolarSpectralLUT(gpu::BufferRef buffer) const;

    // ========================================================================
    // NN Atmosphere (Bindings 17 + 20)
    // ========================================================================
    // Bind the baked NN atmosphere LUT: header (AtmosNNHeaderGPU, binding 17)
    // and the flat float data blob (binding 20). When the atmosphere is
    // disabled, bind a header with enabled = 0 and a small dummy data buffer;
    // shaders must not sample the LUT when header.enabled == 0. An empty ref
    // binds the pipeline's own disabled fallback.
    // ========================================================================

    void BindAtmosphereNN(gpu::BufferRef header, gpu::BufferRef data) const;

    // ========================================================================
    // Instance Geometry Info Buffer (Binding 18)
    // ========================================================================
    // Bind per-instance geometry offset information for multi-BLAS support
    // Each TLAS instance has its own offset into the merged global geometry buffers
    // Structure: InstanceGeometryInfo (32 bytes) with vertex/index/normal/UV/tangent offsets
    // ========================================================================

    // Bind instance geometry info buffer (binding 18)
    void BindInstanceGeometryBuffer(gpu::BufferRef buffer) const;

    // ========================================================================
    // CIE 1931 Color Matching Functions LUT (Binding 19)
    // ========================================================================
    // High-precision CIE XYZ CMFs for VIS_FUSED mode spectral integration
    // 401 samples covering 380-780nm at 1nm resolution
    // Compiled in from core/CIE_CMF_Data.hpp, not read from assets/luts/
    // Each sample is float3(x_bar, y_bar, z_bar)
    // ========================================================================

    // Bind CIE CMF LUT buffer (binding 19)
    void BindCIE_CMF_LUT(gpu::BufferRef buffer) const;

    // ========================================================================
    // Jakob-Hanika RGB -> spectrum coefficient table (Binding 25)
    // ========================================================================
    // StructuredBuffer<float4>, xyz = the three polynomial coefficients. Read
    // by closest hit and miss; see core/RgbToSpectrum.hpp.
    void BindRgbToSpectrumTable(gpu::BufferRef buffer) const;

    // ========================================================================
    // Emissive triangles (Binding 23)
    // ========================================================================
    // World-space emissive triangles with a cumulative-power CDF, so the hit
    // shader can sample a point on a light directly instead of waiting for a
    // BSDF ray to find one. Always bound -- a scene with no emitters gets a
    // single zero entry, and LightingParams::emissiveTriangleCount is what
    // tells the shader not to read it.
    // ========================================================================

    // Bind emissive triangle buffer (binding 23)
    void BindEmissiveTriangleBuffer(gpu::BufferRef buffer) const;
    /// Per-element surface temperatures from the thermal solver (binding 24).
    /// Always bound; a scene with no solve gets a single zero entry, and the
    /// shader reads it only where InstanceGeometryInfo says there is one.
    void BindThermalTemperatureBuffer(gpu::BufferRef buffer) const;
    /// Per-element sun response from the thermal solver (binding 26): a header
    /// record carrying the solve's sun direction, then (dT/dv, v_element) per
    /// element. What lets the shader resolve a shadow finer than the triangle
    /// the solver ran on. Always bound; one zeroed record when there is none.
    void BindThermalSunResponseBuffer(gpu::BufferRef buffer) const;
    void BindThermalTangentBuffer(gpu::BufferRef buffer) const;
    // CameraMeasurement resources. A one-pixel zero image and zero response
    // header are bound by the pipeline itself until a host supplies these.
    void BindCameraMeasurementImage(gpu::TextureRef image) const;  // binding 28
    void BindCameraResponseBuffer(gpu::BufferRef buffer) const;    // binding 29
    void BindCameraLightingBuffer(gpu::BufferRef buffer) const;
    /// 4 x u32 atomic counters (transparent/specular primary hits) accumulated
    /// by raygen in camera-measurement mode; the host zeroes it per
    /// acquisition. Binding 30, camera set only -- PARTIALLY_BOUND so the
    /// visibility and observer sets need not write it.
    void BindCameraDynamicCounterBuffer(gpu::BufferRef buffer) const;  // binding 30
    /// Per-stratum primary-hit depth, same 2D-array layering as the
    /// measurement image. Binding 31, camera set only, PARTIALLY_BOUND.
    void BindCameraMeasurementDepthImage(gpu::TextureRef image) const;   // binding 31
    void BindCameraDepthImage(gpu::TextureRef image) const;         // camera binding 22
    void BindCameraObserverOutputImage(gpu::TextureRef image) const; // observer binding 0
    void BindCameraObserverDepthImage(gpu::TextureRef image) const; // observer binding 22
    void BindCameraAtmosphereNN(gpu::BufferRef header,
                                gpu::BufferRef data) const;        // bindings 17,20 camera set
    void BindCameraObserverAtmosphereNN(gpu::BufferRef header,
                                        gpu::BufferRef data) const;   // observer 17,20
    void SetUseCameraObserverSet(bool enabled) { m_useObserverDescriptorSet = enabled; }

    /// Temporal stratum routing for the next TraceRays. Stratum count
    /// 0/1 writes layer 0, exactly like an unstratified exposure.
    void SetTimeStratum(u32 stratum, u32 count);

    // Update all bindings (call after all Bind* calls)
    static void UpdateDescriptorSets();

    // ========================================================================
    // Rendering
    // ========================================================================

    // Set camera parameters (call before TraceRays)
    void SetCameraData(const struct CameraData& cameraData);
    /// Projection descriptor sets -- one buffer + binding-32 write each, so a
    /// measurement trace and an observer capture can carry different optics at
    /// once.
    static constexpr u32 kProjectionSetOrdinary = 0;     // ordinary/CPU spectral capture
    static constexpr u32 kProjectionSetMeasurement = 1;  // GPU camera measurement
    static constexpr u32 kProjectionSetObserver = 2;     // observer/quantitative capture
    static constexpr u32 kProjectionSetCount = 3;
    /// Projection lives outside the frozen CameraData/push-constant ABI.
    /// When an upload queue is attached the write is recorded into the next
    /// submitted frame instead of landing through the mapping behind a host
    /// wait -- a frame may still be reading the previous projection.
    void SetCameraProjection(const camera::CameraProjection* projection,
                             u32 set = kProjectionSetOrdinary,
                             bool scalarVignetting = false);
    /// Attach the queue deferred buffer writes go through. Owned by the
    /// caller (ExternalRenderContext::Impl); stays null for hosts that upload
    /// synchronously.
    void SetUploadQueue(DeferredBufferUploads* queue) { m_uploadQueue = queue; }
    void SetFusionTransport(const Vector<rendercore::FusionTransportGpu>& records);
    /// Arm path recording for the next trace. The record buffer is grow-only:
    /// a smaller request reuses the allocation. Zeroing and the header write
    /// are recorded ahead of the next TraceRays (vkCmdFillBuffer +
    /// vkCmdUpdateBuffer), not uploaded here -- a per-wavelength sweep does
    /// not pay a host memset + upload per band. (0,0,0,0) disarms; it
    /// allocates nothing unless no record buffer exists yet, in which case a
    /// minimal one is created so the descriptor stays valid.
    void BeginFusionRecording(u32 width, u32 height, u32 spp, u32 maxRays,
                              bool aggregate = false, bool quantitative = false);
    Vector<u8> ReadFusionRecording();
    void SetFusionInitialMedia(const Vector<u32>& media);
    /// raygen.rgen's FusionProbeRay.
    using ProbeRay = shader::FusionProbeRay;
    void SetFusionProbes(const Vector<ProbeRay>& probes);
    /// fusion_geometry.rayq.hlsl's GeometryHit; a default one is a miss.
    using GeometryHit = shader::GeometryHit;
    Vector<GeometryHit> CapturePrimaryGeometry(u32 width, u32 height,
                                               const Vector<glm::vec2>* queryPixels = nullptr);
    [[nodiscard]] const CameraData& GetCameraData() const { return m_pushConstants.cameraAndSampling.camera; }

    // Set accumulation sampling parameters (call before TraceRays).
    //
    // randomSeed is expected to change every sample; sequenceSeed is expected
    // to stay fixed for a whole accumulation round and to be re-rolled only
    // when the round restarts. See PushConstantsRayGen for why.
    void SetSamplingParams(u32 frameIndex, u32 sampleIndex, u32 totalSamples, u32 randomSeed,
                           u32 sequenceSeed);

    // Record the trace into the caller's command list.
    // finalDispatch=true inserts a RT->TRANSFER barrier (for readback).
    // finalDispatch=false inserts a RT->RT barrier (for next sample accumulation).
    void TraceRays(gpu::CommandList& list, u32 width, u32 height, bool finalDispatch = true) const;

    // Select pipeline variant for the given specialization constants.
    // Lazily creates and caches the pipeline + SBT on first use per combination.
    void SetSpecConstants(u32 spectralMode, bool debugEnabled);

private:
    // ========================================================================
    // Initialization steps
    // ========================================================================

    void CreateDescriptorSetLayout();
    /// The same writes to the primary, camera and observer sets, in that order.
    void UpdateSharedDescriptorSets(std::span<const gpu::BindingWrite> writes) const;
    void CreatePipelineLayout();
    void LoadShaders();

    // ========================================================================
    // Helpers
    // ========================================================================

    // Load SPIR-V shader from file
    static std::vector<u32> LoadSPIRV(const std::string& path);

    /// Record the armed record-buffer init -- header update plus zero fill --
    /// into `list` ahead of a trace. Invoked from TraceRays so the barriers
    /// order the writes against whatever read the buffer last.
    void RecordFusionInit(gpu::CommandList& list) const;

    // ========================================================================
    // GPU objects
    // ========================================================================

    // The caller's device; every object below is created through it.
    gpu::Device& m_device;

    // Shader paths
    std::string m_raygenPath;
    std::string m_closestHitPath;
    std::string m_missPath;
    std::string m_shadowMissPath;  // Auto-derived from missPath or explicitly set
    std::string m_anyHitPath;      // Named explicitly, unlike the shadow miss above

    // Pipeline objects
    std::unique_ptr<gpu::BindingLayout> m_bindingLayout;  // the set of RtBindings.hpp, space 0
    std::unique_ptr<gpu::KernelLayout> m_kernelLayout;    // that set plus TracePushConstants
    const gpu::PipelineCache* m_pipelineCache = nullptr;  // External, not owned

    // Descriptor pool and sets
    std::unique_ptr<gpu::BindGroupPool> m_bindGroupPool;
    gpu::BindGroup m_descriptorSet{};
    gpu::BindGroup m_cameraDescriptorSet{};
    gpu::BindGroup m_observerDescriptorSet{};
    std::unique_ptr<gpu::Texture> m_cameraFallbackImage;
    std::unique_ptr<gpu::Texture> m_cameraFallbackDepth;
    std::unique_ptr<gpu::Buffer> m_cameraFallbackResponse;
    std::array<std::unique_ptr<gpu::Buffer>, kProjectionSetCount> m_projectionBuffers;
    // Deferred-upload queue attached by the render context; null means
    // uploads write through the mapping directly.
    DeferredBufferUploads* m_uploadQueue = nullptr;
    std::unique_ptr<gpu::ComputeKernel> m_geometryKernel;  // fusion_geometry on m_kernelLayout
    std::unique_ptr<gpu::Buffer> m_fusionTransportBuffer;
    std::unique_ptr<gpu::Buffer> m_fusionRecordBuffer;
    /// Bytes the last BeginFusionRecording armed; ReadFusionRecording returns
    /// exactly this even when the grow-only allocation is larger.
    u64 m_fusionRecordSize = 0;
    /// The 32-word record header the shader should see. Begin builds it and
    /// the SetFusion* calls patch single fields; it lands on the device with
    /// the next trace.
    std::array<u32, rendercore::kFusionHeaderWords> m_fusionHeader{};
    mutable u64 m_pendingFusionZero = 0;
    mutable bool m_fusionInitPending = false;
    Vector<u32> m_initialFusionMedia;
    std::unique_ptr<gpu::Buffer> m_fusionProbeBuffer;
    std::unique_ptr<gpu::Buffer> m_cameraFallbackAtmosHeader;
    std::unique_ptr<gpu::Buffer> m_cameraFallbackAtmosData;

    // SPIR-V bytecode (kept for lazy pipeline variant creation)
    std::vector<gpu::ShaderCode> m_spirvData;  // [raygen, chit, miss, shadow_miss, anyhit]

    // Pipeline variant cache (keyed by packed spec constants). Each kernel owns
    // its pipeline and shader binding table.
    std::unordered_map<u64, std::unique_ptr<gpu::TraceKernel>> m_variantCache;
    const gpu::TraceKernel* m_activeVariant = nullptr;
    u32 m_activeSpectralMode = 7;
    bool m_useObserverDescriptorSet = false;

    std::unique_ptr<gpu::TraceKernel> CreatePipelineVariant(const SpecConstants& spec);
    static u64 PackSpecKey(u32 spectralMode, u32 debugEnabled) {
        return (static_cast<u64>(debugEnabled) << 32) | spectralMode;
    }

    // Dynamic texture limit (based on device capabilities)
    // 1024 if descriptor indexing available, 32 otherwise
    u32 m_maxTextures = 1024;

    // Push constants (camera + sampling parameters): the shader's
    // PushConstantsRayGen (ql_shared_public.hlsli), whose first 108 bytes are the
    // public PushConstantsRayGen, locked there, and whose last word is
    // textureCount. These hold the host to the same 112 bytes.
    struct TracePushConstants {
        PushConstantsRayGen cameraAndSampling;
        u32 textureCount = 0;
    };
    QL_SHARED_LAYOUT(TracePushConstants, 112)
    QL_SHARED_OFFSET(TracePushConstants, cameraAndSampling, 0)
    QL_SHARED_OFFSET(TracePushConstants, textureCount, 108)
    static_assert(sizeof(TracePushConstants) <= 128);
    mutable TracePushConstants m_pushConstants{};
    std::unique_ptr<gpu::Buffer> m_spectralFallback;
    std::unique_ptr<gpu::Buffer> m_criFallback;
};

} // namespace quantiloom
