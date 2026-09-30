/**
 * @file ExternalRenderContext.cpp
 * @brief Implementation of external Vulkan context injection API
 *
 * @author blitzcolo
 */

// VMA must be included BEFORE GpuBuffer.hpp/GpuImage.hpp to avoid enum redefinition
#include <vk_mem_alloc.h>

#include "renderer/ExternalRenderContext.hpp"
#include "renderer/ConfigResolve.hpp"
#include "renderer/SpectralUnmixer.hpp"
#include "renderer/TemperatureTextureLoader.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/ThermalSunResponse.hpp"
#include "VulkanContextAdapter.hpp"
#include "RayTracingPipeline.hpp"
#include "AccelerationStructure.hpp"
#include "GpuBuffer.hpp"
#include "AsyncPixelReadback.hpp"
#include "GpuImage.hpp"
#include "GpuDisplayRange.hpp"
#include "TextureManager.hpp"
#include "CommandHelper.hpp"
#include "PerformanceLogger.hpp"
#include "BRDFLutGenerator.hpp"
#include "renderer/LightingParams.hpp"
#include "MaterialGpuData.hpp"
#include "atmos/AtmosphereBaker.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "postprocess/CameraConfigIO.hpp"
#include "postprocess/CameraConfigIOInternal.hpp"
#include "postprocess/CameraAutoControl.hpp"
#include "renderer/CameraResponseGpu.hpp"
#include "renderer/GpuCameraPipeline.hpp"

#include "renderer/ThermalEpochBuilder.hpp"
#include "renderer/ThermalPreview.hpp"
#include "renderer/TimelineState.hpp"
#include "renderer/EmissiveInvalidation.hpp"
#include "thermal/ThermalEpochs.hpp"
#include "core/Log.hpp"
#include "core/CacheDirectory.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "core/SpectralData.hpp"
#include "io/GltfLoader.hpp"
#include "io/UsdLoader.hpp"
#include "io/ImageIO.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>

// Not for the cache directory -- that moved to core/CacheDirectory.hpp -- but
// for GetModuleFileNameW, which resolves the DLL's own path further down.
#if defined(_WIN32)
    #include <windows.h>
#endif

namespace quantiloom {

using core::GetDefaultCacheDirectory;

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

    // The facade that owns this Impl. Used by internal steps that need a
    // public facade call (the camera scheduler poses the timeline through
    // SetTimelineTime). Re-pointed by the facade's move operations; never
    // null while the Impl is alive.
    ExternalRenderContext* owner = nullptr;

    // External Vulkan handles (cached for reference)
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    u32 graphicsQueueFamily = 0;
    VkFormat targetColorFormat = VK_FORMAT_B8G8R8A8_SRGB;

    // Current internal render extent -- what the trace, the sensor chain,
    // CLAHE and every extent-sized image are sized to.
    u32 width = 1280;
    u32 height = 720;

    // The extent the host last asked to be presented into, and the factor
    // between the two. The render extent is the target extent scaled down, so
    // a host that wants cheaper frames during a camera drag lowers the scale
    // and keeps presenting into the same swapchain image; the presenting blit
    // magnifies. At scale 1.0 (the default) the two extents are equal and
    // every path below is bit-identical to what it was before the split.
    u32 targetWidth = 1280;
    u32 targetHeight = 720;
    f32 renderScale = 1.0f;

    // Whether the accumulation format can be magnified with VK_FILTER_LINEAR.
    // Queried once at init; NEAREST is the fallback and only looks blockier.
    bool linearBlitSupported = false;

    // Scene data
    std::unique_ptr<Scene> scene;
    Camera camera;
    // Side table for curve slots not stored on the public Material type. Kept
    // in scene-material order and replaced atomically on every scene adoption.
    Vector<rendercore::MaterialGpuIndices> materialGpuIndices;

    // Acceleration structures

    // GPU resources
    std::unique_ptr<GpuImage> outputImage;
    std::unique_ptr<GpuImage> depthAovImage;  // primary-hit distance, R32_SFLOAT (binding 22)
    std::unique_ptr<GpuBuffer> lightingParamsBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    std::unique_ptr<GpuBuffer> spectralCurvesBuffer;
    std::unique_ptr<GpuBuffer> criBuffer;
    std::unique_ptr<GpuBuffer> solarLutBuffer;
    std::optional<std::pair<SpectralCurve, SpectralCurve>> cameraSolarSources;
    std::unique_ptr<GpuBuffer> atmosHeaderBuffer;  // AtmosNNHeaderGPU (binding 17)
    std::unique_ptr<GpuBuffer> atmosDataBuffer;    // Baked LUT blob (binding 20)
    std::unique_ptr<GpuBuffer> cieCmfBuffer;  // CIE 1931 CMF LUT for VIS_Fused mode (binding 19)
    std::unique_ptr<GpuBuffer> rgbToSpectrumBuffer;  // Jakob-Hanika coefficients (binding 25)
    std::unique_ptr<GpuBuffer> emissiveTriangleBuffer;  // world-space emitters for NEE (binding 23)
    bool emissiveTransformDirty = false;
    /// Per-element surface temperatures (binding 24). The interactive path
    /// runs no thermal solve -- a solve is an offline step, and its output is
    /// a state the viewport would have to be told about rather than compute --
    /// so this is one zero entry, which every instance's sentinel keeps the
    /// shader from reading. Bound because an unbound descriptor is not a valid
    /// one, and reading one is a device loss rather than a wrong colour.
    std::unique_ptr<GpuBuffer> thermalTemperatureBuffer;
    std::unique_ptr<GpuBuffer> thermalSunResponseBuffer;
    std::unique_ptr<GpuBuffer> thermalParameterTangentBuffer;
    /// What the tangent buffer currently holds, so a repeat ask is a
    /// no-op rather than a reupload and an accumulation reset.
    ThermalSensitivityParameter whatIfParameter = ThermalSensitivityParameter::Convection;
    f32 whatIfStep = 0.0f;
    std::unique_ptr<rendercore::ThermalPreview> thermalPreview;
    /// Why the last SetThermalTime did not produce temperatures. Held here
    /// rather than only in the preview because the reasons the facade rejects
    /// a solve -- no scene, no acceleration structure -- never reach it, and a
    /// panel that can only say "no result" leaves the user guessing.
    String thermalLastError;
    /// Mirrors what SetThermalSolveEnabled was last told. The preview owns the
    /// same bit, but asking it means building a whole status snapshot, and the
    /// timeline asks on every tick.
    bool thermalEnabled = false;

    /// Which nodes the clock moves, and where they would stand if it did not.
    /// Empty and inert for a scene that declared no [timeline].
    rendercore::TimelineState timeline;

    /// `thermal.timestep_s`, kept because it is the default epoch stride: a
    /// boundary finer than one step is one the solver cannot tell from no
    /// boundary at all.
    f64 thermalTimestep_s = 60.0;

    /// Lends the scene to the epoch builder and takes it back. Declared here
    /// rather than made on demand because ThermalPreview holds the pointer
    /// across solves.
    struct TimelineEpochHost final : rendercore::EpochGeometryHost {
        Impl* owner = nullptr;
        f64 restore_s = 0.0;
        bool captured = false;

        VkAccelerationStructureKHR ApplyEpoch(f64 t_s) override;
        void Restore() override;
    };
    TimelineEpochHost epochHost;
    /// Kept here as well as in the preview, because the preview is rebuilt
    /// with the pipeline and the host registered before either existed.
    std::function<void(u32, u32)> thermalEpochProgress;

    /// Work out where the geometry has to be re-measured, and tell the preview.
    /// Called when the trajectories change -- a config applied, a gizmo drag
    /// finished, a topology edit -- and never when the clock merely moves.
    void RefreshEpochPlan();

    // CRI management (CPU-side copy for rebuild when new entries are added)
    std::vector<ComplexRefractiveIndexGPU> criEntries;
    std::vector<SpectralCurveGPU> spectralCurveEntries;

    // Merged global geometry buffers (for multi-BLAS support)
    // All BLAS geometry data is merged into single global buffers
    // Shader uses InstanceGeometryInfo offsets to index correctly

    // IBL resources
    rendercore::BrdfLut brdfLut;
    rendercore::EnvironmentCubemap envMap;
    rendercore::SceneGeometry geometry;
    bool pendingTlasRefit = false;

    // Texture manager
    std::unique_ptr<TextureManager> textureManager;

    // Ray tracing pipeline
    std::unique_ptr<RayTracingPipeline> pipeline;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::string pipelineCachePath;  // Set in Create() based on InitParams or platform default

    // GPU timestamps around the trace dispatch. Resolved non-blocking at the
    // top of each RenderFrame, since the host submits after we return.
    std::unique_ptr<PerformanceLogger> perfLogger;
    std::unique_ptr<PerformanceLogger> cameraTracePerf;
    std::unique_ptr<PerformanceLogger> cameraFullPerf;
    CameraGpuTimings lastCameraGpuTimings{};

    // Command pool for internal operations
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // Rendering state
    SpectralMode spectralMode = SpectralMode::RGB;  // Default: Fast RGB mode
    DebugVisualizationMode debugMode = DebugVisualizationMode::None;  // Debug visualization mode
    /// The one number a debug view may need. Zero for every view that
    /// needs none, which is all of them but one.
    u32 debugParam = 0;
    f32 wavelength_nm = 550.0f;
    u32 spp = 1;
    LightingParams lightingParams;

    // NN atmosphere state (baked lazily before rendering when the key changes)
    AtmosphereNNConfig atmosphereConfig;              // CPU-side config (default: disabled)
    std::unique_ptr<AtmosModelPack> atmosModelPack;   // Loaded network packs
    uint64_t atmosBakeKey = 0;                        // 0 = nothing baked yet
    bool atmosphereActive = false;                    // A bake is on the GPU

    // Rebakes/uploads the NN atmosphere LUT when the bake key changed
    void UpdateAtmosphereNN();

    // Uploads lightingParams, substituting the atmosphere's air temperature
    // while a bake is active. Every write to lightingParams goes through it.
    void UploadLightingParams();

    /// Pack the solver's per-element sun response into binding 26's layout and
    /// put it on the device. Returns true when the buffer had to be
    /// reallocated and therefore rebound -- which is only when the element
    /// count changed, so a scrub is a plain upload with no wait.
    bool UploadThermalSunResponse(const Vector<f32>& sunSensitivity_K,
                                  const Vector<f32>& sunVisibility,
                                  const glm::vec3& sunDirection,
                                  const Vector<f32>& lagSensitivity_K = {},
                                  const Vector<f32>& lagVisibility = {},
                                  const Vector<glm::vec3>& lagDirection = {});

    /// The what-if tangent field and its step, on binding 27. Same contract as
    /// the sun response above: true when the buffer moved and the descriptor
    /// has to be rewritten.
    bool UploadThermalTangent(const Vector<f32>& tangent, f32 step);

    // Environment map state. True exactly while `envMap` holds a real map that
    // loaded, false while it holds the black placeholder. UploadLightingParams
    // masks enableEnvironmentMap with it, so this is what stops a host raising
    // the flag on a context that has nothing to sample.
    bool hasCustomEnvMap = false;

    // Accumulation
    u32 accumulatedSamples = 0;
    u32 frameIndex = 0;
    u64 pixelImageGeneration = 1;
    u64 rawPixelGeneration = 0;
    u32 rawPixelSamples = 0;

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

    // Seeds the Owen scrambles that stratify the first bounce. Drawn once per
    // accumulation round and then held for the whole round, which is the
    // opposite of what randomSeed above does and the reason they are separate:
    // the stratification is a property of the sequence of samples, so moving
    // its seed between samples would leave nothing to stratify.
    u32 sequenceSeed = constants::DEFAULT_SAMPLING_SEED;

    // Restart the sampling sequence. Called from ResetAccumulation() so the
    // sequence and the accumulation it feeds always begin together.
    void ReseedRng() {
        ++pixelImageGeneration;
        rng.seed(samplingSeed != 0U ? samplingSeed : std::random_device{}());
        sequenceSeed = randDist(rng);
    }

    // Statistics: GPU time of the most recent resolved trace dispatch, in
    // milliseconds. One dispatch = one sample, so this is the per-sample cost.
    // This used to be a steady_clock span across RenderFrame, which only
    // *records* commands -- it measured CPU command recording (microseconds)
    // while every reader treated it as the cost of tracing the scene.
    f32 lastSampleGpuMs = 0.0f;

    // Pixel readback buffer (for debug hover display)
    std::unique_ptr<GpuBuffer> pixelReadbackBuffer;
    std::unique_ptr<rendercore::AsyncPixelReadback> asyncPixelReadback;

    // Display enhancement resources. The pipeline is still the three CLAHE
    // passes -- Linear skips the first two and Equalize sums across tiles.
    DisplayEnhancementParams displayParams;
    std::unique_ptr<GpuImage> displayImage;           // tone-mapped output for display
    std::unique_ptr<GpuBuffer> claheHistogramBuffer;  // Per-tile histograms
    std::unique_ptr<GpuBuffer> claheCdfBuffer;        // Per-tile CDFs
    std::unique_ptr<GpuBuffer> claheMinMaxBuffer;     // Per-tile min/max for normalization
    std::unique_ptr<rendercore::GpuDisplayRange> displayRange;
    bool displayRangeInitAttempted = false;
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

    // Viewport pick (1x1 inline ray-query dispatch; see Pick())
    std::unique_ptr<GpuBuffer> pickOutputBuffer;  // host-visible, one PickResultGpu
    VkDescriptorSetLayout pickDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pickPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool pickDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet pickDescriptorSet = VK_NULL_HANDLE;
    VkPipeline pickPipeline = VK_NULL_HANDLE;
    VkShaderModule pickShader = VK_NULL_HANDLE;
    bool pickInitAttempted = false;  // one failed attempt is not retried

    void CreatePickPipeline();

    // Cached min/max values for CLAHE (computed after frame completion)
    f32 cachedImageMin = 0.0f;
    f32 cachedImageMax = 1.0f;
    bool hasCachedMinMax = false;

    // GPU Sensor simulation resources
    bool gpuSensorEnabled = false;
    SensorParams gpuSensorParams;
    bool gpuSensorWavelengthFromHost = false;
    camera::CameraConfig cameraConfig;
    bool cameraResourcesDirty = true;
    bool cameraCapturePending = false;
    bool cameraCaptureRecorded = false;
    bool cameraCaptureCompleted = false;
    u64 cameraAcquisitionIndex = 0;
    f64 cameraAcquisitionTimeSeconds = 0.0;
    std::unique_ptr<GpuImage> cameraMeasurementImage; // Binding 28, physical pixels, one layer per time stratum.
    std::unique_ptr<GpuImage> cameraMeasurementDepthImage; // Binding 31, per-stratum primary depth.
    std::unique_ptr<GpuImage> cameraBaselineImage;
    std::unique_ptr<GpuImage> cameraDepthImage;
    std::unique_ptr<GpuBuffer> cameraResponseBuffer; // Binding 29, knot/CDF table.
    std::unique_ptr<GpuBuffer> cameraDynamicCounterBuffer; // Binding 30, 4 x u32 raygen counters.
    std::unique_ptr<GpuBuffer> cameraAtmosHeaderBuffer;
    std::unique_ptr<GpuBuffer> cameraAtmosDataBuffer;
    std::unique_ptr<GpuBuffer> cameraBaselineAtmosHeaderBuffer;
    std::unique_ptr<GpuBuffer> cameraBaselineAtmosDataBuffer;
    std::unique_ptr<rendercore::GpuCameraPipeline> cameraGpuPipeline;
    f32 cameraAtmosMinNm = 0.0f;
    f32 cameraAtmosStepNm = 0.0f;
    u32 cameraAtmosCount = 0;
    // Interactive camera acquisition scheduling. When the camera is enabled,
    // RenderFrame drives acquisitions from the timeline clock instead of
    // tracing the visibility image; QueueCameraAcquisition remains the
    // explicit path for headless hosts. One state-advancing tick is one
    // committed acquisition -- redraws, presents and reprocesses never
    // advance it.
    struct CameraScheduler {
        // Set on the first RenderFrame with the camera enabled; before that
        // the explicit Queue/Record/Complete facade owns the acquisition.
        bool drivesAcquisition = false;
        u64 nextAcquisitionIndex = 0;
        f64 nextAcquisitionTimeSeconds = 0.0;
        f64 framePeriodSeconds = 1.0 / 30.0;
        // Scrubbing back before a committed acquisition (or a host reset
        // request) invalidates the detector's temporal history. Applied on
        // the next RenderFrame, which records no acquisition; the reason is
        // kept for GetCameraHistoryStatus.
        bool historyResetPending = false;
        String historyResetReason;
        u64 historyEpoch = 0;
        bool hasCommittedAcquisition = false;
        f64 lastCommittedAcquisitionTime = 0.0;
    };
    CameraScheduler cameraScheduler;

    // M4-2 acquisition history. One checkpoint deep is all the interactive
    // hosts need (rewind to the marked tick). The GPU half of a record holds
    // the device state; the host half carries the CaptureState feedback
    // fields that the M4-4 closed loop fills in.
    struct CameraHistoryRecord {
        rendercore::GpuCameraPipeline::GpuCheckpoint gpu;
        camera::CaptureState host;
    };
    std::vector<CameraHistoryRecord> cameraHistoryStack;
    camera::CaptureState cameraHostState;

    // M4-4 AE/AWB closed loop. The feedback (cameraHostState.next*) is
    // computed lazily once per committed tick from the statistics the last
    // submitted frame left on the device: RunCameraAutoControl runs at the
    // top of RecordCameraMeasurement (before the new tick is recorded, so
    // the readback is queue-ordered after the previous frame) and never
    // between record and submit. `cameraAutoControlDue` marks that the
    // device statistics buffer holds a committed tick's data (Queue clears
    // cameraCaptureCompleted, so that flag cannot serve); `cameraAutoLastAcquisition`
    // marks the index the feedback was last computed for, so a same-tick
    // re-record, a redraw or a display reprocess never moves the loop;
    // `cameraAutoAppliedAcquisition` marks the index the effective config was
    // last pushed to the device for.
    bool cameraAutoControlDue = false;
    u64 cameraAutoLastAcquisition = std::numeric_limits<u64>::max();
    u64 cameraAutoAppliedAcquisition = std::numeric_limits<u64>::max();
    bool cameraAutoThermalNoted = false;

    // M4-1 temporal stratification state. `cameraStratumSampleCounts` carries
    // each layer's running sample count across same-tick re-records so a
    // redraw continues the progressive average instead of restarting it;
    // it resets whenever the acquisition index moves. The remaining fields
    // describe the last committed acquisition for DynamicExposureReport.
    std::array<u32, 8> cameraStratumSampleCounts{};
    u64 cameraStratumRecordedAcquisition =
        std::numeric_limits<u64>::max();
    u32 cameraLastAcquisitionSamples = 0;
    u32 cameraLastStrataCount = 1;
    f64 cameraLastObjectMotion = 0.0;
    DynamicExposureReport lastDynamicReport;
    // Seven-stamp query pool bracketing the camera compute passes
    // (start/dynamic/psf/detector/display/isp/hsv-end). Resolved lazily by
    // GetLastCameraGpuTimings once the GPU has finished the frame.
    VkQueryPool cameraTimingPool = VK_NULL_HANDLE;
    f64 cameraTimestampPeriodNs = 1.0;

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
        perfLogger.reset();  // query pool needs the (external) device alive
        cameraTracePerf.reset();
        cameraFullPerf.reset();
        pipeline.reset();
        cameraGpuPipeline.reset();
        if (cameraTimingPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, cameraTimingPool, nullptr);
            cameraTimingPool = VK_NULL_HANDLE;
        }
        cameraMeasurementImage.reset();
        cameraMeasurementDepthImage.reset();
        cameraBaselineImage.reset();
        cameraDepthImage.reset();
        cameraResponseBuffer.reset();
        cameraDynamicCounterBuffer.reset();
        cameraAtmosHeaderBuffer.reset();
        cameraAtmosDataBuffer.reset();
        cameraBaselineAtmosHeaderBuffer.reset();
        cameraBaselineAtmosDataBuffer.reset();

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
        // The map is gone, so the load state that describes it has to go too --
        // otherwise a re-initialised context reports a custom map it no longer
        // has, and UploadLightingParams stops masking.
        hasCustomEnvMap = false;

        atmosHeaderBuffer.reset();
        atmosDataBuffer.reset();
        cieCmfBuffer.reset();
        rgbToSpectrumBuffer.reset();
        emissiveTriangleBuffer.reset();
        thermalPreview.reset();
        thermalTemperatureBuffer.reset();
        thermalSunResponseBuffer.reset();
        thermalParameterTangentBuffer.reset();
        solarLutBuffer.reset();
        criBuffer.reset();
        spectralCurvesBuffer.reset();
        materialBuffer.reset();
        lightingParamsBuffer.reset();
        asyncPixelReadback.reset();
        outputImage.reset();
        depthAovImage.reset();
        pixelReadbackBuffer.reset();

        // Cleanup CLAHE resources
        displayRange.reset();
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

        // Destroy pick resources
        pickOutputBuffer.reset();
        if (device != VK_NULL_HANDLE) {
            if (pickPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pickPipeline, nullptr);
                pickPipeline = VK_NULL_HANDLE;
            }
            if (pickShader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, pickShader, nullptr);
                pickShader = VK_NULL_HANDLE;
            }
            if (pickDescriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, pickDescriptorPool, nullptr);
                pickDescriptorPool = VK_NULL_HANDLE;
            }
            if (pickPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pickPipelineLayout, nullptr);
                pickPipelineLayout = VK_NULL_HANDLE;
            }
            if (pickDescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, pickDescriptorSetLayout, nullptr);
                pickDescriptorSetLayout = VK_NULL_HANDLE;
            }
        }
        pickInitAttempted = false;

        // Reset merged global geometry buffers
        geometry = {};

        if (commandPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }

        scene.reset();

        // The context adapter is deliberately NOT reset here. It owns the VMA
        // allocator, so anything holding a VMA allocation has to die first --
        // and resetting it at the end of this list only covers the members the
        // list remembers. cieCmfBuffer was missing from it for six months, and
        // freed itself against a dangling allocator every time this context was
        // destroyed: an access violation inside vmaDestroyBuffer, on every
        // minimize of the Studio window.
        //
        // contextAdapter is the first member declared, so it is the last one
        // destroyed, after every other member regardless of whether this
        // function knows about it. Leave it that way and do not add a member
        // above it.
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
    void AdoptScene(Scene&& loaded,
                    Vector<rendercore::MaterialGpuIndices> indices = {});

    void RebuildSceneGpuResources();

    void BuildAccelerationStructures();
    void UpdateGpuResources(bool rebuildEmitters = true);
    void RebuildEmissiveGeometry();

    /// Move the animated nodes to @p t_s and make the GPU agree.
    ///
    /// Deliberately NOT ExternalRenderContext::RefitAccelerationStructure:
    /// that one invalidates the thermal preview's geometry, which is right for
    /// a gizmo drag (it changed a rest pose, so the epoch plan is stale) and
    /// wrong for a scrub (the trajectory the epochs were planned from has not
    /// changed at all). Scrubbing a timeline must not cost an exchange
    /// precompute.
    ///
    /// @return the nodes that actually moved
    Vector<u32> ApplyTimelinePose(f64 t_s, bool deferRefit = false);
    void FlushPendingTlasRefit();
    // Config's renderer.enable_light_sampling. Held here rather than in
    // LightingParams, which has no bits left: turning it off is expressed by
    // publishing an emitter count of zero, which is the same thing the shader
    // already understands as "no lights to sample".
    bool enableLightSampling = true;
    void CreateDummyBuffers();
    void CreateBRDFLut();
    void CreateFallbackEnvMap();
    void CreatePipeline();

    void CreateCLAHEPipeline();
    void ExecuteCLAHE(VkCommandBuffer cmd, u32 width, u32 height);
    void ComputeImageMinMax(f32& outMin, f32& outMax);

    Result<void, String> EnsureCameraResources();
    Result<void, String> UpdateCameraAtmosphere();
    /// Run the interactive camera scheduler for one RenderFrame: commit a new
    /// acquisition when the timeline clock reached the next scheduled slot,
    /// re-record the current tick for convergence on redraws, and apply a
    /// pending history reset (which records nothing). Returns true when a new
    /// acquisition was committed.
    Result<bool, String> StepCameraAcquisition(VkCommandBuffer cmd);
    Result<void, String> RecordCameraMeasurement(VkCommandBuffer cmd);
    Result<void, String> RecordCameraBaseline(VkCommandBuffer cmd);
    Result<CameraData, String> CameraDataForCapture(
        SpectralMode mode, f64 wavelengthNm,
        f64 timeSeconds = std::numeric_limits<f64>::quiet_NaN()) const;

    // M4-4 AE/AWB closed loop. EffectiveCameraConfig mirrors
    // CpuCameraPipeline::EffectiveConfig: a committed acquisition consumes
    // the feedback the previous tick wrote into cameraHostState.next*; a
    // fresh state (nextExposure <= 0, i.e. nothing committed since the last
    // reset) starts from the authored manual values. MaybeRun executes the
    // controller for the last committed tick exactly once; RunCameraAutoControl
    // is the StepAutoControl wrapper (thermal detectors are skipped with the
    // same warning the CPU chain logs).
    [[nodiscard]] camera::CameraConfig EffectiveCameraConfig() const;
    void MaybeRunCameraAutoControl();
    void RunCameraAutoControl();
    /// Record the display-half reprocess (demosaic -> color -> display ->
    /// HSV) over the last acquisition on `cmd`. No trace, no statistics, no
    /// state advance; see ReprocessCameraDisplay.
    [[nodiscard]] Result<void, String> RecordCameraDisplayReprocess(
        VkCommandBuffer cmd);

    /// Which image the swapchain blit should read, and the extent to blit
    /// from: CLAHE's output if the viewport override is on, then the camera
    /// display product, then the raw accumulation. One function because
    /// RenderFrame, PresentAccumulated and ReprocessAccumulated must never
    /// disagree about what "the current image" is.
    struct DisplaySource {
        VkImage image;
        u32 width;
        u32 height;
    };
    struct ImageExtent {
        u32 width;
        u32 height;
    };
    [[nodiscard]] DisplaySource CurrentDisplaySource() const;

    /// The camera display product once an acquisition has completed, else
    /// null (the image exists before the first acquisition but holds no
    /// measurement).
    [[nodiscard]] const GpuImage* CameraDisplayImage() const {
        if (!cameraConfig.enabled || !cameraGpuPipeline || !cameraCaptureCompleted)
            return nullptr;
        return cameraGpuPipeline->GetOutputs().display;
    }

    /// Whether CLAHE has a valid input this frame. It reads the camera's
    /// agcSource (infrared) or corrected product (visible) when the camera is
    /// enabled, which exists only after the first acquisition; until then
    /// there is nothing to enhance and the display falls through to the
    /// camera display (or the raw accumulation).
    [[nodiscard]] bool ClaheInputReady() const {
        if (!cameraConfig.enabled) return true;
        if (!cameraGpuPipeline || !cameraCaptureCompleted) return false;
        const auto outputs = cameraGpuPipeline->GetOutputs();
        return cameraConfig.device.detector == camera::DetectorKind::Thermal
                   ? outputs.agcSource != nullptr
                   : outputs.corrected != nullptr;
    }

    /// The image CLAHE reads: the camera's agcSource (infrared camera -- the
    /// pre-AGC scalar the camera chain would have tone-mapped), the corrected
    /// product for a visible camera, else the raw accumulation.
    [[nodiscard]] const GpuImage* ClaheInputImage() const {
        if (cameraConfig.enabled && cameraGpuPipeline) {
            const auto outputs = cameraGpuPipeline->GetOutputs();
            if (cameraConfig.device.detector == camera::DetectorKind::Thermal &&
                outputs.agcSource)
                return outputs.agcSource;
            if (outputs.corrected) return outputs.corrected;
        }
        return outputImage.get();
    }

    /// The camera config asked for CLAHE as the persistent infrared tone: the
    /// camera display pass writes the pre-AGC scalar to both outputs and the
    /// host's CLAHE pipeline runs over the agcSource, its result becoming the
    /// final display. The viewport CLAHE parameters (tile grid, clip limit,
    /// palette) drive the run, exactly as they drive the viewport override.
    [[nodiscard]] bool CameraClahePersistent() const {
        return cameraConfig.enabled &&
               cameraConfig.device.detector == camera::DetectorKind::Thermal &&
               cameraConfig.isp.infraredTone == DisplayToneMode::Clahe;
    }

    /// The extent CLAHE runs at, and the extent its displayImage is kept at:
    /// the physical sensor array when the camera is enabled, else the render
    /// extent.
    [[nodiscard]] ImageExtent ClaheInputExtent() const {
        if (cameraConfig.enabled) {
            return {cameraConfig.optics.sensorWidthPx,
                    cameraConfig.optics.sensorHeightPx};
        }
        return {width, height};
    }

    /// (Re)create the CLAHE display image at the current ClaheTarget extent
    /// and rebind it. Called when CLAHE initializes, the render extent
    /// changes, or the camera configuration changes the input size.
    void RecreateClaheDisplayImage();

    /// Copy @p source onto the swapchain image and leave it in PRESENT_SRC.
    /// The transitions and the format-converting blit, with no tracing --
    /// which is what lets a frame be drawn without advancing the accumulation.
    /// @p srcWidth/@p srcHeight are the internal render extent, @p dstWidth/
    /// @p dstHeight the target's. They differ whenever the render scale is
    /// below 1.0, and the blit magnifies.
    void BlitToTarget(VkCommandBuffer cmd, VkImage source, VkImage targetImage,
                      VkImageLayout targetLayout,
                      u32 srcWidth, u32 srcHeight,
                      u32 dstWidth, u32 dstHeight,
                      VkPipelineStageFlags sourceStage);

    /// Recreate every render-extent-sized resource at @p renderW x @p renderH.
    /// Waits for the GPU and resets the accumulation: nothing survives a
    /// change of extent.
    void ApplyInternalExtent(u32 renderW, u32 renderH);

    /// Record the extent the host is presenting into and bring the internal
    /// render extent in line with it and the current scale. The single place
    /// the two extents are related; both Resize() and RenderFrame() go
    /// through it.
    void EnsureExtent(u32 targetW, u32 targetH);

    /// One dimension of the target extent, scaled. Never zero.
    [[nodiscard]] u32 ScaledDim(u32 targetDim) const {
        const auto scaled = static_cast<u32>(
            std::lround(static_cast<f64>(targetDim) * renderScale));
        return std::max(1u, scaled);
    }

    /// Whether the internal render extent is the one the current scale and
    /// target extent call for. False between a SetRenderScale and the
    /// RenderFrame that acts on it.
    [[nodiscard]] bool ExtentMatchesScale() const {
        return ScaledDim(targetWidth) == width && ScaledDim(targetHeight) == height;
    }

    /// Map one coordinate from target space into render space, clamped to the
    /// last row/column. The identity at scale 1.0.
    [[nodiscard]] static u32 MapToRender(u32 coord, u32 targetDim, u32 renderDim) {
        if (targetDim == 0U || targetDim == renderDim) {
            return coord;
        }
        const u64 scaled =
            (static_cast<u64>(coord) * renderDim) / static_cast<u64>(targetDim);
        return static_cast<u32>(std::min<u64>(scaled, renderDim - 1U));
    }

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
    m_impl->owner = this;
}

ExternalRenderContext::~ExternalRenderContext() = default;

// The Impl back-points at its facade; a move has to re-point it.
ExternalRenderContext::ExternalRenderContext(ExternalRenderContext&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    if (m_impl) m_impl->owner = this;
}
ExternalRenderContext& ExternalRenderContext::operator=(
    ExternalRenderContext&& other) noexcept {
    m_impl = std::move(other.m_impl);
    if (m_impl) m_impl->owner = this;
    return *this;
}

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
    targetWidth = params.width;
    targetHeight = params.height;

    // Magnifying the accumulation needs a filterable source. Every driver that
    // supports ray tracing supports this for RGBA32F, but the fallback costs
    // one query and turns "wrong" into "blockier".
    VkFormatProperties accumFormatProps{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT,
                                        &accumFormatProps);
    linearBlitSupported =
        (accumFormatProps.optimalTilingFeatures &
         VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0U;
    if (!linearBlitSupported) {
        QL_LOG_WARN("R32G32B32A32_SFLOAT does not support linear filtering: a "
                    "reduced render scale will magnify with NEAREST");
    }

    // The presenting blit is the only thing between linear radiance and the
    // screen, so the target format's transfer function is what encodes it. A
    // non-sRGB target displays linear values uncorrected -- about a stop and a
    // half dark, and easy to mistake for an underexposed scene. This field was
    // stored and never read for a long time; checking it is what it is for.
    switch (targetColorFormat) {
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            break;
        default:
            QL_LOG_WARN("Target colour format {} has no sRGB encoding: the present "
                        "blit will write linear radiance uncorrected and the viewport "
                        "will read darker than an equivalent CLI render. Ask the "
                        "windowing layer for an sRGB swapchain format.",
                        static_cast<int>(targetColorFormat));
            break;
    }

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

    perfLogger = std::make_unique<PerformanceLogger>(*contextAdapter);
    cameraTracePerf = std::make_unique<PerformanceLogger>(*contextAdapter);
    cameraFullPerf = std::make_unique<PerformanceLogger>(*contextAdapter);

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

    outputImage = rendercore::CreateRenderTarget(*contextAdapter, params.width, params.height);
    depthAovImage = rendercore::CreateRenderTarget(*contextAdapter, params.width, params.height,
                                                   VK_FORMAT_R32_SFLOAT);

    // Create lighting params buffer
    lightingParamsBuffer = std::make_unique<GpuBuffer>(
        contextAdapter->GetAllocator(),
        sizeof(LightingParams),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    UploadLightingParams();

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

ConfigApplyReport ExternalRenderContext::ApplyConfig(const Config& config,
                                                     const ConfigApplyOptions& options) {
    ConfigApplyReport report;

    // 1. Read the file before touching any GPU state, so a config that cannot
    //    be honoured leaves the context as it was rather than half-replaced.
    auto resolvedResult = rendercore::ResolveRenderConfig(config, options, report);
    if (!resolvedResult.has_value()) {
        return report;  // The messages already say why; ok() is false.
    }
    auto resolved = std::move(resolvedResult.value());

    // 2. The scene itself, its path resolved the same way every other path in
    //    the file is.
    rendercore::SceneLoadInfo sceneInfo;
    auto sceneResult =
        rendercore::LoadSceneFromConfig(config, options.baseDir, &resolved, &sceneInfo);
    if (!sceneResult.has_value()) {
        report.messages.push_back({ConfigApplyMessage::Severity::Error, "scene",
                                   "Failed to load scene: " + sceneResult.error()});
        QL_LOG_ERROR("ApplyConfig: failed to load scene: {}", sceneResult.error());
        return report;
    }
    Scene loadedScene = std::move(sceneResult.value());

    if (loadedScene.materials.empty()) {
        loadedScene.materials.push_back(
            Material::CreateLambertian(resolved.defaultAlbedo, "DefaultMaterial"));
    }

    // 3. Everything the config says about materials, which needs the scene:
    //    curves, refractive indices, the IR temperature backfill, [[materials]].
    auto spectraResult =
        rendercore::ResolveMaterialSpectra(config, loadedScene, resolved, options, report);
    if (!spectraResult.has_value()) {
        report.messages.push_back({ConfigApplyMessage::Severity::Error, "materials",
                                   spectraResult.error()});
        return report;
    }
    auto spectra = std::move(spectraResult.value());

    // The CLI hands the resolved slots to BuildMaterialBuffer as a side table;
    // this path has no such table, because UpdateGpuResources() reads the
    // indices off each Material. Same mapping, written where this side looks
    // for it -- and it must happen before AdoptScene, which builds the buffer.
    // Endmember weight maps first: it reads the base-colour pixels, which
    // AdoptScene's upload releases, and it fills in the weight texture indices
    // the loop below copies onto the materials.
    rendercore::BuildUnmixWeightTextures(loadedScene, spectra, options.baseDir, report);

    // Authored temperature maps, in the same pre-upload window: the mount
    // appends textures, and AdoptScene's upload fixes the indices.
    rendercore::MountTemperatureTextures(loadedScene, options.baseDir, report);

    // Keep the base colours of curve-bound materials readable. Assigning a new
    // endmember from the library panel re-unmixes them, and by then the only
    // other copy is compressed and mipped on the device.
    for (const auto& mat : loadedScene.materials) {
        if (!spectra.materialNameToEndmembers.contains(mat.name)) continue;
        if (mat.baseColorTextureIndex < 0 ||
            mat.baseColorTextureIndex >= static_cast<i32>(loadedScene.textures.size())) {
            continue;
        }
        loadedScene.textures[static_cast<usize>(mat.baseColorTextureIndex)].retainCpuPixels = true;
    }

    for (auto& mat : loadedScene.materials) {
        if (auto it = spectra.materialNameToCurve.find(mat.name);
            it != spectra.materialNameToCurve.end()) {
            mat.spectralReflectanceCurveIndex = it->second;
        }
        if (auto it = spectra.materialNameToRefractiveIndex.find(mat.name);
            it != spectra.materialNameToRefractiveIndex.end()) {
            mat.complexRefractiveIndexIndex = it->second;
        }
        if (auto it = spectra.materialNameToEndmembers.find(mat.name);
            it != spectra.materialNameToEndmembers.end()) {
            mat.endmemberCurveIndex1 = it->second.curves[1];
            mat.endmemberCurveIndex2 = it->second.curves[2];
            mat.endmemberCurveIndex3 = it->second.curves[3];
            mat.weightTextureIndex = it->second.weightTextureIndex;
        }
        if (auto it = spectra.materialNameToSheenCurve.find(mat.name);
            it != spectra.materialNameToSheenCurve.end()) {
            mat.sheenReflectanceCurveIndex = it->second;
        }
        if (auto it = spectra.materialNameToClearcoatCurve.find(mat.name);
            it != spectra.materialNameToClearcoatCurve.end()) {
            mat.clearcoatReflectanceCurveIndex = it->second;
        }
        if (auto it = spectra.materialNameToDiffuseTransmissionCurve.find(mat.name);
            it != spectra.materialNameToDiffuseTransmissionCurve.end()) {
            mat.diffuseTransmissionColorCurveIndex = it->second;
        }
        // ResolveMaterialSpectra already wrote this onto the material, along
        // with the emissiveFactor it derived from the same curve. Read it back
        // from the map anyway, so the map stays the single authority and a
        // future resolve that stops touching the material in place does not
        // silently leave the interactive path unlit.
        if (auto it = spectra.materialNameToEmissiveCurve.find(mat.name);
            it != spectra.materialNameToEmissiveCurve.end()) {
            mat.emissiveRadianceCurveIndex = it->second;
        }
        if (auto it = spectra.bandAveragedIREmissivity.find(mat.name);
            it != spectra.bandAveragedIREmissivity.end()) {
            mat.bandAveragedIREmissivity = it->second;
        }
    }

    // 4. Render state before the scene, so the one rebuild AdoptScene triggers
    //    already builds the material buffer at the right wavelength. Setting it
    //    afterwards would work and rebuild everything a second time.
    m_impl->spectralMode = resolved.mode;
    m_impl->wavelength_nm = resolved.wavelengthNm;
    m_impl->spp = resolved.spp;
    m_impl->samplingSeed = resolved.seed;
    if (options.applyDebugMode) {
        m_impl->debugMode = static_cast<DebugVisualizationMode>(resolved.debugMode);
    }

    // The curve and refractive-index tables the material indices point into.
    // Uploaded directly rather than through AddSpectralCurve, which rebuilds
    // the whole buffer per curve and would do so before the scene exists.
    m_impl->spectralCurveEntries = spectra.curves;
    m_impl->criEntries = spectra.refractiveIndices;
    if (!m_impl->spectralCurveEntries.empty()) {
        const size_t bytes = m_impl->spectralCurveEntries.size() * sizeof(SpectralCurveGPU);
        m_impl->spectralCurvesBuffer = std::make_unique<GpuBuffer>(
            m_impl->contextAdapter->GetAllocator(), bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        m_impl->spectralCurvesBuffer->Upload(m_impl->spectralCurveEntries.data(), bytes);
    }
    if (!m_impl->criEntries.empty()) {
        const size_t bytes = m_impl->criEntries.size() * sizeof(ComplexRefractiveIndexGPU);
        m_impl->criBuffer = std::make_unique<GpuBuffer>(
            m_impl->contextAdapter->GetAllocator(), bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        m_impl->criBuffer->Upload(m_impl->criEntries.data(), bytes);
    }

    // 5. One GPU rebuild: textures, acceleration structures, material buffer,
    //    pipeline. CreatePipeline binds whatever buffers are current, which is
    //    why the two above are already in place.
    Vector<rendercore::MaterialGpuIndices> materialIndices;
    materialIndices.reserve(loadedScene.materials.size());
    for (const auto& mat : loadedScene.materials) {
        auto slots = rendercore::IndicesFromMaterial(mat);
        if (auto it = spectra.materialNameToIrEmissivityCurve.find(mat.name);
            it != spectra.materialNameToIrEmissivityCurve.end()) {
            if (it->second < 0 || static_cast<usize>(it->second) >= spectra.curves.size()) {
                report.messages.push_back({ConfigApplyMessage::Severity::Error,
                                           "materials", "IR emissivity curve index is invalid for '" +
                                               mat.name + "'"});
                return report;
            }
            slots.irEmissivityCurve = it->second;
        }
        if (auto it = spectra.materialNameToIrTransmittanceCurve.find(mat.name);
            it != spectra.materialNameToIrTransmittanceCurve.end()) {
            if (it->second < 0 || static_cast<usize>(it->second) >= spectra.curves.size()) {
                report.messages.push_back({ConfigApplyMessage::Severity::Error,
                                           "materials", "IR transmittance curve index is invalid for '" +
                                               mat.name + "'"});
                return report;
            }
            slots.irTransmittanceCurve = it->second;
        }
        materialIndices.push_back(slots);
    }
    m_impl->AdoptScene(std::move(loadedScene), std::move(materialIndices));
    report.sceneLoaded = true;

    // 5b. The clock, built against the adopted scene because the rest poses it
    //     snapshots are the ones [[nodes]] just finished writing. Applied
    //     before the thermal block below, so that a solve started here sees the
    //     geometry where `time_s` puts it rather than where the files did.
    m_impl->timeline = rendercore::TimelineState::Build(
        *m_impl->scene, resolved.timeline, resolved.models, sceneInfo, spectra.nodeMotion,
        report);
    m_impl->thermalTimestep_s = resolved.thermal.timestep_s;
    if (m_impl->timeline.Present()) {
        m_impl->ApplyTimelinePose(resolved.timeline.time_s);
    }
    m_impl->RefreshEpochPlan();

    // 6. Illuminant, then lighting. UploadLightingParams is the only legal
    //    writer of that buffer -- it substitutes the atmosphere's air
    //    temperature while a bake is active.
    if (resolved.solarSunSky) {
        SetSolarSpectralLUT(resolved.solarSunSky->first, resolved.solarSunSky->second);
    }
    m_impl->lightingParams = resolved.lighting;
    // resolved.lighting is built from the config alone, so its emissive fields
    // are zero -- assigning it wholesale would switch light sampling off for
    // every scene loaded through a config, silently and only there. The scene
    // is already adopted at this point, so recomputing them is both correct and
    // where the config's own switch takes effect.
    m_impl->enableLightSampling = resolved.enableLightSampling;
    m_impl->RebuildEmissiveGeometry();   // also uploads the lighting params

    // 7. Atmosphere. The bake itself is lazy, on the first frame that needs it.
    SetAtmosphere(resolved.atmosphere);

    // 8. Camera. AdoptScene took the scene file's camera; the config is the
    //    document of record and overrides it. Aspect ratio stays the
    //    viewport's, not the config resolution's -- see the report for that.
    m_impl->camera = resolved.camera;
    // Target extent, not render extent, for the same reason ApplyInternalExtent
    // uses it: the two differ by rounding under a reduced render scale, and the
    // framing must not depend on the scale.
    m_impl->camera.SetAspectRatio(static_cast<f32>(m_impl->targetWidth) /
                                  static_cast<f32>(m_impl->targetHeight));

    // Three conditions, and only the first was here before. Skipping leaves the
    // placeholder bound with the lighting flag at 0, which is what a scene
    // without an environment should render with.
    //
    // A named path, because calling LoadEnvironmentMap("") was never going to
    // load anything -- it just produced a warning about a missing map on every
    // config that has none, which is most of them. An absent map is not an error.
    //
    // RGB, because no spectral branch of closesthit.rchit samples the cubemap
    // (they read solar_lut and the analytic sky instead), so converting an
    // equirect into a cubemap for one is work whose result nothing reads. The
    // resolver has already set the flag to 0 for those modes and warned; this
    // gate has to be here rather than inside LoadEnvironmentMap, because a
    // successful load raises the flag and would otherwise overwrite that 0.
    //
    // One consequence worth naming: a multispectral config is previewed in RGB
    // further down, but the mode is still Multispectral here, so a map it names
    // does not light the preview. That is the invariant working as intended --
    // multispectral is a quantitative mode -- but it is a visible change.
    if (resolved.environmentMapEnabled && !resolved.environmentMap.empty() &&
        resolved.mode == SpectralMode::RGB) {
        if (auto envResult = LoadEnvironmentMap(resolved.environmentMap);
            !envResult.has_value()) {
            report.messages.push_back({ConfigApplyMessage::Severity::Warning,
                                       "renderer.environment_map", envResult.error()});
            QL_LOG_WARN("ApplyConfig: {}", envResult.error());
        }
    }

    m_impl->gpuSensorParams = resolved.sensor; // Legacy getter remains readable.
    if (auto cameraApplied = SetCameraConfig(resolved.cameraConfig);
        !cameraApplied.has_value()) {
        report.messages.push_back({ConfigApplyMessage::Severity::Error,
                                   "sensor", cameraApplied.error()});
        return report;
    }

    // Thermal solve. A failing solve is a report message, not a failed apply.
    if (resolved.thermal.enabled && m_impl->thermalPreview) {
        try {
            ThermalSolveParams tp;
            tp.startTime_h = resolved.thermal.startTime_h;
            tp.timestep_s = resolved.thermal.timestep_s;
            tp.layerCount = resolved.thermal.nodeCount;
            tp.initial = resolved.thermal.initial == thermal::InitialCondition::Steady
                ? ThermalInitialCondition::Steady : ThermalInitialCondition::Uniform;
            tp.initialTemperature_K = resolved.thermal.initialTemperature_K;
            tp.exchangeRays = resolved.thermal.exchangeRays;
            tp.exchangeTopK = resolved.thermal.exchangeTopK;
            tp.airTemperature_K = resolved.thermal.airTemperature_K;
            tp.sunIrradiance_W_m2 = resolved.thermal.sunIrradiance_W_m2;
            tp.diffuseIrradiance_W_m2 = resolved.thermal.diffuseIrradiance_W_m2;
            tp.skyTemperature_K = resolved.thermal.skyTemperature_K;
            tp.relativeHumidity = resolved.thermal.relativeHumidity;
            tp.forcingFile = resolved.thermal.forcingFile;
            tp.checkpointStride_h = resolved.thermal.checkpointStride_h;
            switch (resolved.thermal.convection.model) {
                case thermal::ConvectionModel::Wind:
                    tp.convectionModel = ThermalConvectionModel::Wind;
                    break;
                case thermal::ConvectionModel::Stability:
                    tp.convectionModel = ThermalConvectionModel::Stability;
                    break;
                case thermal::ConvectionModel::Constant:
                    tp.convectionModel = ThermalConvectionModel::Constant;
                    break;
            }
            tp.convectionWindA_W_m2K = resolved.thermal.convection.windIntercept_W_m2K;
            tp.convectionWindB_W_s_m3K = resolved.thermal.convection.windSlope_W_s_m3K;
            tp.convectionFreeC = resolved.thermal.convection.freeCoefficient;
            tp.convectionReferenceHeight_m = resolved.thermal.convection.referenceHeight_m;
            tp.convectionStableDamping = resolved.thermal.convection.stableDamping;
            tp.lateralConduction = resolved.thermal.lateralConduction;
            tp.sunMemoryLags = resolved.thermal.sunMemoryLags;
            // The two measurement switches. sunCorrection changes what the
            // solve carries, so the viewport has to be told or it silently
            // renders the corrected field a config asked not to have;
            // dumpElementsFile only records where a dump would go, since the
            // write here is DumpThermalElements() rather than the solve.
            tp.sunCorrection = resolved.thermal.sunCorrection;
            tp.dumpElementsFile = resolved.thermal.dumpElementsFile;
            tp.parameterSensitivities.clear();
            for (const thermal::ThermalParameter p : resolved.thermal.parameterSensitivities) {
                switch (p) {
                    case thermal::ThermalParameter::Convection:
                        tp.parameterSensitivities.push_back(
                            ThermalSensitivityParameter::Convection);
                        break;
                    case thermal::ThermalParameter::Emissivity:
                        tp.parameterSensitivities.push_back(
                            ThermalSensitivityParameter::Emissivity);
                        break;
                    case thermal::ThermalParameter::Absorptivity:
                        tp.parameterSensitivities.push_back(
                            ThermalSensitivityParameter::Absorptivity);
                        break;
                    case thermal::ThermalParameter::Conductivity:
                        tp.parameterSensitivities.push_back(
                            ThermalSensitivityParameter::Conductivity);
                        break;
                    case thermal::ThermalParameter::HeatCapacity:
                        tp.parameterSensitivities.push_back(
                            ThermalSensitivityParameter::HeatCapacity);
                        break;
                    case thermal::ThermalParameter::Count:
                        break;
                }
            }
            SetThermalSolveParams(tp);

            ClearThermalMaterials();
            for (const auto& [name, mat] : spectra.thermalMaterials) {
                ThermalMaterialParams tmp;
                tmp.isShell = mat.isShell;
                tmp.conductivity_W_mK = mat.conductivity_W_mK;
                tmp.density_kg_m3 = mat.density_kg_m3;
                tmp.specificHeat_J_kgK = mat.specificHeat_J_kgK;
                tmp.thickness_m = mat.thickness_m;
                tmp.convection_W_m2K = mat.convection_W_m2K;
                tmp.shortwaveAbsorptivity = mat.shortwaveAbsorptivity;
                tmp.wetnessFactor = mat.wetnessFactor;
                tmp.internalHeat_W_m2 = mat.internalHeat_W_m2;
                tmp.interiorFixedTemperature =
                    mat.interiorBoundary == thermal::InteriorBoundary::FixedTemperature;
                tmp.interiorAmbient =
                    mat.interiorBoundary == thermal::InteriorBoundary::AmbientInterior;
                tmp.interiorTemperature_K = mat.interiorTemperature_K;
                tmp.interiorConvection_W_m2K = mat.interiorConvection_W_m2K;
                SetThermalMaterial(name, tmp);
                ++report.thermalMaterialsApplied;
            }

            if (m_impl->thermalPreview) {
                m_impl->thermalPreview->SetFallbackSunDirection(resolved.lighting.sunDirection);
            }
            SetThermalSolveEnabled(true);
            report.thermalSolveEnabled = true;

            // With a clock, `thermal.time_h` stops meaning "the hour to
            // render" and starts meaning "the hour at the timeline's start" --
            // the hour to render is whatever `time_s` maps to from there.
            f64 hour = resolved.thermal.time_h;
            if (m_impl->timeline.Present()) {
                m_impl->timeline.SetThermalMapping(resolved.thermal.time_h,
                                                   resolved.timeline.thermalTimeScale);
                hour = m_impl->timeline.HourAt(resolved.timeline.time_s);
            }

            if (auto thermalResult = SetThermalTime(hour); !thermalResult.has_value()) {
                report.messages.push_back({ConfigApplyMessage::Severity::Warning,
                                           "thermal", thermalResult.error()});
                QL_LOG_WARN("ApplyConfig: thermal solve: {}", thermalResult.error());
            }
        } catch (const std::exception& ex) {
            report.messages.push_back({ConfigApplyMessage::Severity::Warning,
                                       "thermal", ex.what()});
            QL_LOG_WARN("ApplyConfig: thermal solve failed: {}", ex.what());
            SetThermalSolveEnabled(false);
        }
        report.thermalEpochs = GetThermalSolveStatus().thermalEpochCount;
    } else {
        SetThermalSolveEnabled(false);
    }

    if (resolved.hasHyperspectralSection) {
        report.messages.push_back(
            {ConfigApplyMessage::Severity::Info, "hyperspectral",
             "[hyperspectral] describes a cube render, which is a separate "
             "non-progressive renderer -- this context ignores it and previews "
             "the scene in its spectral mode instead."});
    }
    if (resolved.mode == SpectralMode::Multispectral) {
        report.messages.push_back(
            {ConfigApplyMessage::Severity::Warning, "spectral.mode",
             "spectral.mode = \"multispectral\" renders a cube, which this "
             "context cannot do progressively. Previewing in RGB instead."});
        QL_LOG_WARN("ApplyConfig: multispectral previewed as RGB");
        m_impl->spectralMode = SpectralMode::RGB;
    }

    ResetAccumulation();

    QL_LOG_INFO("ApplyConfig: {} spectral curve(s), {} refractive index(es), "
                "{} material(s) backfilled, {} overridden, atmosphere {}",
                report.spectralCurvesLoaded, report.refractiveIndicesLoaded,
                report.materialsTemperatureBackfilled, report.materialsOverridden,
                report.atmosphereEnabled ? "on" : "off");
    return report;
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
void ExternalRenderContext::Impl::AdoptScene(
    Scene&& loaded, Vector<rendercore::MaterialGpuIndices> indices) {
    scene = std::make_unique<Scene>(std::move(loaded));
    materialGpuIndices = std::move(indices);
    if (materialGpuIndices.size() != scene->materials.size()) {
        materialGpuIndices.clear();
    }

    // Setup camera from scene
    camera = scene->camera;
    camera.SetAspectRatio(static_cast<f32>(width) / static_cast<f32>(height));

    if (thermalPreview) thermalPreview->InvalidateGeometry();

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
    return LoadSceneFromGltf(gltfPath, GltfSceneOptions{});
}

Result<void, String> ExternalRenderContext::LoadSceneFromGltf(const String& gltfPath,
                                                              const GltfSceneOptions& options) {
    QL_LOG_INFO("Loading glTF scene: {}", gltfPath);

    GltfLoadOptions loadOptions;
    loadOptions.variant = options.variant;

    auto result = GltfLoader::LoadFromFile(gltfPath, loadOptions);
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
    return LoadSceneFromUsd(usdPath, UsdSceneOptions{});
}

Result<void, String> ExternalRenderContext::LoadSceneFromUsd(const String& usdPath,
                                                             const UsdSceneOptions& options) {
    QL_LOG_INFO("Loading USD scene: {}", usdPath);

    UsdLoadOptions loadOptions;
    if (!options.variantSelections.empty()) {
        // The same one-string syntax a config uses, so a host and a .toml
        // cannot disagree about what a selection means.
        String spec;
        for (const String& selection : options.variantSelections) {
            if (!spec.empty()) {
                spec += ',';
            }
            spec += selection;
        }
        auto parsed = ParseUsdVariantSpec(spec);
        if (!parsed.has_value()) {
            return Result<void, String>::Err(parsed.error());
        }
        loadOptions.variantSelections = std::move(parsed.value());
    }
    loadOptions.timeCode = options.timeCode;
    loadOptions.useDefaultTime = options.useDefaultTime;
    loadOptions.payloadPolicy = options.loadPayloads ? UsdLoadOptions::PayloadPolicy::LoadAll
                                                     : UsdLoadOptions::PayloadPolicy::LoadNone;
    loadOptions.applyStageMetrics = options.applyStageMetrics;

    auto result = UsdLoader::LoadFromFile(usdPath, loadOptions);
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

// Uploads the lighting parameters, substituting the NN atmosphere's ground
// temperature for the thermal-sky fallback while a bake is on the GPU.
//
// This is not cosmetic. AtmosSkyRadianceIR divides the network's zenith
// downwelling by B(T_air) to recover an emissivity, so a T_air unrelated to
// the bake distorts the entire horizon ramp, and the max-depth Planck
// fallback in closesthit.rchit with it. src/app/main.cpp makes the same
// substitution; without this the two front ends rendered the same scene
// differently. The host's own value is left intact in lightingParams so it
// returns when the atmosphere is switched off, and so GetLightingParams keeps
// reporting what the host set.
//
// It also masks enableEnvironmentMap with whether a map is actually loaded, for
// the same reason and in the same way. The flag says the shader may sample
// binding 10 as a light source, and binding 10 holds a black placeholder
// whenever no map loaded -- so an unmasked 1 there does not brighten the scene,
// it darkens it: the shader takes the split-sum path, adds black, and zeroes the
// traced specular residual (qSpec) that would otherwise have carried the
// reflection. A host can raise the flag through SetLightingParams at any time
// and this is the one place every upload passes through, so the check lives here
// rather than at each of the nine call sites.
bool ExternalRenderContext::Impl::UploadThermalSunResponse(
    const Vector<f32>& sunSensitivity_K, const Vector<f32>& sunVisibility,
    const glm::vec3& sunDirection, const Vector<f32>& lagSensitivity_K,
    const Vector<f32>& lagVisibility, const Vector<glm::vec3>& lagDirection) {
    const auto records = rendercore::MakeThermalSunResponse(
        sunSensitivity_K, sunVisibility, sunDirection, lagSensitivity_K, lagVisibility,
        lagDirection);
    const VkDeviceSize bytes =
        records.size() * sizeof(rendercore::ThermalSunResponseGpu);

    const bool reallocate =
        !thermalSunResponseBuffer || thermalSunResponseBuffer->GetSize() != bytes;
    if (reallocate) {
        if (thermalSunResponseBuffer) vkDeviceWaitIdle(device);
        thermalSunResponseBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
    }
    thermalSunResponseBuffer->Upload(records.data(), bytes);
    return reallocate;
}

/// Put a per-element tangent field and its step on binding 27.
///
/// One float of step at index 0 and one tangent per element after it. Empty
/// tangents mean nothing is being previewed, which is a single zero: the
/// shader's multiply by the step then costs nothing and needs no branch.
///
/// @return true when the buffer was reallocated, which is what tells the
///         caller the descriptor has to be rewritten.
bool ExternalRenderContext::Impl::UploadThermalTangent(const Vector<f32>& tangent,
                                                       const f32 step) {
    Vector<f32> records;
    records.reserve(tangent.size() + 1);
    records.push_back(tangent.empty() ? 0.0f : step);
    records.insert(records.end(), tangent.begin(), tangent.end());
    if (records.size() < 2) records.push_back(0.0f);

    const VkDeviceSize bytes = records.size() * sizeof(f32);
    const bool reallocate =
        !thermalParameterTangentBuffer || thermalParameterTangentBuffer->GetSize() != bytes;
    if (reallocate) {
        if (thermalParameterTangentBuffer) vkDeviceWaitIdle(device);
        thermalParameterTangentBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
    }
    thermalParameterTangentBuffer->Upload(records.data(), bytes);
    return reallocate;
}

void ExternalRenderContext::Impl::UploadLightingParams() {
    LightingParams effective = lightingParams;
    if (atmosphereActive) {
        effective.atmosphereTemperature_K =
            static_cast<f32>(atmosphereConfig.tGroundK);
    }
    effective.enableEnvironmentMap =
        (lightingParams.enableEnvironmentMap != 0 && hasCustomEnvMap) ? 1u : 0u;
    lightingParamsBuffer->Upload(&effective, sizeof(LightingParams));
}

// Lazily (re)bakes the NN atmosphere LUT when the bake key changed and
// uploads header + data. On bake failure (missing network files etc.) the
// atmosphere is disabled with a critical log -- no analytic fallback exists.
void ExternalRenderContext::Impl::UpdateAtmosphereNN() {
    constexpr uint64_t kDisabledKey = 1;  // 0 = dirty, 1 = disabled uploaded

    auto uploadDisabled = [this]() {
        AtmosNNHeaderGPU disabledHeader{};
        atmosHeaderBuffer->Upload(&disabledHeader, sizeof(disabledHeader));
        atmosBakeKey = kDisabledKey;
        atmosphereActive = false;
        UploadLightingParams();  // Hand the fallback temperature back to the host
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
        atmosphereActive = true;
        UploadLightingParams();  // T_air must match what this LUT was baked at
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

    // Ordinary poses are uploaded and refitted in this very submission, so
    // prior traces finish before UPDATE and the new trace sees the new pose.
    // Camera capture retains its synchronous pose/stratum contract.
    if (m_impl->pendingTlasRefit) {
        if (m_impl->cameraConfig.enabled) {
            m_impl->FlushPendingTlasRefit();
        } else {
            m_impl->geometry.RecordPreparedTlasRefit(cmd);
            m_impl->pendingTlasRefit = false;
        }
    }

    // Collect GPU timings from frames the GPU has finished by now. Never
    // blocks: with the host's frames-in-flight the previous dispatch may
    // still be running, and its result is simply picked up next time.
    if (m_impl->perfLogger && m_impl->perfLogger->TryResolvePending()) {
        m_impl->lastSampleGpuMs = m_impl->perfLogger->GetLastFrameGpuMs();
    }

    // Update CLAHE min/max cache from previous frame's output image
    // This is safe here because QVulkanWindow ensures the previous frame's
    // GPU work is complete before calling startNextFrame/RenderFrame again
    // Only update every N frames to reduce readback overhead, but always update
    // on frame 1 (after first render) and whenever cache is invalid
    constexpr u32 minMaxUpdateInterval = 10; // Update every 10 frames
    if (m_impl->claheInitialized &&
        (m_impl->displayParams.enabled || m_impl->CameraClahePersistent()) &&
        m_impl->accumulatedSamples > 0 &&
        (!m_impl->hasCachedMinMax ||
         m_impl->accumulatedSamples == 1 ||  // Always update after first frame
         m_impl->frameIndex % minMaxUpdateInterval == 0)) {
        m_impl->ComputeImageMinMax(m_impl->cachedImageMin, m_impl->cachedImageMax);
        m_impl->hasCachedMinMax = true;
        //QL_LOG_DEBUG("CLAHE: Updated min/max cache: [{}, {}]",
        //             m_impl->cachedImageMin, m_impl->cachedImageMax);
    }

    // Record the target extent and bring the render extent in line with it and
    // the current scale. Everything below traces and post-processes at the
    // render extent; only the closing blit knows about the target's.
    m_impl->EnsureExtent(width, height);
    const u32 renderW = m_impl->width;
    const u32 renderH = m_impl->height;

    // Rebake the NN atmosphere LUT if the bake key changed (band, weather,
    // quantized altitude / sun geometry)
    m_impl->UpdateAtmosphereNN();

    // Update camera data with current state
    CameraData cameraData = m_impl->camera.GetCameraData();
    cameraData.wavelength_nm = m_impl->wavelength_nm;
    cameraData.spectral_mode = static_cast<u32>(m_impl->spectralMode);
    cameraData.debug_mode = static_cast<u32>(m_impl->debugMode);
    cameraData.debugParam = m_impl->debugParam;
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
        randomSeed,
        m_impl->sequenceSeed
    );

    if (m_impl->cameraConfig.enabled) {
        // The camera chain owns the frame: it traces at the physical sensor
        // extent and produces the display product. The visibility
        // accumulation in outputImage is not traced in this mode; hosts that
        // read CaptureScreenshot with the camera enabled get the last
        // non-camera frame, if any.
        if (auto stepped = m_impl->StepCameraAcquisition(cmd); !stepped) {
            QL_LOG_WARN("Camera scheduler: {}", stepped.error());
        }
    } else {
        // Execute ray tracing (writes to internal outputImage in GENERAL
        // layout). Timestamps bracket the trace alone -- not CLAHE or the
        // blit -- so what gets measured is the cost of one sample of *this
        // scene*. The camera trace is bracketed by the camera perf queries
        // inside RecordCameraMeasurement.
        if (m_impl->perfLogger) m_impl->perfLogger->BeginFrame(cmd);
        m_impl->pipeline->TraceRays(cmd, renderW, renderH);
        if (m_impl->perfLogger) m_impl->perfLogger->EndFrame(cmd);
        m_impl->rawPixelGeneration = m_impl->pixelImageGeneration;
        m_impl->rawPixelSamples = m_impl->accumulatedSamples + 1;
    }

    // Post-processing, then the blit. Both halves are shared with
    // PresentAccumulated, which does them without the trace above. They run
    // at the render extent, before the magnifying blit, so a reduced scale
    // makes them cheaper too -- and CLAHE never sees an upsampled image.
    if ((m_impl->displayParams.enabled || m_impl->CameraClahePersistent()) &&
        m_impl->claheInitialized && m_impl->displayImage &&
        m_impl->ClaheInputReady()) {
        const auto claheExtent = m_impl->ClaheInputExtent();
        m_impl->ExecuteCLAHE(cmd, claheExtent.width, claheExtent.height);
    }

    const auto blitSource = m_impl->CurrentDisplaySource();
    // Whichever pass wrote it last is what the barrier has to wait on.
    const VkPipelineStageFlags srcStage =
        (blitSource.image == m_impl->outputImage->GetImage())
            ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    m_impl->BlitToTarget(cmd, blitSource.image, targetImage, targetLayout,
                         blitSource.width, blitSource.height, width, height,
                         srcStage);

    m_impl->accumulatedSamples++;
    m_impl->frameIndex++;
}

ExternalRenderContext::Impl::DisplaySource
ExternalRenderContext::Impl::CurrentDisplaySource() const {
    // Priority: CLAHE (viewport override or the camera's persistent Clahe
    // tone) -> camera display -> raw accumulation. The camera product sits at
    // the physical sensor extent, which BlitToTarget magnifies to the target
    // like any other source.
    if (displayImage && claheInitialized && ClaheInputReady() &&
        (displayParams.enabled || CameraClahePersistent())) {
        const auto extent = ClaheInputExtent();
        return {displayImage->GetImage(), extent.width, extent.height};
    }
    if (const GpuImage* cameraDisplay = CameraDisplayImage()) {
        return {cameraDisplay->GetImage(), cameraConfig.optics.sensorWidthPx,
                cameraConfig.optics.sensorHeightPx};
    }
    return {outputImage->GetImage(), width, height};
}

void ExternalRenderContext::Impl::BlitToTarget(VkCommandBuffer cmd, VkImage source,
                                               VkImage targetImage,
                                               VkImageLayout targetLayout,
                                               u32 srcWidth, u32 srcHeight,
                                               u32 dstWidth, u32 dstHeight,
                                               VkPipelineStageFlags sourceStage) {
    // Step 1: source GENERAL -> TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier outputBarrier{};
    outputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.image = source;
    outputBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputBarrier.subresourceRange.baseMipLevel = 0;
    outputBarrier.subresourceRange.levelCount = 1;
    outputBarrier.subresourceRange.baseArrayLayer = 0;
    outputBarrier.subresourceRange.layerCount = 1;
    outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // Step 2: target -> TRANSFER_DST_OPTIMAL
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
    vkCmdPipelineBarrier(cmd, sourceStage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 2, barriers);

    // Step 3: blit, converting R32G32B32A32_SFLOAT to the target format and
    // magnifying when the render extent is below the target's.
    // vkCmdBlitImage clamps anything above 1.0.
    VkImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.mipLevel = 0;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0] = {0, 0, 0};
    blitRegion.srcOffsets[1] = {static_cast<i32>(srcWidth), static_cast<i32>(srcHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.mipLevel = 0;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0] = {0, 0, 0};
    blitRegion.dstOffsets[1] = {static_cast<i32>(dstWidth), static_cast<i32>(dstHeight), 1};

    // NEAREST when the extents match: a 1:1 blit resolves to a copy either
    // way, and this is the path every full-scale frame takes.
    const bool magnifying = (srcWidth != dstWidth) || (srcHeight != dstHeight);
    const VkFilter filter = (magnifying && linearBlitSupported)
                                ? VK_FILTER_LINEAR
                                : VK_FILTER_NEAREST;

    vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blitRegion, filter);

    // Step 4: source back to GENERAL for the next pass to write
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    // Step 5: target to PRESENT_SRC_KHR
    targetBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    targetBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    targetBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    targetBarrier.dstAccessMask = 0;

    barriers[0] = outputBarrier;
    barriers[1] = targetBarrier;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         sourceStage | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 2, barriers);
}

bool ExternalRenderContext::PresentAccumulated(
    VkCommandBuffer cmd,
    VkImage targetImage,
    VkImageLayout targetLayout,
    u32 width,
    u32 height) {

    if (!m_impl->isReady || !m_impl->outputImage) {
        return false;
    }
    // Nothing has been traced, so there is no accumulation to show. The caller
    // has to draw a real frame instead -- returning true here would present an
    // image whose contents are undefined.
    if (m_impl->accumulatedSamples == 0) {
        return false;
    }
    // A change of target extent invalidates the accumulation the caller is
    // asking to re-present, and reacting to one is RenderFrame's job. Same
    // answer: draw a real frame. Compared against the target extent, not the
    // render extent -- below a scale of 1.0 the two differ by design, and it
    // is the target the caller is talking about.
    if (width != m_impl->targetWidth || height != m_impl->targetHeight) {
        return false;
    }
    // A render scale set since the last trace has not taken effect yet --
    // reacting to one is also RenderFrame's job, and until it does the
    // accumulation is at the wrong extent. Without this a host whose loop has
    // stopped at its target sample count would re-present forever and never
    // pick up the new scale.
    if (!m_impl->ExtentMatchesScale()) {
        return false;
    }

    const auto source = m_impl->CurrentDisplaySource();
    // The post-processed images still hold the last frame's result -- nothing
    // has changed the accumulation since -- so the chains are not re-run.
    const VkPipelineStageFlags sourceStage =
        (source.image == m_impl->outputImage->GetImage())
            ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    m_impl->BlitToTarget(cmd, source.image, targetImage, targetLayout,
                         source.width, source.height, width, height,
                         sourceStage);
    // Deliberately no accumulatedSamples++: that is the whole point.
    return true;
}

bool ExternalRenderContext::ReprocessAccumulated(
    VkCommandBuffer cmd,
    VkImage targetImage,
    VkImageLayout targetLayout,
    u32 width,
    u32 height) {

    // The same three refusals as PresentAccumulated, for the same reasons.
    if (!m_impl->isReady || !m_impl->outputImage) {
        return false;
    }
    if (m_impl->accumulatedSamples == 0) {
        return false;
    }
    if (width != m_impl->targetWidth || height != m_impl->targetHeight) {
        return false;
    }
    // A render scale set since the last trace has not taken effect yet --
    // reacting to one is also RenderFrame's job, and until it does the
    // accumulation is at the wrong extent. Without this a host whose loop has
    // stopped at its target sample count would re-present forever and never
    // pick up the new scale.
    if (!m_impl->ExtentMatchesScale()) {
        return false;
    }

    // The min/max cache reads the CLAHE input image, which this path never
    // changes -- so a filled cache is still right, whatever display setting
    // prompted the reprocess. Only a cache that has never been filled needs
    // computing: CLAHE enabled for the first time after the trace stopped.
    if ((m_impl->displayParams.enabled || m_impl->CameraClahePersistent()) &&
        m_impl->claheInitialized && !m_impl->hasCachedMinMax &&
        m_impl->ClaheInputReady()) {
        m_impl->ComputeImageMinMax(m_impl->cachedImageMin, m_impl->cachedImageMax);
        m_impl->hasCachedMinMax = true;
    }

    // The post-processing half of RenderFrame, over the accumulation as it
    // stands. When the camera owns the frame, a display-setting change only
    // re-runs the display half of the camera chain (demosaic through HSV):
    // no trace, no statistics, no AE/AWB, no thermal-state noise; the AGC
    // window keeps riding in the statistics buffer.
    if (m_impl->cameraConfig.enabled && m_impl->cameraCaptureCompleted &&
        !m_impl->cameraCapturePending && !m_impl->cameraCaptureRecorded) {
        if (auto reprocessed = m_impl->RecordCameraDisplayReprocess(cmd);
            !reprocessed)
            QL_LOG_WARN("Camera display reprocess: {}", reprocessed.error());
    }

    if ((m_impl->displayParams.enabled || m_impl->CameraClahePersistent()) &&
        m_impl->claheInitialized && m_impl->displayImage &&
        m_impl->ClaheInputReady()) {
        const auto claheExtent = m_impl->ClaheInputExtent();
        m_impl->ExecuteCLAHE(cmd, claheExtent.width, claheExtent.height);
    }

    const auto source = m_impl->CurrentDisplaySource();
    const VkPipelineStageFlags sourceStage =
        (source.image == m_impl->outputImage->GetImage())
            ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    m_impl->BlitToTarget(cmd, source.image, targetImage, targetLayout,
                         source.width, source.height, width, height,
                         sourceStage);
    // Like PresentAccumulated: no accumulatedSamples++.
    return true;
}

Result<void, String> ExternalRenderContext::ReprocessCameraDisplay() {
    if (!m_impl->cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (!m_impl->cameraGpuPipeline || !m_impl->cameraCaptureCompleted)
        return Result<void, String>::Err(
            "camera display reprocess needs a completed camera acquisition");
    if (m_impl->cameraCapturePending || m_impl->cameraCaptureRecorded)
        return Result<void, String>::Err(
            "complete the queued camera acquisition before reprocessing");
    Result<void, String> status = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                    [&](VkCommandBuffer cmd) {
        status = m_impl->RecordCameraDisplayReprocess(cmd);
    });
    return status;
}

void ExternalRenderContext::BlitDepthTo(VkCommandBuffer cmd, VkImage targetImage,
                                        VkImageLayout targetCurrentLayout,
                                        u32 width, u32 height) {
    if (!m_impl->depthAovImage) {
        return;
    }

    VkImageSubresourceRange colorRange{};
    colorRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorRange.baseMipLevel = 0;
    colorRange.levelCount = 1;
    colorRange.baseArrayLayer = 0;
    colorRange.layerCount = 1;

    // Source: depth AOV GENERAL (raygen storage write) -> TRANSFER_SRC
    VkImageMemoryBarrier srcBarrier{};
    srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.image = m_impl->depthAovImage->GetImage();
    srcBarrier.subresourceRange = colorRange;
    srcBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // Target: caller's layout -> TRANSFER_DST
    VkImageMemoryBarrier dstBarrier{};
    dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    dstBarrier.oldLayout = targetCurrentLayout;
    dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.image = targetImage;
    dstBarrier.subresourceRange = colorRange;
    dstBarrier.srcAccessMask = 0;  // previous reads need no flush
    dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkImageMemoryBarrier preBarriers[2] = {srcBarrier, dstBarrier};
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 2, preBarriers);

    // NEAREST on purpose: depth values must never be mixed across silhouettes.
    // That holds all the more when the render extent is below the target's and
    // this is a magnification -- an interpolated depth there would be a
    // surface that is not in the scene.
    VkImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.mipLevel = 0;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0] = {0, 0, 0};
    blitRegion.srcOffsets[1] = {static_cast<i32>(m_impl->width),
                                static_cast<i32>(m_impl->height), 1};
    blitRegion.dstSubresource = blitRegion.srcSubresource;
    blitRegion.dstOffsets[0] = {0, 0, 0};
    blitRegion.dstOffsets[1] = {static_cast<i32>(width), static_cast<i32>(height), 1};

    vkCmdBlitImage(cmd,
                   m_impl->depthAovImage->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blitRegion, VK_FILTER_NEAREST);

    // Source back to GENERAL for next frame's raygen write
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    // Target ready for sampling in the caller's overlay fragment shader
    dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dstBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dstBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkImageMemoryBarrier postBarriers[2] = {srcBarrier, dstBarrier};
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 2, postBarriers);
}

void ExternalRenderContext::Resize(u32 width, u32 height) {
    m_impl->EnsureExtent(width, height);
}

void ExternalRenderContext::SetRenderScale(f32 scale) {
    const f32 clamped = std::clamp(scale, 0.25f, 1.0f);
    if (clamped == m_impl->renderScale) {
        return;
    }
    m_impl->renderScale = clamped;
    // Deliberately not resizing here. The extent changes on the next
    // RenderFrame, where the host is between frames and a vkDeviceWaitIdle is
    // already what a resize costs; doing it from an input handler would wait
    // on a frame the host is still recording.
}

f32 ExternalRenderContext::GetRenderScale() const {
    return m_impl->renderScale;
}

u32 ExternalRenderContext::GetRenderWidth() const {
    return m_impl->width;
}

u32 ExternalRenderContext::GetRenderHeight() const {
    return m_impl->height;
}

void ExternalRenderContext::Impl::EnsureExtent(u32 targetW, u32 targetH) {
    targetWidth = targetW;
    targetHeight = targetH;

    const u32 renderW = ScaledDim(targetW);
    const u32 renderH = ScaledDim(targetH);
    if (renderW != width || renderH != height) {
        ApplyInternalExtent(renderW, renderH);
    }
}

void ExternalRenderContext::Impl::ApplyInternalExtent(u32 renderW, u32 renderH) {
    QL_LOG_INFO("ExternalRenderContext render extent {}x{} -> {}x{} "
                "(target {}x{}, scale {:.3f})",
                width, height, renderW, renderH,
                targetWidth, targetHeight, renderScale);

    width = renderW;
    height = renderH;

    // Wait for GPU
    vkDeviceWaitIdle(device);

    outputImage =
        rendercore::CreateRenderTarget(*contextAdapter, width, height);
    depthAovImage = rendercore::CreateRenderTarget(
        *contextAdapter, width, height, VK_FORMAT_R32_SFLOAT);

    // Re-bind output and depth AOV images
    if (pipeline) {
        pipeline->BindOutputImage(*outputImage);
        pipeline->BindDepthImage(*depthAovImage);
    }

    // Recreate CLAHE display image if initialized. Waits on the device, which
    // ApplyInternalExtent already did above; the extent follows the CLAHE
    // input (physical sensor array when the camera is enabled).
    if (claheInitialized && displayImage) {
        RecreateClaheDisplayImage();
    }

    // Aspect comes from the target extent, not the render extent. Rounding
    // makes the two disagree by a fraction of a pixel, and taking it from the
    // render extent would nudge the framing every time the scale changed --
    // which is once at the start of every camera drag.
    if (targetHeight > 0U) {
        camera.SetAspectRatio(static_cast<f32>(targetWidth) /
                              static_cast<f32>(targetHeight));
    }

    // Nothing survives a change of extent. Same two steps as
    // ExternalRenderContext::ResetAccumulation.
    accumulatedSamples = 0;
    ReseedRng();
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
        // UpdateGpuResources replaces descriptor-backed buffers. A viewport
        // frame recorded before this setter may still read the old allocation.
        if (m_impl->device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(m_impl->device);
        // The emitter CDF contains positions, areas and material emission,
        // none of which depends on the wavelength used to sample IR curves.
        m_impl->UpdateGpuResources(false);
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

void ExternalRenderContext::SetDebugParameter(const u32 value) {
    if (m_impl->debugParam != value) {
        m_impl->debugParam = value;
        ResetAccumulation();
    }
}

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
    // The emissive fields describe the scene, not the caller's lighting
    // intent -- a host that round-trips GetLightingParams through its own
    // struct would otherwise zero the emitter count and silently turn off
    // light sampling. They are the renderer's to set; preserve them.
    const u32 emissiveCount = m_impl->lightingParams.emissiveTriangleCount;
    const f32 emissivePower = m_impl->lightingParams.emissiveTotalPower;

    m_impl->lightingParams = params;
    m_impl->lightingParams.emissiveTriangleCount = emissiveCount;
    m_impl->lightingParams.emissiveTotalPower = emissivePower;

    m_impl->UploadLightingParams();
    if (m_impl->thermalPreview) {
        m_impl->thermalPreview->SetFallbackSunDirection(m_impl->lightingParams.sunDirection);
    }
    m_impl->cameraResourcesDirty = true; // NN camera LUT tracks sun geometry.
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;
    ResetAccumulation();
}

void ExternalRenderContext::SetSunDirection(const glm::vec3& direction) {
    m_impl->lightingParams.sunDirection = glm::normalize(direction);
    m_impl->UploadLightingParams();
    if (m_impl->thermalPreview) {
        m_impl->thermalPreview->SetFallbackSunDirection(m_impl->lightingParams.sunDirection);
    }
    m_impl->cameraResourcesDirty = true;
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;
    ResetAccumulation();
}

void ExternalRenderContext::SetSunRadiance(const glm::vec3& radiance) {
    m_impl->lightingParams.sunRadiance_rgb = radiance;
    m_impl->lightingParams.sunRadiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0f;
    m_impl->UploadLightingParams();
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;
    ResetAccumulation();
}

void ExternalRenderContext::SetSkyRadiance(const glm::vec3& radiance) {
    m_impl->lightingParams.skyRadiance_rgb = radiance;
    m_impl->lightingParams.skyRadiance_spectral = (radiance.r + radiance.g + radiance.b) / 3.0f;
    m_impl->UploadLightingParams();
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;
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
// Scene Editing
// ============================================================================

u32 ExternalRenderContext::AddMesh(const Mesh& mesh, const glm::mat4& transform) {
    // TODO: Implement in Phase 2
    (void)mesh;
    (void)transform;
    QL_LOG_WARN("ExternalRenderContext::AddMesh not implemented yet");
    return 0;
}

Result<u32, String> ExternalRenderContext::DuplicateNode(u32 sourceNodeIndex,
                                                         const String& newName) {
    if (!m_impl->scene) {
        return Result<u32, String>::Err("DuplicateNode: no scene loaded");
    }
    auto& nodes = m_impl->scene->nodes;
    if (sourceNodeIndex >= nodes.size()) {
        return Result<u32, String>::Err("DuplicateNode: invalid node index " +
                                        std::to_string(sourceNodeIndex));
    }

    SceneNode copy = nodes[sourceNodeIndex];
    copy.name = newName;
    copy.active = true;
    nodes.push_back(std::move(copy));

    const u32 newIndex = static_cast<u32>(nodes.size() - 1);
    QL_LOG_DEBUG("DuplicateNode: node {} -> {} ('{}')", sourceNodeIndex, newIndex, newName);
    return newIndex;
}

bool ExternalRenderContext::RemoveNode(u32 nodeIndex) {
    if (!m_impl->scene || nodeIndex >= m_impl->scene->nodes.size()) {
        QL_LOG_WARN("RemoveNode: invalid node index {}", nodeIndex);
        return false;
    }
    auto& node = m_impl->scene->nodes[nodeIndex];
    if (!node.active) {
        return false;  // already tombstoned
    }
    node.active = false;
    QL_LOG_DEBUG("RemoveNode: node {} ('{}') tombstoned", nodeIndex, node.name);
    return true;
}

bool ExternalRenderContext::RestoreNode(u32 nodeIndex) {
    if (!m_impl->scene || nodeIndex >= m_impl->scene->nodes.size()) {
        QL_LOG_WARN("RestoreNode: invalid node index {}", nodeIndex);
        return false;
    }
    auto& node = m_impl->scene->nodes[nodeIndex];
    if (node.active) {
        return false;  // nothing to restore
    }
    node.active = true;
    QL_LOG_DEBUG("RestoreNode: node {} ('{}') reactivated", nodeIndex, node.name);
    return true;
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

    const auto& node = m_impl->scene->nodes[nodeIndex];
    if (node.transform != transform &&
        rendercore::NodeHasSampledEmission(*m_impl->scene, node))
        m_impl->emissiveTransformDirty = true;

    // Invalidate delayed readings as soon as the host changes the pose.
    ++m_impl->pixelImageGeneration;
    // Update the node's transform in the scene
    m_impl->scene->nodes[nodeIndex].transform = transform;

    // If the clock moves this node, the edit is about its REST pose, not about
    // this instant: the trajectory will overwrite `transform` on the next tick
    // either way. Solving the composition for R_node here is what makes a
    // gizmo drag at t = 30 still be there at t = 0.
    m_impl->timeline.SetWorldPose(nodeIndex, transform);

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

    // Whether this material's geometry may skip the any-hit shader is baked
    // into the acceleration structure, so an edit that crosses that line needs
    // the BLAS rebuilt -- otherwise turning a surface into a cut-out updates the
    // material and leaves the geometry solid. Compared before the write, and
    // acted on after it, because the classifier reads the scene.
    const bool wasOpaque =
        rendercore::IsOpaqueForRayTracing(*m_impl->scene, materialIndex);

    // 1. Update CPU-side scene data. A newly supplied IR curve must not keep
    // reading the old resolved curve index; full config application is what
    // uploads new curve data, so this immediate edit uses scalar fallback.
    const Material& previous = m_impl->scene->materials[materialIndex];
    const bool emissionChanged = rendercore::SampledEmissionChanged(previous, material);
    if (emissionChanged) {
        // Both the material and emitter buffers may be read by submitted frames.
        // Their contents change together, after those readers have completed.
        vkDeviceWaitIdle(m_impl->device);
    }
    auto indices = rendercore::IndicesFromMaterial(material);
    if (m_impl->materialGpuIndices.size() == m_impl->scene->materials.size()) {
        const auto& oldSlots = m_impl->materialGpuIndices[materialIndex];
        indices.irEmissivityCurve =
            material.irEmissivityCurve == previous.irEmissivityCurve
                ? oldSlots.irEmissivityCurve : -1;
        indices.irTransmittanceCurve =
            material.irTransmittanceCurve == previous.irTransmittanceCurve
                ? oldSlots.irTransmittanceCurve : -1;
        const auto validIndex = [&](i32 curve) {
            return curve < 0 ||
                static_cast<usize>(curve) < m_impl->spectralCurveEntries.size();
        };
        if (!validIndex(indices.irEmissivityCurve)) indices.irEmissivityCurve = -1;
        if (!validIndex(indices.irTransmittanceCurve)) indices.irTransmittanceCurve = -1;
        m_impl->materialGpuIndices[materialIndex] = indices;
    }
    m_impl->scene->materials[materialIndex] = material;

    // 2. Convert to GPU format
    MaterialDataCPU cpuMat = rendercore::ConvertMaterial(
        material, m_impl->wavelength_nm, indices);

    // 3. Partial upload at offset
    VkDeviceSize offset = materialIndex * sizeof(MaterialDataCPU);
    m_impl->materialBuffer->Upload(&cpuMat, sizeof(MaterialDataCPU), offset);

    // 4. Rebuild the geometry only when the opacity classification actually
    //    moved. Doing it on every edit would put an acceleration-structure
    //    rebuild behind a roughness slider.
    if (m_impl->geometry.IsValid() &&
        rendercore::IsOpaqueForRayTracing(*m_impl->scene, materialIndex) != wasOpaque) {
        // A frame the host submitted may still be tracing the structures being
        // replaced, as RebuildAccelerationStructure has to assume too.
        vkDeviceWaitIdle(m_impl->device);
        if (m_impl->geometry.RefreshMaterialOpacity(*m_impl->contextAdapter,
                                                    *m_impl->scene) &&
            m_impl->pipeline) {
            m_impl->pipeline->BindAccelerationStructure(m_impl->geometry.Tlas().GetHandle());
            if (m_impl->geometry.InstanceCount() > 0) {
                m_impl->pipeline->BindInstanceGeometryBuffer(m_impl->geometry.InstanceInfo());
            }
        }
    }

    if (emissionChanged) m_impl->RebuildEmissiveGeometry();

    // 5. Reset accumulation (visual feedback)
    ResetAccumulation();

    if (m_impl->thermalPreview) m_impl->thermalPreview->InvalidateMaterialEmissivity();

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

Result<Vector<String>, String> ExternalRenderContext::SetMaterialEmissionSpectrum(
    u32 materialIndex, const String& sourceOrEmpty, const String& scale,
    const String& baseDir) {
    using EmissionResult = Result<Vector<String>, String>;

    if (!m_impl->scene) {
        return EmissionResult::Err("SetMaterialEmissionSpectrum: no scene loaded");
    }
    if (materialIndex >= m_impl->scene->materials.size()) {
        return EmissionResult::Err("SetMaterialEmissionSpectrum: invalid material index " +
                                   std::to_string(materialIndex));
    }

    Material material = m_impl->scene->materials[materialIndex];

    if (sourceOrEmpty.empty()) {
        // Unbinding leaves emissiveFactor where it is. It was derived from the
        // curve, and the triple the author originally typed is gone -- but the
        // colour on screen is the one they have been looking at, and silently
        // changing a light's brightness because a dropdown moved to "None" is
        // worse than keeping a number whose provenance changed.
        material.emissiveRadianceCurveIndex = -1;
        material.emissiveCurveSource.clear();
        UpdateMaterial(materialIndex, material);
        return EmissionResult(Vector<String>{});
    }

    // The band this viewport is rendering, so the curve is resampled onto the
    // window it will actually be sampled in -- the same rule the config path
    // follows, and the reason a cross-band lamp does not arrive as 64 samples
    // spread from the ultraviolet to the thermal.
    rendercore::EmissionBindingRequest request;
    request.source = sourceOrEmpty;
    request.scale = scale;
    request.authoredEmissive = material.emissiveFactor;
    if (const auto band = GetFusedBandInfo(m_impl->spectralMode)) {
        request.bandMinNm = band->lambdaMinNm;
        request.bandMaxNm = band->lambdaMaxNm;
    }

    auto bound = rendercore::ResolveEmissionSpectrum(request, baseDir);
    if (!bound) {
        return EmissionResult::Err(bound.error());
    }
    const auto& resolved = bound.value();

    // Appended like AddSpectralCurve, but from a SpectralCurveGPU that is
    // already resampled and levelled -- going back through the CPU curve would
    // discard the band clipping and the band averaging that make it right.
    const auto index = static_cast<i32>(m_impl->spectralCurveEntries.size());
    m_impl->spectralCurveEntries.push_back(resolved.curve);

    const size_t bytes = m_impl->spectralCurveEntries.size() * sizeof(SpectralCurveGPU);
    // The new emitter record refers to this curve. Complete prior descriptor
    // readers before replacing the curve buffer, then publish both inputs.
    vkDeviceWaitIdle(m_impl->device);
    m_impl->spectralCurvesBuffer = std::make_unique<GpuBuffer>(
        m_impl->contextAdapter->GetAllocator(), bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_impl->spectralCurvesBuffer->Upload(m_impl->spectralCurveEntries.data(), bytes);
    if (m_impl->pipeline) {
        m_impl->pipeline->BindSpectralCurvesBuffer(m_impl->spectralCurvesBuffer.get());
    }

    material.emissiveRadianceCurveIndex = index;
    material.emissiveCurveSource = sourceOrEmpty;
    if (resolved.rewriteRgb) {
        material.emissiveFactor = resolved.renderedRgb;
    }
    UpdateMaterial(materialIndex, material);

    QL_LOG_INFO("SetMaterialEmissionSpectrum: '{}' on material {} ('{}'), curve index {}, "
                "colour [{:.4g}, {:.4g}, {:.4g}]",
                sourceOrEmpty, materialIndex, material.name, index,
                material.emissiveFactor.r, material.emissiveFactor.g,
                material.emissiveFactor.b);
    for (const auto& warning : resolved.warnings) {
        QL_LOG_WARN("  {}", warning);
    }
    return EmissionResult(Vector<String>(resolved.warnings));
}

i32 ExternalRenderContext::AddSpectralCurve(const SpectralCurve& curve) {
    if (!curve.IsValid()) {
        QL_LOG_WARN("AddSpectralCurve: Invalid curve data");
        return -1;
    }

    const i32 index = static_cast<i32>(m_impl->spectralCurveEntries.size());
    m_impl->spectralCurveEntries.push_back(SpectralCurveGPU::FromCPU(curve));

    const size_t bytes =
        m_impl->spectralCurveEntries.size() * sizeof(SpectralCurveGPU);
    m_impl->spectralCurvesBuffer = std::make_unique<GpuBuffer>(
        m_impl->contextAdapter->GetAllocator(), bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_impl->spectralCurvesBuffer->Upload(m_impl->spectralCurveEntries.data(), bytes);

    if (m_impl->pipeline) {
        m_impl->pipeline->BindSpectralCurvesBuffer(m_impl->spectralCurvesBuffer.get());
    }

    QL_LOG_INFO("AddSpectralCurve: Added curve at index {} ({} curves total)",
                index, m_impl->spectralCurveEntries.size());
    return index;
}

Result<i32, String> ExternalRenderContext::BuildEndmemberWeightTexture(
    const u32 materialIndex, const Vector<glm::vec3>& endmemberColors) {
    using WeightResult = Result<i32, String>;

    if (!m_impl->scene || materialIndex >= m_impl->scene->materials.size()) {
        return WeightResult::Err("BuildEndmemberWeightTexture: material index out of range");
    }
    if (endmemberColors.empty() ||
        endmemberColors.size() > static_cast<usize>(Material::MAX_ENDMEMBERS)) {
        return WeightResult::Err("BuildEndmemberWeightTexture: need 1 to " +
                                 std::to_string(Material::MAX_ENDMEMBERS) + " endmember colours");
    }
    if (!m_impl->textureManager) {
        return WeightResult::Err("BuildEndmemberWeightTexture: no textures uploaded yet");
    }

    const Material& material = m_impl->scene->materials[materialIndex];
    if (material.baseColorTextureIndex < 0 ||
        material.baseColorTextureIndex >= static_cast<i32>(m_impl->scene->textures.size())) {
        return WeightResult::Err("Material '" + material.name +
                                 "' has no base-colour texture to unmix");
    }

    const Texture& source =
        m_impl->scene->textures[static_cast<usize>(material.baseColorTextureIndex)];
    if (source.pixels.empty()) {
        // Only the base colours of materials that were curve-bound when the
        // scene loaded are kept on the CPU. Binding a curve to a material that
        // had none lands here, and the honest answer is that the pixels are
        // gone rather than a weight map fitted to nothing.
        return WeightResult::Err("Base-colour pixels for '" + material.name +
                                 "' were released after upload; reload the scene to unmix it");
    }
    if (source.channels != 4) {
        return WeightResult::Err("Base-colour texture for '" + material.name +
                                 "' is not RGBA");
    }

    const u64 texelCount = static_cast<u64>(source.width) * source.height;

    Texture weights;
    weights.name = "__unmix_" + material.name;
    weights.width = source.width;
    weights.height = source.height;
    weights.channels = 4;
    weights.isSRGB = false;
    weights.skipBlockCompression = true;
    weights.retainCpuPixels = true;  // a further reassignment rebuilds from it
    weights.pixels.resize(static_cast<usize>(texelCount) * 4);

    rendercore::UnmixTexels(source.pixels.data(), texelCount, source.isSRGB,
                            glm::vec3(material.baseColorFactor), endmemberColors.data(),
                            static_cast<i32>(endmemberColors.size()), weights.pixels.data());

    // The texture array is what the shader indexes, so nothing may be reading
    // it while the descriptor is rewritten.
    vkDeviceWaitIdle(m_impl->contextAdapter->GetDevice());

    const i32 index = m_impl->textureManager->AppendTexture(weights);
    if (index < 0) {
        return WeightResult::Err("Failed to upload the weight texture for '" + material.name + "'");
    }

    // Kept on the scene too: a later AdoptScene or full rebuild re-uploads
    // from here, and a weight map that existed only on the device would
    // silently disappear.
    m_impl->scene->textures.push_back(std::move(weights));

    if (m_impl->pipeline) {
        m_impl->pipeline->BindTextures(m_impl->textureManager->GetImageViews(),
                                       m_impl->textureManager->GetSamplers());
    }

    QL_LOG_INFO("BuildEndmemberWeightTexture: material '{}' unmixed into {} endmember(s), "
                "texture index {}",
                material.name, endmemberColors.size(), index);
    return index;
}

void ExternalRenderContext::SetSolarSpectralLUT(const SpectralCurve& sunIrradiance,
                                                const SpectralCurve& skyIrradiance) {
    if (!sunIrradiance.IsValid() && !skyIrradiance.IsValid()) {
        QL_LOG_WARN("SetSolarSpectralLUT: Neither curve is usable, leaving the LUT as it was");
        return;
    }

    const SolarSpectralLUT lut = SolarSpectralLUT::FromCPU(sunIrradiance, skyIrradiance);
    m_impl->cameraSolarSources = std::make_pair(sunIrradiance, skyIrradiance);
    // Replacing a descriptor-backed buffer while an earlier QVulkanWindow
    // frame may still read it is invalid; solar edits are infrequent.
    if (m_impl->solarLutBuffer && m_impl->device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(m_impl->device);

    m_impl->solarLutBuffer = std::make_unique<GpuBuffer>(
        m_impl->contextAdapter->GetAllocator(), sizeof(SolarSpectralLUT),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_impl->solarLutBuffer->Upload(&lut, sizeof(SolarSpectralLUT));

    if (m_impl->pipeline) {
        m_impl->pipeline->BindSolarSpectralLUT(m_impl->solarLutBuffer.get());
    }
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;
    ResetAccumulation();

    QL_LOG_INFO("SetSolarSpectralLUT: sun {} samples, sky {} samples",
                lut.sunIrradiance.numSamples, lut.skyIrradiance.numSamples);
}

void ExternalRenderContext::SetCameraProjection(CameraProjection projection, f32 orthoHeight) {
    m_impl->camera.SetProjection(projection == CameraProjection::Orthographic
                                     ? Camera::Projection::Orthographic
                                     : Camera::Projection::Perspective);
    if (orthoHeight > 0.0f) {
        m_impl->camera.SetOrthoHeight(orthoHeight);
    }
    ResetAccumulation();
}

Result<void, String> ExternalRenderContext::SetSolarSpectralLUTFromSpec(
    const SolarLutSpec& spec, const String& baseDir) {
    // The same function ResolveRenderConfig calls for [lighting] solar_lut*.
    // Reading the spec here instead would be the second reading this facade
    // exists to prevent.
    rendercore::SolarLutRequest request;
    request.pathOrEqualEnergy = spec.pathOrEqualEnergy;
    request.directColumn = spec.directColumn;
    request.diffuseColumn = spec.diffuseColumn;
    request.diffuseIsGlobal = spec.diffuseIsGlobal;
    request.normaliseUnitLuminance = spec.normaliseUnitLuminance;

    auto resolved = rendercore::ResolveSolarLut(request, baseDir, m_impl->spectralMode);
    if (!resolved.has_value()) {
        return Result<void, String>::Err(resolved.error());
    }
    auto& lut = resolved.value();
    for (const auto& warning : lut.warnings) {
        QL_LOG_WARN("{}", warning);
    }

    SetSolarSpectralLUT(lut.sun, lut.sky);

    // The colour half, so the non-spectral paths describe the same sun as the
    // spectral ones. ApplyConfig does this through ResolvedRenderConfig; here
    // there is no config to route it through.
    m_impl->lightingParams.sunRadiance_rgb = lut.sunRadianceRgb;
    m_impl->lightingParams.skyRadiance_rgb = lut.skyRadianceRgb;
    m_impl->lightingParams.sunRadiance_spectral = lut.sunRadianceSpectral;
    m_impl->lightingParams.skyRadiance_spectral = lut.skyRadianceSpectral;
    m_impl->UploadLightingParams();

    ResetAccumulation();
    return Result<void, String>();
}

// ============================================================================
// Thermal Solve
// ============================================================================

void ExternalRenderContext::SetThermalSolveParams(const ThermalSolveParams& params) {
    if (m_impl->thermalPreview) m_impl->thermalPreview->SetParams(params);
}

void ExternalRenderContext::SetThermalMaterial(const String& materialName,
                                               const ThermalMaterialParams& params) {
    if (m_impl->thermalPreview) m_impl->thermalPreview->SetMaterial(materialName, params);
}

void ExternalRenderContext::ClearThermalMaterials() {
    if (m_impl->thermalPreview) m_impl->thermalPreview->ClearMaterials();
}

Result<u32, String> ExternalRenderContext::ThermalElementAt(const PickResult& pick) const {
    using ElementResult = Result<u32, String>;
    if (!m_impl->thermalPreview) {
        return ElementResult::Err("this context has no thermal solve");
    }
    if (!pick.hit) {
        return ElementResult::Err("that ray reached the sky");
    }
    u32 element = 0;
    if (!m_impl->thermalPreview->ElementFor(pick.instanceIndex, pick.primitiveIndex, element)) {
        return ElementResult::Err(
            "that surface is not in the thermal solve -- its material names no "
            "conductivity, so it keeps whatever temperature it was given");
    }
    return ElementResult(element);
}

Result<Vector<f32>, String> ExternalRenderContext::GetThermalParameterSensitivity(
    const ThermalSensitivityParameter parameter) const {
    using FieldResult = Result<Vector<f32>, String>;
    if (!m_impl->thermalPreview) {
        return FieldResult::Err("this context has no thermal solve");
    }
    auto field = m_impl->thermalPreview->ParameterSensitivityField(parameter);
    if (field.empty()) {
        return FieldResult::Err(
            "this solve does not carry a derivative with respect to that parameter -- "
            "ask for it in ThermalSolveParams::parameterSensitivities, which rebuilds "
            "the trajectory");
    }
    return FieldResult(std::move(field));
}

Result<void, String> ExternalRenderContext::SetThermalWhatIf(
    const ThermalSensitivityParameter parameter, const f64 step) {
    using WhatIfResult = Result<void, String>;
    if (!m_impl->thermalPreview) {
        return WhatIfResult::Err("this context has no thermal solve");
    }

    m_impl->whatIfParameter = parameter;
    m_impl->whatIfStep = static_cast<f32>(step);

    Vector<f32> tangent;
    if (step != 0.0) {
        tangent = m_impl->thermalPreview->ParameterSensitivityField(parameter);
        if (tangent.empty()) {
            m_impl->whatIfStep = 0.0f;
            m_impl->UploadThermalTangent({}, 0.0f);
            if (m_impl->pipeline) {
                m_impl->pipeline->BindThermalTangentBuffer(
                    *m_impl->thermalParameterTangentBuffer);
            }
            return WhatIfResult::Err(
                "this solve does not carry a derivative with respect to that "
                "parameter -- ask for it in ThermalSolveParams::parameterSensitivities, "
                "which rebuilds the trajectory");
        }
    }

    if (m_impl->UploadThermalTangent(tangent, m_impl->whatIfStep) && m_impl->pipeline) {
        m_impl->pipeline->BindThermalTangentBuffer(*m_impl->thermalParameterTangentBuffer);
    }
    ResetAccumulation();
    return WhatIfResult();
}

Result<ThermalElementTrajectory, String> ExternalRenderContext::GetElementTrajectory(
    const u32 element, const f64 fromHour, const f64 toHour, const u32 samples) {
    if (!m_impl->thermalPreview) {
        return Result<ThermalElementTrajectory, String>::Err("this context has no thermal solve");
    }
    return m_impl->thermalPreview->ElementTrajectory(element, fromHour, toHour, samples);
}

Result<String, String> ExternalRenderContext::DumpThermalElements(const String& pathOrEmpty) {
    if (!m_impl->thermalPreview) {
        return Result<String, String>::Err("this context has no thermal solve");
    }
    return m_impl->thermalPreview->DumpElements(pathOrEmpty);
}

void ExternalRenderContext::SetThermalSolveEnabled(const bool enabled) {
    if (!m_impl->thermalPreview) return;
    m_impl->thermalEnabled = enabled;
    m_impl->thermalPreview->SetEnabled(enabled);

    if (!enabled && m_impl->thermalTemperatureBuffer) {
        try {
            vkDeviceWaitIdle(m_impl->device);
            const f32 zero = 0.0f;
            m_impl->thermalTemperatureBuffer = std::make_unique<GpuBuffer>(
                m_impl->contextAdapter->GetAllocator(), sizeof(f32),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            m_impl->thermalTemperatureBuffer->Upload(&zero, sizeof(zero));
            m_impl->UploadThermalSunResponse({}, {}, glm::vec3(0.0f));
            m_impl->UploadThermalTangent({}, 0.0f);
            m_impl->whatIfStep = 0.0f;
            if (m_impl->pipeline) {
                m_impl->pipeline->BindThermalTemperatureBuffer(*m_impl->thermalTemperatureBuffer);
                m_impl->pipeline->BindThermalSunResponseBuffer(*m_impl->thermalSunResponseBuffer);
                m_impl->pipeline->BindThermalTangentBuffer(*m_impl->thermalParameterTangentBuffer);
            }

            m_impl->geometry.SetThermalElementBases({});
            ResetAccumulation();
        } catch (const std::exception& ex) {
            QL_LOG_WARN("SetThermalSolveEnabled: failed to restore dummy buffer: {}", ex.what());
        }
    }
}

Result<void, String> ExternalRenderContext::SetThermalTime(const f64 time_h) {
    const auto fail = [&](String reason) {
        m_impl->thermalLastError = reason;
        return Result<void, String>::Err(std::move(reason));
    };

    if (!m_impl->thermalPreview || !m_impl->scene || !m_impl->geometry.IsValid()) {
        return fail("no scene loaded");
    }

    const VkAccelerationStructureKHR tlas = m_impl->geometry.Tlas().GetHandle();
    rendercore::ThermalPreview::SolveResult result;
    try {
        // The precompute consumes the current TLAS, and the field upload
        // below also needs old frame readers completed before host writes.
        m_impl->FlushPendingTlasRefit();
        result = m_impl->thermalPreview->SolveAt(time_h, *m_impl->scene, tlas);
    } catch (const std::exception& ex) {
        return fail(String("thermal solve failed: ") + ex.what());
    }
    if (!result.error.empty()) {
        return fail(result.error);
    }
    m_impl->thermalLastError.clear();

    if (result.elementCountChanged || result.surfaceTemperature_K.size() * sizeof(f32) !=
            (m_impl->thermalTemperatureBuffer ? m_impl->thermalTemperatureBuffer->GetSize() : 0)) {
        vkDeviceWaitIdle(m_impl->device);
        m_impl->thermalTemperatureBuffer = std::make_unique<GpuBuffer>(
            m_impl->contextAdapter->GetAllocator(),
            result.surfaceTemperature_K.size() * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        m_impl->thermalTemperatureBuffer->Upload(
            result.surfaceTemperature_K.data(),
            result.surfaceTemperature_K.size() * sizeof(f32));
        if (m_impl->pipeline) {
            m_impl->pipeline->BindThermalTemperatureBuffer(*m_impl->thermalTemperatureBuffer);
        }
        m_impl->geometry.SetThermalElementBases(result.instanceElementBase);
    } else {
        m_impl->thermalTemperatureBuffer->Upload(
            result.surfaceTemperature_K.data(),
            result.surfaceTemperature_K.size() * sizeof(f32));
    }

    // Refreshed on every scrub, not only when the element count moves: dT/dv
    // and v_element both belong to the instant being rendered, and a stale
    // pair would correct this frame's shadow with the last hour's response.
    // Reallocating is what needs the wait and the rebind, and that only
    // happens when the element count changes -- the helper says which it did.
    if (m_impl->UploadThermalSunResponse(result.sunSensitivity_K, result.sunVisibility,
                                         result.sunDirection, result.lagSensitivity_K,
                                         result.lagVisibility, result.lagDirection) &&
        m_impl->pipeline) {
        m_impl->pipeline->BindThermalSunResponseBuffer(*m_impl->thermalSunResponseBuffer);
    }

    // The what-if tangent is a field of the hour exactly as the temperatures
    // are, so a scrub has to move it too -- otherwise the preview would show
    // noon's derivative over midnight's field.
    if (m_impl->whatIfStep != 0.0f) {
        const auto tangent =
            m_impl->thermalPreview->ParameterSensitivityField(m_impl->whatIfParameter);
        if (m_impl->UploadThermalTangent(tangent, m_impl->whatIfStep) && m_impl->pipeline) {
            m_impl->pipeline->BindThermalTangentBuffer(*m_impl->thermalParameterTangentBuffer);
        }
    }

    ResetAccumulation();
    return Result<void, String>::Ok();
}

// ============================================================================
// The timeline
// ============================================================================

void ExternalRenderContext::Impl::FlushPendingTlasRefit() {
    if (!pendingTlasRefit) return;
    CommandHelper::ExecuteImmediate(*contextAdapter, [this](VkCommandBuffer cmd) {
        geometry.RecordPreparedTlasRefit(cmd);
    });
    pendingTlasRefit = false;
}

Vector<u32> ExternalRenderContext::Impl::ApplyTimelinePose(const f64 t_s, bool deferRefit) {
    if (!scene) return {};

    Vector<u32> moved = timeline.Apply(*scene, t_s);
    if (moved.empty()) {
        if (!deferRefit) FlushPendingTlasRefit();
        return moved;
    }

    if (geometry.IsValid()) {
        // Same handle, updated in place -- no rebind, no device idle. The
        // refit's own barriers order it against tracing already in flight.
        const bool refitted = deferRefit
            ? geometry.PrepareTlasRefit(*scene)
            : geometry.RefitTlas(*contextAdapter, *scene);
        pendingTlasRefit = deferRefit && refitted;
        if (!refitted) {
            // A refit is refused only when the instance count moved, which a
            // trajectory cannot do. Rebuilding is the honest recovery for
            // whatever did.
            QL_LOG_DEBUG("Timeline: refit refused, rebuilding the TLAS");
            vkDeviceWaitIdle(device);
            geometry.RebuildTlas(*contextAdapter, *scene);
            if (pipeline) {
                pipeline->BindAccelerationStructure(geometry.Tlas().GetHandle());
                if (geometry.InstanceCount() > 0) {
                    pipeline->BindInstanceGeometryBuffer(geometry.InstanceInfo());
                }
            }
        }
    }

    // The emitter list holds world-space triangles, so a lamp that moved is
    // sampled where it used to be until the list is rebuilt. Only a lamp: a
    // rock that moved changes nothing in it, and rebuilding per tick for every
    // scene would put a full walk of the geometry on the scrub path.
    const bool emitterMoved =
        std::any_of(moved.begin(), moved.end(), [this](u32 node) {
            return node < scene->nodes.size() &&
                   rendercore::NodeHasSampledEmission(*scene, scene->nodes[node]);
        });
    if (emitterMoved) {
        FlushPendingTlasRefit();
        RebuildEmissiveGeometry();
    }

    return moved;
}

VkAccelerationStructureKHR ExternalRenderContext::Impl::TimelineEpochHost::ApplyEpoch(
    const f64 t_s) {
    if (!captured) {
        // Where the clock actually stands, so Restore can put it back. Taken
        // on the first call rather than handed in, because the builder is
        // reached through the preview and the preview does not know about
        // timelines.
        restore_s = owner->timeline.Current_s();
        captured = true;
    }
    owner->ApplyTimelinePose(t_s);
    return owner->geometry.IsValid() ? owner->geometry.Tlas().GetHandle() : VK_NULL_HANDLE;
}

void ExternalRenderContext::Impl::TimelineEpochHost::Restore() {
    if (!captured) return;
    owner->ApplyTimelinePose(restore_s);
    captured = false;
}

void ExternalRenderContext::Impl::RefreshEpochPlan() {
    if (!thermalPreview) return;

    const rendercore::TimelineConfig& config = timeline.Config();
    const bool wantEpochs =
        timeline.Present() && timeline.HasMotion() &&
        config.thermalGeometry == rendercore::TimelineConfig::ThermalGeometry::Epochs;
    if (!wantEpochs) {
        // Reference mode, or nothing moves: measure the world once, where the
        // scene stands. Which is also what every scene did before the clock
        // existed.
        thermalPreview->SetEpochHost(nullptr);
        thermalPreview->SetEpochPlan({}, {});
        return;
    }

    thermal::EpochPlanInput input;
    input.start_s = config.start_s;
    input.end_s = config.end_s;
    input.minMove_m = static_cast<f32>(config.thermalEpochMinMove_m);
    // One thermal timestep is the finest division worth making: the stepper
    // integrates one geometry per step, so two boundaries inside one step are
    // the same as one. `thermal_time_scale` is what turns a thermal second
    // into a timeline second.
    input.stride_s = (config.thermalEpochStride_s > 0.0)
                         ? config.thermalEpochStride_s
                         : thermalTimestep_s / std::max(config.thermalTimeScale, 1e-9);

    // The discontinuities go on the first node: the planner unions the
    // candidates anyway, and repeating them per node would only make the list
    // longer.
    Vector<f64> changeTimes = timeline.ChangeTimes(config.start_s, config.end_s);
    for (const rendercore::AnimatedNode& animated : timeline.Animated()) {
        thermal::EpochPlanInput::Node node;
        node.boundRadius_m = animated.boundRadius_m;
        node.poseAt = [this, &animated](f64 t) { return timeline.PoseAt(animated, t); };
        node.changeTimes = std::move(changeTimes);
        changeTimes.clear();
        input.nodes.push_back(std::move(node));
    }

    const Vector<f64> times = thermal::PlanEpochTimes(input);
    Vector<f64> from_h;
    from_h.reserve(times.size());
    for (const f64 t : times) from_h.push_back(timeline.HourAt(t));

    QL_LOG_INFO("  Thermal epochs: {} planned over {:.3f} s (stride {:g} s, min move {:g} m)",
                times.size(), config.end_s - config.start_s, input.stride_s,
                config.thermalEpochMinMove_m);

    epochHost.owner = this;
    thermalPreview->SetEpochHost(&epochHost);
    thermalPreview->SetEpochPlan(times, std::move(from_h));
}

Result<void, String> ExternalRenderContext::SetTimelineTime(const f64 t_s) {
    if (!m_impl->scene) {
        return Result<void, String>::Err("no scene loaded");
    }
    // Scrubbing back before a committed camera acquisition invalidates the
    // detector's temporal history (thermal detector state, acquisition-keyed
    // noise). The scheduler applies a reset on the next RenderFrame, which
    // then records no acquisition; forward moves never reset.
    if (m_impl->cameraConfig.enabled &&
        m_impl->cameraScheduler.hasCommittedAcquisition &&
        t_s < m_impl->cameraScheduler.lastCommittedAcquisitionTime - 1e-9) {
        auto& s = m_impl->cameraScheduler;
        s.historyResetPending = true;
        s.historyResetReason =
            String("timeline scrubbed back before the committed camera ") +
            "acquisition at t=" +
            std::to_string(m_impl->cameraScheduler.lastCommittedAcquisitionTime);
    }
    if (!m_impl->timeline.Present()) {
        // Legal and quiet: a static scene has a clock that does nothing, and a
        // host driving one transport for every document should not have to ask
        // first.
        m_impl->timeline.SetCurrent(t_s);
        return Result<void, String>::Ok();
    }

    m_impl->ApplyTimelinePose(t_s, !m_impl->cameraConfig.enabled);

    // The hour follows the clock when the two are mapped. SetThermalTime
    // resets accumulation itself, so this does not do it twice.
    if (m_impl->thermalEnabled && m_impl->timeline.ThermalMapped()) {
        auto solved = SetThermalTime(m_impl->timeline.HourAt(t_s));
        if (!solved.has_value()) {
            ResetAccumulation();
            return solved;
        }
        return Result<void, String>::Ok();
    }

    ResetAccumulation();
    return Result<void, String>::Ok();
}

TimelineInfo ExternalRenderContext::GetTimelineInfo() const {
    TimelineInfo info = m_impl->timeline.Info();
    if (m_impl->thermalPreview && m_impl->thermalEnabled) {
        try {
            const ThermalSolveStatus status = m_impl->thermalPreview->Status();
            info.thermalEpochCount = status.thermalEpochCount;
            info.currentThermalEpoch = status.currentThermalEpoch;
        } catch (...) {
            // A status snapshot that throws says nothing about the clock.
        }
    }
    return info;
}

Result<void, String> ExternalRenderContext::SetTimelineThermalMapping(const f64 hourAtStart_h,
                                                                      const f64 scale) {
    if (!m_impl->timeline.Present()) {
        return Result<void, String>::Err("this scene has no timeline");
    }
    m_impl->timeline.SetThermalMapping(hourAtStart_h, scale);
    return SetTimelineTime(m_impl->timeline.Current_s());
}

glm::mat4 ExternalRenderContext::GetNodeRestTransform(const u32 nodeIndex) const {
    if (!m_impl->scene) return glm::mat4(1.0f);
    return m_impl->timeline.RestOf(*m_impl->scene, nodeIndex);
}

void ExternalRenderContext::SetThermalEpochProgressCallback(
    std::function<void(u32 epoch, u32 count)> callback) {
    m_impl->thermalEpochProgress = std::move(callback);
    if (m_impl->thermalPreview) {
        m_impl->thermalPreview->SetEpochProgressCallback(m_impl->thermalEpochProgress);
    }
}

ThermalSolveStatus ExternalRenderContext::GetThermalSolveStatus() const {
    ThermalSolveStatus status;
    if (m_impl->thermalPreview) {
        try {
            status = m_impl->thermalPreview->Status();
        } catch (...) {
            status = {};
        }
    }
    if (!m_impl->thermalLastError.empty()) {
        status.error = m_impl->thermalLastError;
        status.solveValid = false;
    }
    return status;
}

void ExternalRenderContext::RebuildAccelerationStructure() {
    if (!m_impl->scene || !m_impl->geometry.IsValid()) {
        QL_LOG_WARN("RebuildAccelerationStructure: No scene or geometry available");
        return;
    }

    // A frame the host submitted may still be tracing the TLAS being replaced.
    m_impl->pendingTlasRefit = false;
    vkDeviceWaitIdle(m_impl->device);

    m_impl->geometry.RebuildTlas(*m_impl->contextAdapter, *m_impl->scene);

    if (m_impl->pipeline) {
        m_impl->pipeline->BindAccelerationStructure(m_impl->geometry.Tlas().GetHandle());
        // A topology edit makes RebuildTlas replace the instance info buffer;
        // rebinding unconditionally is cheap on the already-idle device
        if (m_impl->geometry.InstanceCount() > 0) {
            m_impl->pipeline->BindInstanceGeometryBuffer(m_impl->geometry.InstanceInfo());
        }
    }

    // World-space emitters, so a topology or transform edit invalidates them
    // exactly as it does the TLAS.
    m_impl->RebuildEmissiveGeometry();

    if (m_impl->thermalPreview) {
        m_impl->thermalPreview->InvalidateGeometry();
        // A rest pose moved, so where the trajectory takes it moved too.
        m_impl->RefreshEpochPlan();
    }
}

void ExternalRenderContext::RefitAccelerationStructure() {
    if (!m_impl->scene || !m_impl->geometry.IsValid()) {
        QL_LOG_WARN("RefitAccelerationStructure: No scene or geometry available");
        return;
    }

    // Same handle, updated in place: no rebind, no device idle. The refit
    // command's own barriers order it against in-flight tracing.
    if (!m_impl->geometry.PrepareTlasRefit(*m_impl->scene)) {
        QL_LOG_DEBUG("RefitAccelerationStructure: topology changed, rebuilding");
        RebuildAccelerationStructure();  // redoes the emitters itself
        return;
    }

    // The emitter list holds world-space positions, so a moved node leaves it
    // describing where the light used to be. Nothing else notices -- the TLAS
    // is correct, so the light renders in its new place while being sampled at
    // its old one, which reads as a light that has stopped illuminating.
    m_impl->pendingTlasRefit = true;
    if (m_impl->emissiveTransformDirty) {
        // The world-space emitter buffer is still updated synchronously;
        // normal nodes need no emitter upload and take the deferred path.
        m_impl->FlushPendingTlasRefit();
        m_impl->RebuildEmissiveGeometry();
    }

    if (m_impl->thermalPreview) {
        m_impl->thermalPreview->InvalidateGeometry();
        m_impl->RefreshEpochPlan();
    }
}

// ============================================================================
// Status and Statistics
// ============================================================================

u32 ExternalRenderContext::GetAccumulatedSamples() const {
    return m_impl->accumulatedSamples;
}

f32 ExternalRenderContext::GetLastFrameTimeMs() const {
    if (m_impl->perfLogger && m_impl->perfLogger->TryResolvePending())
        m_impl->lastSampleGpuMs = m_impl->perfLogger->GetLastFrameGpuMs();
    return m_impl->lastSampleGpuMs;
}

bool ExternalRenderContext::IsReady() const {
    return m_impl->isReady;
}

const String& ExternalRenderContext::GetPipelineCachePath() const {
    return m_impl->pipelineCachePath;
}

Result<glm::vec4, String> ExternalRenderContext::ReadPixelValue(u32 x, u32 y) {
    // Coordinates are in the target extent -- the pixels the host presented
    // into and the user clicked on. Below a render scale of 1.0 they index a
    // larger grid than the accumulation has, so map before reading.
    if (x >= m_impl->targetWidth || y >= m_impl->targetHeight) {
        return Result<glm::vec4, String>::Err("Pixel coordinates out of bounds");
    }
    x = Impl::MapToRender(x, m_impl->targetWidth, m_impl->width);
    y = Impl::MapToRender(y, m_impl->targetHeight, m_impl->height);

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
    const void* mappedData = m_impl->pixelReadbackBuffer->MapRead();
    if (mappedData == nullptr) {
        return Result<glm::vec4, String>::Err("Failed to map pixel readback buffer");
    }

    const f32* pixelData = static_cast<const f32*>(mappedData);
    glm::vec4 result(pixelData[0], pixelData[1], pixelData[2], pixelData[3]);

    m_impl->pixelReadbackBuffer->Unmap();

    return result;
}

Result<bool, String> ExternalRenderContext::RequestPixelValue(u32 x, u32 y, u64 requestId) {
    if (x >= m_impl->targetWidth || y >= m_impl->targetHeight)
        return Result<bool, String>::Err("Pixel coordinates out of bounds");
    if (!m_impl->isReady || !m_impl->outputImage ||
        m_impl->cameraConfig.enabled || m_impl->rawPixelSamples == 0 ||
        m_impl->rawPixelGeneration != m_impl->pixelImageGeneration)
        return Result<bool, String>::Err("No current raw accumulation pixel is available");
    try {
        if (!m_impl->asyncPixelReadback)
            m_impl->asyncPixelReadback = std::make_unique<rendercore::AsyncPixelReadback>(
                *m_impl->contextAdapter);
        PixelReading reading;
        reading.requestId = requestId;
        reading.imageGeneration = m_impl->pixelImageGeneration;
        reading.acquisitionIndex = m_impl->cameraAcquisitionIndex;
        reading.frameIndex = m_impl->frameIndex;
        reading.accumulatedSamples = m_impl->rawPixelSamples;
        reading.x = x;
        reading.y = y;
        return m_impl->asyncPixelReadback->Submit(m_impl->outputImage->GetImage(),
            Impl::MapToRender(x, m_impl->targetWidth, m_impl->width),
            Impl::MapToRender(y, m_impl->targetHeight, m_impl->height), reading);
    } catch (const std::exception& e) {
        return Result<bool, String>::Err(e.what());
    }
}

Result<Optional<PixelReading>, String> ExternalRenderContext::PollPixelValue() {
    if (!m_impl->asyncPixelReadback) return Optional<PixelReading>{};
    try {
        return m_impl->asyncPixelReadback->Poll(m_impl->pixelImageGeneration,
                                              m_impl->cameraAcquisitionIndex);
    } catch (const std::exception& e) {
        return Result<Optional<PixelReading>, String>::Err(e.what());
    }
}

namespace {

// Must match PickResultGpu in pick.rayq.hlsl (std430, 32 bytes)
struct PickResultGpu {
    u32 hit;
    u32 instanceIndex;
    u32 primitiveIndex;
    f32 hitT;
    glm::vec3 worldPosition;
    u32 _pad;
};
static_assert(sizeof(PickResultGpu) == 32, "PickResultGpu size mismatch");

// Must match PickPushConstants in pick.rayq.hlsl (80 bytes)
struct PickPushConstants {
    glm::vec3 origin;
    f32 fovScale;
    glm::vec3 forward;
    f32 aspectRatio;
    glm::vec3 right;
    u32 pixelX;
    glm::vec3 up;
    u32 pixelY;
    u32 width;
    u32 height;
    // Same two words the shader repurposed; the struct is still 80 bytes.
    u32 projection;
    f32 orthoHeight;
};
static_assert(sizeof(PickPushConstants) == 80, "PickPushConstants size mismatch");

}  // namespace

Result<PickResult, String> ExternalRenderContext::Pick(u32 x, u32 y) {
    try {
        m_impl->FlushPendingTlasRefit();
    } catch (const std::exception& ex) {
        return Result<PickResult, String>::Err(String("Pick: ") + ex.what());
    }

    if (!m_impl->isReady || !m_impl->geometry.IsValid()) {
        return Result<PickResult, String>::Err("Pick: no scene loaded");
    }
    // Target-extent coordinates, like ReadPixelValue's -- the pick ray is
    // reconstructed from the render grid, so map into it first.
    if (x >= m_impl->targetWidth || y >= m_impl->targetHeight) {
        return Result<PickResult, String>::Err("Pick: pixel out of bounds");
    }
    x = Impl::MapToRender(x, m_impl->targetWidth, m_impl->width);
    y = Impl::MapToRender(y, m_impl->targetHeight, m_impl->height);

    m_impl->CreatePickPipeline();
    if (m_impl->pickPipeline == VK_NULL_HANDLE || !m_impl->pickOutputBuffer) {
        return Result<PickResult, String>::Err("Pick: pipeline unavailable");
    }

    // Rebind every call: RebuildTlas replaces the TLAS object outright
    VkAccelerationStructureKHR tlas = m_impl->geometry.Tlas().GetHandle();
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_impl->pickOutputBuffer->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asInfo;
    writes[0].dstSet = m_impl->pickDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[0].descriptorCount = 1;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_impl->pickDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(m_impl->device, static_cast<u32>(writes.size()),
                           writes.data(), 0, nullptr);

    // The exact camera the raygen shader gets, so the pick ray matches the frame
    const CameraData cameraData = m_impl->camera.GetCameraData();
    PickPushConstants pc{};
    pc.origin = cameraData.origin;
    pc.fovScale = cameraData.fovScale;
    pc.forward = cameraData.forward;
    pc.aspectRatio = cameraData.aspectRatio;
    pc.right = cameraData.right;
    pc.pixelX = x;
    pc.up = cameraData.up;
    pc.pixelY = y;
    pc.width = m_impl->width;
    pc.height = m_impl->height;
    // From the same CameraData raygen is given, so a pick and a render agree
    // about which projection is in use.
    pc.projection = cameraData.projection;
    pc.orthoHeight = cameraData.orthoHeight;

    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pickPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_impl->pickPipelineLayout, 0, 1,
                                &m_impl->pickDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_impl->pickPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = m_impl->pickOutputBuffer->GetHandle();
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier,
                             0, nullptr);
    });

    PickResultGpu gpu{};
    const void* mapped = m_impl->pickOutputBuffer->MapRead();
    if (!mapped) return Result<PickResult, String>::Err("Pick: cannot map readback memory");
    std::memcpy(&gpu, mapped, sizeof(gpu));
    m_impl->pickOutputBuffer->Unmap();

    PickResult result;
    result.hit = gpu.hit != 0;
    if (result.hit) {
        const auto& toNode = m_impl->geometry.InstanceToNode();
        if (gpu.instanceIndex >= toNode.size()) {
            return Result<PickResult, String>::Err("Pick: instance index out of range");
        }
        result.nodeIndex = toNode[gpu.instanceIndex];
        result.instanceIndex = gpu.instanceIndex;
        result.primitiveIndex = gpu.primitiveIndex;
        result.hitT = gpu.hitT;
        result.worldPosition = gpu.worldPosition;
    }
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

void ExternalRenderContext::SetDisplayEnhancementParams(
    const DisplayEnhancementParams& params) {
    const bool wasEnabled = m_impl->displayParams.enabled;
    // The percentile window is computed on the host and cached; moving it has
    // to invalidate that cache or the new window takes effect ten frames late,
    // or never once the trace has stopped.
    const bool windowMoved = params.percentileLow != m_impl->displayParams.percentileLow ||
                             params.percentileHigh != m_impl->displayParams.percentileHigh;
    m_impl->displayParams = params;

    if (windowMoved) {
        m_impl->hasCachedMinMax = false;
    }

    // Initialize the pipeline if enabling for the first time
    if (params.enabled && !wasEnabled && !m_impl->claheInitialized) {
        m_impl->CreateCLAHEPipeline();
    }

    QL_LOG_DEBUG("Display enhancement: enabled={}, tone={}, palette={}, clipLimit={}, "
                 "tileSize={}, window=[{}, {}]",
                 params.enabled, static_cast<u32>(params.toneMode),
                 static_cast<u32>(params.palette), params.clipLimit, params.tileSize,
                 params.percentileLow, params.percentileHigh);
}

const DisplayEnhancementParams& ExternalRenderContext::GetDisplayEnhancementParams() const {
    return m_impl->displayParams;
}

// ============================================================================
// GPU Sensor Simulation API
// ============================================================================

Result<void, String> ExternalRenderContext::SetCameraConfig(
    const camera::CameraConfig& config) {
    camera::CameraConfig normalized = config;
    if (config.enabled) {
        const auto valid = camera::ValidateCameraConfig(config);
        if (!valid) return Result<void, String>::Err(valid.error());
        if (config.device.channels.size() > camera::kCameraResponseMaxChannels)
            return Result<void, String>::Err(
                "GPU camera supports at most three response channels");
        if (m_impl->cameraSolarSources) {
            const auto& [sun, sky] = *m_impl->cameraSolarSources;
            for (const auto& channel : config.device.channels) {
                const auto& stack = channel.response;
                const auto& base = stack.systemResponse ? *stack.systemResponse :
                    config.device.detector == camera::DetectorKind::Photon ?
                    *stack.quantumEfficiency : *stack.thermalAbsorptance;
                if (sun.samples.empty() || sky.samples.empty() ||
                    sun.samples.front().first > base.MinNm() ||
                    sun.samples.back().first < base.MaxNm() ||
                    sky.samples.front().first > base.MinNm() ||
                    sky.samples.back().first < base.MaxNm())
                    return Result<void, String>::Err(
                        "GPU camera response exceeds solar/sky LUT coverage");
            }
        }
        if (normalized.device.effectiveMinNm == 0.0 &&
            normalized.device.effectiveMaxNm == 0.0) {
            normalized.device.effectiveMinNm = std::numeric_limits<f64>::infinity();
            for (const auto& channel : normalized.device.channels) {
                const auto& stack = channel.response;
                const auto& base = stack.systemResponse ? *stack.systemResponse :
                    normalized.device.detector == camera::DetectorKind::Photon ?
                    *stack.quantumEfficiency : *stack.thermalAbsorptance;
                normalized.device.effectiveMinNm =
                    std::min(normalized.device.effectiveMinNm, base.MinNm());
                normalized.device.effectiveMaxNm =
                    std::max(normalized.device.effectiveMaxNm, base.MaxNm());
            }
        }
    }
    m_impl->cameraConfig = std::move(normalized);
    m_impl->gpuSensorEnabled = config.enabled;
    m_impl->cameraResourcesDirty = true;
    m_impl->cameraCaptureCompleted = false;
    m_impl->cameraCaptureRecorded = false;
    m_impl->cameraCapturePending = config.enabled;
    m_impl->cameraAcquisitionIndex = 0;
    m_impl->cameraAcquisitionTimeSeconds = m_impl->timeline.Current_s();
    // The interactive scheduler restarts from the new configuration: the
    // first RenderFrame with the camera enabled takes scheduling ownership
    // again and the first acquisition lands at the current clock time.
    m_impl->cameraScheduler.drivesAcquisition = false;
    m_impl->cameraScheduler.nextAcquisitionIndex = 0;
    m_impl->cameraScheduler.nextAcquisitionTimeSeconds =
        m_impl->timeline.Current_s();
    // The CLAHE input extent may have changed (camera on/off, sensor size);
    // its display image follows it.
    m_impl->hasCachedMinMax = false;
    m_impl->RecreateClaheDisplayImage();
    // A new configuration restarts the AE/AWB loop from its authored manual
    // values; the first acquisition under it is fresh state.
    m_impl->cameraAutoControlDue = true;
    m_impl->cameraAutoLastAcquisition = std::numeric_limits<u64>::max();
    m_impl->cameraAutoAppliedAcquisition = std::numeric_limits<u64>::max();
    m_impl->cameraHostState = camera::CaptureState{};
    return Result<void, String>::Ok();
}

const camera::CameraConfig& ExternalRenderContext::GetCameraConfig() const {
    return m_impl->cameraConfig;
}

Result<void, String> ExternalRenderContext::UpdateCameraDisplayConfig(
    const camera::CameraConfig& config) {
    if (!m_impl->cameraConfig.enabled || !config.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (m_impl->cameraCaptureRecorded)
        return Result<void, String>::Err(
            "complete the queued camera acquisition before editing display");
    if (const auto valid = camera::ValidateCameraConfig(config); !valid)
        return valid;

    camera::CameraConfig expected = m_impl->cameraConfig;
    auto& display = expected.isp;
    const auto& requested = config.isp;
    display.whiteBalance = requested.whiteBalance;
    display.deviceToLinearSrgb = requested.deviceToLinearSrgb;
    display.denoise = requested.denoise;
    display.denoiseStrength = requested.denoiseStrength;
    display.sharpen = requested.sharpen;
    display.sharpenStrength = requested.sharpenStrength;
    display.toneGamma = requested.toneGamma;
    display.clipOutOfGamut = requested.clipOutOfGamut;
    display.infraredTone = requested.infraredTone;
    display.infraredPalette = requested.infraredPalette;
    display.contrastLowPercentile = requested.contrastLowPercentile;
    display.contrastHighPercentile = requested.contrastHighPercentile;
    display.hsv = requested.hsv;

    // SetCameraConfig normalizes a missing effective span from the channel
    // curves. Studio keeps its authored config, which may still have zeros.
    camera::CameraConfig authored = config;
    if (authored.device.effectiveMinNm == 0.0 &&
        authored.device.effectiveMaxNm == 0.0) {
        authored.device.effectiveMinNm = expected.device.effectiveMinNm;
        authored.device.effectiveMaxNm = expected.device.effectiveMaxNm;
    }
    if (CameraConfigToToml(authored) != CameraConfigToToml(expected))
        return Result<void, String>::Err(
            "display update also changes measurement, readout or control fields");

    // RecordDisplayReprocess uploads the new ISP constants at its next call.
    // The completed products, thermal state, AE/AWB feedback and scheduler
    // remain untouched here.
    m_impl->cameraConfig = std::move(expected);
    return Result<void, String>::Ok();
}

Result<bool, String> ExternalRenderContext::TryUpdateCameraReadoutConfig(
    const camera::CameraConfig& config) {
    if (!m_impl->cameraConfig.enabled || !config.enabled ||
        m_impl->cameraCaptureRecorded)
        return false;
    if (const auto valid = camera::ValidateCameraConfig(config); !valid)
        return Result<bool, String>::Err(valid.error());

    camera::CameraConfig expected = m_impl->cameraConfig;
    expected.readout.analogGain = config.readout.analogGain;
    expected.readout.electronsPerDn = config.readout.electronsPerDn;
    expected.readout.blackLevelDn = config.readout.blackLevelDn;
    expected.readout.adcBits = config.readout.adcBits;
    expected.readout.outputBits = config.readout.outputBits;
    // These parameters act after the measured optical rate. Calibration
    // maps and response curves are excluded: they need resource uploads.
    expected.photon.fullWellElectrons = config.photon.fullWellElectrons;
    expected.photon.darkCurrentElectronsPerSecond =
        config.photon.darkCurrentElectronsPerSecond;
    expected.photon.readNoiseElectronsRms = config.photon.readNoiseElectronsRms;
    expected.photon.prnuSigma = config.photon.prnuSigma;
    expected.photon.dsnuElectronsRms = config.photon.dsnuElectronsRms;
    expected.photon.dsnuReferenceExposureSeconds =
        config.photon.dsnuReferenceExposureSeconds;
    expected.photon.biasDnRms = config.photon.biasDnRms;
    expected.photon.nucResidualFraction = config.photon.nucResidualFraction;
    expected.photon.enableShotNoise = config.photon.enableShotNoise;
    expected.photon.enableDarkCurrent = config.photon.enableDarkCurrent;
    expected.photon.enableDarkShotNoise = config.photon.enableDarkShotNoise;
    expected.photon.enableReadNoise = config.photon.enableReadNoise;
    expected.photon.enableFpn = config.photon.enableFpn;
    expected.photon.applyNuc = config.photon.applyNuc;
    expected.thermal.responsivityDnPerWatt = config.thermal.responsivityDnPerWatt;
    expected.thermal.readNoiseDnRms = config.thermal.readNoiseDnRms;
    expected.thermal.driftDnPerSecond = config.thermal.driftDnPerSecond;
    expected.thermal.netdKelvin = config.thermal.netdKelvin;
    expected.thermal.netdReferenceTemperatureK =
        config.thermal.netdReferenceTemperatureK;
    expected.thermal.netdNoiseBandwidthHz = config.thermal.netdNoiseBandwidthHz;
    expected.thermal.readoutWindowSeconds = config.thermal.readoutWindowSeconds;
    expected.thermal.netdOpticalCondition = config.thermal.netdOpticalCondition;
    expected.products = config.products;

    camera::CameraConfig authored = config;
    if (authored.device.effectiveMinNm == 0.0 &&
        authored.device.effectiveMaxNm == 0.0) {
        authored.device.effectiveMinNm = expected.device.effectiveMinNm;
        authored.device.effectiveMaxNm = expected.device.effectiveMaxNm;
    }
    if (CameraConfigToToml(authored) != CameraConfigToToml(expected))
        return false;

    camera::CameraConfig previous = std::move(m_impl->cameraConfig);
    m_impl->cameraConfig = std::move(expected);
    if (m_impl->cameraGpuPipeline && m_impl->cameraCaptureCompleted) {
        Result<void, String> status = Result<void, String>::Ok();
        CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                        [&](VkCommandBuffer cmd) {
            status = m_impl->cameraGpuPipeline->RecordReadoutReprocess(
                cmd, m_impl->EffectiveCameraConfig());
        });
        if (!status) {
            m_impl->cameraConfig = std::move(previous);
            return Result<bool, String>::Err(status.error());
        }
    }
    return true;
}

Result<void, String> ExternalRenderContext::QueueCameraAcquisition(
    u64 acquisitionIndex, f64 firstRowMidpointSeconds) {
    if (!m_impl->cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (!std::isfinite(firstRowMidpointSeconds))
        return Result<void, String>::Err("camera acquisition time must be finite");
    if (m_impl->cameraCaptureRecorded)
        return Result<void, String>::Err(
            "complete or re-record the queued camera acquisition before "
            "queuing another one");
    if (m_impl->cameraCaptureCompleted &&
        acquisitionIndex < m_impl->cameraAcquisitionIndex)
        return Result<void, String>::Err(
            "camera acquisition cannot move backwards without resetting state");
    m_impl->cameraAcquisitionIndex = acquisitionIndex;
    m_impl->cameraAcquisitionTimeSeconds = firstRowMidpointSeconds;
    m_impl->cameraCapturePending = true;
    m_impl->cameraCaptureCompleted = false;
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::RecordQueuedCameraAcquisition(
    VkCommandBuffer cmd) {
    if (!m_impl->isReady || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "camera acquisition needs a ready context and command buffer");
    if (!m_impl->cameraCapturePending)
        return Result<void, String>::Err("no camera acquisition is queued");
    if (auto moved = SetTimelineTime(m_impl->cameraAcquisitionTimeSeconds); !moved)
        return moved;
    return m_impl->RecordCameraMeasurement(cmd);
}

Result<void, String> ExternalRenderContext::CompleteQueuedCameraAcquisition() {
    if (!m_impl->cameraCaptureRecorded)
        return Result<void, String>::Err(
            "no recorded camera acquisition is awaiting completion");
    m_impl->cameraCaptureRecorded = false;
    m_impl->cameraCapturePending = false;
    m_impl->cameraCaptureCompleted = true;
    // The committed tick's statistics are on the device (the host submitted
    // before completing): the AE/AWB loop consumes them when the next tick
    // is recorded.
    m_impl->cameraAutoControlDue = true;
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::ResetCameraHistory() {
    // Discard the detector's temporal history and restart the interactive
    // scheduler at acquisition 0 of a new epoch. Safe to call between frames;
    // the explicit Queue/Record/Complete path is unaffected apart from the
    // GpuCameraPipeline counter restart.
    if (m_impl->cameraGpuPipeline) m_impl->cameraGpuPipeline->ResetState();
    m_impl->cameraStratumSampleCounts.fill(0);
    m_impl->cameraStratumRecordedAcquisition =
        std::numeric_limits<u64>::max();
    auto& s = m_impl->cameraScheduler;
    s.historyResetPending = false;
    s.historyResetReason = "camera history reset requested by host";
    s.historyEpoch += 1;
    s.hasCommittedAcquisition = false;
    s.lastCommittedAcquisitionTime = 0.0;
    s.nextAcquisitionIndex = 0;
    s.nextAcquisitionTimeSeconds = m_impl->timeline.Current_s();
    m_impl->cameraAcquisitionIndex = 0;
    m_impl->cameraAcquisitionTimeSeconds = m_impl->timeline.Current_s();
    m_impl->cameraCapturePending = m_impl->cameraConfig.enabled;
    m_impl->cameraCaptureRecorded = false;
    m_impl->cameraCaptureCompleted = false;
    // The AE/AWB loop restarts from the authored manual values; the zeroed
    // statistics buffer is a valid (empty) measurement, so the loop may run.
    m_impl->cameraAutoControlDue = true;
    m_impl->cameraAutoLastAcquisition = std::numeric_limits<u64>::max();
    m_impl->cameraAutoAppliedAcquisition = std::numeric_limits<u64>::max();
    m_impl->cameraHostState = camera::CaptureState{};
    return Result<void, String>::Ok();
}

CameraHistoryStatus ExternalRenderContext::GetCameraHistoryStatus() const {
    CameraHistoryStatus status;
    const auto& s = m_impl->cameraScheduler;
    status.epoch = s.historyEpoch;
    status.acquisitionIndex = m_impl->cameraAcquisitionIndex;
    status.frameTimeSeconds = m_impl->cameraAcquisitionTimeSeconds;
    status.historyReset = s.historyResetPending;
    status.lastResetReason = s.historyResetReason;
    // The AE/AWB feedback the next committed acquisition will consume
    // (cameraHostState.next*); zero exposure means fresh state, i.e. the
    // next acquisition starts from the authored manual values.
    status.nextExposureSeconds = m_impl->cameraHostState.nextExposureSeconds;
    status.nextAnalogGain = m_impl->cameraHostState.nextAnalogGain;
    status.nextWhiteBalance = m_impl->cameraHostState.nextWhiteBalance;
    return status;
}

Result<void, String> ExternalRenderContext::CheckpointCameraHistory() {
    if (!m_impl->cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (auto ready = m_impl->EnsureCameraResources(); !ready) return ready;

    Result<rendercore::GpuCameraPipeline::GpuCheckpoint, String> recorded =
        Result<rendercore::GpuCameraPipeline::GpuCheckpoint, String>::Err(
            "camera history checkpoint was not recorded");
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                    [&](VkCommandBuffer cmd) {
        recorded = m_impl->cameraGpuPipeline->RecordStateCheckpoint(cmd);
    });
    if (!recorded) return Result<void, String>::Err(recorded.error());

    auto& s = m_impl->cameraScheduler;
    // The host half of the record is the feedback carrier: the AE/AWB next*
    // fields stay exactly as the committed ticks left them, so a restore
    // rewinds the closed loop and a replayed tick sequence reproduces the
    // original feedback bit for bit.
    m_impl->cameraHostState.acquisitionIndex = m_impl->cameraAcquisitionIndex;
    m_impl->cameraHostState.frameTimeSeconds =
        m_impl->cameraAcquisitionTimeSeconds;
    m_impl->cameraHostState.historyEpoch = s.historyEpoch;

    ExternalRenderContext::Impl::CameraHistoryRecord record;
    record.gpu = *recorded;
    record.gpu.historyEpoch = s.historyEpoch;
    record.host = m_impl->cameraHostState;
    // Depth one: the newest checkpoint wins.
    m_impl->cameraHistoryStack.clear();
    m_impl->cameraHistoryStack.push_back(std::move(record));
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::RestoreCameraHistoryCheckpoint() {
    if (m_impl->cameraHistoryStack.empty())
        return Result<void, String>::Err(
            "no camera history checkpoint has been recorded");
    ExternalRenderContext::Impl::CameraHistoryRecord record =
        std::move(m_impl->cameraHistoryStack.back());
    m_impl->cameraHistoryStack.pop_back();
    if (!m_impl->cameraGpuPipeline)
        return Result<void, String>::Err(
            "camera history checkpoint cannot restore without the GPU camera "
            "pipeline");

    Result<void, String> restored = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                    [&](VkCommandBuffer cmd) {
        restored = m_impl->cameraGpuPipeline->RestoreStateCheckpoint(
            record.gpu, cmd);
    });
    if (!restored) return restored;
    m_impl->cameraHostState = record.host;
    // The replayed ticks re-run the controller from the restored feedback,
    // consuming the restored statistics buffer, and the effective config is
    // pushed to the device again on the next record.
    m_impl->cameraAutoControlDue = true;
    m_impl->cameraAutoLastAcquisition = std::numeric_limits<u64>::max();
    m_impl->cameraAutoAppliedAcquisition = std::numeric_limits<u64>::max();

    auto& s = m_impl->cameraScheduler;
    // A restore rewinds history, so the epoch moves on and the reason is
    // recorded for GetCameraHistoryStatus. The restore is applied
    // synchronously, though: unlike a timeline scrub there is nothing left
    // pending, and a pending reset on the next RenderFrame would discard
    // the state this call just restored.
    s.historyEpoch += 1;
    s.historyResetPending = false;
    s.historyResetReason = "checkpoint restore";
    s.hasCommittedAcquisition = false;
    s.lastCommittedAcquisitionTime = 0.0;
    if (s.drivesAcquisition) {
        // The scheduler resumes at the checkpoint's tick: replaying it is a
        // same-index re-record, and the slots after it advance normally.
        s.nextAcquisitionIndex = record.host.acquisitionIndex;
        s.nextAcquisitionTimeSeconds = record.host.frameTimeSeconds;
    }
    // The explicit path resumes at the checkpoint's exit as well: the next
    // queued acquisition may re-record the checkpoint's tick or advance from
    // it, both of which the restored GPU state accepts.
    m_impl->cameraAcquisitionIndex = record.host.acquisitionIndex;
    m_impl->cameraAcquisitionTimeSeconds = record.host.frameTimeSeconds;
    m_impl->cameraCapturePending = false;
    m_impl->cameraCaptureRecorded = false;
    m_impl->cameraCaptureCompleted = true;
    m_impl->cameraHostState.acquisitionIndex = m_impl->cameraAcquisitionIndex;
    m_impl->cameraHostState.frameTimeSeconds =
        m_impl->cameraAcquisitionTimeSeconds;
    m_impl->cameraHostState.historyEpoch = s.historyEpoch;
    // Per-tick progressive accumulation starts over on replay: sampleIndex
    // 0 overwrites the measurement image, so a replayed tick converges to
    // the identical average rather than continuing the discarded run's.
    m_impl->cameraStratumSampleCounts.fill(0);
    m_impl->cameraStratumRecordedAcquisition =
        std::numeric_limits<u64>::max();
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::WarmUpCamera(f64 seconds) {
    if (!m_impl->cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    const f64 period = m_impl->cameraConfig.readout.framePeriodSeconds;
    if (!std::isfinite(seconds) || seconds < 0.0)
        return Result<void, String>::Err(
            "camera warmup seconds must be finite and nonnegative");
    if (!std::isfinite(period) || period <= 0.0)
        return Result<void, String>::Err(
            "camera warmup frame period must be finite and positive");
    const f64 now = m_impl->cameraAcquisitionTimeSeconds;
    if (!std::isfinite(now))
        return Result<void, String>::Err(
            "camera warmup needs a finite current frame time");
    // Same grid as the offline CPU reference (camera::WarmUpCamera): the
    // synthetic acquisitions sit on the frame grid that ends one period
    // below `now`, so the first real acquisition afterwards measures a full
    // period of elapsed history. A clamped step lands on the same instant as
    // its predecessor: it moves the index and the noise streams, not the
    // thermal state.
    const u64 steps = static_cast<u64>(std::llround(seconds / period));
    if (steps == 0) return Result<void, String>::Ok();
    const f64 start = std::max(now - static_cast<f64>(steps) * period, 0.0);
    for (u64 i = 0; i < steps; ++i) {
        const f64 nominal = now - static_cast<f64>(steps) * period +
                            static_cast<f64>(i) * period;
        const f64 time = std::max(nominal, start);
        const bool havePrior = m_impl->cameraCaptureCompleted ||
                               m_impl->cameraCaptureRecorded;
        const u64 index = havePrior ? m_impl->cameraAcquisitionIndex + 1
                                    : m_impl->cameraAcquisitionIndex;
        auto queued = QueueCameraAcquisition(index, time);
        if (!queued) return queued;
        Result<void, String> recorded = Result<void, String>::Ok();
        CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                        [&](VkCommandBuffer cmd) {
            recorded = RecordQueuedCameraAcquisition(cmd);
        });
        if (!recorded) return recorded;
        auto completed = CompleteQueuedCameraAcquisition();
        if (!completed) return completed;
    }
    m_impl->cameraHostState.acquisitionIndex = m_impl->cameraAcquisitionIndex;
    m_impl->cameraHostState.frameTimeSeconds =
        m_impl->cameraAcquisitionTimeSeconds;
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::AdvanceCameraTo(f64 timeSeconds) {
    if (!m_impl->cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (!std::isfinite(timeSeconds))
        return Result<void, String>::Err(
            "camera advance needs a finite frame time");
    const bool havePrior = m_impl->cameraCaptureCompleted ||
                           m_impl->cameraCaptureRecorded;
    const u64 index = havePrior ? m_impl->cameraAcquisitionIndex + 1
                                : m_impl->cameraAcquisitionIndex;
    auto queued = QueueCameraAcquisition(index, timeSeconds);
    if (!queued) return queued;
    Result<void, String> recorded = Result<void, String>::Ok();
    CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                    [&](VkCommandBuffer cmd) {
        recorded = RecordQueuedCameraAcquisition(cmd);
    });
    if (!recorded) return recorded;
    auto completed = CompleteQueuedCameraAcquisition();
    if (!completed) return completed;
    m_impl->cameraHostState.acquisitionIndex = m_impl->cameraAcquisitionIndex;
    m_impl->cameraHostState.frameTimeSeconds =
        m_impl->cameraAcquisitionTimeSeconds;
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::RecordCameraBaseline(
    VkCommandBuffer cmd) {
    if (!m_impl->isReady || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "camera baseline needs a ready context and command buffer");
    if (auto moved = SetTimelineTime(m_impl->cameraAcquisitionTimeSeconds); !moved)
        return moved;
    return m_impl->RecordCameraBaseline(cmd);
}

CameraGpuTimings ExternalRenderContext::GetLastCameraGpuTimings() const {
    const bool traceReady = m_impl->cameraTracePerf &&
                            m_impl->cameraTracePerf->TryResolvePending();
    const bool fullReady = m_impl->cameraFullPerf &&
                           m_impl->cameraFullPerf->TryResolvePending();
    if (traceReady && fullReady) {
        auto& timing = m_impl->lastCameraGpuTimings;
        timing.traceMs = m_impl->cameraTracePerf->GetLastFrameGpuMs();
        timing.fullMs = m_impl->cameraFullPerf->GetLastFrameGpuMs();
        timing.postMs = std::max(0.0f, timing.fullMs - timing.traceMs);
        timing.valid = timing.traceMs > 0.0f && timing.fullMs >= timing.traceMs;
    }
    // Per-pass breakdown from the camera pipeline's seven stamps. Never
    // blocks: a frame still in flight leaves the previous values standing.
    if (m_impl->cameraTimingPool != VK_NULL_HANDLE) {
        std::array<u64, 14> results{};
        const VkResult resolved = vkGetQueryPoolResults(
            m_impl->device, m_impl->cameraTimingPool, 0,
            rendercore::GpuCameraPipeline::kTimingQueryCount,
            results.size() * sizeof(u64), results.data(), 2 * sizeof(u64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        bool available = resolved == VK_SUCCESS;
        for (u32 i = 0; available && i < 7; ++i)
            available = results[2 * i + 1] != 0;
        if (available) {
            const f64 period = m_impl->cameraTimestampPeriodNs;
            auto& timing = m_impl->lastCameraGpuTimings;
            const auto span = [&](u32 from, u32 to) {
                return static_cast<f32>(
                    (results[2 * to] - results[2 * from]) * period / 1e6);
            };
            timing.dynamicMs = std::max(0.0f, span(0, 1));
            timing.psfMs = std::max(0.0f, span(1, 2));
            timing.detectorMs = std::max(0.0f, span(2, 3));
            // Stamps 3..4 bracket the full M4-3 ISP (statistics, CDF,
            // demosaic, color, display); stamps 4..5 bracket the M4-4 HSV
            // pass (back to back when HSV is off, so hsvMs reads ~0).
            timing.ispMs = std::max(0.0f, span(3, 4));
            timing.hsvMs = std::max(0.0f, span(4, 5));
        }
    }
    return m_impl->lastCameraGpuTimings;
}

DynamicExposureReport ExternalRenderContext::GetLastDynamicExposureReport() const {
    m_impl->lastDynamicReport = DynamicExposureReport{};
    auto& report = m_impl->lastDynamicReport;
    if (!m_impl->cameraCaptureCompleted || !m_impl->cameraGpuPipeline ||
        m_impl->cameraLastAcquisitionSamples == 0u)
        return report;

    // Both counter sets are read back synchronously; the call must happen
    // between submitted frames, never between record and submit.
    std::array<u32, 4> raygenCounts{};
    bool haveRaygen = false;
    if (m_impl->cameraDynamicCounterBuffer &&
        m_impl->cameraDynamicCounterBuffer->IsValid()) {
        GpuBuffer staging(m_impl->contextAdapter->GetAllocator(),
                          sizeof(raygenCounts), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_TO_CPU);
        if (staging.IsValid()) {
            CommandHelper::ExecuteImmediate(*m_impl->contextAdapter,
                                            [&](VkCommandBuffer cmd) {
                VkBufferMemoryBarrier ready{};
                ready.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ready.buffer = m_impl->cameraDynamicCounterBuffer->GetHandle();
                ready.offset = 0;
                ready.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(cmd,
                                     VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 1, &ready, 0, nullptr);
                VkBufferCopy copy{};
                copy.size = sizeof(raygenCounts);
                vkCmdCopyBuffer(cmd,
                                m_impl->cameraDynamicCounterBuffer->GetHandle(),
                                staging.GetHandle(), 1, &copy);
            });
            if (const void* data = staging.MapRead()) {
                std::memcpy(raygenCounts.data(), data, sizeof(raygenCounts));
                staging.Unmap();
                haveRaygen = true;
            }
        }
    }
    const auto dynamicCounts = m_impl->cameraGpuPipeline->ReadDynamicCounters();
    const bool haveDynamic = dynamicCounts.has_value();

    const f64 pixels = static_cast<f64>(m_impl->cameraConfig.optics.sensorWidthPx) *
                       m_impl->cameraConfig.optics.sensorHeightPx;
    const f64 samples = static_cast<f64>(
        std::max(1u, m_impl->cameraLastAcquisitionSamples));
    const f64 perPixelSample = pixels * samples;
    if (haveRaygen) {
        report.transparentFraction =
            std::min(1.0, static_cast<f64>(raygenCounts[0]) / perPixelSample);
        const f64 specular =
            std::min(1.0, static_cast<f64>(raygenCounts[1]) / perPixelSample);
        // View-dependent pixels are the ones a depth reprojection between
        // strata cannot vouch for: disoccluded, specular, or transparent.
        const f64 disoccluded = haveDynamic
            ? static_cast<f64>((*dynamicCounts)[0]) / pixels : 0.0;
        report.viewDependentFraction = std::min(
            1.0, disoccluded + specular + report.transparentFraction);
    }
    if (haveDynamic)
        report.disoccludedFraction =
            std::min(1.0, static_cast<f64>((*dynamicCounts)[0]) / pixels);
    report.timeSampleCoverage =
        static_cast<f64>(m_impl->cameraLastStrataCount) /
        std::clamp<u32>(m_impl->cameraConfig.quality.gpuTimePositions, 1u,
                        rendercore::kCameraTimeStrataMax);
    report.objectMotionApproximation = m_impl->cameraLastObjectMotion;
    report.strataCount = m_impl->cameraLastStrataCount;
    report.valid = true;
    return report;
}

Result<camera::CameraOutput, String>
ExternalRenderContext::CaptureCameraProducts() {
    if (!m_impl->cameraCaptureCompleted || !m_impl->cameraGpuPipeline)
        return Result<camera::CameraOutput, String>::Err(
            "camera products require a completed submitted acquisition");
    const auto& config = m_impl->cameraConfig;
    if (config.products.tracedRadiance || config.products.cieLinearSrgb ||
        config.products.apparentTemperature)
        return Result<camera::CameraOutput, String>::Err(
            "the requested observer/temperature product is not available from "
            "the M3 GPU preview backend");
    const u32 width = config.optics.sensorWidthPx;
    const u32 height = config.optics.sensorHeightPx;
    const u32 channels = config.device.cfa == camera::CfaPattern::MultiChannel
        ? static_cast<u32>(config.device.channels.size()) : 1u;
    const auto outputs = m_impl->cameraGpuPipeline->GetOutputs();

    const auto readProductImage = [&](const GpuImage* source)
        -> Result<Image, String> {
        if (!source) return Result<Image, String>::Err(
            "GPU camera product image is unavailable");
        const auto rgba = CommandHelper::ReadbackImage(
            *m_impl->contextAdapter, source->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT, width, height);
        if (rgba.size() != static_cast<size_t>(width) * height * 4u)
            return Result<Image, String>::Err(
                "GPU camera product readback has the wrong size");
        Image image(width, height, channels);
        image.channelNames = channels == 1u
            ? std::vector<String>{"Y"}
            : std::vector<String>{"R", "G", "B"};
        for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel)
            for (u32 channel = 0; channel < channels; ++channel)
                image.data[pixel * channels + channel] = rgba[pixel * 4u + channel];
        return image;
    };
    const auto descriptor = [&](camera::SignalKind kind, String unit) {
        camera::SignalDescriptor signal;
        signal.kind = kind;
        signal.unit = std::move(unit);
        signal.responseProfileId = config.device.id.empty()
            ? "generic" : config.device.id;
        signal.calibration = config.device.calibration;
        signal.acquisitionIndex = m_impl->cameraAcquisitionIndex;
        signal.exposureStartSeconds = m_impl->cameraAcquisitionTimeSeconds -
            config.readout.exposureSeconds * 0.5;
        signal.exposureEndSeconds = m_impl->cameraAcquisitionTimeSeconds +
            config.readout.exposureSeconds * 0.5 +
            (config.readout.shutter == camera::ShutterKind::Rolling
                ? (height - 1u) * config.readout.rowDelaySeconds : 0.0);
        signal.cfa = config.device.cfa;
        signal.channelsPerPixel = channels;
        signal.responseMinNm = std::numeric_limits<f64>::infinity();
        signal.responseMaxNm = 0.0;
        for (const auto& channel : config.device.channels) {
            const auto& stack = channel.response;
            const auto& response = stack.systemResponse ? *stack.systemResponse :
                config.device.detector == camera::DetectorKind::Photon
                    ? *stack.quantumEfficiency : *stack.thermalAbsorptance;
            signal.responseMinNm = std::min(signal.responseMinNm, response.MinNm());
            signal.responseMaxNm = std::max(signal.responseMaxNm, response.MaxNm());
            if (config.device.cfa == camera::CfaPattern::MultiChannel) {
                signal.channelResponseIds.push_back(channel.name);
                signal.channelResponseSpanNm.push_back(
                    {response.MinNm(), response.MaxNm()});
            }
        }
        return signal;
    };
    const auto makeProduct = [&](const GpuImage* source,
                                 camera::SignalDescriptor signal)
        -> Result<camera::CameraProduct, String> {
        auto image = readProductImage(source);
        if (!image) return Result<camera::CameraProduct, String>::Err(image.error());
        camera::CameraProduct product{std::move(image.value()), std::move(signal)};
        if (auto annotated = camera::AnnotateProductMetadata(product); !annotated)
            return Result<camera::CameraProduct, String>::Err(annotated.error());
        product.image.metadata["camera_input_kind"] =
            config.inputKind == camera::CameraInputKind::FastRgbApproximation
                ? "fast_rgb_approximation" : "spectral_measurement";
        product.image.metadata["camera_random_seed"] =
            std::to_string(config.randomSeed);
        product.image.metadata["camera_effective_random_seed"] =
            std::to_string(camera::DeviceRandomSeed(config));
        if (m_impl->cameraLastStrataCount > 1) {
            // The band product integrates a moving exposure from T time
            // strata reprojected through the anchor depth: a preview
            // approximation, not the CPU reference's continuous integral.
            product.image.metadata["camera_dynamic_approximation"] =
                "time_strata=" +
                std::to_string(m_impl->cameraLastStrataCount);
        }
        return product;
    };

    camera::CameraOutput result;
    if (config.products.bandMeasurement) {
        auto measured = makeProduct(
            outputs.expectedElectrons,
            descriptor(camera::SignalKind::BandMeasurement,
                config.device.detector == camera::DetectorKind::Photon ? "e-/s" : "W"));
        if (!measured) return Result<camera::CameraOutput, String>::Err(measured.error());
        if (config.device.detector == camera::DetectorKind::Photon) {
            const f32 exposure = static_cast<f32>(config.readout.exposureSeconds);
            for (auto& value : measured.value().image.data) value /= exposure;
        }
        result.bandMeasurement = std::move(measured.value());
    }
    if (config.products.rawDn) {
        auto raw = makeProduct(outputs.rawDn,
            descriptor(camera::SignalKind::RawDN, "DN"));
        if (!raw) return Result<camera::CameraOutput, String>::Err(raw.error());
        result.rawDn = std::move(raw.value());
    }
    if (config.products.correctedDeviceSignal) {
        auto corrected = makeProduct(outputs.corrected,
            descriptor(camera::SignalKind::DeviceLinear,
                config.device.detector == camera::DetectorKind::Photon ? "e-" : "W"));
        if (!corrected)
            return Result<camera::CameraOutput, String>::Err(corrected.error());
        result.correctedDeviceSignal = std::move(corrected.value());
    }
    if (config.products.display) {
        // In the persistent-Clahe mode the on-screen display is the CLAHE
        // pipeline's output, not the camera chain's grey scalar; export what
        // the viewer sees.
        const GpuImage* displaySource =
            (m_impl->CameraClahePersistent() && m_impl->displayImage)
                ? m_impl->displayImage.get()
                : outputs.display;
        auto display = makeProduct(displaySource,
            descriptor(camera::SignalKind::DisplaySrgb, "encoded sRGB"));
        if (!display) return Result<camera::CameraOutput, String>::Err(display.error());
        result.display = std::move(display.value());
    }
    return result;
}

void ExternalRenderContext::SetGPUSensorEnabled(bool enabled) {
    camera::CameraConfig config = m_impl->cameraConfig;
    config.enabled = enabled;
    if (auto updated = SetCameraConfig(config); !updated)
        QL_LOG_ERROR("GPU camera enable failed: {}", updated.error());
}

void ExternalRenderContext::SetGPUSensorParams(const SensorParams& params) {
    m_impl->gpuSensorParams = params;

    // A fused IR band supplies its own photon-energy wavelength unless the host
    // asked for one. There is no "was it set" bit on SensorParams, so anything
    // other than the default counts as deliberate -- which is the same reading the
    // CLI takes from the config naming spectral.wavelength_nm or not.
    m_impl->gpuSensorWavelengthFromHost =
        params.wavelength_nm != SensorParams{}.wavelength_nm;

    // Translate the compatibility facade through the same migration as old
    // TOML, then run the one GPU camera implementation.
    const auto converted = CameraConfigFromSensorParams(
        params, m_impl->spectralMode, m_impl->width, m_impl->height,
        m_impl->camera.GetFovY(), m_impl->wavelength_nm);
    if (!converted) {
        QL_LOG_ERROR("GPU sensor parameter migration failed: {}", converted.error());
        return;
    }
    if (auto updated = SetCameraConfig(converted.value()); !updated)
        QL_LOG_ERROR("GPU sensor parameter update failed: {}", updated.error());
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

    // Determine source image: priority displayImage (CLAHE) -> camera display
    // -> outputImage. Each source carries its own extent: the CLAHE image
    // sits at the CLAHE input extent and the camera product at the physical
    // sensor array extent, which need not match the render extent.
    const GpuImage* sourceImage = m_impl->outputImage.get();
    u32 sourceWidth = m_impl->width;
    u32 sourceHeight = m_impl->height;

    // Priority 1: CLAHE output (includes all effects)
    if ((m_impl->displayParams.enabled || m_impl->CameraClahePersistent()) &&
        m_impl->claheInitialized && m_impl->displayImage &&
        m_impl->ClaheInputReady()) {
        sourceImage = m_impl->displayImage.get();
        const auto extent = m_impl->ClaheInputExtent();
        sourceWidth = extent.width;
        sourceHeight = extent.height;
        QL_LOG_DEBUG("CaptureDisplayImage: Using displayImage (CLAHE enabled)");
    }
    // Priority 2: Camera display product (physical sensor extent)
    else if (const GpuImage* camera = m_impl->CameraDisplayImage()) {
        sourceImage = camera;
        sourceWidth = m_impl->cameraConfig.optics.sensorWidthPx;
        sourceHeight = m_impl->cameraConfig.optics.sensorHeightPx;
        QL_LOG_DEBUG("CaptureDisplayImage: Using camera display (camera enabled)");
    }
    // Priority 3: Raw output
    else {
        QL_LOG_DEBUG("CaptureDisplayImage: Using outputImage (no post-processing)");
    }

    // Read back the appropriate image using CommandHelper
    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        *m_impl->contextAdapter,
        sourceImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        sourceWidth,
        sourceHeight
    );

    // Create Image from pixel data
    Image displayImage(sourceWidth, sourceHeight, 4);  // RGBA
    displayImage.data = std::move(pixels);
    displayImage.channelNames = {"R", "G", "B", "A"};

    // Add metadata
    displayImage.metadata["spectral_mode"] = std::to_string(static_cast<int>(m_impl->spectralMode));
    displayImage.metadata["wavelength_nm"] = std::to_string(m_impl->wavelength_nm);
    displayImage.metadata["accumulated_samples"] = std::to_string(m_impl->accumulatedSamples);
    displayImage.metadata["spp_target"] = std::to_string(m_impl->spp);
    // What was done to the pixels, recorded beside them: a display capture is
    // not radiometric, and the metadata is where that is admitted.
    displayImage.metadata["display_enhancement"] =
        m_impl->displayParams.enabled ? "true" : "false";
    if (m_impl->displayParams.enabled) {
        static constexpr const char* kToneNames[] = {"linear", "equalize", "clahe"};
        static constexpr const char* kPaletteNames[] = {"grey", "grey_inverted", "ironbow",
                                                        "rainbow", "viridis"};
        displayImage.metadata["display_tone_mode"] =
            kToneNames[std::min<usize>(static_cast<usize>(m_impl->displayParams.toneMode),
                                       std::size(kToneNames) - 1)];
        displayImage.metadata["display_palette"] =
            kPaletteNames[std::min<usize>(static_cast<usize>(m_impl->displayParams.palette),
                                          std::size(kPaletteNames) - 1)];
        displayImage.metadata["display_clip_limit"] =
            std::to_string(m_impl->displayParams.clipLimit);
        displayImage.metadata["display_tile_size"] =
            std::to_string(m_impl->displayParams.tileSize);
        displayImage.metadata["display_percentile_window"] =
            std::to_string(m_impl->displayParams.percentileLow) + ", " +
            std::to_string(m_impl->displayParams.percentileHigh);
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

void ExternalRenderContext::Impl::UpdateGpuResources(bool rebuildEmitters) {
    if (!scene) return;

    QL_LOG_INFO("Updating GPU resources...");
    materialBuffer = rendercore::BuildMaterialBuffer(
        *contextAdapter, *scene, wavelength_nm, materialGpuIndices);
    if (rebuildEmitters) RebuildEmissiveGeometry();
    QL_LOG_INFO("  GPU resources updated");
}

// The emitter list is world space, so it is invalidated by anything that moves
// a node -- not only by loading a scene. RefitAccelerationStructure calls this
// for the same reason it re-uploads the TLAS instances.
//
// Rebinds the descriptor itself rather than leaving that to CreatePipeline.
// Most callers do go on to build a pipeline, but SetWavelength does not: it
// calls UpdateGpuResources to refresh the material buffer and then rebinds only
// that. Allocating a new buffer here and returning would leave binding 23
// pointing at the freed one.
void ExternalRenderContext::Impl::RebuildEmissiveGeometry() {
    if (!scene) return;

    const auto triangles = enableLightSampling
        ? rendercore::CollectEmissiveTriangles(*scene)
        : Vector<rendercore::EmissiveTriangleGPU>{};
    const size_t bytes = std::max(size_t{1}, triangles.size()) *
                         sizeof(rendercore::EmissiveTriangleGPU);
    const bool moved = !emissiveTriangleBuffer || emissiveTriangleBuffer->GetSize() < bytes;
    if (moved) {
        emissiveTriangleBuffer =
            rendercore::CreateEmissiveTriangleBuffer(*contextAdapter, triangles);
    } else {
        const rendercore::EmissiveTriangleGPU empty{};
        const void* data = triangles.empty() ? static_cast<const void*>(&empty)
                                             : static_cast<const void*>(triangles.data());
        emissiveTriangleBuffer->Upload(data, bytes);
    }

    f32 totalPower = 0.0f;
    if (!triangles.empty()) {
        totalPower = triangles.back().cumulativePower;  // the CDF's last entry
    }
    lightingParams.emissiveTriangleCount = static_cast<u32>(triangles.size());
    lightingParams.emissiveTotalPower = totalPower;
    UploadLightingParams();

    if (pipeline && moved) {
        pipeline->BindEmissiveTriangleBuffer(*emissiveTriangleBuffer);
    }
    emissiveTransformDirty = false;
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

    cieCmfBuffer = rendercore::CreateCieColourMatchingBuffer(*contextAdapter);
    rgbToSpectrumBuffer = rendercore::CreateRgbToSpectrumBuffer(*contextAdapter);

    // No scene yet, so no emitters -- but the descriptor still has to be
    // written, and CreatePipeline can run before any scene is loaded.
    emissiveTriangleBuffer = rendercore::CreateEmissiveTriangleBuffer(*contextAdapter, {});
    {
        const f32 zero = 0.0f;
        thermalTemperatureBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), sizeof(f32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        thermalTemperatureBuffer->Upload(&zero, sizeof(zero));
    }
    UploadThermalSunResponse({}, {}, glm::vec3(0.0f));
    // Binding 27 the same way: a descriptor the shader reads has to be written
    // whether or not anything is previewing, and a step of zero is what "not
    // previewing" looks like from the shader's side.
    UploadThermalTangent({}, 0.0f);
    thermalPreview = std::make_unique<rendercore::ThermalPreview>(*contextAdapter);
    thermalPreview->SetEpochProgressCallback(thermalEpochProgress);
}

void ExternalRenderContext::Impl::CreateBRDFLut() {
    brdfLut = rendercore::BrdfLut::Create(*contextAdapter);
}

void ExternalRenderContext::Impl::CreateFallbackEnvMap() {
    envMap = rendercore::EnvironmentCubemap::Fallback(*contextAdapter);
}

void ExternalRenderContext::Impl::CreatePipeline() {
    if (pipelineCache == VK_NULL_HANDLE) {
        pipelineCache = RayTracingPipeline::LoadPipelineCache(*contextAdapter,
                                                              pipelineCachePath);
    }

    rendercore::PipelineBindings bindings;
    bindings.outputImage = outputImage.get();
    bindings.depthImage = depthAovImage.get();
    bindings.geometry = &geometry;
    bindings.lightingParams = lightingParamsBuffer.get();
    bindings.materials = materialBuffer.get();
    bindings.textures = textureManager.get();
    bindings.environment = &envMap;
    bindings.brdfLut = &brdfLut;
    bindings.spectralCurves = spectralCurvesBuffer.get();
    bindings.complexRefractiveIndex = criBuffer.get();
    bindings.solarLut = solarLutBuffer.get();
    bindings.atmosphereHeader = atmosHeaderBuffer.get();
    bindings.atmosphereData = atmosDataBuffer.get();
    bindings.cieColourMatching = cieCmfBuffer.get();
    bindings.rgbToSpectrum = rgbToSpectrumBuffer.get();
    bindings.emissiveTriangles = emissiveTriangleBuffer.get();
    bindings.thermalTemperatures = thermalTemperatureBuffer.get();
    bindings.thermalSunResponse = thermalSunResponseBuffer.get();
    bindings.thermalParameterTangent = thermalParameterTangentBuffer.get();

    pipeline = rendercore::CreateRayTracingPipeline(*contextAdapter, pipelineCache,
                                                    bindings);
}

Result<void, String> ExternalRenderContext::Impl::UpdateCameraAtmosphere() {
    cameraAtmosMinNm = 0.0f;
    cameraAtmosStepNm = 0.0f;
    cameraAtmosCount = 0;
    if (!atmosphereConfig.enabled) {
        cameraAtmosHeaderBuffer.reset();
        cameraAtmosDataBuffer.reset();
        cameraBaselineAtmosHeaderBuffer.reset();
        cameraBaselineAtmosDataBuffer.reset();
        return Result<void, String>::Ok();
    }
    if (!atmosModelPack)
        return Result<void, String>::Err(
            "camera atmosphere is enabled but its model pack is unavailable");
    f64 lo = std::numeric_limits<f64>::infinity(), hi = 0.0;
    for (const auto& channel : cameraConfig.device.channels) {
        const auto& stack = channel.response;
        const auto& curve = stack.systemResponse ? *stack.systemResponse :
            cameraConfig.device.detector == camera::DetectorKind::Photon ?
            *stack.quantumEfficiency : *stack.thermalAbsorptance;
        lo = std::min(lo, curve.MinNm());
        hi = std::max(hi, curve.MaxNm());
    }
    const auto firstBand = RenderBandLambdaGrid(SpectralMode::Single, lo);
    const auto lastBand = RenderBandLambdaGrid(SpectralMode::Single, hi);
    if (firstBand.band.empty() || firstBand.band != lastBand.band)
        return Result<void, String>::Err(
            "GPU camera response crosses unsupported NN atmosphere bands");
    constexpr u32 kCameraAtmosSamples = 128;
    std::vector<f64> wavelengths(kCameraAtmosSamples);
    for (u32 i = 0; i < kCameraAtmosSamples; ++i)
        wavelengths[i] = lo + (hi - lo) * i / (kCameraAtmosSamples - 1);
    // A single NN band must cover the entire uniform grid. The shader reads
    // the nearest baked bin mean, never interpolates those means.
    for (f64 nm : wavelengths)
        if (RenderBandLambdaGrid(SpectralMode::Single, nm).band != firstBand.band)
            return Result<void, String>::Err(
                "GPU camera atmosphere has a wavelength coverage gap");
    AtmosphereNNConfig effective = atmosphereConfig;
    const glm::vec3 sunDir = lightingParams.sunDirection;
    if (effective.sunFromLighting && glm::length(sunDir) > 1e-6f) {
        const glm::vec3 sun = glm::normalize(sunDir);
        effective.sunZenithDeg = glm::degrees(std::acos(std::clamp(sun.y, -1.0f, 1.0f)));
        effective.sunAzimuthDeg = glm::degrees(std::atan2(sun.x, sun.z));
    }
    if (effective.h1FromCamera) {
        const f32 metersPerUnit = lightingParams.worldUnitsToMeters > 0.0f ?
                                  lightingParams.worldUnitsToMeters : 1.0f;
        f64 cameraHeightWorld = camera.GetPosition().y;
        if (!cameraConfig.motion.keys.empty()) {
            const auto pose = camera::CameraPoseAt(
                cameraConfig.motion, cameraAcquisitionTimeSeconds);
            if (!pose) return Result<void, String>::Err(pose.error());
            cameraHeightWorld = pose.value().position[1];
        }
        effective.h1Km = std::max(
            cameraHeightWorld * metersPerUnit / 1000.0, 0.0);
    }
    try {
        AtmosphereBaker baker(*atmosModelPack);
        auto baked = baker.Bake(effective, firstBand.band, wavelengths, 0.0);
        const glm::vec3 normalizedSun = glm::length(sunDir) > 1e-6f ?
                                        glm::normalize(sunDir) : glm::vec3(0, 1, 0);
        baked.header.sunDirWorld[0] = normalizedSun.x;
        baked.header.sunDirWorld[1] = normalizedSun.y;
        baked.header.sunDirWorld[2] = normalizedSun.z;
        baked.header.worldUnitsToMeters =
            lightingParams.worldUnitsToMeters > 0.0f ?
            lightingParams.worldUnitsToMeters : 1.0f;
        cameraAtmosHeaderBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), sizeof(AtmosNNHeaderGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        cameraAtmosHeaderBuffer->Upload(&baked.header, sizeof(baked.header));
        cameraAtmosDataBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), baked.data.size() * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        cameraAtmosDataBuffer->Upload(
            baked.data.data(), baked.data.size() * sizeof(f32));
        cameraAtmosMinNm = static_cast<f32>(lo);
        cameraAtmosStepNm = static_cast<f32>(
            (hi - lo) / (kCameraAtmosSamples - 1));
        cameraAtmosCount = kCameraAtmosSamples;
        const f64 middleNm = 0.5 * (lo + hi);
        const auto middleGrid = RenderBandLambdaGrid(
            SpectralMode::Single, middleNm);
        auto baseline = baker.Bake(
            effective, middleGrid.band, middleGrid.lambdasNm,
            middleGrid.windowHalfWidthNm);
        baseline.header.sunDirWorld[0] = normalizedSun.x;
        baseline.header.sunDirWorld[1] = normalizedSun.y;
        baseline.header.sunDirWorld[2] = normalizedSun.z;
        baseline.header.worldUnitsToMeters = baked.header.worldUnitsToMeters;
        cameraBaselineAtmosHeaderBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), sizeof(AtmosNNHeaderGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        cameraBaselineAtmosHeaderBuffer->Upload(
            &baseline.header, sizeof(baseline.header));
        cameraBaselineAtmosDataBuffer = std::make_unique<GpuBuffer>(
            contextAdapter->GetAllocator(), baseline.data.size() * sizeof(f32),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        cameraBaselineAtmosDataBuffer->Upload(
            baseline.data.data(), baseline.data.size() * sizeof(f32));
        return Result<void, String>::Ok();
    } catch (const std::exception& e) {
        return Result<void, String>::Err(
            String("camera atmosphere bake failed: ") + e.what());
    }
}

Result<void, String> ExternalRenderContext::Impl::EnsureCameraResources() {
    if (!cameraConfig.enabled)
        return Result<void, String>::Err("camera capture is disabled");
    if (!contextAdapter || !pipeline)
        return Result<void, String>::Err("camera rendering pipeline is not ready");
    if (!cameraResourcesDirty && cameraMeasurementImage && cameraGpuPipeline)
        return Result<void, String>::Ok();
    const auto valid = camera::ValidateCameraConfig(cameraConfig);
    if (!valid) return Result<void, String>::Err(valid.error());
    // Configuration changes happen between frames; wait before replacing
    // descriptor-backed images/buffers that a submitted frame may still use.
    vkDeviceWaitIdle(device);
    const u32 physicalW = cameraConfig.optics.sensorWidthPx;
    const u32 physicalH = cameraConfig.optics.sensorHeightPx;
    const auto allocator = contextAdapter->GetAllocator();
    const auto makeImage = [&](VkFormat format) {
        auto image = std::make_unique<GpuImage>(
            allocator, device, physicalW, physicalH, format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        TransitionImageLayoutImmediate(
            image->GetImage(), format, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL);
        return image;
    };
    try {
        cameraMeasurementImage = std::make_unique<GpuImage>(
            allocator, device, physicalW, physicalH,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, 1u, rendercore::kCameraTimeStrataMax,
            0, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        cameraMeasurementDepthImage = std::make_unique<GpuImage>(
            allocator, device, physicalW, physicalH, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, 1u, rendercore::kCameraTimeStrataMax,
            0, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        CommandHelper::TransitionImageLayoutImmediate(
            *contextAdapter, cameraMeasurementImage->GetImage(),
            VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, 1u, rendercore::kCameraTimeStrataMax);
        CommandHelper::TransitionImageLayoutImmediate(
            *contextAdapter, cameraMeasurementDepthImage->GetImage(),
            VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, 1u, rendercore::kCameraTimeStrataMax);
        cameraBaselineImage = makeImage(VK_FORMAT_R32G32B32A32_SFLOAT);
        cameraDepthImage = makeImage(VK_FORMAT_R32_SFLOAT);
    } catch (const std::exception& e) {
        return Result<void, String>::Err(
            String("camera image allocation failed: ") + e.what());
    }
    if (!cameraMeasurementImage->IsValid() ||
        !cameraMeasurementDepthImage->IsValid())
        return Result<void, String>::Err("camera image allocation failed");
    if (auto baked = UpdateCameraAtmosphere(); !baked)
        return baked;
    const auto encoded = camera::EncodeCameraResponseGpu(
        cameraConfig, cameraAtmosMinNm, cameraAtmosStepNm, cameraAtmosCount);
    if (!encoded) return Result<void, String>::Err(encoded.error());
    const VkDeviceSize bytes = encoded.value().size() * sizeof(u32);
    cameraResponseBuffer = std::make_unique<GpuBuffer>(
        allocator, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    cameraResponseBuffer->Upload(encoded.value().data(), bytes);
    pipeline->BindCameraMeasurementImage(*cameraMeasurementImage);
    pipeline->BindCameraResponseBuffer(*cameraResponseBuffer);
    pipeline->BindCameraMeasurementDepthImage(*cameraMeasurementDepthImage);
    cameraDynamicCounterBuffer = std::make_unique<GpuBuffer>(
        allocator, 4u * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    if (!cameraDynamicCounterBuffer->IsValid())
        return Result<void, String>::Err(
            "camera dynamic counter allocation failed");
    CommandHelper::ExecuteImmediate(*contextAdapter, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, cameraDynamicCounterBuffer->GetHandle(), 0,
                        VK_WHOLE_SIZE, 0);
    });
    pipeline->BindCameraDynamicCounterBuffer(*cameraDynamicCounterBuffer);
    pipeline->BindCameraDepthImage(*cameraDepthImage);
    pipeline->BindCameraObserverOutputImage(*cameraBaselineImage);
    pipeline->BindCameraObserverDepthImage(*cameraDepthImage);
    pipeline->BindCameraAtmosphereNN(
        cameraAtmosHeaderBuffer ? cameraAtmosHeaderBuffer.get() :
                                  atmosHeaderBuffer.get(),
        cameraAtmosDataBuffer ? cameraAtmosDataBuffer.get() :
                                atmosDataBuffer.get());
    pipeline->BindCameraObserverAtmosphereNN(
        cameraBaselineAtmosHeaderBuffer ? cameraBaselineAtmosHeaderBuffer.get() :
                                          atmosHeaderBuffer.get(),
        cameraBaselineAtmosDataBuffer ? cameraBaselineAtmosDataBuffer.get() :
                                        atmosDataBuffer.get());
    auto gpu = rendercore::GpuCameraPipeline::Create(
        *contextAdapter, physicalW, physicalH);
    if (!gpu) return Result<void, String>::Err(gpu.error());
    if (auto configured = gpu.value()->Configure(cameraConfig); !configured)
        return configured;
    cameraGpuPipeline = std::move(gpu.value());
    if (cameraTimingPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, cameraTimingPool, nullptr);
        cameraTimingPool = VK_NULL_HANDLE;
    }
    VkQueryPoolCreateInfo timingInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    timingInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    timingInfo.queryCount = rendercore::GpuCameraPipeline::kTimingQueryCount;
    if (vkCreateQueryPool(device, &timingInfo, nullptr,
                          &cameraTimingPool) != VK_SUCCESS)
        return Result<void, String>::Err("cannot create camera timing pool");
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(contextAdapter->GetPhysicalDevice(),
                                  &properties);
    cameraTimestampPeriodNs = static_cast<f64>(properties.limits.timestampPeriod);
    cameraStratumSampleCounts.fill(0);
    cameraStratumRecordedAcquisition = std::numeric_limits<u64>::max();
    cameraResourcesDirty = false;
    cameraCaptureCompleted = false;
    return Result<void, String>::Ok();
}

Result<CameraData, String> ExternalRenderContext::Impl::CameraDataForCapture(
    SpectralMode mode, f64 wavelengthNm, f64 timeSeconds) const {
    Camera poseCamera = camera;
    // NaN means "the committed acquisition time"; explicit times serve the
    // time-stratified trace, where each layer poses the camera at its own
    // instant inside the exposure window.
    const f64 poseTime = std::isnan(timeSeconds)
                             ? cameraAcquisitionTimeSeconds
                             : timeSeconds;
    if (!cameraConfig.motion.keys.empty()) {
        const auto pose = camera::CameraPoseAt(
            cameraConfig.motion, poseTime);
        if (!pose) return Result<CameraData, String>::Err(pose.error());
        poseCamera = Camera(
            glm::vec3(static_cast<f32>(pose.value().position[0]),
                      static_cast<f32>(pose.value().position[1]),
                      static_cast<f32>(pose.value().position[2])),
            glm::vec3(static_cast<f32>(pose.value().lookAt[0]),
                      static_cast<f32>(pose.value().lookAt[1]),
                      static_cast<f32>(pose.value().lookAt[2])),
            camera.GetUpReference(), camera.GetFovY(), camera.GetAspectRatio());
        poseCamera.SetProjection(camera.GetProjection());
        poseCamera.SetOrthoHeight(camera.GetOrthoHeight());
    }
    if (poseCamera.GetProjection() != Camera::Projection::Perspective)
        return Result<CameraData, String>::Err(
            "physical camera capture requires a perspective projection");
    const auto& optics = cameraConfig.optics;
    const auto fovX = camera::HorizontalFovRadians(
        optics.focalLengthMm, optics.pixelPitchUm, optics.sensorWidthPx);
    if (!fovX) return Result<CameraData, String>::Err(fovX.error());
    CameraData data = poseCamera.GetCameraData();
    const f64 aspect = static_cast<f64>(optics.sensorWidthPx) /
                       optics.sensorHeightPx;
    data.aspectRatio = static_cast<f32>(aspect);
    data.fovScale = static_cast<f32>(std::tan(fovX.value() / 2.0) / aspect);
    data.spectral_mode = static_cast<u32>(mode);
    data.wavelength_nm = static_cast<f32>(wavelengthNm);
    data.debug_mode = 0;
    data.debugParam = 0;
    return data;
}

Result<bool, String> ExternalRenderContext::Impl::StepCameraAcquisition(
    VkCommandBuffer cmd) {
    auto& s = cameraScheduler;
    if (!s.drivesAcquisition) {
        s.drivesAcquisition = true;
        // Continue the index from the explicit path if a host queued
        // acquisitions before the viewport took over; GpuCameraPipeline
        // requires monotonic ticks.
        s.nextAcquisitionIndex = cameraAcquisitionIndex;
        s.nextAcquisitionTimeSeconds = timeline.Current_s();
    }

    if (s.historyResetPending) {
        // Applied here, between frames: the reset discards GPU detector
        // history, which is only safe once submitted work has drained. The
        // reset frame records no acquisition.
        if (cameraGpuPipeline) cameraGpuPipeline->ResetState();
        s.historyResetPending = false;
        s.historyEpoch += 1;
        s.hasCommittedAcquisition = false;
        s.lastCommittedAcquisitionTime = 0.0;
        cameraAcquisitionIndex = 0;
        cameraCapturePending = false;
        cameraCaptureRecorded = false;
        cameraCaptureCompleted = false;
        s.nextAcquisitionIndex = 0;
        s.nextAcquisitionTimeSeconds = timeline.Current_s();
        // The detector history is gone, so the AE/AWB feedback is too: the
        // next epoch starts from the authored manual values. The zeroed
        // statistics buffer is a valid (empty) measurement.
        cameraAutoControlDue = true;
        cameraAutoLastAcquisition = std::numeric_limits<u64>::max();
        cameraAutoAppliedAcquisition = std::numeric_limits<u64>::max();
        cameraHostState = camera::CaptureState{};
        return Result<bool, String>(false);
    }

    const f64 now = timeline.Current_s();
    if (now + 1e-12 >= s.nextAcquisitionTimeSeconds) {
        // The clock reached the next scheduled slot: pose the scene at the
        // slot and commit one acquisition there.
        const f64 slot = s.nextAcquisitionTimeSeconds;
        if (auto moved = owner->SetTimelineTime(slot); !moved) {
            return Result<bool, String>(
                typename Result<bool, String>::Err(moved.error()));
        }
        cameraAcquisitionIndex = s.nextAcquisitionIndex;
        cameraAcquisitionTimeSeconds = slot;
        cameraCapturePending = true;
        cameraCaptureRecorded = false;
        if (auto recorded = RecordCameraMeasurement(cmd); !recorded) {
            cameraCapturePending = false;
            return Result<bool, String>(
                typename Result<bool, String>::Err(recorded.error()));
        }
        cameraCapturePending = false;
        cameraCaptureRecorded = false;
        cameraCaptureCompleted = true;
        // The committed tick's statistics wait on the device for the loop.
        cameraAutoControlDue = true;
        s.nextAcquisitionIndex += 1;
        s.nextAcquisitionTimeSeconds = slot + s.framePeriodSeconds;
        s.hasCommittedAcquisition = true;
        s.lastCommittedAcquisitionTime = slot;
        return Result<bool, String>(true);
    }

    if (s.hasCommittedAcquisition) {
        // Same tick, more samples: convergence without state advancement.
        // The acquisition keeps the committed index and time; the scene pose
        // is wherever the host left the clock.
        cameraCapturePending = true;
        cameraCaptureRecorded = false;
        if (auto recorded = RecordCameraMeasurement(cmd); !recorded) {
            cameraCapturePending = false;
            return Result<bool, String>(
                typename Result<bool, String>::Err(recorded.error()));
        }
        cameraCapturePending = false;
        cameraCaptureRecorded = false;
        cameraCaptureCompleted = true;
    }
    return Result<bool, String>(false);
}

camera::CameraConfig ExternalRenderContext::Impl::EffectiveCameraConfig() const {
    camera::CameraConfig effective = cameraConfig;
    const camera::IspConfig& isp = cameraConfig.isp;
    const bool fresh = cameraHostState.nextExposureSeconds <= 0.0;
    if (isp.autoExposure && !fresh) {
        effective.readout.exposureSeconds = cameraHostState.nextExposureSeconds;
        if (cameraHostState.nextAnalogGain > 0.0)
            effective.readout.analogGain = cameraHostState.nextAnalogGain;
    }
    if (isp.autoWhiteBalance && !fresh)
        effective.isp.whiteBalance = cameraHostState.nextWhiteBalance;
    return effective;
}

void ExternalRenderContext::Impl::MaybeRunCameraAutoControl() {
    if (!cameraConfig.isp.autoExposure && !cameraConfig.isp.autoWhiteBalance)
        return;
    if (!cameraAutoControlDue || !cameraGpuPipeline) return;
    if (cameraAcquisitionIndex == cameraAutoLastAcquisition) return;
    cameraAutoControlDue = false;
    cameraAutoLastAcquisition = cameraAcquisitionIndex;
    RunCameraAutoControl();
}

void ExternalRenderContext::Impl::RunCameraAutoControl() {
    const camera::IspConfig& isp = cameraConfig.isp;
    const bool photon =
        cameraConfig.device.detector == camera::DetectorKind::Photon;
    if (!photon) {
        // A thermal detector responds to absorbed power, not scene luminance:
        // AE has no physical meaning there, exactly as on the CPU chain.
        if (!cameraAutoThermalNoted) {
            cameraAutoThermalNoted = true;
            QL_LOG_WARN(
                "camera auto_exposure/auto_white_balance ignored: the thermal "
                "detector responds to absorbed power, not scene luminance");
        }
        return;
    }
    const auto stats = cameraGpuPipeline->ReadIspStats();
    if (!stats) {
        QL_LOG_WARN("camera auto control cannot read statistics: {}",
                    stats.error());
        return;
    }
    const auto& measured = stats.value();
    camera::AutoControlState previous;
    const bool fresh = cameraHostState.nextExposureSeconds <= 0.0;
    previous.exposure = fresh ? cameraConfig.readout.exposureSeconds
                              : cameraHostState.nextExposureSeconds;
    previous.analogGain = fresh ? cameraConfig.readout.analogGain
                                : cameraHostState.nextAnalogGain;
    previous.whiteBalance = fresh ? cameraConfig.isp.whiteBalance
                                  : cameraHostState.nextWhiteBalance;
    camera::AutoControlInput input;
    input.lumaMean = measured.meanValue;
    const f64 total = static_cast<f64>(measured.saturatedCount) +
                      static_cast<f64>(measured.unsaturatedCount);
    input.saturatedFraction = total > 0.0
        ? static_cast<f64>(measured.saturatedCount) / total : 0.0;
    for (u32 c = 0; c < 3u; ++c)
        input.channelMeans[c] = measured.channelCounts[c] > 0u
            ? static_cast<f64>(measured.channelSums[c]) /
                  static_cast<f64>(measured.channelCounts[c])
            : 0.0;
    const camera::AutoControlState next = camera::StepAutoControl(
        isp, previous, input, isp.autoExposure, isp.autoWhiteBalance);
    cameraHostState.nextExposureSeconds = next.exposure;
    cameraHostState.nextAnalogGain = next.analogGain;
    cameraHostState.nextWhiteBalance = next.whiteBalance;
}

Result<void, String> ExternalRenderContext::Impl::RecordCameraDisplayReprocess(
    VkCommandBuffer cmd) {
    if (!cameraGpuPipeline || cmd == VK_NULL_HANDLE)
        return Result<void, String>::Err(
            "camera display reprocess needs the GPU camera pipeline");
    return cameraGpuPipeline->RecordDisplayReprocess(
        cmd, EffectiveCameraConfig());
}

Result<void, String> ExternalRenderContext::Impl::RecordCameraMeasurement(
    VkCommandBuffer cmd) {
    if (!cameraCapturePending)
        return Result<void, String>::Err("no camera acquisition is queued");
    if (auto ready = EnsureCameraResources(); !ready) return ready;
    // M4-4: settle the previous committed tick's AE/AWB feedback first. The
    // statistics readback is queue-ordered after the last submitted frame,
    // so the feedback the new tick consumes always describes the tick that
    // came before it. A same-tick re-record finds the index already marked
    // and leaves the loop alone.
    MaybeRunCameraAutoControl();
    const camera::CameraConfig captureConfig = EffectiveCameraConfig();
    if (cameraAcquisitionIndex != cameraAutoAppliedAcquisition) {
        if (auto applied = cameraGpuPipeline->ApplyEffectiveConfig(captureConfig);
            !applied)
            return applied;
        cameraAutoAppliedAcquisition = cameraAcquisitionIndex;
    }
    // The committed acquisition's operating point becomes the previous value
    // of the next controller step, exactly as CpuCameraPipeline's write-back.
    // From the first capture on, the state is no longer fresh.
    cameraHostState.nextExposureSeconds = captureConfig.readout.exposureSeconds;
    cameraHostState.nextAnalogGain = captureConfig.readout.analogGain;
    cameraHostState.nextWhiteBalance = captureConfig.isp.whiteBalance;
    const bool fastRgb = captureConfig.inputKind ==
        camera::CameraInputKind::FastRgbApproximation;
    const SpectralMode captureMode = fastRgb
        ? SpectralMode::RGB : SpectralMode::CameraMeasurement;
    const f64 centerNm = 0.5 * (cameraConfig.device.effectiveMinNm +
                                cameraConfig.device.effectiveMaxNm);
    const u32 effectiveSeed = camera::DeviceRandomSeed(cameraConfig);
    const u32 randomSeed = camera::CounterRandomU32(
        effectiveSeed, 0u, cameraAcquisitionIndex,
        camera::NoiseClass::EmpiricalEffect, 0u);
    if (cameraFullPerf) cameraFullPerf->BeginFrame(cmd);
    if (cameraTracePerf) cameraTracePerf->BeginFrame(cmd);
    const u32 physicalW = cameraConfig.optics.sensorWidthPx;
    const u32 physicalH = cameraConfig.optics.sensorHeightPx;
    // Raygen progressively averages the device rate in binding 28. These
    // wavelength/path samples belong to one acquisition: the detector sees
    // their mean and draws noise exactly once after the loop.
    // CPU wavelengthSamples controls deterministic quadrature refinement.
    // GPU preview draws one response-distributed wavelength per path, so its
    // convergence knob is the viewport SPP. Reusing the CPU default (32) here
    // would launch 32 full 1080p traces for one preview acquisition.
    const u32 spectralSamples = fastRgb ? 1u : std::max(spp, 1u);

    // M4-1 temporal stratification: cut the exposure window into T strata and
    // trace each batch of samples at its own instant, the scene posed by the
    // timeline and the camera by the motion track at that instant. The
    // viewport pose is restored when the loop ends, on every exit path.
    u32 strataCount = 1;
    std::vector<f64> stratumTimes;
    if (!fastRgb) {
        const u32 requested = std::clamp<u32>(
            cameraConfig.quality.gpuTimePositions, 1u,
            rendercore::kCameraTimeStrataMax);
        const bool anythingMoves = !cameraConfig.motion.keys.empty() ||
                                   timeline.HasMotion();
        strataCount = anythingMoves
                          ? std::min<u32>(requested, spectralSamples)
                          : 1u;
        if (cameraStratumRecordedAcquisition != cameraAcquisitionIndex) {
            cameraStratumSampleCounts.fill(0);
            cameraStratumRecordedAcquisition = cameraAcquisitionIndex;
        }
        // Zero the transparent/specular counters for this acquisition.
        vkCmdFillBuffer(cmd, cameraDynamicCounterBuffer->GetHandle(), 0,
                        VK_WHOLE_SIZE, 0);
        VkBufferMemoryBarrier zeroed{};
        zeroed.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        zeroed.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        zeroed.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                               VK_ACCESS_SHADER_WRITE_BIT;
        zeroed.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zeroed.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        zeroed.buffer = cameraDynamicCounterBuffer->GetHandle();
        zeroed.offset = 0;
        zeroed.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
                             0, nullptr, 1, &zeroed, 0, nullptr);
    }
    stratumTimes = camera::ExposureStratumTimes(
        cameraAcquisitionTimeSeconds, captureConfig.readout.exposureSeconds,
        strataCount);
    const f64 restoreTime = timeline.Current_s();
    struct TimelineRestore {
        Impl* self;
        f64 time;
        ~TimelineRestore() { self->ApplyTimelinePose(time); }
    } restoreGuard{this, restoreTime};

    if (fastRgb) pipeline->SetUseCameraObserverSet(true);
    pipeline->SetSpecConstants(static_cast<u32>(captureMode), false);
    std::array<rendercore::DynamicLayerCamera,
               rendercore::kCameraTimeStrataMax> layerCameras{};
    u32 totalTraced = 0;
    for (u32 stratum = 0; stratum < strataCount; ++stratum) {
        const f64 stratumTime = stratumTimes[stratum];
        // Pose the scene directly rather than through SetTimelineTime: the
        // strata around an acquisition reach back before the committed tick,
        // which the public scrub path would read as a history-invalidating
        // rewind (and it would remap the thermal hour, which the trace must
        // not touch).
        ApplyTimelinePose(stratumTime);
        const auto pose = CameraDataForCapture(captureMode, centerNm,
                                               stratumTime);
        if (!pose) return Result<void, String>::Err(pose.error());
        pipeline->SetCameraData(pose.value());
        pipeline->SetTimeStratum(stratum, strataCount);
        const auto& cameraData = pose.value();
        auto& frame = layerCameras[stratum];
        frame.origin = {cameraData.origin.x, cameraData.origin.y,
                        cameraData.origin.z, 0.0f};
        frame.forward = {cameraData.forward.x, cameraData.forward.y,
                         cameraData.forward.z, 0.0f};
        frame.right = {cameraData.right.x, cameraData.right.y,
                       cameraData.right.z, 0.0f};
        frame.up = {cameraData.up.x, cameraData.up.y, cameraData.up.z, 0.0f};
        frame.params = {cameraData.fovScale, cameraData.aspectRatio,
                        static_cast<f32>(stratumTime), 0.0f};

        // Round-robin with the remainder to the earlier strata: each stratum
        // gets spectralSamples/T paths, one layer of the measurement array
        // per stratum, averaged progressively there. The running per-layer
        // count keeps a same-tick re-record converging where the previous
        // record left off instead of restarting the average.
        const u32 base = spectralSamples / strataCount;
        const u32 extra = spectralSamples % strataCount;
        const u32 stratumSamples = base + (stratum < extra ? 1u : 0u);
        const u32 running = cameraStratumSampleCounts[stratum];
        for (u32 sample = 0; sample < stratumSamples; ++sample) {
            const u32 sampleIndex = running + sample;
            pipeline->SetSamplingParams(
                static_cast<u32>(cameraAcquisitionIndex), sampleIndex,
                stratumSamples,
                randomSeed ^ (sampleIndex * 0x9e3779b9u), effectiveSeed);
            pipeline->TraceRays(cmd, physicalW, physicalH);
            if (sample + 1u < stratumSamples || stratum + 1u < strataCount) {
                VkMemoryBarrier accumulation{};
                accumulation.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                accumulation.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                accumulation.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                             VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(
                    cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
                    1, &accumulation, 0, nullptr, 0, nullptr);
            }
        }
        cameraStratumSampleCounts[stratum] = running + stratumSamples;
        totalTraced += stratumSamples;
    }
    if (fastRgb) pipeline->SetUseCameraObserverSet(false);
    if (cameraTracePerf) cameraTracePerf->EndFrame(cmd);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        1, &barrier, 0, nullptr, 0, nullptr);

    // Object-motion approximation for the report: the mean adjacent-stratum
    // displacement of the scene's animated nodes, measured on the trajectory
    // the strata actually traced. Camera-only motion reads 0 here; the
    // reprojection knows the camera moved because the host tells it.
    cameraLastObjectMotion = 0.0;
    if (strataCount > 1 && timeline.HasMotion()) {
        f64 accumulated = 0.0;
        u64 pairs = 0;
        for (const auto& animated : timeline.Animated()) {
            for (u32 k = 0; k + 1 < strataCount; ++k) {
                const glm::vec3 a = glm::vec3(
                    timeline.PoseAt(animated, stratumTimes[k])[3]);
                const glm::vec3 b = glm::vec3(
                    timeline.PoseAt(animated, stratumTimes[k + 1])[3]);
                accumulated += static_cast<f64>(glm::length(b - a));
                ++pairs;
            }
        }
        if (pairs > 0) cameraLastObjectMotion = accumulated / pairs;
    }

    rendercore::DynamicExposureInput dynamic;
    const auto& readoutConfig = captureConfig.readout;
    const bool rolling = readoutConfig.shutter ==
                         camera::ShutterKind::Rolling;
    dynamic.strataRate = cameraMeasurementImage.get();
    dynamic.strataDepth = cameraMeasurementDepthImage.get();
    dynamic.layerCameras = layerCameras.data();
    dynamic.strataCount = strataCount;
    dynamic.firstRowMidSeconds = cameraAcquisitionTimeSeconds;
    dynamic.exposureSeconds = readoutConfig.exposureSeconds;
    dynamic.rowDelaySeconds = rolling ? readoutConfig.rowDelaySeconds : 0.0;
    const rendercore::GpuCameraPipeline::TimingQueries timing{
        cameraTimingPool, 0u};
    const auto readout = fastRgb
        ? cameraGpuPipeline->RecordFastRgbMeasurement(
              cmd, *cameraBaselineImage, cameraAcquisitionIndex,
              cameraAcquisitionTimeSeconds)
        : cameraGpuPipeline->RecordMeasurement(
              cmd, *cameraMeasurementImage, cameraAcquisitionIndex,
              cameraAcquisitionTimeSeconds, timing, dynamic);
    if (cameraFullPerf) cameraFullPerf->EndFrame(cmd);
    if (!readout) return readout;
    cameraLastAcquisitionSamples = totalTraced;
    cameraLastStrataCount = strataCount;
    cameraCaptureRecorded = true;
    return Result<void, String>::Ok();
}

Result<void, String> ExternalRenderContext::Impl::RecordCameraBaseline(
    VkCommandBuffer cmd) {
    if (auto ready = EnsureCameraResources(); !ready) return ready;
    const f64 centerNm = 0.5 * (cameraConfig.device.effectiveMinNm +
                                 cameraConfig.device.effectiveMaxNm);
    const auto pose = CameraDataForCapture(SpectralMode::Single, centerNm);
    if (!pose) return Result<void, String>::Err(pose.error());
    pipeline->SetUseCameraObserverSet(true);
    pipeline->SetCameraData(pose.value());
    pipeline->SetSpecConstants(static_cast<u32>(SpectralMode::Single), false);
    const u32 effectiveSeed = camera::DeviceRandomSeed(cameraConfig);
    const u32 randomSeed = camera::CounterRandomU32(
        effectiveSeed, 0u, cameraAcquisitionIndex,
        camera::NoiseClass::EmpiricalEffect, 0u);
    pipeline->SetSamplingParams(
        static_cast<u32>(cameraAcquisitionIndex), 0u, 1u,
        randomSeed, effectiveSeed);
    if (perfLogger) perfLogger->BeginFrame(cmd);
    pipeline->TraceRays(cmd,
        cameraConfig.optics.sensorWidthPx,
        cameraConfig.optics.sensorHeightPx);
    if (perfLogger) perfLogger->EndFrame(cmd);
    pipeline->SetUseCameraObserverSet(false);
    return Result<void, String>::Ok();
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

// ============================================================================
// Viewport pick pipeline (1x1 inline ray-query dispatch)
// ============================================================================

void ExternalRenderContext::Impl::CreatePickPipeline() {
    if (pickInitAttempted) return;
    pickInitAttempted = true;

    QL_LOG_INFO("Creating pick compute pipeline...");

    auto loadShaderFile = [](const String& path) -> std::vector<u32> {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return {};
        size_t fileSize = static_cast<size_t>(file.tellg());
        if (fileSize == 0 || fileSize % 4 != 0) return {};
        std::vector<u32> code(fileSize / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), fileSize);
        return code;
    };

    auto exeDir = GetExecutableDirectory();
    const std::vector<std::filesystem::path> shaderPaths = {
        "pick.spv",
        exeDir / "pick.spv",
        "shaders/pick.spv",
        exeDir / "shaders" / "pick.spv",
        "../shaders/pick.spv",
        "src/shaders/pick.spv",
    };
    std::vector<u32> code;
    for (const auto& path : shaderPaths) {
        code = loadShaderFile(path.string());
        if (!code.empty()) {
            QL_LOG_DEBUG("Pick: loaded shader from {}", path.string());
            break;
        }
    }
    if (code.empty()) {
        QL_LOG_WARN("Pick: could not load pick.spv, picking disabled");
        return;
    }

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size() * sizeof(u32);
    moduleInfo.pCode = code.data();
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &pickShader) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to create shader module");
        return;
    }

    // binding 0: TLAS, binding 1: result buffer
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &pickDescriptorSetLayout) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to create descriptor set layout");
        return;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PickPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pickDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pickPipelineLayout) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to create pipeline layout");
        return;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = pickShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pickPipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                 &pickPipeline) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to create compute pipeline");
        pickPipeline = VK_NULL_HANDLE;
        return;
    }

    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pickDescriptorPool) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to create descriptor pool");
        return;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pickDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pickDescriptorSetLayout;
    if (vkAllocateDescriptorSets(device, &allocInfo, &pickDescriptorSet) != VK_SUCCESS) {
        QL_LOG_WARN("Pick: failed to allocate descriptor set");
        pickDescriptorSet = VK_NULL_HANDLE;
        return;
    }

    pickOutputBuffer = std::make_unique<GpuBuffer>(
        contextAdapter->GetAllocator(), sizeof(PickResultGpu),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
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

    // Create display image (same format as outputImage) at the CLAHE input
    // extent: the physical sensor array when the camera is enabled, else the
    // render extent.
    const auto claheExtent = ClaheInputExtent();
    displayImage = std::make_unique<GpuImage>(
        allocator,
        device,
        claheExtent.width, claheExtent.height,
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

void ExternalRenderContext::Impl::RecreateClaheDisplayImage() {
    if (!claheInitialized || claheDescriptorSet == VK_NULL_HANDLE ||
        !contextAdapter) {
        return;
    }
    // Replacing a descriptor-backed image a submitted frame may still read;
    // configuration changes happen between frames, but a resize or a camera
    // config change is not guaranteed to.
    vkDeviceWaitIdle(device);
    const auto extent = ClaheInputExtent();
    displayImage = std::make_unique<GpuImage>(
        contextAdapter->GetAllocator(),
        device,
        extent.width, extent.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );
    TransitionImageLayoutImmediate(
        displayImage->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL
    );

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = displayImage->GetView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = claheDescriptorSet;
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &outputImageInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void ExternalRenderContext::Impl::ComputeImageMinMax(f32& outMin, f32& outMax) {
    const GpuImage* input = ClaheInputImage();
    const auto extent = ClaheInputExtent();
    if (!displayRangeInitAttempted) {
        displayRangeInitAttempted = true;
        auto created = rendercore::GpuDisplayRange::Create(*contextAdapter);
        if (created) {
            displayRange = std::move(created.value());
        } else {
            QL_LOG_WARN("GPU display range unavailable; using full image readback: {}",
                        created.error());
        }
    }
    if (displayRange) {
        auto range = displayRange->Compute(
            *input, extent.width, extent.height,
            displayParams.percentileLow, displayParams.percentileHigh);
        if (range) {
            outMin = range.value().min;
            outMax = range.value().max;
            return;
        }
        QL_LOG_WARN("GPU display range failed; using full image readback: {}",
                    range.error());
    }

    // Read back the CLAHE input image -- the camera's corrected product when
    // the camera is enabled, else the raw accumulation.
    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        *contextAdapter,
        input->GetImage(),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        extent.width,
        extent.height
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

    // The display window is a percentile window, always. This used to be
    // gated on the max/min ratio exceeding 100, which reads as "only clip when
    // the range is wide" -- but an infrared render's ratio is 1.01 to 3, so
    // the guard held for exactly the images that need it least protected: the
    // whole window was then set by one hot pixel and one cold one, and the
    // scene got whatever contrast they left over.
    //
    // Build a histogram rather than sorting: same percentile, one pass.
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
    const f64 lowFraction =
        std::clamp(static_cast<f64>(displayParams.percentileLow), 0.0, 100.0) / 100.0;
    const f64 highFraction =
        std::clamp(static_cast<f64>(displayParams.percentileHigh), 0.0, 100.0) / 100.0;

    const size_t totalPixels = luminances.size();
    const auto targetLow = static_cast<size_t>(static_cast<f64>(totalPixels) * lowFraction);
    const auto targetHigh = static_cast<size_t>(static_cast<f64>(totalPixels) * highFraction);

    size_t cumulative = 0;
    size_t binLow = 0;
    size_t binHigh = histBins - 1;
    bool haveLow = false;

    for (size_t i = 0; i < histBins; ++i) {
        cumulative += histogram[i];
        if (!haveLow && cumulative >= targetLow) {
            binLow = i;
            haveLow = true;
        }
        if (cumulative >= targetHigh) {
            binHigh = i;
            break;
        }
    }

    // Convert bins back to luminance values
    const f32 invScale = (absMax - absMin) / (histBins - 1);
    outMin = absMin + static_cast<f32>(binLow) * invScale;
    outMax = absMin + static_cast<f32>(binHigh) * invScale;

    // A window the percentiles collapsed to nothing is an image with no
    // contrast in it -- an isothermal cavity, most often -- and stretching it
    // would turn rounding noise into a picture. Fall back to what there is.
    if (!(outMax - outMin > (absMax - absMin) * 1e-3f)) {
        outMin = absMin;
        outMax = absMax;
    }
}

void ExternalRenderContext::Impl::ExecuteCLAHE(VkCommandBuffer cmd, u32 width, u32 height) {
    if (!claheInitialized) return;

    // The camera's agcSource product (infrared) or corrected product
    // (visible) feeds CLAHE when the camera is enabled; otherwise the raw
    // accumulation. Callers only run CLAHE once a camera acquisition has
    // completed, so the input product is present whenever the camera is.
    VkImageView inputView = ClaheInputImage()->GetView();

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
    u32 tileCountX = static_cast<u32>(displayParams.tileSize);
    u32 tileCountY = static_cast<u32>(displayParams.tileSize);

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
        u32 toneMode;
        u32 palette;
        u32 padding;
    };

    CLAHEPushConstants pushConstants{};
    pushConstants.imageWidth = width;
    pushConstants.imageHeight = height;
    pushConstants.tileCountX = tileCountX;
    pushConstants.tileCountY = tileCountY;
    pushConstants.clipLimit = displayParams.clipLimit;
    pushConstants.luminanceOnly = displayParams.luminanceOnly ? 1 : 0;
    pushConstants.toneMode = static_cast<u32>(displayParams.toneMode);
    pushConstants.palette = static_cast<u32>(displayParams.palette);
    pushConstants.inputMin = inputMin;
    pushConstants.inputMax = inputMax;

    // A linear stretch is the window and nothing else, so the two passes that
    // exist to build a CDF have nothing to contribute -- and skipping them
    // makes the cheapest tone operator also the cheapest to run.
    const bool needsHistogram = displayParams.toneMode != DisplayToneMode::Linear;

    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

    // Memory barrier: wait for the input's last writer. Without the histogram
    // passes the next reader is pass 3's compute, not the buffer fill. A
    // camera-enabled frame's input is a compute product (the camera chain),
    // not the ray tracing output.
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = needsHistogram
                                   ? VK_ACCESS_TRANSFER_WRITE_BIT
                                   : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    vkCmdPipelineBarrier(
        cmd,
        cameraConfig.enabled ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                             : VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        needsHistogram ? VK_PIPELINE_STAGE_TRANSFER_BIT
                       : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );

    if (needsHistogram) {

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

    }  // needsHistogram

    // Pass 3: Apply the mapping, then the palette
    pushConstants.passIndex = 2;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, claheApplyPipeline);
    // Bound here as well as in pass 1, because pass 1 does not always run.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            clahePipelineLayout, 0, 1,
                            &claheDescriptorSet, 0, nullptr);
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
    m_impl->cameraResourcesDirty = true;
    if (m_impl->cameraConfig.enabled) m_impl->cameraCapturePending = true;

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
        // Whatever was bound stays bound -- freeing it would leave binding 10
        // dangling -- but it is no longer what the host asked for, so it stops
        // being a light source. Lowering the flag rather than leaving the old
        // map lighting the scene is what keeps a failed load from looking like a
        // successful one, and it is what GetLightingParams reports afterwards.
        // Accumulated frames were integrated under the old lighting, so they go.
        m_impl->hasCustomEnvMap = false;
        m_impl->lightingParams.enableEnvironmentMap = 0u;
        m_impl->UploadLightingParams();
        ResetAccumulation();
        return Result<void, String>::Err(loaded.error());
    }
    m_impl->envMap = std::move(loaded.value());

    if (m_impl->pipeline) {
        m_impl->pipeline->BindPrefilteredEnvMap(m_impl->envMap.View(), m_impl->envMap.Sampler());
    }
    // A map that loaded is a map that lights: raising the flag here is what makes
    // a bare LoadEnvironmentMap call work on its own, instead of depending on
    // whatever the flag happened to be already.
    m_impl->hasCustomEnvMap = true;
    m_impl->lightingParams.enableEnvironmentMap = 1u;
    m_impl->UploadLightingParams();
    ResetAccumulation();
    return Result<void, String>::Ok();
}

bool ExternalRenderContext::HasEnvironmentMap() const {
    return m_impl->hasCustomEnvMap;
}

} // namespace quantiloom
