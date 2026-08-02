/**
 * @file OfflineRenderer.cpp
 * @brief The offline render facade: config in, radiance out
 *
 * This is `src/app/main.cpp`'s former middle, moved behind the public API. It
 * ran as a ~25-stage sequence of library internals -- GpuBuffer, CommandHelper,
 * BLAS/TLAS, RayTracingPipeline -- which is why the CLI linked the core's
 * objects rather than the DLL and why ~270 symbols stayed exported for its sake.
 *
 * Two things about the code below are load-bearing and easy to undo by accident:
 *
 * **Member order is destruction order.** Everything here used to be a local in
 * one function, destroyed in reverse declaration order with the VulkanContext
 * declared first and therefore destroyed last. The members are declared in that
 * same order for the same reason: a GPU buffer outliving its allocator is not a
 * compile error, and the failure it produces points nowhere near the cause.
 *
 * **The setup order itself is a dependency graph**, not a narrative. The solar
 * LUT is read before the atmosphere because the atmosphere overwrites the
 * lighting temperature; the material buffer is built after the spectral curves
 * because it resolves indices into them.
 *
 * What this file no longer does is *interpret* the config. Reading the ~50 keys
 * lives in rendercore::ResolveRenderConfig / ResolveMaterialSpectra, which
 * Quantiloom Studio reads the same file through -- see ConfigResolve.hpp for
 * what the two readings had drifted into. Everything below takes resolved data
 * and puts it on a device, in the order above.
 *
 * @author blitzcolo
 */

#include "renderer/OfflineRenderer.hpp"

#include "core/Log.hpp"
#include "core/SpectralData.hpp"
#include "io/SpectralIO.hpp"
#include "renderer/ConfigResolve.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/PerformanceLogger.hpp"
#include "renderer/LightingParams.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/RenderDeviceImpl.hpp"
#include "renderer/MaterialGpuData.hpp"
#include "atmos/AtmosphereBaker.hpp"
#include "scene/Camera.hpp"
#include "scene/Material.hpp"
#include "hs_core/HyperspectralRenderer.hpp"
#include "hs_core/HyperspectralConfig.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quantiloom {

namespace {
using SetupResult = Result<void, String>;
}

// ============================================================================
// Impl
// ============================================================================

struct OfflineRenderer::Impl {
    // ------------------------------------------------------------------
    // Not GPU state; position among the members is free.
    // ------------------------------------------------------------------
    Config config;
    InitParams init;
    OfflineRenderParams params;
    /// The config, read. Everything below consumes this rather than the TOML.
    ConfigApplyOptions configOptions;
    rendercore::ResolvedRenderConfig resolved;
    rendercore::ResolvedMaterialSpectra spectra;
    /// Diagnostics from the two resolvers. The CLI logs as it goes, so this is
    /// kept for the record rather than read back.
    ConfigApplyReport configReport;

    // ------------------------------------------------------------------
    // GPU state. DECLARATION ORDER IS DESTRUCTION ORDER -- see the file
    // comment. This is the order the same objects had as locals, context
    // first so that it dies last.
    //
    // Held by pointer only so that it is created where the original code
    // created it -- after the config is parsed. A config error should not
    // cost a device. Each method below binds it back to a plain `context`
    // reference, which is also why nothing here is *named* context: a member
    // of that name would be shadowed rather than referenced (MSVC C4458).
    //
    // Null when InitParams::sharedDevice was given: the device is then a
    // RenderDevice's and outlives this instance. `contextRef` is the one to
    // read; it points at whichever of the two is in play.
    // ------------------------------------------------------------------
    std::unique_ptr<VulkanContext> contextPtr;
    Scene loadedScene;
    rendercore::SceneGeometry geometry;
    std::unique_ptr<GpuImage> outputImage;
    LightingParams lightingParams{};
    std::unique_ptr<GpuBuffer> lightingParamsBuffer;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<GpuBuffer> spectralCurvesBuffer;
    std::unique_ptr<GpuBuffer> criBuffer;
    std::unique_ptr<GpuBuffer> solarSpectralLUTBuffer;
    std::unique_ptr<GpuBuffer> atmosHeaderBuffer;
    std::unique_ptr<GpuBuffer> atmosDataBuffer;
    std::unique_ptr<GpuBuffer> cieCMF_LUTBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    rendercore::BrdfLut brdfLut;
    rendercore::EnvironmentCubemap envMap;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::unique_ptr<RayTracingPipeline> pipeline;
    std::unique_ptr<PerformanceLogger> perfLogger;

    // ------------------------------------------------------------------
    // Borrowed from a shared RenderDevice, or all null. Not GPU state this
    // instance owns, so position among the members above does not matter --
    // none of it is destroyed here.
    //
    // Filled in by Create(), which is the friend RenderDevice named; keeping
    // the reach into RenderDevice::Impl in one place is why these are five
    // plain pointers rather than one pointer to that Impl.
    // ------------------------------------------------------------------
    VulkanContext* contextRef = nullptr;
    const rendercore::BrdfLut* brdfLutRef = nullptr;
    const rendercore::EnvironmentCubemap* fallbackEnvMapRef = nullptr;
    /// Either the shared fallback or `envMap` above, decided in BuildPipeline.
    const rendercore::EnvironmentCubemap* envMapRef = nullptr;
    const GpuBuffer* cieCMF_LUTRef = nullptr;
    bool borrowingDevice = false;

    ~Impl() {
        // Before the members go, because it was created against the context and
        // the context is one of them. Saved on the way out so the next run
        // starts warm. A borrowed cache belongs to the RenderDevice, which saves
        // and destroys it once for the whole batch.
        if (!borrowingDevice && pipelineCache != VK_NULL_HANDLE && contextPtr) {
            if (!init.pipelineCachePath.empty()) {
                RayTracingPipeline::SavePipelineCache(*contextPtr, pipelineCache,
                                                      init.pipelineCachePath);
            }
            RayTracingPipeline::DestroyPipelineCache(*contextPtr, pipelineCache);
        }
    }

    SetupResult BuildScene();
    SetupResult BuildIlluminants();
    SetupResult BuildPipeline();

    OfflineRenderOutput RenderHyperspectral();
    OfflineRenderOutput RenderSingleFrame();
};

// ============================================================================
// Scene and lighting
// ============================================================================

SetupResult OfflineRenderer::Impl::BuildScene() {
    VulkanContext& context = *contextRef;

    QL_LOG_INFO("Loading scene...");

    auto sceneResult = rendercore::LoadSceneFromConfig(config, configOptions.baseDir);
    if (!sceneResult.has_value()) {
        return SetupResult::Err("Failed to load scene: " + sceneResult.error());
    }

    loadedScene = sceneResult.value();

    // A procedural scene brings no materials of its own; material.albedo is
    // what the config offered for that case.
    if (loadedScene.materials.empty()) {
        Material defaultMaterial =
            Material::CreateLambertian(resolved.defaultAlbedo, "DefaultMaterial");
        loadedScene.materials.push_back(defaultMaterial);
        QL_LOG_INFO("  Created default material (spectral albedo: {:.3f})",
                    defaultMaterial.spectralAlbedo);
    }

    QL_LOG_INFO("  Scene loaded: {} meshes, {} nodes, {} materials",
                loadedScene.meshes.size(), loadedScene.nodes.size(),
                loadedScene.materials.size());

    // The rest of what the config says about materials: the IR temperature
    // backfill, [[materials]] overrides, the sRGB-upsampling gate, the
    // reflectance curves and the refractive indices. Reading it needs the scene
    // because all of it is matched to materials by name.
    auto spectraResult = rendercore::ResolveMaterialSpectra(
        config, loadedScene, resolved, configOptions, configReport);
    if (!spectraResult.has_value()) {
        return SetupResult::Err(std::move(spectraResult).error());
    }
    spectra = std::move(spectraResult.value());

    // Merged geometry buffers, one BLAS per primitive, a TLAS over the node
    // instances, and the per-instance offset table the closest-hit shader indexes
    // with InstanceIndex() -- all of it derived from the scene, all of it shared
    // with the interactive context.
    geometry = rendercore::SceneGeometry::Build(context, loadedScene);
    if (!geometry.IsValid()) {
        return SetupResult::Err("Scene has no geometry to trace");
    }

    outputImage = rendercore::CreateRenderTarget(context, params.width, params.height);

    return SetupResult::Ok();
}

SetupResult OfflineRenderer::Impl::BuildIlluminants() {
    VulkanContext& context = *contextRef;

    // Nothing here reads the config any more: the curves, the illuminant and
    // the atmosphere configuration all arrived resolved. What is left is
    // putting them on the device, in the order the file comment describes.

    QL_LOG_INFO("Uploading textures to GPU...");
    textureManager = std::make_unique<TextureManager>(context);
    textureManager->UploadTextures(loadedScene.textures);
    QL_LOG_INFO("  {} textures uploaded", textureManager->GetTextureCount());

    // ====================================================================
    // Spectral reflectance curves
    // ====================================================================
    // A buffer is created either way: an empty binding is not a valid one, so
    // an unconfigured scene gets a single zeroed curve rather than nothing.
    if (!spectra.curves.empty()) {
        spectralCurvesBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            spectra.curves.size() * sizeof(SpectralCurveGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        spectralCurvesBuffer->Upload(spectra.curves.data(),
                                     spectra.curves.size() * sizeof(SpectralCurveGPU));
        QL_LOG_INFO("  Uploaded {} bytes to GPU",
                    spectra.curves.size() * sizeof(SpectralCurveGPU));
    } else {
        SpectralCurveGPU dummyCurve{};
        spectralCurvesBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            sizeof(SpectralCurveGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        spectralCurvesBuffer->Upload(&dummyCurve, sizeof(SpectralCurveGPU));
        QL_LOG_INFO("  Created dummy spectral curves buffer (no curves loaded)");
    }

    // ====================================================================
    // Complex refractive indices
    // ====================================================================
    if (!spectra.refractiveIndices.empty()) {
        criBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            spectra.refractiveIndices.size() * sizeof(ComplexRefractiveIndexGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        criBuffer->Upload(
            spectra.refractiveIndices.data(),
            spectra.refractiveIndices.size() * sizeof(ComplexRefractiveIndexGPU));
        QL_LOG_INFO("  Uploaded {} bytes to GPU ({} entries x {} bytes)",
                    spectra.refractiveIndices.size() * sizeof(ComplexRefractiveIndexGPU),
                    spectra.refractiveIndices.size(),
                    sizeof(ComplexRefractiveIndexGPU));
    } else {
        ComplexRefractiveIndexGPU dummyCRI{};
        criBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            sizeof(ComplexRefractiveIndexGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        criBuffer->Upload(&dummyCRI, sizeof(ComplexRefractiveIndexGPU));
        QL_LOG_INFO("  Created dummy CRI buffer (no data loaded)");
        QL_LOG_INFO("  NOTE: Add [refractive_index] section to config for physical "
                    "metal Fresnel");
    }

    // ====================================================================
    // Solar spectral LUT
    // ====================================================================
    SolarSpectralLUT solarLUT{};  // Zero-initialised: numSamples = 0 is "none"
    if (resolved.solarSunSky) {
        const auto& [sunCurve, skyCurve] = *resolved.solarSunSky;
        solarLUT = SolarSpectralLUT::FromCPU(sunCurve, skyCurve);
        if (solarLUT.IsValid()) {
            auto [minWl, maxWl] = solarLUT.GetWavelengthRange();
            QL_LOG_INFO("  Solar LUT loaded: {} samples, lambda=[{:.1f}, {:.1f}] nm",
                        solarLUT.sunIrradiance.numSamples, minWl, maxWl);
        } else {
            return SetupResult::Err(
                "Solar LUT conversion failed. There is no RGB fallback any more "
                "-- the scene would render unlit, so this is fatal.");
        }
    }

    solarSpectralLUTBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(),
        sizeof(SolarSpectralLUT),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    solarSpectralLUTBuffer->Upload(&solarLUT, sizeof(SolarSpectralLUT));

    // ====================================================================
    // NN atmosphere: bake the resolved configuration for the active band
    // ====================================================================
    // Missing network files or out-of-coverage wavelengths are hard errors --
    // no fallback.
    AtmosNNHeaderGPU atmosHeader{};  // enabled = 0
    // Sized so that even unconditionally-evaluated LUT reads (HLSL ternary is a
    // select) stay in bounds when the atmosphere is disabled
    std::vector<f32> atmosData(2048, 0.0f);
    if (resolved.atmosphere.enabled) {
        AtmosLambdaGrid grid = RenderBandLambdaGrid(
            params.mode, static_cast<double>(params.wavelengthNm));
        if (!grid.error.empty()) {
            return SetupResult::Err("NN atmosphere: " + grid.error);
        }
        if (grid.band.empty()) {
            return SetupResult::Err("NN atmosphere: spectral mode has no NN coverage");
        }
        try {
            AtmosModelPack pack(resolved.atmosphere.modelPackDir);
            AtmosphereBaker baker(pack);
            AtmosBakeResult baked = baker.Bake(
                resolved.atmosphere, grid.band, grid.lambdasNm, grid.windowHalfWidthNm);
            const glm::vec3& sunDir = resolved.lighting.sunDirection;
            baked.header.sunDirWorld[0] = sunDir.x;
            baked.header.sunDirWorld[1] = sunDir.y;
            baked.header.sunDirWorld[2] = sunDir.z;
            baked.header.worldUnitsToMeters = resolved.worldUnitsToMeters;
            atmosHeader = baked.header;
            atmosData = std::move(baked.data);
        } catch (const std::exception& e) {
            return SetupResult::Err(String("NN atmosphere setup failed: ") + e.what());
        }
    }

    atmosHeaderBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(),
        sizeof(AtmosNNHeaderGPU),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    atmosHeaderBuffer->Upload(&atmosHeader, sizeof(AtmosNNHeaderGPU));

    atmosDataBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(),
        atmosData.size() * sizeof(f32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    atmosDataBuffer->Upload(atmosData.data(), atmosData.size() * sizeof(f32));

    // ====================================================================
    // CIE 1931 colour matching functions (VIS_Fused)
    // ====================================================================
    // Compiled in rather than parsed from assets/luts/, so there is no path to
    // get wrong and no "VIS_FUSED will produce INCORRECT colors" fallback to hit.
    // The table is the same for every scene, so a shared device already has one.
    if (!cieCMF_LUTRef) {
        cieCMF_LUTBuffer = rendercore::CreateCieColourMatchingBuffer(context);
        cieCMF_LUTRef = cieCMF_LUTBuffer.get();
    }

    // ====================================================================
    // Material buffer
    // ====================================================================
    QL_LOG_INFO("Creating PBR material buffer...");

    // The curve and refractive-index slots are resolved here rather than inside
    // the conversion: the CLI matches material names against what the config
    // loaded, while the interactive context reads the indices off the Material.
    Vector<rendercore::MaterialGpuIndices> materialIndices;
    materialIndices.reserve(loadedScene.materials.size());
    for (const auto& mat : loadedScene.materials) {
        rendercore::MaterialGpuIndices slots;

        if (auto it = spectra.materialNameToCurve.find(mat.name);
            it != spectra.materialNameToCurve.end()) {
            slots.spectralReflectanceCurve = it->second;
            QL_LOG_INFO("  Material '{}': using spectral curve index {}", mat.name,
                        it->second);
        }
        if (auto it = spectra.materialNameToRefractiveIndex.find(mat.name);
            it != spectra.materialNameToRefractiveIndex.end()) {
            slots.complexRefractiveIndex = it->second;
            QL_LOG_INFO("  Material '{}': using physical Fresnel (CRI index {})",
                        mat.name, it->second);
        }

        materialIndices.push_back(slots);
    }

    materialBuffer = rendercore::BuildMaterialBuffer(
        context, loadedScene, params.wavelengthNm, materialIndices);
    if (!materialBuffer) {
        return SetupResult::Err("Scene has no materials");
    }

    return SetupResult::Ok();
}

// ============================================================================
// Pipeline
// ============================================================================

SetupResult OfflineRenderer::Impl::BuildPipeline() {
    VulkanContext& context = *contextRef;

    // ====================================================================
    // Load or Generate BRDF Integration LUT for IBL (with Disk Caching)
    // ====================================================================
    // The BRDF LUT is scene-independent and can be cached to disk.
    // First run: Generate (5-10 seconds) and save to cache
    // Subsequent runs: Load from cache (instant)
    // ====================================================================

    // A shared device generated it once for the whole batch.
    if (!brdfLutRef) {
        brdfLut = rendercore::BrdfLut::Create(context);
        if (!brdfLut.IsValid()) {
            return SetupResult::Err("Failed to create BRDF LUT sampler");
        }
        brdfLutRef = &brdfLut;
    }

    // ====================================================================
    // Create Prefiltered Environment Map for IBL Specular
    // ====================================================================
    QL_LOG_INFO("Creating prefiltered environment map for IBL...");

    // Falls back to sky blue when the config names no map, or names one that
    // will not load. Load() reads through ImageIO::ReadImage, so .hdr and the
    // LDR formats work as well as .exr -- this used to call ReadEXR directly
    // and take the fallback for anything else.
    // A disabled map is not loaded at all: the binding still has to be valid, so
    // the fallback is bound and the shader skips it on the lighting flag. Off
    // means it contributes nothing, not that it is replaced by a sky.
    // The fallback is uniform sky blue and identical for every scene, so a shared
    // device supplies one. A map the config *names* is this render's own.
    const auto useFallback = [&] {
        if (fallbackEnvMapRef) {
            envMapRef = fallbackEnvMapRef;
        } else {
            envMap = rendercore::EnvironmentCubemap::Fallback(context);
            envMapRef = &envMap;
        }
    };

    const String envMapPath = resolved.environmentMapEnabled ? resolved.environmentMap : String{};
    if (envMapPath.empty()) {
        QL_LOG_INFO("  No environment map specified in config, using fallback");
        useFallback();
    } else {
        auto loaded = rendercore::EnvironmentCubemap::Load(context, envMapPath);
        if (loaded.has_value()) {
            envMap = std::move(loaded.value());
            envMapRef = &envMap;
        } else {
            QL_LOG_WARN("  {}, using fallback", loaded.error());
            useFallback();
        }
    }

    QL_LOG_INFO("  Prefiltered environment map ready ({}x{} per face, {} mip levels)",
                envMapRef->FaceSize(), envMapRef->FaceSize(), envMapRef->MipLevels());

    // ====================================================================
    // Create Ray Tracing Pipeline
    // ====================================================================
    // A borrowed cache arrived already loaded and is saved once by the device;
    // Create() put it in `pipelineCache` and set borrowingDevice so that ~Impl
    // leaves it alone.
    if (!borrowingDevice && !init.pipelineCachePath.empty()) {
        pipelineCache =
            RayTracingPipeline::LoadPipelineCache(context, init.pipelineCachePath);
    }

    rendercore::PipelineBindings bindings;
    bindings.outputImage = outputImage.get();
    bindings.geometry = &geometry;
    bindings.lightingParams = lightingParamsBuffer.get();
    bindings.materials = materialBuffer.get();
    bindings.textures = textureManager.get();
    bindings.environment = envMapRef;
    bindings.brdfLut = brdfLutRef;
    bindings.spectralCurves = spectralCurvesBuffer.get();
    bindings.complexRefractiveIndex = criBuffer.get();
    bindings.solarLut = solarSpectralLUTBuffer.get();
    bindings.atmosphereHeader = atmosHeaderBuffer.get();
    bindings.atmosphereData = atmosDataBuffer.get();
    bindings.cieColourMatching = cieCMF_LUTRef;

    pipeline = rendercore::CreateRayTracingPipeline(context, pipelineCache, bindings);

    CameraData cameraData = resolved.camera.GetCameraData();
    cameraData.wavelength_nm = params.wavelengthNm;         // Override with config wavelength
    cameraData.spectral_mode = static_cast<u32>(params.mode);  // Set rendering mode
    cameraData.debug_mode = static_cast<u32>(resolved.debugMode);
    pipeline->SetCameraData(cameraData);

    pipeline->SetSpecConstants(
        static_cast<u32>(params.mode),
        cameraData.debug_mode != 0);

    QL_LOG_INFO("  Pipeline created and resources bound");

    // ====================================================================
    // Initialize Performance Logger
    // ====================================================================
    QL_LOG_INFO("Initializing performance logger...");

    PerformanceLogger::Config perfConfig;
    perfConfig.csvFilePath = init.performanceCsvPath;
    perfConfig.enableLogging = !init.performanceCsvPath.empty();
    perfLogger = std::make_unique<PerformanceLogger>(context, perfConfig);

    return SetupResult::Ok();
}

// ============================================================================
// Create
// ============================================================================

OfflineRenderer::OfflineRenderer() : m_impl(std::make_unique<Impl>()) {}
OfflineRenderer::~OfflineRenderer() = default;

const OfflineRenderParams& OfflineRenderer::Params() const {
    return m_impl->params;
}

Result<std::unique_ptr<OfflineRenderer>, String> OfflineRenderer::Create(
    const Config& config, const InitParams& params) {
    using CreateResult = Result<std::unique_ptr<OfflineRenderer>, String>;

    // Not make_unique: the constructor is private, and a facade whose only
    // construction path is Create() is the point.
    std::unique_ptr<OfflineRenderer> self(new OfflineRenderer());
    Impl& impl = *self->m_impl;
    impl.config = config;
    impl.init = params;

    // The whole file, read, before a device exists: a config that cannot be
    // honoured should cost nothing. Error policy here is the CLI's -- it
    // renders once and writes a file, so a key it cannot honour is a render it
    // should refuse rather than quietly substitute a default into.
    impl.configOptions.missingRequired = ConfigApplyOptions::MissingKeyPolicy::Error;
    impl.configOptions.atmosphereModelPackFallback = params.atmosphereModelPackFallback;
    // Empty until this field existed, which meant every relative asset path in a
    // CLI-rendered config resolved against the working directory rather than
    // against the config -- the interactive context had always passed it.
    impl.configOptions.baseDir = params.baseDir;
    // One camera, one image: sun angles and observer altitude are pinned at
    // load time rather than following a camera that will not move.
    impl.configOptions.freezeDerivedAtmosGeometry = true;
    impl.configOptions.applyDebugMode = true;

    QL_LOG_INFO("Parsing configuration...");
    auto resolvedResult =
        rendercore::ResolveRenderConfig(config, impl.configOptions, impl.configReport);
    if (!resolvedResult.has_value()) {
        return CreateResult::Err(std::move(resolvedResult).error());
    }
    impl.resolved = std::move(resolvedResult.value());

    // The public params are the subset a caller reads back rather than
    // re-parsing -- see OfflineRenderer.hpp on why it reads them from here.
    impl.params.width = impl.resolved.width;
    impl.params.height = impl.resolved.height;
    impl.params.spp = impl.resolved.spp;
    impl.params.mode = impl.resolved.mode;
    impl.params.modeName = impl.resolved.modeName;
    impl.params.wavelengthNm = impl.resolved.wavelengthNm;
    impl.params.outputPath = impl.resolved.outputPath;

    // Already carries the illuminant's colour and, when an atmosphere is
    // enabled, its ground temperature -- the resolver applied both.
    impl.lightingParams = impl.resolved.lighting;

    // ====================================================================
    // Initialize Vulkan Context
    // ====================================================================
    if (params.sharedDevice) {
        // The one place that reaches into RenderDevice::Impl -- this function is
        // the friend it named. Everything below works off the plain pointers.
        RenderDevice::Impl& shared = *params.sharedDevice->m_impl;
        impl.borrowingDevice = true;
        impl.contextRef = shared.context.get();
        impl.brdfLutRef = &shared.brdfLut;
        impl.fallbackEnvMapRef = &shared.fallbackEnvMap;
        impl.cieCMF_LUTRef = shared.cieCMF_LUTBuffer.get();
        impl.pipelineCache = shared.pipelineCache;
    } else {
        QL_LOG_INFO("Initializing Vulkan context...");
        impl.contextPtr = std::make_unique<VulkanContext>();
        impl.contextRef = impl.contextPtr.get();

        if (!impl.contextPtr->IsRayTracingSupported()) {
            return CreateResult::Err("Ray tracing not supported on this device");
        }
    }

    if (auto r = impl.BuildScene(); !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    impl.lightingParamsBuffer = std::make_unique<GpuBuffer>(
        impl.contextRef->GetAllocator(),
        sizeof(LightingParams),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    impl.lightingParamsBuffer->Upload(&impl.lightingParams, sizeof(LightingParams));

    if (auto r = impl.BuildIlluminants(); !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    if (auto r = impl.BuildPipeline(); !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    return CreateResult(std::move(self));
}

// ============================================================================
// Render
// ============================================================================

OfflineRenderOutput OfflineRenderer::Render() {
    if (m_impl->params.mode == SpectralMode::Multispectral) {
        return m_impl->RenderHyperspectral();
    }
    return m_impl->RenderSingleFrame();
}

OfflineRenderOutput OfflineRenderer::Impl::RenderHyperspectral() {
    VulkanContext& context = *contextRef;

    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  MULTISPECTRAL RENDERING MODE");
    QL_LOG_INFO("========================================");

    // Parse hyperspectral configuration from TOML
    HyperspectralConfig hsConfig;
    hsConfig.wavelengthMin_nm = config.Get<f32>("hyperspectral.wavelength_min_nm", 400.0f);
    hsConfig.wavelengthMax_nm = config.Get<f32>("hyperspectral.wavelength_max_nm", 2500.0f);
    hsConfig.wavelengthStep_nm = config.Get<f32>("hyperspectral.wavelength_step_nm", 10.0f);
    hsConfig.spp = params.spp;
    hsConfig.useGpuReconstruction = config.Get<bool>("hyperspectral.use_gpu_reconstruction", true);
    hsConfig.saveIntermediates = config.Get<bool>("hyperspectral.save_intermediates", false);

    // Determine output path (without extension)
    std::filesystem::path outPath(params.outputPath);
    String hsOutputPath = outPath.parent_path().string();
    if (!hsOutputPath.empty()) hsOutputPath += "/";
    hsOutputPath += outPath.stem().string();
    hsConfig.outputPath = hsOutputPath;

    // Output format (default: ENVI_BSQ - standard remote sensing format)
    String formatStr = config.Get<String>("hyperspectral.output_format", "envi_bsq");
    if (formatStr == "envi_bsq" || formatStr == "ENVI_BSQ") {
        hsConfig.outputFormat = HyperspectralOutputFormat::ENVI_BSQ;
    } else if (formatStr == "envi_bil" || formatStr == "ENVI_BIL") {
        hsConfig.outputFormat = HyperspectralOutputFormat::ENVI_BIL;
    } else if (formatStr == "envi_bip" || formatStr == "ENVI_BIP") {
        hsConfig.outputFormat = HyperspectralOutputFormat::ENVI_BIP;
    } else if (formatStr == "geotiff" || formatStr == "GeoTIFF") {
        hsConfig.outputFormat = HyperspectralOutputFormat::GeoTIFF;
    } else {
        QL_LOG_WARN("Unknown hyperspectral format '{}', using ENVI_BSQ", formatStr);
        hsConfig.outputFormat = HyperspectralOutputFormat::ENVI_BSQ;
    }

    QL_LOG_INFO("  Wavelength range: {:.1f} - {:.1f} nm", hsConfig.wavelengthMin_nm, hsConfig.wavelengthMax_nm);
    QL_LOG_INFO("  Wavelength step: {:.1f} nm", hsConfig.wavelengthStep_nm);
    QL_LOG_INFO("  Total bands: {}", hsConfig.GetNumBands());
    QL_LOG_INFO("  SPP per band: {}", hsConfig.spp);
    QL_LOG_INFO("  Output: {}", hsConfig.outputPath);
    QL_LOG_INFO("  GPU reconstruction: {}", hsConfig.useGpuReconstruction ? "enabled" : "disabled");
    QL_LOG_INFO("  Save intermediates: {}", hsConfig.saveIntermediates ? "enabled" : "disabled");
    QL_LOG_INFO("========================================");

    // Create hyperspectral renderer
    HyperspectralRenderer hsRenderer(context, *pipeline, loadedScene);

    // Progress: the log line every ten bands as before, plus the host's
    // callback on every band when it asked for one. The void* the internal
    // signature carries is how the host callback reaches this lambda -- it is
    // called from the band loop, so capturing `this` would be no safer and
    // less explicit.
    auto progressCallback = [](const HyperspectralProgress& p, void* userData) {
        if (p.currentBand % 10 == 0 || p.currentBand == p.totalBands) {
            QL_LOG_INFO("  Band {}/{} ({:.1f} nm) - {:.1f}% - ETA: {:.1f}s",
                        p.currentBand, p.totalBands, p.currentWavelength_nm,
                        p.GetPercentage(), p.GetRemainingSeconds());
        }
        if (userData) {
            const auto& hostCallback =
                *static_cast<const std::function<void(const OfflineProgress&)>*>(userData);
            OfflineProgress out;
            out.currentBand = p.currentBand;
            out.totalBands = p.totalBands;
            out.currentWavelength_nm = p.currentWavelength_nm;
            out.elapsedSeconds = p.elapsedSeconds;
            out.estimatedTotalSeconds = p.estimatedTotalSeconds;
            hostCallback(out);
        }
    };
    // Null when the host wants no callback, which is what the lambda tests.
    void* const progressUserData =
        init.onProgress ? static_cast<void*>(&init.onProgress) : nullptr;

    OfflineRenderOutput output;
    // The cube is streamed to disk band by band rather than assembled in
    // memory: at 512x512 and 211 bands it is 200 MB, and nothing downstream
    // wants it as one array.
    output.wroteItsOwnOutput = true;

    // Execute hyperspectral rendering
    auto status = hsRenderer.Render(hsConfig, progressCallback, progressUserData);

    if (status == HyperspectralStatus::Success) {
        QL_LOG_INFO("  Hyperspectral rendering complete!");
        QL_LOG_INFO("  Total render time: {:.2f} seconds", hsRenderer.GetLastRenderTime());
        QL_LOG_INFO("  Average time per band: {:.3f} seconds", hsRenderer.GetAverageTimePerBand());

        // Output is already written by HyperspectralRenderer::Render()
        QL_LOG_INFO("  Output written to: {}.hdr/.dat", hsConfig.outputPath);
    } else {
        output.error = String("Hyperspectral rendering failed: ") +
                       HyperspectralStatusToString(status);
    }

    return output;
}

OfflineRenderOutput OfflineRenderer::Impl::RenderSingleFrame() {
    VulkanContext& context = *contextRef;

    const u32 width = params.width;
    const u32 height = params.height;
    const u32 spp = params.spp;

    if (params.mode == SpectralMode::Single ||
        params.mode == SpectralMode::MWIR_Fused ||
        params.mode == SpectralMode::LWIR_Fused ||
        params.mode == SpectralMode::SWIR_Fused) {
        QL_LOG_INFO("Rendering frame at wavelength {:.1f} nm with {} samples per pixel...",
                    params.wavelengthNm, spp);
        QL_LOG_WARN("  ⚠️  PREVIEW MODE: Using RGB-averaged spectral albedo.");
        QL_LOG_WARN("  ⚠️  NOT suitable for quantitative analysis.");
        QL_LOG_WARN("  ⚠️  For quantitative results, provide measured spectral curves.");
    } else {
        QL_LOG_INFO("Rendering frame in RGB mode with {} samples per pixel...", spp);
    }

    try {
        u32 frameIndex = 0;
        f32 totalGpuMs = 0.0f;

        // Per-sample seeds for the path tracer. Deterministic by default so
        // two runs of one scene produce the same image -- the sensor seed
        // alone cannot deliver that, because this stage runs before it and
        // used to draw from random_device. Set renderer.seed = 0 for
        // nondeterministic sampling. The sequence still varies per sample
        // either way, so accumulation quality is unchanged.
        const u32 configuredSeed =
            config.Get<u32>("renderer.seed", constants::DEFAULT_SAMPLING_SEED);
        const u32 renderSeed =
            (configuredSeed != 0U) ? configuredSeed : std::random_device{}();
        if (configuredSeed == 0U) {
            QL_LOG_INFO("  Sampling seed: {} (nondeterministic, renderer.seed = 0)",
                        renderSeed);
        }
        std::mt19937 rng(renderSeed);
        std::uniform_int_distribution<u32> dist(0, std::numeric_limits<u32>::max());

        // Batch samples into groups. Each submit must stay WELL under the
        // Windows TDR limit (~2s): heavy IR scenes run ~500ms/sample, so
        // 2 per batch keeps a submit around 1s with safety margin.
        constexpr u32 BATCH_SIZE = 2;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = context.GetGraphicsQueueFamily();

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(context.GetDevice(), &poolInfo, nullptr, &cmdPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool for batched rendering");
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(context.GetDevice(), &allocInfo, &cmd);

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence);

        for (u32 batchStart = 0; batchStart < spp; batchStart += BATCH_SIZE) {
            u32 batchEnd = std::min(batchStart + BATCH_SIZE, spp);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            for (u32 sampleIndex = batchStart; sampleIndex < batchEnd; ++sampleIndex) {
                u32 randomSeed = dist(rng) ^ (frameIndex * 997 + sampleIndex * 1009);
                pipeline->SetSamplingParams(frameIndex, sampleIndex, spp, randomSeed);

                perfLogger->BeginFrame(cmd);
                bool isFinal = (sampleIndex == spp - 1);
                pipeline->TraceRays(cmd, width, height, isFinal);
                perfLogger->EndFrame(cmd);
            }

            vkEndCommandBuffer(cmd);

            vkResetFences(context.GetDevice(), 1, &fence);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            if (vkQueueSubmit(context.GetGraphicsQueue(), 1, &submitInfo, fence) != VK_SUCCESS) {
                throw std::runtime_error("Queue submit failed");
            }
            if (vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
                throw std::runtime_error("Fence wait failed (possible GPU timeout / device lost)");
            }

            for (u32 i = batchStart; i < batchEnd; ++i) {
                totalGpuMs += perfLogger->ResolveLastGpuMs();
            }
        }

        vkDestroyFence(context.GetDevice(), fence, nullptr);
        vkDestroyCommandPool(context.GetDevice(), cmdPool, nullptr);
        perfLogger->Flush();

        QL_LOG_INFO("  All samples completed!");
        QL_LOG_INFO("  Total GPU time: {:.2f} ms ({:.2f} ms/sample)",
                    totalGpuMs, totalGpuMs / spp);

        f64 totalRayCount = static_cast<f64>(width) * height * spp;
        f64 totalSeconds = static_cast<f64>(totalGpuMs) / 1000.0;
        f64 mraysPerSec = totalSeconds > 0.0 ? totalRayCount / totalSeconds / 1e6 : 0.0;
        QL_LOG_INFO("  Average throughput: {:.2f} Mrays/s", mraysPerSec);

    } catch (const std::exception& e) {
        QL_LOG_ERROR("  [DEBUG] GPU execution FAILED: {}", e.what());
        throw;
    }

    // ====================================================================
    // Readback
    // ====================================================================
    QL_LOG_INFO("Reading back and saving image...");
    QL_LOG_DEBUG("  [DEBUG] Starting image readback...");

    std::vector<f32> pixels = CommandHelper::ReadbackImage(
        context,
        outputImage->GetImage(),
        outputImage->GetFormat(),
        width,
        height
    );

    OfflineRenderOutput output;

    // The renderer knows which band it integrated, so it reports the sensor
    // conditioning rather than making the caller derive it. Shared with the
    // interactive context, which applies the same adjustment internally.
    const auto adjustment = rendercore::SensorAdjustmentForMode(
        params.mode, config.Has("spectral.wavelength_nm"));
    output.sensorRadianceScale = adjustment.radianceScale;
    output.sensorWavelengthNm = adjustment.wavelengthNm;

    // Convert to Image object (4 channels: RGBA)
    Image img(width, height, 4);
    img.channelNames = {"R", "G", "B", "A"};
    img.metadata["renderer"] = "Quantiloom Spectral";
    img.metadata["mode"] = params.modeName;
    if (params.mode == SpectralMode::Single ||
        params.mode == SpectralMode::MWIR_Fused ||
        params.mode == SpectralMode::LWIR_Fused ||
        params.mode == SpectralMode::SWIR_Fused) {
        img.metadata["wavelength_nm"] = std::to_string(params.wavelengthNm);
        img.metadata["quality_level"] = "PREVIEW_ONLY";
        img.metadata["warning"] = "RGB-averaged spectral albedo, not quantitative";
        img.metadata["note"] = "For quantitative results provide measured spectral curves";
    } else if (params.mode == SpectralMode::RGB) {
        img.metadata["quality_level"] = "PREVIEW";
        img.metadata["note"] = "RGB rendering (fast, no spectral integration)";
    } else if (params.mode == SpectralMode::VIS_Fused) {
        img.metadata["quality_level"] = "SPECTRAL";
        img.metadata["note"] = "32-wavelength spectral integration";
    }
    img.metadata["resolution"] = std::to_string(width) + "x" + std::to_string(height);
    img.metadata["spp"] = std::to_string(spp);

    // Copy pixel data
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            u32 pixelIndex = (y * width + x) * 4;
            img(x, y, 0) = pixels[pixelIndex + 0];  // R
            img(x, y, 1) = pixels[pixelIndex + 1];  // G
            img(x, y, 2) = pixels[pixelIndex + 2];  // B
            img(x, y, 3) = pixels[pixelIndex + 3];  // A
        }
    }

    output.radiance = std::move(img);
    return output;
}

}  // namespace quantiloom
