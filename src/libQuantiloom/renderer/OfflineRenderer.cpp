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
#include "core/Sha256.hpp"
#include "dataset/ProductGeometry.hpp"
#include "dataset/ExportSession.hpp"
#include "postprocess/CameraConfigIO.hpp"
#include <nlohmann/json.hpp>
#include "core/SpectralData.hpp"
#include "io/SpectralIO.hpp"
#include "renderer/ConfigResolve.hpp"
#include "renderer/SpectralUnmixer.hpp"
#include "renderer/TemperatureTextureLoader.hpp"
#include "renderer/ThermalEpochBuilder.hpp"
#include "renderer/ThermalExchangePrecompute.hpp"
#include "renderer/TimelineState.hpp"
#include "thermal/ThermalEpochs.hpp"
#include "core/LibVersion.hpp"
#include "renderer/GpuThermalStepper.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalSolveCache.hpp"
#include "thermal/ThermalSolver.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/OfflineBatchScheduler.hpp"
#include "renderer/PerformanceLogger.hpp"
#include "renderer/LightingParams.hpp"
#include "renderer/RenderCore.hpp"
#include "renderer/ThermalSunResponse.hpp"
#include "renderer/RenderDeviceImpl.hpp"
#include "renderer/MaterialGpuData.hpp"
#include "atmos/AtmosphereBaker.hpp"
#include "scene/Camera.hpp"
#include "scene/Material.hpp"
#include "hs_core/HyperspectralRenderer.hpp"
#include "hs_core/HyperspectralConfig.hpp"
#include "hs_core/BatchRenderer.hpp"
#include "postprocess/CpuCameraPipeline.hpp"
#include "postprocess/CameraPhysics.hpp"

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

/// An environment switch that is off unless someone deliberately turned it on:
/// unset reads as off, and so does any spelling that is not an affirmative.
bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    String lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered == "1" || lowered == "on" || lowered == "true" || lowered == "yes";
}
}  // namespace

// ============================================================================
// Impl
// ============================================================================

struct OfflineRenderer::Impl {
    // ------------------------------------------------------------------
    // Not GPU state; position among the members is free.
    // ------------------------------------------------------------------
    String ProductSnapshot(f64 timeSeconds, u32 width, u32 height, bool physical, bool staticSceneCamera = false, const CameraData* boundCamera = nullptr) const {
        using Json = nlohmann::json;
        CameraData actualCamera{};
        if (boundCamera) actualCamera = *boundCamera;
        else if (staticSceneCamera) actualCamera = loadedScene.camera.GetCameraData();
        else {
            const auto pose = CameraDataAt(timeSeconds, params.mode, params.wavelengthNm, physical);
            if (!pose) throw std::runtime_error(pose.error());
            actualCamera = pose.value();
        }
        dataset::ProductGeometry productGeometry;
        productGeometry.width = width;
        productGeometry.height = height;
        productGeometry.referenceTimeSeconds = timeSeconds;
        productGeometry.worldUnitsToMeters = lightingParams.worldUnitsToMeters;
        productGeometry.camera = actualCamera;
        const auto calibration = productGeometry.ToJson();
        if (!calibration) throw std::runtime_error(calibration.error());
        const auto& gpu = contextRef->GetDeviceProperties();
        const auto vector = [](const glm::vec3& v) { return Json::array({v.x,v.y,v.z}); };
        const auto& sun = lightingParams.sunDirection;
        f64 azimuth = std::atan2(sun.x, sun.z) * 180.0 / constants::PI;
        if (azimuth < 0.0) azimuth += 360.0;
        Json snapshot = {
            {"version", 1}, {"geometry", Json::parse(calibration.value())},
            {"software", {{"sdk_version", version::LibVersionString}}},
            {"device", {{"name", gpu.deviceName}, {"vendor_id", gpu.vendorID},
                        {"device_id", gpu.deviceID}, {"driver_version", gpu.driverVersion},
                        {"vulkan_api_version", gpu.apiVersion}}},
            {"lighting", {{"sun_direction_to_sun", vector(sun)},
                          {"sun_azimuth_deg", azimuth},
                          {"sun_elevation_deg", std::asin(std::clamp(static_cast<f64>(sun.y),-1.0,1.0))*180.0/constants::PI},
                          {"sun_radiance_rgb", vector(lightingParams.sunRadiance_rgb)},
                          {"sky_radiance_rgb", vector(lightingParams.skyRadiance_rgb)},
                          {"environment_map_enabled", lightingParams.enableEnvironmentMap != 0},
                          {"solar_sky_lut_enabled", resolved.solarSunSky.has_value()}}},
            {"sampling", {{"requested_spp", params.spp}, {"mode", params.modeName},
                          {"wavelength_nm", params.wavelengthNm}}},
            {"reproducibility", {{"verified", false},
                {"reason", "Resource-load and execution-binary fingerprints are not yet captured"}}}
        };
        const auto clock = timeline.Info();
        snapshot["timeline"] = {{"present", clock.present}, {"start_s", clock.start_s},
            {"end_s", clock.end_s}, {"current_s", timeline.Current_s()},
            {"ticks_per_second", clock.ticksPerSecond}};
        snapshot["thermal"] = {{"hour", ThermalHourNow()},
            {"epoch_count", thermalSession ? thermalSession->EpochCount() : 0u},
            {"epoch_index", thermalSession ? thermalSession->EpochAt(ThermalHourNow()) : 0u}};
        snapshot["state_snapshot_time_s"] = timeline.Current_s();
        snapshot["input_base_dir"] = init.baseDir;
        snapshot["diagnostics"] = Json::array();
        for (const auto& message : configReport.messages)
            snapshot["diagnostics"].push_back({{"key", message.key}, {"text", message.text},
                {"severity", message.severity == ConfigApplyMessage::Severity::Error ? "error" :
                    message.severity == ConfigApplyMessage::Severity::Warning ? "warning" : "info"}});
        return snapshot.dump();
    }
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
    std::unique_ptr<GpuBuffer> rgbToSpectrumBuffer;
    std::unique_ptr<GpuBuffer> emissiveTriangleBuffer;
    /// Per-element surface temperatures from the thermal solver. Always
    /// created -- an unbound descriptor is not a valid one -- with a single
    /// zero entry when no solve ran.
    std::unique_ptr<GpuBuffer> thermalTemperatureBuffer;
    std::unique_ptr<GpuBuffer> thermalSunResponseBuffer;
    std::unique_ptr<GpuBuffer> materialBuffer;
    rendercore::BrdfLut brdfLut;
    rendercore::EnvironmentCubemap envMap;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::unique_ptr<RayTracingPipeline> pipeline;
    std::unique_ptr<PerformanceLogger> perfLogger;
    std::unique_ptr<BatchRenderer> cameraBatchRenderer;
    std::unique_ptr<GpuBuffer> cameraAtmosHeaderBuffer;
    std::unique_ptr<GpuBuffer> cameraAtmosDataBuffer;

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
    const GpuBuffer* rgbToSpectrumRef = nullptr;
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

    // ------------------------------------------------------------------
    // The clock, and the thermal solve it drives
    // ------------------------------------------------------------------

    /// Which nodes move, and where they would stand if they did not. Inert for
    /// a config with no [timeline].
    rendercore::TimelineState timeline;

    /// Kept alive across frames: setting up a solve is a mesh, a stepper, a
    /// steady state and a schedule, and a sequence would otherwise pay for all
    /// of it per frame -- and lose the checkpoints that make the next hour
    /// cheap.
    std::unique_ptr<thermal::ThermalSolveSession> thermalSession;
    /// Outlives the session, which holds a pointer to it.
    std::unique_ptr<rendercore::GpuThermalStepper> thermalGpuStepper;
    thermal::ThermalConfig thermalConfig;
    thermal::ThermalMesh thermalMesh;
    Vector<thermal::ThermalMaterial> thermalSolvedMaterials;
    String thermalStepperName;
    String thermalGpuIdentity;
    bool thermalCacheEligible = false;
    std::filesystem::path thermalCacheDir;
    /// What the last upload put on the GPU, so an hour that lands on the same
    /// field is not re-uploaded.
    f64 thermalUploadedHour = std::numeric_limits<f64>::quiet_NaN();

    /// Lends the scene to the epoch builder.
    struct OfflineEpochHost final : rendercore::EpochGeometryHost {
        Impl* owner = nullptr;
        f64 restore_s = 0.0;
        bool captured = false;
        VkAccelerationStructureKHR ApplyEpoch(f64 t_s) override;
        void Restore() override;
    };
    OfflineEpochHost epochHost;

    SetupResult BuildScene();
    SetupResult BuildIlluminants();
    SetupResult BuildPipeline();

    /// Set up the surface energy balance, if the scene asked for one: mesh,
    /// stepper, geometry schedule, trajectory. Leaves nothing on the GPU --
    /// that is UploadThermalFieldAt's job -- except on the cache-hit path,
    /// where the field is all there was to fetch.
    void BuildThermalSession();

    /// Put the field at one hour where the shader reads it. Always leaves a
    /// valid buffer behind -- a single zero entry when there was no solve --
    /// because an unbound descriptor is not a valid one.
    void UploadThermalFieldAt(f64 time_h);

    /// Put one field on the GPU, rebinding only when the buffer had to grow.
    /// An empty result is the "no solve" record every path must still leave.
    void UploadThermalResult(const thermal::ThermalResult& result);

    /// The hour the timeline's current second maps to, or the config's own
    /// `thermal.time_h` when there is no timeline.
    [[nodiscard]] f64 ThermalHourNow() const;

    OfflineRenderOutput RenderHyperspectral();
    OfflineRenderOutput RenderSingleFrame();
    Result<Image, String> RenderCameraWavelength(f64 wavelengthNm,
                                                 u64 acquisitionIndex,
                                                 u32 seed);
    Result<CameraData, String> CameraDataAt(f64 timeSeconds, SpectralMode mode,
                                           f64 wavelengthNm,
                                           bool physicalSensorProjection) const;
};

// ============================================================================
// Scene and lighting
// ============================================================================

SetupResult OfflineRenderer::Impl::BuildScene() {
    VulkanContext& context = *contextRef;

    QL_LOG_INFO("Loading scene...");

    rendercore::SceneLoadInfo sceneInfo;
    auto sceneResult =
        rendercore::LoadSceneFromConfig(config, configOptions.baseDir, &resolved, &sceneInfo);
    if (!sceneResult.has_value()) {
        return SetupResult::Err("Failed to load scene: " + sceneResult.error());
    }

    loadedScene = sceneResult.value();

    // The camera and the resolution the config asked for, onto the scene.
    //
    // Every mode but one reads them off `resolved` and never looks at the
    // Scene's copies, which is why those copies sat at their defaults --
    // rendercore::LoadSceneFromConfig loads geometry and nothing else. The
    // exception is the hyperspectral path: BatchRenderer takes a `Scene&` and
    // reads `scene.camera`, `scene.width` and `scene.height` from it. So a
    // cube was rendered at 1280x720 from the default camera while the log
    // above printed the resolution the config asked for, and the run exited 0.
    loadedScene.camera = resolved.camera;
    loadedScene.width = resolved.width;
    loadedScene.height = resolved.height;

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

    for (auto& mat : loadedScene.materials) {
        if (auto it = spectra.bandAveragedIREmissivity.find(mat.name);
            it != spectra.bandAveragedIREmissivity.end()) {
            mat.bandAveragedIREmissivity = it->second;
        }
    }

    // The clock, once [[nodes]] has had its say: the rest poses it snapshots
    // are the ones that pass just wrote.
    timeline = rendercore::TimelineState::Build(loadedScene, resolved.timeline,
                                                resolved.models, sceneInfo,
                                                spectra.nodeMotion, configReport);
    if (timeline.Present()) {
        timeline.Apply(loadedScene, resolved.timeline.time_s);
    }

    // Endmember weight maps, unmixed out of the base colours. Here because it
    // needs both what the resolve just produced (the endmember colours) and
    // what the upload is about to destroy (the base-colour pixels): the window
    // is exactly between the two.
    rendercore::BuildUnmixWeightTextures(loadedScene, spectra, configOptions.baseDir,
                                         configReport);

    // Authored temperature maps, in the same pre-upload window: the mount
    // appends textures, and the upload fixes the indices.
    rendercore::MountTemperatureTextures(loadedScene, configOptions.baseDir, configReport);

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

void OfflineRenderer::Impl::UploadThermalResult(const thermal::ThermalResult& result) {
    VulkanContext& context = *contextRef;

    // Uploaded in place whenever the size did not move, which is every frame of
    // a sequence: the pipeline's descriptor points at the buffer OBJECT, so
    // replacing it without rebinding would leave the shader reading memory that
    // has been freed.
    const auto upload = [&](std::unique_ptr<GpuBuffer>& buffer, const void* data,
                            usize bytes, auto&& rebind) {
        if (buffer && buffer->GetSize() == bytes) {
            buffer->Upload(data, bytes);
            return;
        }
        buffer = std::make_unique<GpuBuffer>(context.GetAllocator(), bytes,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VMA_MEMORY_USAGE_CPU_TO_GPU);
        buffer->Upload(data, bytes);
        // Null before BuildPipeline runs, which is the first call; the pipeline
        // picks these up from the bindings struct then.
        if (pipeline) rebind(*buffer);
    };

    const f32 zero = 0.0f;
    const bool empty = result.surfaceTemperature_K.empty();
    upload(thermalTemperatureBuffer,
           empty ? static_cast<const void*>(&zero)
                 : static_cast<const void*>(result.surfaceTemperature_K.data()),
           empty ? sizeof(f32) : result.surfaceTemperature_K.size() * sizeof(f32),
           [this](const GpuBuffer& b) { pipeline->BindThermalTemperatureBuffer(b); });

    // The sun response goes up wherever the temperatures do, including on
    // every failure path: binding 26 has no partially-bound flag either, so
    // "no solve" is one inert record rather than no descriptor.
    const auto records = rendercore::MakeThermalSunResponse(
        result.sunSensitivity_K, result.sunVisibility, result.sunDirection,
        result.lagSensitivity_K, result.lagVisibility, result.lagDirection);
    upload(thermalSunResponseBuffer, records.data(),
           records.size() * sizeof(rendercore::ThermalSunResponseGpu),
           [this](const GpuBuffer& b) { pipeline->BindThermalSunResponseBuffer(b); });

    // Point the instances at their elements. This is the whole of how a
    // triangle in the shader finds the temperature the balance gave it.
    if (!result.instanceElementBase.empty()) {
        geometry.SetThermalElementBases(result.instanceElementBase);
    }
}

VkAccelerationStructureKHR OfflineRenderer::Impl::OfflineEpochHost::ApplyEpoch(const f64 t_s) {
    if (!captured) {
        restore_s = owner->timeline.Current_s();
        captured = true;
    }
    owner->timeline.Apply(owner->loadedScene, t_s);
    if (owner->geometry.IsValid()) {
        if (!owner->geometry.RefitTlas(*owner->contextRef, owner->loadedScene)) {
            owner->geometry.RebuildTlas(*owner->contextRef, owner->loadedScene);
        }
    }
    return owner->geometry.IsValid() ? owner->geometry.Tlas().GetHandle() : VK_NULL_HANDLE;
}

void OfflineRenderer::Impl::OfflineEpochHost::Restore() {
    if (!captured) return;
    owner->timeline.Apply(owner->loadedScene, restore_s);
    if (owner->geometry.IsValid()) {
        if (!owner->geometry.RefitTlas(*owner->contextRef, owner->loadedScene)) {
            owner->geometry.RebuildTlas(*owner->contextRef, owner->loadedScene);
        }
    }
    captured = false;
}

f64 OfflineRenderer::Impl::ThermalHourNow() const {
    if (timeline.Present() && timeline.ThermalMapped()) {
        return timeline.HourAt(timeline.Current_s());
    }
    return resolved.thermal.time_h;
}

void OfflineRenderer::Impl::BuildThermalSession() {
    VulkanContext& context = *contextRef;

    if (!resolved.thermal.enabled) {
        return;
    }

    QL_LOG_INFO("Running the thermal solver...");

    thermalConfig = resolved.thermal;
    thermalConfig.materials = spectra.thermalMaterials;
    if (timeline.Present()) {
        // With a clock, `thermal.time_h` is the hour the timeline starts at.
        timeline.SetThermalMapping(resolved.thermal.time_h,
                                   resolved.timeline.thermalTimeScale);
    }

    thermalMesh = thermal::BuildThermalMesh(loadedScene);

    // Which stepper will run, decided before the key is built: CPU and GPU
    // differ in f64 versus f32, so the answer they give is part of what an
    // entry stands for.
    //
    // The GPU one is opt-in and off by default, and will stay that way until
    // somebody states a per-band tolerance for it. A different reduction order
    // gives different floats, so it cannot be judged by the byte-equality the
    // rest of this path is held to -- what it is actually for is the wait
    // while authoring a scene, where 165 s per material edit is the cost.
    thermal::IThermalStepper* stepper = nullptr;
    thermalStepperName = thermal::CpuCrankNicolsonStepper::kName;
    if (EnvFlagEnabled("QUANTILOOM_THERMAL_GPU_STEPPER")) {
        if (thermalConfig.nodeCount > rendercore::GpuThermalStepper::kMaxNodes) {
            QL_LOG_INFO("  Thermal stepper: CPU ({} layers is past the GPU stepper's {})",
                        thermalConfig.nodeCount, rendercore::GpuThermalStepper::kMaxNodes);
        } else {
            thermalGpuStepper = std::make_unique<rendercore::GpuThermalStepper>(context);
            if (thermalGpuStepper->IsValid()) {
                stepper = thermalGpuStepper.get();
                thermalStepperName = stepper->Name();
            } else {
                thermalGpuStepper.reset();
                QL_LOG_INFO("  Thermal stepper: CPU (the GPU stepper would not start)");
            }
        }
    }
    QL_LOG_INFO("  Thermal stepper: {}", thermalStepperName);

    const auto cacheSettings = thermal::ResolveThermalSolveCacheSettings();
    thermalCacheEligible = cacheSettings.enabled && thermalConfig.dumpElementsFile.empty();
    thermalCacheDir = cacheSettings.directory;
    thermalSolvedMaterials = thermal::BuildSolvedMaterials(loadedScene, thermalConfig);
    {
        const VkPhysicalDeviceProperties& gpu = context.GetDeviceProperties();
        thermalGpuIdentity =
            thermal::MakeGpuIdentity(StringView(gpu.deviceName, sizeof(gpu.deviceName)),
                                     gpu.vendorID, gpu.deviceID, gpu.driverVersion);
    }

    // A static scene can ask the cache before measuring anything, which is
    // where the saving is: a hit skips the view-factor precompute as well as
    // the trajectory. A scene with a clock cannot -- the schedule is part of
    // what the key describes, and there is no way to know it without building
    // it -- so there the cache saves the stepping and not the measuring.
    if (!timeline.Present() && thermalCacheEligible) {
        thermal::ThermalSolveCacheKeyInputs keyInputs;
        keyInputs.mesh = &thermalMesh;
        keyInputs.solvedMaterials = &thermalSolvedMaterials;
        keyInputs.config = &thermalConfig;
        keyInputs.exchangeSunDirection = resolved.lighting.sunDirection;
        keyInputs.gpuIdentity = thermalGpuIdentity;
        keyInputs.stepperName = thermalStepperName;
        keyInputs.libVersion = version::LibVersionString;
        const String cacheKey = thermal::ComputeThermalSolveCacheKey(keyInputs);

        if (!cacheKey.empty()) {
            const std::filesystem::path cacheFile = thermalCacheDir / (cacheKey + ".qltc");
            if (auto cached = thermal::LoadThermalSolveCache(cacheFile, cacheKey, &thermalMesh)) {
                QL_LOG_INFO("  Thermal cache: hit ({}...)", cacheKey.substr(0, 12));
                // The gate line, from the entry rather than from a solve. A
                // render served from cache has to be indistinguishable in the
                // log from one that was not.
                thermal::LogThermalSolveSummary(*cached);
                UploadThermalResult(*cached);
                thermalUploadedHour = thermalConfig.time_h;
                return;
            }
            QL_LOG_INFO("  Thermal cache: miss; solving");
        }
    }

    // Where the geometry has to be measured. One entry unless the clock moves
    // something far enough to matter.
    Vector<f64> epochTimes_s{0.0};
    Vector<f64> epochFrom_h{0.0};
    const bool wantEpochs =
        timeline.Present() && timeline.HasMotion() &&
        resolved.timeline.thermalGeometry ==
            rendercore::TimelineConfig::ThermalGeometry::Epochs;
    if (timeline.Present()) {
        thermal::EpochPlanInput plan;
        plan.start_s = resolved.timeline.start_s;
        plan.end_s = resolved.timeline.end_s;
        plan.minMove_m = static_cast<f32>(resolved.timeline.thermalEpochMinMove_m);
        plan.stride_s = (resolved.timeline.thermalEpochStride_s > 0.0)
                            ? resolved.timeline.thermalEpochStride_s
                            : thermalConfig.timestep_s /
                                  std::max(resolved.timeline.thermalTimeScale, 1e-9);
        if (wantEpochs) {
            Vector<f64> changeTimes =
                timeline.ChangeTimes(resolved.timeline.start_s, resolved.timeline.end_s);
            for (const rendercore::AnimatedNode& animated : timeline.Animated()) {
                thermal::EpochPlanInput::Node node;
                node.boundRadius_m = animated.boundRadius_m;
                node.poseAt = [this, &animated](f64 t) { return timeline.PoseAt(animated, t); };
                node.changeTimes = std::move(changeTimes);
                changeTimes.clear();
                plan.nodes.push_back(std::move(node));
            }
            epochTimes_s = thermal::PlanEpochTimes(plan);
        } else {
            // Reference mode: one geometry, frozen at the instant the config
            // named.
            epochTimes_s = {resolved.timeline.thermalReference_s};
        }
        epochFrom_h.clear();
        epochFrom_h.reserve(epochTimes_s.size());
        for (const f64 t : epochTimes_s) epochFrom_h.push_back(timeline.HourAt(t));
    }

    // The view factors. On the GPU where there is one to run them on: the same
    // rays Aguerre et al. cast with Embree, against the acceleration structure
    // the render already built. A failure here is not a failed render -- the
    // solver falls back to treating every surface as seeing open sky, which
    // is right for a scene with nothing to shade anything else and too cold at
    // night for a street.
    epochHost.owner = this;
    rendercore::EpochBuildInput buildInput;
    buildInput.epochTimes_s = epochTimes_s;
    buildInput.epochFrom_h = epochFrom_h;
    buildInput.meshOptions = thermal::MeshOptionsFor(thermalSolvedMaterials,
                                                     thermalConfig.lateralConduction);
    buildInput.materials = &thermalSolvedMaterials;
    buildInput.precompute.hemisphereRays = resolved.thermal.exchangeRays;
    buildInput.precompute.topK = resolved.thermal.exchangeTopK;
    buildInput.precompute.sunDirection = resolved.lighting.sunDirection;
    const auto forcingSeries = thermal::LoadForcingCsv(thermalConfig.forcingFile);
    buildInput.forcingSeries = &forcingSeries;
    buildInput.fallbackSunDirection = resolved.lighting.sunDirection;
    buildInput.sunMemoryLags =
        thermalConfig.sunCorrection ? thermalConfig.sunMemoryLags : 0u;

    thermal::ThermalGeometrySchedule schedule = rendercore::BuildThermalGeometrySchedule(
        context, epochHost, loadedScene, buildInput, nullptr);
    if (schedule.Empty() || schedule.epochs.front().exchange.skyFraction.empty()) {
        QL_LOG_WARN("  Thermal: no view factors; every surface will be treated as "
                    "seeing open sky");
    }

    auto session = thermal::ThermalSolveSession::Build(loadedScene, thermalConfig,
                                                       std::move(schedule), stepper);
    if (!session) {
        QL_LOG_WARN("  Thermal: {}; the scene keeps the temperatures it was given",
                    session.error());
        return;
    }
    thermalSession = std::move(*session);
    if (thermalSession->EpochCount() > 1) {
        QL_LOG_INFO("  Thermal: {} geometry epoch(s) over the timeline",
                    thermalSession->EpochCount());
    }
}

void OfflineRenderer::Impl::UploadThermalFieldAt(const f64 time_h) {
    if (thermalSession == nullptr) {
        // Either there is no solve, or BuildThermalSession already uploaded a
        // cached field. The buffers must exist either way.
        if (!thermalTemperatureBuffer) UploadThermalResult({});
        return;
    }
    if (thermalUploadedHour == time_h) return;

    thermal::ThermalResult result;
    bool served = false;

    String cacheKey;
    std::filesystem::path cacheFile;
    if (thermalCacheEligible) {
        thermalConfig.time_h = time_h;
        thermal::ThermalSolveCacheKeyInputs keyInputs;
        keyInputs.mesh = &thermalMesh;
        keyInputs.solvedMaterials = &thermalSolvedMaterials;
        keyInputs.config = &thermalConfig;
        keyInputs.schedule = &thermalSession->Schedule();
        keyInputs.exchangeSunDirection = resolved.lighting.sunDirection;
        keyInputs.gpuIdentity = thermalGpuIdentity;
        keyInputs.stepperName = thermalStepperName;
        keyInputs.libVersion = version::LibVersionString;
        cacheKey = thermal::ComputeThermalSolveCacheKey(keyInputs);
        if (!cacheKey.empty()) {
            cacheFile = thermalCacheDir / (cacheKey + ".qltc");
            if (auto cached = thermal::LoadThermalSolveCache(cacheFile, cacheKey, &thermalMesh)) {
                QL_LOG_INFO("  Thermal cache: hit ({}...)", cacheKey.substr(0, 12));
                result = std::move(*cached);
                served = true;
            }
        }
    }

    if (!served) {
        result = thermalSession->FieldAt(time_h);
        if (!result.error.empty()) {
            QL_LOG_WARN("  Thermal: {}; the scene keeps the temperatures it was given",
                        result.error);
            UploadThermalResult({});
            return;
        }
        // Only reachable with an empty error, so a failed solve is never
        // stored. Nor is one carrying material-parameter tangents: the entry
        // format does not hold them -- they are a diagnostic output rather than
        // something the render reads -- and an entry that came back without
        // them would be a silently incomplete answer to a run that asked.
        if (thermalCacheEligible && !cacheKey.empty() && result.parameters.empty()) {
            thermal::StoreThermalSolveCache(cacheFile, cacheKey, result);
        }
    }

    thermal::LogThermalSolveSummary(result);
    UploadThermalResult(result);
    thermalUploadedHour = time_h;
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
    // RGB -> spectrum coefficients
    // ====================================================================
    // Fitted, not compiled in: see core/RgbToSpectrum.hpp. Also scene
    // independent, so a shared device fits it once for a whole batch.
    if (!rgbToSpectrumRef) {
        rgbToSpectrumBuffer = rendercore::CreateRgbToSpectrumBuffer(context);
        rgbToSpectrumRef = rgbToSpectrumBuffer.get();
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
        if (auto it = spectra.materialNameToEmissiveCurve.find(mat.name);
            it != spectra.materialNameToEmissiveCurve.end()) {
            slots.emissiveRadianceCurve = it->second;
            QL_LOG_INFO("  Material '{}': emitting from spectral curve index {}",
                        mat.name, it->second);
        }
        if (auto it = spectra.materialNameToEndmembers.find(mat.name);
            it != spectra.materialNameToEndmembers.end()) {
            const auto& slotsIn = it->second;
            slots.endmemberCurve1 = slotsIn.curves[1];
            slots.endmemberCurve2 = slotsIn.curves[2];
            slots.endmemberCurve3 = slotsIn.curves[3];
            slots.weightTexture = slotsIn.weightTextureIndex;
        }
        if (auto it = spectra.materialNameToSheenCurve.find(mat.name);
            it != spectra.materialNameToSheenCurve.end()) {
            slots.sheenReflectanceCurve = it->second;
            QL_LOG_INFO("  Material '{}': using sheen curve index {}", mat.name, it->second);
        }
        if (auto it = spectra.materialNameToClearcoatCurve.find(mat.name);
            it != spectra.materialNameToClearcoatCurve.end()) {
            slots.clearcoatReflectanceCurve = it->second;
            QL_LOG_INFO("  Material '{}': using clearcoat curve index {}", mat.name, it->second);
        }
        if (auto it = spectra.materialNameToDiffuseTransmissionCurve.find(mat.name);
            it != spectra.materialNameToDiffuseTransmissionCurve.end()) {
            slots.diffuseTransmissionColorCurve = it->second;
            QL_LOG_INFO("  Material '{}': using diffuse transmission curve index {}",
                        mat.name, it->second);
        }
        if (auto it = spectra.materialNameToIrEmissivityCurve.find(mat.name);
            it != spectra.materialNameToIrEmissivityCurve.end())
            slots.irEmissivityCurve = it->second;
        if (auto it = spectra.materialNameToIrTransmittanceCurve.find(mat.name);
            it != spectra.materialNameToIrTransmittanceCurve.end())
            slots.irTransmittanceCurve = it->second;
        if (auto it = spectra.materialNameToFluorescence.find(mat.name);
            it != spectra.materialNameToFluorescence.end()) {
            slots.fluorescenceExcitationCurve = it->second.excitationCurve;
            slots.fluorescenceEmissionCurve = it->second.emissionCurve;
            QL_LOG_INFO("  Material '{}': fluorescing, excitation curve {} and "
                        "emission curve {}",
                        mat.name, it->second.excitationCurve, it->second.emissionCurve);
        }

        materialIndices.push_back(slots);
    }

    const rendercore::MaterialResourceCounts materialResources{
        textureManager ? textureManager->GetTextureCount() : 0,
        spectra.curves.size(), spectra.refractiveIndices.size()};
    materialBuffer = rendercore::BuildMaterialBuffer(
        context, loadedScene, params.wavelengthNm, materialIndices, &materialResources);
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

    // Binding 10 has to hold a valid descriptor whether or not this scene has an
    // environment, so something is always bound. When that something is the
    // placeholder -- no map named, map disabled, spectral mode, or a load that
    // failed -- the lighting flag is 0 and the shader never reads it. The
    // placeholder is one black texel, identical for every scene, so a shared
    // device supplies one. A map the config *names* is this render's own.
    //
    // Load() reads through ImageIO::ReadImage, so .hdr and the LDR formats work
    // as well as .exr -- this used to call ReadEXR directly and take the
    // fallback for anything else.
    const auto useFallback = [&] {
        if (fallbackEnvMapRef) {
            envMapRef = fallbackEnvMapRef;
        } else {
            envMap = rendercore::EnvironmentCubemap::Fallback(context);
            envMapRef = &envMap;
        }
        // Resolve assumed a map it can see named would load. The lighting buffer
        // was filled and uploaded from that assumption before this function ran,
        // so the correction belongs here, where the placeholder is actually
        // chosen -- anything else leaves the shader sampling a black cubemap as
        // if it were sky. For the paths that reach here with the flag already 0
        // (no map, disabled, spectral) this re-upload is a no-op that costs one
        // memcpy at startup.
        lightingParams.enableEnvironmentMap = 0u;
        lightingParamsBuffer->Upload(&lightingParams, sizeof(LightingParams));
    };

    // The mode gate mirrors the resolver, which sets the flag to 0 outside RGB:
    // without it we would convert an equirect to a cubemap that no shader branch
    // will ever sample. ApplyConfig gates the interactive path the same way.
    const String envMapPath =
        (resolved.environmentMapEnabled && resolved.mode == SpectralMode::RGB)
            ? resolved.environmentMap
            : String{};
    if (envMapPath.empty()) {
        QL_LOG_INFO("  No environment map for this render, binding the placeholder "
                    "-- image-based lighting is off");
        useFallback();
    } else {
        auto loaded = rendercore::EnvironmentCubemap::Load(context, envMapPath);
        if (loaded.has_value()) {
            envMap = std::move(loaded.value());
            envMapRef = &envMap;
        } else {
            QL_LOG_WARN("  {} -- binding the placeholder and rendering without "
                        "image-based lighting", loaded.error());
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
    bindings.rgbToSpectrum = rgbToSpectrumRef;
    bindings.emissiveTriangles = emissiveTriangleBuffer.get();
    bindings.thermalTemperatures = thermalTemperatureBuffer.get();
    bindings.thermalSunResponse = thermalSunResponseBuffer.get();

    pipeline = rendercore::CreateRayTracingPipeline(context, pipelineCache, bindings);

    CameraData cameraData = resolved.camera.GetCameraData();
    cameraData.wavelength_nm = params.wavelengthNm;         // Override with config wavelength
    cameraData.spectral_mode = static_cast<u32>(params.mode);  // Set rendering mode
    cameraData.debug_mode = static_cast<u32>(resolved.debugMode);
    // Nothing in a config selects a debug view's parameter; the offline
    // path renders slot zero, which is the whole-day response.
    cameraData.debugParam = 0;
    pipeline->SetCameraData(cameraData);

    pipeline->SetSpecConstants(
        static_cast<u32>(params.mode),
        cameraData.debug_mode != 0);

    QL_LOG_INFO("  Pipeline created and resources bound");

    // GPU timestamps around each trace; per-sample cost is logged after
    // every fence wait in the render loop
    perfLogger = std::make_unique<PerformanceLogger>(context);

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
        impl.rgbToSpectrumRef = shared.rgbToSpectrumBuffer.get();
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

    // Emissive geometry for next-event estimation. Before the upload below,
    // because the triangle count and total power travel in LightingParams --
    // and after BuildScene, because they are world-space and need the node
    // transforms. Nothing moves during an offline render, so this runs once.
    {
        const auto emissiveTris = impl.resolved.enableLightSampling
            ? rendercore::CollectEmissiveTriangles(impl.loadedScene)
            : Vector<rendercore::EmissiveTriangleGPU>{};
        impl.emissiveTriangleBuffer =
            rendercore::CreateEmissiveTriangleBuffer(*impl.contextRef, emissiveTris);
        impl.lightingParams.emissiveTriangleCount =
            static_cast<u32>(emissiveTris.size());
        impl.lightingParams.emissiveTotalPower =
            emissiveTris.empty() ? 0.0f : emissiveTris.back().cumulativePower;
    }

    // The surface energy balance, if the scene asked for one. After the
    // geometry, because the view factors are cast against the acceleration
    // structure it built; before the pipeline, because what comes out is a
    // buffer the pipeline binds.
    impl.BuildThermalSession();
    impl.UploadThermalFieldAt(impl.ThermalHourNow());

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
    const auto pose = m_impl->CameraDataAt(m_impl->timeline.Current_s(),
        m_impl->params.mode, m_impl->params.wavelengthNm, false);
    if (!pose) return OfflineRenderOutput{{}, 1.0f, 0.0f, false, pose.error()};
    m_impl->pipeline->SetCameraData(pose.value());
    if (m_impl->params.mode == SpectralMode::Multispectral) {
        return m_impl->RenderHyperspectral();
    }
    return m_impl->RenderSingleFrame();
}

const camera::CameraConfig& OfflineRenderer::GetCameraConfig() const {
    return m_impl->resolved.cameraConfig;
}

Result<CameraData, String> OfflineRenderer::Impl::CameraDataAt(
    f64 timeSeconds, SpectralMode mode, f64 wavelengthNm,
    bool physicalSensorProjection) const {
    Camera poseCamera = loadedScene.camera;
    const auto& motion = resolved.cameraConfig.motion;
    if (!motion.keys.empty()) {
        const auto pose = camera::CameraPoseAt(motion, timeSeconds);
        if (!pose) return Result<CameraData, String>::Err(pose.error());
        const auto position = glm::vec3(
            static_cast<f32>(pose.value().position[0]),
            static_cast<f32>(pose.value().position[1]),
            static_cast<f32>(pose.value().position[2]));
        const auto lookAt = glm::vec3(
            static_cast<f32>(pose.value().lookAt[0]),
            static_cast<f32>(pose.value().lookAt[1]),
            static_cast<f32>(pose.value().lookAt[2]));
        poseCamera = Camera(position, lookAt, loadedScene.camera.GetUpReference(),
                            loadedScene.camera.GetFovY(),
                            loadedScene.camera.GetAspectRatio());
        poseCamera.SetProjection(loadedScene.camera.GetProjection());
        poseCamera.SetOrthoHeight(loadedScene.camera.GetOrthoHeight());
    }
    CameraData data = poseCamera.GetCameraData();
    data.wavelength_nm = static_cast<f32>(wavelengthNm);
    data.spectral_mode = static_cast<u32>(mode);
    data.debug_mode = physicalSensorProjection ? 0u :
                      static_cast<u32>(resolved.debugMode);
    data.debugParam = 0u;
    if (physicalSensorProjection) {
        if (poseCamera.GetProjection() != Camera::Projection::Perspective)
            return Result<CameraData, String>::Err(
                "physical camera capture requires a perspective projection");
        const auto& optics = resolved.cameraConfig.optics;
        const auto fovX = camera::HorizontalFovRadians(
            optics.focalLengthMm, optics.pixelPitchUm, optics.sensorWidthPx);
        if (!fovX) return Result<CameraData, String>::Err(fovX.error());
        const f64 aspect = static_cast<f64>(optics.sensorWidthPx) /
                           optics.sensorHeightPx;
        data.aspectRatio = static_cast<f32>(aspect);
        data.fovScale = static_cast<f32>(std::tan(fovX.value() / 2.0) / aspect);
    }
    return data;
}

Result<camera::CameraOutput, String>
OfflineRenderer::CaptureCamera(camera::CaptureState& state, f64 frameTimeSeconds) {
    return CaptureCameraInternal(state, frameTimeSeconds, /*suppressProducts=*/false);
}

Result<camera::CameraOutput, String>
OfflineRenderer::CaptureCameraInternal(camera::CaptureState& state,
                                       f64 frameTimeSeconds,
                                       bool suppressProducts) {
    Impl& impl = *m_impl;
    const f64 enteredTime = impl.timeline.Current_s();
    struct RestoreSceneTime {
        OfflineRenderer& renderer;
        Impl& impl;
        f64 timeSeconds;
        ~RestoreSceneTime() {
            (void)renderer.SetTimelineTime(timeSeconds);
            const auto cameraData = impl.CameraDataAt(
                timeSeconds, impl.params.mode, impl.params.wavelengthNm, false);
            if (cameraData && impl.pipeline) impl.pipeline->SetCameraData(cameraData.value());
        }
    } restoreTime{*this, impl, enteredTime};
    camera::CameraConfig config = impl.resolved.cameraConfig;
    if (!config.enabled)
        return Result<camera::CameraOutput, String>::Err("camera capture is disabled");
    if (auto checked = camera::ValidateCameraConfig(config); !checked)
        return Result<camera::CameraOutput, String>::Err(checked.error());
    // The shader's solar/sky LUT clamps outside its supplied span. That is a
    // useful old-scene preview fallback, but cannot support an exact device
    // measurement of a response whose tail was never supplied.
    if (config.inputKind == camera::CameraInputKind::SpectralMeasurement &&
        impl.resolved.solarSunSky) {
        const auto& [sun, sky] = *impl.resolved.solarSunSky;
        for (const auto& channel : config.device.channels) {
            const auto& response = channel.response;
            const auto& base = response.systemResponse ? *response.systemResponse :
                (config.device.detector == camera::DetectorKind::Photon ?
                 *response.quantumEfficiency : *response.thermalAbsorptance);
            if (sun.samples.empty() || sky.samples.empty() ||
                sun.samples.front().first > base.MinNm() ||
                sun.samples.back().first < base.MaxNm() ||
                sky.samples.front().first > base.MinNm() ||
                sky.samples.back().first < base.MaxNm())
                return Result<camera::CameraOutput, String>::Err(
                    "camera response exceeds solar/sky LUT coverage");
        }
    }
    // Observer products are rendered by the facade, not fabricated by the
    // sensor from its measurement. The CPU chain owns only detector products.
    const bool wantsObserver = config.products.cieLinearSrgb;
    const bool wantsTraced = config.products.tracedRadiance;
    if (wantsObserver && !IsVisMode(impl.params.mode))
        return Result<camera::CameraOutput, String>::Err(
            "CIE observer product requires a visible spectral render mode");
    if (wantsTraced && impl.params.mode != SpectralMode::Single)
        return Result<camera::CameraOutput, String>::Err(
            "spectral radiance cube request requires single-wavelength mode");
    config.products.cieLinearSrgb = false;
    config.products.tracedRadiance = false;
    if (suppressProducts) {
        // A skipped-tick advance: the acquisition itself must run, because
        // the detector state feeds back from it, but nothing is written and
        // the observer/trace renders below are skipped entirely.
        config.products.bandMeasurement = false;
        config.products.rawDn = false;
        config.products.correctedDeviceSignal = false;
        config.products.apparentTemperature = false;
        config.products.display = false;
    }
    camera::CpuCameraPipeline camera(std::move(config));
    camera::CaptureState nextState = state;
    const auto sampler = [&](f64 timeSeconds, f64 wavelengthNm)
        -> Result<Image, String> {
        auto moved = SetTimelineTime(timeSeconds);
        if (!moved) return Result<Image, String>::Err(moved.error());
        return impl.RenderCameraWavelength(wavelengthNm, state.acquisitionIndex,
                                           impl.resolved.cameraConfig.randomSeed);
    };
    Result<camera::CameraOutput, String> captured =
        Result<camera::CameraOutput, String>::Err("camera capture did not run");
    if (impl.resolved.cameraConfig.inputKind ==
        camera::CameraInputKind::FastRgbApproximation) {
        auto moved = SetTimelineTime(frameTimeSeconds);
        if (!moved) return Result<camera::CameraOutput, String>::Err(moved.error());
        const auto pose = impl.CameraDataAt(frameTimeSeconds, SpectralMode::RGB,
                                            impl.params.wavelengthNm, false);
        if (!pose) return Result<camera::CameraOutput, String>::Err(pose.error());
        impl.pipeline->SetCameraData(pose.value());
        OfflineRenderOutput fast = impl.RenderSingleFrame();
        if (!fast.error.empty())
            return Result<camera::CameraOutput, String>::Err(fast.error);
        const u32 sensorWidth = impl.resolved.cameraConfig.optics.sensorWidthPx;
        const u32 sensorHeight = impl.resolved.cameraConfig.optics.sensorHeightPx;
        Image rgb(sensorWidth, sensorHeight, 3);
        rgb.channelNames = {"R", "G", "B"};
        // The fast entrance is explicitly approximate. Resize its preview
        // radiance onto the physical pixel grid, leaving pixel collection area
        // tied solely to pitch and fill factor in the detector calculation.
        for (u32 y = 0; y < sensorHeight; ++y)
            for (u32 x = 0; x < sensorWidth; ++x) {
                const f64 sourceX = (x + 0.5) * fast.radiance.width / sensorWidth - 0.5;
                const f64 sourceY = (y + 0.5) * fast.radiance.height / sensorHeight - 0.5;
                const i32 x0 = std::clamp(static_cast<i32>(std::floor(sourceX)), 0,
                                          static_cast<i32>(fast.radiance.width) - 1);
                const i32 y0 = std::clamp(static_cast<i32>(std::floor(sourceY)), 0,
                                          static_cast<i32>(fast.radiance.height) - 1);
                const i32 x1 = std::min(x0 + 1, static_cast<i32>(fast.radiance.width) - 1);
                const i32 y1 = std::min(y0 + 1, static_cast<i32>(fast.radiance.height) - 1);
                const f64 fx = std::clamp(sourceX - x0, 0.0, 1.0);
                const f64 fy = std::clamp(sourceY - y0, 0.0, 1.0);
                for (u32 c = 0; c < 3; ++c) {
                    const f64 top = (1.0 - fx) * fast.radiance(x0, y0, c) +
                                    fx * fast.radiance(x1, y0, c);
                    const f64 bottom = (1.0 - fx) * fast.radiance(x0, y1, c) +
                                       fx * fast.radiance(x1, y1, c);
                    rgb(x, y, c) = static_cast<f32>((1.0 - fy) * top + fy * bottom);
                }
            }
        captured = camera.CaptureFastRgb(nextState, frameTimeSeconds, rgb);
    } else {
        captured = camera.Capture(nextState, frameTimeSeconds, sampler);
    }
    if (!captured) return captured;
    // Freeze sensor-product state before an optional observer render moves the
    // clock again. Asking for a CIE/traced product must not rewrite the capture
    // description of a sensor image that has already been generated.
    if (!suppressProducts && impl.config.GetBool("dataset.metadata", true)) {
        const bool physicalProjection = impl.resolved.cameraConfig.inputKind !=
            camera::CameraInputKind::FastRgbApproximation;
        const auto freeze = [&](std::optional<camera::CameraProduct>& product) {
            if (product) product->image.metadata["quantiloom_provenance"] =
                impl.ProductSnapshot(frameTimeSeconds, product->image.width,
                    product->image.height, physicalProjection);
        };
        freeze(captured.value().bandMeasurement);
        freeze(captured.value().rawDn);
        freeze(captured.value().correctedDeviceSignal);
        freeze(captured.value().apparentTemperature);
        freeze(captured.value().display);
    }
    if (!suppressProducts && (wantsObserver || wantsTraced)) {
        auto moved = SetTimelineTime(frameTimeSeconds);
        if (!moved) return Result<camera::CameraOutput, String>::Err(moved.error());
        const auto pose = impl.CameraDataAt(frameTimeSeconds, impl.params.mode,
                                            impl.params.wavelengthNm, false);
        if (!pose) return Result<camera::CameraOutput, String>::Err(pose.error());
        impl.pipeline->SetCameraData(pose.value());
        OfflineRenderOutput observer = impl.RenderSingleFrame();
        if (!observer.error.empty())
            return Result<camera::CameraOutput, String>::Err(observer.error);
        if (wantsObserver) {
            camera::SignalDescriptor signal;
            signal.kind = camera::SignalKind::CieLinearSrgb;
            signal.unit = "linear_sRGB";
            signal.calibration = impl.resolved.cameraConfig.device.calibration;
            signal.acquisitionIndex = state.acquisitionIndex;
            signal.exposureStartSeconds = frameTimeSeconds;
            signal.exposureEndSeconds = frameTimeSeconds;
            Image image(observer.radiance.width, observer.radiance.height, 3);
            image.channelNames = {"R", "G", "B"};
            for (u32 y = 0; y < image.height; ++y)
                for (u32 x = 0; x < image.width; ++x)
                    for (u32 c = 0; c < 3; ++c)
                        image(x, y, c) = observer.radiance(x, y, c);
            if (const auto frozen = observer.radiance.metadata.find("quantiloom_provenance"); frozen != observer.radiance.metadata.end())
                image.metadata["quantiloom_provenance"] = frozen->second;
            camera::CameraProduct product{std::move(image), std::move(signal)};
            auto annotated = camera::AnnotateProductMetadata(product);
            if (!annotated)
                return Result<camera::CameraOutput, String>::Err(annotated.error());
            captured.value().cieLinearSrgb = std::move(product);
        }
        if (wantsTraced) {
            camera::SignalDescriptor signal;
            signal.kind = camera::SignalKind::SpectralRadiance;
            signal.unit = "W/m^2/sr/nm";
            signal.channelWavelengthNm = {impl.params.wavelengthNm};
            signal.calibration = impl.resolved.cameraConfig.device.calibration;
            signal.acquisitionIndex = state.acquisitionIndex;
            signal.exposureStartSeconds = frameTimeSeconds;
            signal.exposureEndSeconds = frameTimeSeconds;
            Image image(observer.radiance.width, observer.radiance.height, 1);
            image.channelNames = {"L_" + std::to_string(impl.params.wavelengthNm)};
            for (u32 y = 0; y < image.height; ++y)
                for (u32 x = 0; x < image.width; ++x)
                    image(x, y, 0) = observer.radiance(x, y, 0);
            if (const auto frozen = observer.radiance.metadata.find("quantiloom_provenance"); frozen != observer.radiance.metadata.end())
                image.metadata["quantiloom_provenance"] = frozen->second;
            camera::CameraProduct product{std::move(image), std::move(signal)};
            auto annotated = camera::AnnotateProductMetadata(product);
            if (!annotated)
                return Result<camera::CameraOutput, String>::Err(annotated.error());
            captured.value().tracedRadiance = std::move(product);
        }
    }
    const bool volumeRgbAssumption = std::any_of(
        impl.loadedScene.materials.begin(), impl.loadedScene.materials.end(),
        [](const Material& material) {
            return material.volumeDensity > 0.0f && material.scatteringCoeff > 0.0f;
        });
    if (!impl.resolved.solarSunSky || impl.spectra.rgbUpsampledMaterials > 0 ||
        volumeRgbAssumption ||
        (impl.timeline.Present() &&
         frameTimeSeconds - impl.resolved.cameraConfig.readout.exposureSeconds / 2.0 <
             impl.timeline.Info().start_s)) {
        String assumptions;
        if (!impl.resolved.solarSunSky)
            assumptions = "no_measured_solar_sky_lut";
        if (impl.spectra.rgbUpsampledMaterials > 0) {
            if (!assumptions.empty()) assumptions += ",";
            assumptions += "rgb_upsampled_materials:" +
                std::to_string(impl.spectra.rgbUpsampledMaterials);
        }
        if (volumeRgbAssumption) {
            if (!assumptions.empty()) assumptions += ",";
            assumptions += "volume_extinction_rgb_mean";
        }
        const bool preTimeline = impl.timeline.Present() &&
            frameTimeSeconds - impl.resolved.cameraConfig.readout.exposureSeconds / 2.0 <
                impl.timeline.Info().start_s;
        if (preTimeline) {
            if (!assumptions.empty()) assumptions += ",";
            assumptions += "pre_timeline_initial_hold";
        }
        const auto annotate = [&assumptions](std::optional<camera::CameraProduct>& product) {
            if (product) product->image.metadata["camera_scene_assumptions"] = assumptions;
        };
        annotate(captured.value().tracedRadiance);
        annotate(captured.value().cieLinearSrgb);
        annotate(captured.value().bandMeasurement);
        annotate(captured.value().rawDn);
        annotate(captured.value().correctedDeviceSignal);
        annotate(captured.value().apparentTemperature);
        annotate(captured.value().display);
        if (preTimeline) {
            const auto mark = [](std::optional<camera::CameraProduct>& product) {
                if (product) product->image.metadata["camera_pre_timeline_policy"] =
                    "initial_scene_hold";
            };
            mark(captured.value().tracedRadiance);
            mark(captured.value().cieLinearSrgb);
            mark(captured.value().bandMeasurement);
            mark(captured.value().rawDn);
            mark(captured.value().correctedDeviceSignal);
            mark(captured.value().apparentTemperature);
            mark(captured.value().display);
        }
    }
    // The scene ray tracer still stores reflectance and n,k as a 64-point GPU
    // grid. This is separate from the CPU camera's convergent wavelength
    // integration and from the exact source-knot chains for lamps/fluorescence.
    const auto markSceneGrid = [](std::optional<camera::CameraProduct>& product) {
        if (!product) return;
        product->image.metadata["camera_scene_reflectance_nk_grid_samples"] = "64";
        product->image.metadata["camera_scene_reflectance_nk_resampling"] =
            "uniform_gpu_grid";
    };
    markSceneGrid(captured.value().tracedRadiance);
    markSceneGrid(captured.value().cieLinearSrgb);
    markSceneGrid(captured.value().bandMeasurement);
    markSceneGrid(captured.value().rawDn);
    markSceneGrid(captured.value().correctedDeviceSignal);
    markSceneGrid(captured.value().apparentTemperature);
    markSceneGrid(captured.value().display);
    if (!suppressProducts && impl.config.GetBool("dataset.metadata", true)) {
        using Json = nlohmann::json;
        const auto summarizeState = [](const camera::CaptureState& value) {
            core::Sha256 hash;
            hash.UpdateU32(value.version);
            hash.UpdateU64(value.acquisitionIndex);
            hash.UpdateF64(value.frameTimeSeconds);
            hash.UpdateF64(value.nextExposureSeconds);
            hash.UpdateF64(value.nextAnalogGain);
            for (f64 channel : value.nextWhiteBalance) hash.UpdateF64(channel);
            hash.UpdateU64(value.thermalPixelStateW.size());
            for (f64 pixel : value.thermalPixelStateW) hash.UpdateF64(pixel);
            hash.UpdateU64(value.historyEpoch);
            return Json{{"sha256", hash.FinalizeHex()}, {"acquisition_index", value.acquisitionIndex},
                        {"frame_time_s", value.frameTimeSeconds}, {"history_epoch", value.historyEpoch}};
        };
        const auto before = summarizeState(state), after = summarizeState(nextState);
        const auto annotate = [&](std::optional<camera::CameraProduct>& product, bool physical) {
            if (!product) return;
            const bool fastRgb = impl.resolved.cameraConfig.inputKind == camera::CameraInputKind::FastRgbApproximation;
            const auto frozen = product->image.metadata.find("quantiloom_provenance");
            auto snapshot = frozen != product->image.metadata.end()
                ? Json::parse(frozen->second)
                : Json::parse(impl.ProductSnapshot(frameTimeSeconds,
                    product->image.width, product->image.height, physical && !fastRgb));
            if (physical && fastRgb) snapshot["processing"] = "fast_rgb_bilinear_resample_to_sensor_grid";
            snapshot["capture"] = {{"acquisition_index", product->signal.acquisitionIndex},
                {"reference_time_s", frameTimeSeconds},
                {"reference_time_definition", "first_row_exposure_midpoint"},
                {"exposure_start_s", product->signal.exposureStartSeconds},
                {"exposure_end_s", product->signal.exposureEndSeconds},
                {"row_delay_s", physical && impl.resolved.cameraConfig.readout.shutter == camera::ShutterKind::Rolling ?
                    impl.resolved.cameraConfig.readout.rowDelaySeconds : 0.0},
                {"first_row_exposure_s", physical ? 2.0 * (frameTimeSeconds - product->signal.exposureStartSeconds) : 0.0},
                {"row_midpoint_formula", "reference_time_s + row_index * row_delay_s"},
                {"state_before", before}, {"state_after", after}};
            snapshot["units"] = product->signal.unit;
            snapshot["channel_wavelength_nm"] = product->signal.channelWavelengthNm;
            snapshot["camera_config_toml"] = CameraConfigToToml(impl.resolved.cameraConfig);
            product->image.metadata["quantiloom_provenance"] = snapshot.dump();
        };
        annotate(captured.value().tracedRadiance, false);
        annotate(captured.value().cieLinearSrgb, false);
        annotate(captured.value().bandMeasurement, true);
        annotate(captured.value().rawDn, true);
        annotate(captured.value().correctedDeviceSignal, true);
        annotate(captured.value().apparentTemperature, true);
        annotate(captured.value().display, true);
    }
    state = std::move(nextState);
    return captured;
}

Result<camera::CaptureCheckpoint, String>
OfflineRenderer::CheckpointCamera(const camera::CaptureState& state) {
    return camera::CheckpointCamera(state);
}

Result<void, String>
OfflineRenderer::RestoreCamera(camera::CaptureState& state,
                               const camera::CaptureCheckpoint& checkpoint) {
    return camera::RestoreCamera(state, checkpoint);
}

Result<void, String>
OfflineRenderer::AdvanceCameraState(camera::CaptureState& state, f64 timeSeconds) {
    const auto advance = [this](camera::CaptureState& working,
                                f64 t) -> Result<void, String> {
        auto captured = CaptureCameraInternal(working, t, /*suppressProducts=*/true);
        if (!captured)
            return Result<void, String>::Err(captured.error());
        return Result<void, String>::Ok();
    };
    return camera::AdvanceCameraState(state, timeSeconds, advance);
}

Result<void, String>
OfflineRenderer::WarmUpCamera(camera::CaptureState& state, f64 seconds,
                              f64 framePeriodSeconds) {
    const auto advance = [this](camera::CaptureState& working,
                                f64 t) -> Result<void, String> {
        auto captured = CaptureCameraInternal(working, t, /*suppressProducts=*/true);
        if (!captured)
            return Result<void, String>::Err(captured.error());
        return Result<void, String>::Ok();
    };
    return camera::WarmUpCamera(state, seconds, framePeriodSeconds, advance);
}

Result<Image, String> OfflineRenderer::Impl::RenderCameraWavelength(
    f64 wavelengthNm, u64 acquisitionIndex, u32 seed) {
    if (!std::isfinite(wavelengthNm) || wavelengthNm <= 0.0 ||
        wavelengthNm > std::numeric_limits<f32>::max())
        return Result<Image, String>::Err("invalid camera spectral sample wavelength");
    const auto originalPose = CameraDataAt(
        timeline.Current_s(), params.mode, params.wavelengthNm, false);
    if (!originalPose) return Result<Image, String>::Err(originalPose.error());
    CameraData original = originalPose.value();
    struct RestoreBindings {
        Impl& host;
        CameraData original;
        ~RestoreBindings() {
            host.pipeline->BindOutputImage(*host.outputImage);
            host.pipeline->SetCameraData(original);
            host.pipeline->SetSpecConstants(static_cast<u32>(host.params.mode),
                                            original.debug_mode != 0);
            host.pipeline->BindAtmosphereNN(host.atmosHeaderBuffer.get(),
                                            host.atmosDataBuffer.get());
        }
    } restore{*this, original};
    if (resolved.atmosphere.enabled) {
        const auto grid = RenderBandLambdaGrid(SpectralMode::Single, wavelengthNm);
        if (!grid.error.empty() || grid.band.empty())
            return Result<Image, String>::Err("camera wavelength lacks atmosphere coverage: " +
                (grid.error.empty() ? std::to_string(wavelengthNm) : grid.error));
        try {
            AtmosModelPack pack(resolved.atmosphere.modelPackDir);
            AtmosphereBaker baker(pack);
            auto baked = baker.Bake(resolved.atmosphere, grid.band, grid.lambdasNm,
                                    grid.windowHalfWidthNm);
            const glm::vec3& sunDir = resolved.lighting.sunDirection;
            baked.header.sunDirWorld[0] = sunDir.x;
            baked.header.sunDirWorld[1] = sunDir.y;
            baked.header.sunDirWorld[2] = sunDir.z;
            baked.header.worldUnitsToMeters = resolved.worldUnitsToMeters;
            cameraAtmosHeaderBuffer = std::make_unique<GpuBuffer>(
                contextRef->GetAllocator(), sizeof(AtmosNNHeaderGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            cameraAtmosHeaderBuffer->Upload(&baked.header, sizeof(AtmosNNHeaderGPU));
            cameraAtmosDataBuffer = std::make_unique<GpuBuffer>(
                contextRef->GetAllocator(), baked.data.size() * sizeof(f32),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            cameraAtmosDataBuffer->Upload(baked.data.data(),
                                           baked.data.size() * sizeof(f32));
            pipeline->BindAtmosphereNN(cameraAtmosHeaderBuffer.get(),
                                        cameraAtmosDataBuffer.get());
        } catch (const std::exception& e) {
            return Result<Image, String>::Err(
                String("camera atmosphere bake failed: ") + e.what());
        }
    }
    if (!cameraBatchRenderer)
        cameraBatchRenderer =
            std::make_unique<BatchRenderer>(*contextRef, *pipeline, loadedScene);
    BatchRenderParams sample;
    sample.spp = params.spp;
    sample.frameIndex = static_cast<u32>(acquisitionIndex);
    sample.frameIndexHigh = static_cast<u32>(acquisitionIndex >> 32);
    sample.renderSeed = seed;
    sample.outputWidth = resolved.cameraConfig.optics.sensorWidthPx;
    sample.outputHeight = resolved.cameraConfig.optics.sensorHeightPx;
    const auto cameraPose = CameraDataAt(
        timeline.Current_s(), SpectralMode::Single, wavelengthNm, true);
    if (!cameraPose) return Result<Image, String>::Err(cameraPose.error());
    CameraData cameraData = cameraPose.value();
    Image image;
    if (!cameraBatchRenderer->RenderSingleBand(
            static_cast<f32>(wavelengthNm), sample, cameraData, image))
        return Result<Image, String>::Err("single-wavelength camera render failed");
    image.channelNames = {"L_" + std::to_string(wavelengthNm)};
    image.metadata["wavelength_nm"] = std::to_string(wavelengthNm);
    image.metadata["units"] = "W/m^2/sr/nm";
    return image;
}

Result<void, String> OfflineRenderer::SetTimelineTime(const f64 t_s) {
    Impl& impl = *m_impl;
    if (!impl.timeline.Present()) {
        impl.timeline.SetCurrent(t_s);
        return Result<void, String>::Ok();
    }

    const Vector<u32> moved = impl.timeline.Apply(impl.loadedScene, t_s);
    if (!moved.empty() && impl.geometry.IsValid()) {
        if (!impl.geometry.RefitTlas(*impl.contextRef, impl.loadedScene)) {
            impl.geometry.RebuildTlas(*impl.contextRef, impl.loadedScene);
            if (impl.pipeline) {
                impl.pipeline->BindAccelerationStructure(impl.geometry.Tlas().GetHandle());
                if (impl.geometry.InstanceCount() > 0) {
                    impl.pipeline->BindInstanceGeometryBuffer(impl.geometry.InstanceInfo());
                }
            }
        }

        // The emitter list holds world-space triangles, so a lamp that moved is
        // sampled where it used to be until it is rebuilt. Only a lamp: a rock
        // that moved changes nothing in it.
        const bool emitterMoved =
            std::any_of(impl.timeline.Animated().begin(), impl.timeline.Animated().end(),
                        [&moved](const rendercore::AnimatedNode& animated) {
                            return animated.emissive &&
                                   std::find(moved.begin(), moved.end(), animated.node) !=
                                       moved.end();
                        });
        if (emitterMoved) {
            const auto emissiveTris =
                impl.resolved.enableLightSampling
                    ? rendercore::CollectEmissiveTriangles(impl.loadedScene)
                    : Vector<rendercore::EmissiveTriangleGPU>{};
            impl.emissiveTriangleBuffer =
                rendercore::CreateEmissiveTriangleBuffer(*impl.contextRef, emissiveTris);
            impl.lightingParams.emissiveTriangleCount =
                static_cast<u32>(emissiveTris.size());
            impl.lightingParams.emissiveTotalPower =
                emissiveTris.empty() ? 0.0f : emissiveTris.back().cumulativePower;
            if (impl.lightingParamsBuffer) {
                impl.lightingParamsBuffer->Upload(&impl.lightingParams, sizeof(LightingParams));
            }
            if (impl.pipeline) {
                impl.pipeline->BindEmissiveTriangleBuffer(*impl.emissiveTriangleBuffer);
            }
        }
    }

    impl.UploadThermalFieldAt(impl.ThermalHourNow());
    return Result<void, String>::Ok();
}

TimelineInfo OfflineRenderer::GetTimelineInfo() const {
    TimelineInfo info = m_impl->timeline.Info();
    if (m_impl->thermalSession) {
        info.thermalEpochCount = m_impl->thermalSession->EpochCount();
        info.currentThermalEpoch =
            m_impl->thermalSession->EpochAt(m_impl->ThermalHourNow());
    }
    return info;
}

OfflineRenderOutput OfflineRenderer::Impl::RenderHyperspectral() {
    VulkanContext& context = *contextRef;

    // BatchRenderer reads Scene::camera for every wavelength. Bind the current
    // authored pose for this cube and restore the scene camera on every exit,
    // including cancellation and writer failures.
    struct RestoreCamera {
        Camera& target;
        Camera original;
        ~RestoreCamera() { target = original; }
    } restore{loadedScene.camera, loadedScene.camera};
    if (!resolved.cameraConfig.motion.keys.empty()) {
        const auto pose = camera::CameraPoseAt(resolved.cameraConfig.motion, timeline.Current_s());
        if (!pose) return OfflineRenderOutput{{}, 1.0f, 0.0f, true, pose.error()};
        const auto& position = pose.value().position;
        const auto& lookAt = pose.value().lookAt;
        loadedScene.camera = Camera(
            glm::vec3(static_cast<f32>(position[0]), static_cast<f32>(position[1]), static_cast<f32>(position[2])),
            glm::vec3(static_cast<f32>(lookAt[0]), static_cast<f32>(lookAt[1]), static_cast<f32>(lookAt[2])),
            restore.original.GetUpReference(), restore.original.GetFovY(), restore.original.GetAspectRatio());
        loadedScene.camera.SetProjection(restore.original.GetProjection());
        loadedScene.camera.SetOrthoHeight(restore.original.GetOrthoHeight());
    }

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
    } else if (formatStr == "exr_spectral" || formatStr == "exr" || formatStr == "EXR") {
        hsConfig.outputFormat = HyperspectralOutputFormat::EXR_Spectral;
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

    std::unique_ptr<dataset::ExportSession> exportSession;
    if (config.GetBool("dataset.metadata", true)) {
        hsConfig.exportProvenance = ProductSnapshot(timeline.Current_s(), params.width, params.height, false, true);
        auto created = dataset::ExportSession::Create(params.outputPath, config, {hsConfig.exportProvenance});
        if (!created) return OfflineRenderOutput{{}, 1.0f, 0.0f, true, created.error()};
        exportSession = std::move(created.value());
        const auto staged = exportSession->StagingPath(outPath.stem().string());
        if (!staged) return OfflineRenderOutput{{}, 1.0f, 0.0f, true, staged.error()};
        hsConfig.outputPath = staged.value();
        hsConfig.exportRecordId = exportSession->RecordId();
        hsConfig.exportSidecar = exportSession->SidecarName();
        hsConfig.reserveBandArtifact = [&exportSession](const String& name) {
            return exportSession->StagingPath(name);
        };
    }

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
        if (exportSession) {
            namespace fs = std::filesystem;
            const auto& cube = hsRenderer.GetResult();
            const fs::path stagedRoot = fs::path(hsConfig.outputPath).parent_path();
            for (const auto& file : fs::recursive_directory_iterator(stagedRoot)) {
                if (!file.is_regular_file()) continue;
                const auto name = file.path().lexically_relative(stagedRoot).generic_string();
                if (name.starts_with(".internal/") || name.ends_with(".replay.toml")) continue;
                nlohmann::json description{{"width", cube.width}, {"height", cube.height},
                    {"channels", cube.nbands}, {"provenance", nlohmann::json::parse(hsConfig.exportProvenance)},
                    {"spectral", nlohmann::json::parse(cube.metadata.at("quantiloom_spectral_provenance"))}};
                if (name.find("_bands/") != String::npos) description["channels"] = 1;
                const auto added = exportSession->RegisterFile(name, name, description.dump());
                if (!added) { output.error = added.error(); return output; }
            }
            const auto committed = exportSession->Commit();
            if (!committed) { output.error = committed.error(); return output; }
        }
        QL_LOG_INFO("  Hyperspectral rendering complete!");
        QL_LOG_INFO("  Total render time: {:.2f} seconds", hsRenderer.GetLastRenderTime());
        QL_LOG_INFO("  Average time per band: {:.3f} seconds", hsRenderer.GetAverageTimePerBand());

        // Output is already written by HyperspectralRenderer::Render(), under
        // whichever extensions its format uses.
        const char* extensions = ".hdr/.dat";
        switch (hsConfig.outputFormat) {
            case HyperspectralOutputFormat::GeoTIFF:     extensions = ".tif"; break;
            case HyperspectralOutputFormat::EXR_Spectral: extensions = ".exr"; break;
            default: break;
        }
        QL_LOG_INFO("  Output written to: {}{}", hsConfig.outputPath, extensions);
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
    const bool recordMetadata = config.GetBool("dataset.metadata", true);
    String productSnapshot;
    if (recordMetadata) productSnapshot = ProductSnapshot(timeline.Current_s(), width, height, false, false, &pipeline->GetCameraData());

    // No preview warning here. Whether a render is quantitative is a property
    // of its materials, not of its mode, and ResolveMaterialSpectra already
    // decided it -- per material, after the config's own spectral assignments,
    // and it names the offenders. Repeating a blanket version at render time
    // told a fully measured LWIR scene that it was RGB-averaged, which is both
    // false and the loudest thing in the log.
    if (params.mode == SpectralMode::Single ||
        params.mode == SpectralMode::MWIR_Fused ||
        params.mode == SpectralMode::LWIR_Fused ||
        params.mode == SpectralMode::SWIR_Fused) {
        QL_LOG_INFO("Rendering frame at wavelength {:.1f} nm with {} samples per pixel...",
                    params.wavelengthNm, spp);
    } else {
        QL_LOG_INFO("Rendering frame in RGB mode with {} samples per pixel...", spp);
    }

    try {
        // Which frame of the clock this is. Zero for a still render, and the
        // tick index for a sequence -- so two frames of one sequence get
        // different sampling patterns while a re-run of either gets the same
        // one it got before.
        const TimelineInfo timelineInfo = timeline.Info();
        const u32 frameIndex =
            timelineInfo.present
                ? static_cast<u32>(std::max<i64>(timelineInfo.TickOf(timeline.Current_s()), 0))
                : 0u;
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

        // Drawn once, before the sample loop, and held for all spp: this seeds
        // the Owen scrambles that stratify the first bounce, and they have to
        // agree with each other across the samples they are stratifying. Mixed
        // with frameIndex so an animation does not reuse one pattern for every
        // frame.
        const u32 sequenceSeed = dist(rng) ^ (frameIndex * 0x9e3779b9U);
        if (recordMetadata) {
            auto snapshot = nlohmann::json::parse(productSnapshot);
            snapshot["sampling"]["actual_render_seed"] = renderSeed;
            snapshot["sampling"]["sequence_seed"] = sequenceSeed;
            snapshot["sampling"]["frame_index"] = frameIndex;
            snapshot["sampling"]["random_stream_version"] = 1;
            snapshot["sampling"]["generator"] = "std::mt19937_uniform_u32";
            productSnapshot = snapshot.dump();
        }

        // The first submit keeps the historical two-sample size. Completed GPU
        // timestamps then choose a conservative batch under a 100 ms budget:
        // heavy scenes shrink to one, while cheap scenes grow gradually. This
        // stays well below Windows' TDR interval and keeps each blocking fence
        // wait short enough for a future cancellation hook to poll between them.
        rendercore::OfflineBatchScheduler batchScheduler;
        Vector<f32> batchGpuMs;
        batchGpuMs.reserve(rendercore::OfflineBatchScheduler::kMaxSamples);
        u32 submitCount = 0;
        u32 largestBatch = 0;

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

        for (u32 batchStart = 0; batchStart < spp;) {
            const u32 batchSize = batchScheduler.NextBatchSize(spp - batchStart);
            const u32 batchEnd = batchStart + batchSize;

            // A submitted primary command buffer is EXECUTABLE, not INITIAL.
            // Reset it before recording the next batch into the same handle.
            if (batchStart != 0 && vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) {
                throw std::runtime_error("Failed to reset offline render command buffer");
            }

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            for (u32 sampleIndex = batchStart; sampleIndex < batchEnd; ++sampleIndex) {
                u32 randomSeed = dist(rng) ^ (frameIndex * 997 + sampleIndex * 1009);
                pipeline->SetSamplingParams(frameIndex, sampleIndex, spp, randomSeed, sequenceSeed);

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

            batchGpuMs.clear();
            for (u32 i = batchStart; i < batchEnd; ++i) {
                const f32 gpuMs = perfLogger->ResolveLastGpuMs();
                totalGpuMs += gpuMs;
                batchGpuMs.push_back(gpuMs);
            }
            batchScheduler.ObserveCompletedBatch(batchGpuMs);
            ++submitCount;
            largestBatch = std::max(largestBatch, batchSize);
            batchStart = batchEnd;
        }

        vkDestroyFence(context.GetDevice(), fence, nullptr);
        vkDestroyCommandPool(context.GetDevice(), cmdPool, nullptr);

        QL_LOG_INFO("  All samples completed!");
        QL_LOG_INFO("  GPU submits: {} (largest batch: {} samples)",
                    submitCount, largestBatch);
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
    }

    // quality_level answers "can this file be measured off", so it reads the
    // material resolve rather than the mode. A band on the gate's list with
    // every material carrying a measured spectrum is quantitative and says so;
    // one with materials falling back to a flat reflectance is not, and names
    // how many. Stamping PREVIEW_ONLY on the mode alone marked the repo's own
    // furnace and thermography scenes non-quantitative while they were the
    // ones being measured.
    if (params.mode == SpectralMode::RGB) {
        img.metadata["quality_level"] = "PREVIEW";
        img.metadata["note"] = "RGB rendering (fast, no spectral integration)";
    } else if (params.mode == SpectralMode::VIS_Fused) {
        img.metadata["quality_level"] = "SPECTRAL";
        img.metadata["note"] = "32-wavelength spectral integration";
    } else if (params.mode == SpectralMode::VIS_Hero) {
        img.metadata["quality_level"] = "SPECTRAL";
        img.metadata["note"] = "hero-wavelength spectral sampling";
    } else if (spectra.rgbUpsampledMaterials > 0) {
        img.metadata["quality_level"] = "PREVIEW_ONLY";
        img.metadata["warning"] =
            std::to_string(spectra.rgbUpsampledMaterials) +
            " material(s) have no measured spectrum in this band; they fall back "
            "to a flat reflectance";
        img.metadata["note"] = "For quantitative results provide measured spectral curves";
    } else {
        img.metadata["quality_level"] = "SPECTRAL";
    }
    img.metadata["resolution"] = std::to_string(width) + "x" + std::to_string(height);
    img.metadata["spp"] = std::to_string(spp);
    if (recordMetadata) {
        auto snapshot = nlohmann::json::parse(productSnapshot);
        snapshot["sampling"]["actual_spp"] = spp;
        img.metadata["quantiloom_provenance"] = snapshot.dump();
    }

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
