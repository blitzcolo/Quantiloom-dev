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
 * @author blitzcolo
 */

#include "renderer/OfflineRenderer.hpp"

#include "core/Log.hpp"
#include "core/SpectralData.hpp"
#include "io/SpectralIO.hpp"
#include "io/SpectralBasisLoader.hpp"
#include "renderer/VulkanContext.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/PerformanceLogger.hpp"
#include "renderer/LightingParams.hpp"
#include "renderer/RenderCore.hpp"
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

    ~Impl() {
        // Before the members go, because it was created against the context and
        // the context is one of them. Saved on the way out so the next run
        // starts warm.
        if (pipelineCache != VK_NULL_HANDLE && contextPtr) {
            if (!init.pipelineCachePath.empty()) {
                RayTracingPipeline::SavePipelineCache(*contextPtr, pipelineCache,
                                                      init.pipelineCachePath);
            }
            RayTracingPipeline::DestroyPipelineCache(*contextPtr, pipelineCache);
        }
    }

    SetupResult ParseConfig();
    SetupResult BuildScene(const glm::vec3& albedo);
    SetupResult BuildIlluminants(const Camera& camera, const glm::vec3& sunDirection,
                                 f32 worldUnitsToMeters);
    SetupResult BuildPipeline(const Camera& camera);

    OfflineRenderOutput RenderHyperspectral();
    OfflineRenderOutput RenderSingleFrame();
};

// ============================================================================
// Configuration
// ============================================================================

SetupResult OfflineRenderer::Impl::ParseConfig() {
    QL_LOG_INFO("Parsing configuration...");

    // Renderer settings
    auto resArray = config.GetArray<u32>("renderer.resolution");
    if (resArray.size() != 2) {
        // Reporting the values it did get would read them out of bounds when a
        // config omits the key entirely, which is the most likely way to get
        // here.
        return SetupResult::Err(
            "renderer.resolution must be an array of 2 integers, but has " +
            std::to_string(resArray.size()) + " element(s)");
    }
    params.width = resArray[0];
    params.height = resArray[1];
    params.spp = config.Get<u32>("renderer.spp", 1);
    params.outputPath = config.Get<String>("renderer.output", "spectral_output.exr");

    QL_LOG_INFO("  Resolution: {}x{}", params.width, params.height);
    QL_LOG_INFO("  Samples per pixel: {}", params.spp);
    QL_LOG_INFO("  Output: {}", params.outputPath);

    // Spectral settings
    params.modeName = config.Get<String>("spectral.mode", "rgb");  // Default to RGB

    auto spectralModeResult = ParseSpectralMode(params.modeName);
    if (!spectralModeResult.has_value()) {
        return SetupResult::Err(
            "Invalid spectral mode: " + params.modeName +
            ". Supported modes: single, rgb, vis_fused, mwir_fused, lwir_fused, "
            "swir_fused, multispectral");
    }
    params.mode = *spectralModeResult;

    QL_LOG_INFO("  Spectral mode: {}", params.modeName);

    // Only read wavelength_nm for modes that require it (single, MWIR, LWIR, SWIR)
    if (params.mode == SpectralMode::Single ||
        params.mode == SpectralMode::MWIR_Fused ||
        params.mode == SpectralMode::LWIR_Fused ||
        params.mode == SpectralMode::SWIR_Fused) {
        if (config.Has("spectral.wavelength_nm")) {
            params.wavelengthNm = config.Get<f32>("spectral.wavelength_nm", 550.0f);
        } else if (auto band = GetFusedBandInfo(params.mode)) {
            params.wavelengthNm = band->CenterNm();
        }
        QL_LOG_INFO("  Wavelength: {:.1f} nm", params.wavelengthNm);
    }

    return SetupResult::Ok();
}

// ============================================================================
// Scene and lighting
// ============================================================================

SetupResult OfflineRenderer::Impl::BuildScene(const glm::vec3& albedo) {
    VulkanContext& context = *contextPtr;

    // ====================================================================
    // Load Scene Geometry
    // ====================================================================
    QL_LOG_INFO("Loading scene...");

    auto sceneResult = rendercore::LoadSceneFromConfig(config);
    if (!sceneResult.has_value()) {
        return SetupResult::Err("Failed to load scene: " + sceneResult.error());
    }

    loadedScene = sceneResult.value();

    // If scene has no materials (procedural), create default from config
    if (loadedScene.materials.empty()) {
        Material defaultMaterial = Material::CreateLambertian(albedo, "DefaultMaterial");
        loadedScene.materials.push_back(defaultMaterial);
        QL_LOG_INFO("  Created default material (spectral albedo: {:.3f})",
                    defaultMaterial.spectralAlbedo);
    }

    QL_LOG_INFO("  Scene loaded: {} meshes, {} nodes, {} materials",
                loadedScene.meshes.size(), loadedScene.nodes.size(),
                loadedScene.materials.size());

    // ====================================================================
    // Default IR Surface Temperature (thermal emission source)
    // ====================================================================
    // Standard glTF/USD materials carry no temperature, which silences
    // the Planck emission term entirely in MWIR/LWIR. Backfill a
    // scene-wide ambient temperature for materials without their own
    // temperature source. Configurable via scene.default_temperature_k.
    if (IsIRFusedMode(params.mode) || params.mode == SpectralMode::Single) {
        f32 defaultTemperature_K = config.Get<f32>("scene.default_temperature_k", 300.0f);
        if (defaultTemperature_K < 150.0f || defaultTemperature_K > 1000.0f) {
            QL_LOG_WARN("scene.default_temperature_k={:.1f}K is outside typical range [150, 1000], check config",
                        defaultTemperature_K);
        }
        u32 modified = ApplyDefaultIRTemperature(loadedScene.materials, defaultTemperature_K);
        if (modified > 0) {
            QL_LOG_INFO("  Applied default surface temperature {:.1f} K to {} material(s) without temperature data",
                        defaultTemperature_K, modified);
        }
    }

    // ====================================================================
    // Validate Material Spectral Sources (sRGB upsampling gate - R5)
    // ====================================================================
    bool requireQuantitative = (params.mode == SpectralMode::Multispectral ||
                                params.mode == SpectralMode::MWIR_Fused ||
                                params.mode == SpectralMode::LWIR_Fused ||
                                params.mode == SpectralMode::SWIR_Fused);

    bool failOnSRGB = config.Get<bool>("quality.fail_on_srgb_upsample", false);
    bool logSources = config.Get<bool>("quality.log_material_sources", false);

    if (requireQuantitative && (failOnSRGB || logSources)) {
        QL_LOG_INFO("Validating material spectral sources for quantitative mode...");

        bool hasInvalidMaterials = false;
        for (const auto& mat : loadedScene.materials) {
            const char* sourceStr = "Unknown";
            switch (mat.spectralSource) {
                case Material::SpectralSource::Measured:
                    sourceStr = "Measured (quantitative)";
                    break;
                case Material::SpectralSource::RGBUpsampled:
                    sourceStr = "RGB-upsampled (NOT quantitative)";
                    hasInvalidMaterials = true;
                    break;
                case Material::SpectralSource::Procedural:
                    sourceStr = "Procedural";
                    break;
                default:
                    sourceStr = "Unknown";
                    break;
            }

            if (logSources) {
                QL_LOG_INFO("  Material '{}': source = {}", mat.name, sourceStr);
            }

            if (mat.spectralSource == Material::SpectralSource::RGBUpsampled) {
                QL_LOG_WARN("  ⚠️  Material '{}' uses RGB-upsampled spectra (not quantitative)",
                            mat.name);
            }
        }

        if (hasInvalidMaterials && failOnSRGB) {
            return SetupResult::Err(
                "ABORTED: RGB-upsampled materials detected. They are NOT suitable "
                "for quantitative analysis. To proceed with a non-quantitative "
                "preview, set quality.fail_on_srgb_upsample = false; for "
                "quantitative results, provide measured spectral material data.");
        }

        if (hasInvalidMaterials && !failOnSRGB) {
            QL_LOG_WARN("⚠️  WARNING: Proceeding with RGB-upsampled materials (non-quantitative preview).");
        }
    }

    // Merged geometry buffers, one BLAS per primitive, a TLAS over the node
    // instances, and the per-instance offset table the closest-hit shader indexes
    // with InstanceIndex() -- all of it derived from the scene, all of it shared
    // with the interactive context.
    geometry = rendercore::SceneGeometry::Build(context, loadedScene);
    if (!geometry.IsValid()) {
        return SetupResult::Err("Scene has no geometry to trace");
    }

    // ====================================================================
    // Create Output Image
    // ====================================================================
    outputImage = rendercore::CreateRenderTarget(context, params.width, params.height);

    return SetupResult::Ok();
}

SetupResult OfflineRenderer::Impl::BuildIlluminants(const Camera& camera,
                                                    const glm::vec3& sunDirection,
                                                    f32 worldUnitsToMeters) {
    VulkanContext& context = *contextPtr;

    // ====================================================================
    // Create LUT Buffer (Dual Mode: RGB + Spectral)
    // ====================================================================
    QL_LOG_INFO("Creating LUT buffer...");

    // ====================================================================
    // Upload Textures to GPU
    // ====================================================================
    QL_LOG_INFO("Uploading textures to GPU...");

    textureManager = std::make_unique<TextureManager>(context);
    textureManager->UploadTextures(loadedScene.textures);

    QL_LOG_INFO("  {} textures uploaded", textureManager->GetTextureCount());

    // ====================================================================
    // Load Spectral Curves from CSV (for quantitative spectral rendering)
    // ====================================================================
    // Config format:
    //   [spectral_curves]
    //   "Material_Name" = "path/to/reflectance.csv"
    //   "Another_Material" = "path/to/another.csv"
    // ====================================================================
    QL_LOG_INFO("Loading spectral curves...");

    std::vector<SpectralCurveGPU> spectralCurvesData;
    std::unordered_map<std::string, i32> materialNameToSpectralIndex;

    // Check if spectral_curves section exists in config
    if (config.HasSection("spectral_curves")) {
        auto curveEntries = config.GetSection("spectral_curves");

        for (const auto& [materialName, csvPath] : curveEntries) {
            QL_LOG_INFO("  Loading spectral curve for '{}' from '{}'", materialName, csvPath);

            // Load CSV file
            auto result = SpectralIO::LoadSpectralCurveCSV(csvPath);

            if (!result) {
                QL_LOG_WARN("    Failed to load: {}", result.error());
                continue;
            }

            // Convert to SpectralCurve
            SpectralCurve curve;
            curve.samples = result.value();

            // Convert to GPU format (uniform resampling)
            SpectralCurveGPU gpuCurve = SpectralCurveGPU::FromCPU(curve);

            // Store index mapping
            i32 curveIndex = static_cast<i32>(spectralCurvesData.size());
            materialNameToSpectralIndex[materialName] = curveIndex;
            spectralCurvesData.push_back(gpuCurve);

            QL_LOG_INFO("    Loaded: {} samples, λ=[{:.1f}, {:.1f}] nm → curve index {}",
                        gpuCurve.numSamples,
                        gpuCurve.startWavelength_nm,
                        gpuCurve.GetWavelength(gpuCurve.numSamples - 1),
                        curveIndex);
        }
    }

    QL_LOG_INFO("  Total spectral curves from CSV: {}", spectralCurvesData.size());

    // ====================================================================
    // Load SpectralBaker NMF Basis Data (for quantitative spectral rendering)
    // ====================================================================
    // Config format:
    //   [spectral]
    //   basis_file = "assets/spectral/quantiloom_basis_v1.bin"
    //   materials_json = "assets/spectral/quantiloom_materials.json"
    //   band = "VIS"  # Which band to use for rendering (VIS, NIR, SWIR)
    //
    // Materials reference the database via glTF extras:
    //   "extras": { "quantiloom_material": "Gold_HS111.3B" }
    // ====================================================================

    SpectralBasisLoader basisLoader;
    String activeBand = "VIS";  // Default to visible band

    // Present but empty means "not configured": several shipped configs carry the
    // keys with "" as a placeholder, and passing that through logged a loader
    // error on every render, which teaches people that error lines are noise.
    const auto basisFilePath = config.Get<String>("spectral.basis_file", "");
    const auto materialsJsonPath = config.Get<String>("spectral.materials_json", "");

    if (!basisFilePath.empty() && !materialsJsonPath.empty()) {
        activeBand = config.Get<String>("spectral.band", "VIS");

        QL_LOG_INFO("Loading SpectralBaker NMF basis data...");
        QL_LOG_INFO("  Basis file: {}", basisFilePath);
        QL_LOG_INFO("  Materials JSON: {}", materialsJsonPath);
        QL_LOG_INFO("  Active band: {}", activeBand);

        if (basisLoader.Load(basisFilePath, materialsJsonPath)) {
            QL_LOG_INFO("  SpectralBaker data loaded: {} materials, {} bands",
                        basisLoader.GetMaterialCount(), basisLoader.GetNumBands());

            // Process materials with Quantiloom spectral references
            for (const auto& mat : loadedScene.materials) {
                if (!mat.HasQuantiloomRef()) continue;

                // Accept any quantiloom_* type (usgs, ecostress, rii, etc.)
                if (mat.quantiloomMaterialType.find("quantiloom_") != 0) {
                    QL_LOG_WARN("  Unsupported spectral material type: '{}' (expected 'quantiloom_*')",
                                mat.quantiloomMaterialType);
                    continue;
                }

                QL_LOG_INFO("  Processing Quantiloom material: '{}' -> type='{}', name='{}'",
                            mat.name, mat.quantiloomMaterialType, mat.quantiloomMaterialRef);

                // Try exact match first, then partial match
                const MaterialSpectralData* spectralData = basisLoader.FindMaterial(mat.quantiloomMaterialRef);
                if (!spectralData) {
                    spectralData = basisLoader.FindMaterialPartial(mat.quantiloomMaterialRef);
                    if (spectralData) {
                        QL_LOG_INFO("    Matched via partial search: '{}'", spectralData->name);
                    }
                }

                if (!spectralData) {
                    QL_LOG_WARN("    Material '{}' not found in SpectralBaker database", mat.quantiloomMaterialRef);
                    continue;
                }

                // Reconstruct spectral curve for the active band
                SpectralCurveGPU gpuCurve = basisLoader.ReconstructCurveGPU(spectralData->name, activeBand);
                if (gpuCurve.numSamples == 0) {
                    QL_LOG_WARN("    Failed to reconstruct curve for band '{}'", activeBand);
                    continue;
                }

                // Store index mapping (use glTF material name, not the spectral ref)
                i32 curveIndex = static_cast<i32>(spectralCurvesData.size());
                materialNameToSpectralIndex[mat.name] = curveIndex;
                spectralCurvesData.push_back(gpuCurve);

                // Find band data for quality metrics
                const MaterialSpectralData::BandData* bandData = nullptr;
                auto bandIt = spectralData->bands.find(activeBand);
                if (bandIt != spectralData->bands.end()) {
                    bandData = &bandIt->second;
                }

                QL_LOG_INFO("    Reconstructed: {} samples, λ=[{:.1f}, {:.1f}] nm, RMSE={:.4f} → index {}",
                            gpuCurve.numSamples,
                            gpuCurve.startWavelength_nm,
                            gpuCurve.GetWavelength(gpuCurve.numSamples - 1),
                            bandData ? bandData->rmse : 0.0f,
                            curveIndex);
            }
        } else {
            QL_LOG_WARN("  Failed to load SpectralBaker data, using fallback");
        }
    } else {
        QL_LOG_INFO("  No SpectralBaker basis configured");
        QL_LOG_INFO("  NOTE: Add [spectral] basis_file and materials_json for NMF spectral data");
    }

    QL_LOG_INFO("  Total spectral curves (CSV + SpectralBaker): {}", spectralCurvesData.size());

    // Create GPU buffer for spectral curves (even if empty - need valid binding)
    if (!spectralCurvesData.empty()) {
        spectralCurvesBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            spectralCurvesData.size() * sizeof(SpectralCurveGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        spectralCurvesBuffer->Upload(spectralCurvesData.data(),
                                      spectralCurvesData.size() * sizeof(SpectralCurveGPU));
        QL_LOG_INFO("  Uploaded {} bytes to GPU", spectralCurvesData.size() * sizeof(SpectralCurveGPU));
    } else {
        // Create a dummy buffer with one empty curve for valid binding
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
    // Load Complex Refractive Index Data (for physical Fresnel)
    // ====================================================================
    // Config format:
    //   [refractive_index]
    //   "Gold_Material" = "data/refractiveindex/Au_Johnson.yml"
    //   "Silver_Material" = "data/refractiveindex/Ag_Johnson.yml"
    //
    // The material name (left side) must match the glTF material name
    // The path (right side) points to RefractiveIndex.INFO YAML file
    // ====================================================================
    QL_LOG_INFO("Loading complex refractive index data...");

    std::vector<ComplexRefractiveIndexGPU> criData;
    std::unordered_map<std::string, i32> materialNameToCRIIndex;

    // Check if refractive_index section exists in config
    if (config.HasSection("refractive_index")) {
        auto criEntries = config.GetSection("refractive_index");

        for (const auto& [materialName, yamlPath] : criEntries) {
            QL_LOG_INFO("  Loading n,k data for '{}' from '{}'", materialName, yamlPath);

            // Load YAML file (RefractiveIndex.INFO format)
            auto result = SpectralIO::LoadRefractiveIndexYAML(yamlPath);

            if (!result) {
                QL_LOG_WARN("    Failed to load: {}", result.error());
                continue;
            }

            // Convert to GPU format (uniform resampling)
            ComplexRefractiveIndex cri = result.value();
            ComplexRefractiveIndexGPU gpuCRI = ComplexRefractiveIndexGPU::FromCPU(cri);

            // Store index mapping
            i32 criIndex = static_cast<i32>(criData.size());
            materialNameToCRIIndex[materialName] = criIndex;
            criData.push_back(gpuCRI);

            // Log wavelength range and sample F0 at 550nm for reference
            auto [lambda_min, lambda_max] = cri.GetWavelengthRange();
            f32 F0_550 = cri.FresnelR0(550.0f);

            QL_LOG_INFO("    Loaded: {} samples, λ=[{:.1f}, {:.1f}] nm, F0@550nm={:.3f} → CRI index {}",
                        gpuCRI.numSamples,
                        lambda_min, lambda_max,
                        F0_550,
                        criIndex);
        }
    }

    QL_LOG_INFO("  Total complex refractive index entries loaded: {}", criData.size());

    // Create GPU buffer for complex refractive index
    if (!criData.empty()) {
        criBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            criData.size() * sizeof(ComplexRefractiveIndexGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        criBuffer->Upload(criData.data(), criData.size() * sizeof(ComplexRefractiveIndexGPU));
        QL_LOG_INFO("  Uploaded {} bytes to GPU ({} entries × {} bytes)",
                    criData.size() * sizeof(ComplexRefractiveIndexGPU),
                    criData.size(),
                    sizeof(ComplexRefractiveIndexGPU));
    } else {
        // Create a dummy buffer with one empty entry for valid binding
        ComplexRefractiveIndexGPU dummyCRI{};
        criBuffer = std::make_unique<GpuBuffer>(
            context.GetAllocator(),
            sizeof(ComplexRefractiveIndexGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );
        criBuffer->Upload(&dummyCRI, sizeof(ComplexRefractiveIndexGPU));
        QL_LOG_INFO("  Created dummy CRI buffer (no data loaded)");
        QL_LOG_INFO("  NOTE: Add [refractive_index] section to config for physical metal Fresnel");
    }

    // ====================================================================
    // Create Solar Spectral LUT Buffer (MODTRAN / libRadtran)
    // ====================================================================
    // Contains sun and sky irradiance curves for spectral rendering.
    //
    // The sun's spectrum is user-supplied data, like a material's
    // reflectance curve -- the renderer does not know its shape and does
    // not carry one. Any tabulated file will do; the scene says which
    // columns hold what.
    //
    // Config format:
    //   [lighting]
    //   solar_lut = "assets/luts/modtran/AM1.0_VIS23.txt"
    //   # optional, 1-based, wavelength counts as column 1.
    //   # Defaults below are libRadtran uvspec's layout.
    //   solar_lut_columns = [2, 3]              # direct, diffuse
    //   solar_lut_diffuse_is_global = false
    //
    // File format: whitespace- or comma-separated, one row per wavelength,
    // '#' comments and a non-numeric header row both skipped.
    //
    //   # libRadtran uvspec
    //   wavelength(nm)  edir(W/m2/nm)  edn(W/m2/nm)  trans
    //   380.0  1.234e+00  5.678e-02  0.9876
    //
    // For ASTM G-173 (assets/luts/astmg173.csv), whose columns are
    // wavelength, extraterrestrial, global-tilt, direct+circumsolar:
    //
    //   solar_lut = "assets/luts/astmg173.csv"
    //   solar_lut_columns = [4, 3]
    //   solar_lut_diffuse_is_global = true      # col 3 includes the beam
    // ====================================================================
    QL_LOG_INFO("Loading solar spectral LUT...");

    SolarSpectralLUT solarLUT{};  // Zero-initialized (numSamples=0 = fallback mode)

    // A spectral mode with a sun needs a spectrum for it. The shaders no
    // longer invent one from lighting.sun_radiance -- that field is an RGB
    // triple in arbitrary units, and spreading it across SWIR or MWIR made
    // the same scene render differently depending on whether its spectrum
    // happened to be present. Refusing here is the point of "the sun's
    // spectrum is user-supplied data": a missing illuminant is a scene
    // that is not finished, not a scene to guess at.
    // Both illuminants, not just the sun: solar_lut carries the diffuse
    // sky as well, so a scene lit only by sky_radiance loses its light
    // just as completely. Checking one and not the other is how
    // cornell_box_vis quietly went 55% darker instead of saying why.
    const auto nonZero = [](const glm::vec3& c) {
        return c.r > 0.0f || c.g > 0.0f || c.b > 0.0f;
    };
    const bool spectralIlluminantNeeded =
        params.mode != SpectralMode::RGB &&
        (nonZero(lightingParams.sunRadiance_rgb) ||
         nonZero(lightingParams.skyRadiance_rgb));
    if (spectralIlluminantNeeded && !config.Has("lighting.solar_lut")) {
        return SetupResult::Err(
            "[lighting] sun_radiance or sky_radiance is non-zero in a spectral "
            "mode, but no solar_lut is given. An illuminant's spectrum is scene "
            "data, like a reflectance curve -- name one, or zero both for a scene "
            "lit only by emissive materials.");
    }

    if (config.Has("lighting.solar_lut")) {
        auto solarLutPath = config.Get<String>("lighting.solar_lut");
        QL_LOG_INFO("  Loading solar LUT from: {}", solarLutPath);

        // The sun's spectrum is user-supplied data, like a reflectance
        // curve, so the file's layout is the user's to declare. Defaults
        // are libRadtran uvspec's; ASTM G-173 wants [4, 3] with
        // diffuse_is_global, its column 3 being global rather than sky.
        const auto cols = config.GetArray<i32>("lighting.solar_lut_columns");
        const u32 directCol  = cols.size() >= 1 ? static_cast<u32>(cols[0]) : 2u;
        const u32 diffuseCol = cols.size() >= 2 ? static_cast<u32>(cols[1]) : 3u;
        // Reference illuminants are published as relative spectra -- D65 is
        // normalised to 100 at 560 nm -- so their absolute level is
        // arbitrary and colour-matches to five figures of nothing in
        // particular. Normalising to unit luminance puts the illuminant at
        // Y = 1, which is what makes D65 come out as sRGB (1, 1, 1)
        // exactly. It is also the honest separation of white balance from
        // exposure, and it applies just as well to a measured spectrum
        // whose absolute level you would rather set yourself.
        const auto normalise = config.Get<String>("lighting.solar_lut_normalise", "");
        const bool diffuseIsGlobal =
            config.Get<bool>("lighting.solar_lut_diffuse_is_global", false);

        // "equal_energy" is the one illuminant that is not a file: a flat
        // spectrum at unit luminance, CIE illuminant E. It is the neutral
        // reference -- what a material looks like under light that favours
        // no wavelength -- and it is not sRGB white, which is D65. Any
        // other spectrum, D65 included, is a table the scene points at.
        Result<std::pair<SpectralCurve, SpectralCurve>, String> result =
            solarLutPath == "equal_energy"
                ? Result<std::pair<SpectralCurve, SpectralCurve>, String>(
                      std::make_pair(MakeEqualEnergyIlluminant(),
                                     MakeEqualEnergyIlluminant()))
                : SpectralIO::LoadLibRadtranSunAndSky(
                      solarLutPath, "nm", directCol, diffuseCol, diffuseIsGlobal);

        if (result.has_value()) {
            auto& [sunCurve, skyCurve] = result.value();

            // One illuminant, every mode. The spectral paths sample these
            // curves per wavelength; RGB and VIS_FUSED need a colour, and
            // taking it from the same curve is what stops the two halves
            // of the renderer describing different suns. It replaces
            // whatever sun_radiance said, which was a triple in arbitrary
            // units with no defined relationship to the spectrum beside it.
            if (normalise == "unit_luminance") {
                const auto rgb = SpectralIrradianceToLinearSrgb(sunCurve);
                const f32 Y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
                if (Y > 0.0f) {
                    // Both curves by the sun's luminance, not each by its
                    // own: scaling them separately would discard the ratio
                    // between sun and sky, which is the one thing a
                    // measured pair actually tells you.
                    for (auto& v : sunCurve.samples) v.second /= Y;
                    for (auto& v : skyCurve.samples) v.second /= Y;
                    solarLUT = SolarSpectralLUT::FromCPU(sunCurve, skyCurve);
                    QL_LOG_INFO("  Illuminant normalised to unit luminance (was Y={:.4g})", Y);
                }
            } else if (!normalise.empty()) {
                QL_LOG_WARN("  Unknown solar_lut_normalise '{}' (known: unit_luminance)",
                            normalise);
            }

            // Does the spectrum actually cover the band being rendered?
            // SpectralCurve::Evaluate clamps to its endpoints rather than
            // returning zero, so a curve that stops short does not fail --
            // it holds its last value flat across everything above it, and
            // the render looks plausible. CIE D65 stops at 830 nm, which
            // makes it a fine reference illuminant for RGB and a silently
            // wrong one for SWIR upward.
            if (const auto band = GetFusedBandInfo(params.mode);
                band && !sunCurve.samples.empty()) {
                const f32 curveMin = sunCurve.samples.front().first;
                const f32 curveMax = sunCurve.samples.back().first;
                if (curveMin > band->lambdaMinNm || curveMax < band->lambdaMaxNm) {
                    QL_LOG_WARN("  Illuminant spans [{:.0f}, {:.0f}] nm but this mode "
                                "renders [{:.0f}, {:.0f}] nm. Evaluate() clamps, so the "
                                "uncovered part is held flat at the nearest endpoint "
                                "rather than left dark -- the result will look "
                                "reasonable and mean nothing.",
                                curveMin, curveMax,
                                band->lambdaMinNm, band->lambdaMaxNm);
                }
            }

            lightingParams.sunRadiance_rgb = SpectralIrradianceToLinearSrgb(sunCurve);
            lightingParams.skyRadiance_rgb = SpectralIrradianceToLinearSrgb(skyCurve);
            lightingParams.sunRadiance_spectral =
                (lightingParams.sunRadiance_rgb.r + lightingParams.sunRadiance_rgb.g +
                 lightingParams.sunRadiance_rgb.b) / 3.0f;
            lightingParams.skyRadiance_spectral =
                (lightingParams.skyRadiance_rgb.r + lightingParams.skyRadiance_rgb.g +
                 lightingParams.skyRadiance_rgb.b) / 3.0f;
            lightingParamsBuffer->Upload(&lightingParams, sizeof(LightingParams));
            QL_LOG_INFO("  Illuminant colour from the spectrum: sun [{:.4g}, {:.4g}, "
                        "{:.4g}], sky [{:.4g}, {:.4g}, {:.4g}]",
                        lightingParams.sunRadiance_rgb.r, lightingParams.sunRadiance_rgb.g,
                        lightingParams.sunRadiance_rgb.b, lightingParams.skyRadiance_rgb.r,
                        lightingParams.skyRadiance_rgb.g, lightingParams.skyRadiance_rgb.b);

            // Convert to GPU format (uniform resampling to 64 samples)
            solarLUT = SolarSpectralLUT::FromCPU(sunCurve, skyCurve);

            if (solarLUT.IsValid()) {
                auto [minWl, maxWl] = solarLUT.GetWavelengthRange();
                QL_LOG_INFO("  ✓ Solar LUT loaded: {} samples, λ=[{:.1f}, {:.1f}] nm",
                            solarLUT.sunIrradiance.numSamples, minWl, maxWl);
            } else {
                return SetupResult::Err(
                    "Solar LUT conversion failed. There is no RGB fallback any more "
                    "-- the scene would render unlit, so this is fatal.");
            }
        } else {
            QL_LOG_WARN("  ⚠ Failed to load solar LUT: {}", result.error());
            QL_LOG_WARN("  ⚠ Using LightingParams RGB fallback");
        }
    } else {
        QL_LOG_INFO("  No solar_lut specified in config, using LightingParams RGB fallback");
        QL_LOG_INFO("  NOTE: Add [lighting] solar_lut = \"path/to/file.txt\" for spectral illumination");
    }

    solarSpectralLUTBuffer = std::make_unique<GpuBuffer>(
        context.GetAllocator(),
        sizeof(SolarSpectralLUT),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    solarSpectralLUTBuffer->Upload(&solarLUT, sizeof(SolarSpectralLUT));

    // ====================================================================
    // NN Atmosphere ([atmosphere] TOML -> baked spectral LUT, bindings 17+20)
    // ====================================================================
    QL_LOG_INFO("Configuring NN atmosphere...");

    // Deprecation warnings for the removed analytic atmosphere keys
    if (config.Has("atmospheric.preset") || config.Has("atmospheric.rayleigh_enabled") ||
        config.Has("atmospheric.mie_enabled") || config.Has("atmospheric.rayleigh_beta_550nm") ||
        config.Has("atmospheric.mie_beta_550nm")) {
        QL_LOG_WARN("  [atmospheric] is DEPRECATED and ignored; the analytic "
                    "Rayleigh/Mie atmosphere was replaced by the NN atmosphere. "
                    "Use [atmosphere] with model_pack instead.");
    }
    if (config.Has("lighting.transmittance")) {
        QL_LOG_WARN("  lighting.transmittance is DEPRECATED and no longer used; "
                    "view-path transmittance comes from the NN atmosphere");
    }
    if (config.Has("lighting.atmosphere_temperature_k")) {
        QL_LOG_WARN("  lighting.atmosphere_temperature_k is DEPRECATED; used only "
                    "as thermal-sky fallback when the NN atmosphere is disabled");
    }

    AtmosphereNNConfig atmosphereConfig;
    // A scene opts in by naming either key. model_pack wins when both are
    // present; naming only a preset falls back to the shipped weights, so
    // that a config renders the same atmosphere here as it does in Studio.
    if (config.Has("atmosphere.model_pack") || config.Has("atmosphere.preset")) {
        atmosphereConfig.modelPackDir =
            config.Has("atmosphere.model_pack")
                ? config.Get<String>("atmosphere.model_pack")
                : init.atmosphereModelPackFallback;
        if (atmosphereConfig.modelPackDir.empty()) {
            return SetupResult::Err(
                "[atmosphere] names a preset but no model pack was found. Set "
                "atmosphere.model_pack, or point QUANTILOOM_ATMOS_MODELS at the "
                "weights.");
        }
        atmosphereConfig.enabled = true;

        auto presetName = config.Get<String>("atmosphere.preset", "clear");
        if (!atmosphereConfig.ApplyPreset(presetName)) {
            QL_LOG_WARN("  Unknown atmosphere preset '{}', using 'clear'", presetName);
            atmosphereConfig.ApplyPreset("clear");
        }

        // Optional per-feature overrides (out-of-domain values are clamped
        // by the network input spec with a warning)
        auto overrideD = [&](const char* key, double& field) {
            if (config.Has(key)) field = static_cast<double>(config.Get<f32>(key));
        };
        overrideD("atmosphere.atmos_model", atmosphereConfig.atmosModel);
        overrideD("atmosphere.ihaze", atmosphereConfig.ihaze);
        overrideD("atmosphere.icld", atmosphereConfig.icld);
        overrideD("atmosphere.vis_km", atmosphereConfig.visKm);
        overrideD("atmosphere.rainrt_mm_h", atmosphereConfig.rainrtMmH);
        overrideD("atmosphere.t_ground_K", atmosphereConfig.tGroundK);
        overrideD("atmosphere.rh", atmosphereConfig.rh);
        overrideD("atmosphere.p_hPa", atmosphereConfig.pHPa);
        overrideD("atmosphere.h2o_scale", atmosphereConfig.h2oScale);
        if (config.Has("atmosphere.lut_a_samples"))
            atmosphereConfig.lutASamples = config.Get<i32>("atmosphere.lut_a_samples");
        if (config.Has("atmosphere.lut_az_samples"))
            atmosphereConfig.lutAzSamples = config.Get<i32>("atmosphere.lut_az_samples");

        // Sun geometry: default derives from lighting.sun_direction (Y-up);
        // an explicit sun_zenith_deg wins with a mismatch warning
        const double lightingZenith =
            glm::degrees(std::acos(std::clamp(sunDirection.y, -1.0f, 1.0f)));
        const double lightingAzimuth =
            glm::degrees(std::atan2(sunDirection.x, sunDirection.z));
        if (config.Has("atmosphere.sun_zenith_deg")) {
            atmosphereConfig.sunFromLighting = false;
            atmosphereConfig.sunZenithDeg =
                static_cast<double>(config.Get<f32>("atmosphere.sun_zenith_deg"));
            atmosphereConfig.sunAzimuthDeg = static_cast<double>(
                config.Get<f32>("atmosphere.sun_azimuth_deg",
                                static_cast<f32>(lightingAzimuth)));
            if (std::abs(atmosphereConfig.sunZenithDeg - lightingZenith) > 2.0) {
                QL_LOG_WARN("  atmosphere.sun_zenith_deg = {:.1f} differs from "
                            "lighting.sun_direction zenith {:.1f} by > 2 deg; "
                            "using the [atmosphere] value for the NN inputs",
                            atmosphereConfig.sunZenithDeg, lightingZenith);
            }
        } else {
            atmosphereConfig.sunFromLighting = false;  // Resolve here, once
            atmosphereConfig.sunZenithDeg = lightingZenith;
            atmosphereConfig.sunAzimuthDeg = lightingAzimuth;
        }

        // Observer altitude: default derives from camera height
        if (config.Has("atmosphere.h1_km")) {
            atmosphereConfig.h1FromCamera = false;
            atmosphereConfig.h1Km =
                static_cast<double>(config.Get<f32>("atmosphere.h1_km"));
        } else {
            atmosphereConfig.h1FromCamera = false;  // Resolve here, once
            atmosphereConfig.h1Km = std::max(
                static_cast<double>(camera.GetPosition().y * worldUnitsToMeters) / 1000.0,
                0.0);
        }

        QL_LOG_INFO("  NN atmosphere: preset '{}', model pack '{}'{}",
                    atmosphereConfig.preset, atmosphereConfig.modelPackDir,
                    config.Has("atmosphere.model_pack") ? "" : " (resolved)");
        QL_LOG_INFO("  Sun zenith {:.1f} deg, h1 {:.3f} km",
                    atmosphereConfig.sunZenithDeg, atmosphereConfig.h1Km);
    } else {
        QL_LOG_INFO("  No [atmosphere] section - atmosphere disabled");
    }
    if (atmosphereConfig.preset == "disabled") atmosphereConfig.enabled = false;

    // Bake the spectral LUT for the active render band. Missing network
    // files or out-of-coverage wavelengths are hard errors -- no fallback.
    AtmosNNHeaderGPU atmosHeader{};  // enabled = 0
    // Sized so that even unconditionally-evaluated LUT reads (HLSL ternary
    // is a select) stay in bounds when the atmosphere is disabled
    std::vector<f32> atmosData(2048, 0.0f);
    if (atmosphereConfig.enabled) {
        AtmosLambdaGrid grid = RenderBandLambdaGrid(
            params.mode, static_cast<double>(params.wavelengthNm));
        if (!grid.error.empty()) {
            return SetupResult::Err("NN atmosphere: " + grid.error);
        }
        if (grid.band.empty()) {
            return SetupResult::Err("NN atmosphere: spectral mode has no NN coverage");
        }
        try {
            AtmosModelPack pack(atmosphereConfig.modelPackDir);
            AtmosphereBaker baker(pack);
            AtmosBakeResult baked = baker.Bake(
                atmosphereConfig, grid.band, grid.lambdasNm, grid.windowHalfWidthNm);
            baked.header.sunDirWorld[0] = sunDirection.x;
            baked.header.sunDirWorld[1] = sunDirection.y;
            baked.header.sunDirWorld[2] = sunDirection.z;
            baked.header.worldUnitsToMeters = worldUnitsToMeters;
            atmosHeader = baked.header;
            atmosData = std::move(baked.data);
        } catch (const std::exception& e) {
            return SetupResult::Err(String("NN atmosphere setup failed: ") + e.what());
        }
    }

    // When NN atmosphere is active, use its ground temperature for the
    // thermal-sky fallback (depth-limited reflections, non-NN code paths).
    if (atmosphereConfig.enabled) {
        lightingParams.atmosphereTemperature_K = static_cast<f32>(atmosphereConfig.tGroundK);
        lightingParamsBuffer->Upload(&lightingParams, sizeof(LightingParams));
        QL_LOG_INFO("  Updated atmosphere temperature from NN config: {:.1f} K",
                    lightingParams.atmosphereTemperature_K);
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

    QL_LOG_INFO("  NN atmosphere: {}", atmosphereConfig.enabled ? "ENABLED" : "DISABLED");

    // ====================================================================
    // Load CIE 1931 Color Matching Functions LUT (for VIS_FUSED mode)
    // ====================================================================
    // Compiled in rather than parsed from assets/luts/, so there is no path to
    // get wrong and no "VIS_FUSED will produce INCORRECT colors" fallback to hit.
    cieCMF_LUTBuffer = rendercore::CreateCieColourMatchingBuffer(context);

    // ====================================================================
    // Create Material Buffer (PBR)
    // ====================================================================
    QL_LOG_INFO("Creating PBR material buffer...");

    // The curve and refractive-index slots are resolved here rather than inside
    // the conversion: the CLI matches material names against what the config
    // loaded, while the interactive context reads the indices off the Material.
    Vector<rendercore::MaterialGpuIndices> materialIndices;
    materialIndices.reserve(loadedScene.materials.size());
    for (const auto& mat : loadedScene.materials) {
        rendercore::MaterialGpuIndices slots;

        if (auto it = materialNameToSpectralIndex.find(mat.name);
            it != materialNameToSpectralIndex.end()) {
            slots.spectralReflectanceCurve = it->second;
            QL_LOG_INFO("  Material '{}': using spectral curve index {}", mat.name,
                        it->second);
        }
        if (auto it = materialNameToCRIIndex.find(mat.name);
            it != materialNameToCRIIndex.end()) {
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

SetupResult OfflineRenderer::Impl::BuildPipeline(const Camera& camera) {
    VulkanContext& context = *contextPtr;

    // ====================================================================
    // Load or Generate BRDF Integration LUT for IBL (with Disk Caching)
    // ====================================================================
    // The BRDF LUT is scene-independent and can be cached to disk.
    // First run: Generate (5-10 seconds) and save to cache
    // Subsequent runs: Load from cache (instant)
    // ====================================================================

    brdfLut = rendercore::BrdfLut::Create(context);
    if (!brdfLut.IsValid()) {
        return SetupResult::Err("Failed to create BRDF LUT sampler");
    }

    // ====================================================================
    // Create Prefiltered Environment Map for IBL Specular
    // ====================================================================
    QL_LOG_INFO("Creating prefiltered environment map for IBL...");

    // Falls back to sky blue when the config names no map, or names one that
    // will not load. Load() reads through ImageIO::ReadImage, so .hdr and the
    // LDR formats work as well as .exr -- this used to call ReadEXR directly
    // and take the fallback for anything else.
    const auto envMapPath = config.Get<String>("renderer.environment_map", "");
    if (envMapPath.empty()) {
        QL_LOG_INFO("  No environment map specified in config, using fallback");
        envMap = rendercore::EnvironmentCubemap::Fallback(context);
    } else {
        auto loaded = rendercore::EnvironmentCubemap::Load(context, envMapPath);
        if (loaded.has_value()) {
            envMap = std::move(loaded.value());
        } else {
            QL_LOG_WARN("  {}, using fallback", loaded.error());
            envMap = rendercore::EnvironmentCubemap::Fallback(context);
        }
    }

    QL_LOG_INFO("  Prefiltered environment map ready ({}x{} per face, {} mip levels)",
                envMap.FaceSize(), envMap.FaceSize(), envMap.MipLevels());

    // ====================================================================
    // Create Ray Tracing Pipeline
    // ====================================================================
    if (!init.pipelineCachePath.empty()) {
        pipelineCache =
            RayTracingPipeline::LoadPipelineCache(context, init.pipelineCachePath);
    }

    rendercore::PipelineBindings bindings;
    bindings.outputImage = outputImage.get();
    bindings.geometry = &geometry;
    bindings.lightingParams = lightingParamsBuffer.get();
    bindings.materials = materialBuffer.get();
    bindings.textures = textureManager.get();
    bindings.environment = &envMap;
    bindings.brdfLut = &brdfLut;
    bindings.spectralCurves = spectralCurvesBuffer.get();
    bindings.complexRefractiveIndex = criBuffer.get();
    bindings.solarLut = solarSpectralLUTBuffer.get();
    bindings.atmosphereHeader = atmosHeaderBuffer.get();
    bindings.atmosphereData = atmosDataBuffer.get();
    bindings.cieColourMatching = cieCMF_LUTBuffer.get();

    pipeline = rendercore::CreateRayTracingPipeline(context, pipelineCache, bindings);

    CameraData cameraData = camera.GetCameraData();
    cameraData.wavelength_nm = params.wavelengthNm;         // Override with config wavelength
    cameraData.spectral_mode = static_cast<u32>(params.mode);  // Set rendering mode
    cameraData.debug_mode = static_cast<u32>(config.Get<i32>("renderer.debug_mode", 0));
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

    if (auto r = impl.ParseConfig(); !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    // ====================================================================
    // Camera and lighting, read before the device so that a config error
    // costs nothing.
    // ====================================================================
    f32 aspectRatio = static_cast<f32>(impl.params.width) /
                      static_cast<f32>(impl.params.height);
    auto cameraResult = Camera::FromConfig(config, aspectRatio);
    if (!cameraResult.has_value()) {
        return CreateResult::Err("Failed to load camera: " + cameraResult.error());
    }
    Camera camera = cameraResult.value();

    auto sunDirArray = config.GetArray<f32>("lighting.sun_direction");
    if (sunDirArray.size() != 3) {
        return CreateResult::Err(
            "lighting.sun_direction must be an array of 3 floats, but has " +
            std::to_string(sunDirArray.size()) + " element(s)");
    }
    glm::vec3 sunDirection =
        glm::normalize(glm::vec3(sunDirArray[0], sunDirArray[1], sunDirArray[2]));

    auto sunRadArray = config.GetArray<f32>("lighting.sun_radiance");
    if (sunRadArray.size() != 3) {
        return CreateResult::Err(
            "lighting.sun_radiance must be an array of 3 floats, but has " +
            std::to_string(sunRadArray.size()) + " element(s)");
    }
    glm::vec3 sunRadiance(sunRadArray[0], sunRadArray[1], sunRadArray[2]);

    auto skyRadArray = config.GetArray<f32>("lighting.sky_radiance");
    if (skyRadArray.size() != 3) {
        return CreateResult::Err(
            "lighting.sky_radiance must be an array of 3 floats, but has " +
            std::to_string(skyRadArray.size()) + " element(s)");
    }
    glm::vec3 skyRadiance(skyRadArray[0], skyRadArray[1], skyRadArray[2]);

    // Atmospheric transmittance (Beer-Lambert law)
    // Default: 0.9 (relatively clear atmosphere, ~10% attenuation)
    f32 transmittance = config.Get<f32>("lighting.transmittance", 0.9f);
    transmittance = std::clamp(transmittance, 0.0f, 1.0f);

    // Effective atmosphere temperature for IR downwelling radiation
    // Used in MWIR/LWIR modes to compute atmospheric thermal emission
    // Default: 260K (clear sky), typical range: 240K (cold/dry) to 290K (hot/humid)
    f32 atmosphereTemperature_K = config.Get<f32>("lighting.atmosphere_temperature_k", 260.0f);
    if (atmosphereTemperature_K < 150.0f || atmosphereTemperature_K > 350.0f) {
        QL_LOG_WARN("lighting.atmosphere_temperature_k={:.1f}K is outside typical range [150, 350], check config",
                    atmosphereTemperature_K);
    }

    // World unit configuration
    // Conversion factor from scene units to meters for physically-correct Beer-Lambert
    // Default: 1.0 (scene units are meters)
    f32 worldUnitsToMeters = config.Get<f32>("scene.world_units_to_meters", 1.0f);
    if (worldUnitsToMeters <= 0.0f) {
        QL_LOG_WARN("scene.world_units_to_meters must be positive, using default 1.0");
        worldUnitsToMeters = 1.0f;
    }

    QL_LOG_INFO("  Sun direction: [{:.2f}, {:.2f}, {:.2f}]",
                sunDirection.x, sunDirection.y, sunDirection.z);
    QL_LOG_INFO("  Sun radiance: [{:.2f}, {:.2f}, {:.2f}]",
                sunRadiance.r, sunRadiance.g, sunRadiance.b);
    QL_LOG_INFO("  Sky radiance: [{:.2f}, {:.2f}, {:.2f}]",
                skyRadiance.r, skyRadiance.g, skyRadiance.b);
    QL_LOG_INFO("  Atmospheric transmittance: {:.3f}", transmittance);
    QL_LOG_INFO("  Atmosphere temperature (IR): {:.1f} K", atmosphereTemperature_K);
    QL_LOG_INFO("  World units to meters: {:.6f}", worldUnitsToMeters);

    // Material settings. Read even for a scene that brings its own materials,
    // in which case nothing uses it -- but a config that misspells the key
    // should say so rather than render, and this is where it always said it.
    auto albedoArray = config.GetArray<f32>("material.albedo");
    if (albedoArray.size() != 3) {
        return CreateResult::Err(
            "material.albedo must be an array of 3 floats, but has " +
            std::to_string(albedoArray.size()) + " element(s)");
    }
    glm::vec3 albedo(albedoArray[0], albedoArray[1], albedoArray[2]);
    QL_LOG_INFO("  Material albedo: [{:.2f}, {:.2f}, {:.2f}]", albedo.x, albedo.y, albedo.z);

    // ====================================================================
    // Initialize Vulkan Context
    // ====================================================================
    QL_LOG_INFO("Initializing Vulkan context...");
    impl.contextPtr = std::make_unique<VulkanContext>();

    if (!impl.contextPtr->IsRayTracingSupported()) {
        return CreateResult::Err("Ray tracing not supported on this device");
    }

    if (auto r = impl.BuildScene(albedo); !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    // For spectral mode: Convert RGB to scalar (average of RGB channels)
    f32 sunRadiance_spectral = (sunRadiance.r + sunRadiance.g + sunRadiance.b) / 3.0f;
    f32 skyRadiance_spectral = (skyRadiance.r + skyRadiance.g + skyRadiance.b) / 3.0f;

    if (impl.params.mode == SpectralMode::RGB ||
        impl.params.mode == SpectralMode::VIS_Fused) {
        QL_LOG_INFO("  Sun RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W*sr^-1*m^-2",
                    sunRadiance.r, sunRadiance.g, sunRadiance.b);
        QL_LOG_INFO("  Sky RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W*sr^-1*m^-2",
                    skyRadiance.r, skyRadiance.g, skyRadiance.b);
    } else {
        // NOTE: These are RGB-average fallback values, NOT true spectral density
        // Unit: W·sr⁻¹·m⁻² (same as RGB), NOT W·sr⁻¹·m⁻²·nm⁻¹
        QL_LOG_INFO("  Sun fallback radiance (RGB avg): {:.3f} W*sr^-1*m^-2", sunRadiance_spectral);
        QL_LOG_INFO("  Sky fallback radiance (RGB avg): {:.3f} W*sr^-1*m^-2", skyRadiance_spectral);
    }

    // Read chromaticity correction factors from config (optional, defaults to standard values)
    f32 chromaR_correction = config.Get<f32>("quality.chroma_r_correction", LightingDefaults::CHROMA_R_CORRECTION);
    f32 chromaB_correction = config.Get<f32>("quality.chroma_b_correction", LightingDefaults::CHROMA_B_CORRECTION);

    // Read shadow ray enable flag from config (optional, defaults to ENABLED)
    // Known GPU crash issue on some drivers when shadow rays are enabled
    // Users can disable via config: renderer.enable_shadow_rays = false
    bool enableShadowRays = config.Get<bool>("renderer.enable_shadow_rays", true);
    if (!enableShadowRays) {
        QL_LOG_INFO("Shadow rays DISABLED via config");
    }

    impl.lightingParams.sunDirection = sunDirection;

    // Fill both spectral and RGB fields for fallback (used when SolarSpectralLUT unavailable)
    impl.lightingParams.sunRadiance_spectral = sunRadiance_spectral;
    impl.lightingParams.skyRadiance_spectral = skyRadiance_spectral;
    impl.lightingParams.sunRadiance_rgb = sunRadiance;
    impl.lightingParams.skyRadiance_rgb = skyRadiance;
    impl.lightingParams.transmittance = transmittance;
    impl.lightingParams.worldUnitsToMeters = worldUnitsToMeters;
    impl.lightingParams.atmosphereTemperature_K = atmosphereTemperature_K;
    impl.lightingParams.chromaR_correction = chromaR_correction;
    impl.lightingParams.chromaB_correction = chromaB_correction;
    impl.lightingParams.enableShadowRays = enableShadowRays ? 1u : 0u;

    impl.lightingParamsBuffer = std::make_unique<GpuBuffer>(
        impl.contextPtr->GetAllocator(),
        sizeof(LightingParams),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    impl.lightingParamsBuffer->Upload(&impl.lightingParams, sizeof(LightingParams));

    if (auto r = impl.BuildIlluminants(camera, sunDirection, worldUnitsToMeters);
        !r.has_value()) {
        return CreateResult::Err(std::move(r).error());
    }

    if (auto r = impl.BuildPipeline(camera); !r.has_value()) {
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
    VulkanContext& context = *contextPtr;

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

    // Progress callback for status updates
    auto progressCallback = [](const HyperspectralProgress& p, void*) {
        if (p.currentBand % 10 == 0 || p.currentBand == p.totalBands) {
            QL_LOG_INFO("  Band {}/{} ({:.1f} nm) - {:.1f}% - ETA: {:.1f}s",
                        p.currentBand, p.totalBands, p.currentWavelength_nm,
                        p.GetPercentage(), p.GetRemainingSeconds());
        }
    };

    OfflineRenderOutput output;
    // The cube is streamed to disk band by band rather than assembled in
    // memory: at 512x512 and 211 bands it is 200 MB, and nothing downstream
    // wants it as one array.
    output.wroteItsOwnOutput = true;

    // Execute hyperspectral rendering
    auto status = hsRenderer.Render(hsConfig, progressCallback, nullptr);

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
    VulkanContext& context = *contextPtr;

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
