/**
 * @file HyperspectralRenderer.cpp
 * @brief Implementation of hyperspectral rendering engine
 *
 * Phase 1 Implementation:
 * - Basic batch rendering over wavelength range
 * - Progress reporting
 * - SpectralCube assembly
 * - ENVI/GeoTIFF output
 *
 * Phase 2 Implementation:
 * - Adaptive spectral sampling based on material features
 * - SpectralAnalyzer integration for feature detection
 * - AdaptiveGridGenerator for non-uniform wavelength grids
 * - SpectralReconstructor for interpolation of missing bands
 *
 * Phase 3 Implementation:
 * - Refactored to use BatchRenderer for core rendering loop
 * - Clean separation: analysis -> batch render -> reconstruction
 *
 * Phase 4 Implementation:
 * - GPU-accelerated spectral reconstruction via compute shaders
 * - GpuSpectralReconstructor for parallel Catmull-Rom interpolation
 * - 10-40x speedup over CPU reconstruction
 *
 * @author wtflmao
 */

#include "hs_core/HyperspectralRenderer.hpp"
#include "hs_core/HyperspectralConfig.hpp"
#include "hs_core/BatchRenderer.hpp"
#include "hs_core/SpectralAnalyzer.hpp"
#include "hs_core/AdaptiveGridGenerator.hpp"
#include "hs_core/SpectralReconstructor.hpp"
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/VulkanContext.hpp"
#include "scene/Scene.hpp"
#include "core/Log.hpp"
#include "core/Image.hpp"
#include "io/SpectralCubeIO.hpp"
#include "io/ImageIO.hpp"

#include <chrono>
#include <filesystem>
#include <atomic>
#include <cstring>

namespace quantiloom {

// ============================================================================
// Status String Conversion
// ============================================================================

const char* HyperspectralStatusToString(HyperspectralStatus status) {
    switch (status) {
        case HyperspectralStatus::Success:          return "Success";
        case HyperspectralStatus::InvalidConfig:    return "Invalid configuration";
        case HyperspectralStatus::PipelineNotReady: return "Pipeline not ready";
        case HyperspectralStatus::RenderFailed:     return "Render failed";
        case HyperspectralStatus::OutputWriteFailed: return "Output write failed";
        case HyperspectralStatus::Cancelled:        return "Cancelled by user";
        case HyperspectralStatus::OutOfMemory:      return "Out of memory";
        default:                                    return "Unknown status";
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

Vector<f32> GenerateWavelengthList(const HyperspectralConfig& config) {
    Vector<f32> wavelengths;

    if (!config.IsValid()) {
        return wavelengths;
    }

    u32 numBands = config.GetNumBands();
    wavelengths.reserve(numBands);

    for (u32 i = 0; i < numBands; ++i) {
        wavelengths.push_back(config.GetWavelength(i));
    }

    return wavelengths;
}

f64 EstimateRenderTime(const HyperspectralConfig& config, f64 singleBandTime) {
    return static_cast<f64>(config.GetNumBands()) * singleBandTime;
}

// ============================================================================
// HyperspectralRenderer::Impl - PIMPL Implementation
// ============================================================================

struct HyperspectralRenderer::Impl {
    VulkanContext& context;
    RayTracingPipeline& pipeline;
    Scene& scene;

    // Phase 3: Use BatchRenderer for core rendering
    std::unique_ptr<BatchRenderer> batchRenderer;

    // Phase 4: GPU-accelerated spectral reconstruction
    std::unique_ptr<GpuSpectralReconstructor> gpuReconstructor;

    SpectralCube result;
    std::atomic<bool> cancelled{false};

    f64 lastRenderTime = 0.0;
    f64 avgTimePerBand = 0.0;

    // Adaptive sampling state (Phase 2)
    AdaptiveGridInfo adaptiveGrid;
    SpectralAnalysisResult analysisResult;

    Impl(VulkanContext& ctx, RayTracingPipeline& pipe, Scene& sc)
        : context(ctx), pipeline(pipe), scene(sc)
    {
        // Create BatchRenderer instance
        batchRenderer = std::make_unique<BatchRenderer>(ctx, pipe, sc);

        // Create GPU reconstructor (lazy initialization on first use)
        gpuReconstructor = std::make_unique<GpuSpectralReconstructor>(ctx);
    }

    /**
     * @brief Perform spectral analysis for adaptive sampling (Phase 2)
     */
    SpectralAnalysisResult AnalyzeSceneSpectra(
        f32 wavelengthMin, f32 wavelengthMax,
        const SpectralAnalysisParams& params
    ) {
        SpectralAnalyzer analyzer;
        return analyzer.AnalyzeScene(scene, wavelengthMin, wavelengthMax, params);
    }

    /**
     * @brief Convert BatchRenderStatus to HyperspectralStatus
     */
    static HyperspectralStatus ConvertStatus(BatchRenderStatus status) {
        switch (status) {
            case BatchRenderStatus::Success:
                return HyperspectralStatus::Success;
            case BatchRenderStatus::InvalidInput:
                return HyperspectralStatus::InvalidConfig;
            case BatchRenderStatus::PipelineError:
                return HyperspectralStatus::PipelineNotReady;
            case BatchRenderStatus::RenderFailed:
                return HyperspectralStatus::RenderFailed;
            case BatchRenderStatus::Cancelled:
                return HyperspectralStatus::Cancelled;
            case BatchRenderStatus::OutOfMemory:
            case BatchRenderStatus::GPUError:
                return HyperspectralStatus::OutOfMemory;
            default:
                return HyperspectralStatus::RenderFailed;
        }
    }
};

// ============================================================================
// HyperspectralRenderer - Public Interface
// ============================================================================

HyperspectralRenderer::HyperspectralRenderer(
    VulkanContext& context,
    RayTracingPipeline& pipeline,
    Scene& scene
) : m_impl(std::make_unique<Impl>(context, pipeline, scene)) {
    LOG_INFO("HyperspectralRenderer initialized (Phase 3 with BatchRenderer)");
}

HyperspectralRenderer::~HyperspectralRenderer() = default;

HyperspectralRenderer::HyperspectralRenderer(HyperspectralRenderer&&) noexcept = default;
HyperspectralRenderer& HyperspectralRenderer::operator=(HyperspectralRenderer&&) noexcept = default;

HyperspectralStatus HyperspectralRenderer::Render(
    const HyperspectralConfig& config,
    HyperspectralProgressCallback progressCallback,
    void* userData
) {
    using Clock = std::chrono::high_resolution_clock;

    // Reset cancellation flag
    m_impl->cancelled.store(false);
    m_impl->batchRenderer->ResetCancellation();

    // Validate configuration
    if (!config.IsValid()) {
        LOG_ERROR("Invalid hyperspectral config: {}", config.GetValidationError());
        return HyperspectralStatus::InvalidConfig;
    }

    // ========================================================================
    // Phase 2: Adaptive Sampling Setup
    // ========================================================================

    Vector<f32> wavelengthsToRender;
    bool useAdaptive = (config.adaptiveMode != AdaptiveSamplingMode::None);
    AdaptiveGridInfo gridInfo;

    if (useAdaptive && config.adaptiveMode == AdaptiveSamplingMode::Spectral) {
        LOG_INFO("Performing spectral analysis for adaptive sampling...");

        // Analyze scene spectral content
        SpectralAnalysisParams analysisParams;
        analysisParams.slopeThreshold = config.adaptiveDerivativeThreshold;
        analysisParams.prominenceThreshold = 0.02f;
        analysisParams.minFeatureSeparation_nm = config.wavelengthStep_nm * 2.0f;

        m_impl->analysisResult = m_impl->AnalyzeSceneSpectra(
            config.wavelengthMin_nm,
            config.wavelengthMax_nm,
            analysisParams
        );

        // Generate adaptive grid
        AdaptiveGridGenerator generator;
        gridInfo = generator.Generate(config, m_impl->analysisResult);
        m_impl->adaptiveGrid = gridInfo;

        wavelengthsToRender = gridInfo.wavelengths;

        LOG_INFO("Adaptive sampling: {} critical features detected, "
                 "rendering {} of {} wavelengths (compression: {:.2f}x)",
                 m_impl->analysisResult.features.size(),
                 gridInfo.totalWavelengths,
                 config.GetNumBands(),
                 gridInfo.compressionRatio);
    } else {
        // Uniform sampling (Phase 1 behavior)
        wavelengthsToRender = GenerateWavelengthList(config);
        gridInfo = AdaptiveGridGenerator::GenerateUniform(config);
    }

    u32 numBandsToRender = static_cast<u32>(wavelengthsToRender.size());

    LOG_INFO("Starting hyperspectral rendering: {} bands to render, {}-{} nm",
             numBandsToRender, config.wavelengthMin_nm, config.wavelengthMax_nm);

    // ========================================================================
    // Phase 3: Use BatchRenderer for Rendering Loop
    // ========================================================================

    auto startTime = Clock::now();

    // Setup batch render parameters
    BatchRenderParams batchParams;
    batchParams.spp = config.spp;
    batchParams.maxBounces = config.maxBounces;
    batchParams.verbose = false;

    // Create progress adapter to convert BatchRenderProgress to HyperspectralProgress
    BatchRenderProgressCallback batchCallback = nullptr;
    if (progressCallback) {
        batchCallback = [&](const BatchRenderProgress& bp, void*) {
            HyperspectralProgress hp;
            hp.currentBand = bp.currentBand;
            hp.totalBands = bp.totalBands;
            hp.currentWavelength_nm = bp.currentWavelength_nm;
            hp.elapsedSeconds = bp.elapsedSeconds;
            hp.estimatedTotalSeconds = bp.estimatedTotalSeconds;
            progressCallback(hp, userData);
        };
    }

    // Execute batch rendering
    auto [batchStatus, sparseCube] = m_impl->batchRenderer->RenderBatch(
        wavelengthsToRender,
        batchParams,
        batchCallback,
        nullptr
    );

    // Check for errors
    if (batchStatus != BatchRenderStatus::Success) {
        if (batchStatus == BatchRenderStatus::Cancelled) {
            LOG_INFO("Hyperspectral rendering cancelled");
        } else {
            LOG_ERROR("Batch rendering failed: {}",
                      BatchRenderStatusToString(batchStatus));
        }
        return Impl::ConvertStatus(batchStatus);
    }

    // Add metadata
    sparseCube.metadata["renderer"] = useAdaptive ?
        "Quantiloom HyperspectralRenderer Phase3 (Adaptive+BatchRenderer)" :
        "Quantiloom HyperspectralRenderer Phase3 (BatchRenderer)";
    if (useAdaptive) {
        sparseCube.metadata["adaptive_compression"] =
            std::to_string(gridInfo.compressionRatio);
        sparseCube.metadata["critical_features"] =
            std::to_string(m_impl->analysisResult.features.size());
    }

    // ========================================================================
    // Phase 2/4: Spectral Reconstruction (if adaptive)
    // ========================================================================

    if (useAdaptive) {
        LOG_INFO("Reconstructing full spectrum from adaptive samples...");

        // Phase 4: Use GPU reconstruction if enabled and available
        if (config.useGpuReconstruction && IsGpuReconstructionSupported(m_impl->context)) {
            LOG_INFO("Using GPU-accelerated reconstruction");

            GpuReconstructorConfig gpuConfig;
            gpuConfig.method = InterpolationMethod::CatmullRom;
            gpuConfig.verbose = true;

            auto [gpuStatus, gpuResult] = m_impl->gpuReconstructor->Reconstruct(
                sparseCube,
                gridInfo,
                config,
                gpuConfig
            );

            if (gpuStatus == GpuReconstructorStatus::Success) {
                m_impl->result = std::move(gpuResult);
                LOG_INFO("GPU reconstruction complete in {:.3f}s ({:.2f} Mpixels/s)",
                         m_impl->gpuReconstructor->GetLastReconstructionTime(),
                         m_impl->gpuReconstructor->GetPixelsPerSecond() / 1e6);
            } else {
                LOG_WARN("GPU reconstruction failed ({}), falling back to CPU",
                         GpuReconstructorStatusToString(gpuStatus));

                // Fallback to CPU reconstruction
                SpectralReconstructor reconstructor;
                m_impl->result = reconstructor.Reconstruct(
                    sparseCube,
                    gridInfo,
                    config,
                    InterpolationMethod::CatmullRom
                );
            }
        } else {
            // CPU reconstruction (Phase 2 behavior)
            SpectralReconstructor reconstructor;
            m_impl->result = reconstructor.Reconstruct(
                sparseCube,
                gridInfo,
                config,
                InterpolationMethod::CatmullRom
            );
        }

        LOG_INFO("Spectral reconstruction complete");
    } else {
        // No reconstruction needed for uniform sampling
        m_impl->result = std::move(sparseCube);
    }

    // Calculate final statistics
    auto endTime = Clock::now();
    m_impl->lastRenderTime = std::chrono::duration<f64>(endTime - startTime).count();
    m_impl->avgTimePerBand = m_impl->batchRenderer->GetAverageTimePerBand();

    LOG_INFO("Hyperspectral rendering complete: {} bands rendered in {:.2f}s ({:.3f}s/band)",
             numBandsToRender, m_impl->lastRenderTime, m_impl->avgTimePerBand);

    if (useAdaptive) {
        f64 uniformTime = m_impl->avgTimePerBand * config.GetNumBands();
        f64 savedTime = uniformTime - m_impl->lastRenderTime;
        LOG_INFO("Adaptive sampling saved {:.1f}s ({:.1f}%% reduction)",
                 savedTime, 100.0 * savedTime / uniformTime);
    }

    // ========================================================================
    // Save Intermediate Band Images (Debug Only)
    // ========================================================================

    if (config.saveIntermediates && !config.outputPath.empty()) {
        LOG_INFO("Saving intermediate band images for debugging...");

        // Create output directory
        std::filesystem::path outDir = std::filesystem::path(config.outputPath).parent_path();
        std::string baseName = std::filesystem::path(config.outputPath).stem().string();

        if (outDir.empty()) {
            outDir = ".";
        }

        std::filesystem::path intermediateDir = outDir / (baseName + "_bands");
        std::filesystem::create_directories(intermediateDir);

        const SpectralCube& cube = m_impl->result;
        u32 width = cube.width;
        u32 height = cube.height;
        u32 numBands = static_cast<u32>(cube.wavelengths.size());

        for (u32 bandIdx = 0; bandIdx < numBands; ++bandIdx) {
            f32 wavelength = cube.wavelengths[bandIdx];

            // Create grayscale Image from band data
            Image bandImage;
            bandImage.Resize(width, height, 1);
            bandImage.channelNames = {"Y"};  // Luminance channel

            const f32* bandPtr = cube.BandPtr(bandIdx);
            std::memcpy(bandImage.data.data(), bandPtr, width * height * sizeof(f32));

            // Add metadata
            bandImage.metadata["wavelength_nm"] = std::to_string(wavelength);
            bandImage.metadata["band_index"] = std::to_string(bandIdx);

            // Generate filename
            char filename[256];
            std::snprintf(filename, sizeof(filename),
                         "band_%03u_%04.0fnm.exr", bandIdx, wavelength);

            std::filesystem::path filePath = intermediateDir / filename;

            if (!ImageIO::WriteEXR(filePath.string(), bandImage)) {
                LOG_WARN("Failed to save intermediate band {}: {}", bandIdx, filePath.string());
            }
        }

        LOG_INFO("Saved {} intermediate band images to: {}",
                 numBands, intermediateDir.string());
    }

    // ========================================================================
    // Write Output
    // ========================================================================

    if (!config.outputPath.empty()) {
        bool writeSuccess = false;

        switch (config.outputFormat) {
            case HyperspectralOutputFormat::ENVI_BSQ:
            case HyperspectralOutputFormat::ENVI_BIL:
            case HyperspectralOutputFormat::ENVI_BIP:
                writeSuccess = WriteENVI(config.outputPath, config.outputFormat);
                break;
            case HyperspectralOutputFormat::GeoTIFF:
                writeSuccess = WriteGeoTIFF(config.outputPath + ".tif");
                break;
            case HyperspectralOutputFormat::EXR_Multipart:
                writeSuccess = WriteEXR(config.outputPath + ".exr");
                break;
        }

        if (!writeSuccess) {
            LOG_ERROR("Failed to write hyperspectral output to {}", config.outputPath);
            return HyperspectralStatus::OutputWriteFailed;
        }
    }

    return HyperspectralStatus::Success;
}

bool HyperspectralRenderer::RenderSingleWavelength(
    f32 wavelength_nm,
    u32 spp,
    Image& outImage
) {
    BatchRenderParams params;
    params.spp = spp;

    return m_impl->batchRenderer->RenderSingleBand(wavelength_nm, params, outImage);
}

void HyperspectralRenderer::Cancel() {
    m_impl->cancelled.store(true);
    m_impl->batchRenderer->Cancel();
    LOG_INFO("Hyperspectral rendering cancellation requested");
}

bool HyperspectralRenderer::IsCancelled() const {
    return m_impl->cancelled.load();
}

const SpectralCube& HyperspectralRenderer::GetResult() const {
    return m_impl->result;
}

SpectralCube HyperspectralRenderer::TakeResult() {
    return std::move(m_impl->result);
}

bool HyperspectralRenderer::HasResult() const {
    return m_impl->result.IsValid();
}

bool HyperspectralRenderer::WriteENVI(
    const String& basePath,
    HyperspectralOutputFormat interleave
) const {
    if (!HasResult()) {
        LOG_ERROR("No result to write");
        return false;
    }

    ENVIInterleave enviInterleave;
    switch (interleave) {
        case HyperspectralOutputFormat::ENVI_BSQ:
            enviInterleave = ENVIInterleave::BSQ;
            break;
        case HyperspectralOutputFormat::ENVI_BIL:
            enviInterleave = ENVIInterleave::BIL;
            break;
        case HyperspectralOutputFormat::ENVI_BIP:
            enviInterleave = ENVIInterleave::BIP;
            break;
        default:
            enviInterleave = ENVIInterleave::BSQ;
    }

    return SpectralCubeIO::WriteENVI(m_impl->result, basePath, enviInterleave);
}

bool HyperspectralRenderer::WriteGeoTIFF(const String& path) const {
    if (!HasResult()) {
        LOG_ERROR("No result to write");
        return false;
    }

    return SpectralCubeIO::WriteGeoTIFF(m_impl->result, path);
}

bool HyperspectralRenderer::WriteEXR(const String& path) const {
    if (!HasResult()) {
        LOG_ERROR("No result to write");
        return false;
    }

    return SpectralCubeIO::WriteEXR(m_impl->result, path);
}

f64 HyperspectralRenderer::GetLastRenderTime() const {
    return m_impl->lastRenderTime;
}

f64 HyperspectralRenderer::GetAverageTimePerBand() const {
    return m_impl->avgTimePerBand;
}

} // namespace quantiloom
