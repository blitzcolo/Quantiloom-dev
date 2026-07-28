/**
 * @file HyperspectralRenderer.hpp
 * @brief Core hyperspectral rendering engine
 *
 * HyperspectralRenderer orchestrates multi-wavelength rendering:
 * - Iterates over wavelength range from HyperspectralConfig
 * - Invokes existing single-wavelength renderer for each band
 * - Assembles results into SpectralCube
 * - Exports to ENVI/GeoTIFF format
 *
 * Phase 1: Basic batch rendering (no adaptive sampling)
 * Phase 2+: Adaptive spectral sampling based on material features
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralCube.hpp"
#include "core/Image.hpp"
#include "hs_core/HyperspectralConfig.hpp"

#include <memory>
#include <functional>
#include <atomic>

namespace quantiloom {

// Forward declarations
class RayTracingPipeline;
class Scene;
class VulkanContext;

// ============================================================================
// Rendering Status
// ============================================================================

/**
 * @enum HyperspectralStatus
 * @brief Status codes for hyperspectral rendering operations
 */
enum class HyperspectralStatus : u32 {
    Success = 0,
    InvalidConfig,
    PipelineNotReady,
    RenderFailed,
    OutputWriteFailed,
    Cancelled,
    OutOfMemory
};

/**
 * @brief Convert status to human-readable string
 */
QL_API const char* HyperspectralStatusToString(HyperspectralStatus status);

// ============================================================================
// HyperspectralRenderer - Core Rendering Engine
// ============================================================================

/**
 * @class HyperspectralRenderer
 * @brief Orchestrates hyperspectral (multi-wavelength) rendering
 *
 * This class wraps the existing single-wavelength RayTracingPipeline
 * to produce hyperspectral data cubes. It handles:
 * - Wavelength iteration according to HyperspectralConfig
 * - Progress reporting via callbacks
 * - Result assembly into SpectralCube
 * - Output to ENVI/GeoTIFF formats
 *
 * Example usage:
 * @code
 * HyperspectralRenderer renderer(vulkanContext, pipeline, scene);
 *
 * auto config = HyperspectralConfig::MWIR(50.0f);
 * config.spp = 64;
 * config.outputPath = "thermal_cube";
 *
 * auto status = renderer.Render(config, [](const HyperspectralProgress& p, void*) {
 *     std::cout << "Progress: " << p.GetPercentage() << "%" << std::endl;
 * });
 *
 * if (status == HyperspectralStatus::Success) {
 *     const auto& cube = renderer.GetResult();
 *     // Process cube...
 * }
 * @endcode
 */
class QL_API HyperspectralRenderer {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    /**
     * @brief Construct hyperspectral renderer
     * @param context Vulkan context (must remain valid during rendering)
     * @param pipeline Ray tracing pipeline (must support single-wavelength mode)
     * @param scene Scene to render (must remain valid during rendering)
     */
    HyperspectralRenderer(
        VulkanContext& context,
        RayTracingPipeline& pipeline,
        Scene& scene
    );

    ~HyperspectralRenderer();

    // Non-copyable, movable
    HyperspectralRenderer(const HyperspectralRenderer&) = delete;
    HyperspectralRenderer& operator=(const HyperspectralRenderer&) = delete;
    HyperspectralRenderer(HyperspectralRenderer&&) noexcept;
    HyperspectralRenderer& operator=(HyperspectralRenderer&&) noexcept;

    // ========================================================================
    // Core Rendering Interface
    // ========================================================================

    /**
     * @brief Render hyperspectral data cube
     *
     * This is the main entry point for hyperspectral rendering. It:
     * 1. Validates configuration
     * 2. Allocates SpectralCube for results
     * 3. Iterates over wavelengths, calling RenderSingleWavelength for each
     * 4. Reports progress via callback
     * 5. Writes output to file if outputPath is set
     *
     * @param config Hyperspectral rendering configuration
     * @param progressCallback Optional callback for progress updates (can be nullptr)
     * @param userData User data passed to progress callback
     * @return Status code indicating success or failure reason
     *
     * @note This is a blocking call. For async rendering, use RenderAsync.
     * @note The result can be retrieved via GetResult() after successful completion.
     */
    HyperspectralStatus Render(
        const HyperspectralConfig& config,
        HyperspectralProgressCallback progressCallback = nullptr,
        void* userData = nullptr
    );

    /**
     * @brief Render single wavelength band
     *
     * Low-level method to render a single wavelength. Used internally
     * by Render() but exposed for custom wavelength iteration patterns.
     *
     * @param wavelength_nm Wavelength to render (nm)
     * @param spp Samples per pixel
     * @param[out] outImage Output image (width x height, grayscale radiance)
     * @return true on success
     */
    bool RenderSingleWavelength(
        f32 wavelength_nm,
        u32 spp,
        Image& outImage
    );

    /**
     * @brief Cancel ongoing rendering
     *
     * Thread-safe cancellation. Sets internal flag that is checked
     * between wavelength renders. Rendering will stop at the next
     * wavelength boundary.
     */
    void Cancel();

    /**
     * @brief Check if rendering was cancelled
     */
    [[nodiscard]] bool IsCancelled() const;

    // ========================================================================
    // Result Access
    // ========================================================================

    /**
     * @brief Get rendered hyperspectral cube
     * @return Reference to result SpectralCube (valid after successful Render())
     *
     * @note Returns empty cube if Render() hasn't been called or failed.
     */
    [[nodiscard]] const SpectralCube& GetResult() const;

    /**
     * @brief Move result out of renderer
     * @return SpectralCube (moved, renderer's copy becomes empty)
     *
     * Use this to take ownership of the result without copying.
     */
    [[nodiscard]] SpectralCube TakeResult();

    /**
     * @brief Check if result is available
     */
    [[nodiscard]] bool HasResult() const;

    // ========================================================================
    // Output Methods
    // ========================================================================

    /**
     * @brief Write result to ENVI format
     * @param basePath Output path (without extension, will add .hdr and .dat)
     * @param interleave Interleave format (BSQ, BIL, or BIP)
     * @return true on success
     */
    bool WriteENVI(
        const String& basePath,
        HyperspectralOutputFormat interleave = HyperspectralOutputFormat::ENVI_BSQ
    ) const;

    /**
     * @brief Write result to GeoTIFF format
     * @param path Output path (with .tif extension)
     * @return true on success
     */
    bool WriteGeoTIFF(const String& path) const;

    /**
     * @brief Write result to EXR multipart format
     * @param path Output path (with .exr extension)
     * @return true on success
     */
    bool WriteEXR(const String& path) const;

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * @brief Get total rendering time for last Render() call
     * @return Time in seconds
     */
    [[nodiscard]] f64 GetLastRenderTime() const;

    /**
     * @brief Get average time per wavelength band
     * @return Time in seconds
     */
    [[nodiscard]] f64 GetAverageTimePerBand() const;

private:
    // Implementation details hidden via PIMPL
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Generate wavelength list from config
 * @param config Hyperspectral configuration
 * @return Vector of wavelengths in nm
 */
Vector<f32> GenerateWavelengthList(const HyperspectralConfig& config);

/**
 * @brief Estimate rendering time based on single-band benchmark
 * @param config Hyperspectral configuration
 * @param singleBandTime Time to render one band (seconds)
 * @return Estimated total time in seconds
 */
f64 EstimateRenderTime(const HyperspectralConfig& config, f64 singleBandTime);

} // namespace quantiloom
