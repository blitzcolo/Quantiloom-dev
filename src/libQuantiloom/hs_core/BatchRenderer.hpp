/**
 * @file BatchRenderer.hpp
 * @brief Batch wavelength rendering for hyperspectral data cube generation
 *
 * BatchRenderer is a low-level component that iterates over an arbitrary
 * wavelength list and renders each band using the single-wavelength mode
 * of the ray tracing pipeline.
 *
 * Design philosophy:
 * - Single responsibility: only handles batch rendering loop
 * - No adaptive sampling logic (handled by AdaptiveGridGenerator)
 * - No reconstruction logic (handled by SpectralReconstructor)
 * - Outputs raw sparse SpectralCube for downstream processing
 *
 * Usage pattern:
 * 1. AdaptiveGridGenerator decides which wavelengths to render
 * 2. BatchRenderer renders those wavelengths -> sparse_cube
 * 3. SpectralReconstructor interpolates sparse_cube -> full_cube
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralCube.hpp"
#include "core/Image.hpp"

#include <memory>
#include <functional>
#include <atomic>

namespace quantiloom {

// Forward declarations
class RayTracingPipeline;
class Scene;
class VulkanContext;

// ============================================================================
// Batch Rendering Status
// ============================================================================

/**
 * @enum BatchRenderStatus
 * @brief Status codes for batch rendering operations
 */
enum class BatchRenderStatus : u32 {
    Success = 0,        ///< Rendering completed successfully
    InvalidInput,       ///< Invalid wavelength list or parameters
    PipelineError,      ///< Ray tracing pipeline error
    RenderFailed,       ///< Individual band render failed
    Cancelled,          ///< Rendering cancelled by user
    OutOfMemory,        ///< Failed to allocate output cube
    GPUError            ///< GPU resource allocation/execution error
};

/**
 * @brief Convert status to human-readable string
 */
QL_API const char* BatchRenderStatusToString(BatchRenderStatus status);

// ============================================================================
// Batch Rendering Progress
// ============================================================================

/**
 * @struct BatchRenderProgress
 * @brief Progress information for batch rendering
 */
struct BatchRenderProgress {
    u32 currentBand;              ///< Current band being rendered (1-based)
    u32 totalBands;               ///< Total bands to render
    f32 currentWavelength_nm;     ///< Current wavelength in nm
    f32 elapsedSeconds;           ///< Time elapsed since start
    f32 estimatedTotalSeconds;    ///< Estimated total time (0 if unknown)

    /**
     * @brief Get completion percentage
     */
    [[nodiscard]] f32 GetPercentage() const {
        return (totalBands > 0)
            ? 100.0f * static_cast<f32>(currentBand) / static_cast<f32>(totalBands)
            : 0.0f;
    }

    /**
     * @brief Get estimated remaining time in seconds
     */
    [[nodiscard]] f32 GetRemainingSeconds() const {
        return estimatedTotalSeconds - elapsedSeconds;
    }
};

/**
 * @brief Progress callback type for batch rendering
 * @param progress Current progress information
 * @param userData User-provided context pointer
 */
using BatchRenderProgressCallback = std::function<void(const BatchRenderProgress&, void*)>;

// ============================================================================
// BatchRenderParams - Rendering Parameters
// ============================================================================

/**
 * @struct BatchRenderParams
 * @brief Parameters for batch wavelength rendering
 */
struct BatchRenderParams {
    u32 spp = 16;                 ///< Samples per pixel
    u32 maxBounces = 4;           ///< Maximum ray bounces
    bool enableAccumulation = true; ///< Accumulate samples across frames
    bool verbose = false;         ///< Enable verbose logging
};

// ============================================================================
// BatchRenderer - Core Batch Rendering Class
// ============================================================================

/**
 * @class BatchRenderer
 * @brief Renders multiple wavelength bands into a sparse SpectralCube
 *
 * BatchRenderer provides the core rendering loop for hyperspectral imaging.
 * Given an arbitrary list of wavelengths (not necessarily uniformly spaced),
 * it renders each wavelength using the single-wavelength ray tracing mode
 * and assembles results into a SpectralCube.
 *
 * Key features:
 * - Accepts arbitrary wavelength lists (supports adaptive sampling)
 * - Thread-safe cancellation support
 * - Progress reporting with ETA estimation
 * - Automatic GPU resource management
 *
 * Example usage:
 * @code
 * // Create renderer
 * BatchRenderer batchRenderer(vulkanContext, pipeline, scene);
 *
 * // Define wavelengths to render (could be from AdaptiveGridGenerator)
 * Vector<f32> wavelengths = {3000.0f, 3100.0f, 3200.0f, 3400.0f, 3600.0f};
 *
 * // Render batch
 * BatchRenderParams params;
 * params.spp = 64;
 *
 * auto [status, sparseCube] = batchRenderer.RenderBatch(
 *     wavelengths, params,
 *     [](const BatchRenderProgress& p, void*) {
 *         std::cout << "Band " << p.currentBand << "/" << p.totalBands
 *                   << " (" << p.currentWavelength_nm << " nm)" << std::endl;
 *     }
 * );
 *
 * if (status == BatchRenderStatus::Success) {
 *     // sparseCube now contains rendered bands
 *     // Pass to SpectralReconstructor if interpolation needed
 * }
 * @endcode
 */
class QL_API BatchRenderer {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    /**
     * @brief Construct batch renderer
     * @param context Vulkan context (must remain valid during rendering)
     * @param pipeline Ray tracing pipeline (must support single-wavelength mode)
     * @param scene Scene to render (must remain valid during rendering)
     */
    BatchRenderer(
        VulkanContext& context,
        RayTracingPipeline& pipeline,
        Scene& scene
    );

    ~BatchRenderer();

    // Non-copyable, movable
    BatchRenderer(const BatchRenderer&) = delete;
    BatchRenderer& operator=(const BatchRenderer&) = delete;
    BatchRenderer(BatchRenderer&&) noexcept;
    BatchRenderer& operator=(BatchRenderer&&) noexcept;

    // ========================================================================
    // Core Rendering Interface
    // ========================================================================

    /**
     * @brief Render batch of wavelengths into sparse SpectralCube
     *
     * Main entry point for batch rendering. Iterates over the provided
     * wavelength list, rendering each band using single-wavelength mode.
     *
     * @param wavelengths List of wavelengths to render (nm), must be sorted
     * @param params Rendering parameters (SPP, bounces, etc.)
     * @param progressCallback Optional progress callback
     * @param userData User data for progress callback
     * @return Pair of (status, SpectralCube) - cube is valid only if Success
     *
     * @note Wavelengths need not be uniformly spaced
     * @note Output cube's wavelengths array matches input wavelengths
     */
    std::pair<BatchRenderStatus, SpectralCube> RenderBatch(
        const Vector<f32>& wavelengths,
        const BatchRenderParams& params = BatchRenderParams{},
        BatchRenderProgressCallback progressCallback = nullptr,
        void* userData = nullptr
    );

    /**
     * @brief Render single wavelength band
     *
     * Low-level method to render a single wavelength. Can be used
     * for custom iteration patterns or progressive rendering.
     *
     * @param wavelength_nm Wavelength to render (nm)
     * @param params Rendering parameters
     * @param[out] outImage Output image (width x height, grayscale radiance)
     * @return true on success
     */
    bool RenderSingleBand(
        f32 wavelength_nm,
        const BatchRenderParams& params,
        Image& outImage
    );

    // ========================================================================
    // Cancellation Support
    // ========================================================================

    /**
     * @brief Cancel ongoing batch rendering
     *
     * Thread-safe. Sets internal cancellation flag checked between bands.
     * Current band will complete before cancellation takes effect.
     */
    void Cancel();

    /**
     * @brief Check if rendering was cancelled
     */
    [[nodiscard]] bool IsCancelled() const;

    /**
     * @brief Reset cancellation flag
     *
     * Call before starting a new render if reusing the BatchRenderer.
     */
    void ResetCancellation();

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * @brief Get total rendering time for last RenderBatch() call
     * @return Time in seconds
     */
    [[nodiscard]] f64 GetLastRenderTime() const;

    /**
     * @brief Get average time per wavelength band
     * @return Time in seconds
     */
    [[nodiscard]] f64 GetAverageTimePerBand() const;

    /**
     * @brief Get number of bands rendered in last batch
     */
    [[nodiscard]] u32 GetLastBandCount() const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * @brief Get current image dimensions
     * @return Pair of (width, height) from scene
     */
    [[nodiscard]] std::pair<u32, u32> GetImageDimensions() const;

    /**
     * @brief Check if GPU resources are ready
     * @return true if output image is allocated
     */
    [[nodiscard]] bool IsReady() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Validate wavelength list for batch rendering
 * @param wavelengths List of wavelengths to validate
 * @return true if valid (non-empty, positive values, sorted)
 */
QL_API bool ValidateWavelengthList(const Vector<f32>& wavelengths);

/**
 * @brief Sort wavelength list (in-place)
 * @param wavelengths List to sort
 */
QL_API void SortWavelengths(Vector<f32>& wavelengths);

/**
 * @brief Estimate batch rendering time
 * @param numBands Number of bands to render
 * @param singleBandTime Average time per band (seconds)
 * @return Estimated total time in seconds
 */
QL_API f64 EstimateBatchTime(u32 numBands, f64 singleBandTime);

} // namespace quantiloom
