/**
 * @file GpuSpectralReconstructor.hpp
 * @brief GPU-accelerated spectral cube reconstruction
 *
 * GpuSpectralReconstructor uses Vulkan compute shaders to parallelize
 * spectral interpolation across all pixels. Each pixel's spectrum is
 * reconstructed independently, making this highly parallelizable.
 *
 * Performance comparison (typical 1024x1024 image, 200 bands):
 * - CPU SpectralReconstructor: ~2-5 seconds
 * - GPU GpuSpectralReconstructor: ~50-200 ms (10-40x speedup)
 *
 * Supported interpolation methods:
 * - Linear: Simple, fast, may not preserve spectral shape
 * - Catmull-Rom: Smooth, preserves curvature, recommended
 * - Akima: Robust to outliers, more computation
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralCube.hpp"
#include "hs_core/SpectralReconstructor.hpp"  // For InterpolationMethod enum
#include "hs_core/AdaptiveGridGenerator.hpp"
#include "hs_core/HyperspectralConfig.hpp"

#include <memory>

namespace quantiloom {

// Forward declarations
class VulkanContext;

// ============================================================================
// GpuReconstructorStatus
// ============================================================================

/**
 * @enum GpuReconstructorStatus
 * @brief Status codes for GPU reconstruction operations
 */
enum class GpuReconstructorStatus : u32 {
    Success = 0,          ///< Operation completed successfully
    NotInitialized,       ///< Reconstructor not initialized
    InvalidInput,         ///< Invalid input parameters
    ShaderCompileFailed,  ///< Failed to compile compute shader
    PipelineCreateFailed, ///< Failed to create compute pipeline
    BufferAllocFailed,    ///< Failed to allocate GPU buffers
    DispatchFailed,       ///< Compute dispatch failed
    ReadbackFailed,       ///< Failed to read results from GPU
    OutOfMemory,          ///< GPU memory allocation failed
    Timeout               ///< GPU operation timed out
};

/**
 * @brief Convert status to human-readable string
 */
const char* GpuReconstructorStatusToString(GpuReconstructorStatus status);

// ============================================================================
// GpuReconstructorConfig
// ============================================================================

/**
 * @struct GpuReconstructorConfig
 * @brief Configuration for GPU reconstruction
 */
struct GpuReconstructorConfig {
    InterpolationMethod method = InterpolationMethod::CatmullRom;

    // Workgroup size (should match shader)
    u32 workgroupSizeX = 16;
    u32 workgroupSizeY = 16;

    // Timeout for GPU operations (milliseconds)
    u32 timeoutMs = 10000;  // 10 seconds

    // Enable verbose logging
    bool verbose = false;
};

// ============================================================================
// GpuSpectralReconstructor
// ============================================================================

/**
 * @class GpuSpectralReconstructor
 * @brief GPU-accelerated spectral cube reconstruction
 *
 * Uses Vulkan compute shaders to parallelize spectral interpolation.
 * Each pixel's spectrum is processed in parallel on the GPU.
 *
 * Example usage:
 * @code
 * GpuSpectralReconstructor reconstructor(vulkanContext);
 *
 * auto [status, fullCube] = reconstructor.Reconstruct(
 *     sparseCube,
 *     adaptiveGridInfo,
 *     targetConfig,
 *     GpuReconstructorConfig{.method = InterpolationMethod::CatmullRom}
 * );
 *
 * if (status == GpuReconstructorStatus::Success) {
 *     // Use fullCube
 * }
 * @endcode
 */
class GpuSpectralReconstructor {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    /**
     * @brief Construct GPU reconstructor
     * @param context VulkanContext for GPU operations
     */
    explicit GpuSpectralReconstructor(VulkanContext& context);

    ~GpuSpectralReconstructor();

    // Non-copyable, movable
    GpuSpectralReconstructor(const GpuSpectralReconstructor&) = delete;
    GpuSpectralReconstructor& operator=(const GpuSpectralReconstructor&) = delete;
    GpuSpectralReconstructor(GpuSpectralReconstructor&&) noexcept;
    GpuSpectralReconstructor& operator=(GpuSpectralReconstructor&&) noexcept;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize GPU resources (pipeline, shader)
     *
     * Must be called before Reconstruct(). Can be called multiple times
     * safely (subsequent calls are no-ops if already initialized).
     *
     * @return Success if initialization succeeded
     */
    GpuReconstructorStatus Initialize();

    /**
     * @brief Check if reconstructor is initialized and ready
     */
    bool IsReady() const;

    /**
     * @brief Release GPU resources
     *
     * Called automatically by destructor.
     */
    void Shutdown();

    // ========================================================================
    // Main Reconstruction Interface
    // ========================================================================

    /**
     * @brief Reconstruct full cube from adaptive samples (GPU-accelerated)
     *
     * @param sparseCube SpectralCube with adaptive sampling
     * @param gridInfo Adaptive grid information
     * @param targetConfig Target full-resolution configuration
     * @param config GPU reconstruction configuration
     * @return Pair of (status, reconstructed cube)
     */
    std::pair<GpuReconstructorStatus, SpectralCube> Reconstruct(
        const SpectralCube& sparseCube,
        const AdaptiveGridInfo& gridInfo,
        const HyperspectralConfig& targetConfig,
        const GpuReconstructorConfig& config = GpuReconstructorConfig{}
    );

    /**
     * @brief Reconstruct with custom wavelength grids
     *
     * More flexible version that doesn't require AdaptiveGridInfo.
     *
     * @param sparseCube Source sparse cube
     * @param targetWavelengths Target wavelength grid
     * @param config GPU reconstruction configuration
     * @return Pair of (status, reconstructed cube)
     */
    std::pair<GpuReconstructorStatus, SpectralCube> ReconstructCustom(
        const SpectralCube& sparseCube,
        const Vector<f32>& targetWavelengths,
        const GpuReconstructorConfig& config = GpuReconstructorConfig{}
    );

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * @brief Get time spent on last reconstruction (seconds)
     */
    f64 GetLastReconstructionTime() const;

    /**
     * @brief Get GPU memory usage for last reconstruction (bytes)
     */
    u64 GetLastMemoryUsage() const;

    /**
     * @brief Get number of pixels processed per second
     */
    f64 GetPixelsPerSecond() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if GPU reconstruction is available
 *
 * Returns true if the GPU supports required features for compute-based
 * reconstruction (compute shaders, sufficient buffer sizes).
 *
 * @param context VulkanContext to check
 * @return true if GPU reconstruction is supported
 */
bool IsGpuReconstructionSupported(VulkanContext& context);

/**
 * @brief Estimate GPU memory required for reconstruction
 *
 * @param sparseCube Source sparse cube
 * @param targetBands Number of target bands
 * @return Estimated memory usage in bytes
 */
u64 EstimateReconstructionMemory(
    const SpectralCube& sparseCube,
    u32 targetBands
);

} // namespace quantiloom
