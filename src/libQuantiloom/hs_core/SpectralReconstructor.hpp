/**
 * @file SpectralReconstructor.hpp
 * @brief Reconstructs full hyperspectral cubes from adaptively sampled data
 *
 * SpectralReconstructor interpolates missing wavelength bands to rebuild
 * the complete hyperspectral data cube from sparse adaptive sampling.
 *
 * Interpolation methods:
 * - Linear: Simple, fast, may not preserve spectral shape
 * - Cubic (Catmull-Rom): Smooth, preserves curvature, recommended
 * - Akima: Robust to outliers, reduces overshoots
 *
 * Physical considerations:
 * - Interpolation is only applied in "flat" spectral regions
 * - Critical regions (peaks, valleys, edges) are rendered at full resolution
 * - Reconstruction error is bounded by the spectral analysis thresholds
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralCube.hpp"
#include "hs_core/AdaptiveGridGenerator.hpp"
#include "hs_core/HyperspectralConfig.hpp"

namespace quantiloom {

// ============================================================================
// Interpolation Method
// ============================================================================

/**
 * @enum InterpolationMethod
 * @brief Spectral interpolation algorithm selection
 */
enum class InterpolationMethod : u32 {
    Linear,      ///< Linear interpolation (fastest)
    CatmullRom,  ///< Catmull-Rom cubic spline (smooth, recommended)
    Akima        ///< Akima spline (robust to outliers)
};

// ============================================================================
// SpectralReconstructor - Cube Reconstruction
// ============================================================================

/**
 * @class SpectralReconstructor
 * @brief Reconstructs full hyperspectral cubes from adaptive samples
 *
 * Takes a sparsely-sampled SpectralCube (rendered at adaptive wavelengths)
 * and reconstructs the full-resolution cube by interpolating missing bands.
 *
 * Example usage:
 * @code
 * // After adaptive rendering
 * SpectralCube sparseCube = renderer.TakeResult();
 * AdaptiveGridInfo gridInfo = ...; // from generator
 * HyperspectralConfig fullConfig = ...; // target full resolution
 *
 * SpectralReconstructor reconstructor;
 * SpectralCube fullCube = reconstructor.Reconstruct(
 *     sparseCube, gridInfo, fullConfig, InterpolationMethod::CatmullRom
 * );
 *
 * // fullCube now has all wavelengths, interpolated where needed
 * @endcode
 */
class QL_API SpectralReconstructor {
public:
    SpectralReconstructor() = default;
    ~SpectralReconstructor() = default;

    // ========================================================================
    // Main Reconstruction Interface
    // ========================================================================

    /**
     * @brief Reconstruct full cube from adaptive samples
     *
     * @param sparseCube SpectralCube with adaptive sampling
     * @param gridInfo Adaptive grid information used during rendering
     * @param targetConfig Target full-resolution configuration
     * @param method Interpolation method
     * @return Reconstructed full-resolution SpectralCube
     */
    SpectralCube Reconstruct(
        const SpectralCube& sparseCube,
        const AdaptiveGridInfo& gridInfo,
        const HyperspectralConfig& targetConfig,
        InterpolationMethod method = InterpolationMethod::CatmullRom
    );

    /**
     * @brief Reconstruct using pre-computed mapping
     *
     * @param sparseCube SpectralCube with adaptive sampling
     * @param mapping Pre-computed reconstruction mapping
     * @param targetConfig Target configuration (for cube dimensions)
     * @param method Interpolation method
     * @return Reconstructed SpectralCube
     */
    SpectralCube ReconstructWithMapping(
        const SpectralCube& sparseCube,
        const Vector<ReconstructionMapping>& mapping,
        const HyperspectralConfig& targetConfig,
        InterpolationMethod method = InterpolationMethod::CatmullRom
    );

    // ========================================================================
    // Per-Pixel Interpolation
    // ========================================================================

    /**
     * @brief Interpolate spectrum at a single pixel
     *
     * @param sparseCube Source sparse cube
     * @param x Pixel X coordinate
     * @param y Pixel Y coordinate
     * @param targetWavelengths Target wavelength grid
     * @param method Interpolation method
     * @return Interpolated spectrum values
     */
    Vector<f32> InterpolatePixelSpectrum(
        const SpectralCube& sparseCube,
        u32 x, u32 y,
        const Vector<f32>& targetWavelengths,
        InterpolationMethod method = InterpolationMethod::CatmullRom
    );

    // ========================================================================
    // Error Estimation
    // ========================================================================

    /**
     * @brief Estimate maximum interpolation error
     *
     * Computes expected error bounds based on the adaptive grid spacing
     * and assumed spectral smoothness in flat regions.
     *
     * @param gridInfo Adaptive grid information
     * @param targetConfig Target configuration
     * @return Estimated maximum relative error (0-1)
     */
    static f32 EstimateMaxError(
        const AdaptiveGridInfo& gridInfo,
        const HyperspectralConfig& targetConfig
    );

private:
    // ========================================================================
    // Interpolation Kernels
    // ========================================================================

    /**
     * @brief Linear interpolation between two values
     */
    static f32 LinearInterp(f32 v0, f32 v1, f32 t);

    /**
     * @brief Catmull-Rom cubic spline interpolation
     *
     * Requires 4 points: p0, p1, p2, p3
     * Interpolates between p1 and p2 with parameter t in [0,1]
     */
    static f32 CatmullRomInterp(f32 p0, f32 p1, f32 p2, f32 p3, f32 t);

    /**
     * @brief Akima spline interpolation
     *
     * More robust to outliers than Catmull-Rom
     */
    static f32 AkimaInterp(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        f32 targetWavelength
    );

    /**
     * @brief Get interpolation indices for cubic methods
     *
     * Returns indices of 4 surrounding points (p0, p1, p2, p3)
     * where interpolation is between p1 and p2.
     */
    static void GetCubicIndices(
        const Vector<f32>& wavelengths,
        f32 target,
        usize& idx0, usize& idx1, usize& idx2, usize& idx3,
        f32& t
    );
};

} // namespace quantiloom
