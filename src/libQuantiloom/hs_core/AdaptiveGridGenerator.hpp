/**
 * @file AdaptiveGridGenerator.hpp
 * @brief Generates adaptive wavelength sampling grids based on spectral features
 *
 * AdaptiveGridGenerator creates non-uniform wavelength grids that:
 * - Preserve full resolution at critical spectral features (peaks, valleys, edges)
 * - Use coarse sampling in spectrally flat regions
 * - Guarantee physical accuracy while reducing render time
 *
 * Sampling strategy:
 * - Within criticalRadius of feature: use baseStep (user's original interval)
 * - In flat regions: use baseStep * coarseMultiplier (2x-4x original)
 *
 * @author wtflmao
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "hs_core/HyperspectralConfig.hpp"
#include "hs_core/SpectralAnalyzer.hpp"

namespace quantiloom {

// ============================================================================
// AdaptiveGridInfo - Grid Generation Metadata
// ============================================================================

/**
 * @struct AdaptiveGridInfo
 * @brief Information about the generated adaptive grid
 */
struct AdaptiveGridInfo {
    Vector<f32> wavelengths;        ///< Wavelengths to render (adaptive grid)
    Vector<bool> isCritical;        ///< True if wavelength is in critical region
    u32 totalWavelengths = 0;       ///< Number of wavelengths to render
    u32 criticalWavelengths = 0;    ///< Number in critical regions
    u32 coarseWavelengths = 0;      ///< Number in coarse-sampled regions
    f32 compressionRatio = 0.0f;    ///< Ratio: uniform_count / adaptive_count

    /**
     * @brief Get wavelength at index
     */
    [[nodiscard]] f32 GetWavelength(u32 index) const {
        return (index < wavelengths.size()) ? wavelengths[index] : 0.0f;
    }

    /**
     * @brief Check if wavelength at index is in critical region
     */
    [[nodiscard]] bool IsCritical(u32 index) const {
        return (index < isCritical.size()) ? isCritical[index] : false;
    }
};

// ============================================================================
// AdaptiveGridGenerator - Wavelength Grid Generation
// ============================================================================

/**
 * @class AdaptiveGridGenerator
 * @brief Generates adaptive wavelength grids for efficient hyperspectral rendering
 *
 * Given spectral analysis results and rendering configuration, this class
 * generates a non-uniform wavelength grid that balances accuracy and efficiency.
 *
 * Algorithm:
 * 1. Start with wavelengthMin
 * 2. For each position, check if within criticalRadius of any critical wavelength
 * 3. If critical: advance by baseStep
 * 4. If flat: advance by baseStep * coarseMultiplier
 * 5. Continue until wavelengthMax reached
 *
 * Example:
 * @code
 * // Analyze scene
 * SpectralAnalyzer analyzer;
 * auto analysis = analyzer.AnalyzeScene(scene, 3000, 5000, params);
 *
 * // Generate adaptive grid
 * AdaptiveGridGenerator generator;
 * HyperspectralConfig config = HyperspectralConfig::MWIR(50.0f);
 * config.adaptiveMode = AdaptiveSamplingMode::Spectral;
 *
 * auto gridInfo = generator.Generate(config, analysis);
 *
 * std::cout << "Reduced from " << config.GetNumBands()
 *           << " to " << gridInfo.totalWavelengths
 *           << " wavelengths (compression: " << gridInfo.compressionRatio << "x)"
 *           << std::endl;
 * @endcode
 */
class QL_API AdaptiveGridGenerator {
public:
    AdaptiveGridGenerator() = default;
    ~AdaptiveGridGenerator() = default;

    // ========================================================================
    // Grid Generation
    // ========================================================================

    /**
     * @brief Generate adaptive wavelength grid
     *
     * @param config Hyperspectral rendering configuration
     * @param analysis Spectral analysis results (critical wavelengths)
     * @return AdaptiveGridInfo with wavelengths and metadata
     */
    AdaptiveGridInfo Generate(
        const HyperspectralConfig& config,
        const SpectralAnalysisResult& analysis
    );

    /**
     * @brief Generate adaptive grid from explicit critical wavelengths
     *
     * @param wavelengthMin Start of range (nm)
     * @param wavelengthMax End of range (nm)
     * @param baseStep Base sampling interval (nm)
     * @param criticalWavelengths Wavelengths requiring dense sampling
     * @param criticalRadius Radius around critical wavelengths (nm)
     * @param coarseMultiplier Step multiplier for flat regions
     * @return AdaptiveGridInfo with wavelengths and metadata
     */
    AdaptiveGridInfo Generate(
        f32 wavelengthMin,
        f32 wavelengthMax,
        f32 baseStep,
        const Vector<f32>& criticalWavelengths,
        f32 criticalRadius,
        f32 coarseMultiplier
    );

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * @brief Generate uniform grid (for comparison or non-adaptive mode)
     *
     * @param config Hyperspectral configuration
     * @return AdaptiveGridInfo with uniform wavelengths
     */
    static AdaptiveGridInfo GenerateUniform(const HyperspectralConfig& config);

    /**
     * @brief Calculate expected compression ratio before generation
     *
     * @param wavelengthMin Start of range
     * @param wavelengthMax End of range
     * @param baseStep Base sampling interval
     * @param criticalWavelengths Critical wavelength list
     * @param criticalRadius Radius around critical wavelengths
     * @param coarseMultiplier Step multiplier for flat regions
     * @return Estimated compression ratio (uniform/adaptive)
     */
    static f32 EstimateCompressionRatio(
        f32 wavelengthMin,
        f32 wavelengthMax,
        f32 baseStep,
        const Vector<f32>& criticalWavelengths,
        f32 criticalRadius,
        f32 coarseMultiplier
    );

private:
    /**
     * @brief Check if wavelength is near any critical point
     */
    static bool IsNearCritical(
        f32 wavelength,
        const Vector<f32>& criticalWavelengths,
        f32 radius
    );

    /**
     * @brief Find nearest critical wavelength and distance
     */
    static f32 DistanceToNearestCritical(
        f32 wavelength,
        const Vector<f32>& criticalWavelengths
    );
};

// ============================================================================
// Reconstruction Info - For SpectralReconstructor
// ============================================================================

/**
 * @struct ReconstructionMapping
 * @brief Maps target wavelengths to source (rendered) wavelengths
 *
 * Used by SpectralReconstructor to rebuild the full spectrum from
 * the adaptively sampled wavelengths.
 */
struct ReconstructionMapping {
    u32 targetBand;        ///< Target band index in full cube
    f32 targetWavelength;  ///< Target wavelength (nm)
    bool wasRendered;      ///< True if this wavelength was actually rendered
    u32 sourceBand;        ///< Source band index (if rendered) or left neighbor
    u32 sourceNextBand;    ///< Right neighbor band index (for interpolation)
    f32 interpWeight;      ///< Interpolation weight [0,1] (0=use sourceBand)
};

/**
 * @brief Generate reconstruction mapping from adaptive to full grid
 *
 * @param adaptiveGrid The adaptive grid used for rendering
 * @param config Target full-resolution configuration
 * @return Vector of reconstruction mappings
 */
QL_API Vector<ReconstructionMapping> GenerateReconstructionMapping(
    const AdaptiveGridInfo& adaptiveGrid,
    const HyperspectralConfig& config
);

} // namespace quantiloom
