/**
 * @file SpectralAnalyzer.hpp
 * @brief Spectral curve analysis for adaptive hyperspectral sampling
 *
 * SpectralAnalyzer detects important spectral features in material curves:
 * - Emission/absorption peaks (local maxima)
 * - Absorption valleys (local minima)
 * - Steep slopes (high first derivative regions)
 * - Inflection points (curvature changes)
 *
 * These features define "critical wavelengths" where full sampling is required.
 * Flat spectral regions between features can be sparsely sampled and interpolated.
 *
 * Physical significance:
 * - Peaks/valleys often correspond to molecular vibration modes, electronic
 *   transitions, or atmospheric absorption bands
 * - Missing these features would compromise quantitative spectral analysis
 * - Flat regions contain redundant information that can be reconstructed
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/SpectralData.hpp"
#include <vector>

namespace quantiloom {

// Forward declarations
class Scene;
class Material;

// ============================================================================
// SpectralFeature - Detected Feature Descriptor
// ============================================================================

/**
 * @enum SpectralFeatureType
 * @brief Type of spectral feature detected
 */
enum class SpectralFeatureType : u32 {
    Peak,           ///< Local maximum (emission peak)
    Valley,         ///< Local minimum (absorption valley)
    RisingEdge,     ///< Steep positive slope
    FallingEdge,    ///< Steep negative slope
    Inflection      ///< Curvature sign change
};

/**
 * @brief Convert feature type to string
 */
const char* SpectralFeatureTypeToString(SpectralFeatureType type);

/**
 * @struct SpectralFeature
 * @brief Descriptor for a detected spectral feature
 *
 * Contains location, type, and importance metrics for a spectral feature.
 * Used to determine critical wavelengths for adaptive sampling.
 */
struct SpectralFeature {
    f32 wavelength_nm;           ///< Feature center wavelength
    SpectralFeatureType type;    ///< Type of feature
    f32 importance;              ///< Importance score [0, 1] (higher = more critical)
    f32 width_nm;                ///< Estimated feature width (FWHM-like)
    f32 amplitude;               ///< Feature amplitude (peak-to-baseline)

    /**
     * @brief Compare features by wavelength (for sorting)
     */
    bool operator<(const SpectralFeature& other) const {
        return wavelength_nm < other.wavelength_nm;
    }
};

// ============================================================================
// SpectralAnalysisParams - Analysis Configuration
// ============================================================================

/**
 * @struct SpectralAnalysisParams
 * @brief Parameters controlling spectral feature detection
 */
struct SpectralAnalysisParams {
    /// Minimum first derivative magnitude to detect slope features (dR/dλ per nm)
    f32 slopeThreshold = 0.001f;

    /// Minimum peak/valley prominence (relative to local baseline)
    f32 prominenceThreshold = 0.02f;

    /// Smoothing window size for noise reduction (nm)
    f32 smoothingWindow_nm = 20.0f;

    /// Minimum feature separation (nm) - features closer are merged
    f32 minFeatureSeparation_nm = 30.0f;

    /// Weight for emission curves (often more important than reflectance)
    f32 emissionWeight = 1.5f;

    /// Weight for refractive index features (Fresnel effects)
    f32 refractiveIndexWeight = 1.2f;
};

// ============================================================================
// SpectralAnalysisResult - Analysis Output
// ============================================================================

/**
 * @struct SpectralAnalysisResult
 * @brief Complete analysis result for a scene's spectral content
 */
struct SpectralAnalysisResult {
    Vector<SpectralFeature> features;  ///< All detected features (sorted by wavelength)
    Vector<f32> criticalWavelengths;   ///< Unique wavelengths requiring dense sampling
    f32 analysisRange_min;             ///< Analyzed wavelength range start
    f32 analysisRange_max;             ///< Analyzed wavelength range end

    /**
     * @brief Check if a wavelength is near any critical feature
     * @param wavelength_nm Query wavelength
     * @param radius_nm Search radius
     * @return true if within radius of any critical wavelength
     */
    [[nodiscard]] bool IsNearCritical(f32 wavelength_nm, f32 radius_nm) const {
        for (f32 crit : criticalWavelengths) {
            if (std::abs(wavelength_nm - crit) <= radius_nm) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Get number of critical wavelengths in a range
     */
    [[nodiscard]] u32 CountCriticalInRange(f32 min_nm, f32 max_nm) const {
        u32 count = 0;
        for (f32 crit : criticalWavelengths) {
            if (crit >= min_nm && crit <= max_nm) {
                ++count;
            }
        }
        return count;
    }
};

// ============================================================================
// SpectralAnalyzer - Main Analysis Class
// ============================================================================

/**
 * @class SpectralAnalyzer
 * @brief Analyzes spectral curves to detect features for adaptive sampling
 *
 * SpectralAnalyzer examines all spectral curves in a scene (reflectance,
 * emissivity, refractive index) to identify wavelengths where accurate
 * sampling is critical for physical correctness.
 *
 * Algorithm overview:
 * 1. Collect all spectral curves from scene materials
 * 2. For each curve, compute first and second derivatives
 * 3. Detect peaks (sign change - to + in first derivative)
 * 4. Detect valleys (sign change + to - in first derivative)
 * 5. Detect steep slopes (|dR/dλ| > threshold)
 * 6. Merge nearby features from all curves
 * 7. Output unified critical wavelength list
 *
 * Example usage:
 * @code
 * SpectralAnalyzer analyzer;
 * SpectralAnalysisParams params;
 * params.slopeThreshold = 0.002f;  // Stricter threshold
 *
 * auto result = analyzer.AnalyzeScene(scene, 3000.0f, 5000.0f, params);
 *
 * std::cout << "Found " << result.criticalWavelengths.size()
 *           << " critical wavelengths" << std::endl;
 *
 * for (const auto& feature : result.features) {
 *     std::cout << feature.wavelength_nm << " nm: "
 *               << SpectralFeatureTypeToString(feature.type) << std::endl;
 * }
 * @endcode
 */
class SpectralAnalyzer {
public:
    SpectralAnalyzer() = default;
    ~SpectralAnalyzer() = default;

    // ========================================================================
    // Scene Analysis
    // ========================================================================

    /**
     * @brief Analyze all spectral curves in a scene
     *
     * Scans all materials in the scene and analyzes their spectral properties
     * (reflectance, emissivity, transmittance, refractive index) for features.
     *
     * @param scene Scene containing materials to analyze
     * @param wavelengthMin Analysis range start (nm)
     * @param wavelengthMax Analysis range end (nm)
     * @param params Analysis parameters
     * @return Analysis result with detected features and critical wavelengths
     */
    SpectralAnalysisResult AnalyzeScene(
        const Scene& scene,
        f32 wavelengthMin,
        f32 wavelengthMax,
        const SpectralAnalysisParams& params = SpectralAnalysisParams{}
    );

    // ========================================================================
    // Single Curve Analysis
    // ========================================================================

    /**
     * @brief Analyze a single spectral curve for features
     *
     * @param curve Spectral curve to analyze
     * @param wavelengthMin Analysis range start (nm)
     * @param wavelengthMax Analysis range end (nm)
     * @param params Analysis parameters
     * @return Vector of detected features
     */
    Vector<SpectralFeature> AnalyzeCurve(
        const SpectralCurve& curve,
        f32 wavelengthMin,
        f32 wavelengthMax,
        const SpectralAnalysisParams& params = SpectralAnalysisParams{}
    );

    /**
     * @brief Analyze a sampled spectrum (uniform wavelength grid)
     *
     * @param wavelengths Wavelength values (nm)
     * @param values Spectral values at each wavelength
     * @param params Analysis parameters
     * @return Vector of detected features
     */
    Vector<SpectralFeature> AnalyzeSampled(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        const SpectralAnalysisParams& params = SpectralAnalysisParams{}
    );

    // ========================================================================
    // Feature Merging
    // ========================================================================

    /**
     * @brief Merge features from multiple curves
     *
     * Combines feature lists, removing duplicates and merging nearby features.
     * The most important feature at each location is preserved.
     *
     * @param featureLists Vector of feature vectors (one per curve)
     * @param minSeparation Minimum separation between distinct features (nm)
     * @return Merged and deduplicated feature list
     */
    static Vector<SpectralFeature> MergeFeatures(
        const Vector<Vector<SpectralFeature>>& featureLists,
        f32 minSeparation
    );

    /**
     * @brief Extract critical wavelengths from feature list
     *
     * @param features Detected features
     * @param importanceThreshold Minimum importance to include (0-1)
     * @return Sorted vector of critical wavelengths
     */
    static Vector<f32> ExtractCriticalWavelengths(
        const Vector<SpectralFeature>& features,
        f32 importanceThreshold = 0.1f
    );

private:
    // Internal helpers

    /**
     * @brief Compute first derivative using central differences
     */
    static Vector<f32> ComputeFirstDerivative(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values
    );

    /**
     * @brief Compute second derivative
     */
    static Vector<f32> ComputeSecondDerivative(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values
    );

    /**
     * @brief Apply Gaussian smoothing to reduce noise
     */
    static Vector<f32> SmoothCurve(
        const Vector<f32>& values,
        const Vector<f32>& wavelengths,
        f32 windowSize_nm
    );

    /**
     * @brief Detect peaks (local maxima) in a curve
     */
    static Vector<SpectralFeature> DetectPeaks(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        const Vector<f32>& derivative,
        f32 prominenceThreshold
    );

    /**
     * @brief Detect valleys (local minima) in a curve
     */
    static Vector<SpectralFeature> DetectValleys(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        const Vector<f32>& derivative,
        f32 prominenceThreshold
    );

    /**
     * @brief Detect steep slope regions
     */
    static Vector<SpectralFeature> DetectSlopes(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        const Vector<f32>& derivative,
        f32 slopeThreshold
    );

    /**
     * @brief Estimate feature width (FWHM-like metric)
     */
    static f32 EstimateFeatureWidth(
        const Vector<f32>& wavelengths,
        const Vector<f32>& values,
        usize peakIndex
    );

    /**
     * @brief Calculate prominence of a peak/valley
     */
    static f32 CalculateProminence(
        const Vector<f32>& values,
        usize peakIndex,
        bool isPeak
    );
};

} // namespace quantiloom
