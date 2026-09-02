/**
 * @file HyperspectralConfig.hpp
 * @brief Configuration structures for hyperspectral rendering
 *
 * Provides HyperspectralConfig for specifying:
 * - Wavelength range and sampling parameters
 * - Output format and quality settings
 * - Adaptive sampling options (Phase 2+)
 *
 * Design philosophy:
 * - Simple POD structure for easy serialization
 * - Validation methods to catch configuration errors early
 * - Factory methods for common use cases (MWIR, LWIR, SWIR, etc.)
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include <string>

namespace quantiloom {

// ============================================================================
// Hyperspectral Output Format
// ============================================================================

/**
 * @enum HyperspectralOutputFormat
 * @brief Supported output formats for hyperspectral data cubes
 */
enum class HyperspectralOutputFormat : u32 {
    ENVI_BSQ = 0,   ///< ENVI Band Sequential (most common for remote sensing)
    ENVI_BIL,       ///< ENVI Band Interleaved by Line
    ENVI_BIP,       ///< ENVI Band Interleaved by Pixel
    GeoTIFF,        ///< Multi-band GeoTIFF (supports up to 65535 bands)
    EXR_Spectral    ///< OpenEXR, one channel per band (Fichet et al. 2021 layout)
};

// ============================================================================
// Adaptive Sampling Mode (Phase 2+)
// ============================================================================

/**
 * @enum AdaptiveSamplingMode
 * @brief Controls adaptive wavelength sampling strategy
 */
enum class AdaptiveSamplingMode : u32 {
    None = 0,       ///< No adaptive sampling, render all wavelengths
    Spectral,       ///< Adaptive based on spectral curve features
    Spatial,        ///< Adaptive based on per-pixel material complexity
    Full            ///< Combined spectral + spatial adaptive sampling
};

// ============================================================================
// HyperspectralConfig - Main Configuration Structure
// ============================================================================

/**
 * @struct HyperspectralConfig
 * @brief Configuration for hyperspectral rendering
 *
 * Specifies wavelength range, sampling parameters, and output settings
 * for multi-wavelength rendering.
 *
 * Example usage:
 * @code
 * // MWIR rendering: 3000-5000nm with 50nm spacing
 * auto config = HyperspectralConfig::MWIR(50.0f);
 * config.outputPath = "thermal_cube.hdr";
 * config.spp = 64;
 *
 * // Custom range
 * HyperspectralConfig custom;
 * custom.wavelengthMin_nm = 400.0f;
 * custom.wavelengthMax_nm = 2500.0f;
 * custom.wavelengthStep_nm = 10.0f;
 * @endcode
 */
struct HyperspectralConfig {
    // ========================================================================
    // Wavelength Range Parameters
    // ========================================================================

    f32 wavelengthMin_nm = 400.0f;   ///< Start wavelength (nm)
    f32 wavelengthMax_nm = 2500.0f;  ///< End wavelength (nm)
    f32 wavelengthStep_nm = 10.0f;   ///< Sampling interval (nm)

    // ========================================================================
    // Rendering Parameters
    // ========================================================================

    u32 spp = 16;                    ///< Samples per pixel per wavelength
    u32 maxBounces = 4;              ///< Maximum ray bounces (path depth)

    // ========================================================================
    // Output Parameters
    // ========================================================================

    String outputPath = "hyperspectral_cube";  ///< Output file path (without extension)
    HyperspectralOutputFormat outputFormat = HyperspectralOutputFormat::ENVI_BSQ;
    bool saveIntermediates = false;  ///< Save each wavelength as separate file

    // ========================================================================
    // Adaptive Sampling (Phase 2+)
    // ========================================================================

    AdaptiveSamplingMode adaptiveMode = AdaptiveSamplingMode::None;

    /// Derivative threshold for feature detection (dR/dλ per nm)
    /// Wavelengths where |dR/dλ| > threshold are considered "critical"
    f32 adaptiveDerivativeThreshold = 0.001f;

    /// Radius around critical wavelengths for dense sampling (nm)
    f32 adaptiveCriticalRadius_nm = 100.0f;

    /// Coarse sampling multiplier for flat regions (2x = half the wavelengths)
    f32 adaptiveCoarseMultiplier = 2.0f;

    // ========================================================================
    // GPU Acceleration (Phase 4)
    // ========================================================================

    /// Enable GPU-accelerated spectral reconstruction
    /// When enabled, uses compute shaders for parallel interpolation (10-40x faster)
    bool useGpuReconstruction = true;

    // ========================================================================
    // Computed Properties
    // ========================================================================

    /**
     * @brief Calculate number of wavelength bands
     * @return Number of bands to render
     */
    [[nodiscard]] u32 GetNumBands() const {
        if (wavelengthStep_nm <= 0.0f || wavelengthMax_nm <= wavelengthMin_nm) {
            return 0;
        }
        return static_cast<u32>((wavelengthMax_nm - wavelengthMin_nm) / wavelengthStep_nm) + 1;
    }

    /**
     * @brief Get wavelength at specific band index
     * @param bandIndex Band index (0-based)
     * @return Wavelength in nm
     */
    [[nodiscard]] f32 GetWavelength(u32 bandIndex) const {
        return wavelengthMin_nm + static_cast<f32>(bandIndex) * wavelengthStep_nm;
    }

    /**
     * @brief Validate configuration
     * @return true if configuration is valid
     */
    [[nodiscard]] bool IsValid() const {
        if (wavelengthMin_nm <= 0.0f) return false;
        if (wavelengthMax_nm <= wavelengthMin_nm) return false;
        if (wavelengthStep_nm <= 0.0f) return false;
        if (spp == 0) return false;
        if (GetNumBands() == 0) return false;
        return true;
    }

    /**
     * @brief Get validation error message
     * @return Error description or empty string if valid
     */
    [[nodiscard]] String GetValidationError() const {
        if (wavelengthMin_nm <= 0.0f) {
            return "wavelengthMin_nm must be positive";
        }
        if (wavelengthMax_nm <= wavelengthMin_nm) {
            return "wavelengthMax_nm must be greater than wavelengthMin_nm";
        }
        if (wavelengthStep_nm <= 0.0f) {
            return "wavelengthStep_nm must be positive";
        }
        if (spp == 0) {
            return "spp must be at least 1";
        }
        if (GetNumBands() == 0) {
            return "Configuration results in zero bands";
        }
        return "";
    }

    // ========================================================================
    // Factory Methods for Common Configurations
    // ========================================================================

    /**
     * @brief Create MWIR (Mid-Wave Infrared) configuration
     * @param step Wavelength step in nm (default: 50nm)
     * @return HyperspectralConfig for 3000-5000nm range
     */
    static HyperspectralConfig MWIR(f32 step = 50.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 3000.0f;
        config.wavelengthMax_nm = 5000.0f;
        config.wavelengthStep_nm = step;
        return config;
    }

    /**
     * @brief Create LWIR (Long-Wave Infrared) configuration
     * @param step Wavelength step in nm (default: 100nm)
     * @return HyperspectralConfig for 8000-12000nm range
     */
    static HyperspectralConfig LWIR(f32 step = 100.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 8000.0f;
        config.wavelengthMax_nm = 12000.0f;
        config.wavelengthStep_nm = step;
        return config;
    }

    /**
     * @brief Create SWIR (Short-Wave Infrared) configuration
     * @param step Wavelength step in nm (default: 20nm)
     * @return HyperspectralConfig for 1000-2500nm range
     */
    static HyperspectralConfig SWIR(f32 step = 20.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 1000.0f;
        config.wavelengthMax_nm = 2500.0f;
        config.wavelengthStep_nm = step;
        return config;
    }

    /**
     * @brief Create NIR (Near Infrared) configuration
     * @param step Wavelength step in nm (default: 10nm)
     * @return HyperspectralConfig for 780-1400nm range
     */
    static HyperspectralConfig NIR(f32 step = 10.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 780.0f;
        config.wavelengthMax_nm = 1400.0f;
        config.wavelengthStep_nm = step;
        return config;
    }

    /**
     * @brief Create VIS (Visible) configuration
     * @param step Wavelength step in nm (default: 5nm)
     * @return HyperspectralConfig for 380-780nm range
     */
    static HyperspectralConfig VIS(f32 step = 5.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 380.0f;
        config.wavelengthMax_nm = 780.0f;
        config.wavelengthStep_nm = step;
        return config;
    }

    /**
     * @brief Create VNIR (Visible + Near Infrared) configuration
     * @param step Wavelength step in nm (default: 10nm)
     * @return HyperspectralConfig for 400-2500nm range
     */
    static HyperspectralConfig VNIR(f32 step = 10.0f) {
        HyperspectralConfig config;
        config.wavelengthMin_nm = 400.0f;
        config.wavelengthMax_nm = 2500.0f;
        config.wavelengthStep_nm = step;
        return config;
    }
};

// ============================================================================
// HyperspectralProgress - Progress Callback Structure
// ============================================================================

/**
 * @struct HyperspectralProgress
 * @brief Progress information for hyperspectral rendering
 *
 * Passed to progress callbacks during rendering to report status.
 */
struct HyperspectralProgress {
    u32 currentBand = 0;        ///< Current band being rendered (0-indexed)
    u32 totalBands = 0;         ///< Total number of bands
    f32 currentWavelength_nm = 0.0f;  ///< Current wavelength in nm
    f32 elapsedSeconds = 0.0f;  ///< Time elapsed since start
    f32 estimatedTotalSeconds = 0.0f;  ///< Estimated total time

    /**
     * @brief Get progress as percentage [0, 100]
     */
    [[nodiscard]] f32 GetPercentage() const {
        if (totalBands == 0) return 0.0f;
        return 100.0f * static_cast<f32>(currentBand) / static_cast<f32>(totalBands);
    }

    /**
     * @brief Get estimated remaining time in seconds
     */
    [[nodiscard]] f32 GetRemainingSeconds() const {
        return estimatedTotalSeconds - elapsedSeconds;
    }
};

/// Progress callback function type
using HyperspectralProgressCallback = void(*)(const HyperspectralProgress& progress, void* userData);

} // namespace quantiloom
