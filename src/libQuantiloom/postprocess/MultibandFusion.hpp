/**
 * @file MultibandFusion.hpp
 * @brief Multi-band image fusion for enhanced visualization (VIS/SWIR/MWIR composite)
 *
 * Provides MultibandFusion class for combining multiple spectral bands into enhanced output:
 * - Fusion methods: Weighted average, Laplacian pyramid, max response, pseudo-color
 * - Automatic normalization per band (handles different radiance scales)
 * - Multi-scale processing (Laplacian pyramid preserves detail across scales)
 *
 * Fusion methods:
 * 1. WeightedAverage: Simple linear combination (fast, basic enhancement)
 * 2. LaplacianPyramid: Multi-scale decomposition (recommended, preserves detail)
 * 3. MaxResponse: Per-pixel maximum (highlights hot spots)
 * 4. PseudoColor: RGB false-color mapping (VIS→R, SWIR→G, MWIR→B)
 *
 * Typical use case:
 * Combine visible (VIS), short-wave IR (SWIR), and mid-wave IR (MWIR) bands
 * to create a single enhanced image with:
 * - VIS: Surface reflectance and color
 * - SWIR: Material discrimination (minerals, vegetation)
 * - MWIR: Thermal emission (hot objects, active targets)
 *
 * Usage example:
 * @code
 * // Load three bands
 * Image vis = ImageIO::ReadEXR("vis_550nm.exr").value();
 * Image swir = ImageIO::ReadEXR("swir_1650nm.exr").value();
 * Image mwir = ImageIO::ReadEXR("mwir_3900nm.exr").value();
 *
 * // Configure fusion
 * FusionParams params;
 * params.method = FusionMethod::LaplacianPyramid;
 * params.pyramidLevels = 5;
 * params.autoNormalize = true;
 *
 * // Fuse bands
 * auto result = MultibandFusion::Fuse(vis, swir, mwir, params);
 * if (result.has_value()) {
 *     Image fused = result.value();
 *     ImageIO::WritePNG("fused.png", fused);
 * }
 * @endcode
 *
 * @note All input images must have same width/height (validated at runtime)
 * @note Laplacian pyramid recommended for best detail preservation
 * @note Auto-normalization uses min/max per band (can be overridden)
 *
 * @see FusionMethod for algorithm selection
 * @see FusionParams for configuration parameters
 *
 * @author wtflmao
 */

#pragma once

#include "../core/Image.hpp"
#include "../core/Types.hpp"
#include "../core/Platform.hpp"

namespace quantiloom {

/**
 * @enum FusionMethod
 * @brief Multi-band fusion algorithm selection
 *
 * Available fusion algorithms:
 * - WeightedAverage: Linear combination with user-specified weights
 * - LaplacianPyramid: Multi-scale decomposition (preserves fine details)
 * - MaxResponse: Per-pixel maximum across bands (highlights hotspots)
 * - PseudoColor: False-color RGB mapping (VIS→R, SWIR→G, MWIR→B)
 */
enum class FusionMethod {
    WeightedAverage,    // Simple weighted sum
    LaplacianPyramid,   // Multi-scale Laplacian pyramid (recommended)
    MaxResponse,        // Per-pixel maximum selection
    PseudoColor         // RGB false-color mapping (VIS→R, SWIR→G, MWIR→B)
};

/// Fusion parameters (from TOML config)
struct FusionParams {
    FusionMethod method = FusionMethod::LaplacianPyramid;
    u32 pyramidLevels = 5;

    // Weights for weighted average
    f32 weightVis = 0.4f;
    f32 weightSwir = 0.3f;
    f32 weightMwir = 0.3f;

    // Normalization
    bool autoNormalize = true;
    f32 visMin = 0.0f, visMax = 1.0f;
    f32 swirMin = 0.0f, swirMax = 1e-3f;
    f32 mwirMin = 0.0f, mwirMax = 1e-2f;
};

/// Multiband fusion - combine VIS/SWIR/MWIR into enhanced output
class QL_API MultibandFusion {
public:
    /// Fuse three bands into a single image
    /// @param vis VIS band image (visible)
    /// @param swir SWIR band image (short-wave IR)
    /// @param mwir MWIR band image (mid-wave IR)
    /// @param params Fusion parameters
    /// @return Fused image (grayscale or RGB) or error
    static auto Fuse(const Image& vis, const Image& swir, const Image& mwir,
                     const FusionParams& params) -> Result<Image, String>;

private:
    // Normalization
    static auto NormalizeBand(Image& band, f32 minVal, f32 maxVal) -> void;
    static auto AutoNormalizeBand(Image& band) -> std::pair<f32, f32>;

    // Fusion algorithms
    static auto WeightedAverage(const Image& vis, const Image& swir,
                                const Image& mwir, const FusionParams& p) -> Image;
    static auto LaplacianPyramid(const Image& vis, const Image& swir,
                                 const Image& mwir, u32 levels) -> Image;
    static auto MaxResponse(const Image& vis, const Image& swir,
                            const Image& mwir) -> Image;
    static auto PseudoColor(const Image& vis, const Image& swir,
                            const Image& mwir) -> Image;

    // Pyramid helpers
    static auto BuildGaussianPyramid(const Image& img, u32 levels) -> Vector<Image>;
    static auto BuildLaplacianPyramid(const Image& img, u32 levels) -> Vector<Image>;
    static auto CollapseLaplacianPyramid(const Vector<Image>& pyramid) -> Image;
    static auto Downsample(const Image& img) -> Image;
    static auto Upsample(const Image& img, u32 targetWidth, u32 targetHeight) -> Image;
};

} // namespace quantiloom
