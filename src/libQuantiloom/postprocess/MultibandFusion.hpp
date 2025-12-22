#pragma once

#include "../core/Image.hpp"
#include "../core/Types.hpp"

namespace quantiloom {

/// Fusion method selection
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
class MultibandFusion {
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
