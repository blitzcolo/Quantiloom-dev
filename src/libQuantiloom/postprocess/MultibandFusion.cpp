#include "postprocess/MultibandFusion.hpp"
#include "core/Log.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom {

// ============================================================================
// Main Interface
// ============================================================================

auto MultibandFusion::Fuse(const Image& vis, const Image& swir,
                            const Image& mwir, const FusionParams& params)
    -> Result<Image> {

    // Validate inputs
    if (!vis.IsValid() || !swir.IsValid() || !mwir.IsValid()) {
        return Result<Image>(Result<Image>::Err("Invalid input images"));
    }

    if (vis.width != swir.width || vis.width != mwir.width ||
        vis.height != swir.height || vis.height != mwir.height) {
        return Result<Image>(Result<Image>::Err("Image dimensions must match"));
    }

    if (vis.channels != swir.channels || vis.channels != mwir.channels) {
        return Result<Image>(Result<Image>::Err("Image channel counts must match"));
    }

    Log::Info("Multiband fusion: {}x{} VIS/SWIR/MWIR → {} mode",
              vis.width, vis.height,
              params.method == FusionMethod::LaplacianPyramid ? "Laplacian" :
              params.method == FusionMethod::WeightedAverage ? "Weighted" :
              params.method == FusionMethod::MaxResponse ? "MaxResponse" : "PseudoColor");

    // Make working copies
    Image visNorm = vis;
    Image swirNorm = swir;
    Image mwirNorm = mwir;

    // Normalize bands to [0, 1]
    if (params.autoNormalize) {
        auto [visMin, visMax] = AutoNormalizeBand(visNorm);
        auto [swirMin, swirMax] = AutoNormalizeBand(swirNorm);
        auto [mwirMin, mwirMax] = AutoNormalizeBand(mwirNorm);

        Log::Debug("Auto-normalize: VIS [{:.3e}, {:.3e}], SWIR [{:.3e}, {:.3e}], MWIR [{:.3e}, {:.3e}]",
                   visMin, visMax, swirMin, swirMax, mwirMin, mwirMax);
    }
    // Note: If autoNormalize is false, images are used as-is without normalization

    // Apply fusion algorithm
    Image fused;
    switch (params.method) {
        case FusionMethod::WeightedAverage:
            fused = WeightedAverage(visNorm, swirNorm, mwirNorm, params);
            break;
        case FusionMethod::LaplacianPyramid:
            fused = LaplacianPyramid(visNorm, swirNorm, mwirNorm, params.pyramidLevels);
            break;
        case FusionMethod::MaxResponse:
            fused = MaxResponse(visNorm, swirNorm, mwirNorm);
            break;
        case FusionMethod::PseudoColor:
            fused = PseudoColor(visNorm, swirNorm, mwirNorm);
            break;
        default:
            return Result<Image>(Result<Image>::Err("Unknown fusion method"));
    }

    // Add metadata
    fused.metadata["fusion_method"] = params.method == FusionMethod::LaplacianPyramid ? "LaplacianPyramid" :
                                      params.method == FusionMethod::WeightedAverage ? "WeightedAverage" :
                                      params.method == FusionMethod::MaxResponse ? "MaxResponse" : "PseudoColor";
    fused.metadata["pyramid_levels"] = std::to_string(params.pyramidLevels);

    Log::Info("Fusion complete: {} channels", fused.channels);

    return Result(std::move(fused));  // Implicit conversion to Result
}

// ============================================================================
// Normalization
// ============================================================================

auto MultibandFusion::NormalizeBand(Image& band, const f32 minVal, const f32 maxVal) -> void {
    const f32 range = maxVal - minVal;
    if (range <= 0.0f) return;

    for (auto& pixel : band.data) {
        pixel = (pixel - minVal) / range;
        pixel = std::clamp(pixel, 0.0f, 1.0f);
    }
}

auto MultibandFusion::AutoNormalizeBand(Image& band) -> std::pair<f32, f32> {
    if (band.data.empty()) return {0.0f, 1.0f};

    const f32 minVal = *std::ranges::min_element(band.data);
    const f32 maxVal = *std::ranges::max_element(band.data);

    NormalizeBand(band, minVal, maxVal);

    return {minVal, maxVal};
}

// ============================================================================
// Fusion Algorithms
// ============================================================================

auto MultibandFusion::WeightedAverage(const Image& vis, const Image& swir,
                                       const Image& mwir, const FusionParams& p) -> Image {
    Image result(vis.width, vis.height, 1);  // Grayscale output

    const f32 totalWeight = p.weightVis + p.weightSwir + p.weightMwir;

    for (u32 i = 0; i < vis.TotalElements(); ++i) {
        const f32 fused = (vis.data[i] * p.weightVis +
                           swir.data[i] * p.weightSwir +
                           mwir.data[i] * p.weightMwir) / totalWeight;
        result.data[i] = fused;
    }

    return result;
}

auto MultibandFusion::MaxResponse(const Image& vis, const Image& swir,
                                   const Image& mwir) -> Image {
    Image result(vis.width, vis.height, 1);

    for (u32 i = 0; i < vis.TotalElements(); ++i) {
        result.data[i] = std::max({vis.data[i], swir.data[i], mwir.data[i]});
    }

    return result;
}

auto MultibandFusion::PseudoColor(const Image& vis, const Image& swir,
                                   const Image& mwir) -> Image {
    Image result(vis.width, vis.height, 3);  // RGB output

    for (u32 y = 0; y < vis.height; ++y) {
        for (u32 x = 0; x < vis.width; ++x) {
            result(x, y, 0) = vis(x, y, 0);   // R = VIS
            result(x, y, 1) = swir(x, y, 0);  // G = SWIR
            result(x, y, 2) = mwir(x, y, 0);  // B = MWIR
        }
    }

    return result;
}

// ============================================================================
// Laplacian Pyramid Fusion
// ============================================================================

auto MultibandFusion::LaplacianPyramid(const Image& vis, const Image& swir,
                                        const Image& mwir, const u32 levels) -> Image {
    // Build Laplacian pyramids for each band
    auto L_vis = BuildLaplacianPyramid(vis, levels);
    auto L_swir = BuildLaplacianPyramid(swir, levels);
    auto L_mwir = BuildLaplacianPyramid(mwir, levels);

    // Fuse at each level (max absolute value selection)
    Vector<Image> L_fused;
    L_fused.reserve(levels);

    for (u32 i = 0; i < levels; ++i) {
        Image fused(L_vis[i].width, L_vis[i].height, 1);

        for (u32 j = 0; j < fused.TotalElements(); ++j) {
            // Select coefficient with maximum absolute value
            const f32 v1 = L_vis[i].data[j];
            const f32 v2 = L_swir[i].data[j];
            const f32 v3 = L_mwir[i].data[j];

            const f32 abs1 = std::abs(v1);
            const f32 abs2 = std::abs(v2);
            const f32 abs3 = std::abs(v3);

            if (abs1 >= abs2 && abs1 >= abs3) {
                fused.data[j] = v1;
            } else if (abs2 >= abs3) {
                fused.data[j] = v2;
            } else {
                fused.data[j] = v3;
            }
        }

        L_fused.push_back(std::move(fused));
    }

    // Collapse pyramid to get final image
    return CollapseLaplacianPyramid(L_fused);
}

// ============================================================================
// Pyramid Construction
// ============================================================================

auto MultibandFusion::BuildGaussianPyramid(const Image& img, const u32 levels) -> Vector<Image> {
    Vector<Image> pyramid;
    pyramid.reserve(levels);
    pyramid.push_back(img);

    for (u32 i = 1; i < levels; ++i) {
        pyramid.push_back(Downsample(pyramid.back()));
    }

    return pyramid;
}

auto MultibandFusion::BuildLaplacianPyramid(const Image& img, const u32 levels) -> Vector<Image> {
    auto gaussian = BuildGaussianPyramid(img, levels);
    Vector<Image> laplacian;
    laplacian.reserve(levels);

    for (u32 i = 0; i < levels - 1; ++i) {
        const Image& G_i = gaussian[i];
        const Image expanded = Upsample(gaussian[i + 1], G_i.width, G_i.height);

        Image L_i(G_i.width, G_i.height, 1);
        for (u32 j = 0; j < G_i.TotalElements(); ++j) {
            L_i.data[j] = G_i.data[j] - expanded.data[j];
        }

        laplacian.push_back(std::move(L_i));
    }

    // Top level is just the last Gaussian level
    laplacian.push_back(gaussian.back());

    return laplacian;
}

auto MultibandFusion::CollapseLaplacianPyramid(const Vector<Image>& pyramid) -> Image {
    if (pyramid.empty()) {
        return {};
    }

    // Start from top (smallest) level
    Image result = pyramid.back();

    // Iterate down the pyramid
    for (i32 i = static_cast<i32>(pyramid.size()) - 2; i >= 0; --i) {
        // Expand current result to match next level size
        result = Upsample(result, pyramid[i].width, pyramid[i].height);

        // Add Laplacian detail
        for (u32 j = 0; j < result.TotalElements(); ++j) {
            result.data[j] += pyramid[i].data[j];
        }
    }

    return result;
}

// ============================================================================
// Downsampling / Upsampling
// ============================================================================

auto MultibandFusion::Downsample(const Image& img) -> Image {
    const u32 newWidth = img.width / 2;
    const u32 newHeight = img.height / 2;

    Image result(newWidth, newHeight, img.channels);

    // Simple 2x2 average pooling
    for (u32 y = 0; y < newHeight; ++y) {
        for (u32 x = 0; x < newWidth; ++x) {
            for (u32 c = 0; c < img.channels; ++c) {
                const u32 x0 = x * 2;
                const u32 y0 = y * 2;
                const u32 x1 = std::min(x0 + 1, img.width - 1);
                const u32 y1 = std::min(y0 + 1, img.height - 1);

                const f32 avg = (img(x0, y0, c) + img(x1, y0, c) +
                                 img(x0, y1, c) + img(x1, y1, c)) * 0.25f;

                result(x, y, c) = avg;
            }
        }
    }

    return result;
}

auto MultibandFusion::Upsample(const Image& img, const u32 targetWidth,
                                const u32 targetHeight) -> Image {
    Image result(targetWidth, targetHeight, img.channels);

    // Bilinear interpolation
    const f32 xRatio = static_cast<f32>(img.width - 1) / static_cast<f32>(targetWidth - 1);
    const f32 yRatio = static_cast<f32>(img.height - 1) / static_cast<f32>(targetHeight - 1);

    for (u32 y = 0; y < targetHeight; ++y) {
        for (u32 x = 0; x < targetWidth; ++x) {
            const f32 srcX = static_cast<f32>(x) * xRatio;
            const f32 srcY = static_cast<f32>(y) * yRatio;

            const u32 x0 = static_cast<u32>(srcX);
            const u32 y0 = static_cast<u32>(srcY);
            const u32 x1 = std::min(x0 + 1, img.width - 1);
            const u32 y1 = std::min(y0 + 1, img.height - 1);

            const f32 fx = srcX - static_cast<f32>(x0);
            const f32 fy = srcY - static_cast<f32>(y0);

            for (u32 c = 0; c < img.channels; ++c) {
                const f32 v00 = img(x0, y0, c);
                const f32 v10 = img(x1, y0, c);
                const f32 v01 = img(x0, y1, c);
                const f32 v11 = img(x1, y1, c);

                const f32 v0 = v00 * (1.0f - fx) + v10 * fx;
                const f32 v1 = v01 * (1.0f - fx) + v11 * fx;

                result(x, y, c) = v0 * (1.0f - fy) + v1 * fy;
            }
        }
    }

    return result;
}

} // namespace quantiloom
