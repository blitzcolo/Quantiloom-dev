/**
 * @file SpectralReconstructor.cpp
 * @brief Implementation of spectral cube reconstruction from adaptive samples
 *
 * @author blitzcolo
 */

#include "hs_core/SpectralReconstructor.hpp"
#include "core/Log.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom {

// ============================================================================
// SpectralReconstructor - Main Reconstruction
// ============================================================================

SpectralCube SpectralReconstructor::Reconstruct(
    const SpectralCube& sparseCube,
    const AdaptiveGridInfo& gridInfo,
    const HyperspectralConfig& targetConfig,
    InterpolationMethod method
) {
    // Generate reconstruction mapping
    Vector<ReconstructionMapping> mapping = GenerateReconstructionMapping(
        gridInfo, targetConfig
    );

    return ReconstructWithMapping(sparseCube, mapping, targetConfig, method);
}

SpectralCube SpectralReconstructor::ReconstructWithMapping(
    const SpectralCube& sparseCube,
    const Vector<ReconstructionMapping>& mapping,
    const HyperspectralConfig& targetConfig,
    InterpolationMethod method
) {
    u32 width = sparseCube.width;
    u32 height = sparseCube.height;
    u32 targetBands = targetConfig.GetNumBands();

    // Create output cube
    SpectralCube fullCube(
        width, height, targetBands,
        targetConfig.wavelengthMin_nm,
        targetConfig.wavelengthMax_nm
    );

    // Copy metadata
    fullCube.metadata = sparseCube.metadata;
    fullCube.metadata["reconstruction_method"] =
        (method == InterpolationMethod::Linear) ? "linear" :
        (method == InterpolationMethod::CatmullRom) ? "catmull_rom" : "akima";

    // Get source wavelengths
    const Vector<f32>& srcWavelengths = sparseCube.wavelengths;
    u32 srcBands = sparseCube.nbands;

    LOG_INFO("Reconstructing {} target bands from {} source bands using {}",
             targetBands, srcBands,
             fullCube.metadata["reconstruction_method"]);

    // Process each pixel
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            // Get source spectrum for this pixel
            Vector<f32> srcSpectrum(srcBands);
            for (u32 b = 0; b < srcBands; ++b) {
                srcSpectrum[b] = sparseCube(x, y, b);
            }

            // Reconstruct target spectrum
            for (const auto& map : mapping) {
                f32 value;

                if (map.wasRendered) {
                    // Direct copy
                    value = srcSpectrum[map.sourceBand];
                } else {
                    // Interpolation needed
                    switch (method) {
                        case InterpolationMethod::Linear:
                            value = LinearInterp(
                                srcSpectrum[map.sourceBand],
                                srcSpectrum[map.sourceNextBand],
                                map.interpWeight
                            );
                            break;

                        case InterpolationMethod::CatmullRom: {
                            // Get 4 surrounding points for Catmull-Rom
                            usize idx0, idx1, idx2, idx3;
                            f32 t;
                            GetCubicIndices(srcWavelengths, map.targetWavelength,
                                            idx0, idx1, idx2, idx3, t);

                            value = CatmullRomInterp(
                                srcSpectrum[idx0],
                                srcSpectrum[idx1],
                                srcSpectrum[idx2],
                                srcSpectrum[idx3],
                                t
                            );
                            break;
                        }

                        case InterpolationMethod::Akima:
                            value = AkimaInterp(
                                srcWavelengths,
                                srcSpectrum,
                                map.targetWavelength
                            );
                            break;
                    }
                }

                fullCube(x, y, map.targetBand) = value;
            }
        }
    }

    LOG_INFO("Reconstruction complete: {} x {} x {} cube",
             width, height, targetBands);

    return fullCube;
}

Vector<f32> SpectralReconstructor::InterpolatePixelSpectrum(
    const SpectralCube& sparseCube,
    u32 x, u32 y,
    const Vector<f32>& targetWavelengths,
    InterpolationMethod method
) {
    const Vector<f32>& srcWavelengths = sparseCube.wavelengths;
    u32 srcBands = sparseCube.nbands;

    // Get source spectrum
    Vector<f32> srcSpectrum(srcBands);
    for (u32 b = 0; b < srcBands; ++b) {
        srcSpectrum[b] = sparseCube(x, y, b);
    }

    // Interpolate to target wavelengths
    Vector<f32> targetSpectrum(targetWavelengths.size());

    for (usize i = 0; i < targetWavelengths.size(); ++i) {
        f32 targetWl = targetWavelengths[i];

        switch (method) {
            case InterpolationMethod::Linear: {
                // Find surrounding indices
                usize idx = 0;
                while (idx < srcBands - 1 && srcWavelengths[idx + 1] < targetWl) {
                    ++idx;
                }

                if (idx >= srcBands - 1) {
                    targetSpectrum[i] = srcSpectrum.back();
                } else {
                    f32 t = (targetWl - srcWavelengths[idx]) /
                            (srcWavelengths[idx + 1] - srcWavelengths[idx]);
                    targetSpectrum[i] = LinearInterp(
                        srcSpectrum[idx], srcSpectrum[idx + 1], t
                    );
                }
                break;
            }

            case InterpolationMethod::CatmullRom: {
                usize idx0, idx1, idx2, idx3;
                f32 t;
                GetCubicIndices(srcWavelengths, targetWl, idx0, idx1, idx2, idx3, t);
                targetSpectrum[i] = CatmullRomInterp(
                    srcSpectrum[idx0], srcSpectrum[idx1],
                    srcSpectrum[idx2], srcSpectrum[idx3], t
                );
                break;
            }

            case InterpolationMethod::Akima:
                targetSpectrum[i] = AkimaInterp(srcWavelengths, srcSpectrum, targetWl);
                break;
        }
    }

    return targetSpectrum;
}

// ============================================================================
// Error Estimation
// ============================================================================

f32 SpectralReconstructor::EstimateMaxError(
    const AdaptiveGridInfo& gridInfo,
    const HyperspectralConfig& targetConfig
) {
    // Maximum error occurs in coarsely-sampled regions
    // For smooth functions, cubic interpolation error is O(h^4)
    // where h is the sample spacing

    f32 baseStep = targetConfig.wavelengthStep_nm;
    f32 coarseStep = baseStep * targetConfig.adaptiveCoarseMultiplier;

    // Estimate based on typical spectral curvature in flat regions
    // Assuming flat regions have |d²R/dλ²| < 1e-6 per nm²
    f32 maxCurvature = 1e-6f;
    f32 errorBound = maxCurvature * coarseStep * coarseStep / 8.0f;

    return errorBound;
}

// ============================================================================
// Interpolation Kernels
// ============================================================================

f32 SpectralReconstructor::LinearInterp(f32 v0, f32 v1, f32 t) {
    return v0 + t * (v1 - v0);
}

f32 SpectralReconstructor::CatmullRomInterp(f32 p0, f32 p1, f32 p2, f32 p3, f32 t) {
    // Catmull-Rom spline with tension = 0.5
    f32 t2 = t * t;
    f32 t3 = t2 * t;

    f32 a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    f32 a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    f32 a2 = -0.5f * p0 + 0.5f * p2;
    f32 a3 = p1;

    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

f32 SpectralReconstructor::AkimaInterp(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    f32 targetWavelength
) {
    usize n = wavelengths.size();
    if (n < 2) return values.empty() ? 0.0f : values[0];

    // Find interval
    usize i = 0;
    while (i < n - 1 && wavelengths[i + 1] < targetWavelength) {
        ++i;
    }

    if (i >= n - 1) {
        return values.back();
    }

    // Compute slopes
    Vector<f32> slopes(n - 1);
    for (usize j = 0; j < n - 1; ++j) {
        slopes[j] = (values[j + 1] - values[j]) /
                    (wavelengths[j + 1] - wavelengths[j]);
    }

    // Akima derivative at point i
    auto akimaDerivative = [&](usize idx) -> f32 {
        if (n < 3) return slopes[0];

        // Handle boundaries
        f32 m1, m2, m3, m4;
        if (idx == 0) {
            m1 = 3.0f * slopes[0] - 2.0f * slopes[1];
            m2 = 2.0f * slopes[0] - slopes[1];
            m3 = slopes[0];
            m4 = slopes[1];
        } else if (idx == 1) {
            m1 = 2.0f * slopes[0] - slopes[1];
            m2 = slopes[0];
            m3 = slopes[1];
            m4 = (n > 2) ? slopes[2] : slopes[1];
        } else if (idx >= n - 2) {
            m1 = slopes[n - 3];
            m2 = slopes[n - 2];
            m3 = 2.0f * slopes[n - 2] - slopes[n - 3];
            m4 = 3.0f * slopes[n - 2] - 2.0f * slopes[n - 3];
        } else if (idx == n - 2) {
            m1 = slopes[idx - 2];
            m2 = slopes[idx - 1];
            m3 = slopes[idx];
            m4 = 2.0f * slopes[idx] - slopes[idx - 1];
        } else {
            m1 = slopes[idx - 2];
            m2 = slopes[idx - 1];
            m3 = slopes[idx];
            m4 = slopes[idx + 1];
        }

        f32 w1 = std::abs(m4 - m3);
        f32 w2 = std::abs(m2 - m1);

        if (w1 + w2 < 1e-10f) {
            return 0.5f * (m2 + m3);
        }
        return (w1 * m2 + w2 * m3) / (w1 + w2);
    };

    // Hermite interpolation
    f32 h = wavelengths[i + 1] - wavelengths[i];
    f32 t = (targetWavelength - wavelengths[i]) / h;
    f32 t2 = t * t;
    f32 t3 = t2 * t;

    f32 d0 = akimaDerivative(i);
    f32 d1 = akimaDerivative(i + 1);

    f32 h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    f32 h10 = t3 - 2.0f * t2 + t;
    f32 h01 = -2.0f * t3 + 3.0f * t2;
    f32 h11 = t3 - t2;

    return h00 * values[i] + h10 * h * d0 + h01 * values[i + 1] + h11 * h * d1;
}

void SpectralReconstructor::GetCubicIndices(
    const Vector<f32>& wavelengths,
    f32 target,
    usize& idx0, usize& idx1, usize& idx2, usize& idx3,
    f32& t
) {
    usize n = wavelengths.size();

    // Find idx1 such that wavelengths[idx1] <= target < wavelengths[idx1+1]
    idx1 = 0;
    while (idx1 < n - 1 && wavelengths[idx1 + 1] < target) {
        ++idx1;
    }

    // Set idx2
    idx2 = (idx1 < n - 1) ? idx1 + 1 : idx1;

    // Set idx0 (one before idx1)
    idx0 = (idx1 > 0) ? idx1 - 1 : 0;

    // Set idx3 (one after idx2)
    idx3 = (idx2 < n - 1) ? idx2 + 1 : n - 1;

    // Compute interpolation parameter
    if (idx2 > idx1) {
        t = (target - wavelengths[idx1]) / (wavelengths[idx2] - wavelengths[idx1]);
    } else {
        t = 0.0f;
    }

    // Clamp t to [0, 1]
    t = std::clamp(t, 0.0f, 1.0f);
}

} // namespace quantiloom
