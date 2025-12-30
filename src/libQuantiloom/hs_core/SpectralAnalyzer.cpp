/**
 * @file SpectralAnalyzer.cpp
 * @brief Implementation of spectral curve analysis for adaptive sampling
 *
 * @author wtflmao
 */

#include "hs_core/SpectralAnalyzer.hpp"
#include "scene/Scene.hpp"
#include "scene/Material.hpp"
#include "core/SpectralData.hpp"
#include "core/Log.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace quantiloom {

// ============================================================================
// Utility Functions
// ============================================================================

const char* SpectralFeatureTypeToString(SpectralFeatureType type) {
    switch (type) {
        case SpectralFeatureType::Peak:        return "Peak";
        case SpectralFeatureType::Valley:      return "Valley";
        case SpectralFeatureType::RisingEdge:  return "RisingEdge";
        case SpectralFeatureType::FallingEdge: return "FallingEdge";
        case SpectralFeatureType::Inflection:  return "Inflection";
        default:                                return "Unknown";
    }
}

// Helper: Create SpectralCurve from Vector<pair<f32,f32>>
static SpectralCurve CreateCurveFromPairs(const Vector<std::pair<f32, f32>>& pairs) {
    SpectralCurve curve;
    curve.samples = pairs;
    return curve;
}

// ============================================================================
// SpectralAnalyzer - Scene Analysis
// ============================================================================

SpectralAnalysisResult SpectralAnalyzer::AnalyzeScene(
    const Scene& scene,
    f32 wavelengthMin,
    f32 wavelengthMax,
    const SpectralAnalysisParams& params
) {
    SpectralAnalysisResult result;
    result.analysisRange_min = wavelengthMin;
    result.analysisRange_max = wavelengthMax;

    Vector<Vector<SpectralFeature>> allFeatures;

    // Analyze all materials in scene
    const auto& materials = scene.materials;

    LOG_DEBUG("Analyzing {} materials for spectral features in range {}-{} nm",
              materials.size(), wavelengthMin, wavelengthMax);

    for (const auto& material : materials) {
        // Analyze IR reflectance curve if available
        if (!material.irReflectanceCurve.empty()) {
            SpectralCurve curve = CreateCurveFromPairs(material.irReflectanceCurve);
            auto features = AnalyzeCurve(curve, wavelengthMin, wavelengthMax, params);
            if (!features.empty()) {
                allFeatures.push_back(std::move(features));
            }
        }

        // Analyze IR emissivity curve (weighted higher for thermal IR)
        if (!material.irEmissivityCurve.empty()) {
            SpectralCurve curve = CreateCurveFromPairs(material.irEmissivityCurve);
            auto features = AnalyzeCurve(curve, wavelengthMin, wavelengthMax, params);
            // Apply emission weight
            for (auto& f : features) {
                f.importance *= params.emissionWeight;
            }
            if (!features.empty()) {
                allFeatures.push_back(std::move(features));
            }
        }

        // Analyze IR transmittance curve
        if (!material.irTransmittanceCurve.empty()) {
            SpectralCurve curve = CreateCurveFromPairs(material.irTransmittanceCurve);
            auto features = AnalyzeCurve(curve, wavelengthMin, wavelengthMax, params);
            if (!features.empty()) {
                allFeatures.push_back(std::move(features));
            }
        }
    }

    // Merge all features
    result.features = MergeFeatures(allFeatures, params.minFeatureSeparation_nm);

    // Extract critical wavelengths
    result.criticalWavelengths = ExtractCriticalWavelengths(result.features, 0.1f);

    LOG_INFO("Spectral analysis complete: {} features, {} critical wavelengths",
             result.features.size(), result.criticalWavelengths.size());

    return result;
}

// ============================================================================
// SpectralAnalyzer - Single Curve Analysis
// ============================================================================

Vector<SpectralFeature> SpectralAnalyzer::AnalyzeCurve(
    const SpectralCurve& curve,
    f32 wavelengthMin,
    f32 wavelengthMax,
    const SpectralAnalysisParams& params
) {
    // Resample curve to uniform grid for analysis
    const f32 sampleStep = 5.0f;  // 5nm sampling for analysis
    u32 numSamples = static_cast<u32>((wavelengthMax - wavelengthMin) / sampleStep) + 1;

    Vector<f32> wavelengths(numSamples);
    Vector<f32> values(numSamples);

    for (u32 i = 0; i < numSamples; ++i) {
        wavelengths[i] = wavelengthMin + i * sampleStep;
        values[i] = curve.Evaluate(wavelengths[i]);
    }

    return AnalyzeSampled(wavelengths, values, params);
}

Vector<SpectralFeature> SpectralAnalyzer::AnalyzeSampled(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    const SpectralAnalysisParams& params
) {
    if (wavelengths.size() < 5 || values.size() < 5) {
        return {};  // Not enough points for analysis
    }

    Vector<SpectralFeature> features;

    // Apply smoothing to reduce noise
    Vector<f32> smoothed = SmoothCurve(values, wavelengths, params.smoothingWindow_nm);

    // Compute derivatives
    Vector<f32> firstDeriv = ComputeFirstDerivative(wavelengths, smoothed);
    Vector<f32> secondDeriv = ComputeSecondDerivative(wavelengths, smoothed);

    // Detect peaks
    auto peaks = DetectPeaks(wavelengths, smoothed, firstDeriv, params.prominenceThreshold);
    features.insert(features.end(), peaks.begin(), peaks.end());

    // Detect valleys
    auto valleys = DetectValleys(wavelengths, smoothed, firstDeriv, params.prominenceThreshold);
    features.insert(features.end(), valleys.begin(), valleys.end());

    // Detect steep slopes
    auto slopes = DetectSlopes(wavelengths, smoothed, firstDeriv, params.slopeThreshold);
    features.insert(features.end(), slopes.begin(), slopes.end());

    // Sort by wavelength
    std::sort(features.begin(), features.end());

    return features;
}

// ============================================================================
// Feature Detection Helpers
// ============================================================================

Vector<f32> SpectralAnalyzer::ComputeFirstDerivative(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values
) {
    usize n = values.size();
    Vector<f32> derivative(n);

    // Forward difference for first point
    derivative[0] = (values[1] - values[0]) / (wavelengths[1] - wavelengths[0]);

    // Central differences for interior points
    for (usize i = 1; i < n - 1; ++i) {
        f32 dv = values[i + 1] - values[i - 1];
        f32 dw = wavelengths[i + 1] - wavelengths[i - 1];
        derivative[i] = dv / dw;
    }

    // Backward difference for last point
    derivative[n - 1] = (values[n - 1] - values[n - 2]) /
                        (wavelengths[n - 1] - wavelengths[n - 2]);

    return derivative;
}

Vector<f32> SpectralAnalyzer::ComputeSecondDerivative(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values
) {
    auto firstDeriv = ComputeFirstDerivative(wavelengths, values);
    return ComputeFirstDerivative(wavelengths, firstDeriv);
}

Vector<f32> SpectralAnalyzer::SmoothCurve(
    const Vector<f32>& values,
    const Vector<f32>& wavelengths,
    f32 windowSize_nm
) {
    usize n = values.size();
    Vector<f32> smoothed(n);

    for (usize i = 0; i < n; ++i) {
        f32 sum = 0.0f;
        f32 weightSum = 0.0f;
        f32 centerWl = wavelengths[i];

        for (usize j = 0; j < n; ++j) {
            f32 dist = std::abs(wavelengths[j] - centerWl);
            if (dist <= windowSize_nm) {
                // Gaussian weight
                f32 sigma = windowSize_nm / 3.0f;
                f32 weight = std::exp(-0.5f * (dist / sigma) * (dist / sigma));
                sum += values[j] * weight;
                weightSum += weight;
            }
        }

        smoothed[i] = (weightSum > 0) ? (sum / weightSum) : values[i];
    }

    return smoothed;
}

Vector<SpectralFeature> SpectralAnalyzer::DetectPeaks(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    const Vector<f32>& derivative,
    f32 prominenceThreshold
) {
    Vector<SpectralFeature> peaks;
    usize n = derivative.size();

    for (usize i = 1; i < n - 1; ++i) {
        // Zero-crossing from positive to negative = local maximum
        if (derivative[i - 1] > 0 && derivative[i + 1] < 0) {
            f32 prominence = CalculateProminence(values, i, true);

            if (prominence >= prominenceThreshold) {
                SpectralFeature feature;
                feature.wavelength_nm = wavelengths[i];
                feature.type = SpectralFeatureType::Peak;
                feature.amplitude = prominence;
                feature.width_nm = EstimateFeatureWidth(wavelengths, values, i);
                feature.importance = std::min(1.0f, prominence / 0.5f);  // Normalize

                peaks.push_back(feature);
            }
        }
    }

    return peaks;
}

Vector<SpectralFeature> SpectralAnalyzer::DetectValleys(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    const Vector<f32>& derivative,
    f32 prominenceThreshold
) {
    Vector<SpectralFeature> valleys;
    usize n = derivative.size();

    for (usize i = 1; i < n - 1; ++i) {
        // Zero-crossing from negative to positive = local minimum
        if (derivative[i - 1] < 0 && derivative[i + 1] > 0) {
            f32 prominence = CalculateProminence(values, i, false);

            if (prominence >= prominenceThreshold) {
                SpectralFeature feature;
                feature.wavelength_nm = wavelengths[i];
                feature.type = SpectralFeatureType::Valley;
                feature.amplitude = prominence;
                feature.width_nm = EstimateFeatureWidth(wavelengths, values, i);
                feature.importance = std::min(1.0f, prominence / 0.5f);

                valleys.push_back(feature);
            }
        }
    }

    return valleys;
}

Vector<SpectralFeature> SpectralAnalyzer::DetectSlopes(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    const Vector<f32>& derivative,
    f32 slopeThreshold
) {
    Vector<SpectralFeature> slopes;
    usize n = derivative.size();
    bool inSteepRegion = false;
    usize regionStart = 0;
    f32 maxSlope = 0.0f;
    usize maxSlopeIdx = 0;

    for (usize i = 0; i < n; ++i) {
        f32 absSlope = std::abs(derivative[i]);

        if (absSlope > slopeThreshold) {
            if (!inSteepRegion) {
                inSteepRegion = true;
                regionStart = i;
                maxSlope = absSlope;
                maxSlopeIdx = i;
            } else if (absSlope > maxSlope) {
                maxSlope = absSlope;
                maxSlopeIdx = i;
            }
        } else if (inSteepRegion) {
            // End of steep region - create feature at max slope point
            SpectralFeature feature;
            feature.wavelength_nm = wavelengths[maxSlopeIdx];
            feature.type = (derivative[maxSlopeIdx] > 0) ?
                           SpectralFeatureType::RisingEdge :
                           SpectralFeatureType::FallingEdge;
            feature.amplitude = maxSlope;
            feature.width_nm = wavelengths[i] - wavelengths[regionStart];
            feature.importance = std::min(1.0f, maxSlope / (slopeThreshold * 10.0f));

            slopes.push_back(feature);
            inSteepRegion = false;
        }
    }

    // Handle case where steep region extends to end
    if (inSteepRegion) {
        SpectralFeature feature;
        feature.wavelength_nm = wavelengths[maxSlopeIdx];
        feature.type = (derivative[maxSlopeIdx] > 0) ?
                       SpectralFeatureType::RisingEdge :
                       SpectralFeatureType::FallingEdge;
        feature.amplitude = maxSlope;
        feature.width_nm = wavelengths[n - 1] - wavelengths[regionStart];
        feature.importance = std::min(1.0f, maxSlope / (slopeThreshold * 10.0f));

        slopes.push_back(feature);
    }

    return slopes;
}

f32 SpectralAnalyzer::EstimateFeatureWidth(
    const Vector<f32>& wavelengths,
    const Vector<f32>& values,
    usize peakIndex
) {
    f32 peakValue = values[peakIndex];
    f32 halfMax = peakValue * 0.5f;

    // Find left half-max point
    usize leftIdx = peakIndex;
    while (leftIdx > 0 && values[leftIdx] > halfMax) {
        --leftIdx;
    }

    // Find right half-max point
    usize rightIdx = peakIndex;
    while (rightIdx < values.size() - 1 && values[rightIdx] > halfMax) {
        ++rightIdx;
    }

    return wavelengths[rightIdx] - wavelengths[leftIdx];
}

f32 SpectralAnalyzer::CalculateProminence(
    const Vector<f32>& values,
    usize peakIndex,
    bool isPeak
) {
    f32 peakValue = values[peakIndex];

    // Find local baseline (minimum of adjacent valleys for peak, max for valley)
    f32 leftBaseline = peakValue;
    f32 rightBaseline = peakValue;

    // Search left for baseline
    for (usize i = peakIndex; i > 0; --i) {
        if (isPeak) {
            if (values[i] < leftBaseline) {
                leftBaseline = values[i];
            }
            if (values[i] > peakValue) break;  // Found higher peak
        } else {
            if (values[i] > leftBaseline) {
                leftBaseline = values[i];
            }
            if (values[i] < peakValue) break;  // Found deeper valley
        }
    }

    // Search right for baseline
    for (usize i = peakIndex; i < values.size(); ++i) {
        if (isPeak) {
            if (values[i] < rightBaseline) {
                rightBaseline = values[i];
            }
            if (values[i] > peakValue) break;
        } else {
            if (values[i] > rightBaseline) {
                rightBaseline = values[i];
            }
            if (values[i] < peakValue) break;
        }
    }

    f32 baseline = isPeak ? std::max(leftBaseline, rightBaseline)
                          : std::min(leftBaseline, rightBaseline);

    return std::abs(peakValue - baseline);
}

// ============================================================================
// Feature Merging
// ============================================================================

Vector<SpectralFeature> SpectralAnalyzer::MergeFeatures(
    const Vector<Vector<SpectralFeature>>& featureLists,
    f32 minSeparation
) {
    // Collect all features
    Vector<SpectralFeature> allFeatures;
    for (const auto& list : featureLists) {
        allFeatures.insert(allFeatures.end(), list.begin(), list.end());
    }

    if (allFeatures.empty()) {
        return {};
    }

    // Sort by wavelength
    std::sort(allFeatures.begin(), allFeatures.end());

    // Merge nearby features
    Vector<SpectralFeature> merged;
    merged.push_back(allFeatures[0]);

    for (usize i = 1; i < allFeatures.size(); ++i) {
        SpectralFeature& last = merged.back();
        const SpectralFeature& current = allFeatures[i];

        if (current.wavelength_nm - last.wavelength_nm < minSeparation) {
            // Merge: keep the more important one
            if (current.importance > last.importance) {
                last = current;
            }
            // Update importance to max
            last.importance = std::max(last.importance, current.importance);
        } else {
            merged.push_back(current);
        }
    }

    return merged;
}

Vector<f32> SpectralAnalyzer::ExtractCriticalWavelengths(
    const Vector<SpectralFeature>& features,
    f32 importanceThreshold
) {
    Vector<f32> critical;
    critical.reserve(features.size());

    for (const auto& feature : features) {
        if (feature.importance >= importanceThreshold) {
            critical.push_back(feature.wavelength_nm);
        }
    }

    // Remove duplicates and sort
    std::sort(critical.begin(), critical.end());
    auto last = std::unique(critical.begin(), critical.end());
    critical.erase(last, critical.end());

    return critical;
}

} // namespace quantiloom
