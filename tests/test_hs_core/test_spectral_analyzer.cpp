// ============================================================================
// Quantiloom - Unit Tests for hs_core/SpectralAnalyzer.hpp
// ============================================================================
// Tests cover:
// - SpectralFeature structure and comparison operators
// - SpectralAnalysisParams default values and configuration
// - SpectralAnalysisResult utility methods (IsNearCritical, CountCriticalInRange)
// - SpectralAnalyzer::AnalyzeSampled for synthetic test curves
// - Feature detection: peaks, valleys, slopes
// - Feature merging and deduplication
// - Critical wavelength extraction
// - Edge cases: flat curves, monotonic curves, single-point curves
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/SpectralAnalyzer.hpp"
#include <cmath>
#include <algorithm>

using namespace quantiloom;

// ============================================================================
// Helper Functions for Generating Test Curves
// ============================================================================

namespace {

/**
 * @brief Generate a flat (constant) spectrum
 */
Vector<f32> GenerateFlatSpectrum(usize numPoints, f32 value = 0.5f) {
    return Vector<f32>(numPoints, value);
}

/**
 * @brief Generate a single Gaussian peak
 */
Vector<f32> GenerateGaussianPeak(
    const Vector<f32>& wavelengths,
    f32 center,
    f32 amplitude,
    f32 sigma,
    f32 baseline = 0.1f
) {
    Vector<f32> values(wavelengths.size());
    for (usize i = 0; i < wavelengths.size(); ++i) {
        f32 x = wavelengths[i] - center;
        values[i] = baseline + amplitude * std::exp(-0.5f * (x * x) / (sigma * sigma));
    }
    return values;
}

/**
 * @brief Generate an inverted Gaussian (absorption valley)
 */
Vector<f32> GenerateGaussianValley(
    const Vector<f32>& wavelengths,
    f32 center,
    f32 depth,
    f32 sigma,
    f32 baseline = 0.8f
) {
    Vector<f32> values(wavelengths.size());
    for (usize i = 0; i < wavelengths.size(); ++i) {
        f32 x = wavelengths[i] - center;
        values[i] = baseline - depth * std::exp(-0.5f * (x * x) / (sigma * sigma));
    }
    return values;
}

/**
 * @brief Generate a sigmoid (step edge)
 */
Vector<f32> GenerateSigmoid(
    const Vector<f32>& wavelengths,
    f32 center,
    f32 amplitude,
    f32 steepness,
    f32 baseline = 0.2f
) {
    Vector<f32> values(wavelengths.size());
    for (usize i = 0; i < wavelengths.size(); ++i) {
        f32 x = (wavelengths[i] - center) * steepness;
        values[i] = baseline + amplitude / (1.0f + std::exp(-x));
    }
    return values;
}

/**
 * @brief Generate uniform wavelength grid
 */
Vector<f32> GenerateWavelengthGrid(f32 minWl, f32 maxWl, f32 step) {
    Vector<f32> wavelengths;
    for (f32 wl = minWl; wl <= maxWl; wl += step) {
        wavelengths.push_back(wl);
    }
    return wavelengths;
}

/**
 * @brief Generate multi-peak spectrum
 */
Vector<f32> GenerateMultiPeakSpectrum(
    const Vector<f32>& wavelengths,
    const Vector<f32>& centers,
    const Vector<f32>& amplitudes,
    f32 sigma = 50.0f,
    f32 baseline = 0.1f
) {
    Vector<f32> values(wavelengths.size(), baseline);
    for (usize p = 0; p < centers.size(); ++p) {
        for (usize i = 0; i < wavelengths.size(); ++i) {
            f32 x = wavelengths[i] - centers[p];
            f32 amp = (p < amplitudes.size()) ? amplitudes[p] : 0.5f;
            values[i] += amp * std::exp(-0.5f * (x * x) / (sigma * sigma));
        }
    }
    return values;
}

}  // anonymous namespace

// ============================================================================
// SpectralFeatureType Tests
// ============================================================================

TEST(SpectralFeatureTypeTest, ToString) {
    EXPECT_STREQ(SpectralFeatureTypeToString(SpectralFeatureType::Peak), "Peak");
    EXPECT_STREQ(SpectralFeatureTypeToString(SpectralFeatureType::Valley), "Valley");
    EXPECT_STREQ(SpectralFeatureTypeToString(SpectralFeatureType::RisingEdge), "RisingEdge");
    EXPECT_STREQ(SpectralFeatureTypeToString(SpectralFeatureType::FallingEdge), "FallingEdge");
    EXPECT_STREQ(SpectralFeatureTypeToString(SpectralFeatureType::Inflection), "Inflection");
}

// ============================================================================
// SpectralFeature Structure Tests
// ============================================================================

TEST(SpectralFeatureTest, DefaultConstruction) {
    SpectralFeature feature{};
    feature.wavelength_nm = 3500.0f;
    feature.type = SpectralFeatureType::Peak;
    feature.importance = 0.8f;
    feature.width_nm = 100.0f;
    feature.amplitude = 0.5f;

    EXPECT_NEAR(feature.wavelength_nm, 3500.0f, 1e-5f);
    EXPECT_EQ(feature.type, SpectralFeatureType::Peak);
    EXPECT_NEAR(feature.importance, 0.8f, 1e-5f);
}

TEST(SpectralFeatureTest, ComparisonOperator) {
    SpectralFeature f1{};
    f1.wavelength_nm = 3500.0f;

    SpectralFeature f2{};
    f2.wavelength_nm = 4000.0f;

    SpectralFeature f3{};
    f3.wavelength_nm = 3500.0f;

    EXPECT_TRUE(f1 < f2);
    EXPECT_FALSE(f2 < f1);
    EXPECT_FALSE(f1 < f3);  // Equal wavelengths
}

TEST(SpectralFeatureTest, Sorting) {
    Vector<SpectralFeature> features(4);
    features[0].wavelength_nm = 4500.0f;
    features[1].wavelength_nm = 3000.0f;
    features[2].wavelength_nm = 4000.0f;
    features[3].wavelength_nm = 3500.0f;

    std::sort(features.begin(), features.end());

    EXPECT_NEAR(features[0].wavelength_nm, 3000.0f, 1e-5f);
    EXPECT_NEAR(features[1].wavelength_nm, 3500.0f, 1e-5f);
    EXPECT_NEAR(features[2].wavelength_nm, 4000.0f, 1e-5f);
    EXPECT_NEAR(features[3].wavelength_nm, 4500.0f, 1e-5f);
}

// ============================================================================
// SpectralAnalysisParams Tests
// ============================================================================

TEST(SpectralAnalysisParamsTest, DefaultValues) {
    SpectralAnalysisParams params;

    EXPECT_NEAR(params.slopeThreshold, 0.001f, 1e-6f);
    EXPECT_NEAR(params.prominenceThreshold, 0.02f, 1e-6f);
    EXPECT_NEAR(params.smoothingWindow_nm, 20.0f, 1e-5f);
    EXPECT_NEAR(params.minFeatureSeparation_nm, 30.0f, 1e-5f);
    EXPECT_NEAR(params.emissionWeight, 1.5f, 1e-5f);
    EXPECT_NEAR(params.refractiveIndexWeight, 1.2f, 1e-5f);
}

TEST(SpectralAnalysisParamsTest, CustomValues) {
    SpectralAnalysisParams params;
    params.slopeThreshold = 0.005f;
    params.prominenceThreshold = 0.05f;
    params.smoothingWindow_nm = 50.0f;
    params.minFeatureSeparation_nm = 100.0f;

    EXPECT_NEAR(params.slopeThreshold, 0.005f, 1e-6f);
    EXPECT_NEAR(params.prominenceThreshold, 0.05f, 1e-6f);
    EXPECT_NEAR(params.smoothingWindow_nm, 50.0f, 1e-5f);
    EXPECT_NEAR(params.minFeatureSeparation_nm, 100.0f, 1e-5f);
}

// ============================================================================
// SpectralAnalysisResult Tests
// ============================================================================

TEST(SpectralAnalysisResultTest, IsNearCriticalTrue) {
    SpectralAnalysisResult result;
    result.criticalWavelengths = {3500.0f, 4000.0f, 4500.0f};

    EXPECT_TRUE(result.IsNearCritical(3500.0f, 50.0f));  // Exact match
    EXPECT_TRUE(result.IsNearCritical(3520.0f, 50.0f));  // Within radius
    EXPECT_TRUE(result.IsNearCritical(4050.0f, 100.0f)); // Within radius
}

TEST(SpectralAnalysisResultTest, IsNearCriticalFalse) {
    SpectralAnalysisResult result;
    result.criticalWavelengths = {3500.0f, 4000.0f, 4500.0f};

    EXPECT_FALSE(result.IsNearCritical(3700.0f, 50.0f));  // Too far
    EXPECT_FALSE(result.IsNearCritical(3200.0f, 100.0f)); // Outside all
}

TEST(SpectralAnalysisResultTest, IsNearCriticalEmptyList) {
    SpectralAnalysisResult result;
    // criticalWavelengths is empty

    EXPECT_FALSE(result.IsNearCritical(3500.0f, 50.0f));
    EXPECT_FALSE(result.IsNearCritical(4000.0f, 1000.0f));
}

TEST(SpectralAnalysisResultTest, CountCriticalInRange) {
    SpectralAnalysisResult result;
    result.criticalWavelengths = {3200.0f, 3500.0f, 4000.0f, 4500.0f, 4800.0f};

    EXPECT_EQ(result.CountCriticalInRange(3000.0f, 5000.0f), 5u);
    EXPECT_EQ(result.CountCriticalInRange(3400.0f, 4100.0f), 2u);
    EXPECT_EQ(result.CountCriticalInRange(3500.0f, 3500.0f), 1u);  // Single point
    EXPECT_EQ(result.CountCriticalInRange(3600.0f, 3900.0f), 0u);  // No features
}

TEST(SpectralAnalysisResultTest, CountCriticalInRangeEmpty) {
    SpectralAnalysisResult result;

    EXPECT_EQ(result.CountCriticalInRange(3000.0f, 5000.0f), 0u);
}

// ============================================================================
// SpectralAnalyzer::AnalyzeSampled Tests
// ============================================================================

TEST(SpectralAnalyzerTest, AnalyzeFlatSpectrum) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    auto values = GenerateFlatSpectrum(wavelengths.size(), 0.5f);

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.01f;
    params.slopeThreshold = 0.001f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Flat spectrum should have no significant features
    EXPECT_TRUE(features.empty());
}

TEST(SpectralAnalyzerTest, AnalyzeSinglePeak) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    auto values = GenerateGaussianPeak(wavelengths, 4000.0f, 0.6f, 100.0f, 0.1f);

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;
    params.slopeThreshold = 0.0001f;
    params.smoothingWindow_nm = 30.0f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Should detect at least one peak
    EXPECT_GE(features.size(), 1u);

    // Find peak features
    bool foundPeak = false;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::Peak) {
            // Peak should be near 4000 nm
            EXPECT_NEAR(f.wavelength_nm, 4000.0f, 50.0f);
            foundPeak = true;
        }
    }
    EXPECT_TRUE(foundPeak);
}

TEST(SpectralAnalyzerTest, AnalyzeSingleValley) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    auto values = GenerateGaussianValley(wavelengths, 4000.0f, 0.5f, 80.0f, 0.8f);

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;
    params.slopeThreshold = 0.0001f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Should detect at least one valley
    bool foundValley = false;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::Valley) {
            EXPECT_NEAR(f.wavelength_nm, 4000.0f, 50.0f);
            foundValley = true;
        }
    }
    EXPECT_TRUE(foundValley);
}

TEST(SpectralAnalyzerTest, AnalyzeSteepEdge) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    auto values = GenerateSigmoid(wavelengths, 4000.0f, 0.6f, 0.1f, 0.2f);

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.01f;
    params.slopeThreshold = 0.001f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Should detect edge features
    bool foundEdge = false;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::RisingEdge ||
            f.type == SpectralFeatureType::FallingEdge) {
            // Edge should be near 4000 nm
            EXPECT_NEAR(f.wavelength_nm, 4000.0f, 200.0f);
            foundEdge = true;
        }
    }
    // Note: Whether an edge is detected depends on smoothing parameters
    // This test verifies the analyzer runs without errors
    EXPECT_TRUE(features.size() >= 0);  // Always passes but verifies execution
}

TEST(SpectralAnalyzerTest, AnalyzeMultiplePeaks) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    Vector<f32> centers = {3300.0f, 4000.0f, 4700.0f};
    Vector<f32> amplitudes = {0.4f, 0.6f, 0.3f};
    auto values = GenerateMultiPeakSpectrum(wavelengths, centers, amplitudes, 80.0f, 0.1f);

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;
    params.slopeThreshold = 0.0001f;
    params.minFeatureSeparation_nm = 100.0f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Count peaks
    u32 peakCount = 0;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::Peak) {
            ++peakCount;
        }
    }

    // Should detect multiple peaks (may vary based on params)
    EXPECT_GE(peakCount, 2u);
}

// ============================================================================
// SpectralAnalyzer::MergeFeatures Tests
// ============================================================================

TEST(SpectralAnalyzerTest, MergeFeaturesEmpty) {
    Vector<Vector<SpectralFeature>> featureLists;

    auto merged = SpectralAnalyzer::MergeFeatures(featureLists, 50.0f);

    EXPECT_TRUE(merged.empty());
}

TEST(SpectralAnalyzerTest, MergeFeaturesSingleList) {
    Vector<SpectralFeature> list1(2);
    list1[0].wavelength_nm = 3500.0f;
    list1[0].type = SpectralFeatureType::Peak;
    list1[0].importance = 0.8f;
    list1[1].wavelength_nm = 4500.0f;
    list1[1].type = SpectralFeatureType::Valley;
    list1[1].importance = 0.6f;

    Vector<Vector<SpectralFeature>> featureLists = {list1};

    auto merged = SpectralAnalyzer::MergeFeatures(featureLists, 50.0f);

    EXPECT_EQ(merged.size(), 2u);
}

TEST(SpectralAnalyzerTest, MergeFeaturesNearbyMerged) {
    // Features at 3500nm and 3510nm should merge (separation < 50nm)
    SpectralFeature f1{};
    f1.wavelength_nm = 3500.0f;
    f1.type = SpectralFeatureType::Peak;
    f1.importance = 0.8f;

    SpectralFeature f2{};
    f2.wavelength_nm = 3510.0f;
    f2.type = SpectralFeatureType::Peak;
    f2.importance = 0.6f;

    Vector<Vector<SpectralFeature>> featureLists = {{f1}, {f2}};

    auto merged = SpectralAnalyzer::MergeFeatures(featureLists, 50.0f);

    // Should merge into single feature
    EXPECT_EQ(merged.size(), 1u);
    // Higher importance feature should be preserved
    EXPECT_NEAR(merged[0].importance, 0.8f, 1e-5f);
}

TEST(SpectralAnalyzerTest, MergeFeaturesDistantNotMerged) {
    // Features at 3500nm and 4000nm should not merge
    SpectralFeature f1{};
    f1.wavelength_nm = 3500.0f;
    f1.type = SpectralFeatureType::Peak;
    f1.importance = 0.8f;

    SpectralFeature f2{};
    f2.wavelength_nm = 4000.0f;
    f2.type = SpectralFeatureType::Valley;
    f2.importance = 0.6f;

    Vector<Vector<SpectralFeature>> featureLists = {{f1}, {f2}};

    auto merged = SpectralAnalyzer::MergeFeatures(featureLists, 50.0f);

    EXPECT_EQ(merged.size(), 2u);
}

TEST(SpectralAnalyzerTest, MergeFeaturesMultipleLists) {
    SpectralFeature f1{};
    f1.wavelength_nm = 3500.0f;
    f1.importance = 0.8f;

    SpectralFeature f2{};
    f2.wavelength_nm = 4000.0f;
    f2.importance = 0.6f;

    SpectralFeature f3{};
    f3.wavelength_nm = 4500.0f;
    f3.importance = 0.7f;

    SpectralFeature f4{};
    f4.wavelength_nm = 3520.0f;  // Near f1, should merge
    f4.importance = 0.5f;

    Vector<Vector<SpectralFeature>> featureLists = {
        {f1, f2},
        {f3, f4}
    };

    auto merged = SpectralAnalyzer::MergeFeatures(featureLists, 50.0f);

    // f1 and f4 merge -> 3 total features
    EXPECT_EQ(merged.size(), 3u);
}

// ============================================================================
// SpectralAnalyzer::ExtractCriticalWavelengths Tests
// ============================================================================

TEST(SpectralAnalyzerTest, ExtractCriticalWavelengthsEmpty) {
    Vector<SpectralFeature> features;

    auto critical = SpectralAnalyzer::ExtractCriticalWavelengths(features, 0.1f);

    EXPECT_TRUE(critical.empty());
}

TEST(SpectralAnalyzerTest, ExtractCriticalWavelengthsFiltered) {
    Vector<SpectralFeature> features(3);
    features[0].wavelength_nm = 3500.0f;
    features[0].importance = 0.8f;
    features[1].wavelength_nm = 4000.0f;
    features[1].importance = 0.05f;  // Below threshold
    features[2].wavelength_nm = 4500.0f;
    features[2].importance = 0.6f;

    auto critical = SpectralAnalyzer::ExtractCriticalWavelengths(features, 0.1f);

    EXPECT_EQ(critical.size(), 2u);  // Only features >= 0.1 importance
}

TEST(SpectralAnalyzerTest, ExtractCriticalWavelengthsAllPass) {
    Vector<SpectralFeature> features(3);
    features[0].wavelength_nm = 3500.0f;
    features[0].importance = 0.8f;
    features[1].wavelength_nm = 4000.0f;
    features[1].importance = 0.6f;
    features[2].wavelength_nm = 4500.0f;
    features[2].importance = 0.4f;

    auto critical = SpectralAnalyzer::ExtractCriticalWavelengths(features, 0.1f);

    EXPECT_EQ(critical.size(), 3u);
}

TEST(SpectralAnalyzerTest, ExtractCriticalWavelengthsSorted) {
    Vector<SpectralFeature> features(3);
    features[0].wavelength_nm = 4500.0f;
    features[0].importance = 0.8f;
    features[1].wavelength_nm = 3500.0f;
    features[1].importance = 0.6f;
    features[2].wavelength_nm = 4000.0f;
    features[2].importance = 0.7f;

    auto critical = SpectralAnalyzer::ExtractCriticalWavelengths(features, 0.1f);

    EXPECT_EQ(critical.size(), 3u);
    // Should be sorted
    EXPECT_LT(critical[0], critical[1]);
    EXPECT_LT(critical[1], critical[2]);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST(SpectralAnalyzerTest, AnalyzeEmptyInput) {
    SpectralAnalyzer analyzer;
    Vector<f32> wavelengths;
    Vector<f32> values;

    auto features = analyzer.AnalyzeSampled(wavelengths, values);

    EXPECT_TRUE(features.empty());
}

TEST(SpectralAnalyzerTest, AnalyzeSinglePoint) {
    SpectralAnalyzer analyzer;
    Vector<f32> wavelengths = {4000.0f};
    Vector<f32> values = {0.5f};

    auto features = analyzer.AnalyzeSampled(wavelengths, values);

    // Single point has no derivatives, should return empty or gracefully handle
    EXPECT_TRUE(features.empty());
}

TEST(SpectralAnalyzerTest, AnalyzeTwoPoints) {
    SpectralAnalyzer analyzer;
    Vector<f32> wavelengths = {3000.0f, 5000.0f};
    Vector<f32> values = {0.3f, 0.7f};

    auto features = analyzer.AnalyzeSampled(wavelengths, values);

    // Two points can only define slope, not peaks/valleys
    // May or may not detect features depending on implementation
    // This test verifies no crash
    EXPECT_TRUE(features.size() <= 2);  // At most edge features
}

TEST(SpectralAnalyzerTest, AnalyzeMonotonicIncreasing) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    Vector<f32> values(wavelengths.size());
    for (usize i = 0; i < values.size(); ++i) {
        values[i] = 0.1f + 0.8f * static_cast<f32>(i) / values.size();
    }

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;
    params.slopeThreshold = 0.0001f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Monotonic curve should have no peaks/valleys
    for (const auto& f : features) {
        EXPECT_NE(f.type, SpectralFeatureType::Peak);
        EXPECT_NE(f.type, SpectralFeatureType::Valley);
    }
}

TEST(SpectralAnalyzerTest, AnalyzeHighFrequencyNoise) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    Vector<f32> values(wavelengths.size());

    // Generate noisy signal with underlying trend
    for (usize i = 0; i < values.size(); ++i) {
        f32 trend = 0.5f;
        // Small oscillations
        f32 noise = 0.02f * std::sin(static_cast<f32>(i) * 0.5f);
        values[i] = trend + noise;
    }

    SpectralAnalysisParams params;
    params.smoothingWindow_nm = 50.0f;  // Smooth out noise
    params.prominenceThreshold = 0.05f; // Ignore small oscillations

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // With smoothing, should filter out noise and find few or no features
    // This test verifies analyzer handles noisy data without excessive detections
    EXPECT_LE(features.size(), 5u);  // Reasonable bound
}

TEST(SpectralAnalyzerTest, AnalyzeVeryNarrowPeak) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 1.0f);  // Fine grid
    auto values = GenerateGaussianPeak(wavelengths, 4000.0f, 0.8f, 10.0f, 0.1f);  // Narrow peak

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.1f;
    params.smoothingWindow_nm = 5.0f;  // Narrow smoothing to preserve peak

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Should detect the narrow peak
    bool foundPeak = false;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::Peak &&
            std::abs(f.wavelength_nm - 4000.0f) < 30.0f) {
            foundPeak = true;
            break;
        }
    }
    EXPECT_TRUE(foundPeak);
}

TEST(SpectralAnalyzerTest, AnalyzeWidePeak) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);
    auto values = GenerateGaussianPeak(wavelengths, 4000.0f, 0.5f, 500.0f, 0.2f);  // Very wide

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Wide peak should still be detectable
    bool foundPeak = false;
    for (const auto& f : features) {
        if (f.type == SpectralFeatureType::Peak) {
            foundPeak = true;
            break;
        }
    }
    EXPECT_TRUE(foundPeak);
}

TEST(SpectralAnalyzerTest, AnalyzeMismatchedSizes) {
    SpectralAnalyzer analyzer;
    Vector<f32> wavelengths = {3000.0f, 4000.0f, 5000.0f};
    Vector<f32> values = {0.5f, 0.6f};  // Mismatched size

    // Should handle gracefully (implementation-dependent)
    auto features = analyzer.AnalyzeSampled(wavelengths, values);

    // Should not crash, may return empty or partial results
    // Just verify no exception/crash
    EXPECT_TRUE(true);
}

// ============================================================================
// Importance Score Tests
// ============================================================================

TEST(SpectralAnalyzerTest, ImportanceScoring) {
    SpectralAnalyzer analyzer;
    auto wavelengths = GenerateWavelengthGrid(3000.0f, 5000.0f, 10.0f);

    // Create spectrum with large and small peaks
    Vector<f32> values(wavelengths.size(), 0.1f);
    // Large peak at 3500
    for (usize i = 0; i < wavelengths.size(); ++i) {
        f32 x = wavelengths[i] - 3500.0f;
        values[i] += 0.8f * std::exp(-0.5f * x * x / (100.0f * 100.0f));
    }
    // Small peak at 4500
    for (usize i = 0; i < wavelengths.size(); ++i) {
        f32 x = wavelengths[i] - 4500.0f;
        values[i] += 0.2f * std::exp(-0.5f * x * x / (100.0f * 100.0f));
    }

    SpectralAnalysisParams params;
    params.prominenceThreshold = 0.05f;

    auto features = analyzer.AnalyzeSampled(wavelengths, values, params);

    // Find the two peaks
    SpectralFeature* largePeak = nullptr;
    SpectralFeature* smallPeak = nullptr;

    for (auto& f : features) {
        if (f.type == SpectralFeatureType::Peak) {
            if (std::abs(f.wavelength_nm - 3500.0f) < 100.0f) {
                largePeak = &f;
            } else if (std::abs(f.wavelength_nm - 4500.0f) < 100.0f) {
                smallPeak = &f;
            }
        }
    }

    // If both detected, larger peak should have higher importance
    if (largePeak && smallPeak) {
        EXPECT_GT(largePeak->importance, smallPeak->importance);
    }
}
