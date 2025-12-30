// ============================================================================
// Quantiloom - Unit Tests for hs_core/AdaptiveGridGenerator.hpp
// ============================================================================
// Tests cover:
// - AdaptiveGridInfo structure and utility methods
// - ReconstructionMapping structure
// - AdaptiveGridGenerator::Generate with various configurations
// - AdaptiveGridGenerator::GenerateUniform
// - AdaptiveGridGenerator::EstimateCompressionRatio
// - GenerateReconstructionMapping utility function
// - Edge cases: empty critical lists, single wavelength, full coverage
// - Compression ratio verification
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/AdaptiveGridGenerator.hpp"
#include "hs_core/HyperspectralConfig.hpp"
#include <cmath>
#include <algorithm>

using namespace quantiloom;

// ============================================================================
// AdaptiveGridInfo Structure Tests
// ============================================================================

TEST(AdaptiveGridInfoTest, DefaultConstruction) {
    AdaptiveGridInfo info;

    EXPECT_TRUE(info.wavelengths.empty());
    EXPECT_TRUE(info.isCritical.empty());
    EXPECT_EQ(info.totalWavelengths, 0u);
    EXPECT_EQ(info.criticalWavelengths, 0u);
    EXPECT_EQ(info.coarseWavelengths, 0u);
    EXPECT_NEAR(info.compressionRatio, 0.0f, 1e-5f);
}

TEST(AdaptiveGridInfoTest, GetWavelengthValid) {
    AdaptiveGridInfo info;
    info.wavelengths = {3000.0f, 3100.0f, 3200.0f};

    EXPECT_NEAR(info.GetWavelength(0), 3000.0f, 1e-5f);
    EXPECT_NEAR(info.GetWavelength(1), 3100.0f, 1e-5f);
    EXPECT_NEAR(info.GetWavelength(2), 3200.0f, 1e-5f);
}

TEST(AdaptiveGridInfoTest, GetWavelengthOutOfBounds) {
    AdaptiveGridInfo info;
    info.wavelengths = {3000.0f, 3100.0f};

    EXPECT_NEAR(info.GetWavelength(5), 0.0f, 1e-5f);  // Returns 0 for invalid index
}

TEST(AdaptiveGridInfoTest, IsCriticalValid) {
    AdaptiveGridInfo info;
    info.isCritical = {true, false, true, false};

    EXPECT_TRUE(info.IsCritical(0));
    EXPECT_FALSE(info.IsCritical(1));
    EXPECT_TRUE(info.IsCritical(2));
    EXPECT_FALSE(info.IsCritical(3));
}

TEST(AdaptiveGridInfoTest, IsCriticalOutOfBounds) {
    AdaptiveGridInfo info;
    info.isCritical = {true, false};

    EXPECT_FALSE(info.IsCritical(5));  // Returns false for invalid index
}

// ============================================================================
// ReconstructionMapping Structure Tests
// ============================================================================

TEST(ReconstructionMappingTest, RenderedWavelength) {
    ReconstructionMapping map;
    map.targetBand = 10;
    map.targetWavelength = 4000.0f;
    map.wasRendered = true;
    map.sourceBand = 5;
    map.sourceNextBand = 5;
    map.interpWeight = 0.0f;

    EXPECT_EQ(map.targetBand, 10u);
    EXPECT_NEAR(map.targetWavelength, 4000.0f, 1e-5f);
    EXPECT_TRUE(map.wasRendered);
    EXPECT_EQ(map.sourceBand, 5u);
    EXPECT_NEAR(map.interpWeight, 0.0f, 1e-5f);
}

TEST(ReconstructionMappingTest, InterpolatedWavelength) {
    ReconstructionMapping map;
    map.targetBand = 15;
    map.targetWavelength = 4025.0f;
    map.wasRendered = false;
    map.sourceBand = 5;
    map.sourceNextBand = 6;
    map.interpWeight = 0.5f;

    EXPECT_FALSE(map.wasRendered);
    EXPECT_EQ(map.sourceBand, 5u);
    EXPECT_EQ(map.sourceNextBand, 6u);
    EXPECT_NEAR(map.interpWeight, 0.5f, 1e-5f);
}

// ============================================================================
// AdaptiveGridGenerator::GenerateUniform Tests
// ============================================================================

TEST(AdaptiveGridGeneratorTest, GenerateUniformMWIR) {
    auto config = HyperspectralConfig::MWIR(50.0f);  // 3000-5000nm, 50nm step

    auto grid = AdaptiveGridGenerator::GenerateUniform(config);

    // (5000 - 3000) / 50 + 1 = 41 bands
    EXPECT_EQ(grid.totalWavelengths, 41u);
    EXPECT_EQ(grid.criticalWavelengths, 41u);  // All critical in uniform mode
    EXPECT_EQ(grid.coarseWavelengths, 0u);
    EXPECT_NEAR(grid.compressionRatio, 1.0f, 1e-5f);

    // Verify wavelength values
    EXPECT_NEAR(grid.wavelengths.front(), 3000.0f, 1e-5f);
    EXPECT_NEAR(grid.wavelengths.back(), 5000.0f, 1e-5f);

    // All should be marked critical
    for (bool isCrit : grid.isCritical) {
        EXPECT_TRUE(isCrit);
    }
}

TEST(AdaptiveGridGeneratorTest, GenerateUniformLWIR) {
    auto config = HyperspectralConfig::LWIR(100.0f);  // 8000-12000nm, 100nm step

    auto grid = AdaptiveGridGenerator::GenerateUniform(config);

    // (12000 - 8000) / 100 + 1 = 41 bands
    EXPECT_EQ(grid.totalWavelengths, 41u);
    EXPECT_NEAR(grid.wavelengths.front(), 8000.0f, 1e-5f);
    EXPECT_NEAR(grid.wavelengths.back(), 12000.0f, 1e-5f);
}

TEST(AdaptiveGridGeneratorTest, GenerateUniformSmallRange) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 4000.0f;
    config.wavelengthMax_nm = 4100.0f;
    config.wavelengthStep_nm = 10.0f;

    auto grid = AdaptiveGridGenerator::GenerateUniform(config);

    // (4100 - 4000) / 10 + 1 = 11 bands
    EXPECT_EQ(grid.totalWavelengths, 11u);
}

// ============================================================================
// AdaptiveGridGenerator::Generate Tests
// ============================================================================

TEST(AdaptiveGridGeneratorTest, GenerateNoCriticalWavelengths) {
    AdaptiveGridGenerator generator;

    auto grid = generator.Generate(
        3000.0f,   // min
        5000.0f,   // max
        50.0f,     // baseStep
        {},        // no critical wavelengths
        100.0f,    // criticalRadius
        2.0f       // coarseMultiplier
    );

    // With no critical wavelengths, all sampling should be coarse
    // Uniform: (5000-3000)/50 + 1 = 41
    // Coarse: (5000-3000)/100 + 1 = 21
    EXPECT_LT(grid.totalWavelengths, 41u);
    EXPECT_GT(grid.compressionRatio, 1.0f);

    // All should be non-critical (coarse)
    EXPECT_EQ(grid.criticalWavelengths, 0u);
    EXPECT_GT(grid.coarseWavelengths, 0u);
}

TEST(AdaptiveGridGeneratorTest, GenerateSingleCriticalWavelength) {
    AdaptiveGridGenerator generator;

    Vector<f32> critical = {4000.0f};

    auto grid = generator.Generate(
        3000.0f,   // min
        5000.0f,   // max
        50.0f,     // baseStep
        critical,
        100.0f,    // criticalRadius: ±100nm around 4000
        2.0f       // coarseMultiplier
    );

    EXPECT_GT(grid.totalWavelengths, 0u);
    EXPECT_GT(grid.criticalWavelengths, 0u);
    EXPECT_GT(grid.coarseWavelengths, 0u);

    // Should have dense sampling around 4000nm
    // Count wavelengths in critical region [3900, 4100]
    u32 inCriticalRegion = 0;
    for (usize i = 0; i < grid.wavelengths.size(); ++i) {
        if (grid.wavelengths[i] >= 3900.0f && grid.wavelengths[i] <= 4100.0f) {
            ++inCriticalRegion;
        }
    }

    // Critical region (200nm) with 50nm step should have ~5 samples
    EXPECT_GE(inCriticalRegion, 3u);
}

TEST(AdaptiveGridGeneratorTest, GenerateMultipleCriticalWavelengths) {
    AdaptiveGridGenerator generator;

    Vector<f32> critical = {3300.0f, 4000.0f, 4700.0f};

    auto grid = generator.Generate(
        3000.0f,   // min
        5000.0f,   // max
        50.0f,     // baseStep
        critical,
        100.0f,    // criticalRadius
        2.0f       // coarseMultiplier
    );

    EXPECT_GT(grid.totalWavelengths, 0u);

    // Should have more samples than pure coarse but fewer than uniform
    u32 uniformCount = 41;  // (5000-3000)/50 + 1
    EXPECT_LT(grid.totalWavelengths, uniformCount);

    // Compression ratio should be > 1
    EXPECT_GT(grid.compressionRatio, 1.0f);
}

TEST(AdaptiveGridGeneratorTest, GenerateWithLargeCoarseMultiplier) {
    AdaptiveGridGenerator generator;

    Vector<f32> critical = {4000.0f};

    auto grid = generator.Generate(
        3000.0f,   // min
        5000.0f,   // max
        50.0f,     // baseStep
        critical,
        50.0f,     // small criticalRadius
        4.0f       // large coarseMultiplier
    );

    // With 4x coarse multiplier and small critical region,
    // should achieve high compression
    EXPECT_GT(grid.compressionRatio, 1.5f);
}

TEST(AdaptiveGridGeneratorTest, GenerateFullCriticalCoverage) {
    AdaptiveGridGenerator generator;

    // Critical wavelengths covering entire range
    Vector<f32> critical;
    for (f32 wl = 3000.0f; wl <= 5000.0f; wl += 200.0f) {
        critical.push_back(wl);
    }

    auto grid = generator.Generate(
        3000.0f,
        5000.0f,
        50.0f,
        critical,
        150.0f,  // Large radius = overlapping coverage
        2.0f
    );

    // With full coverage, should be similar to uniform
    // (compression ratio close to 1)
    EXPECT_LT(grid.compressionRatio, 1.5f);
}

TEST(AdaptiveGridGeneratorTest, GenerateWithAnalysisResult) {
    AdaptiveGridGenerator generator;

    SpectralAnalysisResult analysis;
    analysis.criticalWavelengths = {3500.0f, 4200.0f};
    analysis.analysisRange_min = 3000.0f;
    analysis.analysisRange_max = 5000.0f;

    HyperspectralConfig config = HyperspectralConfig::MWIR(50.0f);
    config.adaptiveMode = AdaptiveSamplingMode::Spectral;
    config.adaptiveCriticalRadius_nm = 100.0f;
    config.adaptiveCoarseMultiplier = 2.0f;

    auto grid = generator.Generate(config, analysis);

    EXPECT_GT(grid.totalWavelengths, 0u);
    EXPECT_GT(grid.compressionRatio, 1.0f);
}

// ============================================================================
// AdaptiveGridGenerator::EstimateCompressionRatio Tests
// ============================================================================

TEST(AdaptiveGridGeneratorTest, EstimateCompressionNoCritical) {
    f32 ratio = AdaptiveGridGenerator::EstimateCompressionRatio(
        3000.0f, 5000.0f, 50.0f,
        {},       // No critical
        100.0f,
        2.0f
    );

    // Should be approximately 2x (coarse multiplier)
    EXPECT_GT(ratio, 1.5f);
    EXPECT_LT(ratio, 2.5f);
}

TEST(AdaptiveGridGeneratorTest, EstimateCompressionFullCritical) {
    // Many overlapping critical wavelengths
    Vector<f32> critical;
    for (f32 wl = 3000.0f; wl <= 5000.0f; wl += 100.0f) {
        critical.push_back(wl);
    }

    f32 ratio = AdaptiveGridGenerator::EstimateCompressionRatio(
        3000.0f, 5000.0f, 50.0f,
        critical,
        75.0f,   // overlapping radii
        2.0f
    );

    // Should be close to 1 (nearly full coverage)
    EXPECT_LT(ratio, 1.5f);
}

TEST(AdaptiveGridGeneratorTest, EstimateCompressionPartialCritical) {
    Vector<f32> critical = {4000.0f};

    f32 ratio = AdaptiveGridGenerator::EstimateCompressionRatio(
        3000.0f, 5000.0f, 50.0f,
        critical,
        100.0f,
        2.0f
    );

    // Should be between 1 and coarse multiplier
    EXPECT_GT(ratio, 1.0f);
    EXPECT_LT(ratio, 2.0f);
}

// ============================================================================
// GenerateReconstructionMapping Tests
// ============================================================================

TEST(ReconstructionMappingTest, GenerateMappingUniform) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;

    auto grid = AdaptiveGridGenerator::GenerateUniform(config);

    auto mapping = GenerateReconstructionMapping(grid, config);

    // All wavelengths were rendered in uniform mode
    u32 renderedCount = 0;
    for (const auto& m : mapping) {
        if (m.wasRendered) {
            ++renderedCount;
            EXPECT_NEAR(m.interpWeight, 0.0f, 1e-5f);
        }
    }

    // All should be marked as rendered
    EXPECT_EQ(renderedCount, config.GetNumBands());
}

TEST(ReconstructionMappingTest, GenerateMappingAdaptive) {
    // Create adaptive grid with gaps
    AdaptiveGridInfo grid;
    grid.wavelengths = {3000.0f, 3100.0f, 3200.0f};  // Sparse: missing 3050, 3150
    grid.isCritical = {true, true, true};
    grid.totalWavelengths = 3;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;  // Target: 3000, 3050, 3100, 3150, 3200

    auto mapping = GenerateReconstructionMapping(grid, config);

    EXPECT_EQ(mapping.size(), config.GetNumBands());  // 5 target bands

    // Check that 3050 and 3150 need interpolation
    bool found3050 = false, found3150 = false;
    for (const auto& m : mapping) {
        if (std::abs(m.targetWavelength - 3050.0f) < 1.0f) {
            EXPECT_FALSE(m.wasRendered);
            EXPECT_NEAR(m.interpWeight, 0.5f, 0.1f);  // Midpoint
            found3050 = true;
        }
        if (std::abs(m.targetWavelength - 3150.0f) < 1.0f) {
            EXPECT_FALSE(m.wasRendered);
            found3150 = true;
        }
    }
    EXPECT_TRUE(found3050);
    EXPECT_TRUE(found3150);
}

TEST(ReconstructionMappingTest, GenerateMappingEmptyGrid) {
    AdaptiveGridInfo grid;  // Empty

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;

    auto mapping = GenerateReconstructionMapping(grid, config);

    // Should handle gracefully
    EXPECT_TRUE(mapping.empty());
}

TEST(ReconstructionMappingTest, GenerateMappingInterpolationWeights) {
    AdaptiveGridInfo grid;
    grid.wavelengths = {3000.0f, 3200.0f};  // Only endpoints
    grid.isCritical = {true, true};
    grid.totalWavelengths = 2;

    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 3200.0f;
    config.wavelengthStep_nm = 50.0f;  // 3000, 3050, 3100, 3150, 3200

    auto mapping = GenerateReconstructionMapping(grid, config);

    EXPECT_EQ(mapping.size(), 5u);

    // Check interpolation weights
    for (const auto& m : mapping) {
        if (std::abs(m.targetWavelength - 3000.0f) < 1.0f) {
            EXPECT_TRUE(m.wasRendered);
        } else if (std::abs(m.targetWavelength - 3050.0f) < 1.0f) {
            EXPECT_FALSE(m.wasRendered);
            EXPECT_NEAR(m.interpWeight, 0.25f, 0.05f);  // 50/200
        } else if (std::abs(m.targetWavelength - 3100.0f) < 1.0f) {
            EXPECT_FALSE(m.wasRendered);
            EXPECT_NEAR(m.interpWeight, 0.5f, 0.05f);   // 100/200
        } else if (std::abs(m.targetWavelength - 3150.0f) < 1.0f) {
            EXPECT_FALSE(m.wasRendered);
            EXPECT_NEAR(m.interpWeight, 0.75f, 0.05f);  // 150/200
        } else if (std::abs(m.targetWavelength - 3200.0f) < 1.0f) {
            EXPECT_TRUE(m.wasRendered);
        }
    }
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

TEST(AdaptiveGridGeneratorTest, GenerateInvalidParameters) {
    AdaptiveGridGenerator generator;

    // Max < Min
    auto grid1 = generator.Generate(
        5000.0f, 3000.0f, 50.0f, {}, 100.0f, 2.0f
    );
    EXPECT_TRUE(grid1.wavelengths.empty());

    // Zero step
    auto grid2 = generator.Generate(
        3000.0f, 5000.0f, 0.0f, {}, 100.0f, 2.0f
    );
    EXPECT_TRUE(grid2.wavelengths.empty());

    // Negative step
    auto grid3 = generator.Generate(
        3000.0f, 5000.0f, -50.0f, {}, 100.0f, 2.0f
    );
    EXPECT_TRUE(grid3.wavelengths.empty());
}

TEST(AdaptiveGridGeneratorTest, GenerateSingleWavelength) {
    AdaptiveGridGenerator generator;

    auto grid = generator.Generate(
        4000.0f, 4000.1f, 1.0f, {}, 10.0f, 2.0f
    );

    // Should produce at least one wavelength
    EXPECT_GE(grid.totalWavelengths, 1u);
}

TEST(AdaptiveGridGeneratorTest, GenerateVeryFineStep) {
    AdaptiveGridGenerator generator;

    auto grid = generator.Generate(
        3000.0f, 3010.0f, 0.1f, {3005.0f}, 2.0f, 2.0f
    );

    // Should produce many wavelengths
    EXPECT_GT(grid.totalWavelengths, 50u);
}

TEST(AdaptiveGridGeneratorTest, GenerateCoarseMultiplierClamped) {
    AdaptiveGridGenerator generator;

    // Coarse multiplier should be clamped to reasonable range
    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, {}, 100.0f, 100.0f  // Extreme multiplier
    );

    // Should still produce reasonable output
    EXPECT_GT(grid.totalWavelengths, 0u);
    // Compression should be clamped
    EXPECT_LT(grid.compressionRatio, 10.0f);
}

TEST(AdaptiveGridGeneratorTest, GenerateCriticalAtBoundaries) {
    AdaptiveGridGenerator generator;

    Vector<f32> critical = {3000.0f, 5000.0f};  // At exact boundaries

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, critical, 50.0f, 2.0f
    );

    // Should include boundary wavelengths
    EXPECT_GT(grid.totalWavelengths, 0u);
    EXPECT_NEAR(grid.wavelengths.front(), 3000.0f, 1.0f);
    EXPECT_NEAR(grid.wavelengths.back(), 5000.0f, 1.0f);
}

TEST(AdaptiveGridGeneratorTest, GenerateOverlappingCriticalRegions) {
    AdaptiveGridGenerator generator;

    // Two critical wavelengths with overlapping radii
    Vector<f32> critical = {4000.0f, 4050.0f};

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, critical, 100.0f, 2.0f
    );

    // Overlapping regions should be merged
    EXPECT_GT(grid.totalWavelengths, 0u);
}

TEST(AdaptiveGridGeneratorTest, GenerateCriticalOutsideRange) {
    AdaptiveGridGenerator generator;

    // Critical wavelengths outside the render range
    Vector<f32> critical = {2000.0f, 6000.0f};

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, critical, 100.0f, 2.0f
    );

    // Should still generate grid, ignoring out-of-range criticals
    EXPECT_GT(grid.totalWavelengths, 0u);
    EXPECT_EQ(grid.criticalWavelengths, 0u);  // No critical in range
}

// ============================================================================
// Wavelength Ordering Tests
// ============================================================================

TEST(AdaptiveGridGeneratorTest, WavelengthsAreSorted) {
    AdaptiveGridGenerator generator;

    Vector<f32> critical = {4500.0f, 3500.0f, 4000.0f};  // Unsorted

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, critical, 100.0f, 2.0f
    );

    // Wavelengths should be in ascending order
    for (usize i = 1; i < grid.wavelengths.size(); ++i) {
        EXPECT_GT(grid.wavelengths[i], grid.wavelengths[i - 1]);
    }
}

TEST(AdaptiveGridGeneratorTest, WavelengthsInRange) {
    AdaptiveGridGenerator generator;

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, {4000.0f}, 100.0f, 2.0f
    );

    // All wavelengths should be within range
    for (f32 wl : grid.wavelengths) {
        EXPECT_GE(wl, 3000.0f);
        EXPECT_LE(wl, 5000.0f);
    }
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST(AdaptiveGridGeneratorTest, IsCriticalConsistentWithCount) {
    AdaptiveGridGenerator generator;

    auto grid = generator.Generate(
        3000.0f, 5000.0f, 50.0f, {4000.0f}, 100.0f, 2.0f
    );

    // Count true values in isCritical
    u32 trueCount = 0;
    for (bool b : grid.isCritical) {
        if (b) ++trueCount;
    }

    EXPECT_EQ(trueCount, grid.criticalWavelengths);
    EXPECT_EQ(grid.isCritical.size() - trueCount, grid.coarseWavelengths);
    EXPECT_EQ(grid.criticalWavelengths + grid.coarseWavelengths, grid.totalWavelengths);
}

TEST(AdaptiveGridGeneratorTest, CompressionRatioConsistent) {
    AdaptiveGridGenerator generator;

    HyperspectralConfig config = HyperspectralConfig::MWIR(50.0f);

    auto uniformGrid = AdaptiveGridGenerator::GenerateUniform(config);

    SpectralAnalysisResult analysis;
    analysis.criticalWavelengths = {3500.0f, 4500.0f};

    config.adaptiveMode = AdaptiveSamplingMode::Spectral;
    config.adaptiveCriticalRadius_nm = 100.0f;
    config.adaptiveCoarseMultiplier = 2.0f;

    auto adaptiveGrid = generator.Generate(config, analysis);

    // Compression ratio = uniform / adaptive
    f32 expectedRatio = static_cast<f32>(uniformGrid.totalWavelengths) /
                        static_cast<f32>(adaptiveGrid.totalWavelengths);

    EXPECT_NEAR(adaptiveGrid.compressionRatio, expectedRatio, 0.1f);
}
