// ============================================================================
// Quantiloom - Unit Tests for hs_core/HyperspectralConfig.hpp
// ============================================================================
// Tests cover:
// - HyperspectralConfig construction and validation
// - Wavelength range calculations
// - Factory methods for common spectral bands (MWIR, LWIR, SWIR, NIR, VIS)
// - Configuration parameter validation
// - Edge cases and boundary conditions
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/HyperspectralConfig.hpp"

using namespace quantiloom;

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(HyperspectralConfigTest, DefaultConstruction) {
    HyperspectralConfig config;

    EXPECT_EQ(config.wavelengthMin_nm, 400.0f);
    EXPECT_EQ(config.wavelengthMax_nm, 2500.0f);
    EXPECT_EQ(config.wavelengthStep_nm, 10.0f);
    EXPECT_EQ(config.spp, 16);
    EXPECT_EQ(config.maxBounces, 4);
    EXPECT_EQ(config.outputFormat, HyperspectralOutputFormat::ENVI_BSQ);
    EXPECT_EQ(config.adaptiveMode, AdaptiveSamplingMode::None);
    EXPECT_TRUE(config.IsValid());
}

TEST(HyperspectralConfigTest, GetNumBands) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 700.0f;
    config.wavelengthStep_nm = 10.0f;

    // (700 - 400) / 10 + 1 = 31 bands
    EXPECT_EQ(config.GetNumBands(), 31);
}

TEST(HyperspectralConfigTest, GetNumBandsExactDivision) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 3000.0f;
    config.wavelengthMax_nm = 5000.0f;
    config.wavelengthStep_nm = 50.0f;

    // (5000 - 3000) / 50 + 1 = 41 bands
    EXPECT_EQ(config.GetNumBands(), 41);
}

TEST(HyperspectralConfigTest, GetWavelength) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 800.0f;
    config.wavelengthStep_nm = 100.0f;

    EXPECT_NEAR(config.GetWavelength(0), 400.0f, 1e-5f);
    EXPECT_NEAR(config.GetWavelength(1), 500.0f, 1e-5f);
    EXPECT_NEAR(config.GetWavelength(2), 600.0f, 1e-5f);
    EXPECT_NEAR(config.GetWavelength(3), 700.0f, 1e-5f);
    EXPECT_NEAR(config.GetWavelength(4), 800.0f, 1e-5f);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(HyperspectralConfigTest, IsValidTrue) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 700.0f;
    config.wavelengthStep_nm = 10.0f;
    config.spp = 16;

    EXPECT_TRUE(config.IsValid());
    EXPECT_TRUE(config.GetValidationError().empty());
}

TEST(HyperspectralConfigTest, IsValidFalseNegativeMin) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = -100.0f;
    config.wavelengthMax_nm = 700.0f;

    EXPECT_FALSE(config.IsValid());
    EXPECT_FALSE(config.GetValidationError().empty());
}

TEST(HyperspectralConfigTest, IsValidFalseZeroMin) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 0.0f;
    config.wavelengthMax_nm = 700.0f;

    EXPECT_FALSE(config.IsValid());
}

TEST(HyperspectralConfigTest, IsValidFalseMaxLessThanMin) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 700.0f;
    config.wavelengthMax_nm = 400.0f;

    EXPECT_FALSE(config.IsValid());
    EXPECT_FALSE(config.GetValidationError().empty());
}

TEST(HyperspectralConfigTest, IsValidFalseMaxEqualsMin) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 500.0f;
    config.wavelengthMax_nm = 500.0f;

    EXPECT_FALSE(config.IsValid());
}

TEST(HyperspectralConfigTest, IsValidFalseZeroStep) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 700.0f;
    config.wavelengthStep_nm = 0.0f;

    EXPECT_FALSE(config.IsValid());
}

TEST(HyperspectralConfigTest, IsValidFalseNegativeStep) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 700.0f;
    config.wavelengthStep_nm = -10.0f;

    EXPECT_FALSE(config.IsValid());
}

TEST(HyperspectralConfigTest, IsValidFalseZeroSpp) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 400.0f;
    config.wavelengthMax_nm = 700.0f;
    config.wavelengthStep_nm = 10.0f;
    config.spp = 0;

    EXPECT_FALSE(config.IsValid());
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(HyperspectralConfigTest, FactoryMWIR) {
    auto config = HyperspectralConfig::MWIR(50.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 3000.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 5000.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 50.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (5000 - 3000) / 50 + 1 = 41 bands
    EXPECT_EQ(config.GetNumBands(), 41);
}

TEST(HyperspectralConfigTest, FactoryMWIRDefault) {
    auto config = HyperspectralConfig::MWIR();

    EXPECT_NEAR(config.wavelengthStep_nm, 50.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());
}

TEST(HyperspectralConfigTest, FactoryLWIR) {
    auto config = HyperspectralConfig::LWIR(100.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 8000.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 12000.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 100.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (12000 - 8000) / 100 + 1 = 41 bands
    EXPECT_EQ(config.GetNumBands(), 41);
}

TEST(HyperspectralConfigTest, FactorySWIR) {
    auto config = HyperspectralConfig::SWIR(20.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 1000.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 2500.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 20.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (2500 - 1000) / 20 + 1 = 76 bands
    EXPECT_EQ(config.GetNumBands(), 76);
}

TEST(HyperspectralConfigTest, FactoryNIR) {
    auto config = HyperspectralConfig::NIR(10.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 780.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 1400.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 10.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (1400 - 780) / 10 + 1 = 63 bands
    EXPECT_EQ(config.GetNumBands(), 63);
}

TEST(HyperspectralConfigTest, FactoryVIS) {
    auto config = HyperspectralConfig::VIS(5.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 380.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 780.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 5.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (780 - 380) / 5 + 1 = 81 bands
    EXPECT_EQ(config.GetNumBands(), 81);
}

TEST(HyperspectralConfigTest, FactoryVNIR) {
    auto config = HyperspectralConfig::VNIR(10.0f);

    EXPECT_NEAR(config.wavelengthMin_nm, 400.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthMax_nm, 2500.0f, 1e-5f);
    EXPECT_NEAR(config.wavelengthStep_nm, 10.0f, 1e-5f);
    EXPECT_TRUE(config.IsValid());

    // (2500 - 400) / 10 + 1 = 211 bands
    EXPECT_EQ(config.GetNumBands(), 211);
}

// ============================================================================
// Progress Structure Tests
// ============================================================================

TEST(HyperspectralProgressTest, GetPercentage) {
    HyperspectralProgress progress;
    progress.currentBand = 50;
    progress.totalBands = 100;

    EXPECT_NEAR(progress.GetPercentage(), 50.0f, 1e-5f);
}

TEST(HyperspectralProgressTest, GetPercentageZeroTotal) {
    HyperspectralProgress progress;
    progress.currentBand = 10;
    progress.totalBands = 0;

    EXPECT_NEAR(progress.GetPercentage(), 0.0f, 1e-5f);
}

TEST(HyperspectralProgressTest, GetPercentageComplete) {
    HyperspectralProgress progress;
    progress.currentBand = 100;
    progress.totalBands = 100;

    EXPECT_NEAR(progress.GetPercentage(), 100.0f, 1e-5f);
}

TEST(HyperspectralProgressTest, GetRemainingSeconds) {
    HyperspectralProgress progress;
    progress.elapsedSeconds = 30.0f;
    progress.estimatedTotalSeconds = 100.0f;

    EXPECT_NEAR(progress.GetRemainingSeconds(), 70.0f, 1e-5f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(HyperspectralConfigTest, VerySmallStep) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 550.0f;
    config.wavelengthMax_nm = 551.0f;
    config.wavelengthStep_nm = 0.1f;

    EXPECT_TRUE(config.IsValid());
    EXPECT_EQ(config.GetNumBands(), 11);  // (1.0 / 0.1) + 1 = 11
}

TEST(HyperspectralConfigTest, SingleBand) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 550.0f;
    config.wavelengthMax_nm = 550.1f;
    config.wavelengthStep_nm = 1.0f;

    EXPECT_TRUE(config.IsValid());
    EXPECT_EQ(config.GetNumBands(), 1);
}

TEST(HyperspectralConfigTest, LargeWavelengthRange) {
    HyperspectralConfig config;
    config.wavelengthMin_nm = 200.0f;
    config.wavelengthMax_nm = 25000.0f;  // UV to thermal IR
    config.wavelengthStep_nm = 100.0f;

    EXPECT_TRUE(config.IsValid());
    // (25000 - 200) / 100 + 1 = 249 bands
    EXPECT_EQ(config.GetNumBands(), 249);
}

TEST(HyperspectralConfigTest, HighSPP) {
    HyperspectralConfig config;
    config.spp = 4096;

    EXPECT_TRUE(config.IsValid());
}

// ============================================================================
// Adaptive Sampling Configuration Tests
// ============================================================================

TEST(HyperspectralConfigTest, AdaptiveModeNone) {
    HyperspectralConfig config;
    config.adaptiveMode = AdaptiveSamplingMode::None;

    EXPECT_EQ(config.adaptiveMode, AdaptiveSamplingMode::None);
}

TEST(HyperspectralConfigTest, AdaptiveModeSpectral) {
    HyperspectralConfig config;
    config.adaptiveMode = AdaptiveSamplingMode::Spectral;
    config.adaptiveDerivativeThreshold = 0.002f;
    config.adaptiveCriticalRadius_nm = 50.0f;
    config.adaptiveCoarseMultiplier = 4.0f;

    EXPECT_EQ(config.adaptiveMode, AdaptiveSamplingMode::Spectral);
    EXPECT_NEAR(config.adaptiveDerivativeThreshold, 0.002f, 1e-6f);
    EXPECT_NEAR(config.adaptiveCriticalRadius_nm, 50.0f, 1e-5f);
    EXPECT_NEAR(config.adaptiveCoarseMultiplier, 4.0f, 1e-5f);
}

// ============================================================================
// Output Format Tests
// ============================================================================

TEST(HyperspectralConfigTest, OutputFormatENVI_BSQ) {
    HyperspectralConfig config;
    config.outputFormat = HyperspectralOutputFormat::ENVI_BSQ;

    EXPECT_EQ(config.outputFormat, HyperspectralOutputFormat::ENVI_BSQ);
}

TEST(HyperspectralConfigTest, OutputFormatENVI_BIL) {
    HyperspectralConfig config;
    config.outputFormat = HyperspectralOutputFormat::ENVI_BIL;

    EXPECT_EQ(config.outputFormat, HyperspectralOutputFormat::ENVI_BIL);
}

TEST(HyperspectralConfigTest, OutputFormatGeoTIFF) {
    HyperspectralConfig config;
    config.outputFormat = HyperspectralOutputFormat::GeoTIFF;

    EXPECT_EQ(config.outputFormat, HyperspectralOutputFormat::GeoTIFF);
}

TEST(HyperspectralConfigTest, OutputFormatEXR) {
    HyperspectralConfig config;
    config.outputFormat = HyperspectralOutputFormat::EXR_Multipart;

    EXPECT_EQ(config.outputFormat, HyperspectralOutputFormat::EXR_Multipart);
}
