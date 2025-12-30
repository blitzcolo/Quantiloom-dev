// ============================================================================
// Quantiloom - Unit Tests for hs_core/BatchRenderer.hpp
// ============================================================================
// Tests cover:
// - BatchRenderStatus enum and string conversion
// - BatchRenderProgress structure and utility methods
// - BatchRenderParams default values
// - ValidateWavelengthList utility function
// - SortWavelengths utility function
// - EstimateBatchTime utility function
// - Edge cases for wavelength validation
//
// Note: GPU-dependent tests (actual rendering) require integration tests
// with a valid Vulkan context, which are not included here.
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/BatchRenderer.hpp"
#include <cmath>
#include <algorithm>

using namespace quantiloom;

// ============================================================================
// BatchRenderStatus Tests
// ============================================================================

TEST(BatchRenderStatusTest, ToString) {
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::Success), "Success");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::InvalidInput), "Invalid input");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::PipelineError), "Pipeline error");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::RenderFailed), "Render failed");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::Cancelled), "Cancelled");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::OutOfMemory), "Out of memory");
    EXPECT_STREQ(BatchRenderStatusToString(BatchRenderStatus::GPUError), "GPU error");
}

TEST(BatchRenderStatusTest, EnumValues) {
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::Success), 0u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::InvalidInput), 1u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::PipelineError), 2u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::RenderFailed), 3u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::Cancelled), 4u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::OutOfMemory), 5u);
    EXPECT_EQ(static_cast<u32>(BatchRenderStatus::GPUError), 6u);
}

TEST(BatchRenderStatusTest, UnknownStatus) {
    // Cast an invalid value
    auto unknown = static_cast<BatchRenderStatus>(999);
    const char* str = BatchRenderStatusToString(unknown);
    EXPECT_STREQ(str, "Unknown status");
}

// ============================================================================
// BatchRenderProgress Tests
// ============================================================================

TEST(BatchRenderProgressTest, DefaultValues) {
    BatchRenderProgress progress{};

    // Default-initialized values should be zero
    EXPECT_EQ(progress.currentBand, 0u);
    EXPECT_EQ(progress.totalBands, 0u);
    EXPECT_NEAR(progress.currentWavelength_nm, 0.0f, 1e-5f);
    EXPECT_NEAR(progress.elapsedSeconds, 0.0f, 1e-5f);
    EXPECT_NEAR(progress.estimatedTotalSeconds, 0.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetPercentageNormal) {
    BatchRenderProgress progress;
    progress.currentBand = 25;
    progress.totalBands = 100;

    EXPECT_NEAR(progress.GetPercentage(), 25.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetPercentageComplete) {
    BatchRenderProgress progress;
    progress.currentBand = 100;
    progress.totalBands = 100;

    EXPECT_NEAR(progress.GetPercentage(), 100.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetPercentageZeroTotal) {
    BatchRenderProgress progress;
    progress.currentBand = 10;
    progress.totalBands = 0;

    EXPECT_NEAR(progress.GetPercentage(), 0.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetPercentageStart) {
    BatchRenderProgress progress;
    progress.currentBand = 0;
    progress.totalBands = 50;

    EXPECT_NEAR(progress.GetPercentage(), 0.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetRemainingSeconds) {
    BatchRenderProgress progress;
    progress.elapsedSeconds = 30.0f;
    progress.estimatedTotalSeconds = 100.0f;

    EXPECT_NEAR(progress.GetRemainingSeconds(), 70.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetRemainingSecondsNegative) {
    // When elapsed > estimated (estimation was off)
    BatchRenderProgress progress;
    progress.elapsedSeconds = 120.0f;
    progress.estimatedTotalSeconds = 100.0f;

    EXPECT_NEAR(progress.GetRemainingSeconds(), -20.0f, 1e-5f);
}

TEST(BatchRenderProgressTest, GetRemainingSecondsZero) {
    BatchRenderProgress progress;
    progress.elapsedSeconds = 100.0f;
    progress.estimatedTotalSeconds = 100.0f;

    EXPECT_NEAR(progress.GetRemainingSeconds(), 0.0f, 1e-5f);
}

// ============================================================================
// BatchRenderParams Tests
// ============================================================================

TEST(BatchRenderParamsTest, DefaultValues) {
    BatchRenderParams params;

    EXPECT_EQ(params.spp, 16u);
    EXPECT_EQ(params.maxBounces, 4u);
    EXPECT_TRUE(params.enableAccumulation);
    EXPECT_FALSE(params.verbose);
}

TEST(BatchRenderParamsTest, CustomValues) {
    BatchRenderParams params;
    params.spp = 128;
    params.maxBounces = 8;
    params.enableAccumulation = false;
    params.verbose = true;

    EXPECT_EQ(params.spp, 128u);
    EXPECT_EQ(params.maxBounces, 8u);
    EXPECT_FALSE(params.enableAccumulation);
    EXPECT_TRUE(params.verbose);
}

// ============================================================================
// ValidateWavelengthList Tests
// ============================================================================

TEST(ValidateWavelengthListTest, ValidSorted) {
    Vector<f32> wavelengths = {3000.0f, 3500.0f, 4000.0f, 4500.0f, 5000.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, ValidSingleElement) {
    Vector<f32> wavelengths = {4000.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, ValidTwoElements) {
    Vector<f32> wavelengths = {3000.0f, 5000.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, InvalidEmpty) {
    Vector<f32> wavelengths;

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, InvalidNegative) {
    Vector<f32> wavelengths = {3000.0f, -100.0f, 5000.0f};

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, InvalidZero) {
    Vector<f32> wavelengths = {0.0f, 3000.0f, 5000.0f};

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, InvalidUnsorted) {
    Vector<f32> wavelengths = {3000.0f, 5000.0f, 4000.0f};

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, InvalidDescending) {
    Vector<f32> wavelengths = {5000.0f, 4000.0f, 3000.0f};

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, ValidEqualElements) {
    // Equal adjacent elements should be valid (not strictly ascending)
    Vector<f32> wavelengths = {3000.0f, 3000.0f, 4000.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(ValidateWavelengthListTest, ValidLargeList) {
    Vector<f32> wavelengths;
    for (f32 wl = 3000.0f; wl <= 5000.0f; wl += 10.0f) {
        wavelengths.push_back(wl);
    }

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
    EXPECT_EQ(wavelengths.size(), 201u);  // (5000-3000)/10 + 1
}

TEST(ValidateWavelengthListTest, InvalidAllNegative) {
    Vector<f32> wavelengths = {-500.0f, -400.0f, -300.0f};

    EXPECT_FALSE(ValidateWavelengthList(wavelengths));
}

// ============================================================================
// SortWavelengths Tests
// ============================================================================

TEST(SortWavelengthsTest, AlreadySorted) {
    Vector<f32> wavelengths = {3000.0f, 4000.0f, 5000.0f};

    SortWavelengths(wavelengths);

    EXPECT_EQ(wavelengths.size(), 3u);
    EXPECT_NEAR(wavelengths[0], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[1], 4000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[2], 5000.0f, 1e-5f);
}

TEST(SortWavelengthsTest, ReverseSorted) {
    Vector<f32> wavelengths = {5000.0f, 4000.0f, 3000.0f};

    SortWavelengths(wavelengths);

    EXPECT_NEAR(wavelengths[0], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[1], 4000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[2], 5000.0f, 1e-5f);
}

TEST(SortWavelengthsTest, RandomOrder) {
    Vector<f32> wavelengths = {4500.0f, 3000.0f, 5000.0f, 3500.0f, 4000.0f};

    SortWavelengths(wavelengths);

    EXPECT_NEAR(wavelengths[0], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[1], 3500.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[2], 4000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[3], 4500.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[4], 5000.0f, 1e-5f);
}

TEST(SortWavelengthsTest, EmptyList) {
    Vector<f32> wavelengths;

    SortWavelengths(wavelengths);

    EXPECT_TRUE(wavelengths.empty());
}

TEST(SortWavelengthsTest, SingleElement) {
    Vector<f32> wavelengths = {4000.0f};

    SortWavelengths(wavelengths);

    EXPECT_EQ(wavelengths.size(), 1u);
    EXPECT_NEAR(wavelengths[0], 4000.0f, 1e-5f);
}

TEST(SortWavelengthsTest, WithDuplicates) {
    Vector<f32> wavelengths = {4000.0f, 3000.0f, 4000.0f, 5000.0f, 3000.0f};

    SortWavelengths(wavelengths);

    EXPECT_EQ(wavelengths.size(), 5u);
    EXPECT_NEAR(wavelengths[0], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[1], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[2], 4000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[3], 4000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[4], 5000.0f, 1e-5f);
}

// ============================================================================
// EstimateBatchTime Tests
// ============================================================================

TEST(EstimateBatchTimeTest, Basic) {
    f64 time = EstimateBatchTime(10, 1.0);

    EXPECT_NEAR(time, 10.0, 1e-10);
}

TEST(EstimateBatchTimeTest, ZeroBands) {
    f64 time = EstimateBatchTime(0, 1.0);

    EXPECT_NEAR(time, 0.0, 1e-10);
}

TEST(EstimateBatchTimeTest, ZeroTimePerBand) {
    f64 time = EstimateBatchTime(100, 0.0);

    EXPECT_NEAR(time, 0.0, 1e-10);
}

TEST(EstimateBatchTimeTest, LargeBandCount) {
    f64 time = EstimateBatchTime(1000, 0.5);

    EXPECT_NEAR(time, 500.0, 1e-10);
}

TEST(EstimateBatchTimeTest, SmallTimePerBand) {
    f64 time = EstimateBatchTime(100, 0.001);

    EXPECT_NEAR(time, 0.1, 1e-10);
}

TEST(EstimateBatchTimeTest, FractionalResult) {
    f64 time = EstimateBatchTime(3, 0.33333);

    EXPECT_NEAR(time, 0.99999, 1e-5);
}

// ============================================================================
// Workflow Integration Tests (no GPU)
// ============================================================================

TEST(BatchRendererWorkflowTest, ValidateAndSort) {
    // Simulate typical workflow: receive unsorted list, sort, validate
    Vector<f32> wavelengths = {4500.0f, 3000.0f, 5000.0f, 3500.0f};

    // Initially invalid (unsorted)
    EXPECT_FALSE(ValidateWavelengthList(wavelengths));

    // Sort
    SortWavelengths(wavelengths);

    // Now valid
    EXPECT_TRUE(ValidateWavelengthList(wavelengths));

    // Verify order
    EXPECT_NEAR(wavelengths[0], 3000.0f, 1e-5f);
    EXPECT_NEAR(wavelengths[3], 5000.0f, 1e-5f);
}

TEST(BatchRendererWorkflowTest, EstimateTimeForMWIR) {
    // MWIR: 3000-5000nm, 50nm step = 41 bands
    // Assume 0.5s per band
    u32 numBands = 41;
    f64 timePerBand = 0.5;

    f64 estimated = EstimateBatchTime(numBands, timePerBand);

    EXPECT_NEAR(estimated, 20.5, 1e-10);
}

TEST(BatchRendererWorkflowTest, EstimateTimeForLWIR) {
    // LWIR: 8000-12000nm, 100nm step = 41 bands
    // Assume 0.8s per band (larger wavelength = slower)
    u32 numBands = 41;
    f64 timePerBand = 0.8;

    f64 estimated = EstimateBatchTime(numBands, timePerBand);

    EXPECT_NEAR(estimated, 32.8, 1e-10);
}

TEST(BatchRendererWorkflowTest, ProgressTrackingSimulation) {
    // Simulate progress tracking during batch render
    u32 totalBands = 100;

    Vector<BatchRenderProgress> progressHistory;

    for (u32 band = 1; band <= totalBands; ++band) {
        BatchRenderProgress progress;
        progress.currentBand = band;
        progress.totalBands = totalBands;
        progress.currentWavelength_nm = 3000.0f + (band - 1) * 20.0f;
        progress.elapsedSeconds = static_cast<f32>(band) * 0.1f;  // 0.1s per band
        progress.estimatedTotalSeconds = 10.0f;  // 100 bands * 0.1s

        progressHistory.push_back(progress);
    }

    // Verify first progress
    EXPECT_NEAR(progressHistory[0].GetPercentage(), 1.0f, 1e-5f);
    EXPECT_NEAR(progressHistory[0].GetRemainingSeconds(), 9.9f, 1e-5f);

    // Verify middle progress
    EXPECT_NEAR(progressHistory[49].GetPercentage(), 50.0f, 1e-5f);
    EXPECT_NEAR(progressHistory[49].GetRemainingSeconds(), 5.0f, 1e-5f);

    // Verify last progress
    EXPECT_NEAR(progressHistory[99].GetPercentage(), 100.0f, 1e-5f);
    EXPECT_NEAR(progressHistory[99].GetRemainingSeconds(), 0.0f, 1e-5f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(BatchRendererEdgeCaseTest, VeryLargeWavelengthList) {
    Vector<f32> wavelengths;
    for (f32 wl = 200.0f; wl <= 25000.0f; wl += 1.0f) {
        wavelengths.push_back(wl);
    }

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
    EXPECT_EQ(wavelengths.size(), 24801u);
}

TEST(BatchRendererEdgeCaseTest, NonUniformSpacing) {
    // Adaptive sampling produces non-uniform spacing
    Vector<f32> wavelengths = {
        3000.0f, 3010.0f, 3020.0f,  // Fine sampling
        3100.0f, 3200.0f,           // Coarse sampling
        3210.0f, 3220.0f, 3230.0f,  // Fine again
        3500.0f, 3700.0f, 4000.0f   // Coarse
    };

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(BatchRendererEdgeCaseTest, VerySmallWavelengths) {
    // UV range
    Vector<f32> wavelengths = {200.0f, 250.0f, 300.0f, 350.0f, 380.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(BatchRendererEdgeCaseTest, VeryLargeWavelengths) {
    // Far IR / THz range
    Vector<f32> wavelengths = {10000.0f, 15000.0f, 20000.0f, 25000.0f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}

TEST(BatchRendererEdgeCaseTest, HighPrecisionWavelengths) {
    Vector<f32> wavelengths = {3000.001f, 3000.002f, 3000.003f, 3000.004f};

    EXPECT_TRUE(ValidateWavelengthList(wavelengths));
}
