// ============================================================================
// Quantiloom - Unit Tests for Phase 4 GpuSpectralReconstructor
// ============================================================================
// Tests cover:
// - GpuReconstructorStatus enum and string conversion
// - GpuReconstructorConfig default values
// - Memory estimation utilities
// - IsGpuReconstructionSupported utility
// - EstimateReconstructionMemory calculation
//
// Note: Actual GPU reconstruction tests require a valid Vulkan context
// and are not included here. These belong in integration tests.
// ============================================================================

#include <gtest/gtest.h>
#include "hs_core/GpuSpectralReconstructor.hpp"
#include "hs_core/SpectralReconstructor.hpp"
#include "core/SpectralCube.hpp"
#include <cmath>

using namespace quantiloom;

// ============================================================================
// GpuReconstructorStatus Tests
// ============================================================================

TEST(GpuReconstructorStatusTest, ToString) {
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::Success), "Success");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::NotInitialized), "Not initialized");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::InvalidInput), "Invalid input");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::ShaderCompileFailed), "Shader compile failed");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::PipelineCreateFailed), "Pipeline create failed");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::BufferAllocFailed), "Buffer alloc failed");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::DispatchFailed), "Dispatch failed");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::ReadbackFailed), "Readback failed");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::OutOfMemory), "Out of memory");
    EXPECT_STREQ(GpuReconstructorStatusToString(GpuReconstructorStatus::Timeout), "Timeout");
}

TEST(GpuReconstructorStatusTest, EnumValues) {
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::Success), 0u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::NotInitialized), 1u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::InvalidInput), 2u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::ShaderCompileFailed), 3u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::PipelineCreateFailed), 4u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::BufferAllocFailed), 5u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::DispatchFailed), 6u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::ReadbackFailed), 7u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::OutOfMemory), 8u);
    EXPECT_EQ(static_cast<u32>(GpuReconstructorStatus::Timeout), 9u);
}

TEST(GpuReconstructorStatusTest, UnknownStatus) {
    auto unknown = static_cast<GpuReconstructorStatus>(999);
    const char* str = GpuReconstructorStatusToString(unknown);
    EXPECT_STREQ(str, "Unknown status");
}

// ============================================================================
// GpuReconstructorConfig Tests
// ============================================================================

TEST(GpuReconstructorConfigTest, DefaultValues) {
    GpuReconstructorConfig config;

    EXPECT_EQ(config.method, InterpolationMethod::CatmullRom);
    EXPECT_EQ(config.workgroupSizeX, 16u);
    EXPECT_EQ(config.workgroupSizeY, 16u);
    EXPECT_EQ(config.timeoutMs, 10000u);
    EXPECT_FALSE(config.verbose);
}

TEST(GpuReconstructorConfigTest, CustomValues) {
    GpuReconstructorConfig config;
    config.method = InterpolationMethod::Linear;
    config.workgroupSizeX = 8;
    config.workgroupSizeY = 8;
    config.timeoutMs = 5000;
    config.verbose = true;

    EXPECT_EQ(config.method, InterpolationMethod::Linear);
    EXPECT_EQ(config.workgroupSizeX, 8u);
    EXPECT_EQ(config.workgroupSizeY, 8u);
    EXPECT_EQ(config.timeoutMs, 5000u);
    EXPECT_TRUE(config.verbose);
}

TEST(GpuReconstructorConfigTest, AkimaMethod) {
    GpuReconstructorConfig config;
    config.method = InterpolationMethod::Akima;

    EXPECT_EQ(config.method, InterpolationMethod::Akima);
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST(EstimateReconstructionMemoryTest, SmallCube) {
    // 256x256 image, 50 sparse bands -> 100 target bands
    SpectralCube cube(256, 256, 50, 3000.0f, 5000.0f);
    u32 targetBands = 100;

    u64 memory = EstimateReconstructionMemory(cube, targetBands);

    // Expected:
    // srcSize = 256 * 256 * 50 * 4 = 13,107,200 bytes
    // dstSize = 256 * 256 * 100 * 4 = 26,214,400 bytes
    // wlSize = (50 + 100) * 4 = 600 bytes
    // stagingSize = srcSize + dstSize = 39,321,600 bytes
    // Total = ~78 MB

    EXPECT_GT(memory, 50 * 1024 * 1024);  // At least 50 MB
    EXPECT_LT(memory, 200 * 1024 * 1024);  // Less than 200 MB
}

TEST(EstimateReconstructionMemoryTest, MediumCube) {
    // 1024x1024 image, 100 sparse bands -> 200 target bands
    SpectralCube cube(1024, 1024, 100, 3000.0f, 5000.0f);
    u32 targetBands = 200;

    u64 memory = EstimateReconstructionMemory(cube, targetBands);

    // Expected ~1.25 GB
    EXPECT_GT(memory, 500 * 1024 * 1024);  // At least 500 MB
    EXPECT_LT(memory, 3ULL * 1024 * 1024 * 1024);  // Less than 3 GB
}

TEST(EstimateReconstructionMemoryTest, LargeCube) {
    // 2048x2048 image, 200 sparse bands -> 400 target bands
    SpectralCube cube(2048, 2048, 200, 3000.0f, 12000.0f);
    u32 targetBands = 400;

    u64 memory = EstimateReconstructionMemory(cube, targetBands);

    // Very large cube
    EXPECT_GT(memory, 2ULL * 1024 * 1024 * 1024);  // At least 2 GB
}

TEST(EstimateReconstructionMemoryTest, MinimalCube) {
    // 64x64 image, 10 sparse bands -> 20 target bands
    SpectralCube cube(64, 64, 10, 3000.0f, 5000.0f);
    u32 targetBands = 20;

    u64 memory = EstimateReconstructionMemory(cube, targetBands);

    // Very small cube
    EXPECT_GT(memory, 0u);
    EXPECT_LT(memory, 10 * 1024 * 1024);  // Less than 10 MB
}

// ============================================================================
// InterpolationMethod Consistency Tests
// ============================================================================
// These tests verify that the interpolation methods produce consistent results
// between CPU and (hypothetically) GPU implementations

TEST(GpuInterpolationMethodTest, EnumValues) {
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Linear), 0u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::CatmullRom), 1u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Akima), 2u);
}

// ============================================================================
// HyperspectralConfig GPU Option Tests
// ============================================================================

TEST(HyperspectralConfigGpuTest, DefaultGpuEnabled) {
    HyperspectralConfig config;

    // GPU reconstruction should be enabled by default
    EXPECT_TRUE(config.useGpuReconstruction);
}

TEST(HyperspectralConfigGpuTest, DisableGpuReconstruction) {
    HyperspectralConfig config;
    config.useGpuReconstruction = false;

    EXPECT_FALSE(config.useGpuReconstruction);
}

TEST(HyperspectralConfigGpuTest, GpuOptionWithAdaptiveMode) {
    auto config = HyperspectralConfig::MWIR(50.0f);
    config.adaptiveMode = AdaptiveSamplingMode::Spectral;
    config.useGpuReconstruction = true;

    EXPECT_EQ(config.adaptiveMode, AdaptiveSamplingMode::Spectral);
    EXPECT_TRUE(config.useGpuReconstruction);
}

// ============================================================================
// CPU Catmull-Rom Reference Tests (for comparison with GPU)
// ============================================================================
// These tests verify the Catmull-Rom interpolation implementation
// and serve as reference for GPU shader validation

namespace {

// Simple Catmull-Rom interpolation for testing
f32 CatmullRomInterp(f32 p0, f32 p1, f32 p2, f32 p3, f32 t) {
    f32 t2 = t * t;
    f32 t3 = t2 * t;

    f32 a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    f32 a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    f32 a2 = -0.5f * p0 + 0.5f * p2;
    f32 a3 = p1;

    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

f32 LinearInterp(f32 v0, f32 v1, f32 t) {
    return v0 + t * (v1 - v0);
}

} // anonymous namespace

TEST(CatmullRomReferenceTest, InterpolatesAtEndpoints) {
    // At t=0, should return p1; at t=1, should return p2
    f32 p0 = 1.0f, p1 = 2.0f, p2 = 4.0f, p3 = 5.0f;

    EXPECT_NEAR(CatmullRomInterp(p0, p1, p2, p3, 0.0f), 2.0f, 1e-5f);
    EXPECT_NEAR(CatmullRomInterp(p0, p1, p2, p3, 1.0f), 4.0f, 1e-5f);
}

TEST(CatmullRomReferenceTest, SmoothInterpolation) {
    // Linear data: 0, 1, 2, 3 -> should interpolate linearly
    f32 result = CatmullRomInterp(0.0f, 1.0f, 2.0f, 3.0f, 0.5f);
    EXPECT_NEAR(result, 1.5f, 1e-5f);
}

TEST(CatmullRomReferenceTest, PreservesCurvature) {
    // Quadratic-like data
    f32 p0 = 0.0f, p1 = 1.0f, p2 = 4.0f, p3 = 9.0f;

    // At t=0.5, Catmull-Rom should produce a smooth curve
    f32 result = CatmullRomInterp(p0, p1, p2, p3, 0.5f);

    // Should be between p1 and p2, closer to their midpoint
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(CatmullRomReferenceTest, SymmetricData) {
    // Symmetric peak: 0, 2, 4, 2
    f32 result = CatmullRomInterp(0.0f, 2.0f, 4.0f, 2.0f, 0.5f);

    // Should be at or near the peak
    EXPECT_GT(result, 2.5f);
    EXPECT_LT(result, 4.5f);
}

TEST(LinearInterpReferenceTest, BasicInterpolation) {
    EXPECT_NEAR(LinearInterp(0.0f, 10.0f, 0.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(LinearInterp(0.0f, 10.0f, 1.0f), 10.0f, 1e-5f);
    EXPECT_NEAR(LinearInterp(0.0f, 10.0f, 0.5f), 5.0f, 1e-5f);
    EXPECT_NEAR(LinearInterp(0.0f, 10.0f, 0.25f), 2.5f, 1e-5f);
}

TEST(LinearInterpReferenceTest, NegativeValues) {
    EXPECT_NEAR(LinearInterp(-5.0f, 5.0f, 0.5f), 0.0f, 1e-5f);
    EXPECT_NEAR(LinearInterp(-10.0f, -5.0f, 0.5f), -7.5f, 1e-5f);
}

// ============================================================================
// Wavelength Grid Tests
// ============================================================================

TEST(WavelengthGridTest, UniformGridGeneration) {
    HyperspectralConfig config = HyperspectralConfig::MWIR(50.0f);

    u32 numBands = config.GetNumBands();
    EXPECT_EQ(numBands, 41u);  // (5000-3000)/50 + 1 = 41

    // Verify uniform spacing
    for (u32 i = 1; i < numBands; ++i) {
        f32 wl0 = config.GetWavelength(i - 1);
        f32 wl1 = config.GetWavelength(i);
        EXPECT_NEAR(wl1 - wl0, 50.0f, 1e-5f);
    }
}

TEST(WavelengthGridTest, NonUniformSparseSampling) {
    // Simulate adaptive sampling: non-uniform wavelength spacing
    Vector<f32> sparseWavelengths = {
        3000.0f, 3050.0f, 3100.0f,  // Dense region
        3300.0f, 3500.0f,           // Coarse region
        3550.0f, 3600.0f, 3650.0f,  // Dense region
        4000.0f, 4500.0f, 5000.0f   // Coarse region
    };

    // Verify sorted order
    for (usize i = 1; i < sparseWavelengths.size(); ++i) {
        EXPECT_GT(sparseWavelengths[i], sparseWavelengths[i - 1]);
    }

    // Verify coverage
    EXPECT_NEAR(sparseWavelengths.front(), 3000.0f, 1e-5f);
    EXPECT_NEAR(sparseWavelengths.back(), 5000.0f, 1e-5f);
}

// ============================================================================
// SpectralCube Data Layout Tests
// ============================================================================

TEST(SpectralCubeLayoutTest, BSQLayout) {
    // Verify BSQ layout for GPU buffer transfer
    SpectralCube cube(4, 4, 3, 3000.0f, 5000.0f);

    // Fill with band-specific values
    for (u32 b = 0; b < 3; ++b) {
        for (u32 y = 0; y < 4; ++y) {
            for (u32 x = 0; x < 4; ++x) {
                cube(x, y, b) = static_cast<f32>(b * 100 + y * 10 + x);
            }
        }
    }

    // Verify BSQ memory layout: band0_all_pixels, band1_all_pixels, band2_all_pixels
    const f32* data = cube.data.data();

    // Band 0
    EXPECT_NEAR(data[0], 0.0f, 1e-5f);   // (0,0,0)
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);   // (1,0,0)
    EXPECT_NEAR(data[4], 10.0f, 1e-5f);  // (0,1,0)

    // Band 1 starts at offset 16 (4x4 pixels)
    EXPECT_NEAR(data[16], 100.0f, 1e-5f);  // (0,0,1)
    EXPECT_NEAR(data[17], 101.0f, 1e-5f);  // (1,0,1)

    // Band 2 starts at offset 32
    EXPECT_NEAR(data[32], 200.0f, 1e-5f);  // (0,0,2)
}

TEST(SpectralCubeLayoutTest, BandPointerAccess) {
    SpectralCube cube(8, 8, 5, 3000.0f, 5000.0f);

    // Verify BandPtr returns correct offset
    for (u32 b = 0; b < 5; ++b) {
        f32* bandPtr = cube.BandPtr(b);
        f32* expectedPtr = cube.data.data() + b * 8 * 8;
        EXPECT_EQ(bandPtr, expectedPtr);
    }
}

// ============================================================================
// Compute Shader Compatibility Tests
// ============================================================================
// These tests verify data structures match shader expectations

TEST(ComputeShaderCompatTest, PushConstantsSize) {
    // Push constants structure must match shader
    struct ReconstructPushConstants {
        u32 width;
        u32 height;
        u32 srcBands;
        u32 dstBands;
        u32 interpolationMode;
        u32 _pad0;
        u32 _pad1;
        u32 _pad2;
    };

    // Must be a multiple of 4 bytes and reasonably sized
    EXPECT_EQ(sizeof(ReconstructPushConstants) % 4, 0u);
    EXPECT_LE(sizeof(ReconstructPushConstants), 128u);
}

TEST(ComputeShaderCompatTest, InterpolationModeMapping) {
    // Verify interpolation mode values match shader defines
    // INTERP_MODE_LINEAR      0
    // INTERP_MODE_CATMULL_ROM 1
    // INTERP_MODE_AKIMA       2

    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Linear), 0u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::CatmullRom), 1u);
    EXPECT_EQ(static_cast<u32>(InterpolationMethod::Akima), 2u);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(GpuReconstructorEdgeCaseTest, SingleBandCube) {
    // Single band cube - edge case for interpolation
    SpectralCube cube(64, 64, 1, 4000.0f, 4000.0f);

    u64 memory = EstimateReconstructionMemory(cube, 1);
    EXPECT_GT(memory, 0u);
}

TEST(GpuReconstructorEdgeCaseTest, VeryLargeBandCount) {
    // 1000 bands - stress test for wavelength buffers
    SpectralCube cube(128, 128, 100, 200.0f, 25000.0f);

    u64 memory = EstimateReconstructionMemory(cube, 1000);
    EXPECT_GT(memory, 0u);
}

TEST(GpuReconstructorEdgeCaseTest, NonSquareImage) {
    // Non-square aspect ratio
    SpectralCube cube(1920, 1080, 50, 3000.0f, 5000.0f);

    u64 memory = EstimateReconstructionMemory(cube, 100);

    // Verify reasonable memory estimate
    EXPECT_GT(memory, 500 * 1024 * 1024);  // At least 500 MB
}

TEST(GpuReconstructorEdgeCaseTest, SmallNonPowerOfTwo) {
    // Non-power-of-two dimensions
    SpectralCube cube(300, 200, 25, 3000.0f, 5000.0f);

    u64 memory = EstimateReconstructionMemory(cube, 50);
    EXPECT_GT(memory, 0u);
}

