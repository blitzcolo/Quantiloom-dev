#include "core/Log.hpp"
#include "core/Config.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "core/LibVersion.hpp"
#include "core/Image.hpp"
#include "core/SpectralCube.hpp"
#include "core/LUT.hpp"
#include "io/ImageIO.hpp"
#include "io/SpectralIO.hpp"
#include "io/LUTLoader.hpp"
#include "Version.hpp"

#include <iostream>
#include <filesystem>

using namespace quantiloom;

// ============================================================================
// Quantiloom Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // ========================================================================
    // Initialize Logging System
    // ========================================================================
    Log::Init("quantiloom.log", Log::Level::Info);

    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Quantiloom v{}", version::AppVersionString);
    QL_LOG_INFO("  Unified Spectral Path Tracing System");
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("Application Version: {}", version::AppVersionString);
    QL_LOG_INFO("libQuantiloom Version: {}", version::LibVersionString);
    QL_LOG_INFO("Platform: {}", GetPlatformName());
    QL_LOG_INFO("Compiler: {}", GetCompilerName());
    QL_LOG_INFO("Config:   {}", GetBuildConfig());
    QL_LOG_INFO("C++ Standard: C++{}", __cplusplus);

    // ========================================================================
    // Load Configuration (if provided)
    // ========================================================================
    if (argc < 2) {
        QL_LOG_WARN("No configuration file provided.");
        QL_LOG_INFO("Usage: {} <config.toml>", argv[0]);
        QL_LOG_INFO("Running in demo mode...");
    } else {
        std::filesystem::path configPath(argv[1]);
        QL_LOG_INFO("Loading configuration: {}", configPath.string());

        auto configResult = Config::Load(configPath);
        if (configResult) {
            Config config = std::move(*configResult);
            QL_LOG_INFO("Configuration loaded successfully!");

            // Demonstrate config parsing
            config.Print();

            // Example: Read some values
            if (config.Has("renderer.resolution")) {
                auto resolution = config.GetArray<i32>("renderer.resolution");
                if (resolution.size() == 2) {
                    QL_LOG_INFO("Resolution: {}x{}", resolution[0], resolution[1]);
                }
            }

            if (config.Has("renderer.spp")) {
                i32 spp = config.Get<i32>("renderer.spp", 1);
                QL_LOG_INFO("Samples per pixel: {}", spp);
            }
        } else {
            QL_LOG_ERROR("Failed to load configuration: {}", configResult.error());
            Log::Shutdown();
            return 1;
        }
    }

    // ========================================================================
    // M0 Milestone: IO Module Test
    // ========================================================================
    QL_LOG_INFO("");
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  M0 I/O Module Test");
    QL_LOG_INFO("========================================");

    // Test 1: Image I/O (EXR)
    {
        QL_LOG_INFO("Test 1: Creating and writing test image (EXR)...");

        // Create a simple gradient image (RGB)
        Image testImg(256, 256, 3);
        testImg.channelNames = {"R", "G", "B"};
        testImg.metadata["description"] = "Test gradient image";
        testImg.metadata["mode"] = "M0_test";

        for (u32 y = 0; y < testImg.height; ++y) {
            for (u32 x = 0; x < testImg.width; ++x) {
                testImg(x, y, 0) = static_cast<f32>(x) / 255.0f;  // R
                testImg(x, y, 1) = static_cast<f32>(y) / 255.0f;  // G
                testImg(x, y, 2) = 0.5f;                           // B
            }
        }

        std::filesystem::path testPath = "test_output.exr";
        if (ImageIO::WriteEXR(testPath.string(), testImg)) {
            QL_LOG_INFO("  [OK] Wrote test image to {}", testPath.string());

            // Try reading it back
            auto readImg = ImageIO::ReadEXR(testPath.string());
            if (readImg && readImg->IsValid()) {
                QL_LOG_INFO("  [OK] Read back image: {}x{} with {} channels",
                           readImg->width, readImg->height, readImg->channels);
            } else {
                QL_LOG_ERROR("  [FAIL] Failed to read back image");
            }
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to write test image");
        }
    }

    // Test 2: Spectral Cube I/O (HDF5)
    {
        QL_LOG_INFO("Test 2: Creating and writing test spectral cube (HDF5)...");

        SpectralCube testCube(64, 64, 10, 400.0f, 700.0f);
        testCube.metadata["description"] = "Test hyperspectral cube";

        // Fill with wavelength-dependent data
        for (u32 b = 0; b < testCube.nbands; ++b) {
            f32 lambda = testCube.wavelengths[b];
            for (u32 y = 0; y < testCube.height; ++y) {
                for (u32 x = 0; x < testCube.width; ++x) {
                    testCube(x, y, b) = (lambda / 700.0f) * (x + y) / 128.0f;
                }
            }
        }

        std::filesystem::path cubePath = "test_cube.h5";
        if (SpectralIO::WriteHDF5(cubePath.string(), testCube)) {
            QL_LOG_INFO("  [OK] Wrote test cube to {}", cubePath.string());

            auto readCube = SpectralIO::ReadHDF5(cubePath.string());
            if (readCube && readCube->IsValid()) {
                QL_LOG_INFO("  [OK] Read back cube: {}x{}x{} bands",
                           readCube->width, readCube->height, readCube->nbands);
            } else {
                QL_LOG_ERROR("  [FAIL] Failed to read back cube");
            }
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to write test cube");
        }
    }

    // Test 3: LUT generation and loading
    {
        QL_LOG_INFO("Test 3: Testing LUT I/O...");
        QL_LOG_INFO("  Run: python scripts/utils/generate_dummy_lut.py -o test_lut.h5");
        QL_LOG_INFO("  Then: Load test_lut.h5 to verify");

        // Check if test LUT exists (user needs to generate it first)
        std::filesystem::path lutPath = "test_lut.h5";
        if (std::filesystem::exists(lutPath)) {
            auto lut = LUTLoader::LoadHDF5(lutPath.string());
            if (lut && lut->IsValid()) {
                QL_LOG_INFO("  [OK] Loaded LUT with {} wavelength samples", lut->Size());
                QL_LOG_INFO("       Wavelength range: {:.1f} - {:.1f} nm",
                           lut->wavelengths.front(), lut->wavelengths.back());
            } else {
                QL_LOG_ERROR("  [FAIL] Failed to load LUT");
            }
        } else {
            QL_LOG_WARN("  [SKIP] test_lut.h5 not found. Run Python script first.");
        }
    }

    QL_LOG_INFO("");
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  M0 Milestone Checklist");
    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  [x] CMake build system");
    QL_LOG_INFO("  [x] CPM.cmake dependency management");
    QL_LOG_INFO("  [x] Cross-platform compilation (C++20)");
    QL_LOG_INFO("  [x] Logging system (spdlog)");
    QL_LOG_INFO("  [x] Configuration loader (TOML++)");
    QL_LOG_INFO("  [x] EXR I/O (OpenEXR)");
    QL_LOG_INFO("  [x] HDF5 I/O (HDF5 C++)");
    QL_LOG_INFO("  [x] LUT loader");
    QL_LOG_INFO("  [ ] Vulkan initialization (TODO: M1)");
    QL_LOG_INFO("  [ ] Scene loading (TODO: M1)");
    QL_LOG_INFO("");

    QL_LOG_INFO("M0 completed! Ready for M1 (Vulkan RT + basic rendering)");

    // ========================================================================
    // Shutdown
    // ========================================================================
    Log::Shutdown();
    return 0;
}
