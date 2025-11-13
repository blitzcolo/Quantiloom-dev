#include "core/Log.hpp"
#include "core/Config.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "core/LibVersion.hpp"
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
    // Basic Infrastructure Test
    // ========================================================================
    QL_LOG_INFO("");
    QL_LOG_INFO("M0 Milestone Checklist:");
    QL_LOG_INFO("  [x] CMake build system");
    QL_LOG_INFO("  [x] CPM.cmake dependency management");
    QL_LOG_INFO("  [x] Cross-platform compilation (C++20)");
    QL_LOG_INFO("  [x] Logging system (spdlog)");
    QL_LOG_INFO("  [x] Configuration loader (TOML++)");
    QL_LOG_INFO("  [ ] Vulkan initialization (TODO: M1)");
    QL_LOG_INFO("  [ ] Scene loading (TODO: M1)");
    QL_LOG_INFO("");

    // ========================================================================
    // Future: Renderer Initialization (M1+)
    // ========================================================================
    // TODO: Initialize Vulkan, load scene, run path tracer
    QL_LOG_WARN("Renderer not yet implemented (M1+ milestone)");
    QL_LOG_INFO("M0 skeleton test completed successfully!");

    // ========================================================================
    // Shutdown
    // ========================================================================
    Log::Shutdown();
    return 0;
}
