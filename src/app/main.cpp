// ============================================================================
// Quantiloom - Spectral Path Tracer
// ============================================================================
// Main entry point for Quantiloom spectral rendering system
// Supports single-wavelength and multi-wavelength rendering modes
//
// This file is a host, not a renderer. It reads the command line, loads a
// scene configuration, hands it to OfflineRenderer, and writes what comes
// back. The device, the acceleration structures, the pipeline and the render
// loop used to live here as ~1300 lines driving library internals directly --
// a second orchestrator running in parallel with the one inside
// ExternalRenderContext. They are one now, behind the public API.
// ============================================================================

#include "core/Log.hpp"
#include "core/Config.hpp"
#include "core/Image.hpp"
#include "io/ImageIO.hpp"
#include "renderer/OfflineRenderer.hpp"
#include "postprocess/GenericSensor.hpp"
#include "postprocess/PostprocessConfig.hpp"

#include "Version.hpp"

#include <iostream>
#include <filesystem>
#include <algorithm>  // For std::nth_element
#include <cstdio>
#include <cstring>
#include <cstdlib>  // For std::getenv
#include <vector>

using namespace quantiloom;

// Everything from here to main() is private to this translation unit. The
// anonymous namespace gives it internal linkage without repeating `static`.
namespace {


// ============================================================================
// Main Entry Point
// ============================================================================

void PrintVersion() {
    std::cout << "Quantiloom " << version::AppVersionString << "\n";
}

// Locates the atmosphere network weights when a scene names a preset but no
// atmosphere.model_pack. The pack is a deployment artifact -- it is installed
// with the library, ships in the SDK, and is copied next to Studio's
// executable -- so requiring every scene to spell out a path to it made the
// CLI refuse atmospheres that Studio rendered happily.
//
// Quantiloom-Qt has its own equivalent in QuantiloomVulkanRenderer, reading
// the same environment variable; the two are deliberately host-side, since
// where a host finds its assets is not the library's business.
std::string ResolveDefaultAtmosModelPack(const char* argv0) {
    namespace fs = std::filesystem;

    std::vector<fs::path> candidates;
    if (const char* env = std::getenv("QUANTILOOM_ATMOS_MODELS");
        env && *env) {
        candidates.emplace_back(env);
    }
    candidates.emplace_back("assets/atmos_models");  // Run from the repo root
    if (argv0 && *argv0) {
        std::error_code ec;
        const fs::path exeDir = fs::absolute(fs::path(argv0), ec).parent_path();
        if (!ec) candidates.push_back(exeDir / "assets" / "atmos_models");
    }

    for (const fs::path& dir : candidates) {
        std::error_code ec;
        if (fs::is_directory(dir, ec)) return dir.string();
    }
    return {};
}

void PrintBuildInfo() {
    std::cout
        << "Quantiloom - Spectral Path Tracer\n"
        << "  Version:    " << version::AppVersionString << "\n"
        << "  Built:      " << version::BuildTimestamp << "\n"
        << "  Compiler:   " << version::CompilerId << " " << version::CompilerVer << "\n"
        << "  Platform:   " << version::Platform << " (" << version::Arch << ")\n"
        << "  C++:        C++" << version::CxxStandard << "\n"
        << "  Build type: " << version::BuildType << "\n";
}

void PrintHelp(const char* progname) {
    PrintBuildInfo();
    std::cout
        << "\n"
        << "Usage:\n"
        << "  " << progname << " <config.toml> [options]\n"
        << "  " << progname << " --help\n"
        << "  " << progname << " --version\n"
        << "\n"
        << "Options:\n"
        << "  <config.toml>          Scene configuration file (required)\n"
        << "  -h, --help             Show this help message and exit\n"
        << "  -v, --version          Show version number and exit\n"
        << "  -V, --build-info       Show full build information and exit\n"
        << "\n"
        << "Spectral modes (set in config file [spectral] section):\n"
        << "  rgb                    Standard RGB rendering\n"
        << "  single                 Single-wavelength monochromatic rendering\n"
        << "  vis_fused              Visible band spectral integration (380-780 nm)\n"
        << "  swir_fused             Short-wave infrared (900-1700 nm)\n"
        << "  mwir_fused             Mid-wave infrared (3000-5000 nm)\n"
        << "  lwir_fused             Long-wave infrared (8000-14000 nm)\n"
        << "  multispectral          Hyperspectral data cube output\n"
        << "\n"
        << "Examples:\n"
        << "  " << progname << " assets/configs/cornell_box_vis.toml\n"
        << "  " << progname << " assets/configs/cube_usdc.toml\n"
        << "  " << progname << " assets/configs/cornell_box_lwir.toml\n"
        << "\n"
        << "Homepage: https://github.com/blitzcolo/Quantiloom-dev\n";
}

// The real entry point. main() below is only the last-resort exception barrier;
// everything that needs the logger lives here, behind its own handler.
int RunApp(int argc, char* argv[]) {
    // ========================================================================
    // Command-Line Flags (before logging init — pure stdout)
    // ========================================================================
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            PrintHelp(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--version") == 0) {
            PrintVersion();
            return 0;
        }
        if (std::strcmp(argv[i], "-V") == 0 || std::strcmp(argv[i], "--build-info") == 0) {
            PrintBuildInfo();
            return 0;
        }
    }

    // ========================================================================
    // No arguments → show help and exit (before logging init)
    // ========================================================================
    if (argc < 2) {
        std::cerr << "Error: no configuration file provided.\n\n";
        PrintHelp(argv[0]);
        return 1;
    }

    // ========================================================================
    // Initialize Logging
    // ========================================================================
    Log::Init("quantiloom.log", Log::Level::Info);

    QL_LOG_INFO("========================================");
    QL_LOG_INFO("  Quantiloom Spectral Path Tracer v{}", version::AppVersionString);
    QL_LOG_INFO("  {} {} | {} ({})", version::CompilerId, version::CompilerVer,
                version::Platform, version::Arch);
    QL_LOG_INFO("========================================");

    std::filesystem::path configPath(argv[1]);
    QL_LOG_INFO("Loading configuration: {}", configPath.string());

    auto configResult = Config::Load(configPath);
    if (!configResult.has_value()) {
        QL_LOG_ERROR("Failed to load configuration: {}", configResult.error());
        Log::Shutdown();
        return 1;
    }

    Config config = configResult.value();
    QL_LOG_INFO("Configuration loaded successfully");

    try {
        // ====================================================================
        // Render
        // ====================================================================
        // Everything from the Vulkan device to the frame readback lives behind
        // OfflineRenderer now. What stays here is what a *host* does: decide
        // where the assets are, and write the files.
        OfflineRenderer::InitParams initParams;
        initParams.atmosphereModelPackFallback = ResolveDefaultAtmosModelPack(argv[0]);

        auto rendererResult = OfflineRenderer::Create(config, initParams);
        if (!rendererResult.has_value()) {
            QL_LOG_ERROR("{}", rendererResult.error());
            Log::Shutdown();
            return 1;
        }
        OfflineRenderer& renderer = *rendererResult.value();

        // Read rather than re-parsed. Two readers of one TOML key is the bug
        // class this project keeps closing.
        const u32 width = renderer.Params().width;
        const u32 height = renderer.Params().height;
        const SpectralMode spectral_mode = renderer.Params().mode;
        const String spectralModeStr = renderer.Params().modeName;
        const f32 wavelength_nm = renderer.Params().wavelengthNm;
        const String outputPath = renderer.Params().outputPath;

        OfflineRenderOutput rendered = renderer.Render();
        if (!rendered.error.empty()) {
            QL_LOG_ERROR("{}", rendered.error);
        }

        // The hyperspectral cube streams itself to disk band by band; there is
        // no frame to save. Everything below is the single-frame output stage.
        if (!rendered.wroteItsOwnOutput) {
        Image& img = rendered.radiance;
        // ====================================================================
        // Sensor Simulation (Optional Postprocessing)
        // ====================================================================
        if (PostprocessConfig::IsSensorEnabled(config)) {
            QL_LOG_INFO("Applying sensor simulation...");

            // Parse sensor parameters from config
            SensorParams sensorParams = PostprocessConfig::ParseSensorParams(config);

            // ================================================================
            // IR fused modes: unit fixup for the sensor photon budget
            // ================================================================
            // The renderer stores per-nm AVERAGE spectral radiance
            // (band integral / band width, see closesthit.rchit) while the
            // sensor chain expects band-INTEGRATED radiance (W/sr/m^2).
            // Multiply by the band width here, and use the band center for
            // photon energy instead of the 550 nm visible-light default.
            //
            // Both numbers come back with the frame rather than being derived
            // here: the renderer is the only party that knows which band it
            // just integrated.
            const f32 bandScale = rendered.sensorRadianceScale;
            if (rendered.sensorWavelengthNm > 0.0f) {
                sensorParams.wavelength_nm = rendered.sensorWavelengthNm;
            }
            if (bandScale != 1.0f) {
                QL_LOG_INFO("  IR sensor units: radiance x{:.0f} nm bandwidth, photon wavelength {:.0f} nm",
                            bandScale, sensorParams.wavelength_nm);
            }

            // Create sensor model
            GenericSensor sensor;

            // Extract RGB/grayscale channels (drop alpha for sensor simulation)
            Image hdrInput(width, height, 3);
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    hdrInput(x, y, 0) = img(x, y, 0) * bandScale;  // R
                    hdrInput(x, y, 1) = img(x, y, 1) * bandScale;  // G
                    hdrInput(x, y, 2) = img(x, y, 2) * bandScale;  // B
                }
            }

            // Apply sensor chain
            auto sensorResult = sensor.Apply(hdrInput, sensorParams);
            if (!sensorResult.has_value()) {
                QL_LOG_ERROR("  [FAIL] Sensor simulation failed: {}", sensorResult.error());
            } else {
                QL_LOG_INFO("  [OK] Sensor simulation complete");

                const SensorOutput& sensorOutput = sensorResult.value();

                // Replace image with enhanced preview (noisy radiance, for PNG/visualization).
                // Divide the band scale back out so the EXR keeps the same
                // per-nm average radiance units as the sensor-off path.
                const Image& enhancedPreview = sensorOutput.enhancedPreview;
                const f32 invBandScale = 1.0f / bandScale;
                for (u32 y = 0; y < height; ++y) {
                    for (u32 x = 0; x < width; ++x) {
                        img(x, y, 0) = enhancedPreview(x, y, 0) * invBandScale;  // R
                        img(x, y, 1) = enhancedPreview(x, y, 1) * invBandScale;  // G
                        img(x, y, 2) = enhancedPreview(x, y, 2) * invBandScale;  // B
                        // Alpha unchanged
                    }
                }

                // Update metadata
                img.metadata["postprocess"] = "sensor_simulation_preview";

                // Save raw DN image to separate file
                std::filesystem::path exrPath(outputPath);
                std::string rawDnPath = (exrPath.parent_path() / (exrPath.stem().string() + "_rawdn.exr")).string();

                QL_LOG_INFO("Saving raw DN image to {}...", rawDnPath);
                if (ImageIO::WriteEXR(rawDnPath, sensorOutput.rawDN)) {
                    QL_LOG_INFO("  [OK] Saved raw DN image");
                } else {
                    QL_LOG_WARN("  [WARN] Failed to save raw DN image");
                }
            }
        } else {
            QL_LOG_INFO("Sensor simulation disabled (sensor.enabled = false)");
        }

        // Save as EXR
        if (ImageIO::WriteEXR(outputPath, img)) {
            QL_LOG_INFO("  [OK] Saved spectral image to {}", outputPath);
        } else {
            QL_LOG_ERROR("  [FAIL] Failed to save image to {}", outputPath);
        }

        // For fused modes (RGB, VIS_FUSED, MWIR, LWIR), also save PNG preview
        // These modes output both EXR (HDR/physical) and PNG (LDR preview)
        bool isFusedMode = (spectral_mode == SpectralMode::RGB ||
                           spectral_mode == SpectralMode::VIS_Fused ||
                           spectral_mode == SpectralMode::MWIR_Fused ||
                           spectral_mode == SpectralMode::LWIR_Fused ||
                           spectral_mode == SpectralMode::SWIR_Fused);

        if (isFusedMode) {
            // Generate PNG path from EXR path (replace extension)
            std::filesystem::path exrPath(outputPath);
            std::filesystem::path pngPath = exrPath.parent_path() / (exrPath.stem().string() + ".png");

            // Create RGB image for PNG (drop alpha channel)
            Image pngImg(width, height, 3);
            pngImg.channelNames = {"R", "G", "B"};

            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    pngImg(x, y, 0) = img(x, y, 0);  // R
                    pngImg(x, y, 1) = img(x, y, 1);  // G
                    pngImg(x, y, 2) = img(x, y, 2);  // B
                }
            }

            // Physical radiance does not live in [0, 1], so a raw clamp makes
            // a preview that is all black or all white. IR fused modes are far
            // below it (LWIR ~5e-3 W/sr/m^2/nm); a scene lit by a measured
            // solar spectrum is far above it, since ASTM G-173 integrates to
            // a few hundred rather than to the 5.0 someone used to type into
            // sun_radiance. Same remedy for both: stretch the 1st..99th
            // percentile to [0, 1] for the preview, and leave the EXR alone.
            //
            // Applied only when the values actually leave the range, so a
            // scene that was already displayable previews exactly as before.
            {
                std::vector<f32> values(static_cast<size_t>(width) * height);
                for (u32 y = 0; y < height; ++y) {
                    for (u32 x = 0; x < width; ++x) {
                        values[static_cast<size_t>(y) * width + x] = pngImg(x, y, 0);
                    }
                }
                const size_t loIdx = values.size() / 100;
                const size_t hiIdx = values.size() - 1 - loIdx;
                std::nth_element(values.begin(),
                                 values.begin() + static_cast<std::ptrdiff_t>(loIdx),
                                 values.end());
                const f32 lo = values[loIdx];
                std::nth_element(values.begin(),
                                 values.begin() + static_cast<std::ptrdiff_t>(hiIdx),
                                 values.end());
                const f32 hi = values[hiIdx];
                const f32 range = std::max(hi - lo, 1e-12f);

                if (IsIRFusedMode(spectral_mode) || hi > 1.0f) {
                    for (u32 y = 0; y < height; ++y) {
                        for (u32 x = 0; x < width; ++x) {
                            for (u32 c = 0; c < 3; ++c) {
                                pngImg(x, y, c) = (pngImg(x, y, c) - lo) / range;
                            }
                        }
                    }
                    QL_LOG_INFO("  PNG preview stretched from [{:.4g}, {:.4g}]; "
                                "the EXR keeps the physical values", lo, hi);
                }
                QL_LOG_INFO("  IR PNG preview normalized: [{:.4e}, {:.4e}] -> [0, 1]", lo, hi);
            }

            if (ImageIO::WritePNG(pngPath.string(), pngImg)) {
                QL_LOG_INFO("  [OK] Saved PNG preview to {}", pngPath.string());
            } else {
                QL_LOG_WARN("  [WARN] Failed to save PNG preview to {}", pngPath.string());
            }
        }

        } // End of the single-frame output stage

        // ====================================================================
        // Success
        // ====================================================================
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Rendering COMPLETED");
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Spectral mode: {}", spectralModeStr);
        if (spectral_mode == SpectralMode::Single ||
            spectral_mode == SpectralMode::MWIR_Fused ||
            spectral_mode == SpectralMode::LWIR_Fused ||
            spectral_mode == SpectralMode::SWIR_Fused) {
            QL_LOG_INFO("  Wavelength: {:.1f} nm", wavelength_nm);
        } else if (spectral_mode == SpectralMode::Multispectral) {
            QL_LOG_INFO("  Mode: Hyperspectral data cube");
        }
        QL_LOG_INFO("  Output: {}", outputPath);
        QL_LOG_INFO("========================================");

        // The pipeline cache is written back and the device torn down when
        // `renderer` goes out of scope below.
    } catch (const std::exception& e) {
        QL_LOG_ERROR("FATAL ERROR: {}", e.what());
        Log::Shutdown();
        return 1;
    } catch (...) {
        // Without this, a throw that does not derive from std::exception reaches
        // the runtime as an unhandled exception: std::terminate, no message, and
        // the log left unflushed.
        QL_LOG_ERROR("FATAL ERROR: unknown exception");
        Log::Shutdown();
        return 1;
    }

    Log::Shutdown();
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    // RunApp handles its own errors once logging is up. This barrier exists for
    // the window before that -- Log::Init throws if the log file cannot be
    // created -- and for anything escaping RunApp's own handlers.
    //
    // Reporting goes through std::fputs, not the logger (which may not exist
    // yet) and not std::cerr (whose operator<< can itself throw
    // std::ios_base::failure). A last-resort handler that can throw is not one.
    try {
        return RunApp(argc, argv);
    } catch (const std::exception& e) {
        std::fputs("FATAL ERROR: ", stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("FATAL ERROR: unknown exception\n", stderr);
        return 1;
    }
}
