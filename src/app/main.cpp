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

#include "BatchJob.hpp"
#include "McpServe.hpp"
#include "RenderJob.hpp"
#include "Version.hpp"

#include <iostream>
#include <filesystem>
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
        << "  " << progname << " batch <list.txt> [options]\n"
        << "  " << progname << " serve [--port N]\n"
        << "  " << progname << " --help\n"
        << "  " << progname << " --version\n"
        << "\n"
        << "Options:\n"
        << "  <config.toml>          Scene configuration file (required)\n"
        << "  batch <list.txt>       Render every config the list names, in order,\n"
        << "                         reusing one GPU device across all of them\n"
        << "  serve                  Answer MCP on 127.0.0.1 so an agent can render\n"
        << "  --port N               Port for serve mode (default 8766)\n"
        << "  -h, --help             Show this help message and exit\n"
        << "  -v, --version          Show version number and exit\n"
        << "  -V, --build-info       Show full build information and exit\n"
        << "\n"
        << "Batch options:\n"
        << "  --override FILE        A TOML document layered over every config, key by\n"
        << "                         key: keys it names win, keys it omits keep each\n"
        << "                         config's own value. Tables merge; arrays (and\n"
        << "                         [[materials]]) are replaced whole. It may not set\n"
        << "                         renderer.output -- use --output-dir instead\n"
        << "  --output-dir DIR       Write every output to DIR, named after its config.\n"
        << "                         Without it each config keeps its own\n"
        << "                         renderer.output, resolved beside that config\n"
        << "  --fail-fast            Stop at the first failure (default: finish the\n"
        << "                         list and summarise)\n"
        << "  --dry-run              Print the resolved job table and exit\n"
        << "\n"
        << "  A list is one .toml path per line; blank lines and lines starting with #\n"
        << "  are ignored, and relative paths resolve against the list's own directory.\n"
        << "  Jobs run one after another. Two configs that would write the same file\n"
        << "  get -2, -3 appended, with a warning, rather than overwriting each other.\n"
        << "\n"
        << "  A line may also carry its own overrides, after a | :\n"
        << "\n"
        << "    plate.toml | material_overrides.Plate.ir_temperature_k=320.0 \\\n"
        << "                 renderer.output=\"frame_320K.exr\"\n"
        << "    plate.toml | @noon.toml\n"
        << "\n"
        << "  key=value is a dotted TOML key and a TOML value; @file.toml is a whole\n"
        << "  document. They are layered after --override, so the line wins, and a\n"
        << "  line MAY set renderer.output -- naming one file per line is how a\n"
        << "  sequence is written. Use [material_overrides.<name>] rather than\n"
        << "  [[materials]] in an override: arrays are replaced whole, so an array\n"
        << "  override would delete every material it does not name.\n"
        << "\n"
        << "Spectral modes (set in config file [spectral] section):\n"
        << "  rgb                    Standard RGB rendering\n"
        << "  single                 Single-wavelength monochromatic rendering\n"
        << "  vis_hero               Visible band, four sampled wavelengths a path\n"
        << "                         (400-780 nm; also spelled VIS)\n"
        << "  vis_fused              The same band by 32 fixed wavelengths, with no\n"
        << "                         variance in wavelength: the reference the\n"
        << "                         sampled mode is checked against\n"
        << "  nir_fused              Near infrared (930-1200 nm)\n"
        << "  swir_fused             Short-wave infrared (1400-2400 nm)\n"
        << "  mwir_fused             Mid-wave infrared (3000-5000 nm)\n"
        << "  lwir_fused             Long-wave infrared (8000-12000 nm)\n"
        << "  multispectral          Hyperspectral data cube output\n"
        << "\n"
        << "Examples:\n"
        << "  " << progname << " assets/configs/cornell_box_vis.toml\n"
        << "  " << progname << " assets/configs/cube_usdc.toml\n"
        << "  " << progname << " assets/configs/cornell_box_lwir.toml\n"
        << "  " << progname << " batch scenes.txt --output-dir renders\n"
        << "  " << progname << " batch scenes.txt --override preview.toml --dry-run\n"
        << "  " << progname << " batch assets/configs/thermal_sequence/manifest.txt\n"
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

    // ========================================================================
    // serve → answer MCP instead of rendering one scene and exiting
    // ========================================================================
    if (std::strcmp(argv[1], "serve") == 0) {
        u16 port = 8766;  // Studio defaults to 8765; both can run at once
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
                const long parsed = std::strtol(argv[++i], nullptr, 10);
                if (parsed < 1 || parsed > 65535) {
                    std::cerr << "Error: --port must be between 1 and 65535.\n";
                    Log::Shutdown();
                    return 1;
                }
                port = static_cast<u16>(parsed);
            } else {
                std::cerr << "Error: unrecognised option for serve: " << argv[i] << "\n";
                Log::Shutdown();
                return 1;
            }
        }

        const int code = app::RunMcpServer(port, ResolveDefaultAtmosModelPack(argv[0]));
        Log::Shutdown();
        return code;
    }

    // ========================================================================
    // batch → render a list of configurations on one device
    // ========================================================================
    if (std::strcmp(argv[1], "batch") == 0) {
        if (argc < 3) {
            std::cerr << "Error: batch needs a list file.\n"
                         "Usage: " << argv[0] << " batch <list.txt> [options]\n";
            Log::Shutdown();
            return 1;
        }

        app::BatchOptions options;
        options.manifestPath = argv[2];
        options.atmosphereModelPackFallback = ResolveDefaultAtmosModelPack(argv[0]);

        for (int i = 3; i < argc; ++i) {
            if (std::strcmp(argv[i], "--override") == 0 && i + 1 < argc) {
                options.overridePath = argv[++i];
            } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
                options.outputDir = argv[++i];
            } else if (std::strcmp(argv[i], "--fail-fast") == 0) {
                options.failFast = true;
            } else if (std::strcmp(argv[i], "--dry-run") == 0) {
                options.dryRun = true;
            } else {
                std::cerr << "Error: unrecognised option for batch: " << argv[i] << "\n";
                Log::Shutdown();
                return 1;
            }
        }

        const int code = app::RunBatch(options);
        Log::Shutdown();
        return code;
    }

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
        // OfflineRenderer, and everything from there to the files on disk lives
        // in RenderJob -- shared with serve mode, so an agent's render and a
        // command-line render are the same render.
        OfflineRenderer::InitParams init;
        init.atmosphereModelPackFallback = ResolveDefaultAtmosModelPack(argv[0]);
        // Relative asset paths inside the config resolve against the config's own
        // directory first, then against the working directory as before -- so a
        // self-contained scene folder renders from anywhere, and every config in
        // assets/configs/ keeps resolving its repo-root-relative paths.
        init.baseDir = configPath.parent_path().string();

        const app::RenderOutcome outcome = app::RenderConfigToFiles(config, init);

        if (!outcome.ok && outcome.width == 0) {
            QL_LOG_ERROR("{}", outcome.error);
            Log::Shutdown();
            return 1;
        }

        // ====================================================================
        // Success
        // ====================================================================
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Rendering COMPLETED");
        QL_LOG_INFO("========================================");
        QL_LOG_INFO("  Spectral mode: {}", outcome.modeName);
        const SpectralMode spectralMode =
            ParseSpectralMode(outcome.modeName).has_value()
                ? ParseSpectralMode(outcome.modeName).value()
                : SpectralMode::RGB;
        if (spectralMode == SpectralMode::Single ||
            spectralMode == SpectralMode::MWIR_Fused ||
            spectralMode == SpectralMode::LWIR_Fused ||
            spectralMode == SpectralMode::SWIR_Fused) {
            QL_LOG_INFO("  Wavelength: {:.1f} nm", outcome.wavelengthNm);
        } else if (spectralMode == SpectralMode::Multispectral) {
            QL_LOG_INFO("  Mode: Hyperspectral data cube");
        }
        QL_LOG_INFO("  Output: {}", outcome.exrPath);
        if (!outcome.tappPath.empty()) {
            QL_LOG_INFO("  Temperature: {}", outcome.tappPath);
        }
        QL_LOG_INFO("========================================");
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
