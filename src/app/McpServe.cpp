#include "McpServe.hpp"

#include "RenderJob.hpp"

#include "core/Config.hpp"
#include "core/Log.hpp"
#include "mcp/McpServer.hpp"

#include "Version.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <mutex>

namespace quantiloom::app {
namespace {

using json = nlohmann::json;

std::atomic<bool> g_running{true};
std::condition_variable g_wake;
std::mutex g_wakeMutex;

void HandleInterrupt(int) {
    g_running = false;
    g_wake.notify_all();
}

/// Both tools take a configuration the same two ways, so they read it the same
/// way. Relative paths inside the document resolve against the process working
/// directory either way -- the same convention the one-shot CLI has always had.
Result<Config, String> ConfigFromArguments(const json& args) {
    const bool hasInline = args.contains("config_toml") && args["config_toml"].is_string();
    const bool hasPath = args.contains("config_path") && args["config_path"].is_string();

    if (hasInline == hasPath) {
        return Result<Config, String>::Err(
            hasInline ? "Give either config_toml or config_path, not both."
                      : "Give one of config_toml (the document itself) or config_path (a .toml "
                        "file on this machine).");
    }

    if (hasInline) {
        return Config::Parse(args["config_toml"].get<String>());
    }
    return Config::Load(std::filesystem::path(args["config_path"].get<String>()));
}

// ============================================================================
// ql_render
// ============================================================================

mcp::ToolResult RenderTool(const String& argumentsJson, const String& atmosFallback) {
    json args;
    try {
        args = json::parse(argumentsJson);
    } catch (const json::exception& e) {
        return mcp::ToolResult::Error(String("arguments are not valid JSON: ") + e.what());
    }

    auto config = ConfigFromArguments(args);
    if (!config.has_value()) {
        return mcp::ToolResult::Error(config.error());
    }

    const RenderOutcome outcome = RenderConfigToFiles(config.value(), atmosFallback);

    json report;
    report["ok"] = outcome.ok;
    report["width"] = outcome.width;
    report["height"] = outcome.height;
    report["spp"] = outcome.spp;
    report["spectral_mode"] = outcome.modeName;
    report["wavelength_nm"] = outcome.wavelengthNm;
    report["seconds"] = outcome.seconds;
    report["exr_path"] = outcome.exrPath;
    if (!outcome.pngPath.empty()) {
        report["png_path"] = outcome.pngPath;
    }
    if (!outcome.tappPath.empty()) {
        report["apparent_temperature_path"] = outcome.tappPath;
    }
    if (outcome.wroteItsOwnOutput) {
        report["note"] =
            "A hyperspectral cube streams to disk band by band; there is no single frame to show.";
    }
    if (!outcome.error.empty()) {
        report["error"] = outcome.error;
    }

    mcp::ToolResult result;
    result.text = report.dump(2);
    result.isError = !outcome.ok;

    // The EXR on disk keeps the physical radiance; what comes back here is the
    // same stretched preview that was written as PNG, which is the one worth
    // looking at.
    if (outcome.preview.width > 0) {
        const u32 maxDim = args.value("max_dimension", 768u);
        result.imageBase64 = mcp::EncodeImageContent(outcome.preview, maxDim);
        result.imageMimeType = "image/png";
    }

    return result;
}

constexpr const char* kRenderSchema = R"({
  "type": "object",
  "properties": {
    "config_toml": {
      "type": "string",
      "description": "The scene configuration as a TOML document. Relative paths inside it resolve against the server's working directory."
    },
    "config_path": {
      "type": "string",
      "description": "Path to a .toml scene configuration on this machine. Use instead of config_toml, not as well as."
    },
    "max_dimension": {
      "type": "integer",
      "description": "Long edge of the returned preview image, in pixels. Default 768. Larger costs proportionally more context.",
      "minimum": 64,
      "maximum": 2048
    }
  }
})";

constexpr const char* kRenderDescription =
    "Render a scene configuration and return the frame.\n"
    "\n"
    "Each call is independent: a device is created, one scene is traced, and the device is torn "
    "down. Nothing carries over between calls, so the configuration must say everything about the "
    "render.\n"
    "\n"
    "Writes an EXR of the physical radiance to the path in renderer.output, plus a PNG preview for "
    "the fused modes, and returns a downsampled copy of that preview as an image along with the "
    "resolution, sample count, spectral mode and elapsed time.\n"
    "\n"
    "Slow, and honestly so: minutes for a first call while shaders compile, seconds afterwards, "
    "then however long the sample count takes. Call ql_validate_config first if the configuration "
    "is new -- it catches the mistakes that would otherwise cost a full render to discover.";

// ============================================================================
// ql_validate_config
// ============================================================================

/// Checks a path a configuration refers to, and says so if it is not there.
void CheckPath(const Config& config, const String& key, json& problems) {
    if (!config.Has(key)) {
        return;
    }
    const String path = config.Get<String>(key, "");
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(path), ec)) {
        problems.push_back(key + " names '" + path +
                           "', which does not exist relative to the server's working directory (" +
                           std::filesystem::current_path(ec).string() + ").");
    }
}

mcp::ToolResult ValidateTool(const String& argumentsJson) {
    json args;
    try {
        args = json::parse(argumentsJson);
    } catch (const json::exception& e) {
        return mcp::ToolResult::Error(String("arguments are not valid JSON: ") + e.what());
    }

    auto configResult = ConfigFromArguments(args);
    if (!configResult.has_value()) {
        // A TOML syntax error lands here too, which is the most common thing
        // this tool is asked about.
        json report;
        report["ok"] = false;
        report["problems"] = json::array({configResult.error()});
        mcp::ToolResult result;
        result.text = report.dump(2);
        return result;
    }
    const Config& config = configResult.value();

    json problems = json::array();

    const String modeName = config.Get<String>("spectral.mode", "rgb");
    if (!ParseSpectralMode(modeName).has_value()) {
        problems.push_back("spectral.mode is '" + modeName +
                           "'. Supported: single, rgb, vis_fused, vis_hero, nir_fused, "
                           "swir_fused, mwir_fused, lwir_fused, multispectral.");
    }

    const bool hasGeometry = config.Has("scene.preset") || config.Has("scene.gltf") ||
                             config.Has("scene.usd");
    if (!hasGeometry) {
        problems.push_back(
            "[scene] names no geometry. Give one of scene.preset, scene.gltf or scene.usd.");
    }
    CheckPath(config, "scene.gltf", problems);
    CheckPath(config, "scene.usd", problems);
    CheckPath(config, "renderer.environment_map", problems);
    CheckPath(config, "lighting.solar_lut", problems);
    CheckPath(config, "atmosphere.model_pack", problems);

    if (config.Has("renderer.spp") && config.Get<i32>("renderer.spp", 1) < 1) {
        problems.push_back("renderer.spp must be at least 1.");
    }

    json notes = json::array();
    if (config.Has("atmosphere.preset") && !config.Has("atmosphere.model_pack")) {
        notes.push_back(
            "atmosphere.preset is set without atmosphere.model_pack; the server will look for the "
            "weights where it was told to at start-up.");
    }
    if (!config.Has("renderer.output")) {
        notes.push_back("renderer.output is absent; the render will use its default filename.");
    }

    json report;
    report["ok"] = problems.empty();
    report["problems"] = std::move(problems);
    report["notes"] = std::move(notes);
    report["checked"] =
        "TOML syntax, spectral mode name, scene geometry keys, referenced file paths, sample count";

    mcp::ToolResult result;
    result.text = report.dump(2);
    return result;
}

constexpr const char* kValidateSchema = R"({
  "type": "object",
  "properties": {
    "config_toml": {
      "type": "string",
      "description": "The scene configuration as a TOML document."
    },
    "config_path": {
      "type": "string",
      "description": "Path to a .toml scene configuration on this machine. Use instead of config_toml, not as well as."
    }
  }
})";

constexpr const char* kValidateDescription =
    "Check a scene configuration without rendering it. Costs nothing and touches no GPU.\n"
    "\n"
    "Reports TOML syntax errors, an unrecognised spectral.mode, a [scene] section that names no "
    "geometry, and referenced files that are not where the configuration says they are.\n"
    "\n"
    "This is a static check, not a rehearsal: it does not load the scene, resolve materials or "
    "bake an atmosphere, so a configuration that passes here can still fail in ql_render for a "
    "reason only the renderer can see. Use it to catch the cheap mistakes first.";

}  // namespace

int RunMcpServer(const u16 port, const String& atmosphereModelPackFallback) {
    mcp::ServerOptions options;
    options.port = port;
    options.serverName = "quantiloom-cli";
    options.serverVersion = version::AppVersionString;
    options.onCommandQueued = [] { g_wake.notify_all(); };

    auto serverResult = mcp::Server::Create(options);
    if (!serverResult.has_value()) {
        QL_LOG_ERROR("{}", serverResult.error());
        std::cerr << "Error: " << serverResult.error() << "\n";
        return 1;
    }
    mcp::Server& server = *serverResult.value();

    mcp::ToolDef render;
    render.name = "ql_render";
    render.description = kRenderDescription;
    render.inputSchemaJson = kRenderSchema;
    render.timeoutMs = 900000;  // shader compilation on a cold cache is minutes
    render.handler = [&atmosphereModelPackFallback](const String& args) {
        return RenderTool(args, atmosphereModelPackFallback);
    };
    if (auto added = server.RegisterTool(render); !added.has_value()) {
        QL_LOG_ERROR("{}", added.error());
        return 1;
    }

    mcp::ToolDef validate;
    validate.name = "ql_validate_config";
    validate.description = kValidateDescription;
    validate.inputSchemaJson = kValidateSchema;
    validate.readOnly = true;
    validate.handler = [](const String& args) { return ValidateTool(args); };
    if (auto added = server.RegisterTool(validate); !added.has_value()) {
        QL_LOG_ERROR("{}", added.error());
        return 1;
    }

    std::signal(SIGINT, HandleInterrupt);
    std::signal(SIGTERM, HandleInterrupt);

    std::cout << "Quantiloom MCP server on http://127.0.0.1:" << server.Port() << "/mcp\n"
              << "Tools: ql_render, ql_validate_config\n"
              << "Press Ctrl-C to stop.\n";
    std::cout.flush();
    QL_LOG_INFO("MCP serve mode ready on port {}", server.Port());

    // One thread, and it is this one: a render owns the GPU for its whole
    // duration, so there would be nothing for a second one to do.
    while (g_running) {
        server.Pump();

        std::unique_lock<std::mutex> lock(g_wakeMutex);
        g_wake.wait_for(lock, std::chrono::milliseconds(200));
    }

    std::cout << "Stopping.\n";
    server.Stop();
    return 0;
}

}  // namespace quantiloom::app
