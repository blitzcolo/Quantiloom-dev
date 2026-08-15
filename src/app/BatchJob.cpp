#include "BatchJob.hpp"

#include "BatchManifest.hpp"

#include "RenderJob.hpp"

#include "core/Config.hpp"
#include "core/Log.hpp"
#include "renderer/OfflineRenderer.hpp"
#include "renderer/RenderDevice.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace quantiloom::app {
namespace {

namespace fs = std::filesystem;

/// One line of the manifest, carried from parsing through to the summary.
struct Job {
    fs::path configPath;
    /// The config with the override already layered on. The output path is
    /// injected later, once collisions across the whole list are known.
    Config config;
    fs::path outputPath;

    /// Why this job never reached a renderer. Empty means it did.
    String preflightError;

    /// What this manifest line overrode, for the dry run to print. Empty when
    /// the line was a bare path, which most are.
    String overrideSummary;

    /// False for the tail of the list after --fail-fast stopped it, which is not
    /// the same as failing and is not reported as such.
    bool attempted = false;
    bool ok = false;
    String error;
    f64 seconds = 0.0;
};

String Trim(const String& s) {
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    auto begin = std::find_if(s.begin(), s.end(), notSpace);
    auto end = std::find_if(s.rbegin(), s.rend(), notSpace).base();
    return begin < end ? String(begin, end) : String{};
}

String ToLower(String s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// A TOML basic string. Paths go in as generic (forward-slash) form, so on
/// Windows there are no backslashes left to escape -- but a path may legally
/// contain a quote, so this does not assume.
String TomlQuote(const String& text) {
    String out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
    return out;
}

/**
 * @brief Where this job's EXR goes, before collisions are considered
 *
 * With an --output-dir, every config is named after itself in that directory.
 * Without one, each config keeps its own `renderer.output`, resolved against the
 * directory the config came from rather than the working directory -- so outputs
 * land beside their scenes and two scenes in different folders cannot collide by
 * both defaulting to `spectral_output.exr`.
 */
fs::path ResolveOutputPath(const Job& job, const String& outputDir) {
    if (!outputDir.empty()) {
        return fs::absolute(fs::path(outputDir) / (job.configPath.stem().string() + ".exr"))
            .lexically_normal();
    }

    // The same default ConfigResolve applies when the key is absent.
    fs::path named(job.config.GetString("renderer.output", "spectral_output.exr"));
    if (named.is_relative()) named = job.configPath.parent_path() / named;
    return fs::absolute(named).lexically_normal();
}

/**
 * @brief Give every job a distinct output path, warning about each rename
 *
 * Comparison is case-insensitive because the target is a Windows filesystem,
 * where `Out.exr` and `out.exr` are one file. The derived outputs -- the PNG
 * preview, the raw DN EXR, the hyperspectral cube -- are all named from this
 * path's stem downstream, so de-duplicating it de-duplicates those too.
 */
void ResolveOutputCollisions(std::vector<Job>& jobs) {
    std::unordered_map<String, int> taken;

    for (Job& job : jobs) {
        if (!job.preflightError.empty()) continue;

        const fs::path original = job.outputPath;
        fs::path candidate = original;
        int suffix = 1;
        while (taken.count(ToLower(candidate.string())) != 0) {
            ++suffix;
            candidate = original.parent_path() /
                        (original.stem().string() + "-" + std::to_string(suffix) +
                         original.extension().string());
        }

        if (candidate != original) {
            std::cerr << "warning: " << job.configPath.filename().string()
                      << " wanted " << original.filename().string()
                      << ", which another job in this batch already claims; writing "
                      << candidate.filename().string() << " instead\n";
            QL_LOG_WARN("Batch output collision: {} -> {}", original.string(),
                        candidate.string());
        }

        taken[ToLower(candidate.string())] = suffix;
        job.outputPath = candidate;
    }
}

/// Layer `renderer.output` onto the config, so the renderer resolves the path the
/// batch chose rather than the one the file named.
Config WithOutputPath(const Config& config, const fs::path& outputPath) {
    const String document =
        "[renderer]\noutput = " + TomlQuote(outputPath.generic_string()) + "\n";
    auto injection = Config::Parse(document);
    if (!injection.has_value()) {
        // Only reachable if a path defeats TomlQuote, which would be a bug here
        // rather than bad user input -- but silently rendering to the wrong file
        // is the one outcome worth avoiding, so say so and keep the config's own.
        QL_LOG_ERROR("Could not inject output path '{}': {}", outputPath.string(),
                     injection.error());
        return config;
    }
    return config.MergedWith(injection.value());
}

String FormatSeconds(const f64 seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << seconds << "s";
    return oss.str();
}

}  // namespace

int RunBatch(const BatchOptions& options) {
    const auto batchStarted = std::chrono::steady_clock::now();

    // ========================================================================
    // Pre-flight: everything that can be known before a device exists
    // ========================================================================
    auto manifest = ParseManifest(fs::path(options.manifestPath));
    if (!manifest.has_value()) {
        std::cerr << "Error: " << manifest.error() << "\n";
        QL_LOG_ERROR("{}", manifest.error());
        return 1;
    }

    Config overrides;
    if (!options.overridePath.empty()) {
        auto loaded = Config::Load(fs::path(options.overridePath));
        if (!loaded.has_value()) {
            std::cerr << "Error: cannot load override: " << loaded.error() << "\n";
            QL_LOG_ERROR("Cannot load override: {}", loaded.error());
            return 1;
        }
        overrides = loaded.value();

        // A single output path applied to every config is exactly the mutual
        // overwrite this command exists to prevent, so it is refused rather than
        // worked around.
        if (overrides.Has("renderer.output")) {
            std::cerr << "Error: the override sets renderer.output, which would send "
                         "every job in the batch to one file.\n"
                         "       Use --output-dir DIR to place the outputs instead.\n";
            QL_LOG_ERROR("Override sets renderer.output; refusing to run the batch");
            return 1;
        }
    }

    std::vector<Job> jobs;
    jobs.reserve(manifest.value().size());

    for (const ManifestEntry& entry : manifest.value()) {
        Job job;
        job.configPath = entry.configPath;

        if (!fs::is_regular_file(entry.configPath)) {
            job.preflightError = "no such configuration file";
            jobs.push_back(std::move(job));
            continue;
        }

        auto loaded = Config::Load(entry.configPath);
        if (!loaded.has_value()) {
            job.preflightError = loaded.error();
            jobs.push_back(std::move(job));
            continue;
        }

        // Layering order: the config, then the batch-wide --override, then
        // this line's own. The line is the most specific statement about this
        // job, so it wins -- and unlike the batch-wide override it MAY set
        // renderer.output, because naming one file per line is how a sequence
        // is written rather than a way for every job to overwrite one file.
        Config config = options.overridePath.empty() ? std::move(loaded.value())
                                                     : loaded.value().MergedWith(overrides);
        auto withEntry = ApplyEntryOverrides(config, entry);
        if (!withEntry.has_value()) {
            job.preflightError = withEntry.error();
            jobs.push_back(std::move(job));
            continue;
        }

        job.config = std::move(withEntry.value());
        job.overrideSummary = DescribeEntryOverrides(entry);
        job.outputPath = ResolveOutputPath(job, options.outputDir);
        jobs.push_back(std::move(job));
    }

    ResolveOutputCollisions(jobs);

    // Only after collisions are settled: a directory created for a path that then
    // moved would be an empty directory nobody asked for.
    for (const Job& job : jobs) {
        if (!job.preflightError.empty()) continue;
        const fs::path parent = job.outputPath.parent_path();
        if (parent.empty()) continue;
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            std::cerr << "warning: cannot create output directory " << parent.string()
                      << ": " << ec.message() << "\n";
        }
    }

    // ========================================================================
    // The job table
    // ========================================================================
    std::cout << "Batch: " << jobs.size() << " job" << (jobs.size() == 1 ? "" : "s")
              << " from " << options.manifestPath << "\n";
    if (!options.overridePath.empty()) {
        std::cout << "  Override: " << options.overridePath
                  << " (layered over every config, key by key)\n";
    }
    std::cout << "  Execution: serial, one shared GPU device\n";

    if (options.dryRun) {
        std::cout << "\nDry run -- nothing will be rendered.\n";
        for (size_t i = 0; i < jobs.size(); ++i) {
            const Job& job = jobs[i];
            std::cout << "  [" << (i + 1) << "/" << jobs.size() << "] "
                      << job.configPath.string() << "\n";
            if (!job.preflightError.empty()) {
                std::cout << "        SKIPPED: " << job.preflightError << "\n";
            } else {
                if (!job.overrideSummary.empty()) {
                    std::cout << "        overrides: " << job.overrideSummary << "\n";
                }
                std::cout << "        -> " << job.outputPath.string() << "\n";
            }
        }
        return 0;
    }

    // ========================================================================
    // The device, once
    // ========================================================================
    auto device = RenderDevice::Create();
    if (!device.has_value()) {
        std::cerr << "Error: " << device.error() << "\n";
        QL_LOG_ERROR("{}", device.error());
        return 1;
    }

    // ========================================================================
    // Render, in order
    // ========================================================================
    bool stopped = false;
    for (size_t i = 0; i < jobs.size(); ++i) {
        Job& job = jobs[i];
        job.attempted = true;

        // The progress line goes out complete, after the fact: the library logs
        // to the console as it works, so a half line printed first would come
        // back split across everything the render had to say.
        const String label = "[" + std::to_string(i + 1) + "/" +
                             std::to_string(jobs.size()) + "] " +
                             job.configPath.filename().string() + " ... ";
        QL_LOG_INFO("{}starting", label);

        if (!job.preflightError.empty()) {
            std::cout << label << "FAILED: " << job.preflightError << "\n";
            QL_LOG_ERROR("[{}/{}] {}: {}", i + 1, jobs.size(), job.configPath.string(),
                         job.preflightError);
            if (options.failFast) { stopped = true; break; }
            continue;
        }

        OfflineRenderer::InitParams init;
        init.atmosphereModelPackFallback = options.atmosphereModelPackFallback;
        init.sharedDevice = device.value().get();
        init.baseDir = job.configPath.parent_path().string();

        RenderOutcome outcome;
        try {
            outcome = RenderConfigToFiles(WithOutputPath(job.config, job.outputPath), init);
        } catch (const std::exception& e) {
            // One scene that trips a device loss or a bad allocation should not
            // take the rest of the list with it.
            outcome.ok = false;
            outcome.error = String("exception: ") + e.what();
        }

        job.ok = outcome.ok;
        job.error = outcome.error;
        job.seconds = outcome.seconds;

        if (job.ok) {
            std::cout << label << "ok " << FormatSeconds(job.seconds) << " -> "
                      << job.outputPath.string() << "\n";
            QL_LOG_INFO("[{}/{}] {} ok in {:.1f}s -> {}", i + 1, jobs.size(),
                        job.configPath.string(), job.seconds, job.outputPath.string());
        } else {
            std::cout << label << "FAILED: " << job.error << "\n";
            QL_LOG_ERROR("[{}/{}] {} FAILED: {}", i + 1, jobs.size(),
                         job.configPath.string(), job.error);
            if (options.failFast) { stopped = true; break; }
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    size_t succeeded = 0;
    size_t notRun = 0;
    std::vector<const Job*> failures;
    for (const Job& job : jobs) {
        if (!job.attempted) {
            ++notRun;
        } else if (job.ok) {
            ++succeeded;
        } else {
            failures.push_back(&job);
        }
    }

    const f64 totalSeconds =
        std::chrono::duration<f64>(std::chrono::steady_clock::now() - batchStarted).count();

    std::cout << "\n"
              << succeeded << "/" << jobs.size() << " rendered in "
              << FormatSeconds(totalSeconds);
    if (stopped) {
        std::cout << " (--fail-fast stopped the batch; " << notRun << " not run)";
    }
    std::cout << "\n";

    for (const Job* job : failures) {
        std::cout << "  FAILED  " << job->configPath.string() << ": "
                  << (job->preflightError.empty() ? job->error : job->preflightError) << "\n";
    }

    return failures.empty() ? 0 : 1;
}

}  // namespace quantiloom::app
