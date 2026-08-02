/**
 * @file BatchJob.hpp
 * @brief Render a list of scene configurations in one process, on one device
 *
 * The alternative is a shell loop around the one-shot CLI, which is what
 * scripts/render-tests/capture_baseline.sh does. That pays for a Vulkan device
 * (~250 ms), a process start, and a pipeline-cache round trip per image -- against
 * ~6 ms of actual tracing for a small frame. It also leaves every config's
 * `renderer.output` to fend for itself, so two scenes that both default to
 * `spectral_output.exr` silently overwrite each other.
 *
 * This fixes both: one RenderDevice shared across the whole list, and every output
 * path resolved and de-duplicated before the first render starts.
 *
 * Serial by design. A single TraceRays dispatch is width x height threads and
 * already saturates the card, so two concurrent jobs would time-slice the same
 * execution units for the same total wall clock while paying double the VRAM --
 * and the library submits to one unsynchronised queue anyway.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::app {

/**
 * @brief Everything `batch` takes from the command line
 */
struct BatchOptions {
    /// Text file, one .toml path per line; `#` comments and blank lines skipped.
    String manifestPath;

    /// A TOML document layered over every config, key by key. Empty for none.
    String overridePath;

    /**
     * @brief Write every output into this directory instead, named after its config
     *
     * Empty keeps each config's own `renderer.output`, resolved against the
     * directory that config was loaded from.
     */
    String outputDir;

    /// Stop at the first failure instead of finishing the list and summarising.
    bool failFast = false;

    /// Resolve and print the job table, then exit without rendering.
    bool dryRun = false;

    /// Passed through to every render. @see OfflineRenderer::InitParams
    String atmosphereModelPackFallback;
};

/**
 * @brief Run the batch
 *
 * @return A process exit code: 0 if every job succeeded, 1 otherwise.
 */
int RunBatch(const BatchOptions& options);

}  // namespace quantiloom::app
