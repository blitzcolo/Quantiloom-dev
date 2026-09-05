/**
 * @file RenderJob.hpp
 * @brief One configuration in, one set of files out
 *
 * What the CLI does with a scene, factored out of main() so the MCP serve mode
 * does the same thing rather than its own version of it. The sensor unit fixup
 * for the IR bands and the percentile stretch on the preview are both subtle
 * enough that a second copy would drift, and "two readers of one thing" is the
 * bug class this project keeps closing.
 *
 * Host-side by nature: it decides filenames and writes to disk, which the
 * library deliberately does not.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Config.hpp"
#include "core/Image.hpp"
#include "core/Types.hpp"
#include "renderer/OfflineRenderer.hpp"

namespace quantiloom::app {

/**
 * @brief What a render produced, for a caller that has to report it
 */
struct RenderOutcome {
    bool ok = false;
    /// Why it failed. Also set when the render half-succeeded, in which case
    /// the files below may still exist.
    String error;

    // Resolved by the renderer, read rather than re-parsed from the TOML.
    u32 width = 0;
    u32 height = 0;
    u32 spp = 0;
    String modeName;
    f32 wavelengthNm = 0.0f;

    String exrPath;
    /// Empty for modes that write no preview.
    String pngPath;
    /// The temperature a thermal camera would report, in kelvin. Empty unless
    /// [thermography] enabled it and the mode carries a band to invert.
    String tappPath;
    /// The hyperspectral cube streams itself to disk; there is no frame.
    bool wroteItsOwnOutput = false;

    /**
     * @brief The displayable RGB frame, stretched exactly as the PNG on disk
     *
     * Kept in memory for a caller that wants to show the frame rather than
     * point at it -- an MCP tool answering with image content, say. Empty when
     * the mode wrote no preview.
     */
    Image preview;

    f64 seconds = 0.0;
};

/**
 * @brief Render a configuration and write its outputs
 *
 * @param config Parsed scene configuration.
 * @param init   Everything host-side the renderer needs: the atmosphere weights
 *               fallback, the config's own directory, and -- for batch mode --
 *               the shared device the render runs on.
 */
/**
 * @brief Post-process one rendered frame and write its files
 *
 * The apparent-temperature map when [thermography] asked for one, then the
 * sensor chain, then the EXR, then the PNG preview for the fused modes. Split
 * out of RenderConfigToFiles so that a sequence -- which renders many frames
 * through one renderer and names each of them itself -- writes them exactly
 * the way a single render does.
 *
 * @param outcome  `exrPath`, `width` and `height` are read; `pngPath`,
 *                 `tappPath`, `preview` and `error` are written.
 */
void WriteFrameOutputs(const Config& config, OfflineRenderOutput& rendered,
                       SpectralMode spectralMode, RenderOutcome& outcome);

RenderOutcome RenderConfigToFiles(const Config& config,
                                  const OfflineRenderer::InitParams& init);

/**
 * @brief Render a configuration on a device of its own
 *
 * What a host rendering one scene wants, and what the CLI's single-config path
 * and serve mode both call.
 *
 * @param config                        Parsed scene configuration.
 * @param atmosphereModelPackFallback   Where to find atmosphere weights when
 *                                      the scene names a preset but no path.
 */
RenderOutcome RenderConfigToFiles(const Config& config,
                                  const String& atmosphereModelPackFallback);

}  // namespace quantiloom::app
