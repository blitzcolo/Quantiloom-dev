/**
 * @file OfflineRenderer.hpp
 * @brief Renders a scene configuration to an image, on a device it creates itself
 *
 * The sibling of ExternalRenderContext. That one serves a host that already owns
 * Vulkan and wants to draw into its surface; this one serves a host that owns
 * nothing and wants a finished frame. Both are adapters over the same internal
 * RenderCore, so the two front ends run one renderer rather than two that drift.
 *
 * Everything below the config is private: the device, the acceleration structures,
 * the pipeline, the spectral and atmosphere buffers. A caller supplies a parsed
 * scene configuration and receives radiance.
 *
 * @code
 * auto renderer = OfflineRenderer::Create(config);
 * if (!renderer) { QL_LOG_ERROR("{}", renderer.error()); return 1; }
 *
 * auto out = renderer.value()->Render();
 * if (!out.wroteItsOwnOutput) {
 *     ImageIO::WriteEXR(renderer.value()->Params().outputPath, out.radiance);
 * }
 * @endcode
 *
 * @note One render per instance. The offline case has no reason to re-trace the
 *       same scene, and pretending otherwise would mean specifying what does and
 *       does not carry over between calls.
 * @note Not thread-safe.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "core/Platform.hpp"
#include "core/Config.hpp"
#include "core/Image.hpp"

#include <functional>
#include <memory>

namespace quantiloom {

class RenderDevice;

/**
 * @struct OfflineProgress
 * @brief How far a hyperspectral cube has got
 *
 * The public face of the band-by-band progress the renderer already tracks.
 * A cube is minutes of work that produces nothing until the last band, so a
 * host offering it needs these numbers to say anything at all while it runs.
 */
struct OfflineProgress {
    u32 currentBand = 0;              ///< Bands finished so far
    u32 totalBands = 0;               ///< Bands in this cube
    f32 currentWavelength_nm = 0.0f;  ///< Where the render has got to
    f32 elapsedSeconds = 0.0f;
    /// Extrapolated from the bands done so far, so it is meaningless until a
    /// few have completed and settles as the render proceeds.
    f32 estimatedTotalSeconds = 0.0f;

    [[nodiscard]] f32 GetPercentage() const {
        if (totalBands == 0) return 0.0f;
        return 100.0f * static_cast<f32>(currentBand) / static_cast<f32>(totalBands);
    }
    [[nodiscard]] f32 GetRemainingSeconds() const {
        return estimatedTotalSeconds - elapsedSeconds;
    }
};

/**
 * @brief What the renderer resolved from the configuration
 *
 * Read rather than re-parsed. A caller writing the output needs the resolution,
 * the mode and the wavelength to name and describe its files, and re-reading the
 * same TOML keys on its own side is how two readers of one setting end up
 * disagreeing -- `wavelength_nm` in particular is not simply the config value,
 * since a fused mode with no explicit wavelength takes its band centre.
 */
struct OfflineRenderParams {
    u32 width = 0;
    u32 height = 0;
    u32 spp = 1;

    SpectralMode mode = SpectralMode::RGB;
    /// As spelled in the config, for metadata and messages.
    String modeName;
    /// The config's `spectral.wavelength_nm`, or the mode's band centre.
    f32 wavelengthNm = 550.0f;

    String outputPath;
};

/**
 * @brief The result of a render
 */
struct OfflineRenderOutput {
    /**
     * @brief The traced frame, RGBA, with the metadata an archival EXR carries
     *
     * Values are per-nm **average** spectral radiance in the fused bands, not
     * band-integrated -- see SensorAdjustmentForMode before feeding a sensor
     * chain. Empty when `wroteItsOwnOutput` is set.
     */
    Image radiance;

    /**
     * @brief What `radiance` has to be scaled by before a sensor chain
     *
     * `core/Types.hpp` states the contract beside the band table: a fused-band
     * render writes per-nm **average** spectral radiance, while a sensor chain
     * wants band-**integrated** radiance and the band's own photon energy.
     * Getting either wrong is not subtle -- for LWIR the two together are a
     * factor of ~7e4 -- and the renderer is the only party that knows which
     * band it just integrated, so it says so rather than leaving the caller to
     * work it out.
     *
     * 1.0 outside the fused IR modes.
     */
    f32 sensorRadianceScale = 1.0f;

    /// Wavelength for photon energy, or 0 to keep whatever the caller set.
    f32 sensorWavelengthNm = 0.0f;

    /**
     * @brief The render already wrote its output and there is no image to save
     *
     * True for the hyperspectral cube, which streams band by band to an ENVI
     * file rather than assembling a frame in memory.
     */
    bool wroteItsOwnOutput = false;

    /// Set when the render failed after starting; `radiance` is then unusable.
    String error;
};

/**
 * @class OfflineRenderer
 * @brief A path tracer that owns its own device, for offline rendering
 */
class QL_API OfflineRenderer {
public:
    /**
     * @brief Host-side settings, none of which belong in a scene config
     */
    struct InitParams {
        /**
         * @brief Where to look for atmosphere network weights
         *
         * Used only when a scene names `atmosphere.preset` without a
         * `model_pack`. The pack is a deployment artifact -- installed with the
         * library, shipped in the SDK, copied next to an executable -- so where
         * a given host keeps it is not the library's business to guess.
         */
        String atmosphereModelPackFallback;

        /// Read at start-up and written back afterwards. Empty disables both.
        /// Ignored when `sharedDevice` is set -- the cache belongs to the device.
        String pipelineCachePath = "pipeline_cache.bin";

        /**
         * @brief A device to render on instead of creating one
         *
         * The device, the pipeline cache, the BRDF LUT, the colour matching table
         * and the fallback environment map then come from it rather than being
         * built here -- roughly 250 ms and a pair of LUT generations per render,
         * which is what a batch of a hundred scenes was paying a hundred times.
         *
         * Null keeps the original behaviour: this instance creates and owns
         * everything, as a single-render host wants. Non-null, the device must
         * outlive this renderer, and renders sharing one run one at a time.
         *
         * @see RenderDevice
         */
        RenderDevice* sharedDevice = nullptr;

        /**
         * @brief Directory a relative path inside the config resolves against
         *
         * Tried first, with the path as written as the fallback, so a
         * self-contained scene folder resolves against itself while a
         * repo-root-relative path keeps working from the working directory.
         * Normally the directory the config file was loaded from.
         *
         * Empty -- the default -- means paths resolve against the working
         * directory alone, which is what this renderer did before the field
         * existed.
         */
        String baseDir;

        /// Per-frame GPU timings. Empty disables the logger.
        String performanceCsvPath = "quantiloom_performance.csv";

        /**
         * @brief Called as each hyperspectral band completes
         *
         * A cube is minutes of work with no output until the last band, so a
         * host that offers it needs somewhere to put a progress bar. The
         * renderer has always tracked this -- it logs a line every ten bands
         * -- but the numbers had no way out of the library.
         *
         * Called from the render thread, between bands, with the render
         * paused. Do not call back into the renderer from it. Empty by
         * default, and unused by a non-hyperspectral render.
         */
        std::function<void(const OfflineProgress&)> onProgress;
    };

    /**
     * @brief Parse the configuration and build everything the render needs
     *
     * Creates a device, loads the scene, uploads every buffer and builds the
     * pipeline -- so a successful return means the next call is the trace
     * itself. Errors are returned rather than logged, since a host may want to
     * report them somewhere other than the log.
     *
     * @return The renderer, or why the scene cannot be rendered
     */
    static Result<std::unique_ptr<OfflineRenderer>, String> Create(
        const Config& config, const InitParams& params = {});

    ~OfflineRenderer();

    OfflineRenderer(const OfflineRenderer&) = delete;
    OfflineRenderer& operator=(const OfflineRenderer&) = delete;
    OfflineRenderer(OfflineRenderer&&) = delete;
    OfflineRenderer& operator=(OfflineRenderer&&) = delete;

    /// What Create() resolved from the config.
    [[nodiscard]] const OfflineRenderParams& Params() const;

    /**
     * @brief Trace the scene and read the frame back
     *
     * @throws std::runtime_error if a queue submit or fence wait fails, which on
     *         this path usually means a device loss or a driver timeout -- there
     *         is no partial result worth returning.
     */
    OfflineRenderOutput Render();

private:
    OfflineRenderer();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom
