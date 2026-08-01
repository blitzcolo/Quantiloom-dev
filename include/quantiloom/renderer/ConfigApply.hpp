/**
 * @file ConfigApply.hpp
 * @brief Options and diagnostics for applying a scene configuration
 *
 * A scene TOML has around fifty keys, and until now each host worked out what
 * they meant for itself: the CLI inside OfflineRenderer, Quantiloom Studio in
 * its own ConfigManager. The two drifted -- opposite defaults for shadow rays,
 * a wavelength rule only one of them had, a normalisation key only one of them
 * read -- and the same file rendered differently depending on which program
 * opened it.
 *
 * The interpretation is one function now, and these are the two things that
 * legitimately differ between hosts: how strict to be about a key that is
 * missing, and what to do about the handful of keys a windowed context cannot
 * honour. Everything else is the same for everyone.
 *
 * Nothing here is exported. They are plain structs passed to and returned from
 * ExternalRenderContext::ApplyConfig, which is.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom {

/**
 * @brief How to read a configuration.
 */
struct ConfigApplyOptions {
    /**
     * @brief What a missing required key means.
     *
     * The CLI renders once and writes a file, so a config it cannot fully
     * honour is a config it should refuse -- a render that silently used a
     * default is worse than no render. An editor holds a document being
     * written, where half-finished is a normal state to be in and saying so is
     * more useful than refusing to open it.
     */
    enum class MissingKeyPolicy : u8 {
        Error,           ///< Refuse, and say which key. The CLI's behaviour.
        WarnAndDefault,  ///< Warn, take the documented default, carry on.
    };
    MissingKeyPolicy missingRequired = MissingKeyPolicy::WarnAndDefault;

    /**
     * @brief Directory the config's relative paths resolve against.
     *
     * Empty means the process working directory, which is what the CLI wants:
     * it is run from the repo root and its configs name assets from there.
     * A GUI opening a file from anywhere wants the file's own directory.
     */
    String baseDir;

    /**
     * @brief Model pack to use when [atmosphere] names a preset but no pack.
     *
     * Where the shipped weights live is a property of the installation, not of
     * the scene, so finding them stays with the host -- both already have a
     * resolver that looks at QUANTILOOM_ATMOS_MODELS and its own directories.
     */
    String atmosphereModelPackFallback;

    /**
     * @brief Resolve sun angles and observer altitude once, rather than per frame.
     *
     * The CLI renders one image from one camera, so it pins both at load time.
     * An interactive viewport wants them to follow the camera it is flying,
     * which is what AtmosphereNNConfig's sunFromLighting / h1FromCamera do.
     * Explicit atmosphere.sun_zenith_deg / h1_km keys are honoured either way:
     * a scene that states a geometry means it.
     */
    bool freezeDerivedAtmosGeometry = false;

    /**
     * @brief Apply renderer.debug_mode.
     *
     * Studio treats the debug visualisation as a way of looking at a scene
     * rather than part of it, and keeps it out of the document. The CLI has no
     * other way to ask for one.
     */
    bool applyDebugMode = false;
};

/**
 * @brief One thing worth saying about a configuration that was read.
 */
struct ConfigApplyMessage {
    enum class Severity : u8 { Info, Warning, Error };

    Severity severity = Severity::Info;
    String key;   ///< The TOML key concerned; empty when none is.
    String text;
};

/**
 * @brief What applying a configuration did, and what it could not do.
 *
 * Deliberately not a copy of the state that was applied: that is readable
 * through the context's own getters, and a second copy is a second thing to
 * keep in step. What is here is what no getter can answer -- the diagnostics,
 * how much of each kind of data was loaded, and the keys a windowed context
 * has no way to honour.
 */
struct ConfigApplyReport {
    Vector<ConfigApplyMessage> messages;

    bool sceneLoaded = false;
    bool solarLutLoaded = false;
    bool atmosphereEnabled = false;
    u32 spectralCurvesLoaded = 0;         ///< CSV curves plus NMF reconstructions.
    u32 refractiveIndicesLoaded = 0;
    u32 materialsTemperatureBackfilled = 0;
    u32 materialsOverridden = 0;          ///< From [[materials]] entries.
    u32 nodesTransformed = 0;             ///< From [[nodes]] transform overrides.
    u32 nodesDuplicated = 0;              ///< From [[duplicates]] entries.
    u32 nodesRemoved = 0;                 ///< From scene.removed_nodes.

    /// @name Keys a windowed context cannot honour, echoed for the host
    /// @{
    /// renderer.resolution. A viewport renders at its own size; the CLI does not.
    u32 configWidth = 0;
    u32 configHeight = 0;
    /// renderer.output. Nothing to write to in an interactive session.
    String outputPath;
    /// [hyperspectral]. Cube rendering is a separate, non-progressive renderer.
    bool hasHyperspectralSection = false;
    /// @}

    [[nodiscard]] bool ok() const {
        for (const auto& m : messages) {
            if (m.severity == ConfigApplyMessage::Severity::Error) return false;
        }
        return true;
    }

    /// First error, for a caller that wants one string. Empty when ok().
    [[nodiscard]] String FirstError() const {
        for (const auto& m : messages) {
            if (m.severity == ConfigApplyMessage::Severity::Error) return m.text;
        }
        return {};
    }
};

}  // namespace quantiloom
