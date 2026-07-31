/**
 * @file ConfigResolve.hpp
 * @brief Reading a scene TOML into plain data, once, for every host
 *
 * Internal on purpose: not in include/quantiloom/, nothing here carries QL_API.
 * The CLI reaches it directly; Quantiloom Studio reaches it through
 * ExternalRenderContext::ApplyConfig. The unit tests link quantiloom_core and
 * see it as it is.
 *
 * ## Why this exists
 *
 * The same ~50 keys were being interpreted twice -- once inside OfflineRenderer
 * for the CLI, once inside Quantiloom Studio's own ConfigManager -- and the two
 * readings had drifted far enough that a scene rendered differently depending
 * on which program opened it. Shadow rays defaulted on for one and off for the
 * other. An absent spectral.wavelength_nm meant the band centre for one and
 * 550 nm for the other, so every thermal config rendered at a visible
 * wavelength in the GUI. lighting.solar_lut_normalise was read by one and
 * ignored by the other, a factor of ten thousand on a D65 scene. The NMF basis
 * and [refractive_index] were read by one and not the other at all.
 *
 * None of those were arguable: they were the same file being read two ways.
 * This is the one way.
 *
 * ## What is here and what is not
 *
 * Config in, plain structs out. No device, no GPU buffer, no upload. That
 * boundary is what lets one function serve a batch renderer that owns its
 * device and an interactive context that was handed someone else's -- each
 * takes the resolved data and uploads it the way it needs to.
 *
 * It is also why the two halves are separate. ResolveRenderConfig needs no
 * scene, so a host can call it before spending a device on a config that turns
 * out to be broken. ResolveMaterialSpectra needs the loaded scene, because
 * curve and refractive-index files are matched to materials by name.
 *
 * ## Ordering
 *
 * Two dependencies from the original code survive here, and both are internal
 * to ResolveRenderConfig rather than left to callers: the solar LUT is read
 * before the atmosphere, because an enabled atmosphere overwrites the lighting
 * temperature with its own ground temperature; and the atmosphere's geometry
 * is derived after the camera, because an observer altitude that was not given
 * comes from the camera height.
 *
 * The one ordering rule left to callers is that the material buffer is built
 * after ResolveMaterialSpectra, since it resolves indices into the curves that
 * returns.
 *
 * @author blitzcolo
 */

#pragma once

#include "atmos/AtmosphereNNConfig.hpp"
#include "core/Config.hpp"
#include "core/SpectralData.hpp"
#include "core/Types.hpp"
#include "postprocess/SensorModel.hpp"
#include "renderer/ConfigApply.hpp"
#include "renderer/LightingParams.hpp"
#include "scene/Camera.hpp"
#include "scene/Scene.hpp"

#include <glm/glm.hpp>

#include <optional>
#include <unordered_map>
#include <utility>

namespace quantiloom::rendercore {

/**
 * @brief Everything a scene TOML says that does not depend on the scene file.
 */
struct ResolvedRenderConfig {
    // [renderer]
    u32 width = 0;
    u32 height = 0;
    u32 spp = 1;
    u32 seed = constants::DEFAULT_SAMPLING_SEED;
    String outputPath = "spectral_output.exr";
    String environmentMap;
    /// `renderer.environment_map_enabled`. False means the map lights nothing --
    /// not that it is replaced by a substitute sky. The path is still carried so
    /// a host can keep showing it and turn it back on without re-reading the
    /// file. A host that honours this skips loading the map entirely; the
    /// cubemap binding stays valid because the fallback is bound instead, and
    /// the shader does not sample it either way.
    bool environmentMapEnabled = true;
    i32 debugMode = 0;

    // [spectral]
    SpectralMode mode = SpectralMode::RGB;
    String modeName = "rgb";
    f32 wavelengthNm = 550.0f;

    // [scene]
    f32 worldUnitsToMeters = 1.0f;
    f32 defaultTemperatureK = 300.0f;

    // [camera]
    Camera camera;

    /// [lighting] and [quality] merged, with the solar spectrum's colour and
    /// the atmosphere's ground temperature already substituted in where they
    /// apply -- so this is what goes to the GPU, not a first draft of it.
    LightingParams lighting{};

    /// The sun and sky spectra, normalised if the config asked for it. Absent
    /// when the scene names no solar_lut, which in a spectral mode means it
    /// renders unlit -- the shaders have no RGB fallback.
    std::optional<std::pair<SpectralCurve, SpectralCurve>> solarSunSky;

    /// Assembled from preset plus per-feature overrides. `enabled` is false
    /// unless the scene named [atmosphere] model_pack or preset.
    AtmosphereNNConfig atmosphere;

    // [sensor]
    bool sensorEnabled = false;
    SensorParams sensor{};

    // [quality]
    bool failOnSrgbUpsample = false;
    bool logMaterialSources = false;

    // [material] -- the fallback for a scene that brings no materials of its own
    glm::vec3 defaultAlbedo{0.8f, 0.8f, 0.8f};

    // [hyperspectral], echoed rather than resolved: the cube renderer reads it
    bool hasHyperspectralSection = false;
};

/**
 * @brief Curves and per-material slots resolved against a loaded scene.
 *
 * The name maps are how a host wires slots into materials. The CLI writes them
 * into a MaterialGpuIndices table; the interactive context writes them onto the
 * Material objects themselves, which is where its own buffer builder reads
 * them. Same data, two conventions.
 */
struct ResolvedMaterialSpectra {
    Vector<SpectralCurveGPU> curves;
    std::unordered_map<String, i32> materialNameToCurve;

    Vector<ComplexRefractiveIndexGPU> refractiveIndices;
    std::unordered_map<String, i32> materialNameToRefractiveIndex;

    u32 temperatureBackfilled = 0;
    u32 materialsOverridden = 0;
};

/**
 * @brief Read every scene-independent key.
 *
 * Loads the solar LUT from disk when one is named, and resolves the atmosphere
 * model pack, so this touches the filesystem -- but never a device.
 *
 * @param config    The parsed TOML.
 * @param options   Strictness, base directory, atmosphere fallback.
 * @param report    Appended to. Messages, and the echoed unhonorable keys.
 * @return The resolved configuration, or the first error under Error policy.
 */
Result<ResolvedRenderConfig, String> ResolveRenderConfig(
    const Config& config, const ConfigApplyOptions& options, ConfigApplyReport& report);

/**
 * @brief Read every key that needs the scene to make sense of it.
 *
 * Mutates @p scene: backfills a default surface temperature onto materials
 * carrying none, and applies [[materials]] IR overrides. Both change values,
 * never the shape of anything.
 *
 * @param scene     The loaded scene. Modified in place.
 * @param resolved  Output of ResolveRenderConfig; supplies the mode, the
 *                  default temperature and the sRGB-upsampling gate.
 */
Result<ResolvedMaterialSpectra, String> ResolveMaterialSpectra(
    const Config& config, Scene& scene, const ResolvedRenderConfig& resolved,
    const ConfigApplyOptions& options, ConfigApplyReport& report);

}  // namespace quantiloom::rendercore
