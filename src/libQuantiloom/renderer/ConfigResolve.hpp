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
#include "thermal/ThermalSolver.hpp"
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
 * @brief Resolve a path a config named, the way every other config path is
 *
 * Against the config's own directory when that finds the file, and otherwise
 * left as written -- the two conventions in use are a self-contained scene
 * folder and a repo-root-relative CLI config, and both have to keep working.
 * See the definition for the full reasoning. Declared here because anything
 * that reads a file a config pointed at has to agree about this.
 */
String ResolveConfigPath(const String& path, const String& baseDir);

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
    /// `renderer.environment_map_enabled`. Configuration *intent*, and only that:
    /// it is one of three conditions on image-based lighting, not the switch.
    /// The GPU flag `lighting.enableEnvironmentMap` is 1 only when this is true
    /// AND `environmentMap` names a file AND the mode is RGB (see the resolver;
    /// the spectral branches read a map's RGB as spectral radiance density and
    /// get it wrong by about 12x, so they never sample one). A host applies a
    /// fourth: the map has to actually load.
    ///
    /// False means the map lights nothing -- not that it is replaced by a
    /// substitute sky. The path is still carried so a host can keep showing it
    /// and turn it back on without re-reading the file. A host that honours this
    /// skips loading the map entirely; the cubemap binding stays valid because
    /// the black placeholder is bound instead, and the shader does not sample it
    /// either way.
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

    /// Whether emissive geometry is sampled directly (next-event estimation).
    ///
    /// On by default and there is no reason to turn it off for a picture: it
    /// costs one shadow ray per hit and is worth roughly thirty times the
    /// samples in a scene lit by emissive geometry. The switch exists so that
    /// the two light-transport strategies can be measured against each other --
    /// with it off, emitters are found only by BSDF sampling, and the two must
    /// converge to the same image. scripts/render-tests/check_nee_mis.py is
    /// that comparison.
    bool enableLightSampling = true;

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

    /// [thermal] -- the surface energy balance, when a scene asks for one. Its
    /// per-material properties come from the same [[materials]] entries the IR
    /// overrides do, matched by name.
    thermal::ThermalConfig thermal;

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
/**
 * @brief One material's endmember mixture, resolved against the curve buffer
 *
 * curves[0] is the same index materialNameToCurve holds for this material --
 * endmember 0 is the material's own reference, not a separate thing.
 *
 * colorsLinear is what each curve looks like under D65, and it exists so the
 * unmixer can ask "how much of each of these is this texel". Always taken from
 * the VIS band whatever the render band is: the weights are a spatial
 * abundance, wavelength-independent by construction, while the texture that
 * suggests them is a visible-light image.
 */
struct EndmemberSlots {
    i32 curves[Material::MAX_ENDMEMBERS] = {-1, -1, -1, -1};
    glm::vec3 colorsLinear[Material::MAX_ENDMEMBERS]{};
    i32 count = 0;
    Material::SpectralUnmixMode unmix = Material::SpectralUnmixMode::Auto;
    String weightTexturePath;
    i32 weightTextureIndex = -1;  ///< filled by the unmixer, after this resolve
};

struct ResolvedMaterialSpectra {
    Vector<SpectralCurveGPU> curves;
    std::unordered_map<String, i32> materialNameToCurve;

    /// Endmember mixtures, keyed like materialNameToCurve. A material appears
    /// in both: the first map is endmember 0, this one is the whole mixture.
    std::unordered_map<String, EndmemberSlots> materialNameToEndmembers;

    /// Measured sheen reflectance, keyed like materialNameToCurve and pointing
    /// into the same `curves`. Separate from the endmember mixture: sheen is
    /// one curve with no weight texture, because the mixture's weights are
    /// unmixed from the base-colour texture and describe the base, not the
    /// fibres over it. A material may appear here without appearing in
    /// materialNameToCurve -- velvet over an RGB base colour.
    std::unordered_map<String, i32> materialNameToSheenCurve;

    /// Measured clearcoat reflectance, resolved exactly as the sheen curve is.
    /// The only path by which a coat reaches MWIR or LWIR, where the dielectric
    /// 0.04 the visible bands assume does not hold.
    std::unordered_map<String, i32> materialNameToClearcoatCurve;

    /// Measured diffuse transmission colour, resolved exactly as the sheen
    /// curve is. The only path by which it reaches NIR or SWIR; MWIR and LWIR
    /// do not read it at all, since thermal transmittance is already
    /// irTransmittance and a surface cannot have two.
    std::unordered_map<String, i32> materialNameToDiffuseTransmissionCurve;

    Vector<ComplexRefractiveIndexGPU> refractiveIndices;
    std::unordered_map<String, i32> materialNameToRefractiveIndex;

    /// Thermal properties by material name, from the [[materials]] entries
    /// that named a conductivity. Keyed by name rather than written onto the
    /// Material: that header is a layout contract, and these are solver inputs
    /// nothing in the shader reads.
    std::unordered_map<String, thermal::ThermalMaterial> thermalMaterials;

    /// Band-averaged LWIR emissivity per material, Planck-weighted at 300 K.
    /// Computed during ref resolution from the reconstructed LWIR reflectance
    /// curve. Written onto Material::bandAveragedIREmissivity by whoever
    /// adopts the scene so both ThermalSolver and ThermalPreview see it.
    std::unordered_map<String, f32> bandAveragedIREmissivity;

    /// How many materials reached a quantitative band with no measured
    /// spectrum -- counted by the sRGB-upsampling gate below, after the
    /// config's own assignments have been applied, so it reflects what the
    /// render will actually shade with rather than what the scene file
    /// happened to load with. Zero in RGB and VIS_FUSED, which have no such
    /// requirement, and in SINGLE inside the visible band, where upsampling a
    /// base colour is exactly what VIS_FUSED does per wavelength.
    ///
    /// This is the authority for "is this render quantitative". Callers must
    /// not re-derive it from the spectral mode: every band on the gate's list
    /// *can* be quantitative, and whether one is depends on its materials.
    u32 rgbUpsampledMaterials = 0;

    u32 temperatureBackfilled = 0;
    u32 materialsOverridden = 0;
    u32 nodesTransformed = 0;
    u32 nodesDuplicated = 0;
    u32 nodesRemoved = 0;
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

/**
 * @brief The illuminant, resolved from what a scene declares about it.
 *
 * Everything `lighting.solar_lut*` means, in one place: which file (or the
 * literal "equal_energy"), which columns it keeps the sun and sky in, whether
 * column 3 is global rather than sky, and whether to normalise.
 *
 * Split out of ResolveRenderConfig so that a host can set the illuminant
 * without re-applying a whole configuration -- and so that doing so cannot
 * mean anything different from what the file would have meant. A second
 * reading of these keys is exactly the class of divergence this repository
 * spent a release removing.
 */
struct SolarLutRequest {
    /// A path, or the literal "equal_energy" for CIE illuminant E.
    String pathOrEqualEnergy;
    /// 1-based columns. libRadtran uvspec's layout is the default; ASTM G-173
    /// wants {4, 3} with diffuseIsGlobal.
    u32 directColumn = 2;
    u32 diffuseColumn = 3;
    /// Subtract direct from the diffuse column, clamped at zero.
    bool diffuseIsGlobal = false;
    /// Divide both curves by the sun's luminance, putting the illuminant at
    /// Y = 1. Published reference spectra are relative, so their absolute
    /// level is arbitrary.
    bool normaliseUnitLuminance = false;
};

/// What an illuminant resolves to: the two curves, and the RGB the renderer's
/// non-spectral paths must use so that both halves describe one sun.
struct ResolvedSolarLut {
    SpectralCurve sun;
    SpectralCurve sky;
    glm::vec3 sunRadianceRgb{0.0f};
    glm::vec3 skyRadianceRgb{0.0f};
    f32 sunRadianceSpectral = 0.0f;
    f32 skyRadianceSpectral = 0.0f;
    /// Things the caller should surface: an unknown normalise mode, a spectrum
    /// that does not span the band being rendered. Not errors -- the render
    /// proceeds and looks plausible, which is why they must be said out loud.
    Vector<String> warnings;
};

/**
 * @brief Load, normalise and colour-derive one illuminant.
 *
 * @param request   What the scene (or the host) asked for.
 * @param baseDir   Directory relative paths resolve against.
 * @param mode      Only for the band-coverage warning; pass the mode being
 *                  rendered, or RGB to skip that check.
 */
Result<ResolvedSolarLut, String> ResolveSolarLut(const SolarLutRequest& request,
                                                 const String& baseDir,
                                                 SpectralMode mode);

}  // namespace quantiloom::rendercore
