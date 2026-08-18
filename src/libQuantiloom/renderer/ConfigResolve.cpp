/**
 * @file ConfigResolve.cpp
 * @brief The one reading of a scene TOML -- see ConfigResolve.hpp for why
 *
 * @author blitzcolo
 */

#include "renderer/ConfigResolve.hpp"

#include "core/SkyThermal.hpp"

#include "core/Log.hpp"
#include "io/SpectralBasisLoader.hpp"
#include "io/SpectralIO.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "scene/Material.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <utility>
#include <cmath>
#include <filesystem>

namespace quantiloom::rendercore {

namespace {

using Severity = ConfigApplyMessage::Severity;

/// Collects diagnostics and remembers whether a fatal one was seen, so the
/// resolver can keep reading -- a half-finished config is more useful with all
/// of its problems listed than with only the first.
class Diagnostics {
public:
    Diagnostics(const ConfigApplyOptions& options, ConfigApplyReport& report)
        : m_options(options), m_report(report) {}

    void Info(String key, String text) {
        Add(Severity::Info, std::move(key), std::move(text));
    }

    void Warn(String key, String text) {
        Add(Severity::Warning, std::move(key), std::move(text));
    }

    /// A key the CLI refuses over and an editor carries on past. Returns true
    /// when the caller should treat it as fatal.
    bool Required(String key, String text) {
        const bool fatal =
            m_options.missingRequired == ConfigApplyOptions::MissingKeyPolicy::Error;
        if (fatal) {
            m_firstError = text;
            Add(Severity::Error, std::move(key), std::move(text));
        } else {
            Add(Severity::Warning, std::move(key), std::move(text));
        }
        return fatal;
    }

    /// Wrong whatever the policy: nothing downstream can proceed from it.
    void Fatal(String key, String text) {
        if (m_firstError.empty()) m_firstError = text;
        Add(Severity::Error, std::move(key), std::move(text));
    }

    [[nodiscard]] bool failed() const { return !m_firstError.empty(); }
    [[nodiscard]] const String& firstError() const { return m_firstError; }

private:
    void Add(Severity severity, String key, String text) {
        // Through the core logger as well as into the report: the CLI prints
        // its log and never looks at the report, and a GUI user reading the
        // log expects to find the same lines there.
        switch (severity) {
            case Severity::Error:   QL_LOG_ERROR("{}", text); break;
            case Severity::Warning: QL_LOG_WARN("{}", text); break;
            case Severity::Info:    QL_LOG_INFO("{}", text); break;
        }
        m_report.messages.push_back({severity, std::move(key), std::move(text)});
    }

    const ConfigApplyOptions& m_options;
    ConfigApplyReport& m_report;
    String m_firstError;
};

/// One table's worth of the transform grammar shared by [[nodes]] and
/// [[duplicates]]: `matrix` (16 numbers, column-major) wins; otherwise
/// translation / rotation_euler_degrees / scale compose in that order.
/// Degrees, because a person writes these by hand and a file full of
/// 0.6108 is a file nobody edits twice.
struct TransformKeys {
    bool present = false;   ///< some transform key appeared
    bool valid = false;     ///< ...and parsed cleanly
    usize matrixCount = 0;  ///< for the wrong-size warning
    glm::mat4 transform{1.0f};
};

TransformKeys ParseTransformKeys(const Config& table) {
    TransformKeys out;
    if (const auto matrix = table.GetFloatArray("matrix"); !matrix.empty()) {
        out.present = true;
        out.matrixCount = matrix.size();
        if (matrix.size() != 16) {
            return out;
        }
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                out.transform[c][r] = matrix[static_cast<usize>(c * 4 + r)];
            }
        }
        out.valid = true;
        return out;
    }

    const auto translation = table.GetFloatArray("translation");
    const auto rotation = table.GetFloatArray("rotation_euler_degrees");
    const auto scale = table.GetFloatArray("scale");
    if (translation.empty() && rotation.empty() && scale.empty()) {
        return out;
    }
    out.present = true;

    glm::mat4 transform(1.0f);
    if (translation.size() >= 3) {
        transform = glm::translate(
            transform, glm::vec3(translation[0], translation[1], translation[2]));
    }
    if (rotation.size() >= 3) {
        transform = glm::rotate(transform, glm::radians(rotation[0]), glm::vec3(1, 0, 0));
        transform = glm::rotate(transform, glm::radians(rotation[1]), glm::vec3(0, 1, 0));
        transform = glm::rotate(transform, glm::radians(rotation[2]), glm::vec3(0, 0, 1));
    }
    if (scale.size() >= 3) {
        transform = glm::scale(transform, glm::vec3(scale[0], scale[1], scale[2]));
    }
    out.transform = transform;
    out.valid = true;
    return out;
}

}  // namespace

/// Resolve a path the config named, against the config's own directory when
/// that finds the file and otherwise not at all.
///
/// Two conventions are in use and both have to keep working. The CLI is run
/// from the repo root and its configs name assets from there, so their relative
/// paths are working-directory-relative -- `assets/luts/D65.csv` beside a
/// config that itself lives in `assets/configs/`. A config a GUI user opens
/// from an arbitrary directory has no such root, and expects its paths to be
/// relative to itself.
///
/// Trying the base directory first and falling back to the path as written
/// serves both: a self-contained scene folder resolves against itself, and a
/// repo-root-relative path is left alone to resolve against the working
/// directory as it always did. Both callers pass a real base directory -- the
/// CLI passes the config's own parent -- so the first attempt is the normal
/// case, not an exception.
///
/// When BOTH candidates exist the base directory wins, and that is worth
/// saying out loud. A stale duplicate under the config's directory shadows the
/// canonical asset silently and indefinitely: three copies of the prism models
/// under assets/configs/assets/ did exactly that for six weeks, and because
/// the render still succeeded, the two dispersion checkers went on editing a
/// file nothing read and reporting the resulting no-op as a renderer bug.
String ResolveConfigPath(const String& path, const String& baseDir) {
    if (path.empty() || baseDir.empty()) return path;
    std::filesystem::path p(path);
    if (p.is_absolute()) return path;

    std::error_code ec;
    const auto candidate = (std::filesystem::path(baseDir) / p).lexically_normal();
    if (std::filesystem::exists(candidate, ec)) {
        if (std::filesystem::exists(p, ec)) {
            QL_LOG_WARN("Config path '{}' exists both under the config's directory "
                        "('{}') and as written; using the former. Delete one -- a "
                        "duplicate beside the config shadows the real asset and "
                        "edits to it will appear to do nothing.",
                        path, candidate.string());
        }
        return candidate.string();
    }
    return path;
}

// ============================================================================
// Scene-independent keys
// ============================================================================

Result<ResolvedRenderConfig, String> ResolveRenderConfig(
    const Config& config, const ConfigApplyOptions& options, ConfigApplyReport& report) {
    using ResolveResult = Result<ResolvedRenderConfig, String>;

    ResolvedRenderConfig out;
    Diagnostics diag(options, report);

    // ------------------------------------------------------------------
    // [renderer]
    // ------------------------------------------------------------------
    const auto resolution = config.GetArray<i32>("renderer.resolution");
    if (resolution.size() >= 2) {
        out.width = static_cast<u32>(resolution[0]);
        out.height = static_cast<u32>(resolution[1]);
    } else {
        diag.Required("renderer.resolution",
                      "renderer.resolution must be an array of 2 integers, but has " +
                          std::to_string(resolution.size()) + " element(s)");
    }
    report.configWidth = out.width;
    report.configHeight = out.height;

    out.spp = config.Get<u32>("renderer.spp", 1);
    out.outputPath = config.Get<String>("renderer.output", "spectral_output.exr");
    report.outputPath = out.outputPath;
    out.environmentMap = config.Get<String>("renderer.environment_map", "");
    if (!out.environmentMap.empty()) {
        out.environmentMap = ResolveConfigPath(out.environmentMap, options.baseDir);
    }
    // Intent, not the switch. Still defaults on, and that is harmless now: a
    // scene that names no map gets no image-based lighting regardless, because
    // the GPU flag below also requires a named map. Keeping the default true is
    // what lets "name a map and it lights the scene" stay a one-key operation
    // instead of needing this one turned on as well.
    //
    // The key exists so a scene can keep its path while lighting from something
    // else, which is how you compare the two without editing the path out and
    // back in.
    out.environmentMapEnabled = config.Get<bool>("renderer.environment_map_enabled", true);
    out.seed = config.Get<u32>("renderer.seed", constants::DEFAULT_SAMPLING_SEED);
    out.debugMode = config.Get<i32>("renderer.debug_mode", 0);

    QL_LOG_INFO("  Resolution: {}x{}", out.width, out.height);
    QL_LOG_INFO("  Samples per pixel: {}", out.spp);
    QL_LOG_INFO("  Output: {}", out.outputPath);

    // ------------------------------------------------------------------
    // [spectral]
    // ------------------------------------------------------------------
    out.modeName = config.Get<String>("spectral.mode", "rgb");
    if (auto parsed = ParseSpectralMode(out.modeName); parsed.has_value()) {
        out.mode = *parsed;
    } else {
        diag.Fatal("spectral.mode",
                   "Invalid spectral mode: " + out.modeName +
                       ". Supported modes: single, rgb, vis_fused, mwir_fused, "
                       "lwir_fused, swir_fused, multispectral");
    }
    QL_LOG_INFO("  Spectral mode: {}", out.modeName);

    // Only the modes that render at one wavelength read it. For those, an
    // absent key means the centre of the band being rendered -- 550 nm is a
    // visible wavelength and means nothing to a thermal render.
    if (out.mode == SpectralMode::Single || out.mode == SpectralMode::MWIR_Fused ||
        out.mode == SpectralMode::LWIR_Fused || out.mode == SpectralMode::SWIR_Fused) {
        if (config.Has("spectral.wavelength_nm")) {
            out.wavelengthNm = config.Get<f32>("spectral.wavelength_nm", 550.0f);
        } else if (auto band = GetFusedBandInfo(out.mode)) {
            out.wavelengthNm = band->CenterNm();
        }
        QL_LOG_INFO("  Wavelength: {:.1f} nm", out.wavelengthNm);
    }

    out.hasHyperspectralSection = config.HasSection("hyperspectral");
    report.hasHyperspectralSection = out.hasHyperspectralSection;

    // ------------------------------------------------------------------
    // [scene]
    // ------------------------------------------------------------------
    out.worldUnitsToMeters = config.Get<f32>("scene.world_units_to_meters", 1.0f);
    if (out.worldUnitsToMeters <= 0.0f) {
        QL_LOG_WARN("scene.world_units_to_meters must be positive, using default 1.0");
        out.worldUnitsToMeters = 1.0f;
    }
    out.defaultTemperatureK = config.Get<f32>("scene.default_temperature_k", 300.0f);

    // ------------------------------------------------------------------
    // [camera]
    // ------------------------------------------------------------------
    const f32 aspectRatio = (out.width > 0 && out.height > 0)
                                ? static_cast<f32>(out.width) / static_cast<f32>(out.height)
                                : 16.0f / 9.0f;
    if (auto cameraResult = Camera::FromConfig(config, aspectRatio);
        cameraResult.has_value()) {
        out.camera = cameraResult.value();
    } else {
        diag.Required("camera", "Failed to load camera: " + cameraResult.error());
    }

    // ------------------------------------------------------------------
    // [lighting]
    // ------------------------------------------------------------------
    out.lighting = CreateDefaultLightingParams();

    const auto sunDirArray = config.GetArray<f32>("lighting.sun_direction");
    if (sunDirArray.size() == 3) {
        out.lighting.sunDirection =
            glm::normalize(glm::vec3(sunDirArray[0], sunDirArray[1], sunDirArray[2]));
    } else {
        diag.Required("lighting.sun_direction",
                      "lighting.sun_direction must be an array of 3 floats, but has " +
                          std::to_string(sunDirArray.size()) + " element(s)");
    }

    glm::vec3 sunRadiance(0.0f);
    const auto sunRadArray = config.GetArray<f32>("lighting.sun_radiance");
    if (sunRadArray.size() == 3) {
        sunRadiance = glm::vec3(sunRadArray[0], sunRadArray[1], sunRadArray[2]);
    } else {
        diag.Required("lighting.sun_radiance",
                      "lighting.sun_radiance must be an array of 3 floats, but has " +
                          std::to_string(sunRadArray.size()) + " element(s)");
    }

    glm::vec3 skyRadiance(0.0f);
    const auto skyRadArray = config.GetArray<f32>("lighting.sky_radiance");
    if (skyRadArray.size() == 3) {
        skyRadiance = glm::vec3(skyRadArray[0], skyRadArray[1], skyRadArray[2]);
    } else {
        diag.Required("lighting.sky_radiance",
                      "lighting.sky_radiance must be an array of 3 floats, but has " +
                          std::to_string(skyRadArray.size()) + " element(s)");
    }

    f32 atmosphereTemperature_K =
        config.Get<f32>("lighting.atmosphere_temperature_k", 260.0f);
    if (atmosphereTemperature_K < 150.0f || atmosphereTemperature_K > 350.0f) {
        QL_LOG_WARN("lighting.atmosphere_temperature_k={:.1f}K is outside typical "
                    "range [150, 350], check config",
                    atmosphereTemperature_K);
    }

    // ------------------------------------------------------------------
    // Clear sky
    // ------------------------------------------------------------------
    // Without the NN atmosphere the thermal sky was one isotropic blackbody:
    // as warm at the zenith as at the horizon, which no sky is. sky_model =
    // "clear_sky" replaces it with the flat-slab law driven by a
    // Berdahl-Fromberg emissivity, so a surface facing up cools against a
    // colder sky than one facing sideways -- the effect that puts frost on a
    // car roof and not on its doors.
    //
    // The correlation and the dew point behind it are evaluated here, once,
    // and only the zenith emissivity goes to the GPU. sky_emissivity
    // overrides it for a measured or otherwise known sky.
    f32 skyEmissivityClear = 0.0f;
    const auto skyModel = config.GetString("atmosphere.sky_model", "isotropic");
    if (skyModel == "clear_sky") {
        const f32 airTemperature_K = config.Get<f32>("atmosphere.air_temperature_k", 288.15f);
        const f32 relativeHumidity = config.Get<f32>("atmosphere.relative_humidity", 50.0f);
        const f32 emissivityOverride = config.Get<f32>("atmosphere.sky_emissivity", 0.0f);

        const f64 dewPointC =
            skythermal::DewPointC(static_cast<f64>(airTemperature_K) - 273.15,
                                  static_cast<f64>(relativeHumidity));
        const f64 emissivity = emissivityOverride > 0.0f
                                   ? static_cast<f64>(emissivityOverride)
                                   : skythermal::ClearSkyEmissivity(dewPointC);
        skyEmissivityClear = static_cast<f32>(emissivity);

        // The air temperature IS the sky's Planck temperature under this
        // model; the emissivity is what makes it read colder. Overriding the
        // deprecated key rather than reading both is what keeps one number in
        // charge.
        atmosphereTemperature_K = airTemperature_K;

        QL_LOG_INFO("  Clear sky: T_air {:.1f} K, RH {:.0f}%, dew point {:.1f} C, "
                    "zenith emissivity {:.3f}{}, effective sky {:.1f} K",
                    airTemperature_K, relativeHumidity, dewPointC, emissivity,
                    emissivityOverride > 0.0f ? " (given)" : "",
                    skythermal::EffectiveSkyTemperatureK(
                        static_cast<f64>(airTemperature_K), emissivity));
    } else if (skyModel != "isotropic") {
        diag.Warn("atmosphere.sky_model",
                  "  unknown atmosphere.sky_model '" + skyModel +
                      "', expected isotropic|clear_sky. Using isotropic.");
    }

    QL_LOG_INFO("  Sun direction: [{:.2f}, {:.2f}, {:.2f}]", out.lighting.sunDirection.x,
                out.lighting.sunDirection.y, out.lighting.sunDirection.z);
    QL_LOG_INFO("  Sun radiance: [{:.2f}, {:.2f}, {:.2f}]", sunRadiance.r, sunRadiance.g,
                sunRadiance.b);
    QL_LOG_INFO("  Sky radiance: [{:.2f}, {:.2f}, {:.2f}]", skyRadiance.r, skyRadiance.g,
                skyRadiance.b);
    QL_LOG_INFO("  Atmosphere temperature (IR): {:.1f} K", atmosphereTemperature_K);
    QL_LOG_INFO("  World units to meters: {:.6f}", out.worldUnitsToMeters);

    // ------------------------------------------------------------------
    // [material] -- read even when the scene brings its own materials, so a
    // misspelled key says so rather than rendering.
    // ------------------------------------------------------------------
    const auto albedoArray = config.GetArray<f32>("material.albedo");
    if (albedoArray.size() == 3) {
        out.defaultAlbedo = glm::vec3(albedoArray[0], albedoArray[1], albedoArray[2]);
    } else {
        diag.Required("material.albedo",
                      "material.albedo must be an array of 3 floats, but has " +
                          std::to_string(albedoArray.size()) + " element(s)");
    }
    QL_LOG_INFO("  Material albedo: [{:.2f}, {:.2f}, {:.2f}]", out.defaultAlbedo.x,
                out.defaultAlbedo.y, out.defaultAlbedo.z);

    const f32 sunRadiance_spectral = (sunRadiance.r + sunRadiance.g + sunRadiance.b) / 3.0f;
    const f32 skyRadiance_spectral = (skyRadiance.r + skyRadiance.g + skyRadiance.b) / 3.0f;

    if (out.mode == SpectralMode::RGB || out.mode == SpectralMode::VIS_Fused) {
        QL_LOG_INFO("  Sun RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W*sr^-1*m^-2",
                    sunRadiance.r, sunRadiance.g, sunRadiance.b);
        QL_LOG_INFO("  Sky RGB radiance: [{:.2f}, {:.2f}, {:.2f}] W*sr^-1*m^-2",
                    skyRadiance.r, skyRadiance.g, skyRadiance.b);
    } else {
        // RGB-average fallbacks, in W*sr^-1*m^-2, not spectral density.
        QL_LOG_INFO("  Sun fallback radiance (RGB avg): {:.3f} W*sr^-1*m^-2",
                    sunRadiance_spectral);
        QL_LOG_INFO("  Sky fallback radiance (RGB avg): {:.3f} W*sr^-1*m^-2",
                    skyRadiance_spectral);
    }

    // [quality]
    const f32 chromaR = config.Get<f32>("quality.chroma_r_correction",
                                        LightingDefaults::CHROMA_R_CORRECTION);
    const f32 chromaB = config.Get<f32>("quality.chroma_b_correction",
                                        LightingDefaults::CHROMA_B_CORRECTION);
    out.failOnSrgbUpsample = config.Get<bool>("quality.fail_on_srgb_upsample", false);
    out.logMaterialSources = config.Get<bool>("quality.log_material_sources", false);

    const bool enableShadowRays = config.Get<bool>("renderer.enable_shadow_rays", true);
    if (!enableShadowRays) {
        QL_LOG_INFO("Shadow rays DISABLED via config");
    }

    out.lighting.sunRadiance_spectral = sunRadiance_spectral;
    out.lighting.skyRadiance_spectral = skyRadiance_spectral;
    out.lighting.sunRadiance_rgb = sunRadiance;
    out.lighting.skyRadiance_rgb = skyRadiance;
    out.lighting.skyEmissivityClear = skyEmissivityClear;
    out.lighting.worldUnitsToMeters = out.worldUnitsToMeters;
    out.lighting.atmosphereTemperature_K = atmosphereTemperature_K;
    out.lighting.chromaR_correction = chromaR;
    out.lighting.chromaB_correction = chromaB;
    out.enableLightSampling = config.Get<bool>("renderer.enable_light_sampling", true);
    if (!out.enableLightSampling) {
        QL_LOG_INFO("Light sampling DISABLED via config: emitters are found by "
                    "BSDF sampling alone");
    }

    out.lighting.enableShadowRays = enableShadowRays ? 1u : 0u;

    // The GPU flag is narrower than the config key, on two counts.
    //
    // A named map, because there is no such thing as image-based lighting with
    // no image. What used to stand in was the fallback cubemap, 256x256 of
    // sky blue, and the flag did not distinguish it from a real HDRI -- so every
    // scene that named no map was lit by an invented sky. In a preview that is a
    // look; in a quantitative render it was 46-84% of the signal.
    //
    // RGB only, because the map has no honest interpretation in the other modes.
    // The spectral branches take the sampled RGB through
    // ConvertLinearRGBToIlluminantSpectrum and use the result as spectral
    // radiance density per nanometre, which is about 12x an ASTM G-173 sky --
    // the numbers are not in the units the code treats them as. A spectral scene
    // is lit by lighting.solar_lut and the analytic sky, which are; the map is
    // preview data, and a scene that names one in a spectral mode is warned
    // below rather than silently mis-lit.
    const bool mapNamed = out.environmentMapEnabled && !out.environmentMap.empty();
    out.lighting.enableEnvironmentMap = (mapNamed && out.mode == SpectralMode::RGB) ? 1u : 0u;

    const auto nonZero = [](const glm::vec3& c) {
        return c.r > 0.0f || c.g > 0.0f || c.b > 0.0f;
    };

    if (mapNamed && out.mode != SpectralMode::RGB) {
        diag.Warn("renderer.environment_map",
                  "  Scene names an environment map in a spectral mode, where it "
                  "is preview data only and is ignored. A map is an RGB image in "
                  "arbitrary units; reading it as spectral radiance density "
                  "would overstate a real sky by roughly 12x, so the spectral "
                  "modes do not sample one. This scene is lit by "
                  "lighting.solar_lut and the analytic sky. The path is kept, so "
                  "the same file still lights an RGB preview of this scene.");
    } else if (mapNamed && (nonZero(sunRadiance) || nonZero(skyRadiance))) {
        // An environment map is a light source, and a sky HDRI has a sun painted
        // into it. Adding an analytic sun on top of one is counting the same
        // illumination twice, and because nothing aligns the two directions the
        // usual symptom is two specular highlights on the same surface, in
        // different places. Both hosts warn about it because neither can tell
        // whether it was meant: a synthetic HDRI with no sun in it plus an
        // analytic sun is a legitimate way to light a scene.
        //
        // Only reachable in RGB now -- in the spectral modes the map contributes
        // nothing, so there is nothing to double-count and the branch above has
        // the more useful thing to say.
        diag.Warn("renderer.environment_map",
                  "  Scene has both an environment map and a non-zero analytic "
                  "sun or sky. An HDRI sky already carries its own illumination, "
                  "so the two are added and the same light is counted twice -- "
                  "typically two specular highlights in different places. Zero "
                  "lighting.sun_radiance and sky_radiance to light from the map "
                  "alone, or drop renderer.environment_map to light from the "
                  "analytic sun alone.");
    }

    // ------------------------------------------------------------------
    // Solar spectral LUT. Before the atmosphere -- see the header.
    // ------------------------------------------------------------------
    QL_LOG_INFO("Loading solar spectral LUT...");

    // A spectral mode with a sun needs a spectrum for it. The shaders no longer
    // invent one from lighting.sun_radiance, which is an RGB triple in
    // arbitrary units, so spreading it across SWIR or MWIR made the same scene
    // render differently depending on whether its spectrum happened to be
    // present. Both illuminants, not just the sun: solar_lut carries the sky
    // too, so a scene lit only by sky_radiance loses its light just as
    // completely.
    const bool spectralIlluminantNeeded =
        out.mode != SpectralMode::RGB && (nonZero(sunRadiance) || nonZero(skyRadiance));
    if (spectralIlluminantNeeded && !config.Has("lighting.solar_lut")) {
        diag.Required("lighting.solar_lut",
                      "[lighting] sun_radiance or sky_radiance is non-zero in a "
                      "spectral mode, but no solar_lut is given. An illuminant's "
                      "spectrum is scene data, like a reflectance curve -- name one, "
                      "or zero both for a scene lit only by emissive materials.");
    }

    if (config.Has("lighting.solar_lut")) {
        // Every key below means what ResolveSolarLut says it means, and the
        // facade a host calls to change the illuminant at runtime goes through
        // the same function. Two readings of these keys is the divergence this
        // repository spent a release removing.
        SolarLutRequest request;
        request.pathOrEqualEnergy = config.Get<String>("lighting.solar_lut");
        QL_LOG_INFO("  Loading solar LUT from: {}", request.pathOrEqualEnergy);

        const auto cols = config.GetArray<i32>("lighting.solar_lut_columns");
        if (cols.size() >= 1) request.directColumn = static_cast<u32>(cols[0]);
        if (cols.size() >= 2) request.diffuseColumn = static_cast<u32>(cols[1]);

        const auto normalise = config.Get<String>("lighting.solar_lut_normalise", "");
        request.normaliseUnitLuminance = (normalise == "unit_luminance");
        if (!normalise.empty() && !request.normaliseUnitLuminance) {
            diag.Warn("lighting.solar_lut_normalise",
                      "  Unknown solar_lut_normalise '" + normalise +
                          "' (known: unit_luminance)");
        }
        request.diffuseIsGlobal =
            config.Get<bool>("lighting.solar_lut_diffuse_is_global", false);

        auto result = ResolveSolarLut(request, options.baseDir, out.mode);
        if (result.has_value()) {
            auto& lut = result.value();
            for (const auto& warning : lut.warnings) {
                QL_LOG_WARN("{}", warning);
            }

            out.lighting.sunRadiance_rgb = lut.sunRadianceRgb;
            out.lighting.skyRadiance_rgb = lut.skyRadianceRgb;
            out.lighting.sunRadiance_spectral = lut.sunRadianceSpectral;
            out.lighting.skyRadiance_spectral = lut.skyRadianceSpectral;

            out.solarSunSky = std::make_pair(std::move(lut.sun), std::move(lut.sky));
            report.solarLutLoaded = true;
        } else {
            diag.Warn("lighting.solar_lut",
                      "  Failed to load solar LUT: " + result.error() +
                          " -- using LightingParams RGB fallback");
        }
    } else {
        QL_LOG_INFO("  No solar_lut specified in config, using LightingParams RGB fallback");
        QL_LOG_INFO("  NOTE: Add [lighting] solar_lut = \"path/to/file.txt\" for "
                    "spectral illumination");
    }

    // ------------------------------------------------------------------
    // [atmosphere]
    // ------------------------------------------------------------------
    QL_LOG_INFO("Configuring NN atmosphere...");

    if (config.Has("atmospheric.preset") || config.Has("atmospheric.rayleigh_enabled") ||
        config.Has("atmospheric.mie_enabled") ||
        config.Has("atmospheric.rayleigh_beta_550nm") ||
        config.Has("atmospheric.mie_beta_550nm")) {
        diag.Warn("atmospheric",
                  "  [atmospheric] is DEPRECATED and ignored; the analytic "
                  "Rayleigh/Mie atmosphere was replaced by the NN atmosphere. "
                  "Use [atmosphere] with model_pack instead.");
    }
    if (config.Has("lighting.transmittance")) {
        QL_LOG_WARN("  lighting.transmittance is DEPRECATED and no longer used; "
                    "view-path transmittance comes from the NN atmosphere");
    }
    if (config.Has("lighting.atmosphere_temperature_k")) {
        QL_LOG_WARN("  lighting.atmosphere_temperature_k is DEPRECATED; used only "
                    "as thermal-sky fallback when the NN atmosphere is disabled");
    }

    // A scene opts in by naming either key. model_pack wins when both are
    // present; naming only a preset falls back to the shipped weights, so that
    // a config renders the same atmosphere here as it does in Studio.
    if (config.Has("atmosphere.model_pack") || config.Has("atmosphere.preset")) {
        out.atmosphere.modelPackDir =
            config.Has("atmosphere.model_pack")
                ? ResolveConfigPath(config.Get<String>("atmosphere.model_pack"), options.baseDir)
                : options.atmosphereModelPackFallback;
        if (out.atmosphere.modelPackDir.empty()) {
            diag.Required("atmosphere.model_pack",
                          "[atmosphere] names a preset but no model pack was found. "
                          "Set atmosphere.model_pack, or point QUANTILOOM_ATMOS_MODELS "
                          "at the weights.");
        } else {
            out.atmosphere.enabled = true;
        }

        const auto presetName = config.Get<String>("atmosphere.preset", "clear");
        if (!out.atmosphere.ApplyPreset(presetName)) {
            diag.Warn("atmosphere.preset",
                      "  Unknown atmosphere preset '" + presetName + "', using 'clear'");
            out.atmosphere.ApplyPreset("clear");
        }

        // Out-of-domain values are clamped by the network input spec, with a
        // warning from the baker.
        const auto overrideD = [&](const char* key, double& field) {
            if (config.Has(key)) field = static_cast<double>(config.Get<f32>(key));
        };
        overrideD("atmosphere.atmos_model", out.atmosphere.atmosModel);
        overrideD("atmosphere.ihaze", out.atmosphere.ihaze);
        overrideD("atmosphere.icld", out.atmosphere.icld);
        overrideD("atmosphere.vis_km", out.atmosphere.visKm);
        overrideD("atmosphere.rainrt_mm_h", out.atmosphere.rainrtMmH);
        overrideD("atmosphere.t_ground_K", out.atmosphere.tGroundK);
        overrideD("atmosphere.rh", out.atmosphere.rh);
        overrideD("atmosphere.p_hPa", out.atmosphere.pHPa);
        overrideD("atmosphere.h2o_scale", out.atmosphere.h2oScale);
        if (config.Has("atmosphere.lut_a_samples")) {
            out.atmosphere.lutASamples = config.Get<i32>("atmosphere.lut_a_samples");
        }
        if (config.Has("atmosphere.lut_az_samples")) {
            out.atmosphere.lutAzSamples = config.Get<i32>("atmosphere.lut_az_samples");
        }

        // Sun geometry. A scene that states one means it, whoever is rendering;
        // one that does not gets it derived -- frozen here for a batch render,
        // left to follow the camera for an interactive one.
        const glm::vec3& sunDir = out.lighting.sunDirection;
        const double lightingZenith =
            glm::degrees(std::acos(std::clamp(sunDir.y, -1.0f, 1.0f)));
        const double lightingAzimuth = glm::degrees(std::atan2(sunDir.x, sunDir.z));
        if (config.Has("atmosphere.sun_zenith_deg")) {
            out.atmosphere.sunFromLighting = false;
            out.atmosphere.sunZenithDeg =
                static_cast<double>(config.Get<f32>("atmosphere.sun_zenith_deg"));
            out.atmosphere.sunAzimuthDeg = static_cast<double>(config.Get<f32>(
                "atmosphere.sun_azimuth_deg", static_cast<f32>(lightingAzimuth)));
            if (std::abs(out.atmosphere.sunZenithDeg - lightingZenith) > 2.0) {
                QL_LOG_WARN("  atmosphere.sun_zenith_deg = {:.1f} differs from "
                            "lighting.sun_direction zenith {:.1f} by > 2 deg; "
                            "using the [atmosphere] value for the NN inputs",
                            out.atmosphere.sunZenithDeg, lightingZenith);
            }
        } else if (options.freezeDerivedAtmosGeometry) {
            out.atmosphere.sunFromLighting = false;  // Resolve here, once
            out.atmosphere.sunZenithDeg = lightingZenith;
            out.atmosphere.sunAzimuthDeg = lightingAzimuth;
        }

        // Observer altitude, same rule.
        if (config.Has("atmosphere.h1_km")) {
            out.atmosphere.h1FromCamera = false;
            out.atmosphere.h1Km = static_cast<double>(config.Get<f32>("atmosphere.h1_km"));
        } else if (options.freezeDerivedAtmosGeometry) {
            out.atmosphere.h1FromCamera = false;  // Resolve here, once
            out.atmosphere.h1Km = std::max(
                static_cast<double>(out.camera.GetPosition().y * out.worldUnitsToMeters) /
                    1000.0,
                0.0);
        }

        QL_LOG_INFO("  NN atmosphere: preset '{}', model pack '{}'{}",
                    out.atmosphere.preset, out.atmosphere.modelPackDir,
                    config.Has("atmosphere.model_pack") ? "" : " (resolved)");
        QL_LOG_INFO("  Sun zenith {:.1f} deg, h1 {:.3f} km", out.atmosphere.sunZenithDeg,
                    out.atmosphere.h1Km);
    } else {
        QL_LOG_INFO("  No [atmosphere] section - atmosphere disabled");
    }
    if (out.atmosphere.preset == "disabled") out.atmosphere.enabled = false;
    report.atmosphereEnabled = out.atmosphere.enabled;

    // An active atmosphere owns the thermal-sky temperature: AtmosSkyRadianceIR
    // divides the network's zenith downwelling by B(T_air) to recover an
    // emissivity, so a T_air unrelated to the bake distorts the horizon ramp.
    if (out.atmosphere.enabled) {
        out.lighting.atmosphereTemperature_K = static_cast<f32>(out.atmosphere.tGroundK);
        QL_LOG_INFO("  Updated atmosphere temperature from NN config: {:.1f} K",
                    out.lighting.atmosphereTemperature_K);
        // The network's own zenith downwelling is a measurement of this sky,
        // spectrally resolved; the analytic correlation is the substitute for
        // not having one. Two models of the same sky would be one too many.
        if (out.lighting.skyEmissivityClear > 0.0f) {
            QL_LOG_INFO("  atmosphere.sky_model is ignored: the NN atmosphere "
                        "supplies the sky's downwelling directly");
            out.lighting.skyEmissivityClear = 0.0f;
        }
    }
    QL_LOG_INFO("  NN atmosphere: {}", out.atmosphere.enabled ? "ENABLED" : "DISABLED");

    // ------------------------------------------------------------------
    // [sensor]
    // ------------------------------------------------------------------
    out.sensorEnabled = config.Get<bool>("sensor.enabled", false);
    // Parsed whether or not it is enabled, so a host can turn it on later
    // without rereading the file.
    out.sensor = PostprocessConfig::ParseSensorParams(config);

    // ------------------------------------------------------------------
    // [thermal]
    // ------------------------------------------------------------------
    // The surface energy balance. What it produces is a temperature per
    // triangle, which the renderer uploads and the closest-hit shader reads in
    // place of the material's own -- so a scene that enables this stops
    // needing a temperature typed into it at all.
    //
    // Per-material properties are read from [[materials]] below, beside the IR
    // overrides they belong with.
    out.thermal.enabled = config.Get<bool>("thermal.enabled", false);
    if (out.thermal.enabled) {
        out.thermal.time_h = config.Get<f64>("thermal.time_h", 12.0);
        out.thermal.startTime_h = config.Get<f64>("thermal.start_time_h", 0.0);
        out.thermal.timestep_s = config.Get<f64>("thermal.timestep_s", 60.0);
        out.thermal.nodeCount = config.Get<u32>("thermal.layers", 10);
        out.thermal.initialTemperature_K =
            config.Get<f64>("thermal.initial_temperature_k", 288.15);
        out.thermal.exchangeRays = config.Get<u32>("thermal.exchange_rays", 256);
        out.thermal.exchangeTopK = config.Get<u32>("thermal.exchange_top_k", 32);
        out.thermal.sunIrradiance_W_m2 = config.Get<f64>("thermal.sun_irradiance_w_m2", 0.0);
        out.thermal.diffuseIrradiance_W_m2 =
            config.Get<f64>("thermal.diffuse_irradiance_w_m2", 0.0);
        out.thermal.checkpointStride_h = config.Get<f64>("thermal.checkpoint_stride_h", 1.0);
        out.thermal.forcingFile =
            ResolveConfigPath(config.GetString("thermal.forcing_file", ""), options.baseDir);

        const auto initial = config.GetString("thermal.initial", "steady");
        if (initial == "uniform") {
            out.thermal.initial = thermal::InitialCondition::Uniform;
        } else if (initial == "steady") {
            out.thermal.initial = thermal::InitialCondition::Steady;
        } else {
            diag.Warn("thermal.initial",
                      "  unknown thermal.initial '" + initial +
                          "', expected steady|uniform. Using steady.");
        }

        // The air the surfaces exchange with is the air the sky model uses --
        // one atmosphere per scene. Same for the sun: the balance is lit by
        // whatever lights the render.
        out.thermal.airTemperature_K = config.Get<f64>(
            "thermal.air_temperature_k", static_cast<f64>(out.lighting.atmosphereTemperature_K));
        out.thermal.sunDirection = out.lighting.sunDirection;

        // How wet the air is, which decides how much a wet surface can
        // evaporate. Read from [atmosphere] rather than duplicated here: it is
        // the same humidity the clear-sky model derives its dew point from,
        // and a scene with two of them would be a scene with two atmospheres.
        out.thermal.relativeHumidity = config.Get<f64>("atmosphere.relative_humidity", 50.0);

        // The sky the surfaces radiate against. With the clear-sky model on,
        // that is the effective temperature its emissivity implies rather than
        // the air temperature -- which is the whole difference between a
        // surface that frosts overnight and one that does not.
        out.thermal.skyTemperature_K =
            out.lighting.skyEmissivityClear > 0.0f
                ? skythermal::EffectiveSkyTemperatureK(
                      out.thermal.airTemperature_K,
                      static_cast<f64>(out.lighting.skyEmissivityClear))
                : static_cast<f64>(out.lighting.atmosphereTemperature_K);

        QL_LOG_INFO("  Thermal: {:.1f} h to {:.1f} h at {:.0f} s, {} layers, air {:.1f} K, "
                    "sky {:.1f} K",
                    out.thermal.startTime_h, out.thermal.time_h, out.thermal.timestep_s,
                    out.thermal.nodeCount, out.thermal.airTemperature_K,
                    out.thermal.skyTemperature_K);
    }

    if (diag.failed()) {
        return ResolveResult::Err(diag.firstError());
    }
    return ResolveResult(std::move(out));
}

// ============================================================================
// The illuminant
// ============================================================================

Result<ResolvedSolarLut, String> ResolveSolarLut(const SolarLutRequest& request,
                                                 const String& baseDir,
                                                 SpectralMode mode) {
    using LutResult = Result<ResolvedSolarLut, String>;

    if (request.pathOrEqualEnergy.empty()) {
        return LutResult::Err("No illuminant named");
    }

    // "equal_energy" is the one illuminant that is not a file: a flat spectrum
    // at unit luminance, CIE illuminant E. It is the neutral reference -- what
    // a material looks like under light that favours no wavelength -- and it is
    // not sRGB white, which is D65.
    auto loaded =
        request.pathOrEqualEnergy == "equal_energy"
            ? Result<std::pair<SpectralCurve, SpectralCurve>, String>(
                  std::make_pair(MakeEqualEnergyIlluminant(), MakeEqualEnergyIlluminant()))
            : SpectralIO::LoadLibRadtranSunAndSky(
                  ResolveConfigPath(request.pathOrEqualEnergy, baseDir), "nm",
                  request.directColumn, request.diffuseColumn, request.diffuseIsGlobal);

    if (!loaded.has_value()) {
        return LutResult::Err(loaded.error());
    }

    ResolvedSolarLut out;
    auto& [sunCurve, skyCurve] = loaded.value();

    // Reference illuminants are published as relative spectra -- D65 is
    // normalised to 100 at 560 nm -- so their absolute level is arbitrary.
    // Normalising to unit luminance puts the illuminant at Y = 1, which is what
    // makes D65 come out as sRGB (1, 1, 1) exactly. It is also the honest
    // separation of white balance from exposure.
    if (request.normaliseUnitLuminance) {
        const auto rgb = SpectralIrradianceToLinearSrgb(sunCurve);
        const f32 Y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
        if (Y > 0.0f) {
            // Both curves by the sun's luminance, not each by its own: scaling
            // them separately would discard the ratio between sun and sky,
            // which is the one thing a measured pair actually tells you.
            for (auto& v : sunCurve.samples) v.second /= Y;
            for (auto& v : skyCurve.samples) v.second /= Y;
            QL_LOG_INFO("  Illuminant normalised to unit luminance (was Y={:.4g})", Y);
        }
    }

    // Does the spectrum actually cover the band being rendered?
    // SpectralCurve::Evaluate clamps to its endpoints rather than returning
    // zero, so a curve that stops short does not fail -- it holds its last
    // value flat across everything above it, and the render looks plausible.
    // CIE D65 stops at 830 nm, which makes it a fine reference illuminant for
    // RGB and a silently wrong one for SWIR upward.
    if (const auto band = GetFusedBandInfo(mode); band && !sunCurve.samples.empty()) {
        const f32 curveMin = sunCurve.samples.front().first;
        const f32 curveMax = sunCurve.samples.back().first;
        if (curveMin > band->lambdaMinNm || curveMax < band->lambdaMaxNm) {
            out.warnings.push_back(
                "  Illuminant spans [" + std::to_string(static_cast<int>(curveMin)) +
                ", " + std::to_string(static_cast<int>(curveMax)) +
                "] nm but this mode renders [" +
                std::to_string(static_cast<int>(band->lambdaMinNm)) + ", " +
                std::to_string(static_cast<int>(band->lambdaMaxNm)) +
                "] nm. Evaluate() clamps, so the uncovered part is held flat at the "
                "nearest endpoint rather than left dark -- the result will look "
                "reasonable and mean nothing.");
        }
    }

    // One illuminant, every mode. The spectral paths sample these curves per
    // wavelength; RGB and VIS_Fused need a colour, and taking it from the same
    // curve is what stops the two halves of the renderer describing different
    // suns. It replaces whatever sun_radiance said, which was a triple in
    // arbitrary units with no defined relationship to the spectrum beside it.
    out.sunRadianceRgb = SpectralIrradianceToLinearSrgb(sunCurve);
    out.skyRadianceRgb = SpectralIrradianceToLinearSrgb(skyCurve);
    out.sunRadianceSpectral =
        (out.sunRadianceRgb.r + out.sunRadianceRgb.g + out.sunRadianceRgb.b) / 3.0f;
    out.skyRadianceSpectral =
        (out.skyRadianceRgb.r + out.skyRadianceRgb.g + out.skyRadianceRgb.b) / 3.0f;
    QL_LOG_INFO("  Illuminant colour from the spectrum: sun [{:.4g}, {:.4g}, {:.4g}], "
                "sky [{:.4g}, {:.4g}, {:.4g}]",
                out.sunRadianceRgb.r, out.sunRadianceRgb.g, out.sunRadianceRgb.b,
                out.skyRadianceRgb.r, out.skyRadianceRgb.g, out.skyRadianceRgb.b);

    out.sun = std::move(sunCurve);
    out.sky = std::move(skyCurve);
    return LutResult(std::move(out));
}

// ============================================================================
// Scene-dependent keys
// ============================================================================

Result<ResolvedMaterialSpectra, String> ResolveMaterialSpectra(
    const Config& config, Scene& scene, const ResolvedRenderConfig& resolved,
    const ConfigApplyOptions& options, ConfigApplyReport& report) {
    using SpectraResult = Result<ResolvedMaterialSpectra, String>;

    ResolvedMaterialSpectra out;
    Diagnostics diag(options, report);

    // ------------------------------------------------------------------
    // Default IR surface temperature
    // ------------------------------------------------------------------
    // Standard glTF/USD materials carry no temperature, which silences the
    // Planck emission term entirely in MWIR/LWIR. Backfill a scene-wide ambient
    // temperature for materials without their own.
    if (IsIRFusedMode(resolved.mode) || resolved.mode == SpectralMode::Single) {
        const f32 defaultTemperature_K = resolved.defaultTemperatureK;
        if (defaultTemperature_K < 150.0f || defaultTemperature_K > 1000.0f) {
            diag.Warn("scene.default_temperature_k",
                      "scene.default_temperature_k=" +
                          std::to_string(defaultTemperature_K) +
                          "K is outside typical range [150, 1000], check config");
        }
        out.temperatureBackfilled =
            ApplyDefaultIRTemperature(scene.materials, defaultTemperature_K);
        if (out.temperatureBackfilled > 0) {
            QL_LOG_INFO("  Applied default surface temperature {:.1f} K to {} "
                        "material(s) without temperature data",
                        defaultTemperature_K, out.temperatureBackfilled);
        }
    }
    report.materialsTemperatureBackfilled = out.temperatureBackfilled;

    // ------------------------------------------------------------------
    // [[materials]] IR overrides
    // ------------------------------------------------------------------
    // A flat emissivity and transmittance across the thermal bands, with
    // reflectance from energy conservation. Two sample points is all a constant
    // needs, and they bracket MWIR and LWIR so Evaluate() holds the value flat
    // between and beyond them.
    //
    //   [[materials]]
    //   name = "Panel"
    //   ir_emissivity = 0.92
    //   ir_transmittance = 0.0
    //   ir_temperature_k = 310.0
    //
    // This began as a Studio-only reading of the file, which meant a config the
    // GUI honoured rendered without it from the CLI.
    //
    // [material_overrides] spells the same keys under a table keyed by
    // material name:
    //
    //   [material_overrides.Panel]
    //   ir_temperature_k = 320.0
    //
    // Same vocabulary, different merge behaviour, which is the whole reason it
    // exists. Config::MergedWith replaces arrays whole -- array elements carry
    // no identity to pair them up by -- so an override document naming one
    // material's temperature would delete every other [[materials]] entry.
    // Tables merge key by key, so the table form survives being layered, which
    // is what a per-job override in a batch manifest needs.
    //
    // Applied after the array, so a manifest line wins over the scene file it
    // is overriding. Both forms go through the same loop body: two spellings
    // of one vocabulary, not two vocabularies.
    Vector<std::pair<String, Config>> materialTables;
    for (auto& table : config.GetTableArray("materials")) {
        auto name = table.GetString("name", "");
        if (name.empty()) continue;
        materialTables.emplace_back(std::move(name), std::move(table));
    }
    for (const auto& overrideName : config.GetSubtableNames("material_overrides")) {
        auto table = config.GetTable("material_overrides." + overrideName);
        if (table.has_value()) {
            materialTables.emplace_back(overrideName, std::move(table.value()));
        }
    }

    for (const auto& [name, matTable] : materialTables) {
        auto it = std::find_if(scene.materials.begin(), scene.materials.end(),
                               [&name](const Material& m) { return m.name == name; });
        if (it == scene.materials.end()) {
            diag.Warn("materials",
                      "  a material override names '" + name + "', which the scene "
                      "has no material by");
            continue;
        }

        // PBR overrides. Absent keys leave the loaded value alone, so a
        // [[materials]] entry that only sets a temperature does not silently
        // reset the surface to white plastic.
        if (const auto colour = matTable.GetFloatArray("base_color"); colour.size() >= 3) {
            it->baseColorFactor = glm::vec4(colour[0], colour[1], colour[2],
                                            colour.size() >= 4 ? colour[3]
                                                               : it->baseColorFactor.a);
        }
        if (matTable.Has("metallic")) {
            it->metallicFactor = matTable.GetFloat("metallic", it->metallicFactor);
        }
        if (matTable.Has("roughness")) {
            it->roughnessFactor = matTable.GetFloat("roughness", it->roughnessFactor);
        }
        if (const auto emissive = matTable.GetFloatArray("emissive"); emissive.size() >= 3) {
            it->emissiveFactor = glm::vec3(emissive[0], emissive[1], emissive[2]);
        }

        // Transmission, dispersion and participating media. The fields have
        // always reached the GPU, but only glTF's KHR extensions could set
        // them -- a config could not describe a piece of glass, and the
        // prism and water scenes carried their intent in comments the reader
        // never saw. Same absent-key-leaves-the-loaded-value rule as above.
        if (matTable.Has("ior")) {
            it->ior = matTable.GetFloat("ior", it->ior);
        }
        if (matTable.Has("transmission")) {
            it->transmission = matTable.GetFloat("transmission", it->transmission);
        }
        if (matTable.Has("dispersion")) {
            it->dispersion = matTable.GetFloat("dispersion", it->dispersion);
        }
        if (const auto attenuation = matTable.GetFloatArray("attenuation_color");
            attenuation.size() >= 3) {
            it->attenuationColor = glm::vec3(attenuation[0], attenuation[1], attenuation[2]);
        }
        if (matTable.Has("attenuation_distance")) {
            it->attenuationDistance =
                matTable.GetFloat("attenuation_distance", it->attenuationDistance);
        }
        if (matTable.Has("thickness")) {
            it->thicknessFactor = matTable.GetFloat("thickness", it->thicknessFactor);
        }
        if (matTable.Has("volume_density")) {
            it->volumeDensity = matTable.GetFloat("volume_density", it->volumeDensity);
        }
        if (matTable.Has("scattering_coeff")) {
            it->scatteringCoeff = matTable.GetFloat("scattering_coeff", it->scatteringCoeff);
        }
        if (matTable.Has("absorption_coeff")) {
            it->absorptionCoeff = matTable.GetFloat("absorption_coeff", it->absorptionCoeff);
        }
        if (matTable.Has("phase_g")) {
            it->phaseG = matTable.GetFloat("phase_g", it->phaseG);
        }

        // The measured-material database entry this surface stands for. Set
        // here rather than only from glTF extras, so an assignment made in
        // Studio survives being saved: the NMF reconstruction loop below runs
        // after this block and picks it up with no further plumbing.
        if (matTable.Has("spectral_material_type")) {
            it->quantiloomMaterialType =
                matTable.GetString("spectral_material_type", it->quantiloomMaterialType);
        }
        if (matTable.Has("spectral_material_ref")) {
            it->quantiloomMaterialRef =
                matTable.GetString("spectral_material_ref", it->quantiloomMaterialRef);
        }

        // Endmembers. The plural key holds the whole list: entry 0 is the same
        // slot the singular key writes, so a one-entry list and the singular
        // form mean exactly the same thing. Naming both is refused rather than
        // merged, because either reading of the intent would be a guess.
        if (matTable.Has("spectral_material_refs")) {
            const auto refs = matTable.GetStringArray("spectral_material_refs");
            if (matTable.Has("spectral_material_ref")) {
                diag.Fatal("materials.spectral_material_refs",
                           "  Material '" + name +
                               "': spectral_material_ref and spectral_material_refs are "
                               "both set. The plural form already includes the first "
                               "endmember -- keep one.");
            } else if (refs.empty()) {
                diag.Warn("materials.spectral_material_refs",
                          "  Material '" + name + "': spectral_material_refs is empty");
            } else if (refs.size() > static_cast<size_t>(Material::MAX_ENDMEMBERS)) {
                diag.Fatal("materials.spectral_material_refs",
                           "  Material '" + name + "': " + std::to_string(refs.size()) +
                               " endmembers, at most " + std::to_string(Material::MAX_ENDMEMBERS) +
                               " fit in an RGBA weight texture");
            } else {
                it->quantiloomMaterialRef = refs.front();
                it->quantiloomExtraRefs.assign(refs.begin() + 1, refs.end());
            }
        }

        if (matTable.Has("spectral_unmix")) {
            const auto mode = matTable.GetString("spectral_unmix", "auto");
            if (mode == "auto") {
                it->spectralUnmixMode = Material::SpectralUnmixMode::Auto;
            } else if (mode == "texture") {
                it->spectralUnmixMode = Material::SpectralUnmixMode::Texture;
            } else if (mode == "off") {
                it->spectralUnmixMode = Material::SpectralUnmixMode::Off;
            } else {
                diag.Warn("materials.spectral_unmix",
                          "  Material '" + name + "': unknown spectral_unmix '" + mode +
                              "', expected auto|texture|off. Using auto.");
                it->spectralUnmixMode = Material::SpectralUnmixMode::Auto;
            }
        }

        if (matTable.Has("spectral_weight_texture")) {
            it->spectralWeightTexturePath = matTable.GetString("spectral_weight_texture", "");
            if (it->spectralUnmixMode != Material::SpectralUnmixMode::Texture) {
                diag.Warn("materials.spectral_weight_texture",
                          "  Material '" + name +
                              "': spectral_weight_texture is only read with "
                              "spectral_unmix = \"texture\"; ignoring it");
            }
        } else if (it->spectralUnmixMode == Material::SpectralUnmixMode::Texture) {
            diag.Fatal("materials.spectral_unmix",
                       "  Material '" + name +
                           "': spectral_unmix = \"texture\" needs "
                           "spectral_weight_texture to name one");
        }

        constexpr f32 kMwirNm = 4000.0f;
        constexpr f32 kLwirNm = 10000.0f;
        const f32 emissivity = matTable.GetFloat("ir_emissivity", 0.0f);
        const f32 transmittance = matTable.GetFloat("ir_transmittance", 0.0f);

        if (emissivity > 0.0f) {
            it->irEmissivityCurve = {{kMwirNm, emissivity}, {kLwirNm, emissivity}};
        }
        if (transmittance > 0.0f) {
            it->irTransmittanceCurve = {{kMwirNm, transmittance}, {kLwirNm, transmittance}};
        }
        const f32 reflectance = 1.0f - emissivity - transmittance;
        if (reflectance > 0.0f) {
            it->irReflectanceCurve = {{kMwirNm, reflectance}, {kLwirNm, reflectance}};
        }
        if (matTable.Has("ir_temperature_k")) {
            it->irTemperature_K = matTable.GetFloat("ir_temperature_k", 0.0f);
        }

        // Per-texel temperature. The path is stored raw and resolved against
        // the config directory at mount time, the same split
        // spectral_weight_texture uses. Scale and offset stand on their own so
        // a config can retune a glTF-provided map without renaming it.
        if (matTable.Has("temperature_texture")) {
            it->temperatureTexturePath = matTable.GetString("temperature_texture", "");
        }
        if (matTable.Has("temperature_scale")) {
            it->temperatureScale = matTable.GetFloat("temperature_scale", it->temperatureScale);
        }
        if (matTable.Has("temperature_offset")) {
            it->temperatureOffset =
                matTable.GetFloat("temperature_offset", it->temperatureOffset);
        }

        // Thermal properties, for the surface energy balance. Read into a
        // side table keyed by name rather than onto the Material, which is a
        // layout contract Quantiloom-Qt reads by offset and has no business
        // carrying solver inputs. Naming a conductivity is what opts a
        // material in; everything else has a default that describes masonry.
        if (matTable.Has("thermal_conductivity_w_mk")) {
            thermal::ThermalMaterial props;
            props.conductivity_W_mK = matTable.GetFloat("thermal_conductivity_w_mk", 0.0f);
            props.density_kg_m3 = matTable.GetFloat("density_kg_m3", 2000.0f);
            props.specificHeat_J_kgK = matTable.GetFloat("specific_heat_j_kgk", 900.0f);
            props.thickness_m = matTable.GetFloat("thickness_m", 0.2f);
            props.convection_W_m2K = matTable.GetFloat("convection_h_w_m2k", 5.0f);
            props.shortwaveAbsorptivity =
                matTable.GetFloat("shortwave_absorptivity", 0.7f);
            props.wetnessFactor = matTable.GetFloat("wetness_factor", 0.0f);
            props.interiorTemperature_K =
                matTable.GetFloat("interior_temperature_k", 293.15f);

            const auto boundary = matTable.GetString("interior_bc", "adiabatic");
            if (boundary == "fixed") {
                props.interiorBoundary = thermal::InteriorBoundary::FixedTemperature;
            } else if (boundary == "adiabatic") {
                props.interiorBoundary = thermal::InteriorBoundary::Adiabatic;
            } else {
                diag.Warn("materials.interior_bc",
                          "  Material '" + name + "': unknown interior_bc '" + boundary +
                              "', expected adiabatic|fixed. Using adiabatic.");
            }

            out.thermalMaterials[name] = props;
            QL_LOG_INFO("  Material '{}': thermal k={:.2f} W/mK, rho c={:.0f} J/m3K, "
                        "d={:.3f} m, h={:.1f} W/m2K",
                        name, props.conductivity_W_mK,
                        props.density_kg_m3 * props.specificHeat_J_kgK, props.thickness_m,
                        props.convection_W_m2K);
        }

        ++out.materialsOverridden;
        QL_LOG_INFO("  Material '{}': IR override e={:.3f} t={:.3f} r={:.3f} T={:.1f} K",
                    name, emissivity, transmittance, std::max(reflectance, 0.0f),
                    it->irTemperature_K);
    }
    report.materialsOverridden = out.materialsOverridden;

    // ------------------------------------------------------------------
    // [[duplicates]] -- nodes pasted in Studio
    // ------------------------------------------------------------------
    // A node the scene file did not place: a shallow copy of one it did.
    // Same sharing rule as the editor's copy-paste (geometry and materials
    // shared, transform its own), and it lives here rather than in either
    // host so a config written after a paste renders the same everywhere --
    // Studio and the CLI.
    //
    //   [[duplicates]]
    //   source = "Hull"
    //   name = "Hull.001"
    //   translation = [2.0, 0.0, 0.0]   # [[nodes]] grammar; omitted = in place
    //
    // Resolved before [[nodes]] so a [[nodes]] entry may address a duplicate
    // by name; entries resolve in file order, so a duplicate may itself be
    // the source of a later one.
    for (const auto& dupTable : config.GetTableArray("duplicates")) {
        const auto source = dupTable.GetString("source", "");
        const auto name = dupTable.GetString("name", "");
        if (source.empty() || name.empty()) {
            diag.Warn("duplicates",
                      "  [[duplicates]] needs both source and name");
            continue;
        }

        const auto it = std::find_if(scene.nodes.begin(), scene.nodes.end(),
                                     [&source](const SceneNode& n) { return n.name == source; });
        if (it == scene.nodes.end()) {
            diag.Warn("duplicates",
                      "  [[duplicates]] sources '" + source +
                          "', which the scene has no node by");
            continue;
        }

        const TransformKeys keys = ParseTransformKeys(dupTable);
        if (keys.present && !keys.valid) {
            diag.Warn("duplicates",
                      "  [[duplicates]] '" + name + "' has a matrix of " +
                          std::to_string(keys.matrixCount) + " numbers; 16 are needed");
            continue;
        }

        // Copy before push_back: the insertion invalidates `it`
        SceneNode copy = *it;
        copy.name = name;
        copy.active = true;
        if (keys.present) {
            copy.transform = keys.transform;
        }
        scene.nodes.push_back(std::move(copy));

        ++out.nodesDuplicated;
        QL_LOG_INFO("  Node '{}' duplicated as '{}'", source, name);
    }
    report.nodesDuplicated = out.nodesDuplicated;

    // ------------------------------------------------------------------
    // scene.removed_nodes -- nodes deleted in Studio
    // ------------------------------------------------------------------
    // The other half of runtime scene editing: a node the scene file placed
    // but the user deleted. Tombstoned, not erased, exactly like the editor's
    // RemoveNode -- the node keeps its index and contributes no instances.
    //
    //   [scene]
    //   removed_nodes = ["Backdrop", "Node_3"]
    //
    // Resolved after [[duplicates]] so a duplicate can be removed too (a
    // config written by hand may want that; one written by Studio simply
    // drops the [[duplicates]] entry instead).
    for (const auto& name : config.GetStringArray("scene.removed_nodes")) {
        const auto it = std::find_if(scene.nodes.begin(), scene.nodes.end(),
                                     [&name](const SceneNode& n) { return n.name == name; });
        if (it == scene.nodes.end()) {
            diag.Warn("scene.removed_nodes",
                      "  removed_nodes names '" + name +
                          "', which the scene has no node by");
            continue;
        }
        it->active = false;
        ++out.nodesRemoved;
        QL_LOG_INFO("  Node '{}' removed", name);
    }
    report.nodesRemoved = out.nodesRemoved;

    // ------------------------------------------------------------------
    // [[nodes]] transform overrides
    // ------------------------------------------------------------------
    // A world transform for a node the scene file already placed, so a
    // position arrived at by moving something in Studio can be written down and
    // rendered again -- by Studio, and by the CLI, which is the point of it
    // living here rather than in either host.
    //
    //   [[nodes]]
    //   name = "Hull"
    //   translation = [0.0, 1.5, 0.0]
    //   rotation_euler_degrees = [0.0, 35.0, 0.0]
    //   scale = [1.0, 1.0, 1.0]
    //
    // Or, for something a rotation order cannot express, the matrix itself:
    //
    //   matrix = [ ... 16 numbers, column-major ... ]
    //
    // Grammar shared with [[duplicates]]; see ParseTransformKeys.
    for (const auto& nodeTable : config.GetTableArray("nodes")) {
        const auto name = nodeTable.GetString("name", "");
        if (name.empty()) continue;

        auto it = std::find_if(scene.nodes.begin(), scene.nodes.end(),
                               [&name](const SceneNode& n) { return n.name == name; });
        if (it == scene.nodes.end()) {
            diag.Warn("nodes",
                      "  [[nodes]] names '" + name + "', which the scene has no node by");
            continue;
        }

        const TransformKeys keys = ParseTransformKeys(nodeTable);
        if (!keys.present) {
            diag.Warn("nodes",
                      "  [[nodes]] '" + name + "' sets no transform; give translation, "
                      "rotation_euler_degrees and scale, or matrix");
            continue;
        }
        if (!keys.valid) {
            diag.Warn("nodes",
                      "  [[nodes]] '" + name + "' has a matrix of " +
                          std::to_string(keys.matrixCount) + " numbers; 16 are needed");
            continue;
        }
        it->transform = keys.transform;

        ++out.nodesTransformed;
        QL_LOG_INFO("  Node '{}': transform overridden", name);
    }
    report.nodesTransformed = out.nodesTransformed;

    // ------------------------------------------------------------------
    // sRGB upsampling gate
    // ------------------------------------------------------------------
    const bool requireQuantitative = (resolved.mode == SpectralMode::Multispectral ||
                                      resolved.mode == SpectralMode::MWIR_Fused ||
                                      resolved.mode == SpectralMode::LWIR_Fused ||
                                      resolved.mode == SpectralMode::SWIR_Fused);

    if (requireQuantitative && (resolved.failOnSrgbUpsample || resolved.logMaterialSources)) {
        QL_LOG_INFO("Validating material spectral sources for quantitative mode...");

        bool hasInvalidMaterials = false;
        for (const auto& mat : scene.materials) {
            const char* sourceStr = "Unknown";
            switch (mat.spectralSource) {
                case Material::SpectralSource::Measured:
                    sourceStr = "Measured (quantitative)";
                    break;
                case Material::SpectralSource::RGBUpsampled:
                    sourceStr = "RGB-upsampled (NOT quantitative)";
                    hasInvalidMaterials = true;
                    break;
                case Material::SpectralSource::Procedural:
                    sourceStr = "Procedural";
                    break;
                default:
                    sourceStr = "Unknown";
                    break;
            }

            if (resolved.logMaterialSources) {
                QL_LOG_INFO("  Material '{}': source = {}", mat.name, sourceStr);
            }
            if (mat.spectralSource == Material::SpectralSource::RGBUpsampled) {
                QL_LOG_WARN("  ⚠️  Material '{}' uses RGB-upsampled spectra (not quantitative)",
                            mat.name);
            }
        }

        if (hasInvalidMaterials && resolved.failOnSrgbUpsample) {
            return SpectraResult::Err(
                "ABORTED: RGB-upsampled materials detected. They are NOT suitable "
                "for quantitative analysis. To proceed with a non-quantitative "
                "preview, set quality.fail_on_srgb_upsample = false; for "
                "quantitative results, provide measured spectral material data.");
        }
        if (hasInvalidMaterials && !resolved.failOnSrgbUpsample) {
            QL_LOG_WARN("⚠️  WARNING: Proceeding with RGB-upsampled materials "
                        "(non-quantitative preview).");
        }
    }

    // ------------------------------------------------------------------
    // [spectral_curves]: material name -> reflectance CSV
    // ------------------------------------------------------------------
    QL_LOG_INFO("Loading spectral curves...");

    if (config.HasSection("spectral_curves")) {
        for (const auto& [materialName, csvPath] : config.GetSection("spectral_curves")) {
            QL_LOG_INFO("  Loading spectral curve for '{}' from '{}'", materialName, csvPath);

            auto result =
                SpectralIO::LoadSpectralCurveCSV(ResolveConfigPath(csvPath, options.baseDir));
            if (!result) {
                diag.Warn("spectral_curves",
                          "    Failed to load: " + result.error());
                continue;
            }

            SpectralCurve curve;
            curve.samples = result.value();
            SpectralCurveGPU gpuCurve = SpectralCurveGPU::FromCPU(curve);

            const i32 curveIndex = static_cast<i32>(out.curves.size());
            out.materialNameToCurve[materialName] = curveIndex;
            out.curves.push_back(gpuCurve);

            QL_LOG_INFO("    Loaded: {} samples, λ=[{:.1f}, {:.1f}] nm → curve index {}",
                        gpuCurve.numSamples, gpuCurve.startWavelength_nm,
                        gpuCurve.GetWavelength(gpuCurve.numSamples - 1), curveIndex);
        }
    }

    QL_LOG_INFO("  Total spectral curves from CSV: {}", out.curves.size());

    // ------------------------------------------------------------------
    // [spectral] NMF basis: measured curves reconstructed for materials that
    // carry a quantiloom_* reference in their glTF extras.
    // ------------------------------------------------------------------
    // Present but empty means "not configured": several shipped configs carry
    // the keys with "" as a placeholder, and passing that through logged a
    // loader error on every render, which teaches people that error lines are
    // noise.
    const auto basisFilePath = config.Get<String>("spectral.basis_file", "");
    const auto materialsJsonPath = config.Get<String>("spectral.materials_json", "");

    if (!basisFilePath.empty() && !materialsJsonPath.empty()) {
        const auto activeBand = config.Get<String>("spectral.band", "VIS");

        QL_LOG_INFO("Loading SpectralBaker NMF basis data...");
        QL_LOG_INFO("  Basis file: {}", basisFilePath);
        QL_LOG_INFO("  Materials JSON: {}", materialsJsonPath);
        QL_LOG_INFO("  Active band: {}", activeBand);

        SpectralBasisLoader basisLoader;
        if (basisLoader.Load(ResolveConfigPath(basisFilePath, options.baseDir),
                             ResolveConfigPath(materialsJsonPath, options.baseDir))) {
            QL_LOG_INFO("  SpectralBaker data loaded: {} materials, {} bands",
                        basisLoader.GetMaterialCount(), basisLoader.GetNumBands());

            // One reconstruction per distinct reference, not per material that
            // names it: two surfaces of the same measured concrete are the
            // same curve, and uploading it twice costs a buffer slot and an
            // opportunity for them to disagree.
            struct ResolvedRef {
                i32 curveIndex = -1;
                glm::vec3 colorLinear{0.0f};
                bool hasVisColor = false;
            };
            std::unordered_map<String, ResolvedRef> refCache;

            // A reference resolved to a curve index and to the colour it would
            // read as under D65. Returns nullptr, having logged why, when the
            // reference names nothing or its band cannot be reconstructed.
            const auto resolveRef =
                [&](const String& ref, const String& materialName) -> const ResolvedRef* {
                if (auto cached = refCache.find(ref); cached != refCache.end()) {
                    return cached->second.curveIndex >= 0 ? &cached->second : nullptr;
                }

                const MaterialSpectralData* spectralData = basisLoader.FindMaterial(ref);
                if (!spectralData) {
                    spectralData = basisLoader.FindMaterialPartial(ref);
                    if (spectralData) {
                        // The full name, because a partial match is where a
                        // mixture silently binds two endmembers to the same
                        // library entry, or to the wrong one.
                        QL_LOG_INFO("    '{}' matched via partial search: '{}'",
                                    ref, spectralData->name);
                    }
                }
                if (!spectralData) {
                    QL_LOG_WARN("    Material '{}': '{}' not found in SpectralBaker database",
                                materialName, ref);
                    refCache[ref] = ResolvedRef{};
                    return nullptr;
                }

                SpectralCurveGPU gpuCurve =
                    basisLoader.ReconstructCurveGPU(spectralData->name, activeBand);
                if (gpuCurve.numSamples == 0) {
                    QL_LOG_WARN("    Material '{}': failed to reconstruct '{}' for band '{}'",
                                materialName, ref, activeBand);
                    refCache[ref] = ResolvedRef{};
                    return nullptr;
                }

                ResolvedRef entry;
                entry.curveIndex = static_cast<i32>(out.curves.size());
                out.curves.push_back(gpuCurve);

                // The VIS reconstruction, separately from the render band: the
                // colour is only ever compared against a base-colour texel, and
                // a SWIR curve has no colour to compare.
                const SpectralCurve visCurve = basisLoader.ReconstructCurve(spectralData->name, "VIS");
                if (!visCurve.samples.empty()) {
                    entry.colorLinear = ReflectanceToLinearSrgbD65(visCurve);
                    entry.hasVisColor = true;
                }

                const MaterialSpectralData::BandData* bandData = nullptr;
                if (auto bandIt = spectralData->bands.find(activeBand);
                    bandIt != spectralData->bands.end()) {
                    bandData = &bandIt->second;
                }
                QL_LOG_INFO("    Reconstructed '{}': {} samples, λ=[{:.1f}, {:.1f}] nm, "
                            "RMSE={:.4f}, coverage={:.2f} → index {}",
                            spectralData->name, gpuCurve.numSamples, gpuCurve.startWavelength_nm,
                            gpuCurve.GetWavelength(gpuCurve.numSamples - 1),
                            bandData ? bandData->rmse : 0.0f,
                            bandData ? bandData->coverage : 1.0f, entry.curveIndex);

                auto [it, _] = refCache.emplace(ref, entry);
                return &it->second;
            };

            for (const auto& mat : scene.materials) {
                if (!mat.HasQuantiloomRef()) continue;

                // Any quantiloom_* type (usgs, ecostress, rii, ...)
                if (mat.quantiloomMaterialType.find("quantiloom_") != 0) {
                    QL_LOG_WARN("  Unsupported spectral material type: '{}' "
                                "(expected 'quantiloom_*')",
                                mat.quantiloomMaterialType);
                    continue;
                }

                QL_LOG_INFO("  Processing Quantiloom material: '{}' -> type='{}', name='{}'",
                            mat.name, mat.quantiloomMaterialType, mat.quantiloomMaterialRef);

                // Endmember 0 is the material's own reference. If it does not
                // resolve there is no mixture to speak of, and the material
                // falls back to RGB upsampling exactly as it did before.
                const ResolvedRef* first = resolveRef(mat.quantiloomMaterialRef, mat.name);
                if (!first) continue;

                EndmemberSlots slots;
                slots.unmix = mat.spectralUnmixMode;
                slots.weightTexturePath = mat.spectralWeightTexturePath;
                slots.curves[0] = first->curveIndex;
                slots.colorsLinear[0] = first->colorLinear;
                slots.count = 1;
                bool everyEndmemberHasColor = first->hasVisColor;

                for (const auto& extraRef : mat.quantiloomExtraRefs) {
                    if (slots.count >= Material::MAX_ENDMEMBERS) break;
                    const ResolvedRef* extra = resolveRef(extraRef, mat.name);
                    if (!extra) continue;
                    slots.curves[slots.count] = extra->curveIndex;
                    slots.colorsLinear[slots.count] = extra->colorLinear;
                    everyEndmemberHasColor = everyEndmemberHasColor && extra->hasVisColor;
                    ++slots.count;
                }

                // Unmixing compares endmember colours against a texel, so an
                // endmember with no visible-light spectrum has nothing to
                // compare. Better one flat measured curve than a mixture
                // weighted by a colour that was invented.
                if (slots.unmix != Material::SpectralUnmixMode::Off && !everyEndmemberHasColor) {
                    diag.Warn("materials.spectral_material_refs",
                              "  Material '" + mat.name +
                                  "': an endmember has no VIS band, so its colour is unknown "
                                  "and the weights cannot be derived. Falling back to the "
                                  "first curve, flat.");
                    slots.unmix = Material::SpectralUnmixMode::Off;
                }

                // Keyed by the glTF material name, not the spectral reference.
                out.materialNameToCurve[mat.name] = slots.curves[0];
                out.materialNameToEndmembers[mat.name] = slots;

                if (slots.count > 1) {
                    QL_LOG_INFO("    Mixture of {} endmembers", slots.count);
                }
            }
        } else {
            diag.Warn("spectral.basis_file",
                      "  Failed to load SpectralBaker data, using fallback");
        }
    } else {
        QL_LOG_INFO("  No SpectralBaker basis configured");
        QL_LOG_INFO("  NOTE: Add [spectral] basis_file and materials_json for NMF "
                    "spectral data");
    }

    QL_LOG_INFO("  Total spectral curves (CSV + SpectralBaker): {}", out.curves.size());
    report.spectralCurvesLoaded = static_cast<u32>(out.curves.size());

    // ------------------------------------------------------------------
    // [refractive_index]: material name -> RefractiveIndex.INFO YAML
    // ------------------------------------------------------------------
    QL_LOG_INFO("Loading complex refractive index data...");

    if (config.HasSection("refractive_index")) {
        for (const auto& [materialName, yamlPath] : config.GetSection("refractive_index")) {
            QL_LOG_INFO("  Loading n,k data for '{}' from '{}'", materialName, yamlPath);

            auto result =
                SpectralIO::LoadRefractiveIndexYAML(ResolveConfigPath(yamlPath, options.baseDir));
            if (!result) {
                diag.Warn("refractive_index", "    Failed to load: " + result.error());
                continue;
            }

            ComplexRefractiveIndex cri = result.value();
            ComplexRefractiveIndexGPU gpuCRI = ComplexRefractiveIndexGPU::FromCPU(cri);

            const i32 criIndex = static_cast<i32>(out.refractiveIndices.size());
            out.materialNameToRefractiveIndex[materialName] = criIndex;
            out.refractiveIndices.push_back(gpuCRI);

            const auto [lambda_min, lambda_max] = cri.GetWavelengthRange();
            const f32 F0_550 = cri.FresnelR0(550.0f);
            QL_LOG_INFO("    Loaded: {} samples, λ=[{:.1f}, {:.1f}] nm, F0@550nm={:.3f} "
                        "→ CRI index {}",
                        gpuCRI.numSamples, lambda_min, lambda_max, F0_550, criIndex);
        }
    }

    QL_LOG_INFO("  Total complex refractive index entries loaded: {}",
                out.refractiveIndices.size());
    report.refractiveIndicesLoaded = static_cast<u32>(out.refractiveIndices.size());

    if (diag.failed()) {
        return SpectraResult::Err(diag.firstError());
    }
    return SpectraResult(std::move(out));
}

}  // namespace quantiloom::rendercore
