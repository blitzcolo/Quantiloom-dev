/**
 * @file ConfigResolve.cpp
 * @brief The one reading of a scene TOML -- see ConfigResolve.hpp for why
 *
 * @author blitzcolo
 */

#include "renderer/ConfigResolve.hpp"

#include "core/Log.hpp"
#include "io/SpectralBasisLoader.hpp"
#include "io/SpectralIO.hpp"
#include "postprocess/PostprocessConfig.hpp"
#include "scene/Material.hpp"

#include <algorithm>
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

/// Resolve a path the config named against the config's own directory.
///
/// An empty baseDir means the process working directory, which is the CLI's
/// convention -- it is run from the repo root and its configs name assets from
/// there. A GUI opens files from anywhere, so it passes the file's directory
/// and relative paths keep working.
String ResolvePath(const String& path, const String& baseDir) {
    if (path.empty() || baseDir.empty()) return path;
    std::filesystem::path p(path);
    if (p.is_absolute()) return path;
    return (std::filesystem::path(baseDir) / p).lexically_normal().string();
}

}  // namespace

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
        out.environmentMap = ResolvePath(out.environmentMap, options.baseDir);
    }
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

    f32 transmittance = config.Get<f32>("lighting.transmittance", 0.9f);
    transmittance = std::clamp(transmittance, 0.0f, 1.0f);

    f32 atmosphereTemperature_K =
        config.Get<f32>("lighting.atmosphere_temperature_k", 260.0f);
    if (atmosphereTemperature_K < 150.0f || atmosphereTemperature_K > 350.0f) {
        QL_LOG_WARN("lighting.atmosphere_temperature_k={:.1f}K is outside typical "
                    "range [150, 350], check config",
                    atmosphereTemperature_K);
    }

    QL_LOG_INFO("  Sun direction: [{:.2f}, {:.2f}, {:.2f}]", out.lighting.sunDirection.x,
                out.lighting.sunDirection.y, out.lighting.sunDirection.z);
    QL_LOG_INFO("  Sun radiance: [{:.2f}, {:.2f}, {:.2f}]", sunRadiance.r, sunRadiance.g,
                sunRadiance.b);
    QL_LOG_INFO("  Sky radiance: [{:.2f}, {:.2f}, {:.2f}]", skyRadiance.r, skyRadiance.g,
                skyRadiance.b);
    QL_LOG_INFO("  Atmospheric transmittance: {:.3f}", transmittance);
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
    out.lighting.transmittance = transmittance;
    out.lighting.worldUnitsToMeters = out.worldUnitsToMeters;
    out.lighting.atmosphereTemperature_K = atmosphereTemperature_K;
    out.lighting.chromaR_correction = chromaR;
    out.lighting.chromaB_correction = chromaB;
    out.lighting.enableShadowRays = enableShadowRays ? 1u : 0u;

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
    const auto nonZero = [](const glm::vec3& c) {
        return c.r > 0.0f || c.g > 0.0f || c.b > 0.0f;
    };
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
        const auto namedPath = config.Get<String>("lighting.solar_lut");
        QL_LOG_INFO("  Loading solar LUT from: {}", namedPath);

        // The file's layout is the scene's to declare: the sun's spectrum is
        // user-supplied data and the renderer does not know its shape.
        // Defaults are libRadtran uvspec's; ASTM G-173 wants [4, 3] with
        // diffuse_is_global, its column 3 being global rather than sky.
        const auto cols = config.GetArray<i32>("lighting.solar_lut_columns");
        const u32 directCol = cols.size() >= 1 ? static_cast<u32>(cols[0]) : 2u;
        const u32 diffuseCol = cols.size() >= 2 ? static_cast<u32>(cols[1]) : 3u;
        const auto normalise = config.Get<String>("lighting.solar_lut_normalise", "");
        const bool diffuseIsGlobal =
            config.Get<bool>("lighting.solar_lut_diffuse_is_global", false);

        // "equal_energy" is the one illuminant that is not a file: a flat
        // spectrum at unit luminance, CIE illuminant E. It is the neutral
        // reference -- what a material looks like under light that favours no
        // wavelength -- and it is not sRGB white, which is D65.
        auto result =
            namedPath == "equal_energy"
                ? Result<std::pair<SpectralCurve, SpectralCurve>, String>(
                      std::make_pair(MakeEqualEnergyIlluminant(),
                                     MakeEqualEnergyIlluminant()))
                : SpectralIO::LoadLibRadtranSunAndSky(
                      ResolvePath(namedPath, options.baseDir), "nm", directCol,
                      diffuseCol, diffuseIsGlobal);

        if (result.has_value()) {
            auto& [sunCurve, skyCurve] = result.value();

            // Reference illuminants are published as relative spectra -- D65 is
            // normalised to 100 at 560 nm -- so their absolute level is
            // arbitrary. Normalising to unit luminance puts the illuminant at
            // Y = 1, which is what makes D65 come out as sRGB (1, 1, 1)
            // exactly. It is also the honest separation of white balance from
            // exposure.
            if (normalise == "unit_luminance") {
                const auto rgb = SpectralIrradianceToLinearSrgb(sunCurve);
                const f32 Y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
                if (Y > 0.0f) {
                    // Both curves by the sun's luminance, not each by its own:
                    // scaling them separately would discard the ratio between
                    // sun and sky, which is the one thing a measured pair
                    // actually tells you.
                    for (auto& v : sunCurve.samples) v.second /= Y;
                    for (auto& v : skyCurve.samples) v.second /= Y;
                    QL_LOG_INFO("  Illuminant normalised to unit luminance (was Y={:.4g})",
                                Y);
                }
            } else if (!normalise.empty()) {
                diag.Warn("lighting.solar_lut_normalise",
                          "  Unknown solar_lut_normalise '" + normalise +
                              "' (known: unit_luminance)");
            }

            // Does the spectrum actually cover the band being rendered?
            // SpectralCurve::Evaluate clamps to its endpoints rather than
            // returning zero, so a curve that stops short does not fail -- it
            // holds its last value flat across everything above it, and the
            // render looks plausible. CIE D65 stops at 830 nm, which makes it a
            // fine reference illuminant for RGB and a silently wrong one for
            // SWIR upward.
            if (const auto band = GetFusedBandInfo(out.mode);
                band && !sunCurve.samples.empty()) {
                const f32 curveMin = sunCurve.samples.front().first;
                const f32 curveMax = sunCurve.samples.back().first;
                if (curveMin > band->lambdaMinNm || curveMax < band->lambdaMaxNm) {
                    QL_LOG_WARN("  Illuminant spans [{:.0f}, {:.0f}] nm but this mode "
                                "renders [{:.0f}, {:.0f}] nm. Evaluate() clamps, so the "
                                "uncovered part is held flat at the nearest endpoint "
                                "rather than left dark -- the result will look "
                                "reasonable and mean nothing.",
                                curveMin, curveMax, band->lambdaMinNm, band->lambdaMaxNm);
                }
            }

            // One illuminant, every mode. The spectral paths sample these
            // curves per wavelength; RGB and VIS_Fused need a colour, and
            // taking it from the same curve is what stops the two halves of the
            // renderer describing different suns. It replaces whatever
            // sun_radiance said, which was a triple in arbitrary units with no
            // defined relationship to the spectrum beside it.
            out.lighting.sunRadiance_rgb = SpectralIrradianceToLinearSrgb(sunCurve);
            out.lighting.skyRadiance_rgb = SpectralIrradianceToLinearSrgb(skyCurve);
            out.lighting.sunRadiance_spectral =
                (out.lighting.sunRadiance_rgb.r + out.lighting.sunRadiance_rgb.g +
                 out.lighting.sunRadiance_rgb.b) / 3.0f;
            out.lighting.skyRadiance_spectral =
                (out.lighting.skyRadiance_rgb.r + out.lighting.skyRadiance_rgb.g +
                 out.lighting.skyRadiance_rgb.b) / 3.0f;
            QL_LOG_INFO("  Illuminant colour from the spectrum: sun [{:.4g}, {:.4g}, "
                        "{:.4g}], sky [{:.4g}, {:.4g}, {:.4g}]",
                        out.lighting.sunRadiance_rgb.r, out.lighting.sunRadiance_rgb.g,
                        out.lighting.sunRadiance_rgb.b, out.lighting.skyRadiance_rgb.r,
                        out.lighting.skyRadiance_rgb.g, out.lighting.skyRadiance_rgb.b);

            out.solarSunSky = std::make_pair(std::move(sunCurve), std::move(skyCurve));
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
                ? ResolvePath(config.Get<String>("atmosphere.model_pack"), options.baseDir)
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
    }
    QL_LOG_INFO("  NN atmosphere: {}", out.atmosphere.enabled ? "ENABLED" : "DISABLED");

    // ------------------------------------------------------------------
    // [sensor]
    // ------------------------------------------------------------------
    out.sensorEnabled = config.Get<bool>("sensor.enabled", false);
    // Parsed whether or not it is enabled, so a host can turn it on later
    // without rereading the file.
    out.sensor = PostprocessConfig::ParseSensorParams(config);

    if (diag.failed()) {
        return ResolveResult::Err(diag.firstError());
    }
    return ResolveResult(std::move(out));
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
    for (const auto& matTable : config.GetTableArray("materials")) {
        const auto name = matTable.GetString("name", "");
        if (name.empty()) continue;

        auto it = std::find_if(scene.materials.begin(), scene.materials.end(),
                               [&name](const Material& m) { return m.name == name; });
        if (it == scene.materials.end()) {
            diag.Warn("materials",
                      "  [[materials]] names '" + name + "', which the scene has no "
                      "material by");
            continue;
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

        ++out.materialsOverridden;
        QL_LOG_INFO("  Material '{}': IR override e={:.3f} t={:.3f} r={:.3f} T={:.1f} K",
                    name, emissivity, transmittance, std::max(reflectance, 0.0f),
                    it->irTemperature_K);
    }
    report.materialsOverridden = out.materialsOverridden;

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
                SpectralIO::LoadSpectralCurveCSV(ResolvePath(csvPath, options.baseDir));
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
        if (basisLoader.Load(ResolvePath(basisFilePath, options.baseDir),
                             ResolvePath(materialsJsonPath, options.baseDir))) {
            QL_LOG_INFO("  SpectralBaker data loaded: {} materials, {} bands",
                        basisLoader.GetMaterialCount(), basisLoader.GetNumBands());

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

                const MaterialSpectralData* spectralData =
                    basisLoader.FindMaterial(mat.quantiloomMaterialRef);
                if (!spectralData) {
                    spectralData = basisLoader.FindMaterialPartial(mat.quantiloomMaterialRef);
                    if (spectralData) {
                        QL_LOG_INFO("    Matched via partial search: '{}'", spectralData->name);
                    }
                }
                if (!spectralData) {
                    QL_LOG_WARN("    Material '{}' not found in SpectralBaker database",
                                mat.quantiloomMaterialRef);
                    continue;
                }

                SpectralCurveGPU gpuCurve =
                    basisLoader.ReconstructCurveGPU(spectralData->name, activeBand);
                if (gpuCurve.numSamples == 0) {
                    QL_LOG_WARN("    Failed to reconstruct curve for band '{}'", activeBand);
                    continue;
                }

                // Keyed by the glTF material name, not the spectral reference.
                const i32 curveIndex = static_cast<i32>(out.curves.size());
                out.materialNameToCurve[mat.name] = curveIndex;
                out.curves.push_back(gpuCurve);

                const MaterialSpectralData::BandData* bandData = nullptr;
                if (auto bandIt = spectralData->bands.find(activeBand);
                    bandIt != spectralData->bands.end()) {
                    bandData = &bandIt->second;
                }

                QL_LOG_INFO("    Reconstructed: {} samples, λ=[{:.1f}, {:.1f}] nm, "
                            "RMSE={:.4f} → index {}",
                            gpuCurve.numSamples, gpuCurve.startWavelength_nm,
                            gpuCurve.GetWavelength(gpuCurve.numSamples - 1),
                            bandData ? bandData->rmse : 0.0f, curveIndex);
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
                SpectralIO::LoadRefractiveIndexYAML(ResolvePath(yamlPath, options.baseDir));
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
