// ============================================================================
// Quantiloom - Unit Tests for renderer/ConfigResolve.hpp
// ============================================================================
// Every case here is a rule the CLI and Quantiloom Studio once disagreed on,
// back when each read the TOML for itself. They are pinned rather than merely
// exercised: a change that makes one of them fail is a change that would let
// the same scene render two ways again.
//
// See render_path_divergence.md in the Quantiloom-Qt repo for the audit these
// came from.
// ============================================================================

#include <gtest/gtest.h>

#include "renderer/ConfigResolve.hpp"

#include <filesystem>
#include <fstream>

using namespace quantiloom;
using namespace quantiloom::rendercore;

namespace {

/// A scene carrying the keys the CLI requires, so that a test about one rule is
/// not also a test about missing-key policy.
///
/// Built section by section rather than by concatenating text: TOML will not
/// let a table be opened twice, so a test that wants an extra [lighting] key
/// has to have it emitted inside the one [lighting] the fixture writes.
struct SceneToml {
    std::string rendererKeys;    ///< Extra keys inside [renderer]
    std::string spectralKeys;    ///< Extra keys inside [spectral]
    std::string lightingKeys;    ///< Extra keys inside [lighting]
    std::string sceneKeys;       ///< Extra keys inside [scene]
    std::string atmosphereKeys;  ///< Body of [atmosphere]; the section is omitted when empty
    std::string trailing;        ///< Whole sections appended verbatim
    std::string sunRadiance = "[0.0, 0.0, 0.0]";
    std::string skyRadiance = "[0.0, 0.0, 0.0]";
    bool includeRequired = true;

    [[nodiscard]] std::string Render() const {
        std::string toml;
        if (includeRequired) {
            toml += "[renderer]\nresolution = [64, 64]\n" + rendererKeys +
                    "\n[spectral]\n" + spectralKeys +
                    "\n[scene]\n" + sceneKeys +
                    "\n[camera]\nposition = [0.0, 1.0, 5.0]\n"
                    "look_at = [0.0, 0.0, 0.0]\n"
                    "\n[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                    "sun_radiance = " + sunRadiance + "\n"
                    "sky_radiance = " + skyRadiance + "\n" + lightingKeys +
                    "\n[material]\nalbedo = [0.8, 0.8, 0.8]\n";
        }
        if (!atmosphereKeys.empty()) {
            toml += "\n[atmosphere]\n" + atmosphereKeys;
        }
        toml += "\n" + trailing;
        return toml;
    }
};

class ConfigResolveTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = std::filesystem::temp_directory_path() / "quantiloom_config_resolve";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    Config Parse(const SceneToml& scene) {
        const auto path = testDir / "scene.toml";
        {
            std::ofstream file(path);
            file << scene.Render();
        }
        auto loaded = Config::Load(path.string());
        EXPECT_TRUE(loaded.has_value())
            << "fixture TOML did not parse:\n" << scene.Render();
        return loaded.value();
    }

    /// Resolve under the CLI's strictness.
    Result<ResolvedRenderConfig, String> ResolveStrict(const Config& config) {
        ConfigApplyOptions options;
        options.missingRequired = ConfigApplyOptions::MissingKeyPolicy::Error;
        options.freezeDerivedAtmosGeometry = true;
        return ResolveRenderConfig(config, options, report);
    }

    /// Resolve the way an editor holding a half-finished document does.
    Result<ResolvedRenderConfig, String> ResolveLenient(const Config& config) {
        ConfigApplyOptions options;
        options.missingRequired = ConfigApplyOptions::MissingKeyPolicy::WarnAndDefault;
        return ResolveRenderConfig(config, options, report);
    }

    std::filesystem::path testDir;
    ConfigApplyReport report;
};

}  // namespace

// ============================================================================
// Wavelength: an absent key means the band centre, not 550 nm
// ============================================================================
// Studio defaulted every mode to 550 nm, so a thermal config that did not spell
// the key out -- which is all of them -- rendered at a visible wavelength.

TEST_F(ConfigResolveTest, LwirWavelengthDefaultsToBandCentre) {
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().wavelengthNm, 10000.0f);  // (8000 + 12000) / 2
}

TEST_F(ConfigResolveTest, MwirWavelengthDefaultsToBandCentre) {
    auto config = Parse({.spectralKeys = "mode = \"mwir_fused\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().wavelengthNm, 4000.0f);  // (3000 + 5000) / 2
}

TEST_F(ConfigResolveTest, ExplicitWavelengthWins) {
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\nwavelength_nm = 9200.0\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().wavelengthNm, 9200.0f);
}

TEST_F(ConfigResolveTest, RgbModeKeepsFiveFiftyAndDoesNotTakeABandCentre) {
    auto config = Parse({.spectralKeys = "mode = \"rgb\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().wavelengthNm, 550.0f);
}

// ============================================================================
// Spectral mode aliases
// ============================================================================
// Studio's own parser lower-cased its input, losing every uppercase alias, and
// had no case for "multispectral" -- each of them silently became RGB.

TEST_F(ConfigResolveTest, UppercaseBandAliasesResolve) {
    struct Case { const char* spelling; SpectralMode expected; };
    const Case cases[] = {
        {"VIS", SpectralMode::VIS_Fused},   {"MWIR", SpectralMode::MWIR_Fused},
        {"LWIR", SpectralMode::LWIR_Fused}, {"SWIR", SpectralMode::SWIR_Fused},
        {"NIR", SpectralMode::NIR_Fused},   {"RGB", SpectralMode::RGB},
    };
    for (const auto& c : cases) {
        auto config = Parse({.spectralKeys = std::string("mode = \"") + c.spelling + "\"\n"});
        auto resolved = ResolveStrict(config);
        ASSERT_TRUE(resolved.has_value()) << c.spelling << ": " << resolved.error();
        EXPECT_EQ(resolved.value().mode, c.expected) << "for spelling " << c.spelling;
    }
}

TEST_F(ConfigResolveTest, MultispectralIsAModeNotAFallbackToRgb) {
    auto config = Parse({.spectralKeys = "mode = \"multispectral\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(resolved.value().mode, SpectralMode::Multispectral);
}

TEST_F(ConfigResolveTest, AnUnknownModeIsAnErrorRatherThanRgb) {
    auto config = Parse({.spectralKeys = "mode = \"ultraviolet\"\n"});
    auto resolved = ResolveStrict(config);
    EXPECT_FALSE(resolved.has_value());
}

// ============================================================================
// Defaults that differed
// ============================================================================

TEST_F(ConfigResolveTest, ShadowRaysDefaultOn) {
    auto config = Parse({});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(resolved.value().lighting.enableShadowRays, 1u);
}

TEST_F(ConfigResolveTest, ShadowRaysCanBeTurnedOff) {
    auto config = Parse({.rendererKeys = "enable_shadow_rays = false\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(resolved.value().lighting.enableShadowRays, 0u);
}

TEST_F(ConfigResolveTest, SppDefaultsToOne) {
    auto config = Parse({});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(resolved.value().spp, 1u);
}

TEST_F(ConfigResolveTest, FovDefaultsToSixty) {
    auto config = Parse({});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().camera.GetFovY(), 60.0f);
}

// ============================================================================
// Missing-key policy
// ============================================================================
// The same file, read two ways on purpose: a batch renderer refuses what an
// editor opens and reports on.

TEST_F(ConfigResolveTest, StrictRefusesAMissingResolution) {
    auto config = Parse({.trailing = R"(
[camera]
position = [0.0, 1.0, 5.0]
look_at = [0.0, 0.0, 0.0]
[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance = [1.0, 1.0, 1.0]
sky_radiance = [0.1, 0.1, 0.1]
[material]
albedo = [0.8, 0.8, 0.8]
)", .includeRequired = false});
    auto resolved = ResolveStrict(config);
    EXPECT_FALSE(resolved.has_value());
}

TEST_F(ConfigResolveTest, LenientOpensTheSameFileAndSaysWhatIsMissing) {
    auto config = Parse({.trailing = R"(
[camera]
position = [0.0, 1.0, 5.0]
look_at = [0.0, 0.0, 0.0]
[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance = [1.0, 1.0, 1.0]
sky_radiance = [0.1, 0.1, 0.1]
[material]
albedo = [0.8, 0.8, 0.8]
)", .includeRequired = false});
    auto resolved = ResolveLenient(config);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_TRUE(report.ok()) << "a missing key should warn, not error, when lenient";

    bool mentionsResolution = false;
    for (const auto& m : report.messages) {
        if (m.key == "renderer.resolution" &&
            m.severity == ConfigApplyMessage::Severity::Warning) {
            mentionsResolution = true;
        }
    }
    EXPECT_TRUE(mentionsResolution) << "the report should name the key that was missing";
}

TEST_F(ConfigResolveTest, StrictRefusesASpectralSceneWithNoIlluminantSpectrum) {
    // The shaders have no RGB fallback: this scene would render unlit.
    auto config = Parse({.spectralKeys = "mode = \"vis_fused\"\n",
                         .sunRadiance = "[1.0, 1.0, 1.0]",
                         .skyRadiance = "[0.1, 0.1, 0.1]"});
    auto resolved = ResolveStrict(config);
    EXPECT_FALSE(resolved.has_value());
}

// ============================================================================
// Illuminant normalisation
// ============================================================================
// Read by the CLI and ignored by Studio, which is a factor of Y on every
// rendered value -- ten thousand, for the D65 table shipped with the repo.

TEST_F(ConfigResolveTest, UnitLuminanceNormalisationPutsEqualEnergyAtYEqualsOne) {
    auto config = Parse({.lightingKeys = "solar_lut = \"equal_energy\"\n"
                             "solar_lut_normalise = \"unit_luminance\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    ASSERT_TRUE(resolved.value().solarSunSky.has_value());

    const auto& rgb = resolved.value().lighting.sunRadiance_rgb;
    const float Y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    EXPECT_NEAR(Y, 1.0f, 1e-3f);
}

TEST_F(ConfigResolveTest, AnUnknownNormalisationWarnsRatherThanSilentlyDoingNothing) {
    auto config = Parse({.lightingKeys = "solar_lut = \"equal_energy\"\n"
                             "solar_lut_normalise = \"peak\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    bool warned = false;
    for (const auto& m : report.messages) {
        if (m.key == "lighting.solar_lut_normalise" &&
            m.severity == ConfigApplyMessage::Severity::Warning) {
            warned = true;
        }
    }
    EXPECT_TRUE(warned);
}

TEST_F(ConfigResolveTest, TheIlluminantsColourComesFromItsSpectrum) {
    // Not from lighting.sun_radiance, which is a triple in arbitrary units.
    auto config = Parse({.lightingKeys = "solar_lut = \"equal_energy\"\n",
                         .sunRadiance = "[7.0, 7.0, 7.0]"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_NE(resolved.value().lighting.sunRadiance_rgb.g, 7.0f);
    EXPECT_TRUE(report.solarLutLoaded);
}

// ============================================================================
// Environment map and the double-count it invites
// ============================================================================
// An HDRI sky has a sun painted into it. Adding an analytic sun on top counts
// the same illumination twice, and nothing aligns the two directions -- the
// symptom is two specular highlights on one surface, in different places.

TEST_F(ConfigResolveTest, EnvironmentMapWithAnAnalyticSunWarnsAboutDoubleCounting) {
    auto config = Parse({.rendererKeys = "environment_map = \"sky.exr\"\n",
                         .sunRadiance = "[1.0, 1.0, 1.0]"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    bool warned = false;
    for (const auto& m : report.messages) {
        if (m.key == "renderer.environment_map" &&
            m.severity == ConfigApplyMessage::Severity::Warning) {
            warned = true;
        }
    }
    EXPECT_TRUE(warned);
}

TEST_F(ConfigResolveTest, AnEnvironmentMapAloneDoesNotWarn) {
    // Zero analytic sun and sky: the map is the only light, which is the
    // physically honest way to use one.
    auto config = Parse({.rendererKeys = "environment_map = \"sky.exr\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    for (const auto& m : report.messages) {
        EXPECT_NE(m.key, "renderer.environment_map");
    }
}

TEST_F(ConfigResolveTest, ADisabledMapDoesNotDoubleCountAndDoesNotWarn) {
    auto config = Parse({.rendererKeys = "environment_map = \"sky.exr\"\n"
                                         "environment_map_enabled = false\n",
                         .sunRadiance = "[1.0, 1.0, 1.0]"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    EXPECT_FALSE(resolved.value().environmentMapEnabled);
    EXPECT_EQ(resolved.value().lighting.enableEnvironmentMap, 0u);
    // The path survives being turned off, so a host can switch it back on.
    EXPECT_FALSE(resolved.value().environmentMap.empty());

    for (const auto& m : report.messages) {
        EXPECT_NE(m.key, "renderer.environment_map");
    }
}

TEST_F(ConfigResolveTest, ImageBasedLightingIsOnByDefaultEvenWithNoMapNamed) {
    // With no path the fallback cubemap is what lights the scene, and it always
    // has. Defaulting this off would silently darken every config without a map.
    auto config = Parse({});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_TRUE(resolved.value().environmentMapEnabled);
    EXPECT_EQ(resolved.value().lighting.enableEnvironmentMap, 1u);
}

// ============================================================================
// Atmosphere geometry
// ============================================================================
// A scene that states a geometry means it, whoever renders. One that does not
// gets it frozen for a batch render and left to follow the camera for a
// viewport -- which is the one thing the two hosts legitimately differ on.

TEST_F(ConfigResolveTest, ExplicitSunZenithIsHonouredAndFrozen) {
    auto config = Parse({.atmosphereKeys = R"(model_pack = "/nonexistent/pack"
preset = "clear"
sun_zenith_deg = 33.0
sun_azimuth_deg = 120.0
h1_km = 2.5
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    const auto& atmos = resolved.value().atmosphere;
    EXPECT_DOUBLE_EQ(atmos.sunZenithDeg, 33.0);
    EXPECT_DOUBLE_EQ(atmos.sunAzimuthDeg, 120.0);
    EXPECT_DOUBLE_EQ(atmos.h1Km, 2.5);
    EXPECT_FALSE(atmos.sunFromLighting);
    EXPECT_FALSE(atmos.h1FromCamera);
}

TEST_F(ConfigResolveTest, DerivedGeometryStaysLiveWhenNotFrozen) {
    auto config = Parse({.atmosphereKeys = R"(model_pack = "/nonexistent/pack"
preset = "clear"
)"});
    ConfigApplyOptions options;  // freezeDerivedAtmosGeometry defaults false
    auto resolved = ResolveRenderConfig(config, options, report);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    EXPECT_TRUE(resolved.value().atmosphere.sunFromLighting);
    EXPECT_TRUE(resolved.value().atmosphere.h1FromCamera);
}

TEST_F(ConfigResolveTest, DisabledPresetTurnsTheAtmosphereOff) {
    auto config = Parse({.atmosphereKeys = R"(model_pack = "/nonexistent/pack"
preset = "disabled"
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FALSE(resolved.value().atmosphere.enabled);
}

TEST_F(ConfigResolveTest, LegacyAtmosphericSectionIsIgnoredWithAWarning) {
    // Studio used to map these onto NN presets; the core deprecated them. One
    // policy now, and it says so rather than silently doing nothing.
    auto config = Parse({.trailing = "[atmospheric]\npreset = \"clear_day\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FALSE(resolved.value().atmosphere.enabled);

    bool warned = false;
    for (const auto& m : report.messages) {
        if (m.key == "atmospheric" &&
            m.severity == ConfigApplyMessage::Severity::Warning) {
            warned = true;
        }
    }
    EXPECT_TRUE(warned);
}

TEST_F(ConfigResolveTest, AnEnabledAtmosphereOwnsTheThermalSkyTemperature) {
    // AtmosSkyRadianceIR divides the network's zenith downwelling by B(T_air)
    // to recover an emissivity, so T_air must be the one the bake used.
    auto config = Parse({.lightingKeys = "atmosphere_temperature_k = 260.0\n",
                         .atmosphereKeys = R"(model_pack = "/nonexistent/pack"
preset = "clear"
t_ground_K = 295.0
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_FLOAT_EQ(resolved.value().lighting.atmosphereTemperature_K, 295.0f);
}

// ============================================================================
// Keys a windowed context cannot honour are echoed, not swallowed
// ============================================================================

TEST_F(ConfigResolveTest, ResolutionAndOutputAreReportedForAHostThatCannotUseThem) {
    auto config = Parse({.rendererKeys = "output = \"beauty.exr\"\n"});
    auto resolved = ResolveLenient(config);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(report.configWidth, 64u);
    EXPECT_EQ(report.configHeight, 64u);
    EXPECT_EQ(report.outputPath, "beauty.exr");
}

// ============================================================================
// Scene-dependent: the material half
// ============================================================================

namespace {

Scene MakeSceneWithMaterials(std::initializer_list<const char*> names) {
    Scene scene;
    for (const char* name : names) {
        Material m = Material::CreateLambertian(glm::vec3(0.5f), name);
        scene.materials.push_back(m);
    }
    return scene;
}

/// A scene whose nodes are named, for the [[nodes]] cases. Transforms start at
/// identity so an override is visible.
Scene MakeSceneWithNodes(const std::vector<const char*>& names) {
    Scene scene;
    for (const char* name : names) {
        SceneNode node;
        node.name = name;
        node.transform = glm::mat4(1.0f);
        scene.nodes.push_back(node);
    }
    return scene;
}

}  // namespace

TEST_F(ConfigResolveTest, DefaultTemperatureBackfillsMaterialsInThermalModes) {
    // glTF and USD materials carry no temperature, which silences Planck
    // emission entirely. Studio never did this backfill at all.
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n",
                         .sceneKeys = "default_temperature_k = 310.0\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Wall", "Floor"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(spectra.value().temperatureBackfilled, 2u);
    EXPECT_FLOAT_EQ(scene.materials[0].irTemperature_K, 310.0f);
}

TEST_F(ConfigResolveTest, NoTemperatureBackfillInRgbMode) {
    auto config = Parse({.spectralKeys = "mode = \"rgb\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Wall"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();
    EXPECT_EQ(spectra.value().temperatureBackfilled, 0u);
}

TEST_F(ConfigResolveTest, MaterialsTableAppliesPbrOverrides) {
    // The half of a material Studio can edit but could not write down: an agent
    // or a person changing a colour in the GUI had it disappear on save.
    auto config = Parse({.trailing = R"([[materials]]
name = "Panel"
base_color = [0.9, 0.1, 0.1]
metallic = 0.75
roughness = 0.2
emissive = [1.0, 0.5, 0.0]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel", "Other"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_FLOAT_EQ(scene.materials[0].baseColorFactor.r, 0.9f);
    EXPECT_FLOAT_EQ(scene.materials[0].baseColorFactor.b, 0.1f);
    EXPECT_FLOAT_EQ(scene.materials[0].metallicFactor, 0.75f);
    EXPECT_FLOAT_EQ(scene.materials[0].roughnessFactor, 0.2f);
    EXPECT_FLOAT_EQ(scene.materials[0].emissiveFactor.g, 0.5f);
}

// An entry that sets one thing must not reset the rest: a [[materials]] block
// carrying only a temperature used to be the whole material's new definition.
TEST_F(ConfigResolveTest, MaterialsTableLeavesUnmentionedPbrKeysAlone) {
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n",
                         .trailing = R"([[materials]]
name = "Panel"
ir_temperature_k = 350.0
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel"});
    scene.materials[0].metallicFactor = 0.6f;
    scene.materials[0].roughnessFactor = 0.3f;
    const glm::vec4 loadedColour = scene.materials[0].baseColorFactor;

    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_FLOAT_EQ(scene.materials[0].metallicFactor, 0.6f);
    EXPECT_FLOAT_EQ(scene.materials[0].roughnessFactor, 0.3f);
    EXPECT_FLOAT_EQ(scene.materials[0].baseColorFactor.r, loadedColour.r);
    EXPECT_FLOAT_EQ(scene.materials[0].irTemperature_K, 350.0f);
}

// ============================================================================
// Transmission, dispersion, participating media and the spectral database ref
// ============================================================================
// The fields always reached the GPU; only glTF's KHR extensions could set them.
// A config could not describe a piece of glass, and the prism and water scenes
// carried their intent in comments the reader never saw.

TEST_F(ConfigResolveTest, MaterialsTableAppliesTransmissionAndVolumeKeys) {
    auto config = Parse({.trailing = R"([[materials]]
name = "Glass"
ior = 1.52
transmission = 0.95
dispersion = 0.018
attenuation_color = [0.8, 0.9, 1.0]
attenuation_distance = 0.35
thickness = 0.02
volume_density = 0.4
scattering_coeff = 1.5
absorption_coeff = 0.25
phase_g = 0.6
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Glass"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    const Material& glass = scene.materials[0];
    EXPECT_FLOAT_EQ(glass.ior, 1.52f);
    EXPECT_FLOAT_EQ(glass.transmission, 0.95f);
    EXPECT_FLOAT_EQ(glass.dispersion, 0.018f);
    EXPECT_FLOAT_EQ(glass.attenuationColor.b, 1.0f);
    EXPECT_FLOAT_EQ(glass.attenuationDistance, 0.35f);
    EXPECT_FLOAT_EQ(glass.thicknessFactor, 0.02f);
    EXPECT_FLOAT_EQ(glass.volumeDensity, 0.4f);
    EXPECT_FLOAT_EQ(glass.scatteringCoeff, 1.5f);
    EXPECT_FLOAT_EQ(glass.absorptionCoeff, 0.25f);
    EXPECT_FLOAT_EQ(glass.phaseG, 0.6f);
}

TEST_F(ConfigResolveTest, TransmissionKeysLeaveTheLoadedValueWhenAbsent) {
    // Same absent-key rule the PBR half follows: an entry that sets a colour
    // must not silently make the glass opaque.
    auto config = Parse({.trailing = R"([[materials]]
name = "Glass"
base_color = [0.5, 0.5, 0.5]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Glass"});
    scene.materials[0].transmission = 0.9f;
    scene.materials[0].ior = 1.7f;

    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_FLOAT_EQ(scene.materials[0].transmission, 0.9f);
    EXPECT_FLOAT_EQ(scene.materials[0].ior, 1.7f);
}

TEST_F(ConfigResolveTest, MaterialsTableCarriesTheSpectralDatabaseReference) {
    // What a measured-material assignment made in Studio has to survive a save
    // as. The NMF reconstruction runs later in the same resolve and reads these
    // two fields, so setting them here is the whole of the plumbing.
    auto config = Parse({.trailing = R"([[materials]]
name = "Roof"
spectral_material_type = "quantiloom_usgs"
spectral_material_ref = "Aluminum brushed 293K"
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Roof"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(scene.materials[0].quantiloomMaterialType, "quantiloom_usgs");
    EXPECT_EQ(scene.materials[0].quantiloomMaterialRef, "Aluminum brushed 293K");
    EXPECT_TRUE(scene.materials[0].HasQuantiloomRef());
}

// ============================================================================
// [[nodes]] transform overrides
// ============================================================================

TEST_F(ConfigResolveTest, NodesTableAppliesTrsToTheNamedNode) {
    auto config = Parse({.trailing = R"([[nodes]]
name = "Hull"
translation = [1.0, 2.0, 3.0]
rotation_euler_degrees = [0.0, 90.0, 0.0]
scale = [2.0, 2.0, 2.0]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithNodes({"Hull", "Mast"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(spectra.value().nodesTransformed, 1u);
    const glm::mat4& m = scene.nodes[0].transform;
    // Translation lands in the fourth column.
    EXPECT_FLOAT_EQ(m[3][0], 1.0f);
    EXPECT_FLOAT_EQ(m[3][1], 2.0f);
    EXPECT_FLOAT_EQ(m[3][2], 3.0f);
    // 90 degrees about Y takes the x axis to -z, scaled by 2.
    EXPECT_NEAR(m[0][0], 0.0f, 1e-5f);
    EXPECT_NEAR(m[0][2], -2.0f, 1e-5f);
    // The node not named keeps identity.
    EXPECT_FLOAT_EQ(scene.nodes[1].transform[3][0], 0.0f);
}

// Degrees, not radians: a file full of 0.6108 is a file nobody edits twice, and
// writing degrees into something expecting radians has shipped as a bug here.
TEST_F(ConfigResolveTest, NodesTableRotationIsInDegrees) {
    auto config = Parse({.trailing = R"([[nodes]]
name = "Hull"
rotation_euler_degrees = [0.0, 180.0, 0.0]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithNodes({"Hull"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_NEAR(scene.nodes[0].transform[0][0], -1.0f, 1e-5f);
    EXPECT_NEAR(scene.nodes[0].transform[2][2], -1.0f, 1e-5f);
}

TEST_F(ConfigResolveTest, NodesTableAcceptsAMatrixDirectly) {
    auto config = Parse({.trailing = R"([[nodes]]
name = "Hull"
matrix = [1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          4.0, 5.0, 6.0, 1.0]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithNodes({"Hull"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_FLOAT_EQ(scene.nodes[0].transform[3][0], 4.0f);
    EXPECT_FLOAT_EQ(scene.nodes[0].transform[3][1], 5.0f);
    EXPECT_FLOAT_EQ(scene.nodes[0].transform[3][2], 6.0f);
}

TEST_F(ConfigResolveTest, NodesTableWarnsAboutANameTheSceneDoesNotHave) {
    auto config = Parse({.trailing = R"([[nodes]]
name = "NotInTheScene"
translation = [1.0, 0.0, 0.0]
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithNodes({"Hull"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(spectra.value().nodesTransformed, 0u);
    EXPECT_FALSE(report.messages.empty());
}

TEST_F(ConfigResolveTest, MaterialsTableAppliesIrOverridesToTheNamedMaterial) {
    // This reading of [[materials]] began as Studio-only, so a config the GUI
    // honoured rendered without it from the CLI.
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n",
                         .trailing = R"([[materials]]
name = "Panel"
ir_emissivity = 0.9
ir_transmittance = 0.0
ir_temperature_k = 350.0
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel", "Other"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(spectra.value().materialsOverridden, 1u);
    ASSERT_EQ(scene.materials[0].irEmissivityCurve.size(), 2u);
    EXPECT_FLOAT_EQ(scene.materials[0].irEmissivityCurve[0].second, 0.9f);
    EXPECT_FLOAT_EQ(scene.materials[0].irTemperature_K, 350.0f);

    // Reflectance from energy conservation: 1 - 0.9 - 0.0
    ASSERT_EQ(scene.materials[0].irReflectanceCurve.size(), 2u);
    EXPECT_NEAR(scene.materials[0].irReflectanceCurve[0].second, 0.1f, 1e-6f);

    // The override names one material and leaves the rest alone.
    EXPECT_TRUE(scene.materials[1].irEmissivityCurve.empty());
}

TEST_F(ConfigResolveTest, MaterialsTableCarriesATemperatureTexture) {
    // The field reached the GPU long before a config could name the map; only
    // glTF extras and USD attributes could, so a TOML-only scene was stuck
    // with one temperature per material.
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n",
                         .trailing = R"([[materials]]
name = "Panel"
temperature_texture = "maps/panel_temp.png"
temperature_scale = 60.0
temperature_offset = 270.0
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel", "Other"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    // The path is stored raw: it is resolved against the config directory at
    // mount time, not here.
    EXPECT_EQ(scene.materials[0].temperatureTexturePath, "maps/panel_temp.png");
    EXPECT_FLOAT_EQ(scene.materials[0].temperatureScale, 60.0f);
    EXPECT_FLOAT_EQ(scene.materials[0].temperatureOffset, 270.0f);

    // Untouched materials keep the defaults, which are the full 500 K range.
    EXPECT_TRUE(scene.materials[1].temperatureTexturePath.empty());
    EXPECT_FLOAT_EQ(scene.materials[1].temperatureScale, 500.0f);
    EXPECT_FLOAT_EQ(scene.materials[1].temperatureOffset, 200.0f);
}

TEST_F(ConfigResolveTest, TemperatureScaleAndOffsetStandOnTheirOwn) {
    // Retuning the kelvin mapping of a map a scene file already provides is a
    // legitimate override, so neither key requires temperature_texture.
    auto config = Parse({.spectralKeys = "mode = \"lwir_fused\"\n",
                         .trailing = R"([[materials]]
name = "Panel"
temperature_offset = 250.0
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel"});
    scene.materials[0].temperatureTextureIndex = 3;  // as a loader would leave it
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_FLOAT_EQ(scene.materials[0].temperatureOffset, 250.0f);
    EXPECT_FLOAT_EQ(scene.materials[0].temperatureScale, 500.0f);  // absent key, left alone
    EXPECT_EQ(scene.materials[0].temperatureTextureIndex, 3);      // and the map survives
}

TEST_F(ConfigResolveTest, MaterialsTableNamingAnAbsentMaterialWarns) {
    auto config = Parse({.trailing = R"([[materials]]
name = "NotInThisScene"
ir_emissivity = 0.5
)"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel"});
    ConfigApplyOptions options;
    ConfigApplyReport materialReport;
    auto spectra =
        ResolveMaterialSpectra(config, scene, resolved.value(), options, materialReport);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    EXPECT_EQ(spectra.value().materialsOverridden, 0u);
    bool warned = false;
    for (const auto& m : materialReport.messages) {
        if (m.severity == ConfigApplyMessage::Severity::Warning) warned = true;
    }
    EXPECT_TRUE(warned);
}

TEST_F(ConfigResolveTest, SpectralCurvesAreKeyedByMaterialName) {
    const auto csv = testDir / "reflectance.csv";
    {
        std::ofstream file(csv);
        file << "400,0.10\n500,0.20\n600,0.30\n700,0.40\n";
    }

    auto config = Parse({.trailing = "[spectral_curves]\n\"Panel\" = \"" +
                                    std::filesystem::path(csv).generic_string() +
                                    "\"\n"});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    Scene scene = MakeSceneWithMaterials({"Panel"});
    ConfigApplyOptions options;
    auto spectra = ResolveMaterialSpectra(config, scene, resolved.value(), options, report);
    ASSERT_TRUE(spectra.has_value()) << spectra.error();

    ASSERT_EQ(spectra.value().curves.size(), 1u);
    auto it = spectra.value().materialNameToCurve.find("Panel");
    ASSERT_NE(it, spectra.value().materialNameToCurve.end());
    EXPECT_EQ(it->second, 0);
}

// ============================================================================
// The illuminant resolves the same way for a file and for a host
// ============================================================================
// ResolveSolarLut is what [lighting] solar_lut* means. Both the config path and
// ExternalRenderContext::SetSolarSpectralLUTFromSpec go through it, so a host
// that offers a choice of illuminant cannot make one mean something else.

TEST_F(ConfigResolveTest, EqualEnergyIlluminantNeedsNoFile) {
    SolarLutRequest request;
    request.pathOrEqualEnergy = "equal_energy";

    auto result = ResolveSolarLut(request, "", SpectralMode::RGB);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_FALSE(result.value().sun.samples.empty());
    EXPECT_FALSE(result.value().sky.samples.empty());

    // Illuminant E is flat in *energy* and normalised to unit luminance, which
    // is not the same as being sRGB white: sRGB's white point is D65, and E is
    // slightly warm against it. So Y is exactly 1 and the channels are merely
    // close -- asserting (1, 1, 1) would be asserting that E is D65.
    const glm::vec3 rgb = result.value().sunRadianceRgb;
    const float Y = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    EXPECT_NEAR(Y, 1.0f, 1e-3f);
    // (1.205, 0.948, 0.909) as measured. Note that the doc comment on
    // MakeEqualEnergyIlluminant claims linear sRGB "exactly (1, 1, 1)", which
    // holds only if sRGB's white point were E; it is D65, so E reads warm.
    // Pinned loosely -- the point is that E is near-neutral and on the warm
    // side, not the third decimal of a chromatic adaptation.
    EXPECT_GT(rgb.r, rgb.b) << "E is warm against D65, not cool";
    EXPECT_LT(rgb.r / rgb.b, 1.5f) << "...but still recognisably neutral";
}

TEST_F(ConfigResolveTest, UnitLuminanceNormalisationKeepsTheSunSkyRatio) {
    // Scaling the two curves separately would put each at Y = 1 and discard
    // the ratio between them, which is the one thing a measured pair says.
    SolarLutRequest plain;
    plain.pathOrEqualEnergy = "equal_energy";
    auto unnormalised = ResolveSolarLut(plain, "", SpectralMode::RGB);
    ASSERT_TRUE(unnormalised.has_value());

    SolarLutRequest normalised = plain;
    normalised.normaliseUnitLuminance = true;
    auto scaled = ResolveSolarLut(normalised, "", SpectralMode::RGB);
    ASSERT_TRUE(scaled.has_value());

    const auto ratioOf = [](const ResolvedSolarLut& lut) {
        return lut.skyRadianceRgb.g / lut.sunRadianceRgb.g;
    };
    EXPECT_NEAR(ratioOf(unnormalised.value()), ratioOf(scaled.value()), 1e-4f);

    // ...and the sun itself lands at unit luminance.
    const glm::vec3 sun = scaled.value().sunRadianceRgb;
    const float Y = 0.2126f * sun.r + 0.7152f * sun.g + 0.0722f * sun.b;
    EXPECT_NEAR(Y, 1.0f, 1e-3f);
}

TEST_F(ConfigResolveTest, AnIlluminantTooNarrowForTheBandSaysSo) {
    // Evaluate() clamps rather than returning zero, so a spectrum that stops
    // short renders plausibly and means nothing. CIE D65 stopping at 830 nm is
    // the real case: a fine reference for RGB, silently wrong for SWIR upward.
    const auto spectrum = testDir / "narrow.csv";
    {
        std::ofstream file(spectrum);
        file << "# wavelength direct diffuse\n";
        for (int nm = 400; nm <= 830; nm += 10) {
            file << nm << " 1.0 0.1\n";
        }
    }

    SolarLutRequest request;
    request.pathOrEqualEnergy = spectrum.string();

    auto visible = ResolveSolarLut(request, "", SpectralMode::RGB);
    ASSERT_TRUE(visible.has_value()) << visible.error();
    EXPECT_TRUE(visible.value().warnings.empty());

    auto thermal = ResolveSolarLut(request, "", SpectralMode::LWIR_Fused);
    ASSERT_TRUE(thermal.has_value()) << thermal.error();
    EXPECT_FALSE(thermal.value().warnings.empty())
        << "a spectrum that does not reach the band must not pass silently";
}

TEST_F(ConfigResolveTest, AnUnnamedIlluminantIsAnError) {
    SolarLutRequest request;
    EXPECT_FALSE(ResolveSolarLut(request, "", SpectralMode::RGB).has_value());
    request.pathOrEqualEnergy = "no/such/spectrum.csv";
    EXPECT_FALSE(ResolveSolarLut(request, "", SpectralMode::RGB).has_value());
}

// ============================================================================
// camera.projection
// ============================================================================
// Orthographic rays share a direction and start spread across the film plane,
// which is what makes a front or top view measurable. Absent means
// perspective, so every scene written before the key existed still means what
// it meant.

TEST_F(ConfigResolveTest, CameraIsPerspectiveUnlessTheConfigSaysOtherwise) {
    auto config = Parse({});
    auto resolved = ResolveStrict(config);
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(resolved.value().camera.GetProjection(), Camera::Projection::Perspective);
}

TEST_F(ConfigResolveTest, OrthographicProjectionIsRead) {
    auto config = Parse({.trailing = R"([camera.extra]
)"});
    // The fixture writes [camera] itself, so the key goes in through the
    // camera block it already emits.
    const auto path = testDir / "ortho.toml";
    {
        std::ofstream file(path);
        file << "[renderer]\nresolution = [64, 64]\n"
                "[spectral]\n"
                "[scene]\n"
                "[camera]\nposition = [0.0, 0.0, 5.0]\nlook_at = [0.0, 0.0, 0.0]\n"
                "fov_y = 60.0\nprojection = \"orthographic\"\northo_height = 4.0\n"
                "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                "sun_radiance = [0.0, 0.0, 0.0]\nsky_radiance = [0.0, 0.0, 0.0]\n"
                "[material]\nalbedo = [0.8, 0.8, 0.8]\n";
    }
    auto loaded = Config::Load(path.string());
    ASSERT_TRUE(loaded.has_value());
    auto resolved = ResolveStrict(loaded.value());
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    EXPECT_EQ(resolved.value().camera.GetProjection(), Camera::Projection::Orthographic);
    EXPECT_FLOAT_EQ(resolved.value().camera.GetOrthoHeight(), 4.0f);

    // And it reaches the GPU struct raygen actually reads.
    const CameraData data = resolved.value().camera.GetCameraData();
    EXPECT_EQ(data.projection, 1u);
    EXPECT_FLOAT_EQ(data.orthoHeight, 4.0f);
}

TEST_F(ConfigResolveTest, OrthoHeightDefaultsToWhatThePerspectiveCameraFramed) {
    // Switching projection should not also change how much of the scene is in
    // shot; without a stated height the framing follows from the distance and
    // the field of view.
    const auto path = testDir / "ortho_default.toml";
    {
        std::ofstream file(path);
        file << "[renderer]\nresolution = [64, 64]\n"
                "[spectral]\n"
                "[scene]\n"
                "[camera]\nposition = [0.0, 0.0, 10.0]\nlook_at = [0.0, 0.0, 0.0]\n"
                "fov_y = 60.0\nprojection = \"orthographic\"\n"
                "[lighting]\nsun_direction = [0.0, 1.0, 0.0]\n"
                "sun_radiance = [0.0, 0.0, 0.0]\nsky_radiance = [0.0, 0.0, 0.0]\n"
                "[material]\nalbedo = [0.8, 0.8, 0.8]\n";
    }
    auto loaded = Config::Load(path.string());
    ASSERT_TRUE(loaded.has_value());
    auto resolved = ResolveStrict(loaded.value());
    ASSERT_TRUE(resolved.has_value()) << resolved.error();

    // 2 * 10 * tan(30 deg) = 11.547
    EXPECT_NEAR(resolved.value().camera.GetOrthoHeight(), 11.547f, 0.01f);
}
