/**
 * @file ThermalSolver.cpp
 * @brief From a scene and a time of day to a temperature per triangle
 */

#include "thermal/ThermalSolver.hpp"

#include "core/Log.hpp"
#include "thermal/CpuCrankNicolsonStepper.hpp"
#include "thermal/ShortwaveGains.hpp"
#include "thermal/ThermalMesh.hpp"
#include "thermal/ThermalTimeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numbers>
#include <sstream>

namespace quantiloom::thermal {

namespace {

/// The material's own long-wave emissivity, from the IR curve the renderer
/// uses. Taken rather than typed again so the balance and the camera agree
/// about what a surface radiates.
f32 EmissivityOf(const Material& material) {
    if (material.bandAveragedIREmissivity >= 0.0f) {
        return material.bandAveragedIREmissivity;
    }
    if (!material.irEmissivityCurve.empty()) {
        // At 10 um, where a ~300 K surface radiates most -- not front(), which
        // is the curve's shortest wavelength (3 um for a glTF/USD curve) and
        // says little about the long wave the balance exchanges.
        return material.GetIREmissivity(10000.0f);
    }
    return 0.95f - 0.90f * material.metallicFactor;
}

}  // namespace

void DumpThermalElements(const String& path, const Vector<ThermalElement>& elements,
                         const Vector<ThermalMaterial>& materials,
                         const ExchangeGeometry& geometry,
                         const Vector<f32>& temperature_K,
                         const Vector<f32>& sunSensitivity_K,
                         const Vector<f32>& visibility,
                         const Vector<f32>& lagSensitivity_K,
                         const Vector<f32>& lagVisibility,
                         const Vector<f64>& lagTime_h,
                         const Vector<glm::vec3>& lagDirection,
                         const Vector<f32>& parameterSensitivity,
                         const Vector<ThermalParameter>& parameters) {
    std::ofstream out(path);
    if (!out) {
        QL_LOG_WARN("  Thermal: cannot write thermal.dump_elements to '{}'", path);
        return;
    }

    out << std::setprecision(9);

    // The material properties AS THE SOLVE SAW THEM, not as the config wrote
    // them. The two differ routinely and silently: a material bound to a
    // measured spectrum has its long-wave emissivity replaced by the
    // Planck-weighted band average of that spectrum, so a scene whose config
    // says 0.90 can be solved at 0.9329. Anything reproducing an element's
    // trajectory outside the renderer -- which is how a mesh-resolution study
    // gets a reference -- has to start from these numbers, and reading them off
    // the config instead is a 0.4 K error that looks exactly like a result.
    for (usize m = 0; m < materials.size(); ++m) {
        const ThermalMaterial& mat = materials[m];
        out << "# material " << m << ": k=" << mat.conductivity_W_mK
            << " rho=" << mat.density_kg_m3 << " c=" << mat.specificHeat_J_kgK
            << " d=" << mat.thickness_m << " h=" << mat.convection_W_m2K
            << " alpha_sw=" << mat.shortwaveAbsorptivity
            << " eps_lw=" << mat.longwaveEmissivity
            << " wetness=" << mat.wetnessFactor
            << " q_internal=" << mat.internalHeat_W_m2
            << " interior_bc="
            << (mat.interiorBoundary == InteriorBoundary::FixedTemperature ? "fixed"
                : mat.interiorBoundary == InteriorBoundary::AmbientInterior ? "ambient"
                                                                           : "adiabatic")
            << " interior_K=" << mat.interiorTemperature_K
            << " interior_h=" << mat.interiorConvection_W_m2K
            << " solves=" << (mat.ParticipatesInSolve() ? 1 : 0) << '\n';
    }

    // Which hour each tracked sun column belongs to, and where the sun was
    // then. Without these the per-column pairs below are unattributable, and
    // the study that reads them has to guess.
    const usize lagSlots =
        lagTime_h.size() == lagDirection.size() &&
                lagSensitivity_K.size() == lagTime_h.size() * elements.size() &&
                lagVisibility.size() == lagSensitivity_K.size()
            ? lagTime_h.size()
            : 0;
    for (usize s = 0; s < lagSlots; ++s) {
        out << "# lag column " << s << ": time_h=" << lagTime_h[s]
            << " sun=" << lagDirection[s].x << ',' << lagDirection[s].y << ','
            << lagDirection[s].z << '\n';
    }

    // dTdv_K is empty rather than zero when the tangent was not carried:
    // zero is a temperature that does not move, which is a different claim
    // from not having asked.
    out << "element,centroid_x,centroid_y,centroid_z,normal_x,normal_y,normal_z,"
           "area_m2,material_id,solved,T_K,dTdv_K,v_element,sky_fraction";
    for (usize s = 0; s < lagSlots; ++s) {
        out << ",dTdv_lag" << s << "_K,v_lag" << s;
    }
    const usize parameterCount =
        parameterSensitivity.size() == parameters.size() * elements.size()
            ? parameters.size()
            : 0;
    for (usize p = 0; p < parameterCount; ++p) {
        out << ",dTd_" << ThermalParameterName(parameters[p]);
    }
    out << '\n';

    const bool haveTangent = sunSensitivity_K.size() == elements.size();
    for (usize e = 0; e < elements.size(); ++e) {
        const ThermalElement& element = elements[e];
        out << e << ',' << element.centroid.x << ',' << element.centroid.y << ','
            << element.centroid.z << ',' << element.normal.x << ',' << element.normal.y
            << ',' << element.normal.z << ',' << element.area_m2 << ','
            << element.materialId << ','
            << (e < temperature_K.size() && temperature_K[e] > 0.0f
                    ? 1
                    : 0)
            << ',';
        if (e < temperature_K.size()) out << temperature_K[e];
        out << ',';
        if (haveTangent) out << sunSensitivity_K[e];
        out << ',';
        if (e < visibility.size()) out << visibility[e];
        out << ',';
        if (e < geometry.skyFraction.size()) out << geometry.skyFraction[e];
        for (usize s = 0; s < lagSlots; ++s) {
            out << ',' << lagSensitivity_K[s * elements.size() + e] << ','
                << lagVisibility[s * elements.size() + e];
        }
        for (usize p = 0; p < parameterCount; ++p) {
            out << ',' << parameterSensitivity[p * elements.size() + e];
        }
        out << '\n';
    }
    QL_LOG_INFO("  Thermal: wrote {} elements to {}", elements.size(), path);
}

ExchangeGeometry MakeOpenSkyExchange(const usize elementCount) {
    ExchangeGeometry exchange;
    exchange.viewFactors.rowStart.assign(elementCount + 1, 0);
    exchange.skyFraction.assign(elementCount, 1.0f);
    exchange.sunVisibility.assign(elementCount, 1.0f);
    return exchange;
}

Vector<f32> SampleSunVisibilityAt(const SunVisibilityTable& table,
                                  const ExchangeGeometry& exchange, const f64 time_h,
                                  const usize elementCount) {
    // Shadowed when nothing describes this geometry -- see the declaration.
    Vector<f32> visibility(elementCount, 0.0f);

    if (table.SampleCount() > 0 && table.ElementCount() == elementCount) {
        usize a = 0;
        usize b = 0;
        f64 blend = 0.0;
        table.SampleIndices(time_h, a, b, blend);
        const f32 bf = static_cast<f32>(blend);
        const f32* colA = table.Column(a);
        const f32* colB = table.Column(b);
        for (usize e = 0; e < elementCount; ++e) {
            visibility[e] = colA[e] + bf * (colB[e] - colA[e]);
        }
    } else if (exchange.sunVisibility.size() == elementCount) {
        visibility = exchange.sunVisibility;
    }
    return visibility;
}

std::optional<String> ReadForcingFile(const String& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    String contents{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (in.bad()) return std::nullopt;
    return contents;
}

Vector<std::pair<f64, ThermalForcing>> LoadForcingCsv(const String& path,
                                                      const f64 absentRelativeHumidity) {
    if (path.empty()) return {};
    const std::optional<String> contents = ReadForcingFile(path);
    if (!contents) {
        QL_LOG_WARN("  Thermal: cannot read forcing file '{}'; using constant forcing", path);
        return {};
    }
    return ParseForcingCsv(*contents, path, absentRelativeHumidity);
}

Vector<std::pair<f64, ThermalForcing>> ParseForcingCsv(const StringView contents,
                                                       const StringView source,
                                                       const f64 absentRelativeHumidity) {
    Vector<std::pair<f64, ThermalForcing>> series;
    std::istringstream in{String(contents)};

    String line;
    usize lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber;
        // The bytes are read in binary, so a CRLF file keeps its CR here.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream fields(line);

        f64 time_h = 0.0;
        if (!(fields >> time_h)) {
            if (lineNumber == 1) continue;  // header
            QL_LOG_WARN("  Thermal: forcing line {} has no time; skipped", lineNumber);
            continue;
        }

        ThermalForcing forcing;
        forcing.relativeHumidity = absentRelativeHumidity;
        f64 azimuth_deg = 0.0;
        f64 elevation_deg = 90.0;
        fields >> forcing.airTemperature_K >> forcing.sunIrradiance_W_m2 >> azimuth_deg >>
            elevation_deg >> forcing.skyTemperature_K;

        // Diffuse irradiance and humidity are optional trailing columns: a
        // file written before they existed keeps its meaning, with no diffuse
        // light and the run's own humidity. Read through locals so a row that stops
        // early leaves the defaults alone rather than being handed whatever a
        // failed extraction wrote.
        f64 diffuse_W_m2 = 0.0;
        if (fields >> diffuse_W_m2) forcing.diffuseIrradiance_W_m2 = diffuse_W_m2;
        f64 relativeHumidity = 0.0;
        if (fields >> relativeHumidity) forcing.relativeHumidity = relativeHumidity;
        // Ninth column, optional exactly as the seventh and eighth are: the
        // convective coefficient at this instant. A file written before it
        // existed keeps its meaning, and a zero means the same as an absent
        // column -- use the material's own.
        f64 convection_W_m2K = 0.0;
        if (fields >> convection_W_m2K) forcing.convection_W_m2K = convection_W_m2K;
        // Tenth column, optional on the same terms: the wind speed, which the
        // convection law turns into a coefficient. A station file has this and
        // usually does not have the ninth -- an h measured at a surface is
        // rarer than the wind that sets it.
        f64 windSpeed_m_s = 0.0;
        if (fields >> windSpeed_m_s) forcing.windSpeed_m_s = windSpeed_m_s;

        const f64 az = azimuth_deg * std::numbers::pi / 180.0;
        const f64 el = elevation_deg * std::numbers::pi / 180.0;
        forcing.sunDirection = glm::normalize(glm::vec3(
            static_cast<f32>(std::cos(el) * std::sin(az)), static_cast<f32>(std::sin(el)),
            static_cast<f32>(-std::cos(el) * std::cos(az))));

        series.emplace_back(time_h, forcing);
    }

    std::sort(series.begin(), series.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    QL_LOG_INFO("  Thermal: {} forcing samples from {}", series.size(), source);
    return series;
}

ThermalForcing SampleForcing(const Vector<std::pair<f64, ThermalForcing>>& series,
                             const f64 time_h, const ThermalForcing& fallback) {
    if (series.empty()) return fallback;
    if (time_h <= series.front().first) return series.front().second;
    if (time_h >= series.back().first) return series.back().second;

    // The first sample at or after time_h -- what the linear scan this
    // replaced found, in log time: it runs once per step, and a month of
    // minute-resolution station data at a minute a step was 1e9 comparisons.
    // time_h is strictly inside the series here, so i >= 1.
    const auto at = std::lower_bound(
        series.begin(), series.end(), time_h,
        [](const std::pair<f64, ThermalForcing>& sample, const f64 t) { return sample.first < t; });
    const usize i = static_cast<usize>(at - series.begin());
    const f64 span = series[i].first - series[i - 1].first;
    const f64 t = span > 0.0 ? (time_h - series[i - 1].first) / span : 0.0;
    const ThermalForcing& a = series[i - 1].second;
    const ThermalForcing& b = series[i].second;

    ThermalForcing out;
    out.airTemperature_K = a.airTemperature_K + t * (b.airTemperature_K - a.airTemperature_K);
    out.sunIrradiance_W_m2 =
        a.sunIrradiance_W_m2 + t * (b.sunIrradiance_W_m2 - a.sunIrradiance_W_m2);
    out.diffuseIrradiance_W_m2 =
        a.diffuseIrradiance_W_m2 + t * (b.diffuseIrradiance_W_m2 - a.diffuseIrradiance_W_m2);
    out.skyTemperature_K = a.skyTemperature_K + t * (b.skyTemperature_K - a.skyTemperature_K);
    out.relativeHumidity = a.relativeHumidity + t * (b.relativeHumidity - a.relativeHumidity);
    out.convection_W_m2K = a.convection_W_m2K + t * (b.convection_W_m2K - a.convection_W_m2K);
    out.windSpeed_m_s = a.windSpeed_m_s + t * (b.windSpeed_m_s - a.windSpeed_m_s);
    // Normalised only when the blend has a length. Two opposite directions
    // blend to zero half way, and a zero row blends to nothing near it; either
    // way glm::normalize returns NaN, which every element's cos(theta) then
    // carries. The nearer endpoint is the honest stand-in for a direction the
    // series does not determine.
    const glm::vec3 mixed = glm::mix(a.sunDirection, b.sunDirection, static_cast<f32>(t));
    const f32 length = glm::length(mixed);
    if (length > 1e-6f) {
        out.sunDirection = glm::normalize(mixed);
    } else {
        const ThermalForcing& nearer = t < 0.5 ? a : b;
        const ThermalForcing& farther = t < 0.5 ? b : a;
        out.sunDirection = glm::length(nearer.sunDirection) > 0.0f ? nearer.sunDirection
                                                                   : farther.sunDirection;
    }
    return out;
}

Vector<ThermalMaterial> BuildSolvedMaterials(const Scene& scene, const ThermalConfig& config,
                                             u32* namedCount) {
    Vector<ThermalMaterial> materials(scene.materials.size());
    u32 named = 0;
    for (usize m = 0; m < scene.materials.size(); ++m) {
        materials[m].longwaveEmissivity = EmissivityOf(scene.materials[m]);
        const auto it = config.materials.find(scene.materials[m].name);
        if (it != config.materials.end()) {
            // Copy the config's block wholesale, then put the curve's
            // emissivity back: the config never carries a trustworthy one.
            const f32 emissivity = materials[m].longwaveEmissivity;
            materials[m] = it->second;
            materials[m].longwaveEmissivity = emissivity;
            ++named;
        }
    }
    if (namedCount != nullptr) {
        *namedCount = named;
    }
    return materials;
}

void LogThermalSolveSummary(const ThermalResult& result) {
    QL_LOG_INFO("  Thermal: {} elements ({} solved), {} exchange entries, {} steps, "
                "{:.1f}-{:.1f} K (mean {:.1f} K)",
                result.elementCount, result.participatingElements, result.exchangeNonZeros,
                result.stepsTaken, result.minTemperature_K, result.maxTemperature_K,
                result.meanTemperature_K);
}

// ============================================================================
// A solve kept alive
// ============================================================================
// What used to be the whole of RunThermalSolve, cut where the setup ends and
// the question begins. RunThermalSolve is now Build followed by one FieldAt;
// a sequence is Build followed by many.

struct ThermalSolveSession::Impl {
    ThermalConfig config;
    ThermalMesh mesh;
    Vector<ThermalMaterial> materials;
    ThermalGeometrySchedule schedule;
    Vector<std::pair<f64, ThermalForcing>> forcingSeries;
    /// The forcing file's bytes as forcingSeries was parsed from them; empty
    /// when no file is named, nullopt when one is and could not be read.
    std::optional<String> forcingContents;
    ThermalForcing constantForcing;

    CpuCrankNicolsonStepper cpuStepper;
    IThermalStepper* active = nullptr;
    std::unique_ptr<ThermalTimeline> timeline;

    u32 participatingElements = 0;
    u32 exchangeNonZeros = 0;

    explicit Impl(const ConvectionLaw& law) : cpuStepper(law) {}
};

ThermalSolveSession::ThermalSolveSession() = default;
ThermalSolveSession::~ThermalSolveSession() = default;

Result<std::unique_ptr<ThermalSolveSession>, String> ThermalSolveSession::Build(
    const Scene& scene, const ThermalConfig& config, ThermalGeometrySchedule schedule,
    IThermalStepper* stepper) {
    using BuildResult = Result<std::unique_ptr<ThermalSolveSession>, String>;

    auto session = std::unique_ptr<ThermalSolveSession>(new ThermalSolveSession());
    session->m_impl = std::make_unique<Impl>(config.convection);
    Impl& impl = *session->m_impl;
    impl.config = config;

    // Materials first: the shell pairing needs to know which of them are
    // shells and how thick they are, so the mesh cannot be built before them.
    u32 named = 0;
    impl.materials = BuildSolvedMaterials(scene, config, &named);
    if (named == 0) {
        return BuildResult::Err("no material in the scene has thermal properties; "
                                "set thermal_conductivity_w_mk on at least one");
    }

    // The mesh is built at whatever pose the scene is in now, and only its
    // TOPOLOGY is used from here: the instance bases, the shell partners, the
    // shared edges. All three are invariant under the rigid motion an epoch
    // schedule is made of, so this does not have to match any particular epoch.
    // The per-epoch geometry -- centroids and normals -- lives in the schedule.
    impl.mesh = BuildThermalMesh(scene, MeshOptionsFor(impl.materials,
                                                      config.lateralConduction));
    if (impl.mesh.elements.empty()) {
        return BuildResult::Err("the scene has no triangles to solve on");
    }
    const usize n = impl.mesh.elements.size();

    impl.schedule = std::move(schedule);
    if (impl.schedule.Empty()) {
        impl.schedule = ThermalGeometrySchedule::Single(MakeOpenSkyExchange(n), {});
    }

    // Lateral conduction is mesh-derived and invariant under rigid motion, so
    // it is built once and shared by every epoch's exchange.
    CsrMatrix lateral;
    if (config.lateralConduction) {
        lateral = BuildLateralConduction(impl.mesh, impl.materials);
        if (lateral.NonZeros() == 0) {
            QL_LOG_WARN("  Thermal: lateral_conduction is on but no two triangles of any "
                        "object share an edge whose materials both solve; the term does "
                        "nothing");
        }
        if (impl.mesh.nonManifoldEdgeCount > 0) {
            QL_LOG_WARN("  Thermal: {} edges are shared by more than two triangles; each is "
                        "joined to the first two",
                        impl.mesh.nonManifoldEdgeCount);
        }
    }

    for (usize e = 0; e < impl.schedule.Count(); ++e) {
        ThermalGeometryEpoch& epoch = impl.schedule.epochs[e];

        // The single-geometry case says nothing about elements, because the
        // mesh above already is them.
        if (epoch.elements.empty()) epoch.elements = impl.mesh.elements;
        if (epoch.elements.size() != n) {
            return BuildResult::Err("epoch " + std::to_string(e) + " has " +
                                    std::to_string(epoch.elements.size()) +
                                    " elements where the mesh has " + std::to_string(n));
        }

        if (epoch.exchange.skyFraction.size() != n) {
            if (!epoch.exchange.skyFraction.empty()) {
                QL_LOG_WARN("  Thermal: the exchange geometry has {} rows for {} elements; "
                            "falling back to open sky",
                            epoch.exchange.skyFraction.size(), n);
            }
            epoch.exchange = MakeOpenSkyExchange(n);
        }
        epoch.exchange.lateral = lateral;

        // Synthesise a single-column sun table from the exchange when none was
        // given. This preserves the old behaviour: one sun direction for the
        // whole run.
        if (epoch.sunTable.SampleCount() == 0 && !epoch.exchange.sunVisibility.empty()) {
            epoch.sunTable.sampleTime_h = {config.startTime_h};
            epoch.sunTable.visibility = epoch.exchange.sunVisibility;
            epoch.sunTable.sampleDirection = {config.sunDirection};
        }

        // What the short wave does off the other surfaces, per sun column.
        // Depends on geometry and absorptivity and nothing else, so it is baked
        // once per epoch rather than gathered again on every step -- and skipped
        // when the builder that produced this schedule already did it.
        if (epoch.sunTable.diffuseGain.empty()) {
            BakeShortwaveGains(epoch.exchange, epoch.elements, impl.materials,
                               epoch.sunTable);
        }
    }

    impl.exchangeNonZeros =
        static_cast<u32>(impl.schedule.epochs.front().exchange.viewFactors.NonZeros());
    for (const ThermalElement& element : impl.mesh.elements) {
        if (element.area_m2 > 0.0f && element.materialId < impl.materials.size() &&
            impl.materials[element.materialId].ParticipatesInSolve()) {
            ++impl.participatingElements;
        }
    }

    // Read once. The bytes parsed here are the bytes the solve cache key is
    // taken from (ForcingContents), so a file rewritten while this session
    // lives cannot store a field under a key that describes other forcing.
    if (config.forcingFile.empty()) {
        impl.forcingContents = String{};
    } else {
        impl.forcingContents = ReadForcingFile(config.forcingFile);
        if (impl.forcingContents) {
            impl.forcingSeries = ParseForcingCsv(*impl.forcingContents, config.forcingFile,
                                                 config.relativeHumidity);
        } else {
            QL_LOG_WARN("  Thermal: cannot read forcing file '{}'; using constant forcing",
                        config.forcingFile);
        }
    }

    impl.constantForcing = {};
    impl.constantForcing.airTemperature_K = config.airTemperature_K;
    impl.constantForcing.sunIrradiance_W_m2 = config.sunIrradiance_W_m2;
    impl.constantForcing.diffuseIrradiance_W_m2 = config.diffuseIrradiance_W_m2;
    impl.constantForcing.sunDirection = config.sunDirection;
    impl.constantForcing.skyTemperature_K = config.skyTemperature_K;
    impl.constantForcing.relativeHumidity = config.relativeHumidity;

    // The caller's stepper if it brought one, otherwise our own.
    IThermalStepper* chosen = stepper;
    if (chosen != nullptr && chosen->Convection().model != config.convection.model) {
        // A stepper that does not evaluate the law this run asked for would
        // step a different balance and say nothing about it. The fall back is
        // to the one that does, and it changes the stepper's name, so the
        // solve cache cannot serve the two for each other either.
        QL_LOG_INFO("  Thermal stepper: {} does not carry the requested convection law; "
                    "using {}", chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    if (chosen != nullptr && config.lateralConduction && !chosen->CarriesLateralConduction()) {
        QL_LOG_INFO("  Thermal stepper: {} does not carry lateral conduction; using {}",
                    chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    if (chosen != nullptr && impl.mesh.shellPairCount > 0 && !chosen->CarriesShells()) {
        QL_LOG_INFO("  Thermal stepper: {} solves a shell as two independent slabs; "
                    "using {}", chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    if (chosen != nullptr && NeedsInteriorSources(impl.materials) &&
        !chosen->CarriesInteriorSources()) {
        QL_LOG_INFO("  Thermal stepper: {} does not carry internal heat or an ambient "
                    "interior; using {}", chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    // The two tangents are sized into the state by the Desc below, not by the
    // stepper, so one that does not integrate them hands them back at zero --
    // a derivative, not an absence: a shading pass would trace every lagged
    // shadow against no response and a what-if would predict that no slider
    // moves anything. ThermalPreview's ChooseStepper asks the same two
    // questions for the viewport.
    if (chosen != nullptr && config.sunCorrection && config.sunMemoryLags > 0 &&
        !chosen->CarriesLagSensitivity()) {
        QL_LOG_INFO("  Thermal stepper: {} does not carry the sun_memory_lags tangents; "
                    "using {}", chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    if (chosen != nullptr && !config.parameterSensitivities.empty() &&
        !chosen->CarriesParameterSensitivity()) {
        QL_LOG_INFO("  Thermal stepper: {} does not carry the parameter_sensitivities "
                    "tangents; using {}", chosen->Name(), impl.cpuStepper.Name());
        chosen = nullptr;
    }
    impl.cpuStepper.SetShellPartners(impl.mesh.shellPartner);
    impl.active = (chosen != nullptr) ? chosen : &impl.cpuStepper;

    f64 fastestWind_m_s = 0.0;
    for (const auto& [time_h, forcing] : impl.forcingSeries) {
        fastestWind_m_s = std::max(fastestWind_m_s, forcing.windSpeed_m_s);
    }
    const f64 shortest = CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
        impl.mesh.elements, impl.materials, config.airTemperature_K, config.convection,
        fastestWind_m_s);
    if (std::isfinite(shortest) && config.timestep_s > shortest) {
        QL_LOG_WARN("  Thermal: timestep {:.0f} s is longer than the shortest surface time "
                    "constant ({:.0f} s). The step is stable, but that surface's "
                    "transient is under-resolved and may ring for a few steps after a "
                    "sharp change in forcing -- shorten the step if it matters.",
                    config.timestep_s, shortest);
    }

    // The lateral term is explicit rather than half-implicit, so its limit is
    // a stability one rather than an accuracy one: past about twice this the
    // field oscillates between neighbours instead of smoothing.
    if (config.lateralConduction) {
        const f64 lateralTau =
            LateralTimeConstantSeconds(lateral, impl.mesh.elements, impl.materials);
        if (std::isfinite(lateralTau)) {
            QL_LOG_INFO("  Thermal: lateral conduction over {} joins, shortest lateral "
                        "time constant {:.0f} s",
                        lateral.NonZeros() / 2, lateralTau);
            if (config.timestep_s > 2.0 * lateralTau) {
                QL_LOG_WARN("  Thermal: timestep {:.0f} s is past twice the shortest "
                            "lateral time constant ({:.0f} s). The lateral term is "
                            "explicit, so this one oscillates rather than smooths -- "
                            "shorten the step or coarsen the mesh.",
                            config.timestep_s, lateralTau);
            }
        }
    }

    ThermalTimeline::Desc desc;
    desc.startTime_h = config.startTime_h;
    desc.timestep_s = config.timestep_s;
    desc.checkpointStride_h = config.checkpointStride_h;
    desc.nodeCount = config.nodeCount;
    desc.initial = config.initial;
    desc.initialTemperature_K = config.initialTemperature_K;
    desc.carrySunSensitivity = config.sunCorrection;
    desc.sunMemoryLags = config.sunCorrection ? config.sunMemoryLags : 0u;
    desc.parameters = config.parameterSensitivities;

    impl.timeline = std::make_unique<ThermalTimeline>(desc, impl.schedule, impl.materials,
                                                      impl.forcingSeries,
                                                      impl.constantForcing, *impl.active);
    return BuildResult(std::move(session));
}

u32 ThermalSolveSession::EpochCount() const {
    return static_cast<u32>(m_impl->schedule.Count());
}

u32 ThermalSolveSession::EpochAt(const f64 time_h) const {
    return static_cast<u32>(m_impl->schedule.EpochAt(time_h));
}

u32 ThermalSolveSession::ElementCount() const {
    return static_cast<u32>(m_impl->mesh.elements.size());
}

const ThermalGeometrySchedule& ThermalSolveSession::Schedule() const {
    return m_impl->schedule;
}

const std::optional<String>& ThermalSolveSession::ForcingContents() const {
    return m_impl->forcingContents;
}

ThermalResult ThermalSolveSession::FieldAt(const f64 time_h) {
    Impl& impl = *m_impl;
    ThermalResult result;

    // The geometry in force at this hour, which is where the normals come from:
    // whether the sun is behind an element is a question about where that
    // element is NOW, not about where it started.
    const ThermalGeometryEpoch& epoch = impl.schedule.At(time_h);
    const Vector<ThermalElement>& elements = epoch.elements;
    const SunVisibilityTable& effectiveTable = epoch.sunTable;
    const ExchangeGeometry& geometry = epoch.exchange;

    result.elementCount = static_cast<u32>(elements.size());
    result.instanceElementBase = impl.mesh.instanceElementBase;
    result.participatingElements = impl.participatingElements;
    result.exchangeNonZeros = impl.exchangeNonZeros;

    const ThermalState& state = impl.timeline->StateAt(time_h);
    result.stepsTaken = impl.timeline->LastStepCount();

    // ------------------------------------------------------------------
    // What the renderer reads
    // ------------------------------------------------------------------
    result.surfaceTemperature_K.resize(elements.size());
    // What the shading pass needs to undo the per-triangle quantisation of the
    // shadow: the tangent, and the visibility it was taken about.
    const bool haveTangent = state.HasSensitivity();
    result.sunDirection =
        SampleForcing(impl.forcingSeries, time_h, impl.constantForcing).sunDirection;
    if (haveTangent) {
        result.sunSensitivity_K.assign(elements.size(), 0.0f);
        result.sunVisibility =
            SampleSunVisibilityAt(effectiveTable, geometry, time_h, elements.size());
    }

    // The per-column tangents, and the sun position each belongs to. A slot
    // that was never claimed, or a column the table cannot say where the sun
    // was for, is dropped: the shading pass would have nothing to trace toward,
    // and what a dropped slot holds is already inside sunSensitivity_K.
    const bool haveLags = haveTangent && state.HasLagSensitivity() &&
                          effectiveTable.sampleDirection.size() ==
                              effectiveTable.SampleCount();
    if (haveLags) {
        const usize n = elements.size();
        result.lagSlots = state.LagSlots();
        result.lagSensitivity_K.assign(result.lagSlots * n, 0.0f);
        result.lagVisibility.assign(result.lagSlots * n, 0.0f);
        result.lagDirection.assign(result.lagSlots, glm::vec3(0.0f, 1.0f, 0.0f));

        for (u32 s = 0; s < result.lagSlots; ++s) {
            const u32 column = state.lagColumn[s];
            if (column >= effectiveTable.SampleCount()) continue;

            const glm::vec3 direction = effectiveTable.sampleDirection[column];
            result.lagDirection[s] = direction;
            const f32* visibility = effectiveTable.Column(column);

            for (usize e = 0; e < n; ++e) {
                const u32 id = elements[e].materialId;
                const bool solved = elements[e].area_m2 > 0.0f &&
                                    id < impl.materials.size() &&
                                    impl.materials[id].ParticipatesInSolve();
                if (!solved) continue;
                result.lagVisibility[s * n + e] = visibility[e];
                // Same rule as the whole-day tangent, against this column's own
                // sun: a face the sun was behind then would have been dark
                // however finely the shader resolved it.
                if (glm::dot(elements[e].normal, direction) > 0.0f) {
                    result.lagSensitivity_K[s * n + e] =
                        static_cast<f32>(state.SurfaceLagSensitivity(s, e));
                }
            }
        }
    }

    // The material-parameter tangents, if any were asked for. Unlike the sun's,
    // these are not zeroed for a face the sun is behind: a surface out of the
    // sun still responds to its emissivity and its heat capacity.
    if (state.HasParameterSensitivity()) {
        const usize n = elements.size();
        result.parameters = state.parameters;
        result.parameterSensitivity.assign(result.parameters.size() * n, 0.0f);
        for (usize p = 0; p < result.parameters.size(); ++p) {
            for (usize e = 0; e < n; ++e) {
                const u32 id = elements[e].materialId;
                const bool solved = elements[e].area_m2 > 0.0f &&
                                    id < impl.materials.size() &&
                                    impl.materials[id].ParticipatesInSolve();
                if (!solved) continue;
                result.parameterSensitivity[p * n + e] =
                    static_cast<f32>(state.SurfaceParameterSensitivity(p, e));
            }
        }
    }

    f64 sum = 0.0;
    result.minTemperature_K = std::numeric_limits<f64>::max();
    result.maxTemperature_K = std::numeric_limits<f64>::lowest();

    for (usize e = 0; e < elements.size(); ++e) {
        const u32 id = elements[e].materialId;
        const bool solved = elements[e].area_m2 > 0.0f && id < impl.materials.size() &&
                            impl.materials[id].ParticipatesInSolve();
        const f64 T = solved ? state.Surface(e) : 0.0;
        result.surfaceTemperature_K[e] = static_cast<f32>(T);
        if (haveTangent) {
            result.sunSensitivity_K[e] =
                solved ? static_cast<f32>(state.SurfaceSensitivity(e)) : 0.0f;
            // An element the sun is behind right now has zero visibility and
            // would have zero however finely the shader resolved it, so its
            // correction is zero whatever the tangent says. Zeroing it here
            // rather than testing the normal in the shader is not a shortcut:
            // the shader's geometric normal has been flipped to face the
            // viewer, so it cannot tell this case from a hit on the back of a
            // sun-facing triangle -- which has the same temperature as the
            // front and does want the correction.
            if (glm::dot(elements[e].normal, result.sunDirection) <= 0.0f) {
                result.sunSensitivity_K[e] = 0.0f;
            }
        }
        if (solved) {
            sum += T;
            result.minTemperature_K = std::min(result.minTemperature_K, T);
            result.maxTemperature_K = std::max(result.maxTemperature_K, T);
        }
    }
    if (result.participatingElements > 0) {
        result.meanTemperature_K = sum / result.participatingElements;
    } else {
        result.minTemperature_K = 0.0;
        result.maxTemperature_K = 0.0;
    }

    if (!impl.config.dumpElementsFile.empty()) {
        Vector<f64> lagTime_h;
        if (haveLags) {
            lagTime_h.reserve(result.lagSlots);
            for (u32 s = 0; s < result.lagSlots; ++s) {
                const u32 column = state.lagColumn[s];
                lagTime_h.push_back(column < effectiveTable.SampleCount()
                                        ? effectiveTable.sampleTime_h[column]
                                        : 0.0);
            }
        }
        DumpThermalElements(impl.config.dumpElementsFile, elements, impl.materials, geometry,
                            result.surfaceTemperature_K, result.sunSensitivity_K,
                            SampleSunVisibilityAt(effectiveTable, geometry, time_h,
                                                  elements.size()),
                            result.lagSensitivity_K, result.lagVisibility, lagTime_h,
                            result.lagDirection, result.parameterSensitivity,
                            result.parameters);
    }

    return result;
}

ThermalResult RunThermalSolve(const Scene& scene, const ThermalConfig& config,
                              ThermalGeometrySchedule schedule, IThermalStepper* stepper) {
    auto session = ThermalSolveSession::Build(scene, config, std::move(schedule), stepper);
    if (!session) {
        ThermalResult result;
        result.error = session.error();
        return result;
    }

    ThermalResult result = (*session)->FieldAt(config.time_h);
    LogThermalSolveSummary(result);
    return result;
}

ThermalResult RunThermalSolve(const Scene& scene, const ThermalConfig& config,
                              const ExchangeGeometry& exchange,
                              const SunVisibilityTable& sunTable, IThermalStepper* stepper) {
    // The single-geometry spelling. Its epoch brings no elements of its own:
    // the session's mesh is them, and copying it would be a second copy of the
    // largest thing here.
    return RunThermalSolve(scene, config,
                           ThermalGeometrySchedule::Single(exchange, sunTable), stepper);
}

}  // namespace quantiloom::thermal
