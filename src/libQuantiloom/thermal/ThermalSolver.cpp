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
        return material.irEmissivityCurve.front().second;
    }
    return 0.95f - 0.90f * material.metallicFactor;
}

}  // namespace

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
    Vector<f32> visibility(elementCount, 1.0f);

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

Vector<std::pair<f64, ThermalForcing>> LoadForcingCsv(const String& path) {
    Vector<std::pair<f64, ThermalForcing>> series;
    if (path.empty()) return series;

    std::ifstream in(path);
    if (!in) {
        QL_LOG_WARN("  Thermal: cannot read forcing file '{}'; using constant forcing", path);
        return series;
    }

    String line;
    usize lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber;
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
        f64 azimuth_deg = 0.0;
        f64 elevation_deg = 90.0;
        fields >> forcing.airTemperature_K >> forcing.sunIrradiance_W_m2 >> azimuth_deg >>
            elevation_deg >> forcing.skyTemperature_K;

        // Diffuse irradiance and humidity are optional trailing columns: a
        // file written before they existed keeps its meaning, with no diffuse
        // light and average humidity. Read through locals so a row that stops
        // early leaves the defaults alone rather than being handed whatever a
        // failed extraction wrote.
        f64 diffuse_W_m2 = 0.0;
        if (fields >> diffuse_W_m2) forcing.diffuseIrradiance_W_m2 = diffuse_W_m2;
        f64 relativeHumidity = 0.0;
        if (fields >> relativeHumidity) forcing.relativeHumidity = relativeHumidity;

        const f64 az = azimuth_deg * std::numbers::pi / 180.0;
        const f64 el = elevation_deg * std::numbers::pi / 180.0;
        forcing.sunDirection = glm::normalize(glm::vec3(
            static_cast<f32>(std::cos(el) * std::sin(az)), static_cast<f32>(std::sin(el)),
            static_cast<f32>(-std::cos(el) * std::cos(az))));

        series.emplace_back(time_h, forcing);
    }

    std::sort(series.begin(), series.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    QL_LOG_INFO("  Thermal: {} forcing samples from {}", series.size(), path);
    return series;
}

ThermalForcing SampleForcing(const Vector<std::pair<f64, ThermalForcing>>& series,
                             const f64 time_h, const ThermalForcing& fallback) {
    if (series.empty()) return fallback;
    if (time_h <= series.front().first) return series.front().second;
    if (time_h >= series.back().first) return series.back().second;

    for (usize i = 1; i < series.size(); ++i) {
        if (time_h <= series[i].first) {
            const f64 span = series[i].first - series[i - 1].first;
            const f64 t = span > 0.0 ? (time_h - series[i - 1].first) / span : 0.0;
            const ThermalForcing& a = series[i - 1].second;
            const ThermalForcing& b = series[i].second;

            ThermalForcing out;
            out.airTemperature_K = a.airTemperature_K + t * (b.airTemperature_K - a.airTemperature_K);
            out.sunIrradiance_W_m2 =
                a.sunIrradiance_W_m2 + t * (b.sunIrradiance_W_m2 - a.sunIrradiance_W_m2);
            out.diffuseIrradiance_W_m2 =
                a.diffuseIrradiance_W_m2 +
                t * (b.diffuseIrradiance_W_m2 - a.diffuseIrradiance_W_m2);
            out.skyTemperature_K =
                a.skyTemperature_K + t * (b.skyTemperature_K - a.skyTemperature_K);
            out.relativeHumidity =
                a.relativeHumidity + t * (b.relativeHumidity - a.relativeHumidity);
            out.sunDirection = glm::normalize(
                glm::mix(a.sunDirection, b.sunDirection, static_cast<f32>(t)));
            return out;
        }
    }
    return series.back().second;
}

ThermalResult RunThermalSolve(const Scene& scene, const ThermalConfig& config,
                              const ExchangeGeometry& exchange,
                              const SunVisibilityTable& sunTable) {
    ThermalResult result;

    ThermalMesh mesh = BuildThermalMesh(scene);
    result.elementCount = static_cast<u32>(mesh.elements.size());
    result.instanceElementBase = std::move(mesh.instanceElementBase);
    if (mesh.elements.empty()) {
        result.error = "the scene has no triangles to solve on";
        return result;
    }

    Vector<ThermalMaterial> materials(scene.materials.size());
    u32 named = 0;
    for (usize m = 0; m < scene.materials.size(); ++m) {
        materials[m].longwaveEmissivity = EmissivityOf(scene.materials[m]);
        const auto it = config.materials.find(scene.materials[m].name);
        if (it != config.materials.end()) {
            const f32 emissivity = materials[m].longwaveEmissivity;
            materials[m] = it->second;
            materials[m].longwaveEmissivity = emissivity;
            ++named;
        }
    }
    if (named == 0) {
        result.error = "no material in the scene has thermal properties; "
                       "set thermal_conductivity_w_mk on at least one";
        return result;
    }

    for (const ThermalElement& element : mesh.elements) {
        if (element.area_m2 > 0.0f && element.materialId < materials.size() &&
            materials[element.materialId].ParticipatesInSolve()) {
            ++result.participatingElements;
        }
    }
    result.exchangeNonZeros = static_cast<u32>(exchange.viewFactors.NonZeros());

    const ExchangeGeometry openSky = MakeOpenSkyExchange(mesh.elements.size());
    const ExchangeGeometry& geometry =
        exchange.skyFraction.size() == mesh.elements.size() ? exchange : openSky;
    if (&geometry == &openSky && !exchange.skyFraction.empty()) {
        QL_LOG_WARN("  Thermal: the exchange geometry has {} rows for {} elements; "
                    "falling back to open sky",
                    exchange.skyFraction.size(), mesh.elements.size());
    }

    // Synthesise a single-column sun table from the exchange when no table
    // is provided. This preserves the old behaviour: one sun direction for
    // the entire run.
    SunVisibilityTable effectiveTable = sunTable;
    if (effectiveTable.SampleCount() == 0 && !geometry.sunVisibility.empty()) {
        effectiveTable.sampleTime_h = {config.startTime_h};
        effectiveTable.visibility = geometry.sunVisibility;
        effectiveTable.sampleDirection = {config.sunDirection};
    }

    // What the short wave does off the other surfaces, per sun column. Depends
    // on geometry and absorptivity and nothing else, so it is baked once here
    // rather than gathered again on every step.
    BakeShortwaveGains(geometry, mesh.elements, materials, effectiveTable);

    const auto forcingSeries = LoadForcingCsv(config.forcingFile);

    ThermalForcing constantForcing;
    constantForcing.airTemperature_K = config.airTemperature_K;
    constantForcing.sunIrradiance_W_m2 = config.sunIrradiance_W_m2;
    constantForcing.diffuseIrradiance_W_m2 = config.diffuseIrradiance_W_m2;
    constantForcing.sunDirection = config.sunDirection;
    constantForcing.skyTemperature_K = config.skyTemperature_K;
    constantForcing.relativeHumidity = config.relativeHumidity;

    CpuCrankNicolsonStepper stepper;

    const f64 shortest = CpuCrankNicolsonStepper::ShortestTimeConstantSeconds(
        mesh.elements, materials, config.airTemperature_K);
    if (std::isfinite(shortest) && config.timestep_s > shortest) {
        QL_LOG_WARN("  Thermal: timestep {:.0f} s is longer than the shortest surface time "
                    "constant ({:.0f} s). The radiative coupling is explicit, so the "
                    "result is smoothed rather than unstable -- shorten the step if the "
                    "fastest surface matters.",
                    config.timestep_s, shortest);
    }

    ThermalTimeline::Desc desc;
    desc.startTime_h = config.startTime_h;
    desc.timestep_s = config.timestep_s;
    desc.checkpointStride_h = config.checkpointStride_h;
    desc.nodeCount = config.nodeCount;
    desc.initial = config.initial;
    desc.initialTemperature_K = config.initialTemperature_K;

    ThermalTimeline timeline(desc, mesh.elements, materials, geometry,
                             effectiveTable, forcingSeries, constantForcing, stepper);

    const ThermalState& state = timeline.StateAt(config.time_h);
    result.stepsTaken = timeline.LastStepCount();

    // ------------------------------------------------------------------
    // What the renderer reads
    // ------------------------------------------------------------------
    result.surfaceTemperature_K.resize(mesh.elements.size());
    // What the shading pass needs to undo the per-triangle quantisation of the
    // shadow: the tangent, and the visibility it was taken about.
    const bool haveTangent = state.HasSensitivity();
    result.sunDirection =
        SampleForcing(forcingSeries, config.time_h, constantForcing).sunDirection;
    if (haveTangent) {
        result.sunSensitivity_K.assign(mesh.elements.size(), 0.0f);
        result.sunVisibility =
            SampleSunVisibilityAt(effectiveTable, geometry, config.time_h,
                                  mesh.elements.size());
    }

    f64 sum = 0.0;
    result.minTemperature_K = std::numeric_limits<f64>::max();
    result.maxTemperature_K = std::numeric_limits<f64>::lowest();

    for (usize e = 0; e < mesh.elements.size(); ++e) {
        const u32 id = mesh.elements[e].materialId;
        const bool solved = mesh.elements[e].area_m2 > 0.0f && id < materials.size() &&
                            materials[id].ParticipatesInSolve();
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
            if (glm::dot(mesh.elements[e].normal, result.sunDirection) <= 0.0f) {
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

    QL_LOG_INFO("  Thermal: {} elements ({} solved), {} exchange entries, {} steps, "
                "{:.1f}-{:.1f} K (mean {:.1f} K)",
                result.elementCount, result.participatingElements, result.exchangeNonZeros,
                result.stepsTaken, result.minTemperature_K, result.maxTemperature_K,
                result.meanTemperature_K);
    return result;
}

}  // namespace quantiloom::thermal
