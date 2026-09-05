#include "renderer/ThermalEpochBuilder.hpp"

#include "core/Log.hpp"
#include "thermal/ShortwaveGains.hpp"
#include "thermal/ThermalSolver.hpp"

#include <algorithm>
#include <limits>

namespace quantiloom::rendercore {

namespace {

/// The forcing rows epoch @p e needs, plus one on each side.
///
/// The extra row is not a safety margin, it is the interpolation: a step at
/// the very start of an epoch blends between the row before its span and the
/// first one inside it, and without the earlier row it would clamp to the
/// later one instead -- which puts the sun where it will be rather than where
/// it is.
void SunColumnRange(const Vector<std::pair<f64, thermal::ThermalForcing>>& forcing,
                    const f64 from_h, const f64 to_h, usize& first, usize& last) {
    first = 0;
    last = forcing.empty() ? 0 : forcing.size() - 1;
    if (forcing.size() < 2) return;

    while (first + 1 < forcing.size() && forcing[first + 1].first <= from_h) ++first;
    while (last > 0 && forcing[last - 1].first >= to_h) --last;
    if (last < first) last = first;
}

thermal::SunVisibilityTable BuildSunTable(ThermalExchangePrecompute& precompute,
                                          VkAccelerationStructureKHR tlas,
                                          const thermal::ThermalMesh& mesh,
                                          const thermal::ExchangeGeometry& exchange,
                                          const EpochBuildInput& input, usize firstRow,
                                          usize lastRow) {
    thermal::SunVisibilityTable table;
    const auto& forcing = *input.forcingSeries;

    if (forcing.size() > 1 && tlas != VK_NULL_HANDLE && precompute.IsValid()) {
        Vector<glm::vec3> directions;
        directions.reserve(lastRow - firstRow + 1);
        table.sampleTime_h.reserve(lastRow - firstRow + 1);
        for (usize r = firstRow; r <= lastRow; ++r) {
            table.sampleTime_h.push_back(forcing[r].first);
            directions.push_back(forcing[r].second.sunDirection);
        }

        table.visibility =
            precompute.RunSunVisibility(tlas, mesh.elements, mesh.instanceElementBase,
                                        directions);
        if (table.visibility.size() == directions.size() * mesh.elements.size()) {
            table.sampleDirection = std::move(directions);
        } else {
            table = {};  // the dispatch failed; fall through to one column
        }
    }

    if (table.SampleCount() == 0 && !exchange.sunVisibility.empty()) {
        table.sampleTime_h = {forcing.empty() ? 0.0 : forcing.front().first};
        table.visibility = exchange.sunVisibility;
        table.sampleDirection = {input.fallbackSunDirection};
    }
    return table;
}

}  // namespace

thermal::ThermalGeometrySchedule BuildThermalGeometrySchedule(
    VulkanContext& context, EpochGeometryHost& host, const Scene& scene,
    const EpochBuildInput& input, thermal::ThermalMesh* epoch0MeshOut) {
    thermal::ThermalGeometrySchedule schedule;

    const usize count = input.epochTimes_s.size();
    if (count == 0 || input.materials == nullptr || input.forcingSeries == nullptr) {
        return schedule;
    }

    const Vector<std::pair<f64, thermal::ThermalForcing>>& forcing = *input.forcingSeries;
    const bool keepWholeSunTable = input.sunMemoryLags > 0 && forcing.size() > 1 && count > 1;
    if (keepWholeSunTable) {
        QL_LOG_WARN("  Thermal epochs: sun_memory_lags carries column indices across epoch "
                    "boundaries, so every one of the {} epochs keeps all {} sun columns "
                    "rather than its own span's",
                    count, forcing.size());
    }

    ThermalExchangePrecompute precompute(context);
    precompute.SetMaterialCoverage(input.materialCoverage);

    usize expectedElements = 0;
    schedule.epochs.reserve(count);

    for (usize e = 0; e < count; ++e) {
        const f64 t_s = input.epochTimes_s[e];
        const VkAccelerationStructureKHR tlas = host.ApplyEpoch(t_s);

        // Contacts are the mesh's adjacency, which rigid motion cannot change,
        // so they are found once. Same for the shell pairing and the instance
        // bases, which is why epoch zero's mesh is the one the caller keeps.
        thermal::ThermalMeshOptions options = input.meshOptions;
        if (e > 0) options.contacts = false;
        thermal::ThermalMesh mesh = thermal::BuildThermalMesh(scene, options);

        if (e == 0) {
            expectedElements = mesh.elements.size();
            if (epoch0MeshOut != nullptr) *epoch0MeshOut = mesh;
        } else if (mesh.elements.size() != expectedElements) {
            // Only non-rigid motion or a topology edit can do this, and neither
            // is something the timeline does. Stopping is right: a schedule
            // whose epochs disagree about what element i is would carry one
            // surface's temperature onto another.
            QL_LOG_ERROR("  Thermal epochs: epoch {} at t = {:.3f} s has {} elements where "
                         "epoch 0 had {}; the schedule stops here",
                         e, t_s, mesh.elements.size(), expectedElements);
            break;
        }

        thermal::ExchangeGeometry exchange;
        if (precompute.IsValid() && tlas != VK_NULL_HANDLE && !mesh.elements.empty()) {
            exchange = precompute.Run(tlas, mesh.elements, mesh.instanceElementBase,
                                      input.precompute);
        } else {
            exchange = thermal::MakeOpenSkyExchange(mesh.elements.size());
        }

        usize firstRow = 0;
        usize lastRow = forcing.empty() ? 0 : forcing.size() - 1;
        if (!keepWholeSunTable && forcing.size() > 1 && count > 1) {
            const f64 from_h = (e == 0) ? -std::numeric_limits<f64>::infinity()
                                        : input.epochFrom_h[e];
            const f64 to_h = (e + 1 < count) ? input.epochFrom_h[e + 1]
                                             : std::numeric_limits<f64>::infinity();
            SunColumnRange(forcing, from_h, to_h, firstRow, lastRow);
        }

        thermal::SunVisibilityTable sunTable =
            BuildSunTable(precompute, tlas, mesh, exchange, input, firstRow, lastRow);

        // The gains depend on the geometry, the columns and the absorptivities,
        // and all three are this epoch's -- so this is baked per epoch rather
        // than once for the run.
        thermal::BakeShortwaveGains(exchange, mesh.elements, *input.materials, sunTable);

        thermal::ThermalGeometryEpoch epoch;
        epoch.from_h = (e == 0) ? -std::numeric_limits<f64>::infinity() : input.epochFrom_h[e];
        epoch.timelineTime_s = t_s;
        epoch.elements = std::move(mesh.elements);
        epoch.exchange = std::move(exchange);
        epoch.sunTable = std::move(sunTable);
        schedule.epochs.push_back(std::move(epoch));
    }

    host.Restore();

    if (schedule.Count() > 1) {
        QL_LOG_INFO("  Thermal epochs: {}, {} exchange precompute(s), {} sun column(s) in "
                    "epoch 0",
                    schedule.Count(), schedule.Count(),
                    schedule.epochs.front().sunTable.SampleCount());
    }
    return schedule;
}

}  // namespace quantiloom::rendercore
