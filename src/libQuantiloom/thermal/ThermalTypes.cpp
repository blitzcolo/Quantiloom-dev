/**
 * @file ThermalTypes.cpp
 * @brief Non-trivial members of the thermal POD types
 */

#include "thermal/ThermalTypes.hpp"

#include <algorithm>
#include <limits>

namespace quantiloom::thermal {

const char* ThermalParameterName(const ThermalParameter parameter) {
    switch (parameter) {
        case ThermalParameter::Convection:   return "h";
        case ThermalParameter::Emissivity:   return "epsilon";
        case ThermalParameter::Absorptivity: return "alpha";
        case ThermalParameter::Conductivity: return "k";
        case ThermalParameter::HeatCapacity: return "rhoc";
        case ThermalParameter::Count:        break;
    }
    return "";
}

ThermalParameter ThermalParameterFromName(const StringView name) {
    for (u8 i = 0; i < static_cast<u8>(ThermalParameter::Count); ++i) {
        const auto parameter = static_cast<ThermalParameter>(i);
        if (name == ThermalParameterName(parameter)) return parameter;
    }
    return ThermalParameter::Count;
}

void SunVisibilityTable::SampleIndices(const f64 t, usize& a, usize& b,
                                       f64& blend) const {
    const usize K = sampleTime_h.size();
    if (K <= 1) {
        a = b = 0;
        blend = 0.0;
        return;
    }
    if (t <= sampleTime_h.front()) {
        a = b = 0;
        blend = 0.0;
        return;
    }
    if (t >= sampleTime_h.back()) {
        a = b = K - 1;
        blend = 0.0;
        return;
    }
    const auto it =
        std::upper_bound(sampleTime_h.begin(), sampleTime_h.end(), t);
    b = static_cast<usize>(it - sampleTime_h.begin());
    a = b - 1;
    const f64 span = sampleTime_h[b] - sampleTime_h[a];
    blend = span > 0.0 ? (t - sampleTime_h[a]) / span : 0.0;
}

usize ThermalGeometrySchedule::EpochAt(const f64 time_h) const {
    if (epochs.size() <= 1) return 0;
    // Linear rather than a binary search: the count is the number of times
    // something in the scene moved far enough to matter, which is tens, and a
    // scan of tens of doubles is not worth being clever about.
    usize found = 0;
    for (usize e = 1; e < epochs.size(); ++e) {
        if (epochs[e].from_h <= time_h) {
            found = e;
        } else {
            break;
        }
    }
    return found;
}

ThermalGeometrySchedule ThermalGeometrySchedule::Single(ExchangeGeometry exchange,
                                                        SunVisibilityTable sunTable,
                                                        Vector<ThermalElement> elements) {
    ThermalGeometrySchedule schedule;
    ThermalGeometryEpoch epoch;
    epoch.from_h = -std::numeric_limits<f64>::infinity();
    epoch.elements = std::move(elements);
    epoch.exchange = std::move(exchange);
    epoch.sunTable = std::move(sunTable);
    schedule.epochs.push_back(std::move(epoch));
    return schedule;
}

}  // namespace quantiloom::thermal
