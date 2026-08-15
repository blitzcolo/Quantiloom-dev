/**
 * @file SkyThermal.cpp
 * @brief What a clear sky radiates, from the air temperature and the humidity
 */

#include "core/SkyThermal.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom::skythermal {

namespace {
// Magnus, WMO coefficients. Valid over roughly -45 C to 60 C.
constexpr f64 kMagnusA = 17.62;
constexpr f64 kMagnusB = 243.12;  // C
}  // namespace

f64 DewPointC(const f64 airTemperatureC, const f64 relativeHumidityPercent) {
    // Below a percent or so the logarithm runs away and the correlation has
    // nothing to say anyway; that air is drier than any measured sky.
    const f64 rh = std::clamp(relativeHumidityPercent, 1.0, 100.0);
    const f64 gamma = std::log(rh / 100.0) +
                      kMagnusA * airTemperatureC / (kMagnusB + airTemperatureC);
    return kMagnusB * gamma / (kMagnusA - gamma);
}

f64 ClearSkyEmissivity(const f64 dewPointC) {
    const f64 t = dewPointC / 100.0;
    const f64 eps = 0.711 + 0.56 * t + 0.73 * t * t;
    return std::clamp(eps, 0.0, 1.0);
}

f64 EffectiveSkyTemperatureK(const f64 airTemperatureK, const f64 emissivity) {
    if (airTemperatureK <= 0.0 || emissivity <= 0.0) {
        return 0.0;
    }
    return airTemperatureK * std::pow(std::clamp(emissivity, 0.0, 1.0), 0.25);
}

}  // namespace quantiloom::skythermal
