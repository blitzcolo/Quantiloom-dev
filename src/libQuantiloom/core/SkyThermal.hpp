/**
 * @file SkyThermal.hpp
 * @brief What a clear sky radiates, from the air temperature and the humidity
 *
 * A thermal camera pointed up does not see the air temperature. It sees a
 * partly transparent atmosphere against a background near 3 K, and how cold
 * that reads depends on how much water vapour is in the way -- which is why
 * radiative cooling puts frost on a car roof on a clear night and not on a
 * cloudy one, and why the zenith of an outdoor thermogram is the coldest thing
 * in it.
 *
 * The correlation here is Berdahl and Martin's, which relates the sky's
 * effective emissivity to the dew point alone. Dew point rather than relative
 * humidity because it is the absolute water content that matters, and a dew
 * point of 10 C means the same amount of vapour whatever the air temperature
 * is above it.
 *
 * Berdahl & Martin, "Emissivity of clear skies", Solar Energy 32(5): 663-664,
 * 1984 -- not Berdahl & Fromberg (1982), which this used to cite and whose fit
 * is a different, linear one. The Magnus coefficients are the WMO's.
 *
 * The published fit also carries a diurnal term, +0.013 cos(2 pi t / 24)
 * with t the hour from midnight, and a pressure term, 0.00012 (P - 1000) hPa;
 * neither is used here, so the value is the daily-mean sea-level sky.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::skythermal {

/// Dew point in Celsius, from air temperature in Celsius and relative humidity
/// in percent. Magnus with the WMO coefficients a = 17.62, b = 243.12 C:
///
///   gamma = ln(RH/100) + a T / (b + T)
///   Tdp   = b gamma / (a - gamma)
///
/// Exact at RH = 100, where it returns the air temperature.
[[nodiscard]] f64 DewPointC(f64 airTemperatureC, f64 relativeHumidityPercent);

/// Clear-sky emissivity from the dew point, Berdahl & Martin (1984):
///
///   eps = 0.711 + 0.56 (Tdp/100) + 0.73 (Tdp/100)^2      (Tdp in C)
///
/// without the fit's diurnal (+0.013 cos(2 pi t/24)) and pressure terms.
///
/// Fitted over dew points from -20 C to 30 C; clamped to [0, 1] outside that,
/// which the quadratic would otherwise leave. This is a HEMISPHERICAL
/// emissivity -- it describes the flux onto a horizontal surface, integrated
/// over the whole sky -- and the flat-slab law in the shader treats it as the
/// zenith value instead. That overstates the hemispherical flux by a few
/// percent. The alternative is a second correlation for the angular
/// dependence (Martin-Berdahl), and the slab law is what the NN atmosphere
/// path already uses, so the two paths stay the same shape.
[[nodiscard]] f64 ClearSkyEmissivity(f64 dewPointC);

/// Effective sky temperature: the blackbody that radiates what this sky does.
///
///   T_sky = T_air * eps^(1/4)
///
/// The number a meteorologist quotes and the one a surface energy balance
/// exchanges with, so the thermal solver takes its sky from here.
[[nodiscard]] f64 EffectiveSkyTemperatureK(f64 airTemperatureK, f64 emissivity);

}  // namespace quantiloom::skythermal
