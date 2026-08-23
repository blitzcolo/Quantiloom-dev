#!/usr/bin/env python3
"""Solar position and a clear-sky irradiance model, stdlib only.

Two experiments need the sun's position from a date and a place: the synthetic
desert forcing, and the station-forcing validation, where the measured file
gives irradiance but not always geometry.  Both go through here so that they
cannot disagree about where the sun was at 15:00.

The position is the NOAA solar-position algorithm (the one behind NOAA's own
calculator), accurate to a few hundredths of a degree over the years that
matter -- far inside anything a surface energy balance is sensitive to.

The irradiance model is deliberately simpler and deliberately labelled.  It
produces a *plausible* clear day for a synthetic scene, which is all a
mesh-resolution study needs: that study measures a discretisation error against
a reference integrated from the same forcing, so the forcing has to be
realistic, not measured.  Anywhere a claim rests on the irradiance itself, use
a measured file instead -- that is what the station validation does, and why it
is a separate experiment.
"""

import math

# Solar constant, W/m^2 (WMO).
SOLAR_CONSTANT = 1367.0


def julian_day(year, month, day, hour_utc):
    """Julian day for a civil date and a UTC hour, Gregorian calendar."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1))
            + day + b - 1524.5 + hour_utc / 24.0)


def solar_position(year, month, day, hour_local, latitude_deg, longitude_deg,
                   utc_offset_h):
    """(elevation_deg, azimuth_deg) of the sun, azimuth from north through east.

    Azimuth is returned in the convention the forcing CSV uses, which is also
    the convention LoadForcingCsv turns into a direction: north is 0, east 90.
    Elevation is the refracted (apparent) elevation, since that is what decides
    whether the sun is up.
    """
    jd = julian_day(year, month, day, hour_local - utc_offset_h)
    t = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    m = math.radians(mean_anom)
    centre = (math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
              + math.sin(2 * m) * (0.019993 - 0.000101 * t)
              + math.sin(3 * m) * 0.000289)
    true_long = mean_long + centre
    apparent_long = (true_long - 0.00569
                     - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t)))

    seconds = 21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))
    mean_obliquity = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t))

    declination = math.degrees(math.asin(
        math.sin(math.radians(obliquity)) * math.sin(math.radians(apparent_long))))

    # Equation of time, in minutes.
    y = math.tan(math.radians(obliquity / 2.0)) ** 2
    l0 = math.radians(mean_long)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0)
        - 2.0 * eccentricity * math.sin(m)
        + 4.0 * eccentricity * y * math.sin(m) * math.cos(2 * l0)
        - 0.5 * y * y * math.sin(4 * l0)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * m))

    true_solar_time = (hour_local * 60.0 + eq_time
                       + 4.0 * longitude_deg - 60.0 * utc_offset_h) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    if hour_angle < -180.0:
        hour_angle += 360.0

    lat = math.radians(latitude_deg)
    dec = math.radians(declination)
    ha = math.radians(hour_angle)

    cos_zenith = (math.sin(lat) * math.sin(dec)
                  + math.cos(lat) * math.cos(dec) * math.cos(ha))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elevation = 90.0 - zenith

    # Atmospheric refraction, the NOAA piecewise fit. Matters only near the
    # horizon, where it decides sunrise by about two minutes.
    if elevation > 85.0:
        refraction = 0.0
    elif elevation > 5.0:
        te = math.tan(math.radians(elevation))
        refraction = (58.1 / te - 0.07 / te**3 + 0.000086 / te**5) / 3600.0
    elif elevation > -0.575:
        refraction = (1735.0 + elevation * (-518.2 + elevation * (
            103.4 + elevation * (-12.79 + elevation * 0.711)))) / 3600.0
    else:
        refraction = -20.772 / math.tan(math.radians(elevation)) / 3600.0

    apparent_elevation = elevation + refraction

    denominator = math.cos(lat) * math.sin(math.radians(zenith))
    if abs(denominator) < 1e-9:
        azimuth = 180.0
    else:
        cos_az = (math.sin(lat) * cos_zenith - math.sin(dec)) / denominator
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = math.degrees(math.acos(cos_az))
        azimuth = (180.0 + azimuth) % 360.0 if hour_angle > 0.0 else (180.0 - azimuth) % 360.0

    return apparent_elevation, azimuth


def air_mass(elevation_deg):
    """Kasten-Young relative air mass. Infinite below the horizon."""
    if elevation_deg <= 0.0:
        return float("inf")
    z = 90.0 - elevation_deg
    return 1.0 / (math.sin(math.radians(elevation_deg))
                  + 0.50572 * (96.07995 - z) ** -1.6364)


def clear_sky_irradiance(elevation_deg, transmittance=0.75, diffuse_fraction=0.10):
    """(DNI, DHI) in W/m^2 for a cloudless sky -- SYNTHETIC, not measured.

    DNI follows the Laue/Meinel form `S0 * tau^(AM^0.678)`, which is a two-
    parameter fit rather than a radiative-transfer calculation; the diffuse part
    is a fixed fraction of extraterrestrial horizontal irradiance.  Both are
    standard first approximations and both are wrong in the third digit.  They
    are here to make a synthetic day realistic, and any number that has to be
    defended should come from a measured file instead.
    """
    if elevation_deg <= 0.0:
        return 0.0, 0.0
    am = air_mass(elevation_deg)
    dni = SOLAR_CONSTANT * transmittance ** (am ** 0.678)
    dhi = diffuse_fraction * SOLAR_CONSTANT * math.sin(math.radians(elevation_deg))
    return max(0.0, dni), max(0.0, dhi)


def dew_point_c(air_temperature_c, relative_humidity_percent):
    """Magnus-Tetens dew point with the WMO coefficients -- the same correlation
    core/SkyThermal.hpp uses, so a forcing file written here and a sky computed
    in the renderer agree by construction rather than by coincidence."""
    rh = max(1e-3, min(100.0, relative_humidity_percent)) / 100.0
    a, b = 17.62, 243.12
    gamma = math.log(rh) + a * air_temperature_c / (b + air_temperature_c)
    return b * gamma / (a - gamma)


def clear_sky_temperature_k(air_temperature_k, relative_humidity_percent):
    """Berdahl & Fromberg (1982) effective sky temperature.

    Mirrors skythermal::EffectiveSkyTemperatureK. A forcing CSV can therefore
    carry either this, or a sky temperature inverted from a measured downwelling
    flux -- and the difference between the two runs is exactly what the
    correlation contributes, which is a result rather than a nuisance.
    """
    tdp = dew_point_c(air_temperature_k - 273.15, relative_humidity_percent)
    emissivity = 0.711 + 0.56 * (tdp / 100.0) + 0.73 * (tdp / 100.0) ** 2
    emissivity = max(0.0, min(1.0, emissivity))
    return air_temperature_k * emissivity ** 0.25


def diurnal_air_temperature_k(hour, t_min_k, t_max_k, sunrise_h=6.0, peak_h=15.0):
    """A smooth diurnal air temperature: minimum at sunrise, maximum mid-
    afternoon, cosine between and a slower cosine decay overnight.  Enough
    structure for the surface to lag it, which is the point."""
    amplitude = 0.5 * (t_max_k - t_min_k)
    mean = 0.5 * (t_max_k + t_min_k)
    if sunrise_h <= hour <= peak_h:
        phase = math.pi * (hour - sunrise_h) / (peak_h - sunrise_h)
        return mean - amplitude * math.cos(phase)
    night = (hour - peak_h) % 24.0
    span = 24.0 - (peak_h - sunrise_h)
    phase = math.pi * night / span
    return mean + amplitude * math.cos(phase)
