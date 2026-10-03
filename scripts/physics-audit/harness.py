"""
Physics reference harness for Quantiloom formula audit.

All functions use only the Python standard library so the harness can be run
without installing dependencies.
"""

import math
import random
from dataclasses import dataclass

# CODATA 2018 exact constants
PLANCK_H = 6.62607015e-34       # J*s
SPEED_OF_LIGHT_C = 299792458.0  # m/s
BOLTZMANN_K = 1.380649e-23      # J/K
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4


@dataclass(frozen=True)
class CameraResponse:
    """Piecewise-linear device response, with an explicit calibration meaning.

    Absolute QE and optical transmission are fractions of incident photons or
    power. A system response already includes detector, filter and lens;
    multiplying any of them again is an error. For relative QE, peak_one means
    amplitude is peak QE, while area_one means amplitude has units nm and is
    the integrated QE. Source documents where that amplitude came from.
    """
    points: tuple
    kind: str = "absolute_qe"
    normalization: str = "none"
    amplitude: float = 1.0
    source: str = ""


def _validated_points(points, label, maximum=None):
    points = tuple((float(x), float(y)) for x, y in points)
    if len(points) < 2:
        raise ValueError(f"{label} needs at least two samples")
    for i, (wavelength, value) in enumerate(points):
        if (not math.isfinite(wavelength) or wavelength <= 0 or
                not math.isfinite(value) or value < 0 or
                (maximum is not None and value > maximum) or
                (i and wavelength <= points[i - 1][0])):
            raise ValueError(f"invalid {label} sample")
    return points


def _area(points):
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2
               for a, b in zip(points, points[1:]))


def _validate_response(response):
    if not isinstance(response, CameraResponse):
        response = CameraResponse(tuple(response))
    if response.kind not in ("absolute_qe", "relative_qe", "thermal_absorptance",
                             "lens_transmission", "filter_transmission",
                             "system_photon_qe", "system_thermal_absorptance"):
        raise ValueError("unknown response kind")
    if not math.isfinite(response.amplitude) or response.amplitude < 0:
        raise ValueError("invalid response amplitude")
    if response.kind == "relative_qe":
        if response.normalization not in ("peak_one", "area_one") or not response.source:
            raise ValueError("relative QE needs normalization and amplitude source")
        points = _validated_points(response.points, "response")
        norm = max(value for _, value in points) if response.normalization == "peak_one" else _area(points)
        if not math.isclose(norm, 1.0, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("relative QE shape does not match declared normalization")
        if any(response.amplitude * y > 1.0 + 1e-12 for _, y in points):
            raise ValueError("relative QE amplitude exceeds unity")
    else:
        if response.normalization != "none" or response.amplitude != 1.0:
            raise ValueError("absolute response must not be renormalized")
        points = _validated_points(response.points, "response", maximum=1.0)
    return CameraResponse(points, response.kind, response.normalization,
                          response.amplitude, response.source)


def _interpolate(points, nm):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= nm <= x1:
            return y0 + (y1 - y0) * (nm - x0) / (x1 - x0)
    raise ValueError("spectrum does not cover the response")


def _measurement_setup(response, filters, lens, fill_factor):
    response = _validate_response(response)
    filters = tuple(filters)
    if (not math.isfinite(fill_factor) or fill_factor < 0 or fill_factor > 1):
        raise ValueError("fill factor must be a fraction")
    if response.kind in ("system_photon_qe", "system_thermal_absorptance") and (
            filters or lens is not None or fill_factor != 1):
        raise ValueError("system response already includes optics and fill factor")
    curves = [response]
    for curve in filters:
        curve = _validate_response(curve)
        if curve.kind != "filter_transmission":
            raise ValueError("filters must be filter transmissions")
        if curve.points[0][0] > response.points[0][0] or curve.points[-1][0] < response.points[-1][0]:
            raise ValueError("optical factor does not cover the response")
        curves.append(curve)
    if lens is not None:
        lens = _validate_response(lens)
        if lens.kind != "lens_transmission":
            raise ValueError("lens must be lens transmission")
        if lens.points[0][0] > response.points[0][0] or lens.points[-1][0] < response.points[-1][0]:
            raise ValueError("lens does not cover the response")
        curves.append(lens)
    return curves


# Eight-point Gauss-Legendre quadrature is exact for polynomials through degree
# 15. Within each knot interval every declared curve is linear, so the photon
# integrand is a low-degree polynomial. Nodes/weights: Abramowitz & Stegun 25.4.
_GAUSS_X = (0.09501250983763744, 0.2816035507792589,
            0.4580167776572274, 0.6178762444026438,
            0.7554044083550030, 0.8656312023878318,
            0.9445750230732326, 0.9894009349916499)
_GAUSS_W = (0.1894506104550685, 0.1826034150449236,
            0.1691565193950025, 0.1495959888165767,
            0.1246289712555339, 0.09515851168249278,
            0.06225352393864789, 0.02715245941175410)


def _integrate_camera_density(source, response, pixel_area_m2, photon,
                              filters=(), lens=None, fill_factor=1, source_knots=()):
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("pixel area must be positive and finite")
    curves = _measurement_setup(response, filters, lens, fill_factor)
    low, high = curves[0].points[0][0], curves[0].points[-1][0]
    knots = sorted({x for curve in curves for x, _ in curve.points if low <= x <= high} |
                   {x for x in source_knots if low < x < high})

    def density(nm):
        optical = fill_factor
        for curve in curves:
            optical *= curve.amplitude * _interpolate(curve.points, nm)
        quantum = nm * 1e-9 / (PLANCK_H * SPEED_OF_LIGHT_C) if photon else 1.0
        return source(nm) * optical * quantum

    total = 0.0
    for a, b in zip(knots, knots[1:]):
        middle, half_width = (a + b) / 2, (b - a) / 2
        total += half_width * sum(weight * (density(middle - half_width * node) +
                                            density(middle + half_width * node))
                                  for node, weight in zip(_GAUSS_X, _GAUSS_W))
    return pixel_area_m2 * total


def photon_electron_rate(pre_optics_irradiance, response, pixel_area_m2, *, filters=(),
                         lens=None, fill_factor=1):
    """EMVA 1288 photon conversion, e-/s, from pre-optics W/m^2/nm.

    Input is scene radiance times pupil solid angle, before lens and filter.
    The plan's E_lambda is the detector-plane value AFTER those transmissions;
    use photon_electron_rate_detector_plane for that boundary. Wavelength
    becomes metres only in lambda/(hc).
    """
    samples = _validated_points(pre_optics_irradiance, "pre-optics irradiance")
    response = _validate_response(response)
    if response.kind not in ("absolute_qe", "relative_qe", "system_photon_qe"):
        raise ValueError("photon detector needs QE or system response")
    if samples[0][0] > response.points[0][0] or samples[-1][0] < response.points[-1][0]:
        raise ValueError("irradiance does not cover the response")
    return _integrate_camera_density(lambda nm: _interpolate(samples, nm), response,
                                     pixel_area_m2, True, filters, lens, fill_factor,
                                     (x for x, _ in samples))


def absorbed_power(pre_optics_irradiance, response, pixel_area_m2, *, filters=(),
                   lens=None, fill_factor=1):
    """Thermal watts from pre-optics irradiance; no photon factor."""
    samples = _validated_points(pre_optics_irradiance, "pre-optics irradiance")
    response = _validate_response(response)
    if response.kind not in ("thermal_absorptance", "system_thermal_absorptance"):
        raise ValueError("thermal detector needs absorptance or system response")
    if samples[0][0] > response.points[0][0] or samples[-1][0] < response.points[-1][0]:
        raise ValueError("irradiance does not cover the response")
    return _integrate_camera_density(lambda nm: _interpolate(samples, nm), response,
                                     pixel_area_m2, False, filters, lens, fill_factor,
                                     (x for x, _ in samples))


def photon_electron_rate_detector_plane(detector_plane_irradiance, qe, pixel_area_m2):
    """Plan/EMVA formula with E_lambda already after all optical transmission.

    No lens/filter/fill-factor arguments exist at this boundary, so those
    factors cannot be counted twice. Only detector QE is applied.
    """
    qe = _validate_response(qe)
    if qe.kind not in ("absolute_qe", "relative_qe"):
        raise ValueError("detector-plane signal needs detector QE, not system response")
    return photon_electron_rate(detector_plane_irradiance, qe, pixel_area_m2)


def absorbed_power_detector_plane(detector_plane_irradiance, absorptance,
                                  pixel_area_m2):
    """Thermal detector absorption after lens and filter transmission."""
    absorptance = _validate_response(absorptance)
    if absorptance.kind != "thermal_absorptance":
        raise ValueError("detector-plane power needs thermal absorptance")
    return absorbed_power(detector_plane_irradiance, absorptance, pixel_area_m2)


def expected_photoelectrons(samples, response, pixel_area_m2, exposure_s, **optics):
    """EMVA 1288 constant-irradiance exposure integral, in electrons."""
    if not math.isfinite(exposure_s) or exposure_s < 0:
        raise ValueError("exposure must be nonnegative and finite")
    return exposure_s * photon_electron_rate(samples, response, pixel_area_m2, **optics)


def thermal_detector_step(previous, absorbed_power, dt_s, tau_s):
    """Exact constant-forcing solution of tau*ds/dt+s=Pabsorbed."""
    if (not all(math.isfinite(x) for x in (previous, absorbed_power, dt_s, tau_s)) or
            previous < 0 or absorbed_power < 0 or dt_s < 0 or tau_s < 0):
        raise ValueError("thermal state, power, step and time constant must be nonnegative")
    if dt_s == 0:
        return previous
    if tau_s == 0:
        return absorbed_power
    change = -math.expm1(-dt_s / tau_s)
    return previous + change * (absorbed_power - previous)

# First radiation constant in wavelength form, per nm
C1_NM = 1.191042972e20          # W*nm^4/m^2/sr
# hc/k in meters
C2_M = 1.438776877e-2           # m*K


def planck_blackbody(T_K: float, lambda_nm: float) -> float:
    """
    Spectral radiance of a blackbody per unit wavelength (W sr^-1 m^-2 nm^-1).
    """
    if T_K <= 0.0 or lambda_nm <= 0.0:
        return 0.0
    lambda_m = lambda_nm * 1e-9
    exponent = C2_M / (lambda_m * T_K)
    # Clamp to avoid overflow in exp(). exp(80) ~ 5.5e34, safe for float64.
    if exponent > 700.0:
        return 0.0
    denom = math.exp(exponent) - 1.0
    if denom == 0.0:
        return 0.0
    return C1_NM / (lambda_nm ** 5) / denom


def planck_blackbody_temperature_derivative(T_K: float, lambda_nm: float) -> float:
    """dB_lambda/dT, W/(m^2 sr nm K), from Planck's wavelength law.

    For x=hc/(lambda*k*T), dB/dT=B*x/[T*(1-exp(-x))]. This analytic
    derivative is what a NETD calculation needs, rather than a finite
    difference tied to an arbitrary temperature increment.
    """
    if T_K <= 0 or lambda_nm <= 0:
        return 0.0
    x = C2_M / (lambda_nm * 1e-9 * T_K)
    if x > 700:
        return 0.0
    return planck_blackbody(T_K, lambda_nm) * x / (-T_K * math.expm1(-x))


def _blackbody_measurement(T_K, response, pixel_area_m2, solid_angle_sr,
                           photon, derivative, **optics):
    if not math.isfinite(T_K) or T_K <= 0:
        raise ValueError("blackbody temperature must be positive and finite")
    if not math.isfinite(solid_angle_sr) or solid_angle_sr <= 0:
        raise ValueError("pupil solid angle must be positive and finite")
    response = _validate_response(response)
    allowed = ("absolute_qe", "relative_qe", "system_photon_qe") if photon else (
        "thermal_absorptance", "system_thermal_absorptance")
    if response.kind not in allowed:
        raise ValueError("incompatible detector response")
    planck = planck_blackbody_temperature_derivative if derivative else planck_blackbody
    return _integrate_camera_density(lambda nm: solid_angle_sr * planck(T_K, nm),
                                     response, pixel_area_m2, photon, **optics)


def blackbody_photon_rate_and_derivative(T_K, response, pixel_area_m2,
                                         solid_angle_sr, **optics):
    """Blackbody e-/s and e-/(s K) for the same QE/optics response integral."""
    return (_blackbody_measurement(T_K, response, pixel_area_m2, solid_angle_sr,
                                   True, False, **optics),
            _blackbody_measurement(T_K, response, pixel_area_m2, solid_angle_sr,
                                   True, True, **optics))


def blackbody_power_and_derivative(T_K, response, pixel_area_m2,
                                   solid_angle_sr, **optics):
    """Blackbody absorbed W and W/K for the same thermal response integral."""
    return (_blackbody_measurement(T_K, response, pixel_area_m2, solid_angle_sr,
                                   False, False, **optics),
            _blackbody_measurement(T_K, response, pixel_area_m2, solid_angle_sr,
                                   False, True, **optics))


def physical_horizontal_fov_deg(array_width_px, pixel_pitch_um, focal_length_mm):
    """Rectilinear thin-lens field of view: 2 atan(sensor_width/(2 f))."""
    if (not isinstance(array_width_px, int) or array_width_px <= 0 or
            not math.isfinite(pixel_pitch_um) or pixel_pitch_um <= 0 or
            not math.isfinite(focal_length_mm) or focal_length_mm <= 0):
        raise ValueError("physical array width, pitch and focal length are required")
    width_mm = array_width_px * pixel_pitch_um * 1e-3
    return math.degrees(2 * math.atan(width_mm / (2 * focal_length_mm)))


def projected_pupil_solid_angle_sr(f_number):
    """Projected cone integral int(cos(theta) dOmega)=pi*sin(alpha)^2.

    tan(alpha)=1/(2N). This irradiance factor has steradian units but is NOT
    the geometric cone solid angle 2*pi*(1-cos(alpha)); do not multiply an
    additional cosine after using it. See the radiance-to-irradiance relation.
    """
    if not math.isfinite(f_number) or f_number <= 0:
        raise ValueError("f-number must be positive and finite")
    return math.pi / (1 + 4*f_number*f_number)


def radiance_to_preoptics_irradiance(spectral_radiance, f_number,
                                     field_angle_rad=0.0, cos4_vignetting=False):
    """Convert W/(m^2 sr nm) to pre-transmission W/(m^2 nm).

    Radiance is integrated over the projected pupil cone; the optional cos^4
    field-angle factor is a separate natural-vignetting model assumption.
    Lens and filter transmission
    belong later in the camera response integral. This uses the radiometric
    Lambertian cone relation rather than the paraxial pi/(4N^2) limit.
    """
    if (not math.isfinite(field_angle_rad) or
            abs(field_angle_rad) >= math.pi/2):
        raise ValueError("field angle must face the pupil")
    samples = _validated_points(spectral_radiance, "spectral radiance")
    factor = projected_pupil_solid_angle_sr(f_number)
    if cos4_vignetting:
        factor *= math.cos(field_angle_rad)**4
    return tuple((nm, value*factor) for nm, value in samples)


def _bessel_j(integer_order, x):
    """Bessel J0/J1 from their power series or angular integral (DLMF 10.2, 10.9)."""
    if x < 20:
        term = 1.0 if integer_order == 0 else x / 2
        total = term
        for k in range(1, 100):
            term *= -(x * x / 4) / (k * (k + integer_order))
            total += term
            if abs(term) < 1e-16 * max(1.0, abs(total)):
                break
        return total
    # J_n(x)=(1/pi) integral_0^pi cos(n*t-x*sin(t)) dt. Simpson resolves
    # many oscillations without the cancellation of the large-x power series.
    steps = max(1024, int(64 * x))
    if steps % 2:
        steps += 1
    h = math.pi / steps
    value = math.cos(0.0) + math.cos(integer_order * math.pi)
    for i in range(1, steps):
        angle = i * h
        value += (4 if i % 2 else 2) * math.cos(integer_order * angle - x * math.sin(angle))
    return value * h / (3 * math.pi)


def airy_psf_density_per_m2(radius_m, wavelength_nm, f_number):
    """Normalized diffraction PSF per image-plane m^2.

    Airy intensity is k^2/(4*pi) [2 J1(k*r)/(k*r)]^2,
    k=pi/(lambda*N). The integral over the entire image plane is one;
    aperture throughput belongs in irradiance, not in PSF normalization.
    See Born & Wolf, Principles of Optics, circular-aperture diffraction.
    """
    if (not math.isfinite(radius_m) or radius_m < 0 or
            not math.isfinite(wavelength_nm) or wavelength_nm <= 0 or
            not math.isfinite(f_number) or f_number <= 0):
        raise ValueError("invalid Airy parameters")
    k = math.pi / (wavelength_nm * 1e-9 * f_number)
    x = k * radius_m
    ratio = 1.0 if x == 0 else 2 * _bessel_j(1, x) / x
    return k * k * ratio * ratio / (4 * math.pi)


def airy_encircled_energy(radius_m, wavelength_nm, f_number):
    """Airy energy inside radius: 1-J0(x)^2-J1(x)^2, x=pi*r/(lambda*N)."""
    if (not math.isfinite(radius_m) or radius_m < 0 or
            not math.isfinite(wavelength_nm) or wavelength_nm <= 0 or
            not math.isfinite(f_number) or f_number <= 0):
        raise ValueError("invalid Airy parameters")
    x = math.pi * radius_m / (wavelength_nm * 1e-9 * f_number)
    return 1 - _bessel_j(0, x) ** 2 - _bessel_j(1, x) ** 2


def wien_peak_wavelength(T_K: float) -> float:
    """Peak wavelength (nm) from Wien's displacement law."""
    if T_K <= 0.0:
        return 0.0
    # x where (x-5)*exp(x) + 5 = 0 -> x = 4.965114231744276
    x = 4.965114231744276
    return (C2_M / x) / T_K * 1e9


def stefan_boltzmann_radiance(T_K: float) -> float:
    """Total radiated power per unit area (W/m^2) for a blackbody."""
    if T_K <= 0.0:
        return 0.0
    return STEFAN_BOLTZMANN * T_K ** 4


def fresnel_exact_conductor(cos_theta: float, n: float, k: float) -> float:
    """
    Exact unpolarized Fresnel reflectance for a conductor with complex index
    n + i*k. Matches the formulation in src/shaders/common.hlsli.
    """
    cos_theta = min(1.0, max(0.0, abs(cos_theta)))
    cos2 = cos_theta * cos_theta
    sin2 = 1.0 - cos2

    n2 = n * n
    k2 = k * k

    t0 = n2 - k2 - sin2
    a2b2 = math.sqrt(t0 * t0 + 4.0 * n2 * k2)
    t1 = a2b2 + cos2
    a = math.sqrt(0.5 * (a2b2 + t0))
    t2 = 2.0 * a * cos_theta
    Rs = (t1 - t2) / (t1 + t2)

    t3 = cos2 * a2b2 + sin2 * sin2
    t4 = t2 * sin2
    Rp = Rs * (t3 - t4) / (t3 + t4)

    return 0.5 * (Rs + Rp)


def homogeneous_slab_coefficients(n, sigma_per_m, thickness_m):
    """Normal-incidence parallel dielectric slab, including all internal returns.

    Fresnel interfaces: PBRT 4e, Reflection Models/Dielectric BSDF.
    Beer attenuation: PBRT 4e, Volume Scattering/Transmittance.
    The geometric series gives R,T and the LTE absorption/emission fraction A.
    """
    if n <= 0 or sigma_per_m < 0 or thickness_m <= 0:
        raise ValueError('invalid slab parameters')
    f = fresnel_exact_dielectric(1, 1, n)
    a = math.exp(-sigma_per_m * thickness_m)
    denominator = 1 - f*f*a*a
    t = (1-f)**2*a/denominator
    r = f + (1-f)**2*f*a*a/denominator
    return r, t, 1-r-t


def fresnel_exact_dielectric(cosI: float, n1: float, n2: float) -> float:
    """
    Exact Fresnel reflectance for a dielectric-dielectric interface.
    Returns 1.0 for total internal reflection.
    """
    cosI = min(1.0, max(0.0, abs(cosI)))
    sin2I = 1.0 - cosI * cosI
    eta = n1 / n2
    sin2T = eta * eta * sin2I
    if sin2T > 1.0:
        return 1.0
    cosT = math.sqrt(1.0 - sin2T)
    Rs = ((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)) ** 2
    Rp = ((n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI)) ** 2
    return 0.5 * (Rs + Rp)


def schlick_fresnel(cos_theta: float, F0: float) -> float:
    """Schlick Fresnel approximation."""
    cos_theta = min(1.0, max(0.0, cos_theta))
    return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5


def fresnel_f0(n: float, k: float = 0.0) -> float:
    """Normal-incidence reflectance from n and k."""
    return ((n - 1.0) ** 2 + k * k) / ((n + 1.0) ** 2 + k * k)


def rayleigh_phase(cos_theta: float) -> float:
    """Normalized Rayleigh phase function."""
    return (3.0 / (16.0 * math.pi)) * (1.0 + cos_theta * cos_theta)


def henyey_greenstein(cos_theta: float, g: float) -> float:
    """Normalized Henyey-Greenstein phase function."""
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * math.pi * denom ** 1.5)


def koschmieder_visibility(beta_m_550nm: float) -> float:
    """
    Meteorological visibility (km) from Mie extinction coefficient at 550 nm.
    Uses the classic Koschmieder relation V = 3.912 / beta (m^-1).
    """
    if beta_m_550nm <= 0.0:
        return float('inf')
    return 3.912 / beta_m_550nm / 1000.0


def sphere_angle_integrate(f, n_samples: int = 200000) -> float:
    """
    Monte-Carlo integration of a phase function over the unit sphere.
    Returns the average value times 4*pi.
    """
    total = 0.0
    for _ in range(n_samples):
        # Uniform sphere sampling
        u1 = random.random()
        u2 = random.random()
        theta = math.acos(2.0 * u1 - 1.0)
        phi = 2.0 * math.pi * u2
        cos_theta = math.cos(theta)
        total += f(cos_theta)
    avg = total / n_samples
    return avg * 4.0 * math.pi


def ggx_distribution(NdotH: float, alpha: float) -> float:
    """GGX (Trowbridge-Reitz) normal distribution function."""
    a2 = alpha * alpha
    denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom * denom)


def ggx_smith_g1(NdotX: float, alpha: float) -> float:
    """Smith G1 for GGX."""
    a2 = alpha * alpha
    return (2.0 * NdotX) / (NdotX + math.sqrt(a2 + (1.0 - a2) * NdotX * NdotX))


def ggx_energy_integral(NdotV: float, roughness: float, samples: int = 65536) -> float:
    """
    Estimate the directional-hemispherical reflectance of the Cook-Torrance
    specular lobe for F0 = 1 (white, non-metallic) using importance sampling.
    For a perfectly white, energy-conserving BRDF this should be <= 1.
    """
    alpha = roughness * roughness
    NdotV = min(1.0, max(0.0, NdotV))
    V = (math.sqrt(1.0 - NdotV * NdotV), 0.0, NdotV)
    N = (0.0, 0.0, 1.0)

    total = 0.0
    rng = random.Random(42)
    for _ in range(samples):
        # Importance sample GGX
        xi1 = rng.random()
        xi2 = rng.random()
        phi = 2.0 * math.pi * xi1
        cos_theta = math.sqrt((1.0 - xi2) / (1.0 + (alpha * alpha - 1.0) * xi2))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        H = (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)

        # L = reflect(-V, H)
        VdotH = V[0]*H[0] + V[1]*H[1] + V[2]*H[2]
        L = (2.0 * VdotH * H[0] - V[0],
             2.0 * VdotH * H[1] - V[1],
             2.0 * VdotH * H[2] - V[2])
        NdotL = L[2]
        if NdotL <= 0.0 or VdotH <= 0.0:
            continue
        NdotH = cos_theta

        D = ggx_distribution(NdotH, alpha)
        G1_V = ggx_smith_g1(NdotV, alpha)
        G1_L = ggx_smith_g1(NdotL, alpha)
        G = G1_V * G1_L

        # PDF = D * NdotH / (4 * VdotH)
        # Weight = F * G * VdotH / (NdotH * NdotV)
        # With F = 1:
        weight = G * VdotH / (NdotH * NdotV)
        total += weight

    return total / samples


def optical_depth_exponential(
    planet_radius: float,
    atmosphere_height: float,
    altitude0: float,
    zenith_cos: float,
    beta0: float,
    scale_height: float,
    steps: int = 1024
) -> float:
    """
    Numerical optical depth through an exponential atmosphere along a ray.
    Simple ray-march from altitude0 along direction with cosine zenith angle.
    """
    # Rough upper bound on path length within atmosphere
    # For downward rays we clamp to ground intersection.
    r0 = planet_radius + altitude0
    # solve r^2 = (planet_radius + h)^2; line: r = r0 + s * direction, etc.
    # Simplified: march until altitude < 0 or > atmosphere_height.
    step_len = atmosphere_height / steps
    tau = 0.0
    s = 0.0
    while s < atmosphere_height * 10.0:
        s += step_len
        # altitude approximation along a straight line
        # Use law of cosines for spherical shell
        r2 = r0 * r0 + s * s + 2.0 * r0 * s * zenith_cos
        r = math.sqrt(r2)
        h = r - planet_radius
        if h < 0.0 or h > atmosphere_height:
            break
        rho = math.exp(-h / scale_height)
        tau += beta0 * rho * step_len
    return tau


def run_spot_checks():
    """Run quick sanity checks and print results."""
    print("Planck(300, 9659) =", planck_blackbody(300.0, 9659.0))
    print("Wien peak 300K   =", wien_peak_wavelength(300.0), "nm")
    print("SB 300K          =", stefan_boltzmann_radiance(300.0), "W/m^2")
    print("Fresnel Al F0    =", fresnel_f0(0.27, 3.29))
    print("Fresnel Al @ 45   =", fresnel_exact_conductor(0.7071, 0.27, 3.29))
    print("Schlick Al F0 @45 =", schlick_fresnel(0.7071, fresnel_f0(0.27, 3.29)))
    print("Visibility 2e-6  =", koschmieder_visibility(2.0e-6), "km")
    print("Visibility 2e-4  =", koschmieder_visibility(2.0e-4), "km")
    print("Rayleigh integral=", sphere_angle_integrate(lambda c: rayleigh_phase(c), 200000))
    print("HG integral g=0.76=", sphere_angle_integrate(lambda c: henyey_greenstein(c, 0.76), 200000))
    print("GGX energy rough=0.5 NdotV=1 =", ggx_energy_integral(1.0, 0.5, 16384))


def flat_atmosphere_sky_radiance(eps0: float, cos_zenith: float,
                                 T_air: float, lambda_nm: float) -> float:
    """Flat-slab secant sky radiance model matching AtmosSkyRadianceIR shader.

    eps0       -- zenith effective emissivity [0, 1]
    cos_zenith -- cosine of zenith angle (+1 = overhead, 0 = horizon)
    T_air      -- air temperature (K)
    lambda_nm  -- wavelength (nm)
    """
    SEC_MAX = 5.0
    B_air = planck_blackbody(T_air, lambda_nm)
    if B_air < 1e-30:
        return 0.0
    abscos = max(abs(cos_zenith), 0.01)
    sec_theta = min(1.0 / abscos, SEC_MAX)
    transparency = max(1.0 - eps0, 0.0) ** sec_theta
    return B_air * (1.0 - transparency)


def band_average_radiance(T_K: float, lambda_min_nm: float, lambda_max_nm: float,
                          nodes: int = 16) -> float:
    """Band-average spectral radiance on the renderer's own quadrature.

    The fused thermal bands integrate `nodes` wavelengths by the trapezoid rule
    with half-weight endpoints and divide by the band width (closesthit.rchit,
    NUM_IR_SAMPLES). Reproduced here rather than integrated exactly, because
    the number this checks is what the renderer writes -- a finer rule would
    disagree with the image by the quadrature error and call it a bug.
    """
    if nodes < 2 or lambda_max_nm <= lambda_min_nm or T_K <= 0.0:
        return 0.0
    step = (lambda_max_nm - lambda_min_nm) / (nodes - 1)
    total = 0.0
    for i in range(nodes):
        lam = lambda_min_nm + i * step
        weight = 0.5 if i in (0, nodes - 1) else 1.0
        total += planck_blackbody(T_K, lam) * weight
    return total / (nodes - 1)


def invert_band_average_radiance(radiance: float, lambda_min_nm: float,
                                 lambda_max_nm: float, nodes: int = 16) -> float:
    """Temperature whose band average is `radiance`, by bisection.

    Bisection rather than Newton on purpose: the renderer's inversion uses
    Newton with an analytic derivative, and a reference that shares that
    derivative would not be checking it. Band radiance is strictly increasing
    in T, which is what makes bisection sufficient.
    """
    if radiance <= 0.0:
        return 100.0
    lo, hi = 100.0, 3000.0
    if radiance <= band_average_radiance(lo, lambda_min_nm, lambda_max_nm, nodes):
        return lo
    if radiance >= band_average_radiance(hi, lambda_min_nm, lambda_max_nm, nodes):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if band_average_radiance(mid, lambda_min_nm, lambda_max_nm, nodes) < radiance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def flir_surface_temperature(radiance: float, lambda_min_nm: float, lambda_max_nm: float,
                             emissivity: float = 1.0, reflected_T_K: float = 0.0,
                             tau_atm: float = 1.0, atm_T_K: float = 0.0,
                             nodes: int = 16) -> float:
    """Surface temperature a thermal camera reports, per band.

    Aguerre et al. 2020 eq. 8 with the band average in place of sigma T^4:

        L = tau [eps B(Ts) + (1 - eps) B(Trefl)] + (1 - tau) B(Tatm)

    solved for B(Ts) and inverted. Their whole-spectrum form is the total flux
    of a blackbody, which is not what a band-limited camera collects; using it
    would fold the out-of-band tail into every temperature.
    """
    if emissivity <= 0.0 or tau_atm <= 0.0:
        return 100.0
    b_surface = radiance / tau_atm
    if emissivity < 1.0 and reflected_T_K > 0.0:
        b_surface -= (1.0 - emissivity) * band_average_radiance(
            reflected_T_K, lambda_min_nm, lambda_max_nm, nodes)
    if tau_atm < 1.0 and atm_T_K > 0.0:
        b_surface -= ((1.0 - tau_atm) / tau_atm) * band_average_radiance(
            atm_T_K, lambda_min_nm, lambda_max_nm, nodes)
    b_surface /= emissivity
    return invert_band_average_radiance(b_surface, lambda_min_nm, lambda_max_nm, nodes)


def dew_point_c(air_temperature_c: float, relative_humidity_percent: float) -> float:
    """Dew point in Celsius, Magnus with the WMO coefficients.

    gamma = ln(RH/100) + a T / (b + T),  Tdp = b gamma / (a - gamma)

    Exact at RH = 100, where it returns the air temperature -- which is the
    property worth checking, since everything else here is a fit.
    """
    a, b = 17.62, 243.12
    rh = min(max(relative_humidity_percent, 1.0), 100.0)
    gamma = math.log(rh / 100.0) + a * air_temperature_c / (b + air_temperature_c)
    return b * gamma / (a - gamma)


def clear_sky_emissivity(dew_point_c_value: float) -> float:
    """Berdahl-Fromberg clear-sky emissivity from the dew point.

    eps = 0.711 + 0.56 (Tdp/100) + 0.73 (Tdp/100)^2

    Solar Energy 29(4), 1982. A hemispherical emissivity: it describes the flux
    onto a horizontal surface. The renderer feeds it to the flat-slab law as a
    zenith value instead, which overstates the hemispherical flux by a few
    percent -- the trade is documented in SkyThermal.hpp.
    """
    t = dew_point_c_value / 100.0
    return min(max(0.711 + 0.56 * t + 0.73 * t * t, 0.0), 1.0)


def effective_sky_temperature_k(air_temperature_k: float, emissivity: float) -> float:
    """The blackbody that radiates what this sky does: T_air * eps^(1/4)."""
    if air_temperature_k <= 0.0 or emissivity <= 0.0:
        return 0.0
    return air_temperature_k * min(max(emissivity, 0.0), 1.0) ** 0.25


def clear_sky_radiance(eps0: float, cos_zenith: float, t_air: float,
                       lambda_nm: float) -> float:
    """Flat-slab clear sky, the analytic twin of flat_atmosphere_sky_radiance.

    Same law; eps0 comes from the Berdahl-Fromberg correlation rather than from
    the network's baked zenith downwelling.
    """
    return flat_atmosphere_sky_radiance(eps0, cos_zenith, t_air, lambda_nm)


if __name__ == "__main__":
    run_spot_checks()
