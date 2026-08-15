"""
Physics reference harness for Quantiloom formula audit.

All functions use only the Python standard library so the harness can be run
without installing dependencies.
"""

import math
import random

# CODATA 2018 exact constants
PLANCK_H = 6.62607015e-34       # J*s
SPEED_OF_LIGHT_C = 299792458.0  # m/s
BOLTZMANN_K = 1.380649e-23      # J/K
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4

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


if __name__ == "__main__":
    run_spot_checks()
