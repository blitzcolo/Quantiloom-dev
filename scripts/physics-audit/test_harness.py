"""Spot-check tests for the physics audit harness."""

import math
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])

import harness as H


def approx(a, b, tol=0.05):
    if b == 0.0:
        return abs(a) < tol
    return abs(a - b) / abs(b) < tol


def test_planck_300k_peak():
    val = H.planck_blackbody(300.0, 9659.0)
    assert 0.008 <= val <= 0.012, f"Planck(300,9659) out of range: {val}"


def test_wien_displacement():
    peak = H.wien_peak_wavelength(300.0)
    assert approx(peak, 9659.0, 0.01), f"Wien peak {peak}"


def test_fresnel_aluminum():
    f0 = H.fresnel_f0(0.27, 3.29)
    assert 0.88 <= f0 <= 0.94, f"Al F0 {f0}"
    f45 = H.fresnel_exact_conductor(0.7071, 0.27, 3.29)
    assert approx(f45, f0, 0.05), f"Fresnel @45 {f45}"


def test_visibility():
    assert approx(H.koschmieder_visibility(2.0e-6), 1956.0, 0.01)
    assert approx(H.koschmieder_visibility(2.0e-4), 19.56, 0.01)


def test_phase_functions_normalize():
    # Monte Carlo tolerance is loose
    I_rayleigh = H.sphere_angle_integrate(lambda c: H.rayleigh_phase(c), 200000)
    assert approx(I_rayleigh, 1.0, 0.02), f"Rayleigh integral {I_rayleigh}"
    I_hg = H.sphere_angle_integrate(lambda c: H.henyey_greenstein(c, 0.76), 200000)
    assert approx(I_hg, 1.0, 0.02), f"HG integral {I_hg}"


def test_ggx_energy_conservation():
    # Smooth white surface at normal incidence should reflect ~1
    e = H.ggx_energy_integral(1.0, 0.045, 32768)
    assert 0.9 <= e <= 1.1, f"GGX energy {e}"


if __name__ == "__main__":
    test_planck_300k_peak()
    test_wien_displacement()
    test_fresnel_aluminum()
    test_visibility()
    test_phase_functions_normalize()
    test_ggx_energy_conservation()
    print("All harness spot checks passed.")
