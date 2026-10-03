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


def test_camera_reference():
    got = H.photon_electron_rate([(500, 2), (600, 2)],
                                 [(500, 0), (550, 1), (600, 0)], 1e-12)
    expected = 2 * 50 * 1e-12 * 550e-9 / (H.PLANCK_H * H.SPEED_OF_LIGHT_C)
    assert approx(got, expected, 1e-12)
    assert approx(H.thermal_detector_step(0, 1, 0.008, 0.008),
                  1 - math.exp(-1), 1e-12)


def _rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid camera input was accepted")


def test_camera_response_semantics_and_units():
    # A uniform spectrum and a uniform QE integrate exactly in wavelength.
    # Photon conversion has integral(lambda d lambda), not center-wavelength
    # times a guessed band width. The two units differ by lambda/(hc).
    a, b, irradiance, area, exposure = 500.0, 700.0, 2.0, 1e-12, 0.01
    samples = [(a, irradiance), (b, irradiance)]
    qe = H.CameraResponse(((a, 0.6), (b, 0.6)))
    expected_rate = (area * irradiance * 0.6 * 1e-9 * (b*b-a*a) /
                     (2 * H.PLANCK_H * H.SPEED_OF_LIGHT_C))
    rate = H.photon_electron_rate(samples, qe, area)
    assert approx(rate, expected_rate, 2e-14)
    assert approx(H.expected_photoelectrons(samples, qe, area, exposure),
                  exposure * expected_rate, 2e-14)
    assert approx(H.expected_photoelectrons(samples, qe, 2*area, 3*exposure),
                  6 * exposure * expected_rate, 2e-14)

    lens = H.CameraResponse(((a, 0.8), (b, 0.8)), "lens_transmission")
    filt = H.CameraResponse(((a, 0.5), (b, 0.5)), "filter_transmission")
    relative = H.CameraResponse(((a, 1.0), (b, 1.0)), "relative_qe",
                                "peak_one", 0.6, "calibrated peak QE")
    assert approx(H.photon_electron_rate(samples, relative, area,
                                         lens=lens, filters=(filt,), fill_factor=0.75),
                  expected_rate * 0.8 * 0.5 * 0.75, 2e-14)
    area_shape = H.CameraResponse(((a, 1/(b-a)), (b, 1/(b-a))),
                                  "relative_qe", "area_one", 0.6*(b-a),
                                  "integrated QE")
    assert approx(H.photon_electron_rate(samples, area_shape, area), expected_rate,
                  2e-14)

    # A system response already includes optical transmission; multiplication
    # of the filter or fill factor again would silently understate the signal.
    system = H.CameraResponse(((a, 0.24), (b, 0.24)), "system_photon_qe")
    assert approx(H.photon_electron_rate(samples, system, area),
                  expected_rate * 0.4, 2e-14)
    _rejects(lambda: H.photon_electron_rate(samples, system, area, lens=lens))
    _rejects(lambda: H.photon_electron_rate(samples, system, area, fill_factor=0.8))
    _rejects(lambda: H.absorbed_power(samples, system, area))

    absorber = H.CameraResponse(((a, 0.3), (b, 0.3)), "thermal_absorptance")
    expected_power = area * irradiance * (b-a) * 0.3
    assert approx(H.absorbed_power(samples, absorber, area), expected_power, 2e-14)
    assert approx(H.absorbed_power(samples, absorber, area, lens=lens,
                                   filters=(filt,)), expected_power*0.8*0.5, 2e-14)
    thermal_system = H.CameraResponse(((a, 0.12), (b, 0.12)),
                                      "system_thermal_absorptance")
    assert approx(H.absorbed_power(samples, thermal_system, area),
                  expected_power * 0.4, 2e-14)
    _rejects(lambda: H.photon_electron_rate(samples, thermal_system, area))
    _rejects(lambda: H.absorbed_power(samples, thermal_system, area, filters=(filt,)))
    _rejects(lambda: H.absorbed_power(samples, qe, area))
    _rejects(lambda: H.photon_electron_rate(samples, absorber, area))
    _rejects(lambda: H.photon_electron_rate(samples, qe, area, lens=filt))
    _rejects(lambda: H.photon_electron_rate(samples, qe, area, filters=(lens,)))

    # The two legal unit boundaries agree when the caller has explicitly
    # applied optical transmission to the detector-plane irradiance.
    detector_plane = [(a, irradiance*0.8*0.5), (b, irradiance*0.8*0.5)]
    assert approx(H.photon_electron_rate_detector_plane(detector_plane, qe, area),
                  H.photon_electron_rate(samples, qe, area, lens=lens,
                                         filters=(filt,)), 2e-14)
    assert approx(H.absorbed_power_detector_plane(detector_plane, absorber, area),
                  H.absorbed_power(samples, absorber, area, lens=lens,
                                   filters=(filt,)), 2e-14)
    _rejects(lambda: H.photon_electron_rate_detector_plane(detector_plane,
                                                           system, area))
    _rejects(lambda: H.absorbed_power_detector_plane(detector_plane,
                                                     thermal_system, area))


def test_camera_rejects_incomplete_or_invalid_spectra():
    qe = H.CameraResponse(((500, 1), (600, 1)))
    _rejects(lambda: H.photon_electron_rate([(520, 1), (600, 1)], qe, 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (580, 1)], qe, 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (500, 1)], qe, 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, math.nan), (600, 1)], qe, 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, -1), (600, 1)], qe, 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (600, 1)], qe, 0))
    _rejects(lambda: H.expected_photoelectrons([(500, 1), (600, 1)], qe,
                                               1e-12, -0.1))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (600, 1)],
        H.CameraResponse(((500, 1.1), (600, 1))), 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (600, 1)],
        H.CameraResponse(((500, 1), (600, 1)), "relative_qe"), 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (600, 1)],
        H.CameraResponse(((500, 0.8), (600, 0.9)), "relative_qe",
                         "peak_one", 0.8, "source"), 1e-12))
    _rejects(lambda: H.photon_electron_rate([(500, 1), (600, 1)], qe,
        1e-12, lens=H.CameraResponse(((520, 1), (600, 1)), "lens_transmission")))


def test_camera_blackbody_derivative_and_thermal_step():
    T, nm = 300.0, 10000.0
    # A known 300 K, 10 um Planck value and independent centered finite
    # difference of the blackbody formula guard absolute scale and d/dT.
    assert approx(H.planck_blackbody(T, nm), 0.009924, 0.0002)
    delta = 0.01
    finite_difference = (H.planck_blackbody(T+delta, nm) -
                         H.planck_blackbody(T-delta, nm))/(2*delta)
    assert approx(H.planck_blackbody_temperature_derivative(T, nm),
                  finite_difference, 1e-8)
    response = H.CameraResponse(((9999, 0.5), (10001, 0.5)),
                                "thermal_absorptance")
    area, omega = 1e-10, 0.1
    power, derivative = H.blackbody_power_and_derivative(T, response, area, omega)
    narrow_band_power = H.planck_blackbody(T, nm) * 2 * 0.5 * area * omega
    assert approx(power, narrow_band_power, 1e-8)
    p_plus = H.blackbody_power_and_derivative(T+delta, response, area, omega)[0]
    p_minus = H.blackbody_power_and_derivative(T-delta, response, area, omega)[0]
    assert approx(derivative, (p_plus-p_minus)/(2*delta), 1e-8)
    assert derivative > 0

    photon_response = H.CameraResponse(((9999, 0.5), (10001, 0.5)))
    electron_rate, electron_derivative = H.blackbody_photon_rate_and_derivative(
        T, photon_response, area, omega)
    assert approx(electron_rate, power * nm * 1e-9 /
                  (H.PLANCK_H * H.SPEED_OF_LIGHT_C), 1e-8)
    assert electron_derivative > 0

    # Exact first-order step and semigroup property for constant absorbed power.
    tau = 0.008
    first = H.thermal_detector_step(0.0, 1.0, tau, tau)
    twice = H.thermal_detector_step(first, 1.0, tau, tau)
    assert approx(twice, 1-math.exp(-2), 1e-14)
    assert H.thermal_detector_step(1.0, 0.0, 0.0, tau) == 1.0
    assert H.thermal_detector_step(1.0, 0.0, tau, 0.0) == 0.0
    _rejects(lambda: H.thermal_detector_step(0, 1, -1, tau))
    _rejects(lambda: H.thermal_detector_step(0, 1, 1, math.nan))


def test_camera_physical_fov_and_airy_normalization():
    # 36 mm full-frame sensor at 50 mm focal length; preview dimensions are
    # absent from the formula and therefore cannot change collecting area.
    assert approx(H.physical_horizontal_fov_deg(3600, 10, 50),
                  39.597752709, 1e-10)
    _rejects(lambda: H.physical_horizontal_fov_deg(0, 10, 50))
    assert approx(H.projected_pupil_solid_angle_sr(2.0), math.pi/17, 1e-14)
    incident = ((500.0, 2.0), (700.0, 2.0))
    pre_optics = H.radiance_to_preoptics_irradiance(incident, 2.0)
    assert pre_optics == ((500.0, 2*math.pi/17), (700.0, 2*math.pi/17))
    off_axis = H.radiance_to_preoptics_irradiance(
        incident, 2.0, math.pi/3, cos4_vignetting=True)
    assert approx(off_axis[0][1], pre_optics[0][1]/16, 1e-14)
    _rejects(lambda: H.radiance_to_preoptics_irradiance(incident, 2.0,
                                                       math.pi/2))
    wavelength, f_number = 550.0, 2.0
    first_dark_radius = 1.2196698912665045 * wavelength * 1e-9 * f_number
    enclosed = H.airy_encircled_energy(first_dark_radius, wavelength, f_number)
    assert approx(enclosed, 0.837785, 2e-6)
    k = math.pi/(wavelength*1e-9*f_number)
    assert approx(H.airy_psf_density_per_m2(0, wavelength, f_number),
                  k*k/(4*math.pi), 1e-14)
    # Independently integrate 2*pi*r*PSF(r) through the first dark ring.
    steps = 1000
    dr = first_dark_radius/steps
    radial = 0.0
    for i in range(steps+1):
        r = i*dr
        term = 2*math.pi*r*H.airy_psf_density_per_m2(r, wavelength, f_number)
        radial += (1 if i in (0, steps) else (4 if i % 2 else 2))*term
    assert approx(radial*dr/3, enclosed, 1e-9)


if __name__ == "__main__":
    for n, sigma, thickness in ((1.0, 0.0, .2), (1.5, 0.0, .2), (1.5, 10.0, .2)):
        r,t,a=H.homogeneous_slab_coefficients(n,sigma,thickness)
        assert abs(r+t+a-1)<1e-12 and min(r,t,a)>-1e-12
    assert approx(H.homogeneous_slab_coefficients(1.5,0,.2)[1],12/13,1e-12)
    test_planck_300k_peak()
    test_wien_displacement()
    test_fresnel_aluminum()
    test_visibility()
    test_phase_functions_normalize()
    test_ggx_energy_conservation()
    test_camera_reference()
    test_camera_response_semantics_and_units()
    test_camera_rejects_incomplete_or_invalid_spectra()
    test_camera_blackbody_derivative_and_thermal_step()
    test_camera_physical_fov_and_airy_normalization()
    print("All harness spot checks passed.")
