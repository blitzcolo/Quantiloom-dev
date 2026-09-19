#!/usr/bin/env python3
"""Render the camera's spectral input against independent physical answers.

This checker exercises SINGLE as a wavelength-resolved radiance source for a
detector, before ADC or ISP. It checks the 2000 nm hot-surface Planck tail,
measured n,k at 4/10 um, a flat open sky against both SINGLE and SWIR_FUSED,
visible excitation feeding a 600 nm measurement, emission outside its authored
span, and rejection of an NN atmosphere wavelength outside training coverage.

Run from either WSL or Windows after building the CLI and shaders. The script
uses repo-relative assets and writes temporary configs/outputs under
assets/configs; no tracked fixture is changed. Exit 0 pass, 1 wrong render,
2 no CLI, 3 no usable GPU. All output is ASCII for Windows CJK consoles.
"""

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = pathlib.Path(os.environ.get("CLI", str(ROOT / "build/src/app/Release/Quantiloom.exe")))
BASE = ROOT / "assets/configs/camera_physics_open.toml"
sys.path.insert(0, str(ROOT / "scripts/physics-audit"))
import harness as physics

MATERIAL = 'material_overrides."shadow_scene_open"'
GPU_ABSENT = re.compile(r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")
SUCCESS = "Saved spectral image"
RHO = 0.7
COS_SUN = 0.8


class RenderFailure(Exception):
    pass


class NoGpu(Exception):
    pass


def ascii_safe(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def set_key(doc, section, key, literal):
    """Set one key in our small TOML fixture without a second TOML serializer."""
    lines = doc.splitlines()
    header = f"[{section}]"
    try:
        start = lines.index(header)
    except ValueError:
        lines += ["", header]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("[")), len(lines))
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    for i in range(start + 1, end):
        if key_re.match(lines[i]):
            lines[i] = f"{key} = {literal}"
            break
    else:
        lines.insert(end, f"{key} = {literal}")
    return "\n".join(lines) + "\n"


def remove_key(doc, section, key):
    lines = doc.splitlines()
    header = f"[{section}]"
    start = lines.index(header)
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("[")), len(lines))
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    return "\n".join(line for i, line in enumerate(lines)
                     if not (start < i < end and key_re.match(line))) + "\n"


def case_config(name, work, *, wavelength=2000.0, band="SWIR", mode="single",
                temperature=300.0, emissivity=0.3, spp=32, solar=None,
                extra=None, gltf=None):
    doc = BASE.read_text(encoding="utf-8")
    changes = [
        ("renderer", "resolution", "[64, 64]"),
        ("renderer", "spp", str(spp)),
        ("renderer", "output", json.dumps(relative(work / f"{name}.exr"))),
        ("spectral", "mode", json.dumps(mode)),
        ("spectral", "wavelength_nm", str(wavelength)),
        ("spectral", "band", json.dumps(band)),
        ("scene", "default_temperature_k", str(temperature)),
        (MATERIAL, "ir_temperature_k", str(temperature)),
        (MATERIAL, "ir_emissivity", str(emissivity)),
    ]
    if gltf is not None:
        changes.append(("scene", "gltf", json.dumps(relative(gltf))))
    if solar is not None:
        changes += [
            ("lighting", "sun_radiance", "[5.0, 5.0, 5.0]"),
            ("lighting", "solar_lut", json.dumps(solar)),
            ("lighting", "solar_lut_columns", "[4, 3]"),
            ("lighting", "solar_lut_diffuse_is_global", "true"),
        ]
    changes += extra or []
    for section, key, literal in changes:
        doc = set_key(doc, section, key, literal)
    return doc


def run(name, doc, work, *, expect_failure=None):
    cfg = work / f"{name}.toml"
    cfg.write_text(doc, encoding="utf-8")
    proc = subprocess.run([str(CLI), relative(cfg)], cwd=ROOT,
                          capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if GPU_ABSENT.search(log):
        raise NoGpu("camera physics: no usable GPU, nothing measured")
    if expect_failure is not None:
        if SUCCESS in log or not re.search(expect_failure, log, re.IGNORECASE):
            raise RenderFailure(f"{name}: expected coverage error, got: " +
                                " | ".join(log.splitlines()[-8:]))
        print(f"{name}: rejected with expected coverage diagnostic")
        return None
    output = work / f"{name}.exr"
    if log.count(SUCCESS) != 1 or proc.returncode != 0 or not output.is_file():
        raise RenderFailure(f"{name}: render failed: " +
                            " | ".join(log.splitlines()[-8:]))
    with OpenEXR.File(str(output)) as exr:
        channels = exr.channels()
        pixels = channels[next(iter(channels))].pixels
        if pixels.ndim == 3:
            pixels = pixels[..., 0]
        pixels = pixels.astype(np.float64)
    if not np.all(np.isfinite(pixels)):
        raise RenderFailure(f"{name}: EXR contains nonfinite pixels")
    return pixels


def roi_mean(img):
    h, w = img.shape
    roi = img[h//4:3*h//4, w//4:3*w//4]
    return float(roi.mean())


def near(label, observed, expected, rel_tol, abs_tol=0.0):
    error = abs(observed - expected)
    bound = max(abs_tol, abs(expected) * rel_tol)
    print(f"{label}: observed={observed:.7e} expected={expected:.7e} "
          f"error={error:.3e} bound={bound:.3e}")
    if error > bound:
        raise RenderFailure(f"{label}: outside independent reference")


def read_pairs(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return rows


def read_solar(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        try:
            rows.append((float(parts[0]), float(parts[2]), float(parts[3])))
        except (ValueError, IndexError):
            continue
    return rows


def interp(rows, wavelength, column=1):
    for a, b in zip(rows, rows[1:]):
        if a[0] <= wavelength <= b[0]:
            t = (wavelength - a[0]) / (b[0] - a[0])
            return a[column] + t * (b[column] - a[column])
    return 0.0


def trapezoid(rows):
    return sum((b[0]-a[0]) * (a[1]+b[1])/2 for a, b in zip(rows, rows[1:]))


def excitation_integral(excitation, solar):
    """Independent ex(lambda)*[direct*cos+diffuse] integral, W/m^2."""
    low, high = excitation[0][0], excitation[-1][0]
    knots = sorted({x for x, _ in excitation} |
                   {row[0] for row in solar if low < row[0] < high})
    def signal(wavelength):
        direct = interp(solar, wavelength, 2)
        global_ = interp(solar, wavelength, 1)
        return interp(excitation, wavelength) * (direct * COS_SUN + global_ - direct)
    return sum((b-a) * (signal(a)+4*signal((a+b)/2)+signal(b))/6
               for a, b in zip(knots, knots[1:]))


def uploaded_solar_grid(rows, samples=64):
    """Independent model of SolarSpectralLUT::FromCPU's documented grid.

    The device keeps exact excitation/emission knots, but the scene illuminant
    is a 64-point uniform GPU LUT. The off-diagonal reference must integrate
    what was actually supplied to the renderer rather than the higher-detail
    source CSV that the GPU cannot see.
    """
    low, high = rows[0][0], rows[-1][0]
    return [(wavelength := low + (high-low)*index/(samples-1),
             interp(rows, wavelength, 1), interp(rows, wavelength, 2))
            for index in range(samples)]


def copy_open_scene(work, emissive, strip_ir=False):
    source = ROOT / "assets/models/shadow_scene"
    for suffix in (".bin", "_emissivity.csv"):
        shutil.copy2(source / f"shadow_scene_open{suffix}",
                     work / f"shadow_scene_open{suffix}")
    if emissive > 0:
        (work / "shadow_scene_open_emissivity.csv").write_text(
            "# wavelength_nm,emissivity\n300,0\n15000,0\n", encoding="ascii")
    scene = json.loads((source / "shadow_scene_open.gltf").read_text(encoding="utf-8"))
    scene["materials"][0]["emissiveFactor"] = [emissive] * 3
    if strip_ir:
        scene["materials"][0].get("extensions", {}).pop("QUANTILOOM_material_ir", None)
    target = work / "shadow_scene_open.gltf"
    target.write_text(json.dumps(scene), encoding="utf-8")
    return target


def check_hot_swir(work):
    cfg = case_config("hot_swir", work, temperature=1200, emissivity=1.0, spp=16)
    got = roi_mean(run("hot_swir", cfg, work))
    near("SINGLE 2000nm hot surface", got, physics.planck_blackbody(1200, 2000), 0.01)


def check_nk(work):
    # A synthetic measured n,k constant across both bands. R0=(1^2+1^2)/
    # (3^2+1^2)=0.2, so Kirchhoff gives emissivity 0.8. Without n,k the
    # material's dielectric PBR fallback gives 0.95: ignoring n,k cannot pass.
    nk = work / "constant_nk.yml"
    nk.write_text("DATA:\n  - type: tabulated nk\n    data: |\n"
                  "        3.0 2.0 1.0\n        12.0 2.0 1.0\n", encoding="ascii")
    bare_gltf = copy_open_scene(work, emissive=0.0, strip_ir=True)
    for name, wavelength, band, temperature, sky in (
        ("nk_mwir", 4000.0, "MWIR", 600.0, 300.0),
        ("nk_lwir", 10000.0, "LWIR", 400.0, 250.0),
    ):
        extra = [
            ("lighting", "atmosphere_temperature_k", str(sky)),
            ("refractive_index", '"shadow_scene_open"', json.dumps(relative(nk))),
        ]
        cfg = case_config(name, work, wavelength=wavelength, band=band,
                          temperature=temperature, emissivity=0.95, spp=64,
                          gltf=bare_gltf, extra=extra)
        # Either authored IR scalar key synthesizes a reflectance curve in
        # ConfigResolve, even when its value is zero. Remove both so measured
        # n,k is the highest-priority source and the PBR fallback stays 0.95.
        cfg = remove_key(cfg, MATERIAL, "ir_emissivity")
        cfg = remove_key(cfg, MATERIAL, "ir_transmittance")
        got = roi_mean(run(name, cfg, work))
        expected = (0.8 * physics.planck_blackbody(temperature, wavelength) +
                    0.2 * physics.planck_blackbody(sky, wavelength))
        near(f"SINGLE {int(wavelength)}nm measured n,k", got, expected, 0.025)


def check_open_sky(work):
    reflectance = work / "flat_reflectance.csv"
    reflectance.write_text("# wavelength_nm,reflectance\n300,0.7\n15000,0.7\n",
                           encoding="ascii")
    extra = [("spectral_curves", '"shadow_scene_open"',
              json.dumps(relative(reflectance)))]
    values = []
    for i in range(16):
        wavelength = 1400 + i * 1000 / 15
        name = f"open_swir_{i:02d}"
        cfg = case_config(name, work, wavelength=wavelength, temperature=300,
                          emissivity=0.3, spp=8,
                          solar="assets/luts/flat_sun_sky.csv", extra=extra)
        value = roi_mean(run(name, cfg, work))
        expected = RHO * (COS_SUN + 0.5) / math.pi + 0.3 * physics.planck_blackbody(300, wavelength)
        near(f"SINGLE open sky {i:02d}", value, expected, 0.012)
        values.append(value)
    cfg = case_config("open_swir_fused", work, mode="swir_fused", temperature=300,
                      emissivity=0.3, spp=8,
                      solar="assets/luts/flat_sun_sky.csv", extra=extra)
    fused = roi_mean(run("open_swir_fused", cfg, work))
    reintegrated = (sum(values) - (values[0] + values[-1]) / 2) / 15
    near("SWIR fused vs 16 SINGLE wavelengths", fused, reintegrated, 0.012)


def check_fluorescence(work):
    excitation = read_pairs(ROOT / "assets/data/fluorescence/demo_dye_excitation.csv")
    emission = read_pairs(ROOT / "assets/data/fluorescence/demo_dye_emission.csv")
    em_at_600 = interp(emission, 600) / trapezoid(emission)
    increments = {}
    for label, lut in (("flat", "flat_sun_sky.csv"),
                       ("blue", "blue_shifted_sun_sky.csv")):
        solar = f"assets/luts/{lut}"
        for yield_ in (0.0, 0.6):
            name = f"fluor_{label}_{int(yield_*10)}"
            extra = [
                (MATERIAL, "fluorescence_excitation_curve",
                 '"assets/data/fluorescence/demo_dye_excitation.csv"'),
                (MATERIAL, "fluorescence_emission_curve",
                 '"assets/data/fluorescence/demo_dye_emission.csv"'),
                (MATERIAL, "fluorescence_yield", str(yield_)),
            ]
            cfg = case_config(name, work, wavelength=600, band="VIS",
                              temperature=300, emissivity=0.3, spp=512,
                              solar=solar, extra=extra)
            increments[(label, yield_)] = roi_mean(run(name, cfg, work))
        observed = increments[(label, 0.6)] - increments[(label, 0.0)]
        source = read_solar(ROOT / solar)
        expected = 0.6 * em_at_600 * excitation_integral(excitation, source) / math.pi
        near(f"600nm fluorescence {label}", observed, expected, 0.06)
        increments[label] = observed
    expected_ratio = (excitation_integral(excitation, uploaded_solar_grid(
        read_solar(ROOT / "assets/luts/blue_shifted_sun_sky.csv"))) /
        excitation_integral(excitation, uploaded_solar_grid(
        read_solar(ROOT / "assets/luts/flat_sun_sky.csv"))))
    near("off-diagonal excitation ratio", increments["blue"] / increments["flat"],
         expected_ratio, 0.04)


def check_lamp_domain(work):
    gltf = copy_open_scene(work, emissive=5.0)
    curve = work / "visible_lamp.csv"
    curve.write_text("# wavelength_nm,radiance\n400,2\n780,2\n", encoding="ascii")
    extra = [
        (MATERIAL, "emissive_curve", json.dumps(relative(curve))),
        (MATERIAL, "emissive_scale", '"absolute"'),
    ]
    for name, wavelength, band in (("lamp_inside", 600, "VIS"),
                                   ("lamp_outside", 2000, "SWIR")):
        cfg = case_config(name, work, wavelength=wavelength, band=band,
                          temperature=300, emissivity=0.0, spp=16,
                          gltf=gltf, extra=extra)
        value = roi_mean(run(name, cfg, work))
        if wavelength == 600:
            near("bound lamp inside authored span", value, 2.0, 0.015)
        else:
            near("bound lamp outside authored span", value, 0.0, 0.0, 1e-6)

    # The scene's VIS preview band ends at 780 nm, but a device can ask for a
    # wavelength beyond it. The source curve really extends to 900 nm; using
    # only its VIS-resampled GPU slot would make this valid 850 nm measurement
    # dark. The exact source-knot chain must retain the original span.
    extended = work / "extended_lamp.csv"
    extended.write_text("# wavelength_nm,radiance\n400,2\n900,2\n", encoding="ascii")
    extended_extra = [
        (MATERIAL, "emissive_curve", json.dumps(relative(extended))),
        (MATERIAL, "emissive_scale", '"absolute"'),
    ]
    cfg = case_config("lamp_beyond_preview", work, wavelength=850, band="VIS",
                      temperature=300, emissivity=0.0, spp=16,
                      gltf=gltf, extra=extended_extra)
    near("lamp 850nm beyond VIS preview band",
         roi_mean(run("lamp_beyond_preview", cfg, work)), 2.0, 0.015)


def check_material_capture(work):
    """One capture must evaluate sloped emissivity and transmission per lambda."""
    gltf = copy_open_scene(work, emissive=0.0)
    (work / "shadow_scene_open_emissivity.csv").write_text(
        "# wavelength_nm,emissivity\n"
        "3000,0.2\n4000,0.2\n10000,0.6\n12000,0.6\n", encoding="ascii")
    (work / "shadow_scene_open_transmittance.csv").write_text(
        "# wavelength_nm,transmittance\n"
        "3000,0.1\n4000,0.1\n10000,0.3\n12000,0.3\n", encoding="ascii")
    scene = json.loads(gltf.read_text(encoding="utf-8"))
    scene["materials"][0]["extensions"]["QUANTILOOM_material_ir"][
        "transmittanceCurve"] = "shadow_scene_open_transmittance.csv"
    gltf.write_text(json.dumps(scene), encoding="utf-8")
    name = "two_band_emissivity"
    extra = [
        # The scene's 48-pixel preview is deliberately not the 64-pixel
        # physical array. Capture must allocate/trace the latter separately.
        ("renderer", "resolution", "[48, 48]"),
        ("camera", "projection", '"perspective"'),
        ("camera", "fov_y", "0.3667"),
        ("lighting", "atmosphere_temperature_k", "250.0"),
        ("sensor", "version", "1"),
        ("sensor", "enabled", "true"),
        ("sensor", "detector", '"photon"'),
        ("sensor", "cfa", '"multi_channel"'),
        ("sensor.optics", "f_number", "2.0"),
        ("sensor.optics", "pixel_pitch_um", "5.0"),
        ("sensor.optics", "fill_factor", "1.0"),
        ("sensor.optics", "sensor_width_px", "64"),
        ("sensor.optics", "sensor_height_px", "64"),
        ("sensor.exposure", "time_s", "0.01"),
        ("sensor.readout", "adc_bits", "12"),
        ("sensor.readout", "output_bits", "12"),
        ("sensor.quality", "wavelength_samples", "8"),
        ("sensor.products", "band_measurement", "true"),
        ("sensor.products", "display", "true"),
    ]
    cfg = case_config(name, work, wavelength=4000, band="MWIR", temperature=500,
                      emissivity=0.95, spp=8, gltf=gltf,
                      solar="assets/luts/flat_sun_sky.csv", extra=extra)
    # Scalar overrides synthesize a flat reflectance curve and would mask the
    # glTF's sloped emissivity. Keep only the temperature override.
    cfg = remove_key(cfg, MATERIAL, "ir_emissivity")
    cfg = remove_key(cfg, MATERIAL, "ir_transmittance")
    for label, centre in (("At4um", 4000), ("At10um", 10000)):
        cfg += (f'\n[[sensor.channels]]\nname = "{label}"\n'
                '[sensor.channels.qe]\nkind = "absolute_qe"\n'
                f'wavelength_nm = [{centre-25}, {centre}, {centre+25}]\n'
                'value = [0, 1, 0]\n')
    run(name, cfg, work)
    measured_path = work / f"{name}_measurement.exr"
    if not measured_path.is_file():
        raise RenderFailure("same-capture wavelength response has no measurement EXR")
    with OpenEXR.File(str(measured_path)) as exr:
        channels = exr.channels()
        if any(channel.pixels.shape != (64, 64) for channel in channels.values()):
            raise RenderFailure("measurement extent follows preview, not physical array")
        unit = exr.header().get("camera_unit")
        if unit not in ("e-/s", b"e-/s"):
            raise RenderFailure(f"measurement unit is {ascii_safe(unit)}, expected e-/s")
        for label, centre in (("At4um", 4000), ("At10um", 10000)):
            if label not in channels:
                raise RenderFailure(f"measurement EXR lacks {label} channel")
            measured = roi_mean(channels[label].pixels.astype(np.float64))
            qe = physics.CameraResponse(((centre-25, 0.0), (centre, 1.0),
                                         (centre+25, 0.0)))
            omega = physics.projected_pupil_solid_angle_sr(2.0)
            samples = []
            for wavelength in range(centre-25, centre+26):
                epsilon = 0.2 + (wavelength - 4000) * 0.4 / 6000
                transmission = 0.1 + (wavelength - 4000) * 0.2 / 6000
                rho = 1 - epsilon - transmission
                sky = physics.planck_blackbody(250, wavelength)
                radiance = (epsilon * physics.planck_blackbody(500, wavelength) +
                            (rho + transmission) * sky +
                            rho * COS_SUN / math.pi)
                samples.append((float(wavelength), radiance * omega))
            expected = physics.photon_electron_rate_detector_plane(
                samples, qe, (5e-6)**2)
            near(f"same capture {label} emissivity", measured, expected, 0.02)


def check_single_dispersion(work):
    """At fixed lambda, analytic and measured BK7 should bend the same ray."""
    original = json.loads((ROOT / "assets/models/prism_dispersion.gltf").read_text(
        encoding="utf-8"))
    images = {}
    for source in ("off", "analytic", "measured"):
        model = json.loads(json.dumps(original))
        if source != "analytic":
            for material in model["materials"]:
                material.get("extensions", {}).pop("KHR_materials_dispersion", None)
        gltf = work / f"prism_{source}.gltf"
        gltf.write_text(json.dumps(model), encoding="utf-8")
        for wavelength in (450, 650):
            name = f"prism_{source}_{wavelength}"
            cfg = (ROOT / "assets/configs/prism_dispersion.toml").read_text(
                encoding="utf-8")
            for section, key, literal in (
                ("renderer", "resolution", "[256, 256]"),
                ("renderer", "spp", "64"),
                ("renderer", "output", json.dumps(relative(work / f"{name}.exr"))),
                ("spectral", "mode", '"single"'),
                ("spectral", "wavelength_nm", str(wavelength)),
                ("spectral", "band", '"VIS"'),
                ("scene", "gltf", json.dumps(relative(gltf))),
                ("quality", "fail_on_srgb_upsample", "false"),
            ):
                cfg = set_key(cfg, section, key, literal)
            if source == "measured":
                cfg = set_key(cfg, "refractive_index", '"PrismGlass_BK7"',
                              '"assets/data/refractiveindex/BK7_sellmeier.yml"')
            images[source, wavelength] = run(name, cfg, work)
    for wavelength in (450, 650):
        off = images["off", wavelength]
        analytic = images["analytic", wavelength]
        measured = images["measured", wavelength]
        scale = max(float(np.abs(off).mean()), 1e-12)
        changed = float(np.abs(analytic - off).mean()) / scale
        agreement = float(np.abs(measured - analytic).mean()) / scale
        print(f"SINGLE n(lambda) {wavelength}nm: effect={changed:.3e} "
              f"analytic/measured={agreement:.3e}")
        if changed <= 1e-5 or agreement > 0.03:
            raise RenderFailure(f"SINGLE {wavelength}nm dispersion source disagreement")


def check_atmosphere_gap(work):
    extra = [("atmosphere", "preset", '"clear"')]
    cfg = case_config("atmosphere_gap", work, wavelength=850, band="VIS",
                      temperature=300, emissivity=0.3, extra=extra)
    run("atmosphere_gap", cfg, work,
        expect_failure=r"outside NN atmosphere coverage|atmosphere.*coverage")


def check_solar_coverage(work):
    """A precise device response may not extrapolate a shorter solar LUT."""
    source = work / "short_sun_sky.csv"
    source.write_text(
        "Wvlgth nm,Etr,Global tilt,Direct+circumsolar\n"
        "550,1,1.5,1\n600,1,1.5,1\n", encoding="ascii")
    name = "sensor_short_solar"
    cfg = case_config(name, work, wavelength=600, band="VIS", solar=relative(source))
    for section, key, literal in (
        ("camera", "projection", '"perspective"'),
        ("camera", "fov_y", "0.3667"),
        ("sensor", "version", "1"),
        ("sensor", "enabled", "true"),
        ("sensor", "detector", '"photon"'),
        ("sensor.optics", "pixel_pitch_um", "5"),
        ("sensor.optics", "sensor_width_px", "64"),
        ("sensor.optics", "sensor_height_px", "64"),
        ("sensor.exposure", "time_s", "0.01"),
        ("sensor.readout", "adc_bits", "12"),
        ("sensor.readout", "output_bits", "12"),
        ("sensor.products", "band_measurement", "true"),
    ):
        cfg = set_key(cfg, section, key, literal)
    cfg += ('\n[[sensor.channels]]\nname = "Mono"\n'
            '[sensor.channels.qe]\nkind = "absolute_qe"\n'
            'wavelength_nm = [550, 600, 650]\nvalue = [0, 0.8, 0]\n')
    path = work / f"{name}.toml"
    path.write_text(cfg, encoding="utf-8")
    proc = subprocess.run([str(CLI), relative(path)], cwd=ROOT,
                          capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if GPU_ABSENT.search(log):
        raise NoGpu("camera physics: no usable GPU, nothing measured")
    coverage_error = ("camera response exceeds solar/sky LUT coverage" in log or
                      "LUT does not cover device channel" in log)
    if not coverage_error or proc.returncode == 0:
        raise RenderFailure("short solar LUT was not rejected by precise capture: " +
                            " | ".join(log.splitlines()[-8:]))
    print("precise response outside solar/sky LUT: rejected")


def check_atmosphere_capture(work):
    """Two response bands in one acquisition must each get their own NN bake."""
    def capture_config(name, response_channels):
        extra = [
            ("camera", "projection", '"perspective"'),
            ("camera", "fov_y", "0.3667"),
            ("atmosphere", "preset", '"clear"'),
            ("lighting", "atmosphere_temperature_k", "250.0"),
            ("sensor", "version", "1"),
            ("sensor", "enabled", "true"),
            ("sensor", "detector", '"photon"'),
            ("sensor", "cfa", '"multi_channel"' if len(response_channels) > 1 else '"mono"'),
            ("sensor.optics", "f_number", "2.0"),
            ("sensor.optics", "pixel_pitch_um", "5.0"),
            ("sensor.optics", "sensor_width_px", "64"),
            ("sensor.optics", "sensor_height_px", "64"),
            ("sensor.exposure", "time_s", "0.01"),
            ("sensor.readout", "adc_bits", "12"),
            ("sensor.readout", "output_bits", "12"),
            ("sensor.quality", "wavelength_samples", "4"),
            ("sensor.products", "band_measurement", "true"),
            ("sensor.products", "display", "true"),
        ]
        cfg = case_config(name, work, wavelength=4000, band="MWIR",
                          temperature=300, emissivity=0.3, spp=8, extra=extra)
        for label, centre in response_channels:
            cfg += (f'\n[[sensor.channels]]\nname = "{label}"\n'
                    '[sensor.channels.qe]\nkind = "absolute_qe"\n'
                    f'wavelength_nm = [{centre-25}, {centre}, {centre+25}]\n'
                    'value = [0, 1, 0]\n')
        return cfg

    def measurement(name, cfg, label):
        run(name, cfg, work)
        path = work / f"{name}_measurement.exr"
        if not path.is_file():
            raise RenderFailure(f"{name}: no measurement EXR")
        with OpenEXR.File(str(path)) as exr:
            channels = exr.channels()
            if label not in channels:
                raise RenderFailure(f"{name}: no {label} measurement channel")
            return roi_mean(channels[label].pixels.astype(np.float64))

    response_channels = (("At4um", 4000), ("At45um", 4500))
    combined_name = "atmosphere_two_channels"
    combined_cfg = capture_config(combined_name, response_channels)
    run(combined_name, combined_cfg, work)
    combined_path = work / f"{combined_name}_measurement.exr"
    if not combined_path.is_file():
        raise RenderFailure("atmosphere combined capture has no measurement EXR")
    with OpenEXR.File(str(combined_path)) as exr:
        combined = {label: roi_mean(exr.channels()[label].pixels.astype(np.float64))
                    for label, _ in response_channels}
    for label, centre in response_channels:
        name = f"atmosphere_only_{centre}"
        separate = measurement(name, capture_config(name, ((label, centre),)),
                               "Measurement")
        near(f"atmosphere rebake {label}", combined[label], separate, 0.001)


def main():
    if not CLI.is_file():
        print(ascii_safe(f"no CLI at {CLI} -- build first"), file=sys.stderr)
        return 2
    if not BASE.is_file():
        print(ascii_safe(f"missing fixture {BASE}"), file=sys.stderr)
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="_camera_gate_",
                                         dir=ROOT / "assets/configs") as directory:
            work = pathlib.Path(directory)
            check_hot_swir(work)
            check_nk(work)
            check_open_sky(work)
            check_fluorescence(work)
            check_material_capture(work)
            check_lamp_domain(work)
            # Dispersion needs a structured colour target so the refracted
            # displacement is observable. The dedicated check_dispersion.py
            # gate owns that fixture and compares both n(lambda) sources.
            check_atmosphere_gap(work)
            check_solar_coverage(work)
            check_atmosphere_capture(work)
    except NoGpu as error:
        print(ascii_safe(error), file=sys.stderr)
        return 3
    except Exception as error:
        print(ascii_safe(f"FAIL: {error}"), file=sys.stderr)
        return 1
    print("camera physics checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
