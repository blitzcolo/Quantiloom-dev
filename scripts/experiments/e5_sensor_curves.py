#!/usr/bin/env python3
"""E5 -- the sensor-facing curves: round-trip, NETD, and surviving FPN.

Three assertions the manuscript makes about the chain from a prescribed
temperature to a reported one, each of which existed as a unit test and none of
which had a curve.

**(a) Round-trip.** An isothermal cavity renders its own blackbody; inverting
that render through the same band quadrature must return the temperature it was
given. The sensor is OFF here, deliberately: this measures the inversion
against the forward integral and nothing else, so a residual is quadrature
rather than noise. Swept over cavity temperature and over both bands, because
a per-band error that varies with band is an interpolation bug and one that
varies with temperature is a quadrature bug, and the two look nothing alike.

**(b) NETD, two ways.** The analytic figure comes from the definition in
Section VI, sigma_L / (dL/dT), computed from the sensor's own responsivity. The
empirical figure comes from actually drawing frames: one detector, many
exposures, the per-pixel standard deviation of raw DN over time, divided by the
dDN/dT the same sensor gives across the temperature grid. They are independent
paths to the same number, and the assertion that the responsivity in the NETD
formula matches the one in the electron conversion is exactly what their
agreement tests.

**(c) Injected FPN, and what survives correction.** PRNU is multiplicative and
DSNU additive, and the manuscript's claim is that modelling them separately is
what lets both be calibrated. Inject known amounts, run non-uniformity
correction at a stated efficiency, average enough frames to drive the temporal
noise below the pattern, and measure what is left. A bright field carries both;
a dark field carries only the additive one. The residual should be
(1 - efficiency) times what went in.

Frame averaging is why this cannot be done by rendering N times: each render
builds a fresh sensor, and a fresh sensor with a fresh seed draws fresh FPN
maps, so the pattern would move with the noise. `sensor_lab` applies one
detector many times.

Usage:
    e5_sensor_curves.py --roundtrip --out-dir evidence/e5
    e5_sensor_curves.py --netd --out-dir evidence/e5
    e5_sensor_curves.py --fpn --out-dir evidence/e5
"""

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runlog import REPO, Run  # noqa: E402

CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
SENSOR_LAB = REPO / "build" / "src" / "tools" / "Release" / "sensor_lab.exe"
CONFIGS = REPO / "assets" / "configs"
WORK = pathlib.Path(r"H:\quantiloom-paper\evidence\e5")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(WORK)

TEMPERATURES = list(range(250, 351, 10))

# The eight cavities of the furnace gate, so the round-trip is measured on the
# same configurations Table IV reports.
CAVITIES = [("lwir", c) for c in ("e1", "e05", "rho1", "spectral", "specular")] + \
           [("mwir", c) for c in ("e1", "e05", "rho1")]

# The cavity's temperature lives on the MATERIAL, not on the scene: the furnace
# glTFs carry temperature_K inside their QUANTILOOM_material_ir extension, and
# scene.default_temperature_k is only the fallback for materials that have none.
# Sweeping the scene key alone leaves every cavity at 300 K and produces a
# perfectly straight line of pure error, which is how this was found.
CAVITY_MATERIAL = {"e1": "furnace_e1", "e05": "furnace_e05", "rho1": "furnace_rho1",
                   "spectral": "furnace_spectral",
                   "specular": "furnace_spectral_smooth"}

# A calibrated LWIR sensor. Every value is stated because a NETD without its
# configuration is not a number: it is the sensitivity that this integration
# time, this aperture and this well would buy.
LWIR_SENSOR = {
    "sensor.enabled": True,
    "sensor.focal_length_mm": 8.675,
    "sensor.f_number": 2.0,
    "sensor.pixel_pitch_um": 12.0,
    "sensor.quantum_efficiency": 0.8,
    "sensor.well_capacity_e": 2.0e7,
    "sensor.read_noise_e_rms": 5000.0,
    "sensor.dark_current_e_s": 1.0e9,
    "sensor.integration_time_s": 2.0e-4,
    "sensor.bit_depth": 16,
    "sensor.gain": 305.18,
    "sensor.psf_sigma_px": 0.0,       # no blur: a uniform field has nothing to blur
    "sensor.detector_temperature_k": 77.0,
    "sensor.enable_poisson_noise": True,
    "sensor.enable_read_noise": True,
    "sensor.enable_dark_current": True,
    "sensor.enable_fpn": False,       # NETD excludes FPN by definition
    "sensor.noise_seed": 0x548C,
}


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "/") + '"'


def render(config, overrides, output, work):
    """One render through the batch path, which is the only one that takes
    per-job dotted overrides."""
    tokens = " ".join(f"{k}={toml_value(v)}" for k, v in
                      {**overrides, "renderer.output": output.as_posix()}.items())
    manifest = work / "_batch.txt"
    manifest.write_text(f"{config.as_posix()} | {tokens}\n", encoding="utf-8")
    result = subprocess.run([str(CLI), "batch", str(manifest)], cwd=str(REPO),
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=7200)
    if result.returncode != 0:
        print(result.stdout[-2500:], file=sys.stderr)
        raise SystemExit(f"render failed: {config.name}")
    return result.stdout


def read_exr_median(path):
    import numpy as np
    import OpenEXR
    pixels = list(OpenEXR.File(str(path)).channels().values())[0].pixels
    array = np.asarray(pixels, dtype=np.float64)
    if array.ndim == 3:
        array = array[:, :, 0]
    return float(np.median(array)), float(np.mean(array)), float(np.std(array))


# ---------------------------------------------------------------------------
# (a) Round-trip
# ---------------------------------------------------------------------------

def roundtrip(args):
    work = WORK / "roundtrip"
    work.mkdir(parents=True, exist_ok=True)
    rows = []

    for band, case in CAVITIES:
        config = CONFIGS / f"furnace_{band}_{case}.toml"
        for temperature in TEMPERATURES:
            output = work / f"{band}_{case}_{temperature}.exr"
            material = CAVITY_MATERIAL[case]
            render(config, {
                "scene.default_temperature_k": float(temperature),
                f"material_overrides.{material}.ir_temperature_k": float(temperature),
                # The ENCLOSURE is what has to be isothermal, not just the wall.
                # Every band keeps an analytic ambient term at this temperature
                # and traces only the residual against it, so leaving it at 300 K
                # while moving the wall builds a cavity with two temperatures in
                # it. The rho1 cavity makes that unmistakable -- its walls have
                # zero emissivity, so its entire radiance is this term reflected,
                # and it reported 300 K at every wall temperature asked for.
                "lighting.atmosphere_temperature_k": float(temperature),
                # Invert as a blackbody. The cavity radiates as one whatever its
                # walls are made of -- that is the invariant being round-tripped
                # -- so assuming an emissivity here would be assuming the answer.
                "thermography.enabled": True,
                "thermography.emissivity": 1.0,
                "thermography.reflected_temperature_k": 0.0,
                "thermography.atmosphere_transmittance": 1.0,
                "sensor.enabled": False,
            }, output, work)

            tapp = output.with_name(output.stem + "_tapp.exr")
            if not tapp.is_file():
                raise SystemExit(f"no apparent-temperature map for {output.name}")
            median, mean, spread = read_exr_median(tapp)
            rows.append({"band": band.upper(), "cavity": case,
                         "T_set_K": float(temperature),
                         "T_recovered_K": median, "T_recovered_mean_K": mean,
                         "spatial_sigma_K": spread,
                         "error_K": median - temperature})
            print(f"  {band}_{case:<9} {temperature} K -> {median:8.4f} K  "
                  f"({median - temperature:+.4f})", flush=True)

    worst = max(abs(r["error_K"]) for r in rows)
    (WORK / "roundtrip.json").write_text(json.dumps(
        {"temperatures_K": TEMPERATURES, "worst_abs_error_K": worst,
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nworst |T_recovered - T_set| = {worst:.4f} K over "
          f"{len(rows)} configurations")


# ---------------------------------------------------------------------------
# (b) NETD, analytic against empirical
# ---------------------------------------------------------------------------

def sensor_toml(path, extra=None):
    """A [sensor] block for sensor_lab, from the same dictionary the renders use."""
    settings = dict(LWIR_SENSOR)
    settings.update(extra or {})
    lines = ["# Auto-generated by e5_sensor_curves.py.", "[spectral]",
             'mode = "lwir_fused"', "wavelength_nm = 10000.0", "", "[sensor]"]
    for key, value in settings.items():
        name = key.split("sensor.", 1)[1]
        lines.append(f"{name} = {toml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_sensor_lab(config, radiance, frames, prefix, uniform=None, size="128x128"):
    command = [str(SENSOR_LAB), str(config), "--frames", str(frames),
               "--out-prefix", str(prefix)]
    command += (["--uniform", repr(uniform), "--size", size] if radiance is None
                else ["--input", str(radiance)])
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=7200)
    if result.returncode != 0:
        print(result.stdout + result.stderr, file=sys.stderr)
        raise SystemExit("sensor_lab failed")
    return json.loads(pathlib.Path(f"{prefix}_stats.json").read_text(encoding="utf-8"))


NETD_LOG = re.compile(r"NETD at ([\d.]+) K: ([\d.]+) mK")


def netd(args):
    """Both NETD paths on one temperature grid.

    The analytic figure is what the renderer reports from the definition; the
    empirical one is measured by drawing frames and dividing the temporal
    spread of raw DN by the dDN/dT the same sensor produces. Nothing links them
    except the responsivity, which is the point: Section VI asserts by unit test
    that the responsivity inside the NETD formula is the one used to convert
    radiance to electrons, and two independent paths agreeing is that assertion
    measured rather than asserted.
    """
    work = WORK / "netd"
    work.mkdir(parents=True, exist_ok=True)
    config = CONFIGS / "furnace_lwir_e1.toml"
    lab_config = sensor_toml(work / "sensor_lwir.toml")

    rows = []
    for temperature in TEMPERATURES:
        isothermal = {
            "scene.default_temperature_k": float(temperature),
            "material_overrides.furnace_e1.ir_temperature_k": float(temperature),
            "lighting.atmosphere_temperature_k": float(temperature),
        }

        # Pass 1: the renderer's own analytic NETD, which needs the sensor on.
        log = render(config, {**isothermal, **LWIR_SENSOR,
                              "thermography.enabled": True,
                              "thermography.emissivity": 1.0,
                              "thermography.report_netd": True},
                     work / f"withsensor_{temperature}.exr", work)
        match = NETD_LOG.search(log)
        analytic_mk = float(match.group(2)) if match else None

        # Pass 2: the clean radiance field, for the frame draw.
        radiance = work / f"radiance_{temperature}.exr"
        render(config, {**isothermal, "sensor.enabled": False}, radiance, work)
        stats = run_sensor_lab(lab_config, radiance, args.frames,
                               work / f"lab_{temperature}")

        rows.append({"T_K": float(temperature),
                     "netd_analytic_mK": analytic_mk,
                     "mean_dn": stats["mean_dn"],
                     "temporal_sigma_dn": stats["temporal_sigma_dn_median"]})
        print(f"  {temperature} K   analytic "
              f"{'n/a' if analytic_mk is None else f'{analytic_mk:7.1f} mK'}   "
              f"mean {stats['mean_dn']:9.1f} DN   "
              f"sigma {stats['temporal_sigma_dn_median']:7.2f} DN", flush=True)

    # dDN/dT by central difference on the same grid, so the empirical NETD uses
    # this sensor's own responsivity rather than an assumed one.
    for index, row in enumerate(rows):
        lo = rows[max(0, index - 1)]
        hi = rows[min(len(rows) - 1, index + 1)]
        span = hi["T_K"] - lo["T_K"]
        row["dDN_dT"] = (hi["mean_dn"] - lo["mean_dn"]) / span if span else None
        row["netd_empirical_mK"] = (
            1000.0 * row["temporal_sigma_dn"] / row["dDN_dT"]
            if row["dDN_dT"] else None)

    comparable = [r for r in rows
                  if r["netd_analytic_mK"] and r["netd_empirical_mK"]]
    ratios = [r["netd_empirical_mK"] / r["netd_analytic_mK"] for r in comparable]
    (WORK / "netd.json").write_text(json.dumps(
        {"frames": args.frames, "sensor": LWIR_SENSOR, "rows": rows,
         "empirical_over_analytic": {"min": min(ratios) if ratios else None,
                                     "max": max(ratios) if ratios else None}},
        indent=2), encoding="utf-8")

    print()
    for row in rows:
        if row["netd_empirical_mK"]:
            print(f"  {row['T_K']:.0f} K   analytic {row['netd_analytic_mK']:7.1f}   "
                  f"empirical {row['netd_empirical_mK']:7.1f} mK   "
                  f"ratio {row['netd_empirical_mK'] / row['netd_analytic_mK']:.3f}")
    if ratios:
        print(f"\nempirical / analytic in [{min(ratios):.3f}, {max(ratios):.3f}]")


# ---------------------------------------------------------------------------
# (c) Injected fixed-pattern noise, and what survives correction
# ---------------------------------------------------------------------------

PRNU_LEVELS = [0.005, 0.01, 0.02]
NUC_LEVELS = [0.95, 0.97, 0.99]
DSNU_E = 20000.0


def fpn(args):
    work = WORK / "fpn"
    work.mkdir(parents=True, exist_ok=True)

    bright = WORK / "netd" / "radiance_300.exr"
    if not bright.is_file():
        raise SystemExit("run --netd first; its 300 K radiance field is the "
                         "bright field here")

    rows = []
    for prnu in PRNU_LEVELS:
        for efficiency in NUC_LEVELS:
            config = sensor_toml(work / f"sensor_p{prnu}_n{efficiency}.toml", {
                "sensor.enable_fpn": True,
                "sensor.fpn.prnu_sigma": prnu,
                "sensor.fpn.dsnu_sigma_e": DSNU_E,
                "sensor.fpn.enable_nuc": True,
                "sensor.fpn.nuc_efficiency": efficiency,
            })
            for field, radiance, uniform in (("bright", bright, None),
                                             ("dark", None, 0.0)):
                stats = run_sensor_lab(
                    config, radiance, args.frames,
                    work / f"{field}_p{prnu}_n{efficiency}", uniform=uniform)
                # PRNU is multiplicative, so what it injects scales with the
                # signal; DSNU is additive and does not. A dark field therefore
                # isolates the additive part, which is the whole reason the two
                # are modelled separately.
                injected = (prnu * stats["mean_dn"] if field == "bright" else 0.0)
                rows.append({
                    "field": field, "prnu_sigma": prnu, "nuc_efficiency": efficiency,
                    "mean_dn": stats["mean_dn"],
                    "residual_fpn_dn": stats["residual_fpn_dn"],
                    "temporal_sigma_dn": stats["temporal_sigma_dn_median"],
                    "single_frame_spatial_sigma_dn":
                        stats["single_frame_spatial_sigma_dn"],
                    "injected_prnu_dn": injected,
                    "predicted_residual_dn": (1.0 - efficiency) * injected,
                })
                print(f"  {field:<6} PRNU {prnu:5.1%} NUC {efficiency:5.1%}   "
                      f"mean {stats['mean_dn']:9.1f}   "
                      f"residual {stats['residual_fpn_dn']:8.2f} DN   "
                      f"predicted {(1.0 - efficiency) * injected:8.2f}", flush=True)

    (WORK / "fpn.json").write_text(json.dumps(
        {"frames": args.frames, "dsnu_sigma_e": DSNU_E, "rows": rows},
        indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--netd", action="store_true")
    parser.add_argument("--fpn", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=WORK)
    parser.add_argument("--frames", type=int, default=100,
                        help="frames per point; the FPN stage wants more, "
                             "since the pattern only emerges once the "
                             "temporal noise has been averaged below it")
    args = parser.parse_args()

    if args.roundtrip:
        roundtrip(args)
    if args.netd:
        netd(args)
    if args.fpn:
        fpn(args)
    if not (args.roundtrip or args.netd or args.fpn):
        parser.error("pick a stage")


if __name__ == "__main__":
    main()
