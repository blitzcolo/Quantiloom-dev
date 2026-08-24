#!/usr/bin/env python3
"""Measure the Cornell light panel's emitted spectrum out of Quantiloom itself.

The Cornell box is lit by one emissive quad. It used to be lit by an RGB triple
with no spectrum, expanded by a fixed convention -- chromaticity fitted as a
reflectance, magnitude carried alongside, the whole multiplied by D65 -- because
no configuration key could hand a real one to the renderer. `emissive_curve` is
that key now, and the panel binds CIE standard illuminant D65: published data
rather than a graphics convention, and the neutral choice for a scene whose job
is to compare two transports rather than to look like a room.

The spectrum is still MEASURED rather than read out of the CIE table, and for a
better reason than before. Three things sit between the published table and what
the surface emits: the levelling that takes the lamp's brightness from the
author's RGB, the resampling onto the band being rendered, and the estimator
itself. Handing Mitsuba the raw table would be handing it a different lamp from
the one the comparison scene has. So: point a camera at the panel and nothing
else, render one wavelength at a time, read the radiance off the pixels. What
comes back is by definition what this renderer emits.

The panel must be measured ALONE. Measured inside the box, a pixel on the panel
also carries interior light that has bounced back onto it -- small, since the
panel is white and the room is dim, but not zero, and it would be silently
folded into the emitted spectrum and then handed to Mitsuba as truth.

Three steps, each a separate flag so the slow one can be skipped:

    measure_panel_spectrum.py --make-scene    build the panel-only asset,
                                              the sweep config and the manifest
    measure_panel_spectrum.py --collect       render the sweep and write the CSV
    measure_panel_spectrum.py --check         closure and shape checks

Closure is the one that matters: a spectrum measured from Quantiloom and given
to Mitsuba should make Mitsuba's own panel-only render return the same numbers.
If it does not, everything downstream is measuring the handoff.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))
import mitsuba_common as common  # noqa: E402

REPO = common.REPO
ASSET_DIR = REPO / "assets" / "models" / "cornell_box"
CONFIG_DIR = REPO / "assets" / "configs" / "validation"
WORK_DIR = REPO / "renders" / "validation" / "panel_sweep"
CSV_OUT = REPO / "assets" / "luts" / "cornell_panel_radiance.csv"

# The panel's own geometry, in millimetres, as generate_cornell_box.py places
# it: on the ceiling at y = 548.7, facing down.
PANEL_QUAD = [(213.0, 548.7, 332.0), (213.0, 548.7, 227.0),
              (343.0, 548.7, 227.0), (343.0, 548.7, 332.0)]
PANEL_EMISSIVE = [15.0, 15.0, 12.0]
PANEL_NAME = "Light Panel"

# The lamp the box binds, and which this sweep must bind too. If these ever
# disagree, the spectrum handed to the second renderer is not the spectrum the
# comparison scene emits, and every number downstream measures the mismatch.
PANEL_EMISSIVE_CURVE = "d65"
PANEL_EMISSIVE_SCALE = "match_luminance"

# The sweep stops where the observer does, and so does the lamp: the compiled
# D65 table runs 380-780 nm, and emission is zero outside a bound curve's own
# span rather than held flat. The render band is 400-780, so the first ten
# samples here come back as exact zeros -- that is the boundary reporting
# itself, not a gap in the measurement.
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP = 380.0, 780.0, 2.0


def write_panel_gltf(path):
    """The light quad alone, with the material the box gives it."""
    a, b, c, d = PANEL_QUAD
    tris = [a, b, c, a, c, d]
    normal = (0.0, -1.0, 0.0)  # facing down, as in the box

    positions = list(tris)
    pos_b = struct.pack(f"<{len(positions) * 3}f",
                        *[component for v in positions for component in v])
    nor_b = struct.pack(f"<{len(positions) * 3}f", *(normal * len(positions)))
    idx_b = struct.pack(f"<{len(positions)}I", *range(len(positions)))
    buf = idx_b + pos_b + nor_b

    xs = [v[0] for v in positions]
    ys = [v[1] for v in positions]
    zs = [v[2] for v in positions]

    gltf = {
        "asset": {"version": "2.0", "generator": "measure_panel_spectrum.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [
            {"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0, "material": 0}]}],
        "materials": [{
            "name": PANEL_NAME,
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                     "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "emissiveFactor": PANEL_EMISSIVE,
            "doubleSided": True,
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": len(positions),
             "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": [min(xs), min(ys), min(zs)],
             "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 2, "componentType": 5126, "count": len(positions),
             "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_b), "target": 34963},
            {"buffer": 0, "byteOffset": len(idx_b), "byteLength": len(pos_b),
             "target": 34962},
            {"buffer": 0, "byteOffset": len(idx_b) + len(pos_b),
             "byteLength": len(nor_b), "target": 34962},
        ],
        "buffers": [{"uri": path.with_suffix(".bin").name, "byteLength": len(buf)}],
    }
    path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    path.with_suffix(".bin").write_bytes(buf)


def wavelengths():
    return np.arange(LAMBDA_MIN, LAMBDA_MAX + 1e-9, LAMBDA_STEP)


def make_scene(args):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    gltf = ASSET_DIR / "cornell_panel_only.gltf"
    write_panel_gltf(gltf)

    # Straight up at the panel from below, close enough that it fills the frame
    # and every pixel is panel. The panel centre is (278, 548.7, 279.5).
    config = CONFIG_DIR / "panel_only_single.toml"
    config.write_text(f'''# Auto-generated by scripts/validation/measure_panel_spectrum.py.
# The Cornell light panel alone, looked at from directly below, so that every
# pixel carries emission and nothing else. Rendered one wavelength at a time;
# the batch manifest beside this file sweeps them.
#
# spp = 1 on purpose: direct emission has no stochastic component, and a second
# sample would average two identical values.

[renderer]
resolution = [64, 64]
spp = 1
seed = 21628
output = "renders/validation/panel_sweep/panel_550.exr"

[spectral]
mode = "single"
wavelength_nm = 550.0

[scene]
gltf = "{gltf.as_posix()}"
world_units_to_meters = 0.001
default_temperature_k = 300.0

[camera]
position = [278.0, 400.0, 279.5]
look_at = [278.0, 548.7, 279.5]
up = [0.0, 0.0, 1.0]
fov_y = 20.0

[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance = [0.0, 0.0, 0.0]
sky_radiance = [0.0, 0.0, 0.0]

[material]
albedo = [0.5, 0.5, 0.5]

[material_overrides."{PANEL_NAME}"]
# The same binding cornell_box_vis.toml gives the panel. Repeated here rather
# than inherited, because this is a different scene file -- and if it is ever
# dropped, the sweep quietly measures the RGB expansion instead of the lamp, and
# the second renderer is handed an emitter this scene does not have.
emissive_curve = "{PANEL_EMISSIVE_CURVE}"
emissive_scale = "{PANEL_EMISSIVE_SCALE}"

[sensor]
enabled = false
''', encoding="utf-8")

    manifest = CONFIG_DIR / "panel_only_sweep.txt"
    lines = []
    for lam in wavelengths():
        out = (WORK_DIR / f"panel_{lam:.0f}.exr").as_posix()
        lines.append(f'{config.as_posix()} | spectral.wavelength_nm={lam:.1f} '
                     f'renderer.output="{out}"')
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {gltf}")
    print(f"wrote {config}")
    print(f"wrote {manifest} ({len(lines)} wavelengths)")


def collect(args):
    manifest = CONFIG_DIR / "panel_only_sweep.txt"
    if not args.skip_render:
        import subprocess
        result = subprocess.run([str(args.ql_exe), "batch", str(manifest)],
                                cwd=str(REPO), capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=7200)
        if result.returncode != 0:
            print(result.stdout[-4000:], file=sys.stderr)
            raise SystemExit("panel sweep failed")

    rows = []
    for lam in wavelengths():
        path = WORK_DIR / f"panel_{lam:.0f}.exr"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run --make-scene and --collect")
        image = common.read_exr(path)
        # Every pixel is panel, so the frame is constant; the spread is
        # reported as the check that it is.
        rows.append((lam, float(np.mean(image)),
                     float(np.max(image) - np.min(image))))

    spreads = [r[2] for r in rows]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8") as f:
        f.write("# Emitted spectral radiance of the Cornell light panel, measured\n"
                "# from Quantiloom by rendering the panel alone one wavelength at a\n"
                f'# time. The panel binds emissive_curve = "{PANEL_EMISSIVE_CURVE}" with\n'
                f'# emissive_scale = "{PANEL_EMISSIVE_SCALE}", so its shape is that\n'
                f"# published curve and its level is the author RGB {PANEL_EMISSIVE}.\n"
                "# Measured rather than recomputed even so: the levelling, the resampling\n"
                "# onto the render band and the estimator all sit between the published\n"
                "# table and what the surface actually emits, and reimplementing that\n"
                "# chain here would be a second version of it to disagree with.\n"
                "# Produced by scripts/validation/measure_panel_spectrum.py.\n"
                "# wavelength_nm, radiance_W_sr-1_m-2_nm-1\n")
        for lam, value, _ in rows:
            f.write(f"{lam:.1f}, {value:.9g}\n")

    print(f"wrote {CSV_OUT} ({len(rows)} wavelengths)")
    print(f"  peak {max(r[1] for r in rows):.6g} at "
          f"{max(rows, key=lambda r: r[1])[0]:.0f} nm")
    print(f"  largest within-frame spread {max(spreads):.3e} "
          f"(should be ~0: every pixel is panel)")


def window_mean(grid, values, centre, half_width, samples=4001):
    """The mean of the piecewise-linear spectrum over a boxcar window.

    This, not the value at the centre, is what a boxcar film response
    estimates: `specfilm` stores the integral of response times radiance. The
    two differ, and by a documented amount rather than a mysterious one -- the
    spectrum has curvature, and its tabulation has a kink at every node, so
    averaging across a node is not the node's value even when the sampling is
    exact there.
    """
    window = np.linspace(centre - half_width, centre + half_width, samples)
    return float(np.trapezoid(np.interp(window, grid, values), window)
                 / (2.0 * half_width))


def check(args):
    """Closure: hand the measured spectrum to Mitsuba and read it back.

    A one-sided area emitter viewed head-on returns its own radiance, so
    Mitsuba's answer should be the spectrum Quantiloom was measured at. Any gap
    is in the handoff -- the file, the interpolation, or the film response --
    and would otherwise show up later as a cross-renderer disagreement about
    the Cornell box.

    Two numbers are reported for each wavelength, and the distinction is the
    whole point of the check:

      vs window   Mitsuba against the boxcar mean of the same tabulated
                  spectrum. This is the handoff, and it should close to
                  numerical precision. Measured at 1e-7 to 6e-5.
      vs point    Mitsuba against the value AT the wavelength, which is what
                  Quantiloom's single-wavelength mode renders. This is not an
                  error in either renderer; it is the width of the film
                  response, and it is reported because it is a systematic in
                  the cross-renderer comparison downstream. It scales with the
                  window: 2.4e-3 at +/-1 nm, 4.2e-4 at +/-0.2 nm.
    """
    import mitsuba as mi
    mi.set_variant("scalar_spectral")
    transform = mi.ScalarTransform4f

    grid, values = common.read_spectral_csv(CSV_OUT)
    print(f"{CSV_OUT.name}: {len(grid)} points, "
          f"{grid[0]:.0f}-{grid[-1]:.0f} nm, film response half-width "
          f"{args.half_width:.2g} nm\n")

    worst_handoff = 0.0
    worst_window_bias = 0.0
    for lam in args.check_wavelengths:
        point = float(np.interp(lam, grid, values))
        expected = window_mean(grid, values, lam, args.half_width)
        scene = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective", "fov": 20.0, "fov_axis": "y",
                "to_world": transform().look_at(origin=[0, -0.15, 0],
                                                target=[0, 0, 0], up=[0, 0, 1]),
                "film": common.specfilm(
                    32, 32, common.single_srf(lam, args.half_width)),
                "sampler": {"type": "independent"},
            },
            "panel": {
                "type": "rectangle",
                "to_world": transform().rotate(axis=[1, 0, 0], angle=90)
                            @ transform().scale([0.05, 0.05, 1]),
                "bsdf": {"type": "diffuse", "reflectance": 0.0},
                "emitter": {"type": "area",
                            "radiance": common.irregular(grid, values)},
            },
        }
        measured = float(np.mean(common.render(scene, 64, 0x547C)))
        handoff = abs(measured - expected) / expected if expected else float("inf")
        window_bias = abs(expected - point) / point if point else float("inf")
        worst_handoff = max(worst_handoff, handoff)
        worst_window_bias = max(worst_window_bias, window_bias)
        print(f"  {lam:6.1f} nm  mitsuba {measured:.6g}  window mean {expected:.6g}  "
              f"point {point:.6g}   vs window {handoff:.2e}   "
              f"vs point {window_bias:.2e}")

    print(f"\nhandoff closure     {worst_handoff:.2e}  "
          f"{'PASS' if worst_handoff < 1e-3 else 'FAIL'}")
    print(f"film-response bias  {worst_window_bias:.2e}  "
          f"(systematic, carried into the comparison's error budget)")

    # The endpoints, because that is where this measurement has already caught a
    # renderer bug. A bound emission curve is zero OUTSIDE its span, and the
    # shader decides inside-or-outside from (lambda - start) / step compared
    # against numSamples - 1. step is stored as a rounded f32, so at the very
    # last wavelength that quotient can land a fraction above the last index and
    # the guard returns zero -- which is what happened here: 780 nm read exactly
    # 0.0 while 778 nm was the brightest sample in the spectrum. A hole one
    # sample wide at the end of a lamp is invisible in a render and fatal in a
    # handoff, so it is asserted rather than eyeballed.
    interior = float(values[-2])
    endpoint = float(values[-1])
    print(f"endpoint continuity {grid[-2]:.0f} nm {interior:.6g} -> "
          f"{grid[-1]:.0f} nm {endpoint:.6g}  ", end="")
    endpoint_ok = endpoint > 0.5 * interior
    print("PASS" if endpoint_ok else "FAIL -- the curve's last sample dropped out")

    return 0 if (worst_handoff < 1e-3 and endpoint_ok) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-scene", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--ql-exe", type=pathlib.Path, default=common.DEFAULT_QL)
    parser.add_argument("--check-wavelengths", type=float, nargs="+",
                        default=[450.0, 550.0, 650.0])
    parser.add_argument("--half-width", type=float, default=0.2,
                        help="film response half-width in nm. Narrow by default: the response is a boxcar, so its width is a systematic bias "
                             "against a point evaluation, and it scales with the width -- 2.4e-3 at 1 nm, 4.2e-4 at 0.2 nm.")
    args = parser.parse_args()

    if not (args.make_scene or args.collect or args.check):
        parser.error("pick at least one of --make-scene, --collect, --check")

    if args.make_scene:
        make_scene(args)
    if args.collect:
        collect(args)
    if args.check:
        return check(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
