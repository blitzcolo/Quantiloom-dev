#!/usr/bin/env python3
"""E3 -- the Cornell box, compared pixel by pixel.

The one scene in the validation set with no closed form. Its answer is an
integral over multiply-scattered light in a closed room, and nobody can write it
down -- which is why it is worth rendering twice, and why it is the only scene
here where the two renderers are compared to each other rather than each to a
third number.

Comparison is at single wavelengths, not in a fused band. Quantiloom's VIS_FUSED
mode integrates 32 wavelengths against the CIE observer and returns linear sRGB;
matching that in Mitsuba would mean matching two colour pipelines as well as two
transports, and a disagreement would have three places to hide. At one
wavelength both renderers point-evaluate the same reflectance file and the
comparison is about transport alone. Three wavelengths are used -- 450, 550 and
650 nm -- chosen because the five wall curves are well separated there, so
colour bleeding is measured rather than assumed.

What is compared, and the error budget it sits in:

  delta(x)  |L_QL - L_M3| / max(L_M3, floor), the floor being 1% of the masked
            mean, so that a dark corner does not report an enormous relative
            error on two nearly-equal small numbers.
  RMSE      full-frame, relative, as Section VIII-B defines it.
  floor     each renderer's own Monte Carlo noise, from two independent seeds:
            RMS(A-B)/sqrt(2). A difference below this is not a difference
            between renderers, and is reported as such rather than interpreted.
  bias      the film-response width, measured at 4.8e-4 by
            measure_panel_spectrum.py --check. Quantiloom evaluates AT the
            wavelength; Mitsuba integrates a narrow boxcar around it.

Usage:
    run_e3_cornell.py --out-dir <paper>/evidence/e1/cornell

The output directory is e1/cornell, not e3: "e3" here is the third scene of the
cross-renderer batch, while evidence/e3/ is Section VIII-G's convergence work,
and the two wrote files into one directory under different meanings of the same
token. They diverged -- a Cornell run landed in e1/cornell while e3/ kept an
older one, and the tracked copy was the stale one, disagreeing with the
manuscript it was evidence for.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mitsuba_common as common  # noqa: E402
import measure_panel_spectrum as panel  # noqa: E402  (one lamp, one definition)

REPO = common.REPO
PORT_DIR = REPO / "build" / "validation" / "cornell_m3"
CONFIG = REPO / "assets" / "configs" / "validation" / "e3_cornell_single.toml"
WORK_DIR = REPO / "renders" / "validation" / "cornell"

# The five wall materials plus the panel, all forced Lambertian. A glTF
# material with no KHR_materials_specular still carries the dielectric default
# F0 = 0.04 in Quantiloom, which Mitsuba's `diffuse` does not have.
MATERIAL_NAMES = ["Construction Concrete", "Red Brick", "Olive Green Paint",
                  "White Marble", "Asphalt", "Light Panel"]


def write_quantiloom_config(path, description):
    # specular = 0 makes the walls Lambertian, matching Mitsuba's `diffuse`.
    # ir_emissivity is written alongside it and is NOT decoration: an override
    # block that mentions neither ir_emissivity nor ir_transmittance leaves the
    # derived reflectance undefined, and this key states the intent explicitly
    # rather than relying on what a defaulted derivation happens to produce.
    # The panel needs its emission spectrum bound as well as its specular
    # zeroed, and the binding has to be the SAME one measure_panel_spectrum.py
    # used -- which is why the token is imported from there rather than spelled
    # again. Mitsuba is handed cornell_panel_radiance.csv, which that script
    # measured out of a panel with this curve bound; if this config renders the
    # panel without it, Quantiloom lights the box with the RGB expansion while
    # Mitsuba lights it with the measured spectrum, and the comparison reports
    # the difference between two lamps as a transport disagreement. It did
    # exactly that once: 16 % apart at 450 nm, uniform across the whole frame.
    def override(name):
        block = f'[material_overrides."{name}"]\nspecular = 0.0'
        if name == panel.PANEL_NAME:
            block += (f'\nemissive_curve = "{panel.PANEL_EMISSIVE_CURVE}"'
                      f'\nemissive_scale = "{panel.PANEL_EMISSIVE_SCALE}"')
        return block

    overrides = "\n\n".join(override(name) for name in MATERIAL_NAMES)

    # Both renderers read the SAME visible-band file. Quantiloom uploads a
    # spectral curve as 64 samples on a uniform grid over the curve's whole
    # range, so the shipped 300-15000 nm CSVs would reach the shader at about
    # 200 nm resolution -- 41.6 % wrong at 550 nm on the olive green paint.
    # Trimmed to 380-780 nm the same 64 samples are 6.35 nm apart and the loss
    # is at most 0.42 %. Binding through [spectral_curves] also overrides what
    # the glTF declares, which is the documented precedence.
    curves = "\n".join(
        f'"{group["material"]}" = "{group["reflectance_csv"]}"'
        for group in description["groups"] if "reflectance_csv" in group)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'''# Auto-generated by scripts/validation/run_e3_cornell.py.
# The Cornell box at one wavelength, for cross-renderer comparison. No spectral
# basis file: the materials carry their measured reflectance through the glTF's
# QUANTILOOM_material_ir extension, and the comparison binds the same CSVs on
# the Mitsuba side, so both renderers read one file per material.

[renderer]
resolution = [512, 512]
spp = 4096
seed = 21628
output = "renders/validation/cornell/ql_550.exr"

[spectral]
mode = "single"
wavelength_nm = 550.0

[scene]
gltf = "{(REPO / "assets" / "models" / "cornell_box" / "cornell_box.gltf").as_posix()}"
world_units_to_meters = 0.001
default_temperature_k = 300.0

[camera]
position = [278.0, 273.0, -800.0]
look_at = [278.0, 273.0, 280.0]
up = [0.0, 1.0, 0.0]
fov_y = 39.0

[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance = [0.0, 0.0, 0.0]
sky_radiance = [0.0, 0.0, 0.0]

[material]
albedo = [0.5, 0.5, 0.5]

[sensor]
enabled = false

[quality]
log_material_sources = true

[spectral_curves]
{curves}

{overrides}
''', encoding="utf-8")


def mitsuba_scene(description, wavelength, resolution, half_width, max_depth):
    import mitsuba as mi
    transform = mi.ScalarTransform4f

    camera = description["camera"]
    scale = 0.001
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": max_depth},
        "sensor": {
            "type": "perspective",
            "fov": camera["fov_y_deg"],
            # Mitsuba's default fov axis is x. The Quantiloom config's 39
            # degrees is vertical, and on a square film the two happen to
            # agree -- which is exactly why this must be set explicitly rather
            # than left to coincide until someone renders a wide frame.
            "fov_axis": "y",
            "to_world": transform().look_at(
                origin=[c * scale for c in camera["position_mm"]],
                target=[c * scale for c in camera["look_at_mm"]],
                up=camera["up"]),
            "film": common.specfilm(resolution, resolution,
                                    common.single_srf(wavelength, half_width)),
            "sampler": {"type": "independent"},
        },
    }

    panel_grid, panel_values = common.read_spectral_csv(
        description["panel_spectrum_csv"])

    for group in description["groups"]:
        shape = {
            "type": "ply",
            "filename": str(PORT_DIR / group["ply"]),
        }
        if "reflectance_csv" in group:
            shape["bsdf"] = {
                "type": "diffuse",
                "reflectance": common.spectrum_from_csv(group["reflectance_csv"]),
            }
        else:
            # The panel: white, and emitting the spectrum measured out of
            # Quantiloom. Its glTF normal faces down into the room, which is
            # the side an area emitter radiates from, so no flip is needed.
            shape["bsdf"] = {"type": "diffuse",
                             "reflectance": float(group["reflectance_flat"])}
            shape["emitter"] = {"type": "area",
                                "radiance": common.irregular(panel_grid, panel_values)}
        scene[group["key"]] = shape
    return scene


def compare(a, b, floor_fraction=0.01):
    """delta(x), RMSE and the summary statistics, over the lit mask."""
    mask = (a > 1e-6) | (b > 1e-6)
    if not np.any(mask):
        return None
    reference = b[mask]
    floor = floor_fraction * float(np.mean(reference))
    delta = np.abs(a[mask] - reference) / np.maximum(reference, floor)
    rmse = float(np.sqrt(np.mean((a[mask] - reference) ** 2))
                 / np.mean(reference))
    return {
        "masked_pixels": int(mask.sum()),
        "floor": floor,
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_p99": float(np.percentile(delta, 99)),
        "delta_max": float(np.max(delta)),
        "rmse_relative": rmse,
        "mean_ql": float(np.mean(a[mask])),
        "mean_m3": float(np.mean(reference)),
        "mean_ratio": float(np.mean(a[mask]) / np.mean(reference)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wavelengths", type=float, nargs="+",
                        default=[450.0, 550.0, 650.0])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--spp", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs=2, default=[0x547C + 1, 0x547C + 2])
    parser.add_argument("--half-width", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--ql-exe", type=pathlib.Path, default=common.DEFAULT_QL)
    parser.add_argument("--skip-quantiloom", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    description = json.loads((PORT_DIR / "scene.json").read_text(encoding="utf-8"))

    import mitsuba as mi
    mi.set_variant("scalar_spectral")

    write_quantiloom_config(CONFIG, description)
    results = {}

    for wavelength in args.wavelengths:
        print(f"\n{wavelength:.0f} nm")

        # --- Quantiloom, two seeds for its own noise floor ---
        ql_images = []
        for seed in args.seeds:
            output = WORK_DIR / f"ql_{wavelength:.0f}_s{seed}.exr"
            if not args.skip_quantiloom:
                manifest = args.out_dir / f"_ql_{wavelength:.0f}_s{seed}.txt"
                manifest.write_text(
                    f'{CONFIG.as_posix()} | spectral.wavelength_nm={wavelength:.1f} '
                    f'renderer.spp={args.spp} renderer.seed={seed} '
                    f'renderer.resolution=[{args.resolution},{args.resolution}] '
                    f'renderer.output="{output.as_posix()}"\n', encoding="utf-8")
                import subprocess
                run = subprocess.run([str(args.ql_exe), "batch", str(manifest)],
                                     cwd=str(REPO), capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=14400)
                if run.returncode != 0:
                    print(run.stdout[-3000:], file=sys.stderr)
                    raise SystemExit(f"Quantiloom failed at {wavelength} nm")
            ql_images.append(common.read_exr(output))

        ql = 0.5 * (ql_images[0] + ql_images[1])
        ql_floor = float(np.sqrt(np.mean((ql_images[0] - ql_images[1]) ** 2) / 2.0))

        # --- Mitsuba, likewise ---
        scene = mitsuba_scene(description, wavelength, args.resolution,
                              args.half_width, args.max_depth)
        m3_floor, m3_a, m3_b = common.half_split_noise(scene, args.spp, args.seeds)
        m3 = 0.5 * (m3_a + m3_b)

        stats = compare(ql, m3)
        stats["quantiloom_noise_floor"] = ql_floor
        stats["mitsuba_noise_floor"] = m3_floor
        stats["mean_signal"] = stats["mean_m3"]
        stats["noise_floor_relative"] = (
            max(ql_floor, m3_floor) / stats["mean_m3"] if stats["mean_m3"] else None)
        results[f"{wavelength:.0f}"] = stats

        print(f"  masked pixels        {stats['masked_pixels']}")
        print(f"  mean QL / mean M3    {stats['mean_ratio']:.6f}")
        print(f"  relative RMSE        {stats['rmse_relative']:.4%}")
        print(f"  delta mean / median  {stats['delta_mean']:.4%} / "
              f"{stats['delta_median']:.4%}")
        print(f"  delta p99 / max      {stats['delta_p99']:.4%} / "
              f"{stats['delta_max']:.4%}")
        print(f"  noise floor QL / M3  {ql_floor / stats['mean_m3']:.4%} / "
              f"{m3_floor / stats['mean_m3']:.4%}")

        np.save(args.out_dir / f"delta_{wavelength:.0f}.npy",
                np.abs(ql - m3) / np.maximum(m3, stats["floor"]))

    common.write_manifest(args.out_dir / "e3_cornell.json", common.manifest(
        "e3_cornell", {"wavelengths": args.wavelengths, "spp": args.spp,
                       "resolution": args.resolution, "seeds": args.seeds,
                       "half_width_nm": args.half_width,
                       "max_depth": args.max_depth, "results": results}))
    (args.out_dir / "e3_summary.json").write_text(json.dumps(results, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'e3_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
