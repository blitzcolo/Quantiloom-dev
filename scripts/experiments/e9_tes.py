#!/usr/bin/env python3
"""E9 -- temperature and emissivity separation, against a scene that knows both.

A thermal camera measures radiance. What anyone wants is temperature, and to
get one you need an emissivity you do not have: N bands give N measurements and
N + 1 unknowns, so the problem is underdetermined by exactly one, always. Every
method for it adds a constraint from somewhere other than the measurement, and
the standard one -- ASTER's NEM/ratio/MMD -- adds an empirical regression
between the spread of a spectrum and its minimum.

Which makes TES a thing to validate rather than a thing to trust, and
validating it needs a scene whose temperature and emissivity are both known.
That is what a renderer is for. This one builds a plate at a stated temperature
with a stated emissivity spectrum, renders it as a multi-band LWIR cube, runs
TES on the bands, and reports what came back against what went in.

The emissivity is deliberately not flat. TES on a grey body is trivial and
tells you nothing: the whole difficulty is separating a spectrum that varies
from a temperature that does not, and the case that separates methods is a
reststrahlen band -- the deep, wide reflectance feature quartz has near 9 um,
which is why silicate mineralogy from orbit is the application TES was built
for. So the fixture has one.

    e9_tes.py                       # the default plate, 12 bands, 8-12 um
    e9_tes.py --bands 6             # fewer bands, which is where MMD strains
    e9_tes.py --temperature-k 320   # a hotter plate
    e9_tes.py --emissivity flat     # the grey case, as a control

Writes a JSON with the recovered temperature and emissivity beside the true
ones, and the errors between them.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runlog import REPO, Run  # noqa: E402
from _winpaths import require_windows_paths  # noqa: E402

# Planck, in the units the renderer reports: W m^-2 sr^-1 nm^-1 against nm.
kPlanckC1 = 1.191042972e20  # 2 h c^2 x 1e36, so lambda goes in as nm
kPlanckC2 = 1.4387768775e7  # h c / k, nm K


def planck(lambda_nm, temperature_k):
    """Spectral radiance of a blackbody."""
    lam = np.asarray(lambda_nm, dtype=np.float64)
    return kPlanckC1 / (lam**5 * (np.exp(kPlanckC2 / (lam * temperature_k)) - 1.0))


def planck_inverse(lambda_nm, radiance):
    """The temperature a blackbody would need to emit that radiance there."""
    lam = np.asarray(lambda_nm, dtype=np.float64)
    radiance = np.maximum(np.asarray(radiance, dtype=np.float64), 1e-30)
    return kPlanckC2 / (lam * np.log1p(kPlanckC1 / (lam**5 * radiance)))


def reststrahlen_reflectance(lambda_nm):
    """A quartz-like LWIR reflectance: flat 0.04, with a band near 9 um.

    Synthetic and labelled so. What matters for this experiment is the SHAPE --
    a wide feature that takes emissivity from 0.96 down to about 0.55 and back
    -- because that spread is what the MMD regression is a regression of. A
    real quartz spectrum has two lobes and a sharper edge; one Gaussian is
    enough to make the separation non-trivial and simple enough that the truth
    is a formula rather than a table.
    """
    lam = np.asarray(lambda_nm, dtype=np.float64)
    return 0.04 + 0.41 * np.exp(-0.5 * ((lam - 9000.0) / 700.0) ** 2)


def emissivity_of(lambda_nm, kind):
    if kind == "flat":
        return np.full_like(np.asarray(lambda_nm, dtype=np.float64), 0.96)
    return 1.0 - reststrahlen_reflectance(lambda_nm)


def tes(lambda_nm, radiance, eps_max=0.99):
    """ASTER's NEM, ratio and MMD, in that order.

    Returns (temperature_K, emissivity[]). The three steps are three different
    admissions of the same problem: NEM guesses the maximum emissivity, the
    ratio throws away the level to keep the shape that guess did not spoil, and
    MMD puts the level back from an empirical relation between spread and
    minimum. Nothing here is derived from radiometry alone, because nothing
    can be.
    """
    lam = np.asarray(lambda_nm, dtype=np.float64)
    L = np.asarray(radiance, dtype=np.float64)

    # NEM: the brightness temperature each band would have at eps_max, and the
    # largest of them. Largest because a band where the true emissivity is
    # closest to eps_max gives the least underestimate, and every other band
    # underestimates.
    t_nem = planck_inverse(lam, L / eps_max)
    temperature = float(np.max(t_nem))
    emissivity = L / planck(lam, temperature)

    # Ratio: the shape, normalised away from the level the NEM guess set.
    beta = emissivity / np.mean(emissivity)

    # MMD: ASTER's regression of the minimum emissivity on the spread. The
    # constants are the published ones and they are the empirical content of
    # the method -- everything above is algebra.
    mmd = float(np.max(beta) - np.min(beta))
    eps_min = 0.994 - 0.687 * (mmd ** 0.737)

    emissivity = beta * (eps_min / float(np.min(beta)))
    # And the temperature the band with the highest emissivity implies, which
    # is where the Planck inversion is least sensitive to an emissivity error.
    best = int(np.argmax(emissivity))
    temperature = float(planck_inverse(lam[best], L[best] / emissivity[best]))
    return temperature, emissivity


def build_config(work, temperature_k, lo_nm, hi_nm, bands, emissivity_kind):
    """A plate at a known temperature with a known emissivity spectrum.

    No solve and no sun: `ir_temperature_k` states the temperature outright and
    a cold sky keeps the reflected term small, so what the cube carries is this
    surface's own emission and the truth is a number in this file rather than
    the output of another calculation. Validating TES against a solved field
    would be validating two things at once.
    """
    grid = np.linspace(lo_nm, hi_nm, 64)
    reflectance = (np.full_like(grid, 0.04) if emissivity_kind == "flat"
                   else reststrahlen_reflectance(grid))
    curve = work / "tes_reflectance.csv"
    with curve.open("w", encoding="utf-8") as f:
        f.write("# Synthetic LWIR reflectance for the TES closed loop.\n")
        f.write("# See e9_tes.py: a quartz-like reststrahlen band, not a measurement.\n")
        f.write("# wavelength_nm,reflectance\n")
        for lam, rho in zip(grid, reflectance):
            f.write(f"{lam:.1f},{rho:.6f}\n")

    step = (hi_nm - lo_nm) / (bands - 1) if bands > 1 else (hi_nm - lo_nm)
    config = work / "tes_plate.toml"
    config.write_text(f"""# Generated by scripts/experiments/e9_tes.py -- do not edit.
[renderer]
resolution = [64, 64]
spp = 16
output = "{(work / 'tes_cube').as_posix()}"

[spectral]
mode = "multispectral"
band = "LWIR"

[hyperspectral]
wavelength_min_nm = {lo_nm}
wavelength_max_nm = {hi_nm}
wavelength_step_nm = {step}
output_format = "exr_spectral"

[scene]
gltf = "{(REPO / 'assets/models/shadow_scene/shadow_scene_open.gltf').as_posix()}"
world_units_to_meters = 1.0
default_temperature_k = {temperature_k}

[camera]
position   = [0.0, 20.0, 0.0]
look_at    = [0.0, 0.0, 0.0]
up         = [0.0, 0.0, 1.0]
projection = "orthographic"
ortho_height = 16.0

[lighting]
sun_direction = [0.6, 0.8, 0.0]
sun_radiance  = [0.0, 0.0, 0.0]
sky_radiance  = [0.0, 0.0, 0.0]
# Cold, so the reflected term is small against the surface's own emission. Not
# zero: a sky at 0 K is not a thing, and TES has to live with a reflected term
# in any case -- what this does is keep it from being the whole story.
atmosphere_temperature_k = 200.0

[material]
albedo = [0.5, 0.5, 0.5]

[[materials]]
name = "shadow_scene_open"
ir_temperature_k = {temperature_k}

[spectral_curves]
"shadow_scene_open" = "{curve.as_posix()}"

[quality]
fail_on_srgb_upsample = false

[sensor]
enabled = false
""", encoding="utf-8")
    return config


def read_cube(path):
    """Band radiances and their wavelengths, out of the spectral EXR."""
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        bands = []
        for name, channel in channels.items():
            # Fichet's layout names a band by its wavelength, with a comma for
            # the decimal point: "S0.9500,0nm". Anything that does not parse is
            # not a band and is skipped rather than guessed at.
            token = name.split(".")[-1]
            if not token.endswith("nm"):
                continue
            try:
                lam = float(token[:-2].replace(",", "."))
            except ValueError:
                continue
            bands.append((lam, np.asarray(channel.pixels, dtype=np.float64)))
    bands.sort(key=lambda b: b[0])
    if not bands:
        raise SystemExit(f"{path} carries no spectral bands this reader recognises")
    return np.array([b[0] for b in bands]), np.stack([b[1] for b in bands])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temperature-k", type=float, default=310.0)
    p.add_argument("--bands", type=int, default=12)
    p.add_argument("--lo-nm", type=float, default=8000.0)
    p.add_argument("--hi-nm", type=float, default=12000.0)
    p.add_argument("--emissivity", choices=("reststrahlen", "flat"),
                   default="reststrahlen")
    p.add_argument("--out", default=str(REPO / "renders/e9_tes/e9_tes.json"))
    args = p.parse_args()

    if args.bands < 3:
        raise SystemExit("TES needs at least three bands to have a spread to regress on")

    out = pathlib.Path(args.out)
    require_windows_paths(out)
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)

    with Run("e9_tes", work) as run:
        config = build_config(work, args.temperature_k, args.lo_nm, args.hi_nm,
                              args.bands, args.emissivity)
        run.render(config)

    cube = work / "tes_cube.exr"
    if not cube.exists():
        raise SystemExit(f"no cube at {cube}; the render wrote something else")
    wavelengths, radiance = read_cube(cube)

    # The plate is uniform, so one spectrum is the whole image and averaging
    # over pixels is averaging away the sampling noise rather than averaging
    # over a scene. A textured surface would need this per pixel.
    spectrum = radiance.reshape(len(wavelengths), -1).mean(axis=1)

    recovered_t, recovered_eps = tes(wavelengths, spectrum)
    true_eps = emissivity_of(wavelengths, args.emissivity)

    report = {
        "temperature_k": {
            "true": args.temperature_k,
            "recovered": recovered_t,
            "error_k": recovered_t - args.temperature_k,
        },
        "emissivity": {
            "wavelength_nm": wavelengths.tolist(),
            "true": true_eps.tolist(),
            "recovered": recovered_eps.tolist(),
            "rmse": float(np.sqrt(np.mean((recovered_eps - true_eps) ** 2))),
            "max_error": float(np.max(np.abs(recovered_eps - true_eps))),
        },
        "bands": int(len(wavelengths)),
        "spectrum_w_m2_sr_nm": spectrum.tolist(),
        "emissivity_model": args.emissivity,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"bands            {len(wavelengths)}  "
          f"[{wavelengths[0]:.0f}, {wavelengths[-1]:.0f}] nm")
    print(f"temperature      true {args.temperature_k:.2f} K, "
          f"recovered {recovered_t:.2f} K, error {recovered_t - args.temperature_k:+.2f} K")
    print(f"emissivity       RMSE {report['emissivity']['rmse']:.4f}, "
          f"worst band {report['emissivity']['max_error']:.4f}")
    print(f"                 true  [{true_eps.min():.3f}, {true_eps.max():.3f}]")
    print(f"                 found [{recovered_eps.min():.3f}, {recovered_eps.max():.3f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
