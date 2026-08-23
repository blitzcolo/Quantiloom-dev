#!/usr/bin/env python3
"""E1 -- uniform-sky Lambertian plane, each renderer against the closed form.

The simplest scene in the set, and the one that has to work before anything
else is believed. A Lambertian plane of reflectance rho under a uniform sky of
hemispherical irradiance E returns

    L = rho * E / pi

from every direction, with no free parameters and no Monte Carlo variance in
the analytic term. Both renderers are compared to that value INDEPENDENTLY --
they are not compared to each other. A cross-renderer difference at this stage
would be uninterpretable: with no third number, there is no way to say which of
the two moved.

Two comparison domains, because Quantiloom writes two different units and both
should be checked:

  band    the SWIR band average, 1400-2400 nm. Legitimate here precisely
          because every spectrum in the scene is flat, so Quantiloom's
          sixteen-point trapezoid over the band is exact rather than
          approximate, and the comparison is not measuring quadrature.
  single  2000 nm, point-evaluated on both sides. No discretisation at all.

Usage:
    run_e1_furnace.py --out-dir evidence/e1
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mitsuba_common as common  # noqa: E402

REPO = common.REPO

# The scene, as both renderers must see it.
RHO = 0.7               # Lambertian reflectance, flat across the band
SKY_IRRADIANCE = 0.5    # hemispherical, W/m^2/nm -- the LUT's Global column
BAND = (1400.0, 2400.0)
SINGLE_NM = 2000.0
ORTHO_HALF_EXTENT = 8.0  # Quantiloom's ortho_height 16 is the full extent

ANALYTIC = RHO * SKY_IRRADIANCE / math.pi

QL_CONFIGS = {
    "band": "assets/configs/validation/e1_furnace_swir.toml",
    "single": "assets/configs/validation/e1_furnace_single2000.toml",
}
QL_OUTPUTS = {
    "band": "renders/validation/e1_furnace_swir.exr",
    "single": "renders/validation/e1_furnace_single2000.exr",
}


def mitsuba_scene(srf, resolution):
    """The same plane under the same sky, described to Mitsuba.

    The ground is a rectangle rotated to face +Y and scaled well past the
    camera's field, so that no pixel sees its edge -- an edge would put
    background into the mean and make the comparison a comparison of framing.
    """
    import mitsuba as mi
    transform = mi.ScalarTransform4f
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 9},
        "sensor": {
            "type": "orthographic",
            # Looking straight down, with the same up vector the Quantiloom
            # config uses, so the two images are the same image.
            "to_world": (transform().look_at(origin=[0, 20, 0], target=[0, 0, 0],
                                             up=[0, 0, 1])
                         @ transform().scale([ORTHO_HALF_EXTENT,
                                              ORTHO_HALF_EXTENT, 1])),
            "film": common.specfilm(resolution, resolution, srf),
            "sampler": {"type": "independent"},
        },
        "ground": {
            "type": "rectangle",
            "to_world": transform().rotate(axis=[1, 0, 0], angle=-90)
                        @ transform().scale([50, 50, 1]),
            "bsdf": {"type": "diffuse", "reflectance": common.flat(RHO)},
        },
        # Uniform sky. Mitsuba's `constant` emitter carries RADIANCE, while the
        # Quantiloom config carries the hemispherical IRRADIANCE its LUT lists,
        # so the conversion L = E / pi happens here rather than being assumed
        # to have happened somewhere.
        "sky": {"type": "constant",
                "radiance": common.flat(SKY_IRRADIANCE / math.pi)},
    }


def report(name, values, tolerance):
    mean = float(np.mean(values))
    spread = float(np.max(values) - np.min(values))
    error = abs(mean - ANALYTIC) / ANALYTIC
    status = "PASS" if error <= tolerance else "FAIL"
    print(f"  {name:<24} {mean:.8f}  rel {error:.3e}  spread {spread:.2e}  {status}")
    return {"mean": mean, "spread": spread, "relative_error": error,
            "passed": error <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spp-ladder", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--seeds", type=int, nargs=2, default=[0x547C + 1, 0x547C + 2])
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--ql-exe", type=pathlib.Path, default=common.DEFAULT_QL)
    parser.add_argument("--skip-quantiloom", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {"analytic": ANALYTIC, "tolerance": args.tolerance,
               "quantiloom": {}, "mitsuba": {}}

    print(f"analytic L = rho*E/pi = {RHO} * {SKY_IRRADIANCE} / pi = {ANALYTIC:.8f} "
          f"W/sr/m^2/nm\n")

    print("Quantiloom")
    for domain, config in QL_CONFIGS.items():
        if not args.skip_quantiloom:
            common.run_quantiloom(REPO / config, cli=args.ql_exe)
        image = common.read_exr(REPO / QL_OUTPUTS[domain])
        results["quantiloom"][domain] = report(domain, image, args.tolerance)
        results["quantiloom"][domain]["config"] = config
        results["quantiloom"][domain]["output_sha256"] = common.sha256(
            REPO / QL_OUTPUTS[domain])

    import mitsuba as mi
    mi.set_variant("scalar_spectral")

    print("\nMitsuba 3")
    srfs = {"band": common.band_srf(*BAND),
            "single": common.single_srf(SINGLE_NM)}
    for domain, srf in srfs.items():
        scene = mitsuba_scene(srf, args.resolution)
        ladder = {}
        for spp in args.spp_ladder:
            floor, a, b = common.half_split_noise(scene, spp, args.seeds)
            entry = report(f"{domain} @ {spp} spp", 0.5 * (a + b), args.tolerance)
            entry["half_split_noise"] = floor
            ladder[str(spp)] = entry
        results["mitsuba"][domain] = ladder

    # The window width must not be a parameter of the answer. If halving it
    # moved the single-wavelength result, the boxcar would be integrating a
    # curve rather than sampling a point.
    narrow = mitsuba_scene(common.single_srf(SINGLE_NM, 0.2), args.resolution)
    narrow_mean = float(np.mean(common.render(narrow, max(args.spp_ladder),
                                              args.seeds[0])))
    wide = results["mitsuba"]["single"][str(max(args.spp_ladder))]["mean"]
    results["mitsuba"]["srf_window_sensitivity"] = {
        "half_width_1nm": wide, "half_width_0p2nm": narrow_mean,
        "relative_difference": abs(narrow_mean - wide) / wide,
    }
    print(f"\n  SRF window +/-1 nm vs +/-0.2 nm: "
          f"{results['mitsuba']['srf_window_sensitivity']['relative_difference']:.2e} "
          f"relative")

    manifest = common.manifest("e1_furnace", {
        "scene": {"rho": RHO, "sky_irradiance_w_m2_nm": SKY_IRRADIANCE,
                  "band_nm": BAND, "single_nm": SINGLE_NM,
                  "ortho_half_extent_m": ORTHO_HALF_EXTENT},
        "resolution": args.resolution, "spp_ladder": args.spp_ladder,
        "seeds": args.seeds, "results": results,
    })
    common.write_manifest(args.out_dir / "e1_furnace.json", manifest)
    (args.out_dir / "e1_summary.json").write_text(json.dumps(results, indent=2),
                                                  encoding="utf-8")

    failed = [f"quantiloom/{k}" for k, v in results["quantiloom"].items()
              if not v["passed"]]
    for domain, ladder in results["mitsuba"].items():
        # The window-sensitivity entry lives beside the ladders and holds
        # floats, not per-spp records; it is a diagnostic rather than a gate.
        if domain == "srf_window_sensitivity":
            continue
        failed += [f"mitsuba/{domain}@{spp}" for spp, entry in ladder.items()
                   if not entry.get("passed", True)]
    print(f"\n{'E1 FAILED: ' + ', '.join(failed) if failed else 'E1 passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
