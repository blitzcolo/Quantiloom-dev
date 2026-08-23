#!/usr/bin/env python3
"""E2 -- open-sky equivalence: the same plane, now with a sun.

E1 checked that a uniform hemisphere is integrated correctly. This adds the
other half of an outdoor illumination model, a collimated beam at an angle, and
checks the sum against

    L = rho * ( E_sun * cos(theta) / pi  +  E_sky / pi )

Both renderers are again compared to that closed form independently, not to
each other. What the sun adds beyond E1 is a cosine and a direction convention,
and both are places where a wrong answer still looks like a rendered image:
Mitsuba's `directional` emitter takes the direction light TRAVELS, while
Quantiloom's `sun_direction` points from the surface toward the sun.

Quantiloom is exact here rather than merely converged, and that is a property of
the renderer's design rather than of this scene being easy. Every band keeps its
analytic ambient terms and traces one ray carrying only the difference between
what the hemisphere actually returns and what those terms assumed; with nothing
above the plane, every such ray escapes and the difference is zero to the bit.
A nonzero spread across this frame would mean the bounce had been added on top
of the analytic sky instead of as a correction to it.

Usage:
    run_e2_skyequiv.py --out-dir evidence/e2
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

RHO = 0.7
SUN_IRRADIANCE = 1.0     # W/m^2/nm, normal to the beam -- the LUT's Direct column
SKY_IRRADIANCE = 0.5     # W/m^2/nm, hemispherical -- Global minus Direct
SUN_DIRECTION = (0.6, 0.8, 0.0)   # surface toward sun, as the config writes it
BAND = (1400.0, 2400.0)
SINGLE_NM = 2000.0

COS_THETA = SUN_DIRECTION[1] / math.sqrt(sum(c * c for c in SUN_DIRECTION))
ANALYTIC = RHO * (SUN_IRRADIANCE * COS_THETA / math.pi + SKY_IRRADIANCE / math.pi)

QL = {
    "band": ("assets/configs/skyequiv_swir.toml",
             "skyequiv_swir_output.exr"),
    "single": ("assets/configs/validation/e2_skyequiv_single2000.toml",
               "renders/validation/e2_skyequiv_single2000.exr"),
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
    results = {"analytic": ANALYTIC, "cos_theta": COS_THETA,
               "tolerance": args.tolerance, "quantiloom": {}, "mitsuba": {}}

    print(f"analytic L = rho*(E_sun*cos/pi + E_sky/pi) = {RHO}*({SUN_IRRADIANCE}*"
          f"{COS_THETA:.1f}/pi + {SKY_IRRADIANCE}/pi) = {ANALYTIC:.8f} W/sr/m^2/nm\n")

    print("Quantiloom")
    for domain, (config, output) in QL.items():
        if not args.skip_quantiloom:
            common.run_quantiloom(REPO / config, cli=args.ql_exe)
        image = common.read_exr(REPO / output)
        entry = report(domain, image, args.tolerance)
        entry["config"] = config
        entry["output_sha256"] = common.sha256(REPO / output)
        results["quantiloom"][domain] = entry

    import mitsuba as mi
    mi.set_variant("scalar_spectral")

    print("\nMitsuba 3")
    srfs = {"band": common.band_srf(*BAND), "single": common.single_srf(SINGLE_NM)}
    for domain, srf in srfs.items():
        scene = common.ground_plane_scene(
            srf, args.resolution, SKY_IRRADIANCE, rho=RHO,
            sun_irradiance=SUN_IRRADIANCE, sun_direction=SUN_DIRECTION)
        ladder = {}
        for spp in args.spp_ladder:
            floor, a, b = common.half_split_noise(scene, spp, args.seeds)
            entry = report(f"{domain} @ {spp} spp", 0.5 * (a + b), args.tolerance)
            entry["half_split_noise"] = floor
            ladder[str(spp)] = entry
        results["mitsuba"][domain] = ladder

    common.write_manifest(args.out_dir / "e2_skyequiv.json", common.manifest(
        "e2_skyequiv", {
            "scene": {"rho": RHO, "sun_irradiance_w_m2_nm": SUN_IRRADIANCE,
                      "sky_irradiance_w_m2_nm": SKY_IRRADIANCE,
                      "sun_direction_surface_to_sun": SUN_DIRECTION,
                      "cos_theta": COS_THETA, "band_nm": BAND,
                      "single_nm": SINGLE_NM},
            "resolution": args.resolution, "spp_ladder": args.spp_ladder,
            "seeds": args.seeds, "results": results,
        }))
    (args.out_dir / "e2_summary.json").write_text(json.dumps(results, indent=2),
                                                  encoding="utf-8")

    failed = [f"quantiloom/{k}" for k, v in results["quantiloom"].items()
              if not v["passed"]]
    for domain, ladder in results["mitsuba"].items():
        failed += [f"mitsuba/{domain}@{spp}" for spp, entry in ladder.items()
                   if not entry.get("passed", True)]
    print(f"\n{'E2 FAILED: ' + ', '.join(failed) if failed else 'E2 passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
