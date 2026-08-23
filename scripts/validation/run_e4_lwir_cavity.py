#!/usr/bin/env python3
"""E4 -- the self-emission domain: an isothermal LWIR cavity, both renderers.

The manuscript reserved a sentence for this experiment failing to run. The
reasoning was that Mitsuba's spectral variant is bounded to the visible range,
so a 300 K cavity radiating at 8-12 um could not be described to it. That is
not the case: the 360-830 nm figure bounds the CIE tables, the RGB upsampling
and the *default* wavelength sampling, and a film with an explicit spectral
response makes the sensor importance-sample the response's own support instead.
So the third renderer does reach the thermal infrared, and the comparison that
was going to be declared impossible is reported here as a number.

The scene is the classical isothermal cavity: a closed box whose walls are all
at one temperature. Its interior radiance is the blackbody function at that
temperature regardless of the wall emissivity -- a surface that emits less also
reflects more, and the two exactly compensate. That is why it is the right test
for the self-emission path: it has a closed form that does not depend on the
one material property most likely to be implemented inconsistently.

Two constructions on the Mitsuba side that are not optional:

  * `flip_normals`, because area emitters are one-sided and a cube's normals
    face outward. Without it the cavity is dark inside and the render is a
    picture of nothing.
  * The emitted spectrum is tabulated as eps * B(lambda, T) into a `regular`
    spectrum rather than built from the `blackbody` plugin, which has no scale
    parameter and is masked outside its declared range.

Usage:
    run_e4_lwir_cavity.py --out-dir evidence/e4
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

# CODATA 2018, as scripts/physics-audit/harness.py carries them.
C1_NM = 1.191042972e20   # 2 h c^2, in W nm^4 / (sr m^2)
C2_M = 1.438776877e-2    # h c / k, in m K

BAND = (8000.0, 12000.0)
CAVITY_T = 300.0
QL_CASES = {
    "eps1.00": ("assets/configs/furnace_lwir_e1.toml", "furnace_lwir_e1_output.exr", 1.0),
    "eps0.50": ("assets/configs/furnace_lwir_e05.toml", "furnace_lwir_e05_output.exr", 0.5),
    "eps0.00": ("assets/configs/furnace_lwir_rho1.toml", "furnace_lwir_rho1_output.exr", 0.0),
}


def planck(temperature_k, wavelength_nm):
    """Spectral radiance, W / (sr m^2 nm)."""
    exponent = C2_M / (wavelength_nm * 1e-9 * temperature_k)
    if exponent > 700.0:
        return 0.0
    return C1_NM / wavelength_nm ** 5 / (math.exp(exponent) - 1.0)


def band_average_exact(temperature_k, band, samples=200001):
    """The band mean of Planck, integrated finely enough to be the truth.

    Distinct from the sixteen-point trapezoid the renderer uses and that
    `check_furnace.py` deliberately mirrors. That checker asks whether the
    shader agrees with the renderer's own quadrature, which is the right
    question for a self-consistency gate; here the question is whether either
    renderer agrees with the physics, so the reference has to be the integral
    rather than the renderer's approximation of it.
    """
    grid = np.linspace(band[0], band[1], samples)
    values = np.array([planck(temperature_k, w) for w in grid])
    return float(np.trapezoid(values, grid) / (band[1] - band[0]))


def band_average_trapezoid16(temperature_k, band):
    """The renderer's own rule, for reporting its quadrature error separately."""
    lo, hi, n = band[0], band[1], 16
    step = (hi - lo) / (n - 1)
    values = [planck(temperature_k, lo + i * step) for i in range(n)]
    return (sum(values) - 0.5 * (values[0] + values[-1])) * step / (hi - lo)


def mitsuba_cavity(emissivity, resolution, srf, temperature_k=CAVITY_T):
    """A closed cube at one temperature, seen from inside.

    The emitted spectrum is tabulated over a range wider than the film's
    response, so that the sensor's wavelength sampling never lands outside the
    emitter's declared support and read zero.
    """
    import mitsuba as mi
    transform = mi.ScalarTransform4f

    grid = np.linspace(7000.0, 13000.0, 601)
    emitted = np.array([emissivity * planck(temperature_k, w) for w in grid])

    cube = {
        "type": "cube",
        # Area emitters radiate from the normal side, and a cube's normals face
        # outward. A cavity is the inside.
        "flip_normals": True,
        "to_world": transform().scale([5, 5, 5]),
        # A plain scalar rather than a flat spectrum. Mitsuba builds a sampling
        # distribution from a `regular` spectrum and refuses one that is
        # everywhere zero, which is exactly the eps = 1 wall -- a perfect
        # absorber is a legitimate surface and should not have to be described
        # as an almost-perfect one. A scalar reflectance is a uniform spectrum
        # with no such construction; the automatic D65 wrapping that would make
        # this wrong applies to emitter values, not to reflectances.
        "bsdf": {"type": "diffuse", "reflectance": float(1.0 - emissivity)},
    }
    if emissivity > 0.0:
        cube["emitter"] = {"type": "area",
                           "radiance": common.irregular(grid, emitted)}

    return {
        "type": "scene",
        # Deep enough that a cavity with rho = 1 still closes: the radiance
        # there arrives entirely by reflection, so a truncated path is a
        # truncated answer rather than a slightly noisy one.
        "integrator": {"type": "path", "max_depth": 200},
        "sensor": {
            "type": "perspective",
            "fov": 90.0,
            "fov_axis": "y",
            "to_world": transform().look_at(origin=[0, 0, 0], target=[0, 0, -1],
                                            up=[0, 1, 0]),
            "film": common.specfilm(resolution, resolution, srf),
            "sampler": {"type": "independent"},
        },
        "cavity": cube,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--spp-ladder", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--seeds", type=int, nargs=2, default=[0x547C + 1, 0x547C + 2])
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--ql-exe", type=pathlib.Path, default=common.DEFAULT_QL)
    parser.add_argument("--skip-quantiloom", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    exact = band_average_exact(CAVITY_T, BAND)
    trapezoid = band_average_trapezoid16(CAVITY_T, BAND)
    print(f"Planck band average at {CAVITY_T:.0f} K over "
          f"{BAND[0]:.0f}-{BAND[1]:.0f} nm")
    print(f"  exact integral            {exact:.8e} W/sr/m^2/nm")
    print(f"  renderer's 16-pt rule     {trapezoid:.8e}  "
          f"({abs(trapezoid - exact) / exact:.3e} relative)\n")

    results = {"analytic_exact": exact, "analytic_trapezoid16": trapezoid,
               "quadrature_relative_error": abs(trapezoid - exact) / exact,
               "band_nm": BAND, "temperature_k": CAVITY_T,
               "tolerance": args.tolerance, "quantiloom": {}, "mitsuba": {}}

    print("Quantiloom")
    for name, (config, output, emissivity) in QL_CASES.items():
        if not args.skip_quantiloom:
            common.run_quantiloom(REPO / config, cli=args.ql_exe)
        image = common.read_exr(REPO / output)
        mean = float(np.mean(image))
        entry = {
            "emissivity": emissivity, "mean": mean, "config": config,
            "relative_error_vs_exact": abs(mean - exact) / exact,
            "relative_error_vs_trapezoid16": abs(mean - trapezoid) / trapezoid,
        }
        entry["passed"] = entry["relative_error_vs_exact"] <= args.tolerance
        results["quantiloom"][name] = entry
        print(f"  {name:<12} {mean:.8e}  vs exact {entry['relative_error_vs_exact']:.3e}"
              f"  vs 16-pt {entry['relative_error_vs_trapezoid16']:.3e}"
              f"  {'PASS' if entry['passed'] else 'FAIL'}")

    import mitsuba as mi
    mi.set_variant("scalar_spectral")

    print("\nMitsuba 3")
    srf = common.band_srf(*BAND)
    for name, (_, _, emissivity) in QL_CASES.items():
        if emissivity <= 0.0:
            # Nothing emits, so there is nothing to compare: a cavity of
            # perfect reflectors with no source is black in any renderer.
            # Quantiloom's rho1 case is not empty in the same way -- its walls
            # carry a temperature that the shader emits from regardless of the
            # material's reflectance -- so the two are not the same scene and
            # are not compared.
            print(f"  {name:<12} not constructed: a Mitsuba cavity with no "
                  f"emitter has no source")
            continue
        scene = mitsuba_cavity(emissivity, args.resolution, srf)
        ladder = {}
        for spp in args.spp_ladder:
            floor, a, b = common.half_split_noise(scene, spp, args.seeds)
            mean = float(np.mean(0.5 * (a + b)))
            error = abs(mean - exact) / exact
            ladder[str(spp)] = {"mean": mean, "relative_error_vs_exact": error,
                                "half_split_noise": floor,
                                "passed": error <= args.tolerance}
            print(f"  {name:<12} @{spp:>5} spp  {mean:.8e}  vs exact {error:.3e}  "
                  f"{'PASS' if error <= args.tolerance else 'FAIL'}")
        results["mitsuba"][name] = ladder

    common.write_manifest(args.out_dir / "e4_lwir_cavity.json", common.manifest(
        "e4_lwir_cavity", {"resolution": args.resolution,
                           "spp_ladder": args.spp_ladder, "seeds": args.seeds,
                           "results": results}))
    (args.out_dir / "e4_summary.json").write_text(json.dumps(results, indent=2),
                                                  encoding="utf-8")

    failed = [f"quantiloom/{k}" for k, v in results["quantiloom"].items()
              if not v["passed"]]
    for name, ladder in results["mitsuba"].items():
        failed += [f"mitsuba/{name}@{spp}" for spp, entry in ladder.items()
                   if not entry["passed"]]
    print(f"\n{'E4 FAILED: ' + ', '.join(failed) if failed else 'E4 passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
