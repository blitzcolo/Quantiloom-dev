#!/usr/bin/env python3
"""Check an unobstructed Lambertian plane against its closed-form radiance.

Usage:
    python check_sky_equiv.py <output.exr> [--tol 0.005]

The scene is one ground plane with nothing above it, so every environment
bounce ray escapes and comes back carrying exactly the analytic sky radiance
the shader's wavelength loop already reflected. The traced correction is
therefore identically zero -- not small, zero -- and the render must equal

    L = rho * ( E_sun/pi * cos(theta) + E_sky/pi )

at any spp, with no Monte Carlo noise anywhere in it.

That is the check that the bounce was added as a CORRECTION to the analytic sky
term and not on top of it. Getting that wrong doubles the ambient contribution
of every open scene, and since the right answer here has no variance, a 2x error
and a 1.02x error are equally visible.

The illuminant is deliberately flat (assets/luts/flat_sun_sky.csv) rather than
ASTM G-173: the renderer resamples a solar curve onto a uniform grid before
uploading it, and matching that here would mean reimplementing the resampler and
then testing it against itself. A flat spectrum survives any resampling, so the
reference below is a closed form and the comparison is against physics.

SWIR because it is the band whose reflectance fallback is exactly 1 - emissivity.
NIR would route through ConvertLinearRGBToSpectrum instead, whose value at 1 um
is not a number this test could state in closed form.

The visible band has no closed form here either, for the same reason: its
reflectance is an upsampled spectrum rather than 1 - emissivity. What the two
visible modes have instead is each other, so --reference names a render to
compare against rather than the formula above. The deterministic mode is the one
that carries the zero-variance claim into the visible band, and --max-spread
asserts it: with the residual exactly zero, an analytic sky and a uniform plane,
every pixel of that frame is the same number. vis_hero draws a quartet per
sample, so its pixels differ by which wavelengths they drew; what it must match
is the deterministic mean.

Exit code 0 = pass, 1 = fail.
"""

import argparse, math, pathlib, sys
import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Scene constants, from assets/models/shadow_scene and the companion TOML.
RHO = 0.7          # 1 - emissivity, with emissivity 0.3 and no transmittance
COS_THETA = 0.8    # sun_direction = normalize(0.6, 0.8, 0) against a +Y normal

# assets/luts/flat_sun_sky.csv, in W/m^2/nm. Flat rather than ASTM G-173 on
# purpose: the renderer resamples a solar curve onto a uniform grid before
# uploading it, and matching that here would mean reimplementing the resampler
# and testing it against itself. A flat spectrum passes through any resampling
# unchanged, so the reference below is a closed form and the comparison is
# against physics rather than against another copy of the code.
E_SUN = 1.0
E_SKY = 0.5        # global 1.5 minus direct 1.0, per solar_lut_diffuse_is_global


def reference_radiance():
    """rho * (E_sun/pi * cos + E_sky/pi), constant in lambda, so the shader's
    trapezoid integration over the band returns it unchanged."""
    return RHO * (E_SUN / math.pi * COS_THETA + E_SKY / math.pi)


def read_exr_channel(path, channel_idx=0):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    pixels = ch[list(ch.keys())[0]].pixels
    if pixels.ndim == 3:
        return pixels[:, :, channel_idx].astype(np.float64)
    return pixels.astype(np.float64)


def main():
    p = argparse.ArgumentParser(description="Open-sky Lambertian equivalence check")
    p.add_argument("exr")
    p.add_argument("--tol", type=float, default=0.005,
                   help="relative tolerance (default 0.5%%)")
    p.add_argument("--reference", metavar="EXR",
                   help="compare against this render instead of the closed form, "
                        "for a band where the reflectance has no closed form")
    p.add_argument("--max-spread", type=float, metavar="REL",
                   help="also require the frame's own max-min to be below this "
                        "fraction of its mean")
    p.add_argument("--reference-max-spread", type=float, metavar="REL",
                   help="the same bound on the --reference frame; 0 asserts the "
                        "deterministic mode's zero-variance claim at zero tolerance")
    args = p.parse_args()

    if not pathlib.Path(args.exr).exists():
        print(f"ERROR: {args.exr} not found")
        sys.exit(1)

    img = read_exr_channel(args.exr)
    if args.reference:
        if not pathlib.Path(args.reference).exists():
            print(f"ERROR: {args.reference} not found")
            sys.exit(1)
        ref_img = read_exr_channel(args.reference)
        ref = float(ref_img.mean())
        ref_spread = float(ref_img.max() - ref_img.min())
        source = f"{args.reference} (mean, spread {ref_spread:.3e})"
    else:
        ref = reference_radiance()
        source = f"closed form, rho={RHO}, cos={COS_THETA}"

    mean_val = float(img.mean())
    spread = float(img.max() - img.min())
    rel_err = abs(mean_val - ref) / ref if ref > 0 else 0.0

    print(f"Reference: {ref:.6e} W/sr/m²/nm  ({source})")
    print(f"Image:     {img.shape[1]}x{img.shape[0]}")
    print(f"ROI mean:  {mean_val:.6e}")
    print(f"Spread:    {spread:.3e}  ({spread / mean_val if mean_val else 0.0:.3e} of the mean)")
    print(f"Rel error: {rel_err:.4%}")

    failed = False
    if rel_err > args.tol:
        print(f"FAIL: error {rel_err:.4%} exceeds tolerance {args.tol:.4%} -- "
              f"a factor near 2 means the traced bounce is being added on top of "
              f"the analytic sky term rather than correcting it")
        failed = True

    def check_spread(label, value, mean, bound):
        if bound is None:
            return False
        rel = value / mean if mean else 0.0
        if rel <= bound:
            return False
        print(f"FAIL: {label} spread {rel:.3e} of the mean exceeds {bound:.3e} -- "
              f"this scene has one material, one illumination and a residual that "
              f"is zero by construction, so anything that varies pixel to pixel is "
              f"variance that should not be here")
        return True

    failed |= check_spread("image", spread, mean_val, args.max_spread)
    if args.reference:
        failed |= check_spread("reference", ref_spread, ref, args.reference_max_spread)
    elif args.reference_max_spread is not None:
        print("ERROR: --reference-max-spread needs --reference")
        sys.exit(1)

    if failed:
        sys.exit(1)
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
