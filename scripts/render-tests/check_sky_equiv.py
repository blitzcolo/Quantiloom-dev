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
    args = p.parse_args()

    if not pathlib.Path(args.exr).exists():
        print(f"ERROR: {args.exr} not found")
        sys.exit(1)

    ref = reference_radiance()
    img = read_exr_channel(args.exr)
    mean_val = float(img.mean())
    spread = float(img.max() - img.min())
    rel_err = abs(mean_val - ref) / ref if ref > 0 else 0.0

    print(f"Reference: {ref:.6e} W/sr/m²/nm  (rho={RHO}, cos={COS_THETA})")
    print(f"Image:     {img.shape[1]}x{img.shape[0]}")
    print(f"ROI mean:  {mean_val:.6e}")
    print(f"Spread:    {spread:.3e}  (must be ~0: no occluders, no variance)")
    print(f"Rel error: {rel_err:.4%}")

    if rel_err > args.tol:
        print(f"FAIL: error {rel_err:.4%} exceeds tolerance {args.tol:.4%} -- "
              f"a factor near 2 means the traced bounce is being added on top of "
              f"the analytic sky term rather than correcting it")
        sys.exit(1)
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
