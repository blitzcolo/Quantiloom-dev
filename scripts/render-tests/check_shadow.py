#!/usr/bin/env python3
"""Verify that a spectral band honours the sun-visibility (shadow) ray.

Usage:
    python check_shadow.py <sun.exr> <nosun.exr> [--min-shadow-frac 0.05]
                                                 [--max-shadow-frac 0.60]
                                                 [--flat-ratio 0.05]

The scene (assets/models/shadow_scene) is built so that sun visibility is the
only quantity that varies across the frame: one material, one normal, an
orthographic top-down camera. The reference render has the sun below the
horizon, so its image is the scene lit by sky and self-emission alone.

    excess = sun_render - nosun_render

is therefore the direct-solar term, pixel by pixel. Two things must hold:

  1. Some pixels have excess ~ 0 -- the slab's shadow. A renderer that
     computes shadowFactor and then discards it (which NIR, SWIR and MWIR did)
     gives every pixel the same nonzero excess, so min(excess)/max(excess) ~ 1
     and this check fails.
  2. The near-zero region is a band, not the whole frame. If the solar term
     vanished everywhere the first check would also pass, and that is a
     different bug.

Comparing against a real sunless render rather than a threshold is what makes
this work in MWIR, where the solar term is a few percent of a signal dominated
by self-emission.

Exit code 0 = pass, 1 = fail.
"""

import argparse, pathlib, sys
import numpy as np
import OpenEXR


def read_exr_channel(path, channel_idx=0):
    """Read one channel from an EXR. Returns 2D numpy array (H, W)."""
    f = OpenEXR.File(str(path))
    ch = f.channels()
    first_key = list(ch.keys())[0]
    pixels = ch[first_key].pixels
    if pixels.ndim == 3:
        return pixels[:, :, channel_idx].astype(np.float64)
    return pixels.astype(np.float64)


def main():
    p = argparse.ArgumentParser(description="Sun-occlusion (shadow ray) check")
    p.add_argument("sun_exr", help="render with the sun above the horizon")
    p.add_argument("nosun_exr", help="same scene, sun below the horizon")
    p.add_argument("--min-shadow-frac", type=float, default=0.05,
                   help="smallest acceptable shadowed fraction of the frame")
    p.add_argument("--max-shadow-frac", type=float, default=0.60,
                   help="largest acceptable shadowed fraction of the frame")
    p.add_argument("--flat-ratio", type=float, default=0.05,
                   help="a pixel counts as shadowed below this fraction of the "
                        "brightest solar excess")
    args = p.parse_args()

    for path in (args.sun_exr, args.nosun_exr):
        if not pathlib.Path(path).exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    sun = read_exr_channel(args.sun_exr)
    nosun = read_exr_channel(args.nosun_exr)
    if sun.shape != nosun.shape:
        print(f"ERROR: shape mismatch {sun.shape} vs {nosun.shape}")
        sys.exit(1)

    excess = sun - nosun
    hi = float(np.percentile(excess, 99))
    print(f"Image: {sun.shape[1]}x{sun.shape[0]}")
    print(f"Solar excess: p99 = {hi:.6e}, min = {excess.min():.6e}, "
          f"mean = {excess.mean():.6e}")

    if hi <= 0:
        print("FAIL: the sun contributes nothing anywhere -- "
              "check the solar LUT covers this band")
        sys.exit(1)

    # Fraction of the frame where the sun was blocked. Measured against the
    # p99 excess rather than the max so a single fireflied pixel cannot set
    # the scale.
    shadow_mask = excess < args.flat_ratio * hi
    frac = float(shadow_mask.mean())
    print(f"Shadowed fraction: {frac:.2%} "
          f"(accept {args.min_shadow_frac:.0%}-{args.max_shadow_frac:.0%})")

    ok = True
    if frac < args.min_shadow_frac:
        print(f"FAIL: no shadow band -- the solar term reaches pixels the slab "
              f"occludes (shadowFactor ignored in this band?)")
        ok = False
    elif frac > args.max_shadow_frac:
        print(f"FAIL: shadowed fraction too large -- the solar term is missing "
              f"almost everywhere, not just behind the slab")
        ok = False

    if ok:
        lit = excess[~shadow_mask]
        print(f"Lit-region excess: mean = {lit.mean():.6e}")
        print("PASS")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
