#!/usr/bin/env python3
"""Check that indirect light exists in VIS_FUSED and that it carries colour.

Usage:
    python check_color_bleed.py <cornell_box_vis.exr>
                                [--min-lit-frac 0.20] [--min-chroma-ratio 1.20]

The Cornell box config has no sun, no sky and no solar LUT: its only emitter is
the ceiling panel, which is emissive geometry. There is no next-event estimation
for emissive geometry anywhere in the renderer, so every photon that reaches any
other surface got there by bouncing. That makes two independent assertions
possible without a reference render.

1. LIT FRACTION. Without an environment bounce the frame is the light panel and
   nothing else -- literally 0.6% of pixels above the noise floor. With one it is
   above 40%. This catches "the bounce does not fire", which is what a wrong
   Russian-roulette probability, a bad depth gate or an over-broad env-map
   condition all look like.

2. CHROMATICITY GRADIENT. The side walls are red brick and olive paint, so light
   arriving at the neutral surfaces between them is tinted by whichever wall it
   came off. R/G therefore has to vary horizontally across the frame. It does not
   vary if the bounce returns a band-averaged radiance instead of a per-
   wavelength one, because then rho(lambda) multiplies a single grey number and
   the bleed comes back colourless. This is the visible-band counterpart of what
   the quartz cavity checks in LWIR.

Neither number is a physical constant, so both thresholds sit well below what a
working renderer produces (0.43 and 1.59 as measured) and well above what a
broken one does (0.006 and 1.00).

Exit code 0 = pass, 1 = fail.
"""

import argparse, pathlib, sys
import numpy as np
import OpenEXR

# Columns sampled for the chromaticity profile, as fractions of image width.
# Wide enough apart to span from beside one wall to beside the other, and away
# from the extreme edges where the walls themselves fill the frame.
PROFILE_X = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
PROFILE_HALF_WIDTH = 0.03
PROFILE_Y = (0.35, 0.55)   # a horizontal band across the middle of the box

FLOOR = 1e-6   # a pixel counts as lit above this radiance


def read_rgb(path):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    px = ch[list(ch.keys())[0]].pixels.astype(np.float64)
    if px.ndim != 3 or px.shape[2] < 3:
        raise SystemExit(f"ERROR: expected an RGB image, got shape {px.shape}")
    return px[:, :, 0], px[:, :, 1], px[:, :, 2]


def main():
    p = argparse.ArgumentParser(description="VIS_FUSED indirect-light check")
    p.add_argument("exr")
    p.add_argument("--min-lit-frac", type=float, default=0.20)
    p.add_argument("--min-chroma-ratio", type=float, default=1.20)
    args = p.parse_args()

    if not pathlib.Path(args.exr).exists():
        print(f"ERROR: {args.exr} not found")
        sys.exit(1)

    R, G, B = read_rgb(args.exr)
    h, w = R.shape

    lit_frac = float(np.mean(R > FLOOR))
    print(f"Image:        {w}x{h}")
    print(f"Lit fraction: {lit_frac:.2%} (need > {args.min_lit_frac:.0%})")

    band = slice(int(h * PROFILE_Y[0]), int(h * PROFILE_Y[1]))
    ratios = []
    for frac in PROFILE_X:
        x0 = int(w * (frac - PROFILE_HALF_WIDTH))
        x1 = int(w * (frac + PROFILE_HALF_WIDTH))
        r = R[band, x0:x1].mean()
        g = G[band, x0:x1].mean()
        if g > FLOOR:
            ratios.append(r / g)

    ok = True
    if lit_frac < args.min_lit_frac:
        print("FAIL: almost nothing is lit -- in a scene whose only emitter is "
              "geometry, that means no light is bouncing at all")
        ok = False

    if len(ratios) < 2:
        print("FAIL: too few lit columns to measure a chromaticity gradient")
        ok = False
    else:
        spread = max(ratios) / min(ratios)
        print(f"R/G across frame: {min(ratios):.3f} .. {max(ratios):.3f} "
              f"(ratio {spread:.3f}, need > {args.min_chroma_ratio:.2f})")
        if spread < args.min_chroma_ratio:
            print("FAIL: indirect light is present but colourless -- the bounce "
                  "is returning a band average rather than a wavelength")
            ok = False

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
