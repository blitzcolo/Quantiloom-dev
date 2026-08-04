#!/usr/bin/env python3
"""Validate endmember mixing against single-curve renders.

Renders assets/configs/endmember_checker.toml three ways:

  a) bound to endmember 0 alone
  b) bound to endmember 1 alone
  c) bound to both, unmixed from the checkerboard base colour

In (c), each checker block is painted with exactly one endmember's colour, so
the unmix must put all the weight on that endmember and the block must render
as the corresponding single-curve image. Comparing (c) against (a) and (b)
rather than against a predicted radiance means the illumination, the
atmosphere and the tone curve all cancel -- if they are wrong they are wrong
identically in all three.

Also asserts the thing that motivated the feature: a mixed render is NOT flat.
A single flat curve has no spatial variation at all, so the block-to-block
difference is the whole point.

Usage:
    python scripts/render-tests/check_endmember_mix.py [--exe PATH] [--tol 0.06]

Exit code 0 = pass, 1 = fail.
"""

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "assets/configs/endmember_checker.toml"
DEFAULT_EXE = REPO / "build/src/app/Release/Quantiloom.exe"

MIXED_REFS = ('spectral_material_refs = ["Spar.patens CRMS322v06 grn.a", '
              '"Grass_Golden_Dry GDS480"]')
SINGLE = ['spectral_material_ref = "Spar.patens CRMS322v06 grn.a"',
          'spectral_material_ref = "Grass_Golden_Dry GDS480"']


def srgb_to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def render(exe, name, refs_line):
    """Render the config with its [[materials]] refs line replaced."""
    text = CONFIG.read_text()
    assert MIXED_REFS in text, "config no longer contains the expected refs line"
    text = text.replace(MIXED_REFS, refs_line)
    text = text.replace('output = "endmember_checker.exr"', f'output = "{name}.exr"')

    tmp = CONFIG.with_name(f"_check_{name}.toml")
    tmp.write_text(text)
    try:
        # Relative to the repo root, not absolute: the renderer is a Windows
        # binary and cannot open a /mnt/... path, but it inherits the working
        # directory through the interop layer.
        proc = subprocess.run([str(exe), str(tmp.relative_to(REPO))],
                              capture_output=True, text=True, cwd=str(REPO), timeout=900)
        if proc.returncode != 0 and "Rendering COMPLETED" not in proc.stdout:
            print(proc.stdout[-2000:], file=sys.stderr)
            raise SystemExit(f"render failed: {name}")
        png = REPO / f"{name}.png"
        if not png.exists():
            raise SystemExit(f"no output image: {png}")
        img = srgb_to_linear(np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255.0)
        return img, proc.stdout
    finally:
        tmp.unlink(missing_ok=True)


def classify(mixed, ref0, ref1):
    """Per pixel, the relative distance to whichever reference is nearer.

    Deliberately not a grid of blocks: where the checker lands in the image
    depends on the camera framing and the UV layout, and a misaligned grid
    averages the two materials together and looks exactly like the failure
    this is testing for. Every pixel should equal ONE of the two single-curve
    renders, and which one it is does not matter.
    """
    px = mixed.reshape(-1, 3)
    d0 = np.abs(px - ref0) / np.maximum(ref0, 1e-9)
    d1 = np.abs(px - ref1) / np.maximum(ref1, 1e-9)
    err0, err1 = d0.max(axis=1), d1.max(axis=1)
    nearest0 = err0 <= err1
    return np.minimum(err0, err1), nearest0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--tol", type=float, default=0.06,
                    help="relative tolerance per channel (default 6%%)")
    args = ap.parse_args()

    exe = pathlib.Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"renderer not built: {exe}")

    single0, _ = render(exe, "endmember_single0", SINGLE[0])
    single1, _ = render(exe, "endmember_single1", SINGLE[1])
    mixed, log = render(exe, "endmember_mixed", MIXED_REFS)

    split = re.search(r"endmember split ([\d]+% / [\d]+%)", log)
    if split:
        print(f"  unmix reported split {split.group(1)} (expect about 50% / 50%)")

    ref0 = single0.reshape(-1, 3).mean(axis=0)
    ref1 = single1.reshape(-1, 3).mean(axis=0)

    print(f"  endmember 0 alone   {ref0[0]:.4f} {ref0[1]:.4f} {ref0[2]:.4f}")
    print(f"  endmember 1 alone   {ref1[0]:.4f} {ref1[1]:.4f} {ref1[2]:.4f}")

    status = 0

    # The two curves must be distinguishable, or the test proves nothing.
    separation = np.abs(ref0 - ref1) / np.maximum(ref0, 1e-9)
    print(f"  separation between them: {separation.max():.1%}")
    if separation.max() < 0.1:
        print("  FAIL: the two endmembers render too similarly to tell apart")
        status = 1

    err, nearest0 = classify(mixed, ref0, ref1)
    p95 = float(np.percentile(err, 95))
    share0 = float(nearest0.mean())
    print(f"  95th pct pixel error vs its nearer endmember: {p95:.2%}")
    print(f"  pixels nearer endmember 0: {share0:.1%}")

    # Almost every pixel must BE one of the two, not something in between. A
    # flat average would sit half the separation away from both and fail this
    # outright -- which is what a first pass with a misaligned block grid
    # looked like, and why this is measured per pixel.
    if p95 > args.tol:
        print(f"  FAIL: pixels do not reproduce a single endmember "
              f"(tolerance {args.tol:.0%})")
        status = 1

    # Both endmembers have to actually appear. One weight stuck at zero would
    # pass the test above trivially.
    if not 0.25 < share0 < 0.75:
        print("  FAIL: the render is dominated by one endmember; the checkerboard "
              "should be an even split")
        status = 1

    print("endmember mixing: " + ("PASS" if status == 0 else "FAIL"))
    return status


if __name__ == "__main__":
    sys.exit(main())
