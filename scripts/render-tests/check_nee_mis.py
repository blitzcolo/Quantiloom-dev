#!/usr/bin/env python3
"""Check that light sampling and BSDF sampling estimate the same image.

Usage:
    python check_nee_mis.py [--spp 2048] [--resolution 192] [--tol 0.03]

A scene lit by emissive geometry can be rendered two ways. The path can bounce
and happen to land on an emitter, which is what this renderer did exclusively
before next-event estimation existed. Or the shading point can pick a point on
an emitter and trace one shadow ray to it. Both estimate the same integral, so
both must converge to the same image -- and the multiple-importance-sampling
weights that combine them must sum to one at every direction, or the light is
counted 1.5 times or 0.5 times and nobody notices, because a brighter Cornell
box still looks like a Cornell box.

That is what this checks, and it is the only check that can. The furnace suite
and check_sky_equiv are both built on scenes where the traced correction is
identically zero, so light sampling never fires in either; check_color_bleed
asserts that indirect light exists, not that there is the right amount of it.

Two renders of assets/configs/cornell_box_vis.toml:

    renderer.enable_light_sampling = true    both strategies, MIS-weighted
    renderer.enable_light_sampling = false   BSDF sampling alone

and their means must agree. The comparison is on region means rather than per
pixel, because the BSDF-only render is far too noisy per pixel to compare
against anything -- that noise is why light sampling was added. The means
converge much faster than the pixels do, which is exactly what makes them a
usable measure of agreement.

Note the asymmetry in what the tolerance is protecting against. Light sampling
converges roughly thirty times faster here, so at any affordable sample count
the BSDF-only render carries almost all of the disagreement, and it carries it
downward: paths that find the light are rare and bright, so the ones that have
not arrived yet are missing energy rather than adding it. A BSDF-only mean that
comes in slightly low is expected. One that comes in high, or low by more than
the tolerance, is a broken MIS weight.

Exit code 0 = pass, 1 = fail.
"""

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np
import OpenEXR

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
CONFIG = REPO / "assets" / "configs" / "cornell_box_vis.toml"

GPU_ABSENT = re.compile(
    r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")

# Regions to compare, as fractions of the frame: (name, x0, x1, y0, y1).
# The light panel itself is deliberately excluded -- it is emission seen
# directly, which neither strategy samples, so it would dilute the comparison
# with a term that cannot disagree.
REGIONS = (
    ("floor",      0.30, 0.70, 0.70, 0.90),
    ("back wall",  0.30, 0.70, 0.35, 0.60),
    ("left wall",  0.05, 0.20, 0.35, 0.75),
    ("right wall", 0.80, 0.95, 0.35, 0.75),
    ("whole frame", 0.02, 0.98, 0.02, 0.98),
)


def read_rgb(path):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    px = ch[list(ch.keys())[0]].pixels.astype(np.float64)
    if px.ndim != 3 or px.shape[2] < 3:
        raise SystemExit(f"ERROR: expected an RGB image, got shape {px.shape}")
    return px[:, :, :3]


def render(cli, work, tag, *, resolution, spp, light_sampling):
    exr = work / f"{tag}.exr"
    cfg = work / f"{tag}.toml"

    text = CONFIG.read_text()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped.split("=")[0].strip() if "=" in stripped else None
        if key == "resolution":
            lines.append(f"resolution = [{resolution}, {resolution}]")
        elif key == "spp":
            lines.append(f"spp = {spp}")
        elif key == "output":
            lines.append(f'output = "{exr.relative_to(REPO).as_posix()}"')
        else:
            lines.append(line)
        if stripped == "[renderer]":
            lines.append("enable_light_sampling = "
                         f"{'true' if light_sampling else 'false'}")
    cfg.write_text("\n".join(lines) + "\n")

    proc = subprocess.run([str(cli), cfg.relative_to(REPO).as_posix()],
                          cwd=REPO, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if log.count("Saved spectral image") != 1:
        if GPU_ABSENT.search(log):
            print("no usable GPU, nothing measured", file=sys.stderr)
            sys.exit(3)
        print(f"ERROR: render failed for {tag}", file=sys.stderr)
        print("\n".join(log.splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)
    return exr


def region_mean(img, box):
    h, w = img.shape[:2]
    _, x0, x1, y0, y1 = box
    patch = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1), :]
    return float(np.mean(patch))


def main():
    p = argparse.ArgumentParser(description="NEE / BSDF agreement check")
    p.add_argument("--spp", type=int, default=2048)
    p.add_argument("--resolution", type=int, default=192)
    p.add_argument("--tol", type=float, default=0.03,
                   help="allowed relative disagreement in a region mean")
    p.add_argument("--work-dir", default=str(REPO / "_convergence"))
    p.add_argument("--cli", default=str(DEFAULT_CLI))
    args = p.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"no CLI at {cli} -- build first", file=sys.stderr)
        sys.exit(2)
    if not CONFIG.exists():
        print(f"ERROR: {CONFIG} not found", file=sys.stderr)
        sys.exit(1)

    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    print(f"rendering {args.spp} spp at {args.resolution}x{args.resolution}, "
          f"both strategies ...", flush=True)
    mis = read_rgb(render(cli, work, "nee_mis_on", resolution=args.resolution,
                          spp=args.spp, light_sampling=True))
    bsdf = read_rgb(render(cli, work, "nee_mis_off", resolution=args.resolution,
                           spp=args.spp, light_sampling=False))

    if mis.shape != bsdf.shape:
        print(f"ERROR: shapes differ, {mis.shape} vs {bsdf.shape}",
              file=sys.stderr)
        sys.exit(1)

    print()
    print(f"{'region':<12} {'MIS':>12} {'BSDF only':>12} {'diff':>9}")
    worst = 0.0
    failed = False
    for box in REGIONS:
        a = region_mean(mis, box)
        b = region_mean(bsdf, box)
        rel = abs(a - b) / a if a > 0 else (0.0 if b == 0 else float("inf"))
        worst = max(worst, rel)
        flag = "" if rel <= args.tol else "  FAIL"
        if rel > args.tol:
            failed = True
        print(f"{box[0]:<12} {a:12.6g} {b:12.6g} {rel:8.2%}{flag}")

    print()
    print(f"worst region disagreement: {worst:.2%} (tolerance {args.tol:.0%})")
    if failed:
        print("FAIL: the two strategies do not agree -- the MIS weights are "
              "not partitioning the light")
        return 1
    print("nee/mis agreement PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
