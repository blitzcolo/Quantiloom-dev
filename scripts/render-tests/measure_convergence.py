#!/usr/bin/env python3
"""Measure how fast a scene converges, in samples rather than in seconds.

Usage:
    python measure_convergence.py assets/configs/cornell_box_vis.toml
    python measure_convergence.py <config.toml> [--spps 16,64,256]
                                  [--reference-spp 4096] [--resolution 256]
                                  [--target-rmse 0.02] [--refresh-reference]

Every other script in this directory asks whether a render is *correct*. This
one asks how many samples correctness costs, which is the question an
optimisation has to answer and none of them can: a change that halves the
variance and a change that halves the time per sample both look like nothing to
a pass/fail check, and a change that quietly biases the estimator looks like an
improvement to a stopwatch.

Method. Render the same scene at a high sample count to serve as ground truth,
then at each of several lower counts, and report

    relative RMSE = sqrt(mean((I - I_ref)^2)) / mean(I_ref)

against it. Pure Monte Carlo error falls as 1/sqrt(spp), so the product
`rel RMSE * sqrt(spp)` is the interesting column: flat means the estimator is
converging normally and the constant is what an optimisation moves; rising
means something is biased and no number of samples will fix it. That is the
same discriminator check_hero_wavelength.py uses, applied to the whole image.

The reference carries its own noise -- about 1/sqrt(reference_spp) of it -- so
readings below roughly a third of that are measuring the reference, not the
render. The script says so when it happens rather than reporting a number it
does not believe.

The reference is cached under the work directory and keyed by config, geometry
and sample count, so repeated runs during an optimisation pay for it once.
Deliberately a different RNG seed from the test renders: sharing a seed would
make the first N samples of the reference literally the same samples being
measured against it.

Exit code 0 = the measurement completed, 1 = it could not be made. This is an
instrument, not a gate -- it does not fail on a number being large.
"""

import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import OpenEXR

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"

# Distinct seeds: see the docstring. Both are derived from the project's default
# sampling seed (0x547C) so that every number in the paper traces to one root;
# they only have to differ from each other and from the default itself.
SEED_REFERENCE = 0x547C + 1
SEED_TEST = 0x547C + 2

GPU_ABSENT = re.compile(
    r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")


def read_rgb(path):
    """The first three channels of an EXR, as float64 HxWx3."""
    f = OpenEXR.File(str(path))
    ch = f.channels()
    px = ch[list(ch.keys())[0]].pixels.astype(np.float64)
    if px.ndim != 3 or px.shape[2] < 3:
        raise SystemExit(f"ERROR: expected an RGB image, got shape {px.shape}")
    return px[:, :, :3]


def patch_config(text, *, resolution, spp, seed, output):
    """Rewrite the [renderer] keys this script controls, adding any that are
    absent. Everything else in the config is left exactly as written -- the
    point is to measure the scene the user has, not a reconstruction of it."""
    replacements = {
        "resolution": f"resolution = [{resolution}, {resolution}]",
        "spp": f"spp = {spp}",
        "seed": f"seed = {seed}",
        "output": f'output = "{output}"',
    }
    lines = text.splitlines()
    section = None
    seen = set()
    out = []
    renderer_end = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == "renderer":
                renderer_end = len(out)
            section = stripped[1:-1]
        if section == "renderer":
            key = stripped.split("=")[0].strip() if "=" in stripped else None
            if key in replacements:
                out.append(replacements[key])
                seen.add(key)
                continue
        out.append(line)
    if section == "renderer":
        renderer_end = len(out)
    if renderer_end is None:
        raise SystemExit("ERROR: config has no [renderer] section")
    missing = [v for k, v in replacements.items() if k not in seen]
    out[renderer_end:renderer_end] = missing
    return "\n".join(out) + "\n"


def render(cli, config_text, work, tag, *, resolution, spp, seed):
    """Render one image, returning its path. Raises SystemExit on failure."""
    exr = work / f"{tag}.exr"
    cfg = work / f"{tag}.toml"
    # Paths are relative to the repo root because that is this process's CWD
    # and the child inherits it -- a Windows binary cannot resolve a WSL path.
    rel_exr = exr.relative_to(REPO).as_posix()
    cfg.write_text(patch_config(config_text, resolution=resolution, spp=spp,
                                seed=seed, output=rel_exr))

    proc = subprocess.run([str(cli), cfg.relative_to(REPO).as_posix()],
                          cwd=REPO, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if log.count("Saved spectral image") != 1:
        if GPU_ABSENT.search(log):
            print("no usable GPU, nothing measured", file=sys.stderr)
            sys.exit(1)
        print(f"ERROR: render failed for {tag}", file=sys.stderr)
        print("\n".join(log.splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)
    if not exr.exists():
        raise SystemExit(f"ERROR: {exr} was reported saved but is not there")
    return exr


def main():
    p = argparse.ArgumentParser(description="Convergence rate of a scene")
    p.add_argument("config")
    p.add_argument("--spps", default="16,64,256",
                   help="comma-separated sample counts to measure")
    p.add_argument("--reference-spp", type=int, default=4096)
    p.add_argument("--resolution", type=int, default=256,
                   help="square render size; overrides the config")
    p.add_argument("--target-rmse", type=float, default=0.02,
                   help="the 'looks converged' threshold the spp estimate uses")
    p.add_argument("--work-dir", default=str(REPO / "_convergence"))
    p.add_argument("--refresh-reference", action="store_true")
    p.add_argument("--cli", default=str(DEFAULT_CLI))
    args = p.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"no CLI at {cli} -- build first", file=sys.stderr)
        sys.exit(1)

    config = pathlib.Path(args.config)
    if not config.exists():
        print(f"ERROR: {config} not found", file=sys.stderr)
        sys.exit(1)
    config_text = config.read_text()

    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not work.resolve().is_relative_to(REPO):
        print(f"ERROR: work dir must be inside {REPO} -- the Windows CLI "
              f"cannot write to a WSL path", file=sys.stderr)
        sys.exit(1)

    spps = sorted(int(s) for s in args.spps.split(","))
    if spps[-1] >= args.reference_spp:
        print(f"ERROR: every measured spp must be below the reference "
              f"({args.reference_spp})", file=sys.stderr)
        sys.exit(1)

    # The reference depends on the scene and the geometry, not on the renderer
    # revision -- refresh it by hand after a change that alters the converged
    # image, which is exactly the change this script is not measuring.
    key = hashlib.sha256(
        f"{config.resolve()}|{args.resolution}|{args.reference_spp}"
        .encode()).hexdigest()[:12]
    ref_exr = work / f"reference_{key}.exr"
    if args.refresh_reference and ref_exr.exists():
        ref_exr.unlink()
    if ref_exr.exists():
        print(f"reference: cached ({args.reference_spp} spp, "
              f"{args.resolution}x{args.resolution})")
    else:
        print(f"reference: rendering {args.reference_spp} spp at "
              f"{args.resolution}x{args.resolution} ...", flush=True)
        produced = render(cli, config_text, work, f"reference_{key}_tmp",
                          resolution=args.resolution, spp=args.reference_spp,
                          seed=SEED_REFERENCE)
        shutil.move(produced, ref_exr)

    ref = read_rgb(ref_exr)
    ref_mean = float(np.mean(ref))
    if ref_mean <= 0.0:
        print("ERROR: the reference render is black -- nothing to converge to",
              file=sys.stderr)
        sys.exit(1)

    # The reference's own residual noise, from its sample count and the
    # constant the test renders reveal. Filled in after the first reading.
    ref_noise = None

    print(f"scene:     {config}")
    print(f"reference: mean radiance {ref_mean:.6g}")
    print()
    print("      spp    rel RMSE    x sqrt(spp)")
    readings = []
    for spp in spps:
        exr = render(cli, config_text, work, f"test_{spp}",
                     resolution=args.resolution, spp=spp, seed=SEED_TEST)
        img = read_rgb(exr)
        if img.shape != ref.shape:
            print(f"ERROR: {spp} spp rendered {img.shape}, reference is "
                  f"{ref.shape}", file=sys.stderr)
            sys.exit(1)
        rmse = float(np.sqrt(np.mean((img - ref) ** 2))) / ref_mean
        readings.append((spp, rmse))
        if ref_noise is None:
            ref_noise = rmse * np.sqrt(spp) / np.sqrt(args.reference_spp)
        flag = "  <- at the reference's own noise floor" \
               if rmse < 3.0 * ref_noise else ""
        print(f"  {spp:7d}    {rmse:8.4%}    {rmse * np.sqrt(spp):10.4f}{flag}")

    # Fit the 1/sqrt(spp) constant on the reading least polluted by the
    # reference's noise, which is always the noisiest -- the lowest spp.
    spp0, rmse0 = readings[0]
    constant = rmse0 * np.sqrt(spp0)
    needed = (constant / args.target_rmse) ** 2

    products = [r * np.sqrt(s) for s, r in readings]
    drift = max(products) / min(products) if min(products) > 0 else float("inf")

    print()
    print(f"1/sqrt(spp) constant: {constant:.4f}")
    print(f"spp for {args.target_rmse:.1%} rel RMSE: {needed:.0f}")
    print(f"product spread across the range: {drift:.2f}x "
          f"({'flat -- noise only' if drift < 1.35 else 'NOT flat -- suspect bias or a reference floor'})")
    print()
    print("Lower constant = fewer samples for the same image. The spread is a "
          "sanity check, not a verdict:\nnoise alone holds it near 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
