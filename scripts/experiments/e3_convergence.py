#!/usr/bin/env python3
"""Property (iii) of Section VIII-G: convergence against wall-clock time.

The first two interactivity properties are latencies -- how long one action
takes. The third is not: progressive rendering is a claim about how fast the
image stops changing, and the honest way to state it is an error curve against
seconds rather than a single "converges in N spp".

Time here is the renderer's own `Total GPU time`, which excludes scene load and
acceleration-structure build. That is the right axis for a progressive claim --
those costs are paid once when the scene opens, not per frame of accumulation --
and the setup time is reported separately rather than folded in, because a
reader deciding whether this is interactive needs both numbers and they answer
different questions.

RMSE is the relative root-mean-square error against a high-sample reference of
the same scene, the same definition Section VIII-B uses. The reference carries
its own noise, about 1/sqrt(reference_spp) of it, so the curve is drawn only
down to where it stops measuring the render and starts measuring the reference;
the floor is marked rather than crossed silently.

The reference uses a different seed from every test render. Sharing one would
make the first N samples of the reference literally the same samples being
measured against it, and the error would fall faster than the estimator does.

Usage:
    e3_convergence.py
    e3_convergence.py --scenes cornell --spps 16,64,256
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
WORK = REPO / "_convergence_t3"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e3")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(EVIDENCE, FIGURES)

sys.path.insert(0, str(REPO / "scripts" / "render-tests"))
from measure_convergence import patch_config  # noqa: E402

# Both derived from 0x547C, as everywhere else in this programme; they only have
# to differ from each other and from the default.
SEED_REFERENCE = 0x547C + 1
SEED_TEST = 0x547C + 2

GPU_TIME = re.compile(r"Total GPU time:\s+([\d.]+) ms")

SCENES = {
    "cornell": dict(
        config="assets/configs/cornell_box_vis.toml",
        label="Cornell box, VIS",
        overrides={}),
    "kv2_vis": dict(
        config="assets/configs/gallery/kv2_desert_lwir.toml",
        label="KV-2 desert, VIS",
        overrides={"spectral.mode": "vis_fused", "spectral.band": "VIS"}),
    "kv2_lwir": dict(
        config="assets/configs/gallery/kv2_desert_lwir.toml",
        label="KV-2 desert, LWIR",
        overrides={}),
}


def read_rgb(path):
    """Channels come back name-sorted, so ask for the group rather than take
    whichever key happens to sort first."""
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)[..., :3]
    planes = [np.asarray(channels[c].pixels, dtype=np.float64)
              for c in ("R", "G", "B") if c in channels]
    if len(planes) == 3:
        return np.stack(planes, axis=-1)
    single = np.asarray(next(iter(channels.values())).pixels, dtype=np.float64)
    return np.repeat(single[..., None], 3, axis=-1) if single.ndim == 2 else single


def apply_overrides(text, overrides):
    """Set dotted keys the config already declares, in place."""
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        lines, current, done = text.splitlines(), None, False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current = stripped[1:-1]
            elif current == section and stripped.split("=")[0].strip() == key:
                lines[index] = f'{key} = "{value}"'
                done = True
        if not done:
            raise SystemExit(f"{dotted} not present in the config to override")
        text = "\n".join(lines) + "\n"
    return text


def render(scene, tag, spp, seed, resolution):
    exr = WORK / f"{tag}.exr"
    cfg = WORK / f"{tag}.toml"
    text = (REPO / scene["config"]).read_text(encoding="utf-8")
    text = apply_overrides(text, scene["overrides"])
    cfg.write_text(patch_config(text, resolution=resolution, spp=spp, seed=seed,
                                output=exr.relative_to(REPO).as_posix()),
                   encoding="utf-8")

    started = time.perf_counter()
    proc = subprocess.run([str(CLI), cfg.relative_to(REPO).as_posix()], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    wall = time.perf_counter() - started
    log = proc.stdout + proc.stderr
    if not exr.exists():
        print("\n".join(log.splitlines()[-20:]), file=sys.stderr)
        raise SystemExit(f"render failed for {tag}")

    match = GPU_TIME.search(log)
    gpu_ms = float(match.group(1)) if match else None
    return exr, gpu_ms, wall


def relative_rmse(image, reference):
    mean = float(reference.mean())
    if mean <= 0:
        return float("nan")
    return float(np.sqrt(((image - reference) ** 2).mean())) / mean


def measure(scene, key, spps, reference_spp, resolution):
    print(f"\n{scene['label']}  ({resolution}x{resolution})", flush=True)
    print(f"  reference at {reference_spp} spp ...", flush=True)
    ref_path, ref_gpu, ref_wall = render(scene, f"{key}_ref", reference_spp,
                                         SEED_REFERENCE, resolution)
    reference = read_rgb(ref_path)
    # Scene open plus acceleration build: the whole invocation minus the part
    # the renderer attributes to tracing.
    setup_s = ref_wall - (ref_gpu or 0.0) / 1000.0
    print(f"    {ref_gpu / 1000:.1f} s of trace, {setup_s:.1f} s of setup", flush=True)

    rows = []
    for spp in spps:
        path, gpu_ms, wall = render(scene, f"{key}_{spp}", spp, SEED_TEST, resolution)
        rmse = relative_rmse(read_rgb(path), reference)
        rows.append({"spp": spp, "gpu_ms": gpu_ms, "wall_s": wall,
                     "relative_rmse": rmse,
                     # Flat means the estimator is converging as 1/sqrt(n);
                     # rising means something is biased and samples will not fix it.
                     "rmse_root_spp": rmse * np.sqrt(spp)})
        print(f"    {spp:>5} spp   {gpu_ms / 1000:7.3f} s   RMSE {100 * rmse:6.3f} %",
              flush=True)

    # The reference's own relative RMSE, extrapolated from this scene's measured
    # constant rather than assumed. 1/sqrt(N) is the *shape*; the constant is
    # what differs between a Cornell box and an LWIR desert by two orders of
    # magnitude, and using the shape alone puts all three floors on one line.
    floor = (rows[-1]["rmse_root_spp"] / np.sqrt(reference_spp)) if rows else None
    return {"label": scene["label"], "config": scene["config"],
            "resolution": resolution, "reference_spp": reference_spp,
            "reference_gpu_s": (ref_gpu or 0.0) / 1000.0,
            "setup_s": setup_s,
            "seed_reference": SEED_REFERENCE, "seed_test": SEED_TEST,
            "reference_noise_floor_relative": floor,
            "ms_per_sample": (rows[-1]["gpu_ms"] / rows[-1]["spp"]) if rows else None,
            "rows": rows}


def crossing(rows, threshold):
    """Trace seconds at which RMSE first falls below the threshold, by log-log
    interpolation between the bracketing measurements.

    Returns (seconds, status). A scene already under the threshold at the
    cheapest sample count measured has not "failed to reach" it -- reporting
    those two cases with one None is how a fast scene gets recorded as a slow
    one.
    """
    if rows and rows[0]["relative_rmse"] <= threshold:
        return rows[0]["gpu_ms"] / 1000.0, "already below at the first sample count"
    for previous, current in zip(rows, rows[1:]):
        if previous["relative_rmse"] > threshold >= current["relative_rmse"]:
            x0, x1 = np.log(previous["gpu_ms"]), np.log(current["gpu_ms"])
            y0, y1 = np.log(previous["relative_rmse"]), np.log(current["relative_rmse"])
            if y1 == y0:
                return current["gpu_ms"] / 1000.0, "interpolated"
            t = (np.log(threshold) - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0)) / 1000.0), "interpolated"
    return None, "not reached in the measured range"


def figure(results, out, threshold):
    colours = ["#2f4b7c", "#a05195", "#c2427a"]
    figure_, axis = plt.subplots(figsize=(4.2, 3.0))

    for index, result in enumerate(results):
        seconds = [r["gpu_ms"] / 1000.0 for r in result["rows"]]
        rmse = [100 * r["relative_rmse"] for r in result["rows"]]
        axis.plot(seconds, rmse, marker="o", ms=4, lw=1.5, color=colours[index % 3],
                  label=result["label"], zorder=4)
        axis.axhline(100 * result["reference_noise_floor_relative"],
                     color=colours[index % 3], lw=0.7, ls=":", alpha=0.55, zorder=2)
        at, status = crossing(result["rows"], threshold)
        result["seconds_to_threshold"] = at
        result["threshold_status"] = status
        if at and status == "interpolated":
            axis.plot([at], [100 * threshold], marker="v", ms=6,
                      color=colours[index % 3], zorder=5)

    axis.axhline(100 * threshold, color="#b03030", lw=1.1, ls="--", zorder=3)
    axis.text(0.985, 100 * threshold * 1.12, f"{100 * threshold:g} % RMSE",
              transform=axis.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=6.8, color="#b03030")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("trace time (s, log) \u2014 scene open excluded", fontsize=8)
    axis.set_ylabel("relative RMSE (%, log)", fontsize=8)
    axis.tick_params(labelsize=7.5)
    axis.grid(which="both", color="0.92", lw=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(fontsize=7, frameon=False, loc="lower left")
    axis.set_title("Dotted: each scene's own reference noise floor, which no curve reaches.\n"
                   "Markers: where 2 % is crossed — the LWIR render begins below it.",
                   fontsize=7.4)
    figure_.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure_.savefig(out, dpi=300, bbox_inches="tight")
    figure_.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure_)
    print(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default="cornell,kv2_vis,kv2_lwir")
    parser.add_argument("--spps", default="8,16,32,64,128,256,512,1024")
    parser.add_argument("--reference-spp", type=int, default=4096)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--target-rmse", type=float, default=0.02)
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved evidence, rendering nothing")
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig9b_convergence.png")
    args = parser.parse_args()

    if args.replot:
        saved = json.loads((EVIDENCE / "convergence.json").read_text(encoding="utf-8"))
        results = saved["scenes"]
        for result in results:
            result["reference_noise_floor_relative"] = (
                result["rows"][-1]["rmse_root_spp"] / np.sqrt(result["reference_spp"]))
        figure(results, args.out, args.target_rmse)
        (EVIDENCE / "convergence.json").write_text(json.dumps(
            {"target_rmse": args.target_rmse, "scenes": results}, indent=2),
            encoding="utf-8")
        for result in results:
            at, status = result.get("seconds_to_threshold"), result.get("threshold_status")
            reached = f"2 % at {at:.3f} s ({status})" if at else f"2 % {status}"
            print(f"  {result['label']:<20} {result['ms_per_sample']:6.2f} ms/sample   "
                  f"setup {result['setup_s']:5.1f} s   {reached}")
        return

    if not CLI.is_file():
        raise SystemExit(f"{CLI} not built")
    WORK.mkdir(parents=True, exist_ok=True)
    spps = [int(s) for s in args.spps.split(",")]

    results = []
    for key in args.scenes.split(","):
        if key not in SCENES:
            raise SystemExit(f"unknown scene {key!r}; have {sorted(SCENES)}")
        results.append(measure(SCENES[key], key, spps, args.reference_spp,
                               args.resolution))

    figure(results, args.out, args.target_rmse)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "convergence.json").write_text(json.dumps(
        {"target_rmse": args.target_rmse, "scenes": results}, indent=2),
        encoding="utf-8")
    print()
    for result in results:
        at, status = result.get("seconds_to_threshold"), result.get("threshold_status")
        reached = f"2 % at {at:.3f} s ({status})" if at else f"2 % {status}"
        print(f"  {result['label']:<20} {result['ms_per_sample']:6.2f} ms/sample   "
              f"setup {result['setup_s']:5.1f} s   {reached}")


if __name__ == "__main__":
    main()
