#!/usr/bin/env python3
"""The first-bounce stratification A/B behind Section VIII-B.

Builds the shaders twice -- once with the six padded Owen-scrambled Sobol' slots
and once with QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE, which routes those same
slots to the PCG stream the deeper bounces already use -- and measures the
samples per pixel each needs to reach a target RMSE on the same scene.

The two arms differ in the sampler and nothing else: same seed, same slot
decomposition, same sample index, same scene, same reference. That is the whole
point; a speed-up measured against a differently configured build is not a
sampler result.

This has to be re-run rather than inherited. The number in the submitted draft
was measured on `cornell_box_vis` while three defects meant its materials were
shading from their base colours instead of their bound reflectance curves (see
Section VIII-D), so it describes a scene the renderer no longer produces.

Usage:
    e1b_stratification.py                       # both arms, restores the build
    e1b_stratification.py --arm stratified      # one arm only
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e1b")
CONFIG = "assets/configs/cornell_box_vis.toml"
OPTION = "QUANTILOOM_UNSTRATIFIED_FIRST_BOUNCE"


def wsl(command, timeout=1800):
    """Run a command in the WSL toolchain shell, which is what drives cmake."""
    full = f'export PATH="$HOME/miniconda3/bin:$PATH" && cd /mnt/h/Quantiloom-dev && {command}'
    return subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc", full],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def set_option(enabled):
    """Reconfigure and rebuild the shaders with the sampler option set."""
    value = "ON" if enabled else "OFF"
    result = wsl(f"cmake.exe -S . -B build -D{OPTION}={value} >/dev/null 2>&1 && "
                 f"cmake.exe --build build --config Release "
                 f"--target CompileShaders CompileComputeShaders -j")
    if result.returncode != 0:
        print(result.stdout + result.stderr, file=sys.stderr)
        raise SystemExit(f"failed to build shaders with {OPTION}={value}")
    print(f"  shaders rebuilt with {OPTION}={value}", flush=True)


SPP_LINE = re.compile(r"(\d+)\s+spp.*?([\d.]+)\s*%")
ESTIMATE = re.compile(r"(?:reach|below)\D*([\d.]+)\s*%[^\n]*?(\d[\d,]*)\s*spp", re.I)


def measure(label, args):
    """One arm: run the convergence instrument and keep its whole report."""
    command = [sys.executable, str(REPO / "scripts" / "render-tests" / "measure_convergence.py"),
               CONFIG, "--spps", args.spps, "--reference-spp", str(args.reference_spp),
               "--resolution", str(args.resolution), "--target-rmse", str(args.target_rmse),
               # A fresh reference per arm would compare each arm against its own
               # noise. Both arms share one, which is why the work dir is shared.
               "--work-dir", str(EVIDENCE / "work")]
    print(f"  measuring [{label}] ...", flush=True)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=14400)
    print(result.stdout[-2000:] if result.stdout else "", flush=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"convergence measurement failed for {label}")

    rows = [{"spp": int(m.group(1)), "rel_rmse_pct": float(m.group(2))}
            for m in SPP_LINE.finditer(result.stdout)]
    estimate = ESTIMATE.search(result.stdout)
    return {
        "arm": label,
        "rows": rows,
        "spp_to_target": int(estimate.group(2).replace(",", "")) if estimate else None,
        "stdout": result.stdout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["stratified", "white-noise", "both"],
                        default="both")
    parser.add_argument("--spps", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--reference-spp", type=int, default=16384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--target-rmse", type=float, default=0.02)
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    arms = ["stratified", "white-noise"] if args.arm == "both" else [args.arm]
    results = {}

    try:
        for arm in arms:
            set_option(arm == "white-noise")
            results[arm] = measure(arm, args)
    finally:
        # The shipping configuration is stratified. Leaving a measurement build
        # in the tree would silently change every later render in this session.
        set_option(False)
        print("  build restored to the shipping sampler", flush=True)

    payload = {"config": CONFIG, "resolution": args.resolution,
               "reference_spp": args.reference_spp,
               "target_rmse_pct": 100.0 * args.target_rmse,
               "arms": {k: {kk: vv for kk, vv in v.items() if kk != "stdout"}
                        for k, v in results.items()}}

    both = all(results.get(a, {}).get("spp_to_target") for a in ("stratified", "white-noise"))
    if both:
        gain = results["white-noise"]["spp_to_target"] / results["stratified"]["spp_to_target"]
        payload["sampling_efficiency_gain"] = gain
        print(f"\n  white noise  {results['white-noise']['spp_to_target']:,} spp"
              f"\n  stratified   {results['stratified']['spp_to_target']:,} spp"
              f"\n  gain         {gain:.2f}x")
    else:
        print("\n  one or both arms did not report an spp-to-target; see the logs",
              file=sys.stderr)

    (EVIDENCE / "stratification.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    for arm, result in results.items():
        (EVIDENCE / f"log_{arm.replace('-', '_')}.txt").write_text(result["stdout"],
                                                                   encoding="utf-8")
    print(f"  wrote {EVIDENCE / 'stratification.json'}")


if __name__ == "__main__":
    main()
