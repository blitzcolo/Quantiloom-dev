#!/usr/bin/env python3
"""Figure 5: the sensor chain's cumulative effect, and why the default AGC is linear.

Two rows over one LWIR render.

The top row adds one stage at a time — bare radiance, plus the Airy-matched
PSF, plus temporal noise, plus fixed-pattern noise with its NUC residual. All
four come from the *same* radiance field, applied by one sensor instance rather
than by four renders: fixed-pattern noise is a property of the detector, and
re-rendering per panel would give each panel a different detector and turn a
cumulative figure into four unrelated ones.

The bottom row is the tone operators, with the statistic that decides which is
the default. Linear AGC is a window and a straight line; Equalize shares one CDF
across all tiles; CLAHE gives each tile its own. The number under each panel is
the fraction of brightness-ordered pixel pairs the operator inverts — sampled,
since 3.16 megapixels is 5 x 10^12 pairs — and it is what a temperature being
readable off an image actually requires.

That row runs on the *radiance field*, not on the chain's output, and the
difference matters. The claim under test is that two pixels at one temperature
must display alike; once temporal noise and a 16-bit quantisation are in the
image they no longer arrive alike, and the measurement would be reporting the
detector rather than the operator. Measured through the full chain the same
statistic reads 0.285 % instead of 1.902 % — quantisation ties are excluded
from an ordered comparison, so the noisier input flatters CLAHE.

The AGC arithmetic is `clahe.comp.hlsl`'s three passes reimplemented in numpy so
that the number can be checked line by line against the shader, and is imported
from e8_agc_monotonicity.py rather than copied.

Usage:
    e5_chain_figure.py --render        # render, apply the chain, draw
    e5_chain_figure.py                 # redraw from what exists
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
LAB = REPO / "build" / "src" / "tools" / "Release" / "sensor_lab.exe"
CONFIG = REPO / "assets" / "configs" / "thermal_solver_lwir.toml"
WORK = REPO / "_convergence_t3" / "e5chain"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e5")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e8_agc_monotonicity import (apply_pass, clipped_cdfs,  # noqa: E402
                                 inverted_pairs, tile_histograms)

# 2048 x 1544 is 3.162 Mpx -- the "3.16-megapixel thermal scene" Section VI
# quotes its inverted-pair statistic on.
RESOLUTION = (2048, 1544)
SEED_PATH = 0x547C
SEED_SENSOR = 0x548C

# The LWIR sensor of Fig. 11 with one deliberate change: 2 ms of integration
# rather than 200 us. At 200 us this scene is photon-starved -- about 280
# electrons per pixel, so Poisson noise alone is 87 % of the scene's own spatial
# modulation and panel three is noise with a picture somewhere inside it. Fig. 11
# measures NETD on a uniform field where that does not matter; a figure whose
# subject is what the chain does to an *image* needs the image to survive it. At
# 2 ms the modulation-to-temporal-noise ratio is 3.4, which is a degraded image
# rather than an absent one, and the caption states the integration time.
SENSOR_BASE = """
[sensor]
enabled = true
focal_length_mm = 50.0
f_number = 2.0
pixel_pitch_um = 12.0
quantum_efficiency = 0.8
integration_time_s = 0.002
well_capacity_e = 20000000.0
read_noise_e_rms = 20.0
bit_depth = 16
gain = 1.0
detector_temperature_k = 300.0
dark_current_e_s = 20000.0
noise_seed = {seed}
psf_sigma_px = {psf}
enable_poisson_noise = {poisson}
enable_read_noise = {read}
enable_dark_current = {dark}
enable_fpn = {fpn}

[sensor.fpn]
prnu_sigma = 0.01
dsnu_sigma_e = 2000.0
enable_nuc = true
nuc_efficiency = 0.97
"""

# Each stage is switched on and the ones before it stay on. `psf_sigma_px = -1`
# is how the model is told to skip the blur entirely.
STAGES = [
    ("bare render", dict(psf=-1.0, poisson="false", read="false", dark="false",
                         fpn="false")),
    ("+ PSF", dict(psf=1.6, poisson="false", read="false", dark="false",
                   fpn="false")),
    ("+ temporal noise", dict(psf=1.6, poisson="true", read="true", dark="true",
                              fpn="false")),
    ("+ FPN and NUC residual", dict(psf=1.6, poisson="true", read="true",
                                    dark="true", fpn="true")),
]


def read_first(path):
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)[..., 0]
    for name in ("R", "Y", "V", "Gray"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)
    array = np.asarray(next(iter(channels.values())).pixels, dtype=np.float64)
    return array[..., 0] if array.ndim == 3 else array


def render_radiance():
    """One LWIR radiance field, the sensor left off — the chain is applied after."""
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "radiance.exr"
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"^resolution\s*=.*$",
                  f"resolution = [{RESOLUTION[0]}, {RESOLUTION[1]}]", text,
                  count=1, flags=re.M)
    text = re.sub(r"^output\s*=.*$",
                  f'output = "{out.relative_to(REPO).as_posix()}"', text,
                  count=1, flags=re.M)
    text = re.sub(r"^spp\s*=.*$", "spp = 64", text, count=1, flags=re.M)
    if re.search(r"^seed\s*=", text, flags=re.M):
        text = re.sub(r"^seed\s*=.*$", f"seed = {SEED_PATH}", text, count=1, flags=re.M)
    else:
        text = text.replace("[renderer]", f"[renderer]\nseed = {SEED_PATH}", 1)

    cfg = WORK / "radiance.toml"
    cfg.write_text(text, encoding="utf-8")
    print(f"rendering {RESOLUTION[0]}x{RESOLUTION[1]} LWIR ...", flush=True)
    proc = subprocess.run([str(CLI), cfg.relative_to(REPO).as_posix()], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=7200)
    if not out.is_file():
        print("\n".join((proc.stdout + proc.stderr).splitlines()[-25:]), file=sys.stderr)
        raise SystemExit("radiance render failed")
    # The relaxation warning is worth surfacing here rather than leaving in a log.
    for line in (proc.stdout + proc.stderr).splitlines():
        if "steady state" in line or "relaxation" in line:
            print("  " + line.split("]")[-1].strip())
    return out


def apply_stage(radiance, label, switches, index):
    cfg = WORK / f"stage{index}.toml"
    cfg.write_text(SENSOR_BASE.format(seed=SEED_SENSOR, **switches), encoding="utf-8")
    prefix = WORK / f"stage{index}"
    proc = subprocess.run(
        [str(LAB), str(cfg), "--input", str(radiance), "--frames", "1",
         "--out-prefix", str(prefix)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=3600)
    out = pathlib.Path(f"{prefix}_mean.exr")
    if not out.is_file():
        print(proc.stdout[-2000:] + proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"sensor_lab failed for {label!r}")
    print(f"  {label}", flush=True)
    return out


def linear_window(values, low=0.5, high=99.5):
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--pairs", type=int, default=20_000_000)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig5_sensor_chain.png")
    args = parser.parse_args()

    # The render and the chain are cached separately: the render is minutes and
    # the four sensor passes are seconds, so a change to the chain must not
    # require the radiance field again, and a stale chain must not survive
    # because the radiance beside it happens to exist.
    radiance = WORK / "radiance.exr"
    if args.render or not radiance.is_file():
        radiance = render_radiance()
    outputs = [pathlib.Path(f"{WORK / f'stage{i}'}_mean.exr") for i in range(len(STAGES))]
    if args.render or not all(path.is_file() for path in outputs):
        print("applying the sensor chain ...", flush=True)
        for index, (label, switches) in enumerate(STAGES):
            apply_stage(radiance, label, switches, index)

    panels = []
    for index, (label, _) in enumerate(STAGES):
        path = pathlib.Path(f"{WORK / f'stage{index}'}_mean.exr")
        if path.is_file():
            panels.append((label, read_first(path)))
    if not panels:
        raise SystemExit(f"no stage outputs under {WORK}; run with --render")

    # Deliberately the radiance field rather than the chain's output. The claim
    # under test is that two pixels at one temperature must display alike, and
    # with temporal noise and a 16-bit quantisation in the image they no longer
    # arrive alike -- the measurement would then be reporting the detector, not
    # the operator. The panels below are the operators applied to that same
    # noiseless field, so the statistic and the picture agree.
    tone_input = read_first(radiance)
    finite = np.where(np.isfinite(tone_input), tone_input, 0.0)
    megapixels = finite.size / 1e6
    lo, hi = float(finite.min()), float(finite.max())
    print(f"\nfinal image {finite.shape[1]} x {finite.shape[0]} = {megapixels:.3f} Mpx, "
          f"radiance {lo:.4g}\u2013{hi:.4g}")

    histograms, bins = tile_histograms(finite, lo, hi)
    rng = np.random.default_rng(SEED_PATH)
    tone = {}
    displays = {}
    for name, display in (
            ("Linear", np.clip((finite - lo) / (hi - lo), 0.0, 1.0)),
            ("Equalize", apply_pass(finite, bins, clipped_cdfs(histograms, True), lo, hi)),
            ("Clahe", apply_pass(finite, bins, clipped_cdfs(histograms, False), lo, hi))):
        fraction, drop = inverted_pairs(finite, display, rng, args.pairs)
        tone[name] = {"inverted_fraction": fraction, "worst_drop": drop,
                      "worst_drop_levels_of_255": drop * 255.0}
        displays[name] = display
        print(f"  {name:9s} inverted {100 * fraction:7.3f} %   "
              f"worst drop {drop * 255:.0f} of 255 display levels")

    figure, axes = plt.subplots(2, 4, figsize=(7.16, 4.5))
    # Each stage reports how much it moved the image, relative to the image's own
    # spatial spread. On a smooth plate a 1.6 px blur is nearly invisible by eye,
    # and a panel a reader cannot distinguish from its neighbour has to say what
    # it did or it is decoration.
    changes = []
    previous = None
    for label, image in panels:
        finite_panel = np.where(np.isfinite(image), image, 0.0)
        if previous is None:
            changes.append(None)
        else:
            spread = float(previous.std()) or 1.0
            changes.append(float(np.sqrt(((finite_panel - previous) ** 2).mean())) / spread)
        previous = finite_panel

    for axis, (label, image), change in zip(axes[0], panels, changes):
        axis.imshow(linear_window(np.where(np.isfinite(image), image, 0.0)),
                    cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(label, fontsize=7.6)
        axis.set_xlabel("radiance field" if change is None
                        else f"RMS change {100 * change:.1f} %\nof the previous panel's spread",
                        fontsize=6.4)
        axis.set_xticks([]); axis.set_yticks([])
    axes[0][0].set_ylabel("cumulative chain\n(linear AGC throughout)", fontsize=7.0)

    for axis, name in zip(axes[1], ("Linear", "Equalize", "Clahe")):
        axis.imshow(displays[name], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name, fontsize=7.6)
        values = tone[name]
        axis.set_xlabel(f"{100 * values['inverted_fraction']:.3f} % inverted\n"
                        f"worst drop {values['worst_drop_levels_of_255']:.0f}/255",
                        fontsize=6.4)
        axis.set_xticks([]); axis.set_yticks([])
    axes[1][0].set_ylabel("tone operators on the\nradiance field (no sensor)", fontsize=7.0)
    axes[1][3].axis("off")
    axes[1][3].text(0.0, 0.5,
                    "Linear AGC is the default\nbecause it is the only\nglobally monotonic "
                    "choice.\n\nTwo pixels at one temperature\nin different CLAHE tiles\n"
                    "display as different greys,\nso a temperature must not\nbe read from "
                    "one.\n\nCLAHE stays available\nfor finding an edge.",
                    ha="left", va="center", fontsize=6.5, color="#333333",
                    linespacing=1.5, transform=axes[1][3].transAxes)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "chain_and_agc.json").write_text(json.dumps(
        {"config": str(CONFIG), "resolution": list(RESOLUTION),
         "megapixels": megapixels, "spp": 64,
         "seed_path": hex(SEED_PATH), "seed_sensor": hex(SEED_SENSOR),
         "pairs_sampled": args.pairs,
         "tone_operator_input": "radiance field, before the sensor chain",
         "stage_rms_change_fraction": changes,
         "stages": [{"label": label, **switches} for label, switches in STAGES],
         "tone_operators": tone}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
