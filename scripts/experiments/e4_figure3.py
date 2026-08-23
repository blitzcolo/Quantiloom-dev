#!/usr/bin/env python3
"""Figure 3: the shadow edge, with and without the tangent.

The two panels are the SAME solve, rendered twice, differing only in whether
`thermal.sun_correction` was on. Both are the apparent-temperature map the
thermography inversion produces, so the colour bar is in kelvin and the two
panels share it — a shared scale is the whole point, since the claim is about
where a temperature boundary lands rather than about how a picture looks.

A difference panel is included because the eye is poor at the comparison the
figure is making: the correction moves a boundary a fraction of a triangle, and
side-by-side panels invite the reader to look for a change in the wrong place.

Usage:
    e4_figure3.py --out figures/fig3_shadow_edge.png
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "validation"))
import mitsuba_common as common  # noqa: E402

EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e4_dtdv")


def crop_to_shadow(image, margin=0.34):
    """The middle of the frame, where the sphere and its shadow are.

    The scene is 120 m of ground and the shadow is about two metres of it; a
    full frame would show a uniform plane with a speck on it.
    """
    h, w = image.shape
    y0, y1 = int(h * margin), int(h * (1.0 - margin))
    x0, x1 = int(w * margin), int(w * (1.0 - margin))
    return image[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\figures"
                                             r"\fig3_shadow_edge.png"))
    args = parser.parse_args()

    raw = crop_to_shadow(common.read_exr(EVIDENCE / "desert_201_raw_fig3_tapp.exr"))
    corrected = crop_to_shadow(
        common.read_exr(EVIDENCE / "desert_201_corr_fig3_tapp.exr"))

    # One scale for both, from the pair, so neither panel is flattered.
    finite = np.concatenate([raw[np.isfinite(raw)], corrected[np.isfinite(corrected)]])
    low, high = np.percentile(finite[finite > 200.0], [1.0, 99.5])

    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))
    for axis, image, title in (
            (axes[0], raw, "(a) per-triangle temperature field"),
            (axes[1], corrected, "(b) with the $dT/dv$ correction")):
        handle = axis.imshow(image, cmap="inferno", vmin=low, vmax=high,
                             interpolation="nearest")
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    difference = corrected - raw
    limit = float(np.percentile(np.abs(difference[np.isfinite(difference)]), 99.8))
    diff_handle = axes[2].imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit,
                                 interpolation="nearest")
    axes[2].set_title("(c) (b) $-$ (a)", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    bar = figure.colorbar(handle, ax=axes[:2], fraction=0.030, pad=0.015)
    bar.set_label("apparent temperature (K)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar2 = figure.colorbar(diff_handle, ax=axes[2], fraction=0.046, pad=0.02)
    bar2.set_label("difference (K)", fontsize=8)
    bar2.ax.tick_params(labelsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}")
    print(f"  shared scale {low:.1f}-{high:.1f} K, "
          f"difference within +/-{limit:.2f} K")


if __name__ == "__main__":
    main()
