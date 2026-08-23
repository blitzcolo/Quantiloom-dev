#!/usr/bin/env python3
"""Figure 8: the same scene in five bands.

The panels are deliberately not equalised against one another. Each is mapped
by the transform its caption names -- sRGB for the visible, linear automatic
gain control for the four infrared bands -- and the point of the figure is that
the *scene* changes character across the strip while the asset, the camera and
the material spectra do not.

Linear AGC and nothing else in the infrared. CLAHE inverts about half of all
brightness-ordered pixel pairs, so a temperature cannot be read off one; a
figure whose subject is the reflection-to-emission transition must not be shown
through an operator that reorders brightness. Each panel therefore carries its
own radiance range in the caption, which is the honest way to say that the
panels are not on a common scale.

Usage:
    e7_figures.py --strip
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")


def read_exr(path):
    """The renderer writes RGBA; channels come back name-sorted, so ask by name."""
    import OpenEXR
    handle = OpenEXR.File(str(path))
    channels = handle.channels()
    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)
    # Separate channels: assemble R, G, B by name rather than by position.
    planes = [np.asarray(channels[c].pixels, dtype=np.float64)
              for c in ("R", "G", "B") if c in channels]
    return np.stack(planes, axis=-1)


def linear_agc(image, low=0.5, high=99.5):
    """What a thermal camera calls linear AGC: a percentile window, then a
    straight line onto [0, 1]. Globally monotonic, so brightness order in the
    display is brightness order in the radiance."""
    grey = image[..., 0] if image.ndim == 3 else image
    lo, hi = np.percentile(grey, [low, high])
    if hi <= lo:
        return np.zeros_like(grey), (float(grey.min()), float(grey.max()))
    return np.clip((grey - lo) / (hi - lo), 0.0, 1.0), (float(lo), float(hi))


def srgb(image):
    """Linear radiance to display, exposure-normalised then gamma-encoded."""
    rgb = image[..., :3]
    scale = np.percentile(rgb, 99.5)
    if scale <= 0:
        scale = 1.0
    linear = np.clip(rgb / scale, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strip", action="store_true", default=True)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig8_five_band_strip.png")
    args = parser.parse_args()

    manifest_path = EVIDENCE / "strip.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} missing; run e7_gallery.py --strip first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    panels = []
    for band in manifest["bands"]:
        path = pathlib.Path(band["output"])
        if not path.is_file():
            print(f"  missing {path}, skipping {band['band']}")
            continue
        image = read_exr(path)
        if band["band"] == "VIS":
            display, span = srgb(image), (float(image.min()), float(image.max()))
        else:
            display, span = linear_agc(image)
        panels.append((band["band"], display, span, band["display"]))

    if not panels:
        raise SystemExit("no panels rendered")

    figure, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 2.4))
    if len(panels) == 1:
        axes = [axes]
    for axis, (name, display, span, transform) in zip(axes, panels):
        axis.imshow(display, cmap=None if display.ndim == 3 else "gray",
                    vmin=None if display.ndim == 3 else 0.0,
                    vmax=None if display.ndim == 3 else 1.0)
        axis.set_title(name, fontsize=10)
        # The window each panel was mapped through, so nobody reads the strip
        # as a common scale.
        axis.set_xlabel(f"{transform}\n{span[0]:.3g}\u2013{span[1]:.3g} W/sr/m$^2$",
                        fontsize=6.5)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}  ({len(panels)} panels)")


if __name__ == "__main__":
    main()
