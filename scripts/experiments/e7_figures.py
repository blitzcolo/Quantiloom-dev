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
    """Linear radiance to display, exposure-normalised then gamma-encoded.

    A spectral renderer writing to sRGB primaries produces negative components
    wherever a spectrum falls outside the sRGB gamut -- about 0.7 % of pixels
    here, on the saturated paint and sand. That is gamut clipping, not negative
    radiance, so it is clamped for display and counted for the caption rather
    than quietly folded into a reported range.
    """
    rgb = image[..., :3]
    out_of_gamut = float((rgb < 0.0).any(axis=-1).mean())
    scale = np.percentile(rgb, 99.5)
    if scale <= 0:
        scale = 1.0
    linear = np.clip(rgb / scale, 0.0, 1.0)
    display = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return display, out_of_gamut


def diurnal(out):
    """The thermal shadow moving through the day.

    One shared display window across all frames, unlike the five-band strip:
    here the whole point is that the scene changes between panels, so mapping
    each one to its own range would erase the very thing being shown.
    """
    manifest_path = EVIDENCE / "diurnal.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} missing; run e7_gallery.py --diurnal")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames = []
    for frame in manifest["frames"]:
        path = pathlib.Path(frame["output"])
        if path.is_file():
            image = read_exr(path)
            frames.append((frame["time_h"], image[..., 0] if image.ndim == 3 else image))
    if not frames:
        raise SystemExit("no diurnal frames rendered")

    stacked = np.concatenate([f[1].ravel() for f in frames])
    lo, hi = np.percentile(stacked, [0.5, 99.5])

    figure, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames), 2.3))
    if len(frames) == 1:
        axes = [axes]
    for axis, (hour, image) in zip(axes, frames):
        axis.imshow(np.clip((image - lo) / (hi - lo), 0, 1), cmap="gray",
                    vmin=0.0, vmax=1.0)
        # Hours into the forcing file, which starts at midnight of day one.
        axis.set_title(f"t = {hour:g} h  ({hour % 24:g}:00, day {int(hour // 24) + 1})",
                       fontsize=8.5)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"LWIR, one shared linear AGC window "
                    f"({lo:.4g}–{hi:.4g} W/sr/m$^2$)", fontsize=8, y=0.04)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}  ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diurnal", action="store_true")
    parser.add_argument("--strip", action="store_true", default=True)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig8_five_band_strip.png")
    args = parser.parse_args()

    if args.diurnal:
        diurnal(FIGURES / "fig8b_diurnal_shadow.png")
        return

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
            display, out_of_gamut = srgb(image)
            rgb = image[..., :3]
            span = (max(0.0, float(rgb.min())), float(rgb.max()))
            note = (band["display"] +
                    (f", {out_of_gamut:.1%} out of gamut" if out_of_gamut > 0.001 else ""))
        else:
            display, span = linear_agc(image)
            note = band["display"]
        panels.append((band["band"], display, span, note))

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
