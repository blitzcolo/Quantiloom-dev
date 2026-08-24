#!/usr/bin/env python3
"""Figure 8: the same scene in five bands.

The panels are deliberately not equalised against one another. Each is mapped
by the transform its caption names -- sRGB for the visible, linear automatic
gain control for the four infrared bands -- and the point of the figure is that
the *scene* changes character across the strip while the asset, the camera and
the material spectra do not.

Linear AGC and nothing else in the infrared. CLAHE is tile-local and therefore
not monotone: it inverts 1.902 % of brightness-ordered pixel pairs with a worst
drop of 17 display levels, so a value read off one is not a measurement. (An
earlier draft of this comment said "about half", from a statistic that was
withdrawn as unreproducible; the qualitative point is unchanged.) A figure whose
subject is the reflection-to-emission transition must not be shown through an
operator that reorders brightness. Each panel therefore carries its own scale
bar, in the units it was mapped from -- which is the honest way to say that the
panels are not on a common scale.

Usage:
    e7_figures.py --strip
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(EVIDENCE, FIGURES)


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
    if len(planes) == 3:
        return np.stack(planes, axis=-1)
    # A single-channel image: the thermography inversion writes one plane named
    # "T", in kelvin. Returned two-dimensional so callers can tell it apart from
    # a colour image by ndim rather than by filename.
    if len(channels) == 1:
        return np.asarray(next(iter(channels.values())).pixels, dtype=np.float64)
    raise SystemExit(f"{path}: cannot read channels {sorted(channels)}")


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



# The palettes are the renderer's, transcribed control point for control point
# from src/shaders/clahe.comp.hlsl -- which says why they are ramps rather than
# fitted curves: "the control points are the specification, someone checking a
# palette against a reference reads numbers". matplotlib's viridis is close but
# is not the one Studio shows.
_PALETTES = {
    "viridis": [(0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
                (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
                (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
                (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
    "ironbow": [(0.000, 0.000, 0.000), (0.110, 0.020, 0.260), (0.300, 0.030, 0.430),
                (0.510, 0.060, 0.430), (0.730, 0.170, 0.310), (0.900, 0.350, 0.130),
                (0.990, 0.640, 0.010), (1.000, 1.000, 0.850)],
}


def renderer_colormap(name, samples=256):
    control = np.asarray(_PALETTES[name])
    x = np.linspace(0.0, 1.0, len(control))
    grid = np.linspace(0.0, 1.0, samples)
    return matplotlib.colors.ListedColormap(
        np.stack([np.interp(grid, x, control[:, i]) for i in range(3)], axis=1),
        name=f"quantiloom_{name}")


# Why the bars do not all say kelvin. LWIR is essentially all self-emission, so
# inverting radiance through Planck recovers a temperature and the axis means
# what it says. NIR and SWIR emit nothing at 300 K -- the same inversion returns
# medians near 926 K and 557 K by reporting reflected sunlight as the object's
# own temperature. MWIR is between: a mid-wave camera does report temperature,
# but roughly half of this band is reflected sun here, so an axis in kelvin
# would present an instrument reading as a property of the object.
BAR_UNITS = {
    "NIR": "linear, from radiance",
    "SWIR": "linear, from radiance",
    "MWIR": "linear, from radiance\n(half this band is reflected sun)",
    "LWIR": "linear, from the apparent-temperature inversion",
}

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
    handle = None
    for axis, (hour, image) in zip(axes, frames):
        # Mapped from radiance directly rather than through a normalised copy,
        # so the colour bar below is the actual scale and not a 0-1 proxy.
        handle = axis.imshow(image, cmap="viridis", vmin=lo, vmax=hi)
        # Hours into the forcing file, which starts at midnight of day one.
        axis.set_title(f"t = {hour:g} h  ({hour % 24:g}:00, day {int(hour // 24) + 1})",
                       fontsize=8.5)
        axis.set_xticks([])
        axis.set_yticks([])
    # One bar for the whole strip: the window is shared, so the panels are
    # comparable and a single scale describes all of them.
    bar = figure.colorbar(handle, ax=axes, fraction=0.020, pad=0.012)
    bar.set_label("spectral radiance (W sr$^{-1}$ m$^{-2}$ nm$^{-1}$)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.formatter.set_powerlimits((0, 0))
    figure.suptitle("LWIR, one shared linear window across the sequence",
                    fontsize=8, y=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    # The prose beside this figure quotes a radiance at three of the hours and
    # asserts the vehicle is darker than the sand at exactly one of them. Those
    # are measurements, so measure them here rather than by hand: a number in
    # the manuscript that no committed script recomputes is a number that
    # silently goes stale the next time the frames are re-rendered, and these
    # already did once.
    #
    # The vehicle is isolated without a hand-drawn mask. The painted and
    # unpainted strip renders differ only in the tank's bound reflectance -- the
    # ground is the same material, rendered from the same camera with the same
    # seed, and comes out bit-identical -- so whatever differs between them is
    # the vehicle. Both are tracked beside this manifest, so the mask is
    # reproducible from the repository and not from a scratch render.
    measurements = _diurnal_radiances(frames)
    if measurements:
        (EVIDENCE / "diurnal_measurements.json").write_text(
            json.dumps(measurements, indent=2), encoding="utf-8")
        print(f"wrote {EVIDENCE / 'diurnal_measurements.json'}")

    print(f"wrote {out}  ({len(frames)} frames)")


def _diurnal_radiances(frames):
    """Mean band radiance over ground and vehicle, per frame.

    Returns None when the painted/unpainted pair is absent, since without it
    there is no principled vehicle mask and a guessed one would be worse than
    no number at all.
    """
    paint = EVIDENCE / "strip" / "kv2_paint_vis.exr"
    bare = EVIDENCE / "strip" / "kv2_bare_vis.exr"
    if not (paint.is_file() and bare.is_file()):
        print("  (skipping radiance measurements: run e7_gallery.py --strip first)")
        return None

    luma = np.array([0.2126, 0.7152, 0.0722])

    def grey(path):
        image = read_exr(path)
        return image[..., :3] @ luma if image.ndim == 3 else image

    # 0.005 absolute, not a fraction of the maximum: the VIS frame has a
    # specular highlight near 3.0, so a relative threshold sets the bar three
    # orders above the paint difference and selects nothing.
    mask = np.abs(grey(paint) - grey(bare)) > 0.005

    rows = []
    for hour, image in frames:
        h, w = image.shape
        ys = (np.arange(h) / (h / mask.shape[0])).astype(int).clip(0, mask.shape[0] - 1)
        xs = (np.arange(w) / (w / mask.shape[1])).astype(int).clip(0, mask.shape[1] - 1)
        vehicle = mask[np.ix_(ys, xs)]
        # Ground is everything below the horizon that is not the vehicle;
        # 0.72 of frame height puts the cut under the skyline for this camera.
        ground = (~vehicle) & (np.arange(h)[:, None] > h * 0.72)
        g, v = float(image[ground].mean()), float(image[vehicle].mean())
        rows.append({"time_h": hour, "clock_h": hour % 24,
                     "ground_mean": g, "vehicle_mean": v,
                     "vehicle_over_ground": v / g})

    inverted = [r["clock_h"] for r in rows if r["vehicle_over_ground"] < 1.0]
    return {"source": "evidence/e7/diurnal.json frames",
            "mask": "|paint - bare| > 0.005 on the VIS strip pair",
            "units": "W sr^-1 m^-2 nm^-1, band mean",
            "rows": rows,
            "hours_sand_brighter_than_vehicle": inverted}


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

    by = {(b["variant"], b["band"]): b for b in manifest["bands"] if "variant" in b}
    if not by:
        raise SystemExit("strip.json predates the painted/unpainted pair; re-run "
                         "e7_gallery.py --strip")

    order = ["VIS", "NIR", "SWIR", "MWIR", "LWIR"]
    rows = [("paint", "PAINTED"), ("bare", "UNPAINTED")]

    def scalar(variant, band):
        """The quantity the panel and its bar are both built from.

        LWIR reads the apparent-temperature inversion; everything else reads
        radiance. See BAR_UNITS for why that split is not arbitrary.
        """
        record = by[(variant, band)]
        path = pathlib.Path(record["output"])
        if band == "LWIR":
            tapp = path.with_name(path.stem + "_tapp.exr")
            if tapp.is_file():
                image = read_exr(tapp)
                return (image[..., 0] if image.ndim == 3 else image), "K"
        image = read_exr(path)
        grey = image[..., :3] @ np.array([0.2126, 0.7152, 0.0722]) \
            if image.ndim == 3 else image
        return grey, "radiance"

    # One window per band, shared by both rows, so a difference between the
    # rows is the vehicle and not the scaling.
    windows = {}
    for band in order:
        if band == "VIS":
            continue
        lo, hi = [], []
        for variant, _ in rows:
            values, _unit = scalar(variant, band)
            finite = values[np.isfinite(values)]
            lo.append(np.percentile(finite, 1.0))
            hi.append(np.percentile(finite, 99.0))
        windows[band] = (float(min(lo)), float(max(hi)))

    for palette_name in ("viridis", "ironbow"):
        cmap = renderer_colormap(palette_name)
        figure, axes = plt.subplots(len(rows), len(order),
                                    figsize=(3.05 * len(order), 2.55 * len(rows)))
        for r, (variant, row_label) in enumerate(rows):
            for c, band in enumerate(order):
                axis = axes[r][c]
                if band == "VIS":
                    image = read_exr(pathlib.Path(by[(variant, band)]["output"]))
                    display, out_of_gamut = srgb(image)
                    axis.imshow(display)
                    axis.set_xlabel("sRGB, scene exposure"
                                    + (f"\n{out_of_gamut:.1%} out of gamut"
                                       if out_of_gamut > 0.001 else ""),
                                    fontsize=6.2)
                else:
                    values, unit = scalar(variant, band)
                    lo, hi = windows[band]
                    handle = axis.imshow(values, cmap=cmap, vmin=lo, vmax=hi)
                    bar = figure.colorbar(handle, ax=axis, fraction=0.043, pad=0.015)
                    bar.ax.tick_params(labelsize=5.6)
                    if unit == "K":
                        bar.set_label("apparent temperature (K)", fontsize=5.8)
                    else:
                        bar.set_label("radiance (W sr$^{-1}$ m$^{-2}$ nm$^{-1}$)",
                                      fontsize=5.8)
                        bar.formatter.set_powerlimits((0, 0))
                        bar.update_ticks()
                    axis.set_xlabel(BAR_UNITS[band], fontsize=6.2)
                axis.set_title(f"{row_label}  {band}" if c == 0 else band, fontsize=8)
                axis.set_xticks([]); axis.set_yticks([])

        figure.suptitle(
            "One asset, two surfaces, five bands. Linear mapping throughout; "
            f"{palette_name} palette; window shared between rows within a band.",
            fontsize=7.5, y=0.02)
        figure.tight_layout()
        out = args.out if palette_name == "viridis" else \
            args.out.with_name(args.out.stem + "_ironbow" + args.out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=300, bbox_inches="tight")
        figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
