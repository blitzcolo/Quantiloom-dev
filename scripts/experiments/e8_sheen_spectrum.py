#!/usr/bin/env python3
"""Figure 2: what an RGB sheen colour becomes outside the band it was fitted in.

Section IV's out-of-band prohibition rests on a concrete failure, and this draws
it. The SheenChair asset's mango velvet carries `sheenColorFactor` =
(1.0, 0.329, 0.1), a warm saturated colour. Jakob & Hanika fit it as a sigmoid
of a quadratic in wavelength over 380-780 nm, which is exact inside that range
and says nothing outside it -- but the quadratic does not stop, and for this
colour it grows, so the sigmoid saturates to one. A velvet becomes a mirror,
Kirchhoff drops its emissivity to zero, and an LWIR render loses the very
radiation it exists to measure.

The arithmetic here is the shader's, line for line: the same trilinear fetch
from the same 64^3 table the renderer binds at binding 25, the same closed-form
double smoothstep inverse, the same sigmoid. The one deliberate difference is
that `RgbSpectrumAt` clamps lambda to the fit domain and this does not -- the
clamp is the guard rail the convention puts in front of the trap, and a figure
about the trap has to show what is behind it.

The peacock velvet from the same asset is drawn beside it as the control. It is
cool rather than warm, its leading coefficient has the other sign, and it
saturates to zero instead: the failure is a property of the colour, not of every
extrapolation, which is why the convention prohibits the operation rather than
correcting it.

Usage:
    e8_sheen_spectrum.py
"""

import argparse
import json
import pathlib
import sys
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TABLE = REPO / "assets" / "luts" / "rgb2spec_srgb_64.bin"
CHAIR = (REPO / "assets" / "models" / "glTF-Sample-Assets" / "Models" /
         "SheenChair" / "glTF" / "SheenChair.gltf")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e8")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(FIGURES, EVIDENCE)

LAMBDA_MIN = 380.0     # RGB2SPEC_LAMBDA_MIN
LAMBDA_RANGE = 400.0   # RGB2SPEC_LAMBDA_RANGE, so the fit ends at 780 nm
RES = 64               # kRgbToSpectrumResolution
CACHE_MAGIC = 4       # header is (magic, version, resolution, coeffCount)


def load_table(path):
    """The renderer's own cache file: a four-u32 header, then 3*res^3 triples."""
    raw = path.read_bytes()
    magic, version, resolution, coeffs = struct.unpack_from("<4I", raw, 0)
    if resolution != RES or coeffs != 3:
        raise SystemExit(f"{path}: unexpected header {resolution=} {coeffs=}")
    data = np.frombuffer(raw, dtype="<f4", offset=4 * CACHE_MAGIC)
    expected = 3 * RES ** 3 * 3
    if data.size != expected:
        raise SystemExit(f"{path}: {data.size} floats, expected {expected}")
    # (maxc, z, y, x, coefficient), matching RgbToSpectrumTable::Fetch's index.
    return data.reshape(3, RES, RES, RES, 3), version


def inv_smoothstep(v):
    """smoothstep(u) = v has the exact root below; the table warps z twice."""
    return 0.5 - np.sin(np.arcsin(np.clip(1.0 - 2.0 * v, -1.0, 1.0)) / 3.0)


def fetch(table, rgb):
    """RgbToSpectrumTable::Fetch, in numpy. Returns (c0, c1, c2)."""
    r, g, b = (float(np.clip(c, 0.0, 1.0)) for c in rgb)
    if r == g == b:
        raise SystemExit("achromatic colours are carried, not fitted")

    comp = (r, g, b)
    maxc = 0 if (r > g and r > b) else (1 if g > b else 2)
    z = comp[maxc]
    res1 = float(RES - 1)

    xf = comp[(maxc + 1) % 3] / z * res1
    yf = comp[(maxc + 2) % 3] / z * res1
    zf = float(np.clip(inv_smoothstep(inv_smoothstep(z)), 0.0, 1.0)) * res1

    xi = min(int(xf), RES - 2)
    yi = min(int(yf), RES - 2)
    zi = min(int(zf), RES - 2)
    dx, dy, dz = xf - xi, yf - yi, zf - zi

    out = np.zeros(3)
    for a in (0, 1):
        for bb in (0, 1):
            for cc in (0, 1):
                w = (dz if a else 1 - dz) * (dy if bb else 1 - dy) * (dx if cc else 1 - dx)
                out += w * table[maxc, zi + a, yi + bb, xi + cc]
    return out


def reflectance(coeffs, lambda_nm, clamp):
    """RgbSpectrumAt. `clamp` selects the shipping guard rail or the raw fit."""
    t = (np.asarray(lambda_nm, dtype=np.float64) - LAMBDA_MIN) / LAMBDA_RANGE
    if clamp:
        t = np.clip(t, 0.0, 1.0)
    x = (coeffs[0] * t + coeffs[1]) * t + coeffs[2]
    return 0.5 + 0.5 * x / np.sqrt(1.0 + x * x)


def sheen_colours():
    materials = json.loads(CHAIR.read_text(encoding="utf-8"))["materials"]
    found = {}
    for m in materials:
        ext = m.get("extensions", {}).get("KHR_materials_sheen")
        if ext and "sheenColorFactor" in ext:
            found[m.get("name", "?")] = ext["sheenColorFactor"]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig2_sheen_upsampling.png")
    args = parser.parse_args()

    table, version = load_table(TABLE)
    colours = sheen_colours()
    print(f"{TABLE.name}: version {version}, {RES}^3 x 3 sub-tables")
    for name, rgb in colours.items():
        print(f"  {name}: {rgb}")

    # Warm first so it draws on top of its control.
    order = [n for n in colours if "Mango" in n] + [n for n in colours if "Mango" not in n]
    styles = {0: dict(colour="#c1440e", label=None), 1: dict(colour="#1f6f78", label=None)}

    grid = np.concatenate([np.linspace(380.0, 780.0, 401),
                           np.geomspace(780.0, 12000.0, 600)[1:]])

    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    axis.axvspan(380.0, 780.0, color="0.90", zorder=0)
    axis.text(462.0, 0.915, "fitted domain\n380\u2013780 nm", ha="center",
              va="center", fontsize=7.5, color="0.35")

    record = {}
    for index, name in enumerate(order):
        rgb = colours[name]
        coeffs = fetch(table, rgb)
        style = styles[index]
        in_band = grid <= 780.0
        curve = reflectance(coeffs, grid, clamp=False)

        short = name.replace("fabric Mystere ", "")
        axis.plot(grid[in_band], curve[in_band], color=style["colour"], lw=1.7,
                  label=f"{short}  RGB {tuple(rgb)}")
        axis.plot(grid[~in_band], curve[~in_band], color=style["colour"], lw=1.4,
                  ls="--", alpha=0.85)

        probes = {f"{w / 1000:g} um": float(reflectance(coeffs, w, clamp=False))
                  for w in (1200.0, 10000.0)}
        record[name] = {"rgb": rgb,
                        "coefficients": [float(c) for c in coeffs],
                        "reflectance_at_780nm": float(reflectance(coeffs, 780.0, False)),
                        "extrapolated": probes}
        print(f"  {short}: c = ({coeffs[0]:+.5g}, {coeffs[1]:+.5g}, {coeffs[2]:+.5g}), "
              f"R(1.2 um) = {probes['1.2 um']:.4f}, R(10 um) = {probes['10 um']:.4f}")

        if "Mango" in name:
            for wl, text_x, text_y in ((1200.0, 830.0, 0.68), (10000.0, 4200.0, 0.78)):
                value = float(reflectance(coeffs, wl, clamp=False))
                axis.plot([wl], [value], "o", ms=4.5, color=style["colour"], zorder=5)
                axis.annotate(f"{value:.4f} at {wl / 1000:g} \u00b5m",
                              xy=(wl, value), xytext=(text_x, text_y),
                              fontsize=7.5, color=style["colour"],
                              arrowprops=dict(arrowstyle="-", lw=0.7,
                                              color=style["colour"], alpha=0.7))

    axis.set_xscale("log")
    axis.set_xlim(380.0, 12000.0)
    axis.set_ylim(-0.02, 1.05)
    axis.set_xlabel("wavelength (nm)")
    axis.set_ylabel("upsampled reflectance $R(\\lambda)$")
    axis.set_xticks([400, 780, 1200, 2000, 5000, 10000])
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.tick_params(axis="both", labelsize=8)
    for band, left, right in (("NIR", 780, 1400), ("SWIR", 1400, 3000),
                              ("MWIR", 3000, 5000), ("LWIR", 8000, 12000)):
        axis.axvline(left, color="0.75", lw=0.5, zorder=0)
        axis.text(np.sqrt(left * right), 1.005, band, ha="center", fontsize=6.5,
                  color="0.45")
    # The 1.4-5 um span is the only large empty region on a log axis whose
    # curves have both saturated by 1 um.
    axis.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.62, 0.44),
                frameon=False)
    axis.set_title("Solid: the fit. Dashed: the extrapolation the convention forbids.",
                   fontsize=8.5)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "sheen_upsampling.json").write_text(json.dumps(
        {"table": str(TABLE), "table_version": version, "resolution": RES,
         "asset": str(CHAIR), "fit_domain_nm": [380.0, 780.0],
         "note": "reflectance evaluated without RgbSpectrumAt's lambda clamp, "
                 "which is the guard rail this figure exists to justify",
         "materials": record}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
