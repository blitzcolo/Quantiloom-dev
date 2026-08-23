#!/usr/bin/env python3
"""The ENVI cube of Section VIII-F, opened the way a downstream tool opens it.

The cube path is easy to demonstrate weakly — write a file, show a picture, say
"hyperspectral". This does the one check that makes the demonstration mean
something: every surface in the scene carries a *named* ECOSTRESS/ASTER curve,
so a spectrum pulled out of the cube at a pixel on the hull can be compared
against the library curve that pixel's material was bound to.

What the figure claims, and what it does not. Dividing the cube by the ASTM
G-173 spectrum the scene was lit with turns radiance into an *apparent*
reflectance, and two things separate that from the library value by a constant
neither this script nor the caption should pretend away. The illumination
geometry is one: G-173 is tabulated for a 37-degree tilted surface and the
ground here is horizontal under a sun at 46 degrees, which alone accounts for
the sand recovering at about 0.7 of its library level. Texture is the other,
and it is larger: the tank's materials are spectral-unmix mixtures modulated by
their own albedo textures, so a hull pixel is a mixture the library curve has
never described -- the renderer says as much at load time, warning that 13.9 %
of the hull's texels were clamped.

So the claim is that the cube separates the scene's materials spectrally and
reads back through a third-party parser, which is what Section VIII-F asks of
it. The library curves are drawn beside the recovered ones for scale, not as a
target the recovered ones are asserted to hit.

The cube is read with spectral-python, the de facto reader for this format, so
the file is exercised through a third-party parser rather than through the code
that wrote it.

Usage:
    e7_cube.py
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CUBE = REPO / "renders" / "gallery" / "kv2_cube.hdr"
SOLAR = REPO / "assets" / "luts" / "astmg173.csv"
LIBRARY = REPO / "assets" / "spectral" / "ecospeclib-all"
MATERIALS = REPO / "assets" / "spectral" / "quantiloom_materials_merged.json"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

# Pixels on each material, in fractions of the frame so they survive a
# resolution change, with the library curve each was bound to in the config.
PROBES = [
    ("turret", 0.505, 0.400, "Olive green gloss paint (0386UUUPNT)", "#4a6b2a"),
    ("hull", 0.470, 0.560, "Olive green gloss paint (0385UUUPNT)", "#6b8f3a"),
    ("sand", 0.180, 0.860, "Brown loamy fine sand (87P3468)", "#c19a5b"),
]
PATCH = 6  # half-width of the averaging window, in pixels


def read_library(name):
    """ECOSTRESS ASCII: a header block, then 'wavelength_um  per-cent' pairs."""
    catalogue = json.loads(MATERIALS.read_text(encoding="utf-8"))["materials"]
    path = LIBRARY / catalogue[name]["source"]["filename"]
    wavelengths, values = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            w, v = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        wavelengths.append(w * 1000.0)
        values.append(v / 100.0)
    order = np.argsort(wavelengths)
    return np.asarray(wavelengths)[order], np.clip(np.asarray(values)[order], 0.0, 1.0)


def solar_irradiance(grid):
    """ASTM G-173 global tilt, column 3, interpolated onto the cube's grid."""
    rows = np.loadtxt(SOLAR, delimiter=",", skiprows=2)
    return np.interp(grid, rows[:, 0], rows[:, 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig8d_envi_cube.png")
    args = parser.parse_args()

    import spectral
    if not CUBE.is_file():
        raise SystemExit(f"{CUBE} missing; render assets/configs/gallery/kv2_cube.toml")
    cube = spectral.open_image(str(CUBE))
    grid = np.asarray([float(w) for w in cube.bands.centers], dtype=np.float64)
    height, width = cube.shape[0], cube.shape[1]
    print(f"{CUBE.name}: {width} x {height} x {len(grid)} bands, "
          f"{grid[0]:.0f}-{grid[-1]:.0f} nm, read by spectral-python")

    irradiance = solar_irradiance(grid)
    # The atmospheric windows, named rather than thresholded. A threshold on the
    # irradiance finds the deepest troughs and misses their shoulders, and the
    # shoulders are where pi L / E is worst behaved: the renderer's sky term does
    # not carry G-173's absorption features, so the quotient climbs steeply on
    # either side of each band and produces spikes that are entirely the
    # inversion's rather than the scene's.
    usable = np.zeros_like(grid, dtype=bool)
    # The first window starts at 1160 rather than at the cube's own 1100: the
    # basis reconstruction is fitted over [1100, 2500] and its first two bands
    # sit on the edge of that fit, where it degrades.
    for lo, hi in ((1160.0, 1330.0), (1500.0, 1780.0), (2050.0, 2350.0)):
        usable |= (grid >= lo) & (grid <= hi)
    print(f"  {int(usable.sum())} of {len(grid)} bands carry usable irradiance; "
          f"{int((~usable).sum())} fall in atmospheric absorption")

    # A false-colour composite: SWIR, NIR, red as R, G, B. This is the standard
    # remote-sensing choice and it is what makes vegetation and painted metal
    # separate at a glance -- neither of which a true-colour view shows.
    def band_at(nm):
        return np.asarray(cube.read_band(int(np.argmin(np.abs(grid - nm)))),
                          dtype=np.float64)

    # All three inside the cube's own window; a nearest-band pick outside it
    # silently returns an endpoint three times and the composite means nothing.
    composite_nm = (2200.0, 1600.0, 1200.0)
    for nm in composite_nm:
        if not grid.min() - 1 <= nm <= grid.max() + 1:
            raise SystemExit(f"{nm} nm is outside the cube's {grid.min():.0f}-"
                             f"{grid.max():.0f} nm range")
    composite = np.dstack([band_at(nm) for nm in composite_nm])
    for c in range(3):
        plane = composite[..., c]
        lo, hi = np.percentile(plane, [1.0, 99.0])
        composite[..., c] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0

    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                                gridspec_kw={"width_ratios": [1.15, 1.0]})
    axes[0].imshow(composite)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("false colour: " + " / ".join(f"{nm:.0f}" for nm in composite_nm)
                  + " nm", fontsize=8)

    record = {}
    for label, fx, fy, curve_name, colour in PROBES:
        x, y = int(fx * width), int(fy * height)
        window = np.asarray(
            cube.read_subregion((max(0, y - PATCH), min(height, y + PATCH + 1)),
                                (max(0, x - PATCH), min(width, x + PATCH + 1))),
            dtype=np.float64)
        radiance = window.reshape(-1, len(grid)).mean(axis=0)
        # Lambertian under a known irradiance: rho = pi L / E. Where E itself
        # is near zero -- the 1400 and 1900 nm water bands, which ASTM G-173
        # takes to the floor -- the quotient is a ratio of two small numbers and
        # says nothing about the surface. Those bands are dropped rather than
        # plotted as the spikes they produce.
        recovered = np.pi * radiance / np.maximum(irradiance, 1e-9)
        recovered = np.where(usable, recovered, np.nan)

        lib_w, lib_v = read_library(curve_name)
        reference = np.interp(grid, lib_w, lib_v)
        inside = usable & (grid >= max(lib_w.min(), grid.min()))             & (grid <= min(lib_w.max(), grid.max()))
        # Reported, but not the claim: with texture modulation and an
        # uncorrected illumination geometry between the two curves, a shape
        # correlation is a number about the confounds as much as about the cube.
        correlation = float(np.corrcoef(recovered[inside], reference[inside])[0, 1])

        axes[0].plot([x], [y], "o", ms=6, mfc="none", mew=1.6, color=colour, zorder=5)
        axes[0].annotate(label, (x, y), xytext=(8, -8), textcoords="offset points",
                         fontsize=7, color=colour, fontweight="bold")

        axes[1].plot(grid, recovered, lw=1.5, color=colour, label=f"{label}, from the cube")
        axes[1].plot(grid, reference, lw=1.1, ls="--", color=colour, alpha=0.7)
        record[label] = {"pixel": [x, y], "library_curve": curve_name,
                         "shape_correlation": correlation,
                         "recovered_mean": float(recovered[inside].mean()),
                         "library_mean": float(reference[inside].mean())}
        print(f"  {label:<7} vs {curve_name[:38]:<38} r = {correlation:+.3f}")

    axes[1].set_xlabel("wavelength (nm)", fontsize=8)
    axes[1].set_ylabel("apparent reflectance", fontsize=8)
    axes[1].set_ylim(0.0, 0.75)
    for left, right in zip(grid[:-1][~usable[:-1]], grid[1:][~usable[:-1]]):
        axes[1].axvspan(left, right, color="0.90", zorder=0, lw=0)
    axes[1].tick_params(labelsize=7.5)
    axes[1].grid(color="0.93", lw=0.6)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[1].set_title("solid: recovered in the atmospheric windows (grey: absorption)\n"
                      "dashed: the bound ECOSTRESS curve, for scale", fontsize=8)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "cube.json").write_text(json.dumps(
        {"cube": str(CUBE), "shape": [width, height, len(grid)],
         "wavelength_nm": [float(grid[0]), float(grid[-1])],
         "step_nm": float(grid[1] - grid[0]),
         "reader": f"spectral-python {spectral.__version__}",
         "note": "recovered curves are apparent reflectance -- pi L / E with E the "
                 "ASTM G-173 global spectrum the scene was lit with -- so they carry "
                 "orientation, shadowing and inter-reflection that the library curve "
                 "does not; the shape is the claim, not the level",
         "probes": record}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
