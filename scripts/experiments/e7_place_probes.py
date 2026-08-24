#!/usr/bin/env python3
"""Where Fig. 8d's three probe pixels come from, and why they are not circular.

e7_cube.py carries three fractional image coordinates as a literal. A literal
is exactly the kind of thing that goes quietly wrong: these were first placed
for a camera at [14, 5, 13], the config later moved to [-7, 3.4, 6.5], and the
coordinates went on sampling sky and sand while still being labelled turret and
hull. The figure then reported a hull ANTI-correlated with the curve its
material was bound to, r = -0.618, and nothing in the tree disagreed.

So the placement is a script, and it is this one. Two properties matter:

  * The vehicle is found, not drawn. kv2_desert_lwir.toml and
    kv2_desert_bare.toml differ only in the reflectance bound to the tank; the
    ground is the same material rendered from the same camera with the same
    seed, so it comes out bit-identical and anything that differs between the
    two frames is the vehicle. Both frames are tracked under evidence/e7/strip,
    so the mask is reproducible from the repository rather than from whatever
    happened to be in a scratch directory.

  * Probes are placed by GEOMETRY and checked against the curves afterwards --
    turret from the top of the silhouette, hull from its middle, sand from the
    open ground beside it. Picking the pixel whose spectrum best matched a
    library curve would assume precisely what Fig. 8d sets out to demonstrate.
    The check is therefore allowed to fail, and the cross-correlation against
    the OTHER materials' curves is printed beside each probe: a probe matching
    its own curve less well than it matches a neighbour's is a misplacement,
    and is reported as one.

The camera in kv2_cube.toml and kv2_desert_lwir.toml is the same position,
target and field of view, so a fractional coordinate carries between them; only
the spectral mode and the resolution differ.

Usage:
    e7_place_probes.py            # place, verify, print
    e7_place_probes.py --write    # also update evidence/e7/probes.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
CUBE = REPO / "renders" / "gallery" / "kv2_cube.hdr"
LIBRARY = REPO / "assets" / "spectral" / "ecospeclib-all"
MATERIALS = REPO / "assets" / "spectral" / "quantiloom_materials_merged.json"
SOLAR = REPO / "assets" / "luts" / "astmg173.csv"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(EVIDENCE)

# The material each probe's surface is bound to in kv2_cube.toml, kept here as
# the expectation the verification pass is allowed to contradict.
BOUND = {
    "turret": "Olive green paint (0408UUUPNT)",
    "hull": "Olive green paint (0407UUUPNT)",
    "sand": "Brown loamy fine sand (87P3468)",
}

# Height fractions within the vehicle silhouette. The KV-2's turret is a tall
# slab box on top of a low hull, so the top fifth is turret and the middle is
# hull, with a wide margin either side of the join.
SLABS = {"turret": (0.00, 0.22), "hull": (0.42, 0.62)}

# What each probe is meant to be distinguishable FROM. 0407 and 0408 are the
# same paint series and are not separable from each other; paint and soil are.
CLASS = {"turret": "paint", "hull": "paint", "sand": "soil"}

# Half-width of the averaging window, matching e7_cube.py.
PATCH = 6

LUMA = np.array([0.2126, 0.7152, 0.0722])


def read_exr_grey(path):
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)[..., :3] @ LUMA
    if len(channels) == 1:
        return np.asarray(next(iter(channels.values())).pixels, dtype=np.float64)
    return np.stack([np.asarray(channels[c].pixels, dtype=np.float64)
                     for c in "RGB"], -1) @ LUMA


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


def vehicle_mask():
    """The tank silhouette, as the painted/unpainted difference."""
    from scipy import ndimage
    paint = EVIDENCE / "strip" / "kv2_paint_vis.exr"
    bare = EVIDENCE / "strip" / "kv2_bare_vis.exr"
    for path in (paint, bare):
        if not path.is_file():
            raise SystemExit(f"{path} missing; run e7_gallery.py --strip first")

    # 0.005 absolute, about a tenth of the ~0.05 ground radiance. Scaled by
    # max() instead, the threshold lands at 0.061 -- bare steel throws a 3.06
    # specular highlight -- and only the tank's brightest 0.6 % survives it.
    mask = np.abs(read_exr_grey(paint) - read_exr_grey(bare)) > 0.005
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((5, 5))))
    labels, count = ndimage.label(mask)
    if count == 0:
        raise SystemExit("the painted and unpainted frames do not differ anywhere")
    # Largest connected component is the tank; the rest is shadow noise.
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes)))


def place(mask):
    from scipy import ndimage
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    top, bottom = ys.min(), ys.max()
    print(f"vehicle mask: {mask.sum():,} px in {width}x{height}, "
          f"x {xs.min()}-{xs.max()}, y {top}-{bottom}")

    probes = {}
    for label, (lo, hi) in SLABS.items():
        rows = np.arange(height)[:, None]
        band = ((rows >= top + (bottom - top) * lo) &
                (rows <= top + (bottom - top) * hi))
        selected = mask & band
        # Erode so the probe sits in the interior. A probe on the silhouette
        # edge straddles the sky and averages two materials.
        inner = ndimage.binary_erosion(selected, np.ones((5, 5)))
        if inner.sum() < 20:
            inner = selected
        yy, xx = np.nonzero(inner)
        probes[label] = (int(np.median(xx)), int(np.median(yy)))

    # Open ground, well clear of the vehicle and of its shadow, which falls to
    # the far side from this camera.
    probes["sand"] = (int(xs.min() * 0.45), int(height * 0.86))
    return {k: (x / width, y / height) for k, (x, y) in probes.items()}


def solar_irradiance(grid):
    """ASTM G-173 global tilt, column 3, interpolated onto the cube's grid."""
    rows = np.loadtxt(SOLAR, delimiter=",", skiprows=2)
    return np.interp(grid, rows[:, 0], rows[:, 2])


def verify(probes):
    """Correlate each probe against every bound curve, not only its own.

    The comparison is the figure's own, and it has to be: correlating raw
    radiance against a reflectance measures the solar spectrum, which every
    pixel shares, and returns about +0.75 for any pair of surfaces. So the
    radiance is turned into apparent reflectance the way Fig. 8d turns it --
    pi L / E over a PATCH-averaged window -- and restricted to the same three
    atmospheric windows, outside which pi L / E is a ratio of two near-zeros.

    What the verdict can and cannot decide. The turret binds 0408UUUPNT and the
    hull 0407UUUPNT, two olive paints from the same series whose curves differ
    by less than either differs from itself between samples; no correlation
    test separates them, and one that claimed to would be measuring noise. The
    discriminating comparison is paint against soil, so that is the one made:
    a probe passes when its own curve beats every curve of a DIFFERENT class.
    """
    import spectral
    if not CUBE.is_file():
        raise SystemExit(f"{CUBE} missing; render assets/configs/gallery/kv2_cube.toml")
    cube = spectral.open_image(str(CUBE))
    height, width = cube.shape[0], cube.shape[1]
    grid = np.asarray([float(v) for v in cube.bands.centers], dtype=np.float64)
    irradiance = solar_irradiance(grid)

    # Same windows as e7_cube.py, named rather than thresholded, and for the
    # same reason: a threshold catches the troughs and misses the shoulders,
    # and the shoulders are where the quotient misbehaves.
    usable = np.zeros_like(grid, dtype=bool)
    for lo, hi in ((1160.0, 1330.0), (1500.0, 1780.0), (2050.0, 2350.0)):
        usable |= (grid >= lo) & (grid <= hi)

    curves = {name: np.interp(grid, *read_library(reference))
              for name, reference in BOUND.items()}

    print()
    print(f"{int(usable.sum())} of {len(grid)} bands usable; "
          f"apparent reflectance, {2 * PATCH + 1}^2 px window")
    print(f"{'probe':8} {'fx':>6} {'fy':>6} {'r vs own':>10}   others")
    rows, ok = [], True
    for label, (fx, fy) in probes.items():
        x, y = int(fx * width), int(fy * height)
        block = np.asarray(
            cube.read_subregion((max(0, y - PATCH), min(height, y + PATCH + 1)),
                                (max(0, x - PATCH), min(width, x + PATCH + 1))),
            dtype=np.float64)
        radiance = block.reshape(-1, len(grid)).mean(axis=0)
        recovered = (np.pi * radiance / np.maximum(irradiance, 1e-9))[usable]

        correlation = {other: float(np.corrcoef(recovered, curve[usable])[0, 1])
                       for other, curve in curves.items()}
        # Only a different-class curve can outrank the probe's own one.
        rivals = [v for k, v in correlation.items() if CLASS[k] != CLASS[label]]
        passed = correlation[label] > max(rivals)
        ok = ok and passed
        others = "  ".join(f"{k}={v:+.3f}"
                           for k, v in correlation.items() if k != label)
        verdict = "OK" if passed else "MISMATCH"
        print(f"{label:8} {fx:6.3f} {fy:6.3f} {correlation[label]:+10.3f}   "
              f"{others}   {verdict}")
        rows.append({"probe": label, "fx": fx, "fy": fy,
                     "bound_curve": BOUND[label], "class": CLASS[label],
                     "correlation": correlation,
                     "beats_every_other_class": bool(passed)})
    return rows, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="update evidence/e7/probes.json")
    args = parser.parse_args()

    probes = place(vehicle_mask())
    rows, ok = verify(probes)

    print("\nPROBES for e7_cube.py:")
    for row in rows:
        print(f'    ("{row["probe"]}", {row["fx"]:.3f}, {row["fy"]:.3f}, '
              f'"{row["bound_curve"]}", ...),')

    if args.write:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / "probes.json").write_text(json.dumps(
            {"placement": "geometric, from the painted/unpainted difference mask",
             "verification": "correlation against every bound curve, not only its own",
             "every_probe_beats_every_other_class": bool(ok),
             "probes": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {EVIDENCE / 'probes.json'}")

    # A mismatch is a placement error, and the caller should hear about it.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
