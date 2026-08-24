#!/usr/bin/env python3
"""Find a real material that hides in the visible and shows in the infrared.

Section VIII-F's camouflage panel claims a target indistinguishable from its
background in the visible band and separated from it in the thermal bands. That
claim is only worth making if the pair is measured rather than arranged, so
this searches the ECOSTRESS/ASTER library for a material whose visible
reflectance tracks the desert sand's and whose MWIR or LWIR reflectance does
not.

The scoring is deliberately blunt. Visible agreement is the RMS difference in
reflectance across 400-780 nm, which is what a broadband visible sensor
integrates; infrared separation is the absolute difference in band-mean
reflectance, which is what sets the emissivity contrast a thermal sensor sees
through Kirchhoff. A candidate has to be close in the first and far in the
second, and the ranking reports both so that a marginal pair can be recognised
as marginal.

If nothing scores well, that is the answer: the figure's caption should then
describe what the fusion actually shows rather than a camouflage narrative the
scene does not support.

Usage:
    e7_find_camouflage_pair.py --against "Brown loamy fine sand (87P3468)"
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "assets" / "spectral" / "ecospeclib-all"
MATERIALS = REPO / "assets" / "spectral" / "quantiloom_materials_merged.json"

VIS = (400.0, 780.0)
MWIR = (3000.0, 5000.0)
LWIR = (8000.0, 12000.0)


def read_spectrum(path):
    """ECOSTRESS ASCII: a header block, then 'wavelength_um  value' pairs.

    The value column is reflectance in per cent for most chapters. Wavelengths
    are microns and sometimes descend, so both are normalised here.
    """
    wavelengths, values = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            w, v = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        wavelengths.append(w * 1000.0)   # um -> nm
        values.append(v / 100.0)         # per cent -> fraction
    if len(wavelengths) < 8:
        return None
    order = np.argsort(wavelengths)
    return np.asarray(wavelengths)[order], np.clip(np.asarray(values)[order], 0.0, 1.0)


def band_mean(spectrum, band):
    wavelengths, values = spectrum
    mask = (wavelengths >= band[0]) & (wavelengths <= band[1])
    return float(values[mask].mean()) if mask.sum() >= 3 else None


def visible_rms(a, b):
    """RMS reflectance difference over the visible, on a common grid."""
    grid = np.linspace(*VIS, 40)
    va = np.interp(grid, a[0], a[1])
    vb = np.interp(grid, b[0], b[1])
    if (a[0].min() > VIS[0] + 50) or (b[0].min() > VIS[0] + 50):
        return None
    return float(np.sqrt(((va - vb) ** 2).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--against", default="Brown loamy fine sand (87P3468)")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\evidence\e7"
                                             r"\camouflage_pair.json"))
    args = parser.parse_args()
    # These defaults are Windows paths; under WSL they are directory NAMES.
    # See scripts/experiments/_winpaths.py for what that silently does.
    require_windows_paths(args.out)

    catalogue = json.loads(MATERIALS.read_text(encoding="utf-8"))["materials"]
    if args.against not in catalogue:
        raise SystemExit(f"{args.against!r} is not in the library")

    target = read_spectrum(LIBRARY / catalogue[args.against]["source"]["filename"])
    if target is None:
        raise SystemExit("could not read the reference spectrum")
    target_mwir = band_mean(target, MWIR)
    target_lwir = band_mean(target, LWIR)
    print(f"reference: {args.against}")
    print(f"  MWIR mean reflectance {target_mwir:.3f}, LWIR {target_lwir:.3f}")

    rows = []
    for name, entry in catalogue.items():
        if name == args.against:
            continue
        path = LIBRARY / entry["source"]["filename"]
        if not path.is_file():
            continue
        spectrum = read_spectrum(path)
        if spectrum is None:
            continue
        rms = visible_rms(target, spectrum)
        mwir = band_mean(spectrum, MWIR)
        lwir = band_mean(spectrum, LWIR)
        if rms is None or mwir is None or lwir is None:
            continue
        rows.append({
            "name": name,
            "visible_rms": rms,
            "mwir_mean": mwir, "lwir_mean": lwir,
            "mwir_separation": abs(mwir - target_mwir),
            "lwir_separation": abs(lwir - target_lwir),
        })

    # Close in the visible first, then as far apart as possible in the infrared.
    close = [r for r in rows if r["visible_rms"] < 0.03]
    close.sort(key=lambda r: -max(r["mwir_separation"], r["lwir_separation"]))
    print(f"\n{len(rows)} materials compared; {len(close)} within 0.03 RMS in the visible\n")
    print(f"  {'material':<52} {'VIS rms':>8} {'MWIR sep':>9} {'LWIR sep':>9}")
    for row in close[:args.top]:
        print(f"  {row['name'][:52]:<52} {row['visible_rms']:8.4f} "
              f"{row['mwir_separation']:9.3f} {row['lwir_separation']:9.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"reference": args.against,
         "reference_mwir_mean": target_mwir, "reference_lwir_mean": target_lwir,
         "compared": len(rows), "visible_rms_threshold": 0.03,
         "candidates": close[:args.top]}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
