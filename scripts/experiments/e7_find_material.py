#!/usr/bin/env python3
"""Screen the spectral library for a sample that describes ONE surface.

Fig. 8 renders the same asset painted and unpainted, so both variants need a
measured curve that spans VIS through LWIR and means the same thing at every
wavelength in between. That second condition is the one that is easy to miss
and it is what this screens for.

The KV-2's original hull sample was "a single coat of olive green alkyd gloss
paint, on an aluminum surface". In LWIR the alkyd is optically thick and the
sample reads as paint, rho = 0.099, agreeing with the config's declared
emissivity of 0.92. Around 4-5 um the coat thins optically, the aluminium shows
through, and rho climbs to 0.658. A KV-2 is thick paint on steel armour, so the
sample described the wrong object in precisely the band the figure exists to
show -- and nothing flagged it, because each band's value is individually
plausible. The signature is only visible ACROSS bands: a substrate showing
through raises MWIR and leaves LWIR alone.

Hence the screen, over the raw library spectra rather than the NMF
reconstruction, since the reconstruction is fitted per band and would smooth
away the very discontinuity being looked for:

  span      must cover VIS through LWIR, or it cannot serve one asset in five
            bands
  opacity   rho(MWIR) close to rho(LWIR) -- the ranking key
  level     an opaque dielectric coating sits low in both, roughly 0.05-0.20
  text      the sample's own description must not name a metal substrate

The ordering by coat count is the confirmation that this measures what it
claims to: across the olive-green series the MWIR/LWIR gap falls as coats are
added -- 0.370 for the thinnest, 0.225, then 0.049 -- which is what optical
thickness does and is not something a screen could impose.

Usage:
    e7_find_material.py --paint          # opaque dielectric coatings
    e7_find_material.py --metal          # bare metal, for the unpainted variant
    e7_find_material.py --paint --write  # also record the shortlist
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "assets" / "spectral" / "ecospeclib-all"
EVIDENCE = pathlib.Path(r"H:\quantiloom-paper\evidence\e7")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _winpaths import require_windows_paths  # noqa: E402

BANDS = {"VIS": (400, 780), "NIR": (930, 1200), "SWIR": (1400, 2400),
         "MWIR": (3000, 5000), "LWIR": (8000, 12000)}

# Words in a sample's own description that mean its curve is a laminate of two
# materials rather than one surface.
SUBSTRATE = re.compile(r"alumin|metal|steel|substrate|panel|foil", re.I)

SEARCHES = {
    "paint": {"select": re.compile(r"paint", re.I),
              "key": lambda b: abs(b["MWIR"] - b["LWIR"]),
              "label": "|rho_MWIR - rho_LWIR|, ascending: opacity"},
    "metal": {"select": re.compile(r"steel|iron|alumin|copper|metal|galvan", re.I),
              "key": lambda b: -b["MWIR"],
              "label": "rho_MWIR, descending: a metal should be reflective"},
}


def read_spectrum(path):
    """ECOSTRESS ASCII: a header block, then 'wavelength_um  per-cent' pairs."""
    header, wavelengths, values = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            header.append(line)
            continue
        try:
            w, v = float(parts[0]), float(parts[1])
        except ValueError:
            header.append(line)
            continue
        wavelengths.append(w * 1000.0)
        values.append(v / 100.0)
    if len(wavelengths) < 20:
        return None
    order = np.argsort(wavelengths)
    return ("\n".join(header),
            np.asarray(wavelengths)[order], np.asarray(values)[order])


def field(header, name):
    for line in header.splitlines():
        if line.startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    return ""


def screen(kind):
    search = SEARCHES[kind]
    rows = []
    for path in sorted(LIBRARY.glob("*spectrum.txt")):
        if not search["select"].search(path.name):
            continue
        parsed = read_spectrum(path)
        if parsed is None:
            continue
        header, wavelengths, reflectance = parsed
        # Span, not coverage: a sample that stops at 2500 nm cannot serve a
        # thermal band, and interpolating it there would be inventing data.
        if wavelengths.min() > 400 or wavelengths.max() < 12000:
            continue
        band = {name: float(reflectance[(wavelengths >= lo) &
                                        (wavelengths <= hi)].mean())
                for name, (lo, hi) in BANDS.items()}
        description = field(header, "Description")
        # The Name field does not carry the sample code, and the olive-green
        # series shares one name across four samples -- which is exactly the
        # set this has to tell apart. The code sits in the filename as a
        # four-digit run followed by letters: manmade.<...>.0408uuupnt.<...>
        found = re.search(r"[.](\d{4}[a-z]+)[.]", path.name)
        code = found.group(1).upper() if found else path.stem
        rows.append({"name": field(header, "Name") or path.name,
                     "sample": code,
                     "file": path.name,
                     "band_mean": band,
                     "mwir_lwir_gap": abs(band["MWIR"] - band["LWIR"]),
                     # Vacuous for the metal search -- every metal names a
                     # metal -- so it is only computed where it discriminates.
                     "names_a_substrate": (kind == "paint"
                                           and bool(SUBSTRATE.search(description))),
                     "description": description[:200]})
    rows.sort(key=lambda r: search["key"](r["band_mean"]))
    return rows, search["label"]


def report(kind, rows, label):
    print(f"\n{len(rows)} {kind} samples spanning VIS..LWIR, sorted by {label}\n")
    print(f"{'name':40} {'VIS':>6} {'NIR':>6} {'SWIR':>6} {'MWIR':>6} {'LWIR':>6}"
          f" {'gap':>7}  substrate?")
    for row in rows:
        b = row["band_mean"]
        flag = "  <-- names a substrate" if row["names_a_substrate"] else ""
        print(f"{(row['name'] + ' (' + row['sample'] + ')')[:40]:40} "
              f"{b['VIS']:6.3f} {b['NIR']:6.3f} "
              f"{b['SWIR']:6.3f} {b['MWIR']:6.3f} {b['LWIR']:6.3f} "
              f"{row['mwir_lwir_gap']:7.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paint", action="store_true")
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="record the shortlist in evidence/e7/material_search.json")
    args = parser.parse_args()
    kinds = [k for k in ("paint", "metal") if getattr(args, k)] or ["paint", "metal"]
    if args.write:
        require_windows_paths(EVIDENCE)

    result = {}
    for kind in kinds:
        rows, label = screen(kind)
        report(kind, rows, label)
        result[kind] = {"sorted_by": label, "compared": len(rows), "candidates": rows}

    if args.write:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / "material_search.json").write_text(
            json.dumps({"library": str(LIBRARY),
                        "bands_nm": BANDS,
                        "chosen": {
                            "hull": "Olive green paint (0407UUUPNT)",
                            "turret": "Olive green paint (0408UUUPNT)",
                            "unpainted": "Galvanized Steel Metal (0525UUUSTLb)"},
                        "searches": result}, indent=2), encoding="utf-8")
        print(f"\nwrote {EVIDENCE / 'material_search.json'}")


if __name__ == "__main__":
    main()
