#!/usr/bin/env python3
"""
check_shortwave_absorptivity.py - audit of the renderer's derived short-wave absorptivity.

What this is
------------
When a scene material carries ``thermal_conductivity_w_mk`` and a measured
spectral reference, ConfigResolve.cpp derives its short-wave absorptivity as

    alpha = 1 - integral(rho * E) / integral(E)

over the illuminant the scene names (``lighting.solar_lut``: the sun column
plus the sky column, ASTM G-173 global-tilt minus direct+circumsolar in these
scenes), integrated on the illuminant's own abscissae with trapezoids. The
reflectance rho is the NMF reconstruction from the SpectralBaker basis
(``spectral.basis_file`` + ``spectral.materials_json``): per band in
VIS/NIR/SWIR, spectrum[i] = sum_j weights[j] * basis[j][i], clamped to [0, 1],
sampled on the band's uniform grid; the three band curves are stitched in
wavelength order, and a mixture of refs is averaged on the first ref's
abscissae. Between the last sample of one band and the first of the next,
SpectralCurve::Evaluate linearly interpolates - that is the whole of the "gap
bridging". The derivation refuses (and the material falls back to 0.7) when
the reflectance spans less than 50% of the illuminant's irradiance.

Why this exists
---------------
The NMF reconstruction is a fit, not the measurement: it is clamped to [0, 1],
sampled on the basis grid, and in the stitched gaps it interpolates. This
script recomputes alpha twice for every measured material of the four env
scenes (Quantiloom-Qt/assets/configs/env_*.toml): once reproducing the
renderer exactly (NMF + stitching), once from the raw continuous library
spectrum the materials JSON points at (assets/spectral/ecospeclib-all in this
repo), and flags any material whose two alphas disagree by more than 0.02.
It changes nothing; it only measures.

Exit codes: 0 every |delta| <= 0.02, 1 an exceedance, 2 a missing input.
Output is ASCII-only (Windows CJK consoles crash on non-ASCII).
"""

import json
import os
import struct
import sys

import tomllib

DEV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QT_ROOT = os.path.join(os.path.dirname(DEV_ROOT), "Quantiloom-Qt")
CONFIG_DIR = os.path.join(QT_ROOT, "assets", "configs")
RAW_LIBRARY_DIR = os.path.join(DEV_ROOT, "assets", "spectral", "ecospeclib-all")
SCENES = ["env_concrete", "env_desert", "env_grassland", "env_seaside"]
DELTA_LIMIT = 0.02
GAP_WINDOWS_NM = [(780.0, 930.0), (1200.0, 1400.0)]


# ---------------------------------------------------------------------------
# SpectralCurve: sorted (wavelength_nm, value) samples, linear interpolation
# with endpoint clamping - a port of include/quantiloom/core/SpectralData.hpp.
# ---------------------------------------------------------------------------

class Curve:
    __slots__ = ("wl", "v")

    def __init__(self, samples):
        self.wl = [s[0] for s in samples]
        self.v = [s[1] for s in samples]

    def evaluate(self, lam):
        if not self.wl:
            return 0.0
        if lam <= self.wl[0]:
            return self.v[0]
        if lam >= self.wl[-1]:
            return self.v[-1]
        lo, hi = 0, len(self.wl) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.wl[mid] < lam:
                lo = mid
            else:
                hi = mid
        t = (lam - self.wl[lo]) / (self.wl[hi] - self.wl[lo])
        return self.v[lo] * (1.0 - t) + self.v[hi] * t


# ---------------------------------------------------------------------------
# Solar LUT - a port of SpectralIO::LoadLibRadtranSunAndSky. Commas and
# semicolons are whitespace, '#' lines are comments, anything that does not
# parse as numbers (the CSV header) is skipped. Columns are 1-based with the
# wavelength as column 1. With diffuse_is_global the sky is
# max(0, global - direct), because ASTM G-173 publishes no diffuse column.
# ---------------------------------------------------------------------------

def load_solar_lut(path, direct_column, diffuse_column, diffuse_is_global):
    sun, sky = [], []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").replace(";", " ").split()
            try:
                values = [float(t) for t in tokens]
            except ValueError:
                continue
            if len(values) < 2:
                continue
            wl = values[0]
            if wl <= 0.0:
                continue
            columns = values[1:]
            direct_index = direct_column - 2
            diffuse_index = diffuse_column - 2
            needed = max(direct_index, diffuse_index) + 1
            if len(columns) < needed:
                continue
            direct = max(0.0, columns[direct_index])
            raw_diffuse = columns[diffuse_index]
            diffuse = max(0.0, raw_diffuse - direct) if diffuse_is_global \
                else max(0.0, raw_diffuse)
            sun.append((wl, direct))
            sky.append((wl, diffuse))
    pairs = sorted(zip(sun, sky), key=lambda p: p[0][0])
    return (Curve([p[0] for p in pairs]), Curve([p[1] for p in pairs]))


# ---------------------------------------------------------------------------
# The derivation itself - a port of DeriveShortwaveAbsorptivity in
# renderer/ConfigResolve.cpp. Trapezoids run on the illuminant's own
# abscissae; a trapezoid straddling the reflectance range edge is skipped
# whole, not clipped. Returns (alpha, covered_fraction) or None on refusal
# (less than 50% of the illuminant's irradiance covered).
# ---------------------------------------------------------------------------

def derive_alpha(reflectance, sun, sky, lo=None, hi=None):
    if len(reflectance.wl) < 2 or len(sun.wl) < 2:
        return None
    if lo is None:
        lo = reflectance.wl[0]
    if hi is None:
        hi = reflectance.wl[-1]
    total = covered = reflected = 0.0
    for i in range(1, len(sun.wl)):
        l0, l1 = sun.wl[i - 1], sun.wl[i]
        if l1 <= l0:
            continue
        e0 = max(0.0, sun.v[i - 1] + sky.evaluate(l0))
        e1 = max(0.0, sun.v[i] + sky.evaluate(l1))
        band = 0.5 * (l1 - l0) * (e0 + e1)
        total += band
        if l0 < lo or l1 > hi:
            continue
        r0 = min(1.0, max(0.0, reflectance.evaluate(l0)))
        r1 = min(1.0, max(0.0, reflectance.evaluate(l1)))
        covered += band
        reflected += 0.5 * (l1 - l0) * (r0 * e0 + r1 * e1)
    if not (total > 0.0) or not (covered > 0.0):
        return None
    fraction = covered / total
    if fraction < 0.5:
        return None
    return (min(1.0, max(0.0, 1.0 - reflected / covered)), fraction)


# ---------------------------------------------------------------------------
# SpectralBaker NMF database - a port of io/SpectralBasisLoader.cpp. Binary
# v3: 64-byte header ("QBAS", version, num_bands), then per band a 16-byte
# header (start_um, end_um, num_samples, num_basis) and num_basis*num_samples
# float32, row-major. Reconstruction clamps to [0, 1] on the uniform grid.
# ---------------------------------------------------------------------------

def load_basis(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"QBAS":
        raise ValueError("not a QBAS basis file: " + path)
    version, num_bands = struct.unpack_from("<II", data, 4)
    if version != 3:
        raise ValueError("unsupported basis version %d: %s" % (version, path))
    bands = {}
    offset = 64
    for index in range(num_bands):
        start, end, n_samples, n_basis = struct.unpack_from("<ffII", data, offset)
        offset += 16
        count = n_basis * n_samples
        values = struct.unpack_from("<%df" % count, data, offset)
        offset += count * 4
        bands["VIS NIR SWIR MWIR LWIR".split()[index]] = (start, end, n_samples, n_basis, values)
    if offset != len(data):
        raise ValueError("trailing bytes in basis file: " + path)
    return bands


def reconstruct_band(bands, weights, band_name):
    """One band's NMF reconstruction, or None when the band is absent."""
    if band_name not in bands or band_name not in weights:
        return None
    start, end, n_samples, n_basis, values = bands[band_name]
    w = weights[band_name]
    if len(w) != n_basis:
        return None
    samples = []
    for i in range(n_samples):
        value = 0.0
        for j in range(n_basis):
            value += w[j] * values[j * n_samples + i]
        value = min(1.0, max(0.0, value))
        t = i / (n_samples - 1) if n_samples > 1 else 0.0
        samples.append(((start + t * (end - start)) * 1000.0, value))
    return Curve(samples)


def stitch_shortwave(bands, weights):
    """VIS then NIR then SWIR, keeping strictly increasing wavelengths -
    the port of the deriveAbsorptivity lambda's stitch. Any space between the
    last sample of one band and the first of the next is bridged by
    Curve.evaluate's linear interpolation."""
    samples = []
    for band_name in ("VIS", "NIR", "SWIR"):
        part = reconstruct_band(bands, weights, band_name)
        if part is None:
            continue
        for wl, value in zip(part.wl, part.v):
            if not samples or wl > samples[-1][0]:
                samples.append((wl, value))
    if len(samples) < 2:
        return None
    return Curve(samples)


def mean_curve(curves):
    """A mixture averaged on the first curve's abscissae - the port of the
    deriveAbsorptivity lambda's mean."""
    mean = curves[0]
    values = []
    for lam in mean.wl:
        values.append(sum(c.evaluate(lam) for c in curves) / len(curves))
    return Curve(list(zip(mean.wl, values)))


# ---------------------------------------------------------------------------
# Raw library spectrum (USGS/ECOSTRESS splib text: a header of "Key: value"
# lines, then "wavelength_um reflectance_percent" rows). This is the original
# measurement, before any NMF fit.
# ---------------------------------------------------------------------------

def load_raw_spectrum(path):
    samples = []
    x_is_micrometers = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("X Units"):
                x_is_micrometers = "micrometer" in line.lower()
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                x, y = float(tokens[0]), float(tokens[1])
            except ValueError:
                continue
            samples.append((x * 1000.0, y / 100.0))
    if not x_is_micrometers:
        raise ValueError("raw spectrum without micrometer X units: " + path)
    samples.sort(key=lambda s: s[0])
    return Curve(samples)


# ---------------------------------------------------------------------------
# Scene parsing
# ---------------------------------------------------------------------------

def parse_scene(path):
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    base = os.path.dirname(path)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    spectral = config.get("spectral", {})
    lighting = config.get("lighting", {})
    columns = lighting.get("solar_lut_columns", [2, 3])
    lut = {
        "path": resolve(lighting["solar_lut"]),
        "direct_column": int(columns[0]),
        "diffuse_column": int(columns[1]),
        "diffuse_is_global": bool(lighting.get("solar_lut_diffuse_is_global", False)),
    }
    materials = []
    for table in config.get("materials", []):
        refs = []
        if "spectral_material_ref" in table:
            refs.append(table["spectral_material_ref"])
        for ref in table.get("spectral_material_refs", []):
            if ref not in refs:
                refs.append(ref)
        if refs:
            materials.append({"name": table.get("name", "?"), "refs": refs})
    return {
        "basis_file": resolve(spectral["basis_file"]),
        "materials_json": resolve(spectral["materials_json"]),
        "lut": lut,
        "materials": materials,
    }


def main():
    # Missing inputs are an exit-2, said once.
    missing = []
    for scene in SCENES:
        path = os.path.join(CONFIG_DIR, scene + ".toml")
        if not os.path.exists(path):
            missing.append(path)
    if not os.path.isdir(RAW_LIBRARY_DIR):
        missing.append(RAW_LIBRARY_DIR)
    if missing:
        for path in missing:
            print("MISSING INPUT: " + path)
        return 2

    # Basis + weights databases, keyed on the (basis, json) pair like the
    # renderer's loader cache.
    databases = {}
    illuminants = {}
    rows = []
    notes = []

    for scene in SCENES:
        scene_path = os.path.join(CONFIG_DIR, scene + ".toml")
        parsed = parse_scene(scene_path)
        for key in ("basis_file", "materials_json", "lut"):
            value = parsed[key] if key != "lut" else parsed["lut"]["path"]
            if not os.path.exists(value):
                print("MISSING INPUT: " + value)
                return 2
        db_key = (parsed["basis_file"], parsed["materials_json"])
        if db_key not in databases:
            bands = load_basis(parsed["basis_file"])
            with open(parsed["materials_json"], "r", encoding="utf-8") as fh:
                materials = json.load(fh)["materials"]
            databases[db_key] = (bands, materials)
        bands, materials = databases[db_key]

        lut_key = (parsed["lut"]["path"], parsed["lut"]["direct_column"],
                   parsed["lut"]["diffuse_column"], parsed["lut"]["diffuse_is_global"])
        if lut_key not in illuminants:
            illuminants[lut_key] = load_solar_lut(**parsed["lut"])
        sun, sky = illuminants[lut_key]

        for material in parsed["materials"]:
            stitched_curves = []
            raw_curves = []
            for ref in material["refs"]:
                entry = materials.get(ref)
                if entry is None:  # renderer falls back to substring match
                    entry = next((m for name, m in materials.items()
                                  if ref.lower() in name.lower()), None)
                if entry is None:
                    print("MISSING INPUT: material '%s' not in %s"
                          % (ref, os.path.basename(parsed["materials_json"])))
                    return 2
                weights = {band: data["basis_weights"]
                           for band, data in entry["bands"].items()}
                stitched = stitch_shortwave(bands, weights)
                if stitched is None:
                    print("MISSING INPUT: '%s' has no short-wave reconstruction" % ref)
                    return 2
                stitched_curves.append(stitched)
                raw_path = os.path.join(RAW_LIBRARY_DIR, entry["source"]["filename"])
                if not os.path.exists(raw_path):
                    print("MISSING INPUT: raw spectrum " + raw_path)
                    return 2
                raw_curves.append(load_raw_spectrum(raw_path))

            mean_stitched = mean_curve(stitched_curves)
            mean_raw = mean_curve(raw_curves)

            derived = derive_alpha(mean_stitched, sun, sky)
            if derived is None:
                print("MISSING INPUT: derivation refused for %s/%s"
                      % (scene, material["name"]))
                return 2
            alpha_derived, covered_fraction = derived

            # Same range and illuminant as the derived side: the stitched
            # curve's span, intersected with what the raw curve actually
            # covers. No gap bridging exists on the raw side - it is one
            # continuous measurement.
            lo, hi = mean_stitched.wl[0], mean_stitched.wl[-1]
            raw_lo, raw_hi = mean_raw.wl[0], mean_raw.wl[-1]
            use_lo, use_hi = max(lo, raw_lo), min(hi, raw_hi)
            raw = derive_alpha(mean_raw, sun, sky, lo=use_lo, hi=use_hi)
            if raw is None:
                print("MISSING INPUT: raw spectrum does not overlap the "
                      "illuminant for %s/%s" % (scene, material["name"]))
                return 2
            alpha_raw, raw_coverage = raw
            if raw_lo > lo + 1e-6 or raw_hi < hi - 1e-6:
                notes.append(
                    "NOTE %s/%s: raw spectrum spans %.0f-%.0f nm, the stitched "
                    "range is %.0f-%.0f nm; raw alpha covers %.1f%% of the "
                    "illuminant over the intersection %.0f-%.0f nm"
                    % (scene, material["name"], raw_lo, raw_hi, lo, hi,
                       raw_coverage * 100.0, use_lo, use_hi))

            delta = alpha_derived - alpha_raw
            rows.append({
                "scene": scene,
                "material": material["name"],
                "refs": " + ".join(material["refs"]),
                "alpha_derived": alpha_derived,
                "alpha_raw": alpha_raw,
                "delta": delta,
                "covered_fraction": covered_fraction,
                "flag": abs(delta) > DELTA_LIMIT,
            })

    # Gap-window fractions of the illuminant's integral, printed once - all
    # four scenes name the same ASTM G-173 LUT with the same columns.
    (sun, sky), = illuminants.values()
    window_sums = [0.0, 0.0]
    total_sum = 0.0
    for i in range(1, len(sun.wl)):
        l0, l1 = sun.wl[i - 1], sun.wl[i]
        if l1 <= l0:
            continue
        e0 = max(0.0, sun.v[i - 1] + sky.evaluate(l0))
        e1 = max(0.0, sun.v[i] + sky.evaluate(l1))
        band = 0.5 * (l1 - l0) * (e0 + e1)
        total_sum += band
        for w, (w0, w1) in enumerate(GAP_WINDOWS_NM):
            if l0 >= w0 and l1 <= w1:
                window_sums[w] += band
    together = window_sums[0] + window_sums[1]

    print("Derived short-wave absorptivity audit (renderer reproduction vs raw library spectrum)")
    print("Basis: Quantiloom-Qt/assets/spectral/quantiloom_basis_v3_merged.qlbin")
    print("Illuminant: ASTM G-173 global tilt (direct+circumsolar + global-direct), 280-4000 nm")
    print("")
    header = "%-13s %-22s %-6s %-6s %-7s %s" % (
        "scene", "material", "a_der", "a_raw", "delta", "refs")
    print(header)
    print("-" * 110)
    exceedances = []
    for row in rows:
        print("%-13s %-22s %6.3f %6.3f %+7.3f %s%s"
              % (row["scene"], row["material"], row["alpha_derived"],
                 row["alpha_raw"], row["delta"], row["refs"],
                 "  <-- |delta| > %.2f" % DELTA_LIMIT if row["flag"] else ""))
        if row["flag"]:
            exceedances.append(row)
    print("")
    for row in rows:
        if row["covered_fraction"] < 0.999:
            print("coverage %s/%s: derived curve spans %.1f%% of the illuminant's irradiance"
                  % (row["scene"], row["material"], row["covered_fraction"] * 100.0))
    for note in notes:
        print(note)
    print("")
    print("Gap-window share of the illuminant's integral (trapezoids fully inside the window):")
    print("  780-930 nm : %.4f" % (window_sums[0] / total_sum))
    print("  1200-1400 nm: %.4f" % (window_sums[1] / total_sum))
    print("  together   : %.4f" % (together / total_sum))
    print("")
    if exceedances:
        print("RESULT: FAIL - %d material(s) with |delta| > %.2f"
              % (len(exceedances), DELTA_LIMIT))
        return 1
    print("RESULT: PASS - every |delta| <= %.2f" % DELTA_LIMIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
