"""ECOSTRESS Spectral Library loader for Quantiloom spectral baker.

Reads .spectrum.txt files from the ECOSTRESS/ASTER spectral library.
Outputs MaterialData objects compatible with the existing NMF pipeline.
"""

import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from usgs_loader import MaterialData


def parse_ecostress_header(lines: List[str]) -> Dict[str, str]:
    """Parse key:value header pairs until the first data line."""
    header: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Data lines start with a digit or whitespace+digit
        if re.match(r"^\s*[\d.+-]", stripped):
            break
        m = re.match(r"^([^:]+):\s*(.*)", stripped)
        if m:
            header[m.group(1).strip()] = m.group(2).strip()
    return header


def load_ecostress_file(path: pathlib.Path) -> Optional[MaterialData]:
    """Load a single ECOSTRESS .spectrum.txt file.

    Returns None on parse failure or insufficient data.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.splitlines()
    header = parse_ecostress_header(lines)

    name = header.get("Name", path.stem)
    sample_no = header.get("Sample No.", "")
    y_units = header.get("Y Units", "").lower()

    is_percent = "percent" in y_units or "percentage" in y_units
    is_reflectance = "reflect" in y_units
    is_transmittance = "transmit" in y_units or "transmission" in y_units

    if not (is_reflectance or is_transmittance):
        return None

    wavelengths = []
    values = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            wl = float(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        if wl <= 0:
            continue
        wavelengths.append(wl)
        values.append(val)

    if len(wavelengths) < 3:
        return None

    wl_arr = np.array(wavelengths)
    val_arr = np.array(values)

    if is_percent:
        val_arr = val_arr / 100.0

    # Ensure ascending wavelength order (some files are descending)
    if wl_arr[0] > wl_arr[-1]:
        wl_arr = wl_arr[::-1]
        val_arr = val_arr[::-1]

    # Filter bad values
    mask = val_arr > -0.1
    if not np.all(mask):
        wl_arr = wl_arr[mask]
        val_arr = val_arr[mask]

    if np.any(val_arr > 1.5):
        warnings.warn(f"{path.name}: values > 1.5 detected, clamping")

    val_arr = np.clip(val_arr, 0.0, 1.0)

    if len(wl_arr) < 3:
        return None

    mat_name = f"{name} ({sample_no})" if sample_no else name

    # Extract chapter from filename (e.g. "manmade", "mineral", "vegetation")
    chapter = path.stem.split(".")[0] if "." in path.stem else ""

    return MaterialData(
        name=mat_name,
        record_id=sample_no,
        instrument="becknic",
        chapter=chapter,
        filename=path.name,
        wavelengths=wl_arr,
        reflectance=val_arr,
    )


def discover_ecostress_materials(
    root: pathlib.Path,
    min_wavelength_um: float = 0.0,
    max_wavelength_um: float = 100.0,
) -> List[MaterialData]:
    """Load all .spectrum.txt files from root, optionally filtering by coverage."""
    files = sorted(root.glob("*.spectrum.txt"))
    materials = []
    skipped = 0

    for f in files:
        mat = load_ecostress_file(f)
        if mat is None:
            skipped += 1
            continue

        wl_min, wl_max = mat.wavelength_range
        if wl_max < min_wavelength_um or wl_min > max_wavelength_um:
            skipped += 1
            continue

        materials.append(mat)

    print(f"ECOSTRESS: loaded {len(materials)} materials, skipped {skipped}")
    return materials
