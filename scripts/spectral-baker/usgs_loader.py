#!/usr/bin/env python3
"""
USGS Spectral Library Data Loader

Parses the USGS splib07a ASCII format for spectral reflectance data.
Handles both wavelength files and material AREF (absolute reflectance) files.

Author: Quantiloom Team
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaterialData:
    """Container for a single material's spectral data."""
    name: str
    record_id: str
    instrument: str
    chapter: str
    filename: str
    wavelengths: np.ndarray  # micrometers
    reflectance: np.ndarray  # dimensionless [0, 1]

    @property
    def num_samples(self) -> int:
        return len(self.wavelengths)

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        return (float(self.wavelengths[0]), float(self.wavelengths[-1]))


def load_wavelengths(filepath: str) -> np.ndarray:
    """
    Load USGS wavelength file.

    Format:
        Line 1: Header (e.g., "splib07a Record=23: Wavelengths ASD 0.35-2.5 microns 2151 ch")
        Lines 2+: One float per line (wavelength in micrometers)

    Args:
        filepath: Path to wavelength file

    Returns:
        numpy array of wavelengths in micrometers
    """
    wavelengths = []

    with open(filepath, 'r') as f:
        # Skip header line
        header = f.readline().strip()
        logger.debug(f"Wavelength file header: {header}")

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                value = float(line)
                wavelengths.append(value)
            except ValueError:
                logger.warning(f"Skipping non-numeric line {line_num}: {line[:50]}")
                continue

    result = np.array(wavelengths, dtype=np.float64)
    logger.info(f"Loaded {len(result)} wavelengths from {Path(filepath).name}")
    logger.info(f"  Range: {result[0]:.4f} - {result[-1]:.4f} µm")

    return result


def parse_material_header(header: str) -> Dict[str, str]:
    """
    Parse USGS material file header line.

    Example header:
        "splib07a Record=18253: Aluminum brushed 293K        ASDFRa AREF"

    Returns dict with keys: record_id, name, instrument
    """
    result = {
        'record_id': '',
        'name': '',
        'instrument': ''
    }

    # Extract record ID
    record_match = re.search(r'Record=(\d+)', header)
    if record_match:
        result['record_id'] = record_match.group(1)

    # Extract instrument (last token before AREF/RREF)
    # Common instruments: ASDFRa, ASDHRa, BECKa, NIC4a
    inst_match = re.search(r'(\w+)\s+[AR]REF\s*$', header)
    if inst_match:
        result['instrument'] = inst_match.group(1)

    # Extract material name (everything between ":" and instrument)
    name_match = re.search(r':\s*(.+?)\s+\w+\s+[AR]REF', header)
    if name_match:
        # Clean up extra whitespace
        result['name'] = re.sub(r'\s+', ' ', name_match.group(1).strip())

    return result


def load_material_reflectance(filepath: str,
                               wavelengths: np.ndarray) -> Optional[MaterialData]:
    """
    Load a single material's reflectance data.

    Args:
        filepath: Path to material AREF file
        wavelengths: Pre-loaded wavelength array (must match file length)

    Returns:
        MaterialData object, or None if loading fails
    """
    reflectance = []
    header_info = {}

    path = Path(filepath)

    try:
        with open(filepath, 'r') as f:
            # Parse header line
            header = f.readline().strip()
            header_info = parse_material_header(header)

            # Read reflectance values
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = float(line)
                    reflectance.append(value)
                except ValueError:
                    logger.warning(f"{path.name}:{line_num} - Invalid value: {line[:30]}")
                    continue

        reflectance = np.array(reflectance, dtype=np.float64)

        # Validate length matches wavelengths
        if len(reflectance) != len(wavelengths):
            logger.error(f"{path.name}: Length mismatch - "
                        f"got {len(reflectance)} values, expected {len(wavelengths)}")
            return None

        # Validate reflectance range (allow some tolerance for calibration errors)
        min_val, max_val = reflectance.min(), reflectance.max()
        if min_val < -0.1:
            logger.warning(f"{path.name}: Negative reflectance detected (min={min_val:.4f})")
        if max_val > 1.5:
            logger.warning(f"{path.name}: Reflectance > 1.5 detected (max={max_val:.4f})")

        # Clamp to valid range
        reflectance = np.clip(reflectance, 0.0, 1.0)

        # Extract chapter from path
        chapter = ""
        for part in path.parts:
            if part.startswith("Chapter"):
                chapter = part
                break

        # Build material name from filename if header parsing failed
        if not header_info['name']:
            # Extract from filename: splib07a_NAME_INSTRUMENT_AREF.txt
            name_match = re.match(r'splib07a_(.+?)_\w+_[AR]REF\.txt', path.name)
            if name_match:
                header_info['name'] = name_match.group(1).replace('_', ' ')
            else:
                header_info['name'] = path.stem

        return MaterialData(
            name=header_info['name'],
            record_id=header_info['record_id'],
            instrument=header_info['instrument'],
            chapter=chapter,
            filename=path.name,
            wavelengths=wavelengths.copy(),
            reflectance=reflectance
        )

    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


def discover_materials(root_dir: str,
                       pattern: str = "**/*_AREF.txt",
                       instrument_filter: Optional[str] = None) -> List[str]:
    """
    Recursively find all material reflectance files.

    Args:
        root_dir: Root directory to search
        pattern: Glob pattern for material files
        instrument_filter: If set, only return files matching this instrument
                          (e.g., "ASD" matches ASDFRa, ASDHRa, etc.)

    Returns:
        List of absolute file paths
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    files = sorted(root.glob(pattern))

    # Filter by instrument if requested
    if instrument_filter:
        files = [f for f in files if instrument_filter in f.name]

    # Exclude wavelength and bandpass files
    files = [f for f in files if "Wavelengths" not in f.name
             and "Bandpass" not in f.name
             and "Wavenumber" not in f.name]

    logger.info(f"Discovered {len(files)} material files in {root_dir}")

    return [str(f) for f in files]


def load_all_materials(root_dir: str,
                       wavelength_file: str,
                       instrument: str = "ASD",
                       max_materials: Optional[int] = None) -> List[MaterialData]:
    """
    Load all materials matching the specified instrument.

    Args:
        root_dir: USGS library root directory
        wavelength_file: Path to wavelength file (relative to root_dir or absolute)
        instrument: Instrument filter (default: "ASD" for VIS+SWIR)
        max_materials: Optional limit for testing

    Returns:
        List of MaterialData objects
    """
    root = Path(root_dir)

    # Load wavelengths
    wl_path = Path(wavelength_file)
    if not wl_path.is_absolute():
        wl_path = root / wavelength_file

    wavelengths = load_wavelengths(str(wl_path))

    # Discover materials
    material_files = discover_materials(root_dir, instrument_filter=instrument)

    if max_materials:
        material_files = material_files[:max_materials]
        logger.info(f"Limited to {max_materials} materials for testing")

    # Load each material
    materials = []
    failed = 0

    for filepath in material_files:
        mat = load_material_reflectance(filepath, wavelengths)
        if mat is not None:
            materials.append(mat)
        else:
            failed += 1

    logger.info(f"Loaded {len(materials)} materials successfully, {failed} failed")

    return materials


def get_material_by_name(materials: List[MaterialData],
                         name_pattern: str) -> Optional[MaterialData]:
    """
    Find a material by name (case-insensitive partial match).
    """
    pattern = name_pattern.lower()
    for mat in materials:
        if pattern in mat.name.lower():
            return mat
    return None


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s: %(message)s')

    # Test paths
    USGS_ROOT = "../../assets/usgs/ASCIIdata_splib07a"
    WL_FILE = "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"

    print("=== USGS Loader Test ===\n")

    # Test wavelength loading
    wl = load_wavelengths(f"{USGS_ROOT}/{WL_FILE}")
    print(f"Wavelengths: {len(wl)} points, {wl[0]:.4f} - {wl[-1]:.4f} µm\n")

    # Test material discovery
    files = discover_materials(USGS_ROOT, instrument_filter="ASD")
    print(f"Found {len(files)} ASD material files\n")

    # Test loading a few materials
    materials = load_all_materials(USGS_ROOT, WL_FILE, max_materials=5)

    print("\nLoaded materials:")
    for mat in materials:
        print(f"  - {mat.name}")
        print(f"    Record: {mat.record_id}, Instrument: {mat.instrument}")
        print(f"    Chapter: {mat.chapter}")
        print(f"    Reflectance range: {mat.reflectance.min():.4f} - {mat.reflectance.max():.4f}")
        print()
