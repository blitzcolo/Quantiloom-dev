#!/usr/bin/env python3
"""
RefractiveIndex.info Database Loader (SpectralBaker v3.0)

Parses the refractiveindex.info YAML database and converts complex refractive
index (n, k) data to reflectance spectra for spectral rendering.

Data source: https://refractiveindex.info/
Format: YAML files with tabulated n,k data

Author: Quantiloom Team
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np

# Use PyYAML for parsing
try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

# Import MaterialData from usgs_loader (shared data structure)
from usgs_loader import MaterialData

logger = logging.getLogger(__name__)


def nk_to_reflectance(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Convert complex refractive index (n, k) to reflectance at normal incidence.

    Uses Fresnel equations for normal incidence:
    R = |r|^2 where r = (n_1 - n_2) / (n_1 + n_2)

    For complex n_2 = n + ik and n_1 = 1 (air):
    R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)

    Args:
        n: Real part of refractive index (dimensionless)
        k: Imaginary part (extinction coefficient)

    Returns:
        Reflectance in [0, 1]
    """
    numerator = (n - 1.0)**2 + k**2
    denominator = (n + 1.0)**2 + k**2

    # Avoid division by zero
    reflectance = np.where(denominator > 1e-10,
                          numerator / denominator,
                          0.0)

    # Clamp to valid range
    reflectance = np.clip(reflectance, 0.0, 1.0)

    return reflectance


def parse_tabulated_nk(data_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse 'tabulated nk' data block.

    Format:
        wavelength_um n k
        0.500 1.45 0.002
        0.600 1.46 0.003
        ...

    Returns:
        Tuple of (wavelengths_um, n_values, k_values)
    """
    lines = data_str.strip().split('\n')

    wavelengths = []
    n_values = []
    k_values = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                wl = float(parts[0])
                n = float(parts[1])
                k = float(parts[2])

                wavelengths.append(wl)
                n_values.append(n)
                k_values.append(k)
            except ValueError:
                logger.warning(f"Skipping invalid nk data line: {line[:50]}")
                continue

    if len(wavelengths) == 0:
        raise ValueError("No valid nk data found")

    return (np.array(wavelengths, dtype=np.float64),
            np.array(n_values, dtype=np.float64),
            np.array(k_values, dtype=np.float64))


def load_material_from_yaml(filepath: str) -> Optional[MaterialData]:
    """
    Load a single material from RefractiveIndex.info YAML file.

    Extracts 'tabulated nk' data, converts to reflectance, and returns
    a MaterialData object compatible with the existing pipeline.

    Args:
        filepath: Path to .yml file

    Returns:
        MaterialData object, or None if file doesn't contain usable nk data
    """
    path = Path(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)

        if not doc or 'DATA' not in doc:
            logger.debug(f"{path.name}: No DATA section")
            return None

        # Extract metadata
        references = doc.get('REFERENCES', '')
        comments = doc.get('COMMENTS', '')

        # Extract author from filename (most reliable method)
        # RefractiveIndex.info naming convention: Author.yml or Author-variant.yml
        # Examples: Babar.yml, Ferrera-298K.yml, Rakic-BB.yml
        author = path.stem  # e.g., "Babar", "Ferrera-298K", "Rakic-BB"

        # Extract material symbol from path
        # Path structure: data/main/MaterialSymbol/nk/Author.yml
        parts = path.parts
        material_symbol = "Unknown"
        for i, part in enumerate(parts):
            if part in ['main', 'organic', 'glass', 'other']:
                if i + 1 < len(parts):
                    material_symbol = parts[i + 1]
                break

        # Look for 'tabulated nk' data
        data_section = doc['DATA']
        if not isinstance(data_section, list):
            data_section = [data_section]

        nk_data = None
        for entry in data_section:
            if entry.get('type') == 'tabulated nk':
                nk_data = entry.get('data', '')
                break

        if not nk_data:
            logger.debug(f"{path.name}: No 'tabulated nk' data found")
            return None

        # Parse nk data
        wavelengths_um, n_values, k_values = parse_tabulated_nk(nk_data)

        if len(wavelengths_um) < 3:
            logger.warning(f"{path.name}: Insufficient data points ({len(wavelengths_um)})")
            return None

        # Convert n,k to reflectance
        reflectance = nk_to_reflectance(n_values, k_values)

        # Build material name
        material_name = f"{material_symbol} ({author})"

        # Extract record ID from filename (use stem as unique identifier)
        record_id = path.stem

        # Determine chapter/category
        chapter = ""
        for part in parts:
            if part in ['main', 'organic', 'glass', 'other']:
                chapter = part
                break

        # Validate wavelength range
        wl_min, wl_max = wavelengths_um[0], wavelengths_um[-1]
        logger.debug(f"{material_name}: {len(wavelengths_um)} points, {wl_min:.3f}-{wl_max:.3f} µm")

        return MaterialData(
            name=material_name,
            record_id=record_id,
            instrument="RefractiveIndex.info",
            chapter=chapter,
            filename=path.name,
            wavelengths=wavelengths_um,
            reflectance=reflectance
        )

    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


def discover_refractiveindex_materials(root_dir: str,
                                       pattern: str = "**/*.yml",
                                       category_filter: Optional[str] = None) -> List[str]:
    """
    Recursively find all RefractiveIndex YAML files.

    Args:
        root_dir: Root directory (typically database/data)
        pattern: Glob pattern for YAML files
        category_filter: If set, only include files from this category
                        (e.g., "main", "glass", "organic")

    Returns:
        List of absolute file paths
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    files = sorted(root.glob(pattern))

    # Filter by category if requested
    if category_filter:
        files = [f for f in files if category_filter in f.parts]

    # Exclude metadata files
    files = [f for f in files if f.name not in ['about.yml', 'catalog-nk.yml', 'catalog-n2.yml']]

    logger.info(f"Discovered {len(files)} RefractiveIndex files in {root_dir}")

    return [str(f) for f in files]


def load_all_refractiveindex_materials(root_dir: str,
                                       pattern: str = "data/main/**/nk/*.yml",
                                       max_materials: Optional[int] = None) -> List[MaterialData]:
    """
    Load all materials from RefractiveIndex.info database.

    Args:
        root_dir: Database root directory (containing 'data' folder)
        pattern: Glob pattern relative to root_dir
        max_materials: Optional limit for testing

    Returns:
        List of MaterialData objects
    """
    root = Path(root_dir)

    # Discover material files
    search_path = root / pattern.split('/')[0]  # Get first component
    remaining_pattern = '/'.join(pattern.split('/')[1:])

    material_files = discover_refractiveindex_materials(
        str(search_path),
        pattern=remaining_pattern,
        category_filter='main'
    )

    if max_materials:
        material_files = material_files[:max_materials]
        logger.info(f"Limited to {max_materials} materials for testing")

    # Load each material
    materials = []
    failed = 0

    for filepath in material_files:
        mat = load_material_from_yaml(filepath)
        if mat is not None:
            materials.append(mat)
        else:
            failed += 1

    logger.info(f"Loaded {len(materials)} materials successfully, {failed} failed/skipped")

    return materials


def filter_by_wavelength_coverage(materials: List[MaterialData],
                                  min_wavelength: float,
                                  max_wavelength: float,
                                  min_coverage: float = 0.8) -> List[MaterialData]:
    """
    Filter materials that have sufficient coverage of a wavelength range.

    Args:
        materials: List of MaterialData objects
        min_wavelength: Minimum required wavelength (micrometers)
        max_wavelength: Maximum required wavelength (micrometers)
        min_coverage: Minimum fraction of range that must be covered (0-1)

    Returns:
        Filtered list of materials
    """
    filtered = []

    for mat in materials:
        wl_min, wl_max = mat.wavelength_range

        # Check if material covers the required range
        overlap_start = max(wl_min, min_wavelength)
        overlap_end = min(wl_max, max_wavelength)

        if overlap_end <= overlap_start:
            continue  # No overlap

        overlap_range = overlap_end - overlap_start
        required_range = max_wavelength - min_wavelength
        coverage = overlap_range / required_range

        if coverage >= min_coverage:
            filtered.append(mat)

    return filtered


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s: %(message)s')

    # Test paths
    REFIDX_ROOT = "../../assets/refractiveindex/database"

    print("=== RefractiveIndex Loader Test ===\n")

    # Test single material (Silver)
    ag_file = Path(REFIDX_ROOT) / "data/main/Ag/nk/Babar.yml"
    if ag_file.exists():
        print(f"Loading test file: {ag_file.name}")
        mat = load_material_from_yaml(str(ag_file))

        if mat:
            print(f"\nMaterial: {mat.name}")
            print(f"  Record ID: {mat.record_id}")
            print(f"  Wavelength range: {mat.wavelength_range[0]:.3f} - {mat.wavelength_range[1]:.3f} µm")
            print(f"  Data points: {mat.num_samples}")
            print(f"  Reflectance range: {mat.reflectance.min():.3f} - {mat.reflectance.max():.3f}")

    # Test batch loading
    print("\n=== Loading all main category materials (first 10) ===")
    materials = load_all_refractiveindex_materials(REFIDX_ROOT, max_materials=10)

    print(f"\nLoaded {len(materials)} materials:")
    for mat in materials[:5]:
        print(f"  - {mat.name}")
        print(f"    Range: {mat.wavelength_range[0]:.3f} - {mat.wavelength_range[1]:.3f} µm")
        print(f"    Reflectance: {mat.reflectance.min():.3f} - {mat.reflectance.max():.3f}")

    # Test wavelength filtering
    print("\n=== Filter materials covering VIS range (0.35-0.78 µm) ===")
    vis_materials = filter_by_wavelength_coverage(materials, 0.35, 0.78, min_coverage=0.8)
    print(f"Found {len(vis_materials)} materials with >80% VIS coverage")
