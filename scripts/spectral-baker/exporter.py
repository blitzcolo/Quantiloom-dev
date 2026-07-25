#!/usr/bin/env python3
"""
Spectral Data Exporter (SpectralBaker v3.0)

Writes NMF basis functions and material weights to binary and JSON formats.
Binary format is optimized for C++ loading.

Author: Quantiloom Team
"""

import json
import struct
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from spectral_processor import BandBasis, MaterialWeights

logger = logging.getLogger(__name__)

# Binary format constants
MAGIC = b'QBAS'
VERSION = 3  # Version 3: multi-source support (USGS + RefractiveIndex), 5 bands, fixed metrics
HEADER_SIZE = 64
RESERVED_SIZE = 52  # Increased since we removed global n_basis


def write_basis_binary(filepath: str,
                       band_bases: Dict[str, BandBasis],
                       band_order: List[str] = None) -> int:
    """
    Write global basis functions to binary file.

    File format v3 (little-endian) - supports per-band basis counts:
        Header (64 bytes):
            - magic: 4 bytes ("QBAS")
            - version: uint32 (3)
            - num_bands: uint32
            - reserved: 52 bytes

        For each band:
            BandHeader (16 bytes):
                - wavelength_start: float32 (micrometers)
                - wavelength_end: float32 (micrometers)
                - num_samples: uint32
                - num_basis: uint32 (per-band basis count)
            BandData:
                - basis_data: float32[num_basis × num_samples] (row-major)

    Args:
        filepath: Output file path
        band_bases: Dict of BandBasis objects
        band_order: Order of bands to write (default: ['VIS', 'NIR', 'SWIR'])

    Returns:
        Total bytes written
    """
    if band_order is None:
        band_order = ['VIS', 'NIR', 'SWIR', 'MWIR', 'LWIR']

    # Filter to only bands that exist
    band_order = [b for b in band_order if b in band_bases]

    if not band_order:
        raise ValueError("No valid bands to write")

    num_bands = len(band_order)

    with open(filepath, 'wb') as f:
        # Write header (no global n_basis in v2)
        header = struct.pack('<4sII',
                            MAGIC,
                            VERSION,
                            num_bands)
        header += b'\x00' * RESERVED_SIZE  # Reserved padding
        assert len(header) == HEADER_SIZE
        f.write(header)

        # Write each band
        for band_name in band_order:
            basis = band_bases[band_name]
            wavelengths = basis.wavelengths
            data = basis.basis_functions.astype(np.float32)
            n_basis = data.shape[0]
            n_samples = data.shape[1]

            # Band header (16 bytes): wavelength_start, wavelength_end, num_samples, num_basis
            band_header = struct.pack('<ffII',
                                     float(wavelengths[0]),   # wavelength_start
                                     float(wavelengths[-1]),  # wavelength_end
                                     n_samples,               # num_samples
                                     n_basis)                 # num_basis (per-band)
            f.write(band_header)

            # Basis data (row-major: [n_basis × n_wavelengths])
            f.write(data.tobytes())

            logger.debug(f"Wrote {band_name}: {n_basis} basis × {n_samples} samples ({data.nbytes} bytes)")

        total_bytes = f.tell()

    logger.info(f"Wrote basis file: {filepath} ({total_bytes:,} bytes)")

    return total_bytes


def read_basis_binary(filepath: str) -> Dict[str, Any]:
    """
    Read binary basis file (for verification).

    Returns dict with:
        - version: int
        - num_bands: int
        - bands: list of dicts with wavelength_range, num_samples, num_basis, basis_data
    """
    with open(filepath, 'rb') as f:
        # Read header
        header = f.read(HEADER_SIZE)
        magic, version, num_bands = struct.unpack('<4sII', header[:12])

        if magic != MAGIC:
            raise ValueError(f"Invalid magic number: {magic}")

        result = {
            'version': version,
            'num_bands': num_bands,
            'bands': []
        }

        # Read each band
        for _ in range(num_bands):
            # Band header (16 bytes in v2)
            band_header = f.read(16)
            wl_start, wl_end, num_samples, n_basis = struct.unpack('<ffII', band_header)

            # Basis data
            data_size = n_basis * num_samples * 4  # float32
            data = np.frombuffer(f.read(data_size), dtype=np.float32)
            data = data.reshape(n_basis, num_samples)

            result['bands'].append({
                'wavelength_range': (wl_start, wl_end),
                'num_samples': num_samples,
                'num_basis': n_basis,
                'basis_data': data
            })

    return result


def write_material_json(filepath: str,
                        material_weights: List[MaterialWeights],
                        metadata: Dict[str, Any] = None) -> None:
    """
    Write material weights to JSON file.

    Schema:
    {
        "metadata": {
            "generator": "SpectralBaker v1.0",
            "source_library": "USGS splib07a",
            "date_generated": "2025-12-23T10:30:00Z",
            "algorithm": "NMF",
            "num_basis": 16,
            "num_materials": 2847
        },
        "materials": {
            "Material_Name": {
                "source": { filename, record_id, instrument, chapter },
                "bands": {
                    "VIS": { wavelength_range_um, basis_weights, rmse, explained_variance },
                    "SWIR": { ... }
                }
            }
        }
    }

    Args:
        filepath: Output JSON path
        material_weights: List of MaterialWeights objects
        metadata: Additional metadata to include
    """
    if metadata is None:
        metadata = {}

    # Build default metadata
    default_meta = {
        'generator': 'SpectralBaker v3.0',
        'source_library': 'USGS splib07a',
        'date_generated': datetime.utcnow().isoformat() + 'Z',
        'algorithm': 'NMF',
        'num_materials': len(material_weights)
    }

    # Infer num_basis from first material
    if material_weights:
        first_band = list(material_weights[0].band_weights.keys())[0]
        default_meta['num_basis'] = len(material_weights[0].band_weights[first_band])

    default_meta.update(metadata)

    # Build materials dict
    materials_dict = {}

    for mat in material_weights:
        # Create unique key (handle duplicates)
        key = mat.name
        if key in materials_dict:
            # Append record_id for uniqueness
            key = f"{mat.name}_{mat.record_id}"

        bands_data = {}
        for band_name, weights in mat.band_weights.items():
            # 'coverage' is the fraction of the band the source actually measured.
            # Below 1.0 the remainder is extrapolated, and rmse/explained_variance
            # are correspondingly optimistic -- consumers must read the two together.
            bands_data[band_name] = {
                'basis_weights': [float(w) for w in weights],
                'rmse': mat.band_rmse.get(band_name, 0.0),
                'explained_variance': mat.band_variance.get(band_name, 0.0),
                'coverage': mat.band_coverage.get(band_name, 1.0)
            }

        materials_dict[key] = {
            'source': {
                'filename': mat.filename,
                'record_id': mat.record_id,
                'instrument': mat.instrument,
                'chapter': mat.chapter
            },
            'bands': bands_data
        }

    output = {
        'metadata': default_meta,
        'materials': materials_dict
    }

    # Write with pretty printing
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size = Path(filepath).stat().st_size
    logger.info(f"Wrote materials JSON: {filepath} ({file_size:,} bytes)")


def write_cpp_header(filepath: str,
                     band_bases: Dict[str, BandBasis],
                     band_order: List[str] = None) -> None:
    """
    Write C++ header with embedded basis data (alternative to binary).

    Useful for small datasets or embedded systems.
    """
    if band_order is None:
        band_order = ['VIS', 'SWIR']

    n_basis = band_bases[band_order[0]].basis_functions.shape[0]

    with open(filepath, 'w') as f:
        f.write("// Auto-generated by SpectralBaker v3.0\n")
        f.write(f"// Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("namespace quantiloom {\n")
        f.write("namespace spectral {\n\n")

        f.write(f"constexpr uint32_t NUM_BASIS = {n_basis};\n")
        f.write(f"constexpr uint32_t NUM_BANDS = {len(band_order)};\n\n")

        for band_name in band_order:
            basis = band_bases[band_name]
            n_samples = len(basis.wavelengths)

            f.write(f"// {band_name} Band: {basis.wavelengths[0]:.3f} - {basis.wavelengths[-1]:.3f} µm\n")
            f.write(f"constexpr uint32_t {band_name}_NUM_SAMPLES = {n_samples};\n")
            f.write(f"constexpr float {band_name}_WAVELENGTH_START = {basis.wavelengths[0]:.6f}f;\n")
            f.write(f"constexpr float {band_name}_WAVELENGTH_END = {basis.wavelengths[-1]:.6f}f;\n\n")

            # Write basis data as C array
            data = basis.basis_functions.astype(np.float32).flatten()
            f.write(f"constexpr float {band_name}_BASIS[{len(data)}] = {{\n")

            # Write 8 values per line
            for i in range(0, len(data), 8):
                line = ", ".join(f"{v:.8f}f" for v in data[i:i+8])
                f.write(f"    {line},\n")

            f.write("};\n\n")

        f.write("}  // namespace spectral\n")
        f.write("}  // namespace quantiloom\n")

    file_size = Path(filepath).stat().st_size
    logger.info(f"Wrote C++ header: {filepath} ({file_size:,} bytes)")


def write_summary_csv(filepath: str,
                      material_weights: List[MaterialWeights]) -> None:
    """
    Write per-material statistics to CSV for analysis.
    """
    bands = list(material_weights[0].band_weights.keys()) if material_weights else []

    with open(filepath, 'w') as f:
        # Header
        headers = ['name', 'record_id', 'instrument', 'chapter', 'filename']
        for band in bands:
            headers.extend([f'{band}_rmse', f'{band}_variance', f'{band}_coverage'])
        f.write(','.join(headers) + '\n')

        # Data rows
        for mat in material_weights:
            row = [
                f'"{mat.name}"',
                mat.record_id,
                mat.instrument,
                mat.chapter,
                mat.filename
            ]
            for band in bands:
                row.extend([
                    f"{mat.band_rmse.get(band, 0.0):.6f}",
                    f"{mat.band_variance.get(band, 0.0):.6f}",
                    f"{mat.band_coverage.get(band, 1.0):.6f}"
                ])
            f.write(','.join(row) + '\n')

    logger.info(f"Wrote summary CSV: {filepath}")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s: %(message)s')

    # Create dummy data for testing
    print("=== Exporter Test ===\n")

    # Fake basis data
    vis_basis = BandBasis(
        name='VIS',
        wavelengths=np.linspace(0.35, 0.78, 216),
        basis_functions=np.random.rand(16, 216).astype(np.float32),
        explained_variance=0.98
    )

    swir_basis = BandBasis(
        name='SWIR',
        wavelengths=np.linspace(0.78, 2.5, 860),
        basis_functions=np.random.rand(16, 860).astype(np.float32),
        explained_variance=0.97
    )

    band_bases = {'VIS': vis_basis, 'SWIR': swir_basis}

    # Fake materials
    materials = [
        MaterialWeights(
            name='Test Material 1',
            record_id='12345',
            instrument='ASDFRa',
            chapter='ChapterA',
            filename='test1.txt',
            band_weights={'VIS': np.random.rand(16), 'SWIR': np.random.rand(16)},
            band_rmse={'VIS': 0.01, 'SWIR': 0.02},
            band_variance={'VIS': 0.98, 'SWIR': 0.97}
        ),
        MaterialWeights(
            name='Test Material 2',
            record_id='67890',
            instrument='ASDFRa',
            chapter='ChapterB',
            filename='test2.txt',
            band_weights={'VIS': np.random.rand(16), 'SWIR': np.random.rand(16)},
            band_rmse={'VIS': 0.015, 'SWIR': 0.025},
            band_variance={'VIS': 0.97, 'SWIR': 0.96}
        )
    ]

    # Test binary export
    test_dir = Path('/tmp/spectral_baker_test')
    test_dir.mkdir(exist_ok=True)

    bin_path = test_dir / 'test_basis.bin'
    write_basis_binary(str(bin_path), band_bases)

    # Verify by reading back
    read_data = read_basis_binary(str(bin_path))
    print(f"\nVerification:")
    print(f"  Version: {read_data['version']}")
    print(f"  Bands: {read_data['num_bands']}")
    print(f"  Basis: {read_data['num_basis']}")
    for i, band in enumerate(read_data['bands']):
        print(f"  Band {i}: {band['wavelength_range']}, {band['num_samples']} samples")

    # Test JSON export
    json_path = test_dir / 'test_materials.json'
    write_material_json(str(json_path), materials)

    # Test CSV export
    csv_path = test_dir / 'test_summary.csv'
    write_summary_csv(str(csv_path), materials)

    print(f"\nTest files written to: {test_dir}")
