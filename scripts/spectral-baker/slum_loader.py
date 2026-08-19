"""SLUM/LUMA Spectral Library loader for Quantiloom spectral baker.

Reads the two CSV files from Kotthaus et al. 2014 (Zenodo 10.5281/zenodo.4263842):
  LUMA_SLUM_SW.csv  — shortwave reflectance, 348–2506 nm, percent
  LUMA_SLUM_IR.csv  — longwave EMISSIVITY, 8.0–14.0 µm, fractional

The IR file is inverted once: rho = 1 - eps (opaque surfaces, tau = 0).
"""

import pathlib
from typing import Dict, List, Optional

import numpy as np

from usgs_loader import MaterialData

# Descriptions from Kotthaus et al. 2014 Table C.1.
# Key = sample ID, value = (short description, material category).
SAMPLE_INFO: Dict[str, tuple] = {
    "A001": ("Dark new asphalt, road", "asphalt"),
    "A002": ("Medium grey asphalt, road", "asphalt"),
    "A003": ("Dark grey asphalt, road", "asphalt"),
    "A004": ("Aged dark asphalt, pavement", "asphalt"),
    "A005": ("Grey asphalt, pavement", "asphalt"),
    "A006": ("Black asphalt, fresh patch", "asphalt"),
    "A007": ("Dark asphalt, road surface", "asphalt"),
    "A008": ("Grey asphalt, kerb", "asphalt"),
    "A009": ("Weathered asphalt, car park", "asphalt"),
    "A010": ("Grey asphalt, car park", "asphalt"),
    "B001": ("Yellow London stock brick", "brick"),
    "B002": ("Dark engineering brick", "brick"),
    "B003": ("Red brick, smooth", "brick"),
    "B004": ("Cream rendered wall", "brick"),
    "B005": ("Light brown brick", "brick"),
    "B006": ("White painted brick", "brick"),
    "B007": ("Yellow painted brick", "brick"),
    "B008": ("White painted render", "brick"),
    "B009": ("Buff brick", "brick"),
    "B010": ("Light grey mortar", "brick"),
    "B011": ("Dark brown brick", "brick"),
    "B012": ("Red-brown brick", "brick"),
    "B013": ("Light render, pebbledash", "brick"),
    "B014": ("Cream brick", "brick"),
    "C001": ("Grey concrete, precast", "concrete"),
    "C002": ("Pale grey concrete, paving", "concrete"),
    "C003": ("Dark grey concrete, paving", "concrete"),
    "C004": ("Light grey concrete, kerb", "concrete"),
    "C005": ("Weathered concrete, wall", "concrete"),
    "C006": ("Light concrete, paving", "concrete"),
    "C008": ("Dark concrete, block", "concrete"),
    "G001": ("York stone, paving", "stone"),
    "G002": ("Pale sandstone, wall", "stone"),
    "G003": ("Portland stone, wall", "stone"),
    "G004": ("White quartzite, chippings", "stone"),
    "G005": ("Grey granite, kerb", "stone"),
    "L001d": ("Dark slate, roof", "slate"),
    "L001u": ("Slate underside, roof", "slate"),
    "L002": ("Grey fibre cement, roof", "slate"),
    "L003": ("Fibre cement sheet", "slate"),
    "R001": ("Red clay roof tile", "roof_tile"),
    "R002": ("Dark red clay tile", "roof_tile"),
    "R003": ("Orange clay pantile", "roof_tile"),
    "R004": ("Brown clay tile", "roof_tile"),
    "R005": ("Dark brown concrete tile", "roof_tile"),
    "R006": ("Grey concrete tile", "roof_tile"),
    "R007": ("Red-brown concrete tile", "roof_tile"),
    "R008": ("Brown concrete tile", "roof_tile"),
    "R009": ("Dark concrete tile", "roof_tile"),
    "R010": ("Terracotta ridge tile", "roof_tile"),
    "R012": ("Weathered clay tile", "roof_tile"),
    "R013": ("Light grey concrete tile", "roof_tile"),
    "S001": ("Natural Welsh slate", "slate"),
    "S002": ("Dark grey slate", "slate"),
    "S003": ("Grey-green slate", "slate"),
    "S004": ("Fibre cement slate", "slate"),
    "S005": ("Grey fibre cement", "slate"),
    "V001": ("PVC membrane, lead grey", "pvc"),
    "V002": ("PVC membrane, light grey", "pvc"),
    "V003": ("PVC membrane, copper brown", "pvc"),
    "V004": ("PVC membrane, sky blue", "pvc"),
    "V005": ("PVC membrane, verdigris", "pvc"),
    "V006": ("PVC membrane, dark grey", "pvc"),
    "X001": ("Grass, dry cut", "miscellaneous"),
    "X002": ("Grass, green", "miscellaneous"),
    "X003": ("Soil, dark brown", "miscellaneous"),
    "Z001": ("Aluminium sheet, bright", "metal"),
    "Z002": ("Aluminium sheet, embossed", "metal"),
    "Z003": ("Painted metal, white", "metal"),
    "Z004": ("Painted metal, dark green", "metal"),
    "Z005": ("Painted metal, slate grey", "metal"),
    "Z006": ("Weathered iron, corrugated", "metal"),
    "Z007": ("Lead sheet, weathered", "metal"),
    "Z008": ("Zinc-aluminium sheet", "metal"),
}


def _read_csv(path: pathlib.Path) -> tuple:
    """Read a SLUM CSV, returning (wavelengths, sample_ids, data_matrix).

    wavelengths: 1-D array (raw units from file — nm for SW, µm for IR)
    sample_ids: list of column-header sample IDs
    data_matrix: 2-D array [n_wavelengths × n_samples]
    """
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_ids = header[1:]
        rows = []
        for row in reader:
            rows.append([float(v) for v in row])

    arr = np.array(rows, dtype=np.float64)
    wavelengths = arr[:, 0]
    data = arr[:, 1:]
    return wavelengths, sample_ids, data


def load_slum_materials(
    data_dir: str,
    sw_file: str = "LUMA_SLUM_SW.csv",
    ir_file: str = "LUMA_SLUM_IR.csv",
) -> List[MaterialData]:
    """Load all 74 SLUM samples as MaterialData objects.

    Each sample's spectrum is the concatenation of:
      SW reflectance (348–2506 nm, converted from percent to fractional)
      IR reflectance (8002–14000 nm, inverted from emissivity: rho = 1 - eps)
    with a 2506–8002 nm gap left empty (no interpolation).

    Returns a list of MaterialData (wavelengths in µm, reflectance in [0, 1]).
    """
    root = pathlib.Path(data_dir)
    sw_path = root / sw_file
    ir_path = root / ir_file

    sw_wl_nm, sw_ids, sw_data = _read_csv(sw_path)
    ir_wl_um, ir_ids, ir_data = _read_csv(ir_path)

    if sw_ids != ir_ids:
        raise ValueError(
            f"Sample ID mismatch between SW ({len(sw_ids)}) and IR ({len(ir_ids)}) files"
        )

    # --- Convert units ---
    # SW: nm -> µm, percent -> fractional
    sw_wl_um = sw_wl_nm / 1000.0
    sw_reflectance = sw_data / 100.0

    # IR: already in µm; values are EMISSIVITY -> invert ONCE to reflectance
    # Guard: assert values look like emissivity before inverting
    dielectric_mask = np.array([
        not sid.startswith("Z") for sid in ir_ids
    ])
    if dielectric_mask.any():
        dielectric_medians = np.median(ir_data[:, dielectric_mask], axis=0)
        if np.median(dielectric_medians) < 0.5:
            raise ValueError(
                "IR data medians < 0.5 for dielectrics — does not look like "
                "emissivity; refusing to invert (would double-invert)"
            )
    ir_reflectance = 1.0 - ir_data

    materials: List[MaterialData] = []

    for col_idx, sample_id in enumerate(sw_ids):
        sw_rho = sw_reflectance[:, col_idx]
        ir_rho = ir_reflectance[:, col_idx]

        # Clamp to [0, 1]
        sw_rho = np.clip(sw_rho, 0.0, 1.0)
        ir_rho = np.clip(ir_rho, 0.0, 1.0)

        # Concatenate the two segments
        wavelengths = np.concatenate([sw_wl_um, ir_wl_um])
        reflectance = np.concatenate([sw_rho, ir_rho])

        info = SAMPLE_INFO.get(sample_id, (sample_id, "unknown"))
        description, category = info

        mat = MaterialData(
            name=f"SLUM {description} ({sample_id})",
            record_id=sample_id,
            instrument="beckman+bruker",
            chapter=f"slum_{category}",
            filename=f"LUMA_SLUM_{sample_id}",
            wavelengths=wavelengths,
            reflectance=reflectance,
        )
        materials.append(mat)

    # Sanity: Z002 (embossed aluminium) should have IR reflectance ~0.84
    z002_idx = sw_ids.index("Z002")
    z002_ir_rho_mean = float(np.mean(ir_reflectance[:, z002_idx]))
    if not (0.75 < z002_ir_rho_mean < 0.90):
        raise ValueError(
            f"Z002 mean IR reflectance = {z002_ir_rho_mean:.3f}, "
            f"expected ~0.84 — inversion may be wrong"
        )

    print(f"SLUM: loaded {len(materials)} samples "
          f"(SW {sw_wl_um[0]:.3f}-{sw_wl_um[-1]:.3f} µm, "
          f"IR {ir_wl_um[0]:.3f}-{ir_wl_um[-1]:.3f} µm)")

    return materials


if __name__ == "__main__":
    import sys

    data_dir = str(pathlib.Path(__file__).parent / "data" / "slum")
    materials = load_slum_materials(data_dir)

    print(f"\nLoaded {len(materials)} materials:")
    for m in materials[:5]:
        print(f"  {m.name}: {m.wavelength_range[0]:.3f}-{m.wavelength_range[1]:.3f} µm, "
              f"{m.num_samples} points, "
              f"rho range [{m.reflectance.min():.3f}, {m.reflectance.max():.3f}]")
    print("  ...")
    # Print metals
    metals = [m for m in materials if m.record_id.startswith("Z")]
    for m in metals:
        ir_rho = m.reflectance[m.wavelengths > 5.0]
        print(f"  {m.name}: IR rho mean = {ir_rho.mean():.3f}")
