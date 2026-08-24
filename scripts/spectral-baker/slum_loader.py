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

# Sample descriptions, transcribed from the LUMA SLUM documentation PDF:
#   https://urban-meteorology-reading.github.io/other%20files/LUMA_SLUM.pdf
# which gives Class / Material / Colour / Status / Dimension for each of the 74
# samples. Key = sample ID, value = (description, material category); the
# trailing comment is the manual's Material field, kept because it distinguishes
# samples the Colour and Class fields do not.
#
# These were WRONG until 2026-08-24, and wrong in a way worth recording. The
# previous table claimed the same source and did not come from it: it called
# B003 "Red brick, smooth" where the manual says black cement brick, and called
# X001-X003 "Grass" and "Soil" where they are quartzite conglomerate. Three
# independent checks agree with the manual and against it -- Kotthaus et al.
# (2014) say in the text "the black cement brick (B006) has a constantly low
# reflectance" and give a class table in which G is Granite, and the measured
# spectra themselves match the manual's colours in 25 of the 26 brick and
# roofing-tile samples. The exception is R001, whose manual entry reads Black
# while its short-wave reflectance is a strong red; that one disagreement is the
# data's, not this table's, and it is left as the manual has it.
#
# The names matter because a scene binds a material BY NAME through
# spectral_material_ref -- so a wrong name here is not a cosmetic slip, it hands
# the renderer a different measurement than the one that was asked for.
SAMPLE_INFO: Dict[str, tuple] = {
    # --- Asphalt ---
    "A001": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A002": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A003": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A004": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A005": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A006": ("Black/grey road asphalt, weathered", "asphalt"),                # Asphalt with stone aggregate
    "A007": ("Grey asphalt roofing, new", "asphalt"),                         # Asphalt roofing shingle with slate chippings
    "A008": ("Black road asphalt, weathered", "asphalt"),                     # Tarmac
    "A009": ("Black road asphalt, weathered", "asphalt"),                     # Tarmac
    "A010": ("Black road asphalt, weathered", "asphalt"),                     # Tarmac
    # --- Brick ---
    "B001": ("Yellow cement brick, new", "brick"),                            # Cement
    "B002": ("Black/light grey cement brick, new, sandy", "brick"),           # Cement
    "B003": ("Black cement brick, new", "brick"),                             # Cement
    "B004": ("Red ceramic brick, weathered", "brick"),                        # Ceramic with cement
    "B005": ("Red cement brick, weathered", "brick"),                         # Cement
    "B006": ("Black cement brick, sandy", "brick"),                           # Cement
    "B007": ("Light red cement brick, new", "brick"),                         # Cement
    "B008": ("Light red ceramic brick, new", "brick"),                        # Ceramic
    "B009": ("Red cement brick, weathered", "brick"),                         # Cement
    "B010": ("Red with beige and grey paint ceramic brick, weathered", "brick"),# Ceramic with paint
    "B011": ("Red/grey ceramic brick, weathered", "brick"),                   # Ceramic brick with cement
    "B012": ("Red with white paint ceramic brick, weathered", "brick"),       # Ceramic brick with paint
    "B013": ("Red ceramic brick, weathered", "brick"),                        # Ceramic brick
    "B014": ("Yellow/grey ceramic brick, weathered", "brick"),                # Ceramic brick
    # --- Concrete and cement ---
    "C001": ("Grey/ochre cement, weathered", "concrete"),                     # Cement
    "C002": ("Grey/white concrete, new", "concrete"),                         # Concrete with small aggregate
    "C003": ("Grey cement, weathered", "concrete"),                           # Cement
    "C004": ("Grey concrete, weathered", "concrete"),                         # Concrete with small stone aggregate
    "C005": ("Grey cement, weathered", "concrete"),                           # Cement
    "C006": ("White concrete, weathered", "concrete"),                        # Concrete with small stone aggregate
    "C008": ("Grey concrete, weathered/rough", "concrete"),                   # Concrete
    # --- Granite ---
    "G001": ("White/black granite, new, rough", "granite"),                   # Granite
    "G002": ("White/red granite, weathered", "granite"),                      # Granite with cement
    "G003": ("White/black granite, weathered", "granite"),                    # Granite with cement
    "G004": ("White/red/black granite, new, dusty", "granite"),               # Granite
    "G005": ("Red/black granite, new", "granite"),                            # Granite
    # --- Roofing shingle ---
    "L001d": ("Grey roofing shingle, clear", "roofing_shingle"),              # Slate
    "L001u": ("Grey roofing shingle, weathered", "roofing_shingle"),          # Slate roofing shingle
    "L002": ("Black roofing shingle, weathered", "roofing_shingle"),          # Fibre cement
    "L003": ("Black roofing shingle, weathered", "roofing_shingle"),          # Fibre cement
    # --- Roofing tile ---
    "R001": ("Black roofing tile, new", "roof_tile"),                         # Ceramic
    "R002": ("Brown roofing tile, new", "roof_tile"),                         # Ceramic
    "R003": ("Rustic red roofing tile, new", "roof_tile"),                    # Cement
    "R004": ("Burnt red roofing tile, new", "roof_tile"),                     # Ceramic
    "R005": ("Rustic red/black shading roofing tile, new/shiny", "roof_tile"),# Cement
    "R006": ("Slate grey roofing tile, new", "roof_tile"),                    # Cement
    "R007": ("Black roofing tile, new", "roof_tile"),                         # Ceramic
    "R008": ("Rustic red roofing tile, new", "roof_tile"),                    # Cement
    "R009": ("Autumn red roofing tile, new/rough", "roof_tile"),              # Cement
    "R010": ("Red roofing tile, weathered", "roof_tile"),                     # Ceramic
    "R012": ("Red roofing tile, weathered", "roof_tile"),                     # Ceramic
    "R013": ("Red roofing tile, weathered", "roof_tile"),                     # Ceramic
    # --- Stone ---
    "S001": ("Beige stone, weathered", "stone"),                              # Sandstone
    "S002": ("Grey stone, weathered", "stone"),                               # Carboniferous coral limestone
    "S003": ("Ochre stone, weathered", "stone"),                              # Sandstone
    "S004": ("Beige stone, weathered", "stone"),                              # Limestone
    "S005": ("Light grey stone, weathered", "stone"),                         # Sandstone
    # --- PVC roofing sheet ---
    "V001": ("Lead grey pvc roofing sheet, new", "pvc"),                      # PVC
    "V002": ("Light grey pvc roofing sheet, new", "pvc"),                     # PVC
    "V003": ("Copper pvc roofing sheet, new/structured", "pvc"),              # PVC
    "V004": ("Azure blue pvc roofing sheet, new", "pvc"),                     # PVC
    "V005": ("Copper brown pvc roofing sheet, new", "pvc"),                   # PVC
    "V006": ("Copper patina pvc roofing sheet, new", "pvc"),                  # PVC
    # --- Quartzite conglomerate ---
    "X001": ("Beige/brown/black/red quartzite conglomerate, new", "quartzite"),# Quartzite
    "X002": ("Beige/brown/black quartzite conglomerate, new", "quartzite"),   # Quartzite
    "X003": ("Beige/brown/black/red quartzite conglomerate, new", "quartzite"),# Quartzite
    # --- Metal ---
    "Z001": ("Dull grey metal, new", "metal"),                                # Aluminium plus zinc
    "Z002": ("Shiny grey metal, new", "metal"),                               # Aluminium, stucco
    "Z003": ("Dark green metal, new", "metal"),                               # Metal with paint
    "Z004": ("Copper patina metal, weathered", "metal"),                      # Metal with paint
    "Z005": ("Slate grey metal, new", "metal"),                               # Metal with paint
    "Z006": ("Grey metal, weathered", "metal"),                               # Aluminium
    "Z007": ("Grey metal, weathered", "metal"),                               # Lead
    "Z008": ("Black metal, weathered", "metal"),                              # Iron
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

    # Sanity: Z002 (stucco aluminium) should have IR reflectance ~0.84
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
