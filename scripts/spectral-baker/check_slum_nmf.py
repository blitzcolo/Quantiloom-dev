#!/usr/bin/env python3
"""§六.1 health check: project SLUM samples onto the existing ECOSTRESS NMF
basis and report per-band reconstruction RMSE.

This checks whether the basis trained on ECOSTRESS can represent SLUM
materials. If PVC/organics are much worse than the concrete benchmark,
the joint re-bake (Phase 1) must retrain the basis — which it does
anyway; re-run this against the merged basis afterwards.

Benchmarks (from SPECTRAL_ASSIGNMENT_PLAN.md):
  Construction Concrete (0598UUUCNC): VIS RMSE 0.0029, LWIR RMSE 0.0013

Pass criterion: SLUM medians within ~3x of the concrete benchmarks.
"""

import pathlib
import sys
import numpy as np
from scipy.optimize import nnls

# Add this directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from slum_loader import load_slum_materials
from exporter import read_basis_binary
from spectral_processor import BandConfig, resample_uniform


ECOSTRESS_BASIS = pathlib.Path(__file__).parent / "../../assets/spectral/quantiloom_basis_v3_ecostress.qlbin"
SLUM_DATA_DIR = pathlib.Path(__file__).parent / "data" / "slum"

BAND_NAMES = ["VIS", "NIR", "SWIR", "MWIR", "LWIR"]

CONCRETE_BENCHMARKS = {
    "VIS": 0.0029,
    "LWIR": 0.0013,
}


def project_onto_basis(spectrum: np.ndarray, basis: np.ndarray) -> tuple:
    """NNLS projection of a spectrum onto a basis.

    Returns (weights, reconstructed, rmse).
    """
    # basis is [n_components x n_wavelengths], spectrum is [n_wavelengths]
    # Solve: min ||B^T w - s||^2, w >= 0
    weights, residual = nnls(basis.T, spectrum)
    reconstructed = weights @ basis
    rmse = float(np.sqrt(np.mean((spectrum - reconstructed) ** 2)))
    return weights, reconstructed, rmse


def main():
    print("=" * 60)
    print("§六.1 SLUM NMF Health Check")
    print("=" * 60)

    # Load SLUM
    materials = load_slum_materials(str(SLUM_DATA_DIR))
    print()

    # Load ECOSTRESS basis
    basis_data = read_basis_binary(str(ECOSTRESS_BASIS))
    print(f"Basis: {basis_data['num_bands']} bands, version {basis_data['version']}")

    # Band configs matching the basis
    band_configs = {
        "VIS":  BandConfig("VIS",  (0.350, 0.780), 2.0, 16),
        "NIR":  BandConfig("NIR",  (0.780, 1.100), 2.0, 16),
        "SWIR": BandConfig("SWIR", (1.100, 2.500), 2.0, 32),
        "MWIR": BandConfig("MWIR", (2.500, 6.500), 5.0, 48),
        "LWIR": BandConfig("LWIR", (6.500, 15.000), 10.0, 48),
    }

    # For each band, project each SLUM sample
    results = {}
    for band_idx, band_name in enumerate(BAND_NAMES):
        band_basis = basis_data["bands"][band_idx]
        basis_matrix = band_basis["basis_data"]  # [n_components x n_wavelengths]
        wl_start, wl_end = band_basis["wavelength_range"]
        n_samples = band_basis["num_samples"]

        target_wl = np.linspace(wl_start, wl_end, n_samples)  # µm

        band_rmses = []
        band_sample_ids = []

        for mat in materials:
            # Check coverage: does this material have data in this band?
            mat_min, mat_max = mat.wavelength_range
            overlap = min(wl_end, mat_max) - max(wl_start, mat_min)
            coverage = max(0.0, overlap / (wl_end - wl_start))
            if coverage < 0.1:
                continue

            # Resample to basis grid
            resampled = resample_uniform(
                mat.wavelengths, mat.reflectance, target_wl
            )
            resampled = np.clip(resampled, 0.0, 1.0)

            # Project
            _, _, rmse = project_onto_basis(resampled, basis_matrix)
            band_rmses.append(rmse)
            band_sample_ids.append(mat.record_id)

        results[band_name] = {
            "rmses": np.array(band_rmses),
            "ids": band_sample_ids,
        }

    # Report
    print()
    print(f"{'Band':<6} {'Count':>5} {'Median':>8} {'P25':>8} {'P75':>8} {'Max':>8}  {'Bench':>8}  {'Ratio':>6}")
    print("-" * 70)

    pass_all = True
    for band_name in ["VIS", "LWIR"]:
        r = results[band_name]
        rmses = r["rmses"]
        if len(rmses) == 0:
            print(f"{band_name:<6} {'N/A':>5}")
            continue

        med = np.median(rmses)
        p25 = np.percentile(rmses, 25)
        p75 = np.percentile(rmses, 75)
        mx = np.max(rmses)
        bench = CONCRETE_BENCHMARKS.get(band_name, float("nan"))
        ratio = med / bench if bench > 0 else float("nan")

        status = "PASS" if ratio < 3.0 else "WARN"
        if ratio >= 3.0:
            pass_all = False

        print(f"{band_name:<6} {len(rmses):>5} {med:>8.4f} {p25:>8.4f} {p75:>8.4f} {mx:>8.4f}  {bench:>8.4f}  {ratio:>5.1f}x  {status}")

    # Also show NIR/SWIR for completeness
    print()
    for band_name in ["NIR", "SWIR", "MWIR"]:
        r = results[band_name]
        rmses = r["rmses"]
        if len(rmses) == 0:
            print(f"{band_name:<6}  no coverage")
            continue
        med = np.median(rmses)
        print(f"{band_name:<6} {len(rmses):>5} median={med:.4f}")

    # Show worst cases per band
    print()
    for band_name in ["VIS", "LWIR"]:
        r = results[band_name]
        rmses = r["rmses"]
        ids = r["ids"]
        if len(rmses) == 0:
            continue
        order = np.argsort(rmses)[::-1]
        print(f"Worst 5 in {band_name}:")
        for i in order[:5]:
            mat = next(m for m in materials if m.record_id == ids[i])
            print(f"  {rmses[i]:.4f}  {mat.name}")

    # Category breakdown for VIS
    print()
    print("VIS RMSE by category:")
    r = results["VIS"]
    categories = {}
    for rmse_val, sid in zip(r["rmses"], r["ids"]):
        cat = sid[0]
        categories.setdefault(cat, []).append(rmse_val)
    for cat in sorted(categories):
        vals = np.array(categories[cat])
        cat_names = {"A": "asphalt", "B": "brick", "C": "concrete", "G": "stone",
                     "L": "slate", "R": "roof tile", "S": "slate/fibre", "V": "PVC",
                     "X": "misc", "Z": "metal"}
        print(f"  {cat} ({cat_names.get(cat, '?'):<12}): n={len(vals):>2}, "
              f"median={np.median(vals):.4f}, max={vals.max():.4f}")

    print()
    if pass_all:
        print("PASS: SLUM materials are adequately represented by the ECOSTRESS basis.")
    else:
        print("WARN: Some bands exceed 3x the concrete benchmark.")
        print("      The joint re-bake will retrain the basis; re-run this check afterwards.")

    return 0 if pass_all else 1


if __name__ == "__main__":
    sys.exit(main())
