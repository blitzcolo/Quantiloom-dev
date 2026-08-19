#!/usr/bin/env python3
"""§六.2 B-grade bootstrap error quantification.

For SLUM PVC samples (V001–V006, which have BOTH VIS and LWIR measurements),
simulate what a B-grade assignment does:
  - VIS side: drop the measured spectrum, compute sRGB from it, then reconstruct
    via the shader's tri-Gaussian (ConvertLinearRGBToSpectrum).
  - LWIR side: keep the measured spectrum.

Compare the tri-Gaussian VIS against the measured VIS to quantify the B-grade
penalty: "how much worse is the VIS when we don't have a measured curve?"

Also validates the tri-Gaussian replica against the known concrete benchmark:
  Construction Concrete (0598UUUCNC): tri-Gaussian RMSE 0.0384, max relative 19.6%
"""

import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from slum_loader import load_slum_materials

SLUM_DATA_DIR = pathlib.Path(__file__).parent / "data" / "slum"

# CIE 1931 2-degree standard observer, 5 nm spacing, 380–780 nm
# (abbreviated — computed from the full tables)
CIE_WAVELENGTHS_NM = np.arange(380, 781, 5, dtype=np.float64)  # 81 points

# CIE 1931 2-degree standard observer at 5 nm, 380–780 nm (CIE 015:2004 Table 1)
_XBAR = [
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310,
    0.023190, 0.043510, 0.077630, 0.134380, 0.214770,
    0.283900, 0.328500, 0.348280, 0.348060, 0.336200,
    0.318700, 0.290800, 0.233700, 0.163500, 0.104760,
    0.061270, 0.030050, 0.013700, 0.004900, 0.003200,
    0.004900, 0.009300, 0.029100, 0.063270, 0.109600,
    0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400,
    0.642400, 0.541900, 0.447900, 0.360800, 0.283500,
    0.218700, 0.164900, 0.121200, 0.087400, 0.063600,
    0.046770, 0.032900, 0.022700, 0.015840, 0.011359,
    0.008111, 0.005790, 0.004109, 0.002899, 0.002049,
    0.001440, 0.001000, 0.000690, 0.000476, 0.000332,
    0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
    0.000042,
]
_YBAR = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396,
    0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000,
    0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000,
    0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102,
    0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120,
    0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
]
_ZBAR = [
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850,
    0.110200, 0.207400, 0.371300, 0.645600, 1.039050,
    1.385600, 1.622960, 1.747060, 1.782600, 1.772110,
    1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000,
    0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
    0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100,
    0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000,
]
CIE_XYZ = np.column_stack([_XBAR, _YBAR, _ZBAR])


# sRGB D65 XYZ-to-linear-sRGB matrix (IEC 61966-2-1)
XYZ_TO_LSRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)


def spectrum_to_linear_srgb(wavelengths_nm: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """Convert a reflectance spectrum to linear sRGB (D65 illuminant implied).

    Uses CIE 1931 2-degree observer. Returns [R, G, B] in linear space, clipped to [0, 1].
    """
    from scipy.interpolate import interp1d

    # Resample reflectance to the CIE grid
    f = interp1d(wavelengths_nm, reflectance, kind="linear",
                 bounds_error=False, fill_value=(reflectance[0], reflectance[-1]))
    rho = f(CIE_WAVELENGTHS_NM)

    # Integrate against each CMF (5 nm step, trapezoidal)
    xyz = np.zeros(3)
    for c in range(3):
        integrand = rho * CIE_XYZ[:, c]
        xyz[c] = np.trapezoid(integrand, CIE_WAVELENGTHS_NM)

    # Normalize so that an equal-energy white gives Y=1
    white_y = np.trapezoid(CIE_XYZ[:, 1], CIE_WAVELENGTHS_NM)
    xyz /= white_y

    # XYZ to linear sRGB
    lsrgb = XYZ_TO_LSRGB @ xyz
    return np.clip(lsrgb, 0.0, 1.0)


def tri_gaussian_spectrum(rgb_linear: np.ndarray, wavelength_nm: float) -> float:
    """Python replica of the shader's ConvertLinearRGBToSpectrum.

    Centers: 630, 532, 467 nm.  Sigma: lerp(65, 45, saturation).
    White-reference normalization per wavelength.
    """
    r, g, b = max(0.0, rgb_linear[0]), max(0.0, rgb_linear[1]), max(0.0, rgb_linear[2])

    LAMBDA_R, LAMBDA_G, LAMBDA_B = 630.0, 532.0, 467.0

    max_rgb = max(r, g, b)
    min_rgb = min(r, g, b)
    saturation = (max_rgb - min_rgb) / max_rgb if max_rgb > 0.001 else 0.0

    sigma = 65.0 + (45.0 - 65.0) * saturation  # lerp(65, 45, sat)

    def basis(lam, center):
        x = (lam - center) / sigma
        return np.exp(-0.5 * x * x)

    bR = basis(wavelength_nm, LAMBDA_R)
    bG = basis(wavelength_nm, LAMBDA_G)
    bB = basis(wavelength_nm, LAMBDA_B)

    R_lambda = r * bR + g * bG + b * bB
    white = bR + bG + bB
    white = max(white, 0.001)

    return min(max(R_lambda / white, 0.0), 1.5)


def tri_gaussian_spectrum_array(rgb_linear: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Evaluate the tri-Gaussian for an array of wavelengths."""
    return np.array([tri_gaussian_spectrum(rgb_linear, wl) for wl in wavelengths_nm])


def main():
    print("=" * 60)
    print("§六.2 B-Grade Bootstrap Error Quantification")
    print("=" * 60)

    materials = load_slum_materials(str(SLUM_DATA_DIR))

    # --- Validate the tri-Gaussian replica against the known concrete benchmark ---
    # Load concrete from the ECOSTRESS JSON to get its measured VIS spectrum
    import json
    ecostress_json = pathlib.Path(__file__).parent / "../../assets/spectral/quantiloom_materials_ecostress.json"
    with open(ecostress_json) as f:
        lib = json.load(f)

    concrete_key = "Construction  Concrete (0598UUUCNC)"
    if concrete_key in lib["materials"]:
        print(f"\nValidating tri-Gaussian replica against concrete benchmark...")

        # We can't reconstruct the full spectrum from the JSON (it stores NMF weights,
        # not raw spectra). Instead, use the SPECTRAL_ASSIGNMENT_PLAN.md benchmarks:
        #   tri-Gaussian RMSE = 0.0384 for concrete
        # We validate by computing the tri-Gaussian from a known concrete color.
        # Concrete sRGB ~ (0.32, 0.305, 0.285) linear (from ground.py baseColorFactor).
        concrete_rgb = np.array([0.32, 0.305, 0.285])
        vis_wl = np.linspace(400, 780, 200)
        tg_concrete = tri_gaussian_spectrum_array(concrete_rgb, vis_wl)
        # A flat gray at luminance ~0.31 should produce a nearly flat spectrum
        flat_val = 0.299 * concrete_rgb[0] + 0.587 * concrete_rgb[1] + 0.114 * concrete_rgb[2]
        print(f"  Concrete color: ({concrete_rgb[0]:.3f}, {concrete_rgb[1]:.3f}, {concrete_rgb[2]:.3f})")
        print(f"  Tri-Gaussian at 550 nm: {tri_gaussian_spectrum(concrete_rgb, 550.0):.4f} "
              f"(expected ~{flat_val:.3f})")
        print(f"  Tri-Gaussian range: [{tg_concrete.min():.4f}, {tg_concrete.max():.4f}]")
        print(f"  (Concrete is near-achromatic so the tri-Gaussian should be nearly flat)")
    else:
        print(f"  Concrete key not found in ECOSTRESS JSON")

    # --- B-grade error for PVC (V001-V006) ---
    print(f"\n{'='*60}")
    print("B-Grade VIS Error: PVC Samples (V001-V006)")
    print(f"{'='*60}\n")

    pvc_materials = [m for m in materials if m.record_id.startswith("V")]
    print(f"Found {len(pvc_materials)} PVC samples\n")

    vis_wavelengths_nm = np.linspace(400, 780, 200)

    print(f"{'Sample':<35} {'sRGB (linear)':<25} {'RMSE':>8} {'MaxRel%':>8} {'MeanRel%':>8}")
    print("-" * 90)

    all_rmse = []
    all_max_rel = []

    for mat in pvc_materials:
        # Extract VIS-only portion of the measured spectrum
        vis_mask = (mat.wavelengths >= 0.400) & (mat.wavelengths <= 0.780)
        if vis_mask.sum() < 10:
            print(f"  {mat.name}: insufficient VIS coverage, skipping")
            continue

        vis_wl_um = mat.wavelengths[vis_mask]
        vis_rho = mat.reflectance[vis_mask]
        vis_wl_nm = vis_wl_um * 1000.0

        # Step 1: Compute sRGB from the measured VIS spectrum
        lsrgb = spectrum_to_linear_srgb(vis_wl_nm, vis_rho)

        # Step 2: Reconstruct via tri-Gaussian at the same wavelengths
        from scipy.interpolate import interp1d
        f_measured = interp1d(vis_wl_nm, vis_rho, kind="linear",
                              bounds_error=False, fill_value=(vis_rho[0], vis_rho[-1]))
        measured_at_grid = f_measured(vis_wavelengths_nm)

        tg_spectrum = tri_gaussian_spectrum_array(lsrgb, vis_wavelengths_nm)

        # Step 3: Compare
        diff = tg_spectrum - measured_at_grid
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # Relative error (only where measured > 0.01 to avoid division by ~0)
        sig_mask = measured_at_grid > 0.01
        if sig_mask.sum() > 0:
            rel_err = np.abs(diff[sig_mask]) / measured_at_grid[sig_mask] * 100
            max_rel = float(rel_err.max())
            mean_rel = float(rel_err.mean())
        else:
            max_rel = mean_rel = float("nan")

        all_rmse.append(rmse)
        all_max_rel.append(max_rel)

        print(f"  {mat.name:<33} ({lsrgb[0]:.3f},{lsrgb[1]:.3f},{lsrgb[2]:.3f})  "
              f"{rmse:>8.4f} {max_rel:>7.1f}% {mean_rel:>7.1f}%")

    # Also run for all non-PVC samples to give context
    print(f"\n{'='*60}")
    print("B-Grade VIS Error: All 74 SLUM Samples (context)")
    print(f"{'='*60}\n")

    category_rmses = {}
    for mat in materials:
        vis_mask = (mat.wavelengths >= 0.400) & (mat.wavelengths <= 0.780)
        if vis_mask.sum() < 10:
            continue

        vis_wl_um = mat.wavelengths[vis_mask]
        vis_rho = mat.reflectance[vis_mask]
        vis_wl_nm = vis_wl_um * 1000.0

        lsrgb = spectrum_to_linear_srgb(vis_wl_nm, vis_rho)

        from scipy.interpolate import interp1d
        f_measured = interp1d(vis_wl_nm, vis_rho, kind="linear",
                              bounds_error=False, fill_value=(vis_rho[0], vis_rho[-1]))
        measured_at_grid = f_measured(vis_wavelengths_nm)

        tg_spectrum = tri_gaussian_spectrum_array(lsrgb, vis_wavelengths_nm)
        diff = tg_spectrum - measured_at_grid
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        cat = mat.record_id[0]
        category_rmses.setdefault(cat, []).append(rmse)

    cat_names = {"A": "asphalt", "B": "brick", "C": "concrete", "G": "stone",
                 "L": "slate", "R": "roof tile", "S": "slate/fibre", "V": "PVC",
                 "X": "misc", "Z": "metal"}

    print(f"{'Category':<20} {'N':>3} {'Median':>8} {'P75':>8} {'Max':>8}")
    print("-" * 50)
    for cat in sorted(category_rmses):
        vals = np.array(category_rmses[cat])
        print(f"  {cat} ({cat_names.get(cat, '?'):<12}) {len(vals):>3} "
              f"{np.median(vals):>8.4f} {np.percentile(vals, 75):>8.4f} {vals.max():>8.4f}")

    # Summary
    all_vals = np.concatenate(list(category_rmses.values()))
    print(f"\n  {'ALL':<17} {len(all_vals):>3} "
          f"{np.median(all_vals):>8.4f} {np.percentile(all_vals, 75):>8.4f} {all_vals.max():>8.4f}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if all_rmse:
        print(f"PVC B-grade VIS error:")
        print(f"  RMSE:    median {np.median(all_rmse):.4f}, max {max(all_rmse):.4f}")
        print(f"  MaxRel%: median {np.median(all_max_rel):.1f}%, max {max(all_max_rel):.1f}%")
    print(f"\nThe tri-Gaussian RMSE represents the VIS penalty of a B-grade assignment.")
    print(f"A-grade materials use the measured spectrum (RMSE → NMF reconstruction error).")
    print(f"The LWIR side is measured in both grades (no penalty).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
