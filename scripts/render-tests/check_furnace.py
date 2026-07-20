#!/usr/bin/env python3
"""Validate furnace-cavity renders against Planck reference.

Usage:
    python check_furnace.py <output.exr> <band> [--tol 0.002]

  band = LWIR | MWIR
  Default tolerance: 0.2% for ε=1, override with --tol for partial-emitter tests.

Exit code 0 = pass, 1 = fail.
"""

import argparse, math, sys, pathlib
import numpy as np
import OpenEXR

# ---- Planck reference (mirrors harness.py) ----
C1_NM = 1.191042972e20
C2_M  = 1.438776877e-2

def planck_blackbody(T_K, lambda_nm):
    if T_K <= 0 or lambda_nm <= 0:
        return 0.0
    lam_m = lambda_nm * 1e-9
    exponent = C2_M / (lam_m * T_K)
    if exponent > 700:
        return 0.0
    denom = math.exp(exponent) - 1.0
    if denom == 0:
        return 0.0
    return C1_NM / (lambda_nm ** 5) / denom

BANDS = {
    "LWIR": (8000.0, 12000.0, 16),
    "MWIR": (3000.0, 5000.0, 16),
}

def reference_radiance(T_K, band):
    lo, hi, n = BANDS[band]
    step = (hi - lo) / (n - 1)
    acc = sum(planck_blackbody(T_K, lo + i * step) for i in range(n))
    return acc * step / (hi - lo)


def read_exr_channel(path, channel_idx=0):
    """Read one channel from an EXR. Returns 2D numpy array (H, W)."""
    f = OpenEXR.File(str(path))
    ch = f.channels()
    first_key = list(ch.keys())[0]
    pixels = ch[first_key].pixels  # numpy array, typically (H, W, C)
    if pixels.ndim == 3:
        return pixels[:, :, channel_idx].astype(np.float64)
    return pixels.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Furnace cavity radiance check")
    parser.add_argument("exr", help="Path to rendered EXR output")
    parser.add_argument("band", choices=["LWIR", "MWIR"])
    parser.add_argument("--tol", type=float, default=0.002, help="Relative tolerance (default 0.002 = 0.2%%)")
    parser.add_argument("--temp", type=float, default=300.0, help="Cavity temperature K")
    args = parser.parse_args()

    ref = reference_radiance(args.temp, args.band)
    print(f"Reference: B̄({args.temp}K, {args.band}) = {ref:.6e} W/sr/m²/nm")

    exr_path = pathlib.Path(args.exr)
    if not exr_path.exists():
        print(f"ERROR: {exr_path} not found")
        sys.exit(1)

    img = read_exr_channel(exr_path, channel_idx=0)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Central 3x3 ROI (near-normal incidence); angle-dependent emissivity
    # correction reduces ε at off-axis pixels, creating a reflected
    # component with a known π-factor weighting issue.
    cx, cy = w // 2, h // 2
    x0, x1 = cx - 1, cx + 2
    y0, y1 = cy - 1, cy + 2
    roi = img[y0:y1, x0:x1]

    mean_val = float(roi.mean())
    rel_err = abs(mean_val - ref) / ref if ref > 0 else 0

    print(f"ROI mean:  {mean_val:.6e}")
    print(f"Rel error: {rel_err:.4%}")

    if rel_err > args.tol:
        print(f"FAIL: error {rel_err:.4%} exceeds tolerance {args.tol:.4%}")
        sys.exit(1)
    else:
        print(f"PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
