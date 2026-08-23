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
    """Band mean of a blackbody, by the same rule the shader integrates with.

    Trapezoid: n samples span n-1 intervals, so the endpoints carry half
    weight. This used to sum n full-width rectangles and divide by the range,
    which double-counts one interval and returns n/(n-1) of the true mean --
    the same off-by-one the shaders had. With the shaders corrected and this
    left alone, every furnace cavity read 5.9% low in LWIR and 8.7% in MWIR,
    identically across emissivities, which is what a reference-side error
    looks like: a constant offset that does not care what it is measuring.
    """
    lo, hi, n = BANDS[band]
    step = (hi - lo) / (n - 1)
    vals = [planck_blackbody(T_K, lo + i * step) for i in range(n)]
    acc = sum(vals) - 0.5 * (vals[0] + vals[-1])
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

    # The whole image. This used to sample a central 3x3 patch to stay at
    # near-normal incidence, because the reflected term carried "a known
    # π-factor weighting issue" that made off-axis pixels wrong -- the missing
    # 1/PI in the IR environment reflection, since fixed. Sampling everything
    # is the stronger test: an isothermal cavity must return B(T) from every
    # direction, so the whole frame is signal, and an orientation-dependent
    # error has nowhere to hide. Restricting the ROI is what let that bug sit.
    roi = img

    mean_val = float(roi.mean())
    rel_err = abs(mean_val - ref) / ref if ref > 0 else 0

    print(f"ROI mean:  {mean_val:.6e}")
    print(f"Rel error: {rel_err:.4%}")
    # The ratio at full width. "Rel error" rounds to 0.0000% for five of the
    # eight cavities, which is a statement about the format rather than about
    # the renderer -- a table that reports it cannot distinguish a cavity that
    # closes to one part in 10^7 from one that closes to one part in 10^5.
    # Purely an addition to what is printed; nothing above this line changed.
    print(f"Ratio:     {mean_val / ref:.8f}" if ref > 0 else "Ratio:     n/a")

    if rel_err > args.tol:
        print(f"FAIL: error {rel_err:.4%} exceeds tolerance {args.tol:.4%}")
        sys.exit(1)
    else:
        print(f"PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
