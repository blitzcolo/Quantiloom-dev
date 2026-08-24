#!/usr/bin/env python3
"""Check that a Lambertian surface returns the same radiance from every direction.

Usage:
    check_view_independence.py <nadir.exr> <oblique.exr> --mode {reflective,thermal}

A surface whose reflectance and emissivity are HEMISPHERICAL quantities -- which
is what a measured directional-hemispherical reflectance is, and what the CPU
thermal solver assumes -- radiates the same in every direction. Moving the
camera cannot change the answer. This checks exactly that, and nothing else in
the build gates can:

    skyequiv_swir.toml     camera [0, 20, 0], orthographic, plane normal +Y
    shadow_*_{sun,nosun}   the same, all six
    furnace_*              isothermal, so eps + rho = 1 holds at every angle

Every one of them views the surface down its own normal, so NdotV is 1 at every
pixel and any law of the form eps(theta) = eps0 * f(cos theta) with f(1) = 1 is
the identity. The renderer carried such a law -- eps0 * cos^0.7(theta) for
dielectrics -- and it was the identity in all eight furnace cavities and all
seven illumination scenes while turning rho = 0.700 into 0.815 at 60 degrees.
On a ground plane seen at grazing angles it reported a 300 K desert at 324 K in
MWIR, with the thermal ordering of the scene inverted: the hottest object in the
frame came out darker than the sand around it.

Two things make that law wrong rather than merely approximate. It is fabricated
-- the exponent is a tuned constant, not Fresnel -- and the reflectance derived
from it is a function of the VIEW angle that then multiplies light arriving from
every other direction (an isotropic sky, a sun at its own incidence angle). A
Lambertian lobe whose albedo depends on the outgoing direction is not
reciprocal, and its directional-hemispherical albedo is not the number that was
measured: for sand it integrates to 0.365 where the measurement says 0.143.

The two renders are the same scene at NdotV = 1.0 and NdotV = 0.5. Both must
equal the closed form, and each other. Orthographic on both sides, so NdotV is
constant across each frame and neither reference carries Monte Carlo variance.

Exit code 0 = pass, 1 = fail, 2 = a file is missing.
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import OpenEXR

# ---- Reflective reference -------------------------------------------------
# assets/models/shadow_scene/shadow_scene_open.gltf with
# assets/luts/flat_sun_sky.csv, the same constants check_sky_equiv.py asserts.
RHO = 0.7           # 1 - emissivity, emissivity 0.3, no transmittance
COS_THETA = 0.8     # sun_direction (0.6, 0.8, 0) against a +Y normal
E_SUN = 1.0
E_SKY = 0.5         # global 1.5 minus direct 1.0

# ---- Thermal reference ----------------------------------------------------
T_SURFACE_K = 300.0
T_SKY_K = 200.0     # [lighting] atmosphere_temperature_k
EPS = 0.3
LWIR = (8000.0, 12000.0, 16)

C1_NM = 1.191042972e20
C2_M = 1.438776877e-2


def planck_blackbody(T_K, lambda_nm):
    if T_K <= 0 or lambda_nm <= 0:
        return 0.0
    lam_m = lambda_nm * 1e-9
    exponent = C2_M / (lam_m * T_K)
    if exponent > 700:
        return 0.0
    denom = math.exp(exponent) - 1.0
    return C1_NM / (lambda_nm ** 5) / denom if denom else 0.0


def band_mean_planck(T_K, band=LWIR):
    """Band mean by the trapezoid the shader integrates with -- n samples span
    n-1 intervals, so the endpoints carry half weight. Same rule as
    check_furnace.py; a rectangle sum here reads n/(n-1) high."""
    lo, hi, n = band
    step = (hi - lo) / (n - 1)
    vals = [planck_blackbody(T_K, lo + i * step) for i in range(n)]
    acc = sum(vals) - 0.5 * (vals[0] + vals[-1])
    return acc * step / (hi - lo)


def reference(mode):
    if mode == "reflective":
        return RHO * (E_SUN / math.pi * COS_THETA + E_SKY / math.pi)
    # eps * B(T_surface) + rho * B(T_sky), both band means
    return EPS * band_mean_planck(T_SURFACE_K) + RHO * band_mean_planck(T_SKY_K)


def read_exr(path):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    px = ch[list(ch.keys())[0]].pixels
    return (px[:, :, 0] if px.ndim == 3 else px).astype(np.float64)


def main():
    p = argparse.ArgumentParser(description="View-independence check")
    p.add_argument("nadir")
    p.add_argument("oblique")
    p.add_argument("--mode", choices=("reflective", "thermal"), required=True)
    p.add_argument("--tol", type=float, default=0.005,
                   help="relative tolerance (default 0.5%%)")
    args = p.parse_args()

    for f in (args.nadir, args.oblique):
        if not pathlib.Path(f).exists():
            print(f"ERROR: {f} not found", file=sys.stderr)
            return 2

    ref = reference(args.mode)
    a, b = read_exr(args.nadir), read_exr(args.oblique)
    ma, mb = float(a.mean()), float(b.mean())
    ea = abs(ma - ref) / ref
    eb = abs(mb - ref) / ref
    between = abs(mb - ma) / ma if ma > 0 else float("inf")

    print(f"Reference:  {ref:.6e} W/sr/m^2/nm  ({args.mode})")
    print(f"NdotV 1.0:  {ma:.6e}  ({ea:.4%})   spread {a.max() - a.min():.3e}")
    print(f"NdotV 0.5:  {mb:.6e}  ({eb:.4%})   spread {b.max() - b.min():.3e}")
    print(f"Rel error:  {max(ea, eb):.4%}   between views: {between:.4%}")

    if max(ea, eb, between) > args.tol:
        print(f"FAIL: exceeds {args.tol:.4%} -- radiance depends on where the "
              f"camera stands, so the reflectance or emissivity being used is a "
              f"function of the view angle rather than the hemispherical "
              f"quantity that was measured")
        return 1
    print("view independence PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
