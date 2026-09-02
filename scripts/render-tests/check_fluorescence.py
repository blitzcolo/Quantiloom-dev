#!/usr/bin/env python3
"""Check that light absorbed at one wavelength comes back at another.

Every other term the renderer computes is diagonal in wavelength: what arrives
at lambda leaves at lambda, which is what lets four wavelengths share one
geometric path. Fluorescence is the term that is not, and being off the
diagonal is what makes it checkable in a way nothing else here is -- a renderer
that quietly dropped the coupling would still produce a plausible frame, and
three of the four checks below would still pass. The second one is the one that
cannot be faked.

The scene is the open plate the sky-equivalence check uses: one material, no
occlusion, a residual that is zero by construction and an illuminant that is
flat, so anything that changes in the frame changes because of the material.

A. LINEARITY. The re-emitted radiance is yield * em(lambda) * M, with M the
   excitation integral, so doubling the yield doubles the contribution and
   nothing else. Same seed, so the comparison is per pixel and exact.

B. OFF-DIAGONAL TRANSFER. Two illuminants of equal power over the band, one
   with twice as much of it below 500 nm. The dye absorbs on [400, 500] and
   emits on [550, 650], two bands that do not overlap, so under the second
   illuminant its contribution RISES while the illuminant's own power in the
   emission band FALLS. Both are measured. A renderer whose transport is
   diagonal has no way to produce the first.

C. THE TWO ESTIMATORS AGREE. vis_fused sweeps 32 fixed wavelengths and
   vis_hero draws four; the fluorescent term is evaluated inside the loop they
   share, so their answers must match. What this catches is the term being
   added somewhere only one of them reaches.

D. ENERGY. A surface that absorbs everything and gives it all back, spread
   evenly, must return exactly the irradiance it received:

       yield * em(lambda) * INTEGRAL ex E dlambda / pi  =  E / pi

   which for this scene is (E_sun cos(theta) + E_sky) / pi, a closed form. This
   is the check that found the emission normalisation reading the band one grid
   step too wide, and returning 1.6% too little light everywhere.

    python3 scripts/render-tests/check_fluorescence.py

Exit code 0 = pass, 1 = fail, 2 = no CLI, 3 = no usable GPU.
"""
import math
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = ROOT / "build/src/app/Release/Quantiloom.exe"
BASE_CFG = ROOT / "assets/configs/fluorescence_plate.toml"
TMP_CFG = ROOT / "assets/configs/_fluorescence_tmp.toml"

GPU_ABSENT = re.compile(
    r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")

FLAT_LUT = "assets/luts/flat_sun_sky.csv"
BLUE_LUT = "assets/luts/blue_shifted_sun_sky.csv"
DYE_EX = "assets/data/fluorescence/demo_dye_excitation.csv"
DYE_EM = "assets/data/fluorescence/demo_dye_emission.csv"
ALL_EX = "assets/data/fluorescence/total_absorber_excitation.csv"
FLAT_EM = "assets/data/fluorescence/flat_emission.csv"

# From the config and assets/luts/flat_sun_sky.csv, the same constants
# check_sky_equiv.py states: a flat 1.0 W/m^2/nm direct beam at cos = 0.8, and
# a sky that is the global 1.5 minus the direct.
E_SUN, E_SKY, COS_THETA = 1.0, 0.5, 0.8
LUMA = np.array([0.2126, 0.7152, 0.0722])


def illuminant_at(lut, lambda_nm):
    """The direct beam the LUT carries at one wavelength, read from the file.

    A property of the fixture rather than of the renderer, so it is asked of the
    fixture: the claim check B rests on is that the second illuminant is DIMMER
    in the emission band, and if that ever stops being true the check would pass
    for the wrong reason.
    """
    rows = []
    for line in (ROOT / lut).read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[0]), float(parts[3])))
        except ValueError:
            continue  # the header
    rows.sort()
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    return float(np.interp(lambda_nm, xs, ys))


def render(yield_, lut=FLAT_LUT, mode="vis_hero", excitation=DYE_EX, emission=DYE_EM):
    """Render the plate and return the frame as float64 RGB."""
    cfg = BASE_CFG.read_text()
    cfg = cfg.replace("fluorescence_yield = 0.6", f"fluorescence_yield = {yield_}")
    cfg = cfg.replace(f'solar_lut = "{FLAT_LUT}"', f'solar_lut = "{lut}"')
    cfg = cfg.replace(f'fluorescence_excitation_curve = "{DYE_EX}"',
                      f'fluorescence_excitation_curve = "{excitation}"')
    cfg = cfg.replace(f'fluorescence_emission_curve = "{DYE_EM}"',
                      f'fluorescence_emission_curve = "{emission}"')
    cfg = cfg.replace('mode = "vis_hero"', f'mode = "{mode}"')
    cfg = cfg.replace('output = "fluorescence_plate_output.exr"', 'output = "_fluor_tmp.exr"')
    TMP_CFG.write_text(cfg)

    proc = subprocess.run(
        [str(CLI), str(TMP_CFG.relative_to(ROOT))],
        # encoding explicitly, not text=True: the renderer's log has bytes the
        # locale codec rejects, and the reader thread then dies with stdout None.
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if "Saved spectral image" not in log:
        if GPU_ABSENT.search(log):
            print("no usable GPU, nothing measured", file=sys.stderr)
            sys.exit(3)
        print("ERROR: render failed", file=sys.stderr)
        print("\n".join(log.splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)

    out = ROOT / "_fluor_tmp.exr"
    with OpenEXR.File(str(out)) as f:
        ch = f.channels()
        pixels = ch[next(iter(ch))].pixels.astype(np.float64)
    out.unlink(missing_ok=True)
    (ROOT / "_fluor_tmp.png").unlink(missing_ok=True)
    return pixels[..., :3] if pixels.ndim == 3 else pixels


if not CLI.exists():
    print(f"no CLI at {CLI} -- build first", file=sys.stderr)
    sys.exit(2)

failures = []
started = time.time()
try:
    # ------------------------------------------------------------------ A ---
    print("A. the contribution is linear in the yield")
    dark = render(0.0)
    half = render(0.3)
    full = render(0.6)
    d_half = half - dark
    d_full = full - dark
    # Same seed and the same scene, so the excitation integral is drawn at the
    # same wavelength in both and the two differ by the factor alone -- not
    # approximately, per pixel.
    scale = max(float(np.abs(d_full).mean()), 1e-12)
    residual = float(np.abs(d_full - 2.0 * d_half).max()) / scale
    print(f"   mean contribution at yield 0.3 = {float(d_half.mean()):.6f}")
    print(f"   mean contribution at yield 0.6 = {float(d_full.mean()):.6f}")
    print(f"   worst pixel |I(0.6) - I(0) - 2 (I(0.3) - I(0))| = {residual:.3e} of the mean")
    if float(d_full.mean()) <= 0.0:
        failures.append("A: binding a fluorescent pair changed nothing")
    if residual > 1e-3:
        failures.append(f"A: doubling the yield did not double the contribution "
                        f"(worst pixel off by {residual:.3e} of the mean)")

    # ------------------------------------------------------------------ B ---
    print()
    print("B. the transfer is off the diagonal")
    blue_dark = render(0.0, lut=BLUE_LUT)
    blue_full = render(0.6, lut=BLUE_LUT)
    d_blue = blue_full - blue_dark
    ratio = float(d_blue.mean()) / max(float(d_full.mean()), 1e-12)

    # What the illuminant itself does where the dye emits. A renderer whose
    # transport is diagonal can only move the emission band by this factor,
    # because that band is all it would be reading.
    illum_600 = illuminant_at(BLUE_LUT, 600.0) / illuminant_at(FLAT_LUT, 600.0)
    illum_450 = illuminant_at(BLUE_LUT, 450.0) / illuminant_at(FLAT_LUT, 450.0)

    print(f"   two illuminants of equal power over the band, the second with")
    print(f"   twice as much of it below 500 nm")
    print(f"   the illuminant at 450 nm, where the dye absorbs  x {illum_450:.4f}")
    print(f"   the illuminant at 600 nm, where the dye emits    x {illum_600:.4f}")
    print(f"   the dye's contribution                           x {ratio:.4f}")
    if illum_600 >= 1.0 or illum_450 <= 1.0:
        failures.append(f"B: the fixture is wrong, not the renderer: the second "
                        f"illuminant reads x{illum_450:.4f} at 450 nm and "
                        f"x{illum_600:.4f} at 600 nm, and the check needs the first "
                        f"above 1 and the second below it")
    if ratio <= 1.0:
        failures.append(f"B: the dye's contribution moved x{ratio:.4f} with the "
                        f"illuminant, inside the x{illum_600:.4f} a diagonal transport "
                        f"can produce. Light is not crossing from the band the dye "
                        f"absorbs in to the band it emits in")
    # The excitation is entirely below 500 nm, where the second illuminant is
    # twice the first, so the contribution should scale with it. Held loosely --
    # the illuminant is resampled onto a uniform grid before it is uploaded,
    # which softens the step -- but far enough above 1 to be a different
    # phenomenon rather than a shifted one.
    if ratio < 1.5:
        failures.append(f"B: the dye's contribution moved only x{ratio:.4f} where the "
                        f"illuminant it absorbs from moved x{illum_450:.4f}")

    # ------------------------------------------------------------------ C ---
    print()
    print("C. the two visible estimators agree")
    fused_dark = render(0.0, mode="vis_fused")
    fused_full = render(0.6, mode="vis_fused")
    d_fused = fused_full - fused_dark
    disagreement = abs(float(d_fused.mean()) - float(d_full.mean())) / \
        max(abs(float(d_fused.mean())), 1e-12)
    print(f"   vis_hero  contribution {float(d_full.mean()):.6f}")
    print(f"   vis_fused contribution {float(d_fused.mean()):.6f}")
    print(f"   disagreement {disagreement:.4%}")
    if disagreement > 0.01:
        failures.append(f"C: the two estimators disagree by {disagreement:.4%} about "
                        f"the fluorescent contribution")

    # ------------------------------------------------------------------ D ---
    print()
    print("D. what is absorbed is what comes back")
    reference = (E_SUN * COS_THETA + E_SKY) / math.pi
    print(f"   closed form (E_sun cos + E_sky) / pi = {reference:.6f}")
    for mode in ("vis_hero", "vis_fused"):
        off = render(0.0, mode=mode, excitation=ALL_EX, emission=FLAT_EM)
        on = render(1.0, mode=mode, excitation=ALL_EX, emission=FLAT_EM)
        added = float(((on - off).reshape(-1, 3) @ LUMA).mean())
        error = abs(added - reference) / reference
        print(f"   {mode:>10}  added luminance {added:.6f}  error {error:.4%}")
        if error > 0.01:
            failures.append(f"D: in {mode}, a surface that absorbs everything and "
                            f"re-emits it flat returned {added:.6f} where it received "
                            f"{reference:.6f} ({error:.4%} out)")
finally:
    TMP_CFG.unlink(missing_ok=True)

print()
print(f"({time.time() - started:.0f} s)")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("fluorescence checks PASS")
