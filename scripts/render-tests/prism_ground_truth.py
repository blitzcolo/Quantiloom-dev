#!/usr/bin/env python3
"""Build a deterministic ground-truth dispersion image by sweeping SINGLE mode.

SINGLE mode renders one wavelength per image and refracts at that wavelength on
BOTH prism faces (closesthit.rchit uses camera.wavelength_nm for refraction in
every non-RGB, non-VIS_FUSED mode), so the two deviations compound the way they
do in a real prism. Sweeping the same 32 wavelengths VIS_FUSED uses and
combining them with the CIE observer therefore gives what VIS_FUSED *should*
produce -- no Monte Carlo in the wavelength dimension, no shared-path
approximation.

It answers "what does air -> prism -> air actually look like" without relying on
any unverified code, and the refraction it does is correct per wavelength on both
faces.

Reproduces the shader's own combination (SpectralConversion.hlsli):
  XYZ = sum_i L(lambda_i) * cmf(lambda_i) * dlambda,  then / CIE_Y_INTEGRAL
  RGB = XYZ_to_linear_sRGB(XYZ),  then R *= chroma_r, B *= chroma_b

The two chroma factors are read from the scene config rather than baked in here.
They are a legacy escape hatch and default to 1.0 on both channels; a constant
pair hardcoded in this file silently white-balances the reference against a
renderer that no longer applies one, which is a bias in the ground truth itself
and the hardest kind to notice.

WHAT TO CHECK BEFORE TRUSTING A NUMBER FROM IT. SINGLE and VIS_FUSED are
separate shading branches. `--flat` is the control: with a constant n(lambda)
nothing disperses and every mode must agree, so any residual between this sweep
and the deterministic VIS_FUSED grid is branch drift rather than dispersion, and
bounds what a dispersing comparison can claim. Run it first; quote its residual
alongside any figure taken from the BK7 run.

    python3 scripts/render-tests/prism_ground_truth.py [--spp N] [--res N]
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tomllib

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
GLTF = ROOT / "assets/models/prism_dispersion.gltf"
BASE_CFG = ROOT / "assets/configs/prism_dispersion.toml"
CLI = ROOT / "build/src/app/Release/Quantiloom.exe"
CMF_CSV = ROOT / "assets/luts/CIE_xyz_1931_2deg.csv"
TMP_CFG = ROOT / "assets/configs/_sweep_tmp.toml"
OUT_DIR = ROOT / "renders/ground-truth"

# Must match closesthit.rchit's VIS_FUSED grid exactly, or this is a reference
# for a different integral.
LAMBDA_MIN, LAMBDA_MAX, N_SAMPLES = 400.0, 780.0, 32
CIE_Y_INTEGRAL = 106.857
BK7_ABBE = 64.17

# The renderer's own defaults, from LightingParams.hpp::LightingDefaults, with
# the scene's override honoured if it sets one -- TMP_CFG inherits every key
# from BASE_CFG, so whatever is read here is what the render applied.
_quality = tomllib.loads(BASE_CFG.read_text()).get("quality", {})
CHROMA_R = float(_quality.get("chroma_r_correction", 1.0))
CHROMA_B = float(_quality.get("chroma_b_correction", 1.0))

ap = argparse.ArgumentParser()
ap.add_argument("--spp", type=int, default=64)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--flat", action="store_true",
                help="use a constant n(lambda): the sweep must then agree with "
                     "the non-dispersing deterministic grid, which validates "
                     "the sweep itself")
args = ap.parse_args()

STEP = (LAMBDA_MAX - LAMBDA_MIN) / (N_SAMPLES - 1)
LAMBDAS = [LAMBDA_MIN + i * STEP for i in range(N_SAMPLES)]


def load_cmf():
    """CIE 1931 2-degree observer at 1 nm, interpolated to our wavelengths."""
    rows = np.loadtxt(CMF_CSV, delimiter=",")
    nm, xb, yb, zb = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    return np.stack([np.interp(LAMBDAS, nm, c) for c in (xb, yb, zb)], axis=1)


def set_material(dispersing):
    doc = json.loads(GLTF.read_text())
    for mat in doc["materials"]:
        # get, not setdefault: setdefault writes the key before the check below
        # decides this material is not the prism, so every opaque material in
        # the file grew an empty "extensions": {} on every run.
        ext = mat.get("extensions")
        if not ext or "KHR_materials_transmission" not in ext:
            continue
        if dispersing:
            ext["KHR_materials_dispersion"] = {"dispersion": 20.0 / BK7_ABBE}
        else:
            ext.pop("KHR_materials_dispersion", None)
    GLTF.write_text(json.dumps(doc, indent=1))


def render(lambda_nm, index):
    cfg = BASE_CFG.read_text()
    cfg = cfg.replace('mode = "rgb"', 'mode = "single"')
    cfg = cfg.replace("resolution = [1024, 1024]", f"resolution = [{args.res}, {args.res}]")
    cfg = cfg.replace("spp = 64", f"spp = {args.spp}")
    cfg = cfg.replace('output = "prism_dispersion.exr"', f'output = "_sweep_{index:02d}.exr"')
    cfg = cfg.replace('mode = "single"',
                      f'mode = "single"\nwavelength_nm = {lambda_nm:.4f}')
    if args.flat:
        cfg += ('\n[refractive_index]\n"PrismGlass_BK7" = '
                '"assets/data/refractiveindex/BK7_flat.yml"\n')
    TMP_CFG.write_text(cfg)

    proc = subprocess.run([str(CLI), str(TMP_CFG.relative_to(ROOT))],
                          # encoding explicitly, not text=True: that decodes with the
                          # locale codec, and the renderer's log has bytes GBK rejects.
                          # The reader thread then dies and stdout returns None, so the
                          # failure arrives as a TypeError with nothing to suggest an
                          # encoding. errors=replace: this is only ever grepped for an
                          # ASCII marker.
                          cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    if "Saved spectral image" not in (proc.stdout or "") + (proc.stderr or ""):
        sys.exit(f"lambda={lambda_nm:.1f}: no render\n{proc.stdout[-1500:]}")

    produced = ROOT / f"_sweep_{index:02d}.exr"
    with OpenEXR.File(str(produced)) as f:
        ch = f.channels()
        px = ch[next(iter(ch))].pixels.astype(np.float64)
    produced.unlink(missing_ok=True)
    (ROOT / f"_sweep_{index:02d}.png").unlink(missing_ok=True)
    # SINGLE mode replicates its scalar spectral radiance across RGB.
    return px[..., 0]


cmf = load_cmf()
# Read the backup BEFORE mutating, or the finally below faithfully restores the
# mutation and this script edits a tracked asset every time it runs.
backup = GLTF.read_text()
set_material(dispersing=not args.flat)

try:
    XYZ = None
    for i, lam in enumerate(LAMBDAS):
        L = render(lam, i)
        if XYZ is None:
            XYZ = np.zeros((*L.shape, 3))
        XYZ += L[..., None] * cmf[i] * STEP
        print(f"  [{i + 1:2d}/{N_SAMPLES}] {lam:7.2f} nm  mean L = {L.mean():.6g}",
              flush=True)
finally:
    GLTF.write_text(backup)
    TMP_CFG.unlink(missing_ok=True)

XYZ /= CIE_Y_INTEGRAL
M = np.array([[3.2406, -1.5372, -0.4986],
              [-0.9689, 1.8758, 0.0415],
              [0.0557, -0.2040, 1.0570]])
rgb = XYZ @ M.T
rgb[..., 0] *= CHROMA_R
rgb[..., 2] *= CHROMA_B
rgb = np.clip(rgb, 0.0, 1000.0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
tag = "flat" if args.flat else "bk7"
dest = OUT_DIR / f"prism_groundtruth_{tag}_{args.res}px_{args.spp}spp.exr"
header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
rgba = np.dstack([rgb.astype(np.float32),
                  np.ones(rgb.shape[:2], dtype=np.float32)])
with OpenEXR.File(header, {"RGBA": rgba}) as f:
    f.write(str(dest))

print()
print(f"wrote {dest.relative_to(ROOT)}")
print(f"  mean RGB = {rgb[..., 0].mean():.6g} {rgb[..., 1].mean():.6g} {rgb[..., 2].mean():.6g}")
m = rgb.reshape(-1, 3)
lit = m[m.max(axis=1) > 1e-6]
if len(lit):
    print(f"  chromatic spread = {float(np.mean(lit.max(axis=1) - lit.min(axis=1))):.6g}")
