#!/usr/bin/env python3
"""Check hero wavelength sampling in VIS_FUSED against the deterministic grid.

Hero sampling replaces "integrate 32 fixed wavelengths along one path" with
"pick one wavelength, follow it, weight by 1/pdf" for rays past a dispersive
refraction. Its estimator is

    XYZ = L(λ_h) · cmf(λ_h) / pdf,     λ_h ~ U(400, 780),  pdf = 1/380 nm

whose expectation is ∫L(λ)cmf(λ)dλ -- the integral the 32-point grid
approximates. So the two must agree, and that gives a real test even though
Monte Carlo means exactness is unavailable:

A. CONVERGENCE. With a CONSTANT n(λ), refraction is achromatic and hero
   sampling must converge to the non-dispersing deterministic render. The error
   must fall roughly as 1/sqrt(spp). A bias would show as an error that stops
   falling -- which is exactly what a wrong 1/pdf weight, a missing cmf, or a
   sky that returns the whole band instead of one wavelength would look like.

B. DISPERSION. A real BK7 prism must come out chromatically different from the
   same prism with dispersion switched off.

The predecessor of this design -- splitting the band into 8 groups with a ray
each -- failed its own version of check A and was reverted. See
docs/participating-media-and-dispersion.md.

    python3 scripts/render-tests/check_hero_wavelength.py
"""
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
GLTF = ROOT / "assets/models/prism_dispersion.gltf"
BASE_CFG = ROOT / "assets/configs/prism_dispersion.toml"
CLI = ROOT / "build/src/app/Release/Quantiloom.exe"
WORK = ROOT / "renders/hero"
TMP_CFG = ROOT / "assets/configs/_hero_tmp.toml"
FLAT = "assets/data/refractiveindex/BK7_flat.yml"
SELLMEIER = "assets/data/refractiveindex/BK7_sellmeier.yml"

BK7_ABBE = 64.17
MATERIAL = "PrismGlass_BK7"
RES = 256           # small: this runs many spp
SPPS = [64, 256, 1024]


def run(name, dispersion, cri, spp, seed=12345):
    doc = json.loads(GLTF.read_text())
    for mat in doc["materials"]:
        ext = mat.setdefault("extensions", {})
        if "KHR_materials_transmission" not in ext:
            continue
        if dispersion is None:
            ext.pop("KHR_materials_dispersion", None)
        else:
            ext["KHR_materials_dispersion"] = {"dispersion": dispersion}
    GLTF.write_text(json.dumps(doc, indent=1))

    cfg = BASE_CFG.read_text()
    cfg = cfg.replace('mode = "rgb"', 'mode = "vis_fused"')
    cfg = cfg.replace("resolution = [1024, 1024]", f"resolution = [{RES}, {RES}]")
    cfg = cfg.replace("spp = 64", f"spp = {spp}")
    cfg = cfg.replace("seed = 12345", f"seed = {seed}")
    cfg = cfg.replace('output = "prism_dispersion.exr"', f'output = "_hero_{name}.exr"')
    if cri:
        cfg += f'\n[refractive_index]\n"{MATERIAL}" = "{cri}"\n'
    TMP_CFG.write_text(cfg)

    proc = subprocess.run([str(CLI), str(TMP_CFG.relative_to(ROOT))],
                          cwd=ROOT, capture_output=True, text=True)
    if "Saved spectral image" not in proc.stdout + proc.stderr:
        sys.exit(f"{name}: render produced no output\n{proc.stdout[-2000:]}")

    produced = ROOT / f"_hero_{name}.exr"
    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"{name}.exr"
    shutil.copy2(produced, dest)
    produced.unlink(missing_ok=True)
    (ROOT / f"_hero_{name}.png").unlink(missing_ok=True)

    with OpenEXR.File(str(dest)) as f:
        ch = f.channels()
        return ch[next(iter(ch))].pixels.astype(np.float64)[..., :3]


def chroma_spread(a):
    m = a.reshape(-1, 3)
    lit = m[m.max(axis=1) > 1e-6]
    return float(np.mean(lit.max(axis=1) - lit.min(axis=1))) if len(lit) else 0.0


backup = GLTF.read_text()
failures = []
try:
    print("reference: no dispersion, deterministic 32-point grid")
    ref = run("ref", None, None, 64)
    scale = max(float(np.abs(ref).mean()), 1e-12)

    print()
    print("A. convergence to the deterministic grid, with a constant n(lambda)")
    print(f"   {'spp':>6}  {'relative error':>15}  {'x 1/sqrt(spp)':>15}")
    errs = []
    for spp in SPPS:
        img = run(f"flat_{spp}", None, FLAT, spp)
        err = float(np.abs(img - ref).mean()) / scale
        errs.append(err)
        print(f"   {spp:6d}  {err:15.4%}  {err * (spp ** 0.5):15.4f}")

    # The error must fall. Monte Carlo halves it per 4x samples; allow slack but
    # require real improvement, because a bias would simply plateau.
    if errs[-1] >= errs[0]:
        failures.append("error did not fall with more samples -- the estimator is biased, "
                        "not merely noisy")
    elif errs[-1] > 0.02:
        failures.append(f"error still {errs[-1]:.2%} at {SPPS[-1]} spp -- too large "
                        f"to be sampling noise alone")

    print()
    print("B. a real BK7 prism disperses")
    plain = run("plain_hi", None, FLAT, SPPS[-1])
    bk7 = run("bk7_hi", 20.0 / BK7_ABBE, None, SPPS[-1])
    rel = float(np.abs(bk7 - plain).mean()) / scale
    print(f"   relative mean |bk7 - constant n| = {rel:.3e}")
    print(f"   chromatic spread  constant n     = {chroma_spread(plain):.6g}")
    print(f"   chromatic spread  bk7            = {chroma_spread(bk7):.6g}")
    if rel <= 1e-4:
        failures.append("BK7 is indistinguishable from a non-dispersing medium")
finally:
    GLTF.write_text(backup)
    TMP_CFG.unlink(missing_ok=True)

print()
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("hero wavelength checks PASS")
