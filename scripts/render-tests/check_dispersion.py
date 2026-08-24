#!/usr/bin/env python3
"""Check refraction's wavelength-dependent index on a BK7 prism.

Two questions, in RGB mode -- the only mode that disperses today. VIS_FUSED
integrates the whole band along one geometric path, so it cannot; see
docs/participating-media-and-dispersion.md.

1. DOES IT DISPERSE. Switching the Abbe number on must change the render and
   raise its chromatic spread.

2. DO THE TWO n(lambda) SOURCES AGREE. The same glass through the analytic
   KHR_materials_dispersion formula and through a measured Sellmeier table
   should land in the same place. They are independent implementations of one
   physical claim, so agreement is evidence for both and a disagreement says
   which to look at. This is the check that caught the formula being wrong:
   before it used the Khronos form, the analytic path was missing the constant
   term and sat 0.0122 high in n for BK7 -- three times that glass's entire
   F-to-C spread.

    python3 scripts/render-tests/check_dispersion.py
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
WORK = ROOT / "renders/dispersion"
TMP_CFG = ROOT / "assets/configs/_dispersion_tmp.toml"
SELLMEIER = "assets/data/refractiveindex/BK7_sellmeier.yml"

BK7_ABBE = 64.17
MATERIAL = "PrismGlass_BK7"

# (name, KHR dispersion = 20/V_d or None to remove, refractive-index table)
CASES = [
    ("off",      None,            None),
    ("analytic", 20.0 / BK7_ABBE, None),
    ("measured", None,            SELLMEIER),
]


def run(name, dispersion, cri):
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

    cfg = BASE_CFG.read_text().replace('output = "prism_dispersion.exr"',
                                       f'output = "_dispersion_{name}.exr"')
    if cri:
        cfg += f'\n[refractive_index]\n"{MATERIAL}" = "{cri}"\n'
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
        sys.exit(f"{name}: render produced no output\n{proc.stdout[-2000:]}")

    produced = ROOT / f"_dispersion_{name}.exr"
    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"prism_{name}.exr"
    shutil.copy2(produced, dest)
    produced.unlink(missing_ok=True)
    (ROOT / f"_dispersion_{name}.png").unlink(missing_ok=True)

    with OpenEXR.File(str(dest)) as f:
        ch = f.channels()
        return ch[next(iter(ch))].pixels.astype(np.float64)[..., :3]


def chroma_spread(a):
    """Mean |max channel - min channel| over lit pixels: how coloured it is."""
    m = a.reshape(-1, 3)
    lit = m[m.max(axis=1) > 1e-6]
    return float(np.mean(lit.max(axis=1) - lit.min(axis=1))) if len(lit) else 0.0


backup = GLTF.read_text()
try:
    images = {name: run(name, d, c) for name, d, c in CASES}
finally:
    GLTF.write_text(backup)
    TMP_CFG.unlink(missing_ok=True)

off, analytic, measured = (images[k] for k in ("off", "analytic", "measured"))
scale = max(float(np.abs(off).mean()), 1e-12)
failures = []

relB = float(np.abs(analytic - off).mean()) / scale
print("1. BK7 disperses")
print(f"   relative mean |analytic - off| = {relB:.3e}")
print(f"   chromatic spread  off          = {chroma_spread(off):.6g}")
print(f"   chromatic spread  analytic     = {chroma_spread(analytic):.6g}")
if relB <= 1e-6:
    failures.append("switching dispersion on changed nothing")
if chroma_spread(analytic) <= chroma_spread(off):
    failures.append("the dispersing render is no more chromatic than the plain one")

relC = float(np.abs(measured - analytic).mean()) / scale
print()
print("2. measured Sellmeier table agrees with the analytic Abbe formula")
print(f"   relative mean |measured - analytic| = {relC:.3e}")
print(f"   chromatic spread  measured         = {chroma_spread(measured):.6g}")
# Loose on purpose: two different n(lambda) curves agreeing to ~1e-5 in n, then
# put through refraction and Monte Carlo. The claim is "same physics", not
# "same bits". Anything above a percent means one path is wrong.
if relC > 0.01:
    failures.append(f"the two n(lambda) sources disagree by {relC:.3e} relative")

print()
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("dispersion checks PASS")
