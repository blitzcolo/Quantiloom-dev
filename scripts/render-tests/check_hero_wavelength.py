#!/usr/bin/env python3
"""Check the hero-wavelength estimator against the deterministic 32-point grid.

Two visible modes live in one binary and answer the same question. vis_fused
sweeps 32 fixed wavelengths along one geometric path and sums them; vis_hero
draws one wavelength, derives three more by rotating it through the band, and
carries the quartet down a single path:

    XYZ = sum_j L(lambda_j) cmf(lambda_j) / S,   S = sum_k p(lambda_k)

whose expectation is the integral the grid approximates. So the two must agree,
and disagreement is the test even though Monte Carlo means exactness is
unavailable. Three checks, each failing a different way:

A. SWITCHING. On a scene with no dispersion, vis_hero must converge to
   vis_fused across the whole frame. Noise falls as 1/sqrt(spp); a bias does
   not fall at all, and the difference between those two behaviours is what
   this measures. A wrong 1/pdf, a missing quartet denominator or a matching
   function applied at the wrong vertex all show up as an error that stops
   falling. Its predecessor -- splitting the band into 8 groups with a ray
   each -- failed exactly here and was reverted.

B. COLLAPSE. A quartet that meets a dispersive interface cannot follow four
   directions, so it refracts by its hero wavelength and returns a scalar. With
   a CONSTANT n(lambda) that collapse changes nothing geometric, so the render
   must still converge to the non-dispersing deterministic one, and the only
   variable is the spectral bookkeeping around the collapse.

C. DISPERSION. n(lambda) must actually reach the refraction. Measured as a
   signal against a floor rather than against a constant: perturbing the
   dispersion by a fraction of a percent gives a difference that is pure
   residual noise, and tripling it gives the physical one. Both come from the
   same renderer, the same seed and the same code path, so the ratio is a
   number this check can hold to -- and it improves with sample count, where a
   hand-set constant on the raw difference is met by noise from below and
   passes for the wrong reason.

The two n(lambda) sources and the RGB path are check_dispersion.py's job.

    python3 scripts/render-tests/check_hero_wavelength.py

Exit code 0 = pass, 1 = fail, 2 = no CLI, 3 = no usable GPU.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
GLTF = ROOT / "assets/models/prism_dispersion.gltf"
PRISM_CFG = ROOT / "assets/configs/prism_dispersion.toml"
CLI = ROOT / "build/src/app/Release/Quantiloom.exe"
WORK = ROOT / "renders/hero"
TMP_CFG = ROOT / "assets/configs/_hero_tmp.toml"
UNIT = "assets/data/refractiveindex/unit_ior.yml"

GPU_ABSENT = re.compile(
    r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")

BK7_ABBE = 64.17
BK7_DISPERSION = 20.0 / BK7_ABBE      # the KHR parameter, 0.3117 for BK7
MATERIAL = "PrismGlass_BK7"
SEED = 12345

# Small on purpose: these run many spp, and both checks are frame statistics
# rather than pixel comparisons.
BOX_RES = 192
PRISM_RES = 256

REF_SPP = 8192          # the deterministic reference for check A
SPPS = [64, 256, 1024]  # 16x between the ends, so noise alone falls 4x
DISPERSION_SPP = 512

# Luminance, because that is the channel whose quadrature error is negligible.
# The 32-point grid's Riemann sum of a D65-shaped illuminant reads X 0.19% high
# and Z 0.60% high against the exact integral, while Y is 0.001% low; the
# reference is therefore the biased one in blue, by about half a percent, and
# check A reports that per channel and holds only luminance to a bound.
LUMA = np.array([0.2126, 0.7152, 0.0722])

# The Cornell box: one emitter, coloured walls, every lit pixel outside the
# panel reached by bouncing. It has no glass and no atmosphere, so check A
# measures the estimator and nothing else. Written out rather than patched from
# assets/configs/cornell_box_vis.toml so the resolution, sample count and mode
# are all in one place.
BOX_CFG = """[renderer]
resolution = [{res}, {res}]
spp = {spp}
seed = {seed}
output = "_hero_{name}.exr"

[spectral]
mode = "{mode}"
basis_file = "H:/Quantiloom-dev/assets/spectral/quantiloom_basis_v3_usgs.qlbin"
materials_json = "H:/Quantiloom-dev/assets/spectral/quantiloom_materials_usgs.json"
band = "VIS"

[scene]
gltf = "H:/Quantiloom-dev/assets/models/cornell_box/cornell_box.gltf"
world_units_to_meters = 0.001

[camera]
position = [278.0, 273.0, -800.0]
look_at  = [278.0, 273.0, 280.0]
up       = [0.0, 1.0, 0.0]
fov_y    = 39.0

[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance  = [0.0, 0.0, 0.0]
sky_radiance  = [0.0, 0.0, 0.0]

[material]
albedo = [0.5, 0.5, 0.5]

[quality]
fail_on_srgb_upsample = false

[sensor]
enabled = false

[material_overrides."Light Panel"]
emissive_curve = "d65"
"""


def render(name):
    """Run the CLI on TMP_CFG and return the frame as float64 RGB."""
    proc = subprocess.run(
        [str(CLI), str(TMP_CFG.relative_to(ROOT))],
        # encoding explicitly, not text=True: that decodes with the locale
        # codec, and the renderer's log has bytes GBK rejects. The reader thread
        # then dies and stdout returns None, so the failure arrives as a
        # TypeError with nothing to suggest an encoding. errors=replace: this is
        # only ever grepped for an ASCII marker.
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + (proc.stderr or "")
    if "Saved spectral image" not in log:
        if GPU_ABSENT.search(log):
            print("no usable GPU, nothing measured", file=sys.stderr)
            sys.exit(3)
        print(f"ERROR: {name} produced no output", file=sys.stderr)
        print("\n".join(log.splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)

    produced = ROOT / f"_hero_{name}.exr"
    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"{name}.exr"
    shutil.copy2(produced, dest)
    produced.unlink(missing_ok=True)
    (ROOT / f"_hero_{name}.png").unlink(missing_ok=True)

    with OpenEXR.File(str(dest)) as f:
        ch = f.channels()
        return ch[next(iter(ch))].pixels.astype(np.float64)[..., :3]


def render_box(name, mode, spp):
    TMP_CFG.write_text(BOX_CFG.format(res=BOX_RES, spp=spp, seed=SEED,
                                      name=name, mode=mode))
    return render(name)


def render_prism(name, mode, spp, dispersion=None, cri=None, ior=None):
    """Render the prism, having first written dispersion and IOR into the glTF.

    dispersion is the KHR_materials_dispersion parameter, or None to remove the
    extension; cri names a refractive-index table to bind instead.
    """
    doc = json.loads(GLTF.read_text())
    for mat in doc["materials"]:
        ext = mat.setdefault("extensions", {})
        if "KHR_materials_transmission" not in ext:
            continue
        if dispersion is None:
            ext.pop("KHR_materials_dispersion", None)
        else:
            ext["KHR_materials_dispersion"] = {"dispersion": dispersion}
        if ior is not None:
            ext["KHR_materials_ior"] = {"ior": ior}
    GLTF.write_text(json.dumps(doc, indent=1))

    cfg = PRISM_CFG.read_text()
    cfg = cfg.replace('mode = "rgb"', f'mode = "{mode}"')
    cfg = cfg.replace("resolution = [1024, 1024]", f"resolution = [{PRISM_RES}, {PRISM_RES}]")
    cfg = cfg.replace("spp = 64", f"spp = {spp}")
    cfg = cfg.replace("seed = 12345", f"seed = {SEED}")
    cfg = cfg.replace('output = "prism_dispersion.exr"', f'output = "_hero_{name}.exr"')
    if cri:
        cfg += f'\n[refractive_index]\n"{MATERIAL}" = "{cri}"\n'
    TMP_CFG.write_text(cfg)
    return render(name)


def luminance(a):
    return a @ LUMA


def relative_l1(a, b, scale):
    return float(np.abs(a - b).mean()) / scale


if not CLI.exists():
    print(f"no CLI at {CLI} -- build first", file=sys.stderr)
    sys.exit(2)

failures = []
backup = GLTF.read_text()
started = time.time()
try:
    # ------------------------------------------------------------------ A ---
    print("A. vis_hero converges to vis_fused on a scene with no dispersion")
    ref = render_box("box_ref", "vis_fused", REF_SPP)
    ref_luma = luminance(ref)
    scale = max(float(np.abs(ref_luma).mean()), 1e-12)
    print(f"   reference: vis_fused at {REF_SPP} spp, mean luminance {scale:.6e}")
    print(f"   {'spp':>6}  {'error (luminance)':>18}  {'x sqrt(spp)':>12}  {'signed':>10}")

    errs = []
    bias = None
    hero = None
    for spp in SPPS:
        hero = render_box(f"box_hero_{spp}", "vis_hero", spp)
        hero_luma = luminance(hero)
        err = relative_l1(hero_luma, ref_luma, scale)
        bias = float((hero_luma - ref_luma).mean()) / scale
        errs.append(err)
        print(f"   {spp:6d}  {err:18.4%}  {err * spp ** 0.5:12.4f}  {bias:+10.4%}")

    # Noise falls as 1/sqrt(spp), so 16x the samples is 4x less of it; a bias
    # does not fall at all. Requiring half is the loose half of that, and it is
    # the check that caught two clamps which each turned this estimator from
    # noisy into biased.
    drop = errs[0] / max(errs[-1], 1e-12)
    if drop < 2.0:
        failures.append(f"A: error fell only {drop:.2f}x over {SPPS[-1] // SPPS[0]}x the "
                        f"samples -- pure noise would fall "
                        f"{(SPPS[-1] / SPPS[0]) ** 0.5:.1f}x, so part of it is bias")
    if abs(bias) > 0.003:
        failures.append(f"A: signed luminance error {bias:+.4%} at {SPPS[-1]} spp is a "
                        f"bias floor, not noise")

    print(f"   error fell {drop:.2f}x over {SPPS[-1] // SPPS[0]}x the samples")
    print("   per channel at the top sample count, against the same reference:")
    for i, c in enumerate("RGB"):
        b = float((hero[..., i] - ref[..., i]).mean()) / float(ref[..., i].mean())
        print(f"     {c}  {b:+.4%}")
    print("   blue reads low by about half a percent because the 32-point grid's")
    print("   Riemann sum of a D65-shaped illuminant reads Z high by that much.")

    # ------------------------------------------------------------------ B ---
    # Reference: IOR = 1.0 so refraction is the identity, and no dispersion
    # extension at all. The collapse renders below use the same IOR, so the
    # geometry is identical on both sides and the only variable is the spectral
    # bookkeeping. Getting this wrong -- leaving the reference at the glass's
    # real IOR while the collapse case used 1.0 -- compares two different scenes
    # and reports a bias that is not there.
    print()
    print("B. the dispersive collapse converges, with a constant n(lambda)")
    ref = render_prism("collapse_ref", "vis_fused", SPPS[-1], ior=1.0)
    scale = max(float(np.abs(ref).mean()), 1e-12)
    print(f"   reference: vis_fused at {SPPS[-1]} spp, no dispersive material")
    print(f"   {'spp':>6}  {'relative error':>15}  {'x sqrt(spp)':>12}")
    errs = []
    for spp in SPPS:
        img = render_prism(f"collapse_{spp}", "vis_fused", spp, cri=UNIT, ior=1.0)
        err = relative_l1(img, ref, scale)
        errs.append(err)
        print(f"   {spp:6d}  {err:15.4%}  {err * spp ** 0.5:12.4f}")
    drop = errs[0] / max(errs[-1], 1e-12)
    print(f"   error fell {drop:.2f}x over {SPPS[-1] // SPPS[0]}x the samples")
    if drop < 2.0:
        failures.append(f"B: collapse error fell only {drop:.2f}x over "
                        f"{SPPS[-1] // SPPS[0]}x the samples, so part of it is bias")

    # ------------------------------------------------------------------ C ---
    print()
    print(f"C. n(lambda) reaches the refraction, at {DISPERSION_SPP} spp")
    print(f"   {'mode':>10}  {'floor':>11}  {'signal':>11}  {'signal/floor':>13}")
    for mode in ("vis_fused", "vis_hero"):
        base = render_prism(f"disp_base_{mode}", mode, DISPERSION_SPP,
                            dispersion=BK7_DISPERSION, ior=1.5168)
        # A dispersion 0.2% away is a change no render can show above its own
        # residual noise, so this difference is the floor the signal is read
        # against. Same seed, same code path, one number apart.
        near = render_prism(f"disp_near_{mode}", mode, DISPERSION_SPP,
                            dispersion=BK7_DISPERSION * 1.002, ior=1.5168)
        # Three times BK7's dispersion is an Abbe number near 21, which is dense
        # flint rather than a fiction, and it moves the refracted beam by an
        # amount the frame can show.
        far = render_prism(f"disp_far_{mode}", mode, DISPERSION_SPP,
                           dispersion=3.0 * BK7_DISPERSION, ior=1.5168)

        scale = max(float(np.abs(base).mean()), 1e-12)
        floor = relative_l1(near, base, scale)
        signal = relative_l1(far, base, scale)
        ratio = signal / max(floor, 1e-15)
        print(f"   {mode:>10}  {floor:11.3e}  {signal:11.3e}  {ratio:13.1f}")
        if ratio < 50.0:
            failures.append(f"C: in {mode}, tripling the dispersion moved the render "
                            f"only {ratio:.1f}x further than a 0.2% change did, so "
                            f"n(lambda) is barely reaching the refraction")

        # Reported, not held to a bound: dispersion separates the channels, so
        # the change is not the same size in each. A number near 1 would mean
        # the render only got brighter.
        per_channel = [float(np.abs(far[..., i] - base[..., i]).mean()) for i in range(3)]
        spread = max(per_channel) / max(min(per_channel), 1e-15)
        print(f"               channel imbalance of that change: {spread:.2f}x")
finally:
    GLTF.write_text(backup)
    TMP_CFG.unlink(missing_ok=True)

print()
print(f"({time.time() - started:.0f} s)")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("hero wavelength checks PASS")
