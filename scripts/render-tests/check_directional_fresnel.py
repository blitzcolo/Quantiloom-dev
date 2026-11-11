#!/usr/bin/env python3
"""Check that a smooth n,k interface is directional in the thermal bands.

    python3 scripts/render-tests/check_directional_fresnel.py [--part dir|energy|sun|all]

A surface described by measured optical constants -- water, glass, polished
metal -- reflects according to Fresnel, and Fresnel depends strongly on the
angle: liquid water at 10 um reflects 1.0% of the sky looking straight down and
55% at 85 degrees from the normal. Kirchhoff, applied per direction, then makes
its emissivity 1 - F(theta). A thermal camera looking across a sea at grazing
angles therefore sees mostly cold sky, not warm water.

The furnace cavities cannot see whether a renderer gets this right -- in an
isothermal cavity every split of emission and reflection returns B(T) -- and
check_view_independence.py checks the opposite property, for the Lambertian
materials that must NOT depend on the view. This checks the directional case
in three parts, each rendered here through the CLI on assets/models/nk_plane:

  dir     A horizontal water plane at 300 K under a uniform 200 K sky (the same
          constants as the viewangle_* configs), orthographic cameras at 0, 60,
          75 and 85 degrees, in LWIR and MWIR. Each frame must equal

              L(theta) = < (1 - F(theta)) B(T_s) + F(theta) B(T_sky) >

          where F is the unpolarised Fresnel reflectance computed from the
          Hale & Querry table and <> is the band mean by the shader's own
          16-node trapezoid. The surface had one reflectance at every angle,
          the normal-incidence one, until this was written.
  energy  The plane made non-emitting (1 K) under a uniform 300 K sky, with
          aluminium and water at roughness 0.2. Whatever the angle, it may not
          send back more than the sky it is lit by.
  sun     MWIR with a sun at 30 and 70 degrees and nothing else lighting the
          scene. Away from the mirror direction a smooth interface returns no
          sunlight at all -- a Lambertian reflection would be rho E cos / pi --
          and the glint integrated over the outgoing hemisphere must equal
          rho_dh(theta_s) E cos(theta_s), which for a single-scattering GGX
          lobe is F(theta_s) less the energy that lobe loses (measured and
          printed, not assumed).

Exit code 0 = pass, 1 = fail, 2 = no CLI, 3 = no usable GPU.
"""

import argparse
import math
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import OpenEXR

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = pathlib.Path(os.environ.get("CLI", str(ROOT / "build/src/app/Release/Quantiloom.exe")))
TMP_CFG = ROOT / "assets/configs/_nk_directional_tmp.toml"
TMP_OUT = ROOT / "_nk_directional_tmp.exr"
NK_DIR = ROOT / "assets/refractiveindex/database/data/main"
WATER = NK_DIR / "H2O/nk/Hale.yml"
ALUMINIUM = NK_DIR / "Al/nk/Rakic.yml"

GPU_ABSENT = re.compile(r"No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable")

BANDS = {"LWIR": (8000.0, 12000.0, 16), "MWIR": (3000.0, 5000.0, 16)}
T_SURFACE_K = 300.0
T_SKY_K = 200.0
ANGLES = (0.0, 60.0, 75.0, 85.0)

STATS = {"dir": 0.0, "energy": 0.0, "glint": []}

C1_NM = 1.191042972e20
C2_M = 1.438776877e-2


def planck(T_K, lam_nm):
    lam_nm = np.asarray(lam_nm, dtype=np.float64)
    if T_K <= 0:
        return np.zeros_like(lam_nm)
    x = C2_M / (lam_nm * 1e-9 * T_K)
    return np.where(x > 700, 0.0, C1_NM / lam_nm ** 5 / np.expm1(np.minimum(x, 700)))


def nodes(band):
    lo, hi, n = BANDS[band]
    lam = np.linspace(lo, hi, n)
    w = np.full(n, 1.0)
    w[0] = w[-1] = 0.5
    return lam, w * (hi - lo) / (n - 1) / (hi - lo)


def band_mean(values, band):
    _, w = nodes(band)
    return float(np.sum(values * w))


def load_nk(path):
    """The tabulated n,k of a refractiveindex.info YAML, wavelength in nm."""
    rows = []
    in_data = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if "type: tabulated nk" in line:
            in_data = True
            continue
        if in_data:
            parts = line.split()
            if len(parts) == 3:
                try:
                    rows.append(tuple(float(p) for p in parts))
                    continue
                except ValueError:
                    pass
            if rows and not line.strip().startswith("data"):
                break
    arr = np.array(rows)
    return arr[:, 0] * 1000.0, arr[:, 1], arr[:, 2]


def fresnel(cos_t, n, k):
    """Unpolarised reflectance of a conductor; FresnelConductor in common.hlsli."""
    cos_t = np.clip(abs(cos_t), 0.0, 1.0)
    c2 = cos_t * cos_t
    s2 = 1.0 - c2
    t0 = n * n - k * k - s2
    a2b2 = np.sqrt(t0 * t0 + 4.0 * n * n * k * k)
    t1 = a2b2 + c2
    a = np.sqrt(0.5 * (a2b2 + t0))
    t2 = 2.0 * a * cos_t
    rs = (t1 - t2) / (t1 + t2)
    t3 = c2 * a2b2 + s2 * s2
    t4 = t2 * s2
    rp = rs * (t3 - t4) / (t3 + t4)
    return 0.5 * (rs + rp)


def fresnel_at(path, band, theta_deg):
    wl, n, k = load_nk(path)
    lam, _ = nodes(band)
    return fresnel(math.cos(math.radians(theta_deg)), np.interp(lam, wl, n), np.interp(lam, wl, k))


def brightness_temperature(L, band):
    lo, hi = 150.0, 400.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if band_mean(planck(mid, nodes(band)[0]), band) < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def render(cfg_text):
    TMP_CFG.write_text(cfg_text)
    try:
        proc = subprocess.run([str(CLI), str(TMP_CFG.relative_to(ROOT))], cwd=ROOT,
                              capture_output=True, encoding="utf-8", errors="replace")
    finally:
        TMP_CFG.unlink(missing_ok=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    if "Saved spectral image" not in log:
        if GPU_ABSENT.search(log):
            print("no usable GPU, nothing measured", file=sys.stderr)
            sys.exit(3)
        print("ERROR: render failed", file=sys.stderr)
        print("\n".join(log.splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)
    with OpenEXR.File(str(TMP_OUT)) as f:
        ch = f.channels()
        px = ch[next(iter(ch))].pixels.astype(np.float64)
    for suffix in (".exr", ".png", ".metadata.json", ".replay.toml"):
        TMP_OUT.with_suffix(suffix).unlink(missing_ok=True)
    return px[:, :, 0] if px.ndim == 3 else px


def config(*, band, nk, camera, t_surface, t_sky, roughness=None, sun=None,
           resolution=(32, 32), spp=64):
    mode = band.lower() + "_fused"
    rough = f"roughness = {roughness}\n" if roughness is not None else ""
    if sun is None:
        lighting = ('sun_direction = [0.0, 1.0, 0.0]\nsun_radiance = [0.0, 0.0, 0.0]\n'
                    'sky_radiance = [0.0, 0.0, 0.0]\n')
    else:
        lighting = (f"sun_direction = [{sun[0]:.9f}, {sun[1]:.9f}, {sun[2]:.9f}]\n"
                    "sun_radiance = [5.0, 5.0, 5.0]\n"
                    'solar_lut = "assets/luts/flat_sun_sky.csv"\n'
                    "solar_lut_columns = [4, 3]\nsolar_lut_diffuse_is_global = true\n"
                    "sky_radiance = [0.0, 0.0, 0.0]\n")
    return f"""[renderer]
resolution = [{resolution[0]}, {resolution[1]}]
spp = {spp}
output = "{TMP_OUT.name}"

[spectral]
mode = "{mode}"
band = "{band}"

[scene]
gltf = "assets/models/nk_plane/nk_plane.gltf"
world_units_to_meters = 1.0
default_temperature_k = {t_surface}

[camera]
{camera}
[lighting]
{lighting}atmosphere_temperature_k = {t_sky}

[material]
albedo = [0.5, 0.5, 0.5]

[[materials]]
name = "nk_plane"
{rough}
[refractive_index]
nk_plane = "{nk.relative_to(ROOT).as_posix()}"

[quality]
fail_on_srgb_upsample = false

[sensor]
enabled = false
"""


def ortho_camera(theta_deg):
    t = math.radians(theta_deg)
    pos = (100.0 * math.sin(t), 100.0 * math.cos(t), 0.0)
    return (f"position = [{pos[0]:.9f}, {pos[1]:.9f}, {pos[2]:.9f}]\n"
            "look_at = [0.0, 0.0, 0.0]\nup = [0.0, 0.0, 1.0]\n"
            'projection = "orthographic"\northo_height = 2.0\n')


def part_dir(tol, failures):
    print("dir: water plane, 300 K under a uniform 200 K sky")
    for band in ("LWIR", "MWIR"):
        lam, _ = nodes(band)
        bs, bk = planck(T_SURFACE_K, lam), planck(T_SKY_K, lam)
        for theta in ANGLES:
            F = fresnel_at(WATER, band, theta)
            ref = band_mean((1.0 - F) * bs + F * bk, band)
            img = render(config(band=band, nk=WATER, camera=ortho_camera(theta),
                                t_surface=T_SURFACE_K, t_sky=T_SKY_K))
            got = float(img.mean())
            err = abs(got - ref) / ref
            STATS["dir"] = max(STATS["dir"], err)
            print(f"  {band} {theta:4.0f} deg  F_mean {band_mean(F, band):.4f}  "
                  f"ref {ref:.6e} (Tb {brightness_temperature(ref, band):6.2f} K)  "
                  f"got {got:.6e} (Tb {brightness_temperature(got, band):6.2f} K)  "
                  f"err {err:.3%}")
            if err > tol:
                failures.append(f"dir {band} {theta:.0f} deg: {err:.3%} off the closed form")


def part_energy(tol, failures):
    print("energy: non-emitting plane under a uniform 300 K sky, roughness 0.2")
    band = "LWIR"
    sky = band_mean(planck(300.0, nodes(band)[0]), band)
    for name, nk in (("water", WATER), ("aluminium", ALUMINIUM)):
        for theta in ANGLES:
            img = render(config(band=band, nk=nk, camera=ortho_camera(theta), t_surface=1.0,
                                t_sky=300.0, roughness=0.2))
            ratio = float(img.mean()) / sky
            STATS["energy"] = max(STATS["energy"], ratio)
            F = band_mean(fresnel_at(nk, band, theta), band)
            print(f"  {name:9s} {theta:4.0f} deg  out/in {ratio:.5f}  (smooth F {F:.5f})")
            if ratio > 1.0 + tol:
                failures.append(f"energy {name} {theta:.0f} deg: returns {ratio:.5f} of the sky")


def part_sun(tol, failures):
    print("sun: MWIR, flat 1 W/m^2/nm sun, no sky, no emission")
    band = "MWIR"
    for name, nk in (("water", WATER), ("aluminium", ALUMINIUM)):
        for theta_s in (30.0, 70.0):
            ts = math.radians(theta_s)
            sun = (math.sin(ts), math.cos(ts), 0.0)
            F = band_mean(fresnel_at(nk, band, theta_s), band)
            lambert_off = F * math.cos(ts) / math.pi

            # Off the mirror direction: looking straight down while the sun is
            # 30 or 70 degrees away from the normal.
            img = render(config(band=band, nk=nk, camera=ortho_camera(0.0), t_surface=1.0,
                                t_sky=1.0, roughness=0.05, sun=sun))
            off = float(img.mean())
            print(f"  {name:9s} sun {theta_s:2.0f} deg  nadir view {off:.3e}  "
                  f"(Lambertian would be {lambert_off:.3e})")
            if off > 1e-3 * lambert_off:
                failures.append(f"sun {name} {theta_s:.0f} deg: {off:.3e} returned off "
                                f"the mirror direction")

            # The glint integrated over outgoing directions. A pinhole looking
            # back along the mirror direction; every pixel is one outgoing
            # direction because the sun is directional and the plane uniform.
            W = H = 384
            fov = 50.0
            d = 10.0
            fwd = np.array([math.sin(ts), -math.cos(ts), 0.0])
            pos = -fwd * d
            camera = (f"position = [{pos[0]:.9f}, {pos[1]:.9f}, {pos[2]:.9f}]\n"
                      "look_at = [0.0, 0.0, 0.0]\nup = [0.0, 0.0, 1.0]\n"
                      f"fov_y = {fov}\n")
            img = render(config(band=band, nk=nk, camera=camera, t_surface=1.0, t_sky=1.0,
                                roughness=0.2, sun=sun, resolution=(W, H), spp=4))
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            cup = np.cross(right, fwd)
            s = math.tan(math.radians(fov) / 2.0)
            u = (np.arange(W) + 0.5) / W * 2.0 - 1.0
            v = -((np.arange(H) + 0.5) / H * 2.0 - 1.0)
            uu, vv = np.meshgrid(u * s, v * s)
            dirs = fwd[None, None, :] + uu[..., None] * right + vv[..., None] * cup
            norm = np.linalg.norm(dirs, axis=-1)
            cos_axis = 1.0 / norm
            domega = cos_axis ** 3 * (2.0 * s / H) ** 2
            cos_out = np.maximum(-dirs[..., 1] / norm, 0.0)
            exitance = float(np.sum(img * cos_out * domega))
            expected = F * math.cos(ts)
            ratio = exitance / expected
            STATS["glint"].append(ratio)
            print(f"  {name:9s} sun {theta_s:2.0f} deg  glint exitance {exitance:.5f}  "
                  f"F E cos {expected:.5f}  ratio {ratio:.4f}")
            # Single-scattering GGX loses energy and cannot gain it; a frame
            # cropped by the field of view loses a little more. More than a
            # few percent either way is a different BRDF.
            if not (1.0 - 0.06 <= ratio <= 1.0 + tol * 4):
                failures.append(f"sun {name} {theta_s:.0f} deg: glint integrates to "
                                f"{ratio:.4f} of F E cos")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--part", choices=("dir", "energy", "sun", "all"), default="all")
    p.add_argument("--tol", type=float, default=0.005)
    args = p.parse_args()
    if not CLI.exists():
        print(f"no CLI at {CLI} -- build first", file=sys.stderr)
        return 2
    failures = []
    if args.part in ("dir", "all"):
        part_dir(args.tol, failures)
    if args.part in ("energy", "all"):
        part_energy(args.tol, failures)
    if args.part in ("sun", "all"):
        part_sun(args.tol, failures)
    glint = STATS["glint"] or [float("nan")]
    print(f"summary: dir worst {STATS['dir']:.3%}, energy max {STATS['energy']:.4f}, "
          f"glint {min(glint):.3f}-{max(glint):.3f}")
    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        return 1
    print("directional Fresnel PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
