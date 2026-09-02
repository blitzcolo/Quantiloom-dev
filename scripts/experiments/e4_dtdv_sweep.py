#!/usr/bin/env python3
"""E4 -- what the sun-visibility tangent is worth, as a function of mesh size.

The solver carries one temperature per triangle, so the temperature field it
produces can only have edges where the mesh has edges. On a 120 m ground at
201 x 201 that is a 0.6 m triangle, and a shadow whose true transition band is
35 mm wide comes out with a staircase edge. The tangent dT/dv exists to undo
that: the shader traces its own ray and evaluates

    T(x) = T_element + (v(x) - v_element) * dT/dv

This measures how much that is worth, against a reference that is exact.

**The reference.** Every element is an independent one-dimensional column with
no lateral conduction, driven by its own sun-visibility history. So the exact
temperature at a POINT is that point's own visibility history integrated
through the same column -- which costs O(number of sample points) and is
completely independent of how finely the scene happens to be tessellated. That
independence is the whole reason a mesh-refinement study can have a ground
truth at all.

Two things make the reference trustworthy rather than merely plausible, and
both are checked rather than assumed:

  * The column integrator reproduces the scene solver's own answer for a fully
    lit element under an unoccluded sky to 0.0001 K (see Batch 0). Any residual
    there would be indistinguishable from the discretisation being measured.
  * The material properties are read from the solver's own element dump, not
    from the config. A material bound to a measured spectrum has its long-wave
    emissivity replaced by the Planck-weighted band average of that spectrum,
    so a config saying 0.90 can be solved at 0.9329 -- a 0.4 K error that would
    look exactly like a result.

**Visibility, three ways, and they are not the same.** The solver samples the
sun as a disc with several rays, so its v_element is fractional at an edge.
The shader traces one ray, so its v(x) is binary. The reference uses the disc
properly, with many rays. At this geometry the solar disc subtends 17.4 mm of
half-penumbra at the shadow, against triangles of 300-2400 mm, so the three
agree everywhere except within a few centimetres of the edge -- but that is
measured here rather than asserted.

Usage:
    e4_dtdv_sweep.py --solve
    e4_dtdv_sweep.py --reference
    e4_dtdv_sweep.py --compare --out-dir evidence/e4_dtdv
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runlog import REPO  # noqa: E402

CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
COLUMN = REPO / "build" / "src" / "tools" / "Release" / "thermal_column_tool.exe"
BASE_CONFIG = REPO / "assets" / "configs" / "desert" / "desert_thermal_lwir.toml"
FORCING = REPO / "assets" / "configs" / "desert" / "desert_day.csv"
WORK = pathlib.Path(r"H:\quantiloom-paper\evidence\e4_dtdv")

# These are Windows paths; under WSL they are directory NAMES, not paths.
# See scripts/experiments/_winpaths.py for what that silently does.
from _winpaths import require_windows_paths  # noqa: E402
require_windows_paths(WORK)

DIVISIONS = [51, 101, 201, 401]
EVALUATE_H = 87.0          # day 4 at 15:00; three days of spin-up ahead of it
TRANSECT_HALF_M = 3.0
TRANSECT_POINTS = 201
DISC_RAYS = 64             # for the reference's fractional visibility
SOLAR_ANGULAR_RADIUS = 0.00465   # rad, as ThermalExchangePrecompute uses


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

ACTIVE_FORCING = [FORCING]


def sun_direction(time_h):
    """From the forcing file, by the same construction LoadForcingCsv uses."""
    rows = []
    for line in ACTIVE_FORCING[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(v) for v in line.split(",")])
    times = np.array([r[0] for r in rows])
    azimuth = np.interp(time_h, times, [r[3] for r in rows])
    elevation = np.interp(time_h, times, [r[4] for r in rows])
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.sin(a), math.sin(e),
                     -math.cos(e) * math.cos(a)]), elevation


def forcing_times():
    times = []
    for line in ACTIVE_FORCING[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        times.append(float(line.split(",")[0]))
    return np.array(times)


def hits_sphere(origin, direction, centre, radius):
    """Ray-sphere test, vectorised over rays."""
    oc = centre - origin
    t_ca = direction @ oc
    perpendicular = (oc @ oc) - t_ca * t_ca
    return (t_ca > 0.0) & (perpendicular < radius * radius)


def disc_directions(sun, rays, angular_radius):
    """Directions sampling the solar disc around `sun`, roughly uniformly."""
    up = np.array([0.0, 0.0, 1.0]) if abs(sun[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(sun, up)
    right /= np.linalg.norm(right)
    down = np.cross(sun, right)

    # Sunflower spiral: even coverage without a random seed to record.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(rays):
        r = angular_radius * math.sqrt((i + 0.5) / rays)
        theta = i * golden
        d = sun + r * (math.cos(theta) * right + math.sin(theta) * down)
        out.append(d / np.linalg.norm(d))
    return np.array(out)


def visibility_series(point, times, centre, radius, rays):
    """Fractional sun visibility at one ground point through the whole run."""
    series = np.zeros(len(times))
    for index, t in enumerate(times):
        sun, elevation = sun_direction(t)
        if elevation <= 0.0:
            continue           # below the horizon: the solver contributes nothing
        directions = disc_directions(sun, rays, SOLAR_ANGULAR_RADIUS)
        blocked = hits_sphere(point, directions, centre, radius)
        series[index] = 1.0 - blocked.mean()
    return series


# ---------------------------------------------------------------------------
# The solver's own numbers
# ---------------------------------------------------------------------------

def read_dump(path):
    """(materials, records) from a thermal.dump_elements CSV."""
    materials, header, rows = {}, None, []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# material "):
            index = int(line.split()[2].rstrip(":"))
            fields = {}
            for token in line.split(":", 1)[1].split():
                key, _, value = token.partition("=")
                fields[key] = value
            materials[index] = fields
        elif header is None:
            header = line.split(",")
        else:
            rows.append(line.split(","))
    data = {name: np.array([r[i] for r in rows]) for i, name in enumerate(header)}
    return materials, data


def element_index(x, z, divisions, extent=120.0):
    """Which triangle of the uniform grid contains (x, z).

    Mirrors generate_desert_scene.py: cells in row-major order over (j, i), two
    triangles each, the first spanning the half where the local z fraction is
    at least the local x fraction. Verified against the dumped centroids rather
    than trusted -- an off-by-one here would silently compare a point against
    its neighbour and look like a discretisation error.
    """
    cells = divisions - 1
    step = extent / cells
    half = extent * 0.5
    u = (x + half) / step
    w = (z + half) / step
    i = int(np.clip(np.floor(u), 0, cells - 1))
    j = int(np.clip(np.floor(w), 0, cells - 1))
    local_u, local_w = u - i, w - j
    return 2 * (j * cells + i) + (0 if local_w >= local_u else 1)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def solve(args):
    WORK.mkdir(parents=True, exist_ok=True)
    base = BASE_CONFIG.read_text(encoding="utf-8")

    # Three arms of the same solve. "raw" is the uncorrected field, one
    # temperature per triangle. "corr" carries the whole-day tangent, which a
    # shading pass applies with the shadow it traces now. "lag" carries a
    # tangent per recent sun column as well, so the pass can trace the pixel's
    # shadow at the hour it was cast -- which is the arm a MOVING shadow is
    # supposed to need and a frozen one is not.
    arms = ["corr", "raw"]
    if args.memory_lags > 0:
        arms.append("lag")

    for divisions in args.divisions:
        for arm in arms:
            tag = f"{divisions}_{arm}{args.suffix}"
            dump = WORK / f"elements_{tag}.csv"
            config = WORK / f"desert_{tag}.toml"
            text = (base
                    .replace("assets/models/desert/desert_201.gltf",
                             f"assets/models/desert/desert_{divisions}.gltf")
                    .replace('dump_elements = "renders/desert/elements_201_1500.csv"',
                             f'dump_elements = "{dump.as_posix()}"')
                    .replace('output = "renders/desert/desert_201_1500.exr"',
                             f'output = "{(WORK / f"desert_{tag}.exr").as_posix()}"')
                    # Absolute, because the generated config does not live beside
                    # the forcing file. A relative name here resolves against the
                    # caller's directory, is not found, and the solver falls back
                    # to CONSTANT forcing -- which produced a uniform 307.8 K field
                    # and a zero tangent, silently, until the warning below was
                    # promoted to a failure.
                    .replace('forcing_file = "desert_day.csv"',
                             f'forcing_file = "{(args.forcing or FORCING).as_posix()}"'))
            if arm == "raw":
                text = text.replace("[thermal]\nenabled = true",
                                    "[thermal]\nenabled = true\nsun_correction = false")
            elif arm == "lag":
                text = text.replace(
                    "[thermal]\nenabled = true",
                    f"[thermal]\nenabled = true\n"
                    f"sun_memory_lags = {args.memory_lags}")
            if not args.render:
                # The dump is what the sweep measures; a full-resolution render
                # is only needed for the figure pair.
                text = text.replace("resolution = [1024, 1024]", "resolution = [128, 128]")
                text = text.replace("spp = 64", "spp = 1")
            config.write_text(text, encoding="utf-8")

            print(f"  solving {tag} ...", flush=True)
            result = subprocess.run([str(CLI), str(config)], cwd=str(REPO),
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=14400)
            if result.returncode != 0 or not dump.is_file():
                print(result.stdout[-3000:], file=sys.stderr)
                raise SystemExit(f"solve failed for {tag}")
            # A forcing file the solver could not read is not an error to it --
            # it falls back to constant forcing and produces a perfectly
            # plausible isothermal field. That is exactly the failure this
            # experiment cannot survive, so it is checked rather than assumed.
            if "using constant forcing" in result.stdout:
                raise SystemExit(f"{tag}: the forcing file was not read; the "
                                 f"solve fell back to constant forcing")
            materials, data = read_dump(dump)
            solved = (data["solved"] == "1").sum()
            print(f"    {len(data['element'])} elements, {solved} solved, "
                  f"eps_lw {materials[0]['eps_lw']}")


def reference(args):
    scene = json.loads((REPO / "assets" / "models" / "desert" /
                        f"desert_{args.divisions[0]}.json").read_text(encoding="utf-8"))
    centre = np.array(scene["sphere_centre_m"])
    radius = scene["sphere_radius_m"]

    sun, elevation = sun_direction(EVALUATE_H)
    ground = sun / sun[1]
    shadow = centre - ground * centre[1]
    horizontal = math.hypot(sun[0], sun[2])
    across = np.array([-sun[2] / horizontal, 0.0, sun[0] / horizontal])

    offsets = np.linspace(-TRANSECT_HALF_M, TRANSECT_HALF_M, TRANSECT_POINTS)
    points = np.array([shadow + across * o for o in offsets])
    points[:, 1] = 0.0

    times = forcing_times()
    print(f"transect: {TRANSECT_POINTS} points across the shadow at "
          f"({shadow[0]:.3f}, {shadow[2]:.3f}), +/-{TRANSECT_HALF_M} m")
    print(f"sun elevation {elevation:.2f} deg, {len(times)} forcing rows")

    # The material as the SOLVER resolved it, from any dump.
    materials, _ = read_dump(
        WORK / f"elements_{args.divisions[0]}_corr{args.suffix}.csv")
    soil = materials[0]
    print(f"column from the dump: k={soil['k']} eps_lw={soil['eps_lw']} "
          f"alpha_sw={soil['alpha_sw']}")

    spec = WORK / f"reference_column{args.suffix}.toml"
    spec.write_text(f"""# Auto-generated by e4_dtdv_sweep.py from the solver's own element dump.
[column]
thermal_conductivity_w_mk = {soil['k']}
density_kg_m3             = {soil['rho']}
specific_heat_j_kgk       = {soil['c']}
thickness_m               = {soil['d']}
convection_h_w_m2k        = {soil['h']}
shortwave_absorptivity    = {soil['alpha_sw']}
ir_emissivity             = {soil['eps_lw']}
wetness_factor            = {soil['wetness']}
interior_bc               = "{soil['interior_bc']}"
interior_temperature_k    = {soil['interior_K']}

[solve]
start_time_h  = 0.0
end_time_h    = {EVALUATE_H}
timestep_s    = 60.0
layers        = 10
initial       = "steady"
output_step_h = {EVALUATE_H}
checkpoint_stride_h = 24.0

[geometry]
normal       = [0.0, 1.0, 0.0]
sky_fraction = 1.0

[forcing]
file = "{ACTIVE_FORCING[0].as_posix()}"
""", encoding="utf-8")

    results = []
    for index, point in enumerate(points):
        series = visibility_series(point, times, centre, radius, DISC_RAYS)
        csv = WORK / f"vis{args.suffix}_{index:04d}.csv"
        csv.write_text("# time_h, visibility\n" + "".join(
            f"{t:.5f}, {v:.6f}\n" for t, v in zip(times, series)), encoding="utf-8")

        out = WORK / f"ref{args.suffix}_{index:04d}.csv"
        run = subprocess.run([str(COLUMN), str(spec), "--visibility", str(csv),
                              "--out", str(out)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=1800)
        if run.returncode != 0:
            print(run.stdout + run.stderr, file=sys.stderr)
            raise SystemExit(f"column failed at point {index}")
        last = out.read_text(encoding="utf-8").splitlines()[-1].split(",")
        # At the EVALUATED hour, not at the last row of the forcing: the file
        # runs to midnight, where the sun is below the horizon and every point
        # reads zero regardless of where the shadow is.
        at_evaluation = float(np.interp(EVALUATE_H, times, series))
        binary = 0.0 if hits_sphere(
            point, sun_direction(EVALUATE_H)[0][None, :], centre, radius)[0] else 1.0
        results.append({"offset_m": float(offsets[index]),
                        "x": float(point[0]), "z": float(point[2]),
                        "visibility_disc": at_evaluation,
                        "visibility_binary": binary,
                        # The whole history, not only the instant: a correction
                        # that traces the shadow at an EARLIER hour needs this
                        # point's visibility then, and recomputing it would
                        # mean re-tracing every ray.
                        "visibility_history": [round(float(v), 6) for v in series],
                        "T_ref_K": float(last[1])})
        if index % 25 == 0:
            print(f"  {index}/{TRANSECT_POINTS}  v={at_evaluation:.3f}  "
                  f"T={float(last[1]):.3f} K", flush=True)

    (WORK / f"reference{args.suffix}.json").write_text(json.dumps({
        "evaluate_h": EVALUATE_H, "shadow_centre": shadow.tolist(),
        "history_times_h": [float(t) for t in times],
        "across": across.tolist(), "sphere_centre": centre.tolist(),
        "sphere_radius": radius, "disc_rays": DISC_RAYS,
        "solar_angular_radius_rad": SOLAR_ANGULAR_RADIUS,
        "penumbra_half_width_m": float(
            np.linalg.norm(shadow - centre) * SOLAR_ANGULAR_RADIUS),
        "points": results}, indent=2), encoding="utf-8")
    print(f"wrote {WORK / f'reference{args.suffix}.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="solve at full resolution, for the figure pair")
    parser.add_argument("--divisions", type=int, nargs="+", default=DIVISIONS)
    parser.add_argument("--forcing", type=pathlib.Path, default=None,
                        help="override the forcing file; used for the "
                             "frozen-sun control, where the shadow does "
                             "not sweep")
    parser.add_argument("--suffix", default="",
                        help="tag appended to the output names")
    parser.add_argument("--memory-lags", type=int, default=0,
                        help="solve a third arm carrying this many per-column "
                             "sun tangents, for the moving-shadow comparison")
    args = parser.parse_args()

    if args.forcing is not None:
        ACTIVE_FORCING[0] = args.forcing.resolve()
    if args.solve:
        solve(args)
    if args.reference:
        reference(args)
    if not (args.solve or args.reference):
        parser.error("pick a stage")


if __name__ == "__main__":
    main()
