#!/usr/bin/env python3
"""Generate the desert scene: a uniformly tessellated ground plane and one sphere.

This is the scene the sun-visibility tangent exists for.  A large flat ground
under an open sky is the case where every surface element is very nearly an
independent 1-D column -- its hemisphere is almost all sky, so its neighbours
barely reach it -- and a single compact object on it casts the one shadow whose
edge the mesh cannot resolve.  Both facts are load-bearing:

  * The sky-dominated hemisphere is what makes a pointwise reference legitimate.
    The exact temperature at a point is that point's own visibility history
    integrated through the same column, and that reference costs O(samples)
    rather than O(elements) -- which is the only reason a mesh-refinement study
    can have a ground truth at all.

  * The shadow edge is where a per-triangle temperature field visibly fails.
    At the default 120 m over 200 cells each triangle is 0.6 m, and a shadowed
    sand element sits tens of kelvin below a sunlit one, so the quantisation is
    the loudest thing in the frame rather than a subtlety.

The grid is UNIFORM, deliberately.  A graded grid concentrates resolution where
the subject is and is the right choice for a render; here the independent
variable *is* the triangle edge length, and a grid whose edge length varies with
radius does not have one.

Usage:
    generate_desert_scene.py                      # the reference 201x201
    generate_desert_scene.py --divisions 51 101 201 401
    generate_desert_scene.py --divisions 201 --extent 120 --sphere-radius 0.7

Each run writes, into assets/models/desert/:
    desert_<N>.gltf / .bin    the scene
    desert_<N>.json           its measured numbers, for a script or a table to
                              read rather than recompute
    desert_ground_emissivity.csv, desert_sphere_emissivity.csv

`--divisions N` is the vertex count per side, so the cell count is N-1 and the
triangle count 2*(N-1)^2.  N=201 over 120 m is the 0.6 m triangle the paper
describes.
"""

import argparse
import json
import math
import pathlib
import struct

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "assets" / "models" / "desert"

# Sand and a painted sphere. Only the fallback: a config binding a measured
# spectrum through spectral_material_ref replaces these, and should.
GROUND_EMISSIVITY = 0.92
SPHERE_EMISSIVITY = 0.90

GROUND_MATERIAL = "DesertGround"
SPHERE_MATERIAL = "ShadowSphere"


def ground_grid(extent, divisions):
    """A uniform (divisions x divisions)-vertex grid centred on the origin, +Y up.

    Vertices are shared between triangles, so the triangle count is exactly
    2*(divisions-1)^2 and every triangle has the same area -- which is what lets
    "triangle edge length" be a single number on the x axis of a scaling plot.
    """
    half = extent * 0.5
    step = extent / (divisions - 1)

    positions = []
    for j in range(divisions):
        z = -half + j * step
        for i in range(divisions):
            positions.append((-half + i * step, 0.0, z))

    indices = []
    for j in range(divisions - 1):
        for i in range(divisions - 1):
            a = j * divisions + i
            b = a + 1
            c = a + divisions
            d = c + 1
            # Counter-clockwise seen from above, so (v1-v0) x (v2-v0) = +Y.
            indices += [a, c, d, a, d, b]

    normals = [(0.0, 1.0, 0.0)] * len(positions)
    return positions, normals, indices


def uv_sphere(centre, radius, segments=48, rings=24):
    """A UV sphere. Its own tessellation is not under study -- it only has to be
    round enough that the shadow it casts has a smooth edge, since a faceted
    silhouette would put its own quantisation into the measurement."""
    cx, cy, cz = centre
    positions = []
    normals = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments + 1):
            theta = 2.0 * math.pi * s / segments
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            normals.append((nx, ny, nz))
            positions.append((cx + radius * nx, cy + radius * ny, cz + radius * nz))

    indices = []
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a = r * row + s
            b = a + row
            if r != 0:
                indices += [a, b, a + 1]
            if r != rings - 1:
                indices += [a + 1, b, b + 1]
    return positions, normals, indices


def write_csv(path, value):
    """A constant spectral curve spanning every band the renderer integrates."""
    with open(path, "w") as f:
        f.write("# wavelength_nm, value\n")
        f.write(f"300, {value}\n")
        f.write(f"15000, {value}\n")


def build(name, extent, divisions, sphere_radius, sphere_height):
    ground_p, ground_n, ground_i = ground_grid(extent, divisions)
    # A zero radius means ground only. The sphere is the shadow-edge study's
    # occluder; the gallery scene of Section VIII-F puts a tank there instead
    # and does not want a sphere hanging two metres over it.
    if sphere_radius > 0.0:
        centre = (0.0, sphere_height + sphere_radius, 0.0)
        sphere_p, sphere_n, sphere_i = uv_sphere(centre, sphere_radius)
    else:
        sphere_p, sphere_n, sphere_i = [], [], []

    # One buffer, two primitives, each with its own accessor range. The sphere's
    # indices are rebased onto the concatenated vertex array.
    base = len(ground_p)
    positions = ground_p + sphere_p
    normals = ground_n + sphere_n
    indices = ground_i + [i + base for i in sphere_i]

    # uint32 throughout: 401 divisions is 160,801 vertices, well past what a
    # uint16 index can name, and picking the width per file would make the two
    # ends of the sweep differ in more than tessellation.
    idx_b = struct.pack(f"<{len(indices)}I", *indices)
    pos_b = struct.pack(f"<{len(positions) * 3}f", *[c for v in positions for c in v])
    nor_b = struct.pack(f"<{len(normals) * 3}f", *[c for v in normals for c in v])

    buf = idx_b + pos_b + nor_b
    pos_off = len(idx_b)
    nor_off = pos_off + len(pos_b)

    xs = [v[0] for v in positions]
    ys = [v[1] for v in positions]
    zs = [v[2] for v in positions]

    ground_csv = "desert_ground_emissivity.csv"
    sphere_csv = "desert_sphere_emissivity.csv"
    write_csv(OUT_DIR / ground_csv, GROUND_EMISSIVITY)
    write_csv(OUT_DIR / sphere_csv, SPHERE_EMISSIVITY)

    def material(mat_name, curve, colour):
        return {
            "name": mat_name,
            "pbrMetallicRoughness": {
                "baseColorFactor": colour,
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": False,
            "extensions": {
                "QUANTILOOM_material_ir": {
                    "emissivityCurve": curve,
                    "temperature_K": 300.0,
                }
            },
        }

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_desert_scene.py"},
        "extensionsUsed": ["QUANTILOOM_material_ir"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Desert"}],
        "meshes": [
            {
                "name": "Desert",
                "primitives": (
                    [{"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0,
                      "material": 0}]
                    + ([{"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 3,
                         "material": 1}] if sphere_i else [])
                ),
            }
        ],
        # The sphere's material is written either way. A scene that carries a
        # material nothing references is inert; one whose material indices
        # shift with a flag is a scene whose configs stop matching.
        "materials": [
            material(GROUND_MATERIAL, ground_csv, [0.62, 0.52, 0.36, 1.0]),
            material(SPHERE_MATERIAL, sphere_csv, [0.35, 0.38, 0.30, 1.0]),
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5125,
                "count": len(ground_i),
                "type": "SCALAR",
                "byteOffset": 0,
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            },
            {"bufferView": 2, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        ] + ([
            # Accessor 3, the sphere's indices. Omitted entirely without a
            # sphere rather than left at count 0, which glTF does not allow
            # even for an accessor nothing references.
            {
                "bufferView": 0,
                "componentType": 5125,
                "count": len(sphere_i),
                "type": "SCALAR",
                "byteOffset": len(ground_i) * 4,
            },
        ] if sphere_i else []),
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_b), "target": 34963},
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": nor_off, "byteLength": len(nor_b), "target": 34962},
        ],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(buf)}],
    }

    with open(OUT_DIR / f"{name}.gltf", "w") as f:
        json.dump(gltf, f, indent=2)
    with open(OUT_DIR / f"{name}.bin", "wb") as f:
        f.write(buf)

    cell = extent / (divisions - 1)
    stats = {
        "name": name,
        "extent_m": extent,
        "divisions": divisions,
        "cells_per_side": divisions - 1,
        "cell_size_m": cell,
        # The right-triangle legs are the cell size; the hypotenuse is what the
        # eye reads as the shadow's staircase, so both are recorded and a plot
        # can say which it plotted.
        "triangle_leg_m": cell,
        "triangle_hypotenuse_m": cell * math.sqrt(2.0),
        "triangle_area_m2": 0.5 * cell * cell,
        "ground_triangles": len(ground_i) // 3,
        "sphere_triangles": len(sphere_i) // 3,
        "total_triangles": len(indices) // 3,
        "vertices": len(positions),
        "materials": [GROUND_MATERIAL, SPHERE_MATERIAL],
        "sphere_centre_m": list(centre) if sphere_i else None,
        "sphere_radius_m": sphere_radius,
        "ground_emissivity_fallback": GROUND_EMISSIVITY,
        "sphere_emissivity_fallback": SPHERE_EMISSIVITY,
    }
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--divisions", type=int, nargs="+", default=[201],
                        help="vertices per side; 201 over 120 m is a 0.6 m triangle")
    parser.add_argument("--extent", type=float, default=120.0, help="ground size in metres")
    parser.add_argument("--sphere-radius", type=float, default=0.7)
    parser.add_argument("--sphere-height", type=float, default=2.0,
                        help="gap between the ground and the bottom of the sphere. "
                             "Non-zero by default, and the reason is the reference "
                             "rather than the picture: a sphere resting on the ground "
                             "occludes a large part of the sky over the very elements "
                             "its shadow falls on -- a third of the hemisphere a metre "
                             "away -- so those elements are no longer the open-sky "
                             "columns a pointwise reference integrates, and the study "
                             "would be measuring the reference's error as well as the "
                             "mesh's. Held clear of the ground, the shadow lands where "
                             "the sky fraction is still about 0.98, which the solver's "
                             "own element dump can confirm rather than assume.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for divisions in args.divisions:
        if divisions < 2:
            raise SystemExit(f"--divisions must be at least 2, got {divisions}")
        stats = build(f"desert_{divisions}", args.extent, divisions,
                      args.sphere_radius, args.sphere_height)
        print(f"desert_{divisions}: {stats['total_triangles']:>8,} triangles, "
              f"{stats['triangle_leg_m']:.4f} m leg, "
              f"{stats['vertices']:>8,} vertices")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
