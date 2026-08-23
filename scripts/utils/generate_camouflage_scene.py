#!/usr/bin/env python3
"""A flat target lying flush in flat ground: the one scene that isolates spectrum.

Section VIII-F's camouflage panel claims a target indistinguishable from its
background in the visible and separated from it in the thermal bands. Testing
that requires the *only* difference between target and background to be the
material, and a compact object on a plane does not provide it: the first attempt
used a sphere resting on the desert scene's ground and found the target more
visible in the visible band (Michelson contrast 0.117) than in MWIR (0.060),
because a sphere's normals point somewhere else than the ground's and shading
alone separates it whatever its spectrum.

So the target here is a square patch of the ground, sharing the ground's
orientation exactly. Same normal, same irradiance, same view angle, no cast
shadow of its own worth the name -- and therefore any contrast that survives is
the spectrum's. The patch is lifted by a hair (default 10 mm) rather than
stitched into the grid: co-planar geometry z-fights, and 10 mm at this sun
elevation casts a shadow far below one pixel.

Usage:
    generate_camouflage_scene.py
    generate_camouflage_scene.py --extent 60 --patch 8
"""

import argparse
import json
import pathlib
import struct

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "assets" / "models" / "desert_camouflage"

GROUND_MATERIAL = "DesertGround"
TARGET_MATERIAL = "CamouflageTarget"
GROUND_EMISSIVITY = 0.90
TARGET_EMISSIVITY = 0.90        # equal on purpose; see the config's header


def quad(half_x, half_z, y):
    """One axis-aligned quad in the ground plane, wound counter-clockwise from
    above so its geometric normal is +Y like the ground's."""
    positions = [(-half_x, y, -half_z), (half_x, y, -half_z),
                 (half_x, y, half_z), (-half_x, y, half_z)]
    normals = [(0.0, 1.0, 0.0)] * 4
    indices = [0, 3, 2, 0, 2, 1]
    return positions, normals, indices


def write_csv(path, value):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# wavelength_nm, value\n")
        f.write(f"300, {value}\n")
        f.write(f"15000, {value}\n")


def material(name, curve, colour):
    return {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": colour,
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": False,
        "extensions": {
            "QUANTILOOM_material_ir": {
                "emissivityCurve": curve,
                "temperature_K": 320.0,
            }
        },
    }


def build(name, extent, patch, lift):
    ground_p, ground_n, ground_i = quad(extent / 2, extent / 2, 0.0)
    target_p, target_n, target_i = quad(patch / 2, patch / 2, lift)

    base = len(ground_p)
    positions = ground_p + target_p
    normals = ground_n + target_n
    indices = ground_i + [i + base for i in target_i]

    idx_b = struct.pack(f"<{len(indices)}I", *indices)
    pos_b = struct.pack(f"<{len(positions) * 3}f", *[c for v in positions for c in v])
    nor_b = struct.pack(f"<{len(normals) * 3}f", *[c for v in normals for c in v])
    buf = idx_b + pos_b + nor_b
    pos_off, nor_off = len(idx_b), len(idx_b) + len(pos_b)

    xs = [v[0] for v in positions]
    ys = [v[1] for v in positions]
    zs = [v[2] for v in positions]

    ground_csv, target_csv = "camo_ground_emissivity.csv", "camo_target_emissivity.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / ground_csv, GROUND_EMISSIVITY)
    write_csv(OUT_DIR / target_csv, TARGET_EMISSIVITY)

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_camouflage_scene.py"},
        "extensionsUsed": ["QUANTILOOM_material_ir"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "CamouflageGround"}],
        "meshes": [{
            "name": "CamouflageGround",
            "primitives": [
                {"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0, "material": 0},
                {"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 3, "material": 1},
            ],
        }],
        "materials": [
            material(GROUND_MATERIAL, ground_csv, [0.62, 0.52, 0.36, 1.0]),
            material(TARGET_MATERIAL, target_csv, [0.62, 0.52, 0.36, 1.0]),
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": len(ground_i),
             "type": "SCALAR", "byteOffset": 0},
            {"bufferView": 1, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": [min(xs), min(ys), min(zs)],
             "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 2, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 0, "componentType": 5125, "count": len(target_i),
             "type": "SCALAR", "byteOffset": len(ground_i) * 4},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_b), "target": 34963},
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": nor_off, "byteLength": len(nor_b), "target": 34962},
        ],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(buf)}],
    }

    (OUT_DIR / f"{name}.gltf").write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    (OUT_DIR / f"{name}.bin").write_bytes(buf)

    stats = {"name": name, "extent_m": extent, "patch_m": patch, "patch_lift_m": lift,
             "materials": [GROUND_MATERIAL, TARGET_MATERIAL],
             "triangles": len(indices) // 3,
             "note": "target and ground are co-planar and co-normal, so any contrast "
                     "between them is spectral"}
    (OUT_DIR / f"{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"{name}: {extent:g} m ground, {patch:g} m target patch, "
          f"{len(indices) // 3} triangles -> {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="camouflage_flat")
    parser.add_argument("--extent", type=float, default=60.0)
    parser.add_argument("--patch", type=float, default=8.0)
    parser.add_argument("--lift", type=float, default=0.01)
    args = parser.parse_args()
    build(args.name, args.extent, args.patch, args.lift)


if __name__ == "__main__":
    main()
