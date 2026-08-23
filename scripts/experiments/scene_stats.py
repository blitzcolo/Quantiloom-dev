#!/usr/bin/env python3
"""Count what a scene is made of: triangles, materials, emitters.

The paper's setup section names six scenes and gives the size of none of them.
This reads the asset the way the renderer does -- walking the node tree so that
a mesh instanced twice counts twice -- and prints a table that can go straight
into it.

Emitters are counted two ways because a scene can have them two ways: as
KHR_lights_punctual nodes, and as geometry whose material emits.  The Cornell
box has no punctual light at all; its only source is one emissive quad, and a
count that looked for lights would report zero.

Usage:
    scene_stats.py assets/models/cornell_box/cornell_box.gltf
    scene_stats.py --config assets/configs/cornell_box_vis.toml
    scene_stats.py --config-dir assets/configs --markdown
"""

import argparse
import json
import pathlib
import re
import struct
import sys


def load_gltf(path):
    """The JSON chunk of a .gltf or .glb, plus the directory to resolve against."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".glb":
        with open(path, "rb") as f:
            magic, _version, _length = struct.unpack("<III", f.read(12))
            if magic != 0x46546C67:  # 'glTF'
                raise ValueError(f"{path} is not a GLB container")
            while True:
                header = f.read(8)
                if len(header) < 8:
                    raise ValueError(f"{path} has no JSON chunk")
                chunk_length, chunk_type = struct.unpack("<II", header)
                data = f.read(chunk_length)
                if chunk_type == 0x4E4F534A:  # 'JSON'
                    return json.loads(data.decode("utf-8")), path.parent
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def primitive_triangles(primitive, accessors):
    """Triangles in one primitive. Mode 4 is TRIANGLES; anything else is not
    counted rather than guessed at, and is reported separately."""
    if primitive.get("mode", 4) != 4:
        return 0, True
    if "indices" in primitive:
        count = accessors[primitive["indices"]]["count"]
    else:
        position = primitive.get("attributes", {}).get("POSITION")
        if position is None:
            return 0, False
        count = accessors[position]["count"]
    return count // 3, False


def stats(gltf):
    accessors = gltf.get("accessors", [])
    meshes = gltf.get("meshes", [])
    nodes = gltf.get("nodes", [])
    materials = gltf.get("materials", [])

    mesh_triangles = []
    mesh_materials = []
    non_triangle = False
    for mesh in meshes:
        total = 0
        used = set()
        for primitive in mesh.get("primitives", []):
            count, skipped = primitive_triangles(primitive, accessors)
            non_triangle = non_triangle or skipped
            total += count
            if "material" in primitive:
                used.add(primitive["material"])
        mesh_triangles.append(total)
        mesh_materials.append(used)

    # Walk the tree rather than summing the mesh list: a mesh referenced by two
    # nodes is two instances, and the renderer builds a BLAS instance for each.
    triangles = 0
    instances = 0
    used_materials = set()
    punctual = 0

    def visit(index, depth=0):
        nonlocal triangles, instances, punctual
        if depth > 64 or index >= len(nodes):
            return
        node = nodes[index]
        mesh = node.get("mesh")
        if mesh is not None and mesh < len(meshes):
            triangles += mesh_triangles[mesh]
            instances += 1
            used_materials.update(mesh_materials[mesh])
        extensions = node.get("extensions", {})
        if "KHR_lights_punctual" in extensions:
            punctual += 1
        for child in node.get("children", []):
            visit(child, depth + 1)

    scenes = gltf.get("scenes", [])
    roots = scenes[gltf.get("scene", 0)].get("nodes", []) if scenes else range(len(nodes))
    for root in roots:
        visit(root)

    emissive = 0
    for index in used_materials:
        if index >= len(materials):
            continue
        material = materials[index]
        factor = material.get("emissiveFactor", [0.0, 0.0, 0.0])
        strength = (material.get("extensions", {})
                    .get("KHR_materials_emissive_strength", {})
                    .get("emissiveStrength", 1.0))
        if any(c > 0.0 for c in factor) and strength > 0.0:
            emissive += 1

    return {
        "triangles": triangles,
        "mesh_instances": instances,
        "meshes": len(meshes),
        "materials_used": len(used_materials),
        "materials_declared": len(materials),
        "emissive_materials": emissive,
        "punctual_lights": punctual,
        "non_triangle_primitives": non_triangle,
    }


def scene_from_config(config_path):
    """The asset a TOML config points at, without a TOML parser.

    Deliberately a regex over two keys rather than a dependency: this runs
    beside experiments whose only requirement is a Python that exists, and the
    two keys it needs are written the same way in every config in the tree.
    """
    text = pathlib.Path(config_path).read_text(encoding="utf-8")
    for key in ("gltf", "usd"):
        match = re.search(rf'^\s*{key}\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match:
            candidate = pathlib.Path(match.group(1))
            if not candidate.is_absolute():
                repo = pathlib.Path(__file__).resolve().parents[2]
                beside = pathlib.Path(config_path).parent / candidate
                candidate = beside if beside.exists() else repo / candidate
            return key, candidate
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", nargs="*", type=pathlib.Path)
    parser.add_argument("--config", action="append", default=[], type=pathlib.Path,
                        help="read the asset path out of a TOML config")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, help="also write JSON here")
    args = parser.parse_args()

    targets = [(str(a.name), a) for a in args.assets]
    for config in args.config:
        kind, asset = scene_from_config(config)
        if asset is None:
            print(f"{config}: no scene.gltf or scene.usd", file=sys.stderr)
            continue
        if kind == "usd":
            print(f"{config}: USD scenes are not counted here ({asset.name})",
                  file=sys.stderr)
            continue
        targets.append((config.stem, asset))

    if not targets:
        parser.error("give an asset path or --config")

    results = {}
    for label, asset in targets:
        if not asset.exists():
            print(f"{label}: {asset} not found", file=sys.stderr)
            continue
        try:
            gltf, _ = load_gltf(asset)
        except (ValueError, json.JSONDecodeError, OSError) as error:
            print(f"{label}: {error}", file=sys.stderr)
            continue
        results[label] = dict(stats(gltf), asset=str(asset))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.markdown:
        print("| Scene | Triangles | Mesh instances | Materials | Emissive | Lights |")
        print("|---|---:|---:|---:|---:|---:|")
        for label, s in results.items():
            print(f"| {label} | {s['triangles']:,} | {s['mesh_instances']} | "
                  f"{s['materials_used']} | {s['emissive_materials']} | "
                  f"{s['punctual_lights']} |")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
