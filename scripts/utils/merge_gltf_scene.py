#!/usr/bin/env python3
"""Merge a .glb into a .gltf scene, so one config can name both.

A Quantiloom config names exactly one scene file, and the gallery scene of the
paper's Section VIII-F is a tank standing on desert ground -- two assets, one
of them generated here and one converted from elsewhere. This composes them
into a single glTF with an external .bin, which is the form the desert
generator already writes.

The merge is index arithmetic. Every glTF array is referenced by position, so
appending one document's arrays to another's means offsetting every reference
by the base length: accessors point at bufferViews, primitives at accessors and
materials, textures at images and samplers, nodes at meshes and children. The
binary is concatenated with 4-byte alignment, and every appended bufferView's
byteOffset moves by where the second buffer landed.

Deliberately not general. It rejects what it has not been taught to renumber --
animations, skins, cameras, morph targets -- rather than dropping them
silently, because a merger that quietly loses a skin produces a scene that
looks almost right.

Usage:
    merge_gltf_scene.py --base desert_201.gltf --add KV-2.glb \
        --out desert_kv2.gltf --scale 1.0 --translate 0 0 0 --rotate-y 30
"""

import argparse
import json
import math
import pathlib
import struct

# Arrays whose entries are referenced by index from elsewhere in the document.
# Order matters only in that every one of them must be offset consistently.
INDEXED = ("bufferViews", "accessors", "meshes", "materials", "textures",
           "images", "samplers", "nodes")

UNSUPPORTED = ("animations", "skins", "cameras")


def read_glb(path):
    """Return (json, binary) from a .glb, or (json, external .bin) from a .gltf."""
    data = path.read_bytes()
    if data[:4] == b"glTF":
        json_length = struct.unpack("<I", data[12:16])[0]
        document = json.loads(data[20:20 + json_length])
        rest = data[20 + json_length:]
        binary = b""
        if len(rest) >= 8:
            bin_length = struct.unpack("<I", rest[0:4])[0]
            if rest[4:8] == b"BIN\x00":
                binary = rest[8:8 + bin_length]
        return document, binary

    document = json.loads(path.read_text(encoding="utf-8"))
    buffers = document.get("buffers", [])
    if not buffers:
        return document, b""
    uri = buffers[0].get("uri")
    if uri is None:
        raise SystemExit(f"{path} has a buffer with no uri and is not a .glb")
    return document, (path.parent / uri).read_bytes()


def offsets_of(base):
    return {key: len(base.get(key, [])) for key in INDEXED}


def shift(entry, key, delta):
    """Add `delta` to entry[key] if it is present. glTF indices are optional."""
    if delta and key in entry and entry[key] is not None:
        entry[key] += delta


def renumber(document, off):
    """Rewrite every cross-reference in `document` by the given offsets."""
    for view in document.get("bufferViews", []):
        # The buffer index is handled by the caller: everything merges into
        # buffer 0, whose data is concatenated.
        view["buffer"] = 0

    for accessor in document.get("accessors", []):
        shift(accessor, "bufferView", off["bufferViews"])
        for sparse_key in ("sparse",):
            sparse = accessor.get(sparse_key)
            if sparse:
                shift(sparse["indices"], "bufferView", off["bufferViews"])
                shift(sparse["values"], "bufferView", off["bufferViews"])

    for image in document.get("images", []):
        shift(image, "bufferView", off["bufferViews"])

    for texture in document.get("textures", []):
        shift(texture, "source", off["images"])
        shift(texture, "sampler", off["samplers"])

    for material in document.get("materials", []):
        # Every texture reference in a material, wherever the spec puts it.
        def walk(node):
            if isinstance(node, dict):
                if "index" in node and "texCoord" in node:
                    shift(node, "index", off["textures"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(material)

    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            primitive["attributes"] = {
                name: index + off["accessors"]
                for name, index in primitive["attributes"].items()
            }
            shift(primitive, "indices", off["accessors"])
            shift(primitive, "material", off["materials"])
            if "targets" in primitive:
                raise SystemExit("morph targets are not handled by this merger")

    for node in document.get("nodes", []):
        shift(node, "mesh", off["meshes"])
        if "children" in node:
            node["children"] = [c + off["nodes"] for c in node["children"]]


def compose_transform(scale, translate, rotate_y_deg):
    """A column-major 4x4 as glTF wants it: scale, then rotate about Y, then move."""
    angle = math.radians(rotate_y_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    # Columns of R*S, then the translation column.
    return [
        scale * cos, 0.0, scale * -sin, 0.0,
        0.0, scale, 0.0, 0.0,
        scale * sin, 0.0, scale * cos, 0.0,
        translate[0], translate[1], translate[2], 1.0,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--add", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--rotate-y", type=float, default=0.0,
                        help="degrees about the vertical axis")
    parser.add_argument("--name", default="Added")
    args = parser.parse_args()

    base, base_bin = read_glb(args.base)
    add, add_bin = read_glb(args.add)

    for key in UNSUPPORTED:
        if add.get(key) or base.get(key):
            raise SystemExit(f"{key} present; this merger does not renumber them")

    off = offsets_of(base)
    renumber(add, off)

    # Concatenate the binaries, 4-byte aligned, and move the appended views.
    padding = (-len(base_bin)) % 4
    binary_offset = len(base_bin) + padding
    merged_bin = base_bin + b"\x00" * padding + add_bin
    for view in add.get("bufferViews", []):
        view["byteOffset"] = view.get("byteOffset", 0) + binary_offset

    for key in INDEXED:
        if add.get(key):
            base.setdefault(key, []).extend(add[key])

    # One node carrying the added asset's roots, so a single transform places
    # the whole thing and the original hierarchy is preserved beneath it.
    roots = [r + off["nodes"] for r in add.get("scenes", [{}])[0].get("nodes", [])]
    base["nodes"].append({
        "name": args.name,
        "children": roots,
        "matrix": compose_transform(args.scale, args.translate, args.rotate_y),
    })
    base["scenes"][0]["nodes"].append(len(base["nodes"]) - 1)

    for used in ("extensionsUsed", "extensionsRequired"):
        if add.get(used):
            base[used] = sorted(set(base.get(used, [])) | set(add[used]))

    bin_name = args.out.stem + ".bin"
    base["buffers"] = [{"uri": bin_name, "byteLength": len(merged_bin)}]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base, indent=2), encoding="utf-8")
    (args.out.parent / bin_name).write_bytes(merged_bin)

    triangles = 0
    for mesh in base["meshes"]:
        for primitive in mesh["primitives"]:
            if "indices" in primitive:
                triangles += base["accessors"][primitive["indices"]]["count"] // 3
    print(f"wrote {args.out}")
    print(f"  {len(base['meshes'])} meshes, {len(base['materials'])} materials, "
          f"{triangles:,} triangles, {len(merged_bin):,} bytes of binary")
    print(f"  materials: {[m.get('name') for m in base['materials']]}")


if __name__ == "__main__":
    main()
