#!/usr/bin/env python3
"""Generate the open ground plane the directional-emissivity gates render.

Writes assets/models/nk_plane/nk_plane.gltf: one 400 m square at y = 0, normal
+Y, with a material that carries NO infrared data of its own -- no emissivity
curve, no reflectance curve, no scalar emissivity. The configs that use it bind
measured n,k by the material name, so the plane is shaded by the directional
Fresnel microfacet lobe and by nothing else.

Why a new plane rather than shadow_scene_open.gltf: that one ships an
emissivity curve in its QUANTILOOM_material_ir extension, which outranks n,k in
the infrared priority order (InfraredSurfaceFractions), so binding n,k to it
would test the curve path again.

400 m because an orthographic camera at 85 degrees spreads a 2 m frame over
2 / cos(85) = 23 m of ground; the plane must reach past every pixel's footprint
and the bounce rays must not find its edge.

Roughness 0: the shaders floor it at MIN_BOUNCE_ROUGHNESS (alpha ~ 0.002), so
the lobe is a mirror to the precision the closed form needs.
"""

import json
import pathlib
import struct

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "models" / "nk_plane"
HALF = 200.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = [-HALF, 0.0, -HALF, HALF, 0.0, -HALF, HALF, 0.0, HALF, -HALF, 0.0, HALF]
    normals = [0.0, 1.0, 0.0] * 4
    # Counter-clockwise seen from +Y.
    indices = [0, 2, 1, 0, 3, 2]

    idx_b = struct.pack("<6H", *indices) + b"\x00\x00"
    pos_b = struct.pack(f"<{len(positions)}f", *positions)
    nor_b = struct.pack(f"<{len(normals)}f", *normals)
    buf = idx_b + pos_b + nor_b

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_nk_plane.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "NORMAL": 2},
                                    "indices": 0, "material": 0}]}],
        "materials": [{
            "name": "nk_plane",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.0,
            },
            "doubleSided": True,
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-HALF, 0.0, -HALF], "max": [HALF, 0.0, HALF]},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx_b), "target": 34963},
            {"buffer": 0, "byteOffset": len(idx_b), "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": len(idx_b) + len(pos_b), "byteLength": len(nor_b),
             "target": 34962},
        ],
        "buffers": [{"uri": "nk_plane.bin", "byteLength": len(buf)}],
    }
    (OUT_DIR / "nk_plane.gltf").write_text(json.dumps(gltf, indent=2) + "\n")
    (OUT_DIR / "nk_plane.bin").write_bytes(buf)
    print(f"Generated {OUT_DIR / 'nk_plane.gltf'}")


if __name__ == "__main__":
    main()
