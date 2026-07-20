#!/usr/bin/env python3
"""Generate furnace-cavity glTF test scenes for IR energy conservation checks.

Outputs three .gltf variants with QUANTILOOM_material_ir extension:
  furnace_e1.gltf   — ε=1 (perfect emitter / absorber)
  furnace_e05.gltf  — ε=0.5
  furnace_rho1.gltf — ε=0, ρ=1 (perfect reflector, white furnace)

Each is a cube with INWARD-facing normals so the camera placed at the origin
sees only cavity walls.  T_surface set to 300K by default via
scene.default_temperature_k in the companion TOML files.
"""

import json, struct, base64, os, pathlib

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "models" / "furnace"

# Inside-out cube: 8 vertices, 12 triangles, normals pointing inward
HALF = 2.0
VERTS = [
    (-HALF, -HALF, -HALF), ( HALF, -HALF, -HALF),
    ( HALF,  HALF, -HALF), (-HALF,  HALF, -HALF),
    (-HALF, -HALF,  HALF), ( HALF, -HALF,  HALF),
    ( HALF,  HALF,  HALF), (-HALF,  HALF,  HALF),
]

# Faces with reversed winding for inward normals
FACES = [
    (0,2,1), (0,3,2),  # -Z face
    (4,5,6), (4,6,7),  # +Z face
    (0,1,5), (0,5,4),  # -Y face
    (2,3,7), (2,7,6),  # +Y face
    (0,4,7), (0,7,3),  # -X face
    (1,2,6), (1,6,5),  # +X face
]

INWARD_NORMALS = {
    0: ( 0, 0, 1), 1: ( 0, 0, 1),
    2: ( 0, 0,-1), 3: ( 0, 0,-1),
    4: ( 0, 1, 0), 5: ( 0, 1, 0),
    6: ( 0,-1, 0), 7: ( 0,-1, 0),
    8: ( 1, 0, 0), 9: ( 1, 0, 0),
   10: (-1, 0, 0),11: (-1, 0, 0),
}


def build_geometry():
    """Build flat-shaded inside-out cube: per-face vertices with inward normals."""
    positions = []
    normals = []
    indices = []
    for fi, (a, b, c) in enumerate(FACES):
        n = INWARD_NORMALS[fi]
        base = fi * 3
        for vi in (a, b, c):
            positions.extend(VERTS[vi])
            normals.extend(n)
        indices.extend([base, base+1, base+2])
    return positions, normals, indices


def pack_binary(positions, normals, indices):
    pos_bytes = struct.pack(f"<{len(positions)}f", *positions)
    nor_bytes = struct.pack(f"<{len(normals)}f", *normals)
    idx_bytes = struct.pack(f"<{len(indices)}H", *indices)
    # Pad index buffer to 4-byte alignment
    if len(idx_bytes) % 4:
        idx_bytes += b'\x00' * (4 - len(idx_bytes) % 4)
    return pos_bytes, nor_bytes, idx_bytes


def write_csv(path, value):
    """Write a trivial constant-value spectral CSV covering 300-15000nm."""
    with open(path, "w") as f:
        f.write("# wavelength_nm, value\n")
        f.write(f"300, {value}\n")
        f.write(f"15000, {value}\n")


def make_gltf(name, emissivity, out_dir):
    positions, normals, indices = build_geometry()
    pos_b, nor_b, idx_b = pack_binary(positions, normals, indices)
    buf = idx_b + pos_b + nor_b
    buf_uri = f"{name}.bin"

    n_verts = len(positions) // 3
    n_idx = len(indices)
    idx_len = len(idx_b)
    pos_off = idx_len
    pos_len = len(pos_b)
    nor_off = pos_off + pos_len
    nor_len = len(nor_b)

    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]

    emiss_csv = f"{name}_emissivity.csv"
    write_csv(out_dir / emiss_csv, emissivity)

    ext = {}
    if emissivity > 0:
        ext["emissivityCurve"] = emiss_csv
    ext["temperature_K"] = 300.0

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_furnace_cavity.py"},
        "extensionsUsed": ["QUANTILOOM_material_ir"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0, "material": 0}]}],
        "materials": [{
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": True,
            "extensions": {"QUANTILOOM_material_ir": ext}
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": n_idx,   "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": n_verts, "type": "VEC3",
             "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 2, "componentType": 5126, "count": n_verts, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,       "byteLength": idx_len, "target": 34963},
            {"buffer": 0, "byteOffset": pos_off,  "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": nor_off,  "byteLength": nor_len, "target": 34962},
        ],
        "buffers": [{"uri": buf_uri, "byteLength": len(buf)}],
    }

    with open(out_dir / f"{name}.gltf", "w") as f:
        json.dump(gltf, f, indent=2)
    with open(out_dir / buf_uri, "wb") as f:
        f.write(buf)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    make_gltf("furnace_e1",   1.0, OUT_DIR)
    make_gltf("furnace_e05",  0.5, OUT_DIR)
    make_gltf("furnace_rho1", 0.0, OUT_DIR)  # ε=0 → ρ=1

    print(f"Generated furnace cavities in {OUT_DIR}")


if __name__ == "__main__":
    main()
