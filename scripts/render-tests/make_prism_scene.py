#!/usr/bin/env python3
"""Generate a dispersion test scene: a glass prism above a diffuse floor.

There was no scene in the repository that disperses -- no config sets
`dispersion`, and no shipped model carries QUANTILOOM_materials_dispersion --
so the VIS_FUSED dispersion path had nothing to render and no way to be
checked. This writes the smallest asset that exercises it.

The prism is an equilateral triangular prism, apex up, so a ray entering one
slanted face and leaving the other is deviated the way a spectroscope prism
deviates it. BK7 is used because its Cauchy coefficient is tabulated and can be
checked by hand: n_d = 1.5168, V_d = 64.17, so dispersion = 1/V_d = 0.015583
and B should come out at 0.00420 um^2.

    python3 scripts/render-tests/make_prism_scene.py

Writes assets/models/prism_dispersion.gltf (self-contained, base64 buffer).
"""
import base64
import json
import math
import pathlib
import struct

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/models/prism_dispersion.gltf"

BK7_IOR = 1.5168
BK7_ABBE = 64.17

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# Equilateral triangular prism, apex up, extruded along Z. Side 1.2, so the
# apex angle is 60 degrees -- the classic dispersing prism.
SIDE = 1.2
H = SIDE * math.sqrt(3) / 2.0
Z0, Z1 = -0.6, 0.6
BASE_Y = 0.2  # lifted off the floor so refracted rays can reach it

tri = [
    (-SIDE / 2.0, BASE_Y, 0.0),   # 0 base left
    (SIDE / 2.0, BASE_Y, 0.0),    # 1 base right
    (0.0, BASE_Y + H, 0.0),       # 2 apex
]

positions = []
indices = []


def add_tri(a, b, c):
    base = len(positions)
    positions.extend([a, b, c])
    indices.extend([base, base + 1, base + 2])


def add_quad(a, b, c, d):
    add_tri(a, b, c)
    add_tri(a, c, d)


front = [(x, y, Z1) for (x, y, _) in tri]
back = [(x, y, Z0) for (x, y, _) in tri]

add_tri(front[0], front[1], front[2])          # front cap
add_tri(back[2], back[1], back[0])             # back cap (reversed winding)
add_quad(front[0], back[0], back[1], front[1])  # base
add_quad(front[1], back[1], back[2], front[2])  # right slanted face
add_quad(front[2], back[2], back[0], front[0])  # left slanted face

# A large diffuse floor to catch the fan of refracted light.
FLOOR = 8.0
floor_start = len(positions)
add_quad((-FLOOR, 0.0, -FLOOR), (FLOOR, 0.0, -FLOOR),
         (FLOOR, 0.0, FLOOR), (-FLOOR, 0.0, FLOOR))

# ---------------------------------------------------------------------------
# Buffer: positions then indices, both tightly packed
# ---------------------------------------------------------------------------
pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
while len(pos_bytes) % 4:
    pos_bytes += b"\0"
while len(idx_bytes) % 4:
    idx_bytes += b"\0"
blob = pos_bytes + idx_bytes

prism_index_count = floor_start  # indices before the floor's first vertex
assert prism_index_count == 24, f"expected 24 prism indices, got {prism_index_count}"

mins = [min(p[i] for p in positions) for i in range(3)]
maxs = [max(p[i] for p in positions) for i in range(3)]

gltf = {
    "asset": {
        "version": "2.0",
        "generator": "Quantiloom scripts/render-tests/make_prism_scene.py",
    },
    "extensionsUsed": [
        "KHR_materials_transmission",
        "KHR_materials_ior",
        "KHR_materials_dispersion",
    ],
    "scene": 0,
    "scenes": [{"nodes": [0, 1]}],
    "nodes": [
        {"mesh": 0, "name": "Prism"},
        {"mesh": 1, "name": "Floor"},
    ],
    "meshes": [
        {
            "name": "Prism",
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "material": 0,
            }],
        },
        {
            "name": "Floor",
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 2,
                "material": 1,
            }],
        },
    ],
    "materials": [
        {
            "name": "PrismGlass_BK7",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.0,
            },
            "extensions": {
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_ior": {"ior": BK7_IOR},
                # The ratified extension stores 20/V_d, not 1/V_d. GltfLoader
                # divides by 20 to reach Material::dispersion. Using the
                # standard spelling on purpose -- this asset should be readable
                # by any glTF viewer, and it also exercises that conversion.
                "KHR_materials_dispersion": {"dispersion": 20.0 / BK7_ABBE},
            },
        },
        {
            "name": "Floor_Diffuse",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        },
    ],
    "accessors": [
        {
            "bufferView": 0, "componentType": 5126, "count": len(positions),
            "type": "VEC3", "min": mins, "max": maxs,
        },
        {
            "bufferView": 1, "componentType": 5123,
            "byteOffset": 0, "count": prism_index_count, "type": "SCALAR",
        },
        {
            "bufferView": 1, "componentType": 5123,
            "byteOffset": prism_index_count * 2,
            "count": len(indices) - prism_index_count, "type": "SCALAR",
        },
    ],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
        {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(idx_bytes),
         "target": 34963},
    ],
    "buffers": [{
        "byteLength": len(blob),
        "uri": "data:application/octet-stream;base64," +
               base64.b64encode(blob).decode("ascii"),
    }],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(gltf, indent=1))
print(f"wrote {OUT}")
print(f"  {len(positions)} vertices, {len(indices)} indices "
      f"({prism_index_count} prism + {len(indices) - prism_index_count} floor)")
print(f"  BK7: ior={BK7_IOR}, KHR dispersion=20/{BK7_ABBE}={20.0/BK7_ABBE:.6f}"
      f"  -> internal 1/V_d = {1.0/BK7_ABBE:.6f}")


def n_of(lam):
    d = 1.0 / BK7_ABBE
    return BK7_IOR + (BK7_IOR - 1.0) * d * (523655.0 / (lam * lam) - 1.5168)


nF, nC = n_of(486.13), n_of(656.27)
print(f"  n(587.56) = {n_of(587.56):.6f} (= n_d), n_F = {nF:.6f}, n_C = {nC:.6f}")
print(f"  round-trip Abbe = {(BK7_IOR - 1.0) / (nF - nC):.2f} (catalogue {BK7_ABBE})")
