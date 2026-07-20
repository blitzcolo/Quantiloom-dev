#!/usr/bin/env python3
"""Generate a Cornell Box glTF with ECOSTRESS spectral material references.

Original Cornell University specification (mm, Y-up):
  Room: ~555 x 548.8 x 559.2 mm (open front at z=0)
  Short block: 165³, rotated -18° Y, at (130, 0, 65)  — LEFT side
  Tall  block: 165x330x165, rotated +22.5° Y, at (265, 0, 295) — RIGHT side
  Light: 130x105 patch on ceiling, facing DOWN
  Camera: (278, 273, -800), FOV ~39°

All room-wall normals point INWARD.  Block normals point OUTWARD.
Light panel is a separate emissive material (white, emissiveFactor > 0).
"""

import json, struct, math, pathlib

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "models" / "cornell_box"
ECO_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "spectral" / "ecospeclib-all"

MATERIALS = {
    "concrete": "manmade.concrete.constructionconcrete.solid.all.0598uuucnc.jhu.becknic.spectrum.txt",
    "brick":    "manmade.generalconstructionmaterial.brick.solid.all.0097uuubrk.jhu.becknic.spectrum.txt",
    "paint":    "manmade.generalconstructionmaterial.paint.solid.all.0385uuupnt.jhu.becknic.spectrum.txt",
    "marble":   "manmade.generalconstructionmaterial.marble.solid.all.0722uuumbl.jhu.becknic.spectrum.txt",
    "asphalt":  "manmade.road.pavingasphalt.solid.all.0095uuuasp.jhu.becknic.spectrum.txt",
}

MAT_NAMES = {
    "concrete": "Construction Concrete",
    "brick":    "Red Brick",
    "paint":    "Olive Green Paint",
    "marble":   "White Marble",
    "asphalt":  "Asphalt",
    "light":    "Light Panel",
}

PBR_COLORS = {
    "concrete": [0.65, 0.65, 0.65, 1.0],
    "brick":    [0.65, 0.15, 0.10, 1.0],
    "paint":    [0.25, 0.40, 0.15, 1.0],
    "marble":   [0.85, 0.85, 0.82, 1.0],
    "asphalt":  [0.15, 0.15, 0.15, 1.0],
    "light":    [1.00, 1.00, 1.00, 1.0],
}

# ---- Geometry helpers ----

def quad_tris(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]

def compute_normal(v0, v1, v2):
    e1 = [v1[i]-v0[i] for i in range(3)]
    e2 = [v2[i]-v0[i] for i in range(3)]
    n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]
    l = math.sqrt(sum(x*x for x in n))
    return [x/l for x in n] if l > 1e-9 else [0, 1, 0]

def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]

def vadd(v, t):
    return [v[i]+t[i] for i in range(3)]


def make_quad(v0, v1, v2, v3, expected_normal=None):
    """Create a quad with verified normal direction.
    If expected_normal is given, flip winding if the computed normal
    points in the wrong half-space."""
    tris = quad_tris(v0, v1, v2, v3)
    n = compute_normal(*tris[0])
    if expected_normal:
        dot = sum(n[i]*expected_normal[i] for i in range(3))
        if dot < 0:
            tris = quad_tris(v3, v2, v1, v0)
            n = compute_normal(*tris[0])
    return (v0, v1, v2, v3), n, tris


def box_faces_outward(sx, sy, sz, rot=0.0, tx=0, ty=0, tz=0, skip_bottom=True):
    """Generate box faces with OUTWARD normals (for solid objects inside the room)."""
    c = [
        [0,0,0],[sx,0,0],[sx,sy,0],[0,sy,0],
        [0,0,sz],[sx,0,sz],[sx,sy,sz],[0,sy,sz],
    ]
    if rot != 0 or tx or ty or tz:
        c = [vadd(rot_y(v, rot), [tx, ty, tz]) for v in c]

    # Outward-facing quads (CCW when viewed from outside)
    faces = [
        (c[0], c[3], c[2], c[1]),  # -Z face, normal -Z
        (c[4], c[5], c[6], c[7]),  # +Z face, normal +Z
        (c[2], c[3], c[7], c[6]),  # +Y top, normal +Y
        (c[0], c[4], c[7], c[3]),  # -X face, normal -X
        (c[1], c[2], c[6], c[5]),  # +X face, normal +X
    ]
    if not skip_bottom:
        faces.append((c[0], c[1], c[5], c[4]))  # -Y bottom
    return faces


def build_tris(quads):
    positions, normals, indices = [], [], []
    idx = 0
    for q in quads:
        tris = quad_tris(*q)
        n = compute_normal(*tris[0])
        for tri in tris:
            for v in tri:
                positions.extend(v)
                normals.extend(n)
                indices.append(idx); idx += 1
    return positions, normals, indices


# ---- Cornell Box geometry (mm, Y-up) ----
# All room walls: normals point INWARD (toward box center ~(278, 274, 280))
# Verified with expected_normal parameter.

CENTER = [278, 274, 280]

def inward(wall_pos):
    """Approximate inward direction from wall center to box center."""
    return [CENTER[i] - wall_pos[i] for i in range(3)]

# Floor: y=0, normal should point +Y (up, inward)
FLOOR = [(
    [0, 0, 0], [0, 0, 559.2], [552.8, 0, 559.2], [552.8, 0, 0]
)]

# Ceiling: y=548.8, normal should point -Y (down, inward)
CEILING = [(
    [0, 548.8, 559.2], [0, 548.8, 0], [556, 548.8, 0], [556, 548.8, 559.2]
)]

# Back wall: z=559.2, normal should point -Z (toward camera, inward)
BACK = [(
    [0, 0, 559.2], [0, 548.8, 559.2], [556, 548.8, 559.2], [552.8, 0, 559.2]
)]

# Left wall: x≈555, normal should point -X (inward)
LEFT = [(
    [552.8, 0, 559.2], [556, 548.8, 559.2], [556, 548.8, 0], [552.8, 0, 0]
)]

# Right wall: x=0, normal should point +X (inward)
RIGHT = [(
    [0, 0, 0], [0, 548.8, 0], [0, 548.8, 559.2], [0, 0, 559.2]
)]

# Light panel: on ceiling (y≈548.7), facing DOWN (-Y)
LIGHT = [(
    [213, 548.7, 332], [213, 548.7, 227], [343, 548.7, 227], [343, 548.7, 332]
)]

# Blocks: outward normals (solid objects)
SHORT_BLOCK = box_faces_outward(165, 165, 165, rot=-0.314, tx=130, ty=0, tz=65)
TALL_BLOCK  = box_faces_outward(165, 330, 165, rot=0.3925, tx=265, ty=0, tz=295)

# Material groups — light panel is a SEPARATE material with emissive
GROUPS = [
    ("concrete", FLOOR + CEILING + BACK),
    ("light",    LIGHT),
    ("brick",    LEFT),
    ("paint",    RIGHT),
    ("marble",   SHORT_BLOCK),
    ("asphalt",  TALL_BLOCK),
]


def verify_normals():
    """Print normals for verification."""
    expected = {
        "Floor": [0, 1, 0], "Ceiling": [0, -1, 0], "Back": [0, 0, -1],
        "Left": [-1, 0, 0], "Right": [1, 0, 0], "Light": [0, -1, 0],
    }
    surfaces = {"Floor": FLOOR, "Ceiling": CEILING, "Back": BACK,
                "Left": LEFT, "Right": RIGHT, "Light": LIGHT}
    ok = True
    for name, quads in surfaces.items():
        tris = quad_tris(*quads[0])
        n = compute_normal(*tris[0])
        exp = expected[name]
        dot = sum(n[i]*exp[i] for i in range(3))
        status = "OK" if dot > 0.9 else "WRONG"
        if status == "WRONG": ok = False
        print(f"  {name:8s}: n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}) expected=({exp[0]:+d},{exp[1]:+d},{exp[2]:+d}) {status}")
    return ok


def ecostress_to_csv(eco_filename, out_path):
    src = ECO_DIR / eco_filename
    text = src.read_text(encoding="utf-8", errors="replace")
    y_units = ""
    rows = []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if s.lower().startswith("y units"): y_units = s.lower()
        if not in_data:
            parts = s.split()
            if len(parts) >= 2:
                try: float(parts[0]); float(parts[1]); in_data = True
                except ValueError: continue
        if in_data:
            parts = s.split()
            if len(parts) >= 2:
                try: rows.append((float(parts[0]), float(parts[1])))
                except ValueError: continue

    is_pct = "percent" in y_units or "percentage" in y_units
    rows.sort()
    with open(out_path, "w") as f:
        f.write("# wavelength_nm, reflectance\n")
        for wl_um, val in rows:
            r = val / 100.0 if is_pct else val
            f.write(f"{wl_um * 1000:.1f}, {max(0, min(1, r)):.6f}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying normals:")
    if not verify_normals():
        print("  FATAL: normal verification failed!")
        return

    csv_map = {}
    for key, eco_file in MATERIALS.items():
        csv_name = f"cornell_{key}_reflectance.csv"
        ecostress_to_csv(eco_file, OUT_DIR / csv_name)
        csv_map[key] = csv_name

    mat_keys = list(dict.fromkeys(k for k, _ in GROUPS))
    mat_idx = {k: i for i, k in enumerate(mat_keys)}

    all_buf = b""
    accessors, buffer_views, meshes = [], [], []
    offset = 0

    for gi, (mat_key, quads) in enumerate(GROUPS):
        pos, nor, idx = build_tris(quads)
        nv, ni = len(pos) // 3, len(idx)

        ib = struct.pack(f"<{ni}I", *idx)
        pb = struct.pack(f"<{len(pos)}f", *pos)
        nb = struct.pack(f"<{len(nor)}f", *nor)

        bv_base, ac_base = len(buffer_views), len(accessors)

        buffer_views.append({"buffer":0,"byteOffset":offset,"byteLength":len(ib),"target":34963})
        offset += len(ib)
        buffer_views.append({"buffer":0,"byteOffset":offset,"byteLength":len(pb),"target":34962,"byteStride":12})
        offset += len(pb)
        buffer_views.append({"buffer":0,"byteOffset":offset,"byteLength":len(nb),"target":34962,"byteStride":12})
        offset += len(nb)

        xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
        accessors.append({"bufferView":bv_base,"componentType":5125,"count":ni,"type":"SCALAR"})
        accessors.append({"bufferView":bv_base+1,"componentType":5126,"count":nv,"type":"VEC3",
                          "min":[min(xs),min(ys),min(zs)],"max":[max(xs),max(ys),max(zs)]})
        accessors.append({"bufferView":bv_base+2,"componentType":5126,"count":nv,"type":"VEC3"})

        meshes.append({"primitives":[{
            "attributes":{"POSITION":ac_base+1,"NORMAL":ac_base+2},
            "indices":ac_base,"material":mat_idx[mat_key]
        }]})
        all_buf += ib + pb + nb

    # Build glTF materials
    gltf_mats = []
    for key in mat_keys:
        mat = {
            "name": MAT_NAMES[key],
            "pbrMetallicRoughness": {
                "baseColorFactor": PBR_COLORS[key],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": True,
        }
        if key == "light":
            # Emissive ceiling light — VIS brightness ~15 W/sr/m²
            mat["emissiveFactor"] = [15.0, 15.0, 12.0]
        if key in csv_map:
            mat["extensions"] = {"QUANTILOOM_material_ir": {"reflectanceCurve": csv_map[key]}}
        gltf_mats.append(mat)

    ext_used = ["QUANTILOOM_material_ir"]
    gltf = {
        "asset": {"version": "2.0", "generator": "generate_cornell_box.py"},
        "extensionsUsed": ext_used,
        "scene": 0,
        "scenes": [{"nodes": list(range(len(meshes))), "name": "Cornell Box IR"}],
        "nodes": [{"mesh": i} for i in range(len(meshes))],
        "meshes": meshes,
        "materials": gltf_mats,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"uri": "cornell_box.bin", "byteLength": len(all_buf)}],
    }

    with open(OUT_DIR / "cornell_box.gltf", "w") as f:
        json.dump(gltf, f, indent=2)
    with open(OUT_DIR / "cornell_box.bin", "wb") as f:
        f.write(all_buf)

    print(f"\nCornell Box: {OUT_DIR / 'cornell_box.gltf'}")
    print(f"  {len(gltf_mats)} materials, {len(meshes)} meshes, {len(all_buf)} bytes")


if __name__ == "__main__":
    main()
