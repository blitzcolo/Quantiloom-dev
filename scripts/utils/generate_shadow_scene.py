#!/usr/bin/env python3
"""Generate the sun-occlusion test scene: one ground plane, one floating slab.

The scene exists to make sun visibility the ONLY thing that varies across the
frame.  Both quads use the same material, both face +Y, and the camera is
orthographic looking straight down, so every visible pixel has the same
material, the same normal and the same view angle.  A renderer that ignores
the shadow ray therefore produces a perfectly flat image; one that honours it
produces exactly two levels.

    ground   y=0,  x in [-8, 8], z in [-8, 8]
    occluder y=4,  x in [-2, 2], z in [-8, 8]

With the sun at normalize(0.6, 0.8, 0) a ground point (x, 0, z) reaches y=4
after t = 5, at x + 3.  So the ground is shadowed for x + 3 in [-2, 2], i.e.
x in [-5, -1]; the slab itself hides x in [-2, 2], leaving a visible shadow
band at x in [-5, -2).  None of that geometry is needed by the checker -- it
compares against a second render with the sun below the horizon -- but it is
what sizes the band at roughly 3/16 of the frame.

Emissivity 0.3 (so rho = 0.7) keeps the reflected-solar term large enough to
measure in MWIR, where self-emission would otherwise dominate.
"""

import json, struct, pathlib

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "models" / "shadow_scene"

EMISSIVITY = 0.3

GROUND_HALF = 8.0
SLAB_HALF_X = 2.0
SLAB_Y = 4.0


def quad(x0, x1, y, z0, z1):
    """Two triangles spanning [x0,x1] x [z0,z1] at height y, normal +Y.

    Winding is counter-clockwise seen from above: (v1-v0) x (v2-v0) = +Y.
    """
    a = (x0, y, z0)
    b = (x0, y, z1)
    c = (x1, y, z1)
    d = (x1, y, z0)
    return [a, b, c, a, c, d]


def build_geometry(with_slab=True):
    verts = quad(-GROUND_HALF, GROUND_HALF, 0.0, -GROUND_HALF, GROUND_HALF)
    if with_slab:
        verts += quad(-SLAB_HALF_X, SLAB_HALF_X, SLAB_Y, -GROUND_HALF, GROUND_HALF)

    positions = [c for v in verts for c in v]
    normals = [c for _ in verts for c in (0.0, 1.0, 0.0)]
    indices = list(range(len(verts)))
    return positions, normals, indices


def write_csv(path, value):
    """Constant-value spectral CSV covering every band the renderer integrates."""
    with open(path, "w") as f:
        f.write("# wavelength_nm, value\n")
        f.write(f"300, {value}\n")
        f.write(f"15000, {value}\n")


def write_scene(name, with_slab):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positions, normals, indices = build_geometry(with_slab)

    pos_b = struct.pack(f"<{len(positions)}f", *positions)
    nor_b = struct.pack(f"<{len(normals)}f", *normals)
    idx_b = struct.pack(f"<{len(indices)}H", *indices)
    if len(idx_b) % 4:
        idx_b += b"\x00" * (4 - len(idx_b) % 4)

    buf = idx_b + pos_b + nor_b
    pos_off = len(idx_b)
    nor_off = pos_off + len(pos_b)

    xs, ys, zs = positions[0::3], positions[1::3], positions[2::3]

    emiss_csv = f"{name}_emissivity.csv"
    write_csv(OUT_DIR / emiss_csv, EMISSIVITY)

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_shadow_scene.py"},
        "extensionsUsed": ["QUANTILOOM_material_ir"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [
            {"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0, "material": 0}
        ]}],
        "materials": [{
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
            "extensions": {"QUANTILOOM_material_ir": {
                "emissivityCurve": emiss_csv,
                "temperature_K": 300.0,
            }},
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(positions) // 3, "type": "VEC3",
             "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 2, "componentType": 5126, "count": len(normals) // 3, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,       "byteLength": len(idx_b), "target": 34963},
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": nor_off, "byteLength": len(nor_b), "target": 34962},
        ],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(buf)}],
    }

    with open(OUT_DIR / f"{name}.gltf", "w") as f:
        json.dump(gltf, f, indent=2)
    with open(OUT_DIR / f"{name}.bin", "wb") as f:
        f.write(buf)


def main():
    write_scene("shadow_scene", with_slab=True)
    # The same ground with nothing above it. Every bounce ray escapes, so the
    # traced correction is identically zero and the render must equal the
    # analytic Lambertian answer rho*(E_sun/pi*cos + E_sky/pi) exactly -- which
    # is the check that the bounce was added as a correction to the analytic sky
    # term and not on top of it.
    write_scene("shadow_scene_open", with_slab=False)
    print(f"Generated shadow scenes in {OUT_DIR}")


if __name__ == "__main__":
    main()
