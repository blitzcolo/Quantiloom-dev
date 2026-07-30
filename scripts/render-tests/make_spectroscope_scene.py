#!/usr/bin/env python3
"""Generate a prism spectroscope: a bright slit, a prism, and a screen.

Why this exists rather than another camera position on prism_dispersion.toml:
a prism in front of a FEATURELESS background shows no dispersion however
dispersive the glass, because every wavelength lands on the same uniform
brightness. What a spectroscope adds is a compact bright source, so the
deviation has something with an edge to spread.

The geometry is solved, not guessed. Two earlier attempts failed for reasons
worth recording:

  - The prism is extruded along Z, so its slanted faces have normals in the XY
    plane and ALL deviation happens there. A camera looking along -Z enters one
    triangular cap and leaves the other -- two parallel surfaces, net deviation
    zero, a glass slab. No dispersion, correctly.
  - Looking along -X fixes that, but at minimum deviation the view is bent 66
    degrees, which from eye level points at either bare sky or bare floor. Still
    nothing to disperse.

So this script traces the minimum-deviation path through the prism with Snell's
law and places the slit where that path actually goes. At minimum deviation the
internal ray runs parallel to the prism's base, which is the symmetric case and
the one where a spectroscope is sharpest.

The prism is also RAISED off the floor: minimum deviation needs a steep
incidence, and with the prism sitting on the ground the camera works out to be
underground.

    python3 scripts/render-tests/make_spectroscope_scene.py --glass SF11

Prints the camera position and look_at to put in the config, and writes a
self-contained glTF.
"""
import argparse
import base64
import json
import math
import pathlib
import struct

GLASSES = {
    "BK7":  (1.5168, 64.17),
    "SF11": (1.7847, 25.76),
    "SF57": (1.8467, 23.83),
}

ap = argparse.ArgumentParser()
ap.add_argument("--glass", choices=sorted(GLASSES), default="SF11")
ap.add_argument("--side", type=float, default=1.6)
ap.add_argument("--base-y", type=float, default=2.2, help="prism base height")
ap.add_argument("--slit-distance", type=float, default=6.0)
ap.add_argument("--slit-half-width", type=float, default=0.09)
ap.add_argument("--emissive", type=float, default=60.0)
ap.add_argument("--out", default="assets/models/prism_spectroscope.gltf")
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[2]
N_D, ABBE = GLASSES[args.glass]
APEX_DEG = 60.0


def n_of(lam):
    """KHR_materials_dispersion's formula, lambda in nm."""
    return N_D + (N_D - 1.0) / ABBE * (523655.0 / (lam * lam) - 1.5168)


def norm(v):
    m = math.hypot(v[0], v[1])
    return (v[0] / m, v[1] / m)


def refract(d, n, eta):
    """d incident (unit), n outward normal (unit), eta = n_from / n_to. 2D."""
    c1 = -(d[0] * n[0] + d[1] * n[1])
    flip = 1.0
    if c1 < 0.0:                      # ray leaving: normal points the wrong way
        n = (-n[0], -n[1]); c1 = -c1; flip = -1.0
    k = 1.0 - eta * eta * (1.0 - c1 * c1)
    if k < 0.0:
        return None                   # total internal reflection
    c2 = math.sqrt(k)
    t = (eta * d[0] + (eta * c1 - c2) * n[0],
         eta * d[1] + (eta * c1 - c2) * n[1])
    del flip
    return norm(t)


# --- prism cross-section, apex up -------------------------------------------
SIDE, BASE_Y = args.side, args.base_y
H = SIDE * math.sqrt(3) / 2.0
P_L = (-SIDE / 2.0, BASE_Y)          # base left
P_R = (SIDE / 2.0, BASE_Y)           # base right
P_A = (0.0, BASE_Y + H)              # apex
N_RIGHT = norm((math.sin(math.radians(60.0)), 0.5))    # outward, right slanted face
N_LEFT = (-N_RIGHT[0], N_RIGHT[1])                     # outward, left slanted face

# --- minimum deviation: incidence such that the internal ray is symmetric ----
n_d = n_of(587.56)
theta_i = math.asin(n_d * math.sin(math.radians(APEX_DEG) / 2.0))
# The view direction that meets N_RIGHT at theta_i. Solving
# cos(theta - 30deg) = -cos(theta_i) gives two roots; take the one travelling -x.
theta = math.radians(30.0) + math.acos(-math.cos(theta_i))
view = norm((math.cos(theta), math.sin(theta)))
if view[0] > 0:
    theta = math.radians(30.0) - math.acos(-math.cos(theta_i))
    view = norm((math.cos(theta), math.sin(theta)))

entry = ((P_R[0] + P_A[0]) / 2.0, (P_R[1] + P_A[1]) / 2.0)   # right face midpoint
inside = refract(view, N_RIGHT, 1.0 / n_d)
assert inside is not None

# Where the internal ray meets the left face, then out.
# Left face param: P_A + s*(P_L - P_A), s in [0,1]. Solve entry + u*inside = that.
ex, ey = P_L[0] - P_A[0], P_L[1] - P_A[1]
den = inside[0] * ey - inside[1] * ex
s = ((entry[0] - P_A[0]) * inside[1] - (entry[1] - P_A[1]) * inside[0]) / (-den) \
    if abs(den) > 1e-12 else 0.5
exit_pt = (P_A[0] + s * ex, P_A[1] + s * ey)
outward = refract(inside, N_LEFT, n_d)
assert outward is not None, "total internal reflection at the exit face"

deviation = math.degrees(math.acos(max(-1.0, min(1.0, view[0] * outward[0] + view[1] * outward[1]))))


def deviation_at(lam):
    a = math.radians(APEX_DEG)
    return 2.0 * math.asin(n_of(lam) * math.sin(a / 2.0)) - a


spread = math.degrees(deviation_at(400.0) - deviation_at(780.0))

# The slit sits along the exit ray, far enough that the fan has separated.
slit_c = (exit_pt[0] + outward[0] * args.slit_distance,
          exit_pt[1] + outward[1] * args.slit_distance)
# Camera sits back along the view direction from the entry point.
cam = (entry[0] - view[0] * 3.2, entry[1] - view[1] * 3.2)

print(f"{args.glass}: n_d={N_D}, Abbe={ABBE}, apex={APEX_DEG:.0f} deg")
print(f"  minimum-deviation incidence   {math.degrees(theta_i):6.2f} deg")
print(f"  deviation (traced)            {deviation:6.2f} deg")
print(f"  400-780 nm angular separation {spread:6.2f} deg")
print(f"  view dir  ({view[0]:+.4f}, {view[1]:+.4f})   internal ({inside[0]:+.4f}, {inside[1]:+.4f})"
      f"   exit ({outward[0]:+.4f}, {outward[1]:+.4f})")
print(f"  internal ray parallel to base? |dy| = {abs(inside[1]):.4f}  (0 = symmetric)")
print()
print(f"  put in the config:")
print(f"    position = [{cam[0]:.3f}, {cam[1]:.3f}, 0.0]")
print(f"    look_at  = [{entry[0]:.3f}, {entry[1]:.3f}, 0.0]")

# --- build the mesh ---------------------------------------------------------
positions, indices = [], []


def add_tri(a, b, c):
    i = len(positions)
    positions.extend([a, b, c])
    indices.extend([i, i + 1, i + 2])


def add_quad(a, b, c, d):
    add_tri(a, b, c)
    add_tri(a, c, d)


Z0, Z1 = -SIDE / 2.0, SIDE / 2.0
tri = [P_L, P_R, P_A]
front = [(x, y, Z1) for (x, y) in tri]
back = [(x, y, Z0) for (x, y) in tri]
add_tri(*front)
add_tri(back[2], back[1], back[0])
add_quad(front[0], back[0], back[1], front[1])   # base
add_quad(front[1], back[1], back[2], front[2])   # right slanted
add_quad(front[2], back[2], back[0], front[0])   # left slanted
prism_idx = len(indices)

# Slit: a narrow quad facing the prism, tall in Z so it reads as a line source.
w = args.slit_half_width
perp = (-outward[1], outward[0])                 # in-plane, across the ray
sa = (slit_c[0] - perp[0] * w, slit_c[1] - perp[1] * w)
sb = (slit_c[0] + perp[0] * w, slit_c[1] + perp[1] * w)
SLIT_H = 1.4
slit_start = len(positions)
add_quad((sa[0], sa[1], -SLIT_H), (sb[0], sb[1], -SLIT_H),
         (sb[0], sb[1], SLIT_H), (sa[0], sa[1], SLIT_H))
slit_idx = len(indices) - prism_idx

gltf = {
    "asset": {"version": "2.0", "generator": "make_spectroscope_scene.py"},
    "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior",
                       "KHR_materials_dispersion", "KHR_materials_emissive_strength"],
    "scene": 0,
    "scenes": [{"nodes": [0, 1]}],
    "nodes": [{"mesh": 0, "name": "Prism"}, {"mesh": 1, "name": "Slit"}],
    "meshes": [
        {"name": "Prism", "primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]},
        {"name": "Slit", "primitives": [
            {"attributes": {"POSITION": 0}, "indices": 2, "material": 1}]},
    ],
    "materials": [
        {
            "name": "PrismGlass_BK7",   # name kept stable: configs reference it
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.0},
            "extensions": {
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_ior": {"ior": N_D},
                "KHR_materials_dispersion": {"dispersion": 20.0 / ABBE},
            },
        },
        {
            "name": "Slit_Emissive",
            "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1],
                                     "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "emissiveFactor": [1.0, 1.0, 1.0],
            "extensions": {
                "KHR_materials_emissive_strength": {"emissiveStrength": args.emissive},
            },
        },
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
         "min": [min(p[i] for p in positions) for i in range(3)],
         "max": [max(p[i] for p in positions) for i in range(3)]},
        {"bufferView": 1, "componentType": 5123, "byteOffset": 0,
         "count": prism_idx, "type": "SCALAR"},
        {"bufferView": 1, "componentType": 5123, "byteOffset": prism_idx * 2,
         "count": slit_idx, "type": "SCALAR"},
    ],
    "bufferViews": [],
    "buffers": [],
}

pos_b = b"".join(struct.pack("<3f", *p) for p in positions)
idx_b = b"".join(struct.pack("<H", i) for i in indices)
while len(pos_b) % 4:
    pos_b += b"\0"
while len(idx_b) % 4:
    idx_b += b"\0"
blob = pos_b + idx_b
gltf["bufferViews"] = [
    {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
    {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(idx_b), "target": 34963},
]
gltf["buffers"] = [{"byteLength": len(blob),
                    "uri": "data:application/octet-stream;base64," +
                           base64.b64encode(blob).decode("ascii")}]

out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(gltf, indent=1))
print(f"\nwrote {out.relative_to(ROOT)}  ({len(positions)} verts)")
print(f"  slit centre ({slit_c[0]:.3f}, {slit_c[1]:.3f}), half-width {w}, "
      f"emissive strength {args.emissive}")
del slit_start
