#!/usr/bin/env python3
"""Generate the endmember-mixing test scene: a plane, two-colour checkerboard.

The scene exists to make endmember unmixing checkable. Its base-colour texture
is painted with exactly the D65 colours of the two endmembers it will be bound
to, in large blocks, so every texel is unambiguously one material or the other
and the unmixer's right answer is known in advance: weight (1, 0) on one block
and (0, 1) on the other.

That makes the render checkable without modelling the illumination at all --
render the same scene bound to each curve alone, and the mixed render must
reproduce each single-curve result inside the matching block. See
scripts/render-tests/check_endmember_mix.py.

Usage:
    python scripts/utils/generate_endmember_checker.py [--out PATH]
"""

import argparse
import json
import struct
import pathlib
import zlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
BASIS = REPO / "assets/spectral/quantiloom_basis_v3_usgs.qlbin"
MATERIALS = REPO / "assets/spectral/quantiloom_materials_usgs.json"
CMF = REPO / "assets/luts/CIE_xyz_1931_2deg.csv"
D65 = REPO / "assets/luts/CIE_std_illum_D65.csv"

# Two materials whose spectra differ where it matters and whose colours differ
# enough for the unmix to be unambiguous: living cordgrass and dry golden grass.
ENDMEMBERS = ["Spar.patens CRMS322v06 grn.a", "Grass_Golden_Dry GDS480"]

BLOCKS = 8  # checker blocks per side
RES = 512

XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])


def load_basis(path):
    """Mirrors SpectralBasisLoader: QBAS v3, per-band header then float32 basis."""
    raw = path.read_bytes()
    assert raw[:4] == b"QBAS", "not a QBAS file"
    num_bands = struct.unpack_from("<I", raw, 4)[0]
    pos = 64
    bands = {}
    for name in ["VIS", "NIR", "SWIR", "MWIR", "LWIR"][:num_bands]:
        start_um, end_um, n_samp, n_basis = struct.unpack_from("<ffII", raw, pos)
        pos += 16
        data = np.frombuffer(raw, dtype="<f4", count=n_basis * n_samp, offset=pos)
        pos += n_basis * n_samp * 4
        bands[name] = {
            "lambda": np.linspace(start_um, end_um, n_samp) * 1000.0,
            "basis": data.reshape(n_basis, n_samp),
        }
    return bands


def reflectance(bands, materials, name, band="VIS"):
    entry = materials["materials"][name]["bands"][band]
    w = np.array(entry["basis_weights"], dtype=np.float64)
    return bands[band]["lambda"], np.clip(w @ bands[band]["basis"], 0.0, 1.0)


def linear_srgb_d65(lam, refl):
    """The same integral as SpectralColour.cpp: D65-weighted, white-normalised."""
    cmf = np.loadtxt(CMF, delimiter=",")
    d65 = np.loadtxt(D65, delimiter=",", skiprows=1)

    grid = np.arange(380.0, 781.0, 1.0)
    r = np.interp(grid, lam, refl, left=refl[0], right=refl[-1])
    xyz_bar = np.stack([np.interp(grid, cmf[:, 0], cmf[:, i]) for i in (1, 2, 3)], axis=1)
    e = np.interp(grid, d65[:, 0], d65[:, 1], left=0.0, right=0.0)

    xyz = np.trapezoid((r * e)[:, None] * xyz_bar, grid, axis=0)
    xyz /= np.trapezoid(e * xyz_bar[:, 1], grid)
    return np.clip(XYZ_TO_RGB @ xyz, 0.0, 1.0)


def to_srgb_bytes(linear):
    s = np.where(linear <= 0.0031308, linear * 12.92,
                 1.055 * np.power(np.clip(linear, 0, None), 1 / 2.4) - 0.055)
    return np.clip(s * 255.0 + 0.5, 0, 255).astype(np.uint8)


def write_png(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def build_glb(png_bytes):
    """A 20x20 m plane at y=0, one material, the checkerboard as base colour."""
    size = 10.0
    pos = np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]],
                   dtype="<f4")
    nrm = np.tile(np.array([0, 1, 0], dtype="<f4"), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4")
    idx = np.array([0, 1, 2, 0, 2, 3], dtype="<u4")

    blob = bytearray()
    views, accessors = [], []

    def add(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
                     | ({"target": target} if target else {}))
        blob.extend(data)
        return len(views) - 1

    def acc(arr, typ, comp, target, minmax=False):
        v = add(arr.tobytes(), target)
        a = {"bufferView": v, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"] = arr.min(axis=0).tolist()
            a["max"] = arr.max(axis=0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    a_pos = acc(pos, "VEC3", 5126, 34962, True)
    a_nrm = acc(nrm, "VEC3", 5126, 34962)
    a_uv = acc(uv, "VEC2", 5126, 34962)
    a_idx = acc(idx, "SCALAR", 5125, 34963)
    v_img = add(png_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_endmember_checker.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "CheckerPlane"}],
        "meshes": [{"name": "CheckerPlane", "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "material": 0}]}],
        "materials": [{
            "name": "CheckerGround",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
        }],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/png"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    bin_ = bytes(blob) + b"\x00" * (-len(blob) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_))
            + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "assets/models/endmember_checker.glb"))
    args = ap.parse_args()

    bands = load_basis(BASIS)
    materials = json.loads(MATERIALS.read_text())

    colors = []
    for name in ENDMEMBERS:
        lam, refl = reflectance(bands, materials, name)
        lin = linear_srgb_d65(lam, refl)
        colors.append(to_srgb_bytes(lin))
        print(f"{name:40s} linear sRGB = {lin[0]:.4f} {lin[1]:.4f} {lin[2]:.4f}"
              f"  bytes = {tuple(colors[-1])}")

    # Nearest-neighbour sampling and hard block edges: a filtered edge texel
    # would be a genuine mixture of the two, which is correct behaviour but
    # makes the block interiors the only place to assert on.
    step = RES // BLOCKS
    img = np.zeros((RES, RES, 3), dtype=np.uint8)
    for by in range(BLOCKS):
        for bx in range(BLOCKS):
            img[by * step:(by + 1) * step, bx * step:(bx + 1) * step] = colors[(bx + by) % 2]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_glb(write_png(img)))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
