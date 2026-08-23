#!/usr/bin/env python3
"""Port the Cornell box geometry to PLY files Mitsuba can load.

The geometry is taken from `scripts/utils/generate_cornell_box.py` by importing
it, not by re-deriving it. That matters more than it looks: the original Cornell
box data is not axis-aligned -- the back and left walls are a few millimetres
out of square, which is a property of the measured room the dataset came from
and not an error to be tidied. Rebuilding the box from idealised planes would
give Mitsuba a *different room*, and the difference would then be reported as a
transport disagreement.

Units are converted here, once. Quantiloom reads the glTF in millimetres and
scales by `world_units_to_meters = 0.001`; Mitsuba has no such key, so the PLYs
are written in metres.

Normals are per-vertex and flat -- each triangle's vertices carry that
triangle's own normal, exactly as the glTF does, because the generator
duplicates vertices per face rather than sharing them.

Usage:
    export_cornell_mitsuba.py --out-dir build/validation/cornell_m3
"""

import argparse
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))
import mitsuba_common as common  # noqa: E402

import generate_cornell_box as cornell  # noqa: E402

REPO = common.REPO
MM_TO_M = 0.001

# Which measured curve each group's material carries. The light panel has none:
# its base colour is white and its role is to emit, so it is given a flat
# reflectance of 1.0 -- which is what the glTF's baseColorFactor of exactly 1.0
# means through Quantiloom's achromatic path.
REFLECTANCE_CSV = {
    "concrete": "cornell_concrete_reflectance.csv",
    "brick": "cornell_brick_reflectance.csv",
    "paint": "cornell_paint_reflectance.csv",
    "marble": "cornell_marble_reflectance.csv",
    "asphalt": "cornell_asphalt_reflectance.csv",
}


def write_ply(path, positions, normals, indices):
    """A little-endian binary PLY with per-vertex normals."""
    vertex_count = len(positions) // 3
    face_count = len(indices) // 3

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment exported by {pathlib.Path(__file__).name} from "
        f"generate_cornell_box.py, millimetres scaled to metres\n"
        f"element vertex {vertex_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    body = bytearray()
    for v in range(vertex_count):
        body += struct.pack(
            "<6f",
            positions[3 * v] * MM_TO_M,
            positions[3 * v + 1] * MM_TO_M,
            positions[3 * v + 2] * MM_TO_M,
            normals[3 * v], normals[3 * v + 1], normals[3 * v + 2])
    for f in range(face_count):
        body += struct.pack("<B3i", 3, indices[3 * f], indices[3 * f + 1],
                            indices[3 * f + 2])

    path.write_bytes(header + bytes(body))
    return vertex_count, face_count


def trim_to_visible(source, destination, lo=380.0, hi=780.0):
    """Write a copy of a reflectance CSV covering only the visible band.

    Not a convenience. Quantiloom uploads a spectral curve as 64 values on a
    UNIFORM grid spanning the curve's whole range, so a CSV measured from 300 nm
    to 15 µm arrives at the shader with roughly 200 nm between samples --
    everywhere, including the visible band, where these materials have
    structure at 10 nm scales. Measured at 550 nm against the file's own
    resolution, that costs 3.3 % on concrete, 10 % on brick, and 41.6 % on the
    olive green paint, whose green peak is the entire point of a
    colour-bleeding scene.

    Trimmed to 380-780 nm the same 64 samples are 6.35 nm apart and the loss
    falls to 0.42 % at worst. Both renderers are given the trimmed file, so
    neither is advantaged; the remaining 0.42 % is carried explicitly in the
    comparison's error budget rather than absorbed into the result.
    """
    grid, values = common.read_spectral_csv(source)
    selected = [(w, v) for w, v in zip(grid, values) if lo <= w <= hi]
    destination.write_text(
        f"# Visible-band excerpt of {pathlib.Path(source).name}, written by\n"
        f"# {pathlib.Path(__file__).name} for the cross-renderer comparison.\n"
        f"# Trimmed because a curve's GPU representation is 64 uniform samples\n"
        f"# across its whole range; over 300-15000 nm that is 200 nm spacing.\n"
        "# wavelength_nm, reflectance\n"
        + "".join(f"{w:.1f}, {v:.6f}\n" for w, v in selected), encoding="utf-8")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO / "build" / "validation" / "cornell_m3")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    asset_dir = REPO / "assets" / "models" / "cornell_box"
    description = {
        "source": "scripts/utils/generate_cornell_box.py",
        "units": "metres (millimetres scaled by 0.001)",
        "groups": [],
    }

    for key, quads in cornell.GROUPS:
        positions, normals, indices = cornell.build_tris(quads)
        ply = args.out_dir / f"{key}.ply"
        vertices, faces = write_ply(ply, positions, normals, indices)

        entry = {
            "key": key,
            "material": cornell.MAT_NAMES[key],
            "ply": ply.name,
            "vertices": vertices,
            "triangles": faces,
        }
        if key in REFLECTANCE_CSV:
            full = asset_dir / REFLECTANCE_CSV[key]
            trimmed = args.out_dir / f"{key}_visible.csv"
            points = trim_to_visible(full, trimmed)
            entry["reflectance_csv_full"] = full.as_posix()
            entry["reflectance_csv"] = trimmed.as_posix()
            entry["reflectance_points_visible"] = points
        else:
            entry["reflectance_flat"] = 1.0
            entry["emissive_rgb"] = cornell.PBR_COLORS[key][:3]
        description["groups"].append(entry)
        print(f"  {key:<10} {faces:>5} triangles -> {ply.name}")

    # The camera, taken from the shipped configs rather than retyped, so that a
    # change to the scene's framing cannot leave the port pointing elsewhere.
    description["camera"] = {
        "position_mm": [278.0, 273.0, -800.0],
        "look_at_mm": [278.0, 273.0, 280.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 39.0,
        "note": "fov is on the Y axis; Mitsuba's perspective sensor defaults to X",
    }
    description["panel_spectrum_csv"] = (
        REPO / "assets" / "luts" / "cornell_panel_radiance.csv").as_posix()

    out = args.out_dir / "scene.json"
    out.write_text(json.dumps(description, indent=2), encoding="utf-8")
    total = sum(g["triangles"] for g in description["groups"])
    print(f"\n{total} triangles in {len(description['groups'])} groups")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
