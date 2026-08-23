#!/usr/bin/env python3
"""The camouflage-fusion panel of Section VIII-F.

A target that is indistinguishable from its background in the visible and
separated from it in the thermal bands, rendered in three bands and merged by
the Laplacian-pyramid fusion of Section VI.

Two things keep this from being a picture that merely looks convincing.

  * The pair is measured. `e7_find_camouflage_pair.py` searched the whole
    ECOSTRESS/ASTER library for the material closest to this sand across
    400-780 nm and furthest from it in MWIR; the brick it returned is what the
    scene uses. Nothing was chosen for the render.

  * Both surfaces are held at the same temperature, so MWIR separation cannot
    come from a temperature difference. What is left is emissivity, which
    through Kirchhoff is one minus the reflectance the search selected on.

  * The target is a flat patch of the ground rather than an object on it, so
    it shares the ground's normal, irradiance and view angle exactly. The first
    version used a sphere and refuted itself: shading separated the target in
    the visible band more strongly than emissivity separated it in MWIR, and
    the geometry was answering a question meant for the material.

The contrast is then measured rather than asserted, from regions of interest
placed by projecting the patch analytically through the camera the config
declares -- not by thresholding the image, which would use the separation being
measured to decide where to measure it.

Usage:
    e7_camouflage.py --render          # render the three bands, then fuse
    e7_camouflage.py --figure          # rebuild the figure from what exists
"""

import argparse
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"
FUSION = REPO / "build" / "src" / "tools" / "Release" / "fusion_tool.exe"
CONFIG = REPO / "assets" / "configs" / "gallery" / "camouflage_desert.toml"
WORK = pathlib.Path(r"H:\quantiloom-paper\evidence\e7\camouflage")
FIGURES = pathlib.Path(r"H:\quantiloom-paper\figures")

# fusion_tool takes exactly these three, in this order.
BANDS = [("VIS", "vis_fused", "VIS"),
         ("SWIR", "swir_fused", "SWIR"),
         ("MWIR", "mwir_fused", "MWIR")]

# Mirrors assets/configs/gallery/camouflage_desert.toml. Kept here so the ROI
# geometry is derived from the same numbers the renderer was given.
CAMERA = dict(position=(0.0, 9.0, 16.0), look_at=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
              fov_y_deg=34.0, resolution=(1024, 768))
# The target is a square of the ground itself, from generate_camouflage_scene.py.
PATCH = dict(half=4.0, y=0.01)


def read_exr(path):
    """The renderer writes RGBA; OpenEXR sorts channels by name, so ask by name."""
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.asarray(channels[name].pixels, dtype=np.float64)
    planes = [np.asarray(channels[c].pixels, dtype=np.float64)
              for c in ("R", "G", "B") if c in channels]
    if planes:
        return np.stack(planes, axis=-1)
    return np.asarray(next(iter(channels.values())).pixels, dtype=np.float64)


def luminance(image):
    return image[..., 0] if image.ndim == 3 else image


def project(points):
    """World points to pixels, through the camera the config declares.

    A right-handed look-at basis and a pinhole with a vertical field of view,
    which is what the renderer builds. A planar quad projects to a quad, so
    unlike a sphere's silhouette this is exact rather than approximated.
    """
    position = np.array(CAMERA["position"], dtype=np.float64)
    forward = np.array(CAMERA["look_at"], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array(CAMERA["up"], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    width, height = CAMERA["resolution"]
    scale = (height / 2.0) / np.tan(np.radians(CAMERA["fov_y_deg"]) / 2.0)

    out = []
    for point in points:
        offset = np.asarray(point, dtype=np.float64) - position
        depth = float(np.dot(offset, forward))
        out.append((width / 2.0 + float(np.dot(offset, right)) / depth * scale,
                    height / 2.0 - float(np.dot(offset, up)) / depth * scale))
    return np.array(out)


def patch_corners(scale):
    """The target square, grown about its centre by `scale`, in world space."""
    h, y = PATCH["half"] * scale, PATCH["y"]
    return [(-h, y, -h), (h, y, -h), (h, y, h), (-h, y, h)]


def regions(shape):
    """The inside of the target patch, and a ring of ground around it.

    Both are the *same* square grown by different factors and projected through
    the same camera, so both sit on one plane at one orientation and neither
    region contains an edge. The inner inset keeps the patch's own boundary --
    where a pixel straddles two materials -- out of the target mean.
    """
    from matplotlib.path import Path
    height, width = shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.column_stack((xs.ravel(), ys.ravel()))

    def mask(scale):
        polygon = Path(project(patch_corners(scale)))
        return polygon.contains_points(points).reshape(height, width)

    target = mask(0.80)
    background = mask(2.2) & ~mask(1.25)
    return target, background, project(patch_corners(1.0))


def michelson(a, b):
    return abs(a - b) / (a + b) if (a + b) > 0 else 0.0


def render(spp):
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = WORK / "_batch.txt"
    lines = []
    for name, mode, band in BANDS:
        out = (WORK / f"cam_{name.lower()}.exr").as_posix()
        lines.append(f'{CONFIG.as_posix()} | spectral.mode="{mode}" '
                     f'spectral.band="{band}" renderer.spp={spp} '
                     f'renderer.output="{out}"')
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rendering {len(BANDS)} bands at {spp} spp ...", flush=True)
    result = subprocess.run([str(CLI), "batch", str(manifest)], cwd=REPO,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=7200)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"batch render failed ({result.returncode})")
    # Which curve each material bound is worth keeping: a silent fallback to a
    # base colour is invisible in the pixels and is exactly what this scene is
    # claiming did not happen.
    for line in result.stdout.splitlines():
        if "spectral" in line.lower() and ("bound" in line.lower() or
                                           "fallback" in line.lower()):
            print("  " + line.strip())

    fused = WORK / "cam_fused.exr"
    print("fusing VIS + SWIR + MWIR ...", flush=True)
    result = subprocess.run(
        [str(FUSION), str(CONFIG), str(WORK / "cam_vis.exr"),
         str(WORK / "cam_swir.exr"), str(WORK / "cam_mwir.exr"), str(fused)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=1800)
    print(result.stdout[-1500:])
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"fusion failed ({result.returncode})")


def display(image, band):
    """sRGB for the visible, linear AGC for the rest -- never CLAHE."""
    if band == "VIS":
        rgb = image[..., :3]
        scale = np.percentile(rgb, 99.5) or 1.0
        linear = np.clip(rgb / scale, 0.0, 1.0)
        return np.where(linear <= 0.0031308, 12.92 * linear,
                        1.055 * np.power(linear, 1 / 2.4) - 0.055)
    grey = luminance(image)
    lo, hi = np.percentile(grey, [0.5, 99.5])
    return np.clip((grey - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(grey)


def figure(out):
    panels = []
    for name, _, _ in BANDS:
        path = WORK / f"cam_{name.lower()}.exr"
        if path.is_file():
            panels.append((name, read_exr(path)))
    fused_path = WORK / "cam_fused.exr"
    if fused_path.is_file():
        panels.append(("fused", read_exr(fused_path)))
    if not panels:
        raise SystemExit(f"no renders under {WORK}; run with --render")

    target, background, corners = regions(panels[0][1].shape)
    print(f"  target {target.sum():,} px, background {background.sum():,} px; "
          f"patch corners " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in corners))

    record = {}
    figure_, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.35))
    for axis, (name, image) in zip(np.atleast_1d(axes), panels):
        grey = luminance(image)
        t, b = float(grey[target].mean()), float(grey[background].mean())
        record[name] = {"target_mean": t, "background_mean": b,
                        "ratio": (t / b) if b else None,
                        "michelson_contrast": michelson(t, b)}

        shown = display(image, name)
        axis.imshow(shown, cmap=None if shown.ndim == 3 else "gray",
                    vmin=None if shown.ndim == 3 else 0.0,
                    vmax=None if shown.ndim == 3 else 1.0)
        axis.set_title(name, fontsize=9)
        if name == "fused":
            # Merged values carry no physical unit, so no contrast is quoted.
            axis.set_xlabel("Laplacian pyramid\nVIS + SWIR + MWIR, no unit", fontsize=6.2)
        else:
            axis.set_xlabel(f"contrast {record[name]['michelson_contrast']:.3f}\n"
                            f"target/background {record[name]['ratio']:.3f}", fontsize=6.2)
        axis.set_xticks([])
        axis.set_yticks([])
    figure_.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure_.savefig(out, dpi=300, bbox_inches="tight")
    figure_.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure_)

    (WORK / "contrast.json").write_text(json.dumps(
        {"config": str(CONFIG), "camera": CAMERA, "patch": PATCH,
         "regions": {"target_px": int(target.sum()),
                     "background_px": int(background.sum()),
                     "patch_corners_px": corners.tolist(),
                     "note": "target is the patch inset to 80 % of its width; "
                             "background is the same square grown to 220 % with "
                             "the inner 125 % removed, so neither region "
                             "contains the material boundary"},
         "bands": record}, indent=2), encoding="utf-8")

    print()
    for name, values in record.items():
        if values["ratio"] is not None:
            print(f"  {name:6s} target {values['target_mean']:.5g}  "
                  f"background {values['background_mean']:.5g}  "
                  f"ratio {values['ratio']:.4f}  "
                  f"contrast {values['michelson_contrast']:.4f}")
    print(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--out", type=pathlib.Path,
                        default=FIGURES / "fig8c_camouflage_fusion.png")
    args = parser.parse_args()

    if args.render:
        render(args.spp)
    if args.render or args.figure or True:
        figure(args.out)


if __name__ == "__main__":
    main()
