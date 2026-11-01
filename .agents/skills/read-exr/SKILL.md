---
name: read-exr
description: Inspect EXR render output — channel layout, pixel statistics, ROI, and spectral-mode metadata. Use when reading, checking, or comparing .exr files, or verifying a Quantiloom render numerically.
---

# Read OpenEXR Files

The `OpenEXR` Python package is installed (3.4.13). This is the tool for the numeric
check in the `render-verify` skill — "it rendered" is not verification.

## Two layouts, and they differ

Quantiloom writes both. Check before indexing:

| Output | `channels()` returns | Shape |
|---|---|---|
| Main render (`*_output.exr`) | one `RGBA` entry | `(H, W, 4)` |
| Sensor DN (`*_rawdn.exr`) | `Channel_0`, `Channel_1`, `Channel_2` | `(H, W)` each |

`list(f.channels().values())[0]` grabs only `Channel_0` on a rawdn file — read the
dict by name instead.

```python
import OpenEXR, numpy as np

f = OpenEXR.File("gltf_pbr_output.exr")
ch = f.channels()
print({k: v.pixels.shape for k, v in ch.items()})

px = ch["RGBA"].pixels            # (H, W, 4) float32
img = px[:, :, 0].astype(np.float64)
```

## Statistics and ROI

```python
h, w = img.shape
roi = img[h//4:3*h//4, w//4:3*w//4]          # central 50%
print(f"ROI  mean={roi.mean():.6e} std={roi.std():.6e}")
print(f"Full min={img.min():.6e} max={img.max():.6e}")
```

All-zero, all-equal, or NaN/inf statistics mean the render failed even when the exit
code was 0. Rendering is seeded from `std::random_device`, so two runs are never
bit-identical — compare means and histograms, never bytes.

## The header records which spectral mode ran

```python
f.header()["note"]
# "32-wavelength spectral integration"  |  "RGB rendering (fast, no spectral integration)"
```

Use this to confirm the config's `[spectral] mode` actually took effect before
attributing a result to a code change. Other header keys: `channels`, `compression`,
`dataWindow`, `displayWindow`, `lineOrder`, `mode`, `pixelAspectRatio`.

## Notes

- Half-float EXRs are promoted to float32 on read.
- Multi-part: `f.parts()`. Tiled: `.pixels` assembles tiles automatically.
- Hyperspectral cubes are ENVI (`.hdr` + `.dat`), not EXR — different reader.
