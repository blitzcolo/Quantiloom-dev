#!/usr/bin/env python3
"""How much of the brightness ordering each tone operator destroys.

The claim in Section VI is that linear AGC is the default because it is the
only globally monotonic choice: two pixels at the same radiance must display as
the same grey, or a temperature cannot be read off the image. CLAHE breaks that
by construction -- its whole mechanism is a different mapping per tile -- and
this measures by how much.

The metric is the fraction of brightness-ordered pixel pairs the operator
inverts: sample pairs at random, and count those where pixel A has more
radiance than B but displays darker. Sampled rather than exhaustive because
3 x 10^6 pixels is 5 x 10^12 pairs; the sample size is reported with the
result and the standard error is well under the third digit.

This reimplements clahe.comp.hlsl's three passes in numpy rather than reading
the GPU's output, so that the number is checkable line by line against the
shader. The details that matter and are easy to get wrong:

  * Pass 2 sums every tile's histogram when equalising, so all tiles share one
    CDF and pass 3's bilinear blend between four identical CDFs is that CDF.
    Linear skips passes 1 and 2 entirely and is the window and nothing else.
  * Clipping is per tile at clipLimit x the mean bin count, and the clipped
    excess is redistributed equally over all bins, remainder to the first few.
  * Pass 3 blends the four surrounding tile CDFs bilinearly about tile centres,
    which is what keeps CLAHE from showing tile seams -- and is also why its
    mapping varies continuously across the frame rather than in blocks.
  * An achromatic pixel bypasses the luminance-ratio path, because
    RGBToLuminance(v, v, v) is not v to the last bit and the ratio wobbles grey
    pixels around each CDF plateau. Every infrared render is achromatic.

Usage:
    e8_agc_monotonicity.py --input <render.exr>
"""

import argparse
import json
import pathlib

import numpy as np

BINS = 256
TILES = 8          # DisplayEnhancementParams::tileSize, a grid dimension
CLIP_LIMIT = 2.0   # DisplayEnhancementParams::clipLimit
PAIRS = 20_000_000
SEED = 0x547C


def read_exr_luminance(path):
    import OpenEXR
    channels = OpenEXR.File(str(path)).channels()
    array = np.asarray(list(channels.values())[0].pixels, dtype=np.float64)
    if array.ndim == 3:
        # Achromatic renders: every colour channel carries the same value, and
        # the shader's own luminance of a grey pixel is that pixel.
        return array[..., 0]
    return array


def tile_histograms(values, lo, hi):
    """Pass 1. One 256-bin histogram per tile, on the normalised value."""
    height, width = values.shape
    tile_w = (width + TILES - 1) // TILES
    tile_h = (height + TILES - 1) // TILES

    normalised = np.clip((values - lo) / (hi - lo), 0.0, 1.0) if hi > lo \
        else np.full_like(values, 0.5)
    bins = np.minimum((normalised * (BINS - 1) + 0.5).astype(np.int64), BINS - 1)

    ys, xs = np.indices(values.shape)
    tile_index = (ys // tile_h) * TILES + (xs // tile_w)
    flat = tile_index.ravel() * BINS + bins.ravel()
    counts = np.bincount(flat, minlength=TILES * TILES * BINS)
    return counts.reshape(TILES, TILES, BINS), bins


def clipped_cdfs(histograms, equalise):
    """Pass 2. Clip at clipLimit x mean, redistribute the excess, prefix-sum.

    When equalising, every tile is first replaced by the sum over all tiles,
    which is how one pipeline serves both operators.
    """
    if equalise:
        total = histograms.sum(axis=(0, 1))
        histograms = np.broadcast_to(total, histograms.shape).copy()

    cdfs = np.empty(histograms.shape, dtype=np.float64)
    for ty in range(TILES):
        for tx in range(TILES):
            raw = histograms[ty, tx].astype(np.int64)
            total_pixels = int(raw.sum())
            if total_pixels == 0:
                cdfs[ty, tx] = np.linspace(0.0, 1.0, BINS)
                continue
            threshold = max(int(CLIP_LIMIT * total_pixels / BINS), 1)
            clipped = np.minimum(raw, threshold)
            excess = int((raw - clipped)[raw > threshold].sum())
            per_bin, remainder = divmod(excess, BINS)
            clipped = clipped + per_bin
            clipped[:remainder] += 1
            cdfs[ty, tx] = np.cumsum(clipped) / float(total_pixels)
    return cdfs


def apply_pass(values, bins, cdfs, lo, hi):
    """Pass 3. Bilinear blend of the four surrounding tile CDFs, per pixel."""
    height, width = values.shape
    tile_w = width / TILES
    tile_h = height / TILES

    ys, xs = np.indices(values.shape)
    fx_all = (xs + 0.5) / tile_w - 0.5
    fy_all = (ys + 0.5) / tile_h - 0.5
    x0 = np.floor(fx_all).astype(np.int64)
    y0 = np.floor(fy_all).astype(np.int64)
    fx = fx_all - x0
    fy = fy_all - y0
    x0c = np.clip(x0, 0, TILES - 1)
    y0c = np.clip(y0, 0, TILES - 1)
    x1c = np.clip(x0 + 1, 0, TILES - 1)
    y1c = np.clip(y0 + 1, 0, TILES - 1)

    c00 = cdfs[y0c, x0c, bins]
    c10 = cdfs[y0c, x1c, bins]
    c01 = cdfs[y1c, x0c, bins]
    c11 = cdfs[y1c, x1c, bins]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + \
           (c01 * (1 - fx) + c11 * fx) * fy


def inverted_pairs(radiance, display, rng, pairs):
    """The fraction of brightness-ordered pairs the operator reverses, and the
    worst display-value drop among them."""
    flat_r = radiance.ravel()
    flat_d = display.ravel()
    a = rng.integers(0, flat_r.size, pairs)
    b = rng.integers(0, flat_r.size, pairs)

    ra, rb = flat_r[a], flat_r[b]
    da, db = flat_d[a], flat_d[b]
    # Only pairs the radiance actually orders; ties carry no obligation.
    ordered = ra != rb
    higher = np.where(ra > rb, da, db)
    lower = np.where(ra > rb, db, da)
    # An inversion smaller than the display can represent is not an inversion.
    # Pass 3 blends four tile CDFs bilinearly, and when those CDFs are
    # identical -- which is exactly what equalising makes them -- the blend
    # returns the same value only to within rounding. A strict comparison
    # counts that rounding as a reordering and reports a fraction of a per cent
    # of inversions for an operator that provably has none.
    epsilon = 0.5 / 255.0
    bad = ordered & (higher < lower - epsilon)
    drop = (lower - higher)[bad]
    return (float(bad.sum()) / float(ordered.sum()),
            float(drop.max()) if drop.size else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(r"H:\quantiloom-paper\evidence\e8"
                                             r"\agc_monotonicity.json"))
    args = parser.parse_args()

    values = read_exr_luminance(args.input)
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)
    lo, hi = float(values.min()), float(values.max())
    print(f"{args.input.name}: {values.shape[1]} x {values.shape[0]} "
          f"= {values.size / 1e6:.2f} Mpx, radiance {lo:.4g}\u2013{hi:.4g}")

    histograms, bins = tile_histograms(values, lo, hi)
    rng = np.random.default_rng(SEED)

    results = {}
    normalised = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    for name, display in (
            ("Linear", normalised),
            ("Equalize", apply_pass(values, bins,
                                    clipped_cdfs(histograms, True), lo, hi)),
            ("Clahe", apply_pass(values, bins,
                                 clipped_cdfs(histograms, False), lo, hi))):
        fraction, drop = inverted_pairs(values, display, rng, args.pairs)
        results[name] = {"inverted_fraction": fraction,
                         "worst_drop": drop,
                         "worst_drop_levels_of_255": drop * 255.0}
        print(f"  {name:9s} inverted {100 * fraction:7.3f} %   "
              f"worst drop {drop:.4f}  ({drop * 255:.0f} of 255 display levels)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"input": str(args.input), "megapixels": values.size / 1e6,
         "pairs_sampled": args.pairs, "seed": hex(SEED),
         "bins": BINS, "tiles": f"{TILES}x{TILES}", "clip_limit": CLIP_LIMIT,
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
