---
name: spectral-bake
description: Bake material spectra into Quantiloom's NMF basis. Use when adding or regenerating spectral materials from USGS, RefractiveIndex.info, or ECOSTRESS data.
---

# Spectral Bake (SpectralBaker v3)

Converts a spectral library into the per-band NMF basis + material weight database
the renderer loads. Run from `scripts/spectral-baker/` — config paths resolve
relative to the config file.

Dependencies are already installed (numpy, scipy, scikit-learn, pyyaml, matplotlib).

## Pick the source, then the config

| Config | `source_type` | Coverage | Scale |
|---|---|---|---|
| `config.toml` | `usgs` | VIS/NIR/SWIR only, 0.35-2.5 µm | 1,374 ASD materials, 6 chapters |
| `config_refidx.toml` | `refractiveindex` | all 5 bands, 0.2-15 µm | 1,074 nk datasets, 255 materials |
| `config_ecostress.toml` | `ecostress` | all 5 bands, 0.30-25.04 µm union | 3,450 spectra |

MWIR and LWIR require RefractiveIndex or ECOSTRESS. A USGS bake **drops** those two
bands and prints `NOT BAKED (no source data)`; they are absent from the basis file
and the materials JSON rather than filled with extrapolation.

Every band reports a **coverage** fraction — how much of it the source actually
measured. Coverage below 1.0 means the rest is edge-clamp extrapolation, which fits
perfectly by construction, so RMSE and explained variance are optimistic by roughly
the uncovered share. **Never quote a metric without its coverage.** ECOSTRESS in
particular covers each band only partially (VIS came out at 50.9% on a 40-material
sample).

## Workflow

```bash
cd scripts/spectral-baker

python3 bake_spectral.py --config config.toml --scan-only        # what's in the library
python3 bake_spectral.py --config config.toml --max-materials 20 # fast trial (~10 s)
python3 bake_spectral.py --config config.toml                    # full bake
```

Useful flags: `--material "Aluminum" --plot` (single material + plots),
`--experiment-basis 4,8,16,32` (basis-count sweep), `--verbose`.

`--scan-only` works for all three sources. For ecostress it also prints the library's
wavelength span — check that before baking to know which bands will be dropped.

## A full bake overwrites the live assets

Each config's `[output]` paths are exactly what the scene TOMLs load
(`assets/spectral/quantiloom_basis_v3_{usgs,rii}.qlbin` and the matching
`quantiloom_materials_*.json` / `material_summary_*.csv`). There is no rename step.

`--max-materials N` is a trial bake; it refuses to overwrite an existing output
unless you add `--force`. Take that refusal seriously — a 25-material basis
replacing a 1,374-material one is not visible at render time.

`assets/spectral/quantiloom_basis_v3_usgs.qlbin` on disk still predates the coverage
rule: 5 bands, 318 KB, with fabricated MWIR/LWIR. A re-bake yields 3 bands, 111 KB.

## Verify

Each run prints per-band coverage, Mean RMSE, explained variance, and the share of
materials under RMSE 0.03; plots land in `scripts/spectral-baker/output/plots*/`.
Targets from the project's own baselines: mean RMSE < 0.03, good materials > 95%,
explained variance > 98% — per band, and meaningful only at coverage 1.0.
