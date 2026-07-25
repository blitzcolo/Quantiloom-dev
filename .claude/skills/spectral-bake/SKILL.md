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
| `config.toml` | `usgs` | VIS/NIR/SWIR only, 0.35-2.5 µm | 1,195 ASD materials, 6 chapters |
| `config_refidx.toml` | `refractiveindex` | all 5 bands, 0.2-15 µm | n,k → reflectance via Fresnel |
| `config_ecostress.toml` | `ecostress` | ECOSTRESS library | `--scan-only` is broken here (see below) |

MWIR and LWIR require RefractiveIndex or ECOSTRESS. **A USGS bake still emits MWIR
and LWIR bands and reports `RMSE 0.000000, explained variance 100%` for them — that
is degenerate output over an empty wavelength range, not a perfect fit.** Never quote
those numbers as quality evidence.

## Workflow

```bash
cd scripts/spectral-baker

python3 bake_spectral.py --config config.toml --scan-only        # what's in the library
python3 bake_spectral.py --config config.toml --max-materials 20 # fast trial (~10 s)
python3 bake_spectral.py --config config.toml                    # full bake
```

Useful flags: `--material "Aluminum" --plot` (single material + plots),
`--experiment-basis 4,8,16,32` (basis-count sweep), `--verbose`.

`--scan-only` only implements `usgs` and `refractiveindex` branches; with
`config_ecostress.toml` it prints a header, logs `Unknown source_type: ecostress`,
and **still exits 0**. The bake path does support ecostress — only the scan is
missing. Do not read that exit code as success.

## Check the output filenames before wiring anything up

The configs and the shipped artifacts disagree, so a bake will not overwrite what the
renderer actually loads:

| Config writes | Renderer loads (e.g. `assets/configs/cornell_box_vis.toml:12`) |
|---|---|
| `quantiloom_basis_v1.bin` (usgs) | `quantiloom_basis_v3_usgs.qlbin` |
| `quantiloom_basis_refidx_v1.bin` | `quantiloom_basis_v3_rii.qlbin` |
| `quantiloom_materials{,_refidx}.json` | `quantiloom_materials_{usgs,rii}.json` |

After a full bake, either rename the artifacts to the `v3_*` names the scene configs
reference, or update `basis_file` / `materials_json` in the scene TOMLs. Scene configs
currently use absolute paths (`D:/Quantiloom-dev/assets/spectral/...`).

`scripts/spectral-baker/readme.md` names a `quantiloom_basis_v2.qlbin` output that no
config produces — ignore it.

## Verify

Each run prints per-band Mean RMSE, explained variance, and the share of materials
under RMSE 0.03; plots land in `scripts/spectral-baker/output/plots/`. Targets from
the project's own baselines: mean RMSE < 0.03, good materials > 95%, explained
variance > 98% — checked per band, and only on bands the source actually covers.
