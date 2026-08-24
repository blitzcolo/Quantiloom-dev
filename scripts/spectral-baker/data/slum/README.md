# SLUM / LUMA Spectral Library

Spectral Urban Materials (SLUM) and London Urban Micromet Archive (LUMA).

**Source**: Kotthaus, Smith, Wooster & Grimmond (2014), *ISPRS J. Photogramm.
Remote Sens.* 94:194-212.  
**Data DOI**: [10.5281/zenodo.4263842](https://doi.org/10.5281/zenodo.4263842)  
**License**: CC-BY 4.0  
**Downloaded**: 2026-08-19  

## Files

| File | Quantity | Wavelengths | Units |
|---|---|---|---|
| `LUMA_SLUM_SW.csv` | Reflectance | 348.1-2505.8 nm | Percent (0-100) |
| `LUMA_SLUM_IR.csv` | **Emissivity** | 8.0-14.0 µm | Fractional (0-1) |

74 samples (same column set in both files). Same physical specimen measured
on two instruments: Beckman spectrophotometer (SW) and Bruker Vertex 70
FTIR (IR).

## Inversion

The IR file records **emissivity**, not reflectance. The renderer's
`EvaluateEndmemberReflectanceW` + Kirchhoff path expects reflectance:
`rho = 1 - eps` (assuming tau = 0 for opaque urban surfaces). The SLUM
loader inverts once at load time.

## Band gap

2500-8000 nm is not measured. The baker's `spectral_processor` computes
per-material coverage per band and skips bands with zero coverage, so MWIR
(2500-6500 nm) is dropped automatically. VIS and LWIR are fully covered.

## Sample categories (from Kotthaus et al. 2014, Table C.1)

| Prefix | Count | Class |
|---|---|---|
| A | 10 | Road asphalt (9) + asphalt roofing (1) |
| B | 14 | Cement brick (7) + ceramic brick (7) |
| C | 7 | Concrete (4) + cement (3) |
| G | 5 | Granite |
| L | 4 | Roofing shingle (slate, fibre cement) |
| R | 12 | Roofing tile (ceramic + cement) |
| S | 5 | Stone (sandstone 3, limestone 2) |
| V | 6 | PVC roofing sheet |
| X | 3 | Quartzite conglomerate |
| Z | 8 | Metal (aluminium, lead, iron, painted) |

These counts are the paper's own class table (Kotthaus et al. 2014, p. 200) and
the documentation PDF agrees with them sample by sample. An earlier version of
this table had L at 3, R at 13, S as roof tiles, G as mixed stone and X as
"miscellaneous" — none of which is right, and the same edit that corrected it
replaced 74 invented sample descriptions in `slum_loader.py`. See that file's
header for the evidence.

## Sample names

The two CSVs carry **only sample IDs** — `A001`, `B001`, and so on. Every
human-readable name comes from the documentation PDF:

<https://urban-meteorology-reading.github.io/other%20files/LUMA_SLUM.pdf>

78 pages, one per sample, each giving Class / Material / Colour / Status /
Dimension. That file is the authority for anything descriptive about a sample;
nothing else in this repository is.
