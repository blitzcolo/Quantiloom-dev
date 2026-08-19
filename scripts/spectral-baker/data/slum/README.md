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

| Prefix | Count | Material |
|---|---|---|
| A | 10 | Asphalt |
| B | 14 | Brick / engineering brick |
| C | 7 | Concrete / cement |
| G | 5 | Stone / granite / quartzite |
| L | 3 | Slate / fibre cement |
| R | 13 | Roof tile (clay + concrete) |
| S | 5 | Roof tile (slate + fibre cement) |
| V | 6 | PVC roofing membrane |
| X | 3 | Miscellaneous |
| Z | 8 | Metal (aluminium, zinc, lead, iron, painted) |
