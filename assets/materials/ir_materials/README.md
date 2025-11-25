# Infrared Material Data

This directory contains spectral curve data for infrared (MWIR/LWIR) material properties.

## File Format

CSV files with the following structure:

```csv
# Comments start with #
wavelength_nm, value
3000.0, 0.05
3500.0, 0.06
...
```

- **wavelength_nm**: Wavelength in nanometers (typically 3000-12000 nm for MWIR/LWIR)
- **value**: Property value [0, 1]
  - Emissivity ε(λ): Fraction of blackbody radiation emitted
  - Reflectance ρ(λ): Fraction of incident radiation reflected
  - Transmittance τ(λ): Fraction of incident radiation transmitted

## Kirchhoff's Law

For materials in thermal equilibrium:

```
ε(λ) + ρ(λ) + τ(λ) = 1
```

Where:
- ε(λ) = absorptance = emissivity (by Kirchhoff's law)
- ρ(λ) = reflectance
- τ(λ) = transmittance

## Usage

To use IR material data in Quantiloom:

1. **Create CSV files** with spectral curves (see examples above)

2. **Extend glTF material** (future feature - currently placeholder):

```json
{
  "materials": [
    {
      "name": "Hot Metal",
      "extensions": {
        "QUANTILOOM_material_ir": {
          "emissivityCurve": "assets/materials/ir_materials/aluminum_6061_emissivity.csv",
          "temperature_K": 500.0
        }
      }
    }
  ]
}
```

3. **Set wavelength and mode** in config:

```toml
[spectral]
mode = "mwir_fused"
wavelength_nm = 4000.0  # 4 μm (MWIR)
```

## Example Materials

| Material | Type | File | Temperature Range | Notes |
|----------|------|------|-------------------|-------|
| Aluminum 6061 | Emissivity | `aluminum_6061_emissivity.csv` | 300-1000 K | Low emissivity metal |
| Asphalt | Reflectance | `asphalt_reflectance.csv` | 250-350 K | Typical road surface |
| Glass BK7 | Transmittance | `glass_bk7_transmittance.csv` | 300-400 K | Optical glass (opaque in far-IR) |

## Data Sources

⚠️ **Note**: The provided curves are **synthetic approximations** for testing purposes only.

For quantitative results, use measured spectral data from:
- NIST spectral database
- Vendor datasheets
- Laboratory measurements (FTIR spectroscopy)

## Loading (Future Implementation)

Currently, IR curve loading is **not implemented**. This is a placeholder for future development (P1.2+).

To implement:
1. Use `SpectralIO::LoadSpectralCurveCSV()` (see `SpectralIO.cpp`)
2. Populate `Material::irEmissivityCurve`, `irReflectanceCurve`, `irTransmittanceCurve`
3. Bind IR data to GPU (future: separate spectral curve buffers)
4. Update shader to interpolate curves at runtime

## References

- "Thermal Infrared Remote Sensing" (John Wiley & Sons)
- FLIR Application Notes on emissivity
- NIST Infrared Spectroscopy Data
