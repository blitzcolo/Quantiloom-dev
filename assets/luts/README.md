# assets/luts/

Reference data, not generated output. Everything here came from a published
source and should keep saying where from — a spectrum with no provenance is a
number nobody can check.

## Illuminants

Both are read by `lighting.solar_lut`, which takes any whitespace- or
comma-separated table and lets the scene say which columns hold what. They are
for different jobs and are not interchangeable.

### `CIE_std_illum_D65.csv` — CIE standard illuminant D65

Average daylight, and the white point sRGB is defined against. **For RGB work.**

- Source: International Commission on Illumination (CIE), Vienna, 2019
- DOI: [10.25039/CIE.DS.hjfjmt59](https://doi.org/10.25039/CIE.DS.hjfjmt59)
- Downloaded from <https://cie.co.at/data-tables>; `*_metadata.json` is CIE's
  own DataCite record, kept beside it unmodified
- 531 rows, 1 nm, 300–830 nm, two columns: wavelength, relative power
- Relative, normalised to 100 at 560 nm

```toml
[lighting]
solar_lut = "assets/luts/CIE_std_illum_D65.csv"
solar_lut_columns = [2, 0]              # one spectrum, no sky column
solar_lut_normalise = "unit_luminance"  # -> linear sRGB (1, 1, 1)
```

**It stops at 830 nm.** `SpectralCurve::Evaluate` clamps rather than returning
zero, so using it above that holds its 830 nm value flat across the whole band
and produces a render that looks reasonable and means nothing. The CLI warns
when the illuminant does not span the mode's band; do not ignore it.

### `astmg173.csv` — ASTM G-173-03 reference terrestrial spectra

Direct-normal and global-tilt solar irradiance at AM1.5. **For spectral work up
to 4000 nm.**

- Four columns: wavelength, extraterrestrial, global tilt, direct+circumsolar
- Column 3 is *global* and includes the beam, hence `diffuse_is_global`

```toml
[lighting]
solar_lut = "assets/luts/astmg173.csv"
solar_lut_columns = [4, 3]
solar_lut_diffuse_is_global = true
```

### Built in, no file

`solar_lut = "equal_energy"` is a flat spectrum at unit luminance, CIE
illuminant E. The neutral reference: it favours no wavelength, which is what
makes it useful for looking at a material without a sun's colour on it. Note
that it is **not** sRGB white — it lands at (1.205, 0.948, 0.909), because sRGB
is referenced to D65 and a flat spectrum is not D65.

## Other

- `CIE_xyz_1931_2deg.csv` — provenance of the colour matching table compiled
  into `src/libQuantiloom/core/CIE_CMF_Data.hpp`. Nothing reads it; it is kept
  so the compiled table has a source. Delete it only together with a note
  saying where those numbers came from.
- `brdf_lut_512_ggx.bin` — generated, not reference data. `BRDFLutGenerator`
  writes it as a disk cache and regenerates it if absent.
