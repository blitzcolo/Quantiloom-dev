# assets/luts/

Reference data, not generated output. Everything here came from a published
source and should keep saying where from — a spectrum with no provenance is a
number nobody can check.

## Illuminants

All four are read the same way, by whichever key names them: `lighting.solar_lut`
for the light arriving from the sky, `emissive_curve` for the light a surface in
the scene emits. Both take any whitespace- or comma-separated table and let the
scene say which column holds what. They are for different jobs and are not
interchangeable — `astmg173.csv` is a sun and nothing else, and the three CIE
tables are lamps and nothing else.

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

### `CIE_std_illum_A_1nm.csv` — CIE standard illuminant A

A tungsten filament at a distribution temperature of 2856 K. **For incandescent
and tungsten-halogen lamps.**

- Source: International Commission on Illumination (CIE), Vienna, 2018
- DOI: [10.25039/CIE.DS.8jsxjrsn](https://doi.org/10.25039/CIE.DS.8jsxjrsn)
- 531 rows, 1 nm, 300–830 nm, two columns: wavelength, relative power
- Relative, normalised to 100 at 560 nm

CIE 015:2018 defines this one by an equation rather than by measurement — a
Planckian at 2848 K with the 1968 value of $c_2$, which lands at 2856 K on
ITS-90. The shipped table agrees with that equation to $5\times10^{-6}$
relative, checked over all 531 rows. So it is reproducible without the file, and
the file is kept anyway because a table that can be diffed beats an equation
that has to be trusted.

A tungsten-halogen lamp runs hotter than illuminant A — 2800 to 3200 K
depending on the envelope — and tungsten is not a grey emitter, so neither
illuminant A nor a Planckian is exactly one. Use `blackbody_<T>k` when the
colour temperature is what matters and this table when the standard is.

### `CIE_illum_FLs.csv` — CIE illuminants FL1–FL12 and FL3.1–FL3.15

Twenty-seven fluorescent lamps: FL1–FL6 are the standard halophosphate types,
FL7–FL9 broadband, FL10–FL12 narrow-band triphosphor, and the FL3.x series
covers the newer lamp families. **For fluorescent sources**, and the only
illuminants here with emission lines rather than a smooth curve — mercury's
405, 436, 546 and 578 nm spikes sit on top of the phosphor continuum, which is
the whole reason an RGB approximation of a fluorescent lamp is wrong in a way
no colour correction can repair.

- Source: International Commission on Illumination (CIE), Vienna, 2018
- DOI: [10.25039/CIE.DS.ukaymjdn](https://doi.org/10.25039/CIE.DS.ukaymjdn)
- 81 rows, 5 nm, 380–780 nm, 28 columns: wavelength then FL1…FL12,
  FL3.1…FL3.15 in that order
- Relative, normalised to 100 at 560 nm
- Original source is CIE 015:2018 tables 10.1, 10.2 and 10.3

**It stops at 780 nm**, which is narrower than any of the others here, so the
band-coverage warning fires for anything but VIS. That is correct and not worth
suppressing: a fluorescent lamp's near-infrared output is not in this table and
holding its 780 nm value flat across SWIR would invent it.

### Built in, no file

`solar_lut = "equal_energy"` is a flat spectrum at unit luminance, CIE
illuminant E. The neutral reference: it favours no wavelength, which is what
makes it useful for looking at a material without a sun's colour on it. Note
that it is **not** sRGB white — it lands at (1.205, 0.948, 0.909), because sRGB
is referenced to D65 and a flat spectrum is not D65.

## Lamps: `emissive_curve`

The three CIE tables above are also the built-in **emission** spectra, which a
material binds by token rather than by path — `SpectralIO::LoadEmissionSpectrum`
resolves them, and they are compiled in (`core/D65Illuminant.hpp`,
`core/FluorescentIlluminants.hpp`, and illuminant A from its own equation) so
that a token cannot resolve to a file that is missing or has been edited. The
CSVs stay here as the provenance those tables are checked against;
`scripts/make_fl_illuminants.py` is the link for the fluorescent one.

| Token | What |
|---|---|
| `equal_energy` / `illuminant_e` | flat |
| `d65` | average daylight |
| `illuminant_a` | 2856 K tungsten |
| `halogen` | alias for `blackbody_3000k` |
| `cie_f1` … `cie_f12` | fluorescent lamps; f2, f7, f11 are CIE's representatives |
| `cie_f3.1` … `cie_f3.15` | the FL3 series |
| `blackbody_<T>k` | Planck at T kelvin |

Everything but the blackbody family is **relative** — normalised to 100 at
560 nm, no absolute level — which is why `emissive_scale = "match_luminance"` is
the default: the curve gives the shape, the material's `emissive` triple gives
the level. `blackbody_<T>k` is the exception in two ways: it is absolute, in
W·m⁻²·sr⁻¹·nm⁻¹, and it is the only one that spans every band, because it is a
formula rather than a table.

That last point is the one to remember. **An emission curve is zero outside its
own span, never held flat**, unlike a reflectance — so `cie_f7` in a SWIR render
is a dark lamp rather than a lamp with invented near-infrared output. The CLI
says which materials that applies to. `assets/configs/cornell_box_lamp_spectrum.toml`
is a worked example.

## Other

- `CIE_xyz_1931_2deg.csv` — provenance of the colour matching table compiled
  into `src/libQuantiloom/core/CIE_CMF_Data.hpp`. Nothing reads it; it is kept
  so the compiled table has a source. Delete it only together with a note
  saying where those numbers came from.
- `brdf_lut_512_ggx.bin` — generated, not reference data. `BRDFLutGenerator`
  writes it as a disk cache and regenerates it if absent.
