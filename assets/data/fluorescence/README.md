# Fluorescence test fixtures

Synthetic, and deliberately so: they are shapes chosen to make a transfer
between two wavelengths unambiguous, not measurements of any dye.

| file | what |
|---|---|
| `demo_dye_excitation.csv` | a raised cosine on [400, 500] nm peaking at 0.8, zero elsewhere |
| `demo_dye_emission.csv` | a raised cosine on [550, 650] nm, zero elsewhere |
| `total_absorber_excitation.csv` | 1.0 across the band: everything arriving enters the channel |
| `flat_emission.csv` | 1.0 across the band, so it normalises to 1/380 per nm |

The two bands do not overlap, which is what makes
`scripts/render-tests/check_fluorescence.py` able to say that light appearing
in the second arrived because of light in the first. Paired with
`../../luts/blue_shifted_sun_sky.csv`, which carries the same total power over
the visible band as `flat_sun_sky.csv` but twice as much of it below 500 nm:
under it the emission band gets BRIGHTER while the illuminant's own power there
falls by 36%, which no renderer whose transport is diagonal in wavelength can
reproduce.

The excitation is a fraction in [0, 1], dimensionless. The emission is a shape
only -- its level is normalised away at binding, because a spectrofluorimeter
reports counts and the quantum yield is what says how much light comes back.
