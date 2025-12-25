# Atmosphere LUT Generator

This directory contains tools and documentation for generating `.qlut` (Quantiloom LUT) files for atmospheric transmittance.

## Overview

Quantiloom uses pre-computed 3D lookup tables for atmospheric effects:
- **Wavelength** (λ): 300-14000 nm
- **Altitude** (h): 0-30000 m
- **Zenith Angle** (θ): 0-85°

These LUTs are generated offline using radiative transfer codes (MODTRAN, libRadtran, etc.) and loaded at runtime for O(1) queries.

---

## File Format Specification (.qlut)

### Structure

```
┌─────────────────────────────────────┐
│  TOML Header (1024 bytes, padded)   │  ← Human-readable metadata
├─────────────────────────────────────┤
│  Transmittance Data (float32[])     │  ← Binary, C-order (row-major)
├─────────────────────────────────────┤
│  Path Radiance Data (float32[])     │  ← Binary, C-order (optional)
└─────────────────────────────────────┘
```

### TOML Header Format

```toml
[metadata]
name = "Midlatitude Summer"
source = "libRadtran 2.0.4"
created = "2024-12-25T10:30:00Z"
atmospheric_model = "US_Standard_1976"
format_version = 1

[grid.wavelength]
start = 300.0      # nm
stop = 14000.0     # nm
step = 10.0        # nm
count = 1371       # number of samples

[grid.altitude]
start = 0.0        # meters
stop = 30000.0     # meters
step = 1000.0      # meters
count = 31         # number of samples

[grid.zenith]
values = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]  # degrees

[data]
header_size = 1024
transmittance_offset = 1024
transmittance_count = 297501        # 1371 × 31 × 7
path_radiance_offset = 2191028      # 1024 + 297501 × 4
path_radiance_count = 297501
dtype = "float32"
byte_order = "little"
layout = "wavelength_altitude_zenith"
```

### Binary Data Layout

Data is stored in **C-order (row-major)**, indexed as `[wavelength][altitude][zenith]`:

```
Index = i_wavelength × (n_altitude × n_zenith) + i_altitude × n_zenith + i_zenith
```

Example for the header above:
- `transmittance[0]` = τ(λ=300nm, h=0m, θ=0°)
- `transmittance[7]` = τ(λ=300nm, h=1000m, θ=0°)
- `transmittance[217]` = τ(λ=310nm, h=0m, θ=0°)

---

## Python Generator Template

### Dependencies

```bash
pip install numpy toml
# For libRadtran integration:
pip install f90nml  # optional, for reading libRadtran outputs
```

### Basic Generator

```python
#!/usr/bin/env python3
"""
Quantiloom Atmosphere LUT Generator
Generates .qlut files from radiative transfer model outputs.
"""

import numpy as np
import struct
from datetime import datetime
from pathlib import Path


class QLUTGenerator:
    """Generates Quantiloom atmosphere LUT files."""

    HEADER_SIZE = 1024  # bytes, padded with null

    def __init__(self):
        self.metadata = {
            "name": "Unknown",
            "source": "Unknown",
            "created": datetime.now().isoformat(),
            "atmospheric_model": "Unknown",
        }
        self.wavelength_nm = None      # 1D array
        self.altitude_m = None          # 1D array
        self.zenith_deg = None          # 1D array
        self.transmittance = None       # 3D array [wave, alt, zen]
        self.path_radiance = None       # 3D array [wave, alt, zen] (optional)

    def set_grid(self, wavelength_nm, altitude_m, zenith_deg):
        """
        Set the grid axes.

        Parameters:
            wavelength_nm: 1D array of wavelengths (must be uniform spacing)
            altitude_m: 1D array of altitudes (must be uniform spacing)
            zenith_deg: 1D array of zenith angles (can be non-uniform)
        """
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=np.float32)
        self.altitude_m = np.asarray(altitude_m, dtype=np.float32)
        self.zenith_deg = np.asarray(zenith_deg, dtype=np.float32)

        # Validate uniform spacing for wavelength and altitude
        wave_step = np.diff(self.wavelength_nm)
        if not np.allclose(wave_step, wave_step[0], rtol=1e-3):
            raise ValueError("Wavelength grid must be uniformly spaced")

        alt_step = np.diff(self.altitude_m)
        if not np.allclose(alt_step, alt_step[0], rtol=1e-3):
            raise ValueError("Altitude grid must be uniformly spaced")

    def set_transmittance(self, data):
        """
        Set transmittance data.

        Parameters:
            data: 3D array with shape [n_wavelength, n_altitude, n_zenith]
                  Values in range [0, 1]
        """
        self.transmittance = np.asarray(data, dtype=np.float32)
        expected_shape = (len(self.wavelength_nm), len(self.altitude_m), len(self.zenith_deg))
        if self.transmittance.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {self.transmittance.shape}")

    def set_path_radiance(self, data):
        """
        Set path radiance data (optional).

        Parameters:
            data: 3D array with shape [n_wavelength, n_altitude, n_zenith]
                  Values in W·sr⁻¹·m⁻²·nm⁻¹
        """
        self.path_radiance = np.asarray(data, dtype=np.float32)

    def _generate_header(self) -> str:
        """Generate TOML header string."""
        n_wave = len(self.wavelength_nm)
        n_alt = len(self.altitude_m)
        n_zen = len(self.zenith_deg)
        total_elements = n_wave * n_alt * n_zen

        wave_step = float(self.wavelength_nm[1] - self.wavelength_nm[0])
        alt_step = float(self.altitude_m[1] - self.altitude_m[0])

        lines = [
            '[metadata]',
            f'name = "{self.metadata["name"]}"',
            f'source = "{self.metadata["source"]}"',
            f'created = "{self.metadata["created"]}"',
            f'atmospheric_model = "{self.metadata["atmospheric_model"]}"',
            'format_version = 1',
            '',
            '[grid.wavelength]',
            f'start = {float(self.wavelength_nm[0])}',
            f'stop = {float(self.wavelength_nm[-1])}',
            f'step = {wave_step}',
            f'count = {n_wave}',
            '',
            '[grid.altitude]',
            f'start = {float(self.altitude_m[0])}',
            f'stop = {float(self.altitude_m[-1])}',
            f'step = {alt_step}',
            f'count = {n_alt}',
            '',
            '[grid.zenith]',
            f'values = [{", ".join(str(float(z)) for z in self.zenith_deg)}]',
            '',
            '[data]',
            f'header_size = {self.HEADER_SIZE}',
            f'transmittance_offset = {self.HEADER_SIZE}',
            f'transmittance_count = {total_elements}',
        ]

        if self.path_radiance is not None:
            path_radiance_offset = self.HEADER_SIZE + total_elements * 4
            lines.extend([
                f'path_radiance_offset = {path_radiance_offset}',
                f'path_radiance_count = {total_elements}',
            ])

        lines.extend([
            'dtype = "float32"',
            'byte_order = "little"',
            'layout = "wavelength_altitude_zenith"',
        ])

        return '\n'.join(lines)

    def save(self, filepath):
        """
        Save LUT to .qlut file.

        Parameters:
            filepath: Output file path
        """
        filepath = Path(filepath)
        if filepath.suffix != '.qlut':
            filepath = filepath.with_suffix('.qlut')

        # Generate header
        header = self._generate_header()
        if len(header) > self.HEADER_SIZE:
            raise ValueError(f"Header too large: {len(header)} > {self.HEADER_SIZE}")

        # Pad header to fixed size
        header_bytes = header.encode('utf-8')
        header_bytes = header_bytes.ljust(self.HEADER_SIZE, b'\x00')

        # Write file
        with open(filepath, 'wb') as f:
            f.write(header_bytes)
            f.write(self.transmittance.astype('<f4').tobytes())
            if self.path_radiance is not None:
                f.write(self.path_radiance.astype('<f4').tobytes())

        print(f"Saved: {filepath} ({filepath.stat().st_size} bytes)")


# =============================================================================
# Example: Generate a simple test LUT (Beer-Lambert approximation)
# =============================================================================

def generate_test_lut(output_path="test_atmosphere.qlut"):
    """
    Generate a simple test LUT using Beer-Lambert law.
    This is NOT physically accurate - use MODTRAN/libRadtran for real data.
    """
    gen = QLUTGenerator()
    gen.metadata = {
        "name": "Test Atmosphere (Beer-Lambert)",
        "source": "Quantiloom test generator",
        "created": datetime.now().isoformat(),
        "atmospheric_model": "Simplified_Beer_Lambert",
    }

    # Define grid
    wavelength_nm = np.arange(300, 14001, 10, dtype=np.float32)   # 300-14000 nm, 10nm step
    altitude_m = np.arange(0, 30001, 1000, dtype=np.float32)       # 0-30km, 1km step
    zenith_deg = np.array([0, 15, 30, 45, 60, 75, 85], dtype=np.float32)

    gen.set_grid(wavelength_nm, altitude_m, zenith_deg)

    n_wave = len(wavelength_nm)
    n_alt = len(altitude_m)
    n_zen = len(zenith_deg)

    # Compute transmittance (simplified Beer-Lambert)
    transmittance = np.zeros((n_wave, n_alt, n_zen), dtype=np.float32)

    # Rayleigh scattering coefficient at sea level (approximate)
    # β_r(λ) ∝ λ^(-4), normalized at 550nm
    beta_rayleigh_550 = 1.2e-5  # m^-1 at sea level
    scale_height_rayleigh = 8500  # m

    # Simple water vapor absorption bands (very approximate)
    def water_absorption(wavelength_nm):
        """Approximate water vapor absorption coefficient."""
        # Major absorption bands at 1.4, 1.9, 2.7, 6.3 μm
        bands = [
            (1400, 100, 0.5),   # 1.4 μm band
            (1900, 150, 0.8),   # 1.9 μm band
            (2700, 200, 1.0),   # 2.7 μm band
            (6300, 500, 1.5),   # 6.3 μm band
        ]
        alpha = 0.0
        for center, width, strength in bands:
            alpha += strength * np.exp(-0.5 * ((wavelength_nm - center) / width) ** 2)
        return alpha * 1e-4  # m^-1

    for i_wave, wave in enumerate(wavelength_nm):
        # Rayleigh scattering (λ^-4 dependence)
        beta_rayleigh = beta_rayleigh_550 * (550.0 / wave) ** 4

        # Water vapor absorption
        beta_water = water_absorption(wave)

        for i_alt, alt in enumerate(altitude_m):
            # Density decreases with altitude (exponential)
            density_factor = np.exp(-alt / scale_height_rayleigh)
            beta_total = (beta_rayleigh + beta_water) * density_factor

            for i_zen, zen in enumerate(zenith_deg):
                # Air mass factor (secant approximation, with correction near horizon)
                cos_zen = np.cos(np.radians(zen))
                if cos_zen > 0.1:
                    air_mass = 1.0 / cos_zen
                else:
                    # Kasten-Young formula for large zenith angles
                    air_mass = 1.0 / (cos_zen + 0.50572 * (96.07995 - zen) ** (-1.6364))

                # Optical depth (integrate from altitude to TOA)
                # Simplified: assume exponential atmosphere
                optical_depth = beta_total * scale_height_rayleigh * air_mass

                # Transmittance
                transmittance[i_wave, i_alt, i_zen] = np.exp(-optical_depth)

    gen.set_transmittance(transmittance)
    gen.save(output_path)

    return gen


if __name__ == "__main__":
    generate_test_lut("test_atmosphere.qlut")
```

---

## Integration with Radiative Transfer Codes

### MODTRAN 5/6

MODTRAN outputs can be parsed and converted to `.qlut` format:

```python
def load_modtran_tape7(filepath):
    """
    Parse MODTRAN tape7 output file.

    Returns:
        wavelength_nm: 1D array of wavelengths
        transmittance: 1D array of transmittance values
        path_radiance: 1D array of path radiance (W/cm²/sr/cm⁻¹)
    """
    # MODTRAN tape7 format varies by version
    # Columns typically:
    #   FREQ (cm⁻¹), TRANS, PTH_THRML, THRML_SCT, SURF_EMIS, ...

    data = np.loadtxt(filepath, skiprows=5)  # Skip header lines

    wavenumber_cm = data[:, 0]
    transmittance = data[:, 1]
    path_radiance = data[:, 2]  # May need unit conversion

    # Convert wavenumber (cm⁻¹) to wavelength (nm)
    wavelength_nm = 1e7 / wavenumber_cm

    # Reverse arrays (MODTRAN outputs high-to-low wavenumber)
    wavelength_nm = wavelength_nm[::-1]
    transmittance = transmittance[::-1]
    path_radiance = path_radiance[::-1]

    return wavelength_nm, transmittance, path_radiance


def generate_modtran_lut(tape7_files, output_path):
    """
    Generate .qlut from multiple MODTRAN runs.

    Parameters:
        tape7_files: Dict mapping (altitude_m, zenith_deg) -> tape7_filepath
        output_path: Output .qlut file path
    """
    gen = QLUTGenerator()
    gen.metadata = {
        "name": "MODTRAN Atmosphere",
        "source": "MODTRAN 5.4",
        "atmospheric_model": "Tropical",
    }

    # Parse first file to get wavelength grid
    first_key = next(iter(tape7_files))
    wavelength_nm, _, _ = load_modtran_tape7(tape7_files[first_key])

    # Collect altitude and zenith values
    altitudes = sorted(set(k[0] for k in tape7_files.keys()))
    zeniths = sorted(set(k[1] for k in tape7_files.keys()))

    # Resample to uniform grid if needed
    wave_uniform = np.arange(300, 14001, 10, dtype=np.float32)
    alt_uniform = np.array(altitudes, dtype=np.float32)
    zen_array = np.array(zeniths, dtype=np.float32)

    gen.set_grid(wave_uniform, alt_uniform, zen_array)

    # Build 3D array
    transmittance = np.zeros((len(wave_uniform), len(alt_uniform), len(zen_array)), dtype=np.float32)

    for (alt, zen), filepath in tape7_files.items():
        i_alt = altitudes.index(alt)
        i_zen = zeniths.index(zen)

        wave, trans, _ = load_modtran_tape7(filepath)

        # Interpolate to uniform grid
        trans_interp = np.interp(wave_uniform, wave, trans)
        transmittance[:, i_alt, i_zen] = trans_interp

    gen.set_transmittance(transmittance)
    gen.save(output_path)
```

### libRadtran

```python
def run_libradtran(wavelength_nm, altitude_m, zenith_deg, output_dir):
    """
    Run libRadtran uvspec for a single configuration.

    Requires libRadtran installed and uvspec in PATH.
    """
    import subprocess
    from pathlib import Path

    # Create input file
    input_content = f"""
atmosphere_file ../data/atmmod/afglms.dat
source solar ../data/solar_flux/atlas_plus_modtran
wavelength {wavelength_nm[0]} {wavelength_nm[-1]}
altitude {altitude_m / 1000.0}  # km
sza {zenith_deg}
output_quantity transmittance
quiet
"""

    input_file = Path(output_dir) / f"uvspec_alt{altitude_m}_zen{zenith_deg}.inp"
    output_file = input_file.with_suffix('.out')

    input_file.write_text(input_content)

    # Run uvspec
    result = subprocess.run(
        ['uvspec'],
        stdin=open(input_file),
        stdout=open(output_file, 'w'),
        stderr=subprocess.PIPE,
        cwd=output_dir
    )

    if result.returncode != 0:
        raise RuntimeError(f"uvspec failed: {result.stderr.decode()}")

    # Parse output
    data = np.loadtxt(output_file)
    return data[:, 0], data[:, 1]  # wavelength, transmittance
```

---

## Recommended LUT Configurations

### Standard Configurations

| Name | Wavelength | Altitude | Zenith | File Size |
|------|------------|----------|--------|-----------|
| **Minimal** | 380-780nm, 5nm | 0-10km, 2km | 0,30,60° | ~100 KB |
| **Visible+NIR** | 380-2500nm, 5nm | 0-15km, 1km | 7 angles | ~1 MB |
| **Full IR** | 300-14000nm, 10nm | 0-30km, 1km | 7 angles | ~12 MB |
| **High-Res MWIR** | 3000-5000nm, 1nm | 0-20km, 500m | 7 angles | ~5 MB |

### Atmospheric Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `US_Standard_1976` | Standard atmosphere | General purpose |
| `Midlatitude_Summer` | Temperate summer | Summer scenes |
| `Midlatitude_Winter` | Temperate winter | Winter scenes |
| `Tropical` | High humidity | Tropical regions |
| `Subarctic_Summer` | Cold, dry | Arctic scenes |
| `Urban` | High aerosol | City environments |

---

## Validation

After generating a `.qlut` file, validate it:

```python
def validate_qlut(filepath):
    """Validate .qlut file integrity."""
    with open(filepath, 'rb') as f:
        # Read header
        header = f.read(1024).decode('utf-8').rstrip('\x00')
        print("Header valid:", 'toml' in header.lower() or '[metadata]' in header)

        # Parse with toml
        import toml
        config = toml.loads(header)

        n_wave = config['grid']['wavelength']['count']
        n_alt = config['grid']['altitude']['count']
        n_zen = len(config['grid']['zenith']['values'])
        expected_bytes = n_wave * n_alt * n_zen * 4

        # Check data size
        data = f.read()
        print(f"Expected: {expected_bytes} bytes, Got: {len(data)} bytes")
        print(f"Has path radiance: {len(data) >= expected_bytes * 2}")

        # Check transmittance range
        trans = np.frombuffer(data[:expected_bytes], dtype='<f4')
        print(f"Transmittance range: [{trans.min():.4f}, {trans.max():.4f}]")
        print(f"Valid range [0,1]: {trans.min() >= 0 and trans.max() <= 1}")


if __name__ == "__main__":
    validate_qlut("test_atmosphere.qlut")
```

---

## Contact

For issues with LUT generation, open an issue on the Quantiloom repository.
