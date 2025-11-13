#!/usr/bin/env python3
"""
Generate a dummy atmosphere LUT for testing Quantiloom.

This script creates a physically-plausible (but simplified) LUT based on:
- Solar irradiance: CIE D65 standard illuminant
- Sky radiance: Simplified Rayleigh scattering model
- Transmittance: Beer-Lambert attenuation with Rayleigh optical depth

Output: HDF5 file compatible with LUTLoader
"""

import numpy as np
import h5py
import argparse
from pathlib import Path


def cie_d65_spectrum(wavelengths_nm):
    """
    Approximate CIE D65 standard illuminant spectrum.

    This is a coarse approximation based on tabulated data.
    Real D65 would require interpolation from standard tables.

    Returns: spectral power distribution (W/m^2/nm)
    """
    # Simplified D65 model: blackbody + UV/blue boost
    # D65 correlated color temperature: ~6504K

    lambda_nm = wavelengths_nm
    lambda_m = lambda_nm * 1e-9

    # Planck's law constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458.0     # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)
    T = 6504.0          # D65 CCT (K)

    # Planck blackbody radiance (W/m^2/sr/m)
    numerator = 2.0 * h * c**2 / lambda_m**5
    denominator = np.exp(h * c / (lambda_m * k * T)) - 1.0
    radiance_per_m = numerator / denominator

    # Convert to W/m^2/nm (assume Lambertian sun disk)
    # Solar solid angle: ~6.8e-5 sr
    solar_solid_angle = 6.8e-5
    irradiance_per_nm = radiance_per_m * solar_solid_angle * 1e-9

    # Scale to match typical solar constant (~1361 W/m^2 integrated)
    # This is a rough normalization
    irradiance_per_nm *= 1.5  # Empirical factor

    return irradiance_per_nm


def rayleigh_sky_radiance(wavelengths_nm):
    """
    Simplified Rayleigh sky radiance at zenith.

    Sky radiance is proportional to 1/lambda^4 (Rayleigh scattering).
    This is a rough approximation for clear sky conditions.

    Returns: spectral radiance (W/m^2/sr/nm)
    """
    lambda_nm = wavelengths_nm

    # Rayleigh scattering coefficient (proportional to 1/lambda^4)
    # Reference wavelength: 550 nm
    lambda_ref = 550.0
    rayleigh_factor = (lambda_ref / lambda_nm) ** 4

    # Base sky radiance at 550 nm (typical clear sky)
    # ~0.1 W/m^2/sr/nm at zenith
    base_radiance = 0.1

    sky_radiance = base_radiance * rayleigh_factor

    return sky_radiance


def atmospheric_transmittance(wavelengths_nm, zenith_angle_deg=30.0):
    """
    Atmospheric transmittance using simplified Beer-Lambert law.

    Assumes Rayleigh scattering dominates (clear atmosphere).
    Optical depth scales with 1/lambda^4.

    Args:
        wavelengths_nm: wavelength array (nm)
        zenith_angle_deg: solar zenith angle (degrees)

    Returns: transmittance [0, 1]
    """
    lambda_nm = wavelengths_nm

    # Rayleigh optical depth at zenith (tau_0)
    # Reference: tau_0(550 nm) ~ 0.1 for sea level
    lambda_ref = 550.0
    tau_0_ref = 0.1

    tau_0 = tau_0_ref * (lambda_ref / lambda_nm) ** 4

    # Air mass approximation (Kasten-Young formula for zenith < 90 deg)
    zenith_rad = np.deg2rad(zenith_angle_deg)
    air_mass = 1.0 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith_angle_deg)**(-1.6364))

    # Transmittance: exp(-tau * air_mass)
    transmittance = np.exp(-tau_0 * air_mass)

    return transmittance


def generate_dummy_lut(
    output_path,
    lambda_min=380.0,
    lambda_max=2500.0,
    num_samples=212,
    zenith_angle=30.0
):
    """
    Generate dummy atmosphere LUT and save to HDF5.

    Args:
        output_path: output HDF5 file path
        lambda_min: minimum wavelength (nm)
        lambda_max: maximum wavelength (nm)
        num_samples: number of wavelength samples
        zenith_angle: solar zenith angle (degrees)
    """
    print(f"Generating dummy LUT:")
    print(f"  Wavelength range: {lambda_min} - {lambda_max} nm")
    print(f"  Number of samples: {num_samples}")
    print(f"  Solar zenith angle: {zenith_angle} deg")

    # Generate wavelength array
    wavelengths = np.linspace(lambda_min, lambda_max, num_samples, dtype=np.float32)

    # Generate spectral data
    solar_irradiance = cie_d65_spectrum(wavelengths).astype(np.float32)
    sky_radiance = rayleigh_sky_radiance(wavelengths).astype(np.float32)
    transmittance = atmospheric_transmittance(wavelengths, zenith_angle).astype(np.float32)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Write datasets
        f.create_dataset('/wavelengths', data=wavelengths)
        f.create_dataset('/solar_irradiance', data=solar_irradiance)
        f.create_dataset('/sky_radiance', data=sky_radiance)
        f.create_dataset('/transmittance', data=transmittance)

        # Write metadata
        meta_group = f.create_group('/metadata')
        meta_group.attrs['model'] = 'Dummy_Rayleigh'
        meta_group.attrs['solar_zenith_deg'] = str(zenith_angle)
        meta_group.attrs['visibility_km'] = '23.0'  # Standard clear atmosphere
        meta_group.attrs['description'] = 'Simplified Rayleigh atmosphere for testing'
        meta_group.attrs['generator'] = 'generate_dummy_lut.py'

    print(f"  Output: {output_path}")
    print(f"  Solar irradiance range: {solar_irradiance.min():.2e} - {solar_irradiance.max():.2e} W/m^2/nm")
    print(f"  Sky radiance range: {sky_radiance.min():.2e} - {sky_radiance.max():.2e} W/m^2/sr/nm")
    print(f"  Transmittance range: {transmittance.min():.4f} - {transmittance.max():.4f}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Generate dummy atmosphere LUT for Quantiloom')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--lambda-min', type=float, default=380.0,
                        help='Minimum wavelength (nm), default: 380')
    parser.add_argument('--lambda-max', type=float, default=2500.0,
                        help='Maximum wavelength (nm), default: 2500')
    parser.add_argument('--num-samples', type=int, default=212,
                        help='Number of wavelength samples, default: 212')
    parser.add_argument('--zenith', type=float, default=30.0,
                        help='Solar zenith angle (degrees), default: 30')

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_dummy_lut(
        output_path=str(output_path),
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        num_samples=args.num_samples,
        zenith_angle=args.zenith
    )


if __name__ == '__main__':
    main()
