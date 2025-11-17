#!/usr/bin/env python3
"""
Generate a minimal dummy LUT for Quantiloom M1 testing.

The LUT contains a single entry with:
- Sun direction: normalized vector pointing down-right (+X, -Y, -Z)
- Sun radiance: moderate solar irradiance (W·sr⁻¹·m⁻²)
- Sky radiance: diffuse hemispherical background (W·sr⁻¹·m⁻²)

Output: dummy_lut.h5 (HDF5 format)
"""

import h5py
import numpy as np
import os

def generate_dummy_lut(output_path):
    """
    Generate a single-entry LUT matching the shader LUTData structure:

    struct LUTData {
        float3 sunDirection;   // Normalized sun direction
        float  _pad0;
        float3 sunRadiance;    // Direct sun radiance (W·sr⁻¹·m⁻²)
        float  _pad1;
        float3 skyRadiance;    // Hemispherical sky radiance (W·sr⁻¹·m⁻²)
        float  _pad2;
    };
    """

    # Sun direction: 45-degree angle (down-right in view)
    sun_dir = np.array([0.7071, -0.7071, -0.3], dtype=np.float32)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)  # Ensure normalized

    # Sun radiance: Moderate solar irradiance
    # For M1 testing, use moderate values to avoid overexposure
    sun_radiance = np.array([2.0, 2.0, 2.0], dtype=np.float32)

    # Sky radiance: Diffuse sky background (gentle blue)
    sky_radiance = np.array([0.3, 0.5, 0.8], dtype=np.float32)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['description'] = 'Dummy LUT for Quantiloom M1 testing'
        f.attrs['version'] = '1.0'
        f.attrs['mode'] = 'M1_simplified'

        # Single-entry LUT (no wavelength or angle dependence)
        f.create_dataset('sun_direction', data=sun_dir, dtype='f4')
        f.create_dataset('sun_radiance', data=sun_radiance, dtype='f4')
        f.create_dataset('sky_radiance', data=sky_radiance, dtype='f4')

        # Optional: Add wavelength metadata
        f.attrs['wavelength_nm'] = 550.0  # Green

    print(f"✓ Dummy LUT generated: {output_path}")
    print(f"  Sun direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
    print(f"  Sun radiance:  [{sun_radiance[0]:.1f}, {sun_radiance[1]:.1f}, {sun_radiance[2]:.1f}] W·sr⁻¹·m⁻²")
    print(f"  Sky radiance:  [{sky_radiance[0]:.1f}, {sky_radiance[1]:.1f}, {sky_radiance[2]:.1f}] W·sr⁻¹·m⁻²")

if __name__ == "__main__":
    # Output to assets/luts/
    output_dir = os.path.join(os.path.dirname(__file__), "../../assets/luts")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "dummy_lut.h5")
    generate_dummy_lut(output_path)

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

