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
