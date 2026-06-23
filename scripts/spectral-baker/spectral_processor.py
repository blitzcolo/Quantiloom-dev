#!/usr/bin/env python3
"""
Spectral Processor (SpectralBaker v3.0)

Resampling, band splitting, and NMF basis extraction for spectral data.
Supports 5 bands: VIS, NIR, SWIR, MWIR, LWIR (0.35-15 µm)

Author: Quantiloom Team
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import interpolate
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


@dataclass
class BandConfig:
    """Configuration for a spectral band."""
    name: str
    range_um: Tuple[float, float]  # (start, end) in micrometers
    interval_nm: float             # Sampling interval in nanometers
    n_components: int = 16         # Number of NMF basis functions for this band

    @property
    def start_nm(self) -> float:
        return self.range_um[0] * 1000

    @property
    def end_nm(self) -> float:
        return self.range_um[1] * 1000

    @property
    def num_samples(self) -> int:
        return int((self.end_nm - self.start_nm) / self.interval_nm) + 1

    def get_wavelength_grid(self) -> np.ndarray:
        """Generate uniform wavelength grid in micrometers."""
        num_samples = int(round((self.end_nm - self.start_nm) / self.interval_nm)) + 1
        wl_nm = np.linspace(self.start_nm, self.end_nm, num_samples)
        return wl_nm / 1000.0  # Convert to micrometers


# Default band configurations (VIS + NIR + SWIR + MWIR + LWIR)
DEFAULT_BANDS = {
    'VIS': BandConfig('VIS', (0.350, 0.780), 2.0, 16),
    'NIR': BandConfig('NIR', (0.780, 1.100), 2.0, 16),
    'SWIR': BandConfig('SWIR', (1.100, 2.500), 2.0, 32),
    'MWIR': BandConfig('MWIR', (2.500, 6.500), 5.0, 32),  # Mid-wave IR: wider sampling
    'LWIR': BandConfig('LWIR', (6.500, 15.000), 10.0, 32), # Long-wave IR: even wider sampling
}


@dataclass
class NMFConfig:
    """Configuration for NMF algorithm."""
    n_components: int = 16
    init: str = 'nndsvda'
    solver: str = 'cd'
    max_iter: int = 1000
    tol: float = 1e-4
    random_state: int = 42


@dataclass
class BandBasis:
    """NMF basis functions for a single band."""
    name: str
    wavelengths: np.ndarray       # [num_samples] in micrometers
    basis_functions: np.ndarray   # [n_components x num_samples]
    explained_variance: float     # Total variance explained


@dataclass
class MaterialWeights:
    """NMF weights for a single material."""
    name: str
    record_id: str
    instrument: str
    chapter: str
    filename: str
    band_weights: Dict[str, np.ndarray]  # band_name -> [n_components]
    band_rmse: Dict[str, float]          # band_name -> reconstruction RMSE
    band_variance: Dict[str, float]      # band_name -> explained variance ratio


def resample_uniform(wavelengths: np.ndarray,
                     reflectance: np.ndarray,
                     target_wavelengths: np.ndarray,
                     kind: str = 'linear') -> np.ndarray:
    """
    Resample spectral data to uniform wavelength grid.

    Args:
        wavelengths: Original wavelength array (micrometers)
        reflectance: Original reflectance values
        target_wavelengths: Target uniform grid (micrometers)
        kind: Interpolation method ('linear', 'cubic', etc.)

    Returns:
        Resampled reflectance values
    """
    # Create interpolation function
    # Use bounds_error=False and fill_value for extrapolation
    interp_fn = interpolate.interp1d(
        wavelengths,
        reflectance,
        kind=kind,
        bounds_error=False,
        fill_value=(reflectance[0], reflectance[-1])  # Clamp at edges
    )

    # Evaluate at target wavelengths
    resampled = interp_fn(target_wavelengths)

    # Ensure non-negative (NMF requirement)
    resampled = np.clip(resampled, 0.0, 1.0)

    return resampled


def split_bands(wavelengths: np.ndarray,
                reflectance: np.ndarray,
                bands: Dict[str, BandConfig] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split spectral data into separate bands and resample to uniform grid.

    Args:
        wavelengths: Original wavelength array (micrometers)
        reflectance: Original reflectance values
        bands: Band configuration dict (default: VIS + SWIR)

    Returns:
        Dict mapping band name to (wavelengths, reflectance) tuple
    """
    if bands is None:
        bands = DEFAULT_BANDS

    result = {}

    for band_name, config in bands.items():
        # Generate target wavelength grid
        target_wl = config.get_wavelength_grid()

        # Check if we have data coverage
        if wavelengths.min() > target_wl[0] or wavelengths.max() < target_wl[-1]:
            logger.warning(f"Band {band_name}: Source data ({wavelengths.min():.3f}-{wavelengths.max():.3f} µm) "
                          f"doesn't fully cover target range ({target_wl[0]:.3f}-{target_wl[-1]:.3f} µm)")

        # Resample
        resampled = resample_uniform(wavelengths, reflectance, target_wl)

        result[band_name] = (target_wl, resampled)

    return result


def build_reflectance_matrix(materials: List,
                             band_config: BandConfig) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Build the reflectance matrix for NMF from a list of materials.

    Args:
        materials: List of MaterialData objects
        band_config: Band configuration for resampling

    Returns:
        Tuple of (wavelengths, reflectance_matrix, valid_materials)
        reflectance_matrix: [n_materials x n_wavelengths]
    """
    target_wl = band_config.get_wavelength_grid()
    n_samples = len(target_wl)

    rows = []
    valid_materials = []

    for mat in materials:
        try:
            # Resample to target grid
            resampled = resample_uniform(mat.wavelengths, mat.reflectance, target_wl)

            # Validate
            if np.any(np.isnan(resampled)) or np.any(np.isinf(resampled)):
                logger.warning(f"Skipping {mat.name}: contains NaN/Inf values")
                continue

            rows.append(resampled)
            valid_materials.append(mat)

        except Exception as e:
            logger.warning(f"Skipping {mat.name}: {e}")
            continue

    if len(rows) == 0:
        raise ValueError("No valid materials found for NMF")

    matrix = np.vstack(rows)  # [n_materials x n_wavelengths]

    logger.info(f"Built reflectance matrix: {matrix.shape[0]} materials × {matrix.shape[1]} wavelengths")

    return target_wl, matrix, valid_materials


def compute_nmf_basis(reflectance_matrix: np.ndarray,
                      config: NMFConfig = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute NMF basis functions and weights.

    Args:
        reflectance_matrix: [n_materials x n_wavelengths] matrix
        config: NMF configuration

    Returns:
        Tuple of (basis_functions, weights, reconstruction_error)
        basis_functions: [n_components x n_wavelengths]
        weights: [n_materials x n_components]
    """
    if config is None:
        config = NMFConfig()

    n_samples, n_features = reflectance_matrix.shape

    # Check if we need to adjust n_components or init method
    n_components = config.n_components
    init_method = config.init

    if n_components > min(n_samples, n_features):
        logger.warning(f"n_components ({n_components}) > min(n_samples, n_features) ({min(n_samples, n_features)})")
        logger.warning(f"Reducing n_components to {min(n_samples, n_features)}")
        n_components = min(n_samples, n_features)

    # 'nndsvda' requires n_components <= min(n_samples, n_features)
    if init_method == 'nndsvda' and n_components > min(n_samples, n_features):
        logger.warning(f"init='nndsvda' not compatible with n_components={n_components}, using 'random' instead")
        init_method = 'random'

    logger.info(f"Running NMF with {n_components} components...")

    # Initialize NMF
    nmf = NMF(
        n_components=n_components,
        init=init_method,
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state
    )

    # Fit and transform
    weights = nmf.fit_transform(reflectance_matrix)  # [n_materials x n_components]
    basis = nmf.components_                           # [n_components x n_wavelengths]

    # Compute reconstruction error
    reconstructed = weights @ basis
    total_variance = np.var(reflectance_matrix)
    residual_variance = np.var(reflectance_matrix - reconstructed)
    explained_variance = 1.0 - (residual_variance / total_variance)

    reconstruction_error = nmf.reconstruction_err_

    logger.info(f"  NMF converged in {nmf.n_iter_} iterations")
    logger.info(f"  Reconstruction error: {reconstruction_error:.6f}")
    logger.info(f"  Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")

    return basis, weights, explained_variance


def compute_per_material_metrics(original: np.ndarray,
                                  basis: np.ndarray,
                                  weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-material reconstruction metrics.

    Args:
        original: [n_materials x n_wavelengths] original data
        basis: [n_components x n_wavelengths] basis functions
        weights: [n_materials x n_components] material weights

    Returns:
        Tuple of (rmse_per_material, explained_variance_per_material)
    """
    reconstructed = weights @ basis

    # Per-material RMSE
    diff = original - reconstructed
    rmse = np.sqrt(np.mean(diff ** 2, axis=1))

    # Per-material explained variance
    # For materials with near-constant spectra (e.g., metals in IR with R≈1),
    # material_var ≈ 0, making explained variance undefined/negative.
    # In these cases, we judge quality by RMSE instead.

    material_var = np.var(original, axis=1)
    residual_var = np.var(diff, axis=1)

    # Threshold for "flat spectrum": if variance < 1e-6, spectrum is essentially constant
    FLAT_THRESHOLD = 1e-6

    explained = np.zeros_like(material_var)

    for i in range(len(material_var)):
        if material_var[i] < FLAT_THRESHOLD:
            # Flat spectrum: judge by RMSE only
            # If reconstruction is good (RMSE < 0.001), report 1.0
            # Otherwise report 0.0 (to indicate reconstruction failed)
            explained[i] = 1.0 if rmse[i] < 0.001 else 0.0
        else:
            # Normal case: compute explained variance
            ev = 1.0 - residual_var[i] / material_var[i]
            # Clamp to reasonable range [-1, 1]
            # Negative values mean reconstruction is worse than a constant model
            explained[i] = np.clip(ev, -1.0, 1.0)

    return rmse, explained


def process_all_bands(materials: List,
                      bands: Dict[str, BandConfig] = None,
                      nmf_config: NMFConfig = None) -> Tuple[Dict[str, BandBasis], List[MaterialWeights]]:
    """
    Process all materials through the full NMF pipeline for all bands.

    Args:
        materials: List of MaterialData objects
        bands: Band configurations (default: VIS + NIR + SWIR)
        nmf_config: Base NMF configuration (n_components overridden per-band)

    Returns:
        Tuple of (band_bases, material_weights)
        band_bases: Dict mapping band_name -> BandBasis
        material_weights: List of MaterialWeights for each material
    """
    if bands is None:
        bands = DEFAULT_BANDS
    if nmf_config is None:
        nmf_config = NMFConfig()

    band_bases = {}
    all_band_data = {}  # band_name -> (wavelengths, matrix, valid_mats, basis, weights, rmse, variance, n_comp)

    # Process each band
    for band_name, band_config in bands.items():
        logger.info(f"\n=== Processing {band_name} band ===")
        logger.info(f"Range: {band_config.range_um[0]:.3f} - {band_config.range_um[1]:.3f} µm")
        logger.info(f"Samples: {band_config.num_samples} @ {band_config.interval_nm} nm interval")
        logger.info(f"Basis functions: {band_config.n_components}")

        # Build matrix
        wavelengths, matrix, valid_mats = build_reflectance_matrix(materials, band_config)

        # Create per-band NMF config with overridden n_components
        band_nmf_config = NMFConfig(
            n_components=band_config.n_components,
            init=nmf_config.init,
            solver=nmf_config.solver,
            max_iter=nmf_config.max_iter,
            tol=nmf_config.tol,
            random_state=nmf_config.random_state
        )

        # Run NMF
        basis, weights, explained_var = compute_nmf_basis(matrix, band_nmf_config)

        # Compute per-material metrics
        rmse, variance = compute_per_material_metrics(matrix, basis, weights)

        # Store results
        band_bases[band_name] = BandBasis(
            name=band_name,
            wavelengths=wavelengths,
            basis_functions=basis,
            explained_variance=explained_var
        )

        all_band_data[band_name] = (wavelengths, matrix, valid_mats, basis, weights, rmse, variance, band_config.n_components)

        logger.info(f"  Mean RMSE: {np.mean(rmse):.6f}")
        logger.info(f"  Max RMSE: {np.max(rmse):.6f}")
        logger.info(f"  Min explained variance: {np.min(variance):.4f}")

    # Build material weights list
    # Use the first band's valid materials as reference
    first_band = list(bands.keys())[0]
    reference_mats = all_band_data[first_band][2]

    material_weights = []
    for i, mat in enumerate(reference_mats):
        band_weights = {}
        band_rmse = {}
        band_variance = {}

        for band_name in bands.keys():
            _, _, valid_mats, _, weights, rmse, variance, n_comp = all_band_data[band_name]

            # Find this material's index in this band's valid list
            try:
                idx = next(j for j, m in enumerate(valid_mats)
                          if m.filename == mat.filename)
                band_weights[band_name] = weights[idx]
                band_rmse[band_name] = float(rmse[idx])
                band_variance[band_name] = float(variance[idx])
            except StopIteration:
                # Material not valid for this band - use correct n_components for zeros
                band_weights[band_name] = np.zeros(n_comp)
                band_rmse[band_name] = 1.0
                band_variance[band_name] = 0.0

        material_weights.append(MaterialWeights(
            name=mat.name,
            record_id=mat.record_id,
            instrument=mat.instrument,
            chapter=mat.chapter,
            filename=mat.filename,
            band_weights=band_weights,
            band_rmse=band_rmse,
            band_variance=band_variance
        ))

    return band_bases, material_weights


def run_basis_experiment(materials: List,
                         n_components_list: List[int],
                         band_config: BandConfig) -> Dict[int, Tuple[float, float, float]]:
    """
    Run NMF with different numbers of components to find optimal basis count.

    Args:
        materials: List of MaterialData objects
        n_components_list: List of component counts to try
        band_config: Band configuration

    Returns:
        Dict mapping n_components to (explained_variance, mean_rmse, max_rmse)
    """
    logger.info(f"\n=== Basis Count Experiment ({band_config.name}) ===")

    # Build matrix once
    wavelengths, matrix, valid_mats = build_reflectance_matrix(materials, band_config)

    results = {}

    for n_comp in n_components_list:
        logger.info(f"\nTrying {n_comp} components...")

        config = NMFConfig(n_components=n_comp)
        basis, weights, explained_var = compute_nmf_basis(matrix, config)
        rmse, _ = compute_per_material_metrics(matrix, basis, weights)

        results[n_comp] = (explained_var, float(np.mean(rmse)), float(np.max(rmse)))

        logger.info(f"  Explained variance: {explained_var:.4f}")
        logger.info(f"  Mean RMSE: {np.mean(rmse):.6f}")
        logger.info(f"  Max RMSE: {np.max(rmse):.6f}")

    return results


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from usgs_loader import load_all_materials

    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s: %(message)s')

    # Test paths
    USGS_ROOT = "../../assets/usgs/ASCIIdata_splib07a"
    WL_FILE = "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"

    print("=== Spectral Processor Test ===\n")

    # Load test materials
    materials = load_all_materials(USGS_ROOT, WL_FILE, max_materials=50)

    # Process
    band_bases, material_weights = process_all_bands(
        materials,
        nmf_config=NMFConfig(n_components=8)
    )

    print("\n=== Results ===")
    for band_name, basis in band_bases.items():
        print(f"\n{band_name} Band:")
        print(f"  Wavelengths: {len(basis.wavelengths)} samples")
        print(f"  Basis shape: {basis.basis_functions.shape}")
        print(f"  Explained variance: {basis.explained_variance:.4f}")

    print(f"\nProcessed {len(material_weights)} materials")
