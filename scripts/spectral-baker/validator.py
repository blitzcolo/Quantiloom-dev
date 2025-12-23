#!/usr/bin/env python3
"""
Spectral Validation and Visualization

Generates quality metrics and diagnostic plots for NMF basis extraction.
Outputs are designed for thesis/paper inclusion.

Author: Quantiloom Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import matplotlib with Agg backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from spectral_processor import BandBasis, MaterialWeights, BandConfig, DEFAULT_BANDS

logger = logging.getLogger(__name__)

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def compute_global_statistics(material_weights: List[MaterialWeights],
                               bands: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute global statistics across all materials.

    Returns dict with per-band statistics:
        - mean_rmse, std_rmse, min_rmse, max_rmse
        - mean_variance, std_variance, min_variance, max_variance
        - percent_good (RMSE < 0.03)
        - percent_bad (RMSE > 0.10)
    """
    if bands is None:
        bands = list(material_weights[0].band_weights.keys()) if material_weights else []

    stats = {}

    for band in bands:
        rmses = [m.band_rmse.get(band, 0.0) for m in material_weights]
        variances = [m.band_variance.get(band, 0.0) for m in material_weights]

        rmses = np.array(rmses)
        variances = np.array(variances)

        stats[band] = {
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
            'min_rmse': float(np.min(rmses)),
            'max_rmse': float(np.max(rmses)),
            'median_rmse': float(np.median(rmses)),
            'mean_variance': float(np.mean(variances)),
            'std_variance': float(np.std(variances)),
            'min_variance': float(np.min(variances)),
            'max_variance': float(np.max(variances)),
            'percent_good': float(np.sum(rmses < 0.03) / len(rmses) * 100),
            'percent_bad': float(np.sum(rmses > 0.10) / len(rmses) * 100),
            'num_materials': len(rmses)
        }

    return stats


def get_worst_materials(material_weights: List[MaterialWeights],
                        band: str,
                        n: int = 10) -> List[MaterialWeights]:
    """Get materials with highest reconstruction error for a band."""
    sorted_mats = sorted(material_weights,
                        key=lambda m: m.band_rmse.get(band, 0.0),
                        reverse=True)
    return sorted_mats[:n]


def get_best_materials(material_weights: List[MaterialWeights],
                       band: str,
                       n: int = 5) -> List[MaterialWeights]:
    """Get materials with lowest reconstruction error for a band."""
    sorted_mats = sorted(material_weights,
                        key=lambda m: m.band_rmse.get(band, 0.0))
    return sorted_mats[:n]


def plot_basis_functions(band_basis: BandBasis,
                         output_path: str) -> None:
    """
    Plot all basis functions for a band.

    Good for thesis: shows the "spectral atoms" learned by NMF.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    wavelengths_nm = band_basis.wavelengths * 1000  # Convert to nm
    n_basis = band_basis.basis_functions.shape[0]

    colors = plt.cm.viridis(np.linspace(0, 1, n_basis))

    for i in range(n_basis):
        ax.plot(wavelengths_nm, band_basis.basis_functions[i],
               color=colors[i], linewidth=1.2, alpha=0.8,
               label=f'Basis {i+1}')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Basis Function Value')
    ax.set_title(f'{band_basis.name} Band - NMF Basis Functions '
                f'(Explained Variance: {band_basis.explained_variance:.2%})')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved basis plot: {output_path}")


def plot_reconstruction_comparison(original_wavelengths: np.ndarray,
                                   original_reflectance: np.ndarray,
                                   band_basis: BandBasis,
                                   weights: np.ndarray,
                                   material_name: str,
                                   output_path: str) -> None:
    """
    Plot original vs reconstructed spectrum for a single material.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Reconstruct
    reconstructed = weights @ band_basis.basis_functions
    wavelengths_nm = band_basis.wavelengths * 1000

    # Get original data resampled to same grid
    from scipy import interpolate
    interp_fn = interpolate.interp1d(original_wavelengths * 1000,
                                     original_reflectance,
                                     kind='linear',
                                     bounds_error=False,
                                     fill_value='extrapolate')
    original_resampled = np.clip(interp_fn(wavelengths_nm), 0, 1)

    # Main plot
    ax1.plot(wavelengths_nm, original_resampled, 'b-', linewidth=1.5,
            label='Original', alpha=0.8)
    ax1.plot(wavelengths_nm, reconstructed, 'r--', linewidth=1.5,
            label='Reconstructed', alpha=0.8)

    rmse = np.sqrt(np.mean((original_resampled - reconstructed) ** 2))

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title(f'{material_name} - {band_basis.name} Band (RMSE: {rmse:.4f})')
    ax1.legend(loc='upper right')
    ax1.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])
    ax1.set_ylim(0, 1)

    # Residual plot
    residual = original_resampled - reconstructed
    ax2.fill_between(wavelengths_nm, residual, 0,
                    where=(residual >= 0), color='green', alpha=0.5)
    ax2.fill_between(wavelengths_nm, residual, 0,
                    where=(residual < 0), color='red', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residual')
    ax2.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])

    # Set symmetric y-limits for residual
    max_residual = np.max(np.abs(residual)) * 1.1
    ax2.set_ylim(-max_residual, max_residual)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.debug(f"Saved reconstruction plot: {output_path}")


def plot_error_histogram(material_weights: List[MaterialWeights],
                         band: str,
                         output_path: str) -> None:
    """
    Plot histogram of reconstruction errors across all materials.
    """
    rmses = [m.band_rmse.get(band, 0.0) for m in material_weights]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(rmses, bins=50, edgecolor='black', alpha=0.7)

    # Color by quality
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        center = (left + right) / 2
        if center < 0.03:
            patch.set_facecolor('green')
        elif center < 0.10:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')

    # Add vertical lines for thresholds
    ax.axvline(x=0.03, color='green', linestyle='--', linewidth=2,
              label=f'Good threshold (RMSE < 0.03)')
    ax.axvline(x=0.10, color='red', linestyle='--', linewidth=2,
              label=f'Bad threshold (RMSE > 0.10)')

    # Statistics text
    mean_rmse = np.mean(rmses)
    median_rmse = np.median(rmses)
    percent_good = np.sum(np.array(rmses) < 0.03) / len(rmses) * 100

    stats_text = f'Mean: {mean_rmse:.4f}\nMedian: {median_rmse:.4f}\nGood: {percent_good:.1f}%'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Reconstruction RMSE')
    ax.set_ylabel('Number of Materials')
    ax.set_title(f'{band} Band - Reconstruction Error Distribution (n={len(rmses)})')
    ax.legend(loc='upper center')

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved error histogram: {output_path}")


def plot_explained_variance_comparison(experiment_results: Dict[int, Tuple[float, float, float]],
                                       band: str,
                                       output_path: str) -> None:
    """
    Plot explained variance and RMSE vs number of basis functions.

    experiment_results: {n_components: (explained_variance, mean_rmse, max_rmse)}
    """
    n_comps = sorted(experiment_results.keys())
    variances = [experiment_results[n][0] for n in n_comps]
    mean_rmses = [experiment_results[n][1] for n in n_comps]
    max_rmses = [experiment_results[n][2] for n in n_comps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Explained variance plot
    ax1.plot(n_comps, [v * 100 for v in variances], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=98, color='green', linestyle='--', label='98% threshold')
    ax1.axhline(y=95, color='orange', linestyle='--', label='95% threshold')

    ax1.set_xlabel('Number of Basis Functions')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title(f'{band} Band - Explained Variance vs. Basis Count')
    ax1.set_xticks(n_comps)
    ax1.legend()
    ax1.set_ylim(min(90, min(variances) * 100 - 2), 100)

    # RMSE plot
    ax2.plot(n_comps, mean_rmses, 'go-', linewidth=2, markersize=8, label='Mean RMSE')
    ax2.plot(n_comps, max_rmses, 'ro-', linewidth=2, markersize=8, label='Max RMSE')
    ax2.axhline(y=0.03, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Number of Basis Functions')
    ax2.set_ylabel('Reconstruction RMSE')
    ax2.set_title(f'{band} Band - Reconstruction Error vs. Basis Count')
    ax2.set_xticks(n_comps)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved variance comparison: {output_path}")


def plot_multi_material_comparison(materials_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                                   band_name: str,
                                   title_prefix: str,
                                   output_path: str) -> None:
    """
    Plot multiple materials' reconstruction in a grid.

    materials_data: List of (name, wavelengths, original, reconstructed) tuples
    """
    n_materials = len(materials_data)
    n_cols = min(3, n_materials)
    n_rows = (n_materials + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_materials == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, wavelengths, original, reconstructed) in enumerate(materials_data):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        wavelengths_nm = wavelengths * 1000
        rmse = np.sqrt(np.mean((original - reconstructed) ** 2))

        ax.plot(wavelengths_nm, original, 'b-', linewidth=1.2,
               label='Original', alpha=0.8)
        ax.plot(wavelengths_nm, reconstructed, 'r--', linewidth=1.2,
               label='Reconstructed', alpha=0.8)

        # Truncate long names
        short_name = name[:30] + '...' if len(name) > 30 else name
        ax.set_title(f'{short_name}\nRMSE: {rmse:.4f}', fontsize=10)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)

    # Hide empty subplots
    for idx in range(n_materials, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'{title_prefix} - {band_name} Band', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved multi-material comparison: {output_path}")


def generate_validation_report(stats: Dict[str, Dict[str, float]],
                               output_path: str) -> None:
    """
    Generate a text report of validation statistics.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SpectralBaker Validation Report\n")
        f.write("=" * 60 + "\n\n")

        for band, s in stats.items():
            f.write(f"--- {band} Band ---\n")
            f.write(f"  Materials processed: {s['num_materials']}\n\n")

            f.write("  Reconstruction RMSE:\n")
            f.write(f"    Mean:   {s['mean_rmse']:.6f}\n")
            f.write(f"    Median: {s['median_rmse']:.6f}\n")
            f.write(f"    Std:    {s['std_rmse']:.6f}\n")
            f.write(f"    Min:    {s['min_rmse']:.6f}\n")
            f.write(f"    Max:    {s['max_rmse']:.6f}\n\n")

            f.write("  Explained Variance:\n")
            f.write(f"    Mean:   {s['mean_variance']:.4f} ({s['mean_variance']*100:.2f}%)\n")
            f.write(f"    Min:    {s['min_variance']:.4f}\n")
            f.write(f"    Max:    {s['max_variance']:.4f}\n\n")

            f.write("  Quality Breakdown:\n")
            f.write(f"    Good (RMSE < 0.03): {s['percent_good']:.1f}%\n")
            f.write(f"    Bad (RMSE > 0.10):  {s['percent_bad']:.1f}%\n\n")

        f.write("=" * 60 + "\n")

    logger.info(f"Saved validation report: {output_path}")


def generate_all_plots(band_bases: Dict[str, BandBasis],
                       material_weights: List[MaterialWeights],
                       original_materials: List,  # List of MaterialData
                       output_dir: str,
                       n_worst: int = 10,
                       n_best: int = 5) -> None:
    """
    Generate all validation plots and reports.

    Args:
        band_bases: Dict of BandBasis objects
        material_weights: List of MaterialWeights
        original_materials: List of original MaterialData for reconstruction plots
        output_dir: Output directory
        n_worst: Number of worst materials to plot
        n_best: Number of best materials to plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build material lookup
    mat_lookup = {m.filename: m for m in original_materials}

    for band_name, basis in band_bases.items():
        logger.info(f"Generating plots for {band_name} band...")

        # 1. Basis functions
        plot_basis_functions(basis, str(output_path / f'basis_{band_name.lower()}.png'))

        # 2. Error histogram
        plot_error_histogram(material_weights, band_name,
                           str(output_path / f'error_histogram_{band_name.lower()}.png'))

        # 3. Worst materials
        worst = get_worst_materials(material_weights, band_name, n_worst)
        worst_data = []
        for mw in worst:
            if mw.filename in mat_lookup:
                orig = mat_lookup[mw.filename]
                reconstructed = mw.band_weights[band_name] @ basis.basis_functions
                # Resample original to basis grid
                from scipy import interpolate
                interp_fn = interpolate.interp1d(orig.wavelengths, orig.reflectance,
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')
                orig_resampled = np.clip(interp_fn(basis.wavelengths), 0, 1)
                worst_data.append((mw.name, basis.wavelengths, orig_resampled, reconstructed))

        if worst_data:
            plot_multi_material_comparison(worst_data, band_name, 'Worst Reconstructions',
                                          str(output_path / f'worst_{band_name.lower()}.png'))

        # 4. Best materials
        best = get_best_materials(material_weights, band_name, n_best)
        best_data = []
        for mw in best:
            if mw.filename in mat_lookup:
                orig = mat_lookup[mw.filename]
                reconstructed = mw.band_weights[band_name] @ basis.basis_functions
                from scipy import interpolate
                interp_fn = interpolate.interp1d(orig.wavelengths, orig.reflectance,
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')
                orig_resampled = np.clip(interp_fn(basis.wavelengths), 0, 1)
                best_data.append((mw.name, basis.wavelengths, orig_resampled, reconstructed))

        if best_data:
            plot_multi_material_comparison(best_data, band_name, 'Best Reconstructions',
                                          str(output_path / f'best_{band_name.lower()}.png'))

    # 5. Global statistics report
    stats = compute_global_statistics(material_weights)
    generate_validation_report(stats, str(output_path / 'validation_report.txt'))

    logger.info(f"All plots saved to: {output_dir}")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s: %(message)s')

    print("=== Validator Test ===\n")

    # Create dummy data
    n_materials = 100
    vis_basis = BandBasis(
        name='VIS',
        wavelengths=np.linspace(0.35, 0.78, 216),
        basis_functions=np.random.rand(16, 216).astype(np.float32),
        explained_variance=0.98
    )

    materials = []
    for i in range(n_materials):
        rmse = np.random.exponential(0.02)  # Most good, some bad
        materials.append(MaterialWeights(
            name=f'Test Material {i}',
            record_id=str(10000 + i),
            instrument='ASDFRa',
            chapter='ChapterA',
            filename=f'test_{i}.txt',
            band_weights={'VIS': np.random.rand(16)},
            band_rmse={'VIS': rmse},
            band_variance={'VIS': max(0.5, 1.0 - rmse * 10)}
        ))

    # Compute statistics
    stats = compute_global_statistics(materials)
    print("Statistics:")
    for band, s in stats.items():
        print(f"  {band}:")
        print(f"    Mean RMSE: {s['mean_rmse']:.4f}")
        print(f"    Good materials: {s['percent_good']:.1f}%")

    # Generate plots
    test_dir = Path('/tmp/spectral_baker_test/plots')
    test_dir.mkdir(parents=True, exist_ok=True)

    plot_basis_functions(vis_basis, str(test_dir / 'basis_test.png'))
    plot_error_histogram(materials, 'VIS', str(test_dir / 'histogram_test.png'))

    print(f"\nTest plots saved to: {test_dir}")
