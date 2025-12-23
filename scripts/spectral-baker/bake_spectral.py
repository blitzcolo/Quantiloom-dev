#!/usr/bin/env python3
"""
SpectralBaker - Main Entry Point

Converts USGS spectral library data into NMF basis functions and material weights
for physically-based spectral rendering.

Usage:
    # Scan available materials
    python bake_spectral.py --scan-only

    # Process single material for testing
    python bake_spectral.py --material "Aluminum" --plot

    # Run basis count experiment
    python bake_spectral.py --experiment-basis 4,8,16,32

    # Full processing
    python bake_spectral.py --config config.toml

Author: Quantiloom Team
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# TOML support (Python 3.11+ has tomllib built-in)
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import numpy as np

from usgs_loader import (
    load_wavelengths,
    load_all_materials,
    discover_materials,
    get_material_by_name,
    MaterialData
)
from spectral_processor import (
    BandConfig,
    NMFConfig,
    process_all_bands,
    run_basis_experiment,
    DEFAULT_BANDS
)
from exporter import (
    write_basis_binary,
    write_material_json,
    write_summary_csv
)
from validator import (
    compute_global_statistics,
    generate_all_plots,
    plot_explained_variance_comparison,
    plot_reconstruction_comparison
)


# Configure logging
def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def parse_band_config(config: dict) -> dict:
    """Parse band configuration from TOML to BandConfig objects."""
    bands = {}
    for band_name, band_cfg in config.get('bands', {}).items():
        range_um = band_cfg.get('range_um', [0.35, 0.78])
        interval_nm = band_cfg.get('interval_nm', 2.0)
        n_components = band_cfg.get('n_components', 16)  # Per-band basis count
        bands[band_name] = BandConfig(band_name, tuple(range_um), interval_nm, n_components)
    return bands if bands else DEFAULT_BANDS


def parse_nmf_config(config: dict) -> NMFConfig:
    """Parse NMF configuration from TOML."""
    nmf_cfg = config.get('nmf', {})
    return NMFConfig(
        n_components=nmf_cfg.get('n_components', 16),
        init=nmf_cfg.get('init', 'nndsvda'),
        solver=nmf_cfg.get('solver', 'cd'),
        max_iter=nmf_cfg.get('max_iter', 1000),
        tol=nmf_cfg.get('tol', 1e-4),
        random_state=nmf_cfg.get('random_state', 42)
    )


def cmd_scan(args, config: dict):
    """Scan available materials and report statistics."""
    logger = logging.getLogger(__name__)

    input_cfg = config.get('input', {})
    usgs_root = input_cfg.get('usgs_root', '../../assets/usgs/ASCIIdata_splib07a')

    # Resolve path relative to config file
    if args.config:
        config_dir = Path(args.config).parent
        usgs_root = str(config_dir / usgs_root)

    logger.info(f"Scanning USGS library: {usgs_root}")

    # Discover all material files
    all_files = discover_materials(usgs_root, pattern="**/*_AREF.txt")
    asd_files = discover_materials(usgs_root, pattern="**/*_AREF.txt", instrument_filter="ASD")

    # Count by chapter
    chapter_counts = {}
    for f in asd_files:
        path = Path(f)
        for part in path.parts:
            if part.startswith("Chapter"):
                chapter_counts[part] = chapter_counts.get(part, 0) + 1
                break

    print("\n" + "=" * 60)
    print("USGS Spectral Library Scan Results")
    print("=" * 60)
    print(f"\nTotal AREF files: {len(all_files)}")
    print(f"ASD files (VIS+SWIR): {len(asd_files)}")
    print(f"\nBy Chapter:")
    for chapter in sorted(chapter_counts.keys()):
        print(f"  {chapter}: {chapter_counts[chapter]}")
    print("=" * 60 + "\n")


def cmd_single_material(args, config: dict):
    """Process a single material for testing/debugging."""
    logger = logging.getLogger(__name__)

    input_cfg = config.get('input', {})
    usgs_root = input_cfg.get('usgs_root', '../../assets/usgs/ASCIIdata_splib07a')
    wl_file = input_cfg.get('wavelength_file', 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')

    # Resolve paths
    if args.config:
        config_dir = Path(args.config).parent
        usgs_root = str(config_dir / usgs_root)

    logger.info(f"Loading materials from: {usgs_root}")

    # Load all materials (needed for NMF to have sufficient data)
    materials = load_all_materials(usgs_root, wl_file, max_materials=args.max_materials)

    # Find target material
    target = get_material_by_name(materials, args.material)
    if target is None:
        logger.error(f"Material not found: {args.material}")
        logger.info("Available materials (first 10):")
        for mat in materials[:10]:
            logger.info(f"  - {mat.name}")
        return 1

    logger.info(f"Found material: {target.name}")
    logger.info(f"  Record ID: {target.record_id}")
    logger.info(f"  Wavelength range: {target.wavelength_range[0]:.3f} - {target.wavelength_range[1]:.3f} µm")

    # Process
    bands = parse_band_config(config.get('processing', {}))
    nmf_config = parse_nmf_config(config.get('processing', {}))

    band_bases, material_weights = process_all_bands(materials, bands, nmf_config)

    # Find target in results
    target_weights = None
    for mw in material_weights:
        if mw.filename == target.filename:
            target_weights = mw
            break

    if target_weights:
        print(f"\n=== {target.name} ===")
        for band_name in bands.keys():
            print(f"\n{band_name} Band:")
            print(f"  RMSE: {target_weights.band_rmse[band_name]:.6f}")
            print(f"  Explained Variance: {target_weights.band_variance[band_name]:.4f}")
            print(f"  Weights: {target_weights.band_weights[band_name][:5]}...")

        # Generate plot if requested
        if args.plot:
            output_dir = Path(config.get('output', {}).get('plots_dir', './output/plots'))
            output_dir.mkdir(parents=True, exist_ok=True)

            for band_name, basis in band_bases.items():
                plot_path = output_dir / f'material_{args.material.replace(" ", "_")}_{band_name}.png'
                plot_reconstruction_comparison(
                    target.wavelengths,
                    target.reflectance,
                    basis,
                    target_weights.band_weights[band_name],
                    target.name,
                    str(plot_path)
                )
                logger.info(f"Saved plot: {plot_path}")

    return 0


def cmd_experiment(args, config: dict):
    """Run basis count experiment."""
    logger = logging.getLogger(__name__)

    input_cfg = config.get('input', {})
    usgs_root = input_cfg.get('usgs_root', '../../assets/usgs/ASCIIdata_splib07a')
    wl_file = input_cfg.get('wavelength_file', 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')

    if args.config:
        config_dir = Path(args.config).parent
        usgs_root = str(config_dir / usgs_root)

    # Parse component counts
    n_components_list = [int(x) for x in args.experiment_basis.split(',')]
    logger.info(f"Running experiment with basis counts: {n_components_list}")

    # Load materials
    materials = load_all_materials(usgs_root, wl_file, max_materials=args.max_materials)

    # Get band configs
    bands = parse_band_config(config.get('processing', {}))

    # Run experiment for each band
    output_dir = Path(config.get('output', {}).get('plots_dir', './output/plots'))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for band_name, band_config in bands.items():
        results = run_basis_experiment(materials, n_components_list, band_config)
        all_results[band_name] = results

        # Generate plot
        plot_path = output_dir / f'experiment_{band_name.lower()}.png'
        plot_explained_variance_comparison(results, band_name, str(plot_path))

    # Print summary table
    print("\n" + "=" * 80)
    print("Basis Count Experiment Results")
    print("=" * 80)

    for band_name, results in all_results.items():
        print(f"\n{band_name} Band:")
        print(f"{'N_Basis':<10} {'Exp.Var':<12} {'Mean RMSE':<12} {'Max RMSE':<12}")
        print("-" * 50)
        for n_comp in sorted(results.keys()):
            var, mean_rmse, max_rmse = results[n_comp]
            print(f"{n_comp:<10} {var*100:>10.2f}% {mean_rmse:>11.6f} {max_rmse:>11.6f}")

    print("\n" + "=" * 80)
    return 0


def cmd_full_bake(args, config: dict):
    """Full baking pipeline."""
    logger = logging.getLogger(__name__)
    start_time = time.time()

    input_cfg = config.get('input', {})
    output_cfg = config.get('output', {})
    proc_cfg = config.get('processing', {})
    valid_cfg = config.get('validation', {})

    usgs_root = input_cfg.get('usgs_root', '../../assets/usgs/ASCIIdata_splib07a')
    wl_file = input_cfg.get('wavelength_file', 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt')

    # Resolve paths relative to config file
    if args.config:
        config_dir = Path(args.config).parent
        usgs_root = str(config_dir / usgs_root)

        # Resolve output paths
        basis_file = str(config_dir / output_cfg.get('basis_file', './output/quantiloom_basis_v1.bin'))
        material_json = str(config_dir / output_cfg.get('material_json', './output/quantiloom_materials.json'))
        plots_dir = str(config_dir / output_cfg.get('plots_dir', './output/plots'))
    else:
        basis_file = output_cfg.get('basis_file', './output/quantiloom_basis_v1.bin')
        material_json = output_cfg.get('material_json', './output/quantiloom_materials.json')
        plots_dir = output_cfg.get('plots_dir', './output/plots')

    # Create output directories
    Path(basis_file).parent.mkdir(parents=True, exist_ok=True)
    Path(material_json).parent.mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Load materials
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Materials")
    logger.info("=" * 60)

    materials = load_all_materials(usgs_root, wl_file, max_materials=args.max_materials)

    if len(materials) == 0:
        logger.error("No materials loaded!")
        return 1

    # Process
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: NMF Basis Extraction")
    logger.info("=" * 60)

    bands = parse_band_config(proc_cfg)
    nmf_config = parse_nmf_config(proc_cfg)

    logger.info(f"Bands: {list(bands.keys())}")
    logger.info(f"Basis functions: {nmf_config.n_components}")

    band_bases, material_weights = process_all_bands(materials, bands, nmf_config)

    # Export
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Exporting Results")
    logger.info("=" * 60)

    write_basis_binary(basis_file, band_bases)
    write_material_json(material_json, material_weights, {
        'num_basis': nmf_config.n_components
    })

    # Export CSV summary
    csv_path = str(Path(material_json).parent / 'material_summary.csv')
    write_summary_csv(csv_path, material_weights)

    # Validation plots
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Generating Validation Plots")
    logger.info("=" * 60)

    generate_all_plots(
        band_bases,
        material_weights,
        materials,
        plots_dir,
        n_worst=valid_cfg.get('plot_worst_n_materials', 10),
        n_best=valid_cfg.get('plot_best_n_materials', 5)
    )

    # Final statistics
    stats = compute_global_statistics(material_weights)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("BAKING COMPLETE")
    print("=" * 60)
    print(f"\nProcessing time: {elapsed:.1f} seconds")
    print(f"Materials processed: {len(material_weights)}")

    for band_name, s in stats.items():
        print(f"\n{band_name} Band:")
        print(f"  Mean RMSE: {s['mean_rmse']:.6f}")
        print(f"  Mean Explained Variance: {s['mean_variance']*100:.2f}%")
        print(f"  Good materials (RMSE < 0.03): {s['percent_good']:.1f}%")

    print(f"\nOutput files:")
    print(f"  Basis: {basis_file}")
    print(f"  Materials: {material_json}")
    print(f"  Summary: {csv_path}")
    print(f"  Plots: {plots_dir}")
    print("=" * 60 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='SpectralBaker - USGS Spectral Data to NMF Basis Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan available materials
    python bake_spectral.py --scan-only

    # Process single material with plot
    python bake_spectral.py --material "Aluminum" --plot

    # Run basis count experiment
    python bake_spectral.py --experiment-basis 4,8,16,32

    # Full processing with config
    python bake_spectral.py --config config.toml
        """
    )

    parser.add_argument('--config', '-c', type=str, default='config.toml',
                       help='Path to TOML configuration file')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan and report available materials')
    parser.add_argument('--material', '-m', type=str,
                       help='Process single material (name pattern)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots for single material mode')
    parser.add_argument('--experiment-basis', type=str,
                       help='Run experiment with comma-separated basis counts (e.g., 4,8,16,32)')
    parser.add_argument('--max-materials', type=int, default=None,
                       help='Limit number of materials for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-file', type=str,
                       help='Write log to file')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file, args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        # Set defaults
        config = {
            'input': {
                'usgs_root': '../../assets/usgs/ASCIIdata_splib07a',
                'wavelength_file': 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt'
            },
            'processing': {
                'bands': {
                    'VIS': {'range_um': [0.350, 0.780], 'interval_nm': 2.0, 'n_components': 16},
                    'NIR': {'range_um': [0.780, 1.100], 'interval_nm': 2.0, 'n_components': 16},
                    'SWIR': {'range_um': [1.100, 2.500], 'interval_nm': 2.0, 'n_components': 32}
                },
                'nmf': {
                    'init': 'nndsvda',
                    'solver': 'cd',
                    'max_iter': 2000,
                    'tol': 1.0e-4,
                    'random_state': 42
                }
            },
            'output': {
                'basis_file': './output/quantiloom_basis_v1.bin',
                'material_json': './output/quantiloom_materials.json',
                'plots_dir': './output/plots'
            },
            'validation': {
                'plot_worst_n_materials': 10,
                'plot_best_n_materials': 5
            }
        }

    # Dispatch to appropriate command
    try:
        if args.scan_only:
            return cmd_scan(args, config)
        elif args.material:
            return cmd_single_material(args, config)
        elif args.experiment_basis:
            return cmd_experiment(args, config)
        else:
            return cmd_full_bake(args, config)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
