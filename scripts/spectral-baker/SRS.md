# SpectralBaker - Software Requirements Specification

> Written for v1.0 (USGS only, VIS+SWIR). Sections 1, 2, 4, 6, 7, 9 and the
> appendices have been brought in line with the v3.1 implementation. Sections 3, 5
> and 8 are original: the algorithm rationale in §3 still holds, but the function
> signatures in §5 and the C++/HLSL sketches in §8 are v1.0 design intent, not a
> description of the current code -- read the source for those.
> `scripts/spectral-baker/readme.md` is the operational guide; this document is
> the *why*.

## 1. Executive Summary

**Purpose**: Convert spectral library data into runtime-ready basis functions and material weights for physically-based spectral rendering.

**Input**: one of three libraries, selected by `[input].source_type`:

| `source_type` | Library | Physical quantity | Span |
|---|---|---|---|
| `usgs` | USGS splib07a (ASD instrument) | measured reflectance | 0.35-2.5 µm |
| `refractiveindex` | RefractiveIndex.info | complex index (n,k) → reflectance via Fresnel | 0.2-15 µm |
| `ecostress` | ECOSTRESS / ASTER | reflectance or transmittance | 0.30-25.04 µm (union) |

**Output**:
1. Global basis function binary file (shared by all materials)
2. glTF-compatible material database (JSON) with per-material basis weights
3. Per-material statistics (CSV)

**Algorithm**: Non-negative Matrix Factorization (NMF) with configurable per-band basis count [4, 8, 16, 32, 48]

**Reconstruction Model**:
```
Reflectance(λ) ≈ Σ w_i × Basis_i(λ)  for i ∈ [1, N_basis]
```

---

## 2. Data Constraints and Scope

### 2.1 Reality Check: Available Data

**USGS splib07a** (`--scan-only`, measured):
- 2459 AREF files total; 1374 usable ASD files (VIS+SWIR), across 6 chapters
- Spans 0.35-2.5 µm only. Nothing beyond SWIR.

**RefractiveIndex.info**: 1074 nk datasets over 255 unique materials, 0.2-15 µm.

**ECOSTRESS**: 3450 usable spectra, 0.30-25.04 µm as a union — but individual
materials cover very different sub-ranges, so per-band coverage is well under 1.

**Decision**: process all five bands, but **only where the source reaches them**
(§2.3). The v1.0 decision to hardcode VIS+SWIR is superseded: the band set is now
config-driven and the coverage rule decides what is actually emitted.

### 2.2 Spectral Band Definitions

| Band | Wavelength Range | Resampling Interval | Output Samples | Default Basis |
|------|------------------|---------------------|----------------|---------------|
| **VIS** | 350 - 780 nm | 2 nm | 216 | 16 |
| **NIR** | 780 - 1100 nm | 2 nm | 161 | 16 |
| **SWIR** | 1100 - 2500 nm | 2 nm | 701 | 32 |
| **MWIR** | 2500 - 6500 nm | 5 nm | 801 | 32 |
| **LWIR** | 6500 - 15000 nm | 10 nm | 851 | 32 |

Sample count is `(end_nm - start_nm) / interval_nm + 1`. Ranges and basis counts
come from `[processing.bands.*]` in the config, not from code.

**Rationale**:
- **350nm start**: Utilize full ASD range (original starts at 350nm, not 380nm)
- **2nm interval**: Balances spectral fidelity with computational cost
- **Separate bands**: VIS and SWIR have distinct physical mechanisms (electronic transitions vs. vibrational overtones)
- **Coarser interval in the IR**: spectral features broaden with wavelength, so
  5 nm (MWIR) and 10 nm (LWIR) hold detail at a third of the sample count

### 2.3 Coverage Rule

`resample_uniform()` extrapolates outside the source range by clamping to the
first/last sample. It does not fail. A band the source never reaches therefore
comes back as a horizontal line for every material — a rank-1 matrix, which NMF
reconstructs exactly and which the flat-spectrum branch of the metrics scores at
explained variance 1.0. The result reads `RMSE 0.000000, explained variance
100.00%`: better than every band backed by real measurement.

Each band therefore carries a **coverage** fraction, the mean over materials of

```
overlap([band_start, band_end], [material_min, material_max]) / (band_end - band_start)
```

- **coverage == 0** — no material reaches the band. It is skipped: absent from the
  basis binary and from the materials JSON. The C++ loader names bands by file
  position and the uncovered bands (MWIR, LWIR) are last, so a shorter file stays
  correctly indexed; `ReconstructCurve` already warns and returns an empty curve
  for a band it cannot find.
- **0 < coverage < 1** — emitted, but the fraction is written to the console, to
  `coverage` in the materials JSON, and to `<band>_coverage` in the summary CSV.
  RMSE and explained variance are optimistic by roughly the uncovered share.

**No quality metric from this tool is meaningful without its coverage.**

---

## 3. Algorithm Specification

### 3.1 Non-negative Matrix Factorization (NMF)

**Why NMF over PCA**:
- **Physical validity**: Reflectance ≥ 0 always (negative reflectance is unphysical)
- **Interpretability**: Basis functions represent "spectral atoms" with clear meaning
- **Direct storage**: No offset/scaling needed for basis weights

**Implementation**: `sklearn.decomposition.NMF`

**Parameters**:
```python
NMF(
    n_components=16,           # [4, 8, 16, 32] - configurable
    init='nndsvda',            # Deterministic initialization (reproducible)
    solver='cd',               # Coordinate descent (faster for sparse data)
    max_iter=1000,             # Sufficient for convergence
    tol=1e-4,                  # Convergence threshold
    random_state=42            # Reproducibility
)
```

### 3.2 Data Preprocessing Pipeline

```
1. Load USGS ASCII data (skip header line)
2. Load wavelength file (splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt)
3. Build wavelength-reflectance pairs
4. Linear interpolation (scipy.interp1d) to uniform 2nm grid
5. Split into VIS [350-780nm] and SWIR [780-2500nm]
6. Stack all materials into matrix: [N_materials × N_wavelengths]
7. Run NMF separately for VIS and SWIR
8. Extract basis functions (NMF.components_) and weights (NMF.transform())
9. Validate reconstruction error (RMSE per material)
```

---

## 4. Output Specifications

### 4.1 Global Basis File Format

**Filename**: `quantiloom_basis_v3_{usgs,rii,ecostress}.qlbin` — set by
`[output].basis_file`, and matching what `assets/configs/*.toml` loads.

**Binary Layout** (little-endian, format version 3):
```c
struct Header {              // 64 bytes
    char magic[4];           // "QBAS"
    uint32_t version;        // 3
    uint32_t num_bands;      // however many bands survived the coverage rule
    uint8_t reserved[52];    // padding
};

struct BandHeader {          // 16 bytes, one per band
    float wavelength_start;  // micrometers
    float wavelength_end;    // micrometers
    uint32_t num_samples;
    uint32_t num_basis;      // per-band, not global
};

// File structure:
// [Header] { [BandHeader] [float basis_data[num_basis * num_samples]] } * num_bands
```

Version 2 moved the basis count out of the global header and into each band, so
bands can differ in resolution. Version 3 additionally allows any band count.

**Size** = `64 + 16 * num_bands + Σ(num_basis * num_samples * 4)`.

| Source | Bands | Size |
|---|---|---|
| RefractiveIndex (all 5) | VIS 16×216, NIR 16×161, SWIR 32×701, MWIR 32×801, LWIR 32×851 | 325,456 B |
| USGS (MWIR/LWIR skipped) | VIS 16×216, NIR 16×161, SWIR 32×701 | 113,968 B |

### 4.2 Material Database Format

**Filename**: `quantiloom_materials_{usgs,rii,ecostress}.json` — set by
`[output].material_json`.

**JSON Schema**:
```json
{
  "metadata": {
    "generator": "SpectralBaker v3.0",
    "source_library": "USGS splib07a",
    "date_generated": "2026-07-25T10:30:00Z",
    "algorithm": "NMF",
    "num_basis": 16,
    "num_materials": 1374
  },
  "materials": {
    "Aluminum brushed 293K": {
      "source": {
        "filename": "splib07a_Aluminum_brushed_293K_ASDFRa_AREF.txt",
        "record_id": "18253",
        "instrument": "ASDFRa",
        "chapter": "A_ArtificialMaterials"
      },
      "bands": {
        "VIS": {
          "basis_weights": [0.821, 0.152, 0.027],  // num_basis floats, per band
          "rmse": 0.000396,                        // reconstruction error
          "explained_variance": 0.9985,            // fraction of variance captured
          "coverage": 1.0                          // fraction actually measured (§2.3)
        }
      }
    }
  }
}
```

Only bands that passed the coverage rule appear. A USGS database has VIS, NIR and
SWIR; MWIR and LWIR are absent rather than present-and-zero. Band count and basis
count are read from the data, never assumed.

### 4.3 Summary CSV

**Filename**: `material_summary_{usgs,rii,ecostress}.csv` — set by
`[output].summary_csv`.

One row per material: `name, record_id, instrument, chapter, filename`, then
`<band>_rmse, <band>_variance, <band>_coverage` for each surviving band.

**Integration with glTF**:
- Material names must match glTF `material.name` field exactly
- glTF loader reads this JSON at load time
- Basis weights stored in `material.extras.quantiloom_spectral`

---

## 5. Module Architecture

Modules as they exist today. The signatures below are v1.0 design sketches —
`refractiveindex_loader.py` and `ecostress_loader.py` arrived in v3.0 and are not
described here; all three loaders return `MaterialData` and are interchangeable
downstream.

| File | Role |
|---|---|
| `bake_spectral.py` | CLI, config parsing, the four command modes |
| `usgs_loader.py` | USGS ASCII → `MaterialData` |
| `refractiveindex_loader.py` | RefractiveIndex.info YAML (n,k) → reflectance → `MaterialData` |
| `ecostress_loader.py` | ECOSTRESS `.spectrum.txt` → `MaterialData` |
| `spectral_processor.py` | Resampling, coverage, band splitting, NMF |
| `exporter.py` | Binary basis, materials JSON, summary CSV |
| `validator.py` | Statistics and diagnostic plots |

### 5.1 Module 1: USGS Data Loader (`usgs_loader.py`)

**Responsibility**: Parse USGS ASCII format

**Functions**:
```python
def load_wavelengths(filepath: str) -> np.ndarray:
    """Load wavelength file, return array of wavelengths in micrometers."""
    pass

def load_material_reflectance(filepath: str, wavelengths: np.ndarray) -> Dict:
    """
    Load material AREF file, return dict:
    {
        'name': str,
        'record_id': str,
        'wavelengths': np.ndarray,
        'reflectance': np.ndarray
    }
    """
    pass

def discover_materials(root_dir: str) -> List[str]:
    """Recursively find all *_AREF.txt files."""
    pass
```

**Edge Cases**:
- Handle malformed header lines (some files have extra spaces)
- Skip files with insufficient data points (<100 samples)
- Validate reflectance ∈ [0, 1] (some files have bad calibration)

### 5.2 Module 2: Spectral Processor (`spectral_processor.py`)

**Responsibility**: Resampling, band splitting, NMF computation

**Functions**:
```python
def resample_uniform(wavelengths: np.ndarray,
                     reflectance: np.ndarray,
                     target_wavelengths: np.ndarray) -> np.ndarray:
    """Linear interpolation to uniform grid."""
    pass

def split_bands(wavelengths: np.ndarray,
                reflectance: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split into VIS [0.35-0.78] and SWIR [0.78-2.5].
    Return dict with keys 'VIS' and 'SWIR'.
    """
    pass

def compute_basis(reflectance_matrix: np.ndarray,
                  n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run NMF, return (basis_functions, weights).
    basis_functions: [n_components × n_wavelengths]
    weights: [n_materials × n_components]
    """
    pass
```

### 5.3 Module 3: Exporter (`exporter.py`)

**Responsibility**: Write binary basis file and JSON material database

**Functions**:
```python
def write_basis_binary(filepath: str, basis_data: Dict) -> None:
    """Write the basis binary following spec in Section 4.1."""
    pass

def write_material_json(filepath: str, materials: List[Dict]) -> None:
    """Write quantiloom_materials.json following spec in Section 4.2."""
    pass
```

### 5.4 Module 4: Validation (`validator.py`)

**Responsibility**: Generate quality metrics and diagnostic plots

**Functions**:
```python
def compute_reconstruction_error(original: np.ndarray,
                                 basis: np.ndarray,
                                 weights: np.ndarray) -> Dict[str, float]:
    """Return RMSE, max error, explained variance."""
    pass

def plot_reconstruction_comparison(original: np.ndarray,
                                   reconstructed: np.ndarray,
                                   wavelengths: np.ndarray,
                                   material_name: str) -> None:
    """Generate comparison plot (original vs reconstructed curves)."""
    pass

def plot_explained_variance(n_components: List[int],
                           variances: List[float]) -> None:
    """Plot cumulative explained variance vs number of basis functions."""
    pass
```

---

## 6. Configuration File

**Format**: TOML, not YAML. One file per data source: `config.toml` (usgs),
`config_refidx.toml`, `config_ecostress.toml`. Paths inside resolve relative to
the config file's own directory.

```toml
[input]
source_type = "usgs"                      # usgs | refractiveindex | ecostress
usgs_root = "../../assets/usgs/ASCIIdata_splib07a"
wavelength_file = "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"

[processing.bands.VIS]                    # one table per band
range_um = [0.350, 0.780]
interval_nm = 2.0
n_components = 16                         # per-band basis count

[processing.nmf]                          # shared NMF settings
init = "nndsvda"
solver = "cd"
max_iter = 2000
tol = 1.0e-4
random_state = 42

[output]
basis_file = "../../assets/spectral/quantiloom_basis_v3_usgs.qlbin"
material_json = "../../assets/spectral/quantiloom_materials_usgs.json"
summary_csv = "../../assets/spectral/material_summary_usgs.csv"
plots_dir = "./output/plots"

[validation]
plot_worst_n_materials = 10
plot_best_n_materials = 5
export_error_histogram = true
```

**The output paths are the live assets.** They are exactly what the scene TOMLs in
`assets/configs/` load, so a full bake replaces production data. This is
deliberate: the earlier arrangement wrote `quantiloom_basis_v1.bin`, which nothing
loaded, so a bake appeared to succeed while the renderer kept using the old basis.
A trial run (`--max-materials N`) refuses to overwrite an existing output unless
`--force` is given.

---

## 7. Execution Workflow

### 7.1 Development Phase

Every invocation takes `--config`; there is no implicit default source.

**Step 1: Data Discovery**
```bash
python bake_spectral.py --config config.toml --scan-only
# Material counts by chapter. For ecostress, also the wavelength span --
# which is what tells you in advance which bands will be skipped.
```

**Step 2: Single Material Test**
```bash
python bake_spectral.py --config config.toml --material "Aluminum" --plot
# Per-band coverage, RMSE, explained variance, first weights; reconstruction plot.
```

**Step 3: Basis Count Experiment**
```bash
python bake_spectral.py --config config.toml --experiment-basis 4,8,16,32
# Explained variance plot and error table per band. Uncovered bands report NO DATA.
```

**Step 4: Full Baking**
```bash
python bake_spectral.py --config config.toml
# Writes the three artefacts named in [output], overwriting the live assets.
```

### 7.2 Validation Outputs

Production artefacts go to the `[output]` paths (i.e. `assets/spectral/`).
`output/` under the script directory holds only plots:

```
output/plots{,_refidx,_ecostress}/
├── basis_*.png                  # Basis functions per band
├── error_histogram_*.png        # Statistical validation
├── worst_*.png                  # Worst reconstructions, quality check
├── best_*.png                   # Best reconstructions
└── validation_report.txt
```

Pass `--log-file` to persist the run log; it is not written by default.

---

## 8. Integration with Quantiloom Renderer

### 8.1 C++ Loader Implementation

**File**: `src/libQuantiloom/io/SpectralBasisLoader.hpp`

```cpp
// Load global basis functions at startup
struct SpectralBasis {
    u32 num_bands;
    u32 num_basis;

    struct BandBasis {
        f32 wavelength_start_um;
        f32 wavelength_end_um;
        u32 num_samples;
        Vector<f32> basis_data;  // [num_basis × num_samples]
    };

    Vector<BandBasis> bands;  // [VIS, SWIR]
};

SpectralBasis LoadBasisFromFile(const String& filepath);
```

### 8.2 glTF Material Extension

**Modify**: `src/libQuantiloom/scene/Material.hpp`

```cpp
struct Material {
    // ... existing PBR fields ...

    // Spectral representation (choose one)
    enum class SpectralMode {
        Scalar,           // M1: single spectralAlbedo value
        FullCurve,        // M2: spectralReflectanceCurveIndex
        BasisWeights      // M3: NMF basis weights (NEW)
    };

    SpectralMode spectral_mode = SpectralMode::Scalar;

    // M3 data (only valid if spectral_mode == BasisWeights)
    Vector<f32> vis_basis_weights;   // [num_basis] floats
    Vector<f32> swir_basis_weights;  // [num_basis] floats
};
```

### 8.3 Shader Reconstruction

**File**: `src/shaders/spectral_basis.hlsli`

```hlsl
// Descriptor binding for global basis
StructuredBuffer<float> g_VisBasis : register(t20);   // [num_basis × 216]
StructuredBuffer<float> g_SwirBasis : register(t21);  // [num_basis × 860]

// Reconstruct reflectance at given wavelength
float EvaluateSpectralBasis(float wavelength_um, float16 weights, uint band_id)
{
    // Determine sample index in uniform grid
    float lambda_start = (band_id == 0) ? 0.350 : 0.780;
    float lambda_step = 0.002;  // 2nm
    int sample_idx = int((wavelength_um - lambda_start) / lambda_step);

    // Bounds check
    int max_samples = (band_id == 0) ? 216 : 860;
    if (sample_idx < 0 || sample_idx >= max_samples) return 0.0;

    // Reconstruct: Σ w_i × Basis_i(λ)
    float reflectance = 0.0;
    for (int i = 0; i < 16; ++i) {
        int basis_offset = (band_id == 0) ? i * 216 : i * 860;
        reflectance += weights[i] * g_VisBasis[basis_offset + sample_idx];
    }

    return saturate(reflectance);  // Clamp to [0, 1]
}
```

---

## 9. Success Criteria

### 9.1 Quantitative Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Mean RMSE (all materials)** | < 0.03 | Reconstruction error below perceptual threshold |
| **Explained Variance (16 basis)** | > 98% | Capture dominant spectral features |
| **Worst-case RMSE** | < 0.10 | Even outliers remain usable |
| **Good materials (RMSE < 0.03)** | > 95% | Tail stays small |
| **Processing Time** | < 5 min | Full dataset on consumer hardware |

**These apply per band, and only to bands with coverage 1.0.** A partially
covered band beats every target by construction — the extrapolated portion is a
straight line and fits perfectly — so its numbers are not evidence. Report
coverage alongside any metric quoted from this tool.

### 9.2 Qualitative Validation

- [ ] Basis functions are interpretable (smooth, non-oscillatory)
- [ ] Material rankings preserved (if Material A is brighter than B in original, same holds in reconstruction)
- [ ] Spectral features preserved (vegetation red edge, water absorption bands)
- [ ] No negative reflectance in reconstructed spectra

---

## 10. Future Extensions (Out of Scope for M1)

**M2 Features** (if MWIR/LWIR data becomes available):
- Add thermal emission basis functions (Planck blackbody components)
- Temperature-dependent material models
- Kirchhoff's law validation (ε + ρ + τ = 1)

**M3 Features** (compression):
- Variable basis count per material (adaptive complexity)
- Sparse basis weights (many materials may use <8 active basis)
- GPU-side basis decompression shader

---

## Appendix A: File Format Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial release (VIS + SWIR, NMF, 16 basis default), global basis count in header |
| 2.0 | 2025-12-23 | 3 bands (VIS/NIR/SWIR); basis count moved into each band header |
| 3.0 | 2025-12-24 | Multi-source (USGS + RefractiveIndex); 5 bands; variable band count |
| 3.1 | 2026-07-25 | Coverage rule (§2.3); output paths point at the live assets; ECOSTRESS scan |

---

## Appendix B: Known Limitations

1. **Edge-clamp extrapolation**: outside the source's measured range,
   `resample_uniform()` repeats the first/last sample rather than failing or
   returning zero. This is what §2.3 exists to contain. Bands that are entirely
   extrapolated are dropped; partially extrapolated bands are emitted with their
   coverage attached.
2. **Linear interpolation**: Uses `scipy.interp1d(kind='linear')`, not spectral convolution
3. **No atmospheric correction**: library data is lab-measured, not scene-radiance
4. **No BRDF modeling**: Assumes Lambertian reflectance (directional effects ignored)
5. **Fresnel conversion assumes normal incidence**: RefractiveIndex n,k become
   reflectance via `R = ((n-1)² + k²) / ((n+1)² + k²)`, valid at normal incidence only
6. **Per-source material naming**: names must match the glTF `material.name` exactly;
   the three sources use different naming conventions and are not interchangeable

---

**End of Specification**

*Last Updated: 2026-07-25*
*Author: Quantiloom Development Team*
