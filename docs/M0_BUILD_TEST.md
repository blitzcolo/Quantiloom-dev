# M0 Milestone: Build and Test Instructions

## What's Completed in M0

- ✅ CMake build system with CPM.cmake
- ✅ Cross-platform C++20 compilation
- ✅ Logging system (spdlog)
- ✅ Configuration loader (TOML++)
- ✅ **EXR I/O** (OpenEXR 3.2.4) - Multi-channel image read/write
- ✅ **HDF5 I/O** (HDF5 1.14.5) - Hyperspectral cube read/write
- ✅ **LUT Loader** - MODTRAN atmosphere LUT support
- ✅ Python script to generate dummy LUT for testing

## Dependencies

### Automatic (via CPM.cmake)
- spdlog 1.14.1
- tomlplusplus 3.4.0
- OpenEXR 3.2.4
- HDF5 1.14.5

CPM will download and build these automatically during CMake configuration.

### Python (for LUT generation)
```bash
pip install numpy h5py
```

## Build Instructions

### Linux

```bash
# Generate build files
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Run
./build/src/app/Quantiloom
```

### Windows (PowerShell)

```bash
# Generate build files
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j

# Run
.\build\src\app\Release\Quantiloom.exe
```

## Test Procedure

### 1. Build and Run Application

After building, run the executable. It will:
- Create `test_output.exr` (256x256 RGB gradient)
- Create `test_cube.h5` (64x64x10 hyperspectral cube)
- Check for `test_lut.h5` (needs manual generation)

### 2. Generate Dummy LUT

```bash
# From project root
python scripts/utils/generate_dummy_lut.py -o test_lut.h5

# Optional: Custom parameters
python scripts/utils/generate_dummy_lut.py \
    -o test_lut.h5 \
    --lambda-min 380 \
    --lambda-max 2500 \
    --num-samples 212 \
    --zenith 30
```

### 3. Re-run Application

Run the executable again. This time it should load `test_lut.h5` and display:
```
[OK] Loaded LUT with 212 wavelength samples
     Wavelength range: 380.0 - 2500.0 nm
```

## Verify Outputs

### Check EXR file

If you have `exrheader` (from OpenEXR tools):
```bash
exrheader test_output.exr
```

Or use any EXR viewer (e.g., djv_view, OpenImageIO's `iv`).

### Check HDF5 files

If you have `h5dump` (from HDF5 tools):
```bash
# View structure
h5dump -H test_cube.h5
h5dump -H test_lut.h5

# View data (first 10 values)
h5dump -d /wavelengths -s 0 -c 10 test_lut.h5
```

Or use Python:
```python
import h5py
import numpy as np

# Check cube
with h5py.File('test_cube.h5', 'r') as f:
    print(f.keys())
    print(f['/data'].shape)  # Should be (10, 64, 64)

# Check LUT
with h5py.File('test_lut.h5', 'r') as f:
    print(f.keys())
    print(f['/wavelengths'][:10])  # First 10 wavelengths
```

## Expected Console Output

```
========================================
  Quantiloom v0.0.1
  Unified Spectral Path Tracing System
========================================
...
========================================
  M0 I/O Module Test
========================================
Test 1: Creating and writing test image (EXR)...
  [OK] Wrote test image to test_output.exr
  [OK] Read back image: 256x256 with 3 channels
Test 2: Creating and writing test spectral cube (HDF5)...
  [OK] Wrote test cube to test_cube.h5
  [OK] Read back cube: 64x64x10 bands
Test 3: Testing LUT I/O...
  [OK] Loaded LUT with 212 wavelength samples
       Wavelength range: 380.0 - 2500.0 nm

========================================
  M0 Milestone Checklist
========================================
  [x] CMake build system
  [x] CPM.cmake dependency management
  [x] Cross-platform compilation (C++20)
  [x] Logging system (spdlog)
  [x] Configuration loader (TOML++)
  [x] EXR I/O (OpenEXR)
  [x] HDF5 I/O (HDF5 C++)
  [x] LUT loader
  [ ] Vulkan initialization (TODO: M1)
  [ ] Scene loading (TODO: M1)

M0 completed! Ready for M1 (Vulkan RT + basic rendering)
```

## Troubleshooting

### OpenEXR/HDF5 build errors

If CPM fails to download or build OpenEXR/HDF5, try:

1. **Use system packages** (Linux):
```bash
sudo apt install libopenexr-dev libhdf5-dev
# Then rebuild with system packages
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

2. **Clear CPM cache**:
```bash
rm -rf .cpm_cache
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### HDF5 linking errors

If you see undefined references to HDF5, ensure HDF5 C++ library is enabled:
```bash
cmake -B build -DHDF5_BUILD_CPP_LIB=ON
```

### Python script fails

Ensure Python 3.8+ and dependencies:
```bash
python --version  # Should be 3.8+
pip install --upgrade numpy h5py
```

## Next Steps: M1

With M0 complete, we can now proceed to M1:
- Vulkan initialization (VulkanContext, Device, Queue)
- Basic scene representation
- BLAS/TLAS construction
- Ray tracing pipeline
- HLSL shaders (raygen, closesthit, miss)
- First rendered frame output

See `M1_IMPLEMENTATION_PLAN.md` for details.
