# M1 Spectral Upgrade - Implementation Summary

This document summarizes the core spectral path tracing infrastructure added to complete M1 milestone requirements.

---

## 🎯 What Was Implemented

### 1. **Spectral Material Curve System** ✅

Replaced scalar `spectralAlbedo` with full wavelength-dependent reflectance curves.

#### Files Modified:
- `src/libQuantiloom/core/SpectralData.hpp` (NEW)
  - `SpectralCurve` - CPU-side variable-length curves
  - `SpectralCurveGPU` - GPU-side fixed-size (64 samples) curves
  - Linear interpolation for continuous wavelength queries

- `src/libQuantiloom/scene/Material.hpp`
  - Added `i32 spectralReflectanceCurveIndex` field
  - Index into scene-wide spectral curve buffer
  - `-1` = fallback to legacy `spectralAlbedo` (M1 compatibility)

- `src/shaders/common.hlsli`
  - Added `SpectralCurveGPU` struct definition
  - Added `spectralReflectanceCurveIndex` to `MaterialData`
  - Updated struct size: 84 → 88 bytes

- `src/app/main.cpp`
  - Updated `MaterialDataCPU` structure
  - Added `spectralReflectanceCurveIndex` field
  - Updated all `static_assert` validations
  - New total size: 88 bytes (matches GPU)

---

### 2. **Spectral Query System** ✅

GPU-side functions for querying spectral curves during ray tracing.

#### Files Created:
- `src/shaders/spectral_query.hlsli` (NEW)
  - `EvaluateSpectralCurve()` - Linear interpolation on GPU
  - `QueryMaterialSpectralAlbedo()` - Material reflectance at λ
  - `QueryAtmosphericTransmittance()` - Placeholder for MODTRAN LUT

#### Integration:
- `src/shaders/closesthit.rchit`
  - Added `#include "spectral_query.hlsli"`
  - Added `[[vk::binding(13, 0)]] StructuredBuffer<SpectralCurveGPU> spectralCurves;`

---

### 3. **LUT-fast Atmospheric Model** ✅

Implemented Beer-Lambert atmospheric attenuation with path-length dependence.

#### Implementation in `closesthit.rchit`:

```hlsl
// Compute path length from camera to hit point
float pathLength_m = RayTCurrent();

// Convert LUT transmittance (vertical optical depth) to extinction coefficient
const float atmosphericScaleHeight_m = 8000.0;  // Rayleigh scale height
float opticalDepth_vertical = -log(max(lut.transmittance, 1e-6));
float extinctionCoeff = opticalDepth_vertical / atmosphericScaleHeight_m;

// Apply Beer-Lambert attenuation: T(λ, d) = exp(-σ_t × d)
float atmosphericTransmittance = exp(-extinctionCoeff * pathLength_m);
atmosphericTransmittance = clamp(atmosphericTransmittance, 0.0, 1.0);

// Apply to sun radiance
sunRadiance *= atmosphericTransmittance;
```

#### Physical Model:
- **Beer-Lambert Law**: `T(λ, d) = exp(-σ_t(λ) × d)`
  - `σ_t(λ)` = wavelength-dependent extinction coefficient (1/m)
  - `d` = path length from camera to surface (m)
- **Conversion**: `σ_t = τ_vertical / H_atm`
  - `τ_vertical` from MODTRAN LUT (vertical optical depth)
  - `H_atm = 8000m` (atmospheric scale height)

---

### 4. **Sky Radiance Hemispherical Integration** ✅

Enhanced sky ambient lighting with physical interpretation.

#### Implementation in `closesthit.rchit`:

```hlsl
// Hemispherical integration with Lambertian BRDF
// L_sky = ∫_Ω L_sky(ω) × BRDF(ω) × (N · ω) dω
//       = (ρ/π) × L_sky × π = ρ × L_sky
//
// For PBR: account for Fresnel term (energy conservation)
float3 kD = (1.0 - FresnelSchlick(F0, max(dot(normal, V), 0.0))) * (1.0 - metallic);
float3 skyAmbient = kD * albedo * skyRadiance;
```

#### Physical Interpretation:
- **Hemispherical integral**: `∫(N·ω)dω = π` (solid angle of hemisphere)
- **Lambertian BRDF**: `f = ρ/π`
- **Result**: Factor of π cancels → `L_out = ρ × L_incident`
- **PBR adjustments**: Fresnel term + metallic masking for energy conservation

---

### 5. **MODTRAN LUT Loader Interface** ✅

Placeholder implementation for user to fill in HDF5 loading logic.

#### File Created:
- `src/libQuantiloom/io/MODTRANLoader.hpp` (NEW)

#### Interface:
```cpp
struct MODTRANLUT {
    Vector<f32> wavelengths_nm;   // Wavelength grid
    Vector<f32> sun_radiance;     // Solar spectral radiance (W·sr⁻¹·m⁻²·nm⁻¹)
    Vector<f32> sky_radiance;     // Sky spectral radiance
    Vector<f32> transmittance;    // Atmospheric transmittance [0, 1]

    // Query methods with linear interpolation
    f32 QuerySunRadiance(f32 lambda_nm) const;
    f32 QuerySkyRadiance(f32 lambda_nm) const;
    f32 QueryTransmittance(f32 lambda_nm) const;
};

class MODTRANLoader {
public:
    MODTRANLUT Load(const String& filepath);  // PLACEHOLDER - user implements
    static MODTRANLUT CreateDummyLUT();       // For M1 testing
};
```

#### Expected HDF5 Format:
```
/wavelengths_nm       - 1D array (nm)
/sun_radiance         - 1D array (W·sr⁻¹·m⁻²·nm⁻¹)
/sky_radiance         - 1D array (W·sr⁻¹·m⁻²·nm⁻¹)
/transmittance        - 1D array [0, 1]
/metadata             - Group with attributes (altitude_m, visibility_km, etc.)
```

---

## 📊 Data Structure Changes Summary

### Material Structure (CPU/GPU aligned):

| Field | Offset | Size | Description |
|-------|--------|------|-------------|
| `spectralAlbedo` | 64 | 4 | **LEGACY** scalar albedo (M1 fallback) |
| `spectralReflectanceCurveIndex` | 68 | 4 | **NEW** index into `spectralCurves` buffer |
| `irEmissivity` | 72 | 4 | IR emissivity ε(λ) |
| `irReflectance` | 76 | 4 | IR reflectance ρ(λ) |
| `irTransmittance` | 80 | 4 | IR transmittance τ(λ) |
| `irTemperature_K` | 84 | 4 | IR surface temperature |
| **Total** | - | **88** | **+4 bytes from M1** |

### Shader Bindings:

| Binding | Type | Description |
|---------|------|-------------|
| 0 | `RWTexture2D` | Output image |
| 1 | `RaytracingAccelerationStructure` | Scene TLAS |
| 2 | `StructuredBuffer<LUTData>` | Atmospheric LUT |
| 3-9 | ... | Geometry/Material buffers |
| 10-12 | ... | IBL resources |
| **13** | `StructuredBuffer<SpectralCurveGPU>` | **NEW** Spectral curves |

---

## 🔧 How to Use the New System

### 1. Loading Spectral Curves (User Implements)

```cpp
#include "core/SpectralData.hpp"
#include "io/MODTRANLoader.hpp"

// Load measured reflectance curve from file
Vector<f32> wavelengths = {380, 400, 450, 500, ..., 780};  // nm
Vector<f32> reflectances = {0.05, 0.08, 0.12, 0.15, ..., 0.20};  // [0, 1]

SpectralCurve curve(wavelengths, reflectances);

// Convert to GPU format
SpectralCurveGPU gpuCurve = SpectralCurveGPU::FromCPU(curve);

// Upload to scene's spectral curve buffer
scene.spectralReflectanceCurves.push_back(gpuCurve);

// Assign to material
material.spectralReflectanceCurveIndex = scene.spectralReflectanceCurves.size() - 1;
```

### 2. Loading MODTRAN LUT (User Implements)

```cpp
#include "io/MODTRANLoader.hpp"

MODTRANLoader loader;

// OPTION 1: Load from HDF5 (user must implement HDF5 loading logic)
MODTRANLUT lut = loader.Load("assets/luts/modtran_clear_sky.h5");

// OPTION 2: Use dummy LUT for M1 testing
MODTRANLUT lut = MODTRANLoader::CreateDummyLUT();

// Query at specific wavelength
float sunRadiance_550 = lut.QuerySunRadiance(550.0f);  // W·sr⁻¹·m⁻²·nm⁻¹
float skyRadiance_550 = lut.QuerySkyRadiance(550.0f);
float transmittance_550 = lut.QueryTransmittance(550.0f);
```

### 3. GPU Pipeline Integration (User Implements)

**User must update `RayTracingPipeline.cpp`:**

```cpp
// Create spectral curve buffer
VkDeviceSize bufferSize = scene.spectralReflectanceCurves.size() * sizeof(SpectralCurveGPU);
spectralCurvesBuffer = CreateBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

// Upload spectral curves to GPU
UploadBuffer(spectralCurvesBuffer, scene.spectralReflectanceCurves.data(), bufferSize);

// Bind to descriptor set (binding = 13)
VkDescriptorBufferInfo spectralCurvesInfo = {
    .buffer = spectralCurvesBuffer,
    .offset = 0,
    .range = bufferSize
};

VkWriteDescriptorSet write = {
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = descriptorSet,
    .dstBinding = 13,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .pBufferInfo = &spectralCurvesInfo
};

vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
```

---

## ✅ M1 Completion Checklist

- [x] **Spectral Curve Data Structure** - CPU/GPU aligned, 64 samples, linear interpolation
- [x] **Material Spectral Reflectance Index** - Added to Material.hpp, common.hlsli, main.cpp
- [x] **Spectral Query Functions** - `spectral_query.hlsli` with GPU interpolation
- [x] **Beer-Lambert Atmospheric Attenuation** - Path-length dependent `T(λ, d) = exp(-σ_t × d)`
- [x] **Sky Radiance Hemispherical Integration** - Physical Lambertian BRDF with energy conservation
- [x] **MODTRAN LUT Loader Interface** - Empty implementation with clear TODO markers
- [x] **Shader Buffer Binding** - Binding 13 for `spectralCurves` buffer
- [x] **Struct Size Validation** - `static_assert` checks for CPU/GPU alignment

---

## 🚧 What User Must Implement

### Critical (Required for M1 validation):

1. **HDF5 MODTRAN LUT Loading** (`MODTRANLoader.hpp`)
   - Implement `MODTRANLoader::Load()` function
   - Read datasets: `/wavelengths_nm`, `/sun_radiance`, `/sky_radiance`, `/transmittance`
   - Validate data consistency

2. **Spectral Curve Buffer Upload** (`RayTracingPipeline.cpp`)
   - Create VkBuffer for `spectralCurves`
   - Upload `scene.spectralReflectanceCurves` to GPU
   - Bind to descriptor set (binding = 13)

3. **Material Upload with New Field** (`main.cpp` or scene loader)
   - When uploading materials to GPU, include `spectralReflectanceCurveIndex`
   - Ensure struct padding matches (88 bytes total)

### Optional (For M1.5 validation):

4. **Load Measured Spectral Curves** (scene asset pipeline)
   - Parse CSV/HDF5 spectral reflectance data
   - Convert to `SpectralCurve` format
   - Assign to materials via `spectralReflectanceCurveIndex`

5. **MODTRAN LUT Generation** (external tool)
   - Run MODTRAN with desired atmospheric parameters
   - Export to HDF5 format matching interface specification
   - Place in `assets/luts/` directory

---

## 🧪 Testing with Dummy Data

For immediate M1 testing without real MODTRAN data:

```cpp
// Use dummy LUT (constant radiance values)
MODTRANLUT dummyLut = MODTRANLoader::CreateDummyLUT();
// Sun: 1000 W·sr⁻¹·m⁻²·nm⁻¹ (constant)
// Sky: 100 W·sr⁻¹·m⁻²·nm⁻¹ (constant)
// Transmittance: 0.8 (20% atmospheric absorption)

// Use legacy spectralAlbedo fallback (no spectral curves needed)
material.spectralReflectanceCurveIndex = -1;  // -1 = use spectralAlbedo
material.spectralAlbedo = 0.5f;  // 50% reflectance (gray surface)
```

This allows rendering without requiring real spectral data, useful for M1 geometry/lighting validation.

---

## 📝 Notes for M1.5 Validation

To pass M1.5 validation gates (comparison with PBRT-v4/Mitsuba 3):

1. **Use Measured Spectral Curves** - Replace RGB upsampling with real measured data
2. **Load Real MODTRAN LUT** - Implement HDF5 loading for wavelength-dependent atmosphere
3. **Validate Energy Conservation** - Ensure `ε + ρ + τ ≤ 1` for all materials
4. **Cross-Check with Reference** - Compare single-wavelength renders (λ=550nm) against PBRT

---

## 🎓 Key Design Decisions

1. **Backward Compatibility**: Legacy `spectralAlbedo` scalar preserved for M1 scenes
2. **Fixed-Size GPU Arrays**: 64 samples max for predictable memory layout
3. **Linear Interpolation**: Simple and fast, sufficient for smooth spectral curves
4. **Empty Interface Pattern**: MODTRAN loader is placeholder - user implements HDF5 logic
5. **Physical Correctness**: Beer-Lambert law with path-length, not just scalar transmittance
6. **Energy Conservation**: Fresnel + metallic factors in sky integration

---

## 📚 References

- **Beer-Lambert Law**: Atmospheric optics textbooks (e.g., Bohren & Huffman)
- **MODTRAN**: https://modtran.spectral.com/ (LUT generation documentation)
- **HDF5 C++ API**: https://portal.hdfgroup.org/display/HDF5/HDF5 (for implementing loader)
- **glTF 2.0 PBR**: https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
- **Spectral Rendering**: PBRT-v4 Chapter 4 (Radiometry and Spectral Representation)

---

**Implementation Date**: 2025-11-28
**Status**: Core infrastructure complete, user integration required
**Next Milestone**: M1.5 validation with PBRT-v4 cross-comparison
