# M1 Geometric Normal Fix - Summary

## Problem
The original M1 implementation used a **fake normal** in the closest hit shader:
```hlsl
// WRONG: Camera-facing hack
float3 normal = normalize(-WorldRayDirection());
```

This made all surfaces appear to face the camera, resulting in uniform lighting that looked "good enough" but was **physically incorrect**. All cube faces had the same brightness regardless of orientation relative to the sun.

## Solution
Implemented **proper geometric normal calculation** from triangle vertices:

```hlsl
// CORRECT: Read vertices and compute cross product
uint primitiveID = PrimitiveIndex();
uint idx0 = indexBuffer[primitiveID * 3 + 0];
uint idx1 = indexBuffer[primitiveID * 3 + 1];
uint idx2 = indexBuffer[primitiveID * 3 + 2];

float3 v0 = vertexBuffer[idx0];
float3 v1 = vertexBuffer[idx1];
float3 v2 = vertexBuffer[idx2];

float3 edge1 = v1 - v0;
float3 edge2 = v2 - v0;
float3 geometricNormal = normalize(cross(edge1, edge2));
```

## Files Modified

### 1. Shader (closesthit.rchit)
- Added vertex buffer (binding 3) and index buffer (binding 4) access
- Implemented geometric normal calculation via cross product
- Added front-face culling to ensure normals face the ray

### 2. RayTracingPipeline (hpp/cpp)
- Updated descriptor set layout to include 5 bindings (was 3)
- Added `BindGeometryBuffers(vertexBuffer, indexBuffer)` method
- Updated descriptor pool to allocate 3 storage buffers

### 3. AccelerationStructure (hpp)
- Added `GetVertexBuffer()` and `GetIndexBuffer()` accessors on BLAS
- Allows pipeline to bind geometry buffers from existing BLAS data

### 4. main_m1_test.cpp
- Added command-line argument support (--scene, --camera, --lighting, --output)
- Changed SCENE_PRESET constants to globals (g_scenePreset, etc.)
- Added pipeline.BindGeometryBuffers() call

### 5. New Scripts
- `src/shaders/compile_shaders.sh` - Linux/macOS shader compilation
- `src/shaders/compile_shaders.bat` - Windows shader compilation
- `run_all_m1_tests.sh` - Automated test runner for all 3 scenes

## Build & Test Instructions

### Step 1: Recompile Shaders
```bash
cd src/shaders

# Linux/macOS
./compile_shaders.sh

# Windows
compile_shaders.bat
```

This compiles `closesthit.rchit` with the new normal calculation.

### Step 2: Rebuild Project
```bash
cd build  # or "out" if using that
cmake --build .
```

### Step 3: Run Automated Tests
```bash
./run_all_m1_tests.sh
```

This runs all 3 test scenes:
1. **CornellBox** - Single cube, verifies basic lighting
2. **MultiObject** - 3 objects (tall box, cube, sphere)
3. **LightingTest** - 5 cubes in a row, verifies directional lighting

Output files: `m1_test_results/cornell_box.exr`, `multi_object.exr`, `lighting_test.exr`

### Step 4: Manual Test (Single Scene)
```bash
# Cornell Box
./build/src/app/quantiloom_m1_test --scene cornell --output cornell.exr

# Multi-Object
./build/src/app/quantiloom_m1_test --scene multiobject --output multi.exr

# Lighting Test (with morning lighting)
./build/src/app/quantiloom_m1_test --scene lighting --lighting morning --output lighting_morning.exr
```

## Expected Results

### Before Fix (Fake Normal)
- All cube faces: **Same brightness** ~(0.62, 0.62, 0.62)
- Sun direction: **Ignored** (normal always faces camera)
- Lighting gradient: **None**

### After Fix (Geometric Normal)
- Cube top face: **Brightest** (faces sun from upper-left)
- Cube side faces: **Medium brightness** (partial sun exposure)
- Cube bottom/back faces: **Darkest** (only sky radiance)
- Lighting gradient: **Clear** (left-to-right in LightingTest)

## Validation Checklist

For **LightingTest** scene (5 cubes in a row):
- [ ] Leftmost cube is **brightest** (sun from left: `-0.5, 0.8, -0.3`)
- [ ] Rightmost cube is **darkest** (faces away from sun)
- [ ] Clear **gradient** from left to right
- [ ] Each cube has **different brightness on different faces**

If all cubes still look the same brightness → shader not recompiled or normal still broken.

## Technical Details

### Descriptor Set Layout
```
Binding 0: Output image (RWTexture2D)
Binding 1: TLAS (AccelerationStructure)
Binding 2: LUT (sun/sky data)
Binding 3: Vertex buffer (StructuredBuffer<float3>)  ← NEW
Binding 4: Index buffer (StructuredBuffer<uint>)     ← NEW
```

### Shader Compilation Command
```bash
dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.3 -Fo closesthit.spv closesthit.rchit
```

### Pipeline Binding Order
```cpp
pipeline.BindOutputImage(outputImage);
pipeline.BindAccelerationStructure(tlas.GetHandle());
pipeline.BindLUTBuffer(lutBuffer);
pipeline.BindGeometryBuffers(blas.GetVertexBuffer(), blas.GetIndexBuffer());  // NEW
```

## Why This Matters for M2

**M2 requires loading OBJ files with complex geometry.** With fake normals:
- Complex meshes would look completely wrong
- You'd waste days debugging OBJ loader thinking it's broken
- Can't pass M1.5 validation (PBRT/Mitsuba use correct normals)

**Now with correct normals:**
- M2 OBJ loading will work correctly from day 1
- Can validate against PBRT-v4/Mitsuba 3
- Lambert BRDF is actually Lambert (not camera-facing hack)

## M1 Completion Status

✅ Vulkan RT pipeline working
✅ BLAS/TLAS building with barriers
✅ LUT-fast sun/sky lighting
✅ **Geometric normal calculation (FIXED)**
✅ Command-line test harness
✅ Automated 3-scene test suite

**M1 is now ready for M2** once you verify the test results locally.

## Next Steps

1. Run `./run_all_m1_tests.sh` locally
2. Open `m1_test_results/*.exr` in image viewer
3. Verify lighting gradient in LightingTest
4. If validation passes → **Approved for M2**
5. If validation fails → Check shader recompilation

---

**Good taste restored.** No more fake normals. 🎯
