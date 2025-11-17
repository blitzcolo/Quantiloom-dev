# Quantiloom M1 Scene Construction Guide

**Purpose:** This guide provides practical instructions for building and testing scenes with the Quantiloom HS-core ray tracing system (M1 milestone).

**Audience:** Developers testing the M1 prototype and preparing for M2+ scene workflow.

---

## Table of Contents

1. [Coordinate System & Units](#coordinate-system--units)
2. [Scene Components](#scene-components)
3. [Camera Configuration](#camera-configuration)
4. [Lighting Setup (Sun & Sky)](#lighting-setup-sun--sky)
5. [Material System (M1 Limitations)](#material-system-m1-limitations)
6. [OBJ File Requirements](#obj-file-requirements)
7. [Example Scenes](#example-scenes)
8. [Validation & Debugging](#validation--debugging)

---

## 1. Coordinate System & Units

### Coordinate System (Right-Handed)
- **X-axis**: Right (increasing X = right)
- **Y-axis**: Up (increasing Y = up)
- **Z-axis**: Forward (increasing Z = toward camera, out of screen)

```
     Y (up)
     |
     |
     +----> X (right)
    /
   Z (forward, toward camera)
```

### Units
- **Position**: meters (m)
- **Radiance**: W·sr⁻¹·m⁻² (spectral radiance in M2+)
- **All geometry should be in world space** (no automatic scaling)

### Convention Check
Run this test to verify coordinate system:
```cpp
// Camera at origin, looking down +Z
// Box at (0, 0, 5) should appear centered
// Box at (1, 0, 5) should appear on the RIGHT side of screen
// Box at (0, 1, 5) should appear on the TOP of screen
```

---

## 2. Scene Components

### Minimum Viable Scene
A M1 test scene requires:
1. **Geometry** (at least one mesh)
2. **Camera** (position + look direction)
3. **Lighting** (sun direction + sky radiance)

### Recommended Test Scene Structure

```
Scene Root
├── Ground Plane (large quad, Y=0)
├── Reference Objects
│   ├── Unit Cube (1x1x1 at origin)
│   ├── Sphere (radius 1.0 at (2, 1, 0))
│   └── Tall Box (1x2x1 at (-2, 0, 0))
└── Camera (positioned to view all objects)
```

### Ground Plane Guidelines
- **Size**: At least 10x10 m to avoid seeing edges
- **Position**: Y = 0 (world origin on ground)
- **Color**: Neutral gray (albedo 0.5) for lighting reference
- **Geometry**: 2 triangles forming a quad

```cpp
// Ground plane vertices (10x10 m, centered at origin)
{-5.0f, 0.0f, -5.0f},  // Far-left
{ 5.0f, 0.0f, -5.0f},  // Far-right
{ 5.0f, 0.0f,  5.0f},  // Near-right
{-5.0f, 0.0f,  5.0f},  // Near-left

// Indices (counter-clockwise winding for upward normal)
0, 1, 2,  0, 2, 3
```

---

## 3. Camera Configuration

### Camera Parameters (M1 Pinhole Model)

```cpp
struct CameraConfig {
    glm::vec3 position;      // Camera eye position (world space)
    glm::vec3 lookAt;        // Point camera is looking at
    glm::vec3 up;            // Up vector (usually (0,1,0))
    float fovY;              // Vertical field-of-view (degrees)
    float aspectRatio;       // width / height
};
```

### Recommended Test Cameras

**1. Default Overview (Good for initial testing)**
```cpp
position:  (0.0f, 2.0f, -8.0f)  // Elevated, behind scene
lookAt:    (0.0f, 1.0f,  0.0f)  // Looking at origin, slightly up
up:        (0.0f, 1.0f,  0.0f)
fovY:      60.0 degrees
```

**2. Ground-Level View (Test occlusion)**
```cpp
position:  (0.0f, 0.5f, -5.0f)  // Low, human eye height
lookAt:    (0.0f, 0.5f,  0.0f)
up:        (0.0f, 1.0f,  0.0f)
fovY:      70.0 degrees          // Wider FOV
```

**3. Top-Down (Verify layout)**
```cpp
position:  (0.0f, 10.0f, 0.0f)  // Directly above
lookAt:    (0.0f,  0.0f, 0.0f)
up:        (0.0f,  0.0f, 1.0f)  // Z-forward is "up" in top-down
fovY:      45.0 degrees
```

### Camera Direction Calculation (Current M1)
```cpp
// In raygen shader, convert pixel to ray direction:
float2 ndc = uv * 2.0 - 1.0;  // [-1, 1]
float3 direction = normalize(
    float3(ndc.x * aspectRatio, -ndc.y, 1.0)
);
```

**Important:** M1 uses hardcoded pinhole camera. For custom camera, you need to modify `raygen.rgen` to pass camera matrix via push constants or SSBO.

---

## 4. Lighting Setup (Sun & Sky)

### Sun Direction Convention

**Direction Vector**: Points FROM surface TO sun (NOT from sun to surface)

```
Sun Direction = normalize(sunPosition - surfacePosition)

Example:
- Sun at upper-left:  (-0.5, 0.8, -0.3)  ✓
- Sun overhead:       ( 0.0, 1.0,  0.0)  ✓
- Sun below horizon:  ( 0.0,-1.0,  0.0)  ✗ (all surfaces dark)
```

### Recommended Sun Configurations

**1. Standard 3-Point Lighting (Key Light)**
```cpp
sunDirection = normalize(vec3(-0.5f, 0.8f, -0.3f));
sunRadiance  = vec3(3.0f, 3.0f, 3.0f);  // Bright white
```

**2. Morning/Evening Light (Warm)**
```cpp
sunDirection = normalize(vec3(0.7f, 0.3f, -0.2f));  // Low angle
sunRadiance  = vec3(4.0f, 3.5f, 2.8f);  // Warm (more red)
```

**3. Noon (Overhead)**
```cpp
sunDirection = normalize(vec3(0.0f, 1.0f, 0.1f));
sunRadiance  = vec3(5.0f, 5.0f, 5.0f);  // Very bright
```

**4. Backlight (Rim Lighting Test)**
```cpp
sunDirection = normalize(vec3(0.0f, 0.5f, 1.0f));  // Behind objects
sunRadiance  = vec3(6.0f, 6.0f, 6.0f);  // Strong for silhouette
```

### Sky Radiance

```cpp
// Clear blue sky (standard)
skyRadiance = vec3(0.3f, 0.5f, 0.8f);

// Overcast (gray, uniform)
skyRadiance = vec3(0.6f, 0.6f, 0.6f);

// Night sky (very dark)
skyRadiance = vec3(0.01f, 0.01f, 0.02f);

// Sunset sky (warm)
skyRadiance = vec3(0.8f, 0.4f, 0.2f);
```

---

## 5. Material System (M1 Limitations)

### M1 Material Model

**Current Implementation:**
- **BRDF**: Lambert only (diffuse)
- **Albedo**: Hardcoded in shader as `float3(0.8, 0.8, 0.8)`
- **Per-Object Materials**: NOT SUPPORTED in M1

**Formula:**
```hlsl
BRDF = albedo / π
L_out = BRDF × L_sun × max(N·L, 0)
```

### Testing Different Albedos

To test different materials in M1, you must **recompile the shader** with different albedo values:

```hlsl
// In closesthit.rchit, line 33:

// White (high reflectance)
float3 albedo = float3(1.0, 1.0, 1.0);

// Gray (neutral)
float3 albedo = float3(0.5, 0.5, 0.5);

// Dark (low reflectance)
float3 albedo = float3(0.2, 0.2, 0.2);

// Red surface
float3 albedo = float3(0.8, 0.2, 0.2);
```

### M2+ Material System (Planned)

M2 will support per-instance materials via `InstanceCustomIndex`:

```hlsl
// M2+ (future):
uint materialID = InstanceCustomIndex();
MaterialData mat = materials[materialID];
float3 albedo = mat.albedo;
```

**Requirements for M2:**
- Material buffer (SSBO) bound to pipeline
- InstanceCustomIndex set during TLAS build
- Material library (HDF5 or binary)

---

## 6. OBJ File Requirements

### Supported OBJ Features (M1)

✓ **Vertices** (`v x y z`)
✓ **Face indices** (`f v1 v2 v3`)
✓ **Multiple objects** (via `o object_name`)
✗ **Normals** (`vn`) - ignored (computed from geometry in M2+)
✗ **Texture coords** (`vt`) - ignored (no textures in M1)
✗ **Materials** (`mtllib`, `usemtl`) - ignored (single material in M1)

### OBJ Format Requirements

```obj
# Ground plane (10x10 m quad)
o ground_plane
v -5.0 0.0 -5.0
v  5.0 0.0 -5.0
v  5.0 0.0  5.0
v -5.0 0.0  5.0
f 1 2 3
f 1 3 4

# Unit cube at origin
o cube
v -0.5 0.0 -0.5
v  0.5 0.0 -0.5
v  0.5 0.0  0.5
v -0.5 0.0  0.5
v -0.5 1.0 -0.5
v  0.5 1.0 -0.5
v  0.5 1.0  0.5
v -0.5 1.0  0.5

# Faces (12 triangles, 6 faces)
f 1 2 3
f 1 3 4
# ... (top, left, right, back, front)
```

### **CRITICAL:** Winding Order

- **Counter-clockwise** (CCW) winding for outward-facing normals
- Looking at a triangle from outside, vertices should go CCW

```
Correct (CCW, normal points out):
   2
   |\
   | \
   |  \
   1---3

Incorrect (CW, normal points in):
   2
   |/
   |
   |  /
   1---3
```

### Coordinate Transform (Blender to Quantiloom)

If exporting from Blender:
- **Blender uses Z-up**, Quantiloom uses Y-up
- Apply transform: Rotate -90° around X-axis
- Or use: `Forward=-Y, Up=Z` in Blender OBJ export settings

---

## 7. Example Scenes

### Example 1: Minimal Test Scene (Cornell Box Style)

**Purpose:** Verify basic lighting and geometry

```
Scene layout (top-down):

   Sky (blue)

   +-------+  ← Back wall (Y=0 to Y=2, Z=3)
   |       |
   | [Box] |  ← Cube (0.5x0.5x0.5 at center)
   |       |
   +-------+  ← Front (open, camera here)

   Ground plane (10x10 m)
```

**Code:**
```cpp
// In CreateTestScene():
Mesh ground = CreateGroundPlane(10.0f);  // 10x10 m
Mesh cube = CreateCube(0.5f);           // 0.5 m sides

// Camera
cameraPos = vec3(0, 1, -3);   // In front
lookAt = vec3(0, 0.5, 0);     // Looking at cube center

// Sun
sunDir = normalize(vec3(-0.5, 0.8, -0.3));  // Upper-left
sunRadiance = vec3(3.0, 3.0, 3.0);
```

**Expected Output:**
- Cube appears gray, lit from upper-left
- Ground visible around cube
- Blue sky in background

---

### Example 2: Multi-Object Scene

**Purpose:** Test occlusion and multiple objects

```
Scene layout (side view):

Sky ┌────────────────┐
    │                │
    │  [Tall]  [Cube] [Sphere]
    │  │        │      O
    └──┴────────┴──────┴────┘ Ground
```

**Objects:**
1. Ground plane (10x10 m)
2. Tall box (1x2x1 at (-2, 0, 0))
3. Cube (1x1x1 at (0, 0, 0))
4. Sphere (radius 0.5 at (2, 0.5, 0)) - approximated with icosphere

**Camera:**
```cpp
position: (0, 1.5, -5)   // Slightly elevated
lookAt:   (0, 0.5, 0)    // Center of scene
fovY:     60.0
```

**Expected:** All three objects visible, with proper occlusion if overlapping from camera view.

---

### Example 3: Lighting Angle Test

**Purpose:** Verify Lambert BRDF correctness

Create 5 identical cubes in a row:
```
[Cube] [Cube] [Cube] [Cube] [Cube]
  -4     -2      0      2      4   (X positions)
```

Sun from left: `sunDir = normalize(vec3(-1, 0.5, 0))`

**Expected:**
- Leftmost cube: Brightest (faces sun)
- Rightmost cube: Darkest (faces away)
- Gradient from left to right

---

## 8. Validation & Debugging

### Visual Checks

**1. Sky Color Sanity**
- Background should match `skyRadiance` exactly
- No sky visible = all rays hit geometry (check camera position)

**2. Lighting Direction**
- Move sun direction, verify objects light/dark changes
- `sunDir.y > 0` = sun above horizon (objects should be lit)
- `sunDir.y < 0` = sun below horizon (objects should be dark)

**3. Geometric Correctness**
- No black spots = normals correct
- No inside-out faces = winding order correct
- Sharp edges visible = geometry not degenerate

### Debugging Workflows

**Problem: All geometry appears black**
- Check `sunDirection.y > 0` (sun above horizon)
- Verify `sunRadiance` not zero
- Check shader normal calculation

**Problem: Some faces black, others lit**
- Check triangle winding order (should be CCW)
- Ensure all faces have outward normals

**Problem: Sky not visible**
- Camera inside geometry? Move camera back
- FOV too narrow? Increase to 60-90 degrees

**Problem: Objects too bright/dark**
- Adjust `sunRadiance` (typical range: 1.0 - 5.0)
- Check albedo value in shader (should be 0.0 - 1.0)

### Performance Profiling

Monitor frame render time (from logs):
```
[info]   Frame rendered (800x600)  <- Check timestamp delta
```

**Expected M1 Performance:**
- 800x600, 10 triangles: < 10 ms
- 800x600, 1000 triangles: < 50 ms
- 1920x1080, 10000 triangles: < 200 ms

If slower:
- Too many triangles in view
- BLAS not compacted
- Validation layers enabled (disable in Release)

---

## Quick Reference Table

| Parameter | Recommended Range | M1 Default |
|-----------|-------------------|------------|
| Ground size | 10x10 to 100x100 m | 10x10 m |
| Object size | 0.5 to 5.0 m | 1.0 m |
| Camera height | 0.5 to 10.0 m | 2.0 m |
| Camera distance | 3.0 to 20.0 m | 8.0 m |
| Sun elevation (Y) | 0.3 to 1.0 | 0.8 |
| Sun radiance | 1.0 to 10.0 | 3.0 |
| Sky radiance | 0.1 to 1.0 | 0.3-0.8 |
| Albedo (M1) | 0.1 to 1.0 | 0.8 |

---

## Next Steps (M2+)

For full scene workflow in M2:
1. Integrate Assimp/tinygltf for OBJ/GLTF loading
2. Implement material system with per-instance or per-triangle materials
3. Add camera control (TOML config or runtime controls)
4. Support multiple light sources
5. Implement shadow rays (NEE)

See `时间表.md` for M2 milestone details.

---

**End of M1 Scene Construction Guide**
