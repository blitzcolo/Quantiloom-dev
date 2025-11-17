# Quantiloom M2 OBJ Loading Workflow

**Purpose:** Guide for loading external OBJ files in M2 milestone, enabling complex scene testing with ground planes, objects, and realistic geometry.

**Status:** M2 Planning (M1 uses procedural geometry only)

---

## Table of Contents

1. [Overview](#overview)
2. [M1 vs M2 Scene Workflow](#m1-vs-m2-scene-workflow)
3. [OBJ File Preparation](#obj-file-preparation)
4. [Loader Integration](#loader-integration)
5. [Multi-Object Scene Assembly](#multi-object-scene-assembly)
6. [Material Mapping](#material-mapping)
7. [Coordinate System Conversion](#coordinate-system-conversion)
8. [Example Workflows](#example-workflows)
9. [Debugging OBJ Files](#debugging-obj-files)

---

## 1. Overview

### Why OBJ Loading in M2?

**M1 Limitations:**
- Hardcoded procedural geometry (boxes, spheres, ground planes)
- No external file loading
- Manual mesh construction in C++

**M2 Goals:**
- Load complex scenes from external OBJ files
- Support multiple objects with transforms
- Enable artist-created content (Blender, Maya, etc.)
- Test realistic geometry (high poly count, complex shapes)

### Recommended OBJ Loader Libraries

**Option 1: tinyobjloader (Recommended)**
- Header-only, lightweight (~2000 LOC)
- MIT license, no dependencies
- URL: https://github.com/tinyobjloader/tinyobjloader
- **Pros:** Simple API, well-tested, fast
- **Cons:** Limited material support (no PBR)

**Option 2: Assimp**
- Full-featured asset importer (OBJ, FBX, GLTF, etc.)
- URL: https://github.com/assimp/assimp
- **Pros:** Supports many formats, coordinate transforms built-in
- **Cons:** Large library, slower build times

**Quantiloom Recommendation:** Start with **tinyobjloader** for M2, migrate to Assimp in M3+ if GLTF support needed.

---

## 2. M1 vs M2 Scene Workflow

### M1 Workflow (Current)

```cpp
// src/app/main_m1_test.cpp
Mesh sceneMesh = TestScenes::CreateCornellBoxScene();  // Procedural
BLAS blas(context, sceneMesh);
TLAS tlas(context);
tlas.AddInstance(blas);
```

**Limitations:**
- All geometry hardcoded
- Single BLAS for entire scene
- No transforms (everything at identity)

### M2 Workflow (Planned)

```cpp
// Load OBJ files
OBJLoader loader;
Mesh ground = loader.Load("assets/ground.obj");
Mesh cube   = loader.Load("assets/cube.obj");
Mesh sphere = loader.Load("assets/sphere.obj");

// Create separate BLAS for each object (for instancing)
BLAS groundBLAS(context, ground);
BLAS cubeBLAS(context, cube);
BLAS sphereBLAS(context, sphere);

// Build TLAS with transforms
TLAS tlas(context);
tlas.AddInstance(groundBLAS, glm::mat4(1.0f));              // Identity
tlas.AddInstance(cubeBLAS, glm::translate(vec3(-2, 1, 0))); // Left
tlas.AddInstance(cubeBLAS, glm::translate(vec3(2, 1, 0)));  // Right (reuse BLAS)
tlas.AddInstance(sphereBLAS, glm::translate(vec3(0, 2, 0)));
tlas.Build(cmd);
```

**Advantages:**
- External content creation
- Per-object transforms
- BLAS reuse (instancing)
- Easy to add/remove objects

---

## 3. OBJ File Preparation

### 3.1 Export Settings (Blender → OBJ)

**Coordinate System:**
- Quantiloom uses Y-up, right-handed
- Blender uses Z-up by default

**Blender OBJ Export Settings:**
```
File > Export > Wavefront (.obj)

☑ Selection Only (if exporting single object)
☐ Animation (not supported in M2)
☑ Apply Modifiers (important!)
☐ Include UVs (M2 has no textures yet)
☐ Write Normals (we compute them in shader)
☐ Write Materials (M2 materials via code)
☐ Triangulate Faces (some loaders require this)
☑ Objects as OBJ Objects (preserves names)

Forward: -Y Forward (converts to Quantiloom coord system)
Up:      Z Up
```

### 3.2 File Structure

**Single Object OBJ:**
```obj
# ground.obj
o ground_plane
v -5.0 0.0 -5.0
v  5.0 0.0 -5.0
v  5.0 0.0  5.0
v -5.0 0.0  5.0
f 1 2 3
f 1 3 4
```

**Multi-Object OBJ:**
```obj
# scene.obj
o ground
v -10.0 0.0 -10.0
# ... ground vertices ...
f 1 2 3
# ... ground faces ...

o cube
v -0.5 0.0 -0.5
# ... cube vertices ...
f N N+1 N+2  # (indices offset by ground vertex count)
# ... cube faces ...

o sphere
v 0.0 1.0 0.0
# ... sphere vertices ...
```

**Quantiloom Approach:**
- **M2:** Export each object as separate OBJ file (easier per-object BLAS)
- **M3+:** Support multi-object OBJ for entire scenes

### 3.3 Validation Checklist

Before importing OBJ to Quantiloom:

- [ ] All faces are triangles (or loader triangulates)
- [ ] Counter-clockwise winding (outward normals)
- [ ] No degenerate triangles (zero area)
- [ ] Vertex positions in meters (1.0 = 1 meter)
- [ ] Coordinate system matches Quantiloom (Y-up)
- [ ] No extremely large polygons (> 10m edges)
- [ ] File encoding is ASCII (not binary FBX, etc.)

**Validation Tool (Linux/macOS):**
```bash
# Check OBJ file sanity
grep "^v " ground.obj | wc -l  # Count vertices
grep "^f " ground.obj | wc -l  # Count faces
grep "^vt" ground.obj | wc -l  # Texture coords (should be 0 for M2)
```

---

## 4. Loader Integration

### 4.1 Add tinyobjloader to Project

**CMakeLists.txt:**
```cmake
# Add tinyobjloader submodule or FetchContent
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobjloader)

# Link to executable
target_link_libraries(quantiloom_m2_test PRIVATE tinyobjloader)
```

### 4.2 Wrapper Class (Recommended)

Create `src/io/OBJLoader.hpp`:

```cpp
#pragma once

#include "scene/Mesh.hpp"
#include <string>
#include <vector>

namespace quantiloom {

class OBJLoader {
public:
    // Load single mesh from OBJ file
    // If OBJ contains multiple objects, returns first object
    static Mesh Load(const std::string& filepath);

    // Load all meshes from multi-object OBJ
    static std::vector<Mesh> LoadAll(const std::string& filepath);

    // Load with coordinate transform (e.g., Blender Z-up → Y-up)
    static Mesh LoadWithTransform(const std::string& filepath,
                                   const glm::mat4& transform);
};

} // namespace quantiloom
```

### 4.3 Implementation Example

`src/io/OBJLoader.cpp`:

```cpp
#include "OBJLoader.hpp"
#include "core/Log.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace quantiloom {

Mesh OBJLoader::Load(const std::string& filepath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Load OBJ file
    bool success = tinyobj::LoadObj(&attrib, &shapes, &materials,
                                     &warn, &err, filepath.c_str());

    if (!warn.empty()) {
        QL_LOG_WARN("OBJ Loader: {}", warn);
    }

    if (!success) {
        QL_LOG_ERROR("Failed to load OBJ: {}", err);
        throw std::runtime_error("OBJ load failed");
    }

    if (shapes.empty()) {
        throw std::runtime_error("OBJ contains no shapes");
    }

    // Convert first shape to Mesh
    Mesh mesh;
    mesh.name = shapes[0].name;

    // Extract vertex positions
    for (size_t v = 0; v < attrib.vertices.size() / 3; ++v) {
        mesh.positions.push_back({
            attrib.vertices[3 * v + 0],  // X
            attrib.vertices[3 * v + 1],  // Y
            attrib.vertices[3 * v + 2]   // Z
        });
    }

    // Extract indices (convert to triangles if needed)
    for (const auto& index : shapes[0].mesh.indices) {
        mesh.indices.push_back(static_cast<u32>(index.vertex_index));
    }

    QL_LOG_INFO("Loaded OBJ: {} ({} verts, {} tris)",
                filepath, mesh.positions.size(), mesh.indices.size() / 3);

    return mesh;
}

std::vector<Mesh> OBJLoader::LoadAll(const std::string& filepath) {
    // Similar to Load(), but iterate over all shapes
    // (Implementation left as exercise)
}

} // namespace quantiloom
```

---

## 5. Multi-Object Scene Assembly

### 5.1 Scene Manifest (Recommended for M2)

Create a simple text file to describe scene layout:

**assets/scenes/test_scene.txt:**
```
# Quantiloom Scene Manifest
# Format: object_name  obj_file  transform

ground    assets/ground.obj   identity
cube1     assets/cube.obj     translate(-2.0, 1.0, 0.0)
cube2     assets/cube.obj     translate(2.0, 1.0, 0.0)
sphere    assets/sphere.obj   translate(0.0, 2.0, 0.0) scale(0.5)
```

### 5.2 Scene Loader (Pseudocode)

```cpp
struct SceneObject {
    std::string name;
    std::string objFile;
    glm::mat4 transform;
};

std::vector<SceneObject> LoadSceneManifest(const std::string& manifestFile) {
    // Parse manifest file
    // For each line:
    //   1. Load OBJ
    //   2. Build BLAS
    //   3. Add to TLAS with transform
}
```

---

## 6. Material Mapping

### M2 Material System

**Challenge:** OBJ materials (MTL files) don't map directly to Quantiloom spectral materials.

**M2 Approach:**
1. Ignore OBJ material names
2. Assign materials via `InstanceCustomIndex` in TLAS
3. Load material properties from separate config (TOML or JSON)

**Example:**

**assets/materials.toml:**
```toml
[[material]]
id = 0
name = "ground_gray"
albedo = [0.5, 0.5, 0.5]

[[material]]
id = 1
name = "cube_red"
albedo = [0.8, 0.2, 0.2]

[[material]]
id = 2
name = "sphere_white"
albedo = [0.9, 0.9, 0.9]
```

**TLAS Setup:**
```cpp
// Assign material IDs via InstanceCustomIndex
tlas.AddInstance(groundBLAS, identity, materialID=0);
tlas.AddInstance(cubeBLAS, translate(-2,1,0), materialID=1);
tlas.AddInstance(sphereBLAS, translate(0,2,0), materialID=2);
```

**Shader Access:**
```hlsl
// In closesthit.rchit
uint materialID = InstanceCustomIndex();
MaterialData mat = materials[materialID];
float3 albedo = mat.albedo;
```

---

## 7. Coordinate System Conversion

### 7.1 Common Transforms

**Blender Z-up → Quantiloom Y-up:**
```cpp
glm::mat4 BlenderToQuantiloom() {
    // Rotate -90° around X-axis
    return glm::rotate(glm::mat4(1.0f),
                       glm::radians(-90.0f),
                       glm::vec3(1.0f, 0.0f, 0.0f));
}

// Usage:
Mesh mesh = OBJLoader::LoadWithTransform("blender_export.obj",
                                          BlenderToQuantiloom());
```

**3ds Max Z-up → Quantiloom Y-up:**
```cpp
// Same as Blender
```

**Maya Y-up → Quantiloom Y-up:**
```cpp
// No conversion needed (Maya default matches Quantiloom)
```

### 7.2 Automatic Detection

```cpp
// Detect coordinate system from OBJ AABB
Mesh DetectAndConvert(const std::string& path) {
    Mesh mesh = OBJLoader::Load(path);

    // Compute bounding box
    vec3 minBounds = ComputeMinBounds(mesh.positions);
    vec3 maxBounds = ComputeMaxBounds(mesh.positions);

    // Heuristic: If Y extent is very small, likely Z-up
    float yExtent = maxBounds.y - minBounds.y;
    float zExtent = maxBounds.z - minBounds.z;

    if (yExtent < 0.1f && zExtent > 1.0f) {
        QL_LOG_WARN("Detected Z-up coordinates, applying transform");
        ApplyTransform(mesh, BlenderToQuantiloom());
    }

    return mesh;
}
```

---

## 8. Example Workflows

### Example 1: Simple Ground + Cube Scene

**Step 1: Create OBJ files in Blender**

```python
# Blender Python script: export_test_scene.py
import bpy

# Create ground plane
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
bpy.ops.export_scene.obj(filepath='ground.obj',
                          use_selection=True,
                          axis_forward='-Y', axis_up='Z')

# Create cube
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bpy.ops.export_scene.obj(filepath='cube.obj',
                          use_selection=True,
                          axis_forward='-Y', axis_up='Z')
```

**Step 2: Load in Quantiloom M2**

```cpp
// src/app/main_m2_test.cpp
Mesh ground = OBJLoader::Load("assets/ground.obj");
Mesh cube = OBJLoader::Load("assets/cube.obj");

BLAS groundBLAS(context, ground);
BLAS cubeBLAS(context, cube);

TLAS tlas(context);
tlas.AddInstance(groundBLAS, glm::mat4(1.0f));
tlas.AddInstance(cubeBLAS, glm::translate(vec3(0, 0.5, 0)));
tlas.Build(cmd);
```

### Example 2: Stanford Bunny Test

**Download Stanford Bunny:**
```bash
wget https://graphics.stanford.edu/data/3Dscanrep/bunny.tar.gz
tar -xzf bunny.tar.gz
# Convert PLY to OBJ (using MeshLab or Blender)
```

**Load in Quantiloom:**
```cpp
Mesh bunny = OBJLoader::Load("assets/bunny.obj");

// Stanford bunny is tiny (in mm), scale up to meters
glm::mat4 scaleUp = glm::scale(vec3(10.0f));

BLAS bunnyBLAS(context, bunny);
TLAS tlas(context);
tlas.AddInstance(bunnyBLAS, scaleUp);
```

### Example 3: Multi-Room Scene

**Workflow:**
1. Model entire scene in Blender (floor, walls, furniture)
2. Export each room as separate OBJ
3. Load all OBJs and combine in Quantiloom with transforms

```cpp
Mesh floor = OBJLoader::Load("room_floor.obj");
Mesh walls = OBJLoader::Load("room_walls.obj");
Mesh table = OBJLoader::Load("furniture_table.obj");
Mesh chair = OBJLoader::Load("furniture_chair.obj");

// Build scene
TLAS tlas(context);
tlas.AddInstance(BLAS(context, floor), identity);
tlas.AddInstance(BLAS(context, walls), identity);
tlas.AddInstance(BLAS(context, table), translate(1, 0, 2));
tlas.AddInstance(BLAS(context, chair), translate(1.5, 0, 1.5));
// ... add more chairs with different transforms
```

---

## 9. Debugging OBJ Files

### 9.1 Validation Tools

**MeshLab (GUI):**
- Open OBJ file
- Check: `Filters > Quality Measures > Compute Geometric Measures`
- Look for: degenerate faces, non-manifold edges

**Blender (Python):**
```python
import bpy

mesh = bpy.data.objects['Cube'].data

# Check for non-manifold vertices
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
selected = [v for v in mesh.vertices if v.select]
print(f"Non-manifold vertices: {len(selected)}")

# Check winding order
bpy.ops.mesh.normals_make_consistent(inside=False)
```

### 9.2 Common Issues

**Issue 1: All faces appear black**

**Cause:** Winding order is reversed (clockwise instead of CCW)

**Fix:**
```cpp
// In OBJLoader.cpp, reverse winding after load
for (size_t i = 0; i < mesh.indices.size(); i += 3) {
    std::swap(mesh.indices[i + 1], mesh.indices[i + 2]);
}
```

**Issue 2: Objects at wrong scale**

**Cause:** Blender units don't match Quantiloom meters

**Fix:**
```cpp
// Apply scale transform
glm::mat4 scaleTransform = glm::scale(vec3(0.01f)); // cm to m
tlas.AddInstance(blas, scaleTransform);
```

**Issue 3: Objects upside-down**

**Cause:** Coordinate system mismatch (Z-up vs Y-up)

**Fix:**
```cpp
Mesh mesh = OBJLoader::LoadWithTransform("model.obj",
                                          BlenderToQuantiloom());
```

### 9.3 Debug Visualization

**Render bounding boxes:**
```cpp
// Add to closesthit shader for debugging
float3 aabbMin = ObjectRayOrigin() + ObjectRayDirection() * RayTMin();
float3 aabbMax = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();

// Color code by AABB size
float size = length(aabbMax - aabbMin);
payload.radiance = float3(size, size, size) * 0.1;
```

---

## Quick Reference

| Task | M1 Status | M2 Status |
|------|-----------|-----------|
| Procedural geometry | ✅ Implemented | ✅ Keep for tests |
| OBJ loading | ❌ Not supported | 🔄 To implement |
| Multi-object scenes | ⚠️ Manual merge | ✅ TLAS instancing |
| Per-object materials | ❌ Single material | ✅ Via InstanceCustomIndex |
| Coordinate transforms | ⚠️ Manual in code | ✅ Per-instance matrix |

---

## Next Steps for M2 Implementation

**Priority 1 (Required for OBJ loading):**
1. Integrate tinyobjloader to CMake
2. Implement `OBJLoader::Load()` wrapper
3. Test with simple cube.obj and ground.obj

**Priority 2 (Scene assembly):**
1. Support per-instance transforms in TLAS
2. Load material configs from TOML
3. Bind material buffer to shader

**Priority 3 (Production):**
1. Implement scene manifest parser
2. Add coordinate system auto-detection
3. Support OBJ validation on load

---

## Related Documentation

- **M1_Scene_Construction_Guide.md**: OBJ file requirements and winding order
- **时间表.md**: M2 milestone timeline and requirements
- **软件需规SRS.md**: Material system architecture (Section 7.3)

---

**End of M2 OBJ Loading Workflow**
