# M1 Validation Phase - Migration Guide

## Summary of Changes

This guide documents all changes made to fix M1 phase issues before moving to M2.

---

## ✅ COMPLETED Changes

### 1. Fixed BLAS Memory Type (GPU_ONLY with Staging Buffers)
**File**: `src/libQuantiloom/renderer/AccelerationStructure.cpp`

**Change**: Modified `BLAS::Build()` to use proper staging buffer workflow:
- Create staging buffers with `VMA_MEMORY_USAGE_CPU_ONLY`
- Create device-local buffers with `VMA_MEMORY_USAGE_GPU_ONLY`
- Copy data using `vkCmdCopyBuffer`
- Insert memory barrier between transfer and AS build

**Why**: Ensures optimal GPU performance for geometry data.

---

### 2. Implemented Icosphere (No Degenerate Triangles)
**File**: `src/app/SceneBuilder.hpp`

**Change**: Replaced UV sphere with icosphere subdivision algorithm:
- Starts with icosahedron (12 vertices, 20 triangles)
- Subdivides edges to create uniform distribution
- No degenerate triangles at poles

**Why**: UV spheres have degenerate triangles that can cause issues on some drivers.

---

### 3. Created DXC Setup Scripts
**Files**:
- `tools/setup_dxc.sh` (Linux/macOS)
- `tools/setup_dxc.ps1` (Windows PowerShell)

**Usage**:
```bash
# Linux/macOS
./tools/setup_dxc.sh

# Windows
powershell -ExecutionPolicy Bypass -File tools/setup_dxc.ps1
```

**Why**: Automates DXC installation for shader compilation.

---

### 4. Integrated DXC into CMake Build
**File**: `src/shaders/CMakeLists.txt` (new file)

**Change**: Shaders now compile automatically during CMake build:
- Finds DXC in `tools/dxc` or Vulkan SDK
- Compiles `.rgen`, `.rchit`, `.rmiss` to `.spv`
- Shaders rebuild when source changes

**Usage**: Just run `cmake --build .` as usual.

---

### 5. Added Camera TOML Configuration
**Files**:
- `src/libQuantiloom/scene/Camera.hpp` (modified)
- `src/libQuantiloom/scene/Camera.cpp` (new file)
- `config/m1_test.toml` (new file)

**New Features**:
- `Camera` class with look-at interface
- `CameraData` struct for GPU push constants
- `Camera::FromConfig()` to load from TOML

**Example TOML**:
```toml
[camera]
position = [0.0, 2.0, -8.0]
lookAt = [0.0, 1.0, 0.0]
up = [0.0, 1.0, 0.0]
fovY = 60.0
```

---

### 6. Added Camera Push Constants to Shaders
**Files**:
- `src/shaders/common.hlsli` (added `CameraData` struct)
- `src/shaders/raygen.rgen` (uses push constants instead of hardcoded camera)

**Change**: Raygen shader now receives camera parameters via push constants.

---

## 🚧 TODO: Remaining Manual Steps

### Step 1: Add Camera.cpp to CMake Build

**File**: `src/libQuantiloom/CMakeLists.txt`

Find the section with `Scene.cpp` and add `Camera.cpp` to the source list:

```cmake
# Scene module
scene/Scene.cpp
scene/Camera.cpp  # ← ADD THIS LINE
```

---

### Step 2: Update RayTracingPipeline to Support Push Constants

**File**: `src/libQuantiloom/renderer/RayTracingPipeline.hpp`

Add these methods to the `RayTracingPipeline` class:

```cpp
public:
    // Set camera parameters (call before TraceRays)
    void SetCameraData(const CameraData& cameraData);

private:
    CameraData m_cameraData{};  // Cached camera data
```

**File**: `src/libQuantiloom/renderer/RayTracingPipeline.cpp`

1. Modify `CreatePipelineLayout()` to include push constants:

```cpp
void RayTracingPipeline::CreatePipelineLayout() {
    VkDevice device = m_context.GetDevice();

    // Push constant range for camera data
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(CameraData);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;  // ← ADD
    layoutInfo.pPushConstantRanges = &pushConstantRange;  // ← ADD

    VkResult result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
}
```

2. Implement `SetCameraData()`:

```cpp
void RayTracingPipeline::SetCameraData(const CameraData& cameraData) {
    m_cameraData = cameraData;
}
```

3. Modify `TraceRays()` to push camera data:

```cpp
void RayTracingPipeline::TraceRays(VkCommandBuffer cmd, u32 width, u32 height) {
    // Push camera constants BEFORE vkCmdTraceRaysKHR
    vkCmdPushConstants(
        cmd,
        m_pipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        0,
        sizeof(CameraData),
        &m_cameraData
    );

    // ... rest of TraceRays() code ...
    vkCmdTraceRaysKHR(cmd, ...);
}
```

---

### Step 3: Update main_m1_test.cpp to Use TOML Camera

**File**: `src/app/main_m1_test.cpp`

1. Add config loading at the start of `main()`:

```cpp
int main(int argc, char** argv) {
    try {
        QL_LOG_INFO("Quantiloom M1 Test - Ray Tracing Validation");

        // Parse command line arguments
        std::string configPath = "config/m1_test.toml";
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--config" && i + 1 < argc) {
                configPath = argv[i + 1];
                i++;
            }
        }

        // Load configuration
        auto configResult = Config::Load(configPath);
        if (!configResult.IsOk()) {
            QL_LOG_ERROR("Failed to load config: {}", configResult.GetError());
            return 1;
        }
        Config config = configResult.Unwrap();

        // Get resolution
        auto resArray = config.GetArray<u32>("renderer.resolution");
        u32 width = resArray.size() >= 1 ? resArray[0] : 1280;
        u32 height = resArray.size() >= 2 ? resArray[1] : 720;
        f32 aspectRatio = static_cast<f32>(width) / static_cast<f32>(height);

        // Create camera from config
        auto cameraResult = Camera::FromConfig(config, aspectRatio);
        if (!cameraResult.IsOk()) {
            QL_LOG_ERROR("Failed to load camera: {}", cameraResult.GetError());
            return 1;
        }
        Camera camera = cameraResult.Unwrap();
```

2. Before `TraceRays()` call:

```cpp
// Set camera parameters
pipeline.SetCameraData(camera.GetCameraData());

// Trace rays
pipeline.TraceRays(cmd, width, height);
```

---

## Building and Testing

### 1. First-Time Setup: Install DXC

```bash
# Linux/macOS
./tools/setup_dxc.sh

# Windows
powershell -ExecutionPolicy Bypass -File tools/setup_dxc.ps1
```

### 2. Build Project

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

This will automatically compile shaders during the build.

### 3. Run Tests

```bash
# With default config
./build/src/app/quantiloom_m1_test

# With custom config
./build/src/app/quantiloom_m1_test --config config/my_config.toml
```

### 4. Verify Shaders Were Compiled

Check that `.spv` files exist:
```bash
ls -lh src/shaders/*.spv
```

Expected output:
```
raygen.spv
closesthit.spv
miss.spv
```

---

## Validation Checklist

Before proceeding to M2, verify:

- [ ] Shaders compile automatically during CMake build
- [ ] Camera parameters work from TOML config
- [ ] Icosphere renders without black spots (no degenerate triangles)
- [ ] Performance is good (GPU_ONLY buffers in use)
- [ ] All M1 test scenes render correctly

---

## Troubleshooting

### "DXC not found" during CMake

Run the setup script first:
```bash
./tools/setup_dxc.sh  # or setup_dxc.ps1 on Windows
```

### Shader compilation errors

Check shader source files for syntax errors. DXC errors will be shown during build.

### Camera not moving

Ensure `SetCameraData()` is called before `TraceRays()` in main loop.

### Performance issues

Verify buffers are `GPU_ONLY` by checking logs:
```
[info] Uploaded geometry via staging buffers: ...
```

If you see "CPU_TO_GPU", the fix wasn't applied correctly.

---

## Next Steps (M2)

Once all validation passes:

1. Implement OBJ file loading (use Assimp or tinyobjloader)
2. Add per-instance material system
3. Implement shadow rays (NEE)
4. Add more BSDF models beyond Lambert

---

**Good taste restored. No more烂摊子. 🎯**
