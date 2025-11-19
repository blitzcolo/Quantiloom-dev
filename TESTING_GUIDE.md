# Quantiloom glTF + PBR 测试指南 (Testing Guide)

## 目录 (Table of Contents)

1. [快速开始 (Quick Start)](#快速开始-quick-start)
2. [详细步骤 (Detailed Steps)](#详细步骤-detailed-steps)
3. [配置文件说明 (Configuration File Reference)](#配置文件说明-configuration-file-reference)
4. [测试场景 (Test Scenarios)](#测试场景-test-scenarios)
5. [故障排除 (Troubleshooting)](#故障排除-troubleshooting)

---

## 快速开始 (Quick Start)

### 方法 1：测试内置场景 (Test Built-in Procedural Scene)

无需下载任何模型，直接测试内置的 Cornell Box：

```bash
cd /home/user/Quantiloom-dev
mkdir -p build && cd build
cmake ..
cmake --build . --config Release -j$(nproc)

# 运行 Cornell Box 测试
./bin/quantiloom ../assets/configs/cornell_box_procedural.toml
```

**预期结果：**
- 生成 `cornell_box_output.exr` 文件
- 日志显示场景加载、渲染进度、输出保存

---

### 方法 2：测试 glTF 模型 (Test glTF Model)

#### Step 1: 下载测试模型 (Download Test Models)

```bash
cd /home/user/Quantiloom-dev
mkdir -p assets/models
cd assets/models
git clone --depth 1 https://github.com/KhronosGroup/glTF-Sample-Models.git
```

#### Step 2: 编译并运行 (Build and Run)

```bash
cd /home/user/Quantiloom-dev/build
cmake --build . --config Release -j$(nproc)

# 运行 DamagedHelmet 测试 (完整 PBR 材质)
./bin/quantiloom ../assets/configs/gltf_pbr_test.toml
```

**预期结果：**
- 加载 `DamagedHelmet.gltf` 模型
- 应用 PBR 材质（base color + metallic-roughness + normal map + emissive）
- 生成 `gltf_pbr_output.exr` 文件

---

## 详细步骤 (Detailed Steps)

### 1. 编译项目 (Build the Project)

```bash
cd /home/user/Quantiloom-dev
mkdir -p build && cd build

# 配置 CMake（Release 模式以获得最佳性能）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译（使用所有 CPU 核心）
cmake --build . -j$(nproc)
```

**编译成功标志：**
- 无错误输出
- 生成可执行文件：`build/bin/quantiloom`
- 着色器编译成功：`*.spv` 文件生成在 `build/bin/` 目录

---

### 2. 验证着色器编译 (Verify Shader Compilation)

```bash
ls -lh build/bin/*.spv
```

**应看到：**
```
raygen.spv
closesthit.spv
miss.spv
```

如果缺少 `.spv` 文件，检查 CMake 输出中的 DXC 编译错误。

---

### 3. 准备测试资源 (Prepare Test Assets)

#### 选项 A：使用内置场景（无需外部文件）

配置文件：`assets/configs/cornell_box_procedural.toml`

```toml
[scene]
preset = "cornell_box"  # 或 "multi_object" / "lighting_test"
```

#### 选项 B：使用 glTF 模型

下载 Khronos 官方测试模型：

```bash
cd /home/user/Quantiloom-dev/assets/models
git clone --depth 1 https://github.com/KhronosGroup/glTF-Sample-Models.git
```

**推荐测试模型：**

| 模型 | 路径 | 测试目的 |
|------|------|----------|
| **DamagedHelmet** | `2.0/DamagedHelmet/glTF/DamagedHelmet.gltf` | 完整 PBR（base, metal-rough, normal, emissive） |
| **MetalRoughSpheres** | `2.0/MetalRoughSpheres/glTF/MetalRoughSpheres.gltf` | 金属度/粗糙度测试（网格） |
| **BoxTextured** | `2.0/BoxTextured/glTF/BoxTextured.gltf` | 简单纹理映射 |
| **Avocado** | `2.0/Avocado/glTF/Avocado.gltf` | 高质量 PBR 资产 |
| **WaterBottle** | `2.0/WaterBottle/glTF/WaterBottle.gltf` | 透明度测试 (M2+) |

---

### 4. 创建配置文件 (Create Configuration File)

我已经为你创建了三个测试配置文件：

#### 配置文件 1：`assets/configs/gltf_pbr_test.toml`
- **场景**：DamagedHelmet（完整 PBR 材质）
- **分辨率**：1920x1080
- **输出**：`gltf_pbr_output.exr`

#### 配置文件 2：`assets/configs/metal_spheres_test.toml`
- **场景**：MetalRoughSpheres（金属度/粗糙度网格）
- **分辨率**：1920x1080
- **输出**：`metal_spheres_output.exr`

#### 配置文件 3：`assets/configs/cornell_box_procedural.toml`
- **场景**：内置 Cornell Box（无需外部模型）
- **分辨率**：1280x720
- **输出**：`cornell_box_output.exr`

---

### 5. 运行渲染器 (Run the Renderer)

#### 基本语法：

```bash
./build/bin/quantiloom <config_file.toml>
```

#### 示例：

```bash
# 测试 1: Cornell Box（内置场景）
./build/bin/quantiloom assets/configs/cornell_box_procedural.toml

# 测试 2: glTF PBR 头盔
./build/bin/quantiloom assets/configs/gltf_pbr_test.toml

# 测试 3: 金属球网格
./build/bin/quantiloom assets/configs/metal_spheres_test.toml
```

---

### 6. 检查输出 (Check Output)

#### 日志输出（控制台）：

**成功运行应显示：**

```
========================================
  Quantiloom Spectral Path Tracer
========================================
[INFO] Loading configuration: assets/configs/gltf_pbr_test.toml
[INFO] Configuration loaded successfully
[INFO] Loading glTF model: assets/models/.../DamagedHelmet.gltf
[INFO]   Scene loaded: 1 meshes, 1 nodes, 1 materials
[INFO]   Created 1 BLAS(es) for 15452 total triangles
[INFO] Building acceleration structures...
[INFO]   TLAS built with 1 instance(s)
[INFO] Uploading textures to GPU...
[INFO]   5 textures uploaded
[INFO] Creating PBR material buffer...
[INFO]   Material 'Material_MR': base=[1.00,1.00,1.00,1.00] metal=1.00 rough=1.00
[INFO] Rendering frame at wavelength 550.0 nm...
[INFO]   Frame rendered (1920x1080)
[INFO]   [OK] Saved spectral image to gltf_pbr_output.exr
========================================
  Rendering COMPLETED
========================================
```

#### 输出文件（图像）：

**生成的 EXR 文件：**
- `gltf_pbr_output.exr` (或配置文件中指定的名称)
- 位置：当前工作目录（`/home/user/Quantiloom-dev`）

**查看 EXR 文件：**

```bash
# 方法 1: 使用 exrheader 查看元数据
exrheader gltf_pbr_output.exr

# 方法 2: 转换为 PNG 查看（需要 ImageMagick）
convert gltf_pbr_output.exr gltf_pbr_output.png

# 方法 3: 使用专业工具
# - Blender (File → Open Image)
# - GIMP with OpenEXR plugin
# - Photoshop
# - DJV (https://dj-view.sourceforge.io/)
```

---

## 配置文件说明 (Configuration File Reference)

### 完整配置示例：

```toml
[renderer]
resolution = [1920, 1080]      # [width, height] in pixels
spp = 1                        # Samples per pixel (M1: always 1)
output = "output.exr"          # Output file path (*.exr)

[spectral]
mode = "single_wavelength"     # M1 支持：single_wavelength
wavelength_nm = 550.0          # Wavelength in nanometers (400-700 nm visible)

[scene]
# 方法 1: 加载 glTF 文件
gltf = "path/to/model.gltf"    # .gltf or .glb

# 方法 2: 使用内置场景（注释掉上面的 gltf 行）
# preset = "cornell_box"       # "cornell_box" | "multi_object" | "lighting_test"

[camera]
position = [0.0, 2.0, 5.0]     # Camera position [x, y, z] (meters)
look_at = [0.0, 0.0, 0.0]      # Look-at target [x, y, z]
up = [0.0, 1.0, 0.0]           # Up vector (usually Y-up)
fov_y = 45.0                   # Vertical field of view (degrees)

[lighting]
sun_direction = [0.5, 0.8, 0.3]  # Direction FROM surface TO sun [x, y, z]
sun_radiance = [5.0, 5.0, 5.0]   # Sun radiance [R, G, B] (W·sr⁻¹·m⁻²)
sky_radiance = [0.5, 0.7, 1.0]   # Sky radiance [R, G, B] (W·sr⁻¹·m⁻²)

[material]
albedo = [0.8, 0.8, 0.8]       # 仅用于内置场景的默认材质
```

### 关键参数说明：

#### `[scene]` 部分：

**glTF 模式（推荐用于 Phase 3 测试）：**
```toml
[scene]
gltf = "assets/models/model.gltf"
```
- 支持 `.gltf` (文本格式) 和 `.glb` (二进制格式)
- 自动加载嵌入的纹理（PNG/JPEG）
- 材质使用 glTF 中定义的 PBR 参数

**Procedural 模式（用于基础测试）：**
```toml
[scene]
preset = "cornell_box"
```
- 不需要外部模型文件
- 使用 `[material].albedo` 定义的材质
- 预设选项：
  - `cornell_box` - 经典 Cornell Box
  - `multi_object` - 多个几何体测试
  - `lighting_test` - 光照测试场景

#### `[camera]` 部分：

**调整相机以适配模型：**

- **小模型** (如 BoxTextured)：
  ```toml
  position = [0.0, 0.0, 2.0]
  look_at = [0.0, 0.0, 0.0]
  ```

- **中等模型** (如 DamagedHelmet)：
  ```toml
  position = [0.0, 0.0, 3.0]
  look_at = [0.0, 0.0, 0.0]
  ```

- **大型场景** (如 Sponza)：
  ```toml
  position = [0.0, 2.0, 10.0]
  look_at = [0.0, 2.0, 0.0]
  fov_y = 60.0
  ```

**提示**：使用 Blender 或 glTF Viewer 查看模型的边界框，以确定合适的相机位置。

#### `[lighting]` 部分：

**sun_direction 理解：**
- 方向是 **FROM 表面 TO 太阳** (与光照方向相反)
- 例如：`[0.0, 1.0, 0.0]` 表示太阳在天顶（正上方）
- 例如：`[-0.5, 0.8, -0.3]` 表示太阳在右上方偏后

**radiance 值调整：**
- `sun_radiance`: 控制高光强度（3.0-10.0 合适）
- `sky_radiance`: 控制环境光强度（0.3-1.0 合适）
- 值越高，场景越亮

#### `[spectral]` 部分：

**wavelength_nm 值：**
- 400 nm - 紫色
- 450 nm - 蓝色
- 550 nm - 绿色 (推荐用于测试)
- 600 nm - 橙色
- 700 nm - 红色

---

## 测试场景 (Test Scenarios)

### 测试 1: 验证基础渲染（内置场景）

**目的**：验证 Vulkan 管线和基础光栅化是否正常工作

```bash
./build/bin/quantiloom assets/configs/cornell_box_procedural.toml
```

**预期结果**：
- Cornell Box 场景渲染成功
- 没有 Vulkan 错误
- 生成 `cornell_box_output.exr`

---

### 测试 2: 验证 glTF 加载（简单模型）

**目的**：验证 glTF 解析和纹理加载

**步骤：**

1. 修改 `assets/configs/gltf_pbr_test.toml`：
   ```toml
   [scene]
   gltf = "assets/models/glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf"
   ```

2. 运行：
   ```bash
   ./build/bin/quantiloom assets/configs/gltf_pbr_test.toml
   ```

**预期日志：**
```
[INFO] Loading glTF model: .../BoxTextured.gltf
[INFO]   Scene loaded: 1 meshes, 1 nodes, 1 materials
[INFO]   5 textures uploaded
```

---

### 测试 3: 验证 PBR 渲染（完整材质）

**目的**：验证 Cook-Torrance BRDF 和完整 PBR 管线

**步骤：**

1. 确保 `gltf_pbr_test.toml` 使用 DamagedHelmet：
   ```toml
   [scene]
   gltf = "assets/models/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf"
   ```

2. 运行：
   ```bash
   ./build/bin/quantiloom assets/configs/gltf_pbr_test.toml
   ```

**预期结果：**
- 头盔表面显示正确的金属光泽（高光反射）
- 粗糙区域显示漫反射
- Normal map 提供表面细节
- Emissive 区域发光

---

### 测试 4: 验证金属度/粗糙度范围（参数测试）

**目的**：验证完整的 metallic (0.0-1.0) 和 roughness (0.0-1.0) 参数范围

**步骤：**

```bash
./build/bin/quantiloom assets/configs/metal_spheres_test.toml
```

**预期结果：**
- 球体网格显示从非金属到金属的渐变
- 球体网格显示从光滑到粗糙的渐变
- 左下角（非金属+光滑）：强漫反射 + 弱高光
- 右上角（金属+粗糙）：强漫反射 + 无高光

---

## 故障排除 (Troubleshooting)

### 问题 1: `No configuration file provided`

**错误信息：**
```
[ERROR] No configuration file provided
Usage: ./quantiloom <config.toml>
```

**解决方案：**
```bash
# 正确用法：
./build/bin/quantiloom assets/configs/gltf_pbr_test.toml
```

---

### 问题 2: `Failed to load glTF: File not found`

**错误信息：**
```
[ERROR] Failed to load glTF: File not found
```

**原因**：glTF 文件路径错误或文件不存在

**解决方案：**

1. 检查路径是否正确（相对于可执行文件的工作目录）：
   ```bash
   ls -lh assets/models/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf
   ```

2. 使用绝对路径：
   ```toml
   gltf = "/home/user/Quantiloom-dev/assets/models/.../DamagedHelmet.gltf"
   ```

---

### 问题 3: `Ray tracing not supported on this device`

**错误信息：**
```
[ERROR] Ray tracing not supported on this device
```

**原因**：GPU 不支持 Vulkan Ray Tracing (VK_KHR_ray_tracing_pipeline)

**解决方案：**

1. 检查 GPU 支持：
   ```bash
   vulkaninfo | grep -i "ray tracing"
   ```

2. 需要以下硬件：
   - NVIDIA: RTX 20 系列或更新（Turing+ 架构）
   - AMD: RX 6000 系列或更新（RDNA 2+ 架构）
   - Intel: Arc A 系列

---

### 问题 4: 着色器编译失败

**错误信息：**
```
dxc: error: sampling with implicit lod is only allowed in fragment and compute shaders
```

**原因**：已在 commit `02f56f8` 中修复，确保使用最新代码

**解决方案：**
```bash
git pull origin claude/linus-code-review-01KRW3gyxgV8J3DRRjy42Jru
cd build
cmake --build . -j$(nproc)
```

---

### 问题 5: 输出图像全黑

**可能原因：**

1. **相机位置不正确**：相机在模型内部或太远
   - 解决：调整 `[camera].position` 和 `look_at`

2. **光照强度太低**：
   - 解决：增大 `[lighting].sun_radiance` 和 `sky_radiance`

3. **材质问题**：所有材质的 albedo 为黑色
   - 解决：检查 glTF 模型的材质定义

**调试步骤：**

```bash
# 1. 先测试内置场景（已知可工作）
./build/bin/quantiloom assets/configs/cornell_box_procedural.toml

# 2. 如果内置场景正常，问题在 glTF 模型或配置
# 3. 尝试简单的 BoxTextured 模型
```

---

### 问题 6: 编译错误（C++ 代码）

**常见错误已在以下 commits 中修复：**

- `9d9f1c3` - GLM experimental, Upload.cpp, Scene.cpp
- `3ba5209` - GltfLoader access violation, Result::Ok
- `1eaff3a` - main.cpp and main_m1_test.cpp

**解决方案：**
```bash
git status
git log --oneline -10

# 确保在正确的分支
git checkout claude/linus-code-review-01KRW3gyxgV8J3DRRjy42Jru

# 清理重新编译
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

---

## 高级用法 (Advanced Usage)

### 自定义 glTF 模型

如果你有自己的 glTF 模型，确保：

1. **文件格式**：`.gltf` (文本) 或 `.glb` (二进制)
2. **纹理格式**：PNG 或 JPEG（嵌入或外部）
3. **材质**：PBR metallic-roughness workflow
4. **坐标系统**：Y-up (glTF 标准)

### 调整 PBR 参数

glTF 模型的 PBR 参数在模型文件中定义。如果需要修改：

1. 使用 Blender 或 glTF Editor 编辑模型
2. 调整以下参数：
   - `pbrMetallicRoughness.baseColorFactor`
   - `pbrMetallicRoughness.metallicFactor`
   - `pbrMetallicRoughness.roughnessFactor`
   - `normalTexture.scale`
   - `emissiveFactor`

---

## 预期输出示例 (Expected Output Examples)

### Cornell Box (内置场景)

**特征：**
- 红色左墙、绿色右墙、白色其他墙面
- 中间两个白色立方体
- 柔和的阴影和间接光照

### DamagedHelmet (glTF PBR)

**特征：**
- 金属光泽的头盔表面
- Normal map 提供的表面凹凸细节
- Roughness 变化（有些区域光滑，有些粗糙）
- Emissive 区域（发光标志）

### MetalRoughSpheres (参数测试)

**特征：**
- 7x7 球体网格
- 从左到右：roughness 从 0.0 到 1.0
- 从下到上：metallic 从 0.0 到 1.0
- 左下角：塑料外观（非金属+光滑）
- 右上角：粗糙金属外观

---

## 联系与反馈 (Contact & Feedback)

如果遇到问题或需要进一步帮助，请提供：

1. **完整错误日志**（从控制台复制）
2. **配置文件内容**（使用的 `.toml` 文件）
3. **系统信息**：
   ```bash
   vulkaninfo --summary
   lspci | grep -i vga
   ```

---

**祝测试顺利！Good luck with testing!** 🚀
