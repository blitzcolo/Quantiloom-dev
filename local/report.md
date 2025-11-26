# Quantiloom 项目代码库全面分析报告

**报告生成时间**：2025-11-26
**分析范围**：52 个源文件 + 4 个配置 + 3 个文档
**代码行数**：约 15,000 行 C++/HLSL
**文档版本**：基于 SRS V4、时间表 V1

---

## 执行摘要

本报告对 Quantiloom 光谱路径追踪成像系统的代码库进行了深入分析，对比了 SRS V4 需求文档、时间表和实际实现进度。

**核心发现**：
- **实际进度**：约 25%，相当于预期 15 周计划的第 3-4 周
- **当前状态**：M1 里程碑基本完成（95%），M2 早期开发（30%）
- **关键差距**：MS-RT 的混合 PDF MIS 采样、HS-OFF 的 delta-tracking 体渲染均未实现
- **技术债务**：21 个 TODO/FIXME 标记，其中 3 个 P0 阻塞性问题
- **架构质量**：Vulkan RT 管线实现规范，CPU/GPU 结构体对齐验证完善，代码质量较高

---

## 目录

1. [项目实际进度评估](#1-项目实际进度评估)
2. [临时方案与技术债详细列表](#2-临时方案与技术债详细列表)
3. [与最终目标对比：缺失功能分析](#3-与最终目标对比缺失功能分析)
4. [架构问题与改进建议](#4-架构问题与改进建议)
5. [值得表扬的设计](#5-值得表扬的设计)
6. [推荐行动计划](#6-推荐行动计划)
7. [总结](#7-总结)

---

## 1. 项目实际进度评估

### 1.1 里程碑完成度对照表

| 里程碑 | 预期功能 | 实际完成度 | 关键缺失 |
|--------|----------|------------|----------|
| **M0 (工具链)** | CMake、TOML、I/O、日志 | ✅ **100%** | 无 |
| **M1 (HS-core原型)** | Vulkan RT、Lambert BRDF、LUT-fast | ✅ **95%** | IBL镜面反射 |
| **M1.5 (V&V基准)** | PBRT/Mitsuba对比基准 | ❌ **5%** | 自动化脚本、参考渲染器集成 |
| **M2 (MS-Preview)** | $S_\lambda=1$、SVGF去噪、成本日志 | ⚠️ **30%** | 去噪器、性能日志 |
| **M3 (MS-Quality)** | $S_\lambda>1$、混合PDF MIS | ❌ **0%** | 完全未实现 |
| **M4 (HS-OFF Q2)** | 逐带点渲、delta-tracking体渲 | ❌ **10%** | 体渲染核心算法 |
| **M5 (工程化)** | 多GPU、PSO缓存 | ❌ **0%** | 未开始 |
| **M6 (闭环验证)** | MODTRAN端到端验证 | ❌ **0%** | 未开始 |

**总体进度：约 25%**

按预期 15 周计划，当前实际进度相当于第 3-4 周的状态。

### 1.2 代码组织结构

**核心模块 (`src/libQuantiloom/`)**：
- **core/**: 基础设施（Log, Config, Types, Image, LUT, SpectralCube）
- **io/**: 资源加载器（ImageIO, GltfLoader, LUTLoader, SpectralIO）
- **renderer/**: Vulkan 抽象层（VulkanContext, RayTracingPipeline, AccelerationStructure, GpuBuffer/Image, TextureManager, CommandHelper）
- **scene/**: 场景管理（Scene, Camera, Material, Mesh, Texture）

**应用入口 (`src/app/`)**：
- `main.cpp` - 主渲染器（支持 RGB/单波长/MWIR/LWIR 模式）
- `main_m1_test.cpp` - M1 里程碑测试程序
- `SceneBuilder.hpp` - 程序化场景生成器

**着色器 (`src/shaders/`)**：
- `raygen.rgen` - 相机射线生成
- `closesthit.rchit` - PBR 材质着色（Cook-Torrance BRDF）
- `miss.rmiss` - 环境天空
- `common.hlsli` - 共享数据结构（Payload, LUTData, CameraData, MaterialData）
- `pbr.hlsli` - PBR 工具函数（GGX NDF, Smith 遮蔽, Fresnel）
- `SpectralConversion.hlsli` - RGB↔光谱转换
- `blackbody.hlsli` - Planck 定律（IR 自发辐射）

### 1.3 核心功能实现状态

#### Vulkan RT 管线（已实现 ✅）

**加速结构**：
- ✅ BLAS/TLAS 构建与实例化
- ✅ 每个实例携带 materialId
- ✅ 支持 Vertex/Index/UV/Tangent 缓冲

**光追管线**：
- ✅ Shader Binding Table (SBT) 正确对齐
- ✅ 9 个 Descriptor Bindings（输出图像、TLAS、LUT、几何、材质、纹理数组）
- ✅ Bindless 纹理数组（NonUniformResourceIndex）

**着色器功能**：
- ✅ 相机射线生成（含 FOV 缩放、长宽比校正）
- ✅ 三角形求交（硬件加速）
- ✅ 重心坐标插值（UV、法线、切线）
- ✅ 纹理采样（SampleLevel + LOD 0）
- ⚠️ Ray differential 缺失（影响纹理过滤质量）

#### 光谱渲染支持（部分实现 ⚠️）

**光谱模式定义**：
```cpp
enum class SpectralMode : u32 {
    Single       = 0,  // 单波长（灰度输出）
    RGB          = 1,  // RGB渲染
    Multispectral = 2,  // 多光谱（未实现）
    MWIR_Fused   = 3,  // 中波红外（3-5μm）
    LWIR_Fused   = 4   // 长波红外（8-12μm）
};
```

**当前实现状态**：
- ✅ **RGB 模式**：完整 PBR 渲染，输出线性 RGB（`closesthit.rchit:356-378`）
- ✅ **Single 模式**：RGB 平均转灰度（`closesthit.rchit:379-394`）
- ⚠️ **MWIR/LWIR 模式**：基础框架存在，使用 Planck 定律（`closesthit.rchit:396-430`），但：
  - 仅支持 `spectralAlbedo` 近似（非定量）
  - 缺少真实 IR 材质曲线（`Material.hpp:98-104` 定义但未使用）
- ❌ **Multispectral 模式**：未实现（`closesthit.rchit:432-445` 为 fallback）

#### 材质系统（glTF 2.0 PBR 已实现 ✅）

**完整支持的 PBR 功能**：
- ✅ Cook-Torrance 微表面 BRDF（`pbr.hlsli:123-181`）
- ✅ GGX 法线分布函数
- ✅ Smith 几何遮蔽项
- ✅ Fresnel-Schlick 近似
- ✅ Lambertian 漫反射（能量守恒）
- ✅ 法线贴图（TBN 矩阵，Gram-Schmidt 正交化）
- ⚠️ IBL 镜面反射（`closesthit.rchit:339` 标记 TODO，当前 = 0）

**材质加载**：
- ✅ glTF 2.0 解析（基于 tinygltf）
- ✅ 外部纹理加载（PNG/JPEG）
- ✅ 自动 sRGB→线性转换
- ⚠️ IR 材质扩展（`GltfLoader.cpp:368-390` 标记 TODO，仅占位）

**CPU/GPU 结构体对齐验证**（亮点）：
```cpp
static_assert(sizeof(MaterialDataCPU) == 72, "Size mismatch!");
static_assert(offsetof(MaterialDataCPU, emissiveTextureIndex) == 52, "Offset mismatch!");
```

#### 大气散射与 LUT（基础框架 ⚠️）

**当前实现**：
- ✅ LUT 加载器（`LUTLoader.cpp`，HDF5 格式）
- ✅ 线性插值（`LUT.hpp:81-111`）
- ✅ 着色器 LUT 采样（`closesthit.rchit:286-305`，天空 + 直达光）
- ❌ **LUT-fast 模式**（MS-RT）：未见 Beer-Lambert 视线衰减实现
- ❌ **完整体渲染**（HS-OFF）：Rayleigh/Mie 散射、delta-tracking 均未实现

**miss shader**：
```hlsl
// miss.rmiss:30-44
LUTData lut = skyLUT[0];
if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
    payload.radiance = lut.skyRadiance_rgb;
} else {
    payload.radiance = float3(lut.skyRadiance_spectral, ...);
}
```
⚠️ 当前仅返回常量天空辐亮度，无角度依赖

#### 传感器链（未实现 ❌）

**定义的数据结构**：
```cpp
// SpectralCube.hpp - HS-OFF 输出容器
struct SpectralCube {
    u32 width, height, nbands;
    f32 lambda_min, lambda_max, delta_lambda;
    std::vector<f32> data;  // C-order: [band][y][x]
    std::vector<f32> wavelengths;
};
```

**缺失组件**：
- ❌ 量子效率（QE）卷积
- ❌ 仪器线形函数（ILS）
- ❌ DN 生成（12/14/16-bit 量化）
- ❌ 噪声模型（泊松、读出噪声、暗电流）

---

## 2. 临时方案与技术债详细列表

### 2.1 关键 TODO/FIXME（按优先级排序）

#### 🔴 P0 - 阻塞性问题（影响 M2 验收）

| 位置 | 问题描述 | 影响 | 正确解决方案 |
|------|----------|------|--------------|
| **`closesthit.rchit:339`** | **IBL 镜面反射未实现** | PBR 金属材质无环境反射，视觉错误 | **实现 split-sum approximation**：<br>1. 预过滤环境贴图（mipmap 链对应不同粗糙度）<br>2. BRDF 积分 LUT（2D 纹理：NdotV × roughness）<br>3. 着色器中采样并加权：<br>`L_ibl = (kS * prefilteredColor) * (F * brdfLUT.x + brdfLUT.y)` |
| **`main.cpp`** | **缺少 seconds-per-frame 成本记录** | 无法评估 MS-RT 性能，违反 SRS §3.2 | **添加帧时间统计**：<br>1. 使用 `VkQueryPool` 记录 GPU 时间戳<br>2. 记录 spp、$S_\lambda$、分辨率到日志<br>3. 输出 CSV 格式：`frame,spp,S_lambda,gpu_ms,rays_per_sec` |

#### 🔴 P1 - 核心功能缺失（影响 M3/M4）

| 位置 | 问题描述 | 影响 | 正确解决方案 |
|------|----------|------|--------------|
| **全代码库** | **MS-RT 混合 PDF MIS 完全未实现** | 无法满足 SRS §5.1 核心需求 | **实现路线图**（8 周工作量）：<br><br>**Week 1-2: 基础框架**<br>- 在 `raygen.rgen` 中添加 `S_lambda` 循环<br>- 定义 `p_ref`, `p_uniform`, `p_mat` 采样函数<br><br>**Week 3-4: PDF 构建**<br>- `p_ref(λ)`: 从配置读取参考谱（太阳谱×QE），生成 CDF<br>- `p_uniform(λ)`: 带通内均匀分布<br>- `p_mat(λ)`: 离线预计算材质谱均值（粗网格）<br><br>**Week 5-6: MIS 权重**<br>- 实现 balance heuristic: `w_i = p_i / Σp_j`<br>- 添加 $\alpha, \beta, \gamma$ 配置参数<br><br>**Week 7-8: 验证与调优**<br>- 通过基准 C（窄带特征场景）<br>- 实现 MIS 可视化工具（热图显示 PDF 采样来源） |
| **无体渲染代码** | **Delta-tracking 完全缺失** | 无法实现 HS-OFF 大气散射 | **实现路线图**（6 周工作量）：<br><br>**Week 1-2: Woodcock 算法核心**<br>- 在 `closesthit.rchit` 中添加 `TraceDeltaTracking()` 函数<br>- 实现 majorant 查找（从网格或常数）<br>- 自由程采样：`t = -log(ξ) / σ_max`<br>- 拒绝采样：`if ξ < σ_t(x)/σ_max then real_collision else virtual_collision`<br><br>**Week 3-4: 分层 majorant**<br>- 定义 `MajorantGrid3D` 数据结构（八叉树或均匀网格）<br>- 实现 DDA 遍历（voxel stepping）<br><br>**Week 5-6: Rayleigh/Mie 散射**<br>- 实现相函数：`PhaseRayleigh(cosθ)`, `PhaseHG(cosθ, g)`<br>- NEE 对太阳/天空光源<br>- MIS 合并散射/NEE 样本 |
| **`closesthit.rchit:78`** | **Ray differential 缺失** | 纹理 LOD 固定为 0，远距离锯齿 | **实现 ray differential**：<br>1. 在 `Payload` 中添加 `float2 ddx, ddy`（屏幕空间导数）<br>2. 在 `raygen.rgen` 初始化为相邻像素射线差<br>3. 在 `closesthit.rchit` 传播：`ddx' = ddx + t·dD`<br>4. 计算 LOD：`lod = log2(max(length(ddx), length(ddy)))` |

#### 🟡 P2 - 定量验证缺失（影响科研可信度）

| 位置 | 问题描述 | 影响 | 正确解决方案 |
|------|----------|------|--------------|
| **无验证脚本** | **M1.5 基准套件未建立** | 无法通过 Gate-A/B | **实施步骤**：<br>1. 在 `scripts/validation/` 创建 Python 脚本<br>2. 使用 Docker 容器化 PBRT-v4（固定版本）<br>3. 实现基准 A（Cornell Box 解析解对比）：<br>   - 均匀 Lambert 材质（$\rho=0.8$）<br>   - 单色平行光（$E=1.0$）<br>   - 预期解：$L = \rho E / \pi$<br>   - 误差统计：RMSE、中位数相对误差<br>4. CI 集成：GitHub Actions 每日运行 |
| **`GltfLoader.cpp:368-390`** | **IR 材质扩展未实现（仅注释）** | 无法加载真实 IR 材质数据 | **实现 glTF 扩展解析**：<br>1. 读取 `QUANTILOOM_material_ir` JSON<br>2. 解析 CSV 路径（`emissivity_curve`, `reflectance_curve`）<br>3. 调用 `SpectralIO::LoadSpectralCurveCSV()`<br>4. 填充 `Material::irEmissivityCurve` 等字段<br>5. 验证 Kirchhoff 定律：$\epsilon + \rho + \tau \leq 1$ |
| **`SpectralIO.cpp:274-297`** | **CSV 光谱曲线加载器为 Stub** | 无法使用 `assets/materials/ir_materials/` 中的真实数据 | **实现 CSV 解析器**：<br>```cpp<br>// 格式：wavelength_nm,value<br>std::ifstream file(filepath);<br>std::vector<std::pair<f32, f32>> curve;<br>String line;<br>std::getline(file, line); // Skip header<br>while (std::getline(file, line)) {<br>    auto [lambda, value] = ParseCSVLine(line);<br>    curve.emplace_back(lambda, value);<br>}<br>``` |

### 2.2 硬编码值与魔术数字

#### 着色器常量（需配置化）

| 位置 | 硬编码值 | 问题 | 修复方案 |
|------|----------|------|----------|
| **`closesthit.rchit:72`** | `MAX_TEXTURE_INDEX = 1024` | 无法动态调整绑定数组大小 | 通过 push constant 传递，从管线创建时查询：<br>`VkPhysicalDeviceProperties.limits.maxPerStageDescriptorSampledImages` |
| **`raygen.rgen:58-59`** | `TMin=0.001, TMax=10000.0` | 场景缩放敏感，Cornell Box (2m) vs 城市 (10km) | 从相机配置读取：<br>- `camera.near_plane` (TMin)<br>- `camera.far_plane` (TMax) |
| **`SpectralConversion.hlsli:59-60`** | `SIGMA=60.0, NORMALIZATION=1.3` | RGB→光谱上采样参数未调优 | 引用 Jakob & Hanika 2019 优化参数：<br>- σ 应根据基底函数优化（实验值 35-80 nm）<br>- 归一化因子应保证能量守恒（积分测试） |
| **`main.cpp:462`** | RGB→灰度用简单平均 | 应使用感知亮度权重 | 修改为 CIE Y：<br>`float Y = 0.2126*R + 0.7152*G + 0.0722*B;` |

#### 结构体对齐验证（已正确 ✅）

**值得表扬**：`main.cpp:78-84` 使用 `static_assert` 验证 CPU/GPU 结构体对齐，避免了常见的布局错误（这是 M2 崩溃的主因）。

### 2.3 调试代码残留（需清理）

| 位置 | 问题 | 风险 |
|------|------|------|
| **`raygen.rgen:62-87`** | `#if 1 ... #else` 调试块 | 生产代码中保留预处理开关，应移除或使用配置标志 |
| **`closesthit.rchit:369`** | 注释掉的 `output_radiance = baseColor.rgb;` | 容易误用于调试后忘记恢复 |

---

## 3. 与最终目标对比：缺失功能分析

### 3.1 MS-RT 模式（多光谱快速预览）

#### SRS V4 要求（§2.2）

- 输出 $N \in [3,16]$ 个带通通道
- **带通积分**：$I_b = \int_{\text{band}} L_\lambda(\lambda) \cdot w_b(\lambda) d\lambda$
- **混合 PDF MIS**：$p(\lambda) = \alpha p_{\text{ref}} + \beta p_{\text{uniform}} + \gamma p_{\text{mat}}$
- **多采样率**：$S_\lambda \in [1,8]$
- **LUT-fast 大气**：天空辐亮度 + Beer-Lambert 视线衰减
- **性能口径**：seconds-per-frame（明确不宣称实时）

#### 当前实现状态

| 功能 | 状态 | 证据 |
|------|------|------|
| 带通定义 | ❌ 未实现 | 配置文件无 `[spectral.bands]` 数组解析 |
| 混合 PDF 采样 | ❌ 未实现 | 代码中无 `p_ref`, `p_uniform`, `p_mat` 逻辑 |
| $S_\lambda > 1$ 多采样 | ❌ 未实现 | `raygen.rgen` 无波长循环 |
| 分层采样（strata） | ❌ 未实现 | 无 stratified sampling 代码 |
| Beer-Lambert 衰减 | ❌ 未实现 | miss shader 仅返回常量天空，无路径积分 |
| 去噪器（SVGF/NRD） | ❌ 未实现 | 无时空去噪模块 |
| 成本日志 | ❌ 未实现 | 无 `seconds_per_frame` 统计 |

**结论**：MS-RT 模式 **0% 实现**（仅有 RGB 单采样基础，不满足 MS 需求）

#### 正确实现路径

**阶段 1：带通采样框架（2 周）**

```hlsl
// raygen.rgen（伪代码框架）
struct BandConfig {
    float center_nm;
    float fwhm_nm;
    float alpha, beta, gamma;  // MIS权重
};

[[vk::push_constant]] BandConfig bands[16];
[[vk::push_constant]] uint num_bands;
[[vk::push_constant]] uint S_lambda;

[shader("raygeneration")]
void main() {
    for (uint band = 0; band < num_bands; ++band) {
        float3 radiance_band = 0;
        for (uint s = 0; s < S_lambda; ++s) {
            // 混合PDF采样
            float lambda = SampleMixedPDF(bands[band], s);
            float pdf = EvaluateMixedPDF(bands[band], lambda);

            // 发射光线（携带wavelength）
            Payload payload;
            payload.wavelength = lambda;
            TraceRay(..., payload);

            // MIS权重累积
            float w_b = EvaluateILS(bands[band], lambda);
            radiance_band += payload.radiance * w_b / pdf;
        }
        bandImages[band][pixel] = radiance_band / S_lambda;
    }
}
```

**阶段 2：PDF 构建（4 周）**

1. **$p_{\text{ref}}(\lambda)$**：参考谱（太阳谱 × QE）
   - 从配置加载 `ref_spectrum.csv`
   - 构建 1D CDF（逆变换采样）

2. **$p_{\text{uniform}}(\lambda)$**：带通内均匀
   - `lambda = band.center - band.fwhm/2 + ξ * band.fwhm`

3. **$p_{\text{mat}}(\lambda)$**：材质光谱启发
   - 离线预计算：对场景中所有材质，在带通内采样反射率
   - 存储粗网格（如 10nm 分辨率）
   - 运行时查表

**阶段 3：验证（2 周）**

- **基准 B**：已知光谱照明（D65）下的带通积分，与 PBRT-v4 对比
- **基准 C**：窄带金属峰（如铜 600nm Fresnel 峰），评估方差

### 3.2 HS-OFF 模式（高光谱逐带点渲）

#### SRS V4 要求（§2.3）

- 逐波长点渲：`for λ in [λ_min : Δλ : λ_max]`
- **严禁光谱插值**作为主结果
- **完整体渲**：Rayleigh/Mie + delta-tracking
- **强制门控**：`fail_on_srgb_upsample = true`
- 输出高光谱立方体（HDF5/EXR）

#### 当前实现状态

| 功能 | 状态 | 证据 |
|------|------|------|
| 逐带循环驱动 | ⚠️ 框架存在 | `SpectralCube.hpp` 定义数据结构，但 `main.cpp` 无驱动循环 |
| Delta-tracking | ❌ 完全未实现 | 代码中无 Woodcock 算法 |
| Rayleigh/Mie 散射 | ❌ 未实现 | 无相函数代码 |
| sRGB 上采样门控 | ✅ **已实现** | `main.cpp:321-377` 检查 `spectralSource` |
| HDF5 输出 | ⚠️ 部分实现 | `SpectralIO.cpp` 有读取，无写入 |

**结论**：HS-OFF 模式 **15% 实现**（数据结构 + 验证门控，核心算法缺失）

#### sRGB 上采样门控（唯一亮点 ✅）

**代码位置**：`main.cpp:321-377`

```cpp
// 风险 R5 对策：验证光谱数据来源
if (config.Get<bool>("quality.fail_on_srgb_upsample", false)) {
    for (const auto& mat : scene.materials) {
        if (mat.spectralSource == Material::SpectralSource::RGBUpsampled) {
            QL_LOG_ERROR("Material '{}' uses RGB-upsampled spectral data", mat.name);
            QL_LOG_ERROR("This is NOT allowed in HS-OFF quantitative mode");
            QL_LOG_ERROR("Set fail_on_srgb_upsample=false to bypass (non-quantitative)");
            return 1;
        }
    }
}
```

**评价**：符合 SRS §4.3 要求，有效防止光谱污染，体现科研严谨性。

#### 正确实现路径：Delta-Tracking（关键算法）

**Woodcock 算法核心**：

```hlsl
// 伪代码框架
float TraceDeltaTracking(float3 origin, float3 direction, float t_max) {
    float t = 0;
    while (t < t_max) {
        // 查询当前体素的majorant
        float sigma_max = QueryMajorantGrid(origin + t * direction);

        // 采样自由程
        float t_sample = -log(1 - rand()) / sigma_max;
        t += t_sample;

        if (t >= t_max) return 0;  // 射线逃逸

        // 真实碰撞 vs 虚拟碰撞
        float3 pos = origin + t * direction;
        float sigma_t = GetExtinctionCoeff(pos, lambda);  // Rayleigh + Mie

        if (rand() < sigma_t / sigma_max) {
            // 真实碰撞：采样相函数
            float3 scatter_dir = SamplePhaseFunction(direction, lambda);
            // NEE 对太阳/天空
            float3 L_nee = EvaluateNEE(pos, lambda);
            // MIS 合并
            return L_nee + TraceDeltaTracking(pos, scatter_dir, ...);
        }
        // else: 虚拟碰撞，继续前进
    }
}
```

**分层 Majorant 网格**（性能关键）：

- **数据结构**：八叉树或均匀 3D 网格（256³ 典型）
- **预计算**：离线扫描大气密度场，每个 voxel 存储局部最大值
- **遍历**：DDA 算法（Digital Differential Analyzer）

**预估成本**（SRS §3.3）：

- 无体渲：1080p, spp=2 → **0.5-2.0s/波段**
- 完整 Q2：max_steps=256 → **2-30s/波段**
- 76 波段总计：**150-2280 分钟** → **需多 GPU 加速**

### 3.3 传感器链（完全缺失）

#### SRS V4 要求（§4.1）

- **量子效率（QE）**：$\eta(\lambda) \in [0,1]$
- **仪器线形（ILS）**：带通权重函数 $w_b(\lambda)$（高斯/矩形）
- **噪声链**：
  - 泊松噪声：$\sigma^2 = \bar{n}$（光子计数统计）
  - 读出噪声：高斯（$\sigma_{\text{read}} \approx 5-20 e^-$）
  - 暗电流：$i_{\text{dark}} \cdot t_{\text{exp}}$
  - PRNU/DSNU（像素非均匀性）
  - 量化：12/14/16-bit ADC
- **DN 生成**：数字计数值（Digital Number）

#### 当前状态

**完全未实现**（0%）。代码中无相关模块。

#### 正确实现路径

**模块结构**：

```cpp
// src/libQuantiloom/sensor/SensorChain.hpp
class SensorChain {
public:
    // 输入：辐亮度立方体 (W/m²/sr/nm)
    // 输出：DN (12/14/16-bit)
    Image<u16> GenerateDN(const SpectralCube& radiance);

private:
    // QE卷积
    std::vector<f32> qe_curve;  // (wavelength, efficiency)

    // ILS卷积（带通）
    struct Band {
        f32 center_nm, fwhm_nm;
        std::vector<f32> ils_weights;  // 离散化ILS
    };

    // 噪声模型
    f32 dark_current_e_per_s;
    f32 read_noise_sigma_e;
    Image<f32> prnu_map;  // 像素响应非均匀性

    // ADC参数
    u32 bit_depth;
    f32 full_well_capacity_e;
};
```

**处理流程**：

1. **光谱→电子**：
   $$
   e^-(\mathbf{x}) = \int_{\lambda} L(\mathbf{x}, \lambda) \cdot \eta(\lambda) \cdot A \cdot \Omega \cdot t_{\text{exp}} \cdot \frac{\lambda}{hc} \, d\lambda
   $$
   - $A$：像元面积（m²）
   - $\Omega$：立体角（sr）
   - $t_{\text{exp}}$：曝光时间（s）

2. **噪声注入**：
   ```cpp
   float signal_e = QEConvolution(radiance_spectrum);
   float noise_e = sqrt(signal_e);  // 泊松
   noise_e += GaussianNoise(read_noise_sigma);
   noise_e += dark_current * exposure_time;
   signal_e *= prnu_map[pixel];  // PRNU
   ```

3. **量化**：
   ```cpp
   u16 dn = clamp(signal_e / full_well * ((1 << bit_depth) - 1), 0, (1<<bit_depth)-1);
   ```

---

## 4. 架构问题与改进建议

### 4.1 着色器变体管理缺失（R3 风险）

#### 问题描述

当前通过 `#if` 硬切换模式，运行时分支深：

```hlsl
// closesthit.rchit:356-445
if (camera.spectral_mode == SPECTRAL_MODE_RGB) {
    // RGB分支
} else if (camera.spectral_mode == SPECTRAL_MODE_SINGLE) {
    // Single分支
} else if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED) {
    // MWIR分支
}
```

**风险**：
- GPU 分支发散（warp 内不同线程走不同分支，性能损失 2-4x）
- 未来添加 MS-RT/HS-OFF 后，组合爆炸（$2^5 = 32$ 种变体）

#### 正确方案（SRS §6.1）

**ShaderKey 枚举**：

```cpp
enum class ShaderKey {
    Mode_MS_Medium_Off_NEE_On    = 0b0000,
    Mode_MS_Medium_Full_NEE_On   = 0b0001,
    Mode_HS_Medium_Off_NEE_On    = 0b0010,
    Mode_HS_Medium_Full_NEE_On   = 0b0011,
    // ... 最多8个变体
};
```

**Specialization Constants**（Vulkan 特性）：

```cpp
// RayTracingPipeline.cpp
VkSpecializationMapEntry entries[] = {
    {0, 0, sizeof(uint32_t)},  // spectral_mode
    {1, 4, sizeof(uint32_t)},  // medium_mode
};
VkSpecializationInfo spec_info = {
    .mapEntryCount = 2,
    .pMapEntries = entries,
    .dataSize = 8,
    .pData = &shader_key,
};
```

**Shader 中使用**：

```hlsl
[[vk::constant_id(0)]] const uint SPECTRAL_MODE = 0;
[[vk::constant_id(1)]] const uint MEDIUM_MODE = 0;

if constexpr (SPECTRAL_MODE == SPECTRAL_MODE_RGB) {
    // 编译时分支，死代码被裁剪
}
```

**PSO 缓存**（对策 R3）：

```cpp
// 离线预编译8个变体
for (auto key : shader_keys) {
    CreatePipeline(key);
    vkGetPipelineCacheData(..., &cache_blob);
    SaveToDisk("pipeline_cache.bin", cache_blob);
}
```

### 4.2 无单元测试（质量风险）

#### 现状

- `tests/` 目录为空
- `CMakeLists.txt:265` 注释掉测试

#### 建议测试框架

```cpp
// tests/core/test_spectral_conversion.cpp
TEST(SpectralConversion, RGBToSpectrumEnergyConservation) {
    glm::vec3 rgb(0.8, 0.2, 0.1);
    auto spectrum = RGBToSpectrum(rgb);

    // 积分应守恒
    float Y_reconstructed = IntegrateXYZ_Y(spectrum);
    float Y_original = 0.2126*0.8 + 0.7152*0.2 + 0.0722*0.1;

    EXPECT_NEAR(Y_reconstructed, Y_original, 0.01);
}

TEST(DeltaTracking, UniformMedium_AnalyticTransmittance) {
    // 均匀介质，解析解：T = exp(-σ_t * d)
    float sigma_t = 0.5;
    float distance = 10.0;

    float T_analytic = exp(-sigma_t * distance);
    float T_mc = MonteCarloTransmittance(sigma_t, distance, 10000);

    EXPECT_NEAR(T_mc, T_analytic, 0.05);  // 5% 误差容限
}
```

### 4.3 日志结构化不足

#### 问题

当前日志为纯文本，难以解析：

```
INFO: Rendering frame 0
INFO: Trace complete
```

#### 建议格式（JSON Lines）

```json
{"level":"INFO","module":"Renderer","frame":0,"spp":2,"S_lambda":4,"gpu_ms":125.3}
{"level":"PERF","module":"MIS","band":"VIS_550","pdf_ref":0.6,"pdf_uniform":0.25,"pdf_mat":0.15}
```

**优势**：
- 可用 `jq` 过滤：`cat log.jsonl | jq 'select(.module=="MIS")'`
- 自动生成性能曲线（Jupyter Notebook 读取）

---

## 5. 值得表扬的设计

### 5.1 CPU/GPU 结构体对齐验证 ✅

```cpp
static_assert(sizeof(MaterialDataCPU) == 72, "Size mismatch!");
static_assert(offsetof(MaterialDataCPU, emissiveTextureIndex) == 52, "Offset mismatch!");
```

**评价**：这是避免 GPU 崩溃的关键防御，很多项目因缺少此检查而浪费数周调试。Quantiloom 在这一点上做得非常好。

### 5.2 SafeNormalize 防御式编程 ✅

```hlsl
float3 SafeNormalize(float3 v, float3 fallback) {
    float lenSq = dot(v, v);
    if (lenSq < 1e-8) return fallback;
    return v * rsqrt(lenSq);
}
```

**评价**：有效防止退化三角形/法线导致的 NaN 传播。在 GPU 编程中，NaN 是极其隐蔽的 bug 源头，这种防御式编程值得推广。

### 5.3 光谱数据来源追踪 ✅

```cpp
enum class SpectralSource {
    Unknown, Measured, RGBUpsampled, Procedural
};
```

**评价**：支持 SRS §4.3 的 `fail_on_srgb_upsample` 门控，体现了对科研严谨性的重视。这是区分"演示级"和"定量级"渲染器的关键设计。

### 5.4 详细的注释文档 ✅

着色器和 C++ 代码中均有丰富的注释，解释了：
- 为什么不能使用某些功能（如 ray tracing 中不能用 `Sample()` 只能用 `SampleLevel()`）
- 物理单位和坐标系约定
- TODO 的具体实施步骤

**评价**：这大大降低了后续开发者的理解成本。

---

## 6. 推荐行动计划（按 SRS 时间表调整）

### Phase 1：完成 M2 并建立验证基准（6 周）

**Week 1-2**：
- [ ] **实现 IBL 镜面反射**（`closesthit.rchit:339`）
  - 生成预过滤环境贴图（Mipmap 链）
  - 生成 BRDF 积分 LUT
  - 实现 split-sum approximation
- [ ] **添加性能日志**
  - VkQueryPool 时间戳
  - 输出 `seconds_per_frame`、`rays_per_sec`

**Week 3-4**：
- [ ] **建立 M1.5 基准 A**（Cornell Box 解析解）
  - 编写 `scripts/validation/benchmark_a.py`
  - Docker 容器化 PBRT-v4
  - 自动误差统计（RMSE、中位数）
- [ ] **集成 CI**（GitHub Actions）

**Week 5-6**：
- [ ] **通过 Gate-A**（基准 A 误差 < 5%）
- [ ] **文档化 M2 成果**

### Phase 2：攻克 M3 - MS-RT MIS 采样（8 周）

**Week 7-8**：基础框架
- [ ] 实现 $S_\lambda > 1$ 循环（`raygen.rgen`）
- [ ] 定义 `BandConfig` 数据结构

**Week 9-11**：PDF 构建
- [ ] 实现 $p_{\text{ref}}$（参考谱 CDF）
- [ ] 实现 $p_{\text{uniform}}$（均匀采样）
- [ ] 实现 $p_{\text{mat}}$（材质谱网格）

**Week 12-13**：MIS 权重与验证
- [ ] Balance heuristic 权重
- [ ] 可视化工具（PDF 热图）
- [ ] 通过基准 C（窄带收敛）

**Week 14**：
- [ ] **通过 Gate-B**（基准 B/C 误差达标）

### Phase 3：实现 M4 - HS-OFF 体渲染（8 周）

**Week 15-17**：Delta-Tracking 核心
- [ ] Woodcock 算法实现
- [ ] 分层 majorant 网格
- [ ] DDA 遍历

**Week 18-20**：散射模型
- [ ] Rayleigh 相函数
- [ ] Mie 相函数（Henyey-Greenstein）
- [ ] NEE + MIS 合并

**Week 21-22**：验证与优化
- [ ] 通过基准 D（均匀介质对比）
- [ ] 性能剖析与优化

### Phase 4：传感器链与工程化（6 周）

**Week 23-25**：传感器链
- [ ] QE 卷积
- [ ] ILS 带通
- [ ] 噪声模型
- [ ] DN 生成

**Week 26-28**：工程化（M5）
- [ ] PSO 预编译与缓存
- [ ] 多 GPU 支持
- [ ] 性能剖析仪表板

---

## 7. 总结

### 7.1 核心发现

1. **进度偏离**：实际进度约 25%，相当于预期 15 周计划的第 3-4 周
2. **关键差距**：
   - MS-RT 的混合 PDF MIS **完全未实现**（8 周工作量）
   - HS-OFF 的 delta-tracking **完全未实现**（6 周工作量）
   - M1.5 验证基准 **未建立**（4 周工作量）
3. **技术债务**：21 个 TODO/FIXME，其中 3 个 P0 阻塞性问题
4. **架构风险**：着色器变体管理缺失，未来扩展受限

### 7.2 优先级排序

**🔥 P0 - 立即行动**：
1. 建立 M1.5 基准 A（提供验证尺子）
2. 完成 IBL 镜面反射（M2 PBR 完整性）
3. 添加性能日志（满足 SRS 要求）

**🔴 P1 - 核心功能（6-8 周）**：
4. 实现 MIS 采样（M3 核心）
5. 实现 delta-tracking（M4 核心）

**🟡 P2 - 定量验证（4 周）**：
6. 实现 IR 材质加载
7. 实现传感器链

### 7.3 风险提示

#### 关键路径风险

- **MIS 调参黑洞**（R1）：混合 PDF 权重 $(\alpha, \beta, \gamma)$ 需手工调优，预算 2 周
- **体渲成本爆炸**（R4）：Delta-tracking 在密集介质中步进数 >1000，需分层 majorant 优化

#### 预期总工期

- **乐观估计**：20 周（5 个月）达到 M4
- **现实估计**：28 周（7 个月）包含调试与迭代
- **风险缓冲**：+25%（7 周）应对 MIS/体渲调优

### 7.4 最终建议

**建议采用"验证驱动"的增量策略**（符合 SRS 方法论）：

1. **先建立 M1.5 基准**（提供验证尺子）
2. **再实现 M3 MIS**（有基准保证质量）
3. **最后攻克 M4 体渲**（最高风险留在后期）

**不建议**跳过 M1.5 直接实现 M3/M4，因为缺少验证基准会导致：
- 实现错误难以发现（可能数月后才发现物理错误）
- 无法量化性能-质量权衡
- 违反 SRS "验证前置" 原则

### 7.5 项目亮点

尽管进度偏离预期，但项目在以下方面表现优秀：

1. **代码质量高**：结构体对齐验证、防御式编程、详细注释
2. **架构清晰**：模块划分合理，Vulkan 抽象层设计良好
3. **科研严谨**：光谱数据来源追踪、sRGB 上采样门控
4. **文档完善**：SRS V4、时间表、代码注释均详细

这些基础工作为后续实现 M3/M4 奠定了坚实基础。

---

## 附录 A：TODO 统计表

| 优先级 | 数量 | 主要位置 |
|--------|------|----------|
| P0 | 3 | IBL 实现、性能日志、Ray differential |
| P1 | 8 | MIS 采样、Delta-tracking、IR 材质加载 |
| P2 | 10 | 验证脚本、传感器链、硬编码值清理 |
| **总计** | **21** | - |

## 附录 B：关键文件清单

### 需要修改的核心文件（M2-M4）

1. **`src/shaders/raygen.rgen`** - 添加带通循环、MIS 采样
2. **`src/shaders/closesthit.rchit`** - 添加 IBL、Delta-tracking、体渲染
3. **`src/app/main.cpp`** - 添加性能日志、驱动 HS-OFF 循环
4. **`src/libQuantiloom/io/GltfLoader.cpp`** - 实现 IR 材质扩展解析
5. **`src/libQuantiloom/io/SpectralIO.cpp`** - 实现 CSV 光谱曲线加载

### 需要新增的文件

1. **`scripts/validation/benchmark_a.py`** - 基准 A（Cornell Box）
2. **`scripts/validation/benchmark_b.py`** - 基准 B（带通积分）
3. **`scripts/validation/benchmark_c.py`** - 基准 C（窄带特征）
4. **`scripts/validation/benchmark_d.py`** - 基准 D（体渲染）
5. **`src/libQuantiloom/sensor/SensorChain.hpp`** - 传感器链模块

---

**报告结束**

此报告为 Quantiloom 项目提供了全面的现状分析和详细的实施路径。建议项目组以此为基础，调整开发计划，优先完成验证基准（M1.5），再逐步攻克核心算法（M3/M4）。
