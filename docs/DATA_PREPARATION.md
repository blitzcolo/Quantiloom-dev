# Quantiloom 数据准备指南

本文档说明运行Quantiloom渲染器需要提前准备的数据文件，以及如何生成/获取这些数据。

## 📦 必需数据 (Required Data)

### 1. **BRDF Integration LUT** ✅ 已自动缓存
**用途**: IBL镜面反射的split-sum approximation
**格式**: EXR (RG32F, 512×512)
**生成时间**: 首次生成约6秒，之后从缓存加载 <100ms
**缓存位置**: `cache/brdf_lut_512x1024.exr`

**说明**:
- 代码已自动处理缓存逻辑
- 首次运行会生成并保存到`cache/`目录
- 后续运行直接加载，启动速度提升60倍
- 无需手动准备

---

## 🌍 可选数据 (Optional Data)

### 2. **大气透射率LUT** (Atmospheric Transmittance LUT)
**用途**: Beer-Lambert法则的大气衰减查找表
**格式**: HDF5 (`.h5`)
**当前状态**: ⚠️ 使用简化的标量值 `lighting.transmittance` (默认0.9)

**HDF5结构** (如果使用完整LUT):
```
/wavelengths        [n_bands] float32  # 波长数组 (nm)
/transmittance      [n_bands] float32  # 透射率 τ(λ) [0, 1]
/solar_irradiance   [n_bands] float32  # 太阳辐照度 (W·m⁻²·nm⁻¹)
/sky_radiance       [n_bands] float32  # 天空辐射率 (W·sr⁻¹·m⁻²·nm⁻¹)
/metadata           group              # 元数据 (observer altitude, aerosol model, etc.)
```

**数据来源**:
- **MODTRAN**: 大气辐射传输模拟软件
  - 商业软件，需要许可证
  - 可生成高精度大气透射率LUT
  - 支持不同海拔、气溶胶模型、视角等参数

- **libRadtran**: 开源替代方案
  - 免费开源 (GPL)
  - 网址: http://www.libradtran.org/
  - 可输出类似MODTRAN的结果

- **简化模型** (当前实现):
  ```toml
  [lighting]
  transmittance = 0.9  # 0.9 = 晴朗天气，0.7 = 轻雾，0.5 = 重雾
  ```

**如何生成** (使用libRadtran示例):
```bash
# 安装libRadtran
wget http://www.libradtran.org/download/libRadtran-2.0.5.tar.gz
tar xzf libRadtran-2.0.5.tar.gz
cd libRadtran-2.0.5
./configure && make

# 生成透射率LUT
cat > atmLUT.inp <<EOF
atmosphere_file afglus.dat
source solar kurudz
wavelength 400 800  # 可见光范围
output_quantity transmittance
quiet
EOF

./bin/uvspec < atmLUT.inp > transmittance.dat
```

然后使用Python脚本转换为HDF5:
```python
import h5py
import numpy as np

# 解析libRadtran输出
data = np.loadtxt('transmittance.dat')
wavelengths = data[:, 0]  # nm
transmittance = data[:, 1]

# 保存为HDF5
with h5py.File('atmosphere_lut.h5', 'w') as f:
    f.create_dataset('wavelengths', data=wavelengths)
    f.create_dataset('transmittance', data=transmittance)

    # 元数据
    meta = f.create_group('metadata')
    meta.attrs['source'] = 'libRadtran 2.0.5'
    meta.attrs['atmosphere_model'] = 'afglus'
    meta.attrs['observer_altitude_km'] = 0.0
```

---

### 3. **太阳光谱辐照度** (Solar Spectral Irradiance)
**用途**: 直射阳光的波长相关强度
**格式**: CSV或嵌入到大气LUT中
**当前状态**: ⚠️ 使用简化的RGB值 `lighting.sun_radiance`

**标准数据集**:
- **ASTM G173-03**: 标准太阳光谱
  - AM1.5 (地表标准，太阳高度角37°)
  - 波长范围: 280-4000 nm
  - 下载: https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

- **Kurucz 2005**: 高分辨率太阳光谱
  - 波长范围: 200-1000000 nm
  - 分辨率: 0.01 nm (可见光)
  - 下载: http://kurucz.harvard.edu/sun.html

**CSV格式示例** (`data/solar_spectrum_am15.csv`):
```csv
# ASTM G173-03 AM1.5 Global Tilt
# wavelength_nm, irradiance_W_m2_nm
400.0, 1.677
410.0, 1.828
420.0, 1.949
...
800.0, 0.966
```

**如何使用**:
```cpp
// 在config.toml中指定路径
[lighting]
solar_spectrum = "data/solar_spectrum_am15.csv"

// 或者使用简化的RGB值（当前方法）
sun_radiance = [100.0, 100.0, 95.0]  # W·sr⁻¹·m⁻²
```

---

### 4. **IR材质光谱曲线** (IR Material Spectral Curves)
**用途**: 红外波段材质的emissivity/reflectance/transmittance
**格式**: CSV (wavelength_nm, value)
**当前状态**: ✅ 已支持，通过glTF扩展加载

**数据来源**:
- **ASTER Spectral Library**: NASA/JPL光谱库
  - 2400+ 材质光谱
  - 波长范围: 0.4-15 μm
  - 下载: https://speclib.jpl.nasa.gov/

- **ECOSTRESS Spectral Library**: 地球材料光谱
  - 覆盖岩石、土壤、植被、水等
  - 下载: https://speclib.jpl.nasa.gov/

- **自测量数据**: 使用FTIR光谱仪测量

**CSV格式** (`materials/aluminum_emissivity.csv`):
```csv
# Polished aluminum emissivity (LWIR)
# wavelength_nm, emissivity
8000, 0.05
8500, 0.06
9000, 0.07
...
12000, 0.10
```

**glTF集成** (在`.gltf`文件中):
```json
{
  "materials": [
    {
      "name": "Aluminum",
      "pbrMetallicRoughness": { ... },
      "extensions": {
        "QUANTILOOM_material_ir": {
          "emissivityCurve": "materials/aluminum_emissivity.csv",
          "reflectanceCurve": "materials/aluminum_reflectance.csv",
          "transmittanceCurve": "materials/aluminum_transmittance.csv",
          "temperature_K": 300.0
        }
      }
    }
  ]
}
```

---

## 📋 数据准备清单

### 最小配置 (能运行):
- ✅ **BRDF LUT**: 自动生成并缓存
- ✅ **配置文件**: `config.toml` (指定场景、分辨率、光照等)
- ✅ **3D模型**: glTF文件 (`.gltf` 或 `.glb`)

### 完整配置 (高保真渲染):
1. ✅ **BRDF LUT** - 自动处理
2. ⚠️ **大气LUT** (`atmosphere_lut.h5`) - 可选，当前用标量近似
3. ⚠️ **太阳光谱** (`solar_spectrum.csv`) - 可选，当前用RGB近似
4. ✅ **IR材质曲线** (`materials/*.csv`) - 已支持加载

---

## 🛠️ 快速开始示例

### 1. RGB模式 (最简单)
```toml
# config.toml
[spectral]
mode = "rgb"

[lighting]
sun_direction = [0.5, 0.8, 0.3]
sun_radiance = [100.0, 100.0, 95.0]
sky_radiance = [30.0, 35.0, 40.0]
transmittance = 0.9  # 晴朗天气

[scene]
model_path = "models/scene.gltf"
```

### 2. 单波长模式 (光谱渲染)
```toml
[spectral]
mode = "single"
wavelength_nm = 550.0  # 绿光

[lighting]
sun_radiance = [100.0, 100.0, 95.0]  # 会转换为550nm的spectral radiance
```

### 3. LWIR模式 (红外成像)
```toml
[spectral]
mode = "lwir_fused"  # 8-12 μm长波红外

[lighting]
transmittance = 0.85  # 红外大气透射率稍高

[scene]
model_path = "models/thermal_scene.gltf"  # 带IR extension的场景
```

确保场景中的材质有IR数据：
```json
{
  "extensions": {
    "QUANTILOOM_material_ir": {
      "emissivityCurve": "data/aster/concrete_emissivity.csv",
      "temperature_K": 310.0  # 37°C (人体/建筑物表面)
    }
  }
}
```

---

## 📚 推荐数据集

| 数据类型 | 推荐来源 | 许可证 | 质量 |
|---------|---------|--------|------|
| 太阳光谱 | ASTM G173-03 | Public | ⭐⭐⭐⭐⭐ |
| 大气透射率 | libRadtran | GPL | ⭐⭐⭐⭐ |
| IR材质 | ASTER库 | Public | ⭐⭐⭐⭐⭐ |
| 环境贴图 | Poly Haven | CC0 | ⭐⭐⭐⭐⭐ |

---

## ⚡ 性能优化

### BRDF LUT缓存 (已实现)
- **首次运行**: 6秒生成 + 保存
- **后续运行**: <100ms 加载
- **加速**: ~60倍

### 建议的缓存策略
```
项目目录/
├── cache/
│   ├── brdf_lut_512x1024.exr        # BRDF LUT (自动)
│   └── atmosphere_lut.h5             # 大气LUT (手动准备)
├── data/
│   ├── solar_spectrum_am15.csv       # 太阳光谱
│   └── materials/                    # IR材质曲线
│       ├── aluminum_emissivity.csv
│       ├── concrete_emissivity.csv
│       └── glass_transmittance.csv
└── models/
    └── scene.gltf                    # 3D模型
```

---

## 🔗 有用的链接

- **NREL太阳光谱数据**: https://www.nrel.gov/grid/solar-resource/spectra.html
- **libRadtran文档**: http://www.libradtran.org/doc/
- **ASTER光谱库**: https://speclib.jpl.nasa.gov/
- **Poly Haven HDR环境**: https://polyhaven.com/hdris
- **glTF规范**: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

---

## ❓ 常见问题

### Q: 为什么BRDF LUT要缓存？
A: 蒙特卡洛积分计算量大（512×512像素 × 1024采样），但结果是固定的（仅依赖于GGX BRDF模型），适合缓存复用。

### Q: 大气LUT一定要用MODTRAN吗？
A: 不是。对于实时预览，简化的标量transmittance已足够。需要高保真度时才用MODTRAN/libRadtran。

### Q: IR材质数据从哪里来？
A: 优先使用ASTER库（免费公开），或使用FTIR光谱仪实测（需硬件投入）。

### Q: 如何验证数据正确性？
A: 检查Kirchhoff定律: ε(λ) + ρ(λ) + τ(λ) ≤ 1.0，代码会在加载时自动验证。
