  scripts/spectral-baker/
  ├── bake_spectral.py        # 主入口 (CLI)
  ├── usgs_loader.py          # USGS ASCII 解析器
  ├── spectral_processor.py   # 重采样 + NMF 核心算法
  ├── exporter.py             # 二进制/JSON 导出
  ├── validator.py            # 验证图表生成
  ├── config.toml             # 配置文件
  ├── requirements.txt        # Python 依赖
  ├── SRS.md                  # 技术规格文档
  └── output/
      └── plots/              # 验证图表
          ├── basis_vis.png
          ├── basis_swir.png
          ├── error_histogram_*.png
          ├── worst_*.png
          ├── best_*.png
          └── validation_report.txt

  测试结果

  | 指标               | VIS Band | SWIR Band | 目标      |
  |--------------------|----------|-----------|-----------|
  | Mean RMSE          | 0.0036   | 0.0085    | < 0.03 ✅ |
  | Good (RMSE < 0.03) | 99.7%    | 97.3%     | > 95% ✅  |
  | Bad (RMSE > 0.10)  | 0.0%     | 0.0%      | < 5% ✅   |
  | Explained Variance | 99.95%   | 99.79%    | > 98% ✅  |

  处理速度：300 材质 → 4.9 秒

  输出文件

  | 文件                      | 大小   | 用途                 |
  |---------------------------|--------|----------------------|
  | quantiloom_basis_v2.qlbin | 69 KB  | 供 C++ 加载的全局基函数 |
  | quantiloom_materials.json | 481 KB | 材质权重数据库       |
  | material_summary.csv      | 47 KB  | 统计分析用           |

  使用方法

  cd scripts/spectral-baker

  # 1. 扫描可用材质
  python bake_spectral.py --scan-only

  # 2. 测试单个材质
  python bake_spectral.py --material "Aluminum" --plot

  # 3. 基数对比实验（论文用）
  python bake_spectral.py --experiment-basis 4,8,16,32

  # 4. 全量处理（1374 个 ASD 材质）
  python bake_spectral.py

  已知问题

  1. USGS 数据质量问题：约 20 个材质有负反射率（已自动 clamp 到 [0, 1]）
  2. NMF 收敛：部分情况下达到最大迭代次数，可在 config.toml 中增加 max_iter

  # glTF 2.0 使用

  ## glTF 材质引用(usgs):
  ```json
  {
    "materials": [{
      "name": "Something_Something_Plastic_Surface",
      "pbrMetallicRoughness": { ... },
      "extras": {
        "quantiloom_material": {
            "name": "Plastic_PETE GDS383 Clrbluis",
            "type": "quantiloom_usgs"
        }
      }
    }]
  }
  ```

  ## glTF 材质引用(refractiveindex):
  ```json
  {
    "materials": [{
      "name": "Silver_Mirror",
      "pbrMetallicRoughness": { ... },
      "extras": {
        "quantiloom_material": {
            "name": "Ag (Babar)",
            "type": "quantiloom_refidx"
        }
      }
    }]
  }
  ```

# 多数据源支持 (NEW)

SpectralBaker 现在支持两种数据源：

## 1. USGS Spectral Library
- **数据格式**：ASCII 文本格式 (.txt)
- **物理量**：直接测量的反射率 (reflectance)
- **波段覆盖**：VIS + NIR + SWIR (0.35-2.5 µm)
- **材质数量**：~1400 个 (ASD 仪器)
- **材质类型**：矿物、岩石、植被、人工材料等
- **配置文件**：`config.toml`

```bash
# 使用 USGS 数据源
python bake_spectral.py --config config.toml
```

## 2. RefractiveIndex.info Database
- **数据格式**：YAML 格式 (.yml)
- **物理量**：复折射率 (n, k) → 自动转换为反射率
- **波段覆盖**：VIS + NIR + SWIR + MWIR + LWIR (0.2-15 µm)
- **材质数量**：~1000+ 个 (nk 数据)
- **材质类型**：金属、半导体、玻璃、有机物等
- **配置文件**：`config_refidx.toml`

```bash
# 使用 RefractiveIndex.info 数据源
python bake_spectral.py --config config_refidx.toml --max-materials 100
```

## 新增波段 (NEW)

现在支持 5 个光谱波段：

| 波段 | 范围 (µm) | 采样间隔 | 采样点数 | Basis 数量 | 数据源 |
|------|----------|---------|---------|-----------|--------|
| **VIS** | 0.35 - 0.78 | 2 nm | 216 | 16 | USGS + RefIdx |
| **NIR** | 0.78 - 1.10 | 2 nm | 161 | 16 | USGS + RefIdx |
| **SWIR** | 1.10 - 2.50 | 2 nm | 701 | 32 | USGS + RefIdx |
| **MWIR** | 2.50 - 6.50 | 5 nm | 801 | 32 | RefIdx only |
| **LWIR** | 6.50 - 15.00 | 10 nm | 851 | 32 | RefIdx only |

**注意**：MWIR 和 LWIR 仅在使用 RefractiveIndex 数据源时可用。

## 物理转换

RefractiveIndex.info 数据库提供复折射率 (n + ik)，通过 Fresnel 方程自动转换为反射率：

```
R = ((n-1)² + k²) / ((n+1)² + k²)
```

这使得金属（高 k 值）和电介质（低 k 值）材质都能正确处理。

## 扫描可用材质

```bash
# 扫描 USGS 数据源
python bake_spectral.py --config config.toml --scan-only

# 扫描 RefractiveIndex 数据源
python bake_spectral.py --config config_refidx.toml --scan-only
```

---

# 版本历史 (Version History)

## v3.0 (2025-12-24)

### 新增功能
- ✅ **多数据源支持**: 新增 RefractiveIndex.info 数据库（1000+ 材质）
- ✅ **扩展波段覆盖**: 新增 MWIR (2.5-6.5µm) 和 LWIR (6.5-15µm) 波段
- ✅ **物理转换**: 自动将复折射率 (n,k) 转换为反射率
- ✅ **5 波段支持**: VIS + NIR + SWIR + MWIR + LWIR (0.35-15 µm)

### 改进
- ✅ **唯一材质名**: 使用 `MaterialSymbol (Author)` 格式，消除重名问题
- ✅ **自适应 NMF**: 材质数不足时自动调整初始化方法
- ✅ **鲁棒统计**: 修复平坦光谱（金属材质）的 explained variance 计算
- ✅ **二进制格式 v3**: 支持动态波段数量

### 文件
- 新增: `refractiveindex_loader.py`
- 新增: `config_refidx.toml`
- 修改: `bake_spectral.py`, `spectral_processor.py`, `exporter.py`, `config.toml`

### 性能
- RefractiveIndex 数据源: 585 材质 × 5 波段 → 18.9 秒
- USGS 数据源: 保持原有性能 (300 材质 → 4.9 秒)

---

## v2.0 (2025-12-23)

- ✅ 每波段独立 basis 数量配置
- ✅ 3 波段支持 (VIS, NIR, SWIR)
- ✅ 二进制格式 v2

## v1.0 (2025-12-22)

- ✅ 初始版本 (仅 USGS 数据源)
- ✅ 固定 basis 数量
- ✅ VIS + SWIR 两波段