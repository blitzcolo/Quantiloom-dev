  scripts/spectral-baker/
  ├── bake_spectral.py            # 主入口 (CLI)
  ├── usgs_loader.py              # USGS ASCII 解析器
  ├── refractiveindex_loader.py   # RefractiveIndex.info YAML (n,k) 解析器
  ├── ecostress_loader.py         # ECOSTRESS .spectrum.txt 解析器
  ├── spectral_processor.py       # 重采样 + 覆盖度判定 + NMF 核心算法
  ├── exporter.py                 # 二进制/JSON/CSV 导出
  ├── validator.py                # 验证图表生成
  ├── config.toml                 # USGS 配置
  ├── config_refidx.toml          # RefractiveIndex.info 配置
  ├── config_ecostress.toml       # ECOSTRESS 配置
  ├── requirements.txt            # Python 依赖
  ├── SRS.md                      # 技术规格文档
  └── output/
      ├── plots/                  # 验证图表 (USGS)
      ├── plots_refidx/           # 验证图表 (RefractiveIndex)
      └── plots_ecostress/        # 验证图表 (ECOSTRESS)
          ├── basis_*.png
          ├── error_histogram_*.png
          ├── worst_*.png
          ├── best_*.png
          └── validation_report.txt

  注意：产物本身不写在 output/ 下。三个 config 的 basis_file / material_json /
  summary_csv 都直接指向 ../../assets/spectral/，也就是场景 TOML 实际加载的路径，
  烘焙会覆盖线上资产。output/ 只放图表。

  输出文件

  产物名由 config 的 [output] 段决定，与 assets/configs/*.toml 里的
  basis_file / materials_json 一一对应：

  | config                | 产物                             | 波段数 | 大小   |
  |-----------------------|----------------------------------|--------|--------|
  | config.toml           | quantiloom_basis_v3_usgs.qlbin   | 3      | 111 KB |
  |                       | quantiloom_materials_usgs.json   |        | 5.6 MB |
  |                       | material_summary_usgs.csv        |        | 269 KB |
  | config_refidx.toml    | quantiloom_basis_v3_rii.qlbin    | 5      | 318 KB |
  |                       | quantiloom_materials_rii.json    |        | 2.7 MB |
  |                       | material_summary_rii.csv         |        | 90 KB  |
  | config_ecostress.toml | quantiloom_basis_v3_ecostress.qlbin | 5   | —      |
  |                       | quantiloom_materials_ecostress.json |     | —      |
  |                       | material_summary_ecostress.csv   |        | —      |

  基函数文件大小 = 64 (头) + 16 × 波段数 + Σ(n_basis × n_samples × 4)。
  ECOSTRESS 尚未做过全量烘焙，故无实测体积。

  ⚠ 表中 USGS 一行是重新烘焙后的预期值，assets/spectral/ 里现存的 usgs 产物是
  覆盖度检查加入之前生成的：basis 仍是 318 KB / 5 波段，其中 MWIR 与 LWIR 是边缘
  钳位外推出来的伪数据，materials json 里也带着这两个波段的权重。重新烘焙会把它们
  去掉。rii 一行是实测值（RefractiveIndex 覆盖全部 5 个波段，不受影响）。
  json 与 csv 重新烘焙后还会各多一个 coverage 字段。

  使用方法

  cd scripts/spectral-baker

  # 1. 扫描可用材质（三种数据源都支持）
  python bake_spectral.py --config config.toml --scan-only

  # 2. 测试单个材质
  python bake_spectral.py --config config.toml --material "Aluminum" --plot

  # 3. 基数对比实验（论文用）
  python bake_spectral.py --config config.toml --experiment-basis 4,8,16,32

  # 4. 全量处理（USGS: 1374 个 ASD 材质）
  python bake_spectral.py --config config.toml

  --max-materials N 做小样本试跑。因为产物路径就是线上资产，试跑不会覆盖已存在的
  产物，除非显式加 --force。

  已知问题

  1. USGS 数据质量问题：约 20 个材质有负反射率（已自动 clamp 到 [0, 1]）
  2. NMF 收敛：部分情况下达到最大迭代次数，可在 config.toml 中增加 max_iter
  3. 重采样对超出数据源范围的波长做边缘钳位外推，不报错。覆盖度为 0 的波段会被
     整段跳过，但部分覆盖的波段仍会产出，其指标偏乐观 —— 看 coverage 字段。

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

# 多数据源支持

SpectralBaker 支持三种数据源：

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

## 3. ECOSTRESS Spectral Library
- **数据格式**：ASCII 文本 (.spectrum.txt)
- **物理量**：反射率或透射率（百分比自动归一化）
- **波段覆盖**：0.30-25.04 µm（并集），单个材质覆盖范围差异很大
- **材质数量**：3450 个可用（另有 1 个解析失败）
- **材质类型**：矿物 1608、植被 1040、岩石 470、非光合植被 123、人造物 72、
  陨石 59、土壤 69、水 9
- **配置文件**：`config_ecostress.toml`

```bash
python bake_spectral.py --config config_ecostress.toml
```

## 波段与覆盖度

支持 5 个光谱波段：

| 波段 | 范围 (µm) | 采样间隔 | 采样点数 | Basis 数量 | 覆盖它的数据源 |
|------|----------|---------|---------|-----------|--------|
| **VIS** | 0.35 - 0.78 | 2 nm | 216 | 16 | USGS + RefIdx + ECOSTRESS |
| **NIR** | 0.78 - 1.10 | 2 nm | 161 | 16 | USGS + RefIdx + ECOSTRESS |
| **SWIR** | 1.10 - 2.50 | 2 nm | 701 | 32 | USGS + RefIdx + ECOSTRESS |
| **MWIR** | 2.50 - 6.50 | 5 nm | 801 | 32 | RefIdx + ECOSTRESS |
| **LWIR** | 6.50 - 15.00 | 10 nm | 851 | 32 | RefIdx + ECOSTRESS |

（ECOSTRESS 的 basis 数量在 config_ecostress.toml 里 MWIR/LWIR 设为 48。）

**重采样对超出数据源范围的波长做边缘钳位，不是报错。** 这会把一个没有数据的波段
变成一条水平线：秩为 1，NMF 完美重建，报出 `RMSE 0.000000 / 解释方差 100%` ——
比真实测量的波段还漂亮。所以每个波段都会算一个覆盖度：

- 覆盖度为 0（没有任何材质触及该波段）→ **整段跳过**，不写进 basis 文件，
  也不写进 materials json，控制台打印 `NOT BAKED (no source data)`。
  用 USGS 烘焙时 MWIR 和 LWIR 就是这种情况。
- 覆盖度介于 0 和 1 之间 → 照常产出，但控制台、json 的 `coverage` 字段和 csv 的
  `<band>_coverage` 列都会标出有多少是外推的。此时 RMSE 和解释方差偏乐观。

**引用质量指标前先看覆盖度。**

## 物理转换

RefractiveIndex.info 数据库提供复折射率 (n + ik)，通过 Fresnel 方程自动转换为反射率：

```
R = ((n-1)² + k²) / ((n+1)² + k²)
```

这使得金属（高 k 值）和电介质（低 k 值）材质都能正确处理。

## 扫描可用材质

```bash
python bake_spectral.py --config config.toml           --scan-only
python bake_spectral.py --config config_refidx.toml    --scan-only
python bake_spectral.py --config config_ecostress.toml --scan-only
```

ECOSTRESS 的扫描还会打印波长覆盖并集 —— 烘焙前先看这个，就能知道哪些波段会被跳过。

---

# 版本历史 (Version History)

## v3.1 (2026-07-25)

### 修复
- ✅ **不再产出伪造波段**: 数据源完全不覆盖的波段被整段跳过，而不是用边缘钳位
  外推的水平线拟合出 `RMSE 0 / 100%` 的假完美结果
- ✅ **覆盖度可见**: 每个波段的覆盖比例进入控制台、materials json (`coverage`)
  和 summary csv (`<band>_coverage`)
- ✅ **产物名与渲染器一致**: config.toml / config_refidx.toml 之前写的
  `quantiloom_basis_v1.bin` 无人加载，现改为场景 TOML 实际读的
  `quantiloom_basis_v3_{usgs,rii}.qlbin`
- ✅ **summary_csv 配置项生效**: 之前被硬编码的 `material_summary.csv` 忽略
- ✅ **试跑保护**: `--max-materials` 不会覆盖已存在的线上产物，除非 `--force`
- ✅ **`--scan-only` 支持 ECOSTRESS**: 之前只有 usgs / refractiveindex 两个分支

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