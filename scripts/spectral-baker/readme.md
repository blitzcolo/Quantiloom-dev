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