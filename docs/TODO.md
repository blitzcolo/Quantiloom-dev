# Quantiloom 待办

2026-07-25 整理。全部条目在整理当天实际复现过,不是推测。
每条只写**问题 / 现状 / 期待目标**,不预设解决方案。

---

## A. 代码正确性

### A1. `SpectralReconstructor.cpp` 可能读取未初始化的变量

**问题** `f32 value;` 在第 76 行声明时未初始化,随后在一个 `switch` 里按插值方式赋值,
第 119 行无条件用于 `fullCube(x, y, map.targetBand) = value;`。若 `switch` 未命中任何
分支,写进光谱立方体的就是栈上的垃圾值。

**现状** MSVC 每次构建都报 `warning C4701: 使用了可能未初始化的局部变量"value"`。
警告一直存在,构建不中断,没有测试覆盖到未命中分支的情形。

**期待目标** 要么证明所有输入下 `switch` 必然命中(并让编译器也能看出来),要么让未命中
时有明确定义的行为。C4701 从构建输出里消失。

### A2. `src/app/main.cpp` 的 28 条 clang-tidy 告警

**问题** 全仓库调优后的 lint 配置下,`main.cpp` 是唯一的异常值。构成:

| 数量 | check |
|---|---|
| 11 | `bugprone-implicit-widening-of-multiplication-result` |
| 8 | `bugprone-narrowing-conversions` |
| 4 | `misc-use-internal-linkage` |
| 3 | `misc-use-anonymous-namespace` |
| 1 | `misc-misplaced-const` |
| 1 | `bugprone-exception-escape` |

前两类合计 19 条,是数值代码里真实的缺陷类别(`int*int` 隐式加宽赋给 `size_t`、
宽类型窄化)。对一个逐像素、逐波段做索引运算的渲染器,这些不是风格问题。

**现状** 其余 5 个抽样文件合计只有 5 条(4 个文件为 0)。`main.cpp` 是约千行的 CLI
入口,大量解析 TOML 后的数值转换。告警通过 clangd 实时可见,但无人处理。

**期待目标** 19 条数值类告警逐条判定:是真缺陷则修,是误报则在该行说明理由。
`main.cpp` 降到与其他文件同一量级。

---

## B. spectral-baker

### B1. `--scan-only` 对 ECOSTRESS 数据源是坏的,且失败不可见

**问题** `bake_spectral.py --config config_ecostress.toml --scan-only` 打印表头后
报 `Unknown source_type: ecostress`,**但退出码仍是 0**。烘焙路径(第 164 行)是支持
ecostress 的,只有扫描路径缺这个分支。

**现状** 扫描函数只实现了 `usgs` 和 `refractiveindex` 两个分支。调用方(人或脚本)
按退出码判断会认为成功。

**期待目标** 扫描支持 ecostress;或至少失败时退出码非 0,让上层能察觉。

### B2. 三个 config 的输出文件名,没有一个与渲染器实际加载的对得上

**问题**

| config 写出 | 场景 TOML 实际加载 |
|---|---|
| `quantiloom_basis_v1.bin` | `quantiloom_basis_v3_usgs.qlbin` |
| `quantiloom_basis_refidx_v1.bin` | `quantiloom_basis_v3_rii.qlbin` |
| `quantiloom_materials{,_refidx}.json` | `quantiloom_materials_{usgs,rii}.json` |

**现状** `assets/configs/cornell_box_vis.toml:12-13` 用绝对路径加载 `v3_*` 产物。
烘焙一次不会覆盖线上文件,中间隐含一个无人记录的改名步骤。烘焙完直接渲染,用的还是旧
basis,而且没有任何提示。

**期待目标** 配置声明的输出名与渲染器加载的名字一致,或改名步骤成为流程的显式一环。

### B3. USGS 源下 MWIR/LWIR 的退化输出看起来像完美拟合

**问题** 用 USGS 数据源烘焙时,MWIR 和 LWIR 波段照常输出,并报
`Mean RMSE: 0.000000`、`Explained Variance: 100.00%`、`Good materials: 100.0%`。
但 USGS 只覆盖 0.35-2.5 µm,这两个波段**根本没有数据** —— 这是空区间上的退化结果,
不是拟合得好。

**现状** 数字比真实有效的 VIS/SWIR 波段还漂亮,极易被当作质量证据引用到报告里。
`spectral-bake` skill 已记录此坑,但程序本身不作任何提示。

**期待目标** 数据源未覆盖的波段,要么不产出,要么在输出中明确标记为"无数据",
不能与真实拟合结果用同一种形式呈现。

### B4. `readme.md` 描述的产物没有任何 config 会生成

**问题** `scripts/spectral-baker/readme.md` 写明输出 `quantiloom_basis_v2.qlbin`
(69 KB)。三个 config 没有一个产出这个名字。

**现状** 文档与代码脱节,新人照 readme 操作会找不到产物。

**期待目标** readme 描述的产物名、体积、路径与实际一致。

### B5. `SRS.md` 未纳入版本管理

**问题** `scripts/spectral-baker/SRS.md`(500 行技术规格)一直是未跟踪状态。

**现状** 它是 spectral-baker 唯一的设计文档,只存在于这台机器上,误删即永久丢失。
本次整理期间已有几个同类未跟踪文档消失。

**期待目标** 明确它的去留:入库,或确认已废弃后删除。

---

## C. 测试

### C1. 24 个测试长期 SKIPPED

**问题** 全量套件 863 个用例中,24 个从不执行,分布在 4 个 suite:
`BC7CompressionActualTest`、`GltfLoaderTest`、`SpectralBasisLoaderTest`、
`SpectralCubeIOTest`。

**现状** 原因是 `build/` 关闭了 BC7、且 glTF 样例资产不可用。这被当作正常基线接受
(`CLAUDE.md` 也这么写),但意味着 glTF 加载和 EXR 光谱立方体读写的实际覆盖为零。

**期待目标** 明确每个 suite 是"环境所限,可接受"还是"应该跑但没跑"。属于后者的补上
执行条件。

### C2. 一个测试被禁用且无说明

**问题** `tests/test_scene/test_camera.cpp:356` 的
`TEST(CameraTest, DISABLED_PositionSameAsLookAt)`。

**现状** GoogleTest 每次运行都提示 `YOU HAVE 1 DISABLED TEST`。代码里没有注释说明
为何禁用、何时能恢复。

**期待目标** 修复后启用,或删除并说明该场景为何不需要覆盖。

### C3. 没有任何 CI 执行测试

**问题** 套件只有 3 秒,但没有自动化执行。

**现状** 唯一的门禁是 `build_wsl.sh` 里的手工构建步骤。SRS 的 NFR-MAINT-03 也记录了
"CI 只构建不运行测试"。这意味着套件可以数月不被执行,测试与实现悄悄脱节 ——
`run-tests` skill 里那段"红灯未必是代码 bug,可能是测试过期"的判断规则,正是这个缺口
的产物。

**期待目标** 每次推送自动执行套件,红灯可见。

---

## D. 文档与代码不符

### D1. `README.md` 严重过期

**问题** 实测 5 处与磁盘不符:

| README 声称 | 实际 |
|---|---|
| 依赖含 **hdf5** | `CMakeLists.txt:187` 明确写 `HDF5 dependency removed` |
| `build_linux.sh` | 不存在,实际是 `build_wsl.sh` |
| `scripts/validation/run_benchmark.py` 等 3 个脚本 | 全部不存在,该目录只有 `.gitkeep` |
| `docs/DATA_PREPARATION.md` | 不存在 |
| 目录树 | 未提及实际存在的 `atmos/`、`libSpectraForge/`、`src/tools/` |

**现状** `CLAUDE.md` 里已加了一行警告"README 过期,依赖前先核对 CMake 和磁盘",
这是绕过而非修复。

**期待目标** README 与磁盘一致,`CLAUDE.md` 里那行警告可以删掉。

### D2. `tests/CMakeLists.txt` 头注释给出的命令跑不通

**问题** 文件开头 15-25 行推荐 `mkdir build && cd build && cmake ..` 和
`ctest --output-on-failure`。前者与实际的 Visual Studio 生成器配置不符,后者在本仓库
只能跑一个聚合测试、无法选中单个用例。

**现状** `tests/CLAUDE.md` 已记录"此注释过期",同样是绕过。读源文件的人仍会被误导。

**期待目标** 注释里的命令与 `build_wsl.sh` 的真实流程一致。

---

## E. 仓库卫生

### E1. `scripts/atmosphere-qlut-gen/` 的删除未提交

**问题** 4 个文件在工作区已删除,但仍是 git 的 ` D` 状态。

**现状** 该工具已废弃。删除动作悬空,`git status` 每次都显示。

**期待目标** 删除进入历史。

### E2. `src/libQuantiloom/renderer/1111.js`

**问题** 一个 3.5 KB 的 JavaScript 文件躺在 C++ 渲染器模块目录里,未跟踪。

**现状** 与项目技术栈无关,疑似临时文件遗留。

**期待目标** 确认无用后删除。

### E3. 未跟踪资产的去留未定

**问题** 以下文件长期未跟踪,既没入库也没进 `.gitignore`:
`assets/maps/{milkyway_2020_4k,moonless_golf_4k,qwantani_noon_puresky_4k}.exr`、
`assets/models/usd_test/*.usdz`、`assets/spectral/material_summary_{rii,usgs}.csv`、
根目录的 `中期报告.md`。

**现状** `material_summary_*.csv` 是查材质的入口(`spectral-bake` skill 引导用它),
但换台机器就没有。环境贴图和 usdz 模型体积大,是否该入库需要判断。

**期待目标** 每个文件有明确归属:入库、gitignore、或删除。

---

## F. 工具链维护

### F1. clangd shim 是机器状态,换机器静默失效

**问题** 代码智能依赖 `~/.local/bin/clangd` 这个包装脚本(转调 Windows clangd.exe
并做 WSL→盘符路径映射)。它不在仓库里。

**现状** 换机器或新人加入时,LSP 会静默不工作,没有任何提示。重建步骤已写在
`docs/CLAUDE_CODE_SETUP.md`,但要人主动去读。

**期待目标** 缺失时能被察觉,而不是安静地退化成 grep。

### F2. `build-cdb` 需要手动重新生成

**问题** 新增源文件或改动 CMakeLists 后,编译数据库不会自动更新。

**现状** clangd 对不在数据库里的文件退回启发式推断,精度下降但**不报错**。
需要人记得跑 `./scripts/gen_compile_commands.sh`(13 秒)。

**期待目标** 数据库过期能被察觉。

### F3. `findReferences` 结果重复一倍

**问题** 每个文件被列两次:一次 WSL 相对路径,一次 `D:/Quantiloom-dev/...`。

**现状** 路径映射的反向转换不对称,因为 `compile_commands.json` 存的是原生 Windows
路径。引用位置本身正确,只是输出翻倍。已记录在 `.clangd` 注释里。

**期待目标** 单次查询只返回一组结果。

### F4. `.claude/settings.json` 含机器特定绝对路径

**问题** `additionalDirectories` 和 `autoMode` 里写着 `/mnt/d/Quantiloom-SDK` 等
本机路径,而该文件现在已入库、会分发给所有人。

**现状** 目前是单人开发,无实际影响。

**期待目标** 第二个开发者加入前,机器特定配置移到 `.claude/settings.local.json`
(已被 gitignore)。
