# Quantiloom 待办

2026-07-25。全部条目实际复现过,不是推测。
每条只写**问题 / 现状 / 期待目标**,不预设解决方案。

已完成的条目已从本文件移除(A、B、D 组及 E1/E2/F2),记录在 git 历史里。
存活条目沿用原编号,新条目使用未用过的字母,以免与 commit message 里的旧引用混淆。

---

## C. 测试

### C1. 24 个测试长期 SKIPPED

**问题** 全量套件 865 个用例中,24 个从不执行:

| suite | 跳过数 |
|---|---|
| `GltfLoaderTest` | 13 |
| `BC7CompressionActualTest` | 8 |
| `SpectralCubeIOTest` | 2 |
| `SpectralBasisLoaderTest` | 1 |

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

**问题** 套件只有 3 秒,但没有自动化执行。仓库里没有 `.github/workflows`。

**现状** 唯一的门禁是 `build_wsl.sh` 里的手工构建步骤。SRS 的 NFR-MAINT-03 也记录了
"CI 只构建不运行测试"。这意味着套件可以数月不被执行,测试与实现悄悄脱节 ——
`run-tests` skill 里那段"红灯未必是代码 bug,可能是测试过期"的判断规则,正是这个缺口
的产物。

**期待目标** 每次推送自动执行套件,红灯可见。

### C4. `test_ir_energy_conservation.cpp` 的注释与实际状态相反

**问题** 第 139-140 行写着
`// Phase C sentinel fix must make this work; DISABLED_ until then.`,
但紧随其后的 `SentinelDerivedEmissivity_Dielectric` 和 `_Metal` 两个测试**并未被禁用,
而且正常通过**。

**现状** 修复应当早已落地,注释没跟着删。读代码的人会以为这块还没做完,或者反过来
去找一个并不存在的 DISABLED_ 测试。

**期待目标** 注释描述的状态与代码一致。

---

## E. 仓库卫生

### E3. 未跟踪资产的去留未定

**问题** 以下文件长期未跟踪,既没入库也没进 `.gitignore`:
`assets/maps/{milkyway_2020_4k,moonless_golf_4k,qwantani_noon_puresky_4k}.exr`、
`assets/models/usd_test/{Caterpillar_Work_Boot,KV-2_heavy_tank_1940}.usdz`。

**现状** 同批的 `material_summary_*.csv` 已入库。剩下这 5 个都是体积较大的二进制
资产,是否值得进 git 历史需要判断 —— 一旦提交就永久留在历史里。

**期待目标** 每个文件有明确归属:入库、gitignore、或删除。

### E4. `quantiloom_materials_ecostress.json` 体积 20 MB

**问题** 它是仓库里最大的单个文件,已入库(`8833607`)。

**现状** 体积来自 3450 材质 × 5 波段 × 最多 48 个权重,且 `json.dump` 用了
`indent=2`。同源的 usgs 为 3.8 MB、rii 为 2.9 MB。三个文件合计约 27 MB。

**期待目标** 确认这个体积可接受;若不可接受,明确是改紧凑序列化、换二进制格式、
还是不入库。

---

## F. 工具链维护

### F1. clangd shim 是机器状态,换机器静默失效

**问题** 代码智能依赖 `~/.local/bin/clangd` 这个包装脚本(转调 Windows clangd.exe
并做 WSL→盘符路径映射)。它不在仓库里。

**现状** 换机器或新人加入时,LSP 会静默不工作,没有任何提示。重建步骤已写在
`docs/CLAUDE_CODE_SETUP.md`,但要人主动去读。

**期待目标** 缺失时能被察觉,而不是安静地退化成 grep。

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

### F5. 增量构建会静默使用过期的目标文件

**问题** 修改 `SpectralReconstructor.cpp` 后运行
`cmake.exe --build build --config Release -j`,构建返回 0、DLL 被重新链接、
源文件与二进制的时间戳关系也正常(二进制更新),但**那个源文件根本没有重新编译**,
DLL 里仍是旧代码。`touch` 该文件后再构建即恢复正常。

**现状** 没有任何输出提示这件事。当时是靠一个有区分度的新测试失败才发现 —— 如果测试
写成"改与不改都通过"的形式(例如只断言两条等价路径相等),就会直接得出"代码没问题"
的错误结论。这与 F2 是同一类隐性过期,但更难察觉:F2 至少可以主动 `--check`,这里
连可查询的对象都没有。

**期待目标** 要么增量构建可信,要么"改了没生效"这件事能被察觉。

---

## G. 一致性与残留

### G1. NIR 波段范围在三处不一致

**问题**

| 位置 | 声称的范围 |
|---|---|
| `src/shaders/common.hlsli:28` 注释 | 780-1400 nm |
| `src/shaders/README.md` | 780-1400 nm |
| `core/Types.hpp:399` `GetFusedBandInfo` **实际返回** | **930-1200 nm** |

**现状** 枚举值(6)本身是一致的,只有注释里的范围不符,所以没有功能影响。根源是
`NIR` 这个名字同时指两个东西:渲染模式 `NIR_Fused`(930-1200 nm)和物理常量
`WAVELENGTH_{MIN,MAX}_NIR`(780-1400 nm)。两处注释拿物理定义去描述渲染模式。
`Types.hpp` 开头写着"CRITICAL: Must match shader defines in common.hlsli exactly"。

**期待目标** 描述渲染模式的地方统一到它真实的范围,或让两个概念不再共用一个名字。

### G2. `main.cpp` 存在未使用的 include

**问题** `src/app/main.cpp:36` 的 `SpectralCubeIO.hpp` 未被直接使用。

**现状** clangd 会报,但 `.clang-tidy` 里 `misc-include-cleaner` 是**故意关掉**的
(注释写明"与 clangd 自身的 unused-include 诊断重复"),所以批量 lint 查不出来,
只有编辑该文件时才看得到。

**期待目标** 确认无用后删除。

---

## H. 光谱产物质量

### H1. USGS 基的解释方差低于项目自定目标

**问题** 全量重烘(1374 材质)后:

| 波段 | Mean RMSE | 好材质占比 | 解释方差 |
|---|---|---|---|
| VIS | 0.0053 | 98.4% | **97.46%** |
| NIR | 0.0029 | 99.3% | **95.33%** |
| SWIR | 0.0068 | 98.7% | 98.46% |

RMSE(目标 < 0.03)和好材质占比(目标 > 95%)都达标,解释方差(目标 > 98%)
VIS 和 NIR 未达标。

**现状** 这不是回归。此前看到的 99.9% 来自 25 材质的试跑,小样本本来就容易拟合;
1374 个材质是第一次的全量真实数字。当前 `n_components` 为 VIS/NIR 各 16、SWIR 32。

**期待目标** 要么调整基函数数量使其达标,要么把目标值改成有依据的数字。

### H2. MWIR/LWIR 没有高覆盖度的数据源

**问题** 三个数据源的实测覆盖度(该波段中真正来自测量、而非边缘钳位外推的比例):

| 数据源 | VIS | NIR | SWIR | MWIR | LWIR |
|---|---|---|---|---|---|
| usgs | 100% | 100% | 100% | 不产出 | 不产出 |
| rii | 83.5% | 75.8% | 54.4% | **32.4%** | **27.9%** |
| ecostress | 52.0% | 56.4% | 65.6% | **55.5%** | **54.3%** |

**现状** 能渲 MWIR/LWIR 的两个源,最好的情况也只有约一半是真实测量。这两个波段的
RMSE 读数(0.0006-0.0025)之所以比有完整测量的波段还漂亮,正是因为外推段是直线、
拟合代价极低 —— 覆盖度字段已经把这件事记进产物,但数据本身的缺口还在。
三个源互有胜负,目前没有单一源在所有波段都最优。

**期待目标** 明确 MWIR/LWIR 渲染结果的可信程度,并决定是否需要补充数据源
(或按波段混合多个源)。
