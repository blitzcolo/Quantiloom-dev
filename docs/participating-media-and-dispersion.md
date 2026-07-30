# 参与介质、透明物体与光谱色散 —— 现状盘点与路线建议

调研日期 2026-07-30，针对 Quantiloom 当前 `master`（`a2574bc` 之后）。
目标场景：雾霾成像、水下成像、水陆跨界面成像、透明/半透明物体的光谱色散。

---

## 0. 摘要：六个结论

1. **雾霾已经做好了**，走的是 NN 大气（MODTRAN 代理），全波段有权重，直接可用。它和下面的问题**完全无关**。
2. **VIS_FUSED 模式做不出色散。** 它把折射率钉死在 550 nm。而 RGB 模式反倒有一个三波长近似。**光谱模式在光谱现象上不如 RGB 模式** —— 这是本次调研最要紧的发现。
3. **根因是架构性的**：VIS_FUSED 用**一根光线**携带 32 个波长。色散意味着每个波长走**不同的几何路径**，两者不能共存。这正是 hero wavelength 采样要解决的问题。
4. **Cauchy 色散系数的标定量级不对**，按文档语义（`dispersion` = 1/阿贝数）代入，比正确值小 **17–41 倍**（随材料而异）。
5. **实测 n,k 数据已经加载进来了，但只用于金属 F0**，没有用于折射方向。玻璃色散的正确解法已经建好了一半。
6. **参与介质整条路不可达**（无人写入 `volumeDensity`），而 `volumetric.hlsli` 里的 delta/ratio tracking 从未被调用。

---

## 1. 术语：什么是参与介质

「参与介质」（participating media）指光在两个表面之间的**路程上**会发生事情 —— 被吸收、被散射、被介质自身发射 —— 而不是像真空里那样原样直达。雾、霾、烟、云、水、牛奶、皮肤都属于这一类。与之相对的是「只在表面交互」的常规渲染。

描述一个均匀介质需要三样东西，都随波长变化：

| 量 | 含义 | 单位 |
|---|---|---|
| σ_a(λ) | 吸收系数 | m⁻¹ |
| σ_s(λ) | 散射系数 | m⁻¹ |
| p(θ, λ) | 相函数（散射方向分布） | sr⁻¹ |

派生量：消光 σ_t = σ_a + σ_s，单次散射反照率 ω = σ_s/σ_t。海洋光学里习惯写成 a(λ)、b(λ)、体散射函数 β(θ,λ)，是同一套东西的不同记号。

**关键点：这三个量都是波长的函数。** 一个光谱渲染器如果把它们存成 RGB 三元组，就等于在源头上放弃了光谱能力 —— 这正是当前 `MediumProperties` 的状况。

---

## 2. 代码现状盘点

### 2.1 三条互不相干的路径

| 路径 | 位置 | 状态 |
|---|---|---|
| **NN 大气** | `atmos/AtmosphereBaker`, 绑定 17+20 | **可用，质量高** |
| **参与介质单次散射** | `closesthit.rchit` 参与介质块 | **不可达**（见 2.3） |
| **电介质透射/折射** | `closesthit.rchit` transmission 块 | 可用，但色散有缺陷（见 3） |

### 2.2 NN 大气：唯一成熟的一块

MODTRAN 代理网络，烘焙逐波长的程辐射 lpath、透过率 tau、下行辐射 ldown。

- **波段覆盖**：`vis / nir / swir / mwir / lwir`，外加 single 模式的 `unify` —— 全波段。
- **参数是 MODTRAN 自己的**：`ihaze`（气溶胶模型）、`vis_km`（能见度）、`icld`、`rainrt_mm_h`、`rh`、`p_hPa`、`h2o_scale`、`t_ground_K`。
- **预设 8 个**：`clear / turbulent_clear / urban_haze / fog / light_rain / heavy_rain / snow / haze`。

雾霾成像直接写配置即可，不需要写代码：

```toml
[atmosphere]
preset = "urban_haze"     # ihaze=5, vis_km=8
# 或者绕过预设自己给
vis_km = 3.0
rh = 0.8
```

**边界**：这是一个*大气*模型，按观察者高度 `h1_km` 和斜程参数化，设计意图是「透过公里级空气看」。用它模拟一个充满烟的房间是用错工具；水更是完全不适用（见第 4 节）。

### 2.3 参与介质：写了，但接不上

`Material::volumeDensity` 和 `scatteringCoeff` 在全仓库只有 **6 处引用**：两处声明（默认 `0.0f`）、两处拷进 `MaterialDataCPU`、两处 GPU 字段声明。**没有任何写入者** —— glTF 加载器没有、USD 加载器没有、TOML 没有、Studio 没有。

所以 `if (material.volumeDensity > 0.0 && material.scatteringCoeff > 0.0)` 永远为假，整块是死代码。

更值得注意的是 `src/shaders/volumetric.hlsli`（400 行）：

| 函数 | 用途 | 是否被调用 |
|---|---|---|
| `DeltaTrackingHomogeneous` | 无偏体积输运 | **否** |
| `RatioTrackingTransmittance` | 无偏透过率估计 | **否** |
| `SampleHenyeyGreenstein` | 相函数重要性采样 | **否** |
| `CreateMediumFromAttenuation` | 从 glTF 衰减参数建介质 | **否** |
| `CreateMediumFromMaterial` | — | 是（死块内） |
| `HenyeyGreenstein` | — | 是（死块内） |
| `BeerLambertTransmittance` | — | 是（死块内） |

也就是说，**做真正体积输运需要的机器已经写好了，但一个都没接上**，而接上的那三个喂给了一段跑不到的单次散射代码。

已修复（2026-07-30）：该块原本在所有模式下读 `sunRadiance_rgb`，在 SWIR 下内散射项偏大约 **86000 倍**且非灰。修复后按模式分支采样太阳光谱 LUT。但这不改变「整块不可达」的事实。

### 2.4 电介质透射

`transmission` 块做了：Snell 折射、`FresnelDielectric`、全内反射检测、俄罗斯轮盘赌在反射/折射间二选一、Beer-Lambert 体吸收（「有色玻璃」）、递归深度上限 `MAX_TRANSMISSION_DEPTH = 8`。

**缺的是介质跟踪**：渲染器没有「这条光线目前在水里/玻璃里」的状态。Beer-Lambert 按段施加，是薄物体近似，不是「相机泡在介质里」。也没有嵌套电介质的优先级概念。

---

## 3. 色散：本次调研的核心发现

### 3.1 现状：光谱模式做不出色散

```
RGB        → 追 3 根光线（650/550/450 nm），各取一个通道  →  有近似彩虹
VIS_FUSED  → effectiveIOR = CauchyIOR(ior, dispersion, 550.0)  →  没有色散
SINGLE     → CauchyIOR(ior, dispersion, camera.wavelength_nm)  →  有（每帧一个波长）
```

`hasDispersion` 的判据本身就把 VIS_FUSED 排除在多波长追踪之外，只留下 550 nm 单值。

**结果是反直觉的**：你举的三棱锥例子，在 RGB 模式下能看到（三条谱线的粗糙近似），在号称「真光谱积分」的 VIS_FUSED 模式下**看不到** —— 白光原样穿过，不分光。

### 3.2 根因：一根光线携带 32 个波长

`VIS_FUSED` 在**一次 closest-hit 调用**里循环 `NUM_WAVELENGTH_SAMPLES = 32` 个波长，累积 XYZ 再转 sRGB。这个结构对表面着色是对的、也是高效的 —— 32 个波长共享同一条几何路径。

**但色散的本质就是「几何路径依赖波长」。** 折射方向由 Snell 决定，n(λ) 不同 → 方向不同 → 之后的路径完全不同。一根光线无法同时走 32 条路。

这不是可以打补丁的问题，是采样结构的问题。

### 3.3 Cauchy 系数的标定量级不对

`common.hlsli` 的实现：

```hlsl
float CauchyIOR(float ior_d, float dispersion, float wavelength_nm) {
    return ior_d + dispersion * 0.01 / lambda_um2;      // B = dispersion × 0.01
}
```

而 `Material::dispersion` 的文档是「Abbe number reciprocal」，即 1/V_d。

两项 Cauchy 式 n(λ) = n_d + B/λ² 与阿贝数的关系为

```
V_d = (n_d − 1) / (n_F − n_C),  n_F − n_C = B·(1/λ_F² − 1/λ_C²)
λ_F = 0.4861 µm, λ_C = 0.6563 µm  →  1/λ_F² − 1/λ_C² = 1.91039 µm⁻²

⇒  B = (n_d − 1) / (V_d × 1.91039) = 0.52346 × (n_d − 1) / V_d
```

代入 BK7（n_d = 1.5168, V_d = 64.17）得 B = 0.004216 µm²，与公开值 0.00420 µm² 吻合。

而代码给出 B = 0.01/V_d = 1.558×10⁻⁴ µm²。**比正确值小 27 倍。**

偏差因子 = 52.346 × (n_d − 1)，随材料变化：

| 材料 | n_d | V_d | 正确 B (µm²) | 代码 B (µm²) | 偏小 |
|---|---|---|---|---|---|
| 水 | 1.333 | 55.7 | 0.003129 | 0.000180 | 17.4× |
| BK7 冕牌玻璃 | 1.5168 | 64.17 | 0.004216 | 0.000156 | 27.1× |
| SF11 重火石 | 1.7847 | 25.76 | 0.015945 | 0.000388 | 41.1× |

也就是说，即便在 RGB 模式下看到了彩虹，那个彩虹的**强度也是错的**，而且错的程度依材料而异。

### 3.4 更好的路已经铺了一半

仓库已经能加载 **refractiveindex.info 的 YAML**（`SpectralIO::LoadRefractiveIndexYAML` → `ComplexRefractiveIndexGPU` → 绑定 14），提供实测的 n(λ)、k(λ)。

**但它只用在 `ComputeF0` / `ComputeF0_Scalar` 里，也就是金属的菲涅尔反射率。** 折射方向用的是 `material.ior` + Cauchy 拟合。

对玻璃色散来说这是本末倒置：实测 n(λ) 比任何两项拟合都准，而且数据通路已经通了。**把折射的 n 改成从 CRI 缓冲采样，是这条线上性价比最高的一步**，同时可以把 Cauchy 降级为「没有实测数据时的回退」。

对红外尤其重要：Cauchy 式在红外区不准，也无法表达反常色散区；Sellmeier 式在可见光区误差可低到 0.05% 并能覆盖 UV–VIS–IR。既然这是一个覆盖 380–15000 nm 的渲染器，实测表 > Sellmeier > Cauchy。

---

## 4. 四个应用场景各自需要什么

### 4.1 雾霾成像 —— 现在就能做

见 2.2。不需要写代码。建议先做一个能见度对比场景把这块坐实。

### 4.2 水下成像 —— 需要光谱 IOP，且有物理上限

水的吸收随波长变化极其剧烈。数量级（纯水吸收系数 a）：

| 波长 | a 数量级 (m⁻¹) | 1/e 穿透深度 |
|---|---|---|
| 418 nm（吸收极小） | 0.0044 | ~230 m |
| 550 nm | ~0.06 | ~15 m |
| 650 nm | ~0.35 | ~3 m |
| 800 nm | ~2 | ~0.5 m |
| 1000 nm | ~40 | ~2.5 cm |
| 1500 nm（SWIR） | ~10³ | 亚毫米 |
| 3–10 µm（MWIR/LWIR） | ~10⁵–10⁶ | **微米级** |

> 418 nm 处的极小值 0.0044 ± 0.0006 m⁻¹ 来自 Pope & Fry 1997 的积分腔测量。其余为数量级，**接入前应从数据文件取准确表**：380–700 nm 用 Pope & Fry (1997)，红外用 Segelstein (1981) / Hale & Querry (1973) / Wieliczka (1989)，omlc.org 的 water compendium 汇总了这些。

**对规划的直接含义**：这是一个光谱/红外渲染器，而**红外根本穿不进水**。SWIR 以上的「水下成像」物理上只到毫米级。这不是渲染器的限制。

能做且有价值的是：
- **可见光/NIR 水下成像** —— 真实，且是水色遥感的核心
- **从空中看水体** —— 水面热红外特征、离水辐亮度、太阳耀斑（sun glint），在 IR 里都成立

散射来自颗粒物/浮游生物，相函数极度前向（Petzold 测量，g ≈ 0.92），标准分类是 Jerlov 水型：开阔水域 I–III、近岸 1–9C，按单次散射反照率排序。Solonenko & Mobley (2015) 给出了各 Jerlov 水型在 300–700 nm 的 a(λ)、b(λ) 实测/反演表 —— **这就是「介质带一条曲线」所需的数据，格式和现有反射率 CSV 同构**。

### 4.3 水陆跨界面 —— 最难，需要介质跟踪

需要三样：

1. **折射** —— 已有一半（Snell + Fresnel + TIR 都在）
2. **介质跟踪** —— 没有。需要知道光线当前在哪个介质里，穿过界面时切换
3. **嵌套/重叠处理** —— 没有。水里放一个玻璃球就是重叠电介质

工业界标准解法是 **Schmidt & Budge (2002) 的嵌套电介质**：把可重叠的电介质都建成闭合实体并赋优先级，优先级不同的边界视为「假界面」（false interface），不做着色，等价于布尔运算。Arnold 和 RenderMan 都用这套。

**免费的正确性判据：Snell 窗。** 从水下往上看，整个天空被压缩进半角 arcsin(1/1.333) = 48.6° 的锥（全角 97.2°），锥外是水底的全内反射。折射算对了它会自动出现，算错了立刻能看出来 —— 这是个比「看起来对」强得多的判据，和 furnace 腔在热辐射上的地位类似。

### 4.4 透明物体 + 色散 —— 见第 3 节

三棱锥分光是光谱渲染器的招牌能力。当前做不到（VIS_FUSED 无色散）。需要采样结构的改动，见下节。

---

## 5. 核心架构问题：光谱几何 vs 单光线多波长

### 5.1 矛盾

| 需求 | 结构 |
|---|---|
| 表面着色的效率 | 一根光线携带 N 个波长，共享几何路径 |
| 色散、水下逐波长衰减 | 每个波长自己的路径 |

这两者不可兼得。当前实现选了前者，于是丢了后者。

### 5.2 业界解法：Hero Wavelength Spectral Sampling

Wilkie et al. (2014, EGSR / CGF 33(4)) 的做法：

- 每条路径**随机采样一个 hero 波长**，**所有方向采样只依据它**
- 另外携带少数几个波长，**等距分布**在 hero 附近，使整条路径的波长集合均匀覆盖可见区
- 用 MIS 组合各波长作为 hero 的可能性，抵消由此产生的方差

优点：实现比原始的 spectral MIS 简单，性能好，且在**色散、次表面散射、体积效应**上表现优异 —— 恰好是这三个场景。这是生产渲染器（如 Manuka）采用的方案。

**关键性质**：当路径上没有波长相关的几何事件时，多波长共享一条路径（保住效率）；一旦遇到色散界面，只有 hero 波长决定方向，其余波长通过 MIS 权重正确降权。**它不要求你为每个波长各追一条光线。**

### 5.3 对 Quantiloom 的含义

现有的 `VIS_FUSED` 32 波长循环是一个「固定波长栅格 + 共享路径」结构。改造方向有二：

**A. 局部改造 —— 试过了，没成功（2026-07-30）**

按「32 波长分 8 组，每组用组中心波长的 n 追一条折射光线，各组只积分自己那 4 个波长，最后相加」实现了一遍，**因为不满足自身的正确性判据而回退**。

判据是这样设计的：VIS_FUSED 的 XYZ 积分对波长是**线性求和**，所以给一个**常数 n(λ)** 的材质（用一张平的实测折射率表），8 组应当各走同一条路径、各返回自己那一片，相加**必须逐位等于**不分组的结果。实测：

| 量 | 结果 |
|---|---|
| 未受影响的像素（92.72%） | 最大差 **恰好 0** —— 分组没有泄漏到非透射路径 |
| 透射区域（7.28%） | 分组结果**亮 16.6%** |
| 残差通道分布 | R +7.4e-4，G +1.6e-4，B +2.2e-4（强 R 偏） |
| 平坦 n 与真实 BK7 的残差 | 四位有效数字相同 —— 与色散无关，是分组本身的产物 |

排查过程中确认无误的部分：波长循环体是纯线性累加（无端点特例、无按采样数归一化）；分组区间 [0,4)…[28,32) 完整覆盖；`NUM_SPECTRAL_GROUPS = 1` 时与不分组相差 9.3e-09（浮点噪声），说明机制在 N=1 时正确。

排查过程中修掉的两个真问题（但都不是主因）：

- **`TraceRay` 的 payload 是 inout**，被调用方的随机数状态会写回。原实现只在循环外设一次 `rngState`，于是第 g+1 组从第 g 组结束的状态开始，**走的是另一条路径**。求和仍无偏，但那是「8 条路径各一片」而不是「一条路径分 8 片」，更吵且不满足判据。改为每组重置后最大残差降 4 倍。
- **窄带出 sRGB 色域**：4 个波长宽的切片转线性 sRGB 时通道可以为负，而末尾 `clamp(…, 0.0, 1000.0)` 会把负值削掉，8 个削过的切片之和大于削过的整体。已改为仅对整带光线施加下限。（在这个测试场景里实测没有分组真的取到负值，属于预防性修正。）

**主因未找到。** 16.6% 的 R 偏亮在两个修正之后依然存在，量级不随分组数线性变化。下一步应当做的实验：临时让 `VisSampleRange` 恒返回全带、并只统计透射区域，检查 8 条光线之和是否**精确等于 8×** —— 若是，则问题在波长限制而非光线机制，与「循环体线性」的结论矛盾，需要重读循环体；若否，则问题在循环机制本身。（第一次做这个实验时统计了整幅图，被 92.7% 未变像素稀释成 1.44，是无效测量。）

**B. 结构改造（成本高，能力完整）**
引入 hero wavelength 采样，把「一次调用积分整个波段」改成「每条路径携带 4 个波长 + MIS」。这会**改变所有光谱模式的收敛行为和噪声特性**，是一次大手术，且现有的 furnace 判据（等温腔）能守住能量正确性但守不住方差。

**建议先 A 后 B**：A 能立刻让三棱锥出彩虹，且给 B 留了对照基准 —— B 做完之后两者应当收敛到同一个答案，这是 B 唯一可靠的验证方式。

---

## 6. 最佳实践要点（文献）

**色散 / 光谱**
- Hero wavelength 采样是当前的实践标准，兼顾色散与效率（Wilkie 2014）
- 折射率优先用实测 n(λ)；其次 Sellmeier；Cauchy 只适合可见区且无法表达反常色散
- 光谱 Monte Carlo 收敛慢，光谱域的重要性采样（按入射光谱能量分布）能显著降噪

**参与介质**
- 异质介质用 delta tracking / ratio tracking（无偏），不要用固定步长 ray marching
- 色差介质（各波长 σ_t 差异大，水正是如此）用 **spectral tracking**：修改自由程分布以压低路径吞吐量的波动，从而降方差（Kutz et al. 2017）
- null-scattering 的路径积分形式给出了各技术的 pdf，使得可以用 MIS 组合互补的无偏技术，在空间和光谱都变化的介质上优于既有方法
- 采样策略选择：简单直接光照用 light guiding；**默认用 MIS**；主光源被遮挡或可见性很紧时换 BDPT

**水体**
- 用 IOP（a、b、β）而非 ad-hoc 颜色衰减；Jerlov 水型是标准分类
- 实时近似可走 Kd（漫射下行衰减系数）的解析近似（Monzon et al. 2024），但那是实时方案；离线渲染器应当走真正的输运
- 相函数用 Petzold 实测或 Fournier-Forand，而非 g≈0 的各向同性

**嵌套电介质**
- 优先级方案（Schmidt & Budge 2002）优于简单的界面计数；重叠几何是推荐的建模方式

---

## 7. 建议路线

按「每一步都有独立判据」排序：

| # | 工作 | 判据 | 状态 |
|---|---|---|---|
| 1 | 雾霾场景验证既有 NN 大气 | 能见度 vs 对比度单调 | 待做，一个 TOML |
| 2 | 折射改用实测 n(λ)（CRI 缓冲），Cauchy 降为回退 | 与解析路径互校 | **已完成** |
| 3 | 采用 KHR_materials_dispersion 的官方 n(λ) 公式 | BK7 往返阿贝数 = 64.17 | **已完成** |
| 4 | VIS_FUSED 分批色散（方案 A） | 常数 n 时分组求和 = 不分组 | **已回退，见上** |
| 5 | 介质区域概念 + 光谱 σ_a/σ_s 曲线 | 纯水 a(λ) 复现穿透深度表 | 大 |
| 6 | 介质跟踪 + 嵌套电介质优先级 | **Snell 窗 97.2°** | 大 |
| 7 | 接上 delta tracking，弃用单次散射块 | 与解析解对比（均匀介质） | 中 |
| 8 | Hero wavelength 采样（方案 B） | 与 #4 收敛到同一答案 | 很大 |

**第 5 步是关键的设计决策**：你要的是「水面以下是介质 A，以上是介质 B」，这是**空间划分**，不是表面属性。现在的 `Material::volumeDensity` 是挂在材质上的标量，方向就不对。这个决定会框住 6、7 两步。

**优先级建议**：2 和 3 立刻做，成本极低且直接提升现有玻璃渲染的正确性。4 让三棱锥能用。5 之前先把设计想清楚，别急着写。

---

## 8. 参考文献

**光谱渲染与色散**
- Wilkie, A., Nawaz, S., Droske, M., Weidlich, A., Hanika, J. (2014). *Hero Wavelength Spectral Sampling*. Computer Graphics Forum 33(4), 123–131. [10.1111/cgf.12419](https://doi.org/10.1111/cgf.12419)
- Fascione, L. et al. (2018). *Manuka: A Batch-Shading Architecture for Spectral Path Tracing in Movie Production*. ACM TOG 37(3). [10.1145/3182161](https://doi.org/10.1145/3182161)
- Cauchy / Sellmeier 对比：[RP Photonics — Sellmeier Formula](https://www.rp-photonics.com/sellmeier_formula.html)；[Horiba — Cauchy and related Empirical Dispersion Formulae](https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Cauchy_and_related_empirical_dispersion_Formulae_for_Transparent_Materials.pdf)

**参与介质**
- Kutz, P., Habel, R., Li, Y.K., Novák, J. (2017). *Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes*. ACM TOG 36(4).
- Miller, B., Georgiev, I., Jarosz, W. (2019). *A Null-Scattering Path Integral Formulation of Light Transport*. ACM TOG 38(4).
- [Rendering Participating Media Using Path Graphs (2024)](https://arxiv.org/pdf/2404.11894)
- [Benchmarking Monte Carlo path tracing methods in fog and complex visibility](https://www.sciencedirect.com/science/article/abs/pii/S0022407326001731)

**嵌套电介质**
- Schmidt, C.M., Budge, B. (2002). *Simple Nested Dielectrics in Ray Traced Images*. Journal of Graphics Tools 7(2). [10.1080/10867651.2002.10487555](https://doi.org/10.1080/10867651.2002.10487555)
- 实现参考：[Pixar RenderMan — Nested Dielectrics](https://rmanwiki.pixar.com/display/REN24/Nested+Dielectrics)；[Yining Karl Li — Nested Dielectrics](https://blog.yiningkarlli.com/2019/05/nested-dielectrics.html)

**水体光学**
- Pope, R.M., Fry, E.S. (1997). *Absorption spectrum (380–700 nm) of pure water. II. Integrating cavity measurements*. Applied Optics 36(33), 8710–8723. [PubMed 18264420](https://pubmed.ncbi.nlm.nih.gov/18264420/)
- Solonenko, M.G., Mobley, C.D. (2015). *Inherent optical properties of Jerlov water types*. Applied Optics 54(17). [PubMed 26192839](https://pubmed.ncbi.nlm.nih.gov/26192839/)
- Monzon, N., Gutierrez, D., Akkaynak, D., Muñoz, A. (2024). *Real-Time Underwater Spectral Rendering*. Computer Graphics Forum. [10.1111/cgf.15009](https://doi.org/10.1111/cgf.15009) · [项目页](https://graphics.unizar.es/projects/EG24Underwater/) · [开放 PDF](https://zaguan.unizar.es/record/135350/files/texto_completo.pdf)
- 数据汇总：[OMLC — Optical Absorption of Water Compendium](https://omlc.org/spectra/water/abs/index.html)（含 Pope & Fry 1997、Segelstein 1981、Wieliczka 1989 的数据文件）
- [Ocean Optics Web Book — Absorption by Oceanic Constituents](https://www.oceanopticsbook.info/view/absorption/absorption-by-oceanic-constituents)
