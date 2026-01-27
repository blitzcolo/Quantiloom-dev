  Quantiloom 红外成像流程与数据流技术文档

  1. 概述

  本文档描述 Quantiloom 渲染引擎中红外（IR）成像的完整物理模型、数据流和实现细节。红外波段按物理特性分为两大类：
  ┌──────────┬──────┬───────────────┬──────────────────────────────┬────────────────────────────────────┐
  │   类别   │ 波段 │   波长范围    │          主导辐射源          │              成像机制              │
  ├──────────┼──────┼───────────────┼──────────────────────────────┼────────────────────────────────────┤
  │ 反射型   │ NIR  │ 780–1400 nm   │ 太阳反射                     │ 类似可见光，BRDF 主导              │
  ├──────────┼──────┼───────────────┼──────────────────────────────┼────────────────────────────────────┤
  │ 反射型   │ SWIR │ 1000–2500 nm  │ 太阳反射                     │ 类似可见光，热辐射可忽略（T<500K） │
  ├──────────┼──────┼───────────────┼──────────────────────────────┼────────────────────────────────────┤
  │ 热辐射型 │ MWIR │ 3000–5000 nm  │ 热自发辐射 + 太阳反射(5-20%) │ 混合模型                           │
  ├──────────┼──────┼───────────────┼──────────────────────────────┼────────────────────────────────────┤
  │ 热辐射型 │ LWIR │ 8000–12000 nm │ 热自发辐射                   │ 纯热辐射，太阳贡献<0.1%            │
  └──────────┴──────┴───────────────┴──────────────────────────────┴────────────────────────────────────┘
  2. 波段定义（ISO 20473）

  源文件: src/libQuantiloom/core/Types.hpp:482-497

  // NIR: Near-Infrared (primarily reflected solar radiation)
  inline constexpr Wavelength WAVELENGTH_MIN_NIR = 780.0f;   // nm
  inline constexpr Wavelength WAVELENGTH_MAX_NIR = 1400.0f;  // nm

  // SWIR: Short-Wave Infrared
  inline constexpr Wavelength WAVELENGTH_MIN_SWIR = 1000.0f;  // nm
  inline constexpr Wavelength WAVELENGTH_MAX_SWIR = 2500.0f;  // nm

  // MWIR: Mid-Wave Infrared
  inline constexpr Wavelength WAVELENGTH_MIN_MWIR = 3000.0f;  // nm
  inline constexpr Wavelength WAVELENGTH_MAX_MWIR = 5000.0f;  // nm

  // LWIR: Long-Wave Infrared
  inline constexpr Wavelength WAVELENGTH_MIN_LWIR = 8000.0f;   // nm
  inline constexpr Wavelength WAVELENGTH_MAX_LWIR = 12000.0f;  // nm

  ---
  3. 物理基础：热辐射定律

  3.1 普朗克黑体辐射定律（Planck's Law）

  源文件: src/shaders/blackbody.hlsli:84-116

  黑体在温度 $T$ 和波长 $\lambda$ 处的光谱辐亮度：

  $$
  L_\lambda(T, \lambda) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{\exp\left(\frac{hc}{\lambda k_B T}\right) - 1}
  $$

  单位: $\text{W} \cdot \text{sr}^{-1} \cdot \text{m}^{-2} \cdot \text{nm}^{-1}$

  实现优化（预计算常数，$\lambda$ 单位为 nm）：

  $$
  L_\lambda = \frac{C_1}{\lambda_{\text{nm}}^5} \cdot \frac{1}{\exp\left(\frac{C_2}{\lambda_m \cdot T}\right) - 1}
  $$

  其中：
  - $C_1 = 2hc^2 \times 10^{36} = 1.191042972 \times 10^{20}$ (W·nm⁴·sr⁻¹·m⁻²)
  - $C_2 = hc/k_B = 1.43877736 \times 10^{-2}$ (m·K)
  - $\lambda_m = \lambda_{\text{nm}} \times 10^{-9}$

  HLSL 实现:
  float IRPlanckRadiance(float temperature_K, float wavelength_nm) {
      float lambda_nm_5 = pow(wavelength_nm, 5);  // λ⁵
      float numerator = C1_NM / lambda_nm_5;

      float lambda_m = wavelength_nm * 1e-9;
      float exponent = C2 / (lambda_m * temperature_K);
      float denominator = exp(exponent) - 1.0;

      return numerator / max(denominator, 1e-30);  // W·sr⁻¹·m⁻²·nm⁻¹
  }

  3.2 维恩位移定律（Wien's Displacement Law）

  源文件: src/shaders/blackbody.hlsli:137-141

  黑体辐射峰值波长：

  $$
  \lambda_{\text{peak}} = \frac{b}{T} = \frac{2.897771955 \times 10^{-3}}{T} \text{ (m)}
  $$
  ┌──────────┬──────────┬──────────┬──────────┐
  │   物体   │ 温度 (K) │ 峰值波长 │ 所属波段 │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 人体     │ 310      │ 9.3 μm   │ LWIR     │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 室温物体 │ 300      │ 9.7 μm   │ LWIR     │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 发动机   │ 600      │ 4.8 μm   │ MWIR     │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 太阳     │ 5800     │ 500 nm   │ 可见光   │
  └──────────┴──────────┴──────────┴──────────┘
  3.3 斯特藩-玻尔兹曼定律（Stefan-Boltzmann Law）

  源文件: src/shaders/blackbody.hlsli:175-187

  黑体总辐射出射度（全波段积分）：

  $$
  M = \sigma T^4
  $$

  其中 $\sigma = 5.670374419 \times 10^{-8}$ W·m⁻²·K⁻⁴

  灰体修正：
  $$
  M_{\text{gray}} = \varepsilon \cdot \sigma T^4
  $$

  3.4 基尔霍夫定律（Kirchhoff's Law）

  热平衡下的能量守恒：

  $$
  \varepsilon(\lambda) + \rho(\lambda) + \tau(\lambda) = 1
  $$

  其中：
  - $\varepsilon(\lambda)$：发射率（emissivity）
  - $\rho(\lambda)$：反射率（reflectance）
  - $\tau(\lambda)$：透过率（transmittance）

  ---
  4. NIR/SWIR 成像模型（反射型）

  4.1 物理模型

  NIR 和 SWIR 波段以太阳反射为主导，成像机制与可见光相似：

  $$
  L_{\text{total}}(\lambda) = \rho(\lambda) \cdot \left[ L_{\text{sun}}(\lambda) \cdot \cos\theta +
  L_{\text{sky}}(\lambda) \right] \cdot \tau_{\text{atm}}(\lambda)
  $$

  其中：
  - $\rho(\lambda)$：材料双向反射分布函数（BRDF）
  - $L_{\text{sun}}(\lambda)$：太阳光谱辐亮度
  - $\theta$：入射角（N·L）
  - $L_{\text{sky}}(\lambda)$：天空漫射辐射
  - $\tau_{\text{atm}}(\lambda)$：大气透过率

  4.2 热辐射贡献判断

  SWIR 波段热辐射阈值：

  根据 Wien 定律，500K 物体的峰值波长 ≈ 5.8 μm。对于 SWIR（1-2.5 μm），只有 T > 500K 的高温物体才有显著热辐射贡献。

  // SWIR mode: thermal emission negligible for T < 500K
  if (material.irTemperature_K > 500.0) {
      float L_blackbody = IRPlanckRadiance(material.irTemperature_K, lambda);
      L_emission = emissivity * L_blackbody;
  }

  4.3 数据流

  ┌─────────────────┐     ┌─────────────────┐
  │ ASTM G-173      │     │ 材料 BRDF       │
  │ 太阳光谱 LUT    │     │ (albedo,metal,  │
  │ (W·m⁻²·nm⁻¹)   │     │  roughness)     │
  └────────┬────────┘     └────────┬────────┘
           │                       │
           ▼                       ▼
      ┌────────────────────────────────────┐
      │     NIR/SWIR Shader Integration    │
      │  L = ρ(λ) × [L_sun × NdotL + L_sky]│
      └────────────────┬───────────────────┘
                       │
                       ▼
                ┌──────────────┐
                │ 大气透过率   │
                │ τ(λ,h,θ)     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ 传感器响应   │
                │ QE, 噪声模型 │
                └──────────────┘

  ---
  5. MWIR/LWIR 成像模型（热辐射型）

  5.1 完整辐射传输方程

  源文件: src/shaders/closesthit.rchit:1452-1610

  MWIR/LWIR 波段的总辐亮度包含四个分量：

  $$
  \boxed{
  L_{\text{total}}(\lambda) = \underbrace{\varepsilon(\lambda) \cdot L_{bb}(T_s, \lambda)}{\text{自发辐射}} +
  \underbrace{\rho(\lambda) \cdot L{\text{atm}\downarrow}(\lambda)}{\text{反射大气辐射}} + \underbrace{\rho(\lambda)
  \cdot L{\text{sun}}(\lambda) \cdot \cos\theta}{\text{反射太阳辐射}} + \underbrace{\tau(\lambda) \cdot
  L{\text{bg}}(\lambda)}_{\text{透射背景辐射}}
  }
  $$

  5.2 各分量详解

  5.2.1 自发辐射（Thermal Self-Emission）

  物体因自身温度 $T_s$ 而发射的热辐射：

  $$
  L_{\text{emission}}(\lambda) = \varepsilon(\lambda) \cdot L_{bb}(T_s, \lambda)
  $$

  实现:
  float L_emission = 0.0;
  if (material.irTemperature_K > 0.0) {
      float L_blackbody = IRPlanckRadiance(material.irTemperature_K, lambda);
      L_emission = emissivity * L_blackbody;
  }

  5.2.2 反射大气下行辐射（Reflected Atmospheric Downwelling）

  天空作为热源向地面发射的辐射被表面反射：

  $$
  L_{\text{refl,atm}}(\lambda) = \rho(\lambda) \cdot L_{bb}(T_{\text{atm}}, \lambda)
  $$

  其中 $T_{\text{atm}}$ 为有效大气温度（典型值 240-290 K）。

  实现:
  float T_atmosphere = lut.atmosphereTemperature_K;  // 默认 260K
  float L_downwelling = IRPlanckRadiance(T_atmosphere, lambda);
  float L_reflected_atm = reflectance * L_downwelling;

  5.2.3 反射太阳辐射（Reflected Solar Radiation）

  仅 MWIR 显著，LWIR 可忽略

  在 4 μm 处，太阳辐照度 ≈ 5 W·m⁻²·μm⁻¹，对 300K 表面可贡献 5-20% 的总辐亮度。

  $$
  L_{\text{refl,sun}}(\lambda) = \rho(\lambda) \cdot \frac{E_{\text{sun}}(\lambda)}{\pi} \cdot \cos\theta \cdot
  \tau_{\text{atm,sun}}(\lambda)
  $$

  太阳辐照度计算（无 LUT 时的黑体近似）：

  $$
  E_{\text{sun}}(\lambda) = L_{bb}(5778\text{K}, \lambda) \cdot \Omega_{\text{sun}}
  $$

  其中太阳立体角 $\Omega_{\text{sun}} \approx 6.8 \times 10^{-5}$ sr

  实现:
  if (includeSolarReflection && NdotL > 0.0) {
      float sun_irr_lambda;
      if (hasSpectralSolarLUT) {
          sun_irr_lambda = SampleSunIrradiance(solarSpectralLUT[0], lambda);
      } else {
          // Planck fallback for sun at 5778K
          float L_sun_surface = IRPlanckRadiance(5778.0, lambda);
          sun_irr_lambda = L_sun_surface * SUN_SOLID_ANGLE_SR;  // W·m⁻²·nm⁻¹
      }

      float sun_radiance_lambda = sun_irr_lambda / PI;  // Lambertian BRDF
      L_reflected_sun = reflectance * sun_radiance_lambda * NdotL;
      L_reflected_sun *= MWIR_ATM_TRANSMITTANCE_APPROX;  // ~0.8
  }

  5.2.4 透射背景辐射（Transmitted Background）

  对于红外透明材料（ZnSe、Ge 窗口等）：

  $$
  L_{\text{trans}}(\lambda) = \tau(\lambda) \cdot L_{\text{bg}}(\lambda)
  $$

  实现:
  if (transmittance > 0.001) {
      float L_background = IRPlanckRadiance(T_atmosphere, lambda);
      L_transmitted = transmittance * L_background;
  }

  5.3 MWIR vs LWIR 差异
  ┌──────────┬───────────────────┬──────────────────┐
  │   特性   │   MWIR (3-5 μm)   │  LWIR (8-12 μm)  │
  ├──────────┼───────────────────┼──────────────────┤
  │ 太阳反射 │ 5-20%（必须计算） │ <0.1%（忽略）    │
  ├──────────┼───────────────────┼──────────────────┤
  │ 热发射   │ 80-95%            │ ~100%            │
  ├──────────┼───────────────────┼──────────────────┤
  │ 大气窗口 │ H₂O/CO₂ 吸收较强  │ 8-13 μm 高透过率 │
  ├──────────┼───────────────────┼──────────────────┤
  │ 典型应用 │ 高温目标检测      │ 常温热成像       │
  └──────────┴───────────────────┴──────────────────┘
  代码中的判断:
  if (camera.spectral_mode == SPECTRAL_MODE_MWIR_FUSED) {
      includeSolarReflection = (NdotL > 0.0);  // MWIR: include solar
  } else {  // LWIR
      includeSolarReflection = false;  // LWIR: skip for performance
  }

  5.4 发射率与反射率的推导

  源文件: src/shaders/common.hlsli

  当材料未提供显式 IR 发射率时，从金属因子推导：

  $$
  \varepsilon = 0.95 - 0.90 \cdot f_{\text{metallic}} + 0.15 \cdot f_{\text{metallic}} \cdot f_{\text{roughness}}
  $$

  物理依据：
  - 抛光金属：$\varepsilon \approx 0.02-0.05$（高反射）
  - 氧化/粗糙金属：$\varepsilon$ 增加（微腔效应）
  - 非金属介质：$\varepsilon \approx 0.90-0.98$

  反射率由基尔霍夫定律：
  $$
  \rho = 1 - \varepsilon - \tau
  $$

  ---
  6. 光谱积分与输出

  6.1 波段积分（Riemann Sum）

  源文件: src/shaders/closesthit.rchit:1518-1594

  $$
  L_{\text{band}} = \int_{\lambda_{\min}}^{\lambda_{\max}} L_{\text{total}}(\lambda)  d\lambda \approx
  \sum_{i=0}^{N-1} L_{\text{total}}(\lambda_i) \cdot \Delta\lambda
  $$

  实现参数:
  - MWIR/LWIR: $N = 16$ 采样点
  - VIS_Fused: $N = 32$ 采样点

  const uint NUM_IR_SAMPLES = 16;
  const float lambda_step = (lambda_max - lambda_min) / float(NUM_IR_SAMPLES - 1);

  float radiance_accum = 0.0;
  for (uint i = 0; i < NUM_IR_SAMPLES; ++i) {
      float lambda = lambda_min + float(i) * lambda_step;
      // ... compute L_lambda ...
      radiance_accum += L_lambda * lambda_step;
  }

  6.2 平均辐亮度归一化

  为便于不同波段比较，输出平均光谱辐亮度：

  $$
  \bar{L} = \frac{L_{\text{band}}}{\lambda_{\max} - \lambda_{\min}} \quad \text{(W·sr}^{-1}\text{·m}^{-2}\text{)}
  $$

  float band_width = lambda_max - lambda_min;
  float radiance_avg = radiance_accum / band_width;
  output_radiance = float3(radiance_avg, radiance_avg, radiance_avg);  // Grayscale

  ---
  7. 外部数据源

  7.1 太阳光谱 LUT
  ┌───────────────┬─────────────┬───────────────────────────┐
  │    数据源     │  波长范围   │           用途            │
  ├───────────────┼─────────────┼───────────────────────────┤
  │ ASTM G-173-03 │ 280-4000 nm │ AM1.5 太阳直射/漫射辐照度 │
  ├───────────────┼─────────────┼───────────────────────────┤
  │ Planck 5778K  │ 任意        │ 无 LUT 时的黑体近似       │
  └───────────────┴─────────────┴───────────────────────────┘
  数据结构 (SolarSpectralLUT):
  struct SolarSpectralLUT {
      SpectralCurveGPU sunIrradiance;   // 直射 (W·m⁻²·nm⁻¹)
      SpectralCurveGPU skyIrradiance;   // 天空漫射 (W·m⁻²·nm⁻¹)
  };

  7.2 大气透过率 LUT
  ┌────────────┬───────────┬─────────────────────────┐
  │   数据源   │   维度    │          用途           │
  ├────────────┼───────────┼─────────────────────────┤
  │ MODTRAN    │ (λ, h, θ) │ 工业级大气辐射传输      │
  ├────────────┼───────────┼─────────────────────────┤
  │ libRadtran │ (λ, h, θ) │ 开源 UV/VIS/IR 大气模型 │
  └────────────┴───────────┴─────────────────────────┘
  数据结构 (AtmosphereTransmittanceLUT):
  struct AtmosphereTransmittanceLUT {
      UniformAxis    wavelength;   // 波长 (nm)
      UniformAxis    altitude;     // 高度 (m)
      NonUniformAxis zenith;       // 天顶角 (°)

      Vector<f32> transmittance;   // τ(λ,h,θ) ∈ [0, 1]
      Vector<f32> path_radiance;   // L_path(λ,h,θ)
  };

  7.3 材料光谱数据
  ┌────────────────────────┬─────────────────────┬──────┐
  │         数据源         │        内容         │ 格式 │
  ├────────────────────────┼─────────────────────┼──────┤
  │ RefractiveIndex.INFO   │ 复折射率 n(λ), k(λ) │ CSV  │
  ├────────────────────────┼─────────────────────┼──────┤
  │ USGS Spectral Library  │ 反射率光谱          │ CSV  │
  ├────────────────────────┼─────────────────────┼──────┤
  │ ASTER Spectral Library │ 发射率光谱          │ CSV  │
  └────────────────────────┴─────────────────────┴──────┘
  ---
  8. 完整数据流图

                             ┌─────────────────────────────────┐
                             │       External Data Sources      │
                             └─────────────────────────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
  ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
  │ ASTM G-173      │           │ MODTRAN/        │           │ RefractiveIndex │
  │ Solar Spectrum  │           │ libRadtran      │           │ .INFO / USGS    │
  │ (280-4000nm)    │           │ Atm. Trans.     │           │ Material Spectra│
  └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
           │                             │                             │
           ▼                             ▼                             ▼
  ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
  │ SolarSpectralLUT│           │ AtmTrans LUT    │           │ SpectralCurveGPU│
  │ (GPU Buffer)    │           │ (GPU Buffer)    │           │ (GPU Buffer)    │
  └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │          GPU Ray Tracing             │
                      │      (closesthit.rchit)              │
                      │                                      │
                      │  ┌────────────────────────────────┐  │
                      │  │ Mode Selection:                │  │
                      │  │  NIR/SWIR → Reflection Model   │  │
                      │  │  MWIR/LWIR → Thermal Model     │  │
                      │  └────────────────────────────────┘  │
                      │                                      │
                      │  ┌────────────────────────────────┐  │
                      │  │ MWIR/LWIR Physics:             │  │
                      │  │  L = ε·L_bb + ρ·L_atm↓        │  │
                      │  │    + ρ·L_sun·cosθ (MWIR only) │  │
                      │  │    + τ·L_bg                    │  │
                      │  └────────────────────────────────┘  │
                      │                                      │
                      │  ┌────────────────────────────────┐  │
                      │  │ Spectral Integration:          │  │
                      │  │  L_band = Σ L(λᵢ) × Δλ        │  │
                      │  └────────────────────────────────┘  │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │       Post-Processing (CPU)          │
                      │                                      │
                      │  ┌────────────────────────────────┐  │
                      │  │ GenericSensor:                 │  │
                      │  │  - PSF (Airy disk blur)        │  │
                      │  │  - QE × Integration time       │  │
                      │  │  - Noise (Poisson + Read)      │  │
                      │  │  - ADC quantization            │  │
                      │  └────────────────────────────────┘  │
                      │                                      │
                      │  ┌────────────────────────────────┐  │
                      │  │ MultibandFusion:               │  │
                      │  │  - Laplacian Pyramid           │  │
                      │  │  - Pseudo-color mapping        │  │
                      │  └────────────────────────────────┘  │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Output Image   │
                                │  (EXR/PNG/ENVI) │
                                └─────────────────┘

  ---
  9. 物理一致性检查清单

  9.1 单位一致性
  ┌────────────┬─────────────────┬────────────────────────────┐
  │     量     │      单位       │           验证点           │
  ├────────────┼─────────────────┼────────────────────────────┤
  │ 波长       │ nm (全流程)     │ lambda_nm 变量命名         │
  ├────────────┼─────────────────┼────────────────────────────┤
  │ 光谱辐亮度 │ W·sr⁻¹·m⁻²·nm⁻¹ │ IRPlanckRadiance 返回值    │
  ├────────────┼─────────────────┼────────────────────────────┤
  │ 辐照度     │ W·m⁻²·nm⁻¹      │ SampleSunIrradiance 返回值 │
  ├────────────┼─────────────────┼────────────────────────────┤
  │ 温度       │ K               │ irTemperature_K 字段       │
  └────────────┴─────────────────┴────────────────────────────┘
  9.2 能量守恒

  - 基尔霍夫定律: $\varepsilon + \rho + \tau = 1$
  - BRDF 归一化: Lambertian 除以 $\pi$
  - Stefan-Boltzmann 积分: $\int_0^\infty L_\lambda  d\lambda = \sigma T^4 / \pi$

  9.3 物理常数精度
  ┌──────────┬────────────────────────────┬─────────────┐
  │   常数   │             值             │ CODATA 2018 │
  ├──────────┼────────────────────────────┼─────────────┤
  │ $h$      │ 6.62607015×10⁻³⁴ J·s       │ ✓ 精确      │
  ├──────────┼────────────────────────────┼─────────────┤
  │ $c$      │ 299792458 m/s              │ ✓ 精确      │
  ├──────────┼────────────────────────────┼─────────────┤
  │ $k_B$    │ 1.380649×10⁻²³ J/K         │ ✓ 精确      │
  ├──────────┼────────────────────────────┼─────────────┤
  │ $\sigma$ │ 5.670374419×10⁻⁸ W·m⁻²·K⁻⁴ │ ✓ 精确      │
  └──────────┴────────────────────────────┴─────────────┘
  9.4 已知近似与限制
  ┌──────────────────┬───────────────────┬───────────────────────────┐
  │       项目       │     当前实现      │         改进方向          │
  ├──────────────────┼───────────────────┼───────────────────────────┤
  │ MWIR 大气透过率  │ 常数 0.8          │ 波长相关 MODTRAN LUT      │
  ├──────────────────┼───────────────────┼───────────────────────────┤
  │ 太阳 MWIR 辐照度 │ Planck 5778K 近似 │ ASTM G-173 扩展到 5μm     │
  ├──────────────────┼───────────────────┼───────────────────────────┤
  │ 多次散射         │ 未实现            │ Delta-Tracking + 能量补偿 │
  ├──────────────────┼───────────────────┼───────────────────────────┤
  │ 发射率光谱依赖   │ 灰体近似          │ 实测 ε(λ) 曲线            │
  └──────────────────┴───────────────────┴───────────────────────────┘
  ---
  10. 关键源文件索引
  ┌───────────────┬───────────────────────────────────────────────────────┬──────────────────────────────────┐
  │     功能      │                       文件路径                        │          关键函数/结构           │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 波段定义      │ src/libQuantiloom/core/Types.hpp                      │ WAVELENGTH_MIN/MAX_*             │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 普朗克辐射    │ src/shaders/blackbody.hlsli                           │ IRPlanckRadiance()               │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ IR 渲染主逻辑 │ src/shaders/closesthit.rchit                          │ lines 1452-1610                  │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 发射率推导    │ src/shaders/common.hlsli                              │ GetEffectiveIREmissivity()       │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 大气 LUT      │ src/libQuantiloom/core/AtmosphereTransmittanceLUT.hpp │ AtmosphereTransmittanceLUT       │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 传感器模型    │ src/libQuantiloom/postprocess/GenericSensor.hpp       │ SensorParams, ApplySensorChain() │
  ├───────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ 照明参数      │ src/libQuantiloom/renderer/LightingParams.hpp         │ atmosphereTemperature_K          │
  └───────────────┴───────────────────────────────────────────────────────┴──────────────────────────────────┘
  ---
  文档版本: 2026-01-27
  对应代码版本: 0.0.3