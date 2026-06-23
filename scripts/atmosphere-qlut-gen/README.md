# 大气 LUT 生成器

本目录包含用于生成 `.qlut`（Quantiloom LUT）大气透过率文件的工具和文档。

## 概述

Quantiloom 使用离线预计算的三维查找表来模拟大气效应：
- **波长**（λ）：300–14000 nm
- **高度**（h）：0–30000 m
- **天顶角**（θ）：0–85°

这些 LUT 由辐射传输代码（MODTRAN、libRadtran 等）离线生成，运行时以 O(1) 复杂度查询。

---

## 文件格式规范（.qlut）

### 结构

```
┌─────────────────────────────────────┐
│  TOML Header (1024 bytes, padded)   │  ← 人类可读的元数据
├─────────────────────────────────────┤
│  Transmittance Data (float32[])     │  ← 二进制，C 顺序（行主序）
├─────────────────────────────────────┤
│  Path Radiance Data (float32[])     │  ← 二进制，C 顺序（可选）
└─────────────────────────────────────┘
```

### TOML 头部格式

```toml
[metadata]
name = "Midlatitude Summer"
source = "libRadtran 2.0.4"
created = "2024-12-25T10:30:00Z"
atmospheric_model = "US_Standard_1976"
format_version = 1

[grid.wavelength]
start = 300.0      # nm
stop = 14000.0     # nm
step = 10.0        # nm
count = 1371       # 采样点数

[grid.altitude]
start = 0.0        # 米
stop = 30000.0     # 米
step = 1000.0      # 米
count = 31         # 采样点数

[grid.zenith]
values = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]  # 度

[data]
header_size = 1024
transmittance_offset = 1024
transmittance_count = 297501        # 1371 × 31 × 7
path_radiance_offset = 2191028      # 1024 + 297501 × 4
path_radiance_count = 297501
dtype = "float32"
byte_order = "little"
layout = "wavelength_altitude_zenith"
```

### 二进制数据布局

数据以 **C 顺序（行主序）** 存储，索引方式为 `[wavelength][altitude][zenith]`：

```
Index = i_wavelength × (n_altitude × n_zenith) + i_altitude × n_zenith + i_zenith
```

以上述头部为例：
- `transmittance[0]` = τ(λ=300nm, h=0m, θ=0°)
- `transmittance[7]` = τ(λ=300nm, h=1000m, θ=0°)
- `transmittance[217]` = τ(λ=310nm, h=0m, θ=0°)

---

## Python 生成器模板

### 依赖

```bash
pip install numpy toml
# libRadtran 集成（可选）：
pip install f90nml
```

### 基础生成器

```python
#!/usr/bin/env python3
"""
Quantiloom Atmosphere LUT Generator
Generates .qlut files from radiative transfer model outputs.
"""

import numpy as np
import struct
from datetime import datetime
from pathlib import Path


class QLUTGenerator:
    """Generates Quantiloom atmosphere LUT files."""

    HEADER_SIZE = 1024  # bytes, padded with null

    def __init__(self):
        self.metadata = {
            "name": "Unknown",
            "source": "Unknown",
            "created": datetime.now().isoformat(),
            "atmospheric_model": "Unknown",
        }
        self.wavelength_nm = None      # 1D array
        self.altitude_m = None          # 1D array
        self.zenith_deg = None          # 1D array
        self.transmittance = None       # 3D array [wave, alt, zen]
        self.path_radiance = None       # 3D array [wave, alt, zen] (optional)

    def set_grid(self, wavelength_nm, altitude_m, zenith_deg):
        """
        Set the grid axes.

        Parameters:
            wavelength_nm: 1D array of wavelengths (must be uniform spacing)
            altitude_m: 1D array of altitudes (must be uniform spacing)
            zenith_deg: 1D array of zenith angles (can be non-uniform)
        """
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=np.float32)
        self.altitude_m = np.asarray(altitude_m, dtype=np.float32)
        self.zenith_deg = np.asarray(zenith_deg, dtype=np.float32)

        wave_step = np.diff(self.wavelength_nm)
        if not np.allclose(wave_step, wave_step[0], rtol=1e-3):
            raise ValueError("Wavelength grid must be uniformly spaced")

        alt_step = np.diff(self.altitude_m)
        if not np.allclose(alt_step, alt_step[0], rtol=1e-3):
            raise ValueError("Altitude grid must be uniformly spaced")

    def set_transmittance(self, data):
        """
        Set transmittance data.

        Parameters:
            data: 3D array with shape [n_wavelength, n_altitude, n_zenith], values in [0, 1]
        """
        self.transmittance = np.asarray(data, dtype=np.float32)
        expected_shape = (len(self.wavelength_nm), len(self.altitude_m), len(self.zenith_deg))
        if self.transmittance.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {self.transmittance.shape}")

    def set_path_radiance(self, data):
        """
        Set path radiance data (optional).

        Parameters:
            data: 3D array with shape [n_wavelength, n_altitude, n_zenith], W·sr⁻¹·m⁻²·nm⁻¹
        """
        self.path_radiance = np.asarray(data, dtype=np.float32)

    def _generate_header(self) -> str:
        """Generate TOML header string."""
        n_wave = len(self.wavelength_nm)
        n_alt = len(self.altitude_m)
        n_zen = len(self.zenith_deg)
        total_elements = n_wave * n_alt * n_zen

        wave_step = float(self.wavelength_nm[1] - self.wavelength_nm[0])
        alt_step = float(self.altitude_m[1] - self.altitude_m[0])

        lines = [
            '[metadata]',
            f'name = "{self.metadata["name"]}"',
            f'source = "{self.metadata["source"]}"',
            f'created = "{self.metadata["created"]}"',
            f'atmospheric_model = "{self.metadata["atmospheric_model"]}"',
            'format_version = 1',
            '',
            '[grid.wavelength]',
            f'start = {float(self.wavelength_nm[0])}',
            f'stop = {float(self.wavelength_nm[-1])}',
            f'step = {wave_step}',
            f'count = {n_wave}',
            '',
            '[grid.altitude]',
            f'start = {float(self.altitude_m[0])}',
            f'stop = {float(self.altitude_m[-1])}',
            f'step = {alt_step}',
            f'count = {n_alt}',
            '',
            '[grid.zenith]',
            f'values = [{", ".join(str(float(z)) for z in self.zenith_deg)}]',
            '',
            '[data]',
            f'header_size = {self.HEADER_SIZE}',
            f'transmittance_offset = {self.HEADER_SIZE}',
            f'transmittance_count = {total_elements}',
        ]

        if self.path_radiance is not None:
            path_radiance_offset = self.HEADER_SIZE + total_elements * 4
            lines.extend([
                f'path_radiance_offset = {path_radiance_offset}',
                f'path_radiance_count = {total_elements}',
            ])

        lines.extend([
            'dtype = "float32"',
            'byte_order = "little"',
            'layout = "wavelength_altitude_zenith"',
        ])

        return '\n'.join(lines)

    def save(self, filepath):
        """Save LUT to .qlut file."""
        filepath = Path(filepath)
        if filepath.suffix != '.qlut':
            filepath = filepath.with_suffix('.qlut')

        header = self._generate_header()
        if len(header) > self.HEADER_SIZE:
            raise ValueError(f"Header too large: {len(header)} > {self.HEADER_SIZE}")

        header_bytes = header.encode('utf-8').ljust(self.HEADER_SIZE, b'\x00')

        with open(filepath, 'wb') as f:
            f.write(header_bytes)
            f.write(self.transmittance.astype('<f4').tobytes())
            if self.path_radiance is not None:
                f.write(self.path_radiance.astype('<f4').tobytes())

        print(f"Saved: {filepath} ({filepath.stat().st_size} bytes)")


# =============================================================================
# Example: Generate a simple test LUT (Beer-Lambert approximation)
# =============================================================================

def generate_test_lut(output_path="test_atmosphere.qlut"):
    """
    Generate a simple test LUT using Beer-Lambert law.
    NOT physically accurate — use MODTRAN/libRadtran for production data.
    """
    gen = QLUTGenerator()
    gen.metadata = {
        "name": "Test Atmosphere (Beer-Lambert)",
        "source": "Quantiloom test generator",
        "created": datetime.now().isoformat(),
        "atmospheric_model": "Simplified_Beer_Lambert",
    }

    wavelength_nm = np.arange(300, 14001, 10, dtype=np.float32)
    altitude_m = np.arange(0, 30001, 1000, dtype=np.float32)
    zenith_deg = np.array([0, 15, 30, 45, 60, 75, 85], dtype=np.float32)

    gen.set_grid(wavelength_nm, altitude_m, zenith_deg)

    n_wave, n_alt, n_zen = len(wavelength_nm), len(altitude_m), len(zenith_deg)
    beta_rayleigh_550 = 1.2e-5
    scale_height = 8500.0

    def water_absorption(wl):
        bands = [(1400,100,0.5),(1900,150,0.8),(2700,200,1.0),(6300,500,1.5)]
        return sum(s*np.exp(-0.5*((wl-c)/w)**2) for c,w,s in bands) * 1e-4

    transmittance = np.zeros((n_wave, n_alt, n_zen), dtype=np.float32)
    for i_w, wave in enumerate(wavelength_nm):
        beta = beta_rayleigh_550 * (550.0/wave)**4 + water_absorption(wave)
        for i_a, alt in enumerate(altitude_m):
            b = beta * np.exp(-alt/scale_height)
            for i_z, zen in enumerate(zenith_deg):
                cos_z = np.cos(np.radians(zen))
                am = 1.0/cos_z if cos_z > 0.1 else 1.0/(cos_z+0.50572*(96.07995-zen)**(-1.6364))
                transmittance[i_w,i_a,i_z] = np.exp(-b*scale_height*am)

    gen.set_transmittance(transmittance)
    gen.save(output_path)
    return gen


if __name__ == "__main__":
    generate_test_lut("test_atmosphere.qlut")
```

---

## 与辐射传输代码的集成

### MODTRAN 5/6

MODTRAN 输出可解析并转换为 `.qlut` 格式：

```python
def load_modtran_tape7(filepath):
    """
    Parse MODTRAN tape7 output file.

    Returns:
        wavelength_nm, transmittance, path_radiance arrays
    """
    data = np.loadtxt(filepath, skiprows=5)
    wavenumber_cm = data[:, 0]
    transmittance = data[:, 1]
    path_radiance = data[:, 2]

    # Convert wavenumber (cm⁻¹) to wavelength (nm)
    wavelength_nm = 1e7 / wavenumber_cm

    # Reverse: MODTRAN outputs high-to-low wavenumber
    return wavelength_nm[::-1], transmittance[::-1], path_radiance[::-1]


def generate_modtran_lut(tape7_files, output_path):
    """
    Generate .qlut from multiple MODTRAN runs.

    Parameters:
        tape7_files: dict mapping (altitude_m, zenith_deg) -> tape7_filepath
        output_path: output .qlut file path
    """
    gen = QLUTGenerator()
    gen.metadata = {"name":"MODTRAN Atmosphere","source":"MODTRAN 5.4","atmospheric_model":"Tropical"}

    first_key = next(iter(tape7_files))
    wavelength_nm, _, _ = load_modtran_tape7(tape7_files[first_key])

    altitudes = sorted(set(k[0] for k in tape7_files))
    zeniths   = sorted(set(k[1] for k in tape7_files))
    wave_uniform = np.arange(300, 14001, 10, dtype=np.float32)

    gen.set_grid(wave_uniform, np.array(altitudes,dtype=np.float32), np.array(zeniths,dtype=np.float32))

    transmittance = np.zeros((len(wave_uniform),len(altitudes),len(zeniths)),dtype=np.float32)
    for (alt,zen), filepath in tape7_files.items():
        wave, trans, _ = load_modtran_tape7(filepath)
        transmittance[:,altitudes.index(alt),zeniths.index(zen)] = np.interp(wave_uniform,wave,trans)

    gen.set_transmittance(transmittance)
    gen.save(output_path)
```

### libRadtran

```python
def run_libradtran(wavelength_nm, altitude_m, zenith_deg, output_dir):
    """Run libRadtran uvspec for a single configuration. Requires uvspec in PATH."""
    import subprocess
    from pathlib import Path

    input_content = f"""
atmosphere_file ../data/atmmod/afglms.dat
source solar ../data/solar_flux/atlas_plus_modtran
wavelength {wavelength_nm[0]} {wavelength_nm[-1]}
altitude {altitude_m / 1000.0}
sza {zenith_deg}
output_quantity transmittance
quiet
"""
    input_file = Path(output_dir) / f"uvspec_alt{altitude_m}_zen{zenith_deg}.inp"
    output_file = input_file.with_suffix('.out')
    input_file.write_text(input_content)

    result = subprocess.run(['uvspec'], stdin=open(input_file),
                            stdout=open(output_file,'w'), stderr=subprocess.PIPE, cwd=output_dir)
    if result.returncode != 0:
        raise RuntimeError(f"uvspec failed: {result.stderr.decode()}")

    data = np.loadtxt(output_file)
    return data[:,0], data[:,1]  # wavelength, transmittance
```

---

## 推荐 LUT 配置

### 标准配置

| 名称 | 波长范围 | 高度范围 | 天顶角 | 文件大小 |
|------|----------|----------|--------|----------|
| **最小** | 380–780nm，5nm 步长 | 0–10km，2km 步长 | 0,30,60° | ~100 KB |
| **可见+近红外** | 380–2500nm，5nm 步长 | 0–15km，1km 步长 | 7 个角度 | ~1 MB |
| **全红外** | 300–14000nm，10nm 步长 | 0–30km，1km 步长 | 7 个角度 | ~12 MB |
| **高分辨率 MWIR** | 3000–5000nm，1nm 步长 | 0–20km，500m 步长 | 7 个角度 | ~5 MB |

### 大气模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| `US_Standard_1976` | 标准大气 | 通用 |
| `Midlatitude_Summer` | 中纬度夏季 | 夏季场景 |
| `Midlatitude_Winter` | 中纬度冬季 | 冬季场景 |
| `Tropical` | 高湿度热带 | 热带地区 |
| `Subarctic_Summer` | 寒冷干燥 | 极地场景 |
| `Urban` | 高气溶胶 | 城市环境 |

---

## 验证

生成 `.qlut` 文件后，可用以下脚本验证完整性：

```python
def validate_qlut(filepath):
    """Validate .qlut file integrity."""
    import toml
    with open(filepath, 'rb') as f:
        header = f.read(1024).decode('utf-8').rstrip('\x00')
        print("Header valid:", '[metadata]' in header)

        config = toml.loads(header)
        n_wave = config['grid']['wavelength']['count']
        n_alt  = config['grid']['altitude']['count']
        n_zen  = len(config['grid']['zenith']['values'])
        expected_bytes = n_wave * n_alt * n_zen * 4

        data = f.read()
        print(f"Expected: {expected_bytes} bytes, Got: {len(data)} bytes")
        print(f"Has path radiance: {len(data) >= expected_bytes * 2}")

        trans = np.frombuffer(data[:expected_bytes], dtype='<f4')
        print(f"Transmittance range: [{trans.min():.4f}, {trans.max():.4f}]")
        print(f"Valid [0,1]: {trans.min() >= 0 and trans.max() <= 1}")


if __name__ == "__main__":
    validate_qlut("test_atmosphere.qlut")
```

---

## 使用 QLTrans 生成 LUT

项目内置了 `QLTrans` 工具（仅 Windows），基于 MOD4v1r1（MODTRAN 4）自动生成分段 LUT：

```powershell
pwsh scripts\atmosphere-qlut-gen\generate_atmosphere_luts.ps1 `
    -QLTransExe .\build\Release\QLTrans.exe `
    -OutDir .\assets\luts\modtran `
    -MaxJobs 4
```

脚本会生成 6 个分段文件（vis / nir / swir / mwir_hi / lwir / vlwir），每段边界有重叠以支持平滑插值。

---

## 问题反馈

如有 LUT 生成相关问题，请在 Quantiloom 仓库提交 Issue。
