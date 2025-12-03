# Quantiloom Unit Tests

This directory contains comprehensive unit tests for the Quantiloom spectral path tracing system, implemented using the Google Test (GTest) framework.

## 测试结构 (Test Structure)

```
tests/
├── test_core/           # 核心模块测试 (Core module tests)
│   ├── test_types.cpp           # Result类型, 光谱模式解析, 错误码
│   ├── test_image.cpp           # 图像容器, 像素访问, 内存布局
│   ├── test_spectral_data.cpp   # 光谱曲线, 插值, GPU转换
│   └── test_config.cpp          # TOML配置加载和解析
│
├── test_scene/          # 场景模块测试 (Scene module tests)
│   ├── test_material.cpp        # 材质验证, PBR参数, 红外属性
│   └── test_camera.cpp          # 相机变换, FOV, 光线生成
│
├── test_io/             # I/O模块测试 (I/O module tests)
│   └── test_image_io.cpp        # EXR读写, HDR保存, 元数据
│
├── test_renderer/       # 渲染器模块测试 (Renderer tests - 可选)
│   └── [待实现]                  # Vulkan组件测试需要GPU环境
│
├── test_hs_core/        # HS-core算法测试 (Algorithm tests - 待实现)
│   └── [待实现]                  # MIS, delta-tracking等
│
├── test_postprocess/    # 后处理测试 (Post-processing tests - 待实现)
│   └── [待实现]                  # 传感器链, 噪声模型
│
├── CMakeLists.txt       # 测试构建配置
└── README.md            # 本文件
```

## 测试覆盖范围 (Test Coverage)

### ✅ 已实现 (Implemented)

#### Core Module (`test_core/`)
- **`test_types.cpp`** (70+ tests)
  - `Result<T, E>` 类型的成功/错误处理
  - 光谱模式解析 (Single, RGB, Multispectral, MWIR, LWIR)
  - 错误码转换和物理常量验证
  - C++20 concepts测试 (Arithmetic, Numeric)

- **`test_image.cpp`** (30+ tests)
  - 图像构造和初始化
  - 行主序 (row-major) 通道后置 (channel-last) 内存布局验证
  - 像素访问和操作 (写入/读取/指针访问)
  - HDR值处理和多通道支持
  - 元数据和通道命名

- **`test_spectral_data.cpp`** (40+ tests)
  - `SpectralCurve` 线性插值精度
  - 边界外处理 (out-of-range wavelengths)
  - 单调性验证和波长范围检查
  - `SpectralCurveGPU` 转换和降采样
  - 物理真实性测试 (可见光反射率, 红外发射率)

- **`test_config.cpp`** (35+ tests)
  - TOML文件加载和解析
  - 键导航 (点分隔路径: `renderer.width`)
  - 类型转换 (i32, f32, String, bool, 数组)
  - 默认值和必需键验证
  - 嵌套表和复杂结构

#### Scene Module (`test_scene/`)
- **`test_material.cpp`** (30+ tests)
  - PBR参数范围验证 (metallic, roughness, baseColor)
  - 光谱反照率计算
  - 红外材质属性 (发射率, 反射率, 透射率)
  - `SpectralSource` 追踪 (Measured, RGBUpsampled, Procedural)
  - Alpha模式 (Opaque, Mask, Blend)
  - 工厂方法 (CreateLambertian)

- **`test_camera.cpp`** (40+ tests)
  - Look-at变换和向量计算
  - 正交归一基 (orthonormal basis) 验证
  - FOV和长宽比处理
  - 相机数据生成 (GPU推送常量)
  - 特殊配置 (向下看, 向上看, 侧视)

#### I/O Module (`test_io/`)
- **`test_image_io.cpp`** (25+ tests)
  - EXR文件读写 (单通道, RGB, 多光谱)
  - HDR值保存 (高达10000+, 负值支持)
  - 元数据保存和恢复
  - 通道名称保存
  - 文件存在性检查和尺寸读取
  - 边界情况 (单像素, 大图像, 200通道)

### 🚧 待实现 (Planned)

#### Renderer Module (`test_renderer/`)
- Vulkan初始化测试 (需要GPU环境或mock)
- GPU缓冲区创建和上传
- 加速结构构建 (BLAS/TLAS)
- 纹理管理和绑定
- 光线追踪管线创建

#### HS-Core Module (`test_hs_core/`)
- 混合PDF MIS采样验证
- Delta-tracking算法正确性
- 分层采样和低差异序列
- 抗萤火虫策略

#### Post-Processing Module (`test_postprocess/`)
- 传感器响应卷积
- 噪声链 (泊松, 读噪, 暗电流)
- DN生成和量化

## 构建和运行 (Build and Run)

### 前提条件 (Prerequisites)

1. **Google Test** 安装:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgtest-dev

   # macOS (Homebrew)
   brew install googletest

   # Windows (vcpkg)
   vcpkg install gtest
   ```

2. **CMake 3.20+** 和 **C++20编译器** (GCC 10+, Clang 12+, MSVC 2019+)

### 构建测试 (Build Tests)

```bash
# 从项目根目录
cd /path/to/Quantiloom

# 创建构建目录
mkdir build && cd build

# 配置CMake (包括测试)
cmake .. -DBUILD_TESTING=ON

# 构建测试可执行文件
cmake --build . --target libquantiloom_tests

# 或使用make (Unix)
make libquantiloom_tests
```

### 运行所有测试 (Run All Tests)

```bash
# 使用CTest (推荐)
ctest --output-on-failure

# 或直接运行可执行文件
./tests/libquantiloom_tests

# Windows
.\tests\Debug\libquantiloom_tests.exe
```

### 运行特定测试 (Run Specific Tests)

```bash
# 只运行核心模块测试
make test_core

# 只运行场景模块测试
make test_scene

# 只运行I/O模块测试
make test_io

# 使用GTest过滤器运行特定测试套件
./tests/libquantiloom_tests --gtest_filter=ImageTest.*

# 运行特定测试用例
./tests/libquantiloom_tests --gtest_filter=ImageTest.PixelWriteRead

# 排除某些测试
./tests/libquantiloom_tests --gtest_filter=-ImageTest.LargeImageStressTest
```

### 详细输出 (Verbose Output)

```bash
# CTest详细输出
ctest --verbose

# GTest详细输出
./tests/libquantiloom_tests --gtest_print_time=1

# 同时显示测试名称和时间
./tests/libquantiloom_tests --gtest_color=yes --gtest_print_time=1
```

## 代码覆盖率 (Code Coverage - Linux/GCC)

```bash
# 启用覆盖率构建
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON

# 运行测试
ctest

# 生成覆盖率报告
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info  # 过滤系统头文件
lcov --list coverage.info  # 查看摘要

# 生成HTML报告
genhtml coverage.info --output-directory coverage_html
firefox coverage_html/index.html  # 打开浏览器查看
```

## 持续集成 (CI/CD)

### GitHub Actions 示例

```yaml
name: Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libgtest-dev cmake

      - name: Build
        run: |
          mkdir build && cd build
          cmake .. -DBUILD_TESTING=ON
          cmake --build . --target libquantiloom_tests

      - name: Run Tests
        run: |
          cd build
          ctest --output-on-failure
```

## 测试命名约定 (Naming Conventions)

- **测试套件 (Test Suite)**: `<ModuleName>Test` (e.g., `ImageTest`, `MaterialTest`)
- **测试用例 (Test Case)**: `<FunctionName><Condition>` (e.g., `PixelWriteRead`, `IsValidMetallicRange`)
- **文件命名**: `test_<module_name>.cpp` (e.g., `test_image.cpp`, `test_material.cpp`)

## 编写新测试 (Writing New Tests)

### 模板示例

```cpp
// test_core/test_your_module.cpp

#include <gtest/gtest.h>
#include "core/YourModule.hpp"

using namespace quantiloom;

// Test fixture (可选, 用于共享设置)
class YourModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试前的初始化
    }

    void TearDown() override {
        // 每个测试后的清理
    }

    // 共享测试数据
    YourModule module;
};

// 简单测试用例
TEST(YourModuleTest, BasicFunctionality) {
    YourModule mod;
    EXPECT_EQ(mod.GetValue(), 42);
}

// 使用fixture的测试用例
TEST_F(YourModuleTest, AdvancedFunctionality) {
    module.SetValue(100);
    EXPECT_GT(module.GetValue(), 50);
}
```

### 常用断言 (Common Assertions)

```cpp
// 相等性断言
EXPECT_EQ(a, b);      // a == b
EXPECT_NE(a, b);      // a != b
ASSERT_EQ(a, b);      // 失败时终止测试

// 比较断言
EXPECT_LT(a, b);      // a < b
EXPECT_LE(a, b);      // a <= b
EXPECT_GT(a, b);      // a > b
EXPECT_GE(a, b);      // a >= b

// 浮点数断言
EXPECT_FLOAT_EQ(a, b);   // 浮点相等 (默认误差)
EXPECT_NEAR(a, b, epsilon);  // 在epsilon范围内相等

// 布尔断言
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// 字符串断言
EXPECT_STREQ("hello", str);
EXPECT_STRNE("hello", str);

// 异常断言
EXPECT_THROW(statement, exception_type);
EXPECT_NO_THROW(statement);
```

## 调试测试 (Debugging Tests)

### GDB (Linux/macOS)

```bash
# 构建Debug版本
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 使用GDB运行
gdb --args ./tests/libquantiloom_tests --gtest_filter=ImageTest.PixelWriteRead

# GDB命令
(gdb) run              # 运行测试
(gdb) break test_image.cpp:42  # 设置断点
(gdb) continue         # 继续执行
(gdb) print variable   # 打印变量
```

### Visual Studio (Windows)

1. 在Visual Studio中打开项目
2. 设置 `libquantiloom_tests` 为启动项目
3. 项目属性 → 调试 → 命令参数: `--gtest_filter=YourTest.*`
4. F5 启动调试

## 性能基准 (Performance Benchmarks)

对于性能敏感的模块 (如光谱插值, delta-tracking), 可以添加基准测试:

```cpp
TEST(SpectralDataTest, InterpolationBenchmark) {
    SpectralCurve curve = CreateLargeCurve();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; ++i) {
        f32 value = curve.Evaluate(500.0f);
        (void)value;  // 防止优化
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "1M evaluations: " << duration.count() << " μs" << std::endl;
    EXPECT_LT(duration.count(), 100000);  // 预期小于100ms
}
```

## 贡献指南 (Contributing)

1. **添加新测试**: 在对应的 `test_<module>/` 目录下创建测试文件
2. **更新CMakeLists.txt**: 将新测试文件添加到对应的 `TEST_*_SOURCES` 变量
3. **运行所有测试**: 确保所有现有测试仍然通过
4. **代码覆盖率**: 新功能应有 ≥80% 的代码覆盖率
5. **文档**: 在测试文件顶部注释说明测试的功能和覆盖范围

## 参考资源 (References)

- **Google Test 官方文档**: https://google.github.io/googletest/
- **CMake Testing**: https://cmake.org/cmake/help/latest/manual/ctest.1.html
- **Quantiloom SRS**: `/local/软件需规SRS.md`
- **验收标准**: SRS §9.2 (M1.5 V&V 基准套件)

---

**测试覆盖率统计** (当前):
- **核心模块 (Core)**: ~175 tests (types, image, spectral_data, config)
- **场景模块 (Scene)**: ~70 tests (material, camera)
- **I/O模块 (I/O)**: ~25 tests (image_io)
- **总计**: ~270 单元测试

**目标**: 根据SRS M1.5要求, 建立自动化、可复现的交叉验证基准套件。
