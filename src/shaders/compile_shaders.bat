@echo off
REM ============================================================================
REM Quantiloom M1 - Shader Compilation Script (Windows)
REM ============================================================================
REM Compiles all HLSL ray tracing shaders to SPIR-V using DXC
REM ============================================================================
REM MANUAL FALLBACK. CMake compiles these same 15 shaders as part of a normal
REM build (src/shaders/CMakeLists.txt), with the same flags, and tracks .hlsli
REM dependencies -- so an ordinary build picks up shader edits on its own.
REM Use this only when building without CMake, or to force a recompile.
REM Keep the flags below in step with DXC_RT_FLAGS / DXC_COMP_FLAGS there.
REM ============================================================================

echo =========================================
echo   Quantiloom Shader Compilation
echo =========================================
echo You need to run it from the root folder of the project (where the main CMakeLists.txt is located)
echo.

REM Check if DXC is available
where dxc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: DXC not found in PATH
    echo.
    echo Please install Vulkan SDK from https://vulkan.lunarg.com/
    echo.
    pause
    exit /b 1
)

echo DXC found
echo.

REM Compilation flags
set FLAGS=-spirv -T lib_6_3 -fspv-target-env="vulkan1.2"
set COMP_FLAGS=-spirv -T cs_6_0 -fspv-target-env="vulkan1.2"
set RQ_FLAGS=-spirv -T cs_6_5 -fspv-target-env="vulkan1.2"

REM Compile raygen shader
echo [1/14] Compiling raygen.rgen...
dxc %FLAGS% -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile raygen.rgen
    pause
    exit /b 1
)
echo       OK src/shaders/raygen.spv created

REM Compile closesthit shader
echo [2/14] Compiling closesthit.rchit...
dxc %FLAGS% -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile closesthit.rchit
    pause
    exit /b 1
)
echo       OK src/shaders/closesthit.spv created

REM Compile miss shader
echo [3/14] Compiling miss.rmiss...
dxc %FLAGS% -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/miss.spv created

REM Compile shadow_miss shader
echo [4/14] Compiling shadow_miss.rmiss...
dxc %FLAGS% -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile shadow_miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/shadow_miss.spv created

REM Compile CLAHE compute shaders (3 passes)
echo [5/14] Compiling clahe_histogram.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_histogram
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_histogram.spv created

echo [6/14] Compiling clahe_cdf.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_cdf
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_cdf.spv created

echo [7/14] Compiling clahe_apply.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_apply
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_apply.spv created

REM Compile GPU Sensor compute shaders (5 passes)
echo [8/14] Compiling sensor_radiance_to_electrons.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/sensor_radiance_to_electrons.spv src/shaders/sensor_radiance_to_electrons.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_radiance_to_electrons
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_radiance_to_electrons.spv created

echo [9/14] Compiling sensor_poisson_noise.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/sensor_poisson_noise.spv src/shaders/sensor_poisson_noise.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_poisson_noise
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_poisson_noise.spv created

echo [10/14] Compiling sensor_psf_blur_horizontal.comp...
dxc %COMP_FLAGS% -E main -D SENSOR_PSF_HORIZONTAL -Fo src/shaders/sensor_psf_blur_horizontal.spv src/shaders/sensor_psf_blur.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_psf_blur_horizontal
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_psf_blur_horizontal.spv created

echo [11/14] Compiling sensor_psf_blur_vertical.comp...
dxc %COMP_FLAGS% -E main -D SENSOR_PSF_VERTICAL -Fo src/shaders/sensor_psf_blur_vertical.spv src/shaders/sensor_psf_blur.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_psf_blur_vertical
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_psf_blur_vertical.spv created

echo [12/14] Compiling sensor_quantize_to_radiance.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/sensor_quantize_to_radiance.spv src/shaders/sensor_quantize_to_radiance.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_quantize_to_radiance
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_quantize_to_radiance.spv created

echo [13/14] Compiling sensor_fpn.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/sensor_fpn.spv src/shaders/sensor_fpn.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile sensor_fpn
    pause
    exit /b 1
)
echo       OK src/shaders/sensor_fpn.spv created

echo [14/15] Compiling pick.rayq...
dxc %RQ_FLAGS% -E main -Fo src/shaders/pick.spv src/shaders/pick.rayq.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile pick
    pause
    exit /b 1
)
echo       OK src/shaders/pick.spv created

echo [15/15] Compiling thermal_exchange.rayq...
dxc %RQ_FLAGS% -E main -Fo src/shaders/thermal_exchange.spv src/shaders/thermal_exchange.rayq.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile thermal_exchange
    pause
    exit /b 1
)
echo       OK src/shaders/thermal_exchange.spv created

echo.
echo =========================================
echo   All shaders compiled successfully!
echo =========================================
echo.


