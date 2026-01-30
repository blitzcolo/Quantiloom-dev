@echo off
REM ============================================================================
REM Quantiloom M1 - Shader Compilation Script (Windows)
REM ============================================================================
REM Compiles all HLSL ray tracing shaders to SPIR-V using DXC
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

REM Compile raygen shader
echo [1/7] Compiling raygen.rgen...
dxc %FLAGS% -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile raygen.rgen
    pause
    exit /b 1
)
echo       OK src/shaders/raygen.spv created

REM Compile closesthit shader
echo [2/7] Compiling closesthit.rchit...
dxc %FLAGS% -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile closesthit.rchit
    pause
    exit /b 1
)
echo       OK src/shaders/closesthit.spv created

REM Compile miss shader
echo [3/7] Compiling miss.rmiss...
dxc %FLAGS% -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/miss.spv created

REM Compile shadow_miss shader
echo [4/7] Compiling shadow_miss.rmiss...
dxc %FLAGS% -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile shadow_miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/shadow_miss.spv created

REM Compile CLAHE compute shaders (3 passes)
echo [5/7] Compiling clahe_histogram.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_histogram
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_histogram.spv created

echo [6/7] Compiling clahe_cdf.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_cdf
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_cdf.spv created

echo [7/7] Compiling clahe_apply.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_apply
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_apply.spv created

echo.
echo =========================================
echo   All shaders compiled successfully!
echo =========================================
echo.


