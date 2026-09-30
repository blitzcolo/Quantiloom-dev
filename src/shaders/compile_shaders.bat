@echo off
REM ============================================================================
REM Quantiloom M1 - Shader Compilation Script (Windows)
REM ============================================================================
REM Compiles all HLSL ray tracing shaders to SPIR-V using DXC
REM ============================================================================
REM MANUAL FALLBACK. CMake compiles these same 24 shaders as part of a normal
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
echo [1/24] Compiling raygen.rgen...
dxc %FLAGS% -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile raygen.rgen
    pause
    exit /b 1
)
echo       OK src/shaders/raygen.spv created

REM Compile closesthit shader
echo [2/24] Compiling closesthit.rchit...
dxc %FLAGS% -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile closesthit.rchit
    pause
    exit /b 1
)
echo       OK src/shaders/closesthit.spv created

REM Compile miss shader
echo [3/24] Compiling anyhit.rahit...
dxc %FLAGS% -Fo src/shaders/anyhit.spv src/shaders/anyhit.rahit
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile anyhit.rahit
    pause
    exit /b 1
)
echo       OK src/shaders/anyhit.spv created

echo [4/24] Compiling miss.rmiss...
dxc %FLAGS% -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/miss.spv created

REM Compile shadow_miss shader
echo [5/24] Compiling shadow_miss.rmiss...
dxc %FLAGS% -Fo src/shaders/shadow_miss.spv src/shaders/shadow_miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile shadow_miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/shadow_miss.spv created

REM Compile CLAHE compute shaders (3 passes)
echo [6/24] Compiling clahe_histogram.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_HISTOGRAM -Fo src/shaders/clahe_histogram.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_histogram
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_histogram.spv created

echo [7/24] Compiling clahe_cdf.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_CDF -Fo src/shaders/clahe_cdf.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_cdf
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_cdf.spv created

echo [8/24] Compiling clahe_apply.comp...
dxc %COMP_FLAGS% -E main -D CLAHE_PASS_APPLY -Fo src/shaders/clahe_apply.spv src/shaders/clahe.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile clahe_apply
    pause
    exit /b 1
)
echo       OK src/shaders/clahe_apply.spv created

REM Compile display-range reduction shaders (2 passes)
echo Compiling display_range_extents.comp...
dxc %COMP_FLAGS% -E main -D DISPLAY_RANGE_PASS_EXTENTS -Fo src/shaders/display_range_extents.comp.spv src/shaders/display_range.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile display_range_extents
    pause
    exit /b 1
)
echo       OK src/shaders/display_range_extents.comp.spv created

echo Compiling display_range_histogram.comp...
dxc %COMP_FLAGS% -E main -D DISPLAY_RANGE_PASS_HISTOGRAM -Fo src/shaders/display_range_histogram.comp.spv src/shaders/display_range.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile display_range_histogram
    pause
    exit /b 1
)
echo       OK src/shaders/display_range_histogram.comp.spv created

echo [9/24] Compiling pick.rayq...
dxc %RQ_FLAGS% -E main -Fo src/shaders/pick.spv src/shaders/pick.rayq.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile pick
    pause
    exit /b 1
)
echo       OK src/shaders/pick.spv created

echo [10/24] Compiling thermal_exchange.rayq...
dxc %RQ_FLAGS% -E main -Fo src/shaders/thermal_exchange.spv src/shaders/thermal_exchange.rayq.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile thermal_exchange
    pause
    exit /b 1
)
echo       OK src/shaders/thermal_exchange.spv created

echo [11/24] Compiling thermal_step.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/thermal_step.spv src/shaders/thermal_step.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile thermal_step
    pause
    exit /b 1
)
echo       OK src/shaders/thermal_step.spv created

REM Unified camera device preview compute shaders
echo [12/24] Compiling camera_psf.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_psf.comp.spv src/shaders/camera_psf.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_psf
    pause
    exit /b 1
)
echo       OK src/shaders/camera_psf.comp.spv created

echo [13/24] Compiling camera_readout.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_readout.comp.spv src/shaders/camera_readout.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_readout
    pause
    exit /b 1
)
echo       OK src/shaders/camera_readout.comp.spv created

echo [14/24] Compiling camera_stats_tiles.comp...
dxc %COMP_FLAGS% -E main -D CAMERA_STATS_PHASE_TILES -Fo src/shaders/camera_stats_tiles.comp.spv src/shaders/camera_stats.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_stats_tiles
    pause
    exit /b 1
)
echo       OK src/shaders/camera_stats_tiles.comp.spv created

echo [15/24] Compiling camera_stats_reduce.comp...
dxc %COMP_FLAGS% -E main -D CAMERA_STATS_PHASE_REDUCE -Fo src/shaders/camera_stats_reduce.comp.spv src/shaders/camera_stats.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_stats_reduce
    pause
    exit /b 1
)
echo       OK src/shaders/camera_stats_reduce.comp.spv created

echo [16/24] Compiling camera_stats_hist.comp...
dxc %COMP_FLAGS% -E main -D CAMERA_STATS_PHASE_HISTOGRAM -Fo src/shaders/camera_stats_hist.comp.spv src/shaders/camera_stats.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_stats_hist
    pause
    exit /b 1
)
echo       OK src/shaders/camera_stats_hist.comp.spv created

echo [17/24] Compiling camera_cdf.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_cdf.comp.spv src/shaders/camera_cdf.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_cdf
    pause
    exit /b 1
)
echo       OK src/shaders/camera_cdf.comp.spv created

echo [18/24] Compiling camera_demosaic.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_demosaic.comp.spv src/shaders/camera_demosaic.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_demosaic
    pause
    exit /b 1
)
echo       OK src/shaders/camera_demosaic.comp.spv created

echo [19/24] Compiling camera_color.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_color.comp.spv src/shaders/camera_color.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_color
    pause
    exit /b 1
)
echo       OK src/shaders/camera_color.comp.spv created

echo [20/24] Compiling camera_rng_vectors.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_rng_vectors.comp.spv src/shaders/camera_rng_vectors.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_rng_vectors
    pause
    exit /b 1
)
echo       OK src/shaders/camera_rng_vectors.comp.spv created

echo [21/24] Compiling camera_fast_rgb.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_fast_rgb.comp.spv src/shaders/camera_fast_rgb.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_fast_rgb
    pause
    exit /b 1
)
echo       OK src/shaders/camera_fast_rgb.comp.spv created

echo [22/24] Compiling camera_display.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_display.comp.spv src/shaders/camera_display.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_display
    pause
    exit /b 1
)
echo       OK src/shaders/camera_display.comp.spv created

echo [23/24] Compiling camera_dynamic.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_dynamic.comp.spv src/shaders/camera_dynamic.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_dynamic
    pause
    exit /b 1
)
echo       OK src/shaders/camera_dynamic.comp.spv created
echo [24/24] Compiling camera_hsv.comp...
dxc %COMP_FLAGS% -E main -Fo src/shaders/camera_hsv.comp.spv src/shaders/camera_hsv.comp.hlsl
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile camera_hsv
    pause
    exit /b 1
)
echo       OK src/shaders/camera_hsv.comp.spv created

echo.
echo =========================================
echo   All shaders compiled successfully!
echo =========================================
echo.
