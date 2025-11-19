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
set FLAGS=-spirv -T lib_6_3 -fspv-target-env="vulkan1.3"

REM Compile raygen shader
echo [1/3] Compiling raygen.rgen...
dxc %FLAGS% -Fo src/shaders/raygen.spv src/shaders/raygen.rgen
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile raygen.rgen
    pause
    exit /b 1
)
echo       OK src/shaders/raygen.spv created

REM Compile closesthit shader
echo [2/3] Compiling closesthit.rchit...
dxc %FLAGS% -Fo src/shaders/closesthit.spv src/shaders/closesthit.rchit
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile closesthit.rchit
    pause
    exit /b 1
)
echo       OK src/shaders/closesthit.spv created

REM Compile miss shader
echo [3/3] Compiling miss.rmiss...
dxc %FLAGS% -Fo src/shaders/miss.spv src/shaders/miss.rmiss
if %ERRORLEVEL% NEQ 0 (
    echo       X Failed to compile miss.rmiss
    pause
    exit /b 1
)
echo       OK src/shaders/miss.spv created

echo.
echo =========================================
echo   All shaders compiled successfully!
echo =========================================
echo.
echo You can now run the M1 test application.

