<#
.SYNOPSIS
    Build OpenUSD (Pixar USD) for Quantiloom - Windows

.DESCRIPTION
    This script builds a minimal OpenUSD installation without Python support,
    imaging (Hydra), or tools. Only core USD libraries are built.

.PARAMETER InstallDir
    Installation directory (default: C:\openusd)

.PARAMETER UsdVersion
    OpenUSD version tag (default: v25.08)

.PARAMETER BuildType
    CMake build type: Release, Debug, RelWithDebInfo (default: Release)

.EXAMPLE
    .\build_openusd.ps1 -InstallDir "D:\openusd" -UsdVersion "v25.11"
#>

param(
    [string]$InstallDir = "C:\openusd",
    [string]$UsdVersion = "v25.08",
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " OpenUSD Build Script for Quantiloom (Windows)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Version:     $UsdVersion"
Write-Host "  Install Dir: $InstallDir"
Write-Host "  Build Type:  $BuildType"
Write-Host ""

# Check prerequisites
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Check Python
$pythonPath = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonPath) {
    Write-Error "Python not found. Please install Python 3.9+ and add to PATH."
    exit 1
}
Write-Host "  Python: $($pythonPath.Source)"

# Check Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Error "Visual Studio not found. Please install Visual Studio 2022."
    exit 1
}
$vsPath = & $vsWhere -latest -property installationPath
Write-Host "  Visual Studio: $vsPath"

# Check CMake
$cmakePath = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakePath) {
    Write-Error "CMake not found. Please install CMake 3.26+ and add to PATH."
    exit 1
}
Write-Host "  CMake: $($cmakePath.Source)"

# Clone or update OpenUSD
$usdSourceDir = Join-Path $env:TEMP "OpenUSD-$UsdVersion"

Write-Host ""
Write-Host "[2/5] Preparing OpenUSD source..." -ForegroundColor Yellow

if (Test-Path $usdSourceDir) {
    Write-Host "  Using existing source at: $usdSourceDir"
} else {
    Write-Host "  Cloning OpenUSD $UsdVersion..."
    git clone --depth 1 --branch $UsdVersion https://github.com/PixarAnimationStudios/OpenUSD.git $usdSourceDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to clone OpenUSD repository"
        exit 1
    }
}

# Build OpenUSD
Write-Host ""
Write-Host "[3/5] Building OpenUSD (this may take 30-60 minutes)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Build options:"
Write-Host "    --no-python       (disable Python bindings)"
Write-Host "    --no-imaging      (disable Hydra/usdview)"
Write-Host "    --no-usdview      (disable usdview tool)"
Write-Host "    --no-examples     (skip examples)"
Write-Host "    --no-tutorials    (skip tutorials)"
Write-Host "    --no-tools        (skip command-line tools)"
Write-Host "    --no-docs         (skip documentation)"
Write-Host ""

$buildScript = Join-Path $usdSourceDir "build_scripts\build_usd.py"

# Run build script
$buildArgs = @(
    $buildScript,
    "--no-python",
    "--no-imaging",
    "--no-usdview",
    "--no-examples",
    "--no-tutorials",
    "--no-tools",
    "--no-docs",
    "--build-variant", $BuildType.ToLower(),
    $InstallDir
)

Write-Host "  Running: python $($buildArgs -join ' ')"
Write-Host ""

& python @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenUSD build failed"
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "[4/5] Verifying installation..." -ForegroundColor Yellow

$requiredLibs = @(
    "usd.lib",
    "usdGeom.lib",
    "usdShade.lib",
    "sdf.lib",
    "tf.lib",
    "gf.lib",
    "ar.lib",
    "plug.lib",
    "vt.lib",
    "work.lib",
    "arch.lib"
)

$libDir = Join-Path $InstallDir "lib"
$missingLibs = @()

foreach ($lib in $requiredLibs) {
    $libPath = Join-Path $libDir $lib
    if (Test-Path $libPath) {
        Write-Host "  [OK] $lib" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $lib" -ForegroundColor Red
        $missingLibs += $lib
    }
}

if ($missingLibs.Count -gt 0) {
    Write-Warning "Some required libraries are missing. Build may have failed partially."
}

# Print summary
Write-Host ""
Write-Host "[5/5] Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " OpenUSD installed to: $InstallDir" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use in Quantiloom CMake:"
Write-Host ""
Write-Host "  cmake -B build -DUSD_ROOT=`"$InstallDir`" ..."
Write-Host ""
Write-Host "Or set environment variable:"
Write-Host ""
Write-Host "  `$env:USD_ROOT = `"$InstallDir`""
Write-Host ""
