# ============================================================================
# DXC Shader Compiler Setup Script (Windows PowerShell)
# ============================================================================
# Downloads and extracts DirectXShaderCompiler (DXC) from GitHub releases
# ============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ToolsDir = Join-Path $ProjectRoot "tools"
$DxcDir = Join-Path $ToolsDir "dxc"

$DxcVersion = "v1.8.2505.1"  # Latest stable release as of 2024
$DxcArchive = "dxc_2025_07_14.zip"
$DxcUrl = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DxcVersion/$DxcArchive"
$DxcBinPath = Join-Path $DxcDir "bin\x64\dxc.exe"

Write-Host "[DXC Setup] Platform: Windows" -ForegroundColor Cyan

# Check if DXC is already installed
if (Test-Path $DxcBinPath) {
    Write-Host "[DXC Setup] DXC already installed at: $DxcBinPath" -ForegroundColor Green
    & $DxcBinPath --version
    Write-Host "[DXC Setup] To reinstall, delete: $DxcDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "[DXC Setup] DXC not found, downloading..." -ForegroundColor Yellow
Write-Host "[DXC Setup] Version: $DxcVersion" -ForegroundColor Cyan
Write-Host "[DXC Setup] URL: $DxcUrl" -ForegroundColor Cyan

# Create DXC directory
New-Item -ItemType Directory -Force -Path $DxcDir | Out-Null

# Download DXC archive
$ArchivePath = Join-Path $DxcDir $DxcArchive
Write-Host "[DXC Setup] Downloading DXC..." -ForegroundColor Cyan

try {
    Invoke-WebRequest -Uri $DxcUrl -OutFile $ArchivePath -UseBasicParsing
} catch {
    Write-Host "[ERROR] Failed to download DXC: $_" -ForegroundColor Red
    exit 1
}

# Extract archive
Write-Host "[DXC Setup] Extracting DXC..." -ForegroundColor Cyan
try {
    Expand-Archive -Path $ArchivePath -DestinationPath $DxcDir -Force
} catch {
    Write-Host "[ERROR] Failed to extract DXC: $_" -ForegroundColor Red
    exit 1
}

# Clean up archive
Remove-Item $ArchivePath -Force

# Verify installation
if (Test-Path $DxcBinPath) {
    Write-Host "[DXC Setup] Installation successful!" -ForegroundColor Green
    Write-Host "[DXC Setup] DXC binary: $DxcBinPath" -ForegroundColor Cyan
    & $DxcBinPath --version
} else {
    Write-Host "[ERROR] DXC binary not found after extraction" -ForegroundColor Red
    Write-Host "        Expected: $DxcBinPath" -ForegroundColor Red
    exit 1
}

Write-Host "[DXC Setup] Done. You can now build shaders with CMake." -ForegroundColor Green
