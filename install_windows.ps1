$ErrorActionPreference = "Stop"

# SDK prefix is derived from where this repo sits, never hard-coded to a drive:
# Quantiloom-Qt resolves the SDK as ../Quantiloom-SDK relative to its own checkout,
# so a literal D:\ here would keep wiping and installing the wrong tree after a move.
# Override with $env:QUANTILOOM_SDK_ROOT if the SDK lives elsewhere.
$SdkRoot = $env:QUANTILOOM_SDK_ROOT
if (-not $SdkRoot) { $SdkRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "Quantiloom-SDK" }
$SdkDir = Join-Path $SdkRoot "windows_amd64"

Set-Location (Join-Path $PSScriptRoot "build")
if (Test-Path $SdkDir) { Remove-Item -Recurse -Force $SdkDir }
cmake --install . --prefix $SdkDir --config Release
Set-Location $PSScriptRoot
Write-Host "Installed SDK: $SdkDir"
