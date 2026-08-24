# Publish the built tree as the SDK Quantiloom-Qt links against.
#
# Refuses unless build_windows.ps1 left a stamp saying its gates passed for THIS
# build. build_wsl.sh gets that ordering from `set -e` in a single script; two
# scripts cannot, and the failure it prevents is quiet -- an SDK with a red test
# suite or an unreviewed export surface looks exactly like a good one from here.
#
#   ./install_windows.ps1           install if the gates passed
#   ./install_windows.ps1 -Force    install anyway, and say so

param([switch]$Force)

$ErrorActionPreference = "Stop"

$BuildDir = Join-Path $PSScriptRoot "build"
$StampFile = Join-Path $BuildDir ".gates-passed.json"
$Dll = Join-Path $BuildDir "src\libQuantiloom\Release\Quantiloom.dll"

if ($Force) {
    Write-Warning "-Force: installing without checking the gate stamp."
} else {
    if (-not (Test-Path $StampFile)) {
        throw "no gate stamp at $StampFile -- run ./build_windows.ps1 first, or -Force to override."
    }
    $stamp = Get-Content $StampFile -Raw | ConvertFrom-Json

    # A stamp older than the DLL means someone rebuilt without re-running the
    # gates, so the stamp describes a build that no longer exists.
    if (Test-Path $Dll) {
        $dllNow = (Get-Item $Dll).LastWriteTimeUtc
        if ($dllNow -gt [datetime]::Parse($stamp.dllWriteUtc).ToUniversalTime().AddSeconds(1)) {
            throw "the build is newer than the gate stamp -- re-run ./build_windows.ps1, or -Force to override."
        }
    }

    if ($stamp.renderGates -ne "run") {
        Write-Warning "gate stamp says render gates were $($stamp.renderGates)."
        Write-Warning "Nothing has checked that this build renders correctly."
    }
    Write-Host "Gates: tests $($stamp.tests), ABI $($stamp.abi), render $($stamp.renderGates) (commit $($stamp.commit))."
}

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
if ($LASTEXITCODE -ne 0) { Set-Location $PSScriptRoot; throw "cmake --install failed (exit $LASTEXITCODE)" }
Set-Location $PSScriptRoot
Write-Host "Installed SDK: $SdkDir"
