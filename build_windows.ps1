# Build Quantiloom on Windows, behind the same four gates build_wsl.sh runs.
#
# This script and build_wsl.sh must stay functionally equivalent, because they
# install the same SDK into the same place and Quantiloom-Qt cannot tell which
# one produced it. They were not equivalent: this one configured with
# QUANTILOOM_BUILD_TESTS=OFF and ran no gate at all, so `build_windows.ps1;
# install_windows.ps1` published a red build, an unreviewed export surface and
# an unrendered physics suite into the SDK without a word.
#
#   ./build_windows.ps1        build + gates
#   ./install_windows.ps1      install, refuses unless the gates passed
#
# Dependencies are MSVC, CMake and PowerShell -- no POSIX shell, and no WSL.
# The gates are PowerShell ports of the shell scripts build_wsl.sh calls, not
# wrappers around them: a Windows build machine need not have bash, so a gate
# reachable only through bash is a gate that silently does not run.
#
# The one thing not ported is the render gates' checkers, which stay Python.
# They compute band-integrated Planck radiance, RMSE and region statistics over
# EXR images with numpy and OpenEXR behind them; that is the measurement itself,
# not shell glue, and reimplementing it in PowerShell would be rewriting the
# thing under test. Python is a portable dependency, unlike bash -- point
# $env:QUANTILOOM_PYTHON at an interpreter that has numpy and OpenEXR if the
# one on PATH does not. Without it the render gates are SKIPPED, loudly, and
# the stamp records that they were.

$ErrorActionPreference = "Stop"

# Native commands here are read by their exit code, not by whether they wrote to
# stderr: the render gates distinguish 3 (no GPU, a skip) from 1 (a real
# failure), and the checkers report through stdout while exiting non-zero. If
# $PSNativeCommandUseErrorActionPreference is left $true -- it is $false by
# default but a profile can set it -- a non-zero exit throws before the code
# below can look at it, and "this machine has no RTX card" becomes "the build
# failed". Pinned rather than assumed.
$PSNativeCommandUseErrorActionPreference = $false
$SourceDir = $PSScriptRoot
$BuildDir  = Join-Path $SourceDir "build"
Set-Location $SourceDir

# --- Stale build directory guard ---
# CMakeCache.txt hard-codes the absolute source path. If the project was moved
# or copied to another drive/folder, the old cache breaks the configure step
# (typically: Unknown CMake command "CPMAddPackage"). Wipe it and start clean.
$CacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $CacheFile) {
    $homeLine = Select-String -Path $CacheFile -Pattern "^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$" | Select-Object -First 1
    if ($homeLine) {
        $cached = $homeLine.Matches[0].Groups[1].Value.Replace("/", "\").TrimEnd("\")
        $actual = $SourceDir.Replace("/", "\").TrimEnd("\")
        if ($cached -ne $actual) {
            Write-Host "Stale build cache: `"$cached`" != `"$actual`" - removing $BuildDir" -ForegroundColor Yellow
            Remove-Item -LiteralPath $BuildDir -Recurse -Force
        }
    }
}

function Assert-ExitZero([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

# The render gates' checkers import numpy and OpenEXR. `python3` on a stock
# Windows is a Microsoft Store stub that imports neither, so the interpreter is
# probed rather than assumed.
function Find-Python {
    $candidates = @()
    if ($env:QUANTILOOM_PYTHON) { $candidates += $env:QUANTILOOM_PYTHON }
    $candidates += @("python", "python3", "py")
    foreach ($c in $candidates) {
        $exe = (Get-Command $c -ErrorAction SilentlyContinue | Select-Object -First 1)
        if (-not $exe) { continue }
        & $exe.Source -c "import numpy, OpenEXR" 2>$null
        if ($LASTEXITCODE -eq 0) { return $exe.Source }
    }
    return $null
}

# --- Configure ---------------------------------------------------------------
# QUANTILOOM_BUILD_TESTS is passed explicitly, not left to its default, because
# this script used to force it OFF and CMake caches it: a tree configured by the
# old script kept tests off afterwards, and build_wsl.sh -- which never passes
# the flag -- would then reach its test gate and find no test binary.
cmake -B build -G "Visual Studio 18 2026" -A x64 `
    -DQUANTILOOM_BUILD_TESTS=ON `
    -DQUANTILOOM_USE_BC7ENC=OFF `
    -DQUANTILOOM_USE_OPENUSD=ON `
    -DUSD_ROOT=C:/openusd
Assert-ExitZero "cmake configure"

# --- Build -------------------------------------------------------------------
# Shaders are compiled by CMake (the CompileShaders target, which Quantiloom and
# the test binary both depend on) and copied next to the executables by a
# POST_BUILD step. This script used to call src\shaders\compile_shaders.bat
# first as well, which compiled every shader twice with a different DXC -- the
# .bat takes whatever is on PATH -- and left the winner decided by timestamps.
cmake --build build --config Release -j -- /v:q /nologo
Assert-ExitZero "cmake build"

# --- Test gate ---------------------------------------------------------------
# No CI runs this suite; the gates here are the only automated ones. A red suite
# must never reach the install step, or broken code lands in the SDK that
# Quantiloom-Qt links against.
& "$BuildDir\tests\Release\libquantiloom_tests.exe" --gtest_brief=1
Assert-ExitZero "test gate"

# --- ABI gate ----------------------------------------------------------------
# The export table is the SDK's contract with Quantiloom-Qt, and it drifts
# quietly: a new class picks up QL_API by habit and is public forever.
& "$SourceDir\scripts\check_exports.ps1"
Assert-ExitZero "ABI gate"

# --- Physics gates -----------------------------------------------------------
# Eight isothermal furnace cavities against Planck, then the illumination
# suite's seven arms: sun occlusion per band, an open sky whose traced bounce
# must contribute exactly zero, the same ground in the visible where the two
# visible modes are held to each other, a Cornell box lit only by emissive
# geometry, the MIS check that light sampling and BSDF sampling estimate the
# same image, the convergence of the sampled visible estimator against the
# deterministic one, a fluorescent transfer between two wavelength bands, and
# view independence.
#
# Exit 3 means no usable GPU. That is a warning, not a failure -- the same call
# the unit suite makes for its GPU cases -- but it does mean this build was
# never checked against anything that renders.
$Python = Find-Python
$RenderGates = "run"
if ($Python) {
    Write-Host "python: $Python"
    $env:PYTHON = $Python
    foreach ($gate in @(@{Name = "physics"; Script = "scripts\render-tests\run_furnace_suite.ps1"},
                        @{Name = "illumination"; Script = "scripts\render-tests\run_illumination_suite.ps1"})) {
        & (Join-Path $SourceDir $gate.Script)
        if ($LASTEXITCODE -eq 3) {
            Write-Warning "$($gate.Name) gate skipped, no GPU on this machine"
            $RenderGates = "skipped: no GPU"
        } elseif ($LASTEXITCODE -ne 0) {
            throw "$($gate.Name) gate failed -- not installing."
        }
    }
} else {
    $RenderGates = "skipped: no Python with numpy + OpenEXR"
    Write-Warning "no Python with numpy and OpenEXR found -- the furnace and"
    Write-Warning "illumination gates did NOT run. Nothing has checked that this"
    Write-Warning "build puts a photon on a surface correctly. Set"
    Write-Warning "`$env:QUANTILOOM_PYTHON to an interpreter that has them."
}

# --- Stamp -------------------------------------------------------------------
# install_windows.ps1 reads this and refuses without it. Two scripts cannot
# enforce an order between themselves the way build_wsl.sh's single `set -e`
# does, so the ordering is recorded instead of assumed.
$Dll = Join-Path $BuildDir "src\libQuantiloom\Release\Quantiloom.dll"
# git is not a build dependency -- a release tarball has no .git and a build
# machine need not have the client -- so the commit is recorded when it can be
# and left null when it cannot, rather than throwing here.
$commit = $null
if (Get-Command git -ErrorAction SilentlyContinue) {
    $commit = (& git rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0) { $commit = $null }
}
$stamp = [ordered]@{
    utc          = (Get-Date).ToUniversalTime().ToString("o")
    commit       = $commit
    dllWriteUtc  = (Get-Item $Dll).LastWriteTimeUtc.ToString("o")
    tests        = "passed"
    abi          = "passed"
    renderGates  = $RenderGates
}
$stamp | ConvertTo-Json | Set-Content -Path (Join-Path $BuildDir ".gates-passed.json") -Encoding utf8

Write-Host ""
Write-Host "Build OK. Gates: tests passed, ABI passed, render gates $RenderGates." -ForegroundColor Green
Write-Host "Run ./install_windows.ps1 to publish the SDK."
