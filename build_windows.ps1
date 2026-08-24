# Build Quantiloom on Windows, behind the same four gates build_wsl.sh runs.
#
# This script and build_wsl.sh must stay functionally equivalent, because they
# install the same SDK into the same place and Quantiloom-Qt cannot tell which
# one produced it. They were not equivalent: this one configured with
# QUANTILOOM_BUILD_TESTS=OFF and ran no gate at all, so `build_windows.ps1;
# install_windows.ps1` published a red build, an unreviewed export surface and
# an unrendered physics suite into the SDK without a word.
#
# The gates themselves are not reimplemented here. They are the same three shell
# scripts build_wsl.sh calls, run through Git Bash, so there is one definition of
# each check rather than two that drift.
#
#   ./build_windows.ps1        build + gates
#   ./install_windows.ps1      install, refuses unless the gates passed
#
# Two prerequisites this needs and WSL supplies for free:
#   * a POSIX shell -- Git for Windows
#   * a Python with numpy and OpenEXR, for the render gates' checkers. Point
#     $env:QUANTILOOM_PYTHON at one if it is not on PATH. Without it the render
#     gates are SKIPPED, loudly, and the stamp records that they were.

$ErrorActionPreference = "Stop"
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

# --- Helpers -----------------------------------------------------------------

function Assert-ExitZero([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

# The gates live in bash. Git for Windows ships one; prefer bin\bash.exe, which
# sets up the MSYS environment, over usr\bin\bash.exe.
function Find-Bash {
    if ($env:QUANTILOOM_BASH) { return $env:QUANTILOOM_BASH }
    $onPath = Get-Command bash.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike "*\WindowsApps\*" -and $_.Source -notlike "*\System32\*" } |
        Select-Object -First 1
    if ($onPath) { return $onPath.Source }
    foreach ($c in @("$env:ProgramFiles\Git\bin\bash.exe",
                     "${env:ProgramFiles(x86)}\Git\bin\bash.exe")) {
        if (Test-Path $c) { return $c }
    }
    # System32\bash.exe is the WSL launcher, not a Windows bash: it would run the
    # gates against the WSL filesystem view and a different toolchain. Refusing is
    # clearer than silently building somewhere else.
    throw "no Git Bash found. Install Git for Windows, or set `$env:QUANTILOOM_BASH."
}

# The render gates' checkers import numpy and OpenEXR. `python3` on a stock
# Windows is a Microsoft Store stub that imports nothing, so the interpreter is
# probed rather than assumed.
function Find-Python {
    $candidates = @()
    if ($env:QUANTILOOM_PYTHON) { $candidates += $env:QUANTILOOM_PYTHON }
    $candidates += @("python", "python3")
    foreach ($c in $candidates) {
        $exe = (Get-Command $c -ErrorAction SilentlyContinue | Select-Object -First 1)
        if (-not $exe) { continue }
        & $exe.Source -c "import numpy, OpenEXR" 2>$null
        if ($LASTEXITCODE -eq 0) { return $exe.Source }
    }
    return $null
}

$Bash = Find-Bash
Write-Host "bash:   $Bash"

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
& $Bash "./scripts/check_exports.sh"
Assert-ExitZero "ABI gate"

# --- Physics gates -----------------------------------------------------------
# Eight isothermal furnace cavities against Planck, then the illumination suite:
# sun occlusion per band, an open sky whose traced bounce must contribute
# exactly zero, a Cornell box lit only by emissive geometry, and the MIS check
# that light sampling and BSDF sampling estimate the same image.
#
# Exit 3 means no usable GPU. That is a warning, not a failure -- the same call
# the unit suite makes for its GPU cases -- but it does mean this build was
# never checked against anything that renders.
$Python = Find-Python
$RenderGates = "run"
if ($Python) {
    Write-Host "python: $Python"
    $env:PYTHON = $Python
    foreach ($gate in @(@{Name = "physics"; Script = "./scripts/render-tests/run_furnace_suite.sh"},
                        @{Name = "illumination"; Script = "./scripts/render-tests/run_illumination_suite.sh"})) {
        & $Bash $gate.Script
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
$stamp = [ordered]@{
    utc          = (Get-Date).ToUniversalTime().ToString("o")
    commit       = (& git rev-parse HEAD 2>$null)
    dllWriteUtc  = (Get-Item $Dll).LastWriteTimeUtc.ToString("o")
    tests        = "passed"
    abi          = "passed"
    renderGates  = $RenderGates
}
$stamp | ConvertTo-Json | Set-Content -Path (Join-Path $BuildDir ".gates-passed.json") -Encoding utf8

Write-Host ""
Write-Host "Build OK. Gates: tests passed, ABI passed, render gates $RenderGates." -ForegroundColor Green
Write-Host "Run ./install_windows.ps1 to publish the SDK."
