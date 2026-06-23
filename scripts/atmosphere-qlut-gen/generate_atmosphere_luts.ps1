# generate_atmosphere_luts.ps1
# Batch-generate segmented .qlut files by running QLTrans.exe in parallel.
# Each segment has a small overlap with adjacent segments for boundary interpolation.
# Iterates over all combinations of: segments × models × weathers × hazes.
# Each LUT is parameterized by wavelength × solar zenith angle (2D texture).
#
# Usage:
#   pwsh generate_atmosphere_luts.ps1 [-QLTransExe <path>] [-OutDir <path>] [-MaxJobs <n>]

param(
    [string]    $QLTransExe = ".\QLTrans.exe",
    [string]    $OutDir     = ".\modtran_luts",
    [string]    $ScriptDir  = "$PSScriptRoot\..\..\scripts\atmosphere-qlut-gen",
    [int]       $MaxJobs    = 4,

    [int[]] $Models   = @(2, 6),       # 2=Midlatitude_Summer  6=US_Standard
    [int[]] $Weathers = @(0),          # 0=clear
    [int[]] $Hazes    = @(1, 4, 5),    # 1=Rural_23km  4=Maritime_23km  5=Urban_5km

    # Solar zenith angle sweep
    [double] $SunZenStart = 0,
    [double] $SunZenStop  = 90,
    [double] $SunZenStep  = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Wavelength segments (with overlap) ────────────────────────────────────────
$Segments = @(
    @{ name="vis";      wave=1; wl_start=378;   wl_stop=782;   wl_step=1  },  # ±2nm overlap
    @{ name="nir";      wave=1; wl_start=776;   wl_stop=1504;  wl_step=2  },  # ±4nm overlap
    @{ name="swir";     wave=2; wl_start=1490;  wl_stop=5025;  wl_step=5  },  # ±10nm overlap
    @{ name="mwir_hi";  wave=2; wl_start=4980;  wl_stop=8020;  wl_step=10 },  # ±20nm overlap
    @{ name="lwir";     wave=3; wl_start=7960;  wl_stop=12040; wl_step=20 },  # ±40nm overlap
    @{ name="vlwir";    wave=3; wl_start=11950; wl_stop=25050; wl_step=50 }   # ±50nm overlap
)

# ── Build job list ─────────────────────────────────────────────────────────────
$jobs = [System.Collections.Generic.List[hashtable]]::new()
foreach ($seg in $Segments) {
    foreach ($model in $Models) {
        foreach ($weather in $Weathers) {
            foreach ($haze in $Hazes) {
                $jobs.Add(@{ seg=$seg; model=$model; weather=$weather; ihaze=$haze })
            }
        }
    }
}

$nSun = [int](($SunZenStop - $SunZenStart) / $SunZenStep) + 1
Write-Host "Total jobs: $($jobs.Count) ($($Segments.Count) segments x $($Models.Count) models x $($Weathers.Count) weathers x $($Hazes.Count) hazes)"
Write-Host "Solar zenith: $SunZenStart-$SunZenStop deg, step $SunZenStep ($nSun points per LUT)"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$converter = Join-Path $ScriptDir "modtran_to_qlut.py"
if (-not (Test-Path $converter)) { Write-Error "Converter not found: $converter"; exit 1 }
if (-not (Test-Path $QLTransExe)) { Write-Error "QLTrans.exe not found: $QLTransExe"; exit 1 }

# Resolve to absolute paths
$QLTransExe = (Resolve-Path $QLTransExe).Path
$OutDir     = (Resolve-Path $OutDir).Path
$converter  = (Resolve-Path $converter).Path

# ── Parallel dispatch (PowerShell 7 ForEach-Object -Parallel) ─────────────────
$jobs | ForEach-Object -ThrottleLimit $MaxJobs -Parallel {
    $job       = $_
    $exe       = $using:QLTransExe
    $outDir    = $using:OutDir
    $conv      = $using:converter
    $szStart   = $using:SunZenStart
    $szStop    = $using:SunZenStop
    $szStep    = $using:SunZenStep

    $seg     = $job.seg
    $model   = $job.model
    $weather = $job.weather
    $ihaze   = $job.ihaze
    $tag     = "$($seg.name)_m${model}_wx${weather}_hz${ihaze}"
    $workDir = Join-Path $outDir "tmp_$tag"
    New-Item -ItemType Directory -Force -Path $workDir | Out-Null
    $absWorkDir = (Resolve-Path $workDir).Path

    $tomlPath = Join-Path $absWorkDir "qltrans.toml"
    $toml = @"
[atmosphere]
wave=$($seg.wave)
weather=$weather
model=$model
ihaze=$ihaze
wl_start=$($seg.wl_start)
wl_stop=$($seg.wl_stop)
wl_step=$($seg.wl_step)

[solar]
zenith_start=$szStart
zenith_stop=$szStop
zenith_step=$szStep

[output]
dir="$($absWorkDir -replace '\\','\\')"
sun_file="sun.txt"
sky_file="sky.txt"
trans_file="trans.txt"
"@
    Set-Content -Path $tomlPath -Value $toml

    # Run QLTrans directly as a process with hidden window
    $psi = [System.Diagnostics.ProcessStartInfo]::new($exe, "`"$tomlPath`"")
    $psi.WorkingDirectory = $absWorkDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardOutput.ReadToEnd() | Out-Null
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        Write-Warning "[$tag] QLTrans failed (exit $($proc.ExitCode)): $stderr"; return
    }

    $qlut    = Join-Path $outDir "${tag}.qlut"
    $sunTxt  = Join-Path $absWorkDir "sun.txt"
    $skyTxt  = Join-Path $absWorkDir "sky.txt"
    $tranTxt = Join-Path $absWorkDir "trans.txt"
    $pyArgs  = @("`"$conv`"",
        "--sun",      "`"$sunTxt`"",
        "--sky",      "`"$skyTxt`"",
        "--trans",    "`"$tranTxt`"",
        "--wave",     $seg.wave,
        "--model",    $model,
        "--weather",  $weather,
        "--ihaze",    $ihaze,
        "--wl-start", $seg.wl_start,
        "--wl-stop",  $seg.wl_stop,
        "--wl-step",  $seg.wl_step,
        "--sun-zen-start", $szStart,
        "--sun-zen-stop",  $szStop,
        "--sun-zen-step",  $szStep,
        "--out",      "`"$qlut`"") -join " "
    $pyPsi = [System.Diagnostics.ProcessStartInfo]::new("python", $pyArgs)
    $pyPsi.UseShellExecute = $false
    $pyPsi.CreateNoWindow = $true
    $pyPsi.RedirectStandardOutput = $true
    $pyPsi.RedirectStandardError  = $true
    $pyProc = [System.Diagnostics.Process]::Start($pyPsi)
    $pyProc.StandardOutput.ReadToEnd() | Out-Null
    $pyErr = $pyProc.StandardError.ReadToEnd()
    $pyProc.WaitForExit()
    if ($pyProc.ExitCode -ne 0) {
        Write-Warning "[$tag] Python converter failed (exit $($pyProc.ExitCode)):`n$pyErr"; return
    }

    Remove-Item -Recurse -Force $workDir
    Write-Host "[$tag] -> $qlut"
}

Write-Host "`nDone. LUTs written to: $OutDir"
