# Render the furnace cavities and check each against Planck.
#
# The PowerShell twin of run_furnace_suite.sh, for a Windows build machine that
# has MSVC and CMake and no POSIX shell. Both read the same cavity list
# (furnace_cases.txt) and call the same checker (check_furnace.py), so the only
# thing duplicated between them is the loop.
#
# check_furnace.py stays Python because it is the measurement -- band-integrated
# Planck radiance against an ROI mean over an EXR. That is numerical work with
# numpy and OpenEXR behind it, not shell glue, and rewriting it in PowerShell
# would be reimplementing the thing under test.
#
#   ./scripts/render-tests/run_furnace_suite.ps1
#
# Exit codes, which build_windows.ps1 distinguishes:
#   0  every cavity within tolerance
#   1  a cavity rendered and was wrong -- a real failure, stop the build
#   2  no CLI built
#   3  no usable GPU, so nothing was measured
#
# 3 is not 1 on purpose. This repo already decided that a missing ray tracing
# device makes a GPU check skip with a reason rather than fail (see
# tests/support/VulkanTestDevice.hpp); a build machine without an RTX card
# should not be told its physics is broken when the truth is that nobody looked.

$ErrorActionPreference = "Stop"

# Native commands here are read by their exit code, not by whether they wrote to
# stderr: the render gates distinguish 3 (no GPU, a skip) from 1 (a real
# failure), and the checkers report through stdout while exiting non-zero. If
# $PSNativeCommandUseErrorActionPreference is left $true -- it is $false by
# default but a profile can set it -- a non-zero exit throws before the code
# below can look at it, and "this machine has no RTX card" becomes "the build
# failed". Pinned rather than assumed.
$PSNativeCommandUseErrorActionPreference = $false

# Diagnostics on the way to an exit code go to stderr directly rather than
# through Write-Error. Write-Error raises a PowerShell error, and under
# $ErrorActionPreference = "Stop" that TERMINATES the script -- so `exit 3`
# below would never run, and "this machine has no ray tracing device", which is
# a skip, would reach build_windows.ps1 as a thrown exception and fail the
# build. The shell twin's `echo >&2; exit 3` has no such second meaning.
function Write-Stderr([string]$Message) { [Console]::Error.WriteLine($Message) }

Set-Location (Join-Path $PSScriptRoot "..\..")

$Cli = if ($env:CLI) { $env:CLI } else { ".\build\src\app\Release\Quantiloom.exe" }
if (-not (Test-Path $Cli)) {
    Write-Stderr "no CLI at $Cli -- build first"
    exit 2
}

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$CasesFile = "scripts\render-tests\furnace_cases.txt"
if (-not (Test-Path $CasesFile)) {
    Write-Stderr "no cavity list at $CasesFile"
    exit 2
}

$GpuAbsent = 'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable'
$fail = 0

foreach ($line in (Get-Content $CasesFile)) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
    $band, $case = $trimmed -split '\s+', 2
    $cfg = "assets/configs/furnace_${band}_${case}.toml"
    $out = "furnace_${band}_${case}_output.exr"

    # Count the success line rather than trusting the exit code: the CLI's is
    # unreliable, and a render that silently wrote nothing would otherwise be
    # checked against a stale file from a previous run.
    $log = (& $Cli $cfg 2>&1 | Out-String)
    if (([regex]::Matches($log, 'Saved spectral image')).Count -ne 1) {
        if ($log -match $GpuAbsent) {
            Write-Stderr "furnace suite: no usable GPU, nothing measured"
            exit 3
        }
        Write-Host "RENDER FAILED  ${band}_${case}"
        Write-Host (($log -split "`n" | Select-Object -Last 5) -join "`n")
        $fail = 1
        continue
    }

    # Capture first, then branch on the checker's own exit code.
    Write-Host ("{0,-22} " -f "${band}_${case}") -NoNewline
    $report = (& $Py "scripts/render-tests/check_furnace.py" $out $band.ToUpper() 2>&1 | Out-String)
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok) {
        Write-Host (((($report -split "`n") | Where-Object { $_ -match 'Rel error' }) -join '').Trim()) -NoNewline
        Write-Host "  PASS"
    } else {
        Write-Host (((($report -split "`n") | Where-Object { $_ -match 'Rel error|FAIL' }) | ForEach-Object { $_.Trim() }) -join ' ')
        $fail = 1
    }
}

if ($fail -eq 0) {
    Write-Host "furnace suite: all cavities within tolerance"
} else {
    Write-Host "furnace suite: FAILURES above" -ForegroundColor Red
}
exit $fail
