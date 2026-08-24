# Compare the built DLLs' export tables against the reviewed baselines in
# docs/abi/. Exports are the SDK's contract: adding one is a SemVer minor and
# a promise to keep it working, removing one is a major. Neither should happen
# because somebody wanted a unit test to link.
#
#   ./scripts/check_exports.ps1            # verify (what build_windows.ps1 runs)
#   ./scripts/check_exports.ps1 -Update    # accept the current tables
#
# The PowerShell twin of check_exports.sh, which build_wsl.sh runs. Both must
# produce byte-identical baselines, so both undecorate, strip the same noise and
# sort the same way -- see the note on ordering below, which is the one place
# where "the same way" is not the obvious thing.
#
# This is a port rather than a wrapper on purpose: a Windows build machine has
# MSVC and CMake and need not have bash, so the gate cannot be reachable only
# through a POSIX shell. It uses nothing but PowerShell and the two MSVC tools
# the repository already requires.

param([switch]$Update)

$ErrorActionPreference = "Stop"

# Native commands here are read by their exit code, not by whether they wrote to
# stderr: the render gates distinguish 3 (no GPU, a skip) from 1 (a real
# failure), and the checkers report through stdout while exiting non-zero. If
# $PSNativeCommandUseErrorActionPreference is left $true -- it is $false by
# default but a profile can set it -- a non-zero exit throws before the code
# below can look at it, and "this machine has no RTX card" becomes "the build
# failed". Pinned rather than assumed.
$PSNativeCommandUseErrorActionPreference = $false
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$BuildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build" }
# Absolute, not "docs/abi": Set-Location only updates PowerShell's $PWD, and on
# a cross-drive `cd` (e.g. C:\Users\... -> H:\...) that does not carry over to
# [Environment]::CurrentDirectory -- Windows tracks a separate current directory
# per drive letter. A relative path here would resolve fine through PowerShell
# cmdlets (which honor $PWD) but wrongly through the raw .NET calls below
# ([System.IO.File]::ReadAllLines / WriteAllText), which resolve against
# [Environment]::CurrentDirectory instead.
$AbiDir = Join-Path $RepoRoot "docs/abi"

# --- Locate the MSVC tools ---------------------------------------------------
# Neither is on PATH outside a developer prompt, so glob the VS installs. Any
# toolset version works; the export table does not depend on which one reads it.
function Find-MsvcTool([string]$Name, [string]$OverrideVar) {
    $override = [Environment]::GetEnvironmentVariable($OverrideVar)
    if ($override) { return $override }

    $onPath = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($onPath) { return $onPath.Source }

    foreach ($root in @("$env:ProgramFiles\Microsoft Visual Studio",
                        "${env:ProgramFiles(x86)}\Microsoft Visual Studio")) {
        if (-not (Test-Path $root)) { continue }
        $found = Get-ChildItem -Path $root -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*Hostx64\x64*" } |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    throw "check_exports: cannot find $Name. Set $OverrideVar=/path/to/$Name"
}

$Dumpbin = Find-MsvcTool "dumpbin.exe" "QL_DUMPBIN"
$Undname = Find-MsvcTool "undname.exe" "QL_UNDNAME"

# --- Read one DLL's export table --------------------------------------------
# dumpbin's export rows are "ordinal hint RVA name"; take the name, undecorate,
# then sort. Sorting is what makes the baseline stable: link order is not.
function Get-Exports([string]$Dll) {
    $raw = & $Dumpbin -nologo -exports $Dll
    if ($LASTEXITCODE -ne 0) { throw "check_exports: dumpbin failed on $Dll" }

    $decorated = New-Object System.Collections.Generic.List[string]
    $inTable = $false
    foreach ($line in $raw) {
        $line = $line -replace "`r", ""
        if ($line -match 'ordinal hint RVA      name') { $inTable = $true; continue }
        if ($line -match '^  Summary') { $inTable = $false }
        if (-not $inTable) { continue }
        if ($line -match '^ +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +(\S+)') {
            $decorated.Add($Matches[1])
        }
    }
    if ($decorated.Count -eq 0) {
        throw "check_exports: no exports found in $Dll -- is it built?"
    }

    # undname reads a file of decorated names, one per line. Write LF-only: it
    # does not mind, and keeping every intermediate LF means the baselines this
    # can produce with -Update are byte-identical to the shell version's.
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText($tmp, ($decorated -join "`n") + "`n")
        $undecorated = & $Undname $tmp
        if ($LASTEXITCODE -ne 0) { throw "check_exports: undname failed" }
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }

    $cleaned = New-Object System.Collections.Generic.List[string]
    foreach ($line in $undecorated) {
        $s = ($line -replace "`r", "") -replace ' __ptr64', '' -replace '__cdecl ', ''
        $s = ($s -replace ' {2,}', ' ').Trim()
        if ($s) { $cleaned.Add($s) }
    }

    # Ordinal sort, explicitly. `sort -u` in the shell version compares bytes;
    # PowerShell's Sort-Object and .NET's default string comparison are culture
    # aware, which orders punctuation and case differently and would rewrite the
    # baseline on the first -Update from Windows -- turning a formatting
    # difference into what reads as a wholesale ABI change.
    $unique = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]$cleaned, [System.StringComparer]::Ordinal)
    $result = [string[]]::new($unique.Count)
    $unique.CopyTo($result)
    [Array]::Sort($result, [System.StringComparer]::Ordinal)
    return $result
}

# --- Compare -----------------------------------------------------------------
$status = 0

function Write-Lf([string]$Path, [string[]]$Lines) {
    # LF and no trailing BOM, so the file matches what the shell version writes
    # and git does not see a whole-file change when the two are used in turn.
    [System.IO.File]::WriteAllText($Path, (($Lines -join "`n") + "`n"),
                                   (New-Object System.Text.UTF8Encoding $false))
}

function Test-One([string]$Label, [string]$Dll, [string]$GoldenName) {
    $golden = Join-Path $AbiDir $GoldenName
    if (-not (Test-Path $Dll)) {
        Write-Host "check_exports: $Dll not built, skipping $Label"
        return
    }
    $actual = Get-Exports (Resolve-Path $Dll).Path

    if ($Update) {
        New-Item -ItemType Directory -Force -Path $AbiDir | Out-Null
        Write-Lf $golden $actual
        Write-Host "check_exports: $Label baseline updated ($($actual.Count) symbols)"
        return
    }

    if (-not (Test-Path $golden)) {
        Write-Host "check_exports: no baseline at $golden. Run with -Update to create it." -ForegroundColor Red
        $script:status = 1
        return
    }

    $expected = [System.IO.File]::ReadAllLines($golden) | Where-Object { $_ -ne "" }
    $diff = Compare-Object -ReferenceObject @($expected) -DifferenceObject @($actual)
    if ($diff) {
        Write-Host ""
        Write-Host "==================== $Label export surface changed ====================" -ForegroundColor Red
        Write-Host "  - removed (breaking, SemVer major)   + added (new public promise)"
        Write-Host ""
        foreach ($d in $diff) {
            $mark = if ($d.SideIndicator -eq "<=") { "  - " } else { "  + " }
            Write-Host "$mark$($d.InputObject)"
        }
        Write-Host ""
        Write-Host "  Intended? ./scripts/check_exports.ps1 -Update and commit $golden."
        Write-Host "  Not intended? A new class probably picked up QL_API by habit; see the"
        Write-Host "  note above the macro in include/quantiloom/core/Platform.hpp."
        Write-Host "======================================================================="
        $script:status = 1
    } else {
        Write-Host "check_exports: $Label OK ($($expected.Count) symbols)"
    }
}

Test-One "libQuantiloom" "$BuildDir/src/libQuantiloom/Release/Quantiloom.dll" "exports.golden"
Test-One "libSpectraForge" "$BuildDir/src/libSpectraForge/Release/SpectraForge.dll" "spectraforge-exports.golden"

exit $status
