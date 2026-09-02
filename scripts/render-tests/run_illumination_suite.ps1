# Check how light reaches a surface: occlusion, and the paths that are not
# straight lines from the sun.
#
# The PowerShell twin of run_illumination_suite.sh, for a Windows build machine
# with no POSIX shell. Same configs, same checkers, same thresholds -- the
# checkers are where every threshold lives, so the only thing duplicated between
# the two runners is the loop. Keep them in step: a check added to one and not
# the other is a gate that exists on one build path and not the other.
#
# The four checks, each failing a different way:
#
#   occlusion   One material on one normal under an orthographic top-down
#               camera, rendered twice with the sun above and below the horizon.
#               The difference is the direct solar term pixel by pixel, and it
#               must be zero behind the slab. shadowFactor was traced for every
#               mode and read by only three of them, so NIR, SWIR and MWIR let
#               the sun through walls; a band that regresses renders a perfectly
#               flat frame.
#   open sky    The same ground with nothing above it, where the traced
#               environment bounce must contribute exactly zero and the render
#               must equal a closed form. Catches a bounce added on top of the
#               analytic sky term rather than as a correction to it, which
#               doubles the ambient of every open scene and passes every
#               occlusion check while doing so.
#   indirect    A Cornell box whose only emitter is geometry, so every lit pixel
#               outside the light panel got there by bouncing, and the side
#               walls tint what they reflect. Catches a bounce that never fires
#               and a bounce that comes back colourless.
#   view        The same surface seen from 60 degrees off its normal as well
#               as down it. A hemispherical reflectance and emissivity do not
#               depend on where the camera stands, so both views must return one
#               closed form. Catches a view-dependent albedo -- which every other
#               check here is blind to, because they all point the camera down
#               the surface normal, where such a law is the identity.
#   mis         The same box rendered with light sampling on and off, which
#               estimate the same integral and so must agree. Catches emitted
#               radiance counted by both strategies at full weight -- which the
#               indirect check cannot see, since a box at twice its brightness
#               is still a lit box. Run for both emission representations, the
#               RGB emissiveFactor and a bound emissive_curve, because they are
#               separate paths in the shader that meet only at the MIS weight.
#
#   ./scripts/render-tests/run_illumination_suite.ps1
#
# Exit codes match run_furnace_suite.ps1, and build_windows.ps1 distinguishes
# them: 0 all passed, 1 a real failure, 2 no CLI, 3 no usable GPU.

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
$GpuAbsent = 'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable'
$script:fail = 0

# Render one config; $null on success, otherwise the suite should move on.
# Exits 3 outright when the machine has no ray tracing device, which is a skip
# rather than a failure.
function Invoke-Render([string]$Config, [string]$Label) {
    $log = (& $Cli $Config 2>&1 | Out-String)
    if (([regex]::Matches($log, 'Saved spectral image')).Count -ne 1) {
        if ($log -match $GpuAbsent) {
            Write-Stderr "illumination suite: no usable GPU, nothing measured"
            exit 3
        }
        Write-Host "RENDER FAILED  $Label"
        Write-Host (($log -split "`n" | Select-Object -Last 5) -join "`n")
        $script:fail = 1
        return $false
    }
    return $true
}

# Run a checker and print its verdict line, mirroring the shell version's
# capture-then-branch: the checker's own exit code decides, not whether a
# pattern matched.
function Invoke-Checker([string]$Label, [string[]]$Command, [string]$PassPattern,
                        [string]$FailPattern) {
    Write-Host ("{0,-8} " -f $Label) -NoNewline
    $report = (& $Py @Command 2>&1 | Out-String)
    if ($LASTEXITCODE -eq 0) {
        Write-Host (((($report -split "`n") | Where-Object { $_ -match $PassPattern }) -join '').Trim()) -NoNewline
        Write-Host "  PASS"
    } else {
        if ($LASTEXITCODE -eq 3) {
            Write-Stderr "illumination suite: no usable GPU, nothing measured"
            exit 3
        }
        Write-Host (((($report -split "`n") | Where-Object { $_ -match $FailPattern }) | ForEach-Object { $_.Trim() }) -join ' ')
        $script:fail = 1
    }
}

foreach ($band in @("nir", "swir", "mwir")) {
    $rendered = $true
    foreach ($variant in @("sun", "nosun")) {
        if (-not (Invoke-Render "assets/configs/shadow_${band}_${variant}.toml" "${band}_${variant}")) {
            $rendered = $false
            break
        }
    }
    if (-not $rendered) { continue }
    Invoke-Checker $band @("scripts/render-tests/check_shadow.py",
                           "shadow_${band}_sun_output.exr",
                           "shadow_${band}_nosun_output.exr") 'Shadowed fraction' 'Shadowed fraction|FAIL'
}

# The complement of the shadow checks: the same ground with nothing above it,
# where the traced environment bounce must contribute exactly zero and the
# render must equal the closed-form Lambertian answer. A bounce added on top of
# the analytic sky term rather than as a correction to it doubles the ambient
# in every open scene, and passes every occlusion check above while doing so.
if (Invoke-Render "assets/configs/skyequiv_swir.toml" "skyequiv_swir") {
    Invoke-Checker "open" @("scripts/render-tests/check_sky_equiv.py",
                            "skyequiv_swir_output.exr") 'Rel error' 'Rel error|FAIL'
}

# The same open ground in the visible band, where the reflectance is an
# upsampled spectrum rather than 1 - emissivity and so has no closed form to
# compare against. The two visible modes have each other instead: vis_fused
# renders this scene with no variance at all, and vis_hero must land on its
# mean. A residual that is not identically zero shows up in the first as a
# frame that is no longer constant, and a bounce added on top of the analytic
# sky rather than correcting it shows up in the second as a factor near two.
$visOpen = $true
foreach ($m in @("fused", "hero")) {
    if (-not (Invoke-Render "assets/configs/skyequiv_vis_${m}.toml" "skyequiv_vis_${m}")) {
        $visOpen = $false
    }
}
if ($visOpen) {
    Invoke-Checker "open:vis" @("scripts/render-tests/check_sky_equiv.py",
                                "skyequiv_vis_hero_output.exr",
                                "--reference", "skyequiv_vis_fused_output.exr",
                                "--reference-max-spread", "0.0") 'Rel error' 'Rel error|FAIL'
}

# Indirect light in the visible band. The only emitter is the ceiling panel and
# the camera cannot see it, so every lit pixel here arrived by bouncing. This
# asserts that indirect light EXISTS and carries the wall's colour; whether
# there is the right AMOUNT of it is the next check's job.
if (Invoke-Render "assets/configs/cornell_box_vis_bleed.toml" "cornell_box_vis_bleed") {
    Invoke-Checker "bleed" @("scripts/render-tests/check_color_bleed.py",
                             "cornell_box_vis_bleed_output.exr") 'Lit fraction' 'Lit fraction|R/G|FAIL'
}

# How much indirect light, rather than whether any. The panel's radiance reaches
# a surface two ways -- a BSDF-sampled bounce that lands on it, and explicit
# light sampling from the previous vertex -- and the MIS weights must split the
# credit, not duplicate it.
#
# Nothing above can see this. The furnace cavities and the open-sky check are
# both scenes where light sampling never fires, and the bleed check asserts that
# bounced light exists, not that there is the right amount of it. When the MIS
# weight was being applied to the RGB triple, which a curve-bound emitter never
# reads, every one of the checks above passed while everything the lamp lit
# rendered at 2x. A second renderer found it; this is what would have.
#
# This one renders for itself, so there is no Invoke-Render above it.
$MisSpp = if ($env:MIS_SPP) { $env:MIS_SPP } else { "2048" }
$MisRes = if ($env:MIS_RES) { $env:MIS_RES } else { "192" }
Invoke-Checker "mis" @("scripts/render-tests/check_nee_mis.py",
                       "--spp", $MisSpp, "--resolution", $MisRes) 'worst region' 'worst region|worst for|FAIL'


# The two visible estimators against each other. vis_fused sweeps 32 fixed
# wavelengths along one path; vis_hero draws one and rotates it into a quartet.
# They estimate the same integral, so on a scene with no dispersion the second
# must converge to the first, and the error must fall like noise rather than
# settle on a floor. The predecessor of this estimator failed exactly there.
#
# This one renders the prism scene too, which means it edits
# assets/models/prism_dispersion.gltf in place and restores it afterwards; it
# cannot run beside anything else reading that model. It renders for itself, so
# there is no Invoke-Render above it.
Invoke-Checker "hero" @("scripts/render-tests/check_hero_wavelength.py") 'error fell .* over' 'FAIL|error fell'


# Whether the answer depends on where the camera stands. Every check above --
# and every furnace cavity -- views its surface down the surface normal, so
# NdotV is 1 at every pixel and an emissivity law of the form eps0*f(cos theta)
# with f(1)=1 is the identity in all of them. It was not the identity anywhere
# else: it turned rho 0.700 into 0.815 at 60 degrees and reported a 300 K desert
# at 324 K in MWIR, with the scene's thermal ordering inverted.
#
# Two regimes, because they fail differently: a reflectance under a sun and sky,
# and an emissivity against a COLD sky. The cold sky is what makes the second
# observable at all -- with the sky at the surface's own temperature,
# eps*B + rho*B = B for any split, which is exactly why the furnace cavities
# cannot see an emissivity error.
if (Invoke-Render "assets/configs/viewangle_swir_oblique.toml" "viewangle_swir_oblique") {
    Invoke-Checker "view:refl" @("scripts/render-tests/check_view_independence.py",
                                 "skyequiv_swir_output.exr",
                                 "viewangle_swir_oblique_output.exr",
                                 "--mode", "reflective") 'Rel error' 'Rel error|FAIL'
}

$thermalOk = $true
foreach ($cfg in @("viewangle_lwir_nadir", "viewangle_lwir_oblique")) {
    if (-not (Invoke-Render "assets/configs/$cfg.toml" $cfg)) { $thermalOk = $false; break }
}
if ($thermalOk) {
    Invoke-Checker "view:therm" @("scripts/render-tests/check_view_independence.py",
                                  "viewangle_lwir_nadir_output.exr",
                                  "viewangle_lwir_oblique_output.exr",
                                  "--mode", "thermal") 'Rel error' 'Rel error|FAIL'
}

if ($script:fail -eq 0) {
    Write-Host "illumination suite: occlusion, open sky in two bands, indirect, MIS, hero convergence and view independence all within tolerance"
} else {
    Write-Host "illumination suite: FAILURES above" -ForegroundColor Red
}
exit $script:fail
