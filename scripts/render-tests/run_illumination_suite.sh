#!/usr/bin/env bash
# Check how light reaches a surface: occlusion, and the paths that are not
# straight lines from the sun.
#
# The furnace cavities cover what a surface does with light once it arrives, and
# they are blind to all of this -- they have no sun, no sky and no scene outside
# the cavity walls. Three checks here, each failing a different way:
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
#               depend on where the camera stands, so both views must return
#               one closed form. Catches a view-dependent albedo -- which every
#               other check here is blind to, because they all point the camera
#               down the surface normal, where such a law is the identity.
#   open:vis    The same open ground in the visible band, rendered by both
#               visible modes. The deterministic one must come out with no
#               variance at all -- one material, one illumination, a residual
#               that is zero per wavelength -- and the sampled one must land on
#               its mean. The SWIR arm above has a closed form to check against;
#               this one has the other estimator, which is what makes it a
#               switching check rather than a second copy of the same sum.
#   hero        The two visible estimators against each other on a full frame,
#               and n(lambda) against a floor measured from the renderer rather
#               than assumed. Catches an estimator whose error stops falling
#               with sample count, which is what every wrong weight, missing
#               density or misplaced matching function looks like.
#   fluor       A dye that absorbs below 500 nm and emits above 550, under two
#               illuminants of equal power over the band. Its contribution must
#               rise as the illuminant's power in the band it emits in falls,
#               which no transport that is diagonal in wavelength can produce,
#               and a total absorber must give back exactly what it took.
#   mis         The same box rendered with light sampling on and off, which
#               estimate the same integral and so must agree. Catches emitted
#               radiance counted by both strategies at full weight -- which the
#               indirect check cannot see, since a box at twice its brightness
#               is still a lit box. Run for both emission representations, the
#               RGB emissiveFactor and a bound emissive_curve, because they are
#               separate paths in the shader that meet only at the MIS weight.
#
#   ./scripts/render-tests/run_illumination_suite.sh
#
# Exit codes match run_furnace_suite.sh, and build_wsl.sh distinguishes them:
#   0  every check passed
#   1  a band rendered and was wrong -- a real failure, stop the build
#   2  no CLI built
#   3  no usable GPU, so nothing was measured
set -uo pipefail
cd "$(dirname "$0")/../.."

CLI="${CLI:-./build/src/app/Release/Quantiloom.exe}"
[ -x "$CLI" ] || { echo "no CLI at $CLI -- build first (build-and-install skill)" >&2; exit 2; }


# The checkers below need numpy and OpenEXR. In WSL those sit on python3; a
# Windows shell usually has no python3 at all (the name resolves to a Store
# stub), so build_windows.ps1 sets PYTHON to an interpreter that has them.
# The default is unchanged, so the WSL path runs exactly as before.
PY="${PYTHON:-python3}"

fail=0
for band in nir swir mwir; do
    for variant in sun nosun; do
        cfg="assets/configs/shadow_${band}_${variant}.toml"

        # Count the success line rather than trusting the exit code, as the
        # furnace suite does: a render that silently wrote nothing would
        # otherwise be checked against a stale file from a previous run.
        log=$("$CLI" "$cfg" 2>&1)
        if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
            if printf '%s' "$log" | grep -qE 'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable'; then
                echo "illumination suite: no usable GPU, nothing measured" >&2
                exit 3
            fi
            echo "RENDER FAILED  ${band}_${variant}"
            printf '%s\n' "$log" | tail -5 >&2
            fail=1
            continue 2
        fi
    done

    printf '%-8s ' "$band"
    if report=$("$PY" scripts/render-tests/check_shadow.py \
                    "shadow_${band}_sun_output.exr" \
                    "shadow_${band}_nosun_output.exr"); then
        echo "$report" | grep -E 'Shadowed fraction' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Shadowed fraction|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
done

# The complement of the shadow checks: the same ground with nothing above it,
# where the traced environment bounce must contribute exactly zero and the
# render must equal the closed-form Lambertian answer. A bounce added on top of
# the analytic sky term rather than as a correction to it doubles the ambient
# in every open scene, and passes every occlusion check above while doing so.
log=$("$CLI" assets/configs/skyequiv_swir.toml 2>&1)
if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
    echo "RENDER FAILED  skyequiv_swir"
    printf '%s\n' "$log" | tail -5 >&2
    fail=1
else
    printf '%-8s ' "open"
    if report=$("$PY" scripts/render-tests/check_sky_equiv.py skyequiv_swir_output.exr); then
        echo "$report" | grep -E 'Rel error' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Rel error|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
fi

# The same open ground in the visible band, where the reflectance is an
# upsampled spectrum rather than 1 - emissivity and so has no closed form to
# compare against. The two visible modes have each other instead: vis_fused
# renders this scene with no variance at all, and vis_hero must land on its
# mean. A residual that is not identically zero shows up in the first as a
# frame that is no longer constant, and a bounce added on top of the analytic
# sky rather than correcting it shows up in the second as a factor near two.
for m in fused hero; do
    log=$("$CLI" "assets/configs/skyequiv_vis_${m}.toml" 2>&1)
    if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
        echo "RENDER FAILED  skyequiv_vis_${m}"
        printf '%s\n' "$log" | tail -5 >&2
        fail=1
    fi
done
printf '%-8s ' "open:vis"
if report=$("$PY" scripts/render-tests/check_sky_equiv.py \
                skyequiv_vis_hero_output.exr \
                --reference skyequiv_vis_fused_output.exr \
                --reference-max-spread 0.0); then
    echo "$report" | grep -E 'Rel error' | tr -d '\n'; echo '  PASS'
else
    echo "$report" | grep -E 'Rel error|FAIL' | tr '\n' ' '; echo
    fail=1
fi

# Indirect light in the visible band. The only emitter is the ceiling panel and
# the camera cannot see it, so every lit pixel here arrived by bouncing. This
# asserts that indirect light EXISTS and carries the wall's colour; whether
# there is the right AMOUNT of it is the next check's job.
log=$("$CLI" assets/configs/cornell_box_vis_bleed.toml 2>&1)
if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
    echo "RENDER FAILED  cornell_box_vis_bleed"
    printf '%s\n' "$log" | tail -5 >&2
    fail=1
else
    printf '%-8s ' "bleed"
    if report=$("$PY" scripts/render-tests/check_color_bleed.py cornell_box_vis_bleed_output.exr); then
        echo "$report" | grep -E 'Lit fraction' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Lit fraction|R/G|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
fi


# How much indirect light, rather than whether any. The panel's radiance reaches
# a surface two ways -- a BSDF-sampled bounce that lands on it, and explicit
# light sampling from the previous vertex -- and the MIS weights must split the
# credit, not duplicate it. Rendering the same scene with light sampling on and
# off estimates the same integral twice, so the two must agree.
#
# Nothing above can see this. The furnace cavities and the open-sky check are
# both scenes where light sampling never fires, and the bleed check asserts that
# bounced light exists, not that there is the right amount of it -- a Cornell box
# at twice its correct brightness still looks like a Cornell box. When the MIS
# weight was being applied to the RGB triple, which a curve-bound emitter never
# reads, every one of the checks above passed while everything the lamp lit
# rendered at 2x. A second renderer found it; this is what would have.
#
# Runs both emission representations, because they are separate shader paths.
printf '%-8s ' "mis"
if report=$("$PY" scripts/render-tests/check_nee_mis.py \
                --spp "${MIS_SPP:-2048}" --resolution "${MIS_RES:-192}"); then
    echo "$report" | grep -E 'worst region' | tr -d '\n'; echo '  PASS'
else
    mis_status=$?
    if [ "$mis_status" = 3 ]; then
        echo "illumination suite: no usable GPU, nothing measured" >&2
        exit 3
    fi
    echo "$report" | grep -E 'worst region|FAIL|worst for' | tr '\n' ' '; echo
    fail=1
fi


# The two visible estimators against each other. vis_fused sweeps 32 fixed
# wavelengths along one path; vis_hero draws one and rotates it into a quartet.
# They estimate the same integral, so on a scene with no dispersion the second
# must converge to the first, and the error must fall like noise rather than
# settle on a floor. The predecessor of this estimator failed exactly there.
#
# This one renders the prism scene too, which means it edits
# assets/models/prism_dispersion.gltf in place and restores it afterwards; it
# cannot run beside anything else reading that model.
printf '%-8s ' "hero"
if report=$("$PY" scripts/render-tests/check_hero_wavelength.py); then
    echo "$report" | grep -E 'error fell .* over' | head -1 | sed 's/^[[:space:]]*//' \
        | tr -d '\n'; echo '  PASS'
else
    hero_status=$?
    if [ "$hero_status" = 3 ]; then
        echo "illumination suite: no usable GPU, nothing measured" >&2
        exit 3
    fi
    echo "$report" | grep -E 'FAIL|error fell' | tr '\n' ' '; echo
    fail=1
fi


# Light that leaves at a wavelength it did not arrive at. Every other check in
# this suite would pass on a renderer whose transport is diagonal in wavelength,
# because every other term is; this one puts two illuminants of equal power over
# the band in front of a dye that absorbs below 500 nm and emits above 550, and
# asks the emission band to get BRIGHTER as the illuminant's own power there
# falls. Nothing diagonal can do that. It also holds a total absorber to a
# closed form -- what it takes in is what it gives back -- which is how the
# emission normalisation was caught reading the band one grid step too wide.
printf '%-8s ' "fluor"
if report=$("$PY" scripts/render-tests/check_fluorescence.py); then
    echo "$report" | grep -E "dye's contribution " | sed 's/^[[:space:]]*//' \
        | tr -s ' ' | tr -d '\n'; echo '  PASS'
else
    fluor_status=$?
    if [ "$fluor_status" = 3 ]; then
        echo "illumination suite: no usable GPU, nothing measured" >&2
        exit 3
    fi
    echo "$report" | grep -E "FAIL|dye's contribution" | tr '\n' ' '; echo
    fail=1
fi


# Whether the answer depends on where the camera stands. Every check above --
# and every furnace cavity -- views its surface down the surface normal, so
# NdotV is 1 at every pixel and an emissivity law of the form eps0*f(cos theta)
# with f(1)=1 is the identity in all of them. It was not the identity anywhere
# else: it turned rho 0.700 into 0.815 at 60 degrees and reported a 300 K
# desert at 324 K in MWIR, with the scene's thermal ordering inverted.
#
# Two regimes, because they fail differently: a reflectance under a sun and sky,
# and an emissivity against a COLD sky. The cold sky is what makes the second
# one observable at all -- with the sky at the surface's own temperature,
# eps*B + rho*B = B for any split, which is exactly why the furnace cavities
# cannot see an emissivity error.
for pair in "reflective:skyequiv_swir_output.exr:viewangle_swir_oblique_output.exr"             "thermal:viewangle_lwir_nadir_output.exr:viewangle_lwir_oblique_output.exr"; do
    mode="${pair%%:*}"; rest="${pair#*:}"; a="${rest%%:*}"; b="${rest#*:}"

    if [ "$mode" = thermal ]; then
        for cfg in viewangle_lwir_nadir viewangle_lwir_oblique; do
            log=$("$CLI" "assets/configs/${cfg}.toml" 2>&1)
            if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
                if printf '%s' "$log" | grep -qE 'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable'; then
                    echo "illumination suite: no usable GPU, nothing measured" >&2
                    exit 3
                fi
                echo "RENDER FAILED  $cfg"; printf '%s
' "$log" | tail -5 >&2; fail=1; continue 2
            fi
        done
    else
        log=$("$CLI" assets/configs/viewangle_swir_oblique.toml 2>&1)
        if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
            echo "RENDER FAILED  viewangle_swir_oblique"; printf '%s
' "$log" | tail -5 >&2; fail=1; continue
        fi
    fi

    printf '%-8s ' "view:$mode"
    if report=$("$PY" scripts/render-tests/check_view_independence.py "$a" "$b" --mode "$mode"); then
        echo "$report" | grep -E 'Rel error' | tr -d '
'; echo '  PASS'
    else
        echo "$report" | grep -E 'Rel error|FAIL' | tr '
' ' '; echo
        fail=1
    fi
done

if [ "$fail" = 0 ]; then
    echo "illumination suite: occlusion, open sky in two bands, indirect, MIS, hero convergence, fluorescence and view independence all within tolerance"
else
    echo "illumination suite: FAILURES above" >&2
fi
exit "$fail"
