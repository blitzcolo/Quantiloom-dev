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
    if report=$(python3 scripts/render-tests/check_shadow.py \
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
    if report=$(python3 scripts/render-tests/check_sky_equiv.py skyequiv_swir_output.exr); then
        echo "$report" | grep -E 'Rel error' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Rel error|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
fi

# Indirect light in the visible band. Only emitter is the ceiling panel, and
# nothing does next-event estimation for emissive geometry, so a lit frame here
# is bounced light by construction.
log=$("$CLI" assets/configs/cornell_box_vis_bleed.toml 2>&1)
if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
    echo "RENDER FAILED  cornell_box_vis_bleed"
    printf '%s\n' "$log" | tail -5 >&2
    fail=1
else
    printf '%-8s ' "bleed"
    if report=$(python3 scripts/render-tests/check_color_bleed.py cornell_box_vis_bleed_output.exr); then
        echo "$report" | grep -E 'Lit fraction' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Lit fraction|R/G|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
fi

if [ "$fail" = 0 ]; then
    echo "illumination suite: occlusion, open sky and indirect all within tolerance"
else
    echo "illumination suite: FAILURES above" >&2
fi
exit "$fail"
