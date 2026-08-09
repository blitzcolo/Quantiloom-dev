#!/usr/bin/env bash
# Render the sun-occlusion pairs and check that each band honours the shadow ray.
#
# The scene is one material on one normal under an orthographic top-down camera,
# so sun visibility is the only quantity that can vary across the frame. Each
# band renders twice -- sun above the horizon, sun below it -- and the
# difference is the direct-solar term pixel by pixel. It must be zero behind the
# slab and nonzero elsewhere.
#
# This exists because shadowFactor was traced for every mode and then read by
# only three of them: NIR, SWIR and MWIR let the sun through walls, and no
# furnace cavity could see it (they have no sun). A band that regresses here
# renders a perfectly flat frame.
#
#   ./scripts/render-tests/run_shadow_suite.sh
#
# Exit codes match run_furnace_suite.sh, and build_wsl.sh distinguishes them:
#   0  every band shows a shadow band
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
                echo "shadow suite: no usable GPU, nothing measured" >&2
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

if [ "$fail" = 0 ]; then
    echo "shadow suite: every band occludes the sun"
else
    echo "shadow suite: FAILURES above" >&2
fi
exit "$fail"
