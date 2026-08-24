#!/usr/bin/env bash
# Render the furnace cavities and check each against Planck.
#
# An isothermal cavity must return B(T) from every direction and for every
# emissivity -- Kirchhoff, with no free parameters. That makes these the only
# renders in the repo with an answer known in advance, and the only ones that
# can fail on their own rather than by comparison with a previous run.
#
# Worth having as a suite: four of the bugs fixed in the shaders were visible
# here and nowhere else. The pairing is the useful part -- an LWIR cavity and
# an MWIR one differing only in a band that adds a solar term, so a failure on
# one side and not the other says where to look.
#
#   ./scripts/render-tests/run_furnace_suite.sh
#
# Exit codes, which build_wsl.sh distinguishes:
#   0  every cavity within tolerance
#   1  a cavity rendered and was wrong -- a real failure, stop the build
#   2  no CLI built
#   3  no usable GPU, so nothing was measured
#
# 3 is not 1 on purpose. This repo already decided that a missing ray tracing
# device makes a GPU check skip with a reason rather than fail (see
# tests/support/VulkanTestDevice.hpp); a build machine without an RTX card
# should not be told its physics is broken when the truth is that nobody
# looked.
set -uo pipefail
cd "$(dirname "$0")/../.."

CLI="${CLI:-./build/src/app/Release/Quantiloom.exe}"
[ -x "$CLI" ] || { echo "no CLI at $CLI -- build first (build-and-install skill)" >&2; exit 2; }


# The checkers below need numpy and OpenEXR. In WSL those sit on python3; a
# Windows shell usually has no python3 at all (the name resolves to a Store
# stub), so build_windows.ps1 sets PYTHON to an interpreter that has them.
# The default is unchanged, so the WSL path runs exactly as before.
PY="${PYTHON:-python3}"

# The cavity list is data, shared with run_furnace_suite.ps1 -- see the header
# of furnace_cases.txt, which also records what each of the two LWIR-only
# cavities covers that a grey wall cannot. A cavity added here but not there
# would be a gate present on one build path and simply absent on the other.
CASES_FILE="scripts/render-tests/furnace_cases.txt"
[ -f "$CASES_FILE" ] || { echo "no cavity list at $CASES_FILE" >&2; exit 2; }

fail=0
# Read on fd 3, not stdin. The CLI inside this loop reads stdin and would
# swallow the rest of the list: the first version of this loop rendered
# lwir_e1, was handed EOF, and reported "all cavities within tolerance" having
# checked one of eight. The old `for` loop over a fixed word list had no stdin
# to lose, so the hazard arrived with the shared list rather than with the list.
while read -r band case <&3; do
    [ -n "$band" ] || continue
    cfg="assets/configs/furnace_${band}_${case}.toml"
    out="furnace_${band}_${case}_output.exr"

    # Count the success line rather than trusting the exit code: the CLI's
    # is unreliable, and a render that silently wrote nothing would
    # otherwise be checked against a stale file from a previous run.
    log=$("$CLI" "$cfg" 2>&1)
    if [ "$(printf '%s' "$log" | grep -c 'Saved spectral image')" != 1 ]; then
        if printf '%s' "$log" | grep -qE 'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable'; then
            echo "furnace suite: no usable GPU, nothing measured" >&2
            exit 3
        fi
        echo "RENDER FAILED  ${band}_${case}"
        printf '%s\n' "$log" | tail -5 >&2
        fail=1
        continue
    fi

    # Capture first, then branch on the checker's own exit code. Piping it
    # straight into grep would test grep's status instead, and grep
    # succeeds precisely when it has found the word FAIL.
    printf '%-22s ' "${band}_${case}"
    if report=$("$PY" scripts/render-tests/check_furnace.py "$out" "${band^^}"); then
        echo "$report" | grep -E 'Rel error' | tr -d '\n'; echo '  PASS'
    else
        echo "$report" | grep -E 'Rel error|FAIL' | tr '\n' ' '; echo
        fail=1
    fi
done 3< <(grep -vE '^[[:space:]]*(#|$)' "$CASES_FILE")

if [ "$fail" = 0 ]; then
    echo "furnace suite: all cavities within tolerance"
else
    echo "furnace suite: FAILURES above" >&2
fi
exit "$fail"
