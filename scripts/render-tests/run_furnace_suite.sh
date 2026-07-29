#!/usr/bin/env bash
# Render the six furnace cavities and check each against Planck.
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
# Needs a built CLI and an RTX GPU. Not wired into build_wsl.sh: it renders,
# so it costs seconds and a GPU that a build machine may not have.
set -uo pipefail
cd "$(dirname "$0")/../.."

CLI="${CLI:-./build/src/app/Release/Quantiloom.exe}"
[ -x "$CLI" ] || { echo "no CLI at $CLI -- build first (build-and-install skill)" >&2; exit 2; }

fail=0
for band in lwir mwir; do
    for case in e1 e05 rho1; do
        cfg="assets/configs/furnace_${band}_${case}.toml"
        out="furnace_${band}_${case}_output.exr"

        # Count the success line rather than trusting the exit code: the CLI's
        # is unreliable, and a render that silently wrote nothing would
        # otherwise be checked against a stale file from a previous run.
        if [ "$("$CLI" "$cfg" 2>&1 | grep -c 'Saved spectral image')" != 1 ]; then
            echo "RENDER FAILED  ${band}_${case}"
            fail=1
            continue
        fi

        # Capture first, then branch on the checker's own exit code. Piping it
        # straight into grep would test grep's status instead, and grep
        # succeeds precisely when it has found the word FAIL.
        printf '%-22s ' "${band}_${case}"
        if report=$(python3 scripts/render-tests/check_furnace.py "$out" "${band^^}"); then
            echo "$report" | grep -E 'Rel error' | tr -d '\n'; echo '  PASS'
        else
            echo "$report" | grep -E 'Rel error|FAIL' | tr '\n' ' '; echo
            fail=1
        fi
    done
done

if [ "$fail" = 0 ]; then
    echo "furnace suite: all six cavities within tolerance"
else
    echo "furnace suite: FAILURES above" >&2
fi
exit "$fail"
