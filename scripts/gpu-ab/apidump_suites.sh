#!/usr/bin/env bash
# Record Vulkan api_dump logs of GPU test suites, one suite per process (P05 evidence tool).
#
#   bash scripts/gpu-ab/apidump_suites.sh <exe> <outdir> <tag> <suite>...
#
#   <exe>     libquantiloom_tests.exe, as a WSL path
#   <outdir>  directory for the logs, as a WSL path on a Windows drive; created
#   <tag>     log-name suffix, e.g. old | new
#   <suite>   either a fixture name, run as --gtest_filter='*<Fixture>.*' (matches
#             PickTest.X before P05 S23 and Gpu/PickTest.X/Vulkan after it), or
#             NAME=FILTER, run as --gtest_filter=FILTER verbatim (design section 9's
#             AD1a/AD1b, whose filters list cases)
#
# Per suite it writes <outdir>/<name>-<tag>.txt (the api_dump log) and
# <outdir>/<name>-<tag>.gtest.txt (the test output), and prints one RESULT line.
# Compare two runs with: python3 scripts/gpu-ab/apidump_diff.py <name>-old.txt <name>-new.txt
#
# One process per suite: gtest registers value-parameterized suites when
# InitGoogleTest runs, after the statically registered ones, so after S23 a filter
# naming several suites could run them in another order and the logs would differ
# for that reason alone (design p05-rt-rhi.md section 6.3). The layer settings are
# p03's (p03-compute-rhi.md 4.0 step 3), handed to the Windows process through
# WSLENV with a Windows path for the log. The tests run from the repo root, as the
# gate runs them. A non-zero test exit is reported and the next suite still runs;
# the script exits 1 if any suite failed or left no log.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <exe> <outdir> <tag> <Fixture | NAME=FILTER>..." >&2
    exit 2
fi

exe="$1"
outdir="$2"
tag="$3"
shift 3

if [[ ! -f "$exe" ]]; then
    echo "no such test executable: $exe" >&2
    exit 2
fi
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "tag must be [A-Za-z0-9_.-]+: $tag" >&2
    exit 2
fi

exe="$(cd "$(dirname "$exe")" && pwd)/$(basename "$exe")"
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
win_outdir="$(wslpath -w "$outdir")"
cd "$(dirname "$0")/../.."

failed=0
for suite in "$@"; do
    if [[ "$suite" == *=* ]]; then
        name="${suite%%=*}"
        filter="${suite#*=}"
    else
        name="$suite"
        filter="*${suite}.*"
    fi
    if [[ ! "$name" =~ ^[A-Za-z0-9_.-]+$ || -z "$filter" ]]; then
        echo "suite must be a fixture name or NAME=FILTER with NAME in [A-Za-z0-9_.-]+: $suite" >&2
        exit 2
    fi

    log="$outdir/$name-$tag.txt"
    gtest_log="$outdir/$name-$tag.gtest.txt"
    rm -f "$log"

    status=0
    VK_LOADER_LAYERS_ENABLE='*api_dump' \
    VK_APIDUMP_OUTPUT_FORMAT=text \
    VK_APIDUMP_DETAILED=true \
    VK_APIDUMP_NO_ADDR=true \
    VK_APIDUMP_LOG_FILENAME="$win_outdir\\$name-$tag.txt" \
    WSLENV="${WSLENV:+$WSLENV:}VK_LOADER_LAYERS_ENABLE:VK_APIDUMP_OUTPUT_FORMAT:VK_APIDUMP_DETAILED:VK_APIDUMP_NO_ADDR:VK_APIDUMP_LOG_FILENAME" \
        "$exe" "--gtest_filter=$filter" </dev/null >"$gtest_log" 2>&1 || status=$?

    if [[ ! -s "$log" ]]; then
        echo "RESULT $name exit=$status filter=$filter log=MISSING"
        failed=1
        continue
    fi
    echo "RESULT $name exit=$status filter=$filter log=$log lines=$(wc -l <"$log" | tr -d ' ')"
    if [[ "$status" -ne 0 ]]; then
        failed=1
    fi
done
exit "$failed"
