#!/usr/bin/env bash
# Compare the built DLLs' export tables against the reviewed baselines in
# docs/abi/. Exports are the SDK's contract: adding one is a SemVer minor and
# a promise to keep it working, removing one is a major. Neither should happen
# because somebody wanted a unit test to link.
#
#   ./scripts/check_exports.sh            # verify (what build_wsl.sh runs)
#   ./scripts/check_exports.sh --update   # accept the current tables
#
# Baselines hold undecorated names so the diff is reviewable. That costs a
# dependency on undname.exe, which ships with every MSVC toolset -- the same
# place dumpbin.exe comes from, and this repo already requires that toolchain.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=check
case "${1:-}" in
    --update) MODE=update ;;
    "")       ;;
    *) echo "usage: $0 [--update]" >&2; exit 2 ;;
esac

BUILD_DIR="${BUILD_DIR:-build}"
ABI_DIR="docs/abi"

# --- Locate the MSVC tools ---------------------------------------------------
# Neither is on PATH outside a developer prompt, so glob the VS installs. Any
# toolset version works; the export table does not depend on which one reads it.
find_msvc_tool() {
    local name="$1" override="$2" found
    if [[ -n "${!override:-}" ]]; then printf '%s' "${!override}"; return 0; fi
    found=$(command -v "$name" 2>/dev/null) && { printf '%s' "$found"; return 0; }
    found=$(find "/mnt/c/Program Files/Microsoft Visual Studio" \
                 "/mnt/c/Program Files (x86)/Microsoft Visual Studio" \
                 -name "$name" -path '*Hostx64/x64*' 2>/dev/null | head -1)
    [[ -n "$found" ]] || {
        echo "check_exports: cannot find $name. Set ${override}=/path/to/$name" >&2
        exit 1
    }
    printf '%s' "$found"
}

DUMPBIN=$(find_msvc_tool dumpbin.exe QL_DUMPBIN)
UNDNAME=$(find_msvc_tool undname.exe QL_UNDNAME)

# --- Read one DLL's export table --------------------------------------------
# dumpbin's export rows are "ordinal hint RVA name"; take the name, undecorate,
# then sort. Sorting is what makes the baseline stable: link order is not.
dump_exports() {
    local dll="$1" decorated
    decorated=$(mktemp)
    "$DUMPBIN" /nologo /exports "$(wslpath -w "$dll")" \
        | tr -d '\r' \
        | awk '/ordinal hint RVA      name/{f=1;next} /^  Summary/{f=0} f' \
        | grep -E '^ +[0-9]+ +[0-9A-F]+ +[0-9A-F]+ +\S+' \
        | awk '{print $4}' > "$decorated"
    if [[ ! -s "$decorated" ]]; then
        echo "check_exports: no exports found in $dll -- is it built?" >&2
        rm -f "$decorated"; exit 1
    fi
    "$UNDNAME" "$(wslpath -w "$decorated")" \
        | tr -d '\r' \
        | sed 's/ __ptr64//g; s/__cdecl //g; s/  */ /g; s/^ //; s/ $//' \
        | grep -v '^$' \
        | sort -u
    rm -f "$decorated"
}

# --- Compare -----------------------------------------------------------------
status=0
check_one() {
    local label="$1" dll="$2" golden="$ABI_DIR/$3" actual
    if [[ ! -f "$dll" ]]; then
        echo "check_exports: $dll not built, skipping $label" >&2
        return 0
    fi
    actual=$(mktemp)
    dump_exports "$dll" > "$actual"

    if [[ "$MODE" == update ]]; then
        mkdir -p "$ABI_DIR"
        mv "$actual" "$golden"
        echo "check_exports: $label baseline updated ($(wc -l < "$golden") symbols)"
        return 0
    fi

    if [[ ! -f "$golden" ]]; then
        echo "check_exports: no baseline at $golden. Run with --update to create it." >&2
        rm -f "$actual"; status=1; return 0
    fi

    if ! diff -q "$golden" "$actual" >/dev/null; then
        echo
        echo "==================== $label export surface changed ====================" >&2
        echo "  - removed (breaking, SemVer major)   + added (new public promise)" >&2
        echo >&2
        diff "$golden" "$actual" | grep -E '^[<>]' | sed 's/^</  - /; s/^>/  + /' >&2
        echo >&2
        echo "  Intended? ./scripts/check_exports.sh --update and commit $golden." >&2
        echo "  Not intended? A new class probably picked up QL_API by habit; see the" >&2
        echo "  note above the macro in include/quantiloom/core/Platform.hpp." >&2
        echo "=======================================================================" >&2
        status=1
    else
        echo "check_exports: $label OK ($(wc -l < "$golden") symbols)"
    fi
    rm -f "$actual"
}

check_one "libQuantiloom" "$BUILD_DIR/src/libQuantiloom/Release/Quantiloom.dll"   exports.golden
check_one "libSpectraForge" "$BUILD_DIR/src/libSpectraForge/Release/SpectraForge.dll" spectraforge-exports.golden

exit $status
