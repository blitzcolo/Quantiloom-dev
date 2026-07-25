#!/usr/bin/env bash
# PostToolUse hook: warn when a new test file is not registered in CMake.
#
# tests/CMakeLists.txt lists test sources explicitly -- there is no glob. A file
# that is not in the matching set(TEST_<MODULE>_SOURCES ...) block is never
# compiled and never run, and the suite still reports green.
#
# Silent when the file is registered. Uses python3, not jq -- jq is not installed
# here, and a hook that silently no-ops is worse than no hook.
set -uo pipefail
cd "$(dirname "$0")/../.."

f=$(python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d={}
print(d.get("tool_input",{}).get("file_path") or d.get("tool_response",{}).get("filePath") or "")' 2>/dev/null)
[ -n "$f" ] || exit 0

case "$f" in */tests/*/test_*.cpp) ;; *) exit 0 ;; esac

base=$(basename "$f")
dir=$(basename "$(dirname "$f")")
grep -qF "$dir/$base" tests/CMakeLists.txt && exit 0

block="TEST_$(printf '%s' "${dir#test_}" | tr '[:lower:]' '[:upper:]')_SOURCES"
python3 -c 'import json,sys
msg=(f"{sys.argv[1]} is not registered in tests/CMakeLists.txt. Test sources are an "
     "explicit list, not a glob, so this file will never be compiled and never run "
     "-- the suite will still report green. Add it to the "
     f"set({sys.argv[2]} ...) block.")
print(json.dumps({"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":msg}}))' \
  "$dir/$base" "$block"
exit 0
