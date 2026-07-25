#!/usr/bin/env bash
# PostToolUse hook: recompile shaders after an HLSL edit.
#
# Editing a shader without recompiling leaves the old .spv in place, so a later
# render silently "verifies" the previous code (see the render-verify skill).
# The compile takes ~1 s, so just always do it.
#
# Silent on success. On failure, feeds the compiler output back to the model.
# Uses python3, not jq -- jq is not installed here.
set -uo pipefail
cd "$(dirname "$0")/../.."

f=$(python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d={}
print(d.get("tool_input",{}).get("file_path") or d.get("tool_response",{}).get("filePath") or "")' 2>/dev/null)
[ -n "$f" ] || exit 0

case "$f" in
  */src/shaders/*.hlsl|*/src/shaders/*.hlsli|*/src/shaders/*.rgen|\
  */src/shaders/*.rchit|*/src/shaders/*.rmiss|*/src/shaders/*.comp) ;;
  *) exit 0 ;;
esac

if ! out=$(cmd.exe /c "src\\shaders\\compile_shaders.bat" </dev/null 2>&1); then
  printf '%s' "$out" | tr -d '\r' | tail -20 | python3 -c 'import json,sys
msg=(f"Shader compile FAILED after editing {sys.argv[1]}. The old .spv files are still "
     "in place, so rendering now would silently verify the previous shader code. "
     "Fix the HLSL and re-run:\n"
     "  cmd.exe /c \"src\\shaders\\compile_shaders.bat\" </dev/null\n\n" + sys.stdin.read())
print(json.dumps({"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":msg}}))' "$f"
fi
exit 0
