#!/usr/bin/env bash
# Generate compile_commands.json for clangd.
#
# The primary build/ tree uses the "Visual Studio 18 2026" generator, which does NOT
# honour CMAKE_EXPORT_COMPILE_COMMANDS -- only the Makefile and Ninja generators do.
# So this configures a second, build-only tree (build-cdb/) with Ninja purely to emit
# the database. It never compiles anything; configure takes ~15 s.
#
# cl.exe needs the MSVC developer environment (INCLUDE/LIB/PATH), so the whole
# configure runs inside vcvars64.bat.
#
# `--check` reports whether the existing database is stale and exits non-zero if
# it is, without regenerating. Nothing else notices: clangd silently falls back
# to heuristic flags for a file it cannot find, so staleness has to be asked for.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--check" ]; then
    CDB=build-cdb/compile_commands.json
    if [ ! -f "$CDB" ]; then
        echo "STALE: $CDB does not exist. clangd is guessing flags for the whole repo."
        exit 1
    fi
    python3 - "$CDB" <<'PY'
import json, os, subprocess, sys

cdb = sys.argv[1]
cdb_mtime = os.path.getmtime(cdb)
stale = False

newer = [f for f in subprocess.run(
    ["git", "ls-files", "*CMakeLists.txt", "*.cmake"],
    capture_output=True, text=True).stdout.split()
    if os.path.exists(f) and os.path.getmtime(f) > cdb_mtime]
if newer:
    stale = True
    print(f"STALE: {len(newer)} CMake file(s) modified after the database was generated:")
    for f in newer[:10]:
        print(f"    {f}")

repo = os.path.basename(os.getcwd())
known = {e.get("file", "").replace("\\", "/").rsplit(repo + "/", 1)[-1]
         for e in json.load(open(cdb, encoding="utf-8"))}
tracked = [f for f in subprocess.run(
    ["git", "ls-files", "src/*.cpp", "tests/*.cpp"],
    capture_output=True, text=True).stdout.split() if f not in known]
if tracked:
    stale = True
    print(f"STALE: {len(tracked)} tracked source file(s) absent from the database:")
    for f in tracked[:10]:
        print(f"    {f}")

if not stale:
    print(f"OK: {cdb} covers every tracked source and postdates every CMake file.")
sys.exit(1 if stale else 0)
PY
    exit $?
fi

VCVARS="C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat"
NINJA="C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"

# The repo path is baked into the generated .bat and into build-cdb's CMakeCache,
# so it has to come from where this checkout actually lives -- hard-coding a drive
# breaks the moment the tree is moved. Wipe build-cdb if its cache was written for
# a different source path.
REPO_WIN="$(wslpath -w "$PWD" 2>/dev/null || printf '%s' "$PWD")"

if [ -f build-cdb/CMakeCache.txt ]; then
    cached="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build-cdb/CMakeCache.txt | head -n 1 || true)"
    repo_fwd="${REPO_WIN//\\//}"; repo_fwd="${repo_fwd%/}"
    cached="${cached//\\//}"; cached="${cached%/}"
    if [ -n "$cached" ] && [ "${cached,,}" != "${repo_fwd,,}" ]; then
        echo "Stale build-cdb cache ($cached != $repo_fwd) -- removing." >&2
        rm -rf build-cdb
    fi
fi

cat > _gen_cdb.bat <<EOF
@echo off
call "${VCVARS}" >nul
if errorlevel 1 exit /b 1
cd /d "${REPO_WIN}"
cmake.exe -B build-cdb -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="${NINJA}" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
  -DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd ^
  -DQUANTILOOM_BUILD_TESTS=ON
exit /b %errorlevel%
EOF

trap 'rm -f _gen_cdb.bat' EXIT
cmd.exe /c "_gen_cdb.bat" </dev/null

echo
echo "compile_commands.json entries: $(python3 -c "import json;print(len(json.load(open('build-cdb/compile_commands.json'))))")"
echo "Re-run this after adding source files or changing CMakeLists.txt."
