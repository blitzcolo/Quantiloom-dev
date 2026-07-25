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
set -euo pipefail
cd "$(dirname "$0")/.."

VCVARS="C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat"
NINJA="C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"

cat > _gen_cdb.bat <<EOF
@echo off
call "${VCVARS}" >nul
if errorlevel 1 exit /b 1
cd /d D:\\Quantiloom-dev
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
