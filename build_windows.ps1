$ErrorActionPreference = "Stop"
$SourceDir = $PSScriptRoot
$BuildDir  = Join-Path $SourceDir "build"

# --- Stale build directory guard ---
# CMakeCache.txt hard-codes the absolute source path. If the project was moved
# or copied to another drive/folder, the old cache breaks the configure step
# (typically: Unknown CMake command "CPMAddPackage"). Wipe it and start clean.
$CacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $CacheFile) {
    $homeLine = Select-String -Path $CacheFile -Pattern "^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$" | Select-Object -First 1
    if ($homeLine) {
        $cached = $homeLine.Matches[0].Groups[1].Value.Replace("/", "\").TrimEnd("\")
        $actual = $SourceDir.Replace("/", "\").TrimEnd("\")
        if ($cached -ne $actual) {
            Write-Host "Stale build cache: `"$cached`" != `"$actual`" - removing $BuildDir" -ForegroundColor Yellow
            Remove-Item -LiteralPath $BuildDir -Recurse -Force
        }
    }
}

./src/shaders/compile_shaders.bat
cmake -B build -G "Visual Studio 18 2026" -A x64 -DQUANTILOOM_BUILD_TESTS=OFF -DQUANTILOOM_USE_BC7ENC=OFF -DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd

#cmake --build build --config Debug -j
cmake --build build --config Release -j