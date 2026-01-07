./src/shaders/compile_shaders.bat
cmake -B build -G "Visual Studio 18 2026" -A x64 -DQUANTILOOM_BUILD_TESTS=OFF -DUSD_ROOT=C:/openusd

#cmake --build build --config Debug -j
cmake --build build --config Release -j