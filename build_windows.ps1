./src/shaders/compile_shaders.bat
cmake -B build -G "Visual Studio 18 2026" -A x64 -DQUANTILOOM_BUILD_TESTS=OFF -DQUANTILOOM_USE_BC7ENC=OFF -DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd

#cmake --build build --config Debug -j
cmake --build build --config Release -j