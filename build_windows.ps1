cmake -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_GENERATOR_PLATFORM=x64
cmake -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR_PLATFORM=x64 
cmake --build build --config Debug --target ALL_BUILD -j
cmake --build build --config Release --target ALL_BUILD -j