./src/shaders/compile_shaders.sh
#cmake -B out/Debug -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
#cmake --build out/Debug -j

cmake -B out/Release -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH cmake --build out/Release -j