./src/shaders/compile_shaders.sh
#cmake -B out/Debug -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
#cmake --build out/Debug -j

cmake -B out/Release -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DQUANTILOOM_BUILD_TESTS=OFF -DQUANTILOOM_USE_BC7ENC=OFF
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH cmake --build out/Release -j