cmake -B out -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake -B out -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build out --config Debug --target libQuantiloom -j
cmake --build out --config Release --target libQuantiloom -j
cmake --build out --config Debug --target Quantiloom -j
cmake --build out --config Release --target Quantiloom -j
