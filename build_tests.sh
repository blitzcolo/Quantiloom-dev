cd out
cmake .. -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH cmake --build . --target libquantiloom_tests -j && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./tests/libquantiloom_tests
cd ..