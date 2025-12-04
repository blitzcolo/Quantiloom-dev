New-Item -ItemType Directory -Force -Path build
Set-Location build
cmake .. -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON && cmake --build . --target libquantiloom_tests -j && .\tests\Debug\libquantiloom_tests.exe
Set-Location ..