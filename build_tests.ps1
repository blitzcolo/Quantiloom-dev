New-Item -ItemType Directory -Force -Path build-test
Set-Location build-test
cmake .. -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON -DUSD_ROOT=C:/openusd && cmake --build . --target libquantiloom_tests --config Release -j && .\tests\Release\libquantiloom_tests.exe
Set-Location ..