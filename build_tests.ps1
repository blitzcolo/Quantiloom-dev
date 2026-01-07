New-Item -ItemType Directory -Force -Path build-test
Set-Location build-test
cmake .. -DBUILD_TESTING=ON -DENABLE_COVERAGE=ON -DQUANTILOOM_USE_BC7ENC=OFF -DQUANTILOOM_USE_OPENUSD=ON -DUSD_ROOT=C:/openusd && cmake --build . --target libquantiloom_tests --config Release -j && .\tests\Release\libquantiloom_tests.exe
Set-Location ..