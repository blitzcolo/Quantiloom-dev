Set-Location .\build
Remove-Item -Recurse -Force D:\Quantiloom-SDK\windows_amd64
cmake --install . --prefix D:\Quantiloom-SDK\windows_amd64 --config Release
Set-Location ..