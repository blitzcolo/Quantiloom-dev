#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "PrivateExecutable.hpp"

#define IDR_MODTRAN_EXE 101

namespace qltrans {

// Get directory containing the current executable
inline std::filesystem::path GetExeDir()
{
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
}

// Create a directory junction from link to target (Windows NTFS)
inline void CreateJunction(const std::filesystem::path& link,
                           const std::filesystem::path& target)
{
    if (std::filesystem::exists(link)) return; // already there
    // mklink /J is the simplest way; no elevation required for junctions
    std::string cmd = "cmd /c mklink /J \"" + link.string() + "\" \"" + target.string() + "\"";
    std::system(cmd.c_str());
}

// Extract embedded MOD4v1r1.EXE resource to %TEMP%, run it in workDir, then delete it.
// Ensures MODTRAN DATA directory is accessible from workDir via junction.
// Throws std::runtime_error on failure.
inline void RunModtran(const std::filesystem::path& workDir)
{
    // Link DATA directory into workDir so MODTRAN can find band model files
    auto dataSource = GetExeDir() / "DATA";
    auto dataLink   = workDir / "DATA";
    if (std::filesystem::is_directory(dataSource))
        CreateJunction(dataLink, dataSource);

    // Locate resource
    HRSRC hRes = FindResourceW(nullptr, MAKEINTRESOURCEW(IDR_MODTRAN_EXE), (LPCWSTR)RT_RCDATA);
    if (!hRes) throw std::runtime_error("FindResource failed: embedded EXE not found");
    HGLOBAL hGlob = LoadResource(nullptr, hRes);
    if (!hGlob) throw std::runtime_error("LoadResource failed");
    DWORD size = SizeofResource(nullptr, hRes);
    const void* data = LockResource(hGlob);
    if (!data || size == 0) throw std::runtime_error("LockResource failed");

    detail::PrivateExecutable extracted(data, size);
    const std::wstring application = extracted.Path().wstring();
    std::wstring cmd = L"\"" + application + L"\"";
    const std::wstring wd = workDir.wstring();
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(application.c_str(), cmd.data(), nullptr, nullptr, FALSE,
                        0, nullptr, wd.c_str(), &si, &pi))
        throw std::runtime_error("CreateProcess failed for LutHelper.exe");
    detail::Handle process, thread;
    process.value = pi.hProcess;
    thread.value = pi.hThread;
    if (WaitForSingleObject(process.value, INFINITE) != WAIT_OBJECT_0)
        throw std::runtime_error("Cannot wait for LutHelper.exe");
    DWORD exitCode = 0;
    if (!GetExitCodeProcess(process.value, &exitCode))
        throw std::runtime_error("Cannot read LutHelper.exe exit code");
    if (exitCode != 0)
        throw std::runtime_error("LutHelper.exe exited with code " + std::to_string(exitCode));
}

} // namespace qltrans
