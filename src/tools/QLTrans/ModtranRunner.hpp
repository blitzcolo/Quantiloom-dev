#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>

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

    // Build temp path: %TEMP%\ql_<pid>_LutHelper.exe
    wchar_t tmpDir[MAX_PATH];
    GetTempPathW(MAX_PATH, tmpDir);
    std::wstring tmpPath = std::wstring(tmpDir) + L"ql_" +
                           std::to_wstring(GetCurrentProcessId()) + L"_LutHelper.exe";

    // Write to temp file
    HANDLE hFile = CreateFileW(tmpPath.c_str(), GENERIC_WRITE, 0, nullptr,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot create temp EXE");
    DWORD written = 0;
    WriteFile(hFile, data, size, &written, nullptr);
    CloseHandle(hFile);

    // Run via CreateProcess in workDir
    std::wstring cmd = L"\"" + tmpPath + L"\"";
    std::wstring wd  = workDir.wstring();
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                             0, nullptr, wd.c_str(), &si, &pi);
    if (!ok) {
        DeleteFileW(tmpPath.c_str());
        throw std::runtime_error("CreateProcess failed for LutHelper.exe");
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    // Cleanup temp EXE
    if (!DeleteFileW(tmpPath.c_str()))
        MoveFileExW(tmpPath.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);

    if (exitCode != 0)
        throw std::runtime_error("LutHelper.exe exited with code " + std::to_string(exitCode));
}

} // namespace qltrans
