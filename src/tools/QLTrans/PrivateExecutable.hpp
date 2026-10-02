#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <sddl.h>
#include <array>
#include <vector>
#include "../../libQuantiloom/dataset/SecureExportTree.hpp"

namespace qltrans::detail {
class Handle {
public:
    HANDLE value = INVALID_HANDLE_VALUE;
    ~Handle() { if (value != INVALID_HANDLE_VALUE && value) CloseHandle(value); }
    Handle() = default;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
class PrivateExecutable {
    quantiloom::dataset::detail::SecureExportTree tree;
    std::filesystem::path directory, executable;
    Handle readLease;
public:
    PrivateExecutable(const void* bytes, DWORD size) {
        try {
            quantiloom::dataset::detail::PrivateDirectorySecurity security;
            auto& attributes = security.Attributes();
            std::vector<wchar_t> temp(32768);
            const DWORD length = GetTempPathW(static_cast<DWORD>(temp.size()), temp.data());
            if (!length || length >= temp.size()) throw std::runtime_error("Cannot find temporary directory");
            auto parent = std::filesystem::absolute(std::filesystem::path(temp.data())).lexically_normal();
            if (parent != parent.root_path() && parent.filename().empty()) parent = parent.parent_path();
            tree.Ensure(parent);
            bool created = false;
            for (int attempt = 0; attempt < 16; ++attempt) {
                std::array<unsigned char, 32> random{};
                if (BCryptGenRandom(nullptr, random.data(), static_cast<ULONG>(random.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
                    throw std::runtime_error("Cannot generate private executable name");
                std::wstring name = L"ql_";
                for (auto value : random) { name += L"0123456789abcdef"[value >> 4]; name += L"0123456789abcdef"[value & 15]; }
                const auto candidate = parent / name;
                if (CreateDirectoryW(quantiloom::dataset::detail::Win32Path(candidate).c_str(), &attributes)) { directory = candidate; created = true; break; }
                if (GetLastError() != ERROR_ALREADY_EXISTS) throw std::runtime_error("Cannot create private executable directory");
            }
            if (!created) throw std::runtime_error("Cannot allocate private executable directory");
            tree.Ensure(directory);
            executable = quantiloom::dataset::detail::Win32Path(directory / L"LutHelper.exe");
            {
                Handle file;
                file.value = CreateFileW(executable.c_str(), GENERIC_WRITE, 0, &attributes,
                    CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
                if (file.value == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot create private executable");
                DWORD offset = 0;
                while (offset < size) {
                    DWORD written = 0;
                    if (!WriteFile(file.value, static_cast<const unsigned char*>(bytes) + offset, size - offset, &written, nullptr) || !written)
                        throw std::runtime_error("Cannot write complete private executable");
                    offset += written;
                }
                if (!FlushFileBuffers(file.value)) throw std::runtime_error("Cannot flush private executable");
            }
            readLease.value = CreateFileW(executable.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            FILE_ATTRIBUTE_TAG_INFO info{};
            if (readLease.value == INVALID_HANDLE_VALUE ||
                !GetFileInformationByHandleEx(readLease.value, FileAttributeTagInfo, &info, sizeof(info)) ||
                (info.FileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)))
                throw std::runtime_error("Cannot lock private executable for launch");
        } catch (...) { Cleanup(); throw; }
    }
    PrivateExecutable(const PrivateExecutable&) = delete;
    PrivateExecutable& operator=(const PrivateExecutable&) = delete;
    ~PrivateExecutable() { Cleanup(); }
    const std::filesystem::path& Path() const { return executable; }
    void Cleanup() noexcept {
        if (readLease.value != INVALID_HANDLE_VALUE) { CloseHandle(readLease.value); readLease.value = INVALID_HANDLE_VALUE; }
        if (!executable.empty()) DeleteFileW(executable.c_str());
        if (!directory.empty()) { try { tree.Remove(directory); } catch (...) {} }
    }
};
} // namespace qltrans::detail
