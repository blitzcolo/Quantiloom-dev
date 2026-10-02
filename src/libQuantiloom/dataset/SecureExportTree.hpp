#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <cerrno>
#include <cstdio>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "PrivateDirectorySecurity.hpp"
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace quantiloom::dataset::detail {
namespace fs = std::filesystem;

#ifdef _WIN32
// The claim + random staging name can exceed Win32's legacy directory limit
// even when the user's output filename itself is short. Use the extended form
// for every native API and writer path, including UNC shares.
inline std::wstring Win32Path(const fs::path& path) {
    auto absolute = fs::absolute(path).lexically_normal();
    absolute.make_preferred();
    const auto value = absolute.wstring();
    if (value.starts_with(L"\\\\?\\")) return value;
    if (value.starts_with(L"\\\\")) return L"\\\\?\\UNC\\" + value.substr(2);
    return L"\\\\?\\" + value;
}
#endif

// Directory capabilities live for the whole transaction. Windows denies rename
// and deletion of every ancestor. Public directories initially deny write sharing
// too, until a pinned child keeps them nonempty; private directories instead use
// an owner-only ACL and matching integrity label. POSIX addresses children by fd.
// POSIX cannot forbid a same-user directory rename. Identity checks reject a
// detected rename; the descriptors keep concurrent operations on the original
// directories instead of following a replacement link into a foreign tree.
class SecureExportTree {
    struct Directory {
#ifdef _WIN32
        HANDLE handle = INVALID_HANDLE_VALUE;
        bool writeShared = false;
        bool privateDirectory = false;
        ~Directory() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
        int handle = -1;
        ~Directory() { if (handle >= 0) ::close(handle); }
#endif
        fs::path path;
    };
    std::map<fs::path, std::unique_ptr<Directory>> directories;
    static constexpr const char* suffix = ".quantiloom-export.lock";

#ifdef _WIN32
    static void ShareWrites(Directory& directory) {
        if (directory.writeShared) return;
        const auto next = CreateFileW(Win32Path(directory.path).c_str(), FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (next == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot retain export directory protection");
        BY_HANDLE_FILE_INFORMATION before{}, after{};
        const bool same = GetFileInformationByHandle(directory.handle, &before) && GetFileInformationByHandle(next, &after) &&
            before.dwVolumeSerialNumber == after.dwVolumeSerialNumber && before.nFileIndexHigh == after.nFileIndexHigh &&
            before.nFileIndexLow == after.nFileIndexLow && !(after.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT);
        if (!same) { CloseHandle(next); throw std::runtime_error("export directory identity changed"); }
        // Both leases deny delete: replacement is impossible throughout.
        CloseHandle(directory.handle);
        directory.handle = next;
        directory.writeShared = true;
    }
#endif
    Directory& Open(const fs::path& path, bool create, bool checkClaims, [[maybe_unused]] bool privateDirectory = false) {
        if (auto it = directories.find(path); it != directories.end()) {
#ifndef _WIN32
            if (path != path.root_path()) {
                [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
                struct stat opened{}, named{};
                if (::fstat(it->second->handle, &opened) != 0 ||
                    ::fstatat(parent.handle, path.filename().c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
                    !S_ISDIR(named.st_mode) || opened.st_dev != named.st_dev || opened.st_ino != named.st_ino)
                    throw std::runtime_error("export directory identity changed: " + path.string());
            }
#endif
            return *it->second;
        }
        Directory* parent = nullptr;
        if (path != path.root_path()) parent = &Open(path.parent_path(), create, checkClaims);
        if (parent && checkClaims) {
            auto claim = path.filename(); claim += suffix;
            if (ExistsIn(*parent, claim) || ExistsIn(*parent, suffix))
                throw std::runtime_error("output ancestor is locked: " + path.string());
        }
        auto dir = std::make_unique<Directory>();
        dir->path = path;
#ifdef _WIN32
        dir->privateDirectory = privateDirectory || (parent && parent->privateDirectory);
        dir->writeShared = dir->privateDirectory;
        if (create && parent && !CreateDirectoryW(Win32Path(path).c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
            throw std::runtime_error("cannot create export directory: " + path.string() +
                " (Windows error " + std::to_string(GetLastError()) + ")");
        dir->handle = CreateFileW(Win32Path(path).c_str(), FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | (dir->writeShared ? FILE_SHARE_WRITE : 0), nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        FILE_ATTRIBUTE_TAG_INFO attributes{};
        if (dir->handle == INVALID_HANDLE_VALUE ||
            !GetFileInformationByHandleEx(dir->handle, FileAttributeTagInfo, &attributes, sizeof(attributes)) ||
            !(attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
            throw std::runtime_error("unsafe export directory: " + path.string() +
                " (Windows error " + std::to_string(GetLastError()) + ")");
#else
        if (create && parent && ::mkdirat(parent->handle, path.filename().c_str(), 0700) != 0 && errno != EEXIST)
            throw std::runtime_error("cannot create export directory: " + path.string());
        dir->handle = parent ? ::openat(parent->handle, path.filename().c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                             : ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (dir->handle < 0) throw std::runtime_error("unsafe export directory: " + path.string());
#endif
        auto& result = *dir;
        directories.emplace(path, std::move(dir));
#ifdef _WIN32
        // A pinned child keeps this parent nonempty. NTFS rejects adding a
        // junction/symlink in-place to a nonempty directory, so sharing writes
        // is now safe and permits other sessions and atomic file publication.
        if (parent) ShareWrites(*parent);
#endif
        return result;
    }
    static bool ExistsIn(const Directory& dir, const fs::path& leaf) {
#ifdef _WIN32
        const auto attrs = GetFileAttributesW(Win32Path(dir.path / leaf).c_str());
        if (attrs != INVALID_FILE_ATTRIBUTES) return true;
        const auto error = GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return false;
#else
        struct stat status{};
        if (::fstatat(dir.handle, leaf.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0) return true;
        if (errno == ENOENT) return false;
#endif
        throw std::runtime_error("cannot inspect export entry");
    }
    // Cleanup never follows a newly inserted link, even for unregistered files.
    void Clear(const fs::path& path) {
        [[maybe_unused]] auto& dir = Open(path, false, false);
#ifdef _WIN32
        WIN32_FIND_DATAW data{};
        const auto find = FindFirstFileW(Win32Path(path / L"*").c_str(), &data);
        if (find == INVALID_HANDLE_VALUE) return;
        try {
            do {
                const fs::path leaf(data.cFileName);
                if (leaf == L"." || leaf == L"..") continue;
                const auto child = path / leaf;
                if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    if (!(data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) Clear(child);
                    directories.erase(child);
                    RemoveDirectoryW(Win32Path(child).c_str());
                } else DeleteFileW(Win32Path(child).c_str());
            } while (FindNextFileW(find, &data));
        } catch (...) { FindClose(find); throw; }
        FindClose(find);
#else
        const int scan = ::openat(dir.handle, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (scan < 0) throw std::runtime_error("cannot scan staging directory");
        DIR* listing = ::fdopendir(scan);
        if (!listing) { ::close(scan); throw std::runtime_error("cannot scan staging directory"); }
        try {
            while (auto* entry = ::readdir(listing)) {
                const fs::path leaf(entry->d_name);
                if (leaf == "." || leaf == "..") continue;
                struct stat status{};
                if (::fstatat(dir.handle, leaf.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) continue;
                if (S_ISDIR(status.st_mode)) {
                    Clear(path / leaf);
                    directories.erase(path / leaf);
                    ::unlinkat(dir.handle, leaf.c_str(), AT_REMOVEDIR);
                } else ::unlinkat(dir.handle, leaf.c_str(), 0);
            }
        } catch (...) { ::closedir(listing); throw; }
        ::closedir(listing);
#endif
    }
public:
    void Ensure(const fs::path& path, bool checkClaims = false) { (void)Open(path, true, checkClaims); }
    fs::path FilePath(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
#ifdef _WIN32
        return fs::path(Win32Path(path));
#else
        // Streaming writers receive a capability path, not a replaceable ancestry.
        return fs::path("/proc/self/fd") / std::to_string(parent.handle) / path.filename();
#endif
    }
    bool Exists(const fs::path& path) { return ExistsIn(Open(path.parent_path(), false, false), path.filename()); }
    bool Regular(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
#ifdef _WIN32
        const auto attrs = GetFileAttributesW(Win32Path(path).c_str());
        return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT));
#else
        struct stat status{};
        return ::fstatat(parent.handle, path.filename().c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(status.st_mode);
#endif
    }
    void CreatePrivate(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
#ifdef _WIN32
        PrivateDirectorySecurity security;
        const bool made = CreateDirectoryW(Win32Path(path).c_str(), &security.Attributes()) != 0;
#else
        const bool made = ::mkdirat(parent.handle, path.filename().c_str(), 0700) == 0;
#endif
        if (!made) throw std::runtime_error("output is locked: " + path.string());
        try { (void)Open(path, false, false, true); }
        catch (...) { Remove(path); throw; }
    }
    void Claim(const fs::path& path) { CreatePrivate(path); }
    void Replace(const fs::path& from, const fs::path& to) {
        [[maybe_unused]] auto& src = Open(from.parent_path(), false, false);
        [[maybe_unused]] auto& dst = Open(to.parent_path(), false, false);
#ifdef _WIN32
        if (!MoveFileExW(Win32Path(from).c_str(), Win32Path(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            const auto error = GetLastError();
            throw std::runtime_error("atomic replacement failed: " + to.string() +
                " (Windows error " + std::to_string(error) + ")");
        }
#else
        if (::renameat(src.handle, from.filename().c_str(), dst.handle, to.filename().c_str()) != 0)
            throw std::runtime_error("atomic replacement failed: " + to.string());
#endif
    }
    void Remove(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
        directories.erase(path);
#ifdef _WIN32
        RemoveDirectoryW(Win32Path(path).c_str());
#else
        ::unlinkat(parent.handle, path.filename().c_str(), AT_REMOVEDIR);
#endif
    }
    void RemoveTree(const fs::path& path) { Clear(path); Remove(path); }
};
} // namespace quantiloom::dataset::detail
