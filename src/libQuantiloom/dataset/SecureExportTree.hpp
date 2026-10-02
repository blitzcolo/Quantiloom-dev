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
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace quantiloom::dataset::detail {
namespace fs = std::filesystem;

// Directory capabilities live for the whole transaction. Windows denies rename
// and deletion of every ancestor; POSIX always addresses children through dirfd.
// POSIX cannot forbid a same-user directory rename. Identity checks reject a
// detected rename; the descriptors keep concurrent operations on the original
// directories instead of following a replacement link into a foreign tree.
class SecureExportTree {
    struct Directory {
#ifdef _WIN32
        HANDLE handle = INVALID_HANDLE_VALUE;
        ~Directory() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
        int handle = -1;
        ~Directory() { if (handle >= 0) ::close(handle); }
#endif
        fs::path path;
    };
    std::map<fs::path, std::unique_ptr<Directory>> directories;
    static constexpr const char* suffix = ".quantiloom-export.lock";

    Directory& Open(const fs::path& path, bool create, bool checkClaims) {
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
        if (create && parent && !CreateDirectoryW(path.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
            throw std::runtime_error("cannot create export directory: " + path.string());
        dir->handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        FILE_ATTRIBUTE_TAG_INFO attributes{};
        if (dir->handle == INVALID_HANDLE_VALUE ||
            !GetFileInformationByHandleEx(dir->handle, FileAttributeTagInfo, &attributes, sizeof(attributes)) ||
            !(attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
            throw std::runtime_error("unsafe export directory: " + path.string());
#else
        if (create && parent && ::mkdirat(parent->handle, path.filename().c_str(), 0700) != 0 && errno != EEXIST)
            throw std::runtime_error("cannot create export directory: " + path.string());
        dir->handle = parent ? ::openat(parent->handle, path.filename().c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                             : ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (dir->handle < 0) throw std::runtime_error("unsafe export directory: " + path.string());
#endif
        auto& result = *dir;
        directories.emplace(path, std::move(dir));
        return result;
    }
    static bool ExistsIn(const Directory& dir, const fs::path& leaf) {
#ifdef _WIN32
        const auto attrs = GetFileAttributesW((dir.path / leaf).c_str());
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
        const auto find = FindFirstFileW((path / L"*").c_str(), &data);
        if (find == INVALID_HANDLE_VALUE) return;
        try {
            do {
                const fs::path leaf(data.cFileName);
                if (leaf == L"." || leaf == L"..") continue;
                const auto child = path / leaf;
                if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    if (!(data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) Clear(child);
                    directories.erase(child);
                    RemoveDirectoryW(child.c_str());
                } else DeleteFileW(child.c_str());
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
        return path;
#else
        // Streaming writers receive a capability path, not a replaceable ancestry.
        return fs::path("/proc/self/fd") / std::to_string(parent.handle) / path.filename();
#endif
    }
    bool Exists(const fs::path& path) { return ExistsIn(Open(path.parent_path(), false, false), path.filename()); }
    bool Regular(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
#ifdef _WIN32
        const auto attrs = GetFileAttributesW(path.c_str());
        return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT));
#else
        struct stat status{};
        return ::fstatat(parent.handle, path.filename().c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(status.st_mode);
#endif
    }
    void Claim(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
#ifdef _WIN32
        const bool made = CreateDirectoryW(path.c_str(), nullptr) != 0;
#else
        const bool made = ::mkdirat(parent.handle, path.filename().c_str(), 0700) == 0;
#endif
        if (!made) throw std::runtime_error("output is locked: " + path.string());
        try { (void)Open(path, false, false); }
        catch (...) { Remove(path); throw; }
    }
    void Replace(const fs::path& from, const fs::path& to) {
        [[maybe_unused]] auto& src = Open(from.parent_path(), false, false);
        [[maybe_unused]] auto& dst = Open(to.parent_path(), false, false);
#ifdef _WIN32
        if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
#else
        if (::renameat(src.handle, from.filename().c_str(), dst.handle, to.filename().c_str()) != 0)
#endif
            throw std::runtime_error("atomic replacement failed: " + to.string());
    }
    void Remove(const fs::path& path) {
        [[maybe_unused]] auto& parent = Open(path.parent_path(), false, false);
        directories.erase(path);
#ifdef _WIN32
        RemoveDirectoryW(path.c_str());
#else
        ::unlinkat(parent.handle, path.filename().c_str(), AT_REMOVEDIR);
#endif
    }
    void RemoveTree(const fs::path& path) { Clear(path); Remove(path); }
};
} // namespace quantiloom::dataset::detail
