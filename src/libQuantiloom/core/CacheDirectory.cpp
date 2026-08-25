#include "CacheDirectory.hpp"

#include "core/Log.hpp"

#include <filesystem>

#if defined(_WIN32)
    #include <shlobj.h>
    #include <windows.h>
#else  // macOS and Linux
    #include <pwd.h>
    #include <unistd.h>
#endif

namespace quantiloom::core {

String GetDefaultCacheDirectory() {
    std::filesystem::path cacheDir;

#if defined(_WIN32)
    // Windows: Use LOCALAPPDATA
    wchar_t* localAppData = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &localAppData))) {
        cacheDir = std::filesystem::path(localAppData) / "Quantiloom" / "cache";
        CoTaskMemFree(localAppData);
    } else {
        // Fallback to temp directory
        cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
    }

#elif defined(__APPLE__)
    // macOS: Use ~/Library/Caches/
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (home) {
        cacheDir = std::filesystem::path(home) / "Library" / "Caches" / "Quantiloom";
    } else {
        cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
    }

#else  // Linux
    // Linux: Use XDG_CACHE_HOME or ~/.cache/
    const char* xdgCache = getenv("XDG_CACHE_HOME");
    if (xdgCache && xdgCache[0] != '\0') {
        cacheDir = std::filesystem::path(xdgCache) / "Quantiloom";
    } else {
        const char* home = getenv("HOME");
        if (!home) {
            struct passwd* pw = getpwuid(getuid());
            if (pw) home = pw->pw_dir;
        }
        if (home) {
            cacheDir = std::filesystem::path(home) / ".cache" / "Quantiloom";
        } else {
            cacheDir = std::filesystem::temp_directory_path() / "Quantiloom" / "cache";
        }
    }
#endif

    // Create directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    if (ec) {
        QL_LOG_WARN("Failed to create cache directory {}: {}", cacheDir.string(), ec.message());
        // Fall back to current directory
        return ".";
    }

    return cacheDir.string();
}

}  // namespace quantiloom::core
