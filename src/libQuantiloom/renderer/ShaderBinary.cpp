#include "ShaderBinary.hpp"

#include <fstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#include <climits>
#endif

namespace quantiloom::rendercore {

std::filesystem::path ShaderExecutableDir() {
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#elif defined(__linux__)
    char buffer[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length > 0) {
        buffer[length] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
    return std::filesystem::current_path();
#else
    return std::filesystem::current_path();
#endif
}

Vector<std::filesystem::path> SpirvSearchPaths(StringView name) {
    const auto exeDir = ShaderExecutableDir();
    const std::filesystem::path file{name};
    return {
        file,
        exeDir / file,
        std::filesystem::path("shaders") / file,
        exeDir / "shaders" / file,
        std::filesystem::path("..") / "shaders" / file,
        std::filesystem::path("src") / "shaders" / file,
    };
}

Vector<u32> LoadSpirv(StringView name) {
    for (const auto& path : SpirvSearchPaths(name)) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) continue;
        const auto bytes = static_cast<size_t>(file.tellg());
        if (bytes == 0 || bytes % sizeof(u32) != 0) continue;
        Vector<u32> words(bytes / sizeof(u32));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()),
                  static_cast<std::streamsize>(bytes));
        // A truncated read leaves garbage in the tail; keep looking rather
        // than hand it to the driver.
        if (file) return words;
    }
    return {};
}

} // namespace quantiloom::rendercore
