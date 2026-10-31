#include "ShaderBinary.hpp"

#include "core/ResourceLocator.hpp"

#include <fstream>
#include <vector>

namespace quantiloom::rendercore {

Vector<std::filesystem::path> SpirvSearchPaths(StringView name) {
    Vector<std::filesystem::path> paths;
    const auto& resources = core::GetResolvedResources();
    if (resources.Found()) {
        paths.push_back(resources.ShaderPath(name));
    }
    return paths;
}

Vector<u32> ReadSpirvFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const auto bytes = static_cast<size_t>(file.tellg());
    if (bytes == 0 || bytes % sizeof(u32) != 0) return {};
    Vector<u32> words(bytes / sizeof(u32));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()),
              static_cast<std::streamsize>(bytes));
    // A truncated read leaves garbage in the tail; keep looking rather
    // than hand it to the driver.
    if (!file) return {};
    return words;
}

Vector<u32> LoadSpirv(StringView name) {
    for (const auto& path : SpirvSearchPaths(name)) {
        Vector<u32> words = ReadSpirvFile(path);
        if (!words.empty()) return words;
    }
    return {};
}

} // namespace quantiloom::rendercore
