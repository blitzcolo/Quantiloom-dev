#include "gpu/ShaderLibrary.hpp"

#include "core/Log.hpp"
#include "core/ResourceLocator.hpp"
#include "renderer/ShaderBinary.hpp"

#include <cstring>
#include <fstream>

namespace quantiloom::gpu {

namespace {

Vector<std::byte> ReadBytes(const std::filesystem::path& path, ShaderFormat format) {
    if (format == ShaderFormat::SpirV) {
        const Vector<u32> words = rendercore::ReadSpirvFile(path);
        Vector<std::byte> bytes(words.size() * sizeof(u32));
        if (!words.empty()) std::memcpy(bytes.data(), words.data(), bytes.size());
        return bytes;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const auto size = static_cast<size_t>(file.tellg());
    if (size == 0) return {};
    Vector<std::byte> bytes(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!file) return {};
    return bytes;
}

}  // namespace

ShaderLibrary::ShaderLibrary(ShaderFormat format) : format_(format) {}

const char* ShaderLibrary::Extension(ShaderFormat format) {
    return format == ShaderFormat::SpirV ? ".spv" : ".metallib";
}

Result<Vector<ShaderCode>, String> ShaderLibrary::Load(std::span<const StringView> names) const {
    const auto& resources = core::GetResolvedResources();
    if (!resources.Found()) {
        return Result<Vector<ShaderCode>, String>::Err(resources.error);
    }
    return LoadFrom(resources.shaderDir, names);
}

Result<Vector<ShaderCode>, String> ShaderLibrary::LoadFrom(const std::filesystem::path& directory,
                                                           std::span<const StringView> names) const {
    Vector<ShaderCode> loaded;
    loaded.reserve(names.size());
    String missing;
    for (const StringView name : names) {
        ShaderCode code;
        code.name = String(name);
        code.path = directory / (code.name + Extension(format_));
        code.bytes = ReadBytes(code.path, format_);
        if (code.bytes.empty()) {
            if (!missing.empty()) missing += ", ";
            missing += code.path.string();
            continue;
        }
        loaded.push_back(std::move(code));
    }
    if (!missing.empty()) {
        return Result<Vector<ShaderCode>, String>::Err("no valid shader file: " + missing);
    }
    QL_LOG_DEBUG("Shaders: {} loaded from {}", loaded.size(), directory.string());
    return loaded;
}

}  // namespace quantiloom::gpu
