#include "gpu/metal/MetalBackend.hpp"

#include "core/ResultFail.hpp"

#include <nlohmann/json.hpp>

#include <fstream>

namespace quantiloom::gpu::metal {

MTLPixelFormat ToMtl(Format format) {
    switch (format) {
        case Format::Undefined: return MTLPixelFormatInvalid;
        case Format::R32Float: return MTLPixelFormatR32Float;
        case Format::RGBA32Float: return MTLPixelFormatRGBA32Float;
        case Format::RGBA32Uint: return MTLPixelFormatRGBA32Uint;
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::RGBA8Srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::BGRA8Srgb: return MTLPixelFormatBGRA8Unorm_sRGB;
        case Format::RG32Float: return MTLPixelFormatRG32Float;
        case Format::RGBA16Float: return MTLPixelFormatRGBA16Float;
        case Format::BC7Unorm: return MTLPixelFormatBC7_RGBAUnorm;
        case Format::BC7Srgb: return MTLPixelFormatBC7_RGBAUnorm_sRGB;
    }
    return MTLPixelFormatInvalid;
}

bool IsBlockCompressed(Format format) {
    return format == Format::BC7Unorm || format == Format::BC7Srgb;
}

u32 FormatBytes(Format format) {
    switch (format) {
        case Format::Undefined: return 0;
        case Format::R32Float: return 4;
        case Format::RGBA32Float: return 16;
        case Format::RGBA32Uint: return 16;
        case Format::RGBA8Unorm:
        case Format::RGBA8Srgb:
        case Format::BGRA8Srgb: return 4;
        case Format::RG32Float: return 8;
        case Format::RGBA16Float: return 8;
        case Format::BC7Unorm:
        case Format::BC7Srgb: return 16;
    }
    return 0;
}

String NsErrorText(NSError* error) {
    if (error == nil) return "unknown Metal error";
    return String([[error localizedDescription] UTF8String]);
}

Result<ShaderReflection, String> LoadReflection(const std::filesystem::path& metallibPath) {
    std::filesystem::path path = metallibPath;
    path.replace_extension(".reflection.json");
    std::ifstream file(path);
    if (!file.is_open()) {
        return Fail<ShaderReflection>("no shader reflection at " + path.string());
    }
    nlohmann::json json;
    try {
        file >> json;
    } catch (const std::exception& e) {
        return Fail<ShaderReflection>("unreadable shader reflection " + path.string() + ": " + e.what());
    }

    ShaderReflection reflection;
    reflection.entryPoint = json.value("EntryPoint", String("main"));
    for (const auto& entry : json.value("TopLevelArgumentBuffer", nlohmann::json::array())) {
        ArgumentSlot slot;
        const String type = entry.value("Type", String());
        if (type == "SRV") {
            slot.type = ArgumentSlot::Type::Srv;
        } else if (type == "UAV") {
            slot.type = ArgumentSlot::Type::Uav;
        } else if (type == "CBV") {
            slot.type = ArgumentSlot::Type::Cbv;
        } else if (type == "Sampler" || type == "SMP") {
            slot.type = ArgumentSlot::Type::Sampler;
        } else if (type == "Table") {
            slot.type = ArgumentSlot::Type::Table;
            reflection.rootSignature = true;
        } else {
            return Fail<ShaderReflection>("shader reflection " + path.string() + " has an argument of type '" +
                                          type + "' the Metal backend does not encode");
        }
        slot.offset = entry.value("EltOffset", 0u);
        slot.size = entry.value("Size", 0u);
        slot.slot = entry.value("Slot", 0u);
        slot.space = entry.value("Space", 0u);
        reflection.topLevelBytes = std::max(reflection.topLevelBytes, slot.offset + slot.size);
        reflection.topLevel.push_back(slot);
    }
    if (json.contains("state") && json["state"].contains("tg_size")) {
        const auto& size = json["state"]["tg_size"];
        for (size_t i = 0; i < 3 && i < size.size(); ++i) {
            reflection.threadgroup[i] = std::max(1u, size[i].get<u32>());
        }
    }
    for (const auto& constant : json.value("FunctionConstants", nlohmann::json::array())) {
        FunctionConstantInfo info;
        info.name = constant.value("name", String());
        const auto underscore = info.name.find_last_of('_');
        if (underscore == String::npos) {
            return Fail<ShaderReflection>("shader reflection " + path.string() + " names a function constant '" +
                                          info.name + "' with no index");
        }
        info.id = static_cast<u32>(std::stoul(info.name.substr(underscore + 1)));
        reflection.functionConstants.push_back(info);
    }
    return reflection;
}

}  // namespace quantiloom::gpu::metal
