/**
 * @file ShaderLibrary.hpp
 * @brief Loading a kernel set through the SDK's one resource locator (internal)
 *
 * Shader lookup is not duplicated here: core/ResourceLocator resolves exactly one shader
 * directory (installed share/quantiloom/shaders/spirv, or the development fallback), and
 * Load() reads every requested name from it, so a set of siblings can never be mixed from
 * two builds. LoadFrom() takes the directory explicitly for tests.
 */

#pragma once

#include "core/Types.hpp"

#include <filesystem>
#include <span>

namespace quantiloom::gpu {

enum class ShaderFormat : u8 { SpirV, MetalLib };  // extension ".spv" | ".metallib"

struct ShaderCode {
    String name;                  // logical name == the CMake OUT_NAME, e.g. "camera_psf.comp"
    std::filesystem::path path;   // where it was found
    Vector<std::byte> bytes;
};

class ShaderLibrary {
public:
    explicit ShaderLibrary(ShaderFormat format);

    /// Every name, from the directory the resource locator resolved. Err names the missing
    /// or invalid files and the directory searched.
    [[nodiscard]] Result<Vector<ShaderCode>, String> Load(std::span<const StringView> names) const;

    /// Every name from @p directory only (the locator is not consulted).
    [[nodiscard]] Result<Vector<ShaderCode>, String> LoadFrom(const std::filesystem::path& directory,
                                                              std::span<const StringView> names) const;

    [[nodiscard]] ShaderFormat Format() const { return format_; }
    [[nodiscard]] static const char* Extension(ShaderFormat format);

private:
    ShaderFormat format_;
};

}  // namespace quantiloom::gpu
