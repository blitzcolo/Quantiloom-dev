/**
 * @file ResourceLocator.hpp
 * @brief One resolution of the SDK's runtime resources (internal)
 *
 * Behind the public core/Resources.hpp. The installed layout is
 * `<root>/shaders/spirv/*.spv` plus a SHADERSET stamp and `<root>/assets/...`;
 * a build tree has neither, so the working-directory development fallback
 * (src/shaders, shaders, ../shaders, assets/) is what serves it.
 */

#pragma once

#include "core/Types.hpp"

#include <filesystem>

namespace quantiloom::core {

enum class ResourceOrigin { None, Host, Environment, Library, Development };

struct ResolvedResources {
    ResourceOrigin origin = ResourceOrigin::None;
    /// Accepted root; empty when the development fallback resolved.
    std::filesystem::path root;
    /// Directory the .spv set is read from.
    std::filesystem::path shaderDir;
    /// Directory holding assets/ contents (atmos_models, luts).
    std::filesystem::path assetDir;
    /// "no Quantiloom resources found; probed: ..." when origin is None.
    String error;

    [[nodiscard]] bool Found() const { return origin != ResourceOrigin::None; }
    [[nodiscard]] std::filesystem::path ShaderPath(StringView name) const;
    [[nodiscard]] std::filesystem::path AssetPath(const std::filesystem::path& relative) const;
};

/// Everything a resolution depends on, so the unit tests can drive it without
/// touching process state.
struct ResourceCandidates {
    std::filesystem::path hostRoot;
    std::filesystem::path envRoot;
    Vector<std::filesystem::path> libraryRoots;
    Vector<std::filesystem::path> devShaderDirs;
    std::filesystem::path devAssetDir = "assets";
    /// Expected SHADERSET digest; empty disables the stamp check.
    String shaderSetDigest;
};

/// Pure resolution over @p candidates. Rejections are appended to @p log when
/// given, and always summarized in ResolvedResources::error on failure.
ResolvedResources ResolveResourceCandidates(const ResourceCandidates& candidates,
                                            Vector<String>* log = nullptr);

/// The SHADERSET digest compiled into this library: what a root's stamp must
/// equal. Internal (no QL_API); the generated header that holds it is private
/// to quantiloom_core, so tests read it through here.
String CompiledShaderSetDigest();

/// Process-wide resolution, cached. SetResourceRoot() invalidates the cache.
const ResolvedResources& GetResolvedResources();

/// The error GetResolvedResources() produced; empty when resources resolved.
String ResourceNotFoundMessage();

/// Directory of the module (DLL or executable) this code was linked into.
/// GetModuleHandleExW(FROM_ADDRESS) on Windows, dladdr elsewhere.
std::filesystem::path LibraryDirectory();

/// Clear the cached resolution. Internal: SetResourceRoot and the tests.
void ResetResourceLocator();

/// SetResourceRoot without the public header, for internal callers.
void SetResourceRootOverride(const std::filesystem::path& root);

}  // namespace quantiloom::core
