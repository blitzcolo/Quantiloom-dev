#include "core/ResourceLocator.hpp"

#include "core/Log.hpp"
#include "core/ShaderSetDigest.hpp"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quantiloom::core {
namespace {

namespace fs = std::filesystem;

std::mutex g_mutex;
std::filesystem::path g_hostRoot;
bool g_resolved = false;
ResolvedResources g_resources;

String Trimmed(String text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == String::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

String JoinProbed(const Vector<String>& lines) {
    String joined;
    for (const auto& line : lines) {
        if (!joined.empty()) joined += "; ";
        joined += line;
    }
    return joined;
}

String ReadShaderSetStamp(const fs::path& root) {
    std::ifstream file(root / "shaders" / "spirv" / "SHADERSET", std::ios::binary);
    if (!file.is_open()) return {};
    std::ostringstream contents;
    contents << file.rdbuf();
    return Trimmed(contents.str());
}

}  // namespace

std::filesystem::path ResolvedResources::ShaderPath(StringView name) const {
    return shaderDir / std::string(name);
}

std::filesystem::path ResolvedResources::AssetPath(const std::filesystem::path& relative) const {
    return assetDir / relative;
}

ResolvedResources ResolveResourceCandidates(const ResourceCandidates& candidates,
                                            Vector<String>* log) {
    Vector<String> probed;
    auto note = [&](const String& line) {
        probed.push_back(line);
        if (log) log->push_back(line);
    };

    auto tryRoot = [&](const fs::path& root, ResourceOrigin origin,
                       const String& description) -> ResolvedResources {
        ResolvedResources result;
        std::error_code ec;
        const fs::path shaderDir = root / "shaders" / "spirv";
        if (!fs::is_directory(shaderDir, ec)) {
            note(description + " (no shaders/spirv)");
            return result;
        }
        if (!candidates.shaderSetDigest.empty()) {
            const String stamp = ReadShaderSetStamp(root);
            if (stamp.empty()) {
                note(description + " (shaders/spirv/SHADERSET is missing or empty; "
                                   "expected shader set " + candidates.shaderSetDigest + ")");
                return result;
            }
            if (stamp != candidates.shaderSetDigest) {
                note(description + " (SHADERSET mismatch: found " + stamp +
                     ", library was built with " + candidates.shaderSetDigest + ")");
                return result;
            }
        }
        result.origin = origin;
        result.root = root;
        result.shaderDir = shaderDir;
        result.assetDir = root / "assets";
        return result;
    };

    if (!candidates.hostRoot.empty()) {
        ResolvedResources result = tryRoot(candidates.hostRoot, ResourceOrigin::Host,
                                           "SetResourceRoot " + candidates.hostRoot.string());
        if (!result.Found()) {
            result.error = "no Quantiloom resources found; probed: " + JoinProbed(probed);
            return result;
        }
        return result;
    }
    if (!candidates.envRoot.empty()) {
        ResolvedResources result = tryRoot(candidates.envRoot, ResourceOrigin::Environment,
                                           "QUANTILOOM_RESOURCE_ROOT " + candidates.envRoot.string());
        if (!result.Found()) {
            result.error = "no Quantiloom resources found; probed: " + JoinProbed(probed);
            return result;
        }
        return result;
    }

    // A library-relative share/ that exists but does not match this build is a
    // deployment error, not a reason to quietly fall back to the working
    // directory: that is exactly the stale-copy case the stamp exists for.
    bool rejectedLibraryRoot = false;
    for (const auto& root : candidates.libraryRoots) {
        std::error_code ec;
        const bool present = fs::is_directory(root, ec);
        ResolvedResources result = tryRoot(root, ResourceOrigin::Library, root.string());
        if (result.Found()) return result;
        rejectedLibraryRoot = rejectedLibraryRoot || present;
    }

    if (!rejectedLibraryRoot) {
        for (const auto& dir : candidates.devShaderDirs) {
            std::error_code ec;
            if (!fs::is_regular_file(dir / "raygen.spv", ec)) continue;
            ResolvedResources result;
            result.origin = ResourceOrigin::Development;
            result.shaderDir = dir;
            result.assetDir = candidates.devAssetDir;
            note("development fallback " + dir.string());
            return result;
        }
        note("development fallback (no raygen.spv in the working-directory candidates)");
    }

    ResolvedResources result;
    result.error = "no Quantiloom resources found; probed: " + JoinProbed(probed);
    return result;
}

std::filesystem::path LibraryDirectory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&LibraryDirectory), &module) != 0) {
        wchar_t buffer[MAX_PATH];
        const DWORD length = GetModuleFileNameW(module, buffer, MAX_PATH);
        if (length > 0 && length < MAX_PATH) {
            return std::filesystem::path(buffer).parent_path();
        }
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&LibraryDirectory), &info) != 0 && info.dli_fname) {
        return std::filesystem::path(info.dli_fname).parent_path();
    }
#endif
    return {};
}

void ResetResourceLocator() {
    std::lock_guard lock(g_mutex);
    g_resolved = false;
    g_resources = ResolvedResources{};
}

void SetResourceRootOverride(const std::filesystem::path& root) {
    std::lock_guard lock(g_mutex);
    g_hostRoot = root;
    g_resolved = false;
    g_resources = ResolvedResources{};
}

String CompiledShaderSetDigest() {
    return kShaderSetDigest;
}

const ResolvedResources& GetResolvedResources() {
    std::lock_guard lock(g_mutex);
    if (g_resolved) return g_resources;

    ResourceCandidates candidates;
    candidates.hostRoot = g_hostRoot;
    if (const char* env = std::getenv("QUANTILOOM_RESOURCE_ROOT"); env && *env) {
        candidates.envRoot = std::filesystem::path(env);
    }
    const std::filesystem::path libraryDir = LibraryDirectory();
    if (!libraryDir.empty()) {
        candidates.libraryRoots = {libraryDir / ".." / "share" / "quantiloom",
                                   libraryDir / "share" / "quantiloom"};
    }
    candidates.devShaderDirs = {"src/shaders", "shaders", "../shaders"};
    candidates.devAssetDir = "assets";
    candidates.shaderSetDigest = CompiledShaderSetDigest();

    Vector<String> log;
    g_resources = ResolveResourceCandidates(candidates, &log);
    g_resolved = true;
    for (const auto& line : log) {
        QL_LOG_INFO("Resource lookup: {}", line);
    }
    if (g_resources.origin == ResourceOrigin::Development) {
        QL_LOG_INFO("Resource lookup: using the working-directory development fallback ({})",
                    g_resources.shaderDir.string());
    } else if (!g_resources.Found()) {
        QL_LOG_ERROR("{}", g_resources.error);
    }
    return g_resources;
}

String ResourceNotFoundMessage() {
    const auto& resources = GetResolvedResources();
    if (resources.Found()) return {};
    return resources.error;
}

}  // namespace quantiloom::core
