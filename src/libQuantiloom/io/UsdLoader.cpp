/**
 * @file UsdLoader.cpp
 * @brief OpenUSD scene loader implementation using Pixar OpenUSD
 *
 * Implementation of UsdLoader class for loading OpenUSD files.
 * Uses Pixar's official OpenUSD library for USD parsing.
 *
 * Supports:
 * - Full USD composition (sublayers, references, payloads, variants, inherits)
 * - Variant selection via UsdLoadOptions
 * - UsdPreviewSurface and MaterialX materials
 * - Texture connection following
 * - GeomSubsets for multi-material meshes
 * - PointInstancer expansion
 *
 * @author blitzcolo
 */

#include "UsdLoader.hpp"
#include "io/SpectralIO.hpp"
#include "io/UsdShadeGraph.hpp"
#include "io/UsdSurfaceTables.hpp"
#include "io/UsdTextureBank.hpp"
#include "io/ImageIO.hpp"
#include "scene/MeshOptimizer.hpp"
#include "scene/NormalGenerator.hpp"
#include "scene/TangentGenerator.hpp"
#include "renderer/TextureCompressor.hpp"
#include "core/Log.hpp"

#include <cctype>
#include <string_view>

// Conditional compilation based on OpenUSD availability
// ============================================================================
// ParseUsdVariantSpec - the config's `variant` string, for a USD file
// ============================================================================
// pxr-free and outside the #if, so a build without OpenUSD still parses the key
// and still reports a malformed one. See the header for the syntax.

namespace quantiloom {

namespace {

StringView TrimSpace(StringView text) {
    const auto notSpace = [](char c) { return std::isspace(static_cast<unsigned char>(c)) == 0; };
    usize begin = 0;
    while (begin < text.size() && !notSpace(text[begin])) {
        ++begin;
    }
    usize end = text.size();
    while (end > begin && !notSpace(text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

}  // namespace

Result<UsdLoadOptions::VariantSelections, String> ParseUsdVariantSpec(StringView spec) {
    using Selections = UsdLoadOptions::VariantSelections;
    using Res = Result<Selections, String>;

    Selections selections;
    const StringView trimmed = TrimSpace(spec);
    if (trimmed.empty()) {
        return Res(std::move(selections));
    }

    usize cursor = 0;
    while (cursor <= trimmed.size()) {
        const usize comma = trimmed.find(',', cursor);
        const StringView entry = TrimSpace(
            trimmed.substr(cursor, comma == StringView::npos ? StringView::npos : comma - cursor));
        cursor = comma == StringView::npos ? trimmed.size() + 1 : comma + 1;

        if (entry.empty()) {
            return Res::Err("USD variant spec has an empty entry: '" + String(spec) + "'");
        }

        StringView primPath;
        StringView assignment = entry;

        if (entry.front() == '/') {
            const usize open = entry.find('{');
            if (open == StringView::npos || entry.back() != '}') {
                return Res::Err("USD variant entry '" + String(entry) +
                                "' names a prim but not a selection; expected "
                                "/Prim{set=variant}");
            }
            primPath = TrimSpace(entry.substr(0, open));
            assignment = TrimSpace(entry.substr(open + 1, entry.size() - open - 2));
            if (primPath.size() < 2) {
                return Res::Err("USD variant entry '" + String(entry) + "' has no prim path");
            }
        }

        if (assignment.find('{') != StringView::npos ||
            assignment.find('}') != StringView::npos) {
            // Braces outside a prim-scoped entry: most likely a prim path that
            // lost its leading slash, which would otherwise parse as a variant
            // set called "{color".
            return Res::Err("USD variant entry '" + String(entry) +
                            "' has braces but no leading '/'; expected "
                            "/Prim{set=variant} or set=variant");
        }

        const usize equals = assignment.find('=');
        if (equals == StringView::npos) {
            return Res::Err("USD variant entry '" + String(entry) +
                            "' is not set=variant");
        }
        const StringView setName = TrimSpace(assignment.substr(0, equals));
        const StringView variantName = TrimSpace(assignment.substr(equals + 1));
        if (setName.empty() || variantName.empty()) {
            return Res::Err("USD variant entry '" + String(entry) +
                            "' has an empty variant set or variant name");
        }
        if (assignment.find('=', equals + 1) != StringView::npos) {
            return Res::Err("USD variant entry '" + String(entry) +
                            "' has more than one '='");
        }

        selections[String(primPath)][String(setName)] = String(variantName);
    }

    return Res(std::move(selections));
}

}  // namespace quantiloom

#if QUANTILOOM_USE_OPENUSD

// OpenUSD headers
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usd/variantSets.h>
#include <pxr/usd/usd/editContext.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/tokens.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/plug/registry.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <mutex>
#include <future>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE

namespace quantiloom {

// ============================================================================
// InitializeUsdPlugins - Ensure USD plugins are discoverable
// ============================================================================

static bool g_usdPluginsInitialized = false;

static void InitializeUsdPlugins() {
    if (g_usdPluginsInitialized) {
        return;
    }
    g_usdPluginsInitialized = true;

    // Check if PXR_PLUGINPATH_NAME is already set
    const char* existingPath = std::getenv("PXR_PLUGINPATH_NAME");
    if (existingPath && existingPath[0] != '\0') {
        QL_LOG_INFO("USD plugin path already set: {}", existingPath);
        return;
    }

    // Try to find usd plugins directory relative to the executable or library
    // Common locations:
    // 1. <exe_dir>/usd
    // 2. <exe_dir>/../lib/usd
    // 3. USD_ROOT environment variable

    std::vector<std::filesystem::path> searchPaths;

#ifdef _WIN32
    // Get the path of the current module (DLL or EXE)
    char modulePath[MAX_PATH];
    HMODULE hModule = nullptr;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCSTR>(&InitializeUsdPlugins),
        &hModule);
    if (GetModuleFileNameA(hModule, modulePath, MAX_PATH) > 0) {
        std::filesystem::path moduleDir = std::filesystem::path(modulePath).parent_path();
        searchPaths.push_back(moduleDir / "usd");
        searchPaths.push_back(moduleDir / ".." / "lib" / "usd");
    }
#else
    // On Linux/macOS, use /proc/self/exe or dladdr
    std::filesystem::path exePath = std::filesystem::read_symlink("/proc/self/exe");
    std::filesystem::path exeDir = exePath.parent_path();
    searchPaths.push_back(exeDir / "usd");
    searchPaths.push_back(exeDir / ".." / "lib" / "usd");
#endif

    // Also check USD_ROOT environment variable
    const char* usdRoot = std::getenv("USD_ROOT");
    if (usdRoot && usdRoot[0] != '\0') {
        searchPaths.push_back(std::filesystem::path(usdRoot) / "lib" / "usd");
    }

    // Find the first valid plugin directory
    for (const auto& searchPath : searchPaths) {
        std::filesystem::path plugInfoPath = searchPath / "plugInfo.json";
        if (std::filesystem::exists(plugInfoPath)) {
            std::string pluginPath = std::filesystem::absolute(searchPath).string();
            std::replace(pluginPath.begin(), pluginPath.end(), '\\', '/');

            QL_LOG_INFO("Found USD plugins at: {}", pluginPath);

            // Register the plugin path with USD
            PlugRegistry::GetInstance().RegisterPlugins(pluginPath);
            return;
        }
    }

    QL_LOG_WARN("Could not find USD plugins directory. USD file loading may fail.");
    QL_LOG_WARN("Set USD_ROOT or PXR_PLUGINPATH_NAME environment variable to fix this.");
}

// ============================================================================
// IsAvailable - Check if OpenUSD support is compiled in
// ============================================================================

bool UsdLoader::IsAvailable() {
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

static glm::mat4 GfMatrix4dToGlm(const GfMatrix4d& mat) {
    glm::mat4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = static_cast<float>(mat[i][j]);
        }
    }
    return result;
}

static UsdTimeCode GetTimeCode(const UsdLoadOptions& options) {
    if (options.useDefaultTime) {
        return UsdTimeCode::Default();
    }
    return UsdTimeCode(options.timeCode);
}

// ============================================================================
// Stage metrics - upAxis and metersPerUnit as one root transform
// ============================================================================

/// The transform that puts a stage's own conventions into Quantiloom's: Y-up,
/// and metres if the stage said what its units are.
///
/// metersPerUnit is folded only when the stage actually authored it. USD's
/// default when the metadata is absent is 0.01, and applying that silently would
/// shrink by a hundred every scene that has been loading correctly, so an
/// unauthored stage is left at 1.0 and says so once.
///
/// This is geometry, which is why it is not `scene.world_units_to_meters`: that
/// key scales lighting, the camera and the thermal solve (ConfigResolve.cpp:340,
/// 507, 751) and never touches a vertex, and one config scalar cannot describe a
/// scene assembled from models authored in different units.
static glm::mat4 StageRootTransform(const UsdStageWeakPtr& stage,
                                    const UsdLoadOptions& options) {
    if (!options.applyStageMetrics || !stage) {
        return glm::mat4(1.0f);
    }

    glm::mat4 root(1.0f);

    if (UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z) {
        root = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                           glm::vec3(1.0f, 0.0f, 0.0f));
        QL_LOG_INFO("  Stage is Z-up; rotating the scene root -90 degrees about X");
    }

    if (stage->HasAuthoredMetadata(UsdGeomTokens->metersPerUnit)) {
        const double mpu = UsdGeomGetStageMetersPerUnit(stage);
        if (mpu > 0.0 && mpu != 1.0) {
            // Uniform, so it commutes with the rotation above.
            root = glm::scale(glm::mat4(1.0f), glm::vec3(static_cast<f32>(mpu))) * root;
            QL_LOG_INFO("  Stage metersPerUnit = {}; folded into the scene root", mpu);
        }
    } else {
        QL_LOG_WARN("  Stage authored no metersPerUnit. USD's default is 0.01, but "
                    "applying it would shrink every scene that loads correctly today "
                    "by a hundred, so 1.0 is used. Author the metadata to say otherwise.");
    }

    return root;
}

/// Whether a prim is something to render.
///
/// `guide` is annotation and `proxy` is the cheap stand-in a render prim
/// replaces, so loading a proxy alongside its render prim puts two copies of the
/// model in the scene; an invisible prim is one the stage asked not to see.
static bool IsRenderablePrim(const UsdPrim& prim, const UsdTimeCode& timeCode) {
    UsdGeomImageable imageable(prim);
    if (!imageable) {
        return true;
    }
    const TfToken purpose = imageable.ComputePurpose();
    if (purpose == UsdGeomTokens->guide || purpose == UsdGeomTokens->proxy) {
        return false;
    }
    return imageable.ComputeVisibility(timeCode) != UsdGeomTokens->invisible;
}

// ============================================================================
// Vertex attributes and face-vertex expansion
// ============================================================================

/// One authored vertex attribute together with the interpolation USD gave it.
///
/// Keeping the two together is the point. A mesh whose normals are face-varying
/// and whose UVs are per-vertex has to expand the normals and re-index the UVs;
/// reading both through one shared assumption is what used to move an empty
/// vector over whichever of the two did not trigger the expansion.
template <typename T>
struct AttrSource {
    const VtArray<T>* data = nullptr;
    TfToken interpolation;

    [[nodiscard]] bool Present() const { return data != nullptr && !data->empty(); }

    /// Face-varying and uniform attributes address something the point list
    /// cannot; everything else is already indexed by the triangulation.
    [[nodiscard]] bool NeedsExpansion() const {
        return Present() && (interpolation == UsdGeomTokens->faceVarying ||
                             interpolation == UsdGeomTokens->uniform);
    }

    /// Where one face vertex reads its value, or -1 when the attribute is absent
    /// or too short to answer.
    [[nodiscard]] i64 IndexFor(size_t fvIndex, size_t faceIdx, size_t pointIndex) const {
        if (!Present()) {
            return -1;
        }
        size_t index;
        if (interpolation == UsdGeomTokens->faceVarying) {
            index = fvIndex;
        } else if (interpolation == UsdGeomTokens->uniform) {
            index = faceIdx;
        } else if (interpolation == UsdGeomTokens->constant) {
            index = 0;
        } else {  // vertex, varying, or an interpolation nobody authored
            index = pointIndex;
        }
        return index < data->size() ? static_cast<i64>(index) : -1;
    }
};

/// A mesh with one vertex per face vertex, every attribute read through its own
/// interpolation.
struct ExpandedMesh {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;
    std::vector<u32> indices;
    std::vector<u32> triangleToFace;
};

static void ExpandToFaceVertices(const VtArray<int>& faceVertexCounts,
                                 const VtArray<int>& faceVertexIndices,
                                 const std::vector<glm::vec3>& positions,
                                 const AttrSource<GfVec3f>& normalSource,
                                 const AttrSource<GfVec2f>& uvSource,
                                 const AttrSource<GfVec4f>& tangentSource,
                                 ExpandedMesh& out) {
    const size_t faceVertexTotal = faceVertexIndices.size();
    out.positions.reserve(faceVertexTotal);
    if (normalSource.Present()) { out.normals.reserve(faceVertexTotal); }
    if (uvSource.Present()) { out.uvs.reserve(faceVertexTotal); }
    if (tangentSource.Present()) { out.tangents.reserve(faceVertexTotal); }

    bool normalsComplete = normalSource.Present();
    bool uvsComplete = uvSource.Present();
    bool tangentsComplete = tangentSource.Present();

    size_t fvOffset = 0;
    u32 nextVertex = 0;

    for (size_t faceIdx = 0; faceIdx < faceVertexCounts.size(); ++faceIdx) {
        const int vertexCount = faceVertexCounts[faceIdx];
        if (vertexCount < 3 ||
            fvOffset + static_cast<size_t>(vertexCount) > faceVertexTotal) {
            fvOffset += static_cast<size_t>(std::max(vertexCount, 0));
            continue;
        }

        // Fan triangulation: corner 0 with every adjacent pair after it.
        for (int i = 1; i < vertexCount - 1; ++i) {
            const size_t fv[3] = {fvOffset,
                                  fvOffset + static_cast<size_t>(i),
                                  fvOffset + static_cast<size_t>(i) + 1};

            // Validate the whole triangle before emitting any of it, or a bad
            // corner would leave a two-vertex triangle behind and shift every
            // index after it.
            size_t point[3];
            bool triangleOk = true;
            for (int c = 0; c < 3; ++c) {
                const int index = faceVertexIndices[fv[c]];
                if (index < 0 || static_cast<size_t>(index) >= positions.size()) {
                    triangleOk = false;
                    break;
                }
                point[c] = static_cast<size_t>(index);
            }
            if (!triangleOk) {
                continue;
            }

            for (int c = 0; c < 3; ++c) {
                out.positions.push_back(positions[point[c]]);

                if (normalSource.Present()) {
                    const i64 n = normalSource.IndexFor(fv[c], faceIdx, point[c]);
                    if (n < 0) {
                        normalsComplete = false;
                    } else {
                        const GfVec3f& value = (*normalSource.data)[static_cast<size_t>(n)];
                        out.normals.emplace_back(value[0], value[1], value[2]);
                    }
                }
                if (uvSource.Present()) {
                    const i64 u = uvSource.IndexFor(fv[c], faceIdx, point[c]);
                    if (u < 0) {
                        uvsComplete = false;
                    } else {
                        const GfVec2f& value = (*uvSource.data)[static_cast<size_t>(u)];
                        out.uvs.push_back(usd::UsdStToUv(value[0], value[1]));
                    }
                }
                if (tangentSource.Present()) {
                    const i64 t = tangentSource.IndexFor(fv[c], faceIdx, point[c]);
                    if (t < 0) {
                        tangentsComplete = false;
                    } else {
                        const GfVec4f& value = (*tangentSource.data)[static_cast<size_t>(t)];
                        out.tangents.emplace_back(value[0], value[1], value[2], value[3]);
                    }
                }
                out.indices.push_back(nextVertex++);
            }

            out.triangleToFace.push_back(static_cast<u32>(faceIdx));
        }

        fvOffset += static_cast<size_t>(vertexCount);
    }

    // An attribute that could not answer for every face vertex is dropped rather
    // than shipped short: Mesh::IsValid() requires it to match the vertex count,
    // and NormalGenerator rebuilds a missing normal set.
    if (!normalsComplete || out.normals.size() != out.positions.size()) {
        if (!out.normals.empty()) {
            QL_LOG_WARN("    Normals do not cover every face vertex; regenerating them");
        }
        out.normals.clear();
    }
    if (!uvsComplete || out.uvs.size() != out.positions.size()) {
        if (!out.uvs.empty()) {
            QL_LOG_WARN("    UVs do not cover every face vertex; dropping them");
        }
        out.uvs.clear();
    }
    if (!tangentsComplete || out.tangents.size() != out.positions.size()) {
        if (!out.tangents.empty()) {
            QL_LOG_WARN("    Tangents do not cover every face vertex; deriving them "
                        "from the UVs instead");
        }
        out.tangents.clear();
    }
}

/// Fill one entry per point, reading through the source's own interpolation.
/// Constant, vertex and varying all answer per point; anything that does not is
/// left empty rather than short.
template <typename T, typename Out, typename Convert>
static void FillPerPoint(const AttrSource<T>& source, size_t pointCount,
                         std::vector<Out>& out, Convert convert) {
    out.clear();
    if (!source.Present()) {
        return;
    }
    out.reserve(pointCount);
    for (size_t i = 0; i < pointCount; ++i) {
        const i64 index = source.IndexFor(i, 0, i);
        if (index < 0) {
            QL_LOG_WARN("    A vertex attribute covers {} of {} points; dropping it",
                        source.data->size(), pointCount);
            out.clear();
            return;
        }
        out.push_back(convert((*source.data)[static_cast<size_t>(index)]));
    }
}

// ============================================================================
// TriangulatePolygons - Convert polygon faces to triangles
// ============================================================================

std::vector<u32> UsdLoader::TriangulatePolygons(
    const std::vector<i32>& faceVertexCounts,
    const std::vector<i32>& faceVertexIndices) {

    std::vector<u32> triangleIndices;

    size_t indexOffset = 0;
    for (const i32 vertexCount : faceVertexCounts) {
        if (vertexCount < 3) {
            indexOffset += vertexCount;
            continue;
        }

        // Fan triangulation for convex polygons
        for (i32 i = 1; i < vertexCount - 1; ++i) {
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i + 1]));
        }

        indexOffset += vertexCount;
    }

    return triangleIndices;
}

std::vector<u32> UsdLoader::TriangulatePolygonsWithFaceMap(
    const std::vector<i32>& faceVertexCounts,
    const std::vector<i32>& faceVertexIndices,
    std::vector<u32>& outTriangleToFace) {

    std::vector<u32> triangleIndices;
    outTriangleToFace.clear();

    size_t indexOffset = 0;
    for (size_t faceIdx = 0; faceIdx < faceVertexCounts.size(); ++faceIdx) {
        i32 vertexCount = faceVertexCounts[faceIdx];
        if (vertexCount < 3) {
            indexOffset += vertexCount;
            continue;
        }

        // Fan triangulation for convex polygons
        for (i32 i = 1; i < vertexCount - 1; ++i) {
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i]));
            triangleIndices.push_back(static_cast<u32>(faceVertexIndices[indexOffset + i + 1]));
            outTriangleToFace.push_back(static_cast<u32>(faceIdx));
        }

        indexOffset += vertexCount;
    }

    return triangleIndices;
}

// ============================================================================
// ParseTexture - Resolve an asset path and decode it
// ============================================================================

Texture UsdLoader::ParseTexture(const void* /* stagePtr */, const String& assetPath,
                                const String& usdFilePath) {
    if (assetPath.empty()) {
        return {};
    }

    std::filesystem::path fullPath(assetPath);
    if (!fullPath.is_absolute()) {
        fullPath = std::filesystem::path(usdFilePath).parent_path() / fullPath;
    }
    fullPath = std::filesystem::weakly_canonical(fullPath);

    if (!std::filesystem::exists(fullPath)) {
        QL_LOG_ERROR("Texture file not found: {}", fullPath.string());
        return {};
    }
    return usd::DecodeTextureFile(fullPath.string());
}

/// A `quantiloom:` asset attribute's absolute path. The resolver's answer wins,
/// so a `.usdz`'s internal layout works.
static String ResolveQuantiloomAsset(const SdfAssetPath& asset, const String& usdFilePath) {
    if (!asset.GetResolvedPath().empty()) {
        return asset.GetResolvedPath();
    }
    std::filesystem::path path(asset.GetAssetPath());
    if (!path.is_absolute()) {
        path = std::filesystem::path(usdFilePath).parent_path() / path;
    }
    return std::filesystem::weakly_canonical(path).string();
}

// ============================================================================
// ParseSpectralExtensions - Parse Quantiloom custom attributes
// ============================================================================

static void ParseSpectralExtensions(Material& mat, const void* primPtr,
                                    const String& usdFilePath) {
    if (!primPtr) {
        return;
    }

    const auto* prim = static_cast<const UsdPrim*>(primPtr);
    std::filesystem::path usdDir = std::filesystem::path(usdFilePath).parent_path();

    // Parse Quantiloom material reference
    if (UsdAttribute typeAttr = prim->GetAttribute(TfToken("quantiloom:materialType"))) {
        std::string typeStr;
        if (typeAttr.Get(&typeStr)) {
            mat.quantiloomMaterialType = typeStr;
        }
    }

    if (UsdAttribute refAttr = prim->GetAttribute(TfToken("quantiloom:materialRef"))) {
        std::string refStr;
        if (refAttr.Get(&refStr)) {
            mat.quantiloomMaterialRef = refStr;
        }
    }

    if (mat.HasQuantiloomRef()) {
        QL_LOG_INFO("    Found Quantiloom material reference: type='{}', name='{}'",
                    mat.quantiloomMaterialType, mat.quantiloomMaterialRef);
        mat.spectralSource = Material::SpectralSource::Measured;
    }

    // Helper lambda to load spectral curve from asset path attribute
    auto loadSpectralCurve = [&](const char* attrName) -> std::optional<std::vector<std::pair<f32, f32>>> {
        if (UsdAttribute curveAttr = prim->GetAttribute(TfToken(attrName))) {
            std::string curvePath;
            if (curveAttr.Get(&curvePath) && !curvePath.empty()) {
                std::filesystem::path fullPath = usdDir / curvePath;
                auto result = SpectralIO::LoadSpectralCurveCSV(fullPath);
                if (result.has_value()) {
                    QL_LOG_INFO("      Loaded {}: {} ({} points)",
                                attrName, curvePath, result.value().size());
                    return result.value();
                } else {
                    QL_LOG_ERROR("      Failed to load {}: '{}': {}",
                                 attrName, curvePath, result.error());
                }
            }
        }
        return std::nullopt;
    };

    // Load spectral curves
    if (auto curve = loadSpectralCurve("quantiloom:emissivityCurve")) {
        mat.irEmissivityCurve = std::move(*curve);
    }
    if (auto curve = loadSpectralCurve("quantiloom:reflectanceCurve")) {
        mat.irReflectanceCurve = std::move(*curve);
    }
    if (auto curve = loadSpectralCurve("quantiloom:transmittanceCurve")) {
        mat.irTransmittanceCurve = std::move(*curve);
    }

    // Load IR temperature
    if (UsdAttribute tempAttr = prim->GetAttribute(TfToken("quantiloom:temperature_K"))) {
        float temp;
        if (tempAttr.Get(&temp)) {
            mat.irTemperature_K = temp;
            QL_LOG_INFO("      IR temperature: {:.1f} K", mat.irTemperature_K);
        }
    }

    // Parse temperature texture scale and offset (texture loaded in ParseMaterial)
    if (UsdAttribute scaleAttr = prim->GetAttribute(TfToken("quantiloom:temperatureScale"))) {
        float scale;
        if (scaleAttr.Get(&scale)) {
            mat.temperatureScale = scale;
            QL_LOG_INFO("      temperatureScale: {:.1f}", scale);
        }
    }
    if (UsdAttribute offsetAttr = prim->GetAttribute(TfToken("quantiloom:temperatureOffset"))) {
        float offset;
        if (offsetAttr.Get(&offset)) {
            mat.temperatureOffset = offset;
            QL_LOG_INFO("      temperatureOffset: {:.1f} K", offset);
        }
    }

    // ========================================================================
    // Fluorescence
    // ========================================================================
    // The same three quantities QUANTILOOM_materials_fluorescence carries in
    // glTF, spelled the same way: light absorbed at one wavelength and given
    // back at another, rank one. The normalisation and the energy rules are
    // ResolveFluorescence's, so both formats and the TOML keys reach one
    // implementation; what lands here is samples.
    //
    // No emissiveFactor is written, deliberately. A fluorescent surface emits
    // only what something else lit it with, and the emitter-sampling table
    // collects triangles by luminance(emissiveFactor): a triple here would have
    // next-event estimation aim at a surface that is dark on its own.
    if (auto curve = loadSpectralCurve("quantiloom:fluorescenceExcitationCurve")) {
        mat.fluorescenceExcitationCurve = std::move(*curve);
    }
    if (auto curve = loadSpectralCurve("quantiloom:fluorescenceEmissionCurve")) {
        mat.fluorescenceEmissionCurve = std::move(*curve);
    }
    if (UsdAttribute yieldAttr = prim->GetAttribute(TfToken("quantiloom:fluorescenceYield"))) {
        float fluorescenceYield = 0.0f;
        if (yieldAttr.Get(&fluorescenceYield)) {
            mat.fluorescenceYield = fluorescenceYield;
            QL_LOG_INFO("      fluorescenceYield: {:.4g}", mat.fluorescenceYield);
        }
    }

    // One half is not a description of anything, and a scene that carries one
    // half is more likely to have a typo than an intention. Said here rather
    // than left to the binder, which never sees a material that dropped a curve
    // on a failed load.
    if (mat.fluorescenceExcitationCurve.empty() != mat.fluorescenceEmissionCurve.empty()) {
        QL_LOG_WARN("    Material '{}': only one of the two fluorescence curves loaded, "
                    "so the material will not fluoresce", mat.name);
    }
    if (!mat.fluorescenceExcitationCurve.empty() && !mat.fluorescenceEmissionCurve.empty()) {
        // Where the pair came from, for the energy warning ResolveFluorescence
        // prints: "the scene file" is no help when the scene is an assembly.
        mat.fluorescenceSource = usdFilePath + "#" + prim->GetPath().GetString();
    }

    // ========================================================================
    // Dispersion
    // ========================================================================
    // Already the reciprocal Abbe number Material holds, unlike
    // KHR_materials_dispersion's 20/V. Read after the surface shader, so it
    // overrides a vocabulary's own dispersion the way the glTF extension
    // overrides the ratified one.
    if (UsdAttribute dispersionAttr = prim->GetAttribute(TfToken("quantiloom:dispersion"))) {
        float dispersion = 0.0f;
        if (dispersionAttr.Get(&dispersion)) {
            mat.dispersion = dispersion;
            QL_LOG_INFO("      dispersion: {:.6f} (1/Abbe)", mat.dispersion);
        }
    }

    // Mark as measured if IR data loaded
    if (mat.HasIRData()) {
        mat.spectralSource = Material::SpectralSource::Measured;
        if (!mat.ValidateIRKirchhoffLaw()) {
            QL_LOG_WARN("    Material '{}' violates Kirchhoff's law", mat.name);
        }
    }
}

// ============================================================================
// ClassifySurface - Which vocabulary a material's surface shader is written in
// ============================================================================

static usd::SurfaceVocabulary ClassifySurface(const UsdShadeShader& surfaceShader,
                                              const String& materialPath) {
    if (!surfaceShader) {
        return usd::SurfaceVocabulary::Unknown;
    }

    TfToken shaderId;
    surfaceShader.GetIdAttr().Get(&shaderId);
    const usd::SurfaceVocabulary vocabulary = usd::ClassifyShaderId(shaderId.GetString());

    if (vocabulary == usd::SurfaceVocabulary::Unknown) {
        // Reading it through a vocabulary it is not written in matches no input
        // name and produces the default grey, which looks like a material that
        // was authored badly rather than one nobody here can read.
        QL_LOG_WARN("    Material '{}': surface shader id '{}' is not a vocabulary this "
                    "loader knows; the material keeps its defaults",
                    materialPath, shaderId.GetString());
    } else {
        QL_LOG_INFO("    Material '{}' is {}", materialPath,
                    usd::VocabularyName(vocabulary));
    }
    return vocabulary;
}

// ============================================================================
// ParseMaterial - Convert UsdShadeMaterial to Quantiloom Material
// ============================================================================
// The surface itself was already read in Pass 0, because the texture files have
// to be known before any of them is decoded. This turns that reading into a
// Material: scalars first, then textures, because whether the base colour entry
// keeps its CPU pixels depends on the alpha mode the scalars decide.

static Material ParseMaterial(const UsdPrim& prim, const usd::SurfaceReading& reading,
                              usd::UsdTextureBank& bank, std::vector<Texture>& textures,
                              const String& usdFilePath) {
    Material mat;

    // Quantiloom's own defaults, for a material whose shader said nothing.
    mat.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
    mat.metallicFactor = 0.0f;
    mat.roughnessFactor = 0.5f;
    mat.emissiveFactor = glm::vec3(0.0f);
    mat.alphaMode = Material::AlphaMode::Opaque;
    mat.alphaCutoff = 0.5f;
    mat.name = prim.GetName().GetString();

    usd::ConvertSurface(reading, mat);
    usd::ApplySlotTextures(reading, mat, bank, textures);

    mat.ComputeSpectralAlbedo();
    mat.spectralSource = Material::SpectralSource::RGBUpsampled;

    // After the assignment above, never before it: ParseSpectralExtensions is
    // what promotes a material to Measured, and the band gate in
    // ConfigResolve reads that.
    ParseSpectralExtensions(mat, &prim, usdFilePath);

    // The temperature texture is data, so it is linear whatever the file is.
    if (UsdAttribute tempTexAttr = prim.GetAttribute(TfToken("quantiloom:temperatureTexture"))) {
        SdfAssetPath texPath;
        if (tempTexAttr.Get(&texPath) && !texPath.GetAssetPath().empty()) {
            usd::SlotRecipe recipe;
            usd::SlotRecipe::Src source;
            source.path = ResolveQuantiloomAsset(texPath, usdFilePath);
            source.channel = usd::ChannelSel::R;
            recipe.dst[0] = source;
            recipe.srgb = false;
            recipe.debugName = "temperature";
            mat.temperatureTextureIndex = bank.Materialise(recipe, textures);
            if (mat.temperatureTextureIndex >= 0) {
                QL_LOG_INFO("    Loaded temperature texture: index {}",
                            mat.temperatureTextureIndex);
            }
        }
    }

    return mat;
}

// ============================================================================
// ParseMesh - Convert UsdGeomMesh to Quantiloom Mesh
// ============================================================================

Mesh UsdLoader::ParseMesh(const void* stagePtr, const void* primPtr,
                          const std::unordered_map<String, int>& materialPathMap,
                          const std::vector<Material>& materials,
                          const String& usdFilePath,
                          const UsdLoadOptions& options) {
    Mesh mesh;

    if (!primPtr) {
        return mesh;
    }

    const auto* prim = static_cast<const UsdPrim*>(primPtr);
    UsdGeomMesh geomMesh(*prim);
    mesh.name = prim->GetName().GetString();

    UsdTimeCode timeCode = GetTimeCode(options);

    // ========================================================================
    // Get vertex positions (points)
    // ========================================================================
    VtArray<GfVec3f> points;
    if (!geomMesh.GetPointsAttr().Get(&points, timeCode)) {
        QL_LOG_WARN("    Mesh '{}' has no points", mesh.name);
        return mesh;
    }

    std::vector<glm::vec3> positions;
    positions.reserve(points.size());
    for (const auto& p : points) {
        positions.emplace_back(p[0], p[1], p[2]);
    }

    // ========================================================================
    // Get face topology
    // ========================================================================
    VtArray<int> faceVertexCounts, faceVertexIndices;
    geomMesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts, timeCode);
    geomMesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices, timeCode);

    std::vector<i32> fvcVec(faceVertexCounts.begin(), faceVertexCounts.end());
    std::vector<i32> fviVec(faceVertexIndices.begin(), faceVertexIndices.end());

    // Triangulate with face mapping for GeomSubsets
    std::vector<u32> triangleToFace;
    std::vector<u32> indices = TriangulatePolygonsWithFaceMap(fvcVec, fviVec, triangleToFace);

    // ========================================================================
    // Check subdivision scheme
    // ========================================================================
    TfToken subdivisionScheme;
    geomMesh.GetSubdivisionSchemeAttr().Get(&subdivisionScheme, timeCode);
    bool isSubdivisionSurface = (subdivisionScheme == UsdGeomTokens->catmullClark ||
                                  subdivisionScheme == UsdGeomTokens->loop ||
                                  subdivisionScheme == UsdGeomTokens->bilinear);

    if (isSubdivisionSurface) {
        QL_LOG_INFO("    Mesh '{}' is subdivision surface (scheme: {}), treating as polygon mesh",
                    mesh.name, subdivisionScheme.GetString());
        // Note: Full subdivision would require a subdivision library (OpenSubdiv)
        // For now, we treat it as a polygon mesh and compute smooth normals later
    }

    // ========================================================================
    // Get normals and UVs
    // ========================================================================
    // Both are read as an AttrSource -- the values plus the interpolation USD
    // authored them with -- rather than being flattened here, because the two
    // may disagree: face-varying normals on a mesh with per-vertex UVs need the
    // geometry expanded for the normals and the UVs re-indexed, not discarded.
    UsdGeomPrimvarsAPI primvarsAPI(*prim);

    // `primvars:normals` takes precedence over the `normals` attribute, which is
    // what UsdGeomPointBased says. The attribute cannot be read through
    // UsdGeomPrimvar at all: wrapping a name outside the `primvars:` namespace
    // gives an object that is never IsDefined(), so the fallback here used to be
    // dead code and every plain `normals` array was dropped and rebuilt by
    // NormalGenerator -- a mesh's authored shading silently replaced by the
    // dihedral-angle guess.
    VtArray<GfVec3f> usdNormals;
    AttrSource<GfVec3f> normalSource;

    if (UsdGeomPrimvar normalsPrimvar = primvarsAPI.GetPrimvar(TfToken("normals"));
        normalsPrimvar && normalsPrimvar.HasValue()) {
        normalsPrimvar.Get(&usdNormals, timeCode);
        normalSource = AttrSource<GfVec3f>{&usdNormals, normalsPrimvar.GetInterpolation()};
    } else if (UsdAttribute normalsAttr = geomMesh.GetNormalsAttr();
               normalsAttr && normalsAttr.HasValue()) {
        normalsAttr.Get(&usdNormals, timeCode);
        normalSource = AttrSource<GfVec3f>{&usdNormals, geomMesh.GetNormalsInterpolation()};
    }

    if (normalSource.Present()) {
        QL_LOG_INFO("    Mesh '{}' normals: {} values, interpolation={}",
                    mesh.name, usdNormals.size(),
                    normalSource.interpolation.GetString());
    }

    UsdGeomPrimvar uvPrimvar = primvarsAPI.GetPrimvar(TfToken("st"));
    if (!uvPrimvar) {
        uvPrimvar = primvarsAPI.GetPrimvar(TfToken("uv"));
    }

    VtArray<GfVec2f> usdUVs;
    AttrSource<GfVec2f> uvSource;
    if (uvPrimvar && uvPrimvar.HasValue()) {
        uvPrimvar.Get(&usdUVs, timeCode);
        uvSource = AttrSource<GfVec2f>{&usdUVs, uvPrimvar.GetInterpolation()};
    }

    // `primvars:tangents` is the authored tangent frame. USD has no convention
    // for its handedness, so a float3 is taken as right-handed (+1) and a float4
    // carries its own sign, which is what glTF's TANGENT does.
    VtArray<GfVec4f> usdTangents;
    AttrSource<GfVec4f> tangentSource;
    if (UsdGeomPrimvar tangentPrimvar = primvarsAPI.GetPrimvar(TfToken("tangents"));
        tangentPrimvar && tangentPrimvar.HasValue()) {
        if (!tangentPrimvar.Get(&usdTangents, timeCode)) {
            VtArray<GfVec3f> threeComponent;
            if (tangentPrimvar.Get(&threeComponent, timeCode)) {
                usdTangents.reserve(threeComponent.size());
                for (const GfVec3f& tangent : threeComponent) {
                    usdTangents.push_back(GfVec4f(tangent[0], tangent[1], tangent[2], 1.0f));
                }
            }
        }
        tangentSource = AttrSource<GfVec4f>{&usdTangents, tangentPrimvar.GetInterpolation()};
    }

    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;

    // ========================================================================
    // Face-varying or uniform attributes need one vertex per face vertex
    // ========================================================================
    if (normalSource.NeedsExpansion() || uvSource.NeedsExpansion() ||
        tangentSource.NeedsExpansion()) {
        ExpandedMesh expanded;
        ExpandToFaceVertices(faceVertexCounts, faceVertexIndices, positions,
                             normalSource, uvSource, tangentSource, expanded);
        positions      = std::move(expanded.positions);
        normals        = std::move(expanded.normals);
        uvs            = std::move(expanded.uvs);
        tangents       = std::move(expanded.tangents);
        indices        = std::move(expanded.indices);
        triangleToFace = std::move(expanded.triangleToFace);
    } else {
        // Constant, vertex and varying all answer per point, which is what the
        // triangulation above already indexes.
        FillPerPoint(normalSource, positions.size(), normals,
                     [](const GfVec3f& n) { return glm::vec3(n[0], n[1], n[2]); });
        FillPerPoint(uvSource, positions.size(), uvs,
                     [](const GfVec2f& uv) { return usd::UsdStToUv(uv[0], uv[1]); });
        FillPerPoint(tangentSource, positions.size(), tangents,
                     [](const GfVec4f& t) { return glm::vec4(t[0], t[1], t[2], t[3]); });
    }

    // ========================================================================
    // Note: Normals will be generated later by NormalGenerator if missing
    // ========================================================================

    // ========================================================================
    // Handle GeomSubsets (multi-material per mesh)
    // ========================================================================
    std::vector<UsdGeomSubset> geomSubsets = UsdGeomSubset::GetAllGeomSubsets(geomMesh);

    // Get default material binding
    UsdShadeMaterialBindingAPI bindingAPI(*prim);
    UsdShadeMaterial defaultBoundMat = bindingAPI.ComputeBoundMaterial();
    int defaultMaterialId = 0;

    if (defaultBoundMat) {
        String matPath = defaultBoundMat.GetPrim().GetPath().GetString();
        if (auto it = materialPathMap.find(matPath); it != materialPathMap.end()) {
            defaultMaterialId = it->second;
        }
    }

    if (options.enableGeomSubsets && !geomSubsets.empty()) {
        // Create separate primitives for each GeomSubset
        std::unordered_set<u32> assignedFaces;

        for (const auto& subset : geomSubsets) {
            // Check if this is a material binding subset
            TfToken familyName;
            if (subset.GetFamilyNameAttr().Get(&familyName)) {
                // Skip non-material subsets (e.g., "materialBind" is the standard family for materials)
                if (!familyName.IsEmpty() && familyName != TfToken("materialBind")) {
                    continue;
                }
            }

            // Get face indices for this subset
            VtArray<int> subsetIndices;
            subset.GetIndicesAttr().Get(&subsetIndices, timeCode);

            if (subsetIndices.empty()) {
                continue;
            }

            // Get material binding for this subset
            UsdShadeMaterialBindingAPI subsetBindingAPI(subset.GetPrim());
            UsdShadeMaterial subsetMat = subsetBindingAPI.ComputeBoundMaterial();
            int subsetMaterialId = defaultMaterialId;

            if (subsetMat) {
                String matPath = subsetMat.GetPrim().GetPath().GetString();
                if (auto it = materialPathMap.find(matPath); it != materialPathMap.end()) {
                    subsetMaterialId = it->second;
                }
            }

            // Create set of faces in this subset
            std::unordered_set<u32> subsetFaces(subsetIndices.begin(), subsetIndices.end());

            // Create primitive for this subset
            GeometryPrimitive primitive;
            primitive.materialId = subsetMaterialId;

            // Collect triangles belonging to this subset
            for (size_t triIdx = 0; triIdx < triangleToFace.size(); ++triIdx) {
                u32 faceIdx = triangleToFace[triIdx];
                if (subsetFaces.count(faceIdx) > 0) {
                    u32 baseIdx = static_cast<u32>(triIdx) * 3;
                    primitive.indices.push_back(indices[baseIdx]);
                    primitive.indices.push_back(indices[baseIdx + 1]);
                    primitive.indices.push_back(indices[baseIdx + 2]);
                    assignedFaces.insert(faceIdx);
                }
            }

            if (!primitive.indices.empty()) {
                primitive.positions = positions;
                primitive.normals = normals;
                primitive.uvs = uvs;
                primitive.tangents = tangents;
                mesh.primitives.push_back(std::move(primitive));
            }
        }

        // Create primitive for unassigned faces
        GeometryPrimitive remainingPrimitive;
        remainingPrimitive.materialId = defaultMaterialId;

        for (size_t triIdx = 0; triIdx < triangleToFace.size(); ++triIdx) {
            u32 faceIdx = triangleToFace[triIdx];
            if (assignedFaces.count(faceIdx) == 0) {
                u32 baseIdx = static_cast<u32>(triIdx) * 3;
                remainingPrimitive.indices.push_back(indices[baseIdx]);
                remainingPrimitive.indices.push_back(indices[baseIdx + 1]);
                remainingPrimitive.indices.push_back(indices[baseIdx + 2]);
            }
        }

        if (!remainingPrimitive.indices.empty()) {
            remainingPrimitive.positions = positions;
            remainingPrimitive.normals = normals;
            remainingPrimitive.uvs = uvs;
            remainingPrimitive.tangents = tangents;
            mesh.primitives.push_back(std::move(remainingPrimitive));
        }
    } else {
        // No GeomSubsets - create single primitive
        GeometryPrimitive primitive;
        primitive.positions = std::move(positions);
        primitive.normals = std::move(normals);
        primitive.uvs = std::move(uvs);
        primitive.tangents = std::move(tangents);
        primitive.indices = std::move(indices);
        primitive.materialId = defaultMaterialId;
        mesh.primitives.push_back(std::move(primitive));
    }

    // ========================================================================
    // Generate normals BEFORE deduplication (CRITICAL ORDER)
    // ========================================================================
    // NormalGenerator::GenerateWithDihedralAngle() duplicates vertices at hard edges.
    // Deduplication must run AFTER to preserve these intentional splits.
    // Swapping this order causes hard edges to be smoothed incorrectly.
    for (auto& primitive : mesh.primitives) {
        if (primitive.normals.empty()) {
            QL_LOG_DEBUG("    Generating normals for USD primitive with dihedral angle threshold");
            NormalGenerator::GenerateWithDihedralAngle(primitive);
        }
    }

    // ========================================================================
    // Tangents, for the materials that read one
    // ========================================================================
    // After the normals, because the frame is orthogonalised against them and
    // NormalGenerator duplicates vertices at hard edges; before deduplication,
    // which keys on the tangent. Only where a material actually reads one: a
    // frame nobody samples is bytes on the GPU for nothing.
    for (auto& primitive : mesh.primitives) {
        if (!primitive.tangents.empty() || primitive.uvs.empty()) {
            continue;
        }
        if (primitive.materialId < 0 ||
            static_cast<usize>(primitive.materialId) >= materials.size()) {
            continue;
        }
        const Material& material = materials[static_cast<usize>(primitive.materialId)];
        const bool wantsTangents = material.HasAnisotropy() ||
                                   material.normalTextureIndex >= 0 ||
                                   material.clearcoatNormalTextureIndex >= 0;
        if (wantsTangents && TangentGenerator::FromUv(primitive)) {
            QL_LOG_DEBUG("    Derived tangents for '{}' from its UVs", mesh.name);
        }
    }

    // ========================================================================
    // Vertex Deduplication (AFTER normal generation)
    // ========================================================================
    // USD face-varying expansion creates many duplicate vertices at shared edges.
    // Deduplicate to reduce memory and improve cache efficiency.
    // This preserves hard edges created by NormalGenerator.
    auto dedupeStats = MeshOptimizer::DeduplicateMesh(mesh);
    if (dedupeStats.WasOptimized()) {
        QL_LOG_INFO("    Vertex dedup for '{}': {} -> {} vertices ({:.1f}% reduction)",
                    mesh.name, dedupeStats.originalVertexCount, dedupeStats.optimizedVertexCount,
                    dedupeStats.vertexReductionPercent);
    }

    size_t totalVerts = 0, totalTris = 0;
    for (const auto& p : mesh.primitives) {
        totalVerts += p.GetVertexCount();
        totalTris += p.GetTriangleCount();
    }
    QL_LOG_INFO("    Loaded mesh '{}': {} vertices, {} triangles, {} primitives",
                mesh.name, totalVerts, totalTris, mesh.primitives.size());

    // ========================================================================
    // DIAGNOSTIC: Dump primitive data for debugging cube rendering artifacts
    // ========================================================================
    // This diagnostic helps identify issues where triangles on the same face
    // show different normals (e.g., diagonal normals instead of axis-aligned).
    // ========================================================================
    for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
        const auto& primitive = mesh.primitives[primIdx];
        QL_LOG_DEBUG("USD Primitive {} dump:", primIdx);
        QL_LOG_DEBUG("  Positions: {} vertices", primitive.positions.size());
        QL_LOG_DEBUG("  Normals: {} normals", primitive.normals.size());
        QL_LOG_DEBUG("  Indices: {} indices ({} triangles)",
                     primitive.indices.size(), primitive.indices.size() / 3);

        // Dump first 8 vertices
        for (size_t i = 0; i < std::min(size_t(8), primitive.positions.size()); ++i) {
            QL_LOG_DEBUG("    pos[{}] = ({:.4f}, {:.4f}, {:.4f})", i,
                primitive.positions[i].x, primitive.positions[i].y, primitive.positions[i].z);
        }

        // Dump first few triangles with their computed geometric normals
        size_t numTrisToDump = std::min(size_t(12), primitive.indices.size() / 3);
        for (size_t triIdx = 0; triIdx < numTrisToDump; ++triIdx) {
            u32 idx0 = primitive.indices[triIdx * 3 + 0];
            u32 idx1 = primitive.indices[triIdx * 3 + 1];
            u32 idx2 = primitive.indices[triIdx * 3 + 2];

            if (idx0 >= primitive.positions.size() ||
                idx1 >= primitive.positions.size() ||
                idx2 >= primitive.positions.size()) {
                QL_LOG_ERROR("  Triangle {}: INVALID INDICES [{}, {}, {}] (max={})",
                             triIdx, idx0, idx1, idx2, primitive.positions.size() - 1);
                continue;
            }

            const glm::vec3& v0 = primitive.positions[idx0];
            const glm::vec3& v1 = primitive.positions[idx1];
            const glm::vec3& v2 = primitive.positions[idx2];

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 geoNormal = glm::normalize(glm::cross(edge1, edge2));

            // Check if geometric normal is axis-aligned (expected for cube)
            bool axisAligned =
                (std::abs(std::abs(geoNormal.x) - 1.0f) < 0.01f &&
                 std::abs(geoNormal.y) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                (std::abs(geoNormal.x) < 0.01f &&
                 std::abs(std::abs(geoNormal.y) - 1.0f) < 0.01f && std::abs(geoNormal.z) < 0.01f) ||
                (std::abs(geoNormal.x) < 0.01f && std::abs(geoNormal.y) < 0.01f &&
                 std::abs(std::abs(geoNormal.z) - 1.0f) < 0.01f);

            const char* status = axisAligned ? "" : " [NON-AXIS-ALIGNED!]";
            QL_LOG_DEBUG("  Triangle {}: indices=[{}, {}, {}], geoNormal=({:.3f}, {:.3f}, {:.3f}){}",
                         triIdx, idx0, idx1, idx2, geoNormal.x, geoNormal.y, geoNormal.z, status);

            if (!axisAligned) {
                QL_LOG_DEBUG("    v0=({:.4f}, {:.4f}, {:.4f})", v0.x, v0.y, v0.z);
                QL_LOG_DEBUG("    v1=({:.4f}, {:.4f}, {:.4f})", v1.x, v1.y, v1.z);
                QL_LOG_DEBUG("    v2=({:.4f}, {:.4f}, {:.4f})", v2.x, v2.y, v2.z);
            }
        }
    }

    return mesh;
}

// ============================================================================
// ParsePointInstancer - Expand PointInstancer to scene nodes
// ============================================================================

void UsdLoader::ParsePointInstancer(const void* stagePtr, const void* primPtr,
                                     Scene& scene,
                                     const std::unordered_map<String, int>& materialPathMap,
                                     const String& usdFilePath,
                                     const UsdLoadOptions& options) {
    if (!primPtr) {
        return;
    }

    const auto* prim = static_cast<const UsdPrim*>(primPtr);
    UsdGeomPointInstancer instancer(*prim);
    UsdTimeCode timeCode = GetTimeCode(options);

    // Get prototype relationships
    SdfPathVector protoPaths;
    instancer.GetPrototypesRel().GetTargets(&protoPaths);
    if (protoPaths.empty()) {
        return;
    }

    // Load prototype meshes
    const UsdStagePtr stage = prim->GetStage();
    std::vector<u32> protoMeshIndices;

    // Instance transforms are authored in the instancer's own space, so the
    // instancer's world transform sits between them and the stage root. Without
    // it a nested instancer placed every instance at the origin of the stage.
    const glm::mat4 instancerRoot =
        StageRootTransform(stage, options) *
        GfMatrix4dToGlm(UsdGeomXformable(*prim).ComputeLocalToWorldTransform(timeCode));

    for (const auto& protoPath : protoPaths) {
        UsdPrim protoPrim = stage->GetPrimAtPath(protoPath);
        if (!protoPrim) {
            continue;
        }

        // Find or create mesh for this prototype
        if (protoPrim.IsA<UsdGeomMesh>()) {
            Mesh protoMesh = ParseMesh(&(*stage), &protoPrim, materialPathMap,
                                       scene.materials, usdFilePath, options);
            protoMeshIndices.push_back(static_cast<u32>(scene.meshes.size()));
            scene.meshes.push_back(std::move(protoMesh));
        }
    }

    if (protoMeshIndices.empty()) {
        return;
    }

    // Get instance data
    VtArray<int> protoIndices;
    VtArray<GfVec3f> positions;
    VtArray<GfQuath> orientations;
    VtArray<GfVec3f> scales;

    instancer.GetProtoIndicesAttr().Get(&protoIndices, timeCode);
    instancer.GetPositionsAttr().Get(&positions, timeCode);
    instancer.GetOrientationsAttr().Get(&orientations, timeCode);
    instancer.GetScalesAttr().Get(&scales, timeCode);

    if (protoIndices.empty() || positions.empty()) {
        return;
    }

    // Create scene nodes for each instance
    for (size_t i = 0; i < protoIndices.size(); ++i) {
        int protoIdx = protoIndices[i];
        if (protoIdx < 0 || protoIdx >= static_cast<int>(protoMeshIndices.size())) {
            continue;
        }

        SceneNode node;
        node.meshIndex = protoMeshIndices[protoIdx];
        node.name = prim->GetName().GetString() + "_instance_" + std::to_string(i);

        // Build transform matrix
        glm::vec3 pos(0.0f);
        glm::quat rot = glm::identity<glm::quat>();
        glm::vec3 scale(1.0f);

        if (i < positions.size()) {
            pos = glm::vec3(positions[i][0], positions[i][1], positions[i][2]);
        }

        if (i < orientations.size()) {
            GfQuath q = orientations[i];
            rot = glm::quat(q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]);
        }

        if (i < scales.size()) {
            scale = glm::vec3(scales[i][0], scales[i][1], scales[i][2]);
        }

        // Compose transform: T * R * S
        glm::mat4 T = glm::translate(glm::mat4(1.0f), pos);
        glm::mat4 R = glm::mat4_cast(rot);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        node.transform = instancerRoot * T * R * S;

        scene.nodes.push_back(node);
    }

    QL_LOG_INFO("    Expanded PointInstancer '{}': {} instances",
                prim->GetName().GetString(), protoIndices.size());
}

// ============================================================================
// FlattenXformHierarchy - Not used, transforms computed in LoadFromFile
// ============================================================================

std::vector<SceneNode> UsdLoader::FlattenXformHierarchy(const void* /* stagePtr */) {
    return {};
}

// ============================================================================
// ListVariants - List available variants for a prim
// ============================================================================

Result<std::unordered_map<String, std::vector<String>>, String>
UsdLoader::ListVariants(const String& path, const String& primPath) {
    InitializeUsdPlugins();

    if (!std::filesystem::exists(path)) {
        return Result<std::unordered_map<String, std::vector<String>>>(
            Result<std::unordered_map<String, std::vector<String>>>::Err("File not found: " + path));
    }

    // Normalize path for OpenUSD
    std::string normalizedPath = std::filesystem::absolute(path).string();
    std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');

    UsdStageRefPtr stage = UsdStage::Open(std::string(normalizedPath.c_str()));
    if (!stage) {
        return Result<std::unordered_map<String, std::vector<String>>>(
            Result<std::unordered_map<String, std::vector<String>>>::Err("Failed to open USD stage"));
    }

    UsdPrim prim = stage->GetPrimAtPath(SdfPath(primPath));
    if (!prim) {
        return Result<std::unordered_map<String, std::vector<String>>>(
            Result<std::unordered_map<String, std::vector<String>>>::Err("Prim not found: " + primPath));
    }

    std::unordered_map<String, std::vector<String>> result;

    UsdVariantSets variantSets = prim.GetVariantSets();
    std::vector<std::string> setNames = variantSets.GetNames();

    for (const auto& setName : setNames) {
        UsdVariantSet vs = variantSets.GetVariantSet(setName);
        std::vector<std::string> variants = vs.GetVariantNames();
        result[setName] = std::vector<String>(variants.begin(), variants.end());
    }

    return Result(std::move(result));
}

// ============================================================================
// ListPrimsWithVariants - Find all prims with variant sets
// ============================================================================

Result<std::vector<String>, String> UsdLoader::ListPrimsWithVariants(const String& path) {
    InitializeUsdPlugins();

    if (!std::filesystem::exists(path)) {
        return Result<std::vector<String>>(
            Result<std::vector<String>>::Err("File not found: " + path));
    }

    // Normalize path for OpenUSD
    std::string normalizedPath = std::filesystem::absolute(path).string();
    std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');

    UsdStageRefPtr stage = UsdStage::Open(std::string(normalizedPath.c_str()));
    if (!stage) {
        return Result<std::vector<String>>(
            Result<std::vector<String>>::Err("Failed to open USD stage"));
    }

    std::vector<String> result;

    for (const UsdPrim& prim : stage->Traverse()) {
        UsdVariantSets variantSets = prim.GetVariantSets();
        if (!variantSets.GetNames().empty()) {
            result.push_back(prim.GetPath().GetString());
        }
    }

    return Result(std::move(result));
}

// ============================================================================
// LoadFromFile - Main entry point (default options)
// ============================================================================

Result<Scene, String> UsdLoader::LoadFromFile(const String& path) {
    return LoadFromFile(path, UsdLoadOptions::Default());
}

// ============================================================================
// LoadFromFile - Main entry point with options
// ============================================================================

Result<Scene, String> UsdLoader::LoadFromFile(const String& path, const UsdLoadOptions& options) {
    // Ensure USD plugins are initialized
    InitializeUsdPlugins();

    QL_LOG_INFO("Loading USD scene from: {}", path);

    if (!std::filesystem::exists(path)) {
        return Result<Scene>(Result<Scene>::Err("File not found: " + path));
    }

    // Determine file type and check extension
    std::filesystem::path filePath(path);
    String ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext != ".usda" && ext != ".usd" && ext != ".usdc" && ext != ".usdz") {
        return Result<Scene>(Result<Scene>::Err(
            "Unsupported USD file extension: " + ext +
            " (supported: .usd, .usda, .usdc, .usdz)"));
    }

    // Normalize path for OpenUSD
    // Convert to absolute path and use forward slashes
    std::string normalizedPath = std::filesystem::absolute(filePath).string();
    std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');

    QL_LOG_INFO("  Normalized path: {}", normalizedPath);

    // ========================================================================
    // Open USD Stage with load policy
    // ========================================================================
    UsdStage::InitialLoadSet loadSet = (options.payloadPolicy == UsdLoadOptions::PayloadPolicy::LoadAll)
        ? UsdStage::LoadAll
        : UsdStage::LoadNone;

    UsdStageRefPtr stage = UsdStage::Open(normalizedPath, loadSet);
    if (!stage) {
        return Result<Scene>(Result<Scene>::Err("Failed to open USD stage: " + path));
    }

    QL_LOG_INFO("  USD stage opened successfully");

    // ========================================================================
    // Apply variant selections
    // ========================================================================
    const auto applyVariants = [](const UsdPrim& prim, const String& primLabel,
                                 const std::unordered_map<String, String>& variantMap,
                                 bool quiet) {
        UsdVariantSets variantSets = prim.GetVariantSets();
        for (const auto& [setName, variantName] : variantMap) {
            if (!variantSets.HasVariantSet(setName)) {
                if (!quiet) {
                    QL_LOG_WARN("  Variant set '{}' not found on '{}'", setName, primLabel);
                }
                continue;
            }
            UsdVariantSet variantSet = variantSets.GetVariantSet(setName);
            if (variantSet.SetVariantSelection(variantName)) {
                QL_LOG_INFO("  Applied variant: {}[{}={}]", primLabel, setName, variantName);
            } else {
                QL_LOG_WARN("  Failed to set variant {}[{}={}]", primLabel, setName, variantName);
            }
        }
    };

    for (const auto& [primPath, variantMap] : options.variantSelections) {
        if (!primPath.empty()) {
            UsdPrim prim = stage->GetPrimAtPath(SdfPath(primPath));
            if (!prim) {
                QL_LOG_WARN("  Variant selection: prim '{}' not found", primPath);
                continue;
            }
            applyVariants(prim, primPath, variantMap, /*quiet=*/false);
            continue;
        }

        // The wildcard: this set, wherever it is. Every owning prim is collected
        // before any selection is made, because a selection changes the
        // composition the traversal is walking -- picking a variant that adds a
        // subtree while iterating over that subtree is undefined.
        std::vector<UsdPrim> owners;
        for (const UsdPrim& prim : stage->Traverse()) {
            UsdVariantSets sets = prim.GetVariantSets();
            for (const auto& [setName, variantName] : variantMap) {
                (void)variantName;
                if (sets.HasVariantSet(setName)) {
                    owners.push_back(prim);
                    break;
                }
            }
        }
        if (owners.empty()) {
            QL_LOG_WARN("  Variant selection: no prim owns any of the named variant sets");
        }
        for (const UsdPrim& prim : owners) {
            applyVariants(prim, prim.GetPath().GetString(), variantMap, /*quiet=*/true);
        }
    }

    // Debug: Print stage structure
    size_t primCount = 0;
    for (const UsdPrim& prim : stage->Traverse()) {
        (void)prim;
        ++primCount;
    }
    QL_LOG_INFO("  Stage contains {} prims", primCount);

    // Build Quantiloom Scene
    Scene scene;
    scene.name = filePath.stem().string();

    // ========================================================================
    // Pass 0: Read every surface, then decode the files they name
    // ========================================================================
    // Reading and decoding are separated because a shader graph has to be
    // walked to find out which files a material uses, and decoding is the
    // load's I/O cost: knowing every path first is what lets them all decode at
    // once. The readings are kept, so Pass 1 does not walk the graphs again.

    usd::UsdTextureBank bank;
    std::unordered_map<SdfPath, usd::SurfaceReading, SdfPath::Hash> readings;
    std::unordered_set<String> texturePaths;
    const String usdDir = filePath.parent_path().string();

    for (const UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<UsdShadeMaterial>()) {
            continue;
        }

        UsdShadeMaterial shadeMaterial(prim);

        // A material translated from a .mtlx document connects its surface
        // through the `mtlx` render context, not the universal one, so asking
        // only for the default finds nothing and the material comes out empty.
        // The universal context stays second: a material that has both is
        // saying the universal output is the portable one.
        UsdShadeShader surfaceShader = shadeMaterial.ComputeSurfaceSource(
            {TfToken("mtlx"), UsdShadeTokens->universalRenderContext});

        usd::BindingContext context;
        context.usdDir = usdDir;
        context.timeCode = options.timeCode;
        context.useDefaultTime = options.useDefaultTime;
        context.loadTextures = options.loadTextures;
        context.materialPath = prim.GetPath().GetString();

        const usd::SurfaceVocabulary vocabulary =
            ClassifySurface(surfaceShader, context.materialPath);
        usd::SurfaceReading reading = usd::ReadSurface(
            surfaceShader ? &surfaceShader : nullptr, vocabulary, context);
        usd::CollectTextureSources(reading, texturePaths);

        if (UsdAttribute tempTexAttr =
                prim.GetAttribute(TfToken("quantiloom:temperatureTexture"))) {
            SdfAssetPath texPath;
            if (tempTexAttr.Get(&texPath) && !texPath.GetAssetPath().empty()) {
                texturePaths.insert(ResolveQuantiloomAsset(texPath, path));
            }
        }

        readings.emplace(prim.GetPath(), std::move(reading));
    }

    if (options.loadTextures || !texturePaths.empty()) {
        bank.Preload(texturePaths);
    }

    // ========================================================================
    // Pass 1: Collect all Materials
    // ========================================================================
    std::unordered_map<String, int> materialPathMap;

    for (const UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<UsdShadeMaterial>()) {
            continue;
        }
        const auto reading = readings.find(prim.GetPath());
        if (reading == readings.end()) {
            continue;
        }

        Material mat = ParseMaterial(prim, reading->second, bank, scene.textures, path);
        materialPathMap[prim.GetPath().GetString()] = static_cast<int>(scene.materials.size());
        scene.materials.push_back(std::move(mat));

        QL_LOG_INFO("  Loaded material '{}' (metallic={:.2f}, roughness={:.2f})",
                    scene.materials.back().name,
                    scene.materials.back().metallicFactor,
                    scene.materials.back().roughnessFactor);
    }

    // The decoded sources have done their job; only the entries in
    // scene.textures are uploaded, and a 4K source is 64 MB.
    bank.ReleaseSources();
    readings.clear();

    // Ensure at least one default material exists
    if (scene.materials.empty()) {
        scene.materials.push_back(Material::CreateLambertian(glm::vec3(0.8f), "DefaultMaterial"));
    }

    // Colour space is decided per entry as it is built, from what the shader
    // graph said or, failing that, from the slot it fills -- not swept over the
    // materials afterwards. One file bound as both a colour and a data map is
    // two entries with two answers, which a pass over shared indices could not
    // express.
    //
    // BC7 compression comes after, always: BC7CompressedData captures isSRGB at
    // compression time (TextureCompressor.cpp:103) and the upload picks its
    // format from that copy (TextureManager.cpp:311), so a texture compressed
    // before its colour space is known is stuck with the wrong one.
    if (TextureCompressor::IsAvailable() && !scene.textures.empty()) {
        TextureCompressor::ParallelCompressTextures(scene.textures, false /* fast mode */);
    }

    // ========================================================================
    // Pass 2: Collect all Meshes and PointInstancers
    // ========================================================================
    const glm::mat4 stageRoot = StageRootTransform(stage, options);
    size_t skippedPrims = 0;

    for (const UsdPrim& prim : stage->Traverse()) {
        if (prim.IsA<UsdGeomMesh>()) {
            // Skip meshes that are prototypes of PointInstancers
            if (prim.IsInPrototype()) {
                continue;
            }

            if (!IsRenderablePrim(prim, GetTimeCode(options))) {
                ++skippedPrims;
                continue;
            }

            Mesh mesh = ParseMesh(&(*stage), &prim, materialPathMap, scene.materials,
                                  path, options);

            // ================================================================
            // Read doubleSided attribute from geometry and propagate to material
            // ================================================================
            // USD stores doubleSided on geometry, not on material. We read it here
            // and update the bound material. If multiple meshes share a material
            // with different doubleSided values, we use OR logic (safer: render both sides).
            UsdGeomMesh geomMesh(prim);
            bool geomDoubleSided = false;
            if (auto attr = geomMesh.GetDoubleSidedAttr()) {
                attr.Get(&geomDoubleSided, GetTimeCode(options));
            }

            if (geomDoubleSided) {
                // Update all materials used by this mesh's primitives
                for (const auto& primitive : mesh.primitives) {
                    if (primitive.materialId >= 0 && static_cast<size_t>(primitive.materialId) < scene.materials.size()) {
                        Material& mat = scene.materials[primitive.materialId];
                        if (!mat.doubleSided) {
                            QL_LOG_DEBUG("  Mesh '{}' has doubleSided=true, updating material '{}'",
                                        mesh.name, mat.name);
                            mat.doubleSided = true;
                        }
                    }
                }
            }

            // Get world transform
            UsdGeomXformable xformable(prim);
            GfMatrix4d worldXform = xformable.ComputeLocalToWorldTransform(GetTimeCode(options));

            // Create scene node
            SceneNode node;
            node.meshIndex = static_cast<u32>(scene.meshes.size());
            node.name = prim.GetName().GetString();
            node.transform = stageRoot * GfMatrix4dToGlm(worldXform);

            scene.meshes.push_back(std::move(mesh));
            scene.nodes.push_back(node);
        }
        else if (options.enablePointInstancer && prim.IsA<UsdGeomPointInstancer>()) {
            if (!IsRenderablePrim(prim, GetTimeCode(options))) {
                ++skippedPrims;
                continue;
            }
            ParsePointInstancer(&(*stage), &prim, scene, materialPathMap, path, options);
        }
    }

    if (skippedPrims > 0) {
        QL_LOG_INFO("  Skipped {} guide, proxy or invisible prims", skippedPrims);
    }

    QL_LOG_INFO("  Scene '{}' loaded: {} meshes, {} nodes, {} materials, {} textures",
                scene.name, scene.meshes.size(), scene.nodes.size(),
                scene.materials.size(), scene.textures.size());

    return Result(std::move(scene));
}

} // namespace quantiloom

#else // QUANTILOOM_USE_OPENUSD not defined

// ============================================================================
// Stub implementation when OpenUSD is not available
// ============================================================================

namespace quantiloom {

bool UsdLoader::IsAvailable() {
    return false;
}

Result<Scene, String> UsdLoader::LoadFromFile(const String& /* path */) {
    return Result<Scene>(Result<Scene>::Err(
        "OpenUSD support not available. "
        "Please set USD_ROOT to OpenUSD installation path and rebuild."));
}

Result<Scene, String> UsdLoader::LoadFromFile(const String& /* path */, const UsdLoadOptions& /* options */) {
    return Result<Scene>(Result<Scene>::Err(
        "OpenUSD support not available. "
        "Please set USD_ROOT to OpenUSD installation path and rebuild."));
}

Result<std::unordered_map<String, std::vector<String>>, String>
UsdLoader::ListVariants(const String& /* path */, const String& /* primPath */) {
    return Result<std::unordered_map<String, std::vector<String>>>(
        Result<std::unordered_map<String, std::vector<String>>>::Err("OpenUSD support not available"));
}

Result<std::vector<String>, String> UsdLoader::ListPrimsWithVariants(const String& /* path */) {
    return Result<std::vector<String>>(
        Result<std::vector<String>>::Err("OpenUSD support not available"));
}

std::vector<u32> UsdLoader::TriangulatePolygons(
    const std::vector<i32>& /* faceVertexCounts */,
    const std::vector<i32>& /* faceVertexIndices */) {
    return {};
}

std::vector<u32> UsdLoader::TriangulatePolygonsWithFaceMap(
    const std::vector<i32>& /* faceVertexCounts */,
    const std::vector<i32>& /* faceVertexIndices */,
    std::vector<u32>& /* outTriangleToFace */) {
    return {};
}

Texture UsdLoader::ParseTexture(const void* /* stage */, const String& /* assetPath */,
                                  const String& /* usdFilePath */) {
    return Texture{};
}


Mesh UsdLoader::ParseMesh(const void* /* stage */, const void* /* prim */,
                          const std::unordered_map<String, int>& /* materialPathMap */,
                          const std::vector<Material>& /* materials */,
                          const String& /* usdFilePath */,
                          const UsdLoadOptions& /* options */) {
    return Mesh{};
}

void UsdLoader::ParsePointInstancer(const void* /* stage */, const void* /* instancer */,
                                     Scene& /* scene */,
                                     const std::unordered_map<String, int>& /* materialPathMap */,
                                     const String& /* usdFilePath */,
                                     const UsdLoadOptions& /* options */) {
}

std::vector<SceneNode> UsdLoader::FlattenXformHierarchy(const void* /* stage */) {
    return {};
}

} // namespace quantiloom

#endif // QUANTILOOM_USE_OPENUSD
