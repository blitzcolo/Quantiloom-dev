/**
 * @file UsdShadeGraph.cpp
 * @brief The shader-graph walk, the slot recipes and the scalar conversions
 */

#include "io/UsdShadeGraph.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#if QUANTILOOM_USE_OPENUSD

#include <pxr/pxr.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/nodeGraph.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/utils.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace quantiloom::usd {

namespace {

constexpr int kMaxGraphDepth = 16;

UsdTimeCode TimeOf(const BindingContext& context) {
    return context.useDefaultTime ? UsdTimeCode::Default() : UsdTimeCode(context.timeCode);
}

bool IdStartsWith(const std::string& id, const char* prefix) {
    const std::string_view p(prefix);
    return id.size() >= p.size() && std::string_view(id).substr(0, p.size()) == p;
}

/// The asset resolver's answer if it has one, otherwise the path as written,
/// made absolute against the USD file. The resolved path is what makes a
/// `.usdz`'s internal layout and a custom resolver work.
String ResolveAsset(const SdfAssetPath& asset, const BindingContext& context) {
    if (!asset.GetResolvedPath().empty()) {
        return asset.GetResolvedPath();
    }
    const std::string& raw = asset.GetAssetPath();
    if (raw.empty()) {
        return {};
    }
    std::filesystem::path path(raw);
    if (!path.is_absolute() && !context.usdDir.empty()) {
        path = std::filesystem::path(context.usdDir) / path;
    }
    return std::filesystem::weakly_canonical(path).string();
}

/// Read a constant off an attribute, widened to a vec4 in the slot's own terms.
std::optional<glm::vec4> ReadConstant(const UsdAttribute& attr, ValueKind kind,
                                      const BindingContext& context) {
    const UsdTimeCode time = TimeOf(context);
    switch (kind) {
        case ValueKind::Float: {
            f32 value = 0.0f;
            if (attr.Get(&value, time)) {
                return glm::vec4(value, 0.0f, 0.0f, 0.0f);
            }
            break;
        }
        case ValueKind::Color3:
        case ValueKind::Vector3: {
            GfVec3f value;
            if (attr.Get(&value, time)) {
                return glm::vec4(value[0], value[1], value[2], 1.0f);
            }
            // A color3 input given a single float is legal MaterialX authoring.
            f32 scalar = 0.0f;
            if (attr.Get(&scalar, time)) {
                return glm::vec4(scalar, scalar, scalar, 1.0f);
            }
            break;
        }
        case ValueKind::Color4: {
            GfVec4f value;
            if (attr.Get(&value, time)) {
                return glm::vec4(value[0], value[1], value[2], value[3]);
            }
            break;
        }
        case ValueKind::Int: {
            int value = 0;
            if (attr.Get(&value, time)) {
                return glm::vec4(static_cast<f32>(value), 0.0f, 0.0f, 0.0f);
            }
            break;
        }
        case ValueKind::Bool: {
            bool value = false;
            if (attr.Get(&value, time)) {
                return glm::vec4(value ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
            }
            break;
        }
    }
    return std::nullopt;
}

template <typename T>
std::optional<T> GetInputValue(const UsdShadeShader& node, const char* name,
                               const BindingContext& context) {
    if (UsdShadeInput input = node.GetInput(TfToken(name))) {
        T value;
        if (input.Get(&value, TimeOf(context))) {
            return value;
        }
    }
    return std::nullopt;
}

String NodeId(const UsdShadeShader& node) {
    TfToken id;
    node.GetIdAttr().Get(&id);
    return id.GetString();
}

// ============================================================================
// UV chains
// ============================================================================

/// What feeds an image node's texture coordinates.
struct UvBinding {
    String primvar = "st";
    UvTransform transform;
    bool transformSeen = false;
};

/// UsdTransform2d: `result = translate(rotate(in * scale))`, rotation in degrees
/// counter-clockwise -- the same order and handedness as Material::UvTransform,
/// so it maps across term for term.
UvTransform Transform2dOf(const UsdShadeShader& node, const BindingContext& context) {
    UvTransform out;
    if (auto scale = GetInputValue<GfVec2f>(node, "scale", context)) {
        out.scale = glm::vec2((*scale)[0], (*scale)[1]);
    }
    if (auto rotation = GetInputValue<f32>(node, "rotation", context)) {
        out.rotation = glm::radians(*rotation);
    }
    if (auto translation = GetInputValue<GfVec2f>(node, "translation", context)) {
        out.offset = glm::vec2((*translation)[0], (*translation)[1]);
    }
    return out;
}

/**
 * @brief MaterialX `place2d`, folded into one affine
 *
 * NG_place2d_vector2 divides by scale and subtracts offset, which is the
 * opposite of both on a first reading, and its `rotate2d` turns clockwise:
 * mx_rotate_vector2 is (ca*x + sa*y, -sa*x + ca*y), i.e. R(-theta). So with
 * S = diag(1/sx, 1/sy) and R the counter-clockwise rotation by -radians(rotate):
 *
 *   order 0 (SRT): uv' = R S (uv - p) - t + p  ->  offset = p - t - R S p
 *   order 1 (TRS): uv' = S R (uv - p - t) + p  ->  offset = p - S R (p + t)
 *
 * Order 0 is exact. Order 1's matrix S R is only a rotate-then-scale when the
 * scale is uniform or there is no rotation; otherwise it shears, which
 * UvTransform cannot say, and the SRT reading is used with a warning.
 */
UvTransform Place2dOf(const UsdShadeShader& node, BindingContext& context) {
    glm::vec2 pivot(0.0f);
    glm::vec2 scale(1.0f);
    glm::vec2 offset(0.0f);
    f32 rotateDegrees = 0.0f;
    int order = 0;

    if (auto value = GetInputValue<GfVec2f>(node, "pivot", context)) {
        pivot = glm::vec2((*value)[0], (*value)[1]);
    }
    if (auto value = GetInputValue<GfVec2f>(node, "scale", context)) {
        scale = glm::vec2((*value)[0], (*value)[1]);
    }
    if (auto value = GetInputValue<GfVec2f>(node, "offset", context)) {
        offset = glm::vec2((*value)[0], (*value)[1]);
    }
    if (auto value = GetInputValue<f32>(node, "rotate", context)) {
        rotateDegrees = *value;
    }
    if (auto value = GetInputValue<int>(node, "operationorder", context)) {
        order = *value;
    }

    UvTransform out;
    out.rotation = -glm::radians(rotateDegrees);
    out.scale = glm::vec2(scale.x != 0.0f ? 1.0f / scale.x : 1.0f,
                          scale.y != 0.0f ? 1.0f / scale.y : 1.0f);

    const f32 c = std::cos(out.rotation);
    const f32 s = std::sin(out.rotation);
    // A = R(out.rotation) * diag(out.scale)
    const glm::vec2 aPivot(out.scale.x * c * pivot.x - out.scale.y * s * pivot.y,
                           out.scale.x * s * pivot.x + out.scale.y * c * pivot.y);

    if (order == 1) {
        if (scale.x != scale.y && rotateDegrees != 0.0f) {
            if (context.WarnOnce("place2d-trs-shear")) {
                QL_LOG_WARN("    Material '{}': place2d in TRS order with a non-uniform "
                            "scale and a rotation shears the UVs, which one affine "
                            "cannot express; the SRT reading is used instead",
                            context.materialPath);
            }
        } else {
            const glm::vec2 pt = pivot + offset;
            const glm::vec2 aPt(out.scale.x * c * pt.x - out.scale.y * s * pt.y,
                                out.scale.x * s * pt.x + out.scale.y * c * pt.y);
            out.offset = pivot - aPt;
            return out;
        }
    }

    out.offset = pivot - offset - aPivot;
    return out;
}

UvBinding ResolveUvChain(const UsdShadeInput& input, BindingContext& context, int depth);

UvBinding ResolveUvSource(const UsdAttribute& producer, BindingContext& context, int depth) {
    UvBinding out;
    if (depth > kMaxGraphDepth) {
        return out;
    }

    UsdShadeShader node(producer.GetPrim());
    if (!node) {
        return out;
    }
    const String id = NodeId(node);

    if (IdStartsWith(id, "UsdPrimvarReader_")) {
        if (auto name = GetInputValue<TfToken>(node, "varname", context)) {
            out.primvar = name->GetString();
        } else if (auto name = GetInputValue<std::string>(node, "varname", context)) {
            out.primvar = *name;
        }
        return out;
    }

    if (id == "UsdTransform2d") {
        out = ResolveUvChain(node.GetInput(TfToken("in")), context, depth + 1);
        if (out.transformSeen && context.WarnOnce("uv-transform-chain")) {
            QL_LOG_WARN("    Material '{}': more than one UV transform in a chain; "
                        "only the outermost is applied", context.materialPath);
        }
        out.transform = Transform2dOf(node, context);
        out.transformSeen = true;
        return out;
    }

    if (id == "ND_place2d_vector2") {
        out = ResolveUvChain(node.GetInput(TfToken("texcoord")), context, depth + 1);
        if (out.transformSeen && context.WarnOnce("uv-transform-chain")) {
            QL_LOG_WARN("    Material '{}': more than one UV transform in a chain; "
                        "only the outermost is applied", context.materialPath);
        }
        out.transform = Place2dOf(node, context);
        out.transformSeen = true;
        return out;
    }

    if (IdStartsWith(id, "ND_texcoord_")) {
        if (auto index = GetInputValue<int>(node, "index", context); index && *index != 0) {
            if (context.WarnOnce("multi-uv")) {
                QL_LOG_WARN("    Material '{}': UV set {} requested, but only one UV set "
                            "is loaded; set 0 is used", context.materialPath, *index);
            }
        }
        return out;
    }

    if (IdStartsWith(id, "ND_geompropvalue_")) {
        if (auto name = GetInputValue<std::string>(node, "geomprop", context)) {
            out.primvar = *name;
        }
        return out;
    }

    if (context.WarnOnce("uv-node:" + id)) {
        QL_LOG_WARN("    Material '{}': UV node '{}' is not understood; the default "
                    "`st` set is used", context.materialPath, id);
    }
    return out;
}

UvBinding ResolveUvChain(const UsdShadeInput& input, BindingContext& context, int depth) {
    UvBinding out;
    if (!input || depth > kMaxGraphDepth) {
        return out;
    }
    const UsdShadeAttributeVector producers =
        UsdShadeUtils::GetValueProducingAttributes(input);
    if (producers.empty()) {
        return out;
    }
    const UsdAttribute& producer = producers.front();
    if (UsdShadeUtils::GetType(producer.GetName()) == UsdShadeAttributeType::Input) {
        return out;
    }
    return ResolveUvSource(producer, context, depth);
}

// ============================================================================
// Node walk
// ============================================================================

struct InputBinding {
    std::optional<glm::vec4> value;
    std::optional<TextureRef> texture;
};

InputBinding ResolveInputBinding(const UsdShadeInput& input, const SurfaceInputSpec& spec,
                                 BindingContext& context, int depth);

InputBinding WalkNode(const UsdShadeShader& node, const TfToken& outputName,
                      const SurfaceInputSpec& spec, BindingContext& context, int depth);

/// A named output of a `UsdUVTexture` or a MaterialX separate/extract node.
ChannelSel ChannelFromOutput(const TfToken& outputName, ChannelSel fallback) {
    const std::string& name = outputName.GetString();
    if (name == "r" || name == "outr" || name == "x" || name == "outx") return ChannelSel::R;
    if (name == "g" || name == "outg" || name == "y" || name == "outy") return ChannelSel::G;
    if (name == "b" || name == "outb" || name == "z" || name == "outz") return ChannelSel::B;
    if (name == "a" || name == "outa" || name == "w" || name == "outw") return ChannelSel::A;
    if (name == "rgb") return ChannelSel::RGB;
    if (name == "rgba") return ChannelSel::RGBA;
    return fallback;
}

/// MaterialX states an image's colour space on the `file` input.
ColourSpaceHint HintFromColorSpace(const TfToken& colorSpace, BindingContext& context) {
    const std::string& name = colorSpace.GetString();
    if (name.empty()) {
        return ColourSpaceHint::Unspecified;
    }
    if (name == "srgb_texture" || name == "sRGB" || name == "gamma22" || name == "g22_rec709") {
        return ColourSpaceHint::Srgb;
    }
    if (name == "lin_rec709" || name == "lin_srgb" || name == "linear" || name == "raw" ||
        name == "none" || name == "auto") {
        return ColourSpaceHint::Linear;
    }
    // Anything else is a working space this loader has no transform for. Guess
    // by family rather than pretend: a name with "srgb" or "gamma" in it is
    // encoded, everything else is treated as linear.
    const bool encoded = name.find("srgb") != std::string::npos ||
                         name.find("sRGB") != std::string::npos ||
                         name.find("gamma") != std::string::npos;
    if (context.WarnOnce("colorspace:" + name)) {
        QL_LOG_WARN("    Material '{}': colour space '{}' has no transform here; "
                    "treating it as {}", context.materialPath, name,
                    encoded ? "sRGB" : "linear");
    }
    return encoded ? ColourSpaceHint::Srgb : ColourSpaceHint::Linear;
}

InputBinding ReadUsdUVTexture(const UsdShadeShader& node, const TfToken& outputName,
                              const SurfaceInputSpec& spec, BindingContext& context,
                              int depth) {
    InputBinding out;

    SdfAssetPath asset;
    if (UsdShadeInput file = node.GetInput(TfToken("file"))) {
        file.Get(&asset, TimeOf(context));
    }
    const String path = ResolveAsset(asset, context);

    if (path.empty() || !std::filesystem::exists(path)) {
        // `fallback` is what UsdUVTexture says to use when the file cannot be
        // read, so a missing texture becomes a constant rather than nothing.
        if (!path.empty()) {
            QL_LOG_WARN("    Material '{}': texture '{}' not found; using the node's "
                        "fallback", context.materialPath, path);
        }
        if (auto fallback = GetInputValue<GfVec4f>(node, "fallback", context)) {
            out.value = glm::vec4((*fallback)[0], (*fallback)[1], (*fallback)[2],
                                  (*fallback)[3]);
        }
        return out;
    }

    TextureRef ref;
    ref.absolutePath = path;
    ref.nodePath = node.GetPrim().GetPath().GetString();
    ref.channels = ChannelFromOutput(outputName, spec.colour ? ChannelSel::RGB : ChannelSel::R);

    if (auto scale = GetInputValue<GfVec4f>(node, "scale", context)) {
        ref.scale = glm::vec4((*scale)[0], (*scale)[1], (*scale)[2], (*scale)[3]);
    }
    if (auto bias = GetInputValue<GfVec4f>(node, "bias", context)) {
        ref.bias = glm::vec4((*bias)[0], (*bias)[1], (*bias)[2], (*bias)[3]);
    }

    bool unsupportedWrap = false;
    bool anyUnsupported = false;
    if (auto wrapS = GetInputValue<TfToken>(node, "wrapS", context)) {
        ref.sampler.wrapS = WrapFromToken(wrapS->GetString(), unsupportedWrap);
        anyUnsupported |= unsupportedWrap;
    }
    if (auto wrapT = GetInputValue<TfToken>(node, "wrapT", context)) {
        ref.sampler.wrapT = WrapFromToken(wrapT->GetString(), unsupportedWrap);
        anyUnsupported |= unsupportedWrap;
    }
    if (anyUnsupported && context.WarnOnce("wrap-black")) {
        QL_LOG_WARN("    Material '{}': wrap mode 'black' has no equivalent; clamping "
                    "to the edge instead", context.materialPath);
    }

    if (auto colorSpace = GetInputValue<TfToken>(node, "sourceColorSpace", context)) {
        const std::string& name = colorSpace->GetString();
        if (name == "sRGB") {
            ref.colourSpace = ColourSpaceHint::Srgb;
        } else if (name == "raw") {
            ref.colourSpace = ColourSpaceHint::Linear;
        }
        // "auto" stays Unspecified: the destination slot decides, which is a
        // better guess than the file extension for a data map saved as PNG.
    }
    if (ref.colourSpace == ColourSpaceHint::Unspecified) {
        // The attribute may also carry USD's own colour-space metadata.
        if (UsdShadeInput file = node.GetInput(TfToken("file"))) {
            ref.colourSpace = HintFromColorSpace(file.GetAttr().GetColorSpace(), context);
        }
    }

    const UvBinding uv = ResolveUvChain(node.GetInput(TfToken("st")), context, depth + 1);
    ref.uvPrimvar = uv.primvar;
    ref.uv = uv.transform;

    out.texture = std::move(ref);
    return out;
}

InputBinding ReadMaterialXImage(const UsdShadeShader& node, const TfToken& outputName,
                                const SurfaceInputSpec& spec, BindingContext& context,
                                int depth, bool tiled) {
    InputBinding out;

    SdfAssetPath asset;
    UsdShadeInput file = node.GetInput(TfToken("file"));
    if (file) {
        file.Get(&asset, TimeOf(context));
    }
    const String path = ResolveAsset(asset, context);

    if (path.empty() || !std::filesystem::exists(path)) {
        if (!path.empty()) {
            QL_LOG_WARN("    Material '{}': texture '{}' not found; using the node's "
                        "default", context.materialPath, path);
        }
        if (auto fallback = GetInputValue<GfVec3f>(node, "default", context)) {
            out.value = glm::vec4((*fallback)[0], (*fallback)[1], (*fallback)[2], 1.0f);
        } else if (auto scalar = GetInputValue<f32>(node, "default", context)) {
            out.value = glm::vec4(*scalar, *scalar, *scalar, 1.0f);
        }
        return out;
    }

    TextureRef ref;
    ref.absolutePath = path;
    ref.nodePath = node.GetPrim().GetPath().GetString();
    ref.channels = ChannelFromOutput(outputName, spec.colour ? ChannelSel::RGB : ChannelSel::R);
    if (file) {
        ref.colourSpace = HintFromColorSpace(file.GetAttr().GetColorSpace(), context);
    }

    bool unsupportedWrap = false;
    bool anyUnsupported = false;
    if (auto mode = GetInputValue<std::string>(node, "uaddressmode", context)) {
        // MaterialX calls USD's `black` `constant`.
        ref.sampler.wrapS = WrapFromToken(*mode == "constant" ? "black" : *mode, unsupportedWrap);
        anyUnsupported |= unsupportedWrap;
    }
    if (auto mode = GetInputValue<std::string>(node, "vaddressmode", context)) {
        ref.sampler.wrapT = WrapFromToken(*mode == "constant" ? "black" : *mode, unsupportedWrap);
        anyUnsupported |= unsupportedWrap;
    }
    if (anyUnsupported && context.WarnOnce("wrap-black")) {
        QL_LOG_WARN("    Material '{}': address mode 'constant' has no equivalent; "
                    "clamping to the edge instead", context.materialPath);
    }
    if (auto filter = GetInputValue<std::string>(node, "filtertype", context)) {
        ref.sampler.minFilter = FilterFromToken(*filter);
        ref.sampler.magFilter = ref.sampler.minFilter;
    }

    const UvBinding uv = ResolveUvChain(node.GetInput(TfToken("texcoord")), context, depth + 1);
    ref.uvPrimvar = uv.primvar;
    ref.uv = uv.transform;

    if (tiled) {
        // uvtiling scales the coordinates and uvoffset shifts them, before any
        // transform already on the chain -- which is the same place a
        // UvTransform's own scale and offset act, so they compose by product.
        if (auto tiling = GetInputValue<GfVec2f>(node, "uvtiling", context)) {
            ref.uv.scale *= glm::vec2((*tiling)[0], (*tiling)[1]);
        }
        if (auto offset = GetInputValue<GfVec2f>(node, "uvoffset", context)) {
            ref.uv.offset += glm::vec2((*offset)[0], (*offset)[1]);
        }
        for (const char* name : {"realworldimagesize", "realworldtilesize"}) {
            if (UsdShadeInput input = node.GetInput(TfToken(name));
                input && input.GetAttr().HasAuthoredValue() &&
                context.WarnOnce(String("tiledimage-realworld:") + name)) {
                QL_LOG_WARN("    Material '{}': tiledimage '{}' is not applied; UV "
                            "tiling is unitless here", context.materialPath, name);
            }
        }
    }

    out.texture = std::move(ref);
    return out;
}

InputBinding WalkNode(const UsdShadeShader& node, const TfToken& outputName,
                      const SurfaceInputSpec& spec, BindingContext& context, int depth) {
    InputBinding out;
    if (depth > kMaxGraphDepth) {
        return out;
    }

    const String id = NodeId(node);

    if (id == "UsdUVTexture") {
        return ReadUsdUVTexture(node, outputName, spec, context, depth);
    }
    if (IdStartsWith(id, "ND_image_")) {
        return ReadMaterialXImage(node, outputName, spec, context, depth, false);
    }
    if (IdStartsWith(id, "ND_tiledimage_")) {
        return ReadMaterialXImage(node, outputName, spec, context, depth, true);
    }

    if (IdStartsWith(id, "ND_normalmap")) {
        out = ResolveInputBinding(node.GetInput(TfToken("in")), spec, context, depth + 1);
        if (out.texture) {
            if (auto scale = GetInputValue<f32>(node, "scale", context)) {
                out.texture->normalScale = *scale;
            } else if (auto scale2 = GetInputValue<GfVec2f>(node, "scale", context)) {
                out.texture->normalScale = (*scale2)[0];
            }
        }
        return out;
    }

    if (IdStartsWith(id, "ND_separate")) {
        out = ResolveInputBinding(node.GetInput(TfToken("in")), spec, context, depth + 1);
        if (out.texture) {
            out.texture->channels = ChannelFromOutput(outputName, out.texture->channels);
        }
        return out;
    }

    if (IdStartsWith(id, "ND_extract_")) {
        out = ResolveInputBinding(node.GetInput(TfToken("in")), spec, context, depth + 1);
        if (out.texture) {
            const int index = GetInputValue<int>(node, "index", context).value_or(0);
            static const ChannelSel kByIndex[4] = {ChannelSel::R, ChannelSel::G,
                                                   ChannelSel::B, ChannelSel::A};
            out.texture->channels = kByIndex[std::clamp(index, 0, 3)];
        }
        return out;
    }

    if (IdStartsWith(id, "ND_convert_")) {
        return ResolveInputBinding(node.GetInput(TfToken("in")), spec, context, depth + 1);
    }

    if (IdStartsWith(id, "ND_constant_")) {
        if (UsdShadeInput value = node.GetInput(TfToken("value"))) {
            out.value = ReadConstant(value.GetAttr(), spec.kind, context);
        }
        return out;
    }

    if (IdStartsWith(id, "ND_multiply_")) {
        // One side a graph, the other a constant: fold the constant into the
        // texture's scale so the product survives without a second entry.
        InputBinding lhs = ResolveInputBinding(node.GetInput(TfToken("in1")), spec, context,
                                               depth + 1);
        InputBinding rhs = ResolveInputBinding(node.GetInput(TfToken("in2")), spec, context,
                                               depth + 1);
        if (lhs.texture && rhs.value) {
            lhs.texture->scale *= *rhs.value;
            return lhs;
        }
        if (rhs.texture && lhs.value) {
            rhs.texture->scale *= *lhs.value;
            return rhs;
        }
        if (lhs.value && rhs.value) {
            out.value = *lhs.value * *rhs.value;
            return out;
        }
        if (context.WarnOnce("multiply-two-graphs")) {
            QL_LOG_WARN("    Material '{}': a multiply of two textures cannot be folded "
                        "into one slot; the first is used", context.materialPath);
        }
        return lhs.texture ? lhs : rhs;
    }

    if (context.WarnOnce("node:" + id)) {
        QL_LOG_WARN("    Material '{}': shader node '{}' is not understood; the input it "
                    "feeds is treated as unconnected", context.materialPath, id);
    }
    return out;
}

InputBinding ResolveInputBinding(const UsdShadeInput& input, const SurfaceInputSpec& spec,
                                 BindingContext& context, int depth) {
    InputBinding out;
    if (!input || depth > kMaxGraphDepth) {
        return out;
    }

    // GetValueProducingAttributes follows connections through NodeGraph
    // interfaces, which is how a MaterialX document's material-level inputs
    // reach the shader that uses them.
    const UsdShadeAttributeVector producers =
        UsdShadeUtils::GetValueProducingAttributes(input);

    if (producers.empty()) {
        if (input.GetAttr().HasAuthoredValue()) {
            out.value = ReadConstant(input.GetAttr(), spec.kind, context);
        }
        return out;
    }

    const UsdAttribute& producer = producers.front();
    if (UsdShadeUtils::GetType(producer.GetName()) == UsdShadeAttributeType::Input) {
        out.value = ReadConstant(producer, spec.kind, context);
        return out;
    }

    if (!context.loadTextures) {
        return out;
    }

    UsdShadeShader node(producer.GetPrim());
    if (!node) {
        return out;
    }
    return WalkNode(node, UsdShadeUtils::GetBaseNameAndType(producer.GetName()).first, spec,
                    context, depth);
}

void WarnUnsupportedInputs(const UsdShadeShader& shader, SurfaceVocabulary vocabulary,
                           BindingContext& context) {
    for (const UnsupportedInput& item : UnsupportedFor(vocabulary)) {
        UsdShadeInput input = shader.GetInput(TfToken(item.input));
        if (!input) {
            continue;
        }
        const bool authored =
            input.GetAttr().HasAuthoredValue() || input.HasConnectedSource();
        if (authored && context.WarnOnce(String("unsupported:") + item.input)) {
            QL_LOG_WARN("    Material '{}': '{}' is authored but not applied -- {}",
                        context.materialPath, item.input, item.why);
        }
    }
}

}  // namespace

SurfaceReading ReadSurface(const void* shaderPtr, SurfaceVocabulary vocabulary,
                           BindingContext& context) {
    SurfaceReading reading;
    reading.vocabulary = vocabulary;
    reading.materialPath = context.materialPath;

    if (!shaderPtr) {
        return reading;
    }
    const auto& shader = *static_cast<const UsdShadeShader*>(shaderPtr);

    const SurfaceTable table = TableFor(vocabulary);

    // Seed every slot with the node definition's default, so a converter reads
    // the value the renderer would have used rather than a zero this loader
    // invented.
    for (const SurfaceInputSpec& spec : table) {
        reading[spec.slot].value = spec.specDefault;
    }

    for (const SurfaceInputSpec& spec : table) {
        UsdShadeInput input = shader.GetInput(TfToken(spec.input));
        if (!input) {
            continue;
        }
        InputBinding binding = ResolveInputBinding(input, spec, context, 0);
        SlotValue& slot = reading[spec.slot];
        if (binding.value) {
            slot.value = *binding.value;
            slot.authored = true;
        }
        if (binding.texture) {
            slot.texture = std::move(binding.texture);
            slot.authored = true;
        }
    }

    WarnUnsupportedInputs(shader, vocabulary, context);
    return reading;
}

}  // namespace quantiloom::usd

#else  // QUANTILOOM_USE_OPENUSD

namespace quantiloom::usd {

SurfaceReading ReadSurface(const void* /* shaderPtr */, SurfaceVocabulary vocabulary,
                           BindingContext& context) {
    SurfaceReading reading;
    reading.vocabulary = vocabulary;
    reading.materialPath = context.materialPath;
    return reading;
}

}  // namespace quantiloom::usd

#endif  // QUANTILOOM_USE_OPENUSD

// ============================================================================
// Everything below is pxr-free: the recipes and the scalar semantics
// ============================================================================

namespace quantiloom::usd {

namespace {

constexpr f32 kTwoPi = 6.283185307179586f;

constexpr ChannelSel kChannelByIndex[4] = {ChannelSel::R, ChannelSel::G, ChannelSel::B,
                                           ChannelSel::A};

/// Which byte of the source an input reads. A colour selector feeding a scalar
/// slot reads red, which is what a grey map broadcast to RGB makes correct.
u32 SourceChannel(ChannelSel channel) {
    switch (channel) {
        case ChannelSel::G: return 1;
        case ChannelSel::B: return 2;
        case ChannelSel::A: return 3;
        default:            return 0;
    }
}

/// A `scale` with no `bias` is exactly a factor, and a factor costs nothing --
/// no decode, no requantisation, and no change to the pixels the spectral
/// unmixer reads. A bias cannot be folded, so that case is baked instead.
bool ScalarScaleFolds(const TextureRef& ref, u32 channel) {
    return ref.bias[static_cast<int>(channel)] == 0.0f;
}

bool ColourScaleFolds(const TextureRef& ref) {
    return ref.bias.x == 0.0f && ref.bias.y == 0.0f && ref.bias.z == 0.0f;
}

void PutChannel(SlotRecipe& recipe, usize dst, const TextureRef& ref, u32 sourceChannel,
                bool foldsIntoFactor) {
    SlotRecipe::Src src;
    src.path = ref.absolutePath;
    src.channel = kChannelByIndex[sourceChannel];
    if (!foldsIntoFactor) {
        src.scale = ref.scale[static_cast<int>(sourceChannel)];
        src.bias = ref.bias[static_cast<int>(sourceChannel)];
    }
    recipe.dst[dst] = std::move(src);
}

bool HasExtension(const String& path, const char* extension) {
    String ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == extension;
}

/**
 * @brief Whether an entry's pixels are sRGB-encoded
 *
 * An explicit token wins. Otherwise the destination slot decides -- a base
 * colour is colour, a roughness map is data -- rather than the file extension,
 * because a roughness map saved as PNG looks exactly like a colour map on disk.
 * A float format is the one exception: .exr and .hdr carry linear values by
 * construction and there is nothing to decode.
 */
bool IsSrgbEntry(const TextureRef& ref, bool colourSlot) {
    if (HasExtension(ref.absolutePath, ".exr") || HasExtension(ref.absolutePath, ".hdr")) {
        return false;
    }
    if (ref.colourSpace == ColourSpaceHint::Srgb) {
        return true;
    }
    if (ref.colourSpace == ColourSpaceHint::Linear) {
        return false;
    }
    return colourSlot;
}

UvTransform UvFor(const TextureRef& ref) {
    return kFlipUsdV ? ConjugateByVFlip(ref.uv) : ref.uv;
}

/// The `scale = (2, 2, 2, 1)`, `bias = (-1, -1, -1, 0)` pair a UsdUVTexture
/// carries on a normal map, because UsdPreviewSurface's `normal` input wants
/// [-1, 1] and the file holds [0, 1]. The shader already does that decode, so
/// baking it would apply it twice.
bool IsNormalDecodeIdiom(const TextureRef& ref) {
    return ref.scale.x == 2.0f && ref.scale.y == 2.0f && ref.scale.z == 2.0f &&
           ref.bias.x == -1.0f && ref.bias.y == -1.0f && ref.bias.z == -1.0f;
}

bool IsAffineDefault(const TextureRef& ref) {
    return ref.scale == glm::vec4(1.0f) && ref.bias == glm::vec4(0.0f);
}

f32 ScalarFactorFor(const SlotValue& slot) {
    if (!slot.HasTexture()) {
        return slot.value.x;
    }
    const TextureRef& ref = *slot.texture;
    const u32 channel = SourceChannel(ref.channels);
    return ScalarScaleFolds(ref, channel) ? ref.scale[static_cast<int>(channel)] : 1.0f;
}

glm::vec3 ColourFactorFor(const SlotValue& slot) {
    if (!slot.HasTexture()) {
        return glm::vec3(slot.value);
    }
    const TextureRef& ref = *slot.texture;
    return ColourScaleFolds(ref) ? glm::vec3(ref.scale) : glm::vec3(1.0f);
}

/// One image into one destination channel, linear, everything else filled.
i32 SingleChannelEntry(const SlotValue& slot, usize dst, const char* debugName,
                       UsdTextureBank& bank, std::vector<Texture>& textures) {
    const TextureRef& ref = *slot.texture;
    const u32 channel = SourceChannel(ref.channels);
    SlotRecipe recipe;
    PutChannel(recipe, dst, ref, channel, ScalarScaleFolds(ref, channel));
    recipe.srgb = IsSrgbEntry(ref, false);
    recipe.sampler = ref.sampler;
    recipe.debugName = debugName;
    return bank.Materialise(recipe, textures);
}

/// An image passed through untouched, for the two normal slots.
i32 NormalEntry(const SurfaceReading& reading, const SlotValue& slot, const char* debugName,
                UsdTextureBank& bank, std::vector<Texture>& textures) {
    const TextureRef& ref = *slot.texture;
    if (!IsAffineDefault(ref) && !IsNormalDecodeIdiom(ref)) {
        QL_LOG_WARN("    Material '{}': a normal map with a scale or bias that is not the "
                    "[0,1] -> [-1,1] idiom is passed through unchanged; the shader does "
                    "that decode itself", reading.materialPath);
    }
    SlotRecipe recipe;
    for (u32 i = 0; i < 4; ++i) {
        PutChannel(recipe, i, ref, i, /*foldsIntoFactor=*/true);
    }
    recipe.srgb = false;
    recipe.sampler = ref.sampler;
    recipe.debugName = debugName;
    return bank.Materialise(recipe, textures);
}

}  // namespace

// ============================================================================
// CollectTextureSources
// ============================================================================

void CollectTextureSources(const SurfaceReading& reading,
                           std::unordered_set<String>& outPaths) {
    for (const SlotValue& slot : reading.slots) {
        if (slot.texture && !slot.texture->absolutePath.empty()) {
            outPaths.insert(slot.texture->absolutePath);
        }
    }
}

// ============================================================================
// ApplySlotTextures
// ============================================================================

void ApplySlotTextures(const SurfaceReading& reading, Material& material,
                       UsdTextureBank& bank, std::vector<Texture>& textures) {
    const SlotValue& baseColour = reading[SurfaceSlot::BaseColor];
    const SlotValue& opacity = reading[SurfaceSlot::Opacity];

    // ------------------------------------------------------------------
    // baseColor: RGB colour, A opacity
    // ------------------------------------------------------------------
    if (baseColour.HasTexture() || opacity.HasTexture()) {
        SlotRecipe recipe;
        const TextureRef* colourRef = baseColour.texture ? &*baseColour.texture : nullptr;

        if (colourRef != nullptr) {
            const bool folds = ColourScaleFolds(*colourRef);
            if (colourRef->channels == ChannelSel::RGB || colourRef->channels == ChannelSel::RGBA) {
                for (u32 i = 0; i < 3; ++i) {
                    PutChannel(recipe, i, *colourRef, i, folds);
                }
            } else {
                const u32 channel = SourceChannel(colourRef->channels);
                for (u32 i = 0; i < 3; ++i) {
                    PutChannel(recipe, i, *colourRef, channel, folds);
                }
            }
            recipe.sampler = colourRef->sampler;
        }

        if (opacity.HasTexture()) {
            const TextureRef& ref = *opacity.texture;
            const u32 channel = SourceChannel(ref.channels);
            PutChannel(recipe, 3, ref, channel, ScalarScaleFolds(ref, channel));
            if (colourRef == nullptr) {
                recipe.sampler = ref.sampler;
            }
        }

        recipe.srgb = colourRef != nullptr && IsSrgbEntry(*colourRef, true);
        // A cut-out or blended material has its alpha read on the CPU too, for
        // the thermal view factors -- the same reason GltfLoader retains them.
        recipe.retainCpuPixels = material.alphaMode != Material::AlphaMode::Opaque;
        recipe.debugName = "baseColor";

        material.baseColorTextureIndex = bank.Materialise(recipe, textures);
        if (material.baseColorTextureIndex >= 0) {
            material.baseColorUv = UvFor(colourRef != nullptr ? *colourRef : *opacity.texture);
        }
    }

    // ------------------------------------------------------------------
    // metallicRoughness: G roughness, B metallic -- the shader's convention,
    // which is why two separate USD images have to be repacked into one entry
    // ------------------------------------------------------------------
    const SlotValue& roughness = reading[SurfaceSlot::Roughness];
    const SlotValue& metallic = reading[SurfaceSlot::Metallic];
    if (roughness.HasTexture() || metallic.HasTexture()) {
        SlotRecipe recipe;
        if (roughness.HasTexture()) {
            const TextureRef& ref = *roughness.texture;
            const u32 channel = SourceChannel(ref.channels);
            PutChannel(recipe, 1, ref, channel, ScalarScaleFolds(ref, channel));
            recipe.sampler = ref.sampler;
        }
        if (metallic.HasTexture()) {
            const TextureRef& ref = *metallic.texture;
            const u32 channel = SourceChannel(ref.channels);
            PutChannel(recipe, 2, ref, channel, ScalarScaleFolds(ref, channel));
            if (!roughness.HasTexture()) {
                recipe.sampler = ref.sampler;
            }
        }
        // 255 is 1.0, so the untextured half of the pair keeps its scalar.
        recipe.srgb = false;
        recipe.debugName = "metallicRoughness";
        material.metallicRoughnessTextureIndex = bank.Materialise(recipe, textures);
        if (material.metallicRoughnessTextureIndex >= 0) {
            material.metallicRoughnessUv =
                UvFor(roughness.HasTexture() ? *roughness.texture : *metallic.texture);
        }
    }

    // ------------------------------------------------------------------
    // normals
    // ------------------------------------------------------------------
    if (const SlotValue& normal = reading[SurfaceSlot::Normal]; normal.HasTexture()) {
        material.normalTextureIndex = NormalEntry(reading, normal, "normal", bank, textures);
        if (material.normalTextureIndex >= 0) {
            material.normalUv = UvFor(*normal.texture);
            material.normalScale = normal.texture->normalScale;
        }
    }
    if (const SlotValue& coatNormal = reading[SurfaceSlot::ClearcoatNormal];
        coatNormal.HasTexture()) {
        material.clearcoatNormalTextureIndex =
            NormalEntry(reading, coatNormal, "clearcoatNormal", bank, textures);
        if (material.clearcoatNormalTextureIndex >= 0) {
            material.clearcoatNormalUv = UvFor(*coatNormal.texture);
            material.clearcoatNormalScale = coatNormal.texture->normalScale;
        }
    }

    // ------------------------------------------------------------------
    // emissive
    // ------------------------------------------------------------------
    if (const SlotValue& emissive = reading[SurfaceSlot::Emissive]; emissive.HasTexture()) {
        const TextureRef& ref = *emissive.texture;
        const bool folds = ColourScaleFolds(ref);
        SlotRecipe recipe;
        if (ref.channels == ChannelSel::RGB || ref.channels == ChannelSel::RGBA) {
            for (u32 i = 0; i < 3; ++i) {
                PutChannel(recipe, i, ref, i, folds);
            }
        } else {
            const u32 channel = SourceChannel(ref.channels);
            for (u32 i = 0; i < 3; ++i) {
                PutChannel(recipe, i, ref, channel, folds);
            }
        }
        recipe.srgb = IsSrgbEntry(ref, true);
        recipe.sampler = ref.sampler;
        recipe.debugName = "emissive";
        material.emissiveTextureIndex = bank.Materialise(recipe, textures);
        if (material.emissiveTextureIndex >= 0) {
            material.emissiveUv = UvFor(ref);
        }
    }

    // ------------------------------------------------------------------
    // sheen: colour in RGB, roughness in A. When both come from one file they
    // are one entry, which the shader reads twice.
    // ------------------------------------------------------------------
    const SlotValue& sheenColour = reading[SurfaceSlot::SheenColor];
    const SlotValue& sheenRoughness = reading[SurfaceSlot::SheenRoughness];
    const bool sheenShareOneFile =
        sheenColour.HasTexture() && sheenRoughness.HasTexture() &&
        sheenColour.texture->absolutePath == sheenRoughness.texture->absolutePath;

    if (sheenColour.HasTexture()) {
        const TextureRef& ref = *sheenColour.texture;
        const bool folds = ColourScaleFolds(ref);
        SlotRecipe recipe;
        for (u32 i = 0; i < 3; ++i) {
            PutChannel(recipe, i, ref, ref.channels == ChannelSel::RGB ||
                                               ref.channels == ChannelSel::RGBA
                                           ? i
                                           : SourceChannel(ref.channels), folds);
        }
        if (sheenShareOneFile) {
            const TextureRef& rough = *sheenRoughness.texture;
            const u32 channel = SourceChannel(rough.channels);
            PutChannel(recipe, 3, rough, channel, ScalarScaleFolds(rough, channel));
        }
        recipe.srgb = IsSrgbEntry(ref, true);
        recipe.sampler = ref.sampler;
        recipe.debugName = "sheenColor";
        material.sheenColorTextureIndex = bank.Materialise(recipe, textures);
        if (material.sheenColorTextureIndex >= 0) {
            material.sheenColorUv = UvFor(ref);
        }
    }
    if (sheenRoughness.HasTexture()) {
        if (sheenShareOneFile) {
            // Alpha is not gamma-encoded, so reading it out of an sRGB entry is
            // the same number it would have had in a linear one.
            material.sheenRoughnessTextureIndex = material.sheenColorTextureIndex;
            material.sheenRoughnessUv = material.sheenColorUv;
        } else {
            material.sheenRoughnessTextureIndex =
                SingleChannelEntry(sheenRoughness, 3, "sheenRoughness", bank, textures);
            if (material.sheenRoughnessTextureIndex >= 0) {
                material.sheenRoughnessUv = UvFor(*sheenRoughness.texture);
            }
        }
    }

    // ------------------------------------------------------------------
    // specular: strength in A, colour in RGB
    // ------------------------------------------------------------------
    if (const SlotValue& specular = reading[SurfaceSlot::Specular]; specular.HasTexture()) {
        material.specularTextureIndex =
            SingleChannelEntry(specular, 3, "specular", bank, textures);
        if (material.specularTextureIndex >= 0) {
            material.specularUv = UvFor(*specular.texture);
        }
    }
    if (const SlotValue& specularColour = reading[SurfaceSlot::SpecularColor];
        specularColour.HasTexture()) {
        const TextureRef& ref = *specularColour.texture;
        const bool folds = ColourScaleFolds(ref);
        SlotRecipe recipe;
        for (u32 i = 0; i < 3; ++i) {
            PutChannel(recipe, i, ref, ref.channels == ChannelSel::RGB ||
                                               ref.channels == ChannelSel::RGBA
                                           ? i
                                           : SourceChannel(ref.channels), folds);
        }
        recipe.srgb = IsSrgbEntry(ref, true);
        recipe.sampler = ref.sampler;
        recipe.debugName = "specularColor";
        material.specularColorTextureIndex = bank.Materialise(recipe, textures);
        if (material.specularColorTextureIndex >= 0) {
            material.specularColorUv = UvFor(ref);
        }
    }

    // ------------------------------------------------------------------
    // anisotropy: RG is the direction, B the strength. RG fills to (1, 0.5),
    // which decodes to the tangent direction (1, 0) -- (0, 0) would leave the
    // shader normalising a zero vector.
    // ------------------------------------------------------------------
    if (const SlotValue& anisotropy = reading[SurfaceSlot::AnisotropyStrength];
        anisotropy.HasTexture()) {
        const TextureRef& ref = *anisotropy.texture;
        const u32 channel = SourceChannel(ref.channels);
        SlotRecipe recipe;
        recipe.fill = {255, 128, 255, 255};
        PutChannel(recipe, 2, ref, channel, ScalarScaleFolds(ref, channel));
        recipe.srgb = false;
        recipe.sampler = ref.sampler;
        recipe.debugName = "anisotropy";
        material.anisotropyTextureIndex = bank.Materialise(recipe, textures);
        if (material.anisotropyTextureIndex >= 0) {
            material.anisotropyUv = UvFor(ref);
        }
    }

    // ------------------------------------------------------------------
    // clearcoat weight in R, roughness in G
    // ------------------------------------------------------------------
    if (const SlotValue& coat = reading[SurfaceSlot::ClearcoatWeight]; coat.HasTexture()) {
        material.clearcoatTextureIndex =
            SingleChannelEntry(coat, 0, "clearcoat", bank, textures);
        if (material.clearcoatTextureIndex >= 0) {
            material.clearcoatUv = UvFor(*coat.texture);
        }
    }
    if (const SlotValue& coatRoughness = reading[SurfaceSlot::ClearcoatRoughness];
        coatRoughness.HasTexture()) {
        material.clearcoatRoughnessTextureIndex =
            SingleChannelEntry(coatRoughness, 1, "clearcoatRoughness", bank, textures);
        if (material.clearcoatRoughnessTextureIndex >= 0) {
            material.clearcoatRoughnessUv = UvFor(*coatRoughness.texture);
        }
    }

    // ------------------------------------------------------------------
    // transmission in R and thickness in G, the glTF channels
    // ------------------------------------------------------------------
    if (const SlotValue& transmission = reading[SurfaceSlot::Transmission];
        transmission.HasTexture()) {
        material.transmissionTextureIndex =
            SingleChannelEntry(transmission, 0, "transmission", bank, textures);
    }
    if (const SlotValue& thickness = reading[SurfaceSlot::Thickness]; thickness.HasTexture()) {
        material.thicknessTextureIndex =
            SingleChannelEntry(thickness, 1, "thickness", bank, textures);
    }

    if (reading.HasTexture(SurfaceSlot::Occlusion)) {
        QL_LOG_DEBUG("    Material '{}': an occlusion map is bound but not used; ambient "
                     "occlusion is a rasteriser's approximation of what this renderer "
                     "traces", reading.materialPath);
    }
}

// ============================================================================
// Scalar semantics
// ============================================================================

namespace {

bool IsGrey(const glm::vec3& colour) {
    return std::abs(colour.r - colour.g) < 1e-6f && std::abs(colour.g - colour.b) < 1e-6f;
}

void ConvertUsdPreviewSurface(const SurfaceReading& reading, Material& material) {
    material.baseColorFactor =
        glm::vec4(ColourFactorFor(reading[SurfaceSlot::BaseColor]), 1.0f);
    material.metallicFactor = ScalarFactorFor(reading[SurfaceSlot::Metallic]);
    material.roughnessFactor = ScalarFactorFor(reading[SurfaceSlot::Roughness]);
    material.emissiveFactor = ColourFactorFor(reading[SurfaceSlot::Emissive]);
    material.ior = reading.Scalar(SurfaceSlot::Ior);
    material.clearcoatFactor = ScalarFactorFor(reading[SurfaceSlot::ClearcoatWeight]);
    material.clearcoatRoughnessFactor =
        ScalarFactorFor(reading[SurfaceSlot::ClearcoatRoughness]);

    // ------------------------------------------------------------------
    // Opacity
    // ------------------------------------------------------------------
    // opacityThreshold turns the surface into a cut-out at that threshold; with
    // no threshold, an opacity below one or an opacity map is a blend.
    const SlotValue& opacity = reading[SurfaceSlot::Opacity];
    const f32 threshold = reading.Scalar(SurfaceSlot::OpacityThreshold);
    const f32 alpha = ScalarFactorFor(opacity);

    material.baseColorFactor.a = alpha;
    if (threshold > 0.0f) {
        material.alphaMode = Material::AlphaMode::Mask;
        material.alphaCutoff = threshold;
    } else if (opacity.HasTexture() || alpha < 1.0f) {
        material.alphaMode = Material::AlphaMode::Blend;
        if (reading.Scalar(SurfaceSlot::OpacityMode) >= 0.5f) {
            // opacityMode = presence with no threshold. The spec says a presence
            // opacity is a cut-out decided per sample; without a threshold there
            // is nothing to cut at, so it blends and says so.
            QL_LOG_WARN("    Material '{}': opacityMode is 'presence' with no "
                        "opacityThreshold; rendering it as a blend",
                        reading.materialPath);
        }
    }

    // ------------------------------------------------------------------
    // The specular workflow
    // ------------------------------------------------------------------
    if (reading.Scalar(SurfaceSlot::UseSpecularWorkflow) >= 0.5f) {
        material.specularColorFactor = ColourFactorFor(reading[SurfaceSlot::SpecularColor]);
        QL_LOG_WARN("    Material '{}': useSpecularWorkflow has no BRDF here; its "
                    "specularColor is applied as KHR_materials_specular's colour, which "
                    "is an approximation", reading.materialPath);
    }

    if (reading[SurfaceSlot::Occlusion].authored) {
        QL_LOG_DEBUG("    Material '{}': occlusion is read and dropped",
                     reading.materialPath);
    }
}

void ConvertStandardSurface(const SurfaceReading& reading, Material& material) {
    // base is a weight on base_color; the two are one factor here.
    const f32 baseWeight = reading.Scalar(SurfaceSlot::BaseWeight);
    material.baseColorFactor =
        glm::vec4(baseWeight * ColourFactorFor(reading[SurfaceSlot::BaseColor]), 1.0f);
    material.metallicFactor = ScalarFactorFor(reading[SurfaceSlot::Metallic]);
    material.roughnessFactor = ScalarFactorFor(reading[SurfaceSlot::Roughness]);
    material.ior = reading.Scalar(SurfaceSlot::Ior);

    material.specularFactor = ScalarFactorFor(reading[SurfaceSlot::Specular]);
    material.specularColorFactor = ColourFactorFor(reading[SurfaceSlot::SpecularColor]);

    material.anisotropyStrength = ScalarFactorFor(reading[SurfaceSlot::AnisotropyStrength]);
    // specular_rotation is in turns, anisotropyRotation in radians.
    material.anisotropyRotation = reading.Scalar(SurfaceSlot::AnisotropyRotation) * kTwoPi;

    // ------------------------------------------------------------------
    // Transmission
    // ------------------------------------------------------------------
    material.transmission = ScalarFactorFor(reading[SurfaceSlot::Transmission]);
    const f32 depth = reading.Scalar(SurfaceSlot::TransmissionDepth);
    const glm::vec3 transmissionColour = reading.Colour(SurfaceSlot::TransmissionColor);
    if (depth > 0.0f) {
        material.attenuationColor = transmissionColour;
        material.attenuationDistance = depth;
    } else if (transmissionColour != glm::vec3(1.0f)) {
        // With depth zero, standard_surface's transmission_color is a tint
        // independent of how far the light travelled. Beer-Lambert is not that,
        // and dressing one up as the other invents an absorption coefficient
        // nobody supplied.
        QL_LOG_WARN("    Material '{}': transmission_color with transmission_depth = 0 is "
                    "a distance-independent tint, which has no volume-absorption "
                    "equivalent; no attenuation is applied", reading.materialPath);
    }

    // transmission_dispersion is an Abbe number; Material::dispersion is its
    // reciprocal, which is the quantity the Cauchy fit uses.
    const f32 abbe = reading.Scalar(SurfaceSlot::Dispersion);
    material.dispersion = abbe > 0.0f ? 1.0f / abbe : 0.0f;

    // ------------------------------------------------------------------
    // Sheen and coat
    // ------------------------------------------------------------------
    material.sheenColorFactor = reading.Scalar(SurfaceSlot::SheenWeight) *
                                ColourFactorFor(reading[SurfaceSlot::SheenColor]);
    material.sheenRoughnessFactor = ScalarFactorFor(reading[SurfaceSlot::SheenRoughness]);

    material.clearcoatFactor = ScalarFactorFor(reading[SurfaceSlot::ClearcoatWeight]);
    material.clearcoatRoughnessFactor =
        ScalarFactorFor(reading[SurfaceSlot::ClearcoatRoughness]);
    if (reading.Colour(SurfaceSlot::ClearcoatColor) != glm::vec3(1.0f)) {
        QL_LOG_WARN("    Material '{}': coat_color is not applied; the clearcoat layer "
                    "here has no absorption", reading.materialPath);
    }
    if (reading.Scalar(SurfaceSlot::ClearcoatIor) != 1.5f) {
        QL_LOG_WARN("    Material '{}': coat_IOR is not applied; the clearcoat layer's "
                    "index is fixed", reading.materialPath);
    }

    // emission is a weight on emission_color, folded the same way
    // KHR_materials_emissive_strength is.
    material.emissiveFactor = reading.Scalar(SurfaceSlot::EmissiveWeight) *
                              ColourFactorFor(reading[SurfaceSlot::Emissive]);

    // ------------------------------------------------------------------
    // Opacity: a colour in standard_surface, a scalar here
    // ------------------------------------------------------------------
    const SlotValue& opacity = reading[SurfaceSlot::Opacity];
    const glm::vec3 opacityColour = reading.Colour(SurfaceSlot::Opacity);
    if (!IsGrey(opacityColour)) {
        QL_LOG_WARN("    Material '{}': a coloured opacity has no equivalent; its mean is "
                    "used as alpha", reading.materialPath);
    }
    const f32 alpha = opacity.HasTexture()
                          ? ScalarFactorFor(opacity)
                          : (opacityColour.r + opacityColour.g + opacityColour.b) / 3.0f;
    material.baseColorFactor.a = alpha;
    if (opacity.HasTexture() || alpha < 1.0f) {
        material.alphaMode = Material::AlphaMode::Blend;
    }

    // thin_walled is geometry's business in USD and the material's here.
    if (reading.Scalar(SurfaceSlot::ThinWalled) >= 0.5f) {
        material.doubleSided = true;
    }
}

}  // namespace

void ConvertSurface(const SurfaceReading& reading, Material& material) {
    switch (reading.vocabulary) {
        case SurfaceVocabulary::UsdPreviewSurface:
            ConvertUsdPreviewSurface(reading, material);
            return;
        case SurfaceVocabulary::StandardSurface:
            ConvertStandardSurface(reading, material);
            return;
        case SurfaceVocabulary::GltfPbr:
        case SurfaceVocabulary::OpenPbrSurface:
        case SurfaceVocabulary::Unknown:
            break;
    }
}

}  // namespace quantiloom::usd
