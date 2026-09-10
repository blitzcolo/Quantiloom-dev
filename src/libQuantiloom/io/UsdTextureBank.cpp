/**
 * @file UsdTextureBank.cpp
 * @brief Implementation of the decoded-source bank and the slot repacking
 */

#include "io/UsdTextureBank.hpp"

#include "core/Log.hpp"
#include "io/ImageIO.hpp"

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <thread>

namespace quantiloom::usd {

namespace {

const char* ChannelName(ChannelSel channel) {
    switch (channel) {
        case ChannelSel::R:    return "r";
        case ChannelSel::G:    return "g";
        case ChannelSel::B:    return "b";
        case ChannelSel::A:    return "a";
        case ChannelSel::RGB:  return "rgb";
        case ChannelSel::RGBA: return "rgba";
    }
    return "?";
}

/// Which byte of an RGBA8 texel a single-channel selector reads. `rgb` and
/// `rgba` collapse to red here: a colour output feeding a scalar input is a
/// separate case ReadSurface warns about and narrows to R before it gets this
/// far.
u32 ChannelByte(ChannelSel channel) {
    switch (channel) {
        case ChannelSel::G: return 1;
        case ChannelSel::B: return 2;
        case ChannelSel::A: return 3;
        default:            return 0;
    }
}

f32 SrgbToLinear(f32 value) {
    return value <= 0.04045f ? value / 12.92f
                             : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

f32 LinearToSrgb(f32 value) {
    return value <= 0.0031308f ? 12.92f * value
                               : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

u8 Quantise(f32 value) {
    return static_cast<u8>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

}  // namespace

// ============================================================================
// SlotRecipe
// ============================================================================

String SlotRecipe::Key() const {
    String key;
    key.reserve(256);
    static const char* kDstName[4] = {"R", "G", "B", "A"};
    for (usize i = 0; i < 4; ++i) {
        key += kDstName[i];
        key += '=';
        if (dst[i]) {
            key += dst[i]->path;
            key += ':';
            key += ChannelName(dst[i]->channel);
            key += '*';
            key += std::to_string(dst[i]->scale);
            key += '+';
            key += std::to_string(dst[i]->bias);
        } else {
            key += "fill";
            key += std::to_string(static_cast<int>(fill[i]));
        }
        key += '|';
    }
    key += srgb ? "srgb|" : "linear|";
    key += retainCpuPixels ? "cpu|" : "gpu|";
    key += std::to_string(static_cast<int>(sampler.wrapS));
    key += std::to_string(static_cast<int>(sampler.wrapT));
    key += std::to_string(static_cast<int>(sampler.minFilter));
    key += std::to_string(static_cast<int>(sampler.magFilter));
    return key;
}

bool SlotRecipe::IsIdentityOf(const String& path) const {
    static const ChannelSel kInOrder[4] = {ChannelSel::R, ChannelSel::G,
                                           ChannelSel::B, ChannelSel::A};
    for (usize i = 0; i < 4; ++i) {
        if (!dst[i] || dst[i]->path != path || dst[i]->channel != kInOrder[i] ||
            !dst[i]->IsAffineIdentity()) {
            return false;
        }
    }
    return true;
}

std::vector<String> SlotRecipe::Sources() const {
    std::vector<String> paths;
    for (const auto& src : dst) {
        if (src && std::find(paths.begin(), paths.end(), src->path) == paths.end()) {
            paths.push_back(src->path);
        }
    }
    return paths;
}

// ============================================================================
// DecodeTextureFile
// ============================================================================

namespace {

/// The RGBA8 repack both decode paths share.
Texture ToRgba8(const Image& img, const String& sourceUri) {
    Texture tex;
    tex.name = std::filesystem::path(sourceUri).filename().string();
    tex.width = img.width;
    tex.height = img.height;
    tex.channels = 4;  // Always output RGBA for renderer compatibility
    tex.sourceUri = sourceUri;

    const usize pixelCount = static_cast<usize>(img.width) * img.height;
    tex.pixels.resize(pixelCount * 4);

    // Positional indices are right for a PNG or JPEG, where stb_image really
    // does hand back R,G,B in that order, and wrong for an .exr, which comes
    // back in OpenEXR's name-sorted channel order. Asking by name is correct
    // for both, since ImageIO names the stb channels too. See
    // Image::ChannelIndex.
    const u32 cr = img.ChannelIndex("R", 0);
    const u32 cg = img.ChannelIndex("G", 1);
    const u32 cb = img.ChannelIndex("B", 2);
    const u32 ca = img.ChannelIndex("A", 3);
    const u32 grey = img.LuminanceChannelIndex();

    for (usize i = 0; i < pixelCount; ++i) {
        f32 r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;

        if (img.channels == 1) {
            r = g = b = std::clamp(img.data[i], 0.0f, 1.0f);
        } else if (img.channels == 2) {
            r = g = b = std::clamp(img.data[i * 2 + grey], 0.0f, 1.0f);
            a = std::clamp(img.data[i * 2 + (grey == 0 ? 1 : 0)], 0.0f, 1.0f);
        } else if (img.channels == 3) {
            r = std::clamp(img.data[i * 3 + cr], 0.0f, 1.0f);
            g = std::clamp(img.data[i * 3 + cg], 0.0f, 1.0f);
            b = std::clamp(img.data[i * 3 + cb], 0.0f, 1.0f);
        } else if (img.channels >= 4) {
            r = std::clamp(img.data[i * img.channels + cr], 0.0f, 1.0f);
            g = std::clamp(img.data[i * img.channels + cg], 0.0f, 1.0f);
            b = std::clamp(img.data[i * img.channels + cb], 0.0f, 1.0f);
            a = std::clamp(img.data[i * img.channels + ca], 0.0f, 1.0f);
        }

        tex.pixels[i * 4 + 0] = Quantise(r);
        tex.pixels[i * 4 + 1] = Quantise(g);
        tex.pixels[i * 4 + 2] = Quantise(b);
        tex.pixels[i * 4 + 3] = Quantise(a);
    }

    QL_LOG_DEBUG("    Decoded texture '{}' ({}x{}, {} -> 4 channels)",
                 tex.name, tex.width, tex.height, img.channels);

    return tex;
}

}  // namespace

Texture DecodeTextureFile(const String& absolutePath, u32* outSourceChannels) {
    if (absolutePath.empty()) {
        return {};
    }
    auto imageResult = ImageIO::ReadImage(absolutePath);
    if (!imageResult.has_value()) {
        QL_LOG_ERROR("Failed to load texture '{}'", absolutePath);
        return {};
    }
    if (outSourceChannels != nullptr) {
        *outSourceChannels = imageResult.value().channels;
    }
    return ToRgba8(imageResult.value(), absolutePath);
}

Texture DecodeTextureBytes(const String& name, const u8* data, usize size,
                           u32* outSourceChannels) {
    if (data == nullptr || size == 0) {
        return {};
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    u8* decoded = stbi_load_from_memory(data, static_cast<int>(size), &width, &height,
                                        &channels, 0);
    if (decoded == nullptr) {
        QL_LOG_ERROR("Failed to decode packaged texture '{}': {}", name,
                     stbi_failure_reason() != nullptr ? stbi_failure_reason() : "unknown");
        return {};
    }

    // Through Image so the channel-name lookup and the clamping are the same
    // code the file path uses; a packaged texture must not decode differently
    // from the same bytes on disk.
    Image img;
    img.width = static_cast<u32>(width);
    img.height = static_cast<u32>(height);
    img.channels = static_cast<u32>(channels);
    if (channels == 1) {
        img.channelNames = {"Gray"};
    } else if (channels == 2) {
        img.channelNames = {"Gray", "Alpha"};
    } else if (channels == 3) {
        img.channelNames = {"R", "G", "B"};
    } else {
        img.channelNames = {"R", "G", "B", "A"};
    }
    img.data.resize(static_cast<usize>(width) * height * channels);
    for (usize i = 0; i < img.data.size(); ++i) {
        img.data[i] = static_cast<f32>(decoded[i]) / 255.0f;
    }
    stbi_image_free(decoded);

    if (outSourceChannels != nullptr) {
        *outSourceChannels = img.channels;
    }
    return ToRgba8(img, name);
}

// ============================================================================
// UsdTextureBank
// ============================================================================

void UsdTextureBank::Preload(const std::unordered_set<String>& absolutePaths) {
    if (absolutePaths.empty()) {
        return;
    }

    unsigned int threads = std::min(8u, std::thread::hardware_concurrency());
    if (threads == 0) {
        threads = 4;
    }
    QL_LOG_INFO("  Decoding {} texture files using up to {} threads",
                absolutePaths.size(), threads);

    std::vector<String> paths;
    paths.reserve(absolutePaths.size());
    for (const String& path : absolutePaths) {
        if (m_sources.find(path) == m_sources.end()) {
            paths.push_back(path);
        }
    }

    struct Decoded {
        String path;
        Texture texture;
        u32 sourceChannels = 4;
    };
    std::vector<std::future<Decoded>> pending;
    pending.reserve(paths.size());
    for (const String& path : paths) {
        // Bytes that were handed over rather than read from disk -- a texture
        // inside a .usdz -- decode here too, so they cost the same as any other.
        if (const auto encoded = m_encoded.find(path); encoded != m_encoded.end()) {
            const std::vector<u8>& bytes = encoded->second;
            pending.push_back(std::async(std::launch::async, [path, &bytes]() {
                Decoded out;
                out.path = path;
                out.texture = DecodeTextureBytes(path, bytes.data(), bytes.size(),
                                                 &out.sourceChannels);
                return out;
            }));
            continue;
        }
        pending.push_back(std::async(std::launch::async, [path]() {
            Decoded out;
            out.path = path;
            out.texture = DecodeTextureFile(path, &out.sourceChannels);
            return out;
        }));
    }

    usize decoded = 0;
    for (auto& future : pending) {
        Decoded out = future.get();
        if (out.texture.width > 0) {
            ++decoded;
        }
        m_sourceChannels[out.path] = out.sourceChannels;
        m_sources.emplace(std::move(out.path), std::move(out.texture));
    }

    QL_LOG_INFO("  Decoded {} of {} texture files", decoded, paths.size());
    m_encoded.clear();
}

i32 UsdTextureBank::Materialise(const SlotRecipe& recipe, std::vector<Texture>& textures) {
    const String key = recipe.Key();
    if (const auto it = m_byKey.find(key); it != m_byKey.end()) {
        return it->second;
    }

    // A source that never decoded is dropped from the recipe rather than
    // failing it: an opacity map that is missing should not also take the base
    // colour with it. A recipe with nothing left has nothing to build.
    SlotRecipe resolved = recipe;
    const Texture* first = nullptr;
    for (auto& src : resolved.dst) {
        if (!src) {
            continue;
        }
        const auto it = m_sources.find(src->path);
        if (it == m_sources.end() || it->second.width == 0) {
            QL_LOG_WARN("    Texture '{}' did not decode; leaving that channel at its fill",
                        src->path);
            src.reset();
            continue;
        }
        if (first == nullptr) {
            first = &it->second;
        }

        // Alpha asked of a file that had none. The repack filled it with 255,
        // so an opacity bound this way is fully opaque everywhere, which is
        // probably not what binding an opacity map meant. Once per file.
        if (src->channel == ChannelSel::A) {
            const auto channels = m_sourceChannels.find(src->path);
            const bool hasAlpha = channels == m_sourceChannels.end() ||
                                  channels->second == 2 || channels->second >= 4;
            if (!hasAlpha && m_warnedNoAlpha.insert(src->path).second) {
                QL_LOG_WARN("    Texture '{}' has {} channels and no alpha; the 'a' "
                            "output reads as 1 everywhere",
                            src->path, channels->second);
            }
        }
    }
    if (first == nullptr) {
        return -1;
    }

    const u32 width = first->width;
    const u32 height = first->height;

    Texture entry;
    if (resolved.dst[0] && resolved.IsIdentityOf(resolved.dst[0]->path)) {
        // Four channels of one image, untouched. Copy the decoded source rather
        // than walking every texel to reproduce it.
        entry = *first;
    } else {
        entry.width = width;
        entry.height = height;
        entry.channels = 4;
        entry.pixels.assign(static_cast<usize>(width) * height * 4, 0);

        // A scale folds into the material's factor and leaves the pixels alone;
        // a bias cannot, so it is baked, and the bytes a spectral unmixer reads
        // off this entry are then the biased ones. Worth a line in the log.
        for (usize channel = 0; channel < 4; ++channel) {
            const auto& src = resolved.dst[channel];
            if (src && src->bias != 0.0f) {
                QL_LOG_INFO("    Texture entry '{}': scale {} and bias {} on '{}' are baked "
                            "into the pixels",
                            resolved.debugName.empty() ? key : resolved.debugName,
                            src->scale, src->bias, src->path);
                break;
            }
        }

        for (usize channel = 0; channel < 4; ++channel) {
            const auto& src = resolved.dst[channel];
            if (!src) {
                for (usize i = 0; i < static_cast<usize>(width) * height; ++i) {
                    entry.pixels[i * 4 + channel] = resolved.fill[channel];
                }
                continue;
            }

            const Texture& source = m_sources.at(src->path);
            if (source.width != width || source.height != height) {
                QL_LOG_WARN("    Repacking '{}' ({}x{}) onto a {}x{} slot by nearest "
                            "sample; the maps in one slot should share a resolution",
                            src->path, source.width, source.height, width, height);
            }

            // Alpha is never gamma-encoded, so an affine on it stays in encoded
            // space whatever the entry's colour space is.
            const bool decodeForAffine =
                resolved.srgb && channel < 3 && !src->IsAffineIdentity();
            const u32 byte = ChannelByte(src->channel);

            for (u32 y = 0; y < height; ++y) {
                const u32 sy = source.height == height
                                   ? y
                                   : std::min(source.height - 1,
                                              y * source.height / std::max(height, 1u));
                for (u32 x = 0; x < width; ++x) {
                    const u32 sx = source.width == width
                                       ? x
                                       : std::min(source.width - 1,
                                                  x * source.width / std::max(width, 1u));
                    const usize si = (static_cast<usize>(sy) * source.width + sx) * 4 + byte;
                    const usize di = (static_cast<usize>(y) * width + x) * 4 + channel;

                    f32 value = static_cast<f32>(source.pixels[si]) / 255.0f;
                    if (!src->IsAffineIdentity()) {
                        if (decodeForAffine) {
                            value = LinearToSrgb(
                                std::clamp(SrgbToLinear(value) * src->scale + src->bias,
                                           0.0f, 1.0f));
                        } else {
                            value = value * src->scale + src->bias;
                        }
                    }
                    entry.pixels[di] = Quantise(value);
                }
            }
        }
    }

    entry.isSRGB = resolved.srgb;
    entry.sampler = resolved.sampler;
    entry.retainCpuPixels = resolved.retainCpuPixels;
    entry.name = resolved.debugName.empty() ? entry.name : resolved.debugName;

    const i32 index = static_cast<i32>(textures.size());
    textures.push_back(std::move(entry));
    m_byKey.emplace(key, index);
    return index;
}

void UsdTextureBank::ReleaseSources() {
    m_sources.clear();
    m_encoded.clear();
}

// ============================================================================
// Conversions
// ============================================================================

UvTransform ConjugateByVFlip(const UvTransform& transform) {
    // With F(u, v) = (u, 1 - v) and M(x) = R(theta) S x + t, the conjugate
    // F o M o F has matrix J R S J = R(-theta) S -- so the scale is unchanged
    // and the rotation negates -- and translation J A e + J t + e with
    // e = (0, 1), which is the offset below.
    UvTransform out;
    out.scale = transform.scale;
    out.rotation = -transform.rotation;
    out.offset.x = transform.offset.x - transform.scale.y * std::sin(transform.rotation);
    out.offset.y = 1.0f - transform.offset.y - transform.scale.y * std::cos(transform.rotation);
    return out;
}

TextureSampler::WrapMode WrapFromToken(StringView token, bool& outUnsupported) {
    outUnsupported = false;
    if (token == "clamp") {
        return TextureSampler::WrapMode::ClampToEdge;
    }
    if (token == "mirror") {
        return TextureSampler::WrapMode::MirroredRepeat;
    }
    if (token == "black") {
        // A border colour of zero has no equivalent here; clamping repeats the
        // edge texel where the scene asked for black, which is visible and
        // wrong in a way worth saying out loud.
        outUnsupported = true;
        return TextureSampler::WrapMode::ClampToEdge;
    }
    // periodic, repeat, useMetadata, and anything unrecognised.
    return TextureSampler::WrapMode::Repeat;
}

TextureSampler::Filter FilterFromToken(StringView token) {
    return token == "closest" ? TextureSampler::Filter::Nearest
                              : TextureSampler::Filter::Linear;
}

}  // namespace quantiloom::usd
