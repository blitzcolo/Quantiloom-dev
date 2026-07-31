#include "mcp/McpImageUtil.hpp"

#include "core/Log.hpp"

// STB_IMAGE_WRITE_IMPLEMENTATION lives in io/ImageIO.cpp. Defining it again here
// would give the linker two copies of every stbi_write_* symbol.
QL_DISABLE_WARNINGS_PUSH
#include <stb_image_write.h>
QL_DISABLE_WARNINGS_POP

#include <algorithm>
#include <cmath>
#include <vector>

namespace quantiloom::mcp {
namespace {

/// IEC 61966-2-1, the same curve ImageIO::WritePNG applies.
f32 LinearToSRGB(const f32 linear) {
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    }
    return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

constexpr const char* kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

String Base64(const std::vector<u8>& bytes) {
    String out;
    out.reserve(((bytes.size() + 2) / 3) * 4);

    usize i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const u32 triple = (static_cast<u32>(bytes[i]) << 16) |
                           (static_cast<u32>(bytes[i + 1]) << 8) | static_cast<u32>(bytes[i + 2]);
        out.push_back(kBase64Alphabet[(triple >> 18) & 0x3F]);
        out.push_back(kBase64Alphabet[(triple >> 12) & 0x3F]);
        out.push_back(kBase64Alphabet[(triple >> 6) & 0x3F]);
        out.push_back(kBase64Alphabet[triple & 0x3F]);
    }

    const usize remaining = bytes.size() - i;
    if (remaining == 1) {
        const u32 triple = static_cast<u32>(bytes[i]) << 16;
        out.push_back(kBase64Alphabet[(triple >> 18) & 0x3F]);
        out.push_back(kBase64Alphabet[(triple >> 12) & 0x3F]);
        out.append("==");
    } else if (remaining == 2) {
        const u32 triple =
            (static_cast<u32>(bytes[i]) << 16) | (static_cast<u32>(bytes[i + 1]) << 8);
        out.push_back(kBase64Alphabet[(triple >> 18) & 0x3F]);
        out.push_back(kBase64Alphabet[(triple >> 12) & 0x3F]);
        out.push_back(kBase64Alphabet[(triple >> 6) & 0x3F]);
        out.push_back('=');
    }

    return out;
}

void AppendToVector(void* context, void* data, const int size) {
    auto* out = static_cast<std::vector<u8>*>(context);
    const auto* bytes = static_cast<const u8*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

}  // namespace

Image Downsample(const Image& image, const u32 maxDimension) {
    if (image.width == 0 || image.height == 0 || image.channels == 0 || maxDimension == 0) {
        return image;
    }
    const u32 longEdge = std::max(image.width, image.height);
    if (longEdge <= maxDimension) {
        return image;
    }

    // Integer box factor. A fractional scale would need interpolation on both
    // axes for a result nobody is measuring; rounding the factor up keeps the
    // output at or below the requested size.
    const u32 factor = (longEdge + maxDimension - 1) / maxDimension;
    const u32 outWidth = std::max(1u, image.width / factor);
    const u32 outHeight = std::max(1u, image.height / factor);

    Image out(outWidth, outHeight, image.channels);
    out.channelNames = image.channelNames;
    out.metadata = image.metadata;

    for (u32 y = 0; y < outHeight; ++y) {
        for (u32 x = 0; x < outWidth; ++x) {
            for (u32 c = 0; c < image.channels; ++c) {
                f32 sum = 0.0f;
                u32 count = 0;
                for (u32 dy = 0; dy < factor; ++dy) {
                    const u32 sy = y * factor + dy;
                    if (sy >= image.height) {
                        break;
                    }
                    for (u32 dx = 0; dx < factor; ++dx) {
                        const u32 sx = x * factor + dx;
                        if (sx >= image.width) {
                            break;
                        }
                        sum += image(sx, sy, c);
                        ++count;
                    }
                }
                out(x, y, c) = count > 0 ? sum / static_cast<f32>(count) : 0.0f;
            }
        }
    }

    return out;
}

String EncodePngBase64(const Image& image) {
    if (image.width == 0 || image.height == 0 || image.channels == 0) {
        QL_LOG_WARN("MCP: cannot encode an empty image");
        return {};
    }

    // PNG takes 1, 3 or 4. A spectral capture with more than that is shown as
    // its first three, which for every mode this renderer produces is the
    // displayable triple.
    const u32 outChannels = image.channels >= 3 ? 3u : 1u;

    std::vector<u8> pixels(static_cast<usize>(image.width) * image.height * outChannels);
    for (u32 y = 0; y < image.height; ++y) {
        for (u32 x = 0; x < image.width; ++x) {
            for (u32 c = 0; c < outChannels; ++c) {
                const f32 clamped = std::clamp(image(x, y, c), 0.0f, 1.0f);
                pixels[(static_cast<usize>(y) * image.width + x) * outChannels + c] =
                    static_cast<u8>(LinearToSRGB(clamped) * 255.0f + 0.5f);
            }
        }
    }

    std::vector<u8> png;
    png.reserve(pixels.size() / 2);
    const int ok = stbi_write_png_to_func(&AppendToVector, &png, static_cast<int>(image.width),
                                          static_cast<int>(image.height),
                                          static_cast<int>(outChannels), pixels.data(),
                                          static_cast<int>(image.width * outChannels));
    if (ok == 0 || png.empty()) {
        QL_LOG_ERROR("MCP: PNG encoding failed for a {}x{} image", image.width, image.height);
        return {};
    }

    return Base64(png);
}

}  // namespace quantiloom::mcp
