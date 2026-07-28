#include "renderer/RenderCore.hpp"

#include "core/Log.hpp"
#include "io/GltfLoader.hpp"
#include "io/ImageIO.hpp"
#include "io/UsdLoader.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/VulkanContext.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace quantiloom::rendercore {

Result<Scene, String> LoadSceneFromConfig(const Config& config) {
    if (config.Has("scene.usd")) {
        const auto usdPath = config.Get<String>("scene.usd");
        QL_LOG_INFO("Loading USD scene: {}", usdPath);

        auto result = UsdLoader::LoadFromFile(usdPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load USD: " + result.error());
        }
        return Result<Scene, String>(std::move(result.value()));
    }

    if (config.Has("scene.gltf")) {
        const auto gltfPath = config.Get<String>("scene.gltf");
        QL_LOG_INFO("Loading glTF model: {}", gltfPath);

        auto result = GltfLoader::LoadFromFile(gltfPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load glTF: " + result.error());
        }
        return Result<Scene, String>(std::move(result.value()));
    }

    return Result<Scene, String>::Err(
        "No scene.usd or scene.gltf in config -- nothing to render");
}

namespace {

// Direction through the centre of a cubemap texel. Face order is Vulkan's:
// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
glm::vec3 CubemapFaceDirection(u32 face, f32 u, f32 v) {
    const f32 uc = 2.0f * u - 1.0f;
    const f32 vc = 2.0f * v - 1.0f;

    switch (face) {
        case 0:  return glm::normalize(glm::vec3( 1.0f,   -vc,   -uc));
        case 1:  return glm::normalize(glm::vec3(-1.0f,   -vc,    uc));
        case 2:  return glm::normalize(glm::vec3(   uc,  1.0f,    vc));
        case 3:  return glm::normalize(glm::vec3(   uc, -1.0f,   -vc));
        case 4:  return glm::normalize(glm::vec3(   uc,   -vc,  1.0f));
        case 5:  return glm::normalize(glm::vec3(  -uc,   -vc, -1.0f));
        default: return glm::vec3(0.0f);
    }
}

// Bilinear lookup into the equirectangular source.
//
// v = acos(y)/pi, so +Y lands on row 0. That is the zenith: OpenEXR stores scanlines
// top-down and ImageIO::ReadEXR does not flip them, which a 4K noon-sky HDRI confirms
// -- its top quarter is ~47x brighter than its bottom quarter.
glm::vec3 SampleEquirect(const Image& equirect, const glm::vec3& dir) {
    const f32 theta = std::acos(glm::clamp(dir.y, -1.0f, 1.0f));  // [0, pi] from +Y
    const f32 phi   = std::atan2(dir.z, dir.x);                   // [-pi, pi]

    const f32 u = (phi + glm::pi<f32>()) / (2.0f * glm::pi<f32>());
    const f32 v = theta / glm::pi<f32>();

    const u32 width  = equirect.width;
    const u32 height = equirect.height;

    const f32 fx = u * static_cast<f32>(width - 1);
    const f32 fy = v * static_cast<f32>(height - 1);

    const u32 x0 = static_cast<u32>(fx) % width;
    const u32 y0 = static_cast<u32>(fy) % height;
    const u32 x1 = (x0 + 1) % width;
    const u32 y1 = std::min(y0 + 1, height - 1);

    const f32 wx = fx - std::floor(fx);
    const f32 wy = fy - std::floor(fy);

    const glm::vec3 c00(equirect(x0, y0, 0), equirect(x0, y0, 1), equirect(x0, y0, 2));
    const glm::vec3 c10(equirect(x1, y0, 0), equirect(x1, y0, 1), equirect(x1, y0, 2));
    const glm::vec3 c01(equirect(x0, y1, 0), equirect(x0, y1, 1), equirect(x0, y1, 2));
    const glm::vec3 c11(equirect(x1, y1, 0), equirect(x1, y1, 1), equirect(x1, y1, 2));

    const glm::vec3 c0 = c00 * (1.0f - wx) + c10 * wx;
    const glm::vec3 c1 = c01 * (1.0f - wx) + c11 * wx;

    return c0 * (1.0f - wy) + c1 * wy;
}

}  // namespace

Vector<Image> EquirectToCubemap(const Image& equirect, const u32 faceSize) {
    QL_LOG_INFO("  Converting equirectangular to cubemap ({}x{} per face)...",
                faceSize, faceSize);

    Vector<Image> faces(6);
    for (u32 face = 0; face < 6; ++face) {
        faces[face] = Image(faceSize, faceSize, 3);

        for (u32 y = 0; y < faceSize; ++y) {
            for (u32 x = 0; x < faceSize; ++x) {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(faceSize);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(faceSize);

                const glm::vec3 color =
                    SampleEquirect(equirect, CubemapFaceDirection(face, u, v));

                faces[face](x, y, 0) = color.r;
                faces[face](x, y, 1) = color.g;
                faces[face](x, y, 2) = color.b;
            }
        }
    }

    return faces;
}

// ============================================================================
// BrdfLut
// ============================================================================

BrdfLut::~BrdfLut() {
    if (m_sampler != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sampler, nullptr);
    }
}

BrdfLut::BrdfLut(BrdfLut&& other) noexcept
    : m_device(other.m_device),
      m_image(std::move(other.m_image)),
      m_sampler(other.m_sampler),
      m_resolution(other.m_resolution) {
    other.m_sampler = VK_NULL_HANDLE;
    other.m_device = VK_NULL_HANDLE;
    other.m_resolution = 0;
}

BrdfLut& BrdfLut::operator=(BrdfLut&& other) noexcept {
    if (this != &other) {
        if (m_sampler != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_sampler, nullptr);
        }
        m_device = other.m_device;
        m_image = std::move(other.m_image);
        m_sampler = other.m_sampler;
        m_resolution = other.m_resolution;
        other.m_sampler = VK_NULL_HANDLE;
        other.m_device = VK_NULL_HANDLE;
        other.m_resolution = 0;
    }
    return *this;
}

VkImageView BrdfLut::View() const {
    return m_image ? m_image->GetView() : VK_NULL_HANDLE;
}

BrdfLut BrdfLut::Create(VulkanContext& ctx,
                        const String& cachePath,
                        const BRDFLutGenerator::Config& config) {
    QL_LOG_INFO("Creating BRDF LUT...");

    BRDFLutGenerator::Config effective = config;
    Image lut;
    if (auto cached = BRDFLutGenerator::LoadFromBinary(cachePath, &effective)) {
        QL_LOG_INFO("  Loaded from cache: {}", cachePath);
        lut = std::move(cached.value());
    } else {
        QL_LOG_INFO("  Generating {}x{} at {} samples (seconds, then cached)...",
                    effective.resolution, effective.resolution, effective.sampleCount);
        lut = BRDFLutGenerator::Generate(effective);
        if (BRDFLutGenerator::SaveToBinary(cachePath, lut, effective)) {
            QL_LOG_INFO("  Cached to {} for future runs", cachePath);
        }
    }

    const u32 size = effective.resolution;

    BrdfLut result;
    result.m_device = ctx.GetDevice();
    result.m_resolution = size;
    result.m_image = std::make_unique<GpuImage>(
        ctx.GetAllocator(),
        ctx.GetDevice(),
        size, size,
        VK_FORMAT_R32G32_SFLOAT,  // R = scale, G = bias
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    CommandHelper::TransitionImageLayoutImmediate(
        ctx, result.m_image->GetImage(), result.m_image->GetFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    {
        std::vector<f32> interleaved(static_cast<size_t>(size) * size * 2);
        for (u32 y = 0; y < size; ++y) {
            for (u32 x = 0; x < size; ++x) {
                const size_t idx = (static_cast<size_t>(y) * size + x) * 2;
                interleaved[idx + 0] = lut(x, y, 0);
                interleaved[idx + 1] = lut(x, y, 1);
            }
        }

        const VkDeviceSize bytes = interleaved.size() * sizeof(f32);
        GpuBuffer staging(ctx.GetAllocator(), bytes,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        staging.Upload(interleaved.data(), bytes);

        CommandHelper::ExecuteImmediate(ctx, [&](VkCommandBuffer cmd) {
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {size, size, 1};

            vkCmdCopyBufferToImage(cmd, staging.GetHandle(), result.m_image->GetImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        });
    }

    CommandHelper::TransitionImageLayoutImmediate(
        ctx, result.m_image->GetImage(), result.m_image->GetFormat(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // CLAMP_TO_EDGE on every axis, so the border colour is never sampled -- the two
    // implementations disagreed on it (opaque white against the zero-initialised
    // transparent black) with no observable difference.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // no mip chain
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(ctx.GetDevice(), &samplerInfo, nullptr, &result.m_sampler)
        != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to create BRDF LUT sampler");
        result.m_sampler = VK_NULL_HANDLE;
    }

    return result;
}

// ============================================================================
// EnvironmentCubemap
// ============================================================================

namespace {

// Upload one face of one mip level. Each call stages and submits on its own; the
// cubemap is built once per scene load, so the simplicity is worth more than the
// batching.
void UploadCubemapLevel(VulkanContext& ctx, GpuImage& target, u32 face, u32 mip,
                        u32 size, const std::vector<f32>& rgba) {
    const VkDeviceSize bytes = rgba.size() * sizeof(f32);
    GpuBuffer staging(ctx.GetAllocator(), bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    staging.Upload(rgba.data(), bytes);

    CommandHelper::ExecuteImmediate(ctx, [&](VkCommandBuffer cmd) {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = mip;
        region.imageSubresource.baseArrayLayer = face;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {size, size, 1};

        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), target.GetImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    });
}

// A face of N texels supports floor(log2(N)) + 1 levels. Asking for more is invalid
// Vulkan, and the downsample below would divide by a zero-width previous level --
// the shipped 512 with 8 levels stays clear of both, which is why neither
// implementation this replaced ever tripped over it.
EnvironmentCubemap::Params ClampToFaceSize(EnvironmentCubemap::Params p) {
    u32 maxLevels = 1;
    for (u32 size = p.faceSize; size > 1; size >>= 1) {
        ++maxLevels;
    }
    if (p.mipLevels > maxLevels) {
        QL_LOG_WARN("Environment cubemap: {} mip levels requested for a {}x{} face, "
                    "clamping to {}", p.mipLevels, p.faceSize, p.faceSize, maxLevels);
        p.mipLevels = maxLevels;
    }
    if (p.mipLevels == 0) {
        p.mipLevels = 1;
    }
    return p;
}

std::unique_ptr<GpuImage> CreateCubemapImage(VulkanContext& ctx,
                                             const EnvironmentCubemap::Params& p) {
    auto image = std::make_unique<GpuImage>(
        ctx.GetAllocator(), ctx.GetDevice(),
        p.faceSize, p.faceSize,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        p.mipLevels,
        6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        VK_IMAGE_VIEW_TYPE_CUBE);

    CommandHelper::TransitionImageLayoutImmediate(
        ctx, image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        p.mipLevels, 6);

    return image;
}

// Box-filter one mip level out of the base face.
//
// Sampling the base rather than the previous level is an approximation both
// implementations made, and it is kept: the whole chain is a placeholder for GGX
// prefiltering. What is not kept is how the CLI addressed it -- it indexed the base
// face's buffer with the *previous mip's* width as the row stride, so from mip 2
// onwards it read a skewed slice of the base face's top-left corner rather than the
// whole face. Stepping in face coordinates is what the context did, and it is right.
std::vector<f32> DownsampleFace(const Image& baseFace, u32 baseSize, u32 mip) {
    u32 mipSize = baseSize >> mip;
    if (mipSize == 0) mipSize = 1;
    const u32 prevMipSize = baseSize >> (mip - 1);
    const u32 stride = baseSize / prevMipSize;

    std::vector<f32> out(static_cast<size_t>(mipSize) * mipSize * 4);
    for (u32 y = 0; y < mipSize; ++y) {
        for (u32 x = 0; x < mipSize; ++x) {
            glm::vec3 sum(0.0f);
            u32 count = 0;
            for (u32 dy = 0; dy < 2 && (y * 2 + dy) < prevMipSize; ++dy) {
                for (u32 dx = 0; dx < 2 && (x * 2 + dx) < prevMipSize; ++dx) {
                    const u32 sx = std::min((x * 2 + dx) * stride, baseSize - 1);
                    const u32 sy = std::min((y * 2 + dy) * stride, baseSize - 1);
                    sum += glm::vec3(baseFace(sx, sy, 0), baseFace(sx, sy, 1),
                                     baseFace(sx, sy, 2));
                    ++count;
                }
            }
            if (count > 0) {
                sum /= static_cast<f32>(count);
            }

            const size_t idx = (static_cast<size_t>(y) * mipSize + x) * 4;
            out[idx + 0] = sum.r;
            out[idx + 1] = sum.g;
            out[idx + 2] = sum.b;
            out[idx + 3] = 1.0f;
        }
    }
    return out;
}

std::vector<f32> FaceToRgba(const Image& face, u32 size) {
    std::vector<f32> out(static_cast<size_t>(size) * size * 4);
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
            out[idx + 0] = face(x, y, 0);
            out[idx + 1] = face(x, y, 1);
            out[idx + 2] = face(x, y, 2);
            out[idx + 3] = 1.0f;
        }
    }
    return out;
}

}  // namespace

EnvironmentCubemap::~EnvironmentCubemap() = default;
EnvironmentCubemap::EnvironmentCubemap(EnvironmentCubemap&&) noexcept = default;
EnvironmentCubemap& EnvironmentCubemap::operator=(EnvironmentCubemap&&) noexcept = default;

VkImageView EnvironmentCubemap::View() const {
    return m_image ? m_image->GetView() : VK_NULL_HANDLE;
}

Result<EnvironmentCubemap, String> EnvironmentCubemap::Load(VulkanContext& ctx,
                                                            const String& path,
                                                            const Params& requested) {
    QL_LOG_INFO("Loading environment map: {}", path);
    const Params params = ClampToFaceSize(requested);

    if (!ImageIO::FileExists(path)) {
        return Result<EnvironmentCubemap, String>::Err(
            "Environment map file not found: " + path);
    }

    auto equirect = ImageIO::ReadImage(path);
    if (!equirect.has_value()) {
        return Result<EnvironmentCubemap, String>::Err(
            "Failed to load environment map: " + path);
    }
    QL_LOG_INFO("  HDR image loaded: {}x{}, {} channels",
                equirect->width, equirect->height, equirect->channels);

    const Vector<Image> faces = EquirectToCubemap(equirect.value(), params.faceSize);

    EnvironmentCubemap result;
    result.m_faceSize = params.faceSize;
    result.m_mipLevels = params.mipLevels;
    result.m_image = CreateCubemapImage(ctx, params);

    QL_LOG_INFO("  Uploading cubemap to GPU...");
    for (u32 face = 0; face < 6; ++face) {
        UploadCubemapLevel(ctx, *result.m_image, face, 0, params.faceSize,
                           FaceToRgba(faces[face], params.faceSize));
    }

    QL_LOG_INFO("  Generating mipmap chain...");
    for (u32 mip = 1; mip < params.mipLevels; ++mip) {
        u32 mipSize = params.faceSize >> mip;
        if (mipSize == 0) mipSize = 1;
        for (u32 face = 0; face < 6; ++face) {
            UploadCubemapLevel(ctx, *result.m_image, face, mip, mipSize,
                               DownsampleFace(faces[face], params.faceSize, mip));
        }
    }

    CommandHelper::TransitionImageLayoutImmediate(
        ctx, result.m_image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        params.mipLevels, 6);

    QL_LOG_INFO("Environment map loaded successfully: {}", path);
    return Result<EnvironmentCubemap, String>(std::move(result));
}

EnvironmentCubemap EnvironmentCubemap::Fallback(VulkanContext& ctx, const Params& requested) {
    const Params params = ClampToFaceSize(requested);
    QL_LOG_INFO("Creating fallback sky-blue environment map ({}x{} per face)...",
                params.faceSize, params.faceSize);

    constexpr f32 kSky[4] = {0.5f, 0.7f, 1.0f, 1.0f};

    EnvironmentCubemap result;
    result.m_faceSize = params.faceSize;
    result.m_mipLevels = params.mipLevels;
    result.m_image = CreateCubemapImage(ctx, params);

    // Uniform colour, so every level is filled directly rather than downsampled.
    for (u32 mip = 0; mip < params.mipLevels; ++mip) {
        u32 mipSize = params.faceSize >> mip;
        if (mipSize == 0) mipSize = 1;

        std::vector<f32> level(static_cast<size_t>(mipSize) * mipSize * 4);
        for (size_t i = 0; i < static_cast<size_t>(mipSize) * mipSize; ++i) {
            level[i * 4 + 0] = kSky[0];
            level[i * 4 + 1] = kSky[1];
            level[i * 4 + 2] = kSky[2];
            level[i * 4 + 3] = kSky[3];
        }

        for (u32 face = 0; face < 6; ++face) {
            UploadCubemapLevel(ctx, *result.m_image, face, mip, mipSize, level);
        }
    }

    CommandHelper::TransitionImageLayoutImmediate(
        ctx, result.m_image->GetImage(), VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        params.mipLevels, 6);

    return result;
}

}  // namespace quantiloom::rendercore
