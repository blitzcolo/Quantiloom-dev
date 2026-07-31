#include "renderer/RenderCore.hpp"

#include "core/Log.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "io/GltfLoader.hpp"
#include "io/ImageIO.hpp"
#include "io/UsdLoader.hpp"
#include "renderer/CommandHelper.hpp"
#include "renderer/GpuBuffer.hpp"
#include "renderer/GpuImage.hpp"
#include "renderer/RayTracingPipeline.hpp"
#include "renderer/TextureManager.hpp"
#include "renderer/VulkanContext.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>

namespace quantiloom::rendercore {

Result<Scene, String> LoadSceneFromConfig(const Config& config, const String& baseDir) {
    if (config.Has("scene.usd")) {
        const auto usdPath = ResolveConfigPath(config.Get<String>("scene.usd"), baseDir);
        QL_LOG_INFO("Loading USD scene: {}", usdPath);

        auto result = UsdLoader::LoadFromFile(usdPath);
        if (!result.has_value()) {
            return Result<Scene, String>::Err("Failed to load USD: " + result.error());
        }
        return Result<Scene, String>(std::move(result.value()));
    }

    if (config.Has("scene.gltf")) {
        const auto gltfPath = ResolveConfigPath(config.Get<String>("scene.gltf"), baseDir);
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

// Trilinear across the chain and unclamped, so the shader's roughness-derived lod
// picks a level instead of being pinned to mip 0 by a maxLod of zero.
VkSampler CreateEnvironmentSampler(VkDevice device) {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.anisotropyEnable = VK_FALSE;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0f;
    info.maxLod = VK_LOD_CLAMP_NONE;
    info.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler = VK_NULL_HANDLE;
    if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
        QL_LOG_ERROR("Failed to create environment map sampler");
        return VK_NULL_HANDLE;
    }
    return sampler;
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

EnvironmentCubemap::~EnvironmentCubemap() {
    if (m_sampler != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sampler, nullptr);
    }
}

EnvironmentCubemap::EnvironmentCubemap(EnvironmentCubemap&& other) noexcept
    : m_device(other.m_device),
      m_image(std::move(other.m_image)),
      m_sampler(other.m_sampler),
      m_faceSize(other.m_faceSize),
      m_mipLevels(other.m_mipLevels) {
    other.m_device = VK_NULL_HANDLE;
    other.m_sampler = VK_NULL_HANDLE;
    other.m_faceSize = 0;
    other.m_mipLevels = 0;
}

EnvironmentCubemap& EnvironmentCubemap::operator=(EnvironmentCubemap&& other) noexcept {
    if (this != &other) {
        if (m_sampler != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_sampler, nullptr);
        }
        m_device = other.m_device;
        m_image = std::move(other.m_image);
        m_sampler = other.m_sampler;
        m_faceSize = other.m_faceSize;
        m_mipLevels = other.m_mipLevels;
        other.m_device = VK_NULL_HANDLE;
        other.m_sampler = VK_NULL_HANDLE;
        other.m_faceSize = 0;
        other.m_mipLevels = 0;
    }
    return *this;
}

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
    result.m_device = ctx.GetDevice();
    result.m_faceSize = params.faceSize;
    result.m_mipLevels = params.mipLevels;
    result.m_image = CreateCubemapImage(ctx, params);
    result.m_sampler = CreateEnvironmentSampler(ctx.GetDevice());

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
    result.m_device = ctx.GetDevice();
    result.m_faceSize = params.faceSize;
    result.m_mipLevels = params.mipLevels;
    result.m_image = CreateCubemapImage(ctx, params);
    result.m_sampler = CreateEnvironmentSampler(ctx.GetDevice());

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

// ============================================================================
// SceneGeometry
// ============================================================================

namespace {

// Spot-check the merged buffers before they reach the GPU. Only the first handful of
// triangles, and only at debug level: a wrong offset shows up as skewed geometric
// normals here rather than as a corrupt image several stages later.
void LogMergeValidation(const Vector<glm::vec3>& vertices,
                        const Vector<u32>& indices,
                        const Vector<glm::vec3>& normals) {
    QL_LOG_DEBUG("=== Buffer Merge Validation ===");
    QL_LOG_DEBUG("  Total vertices: {}, Total indices: {}", vertices.size(), indices.size());

    const size_t triangles = indices.size() / 3;
    QL_LOG_DEBUG("  Total triangles: {}", triangles);

    for (size_t tri = 0; tri < std::min(triangles, size_t{12}); ++tri) {
        const u32 i0 = indices[tri * 3 + 0];
        const u32 i1 = indices[tri * 3 + 1];
        const u32 i2 = indices[tri * 3 + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
            QL_LOG_ERROR("  Triangle {}: INVALID INDICES [{}, {}, {}] (max={})",
                         tri, i0, i1, i2, vertices.size() - 1);
            continue;
        }

        const glm::vec3 n = glm::normalize(
            glm::cross(vertices[i1] - vertices[i0], vertices[i2] - vertices[i0]));
        const auto onAxis = [](f32 a, f32 b, f32 c) {
            return std::abs(std::abs(a) - 1.0f) < 0.01f && std::abs(b) < 0.01f &&
                   std::abs(c) < 0.01f;
        };
        const bool axisAligned =
            onAxis(n.x, n.y, n.z) || onAxis(n.y, n.x, n.z) || onAxis(n.z, n.x, n.y);

        QL_LOG_DEBUG("  Triangle {}: indices=[{}, {}, {}], geoNormal=({:.3f}, {:.3f}, {:.3f}) {}",
                     tri, i0, i1, i2, n.x, n.y, n.z, axisAligned ? "FINE" : "SKEWED");
    }

    if (!normals.empty()) {
        QL_LOG_DEBUG("  --- Stored vertex normals (first 8) ---");
        for (size_t i = 0; i < std::min(size_t{8}, normals.size()); ++i) {
            QL_LOG_DEBUG("    normal[{}] = ({:.3f}, {:.3f}, {:.3f})",
                         i, normals[i].x, normals[i].y, normals[i].z);
        }
    }
    QL_LOG_DEBUG("=== End Buffer Merge Validation ===");
}

// Walk the scene in the order the instance list is indexed: nodes outermost, then
// the primitives of the mesh each node references. Build and RebuildTlas must agree
// on it, because InstanceIndex() in the shader is a position in this walk.
template <class F>
void ForEachInstance(const Scene& scene, F&& fn) {
    Vector<size_t> meshToFirstPrim;
    meshToFirstPrim.reserve(scene.meshes.size());
    size_t firstPrim = 0;
    for (const auto& mesh : scene.meshes) {
        meshToFirstPrim.push_back(firstPrim);
        firstPrim += mesh.primitives.size();
    }

    for (size_t nodeIndex = 0; nodeIndex < scene.nodes.size(); ++nodeIndex) {
        const auto& node = scene.nodes[nodeIndex];
        const Mesh& mesh = scene.meshes[node.meshIndex];
        const size_t base = meshToFirstPrim[node.meshIndex];
        for (size_t prim = 0; prim < mesh.primitives.size(); ++prim) {
            fn(node.transform, base + prim, nodeIndex);
        }
    }
}

// No material is not a reason to cull: a primitive whose materialId is out of range
// renders double-sided rather than disappearing from one side.
bool IsDoubleSided(const Scene& scene, u32 materialId) {
    return materialId < scene.materials.size() ? scene.materials[materialId].doubleSided
                                               : true;
}

std::unique_ptr<GpuBuffer> UploadGeometryBuffer(VmaAllocator allocator,
                                                const void* data, size_t bytes) {
    auto buffer = std::make_unique<GpuBuffer>(
        allocator, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    buffer->Upload(data, bytes);
    return buffer;
}

}  // namespace

SceneGeometry::~SceneGeometry() = default;
SceneGeometry::SceneGeometry(SceneGeometry&&) noexcept = default;
SceneGeometry& SceneGeometry::operator=(SceneGeometry&&) noexcept = default;

SceneGeometry SceneGeometry::Build(VulkanContext& ctx, const Scene& scene) {
    QL_LOG_INFO("Building acceleration structures...");

    SceneGeometry result;

    // Pass one: where each primitive's slice of every merged buffer begins. Attributes
    // the asset omitted still get a slice, sized by the vertex count, so an instance's
    // offset is valid in every buffer.
    Vector<InstanceGeometryInfo> primitiveOffsets;
    u32 nVerts = 0, nIndices = 0, nNormals = 0, nUVs = 0, nTangents = 0;
    for (const auto& mesh : scene.meshes) {
        for (const auto& prim : mesh.primitives) {
            InstanceGeometryInfo offset{};
            offset.vertexOffset = nVerts;
            offset.indexOffset = nIndices;
            offset.normalOffset = nNormals;
            offset.uvOffset = nUVs;
            offset.tangentOffset = nTangents;
            offset.materialId = prim.materialId;
            primitiveOffsets.push_back(offset);

            const auto vcount = static_cast<u32>(prim.positions.size());
            nVerts += vcount;
            nIndices += static_cast<u32>(prim.indices.size());
            nNormals += prim.normals.empty() ? vcount : static_cast<u32>(prim.normals.size());
            nUVs += prim.uvs.empty() ? vcount : static_cast<u32>(prim.uvs.size());
            nTangents += prim.tangents.empty() ? vcount : static_cast<u32>(prim.tangents.size());
        }
    }

    if (primitiveOffsets.empty()) {
        QL_LOG_WARN("  Scene has no primitives, nothing to build");
        return result;
    }

    QL_LOG_INFO("  Merged geometry: {} vertices, {} indices, {} normals, {} UVs, {} tangents",
                nVerts, nIndices, nNormals, nUVs, nTangents);

    // Pass two: fill. Defaults are the initial values, so only present attributes copy.
    Vector<glm::vec3> vertices(nVerts);
    Vector<u32> indices(nIndices);
    Vector<glm::vec3> normals(nNormals, glm::vec3(0.0f, 1.0f, 0.0f));
    Vector<glm::vec2> uvs(nUVs, glm::vec2(0.0f));
    Vector<glm::vec4> tangents(nTangents, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

    size_t primIdx = 0;
    for (const auto& mesh : scene.meshes) {
        for (const auto& prim : mesh.primitives) {
            const InstanceGeometryInfo& off = primitiveOffsets[primIdx++];

            std::copy(prim.positions.begin(), prim.positions.end(),
                      vertices.begin() + off.vertexOffset);
            std::copy(prim.indices.begin(), prim.indices.end(),
                      indices.begin() + off.indexOffset);

            if (!prim.normals.empty()) {
                std::copy(prim.normals.begin(), prim.normals.end(),
                          normals.begin() + off.normalOffset);
            } else {
                QL_LOG_WARN("  Primitive has no normals after loading, using +Y fallback");
            }
            if (!prim.uvs.empty()) {
                std::copy(prim.uvs.begin(), prim.uvs.end(), uvs.begin() + off.uvOffset);
            }
            if (!prim.tangents.empty()) {
                std::copy(prim.tangents.begin(), prim.tangents.end(),
                          tangents.begin() + off.tangentOffset);
            }
        }
    }

    LogMergeValidation(vertices, indices, normals);

    VmaAllocator allocator = ctx.GetAllocator();
    result.m_vertices = UploadGeometryBuffer(allocator, vertices.data(),
                                             vertices.size() * sizeof(glm::vec3));
    result.m_indices = UploadGeometryBuffer(allocator, indices.data(),
                                            indices.size() * sizeof(u32));
    result.m_normals = UploadGeometryBuffer(allocator, normals.data(),
                                            normals.size() * sizeof(glm::vec3));
    result.m_uvs = UploadGeometryBuffer(allocator, uvs.data(),
                                        uvs.size() * sizeof(glm::vec2));
    result.m_tangents = UploadGeometryBuffer(allocator, tangents.data(),
                                             tangents.size() * sizeof(glm::vec4));
    result.m_vertexCount = nVerts;
    result.m_indexCount = nIndices;
    QL_LOG_INFO("  Created merged geometry buffers");

    // One BLAS per primitive, shared by every node that instances the mesh.
    for (const auto& mesh : scene.meshes) {
        for (const auto& prim : mesh.primitives) {
            result.m_blas.emplace_back(std::make_unique<BLAS>(ctx, prim));
        }
    }

    result.m_tlas = std::make_unique<TLAS>(ctx);

    CommandHelper::ExecuteImmediate(ctx, [&](VkCommandBuffer cmd) {
        for (auto& blas : result.m_blas) {
            blas->Build(cmd);
        }

        ForEachInstance(scene, [&](const glm::mat4& transform, size_t globalPrim,
                                   size_t nodeIndex) {
            const InstanceGeometryInfo& info = primitiveOffsets[globalPrim];
            result.m_instances.push_back(info);
            result.m_instanceToNode.push_back(static_cast<u32>(nodeIndex));

            QL_LOG_DEBUG("  Instance {}: vertexOff={}, indexOff={}, normalOff={}, "
                         "uvOff={}, tangentOff={}, matId={}",
                         result.m_instances.size() - 1, info.vertexOffset,
                         info.indexOffset, info.normalOffset, info.uvOffset,
                         info.tangentOffset, info.materialId);

            // The second argument lands in instanceCustomIndex, which no shader reads
            // -- the material reaches them through InstanceInfo(), indexed by
            // InstanceIndex(). The two implementations this replaced passed different
            // things here for exactly that reason.
            result.m_tlas->AddInstance(*result.m_blas[globalPrim], info.materialId,
                                       transform, IsDoubleSided(scene, info.materialId));
        });

        result.m_tlas->Build(cmd);
    });

    result.m_instanceCount = static_cast<u32>(result.m_instances.size());

    if (!result.m_instances.empty()) {
        result.m_instanceInfo = std::make_unique<GpuBuffer>(
            allocator, result.m_instances.size() * sizeof(InstanceGeometryInfo),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        result.m_instanceInfo->Upload(result.m_instances.data(),
                                      result.m_instances.size() * sizeof(InstanceGeometryInfo));
        QL_LOG_INFO("  Created instance geometry info buffer ({} instances)",
                    result.m_instanceCount);
    }

    QL_LOG_INFO("  Built {} BLAS, 1 TLAS, {} instance(s)",
                result.m_blas.size(), result.m_instanceCount);
    return result;
}

void SceneGeometry::RebuildTlas(VulkanContext& ctx, const Scene& scene) {
    if (m_blas.empty()) {
        QL_LOG_WARN("RebuildTlas: nothing built yet");
        return;
    }

    QL_LOG_DEBUG("Rebuilding TLAS with updated transforms...");
    m_tlas = std::make_unique<TLAS>(ctx);

    CommandHelper::ExecuteImmediate(ctx, [&](VkCommandBuffer cmd) {
        size_t instance = 0;
        ForEachInstance(scene, [&](const glm::mat4& transform, size_t globalPrim,
                                   size_t /*nodeIndex*/) {
            // Offsets are a property of the geometry, not of where a node sits, so
            // the instance list from Build still applies.
            const u32 materialId = instance < m_instances.size()
                                       ? m_instances[instance].materialId
                                       : 0;
            ++instance;
            m_tlas->AddInstance(*m_blas[globalPrim], materialId, transform,
                                IsDoubleSided(scene, materialId));
        });
        m_tlas->Build(cmd);
    });
}

// ============================================================================
// Materials
// ============================================================================

MaterialDataCPU ConvertMaterial(const Material& material, const f32 wavelengthNm,
                                const MaterialGpuIndices& indices) {
    MaterialDataCPU cpuMat{};
    cpuMat.baseColorFactor = material.baseColorFactor;
    cpuMat.baseColorTextureIndex = material.baseColorTextureIndex;
    cpuMat.metallicFactor = material.metallicFactor;
    cpuMat.roughnessFactor = material.roughnessFactor;
    cpuMat.metallicRoughnessTextureIndex = material.metallicRoughnessTextureIndex;
    cpuMat.normalTextureIndex = material.normalTextureIndex;
    cpuMat.normalScale = material.normalScale;
    cpuMat.doubleSided = material.doubleSided ? 1u : 0u;
    cpuMat.emissiveFactor = material.emissiveFactor;
    cpuMat.emissiveTextureIndex = material.emissiveTextureIndex;
    cpuMat.alphaMode = static_cast<u32>(material.alphaMode);
    cpuMat.alphaCutoff = material.alphaCutoff;
    cpuMat.spectralAlbedo = material.spectralAlbedo;
    cpuMat.spectralReflectanceCurveIndex = indices.spectralReflectanceCurve;

    // Sentinel: -1.0f when no IR data -> shader derives epsilon from metallic/roughness
    cpuMat.irEmissivity = material.irEmissivityCurve.empty()
        ? -1.0f
        : std::clamp(material.GetIREmissivity(wavelengthNm), 0.0f, 1.0f);
    cpuMat.irTransmittance = std::clamp(material.GetIRTransmittance(wavelengthNm), 0.0f, 1.0f);
    cpuMat.irTemperature_K = material.irTemperature_K;
    cpuMat.complexRefractiveIndexIndex = indices.complexRefractiveIndex;

    // Temperature texture fields (per-pixel temperature map)
    cpuMat.temperatureTextureIndex = material.temperatureTextureIndex;
    cpuMat.temperatureScale = material.temperatureScale;
    cpuMat.temperatureOffset = material.temperatureOffset;
    cpuMat.irEmissivityCurveIndex = -1;    // no per-wavelength curve buffer yet

    // Transmission properties (KHR_materials_transmission + KHR_materials_volume)
    cpuMat.ior = material.ior;
    cpuMat.transmission = material.transmission;
    cpuMat.transmissionTextureIndex = material.transmissionTextureIndex;
    cpuMat.irTransmittanceCurveIndex = -1; // no per-wavelength curve buffer yet
    cpuMat.attenuationColor = material.attenuationColor;
    cpuMat.attenuationDistance = material.attenuationDistance;
    cpuMat.thicknessFactor = material.thicknessFactor;
    cpuMat.thicknessTextureIndex = material.thicknessTextureIndex;
    cpuMat.dispersion = material.dispersion;
    cpuMat._padding2 = 0.0f;

    // Volume properties (fog, smoke, SSS)
    cpuMat.volumeDensity = material.volumeDensity;
    cpuMat.scatteringCoeff = material.scatteringCoeff;
    cpuMat.absorptionCoeff = material.absorptionCoeff;
    cpuMat.phaseG = material.phaseG;

    return cpuMat;
}

std::unique_ptr<GpuBuffer> BuildMaterialBuffer(VulkanContext& ctx, const Scene& scene,
                                               const f32 wavelengthNm,
                                               const Vector<MaterialGpuIndices>& indices) {
    if (scene.materials.empty()) {
        return nullptr;
    }

    Vector<MaterialDataCPU> gpuMaterials;
    gpuMaterials.reserve(scene.materials.size());

    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const Material& material = scene.materials[i];
        const MaterialGpuIndices resolved =
            i < indices.size()
                ? indices[i]
                : MaterialGpuIndices{material.spectralReflectanceCurveIndex,
                                     material.complexRefractiveIndexIndex};
        gpuMaterials.push_back(ConvertMaterial(material, wavelengthNm, resolved));
    }

    const size_t bytes = gpuMaterials.size() * sizeof(MaterialDataCPU);
    auto buffer = std::make_unique<GpuBuffer>(ctx.GetAllocator(), bytes,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              VMA_MEMORY_USAGE_CPU_TO_GPU);
    buffer->Upload(gpuMaterials.data(), bytes);
    return buffer;
}

// ============================================================================
// Render target
// ============================================================================

std::unique_ptr<GpuImage> CreateRenderTarget(VulkanContext& ctx, const u32 width,
                                             const u32 height, const VkFormat format) {
    QL_LOG_INFO("Creating render target ({}x{}, format {})...", width, height,
                static_cast<int>(format));

    auto image = std::make_unique<GpuImage>(
        ctx.GetAllocator(), ctx.GetDevice(),
        width, height,
        format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // GENERAL because the ray generation shader writes it as a storage image; it is
    // never sampled, only blitted or copied out.
    CommandHelper::TransitionImageLayoutImmediate(
        ctx, image->GetImage(), image->GetFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    return image;
}

// ============================================================================
// Ray tracing pipeline
// ============================================================================

std::unique_ptr<RayTracingPipeline> CreateRayTracingPipeline(
    VulkanContext& ctx, VkPipelineCache cache, const PipelineBindings& bindings) {
    QL_LOG_INFO("Creating ray tracing pipeline...");

    auto pipeline = std::make_unique<RayTracingPipeline>(
        ctx, "raygen.spv", "closesthit.spv", "miss.spv", cache);

    if (bindings.outputImage) {
        pipeline->BindOutputImage(*bindings.outputImage);            // 0
    }
    if (bindings.depthImage) {
        pipeline->BindDepthImage(*bindings.depthImage);              // 22
    }
    if (bindings.geometry && bindings.geometry->IsValid()) {
        const SceneGeometry& geometry = *bindings.geometry;
        pipeline->BindAccelerationStructure(geometry.Tlas().GetHandle());  // 1
        pipeline->BindGeometryBuffers(geometry.Vertices(), geometry.Indices(),
                                      &geometry.UVs());              // 3, 4, 8
        pipeline->BindTangentBuffer(geometry.Tangents());            // 9
        pipeline->BindNormalBuffer(geometry.Normals());              // 16
        if (geometry.InstanceCount() > 0) {
            pipeline->BindInstanceGeometryBuffer(geometry.InstanceInfo());  // 18
        }
    }
    if (bindings.lightingParams) {
        pipeline->BindLUTBuffer(*bindings.lightingParams);           // 2
    }
    if (bindings.materials) {
        pipeline->BindMaterialBuffer(*bindings.materials);           // 5
    }
    if (bindings.textures) {
        pipeline->BindTextures(bindings.textures->GetImageViews(),
                               bindings.textures->GetSamplers());    // 6, 7
    }
    if (bindings.environment) {
        pipeline->BindPrefilteredEnvMap(bindings.environment->View(),
                                        bindings.environment->Sampler());  // 10, 21
    }
    if (bindings.brdfLut) {
        pipeline->BindBRDFLut(bindings.brdfLut->View(),
                              bindings.brdfLut->Sampler());          // 11, 12
    }

    // These binders take a pointer and handle null themselves.
    pipeline->BindSpectralCurvesBuffer(bindings.spectralCurves);     // 13
    pipeline->BindComplexRefractiveIndexBuffer(bindings.complexRefractiveIndex);  // 14
    pipeline->BindSolarSpectralLUT(bindings.solarLut);               // 15
    pipeline->BindAtmosphereNN(bindings.atmosphereHeader,
                               bindings.atmosphereData);             // 17, 20

    if (bindings.cieColourMatching) {
        pipeline->BindCIE_CMF_LUT(*bindings.cieColourMatching);      // 19
    }

    QL_LOG_INFO("  Ray tracing pipeline created and bound");
    return pipeline;
}

// ============================================================================
// Sensor band conditioning
// ============================================================================

SensorBandAdjustment SensorAdjustmentForMode(const SpectralMode mode,
                                             const bool hostSetWavelength) {
    SensorBandAdjustment adjustment;

    // VIS_Fused is excluded on purpose: it outputs CIE-integrated RGB rather than
    // scalar band radiance, so neither the scale nor the band centre applies.
    if (!IsIRFusedMode(mode)) {
        return adjustment;
    }

    const auto band = GetFusedBandInfo(mode);
    if (!band.has_value()) {
        return adjustment;
    }

    adjustment.radianceScale = band->WidthNm();
    if (!hostSetWavelength) {
        adjustment.wavelengthNm = band->CenterNm();
    }
    return adjustment;
}

// ============================================================================
// CIE colour matching functions
// ============================================================================

std::unique_ptr<GpuBuffer> CreateCieColourMatchingBuffer(VulkanContext& ctx) {
    Vector<glm::vec4> table;
    table.reserve(CIE_CMF_LUT_SIZE);
    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        // w is padding: the shader reads this as a structured buffer of float4.
        table.emplace_back(CIE_1931_2DEG[i][0], CIE_1931_2DEG[i][1],
                           CIE_1931_2DEG[i][2], 0.0f);
    }

    const size_t bytes = table.size() * sizeof(glm::vec4);
    auto buffer = std::make_unique<GpuBuffer>(ctx.GetAllocator(), bytes,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              VMA_MEMORY_USAGE_CPU_TO_GPU);
    buffer->Upload(table.data(), bytes);

    QL_LOG_INFO("  CIE CMF LUT created ({} samples, 380-780 nm)", CIE_CMF_LUT_SIZE);
    return buffer;
}

}  // namespace quantiloom::rendercore
