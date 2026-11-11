#include "TextureManager.hpp"
#include "TextureCompressor.hpp"
#include "core/Log.hpp"
#include "renderer/GpuAlloc.hpp"
#include "renderer/TextureTransitions.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <cmath>

namespace quantiloom {

namespace {

// Every stage that samples the uploaded textures. The material textures are
// read by the ray-tracing stages (and by compute passes sharing the layout); a
// barrier whose destination names only the fragment stage leaves them
// unsynchronised against the transfer that wrote them.
constexpr gpu::Access kSampleAccess =
    gpu::Access::RayTraceRead | gpu::Access::ComputeRead | gpu::Access::FragmentRead;

}  // namespace

struct TextureManager::PendingUpload {
    std::unique_ptr<gpu::Buffer> staging;
    std::unique_ptr<gpu::Texture> image;
    gpu::Format format = gpu::Format::Undefined;
    u32 width = 0;
    u32 height = 0;
    u32 mipLevels = 1;
    u32 bufferRowLength = 0;
    u32 bufferImageHeight = 0;
    bool compressed = false;
};

// ============================================================================
// Construction / Destruction
// ============================================================================

TextureManager::TextureManager(gpu::Device& device, u64 uploadBatchBytes)
    : m_device(device)
    , m_uploadBatchBytes(std::max<u64>(uploadBatchBytes, 1)) {
    // Resources are allocated lazily in UploadTextures
}

// m_samplerObjects is declared after m_images: every sampler is destroyed, then every
// gpu::Texture destroys its image and view (RAII)
TextureManager::~TextureManager() = default;

// ============================================================================
// Texture Upload
// ============================================================================

void TextureManager::UploadTextures(std::vector<Texture>& textures) {
    // Clear previous state
    m_images.clear();
    m_samplerObjects.clear();
    m_samplerRefs.clear();
    m_textureRefs.clear();
    m_lastUploadBatchCount = 0;

    // Handle empty texture list: create dummy 1x1 white texture
    if (textures.empty()) {
        QL_LOG_INFO("No textures to upload, creating dummy 1x1 white texture");
        Texture dummyTex = CreateDummyTexture();

        auto pending = PrepareTexture(dummyTex);
        m_device.ExecuteImmediate([&](gpu::CommandList& list) {
            RecordTextureUpload(list, *pending);
        });
        ++m_lastUploadBatchCount;
        m_images.push_back(std::move(pending->image));
        m_samplerObjects.push_back(CreateSampler(dummyTex.sampler));
        m_samplerRefs.push_back(m_samplerObjects.back()->Ref());
        m_textureRefs.push_back(m_images.back()->Ref());

        return;
    }

    // Upload all textures
    QL_LOG_INFO("Uploading {} textures to GPU", textures.size());

    struct BatchEntry {
        Texture* source = nullptr;
        std::unique_ptr<PendingUpload> upload;
    };
    std::vector<BatchEntry> batch;
    u64 batchBytes = 0;

    const auto flush = [&]() {
        if (batch.empty()) return;
        m_device.ExecuteImmediate([&](gpu::CommandList& list) {
            for (const BatchEntry& entry : batch) {
                RecordTextureUpload(list, *entry.upload);
            }
        });
        ++m_lastUploadBatchCount;

        for (BatchEntry& entry : batch) {
            std::unique_ptr<gpu::Sampler> sampler = CreateSampler(entry.source->sampler);
            m_textureRefs.push_back(entry.upload->image->Ref());
            m_samplerRefs.push_back(sampler->Ref());
            m_samplerObjects.push_back(std::move(sampler));
            m_images.push_back(std::move(entry.upload->image));

            // Release only after the submission that consumed the staging
            // bytes completed. Retained pixels feed interactive re-unmixing.
            if (!entry.source->retainCpuPixels) {
                entry.source->ReleaseCPUMemory();
            }
        }
        batch.clear();
        batchBytes = 0;
    };

    for (Texture& texture : textures) {
        const u64 rawBytes =
            static_cast<u64>(texture.width) * texture.height * 4;
        const u64 estimate = texture.HasBC7Data()
            ? static_cast<u64>(texture.bc7Data->data.size()) : rawBytes;
        if (!batch.empty() && batchBytes + estimate > m_uploadBatchBytes) {
            flush();
        }

        auto pending = PrepareTexture(texture);
        const u64 uploadBytes = pending->staging->Size();
        if (!batch.empty() && batchBytes + uploadBytes > m_uploadBatchBytes) {
            flush();
        }
        batchBytes += uploadBytes;
        batch.push_back({&texture, std::move(pending)});

        // A texture at or above the cap owns one batch. The cap bounds staging
        // memory for ordinary textures; a single larger texture is unavoidable.
        if (batchBytes >= m_uploadBatchBytes) {
            flush();
        }
    }
    flush();

    QL_LOG_INFO("  Texture upload complete: {} textures, {} samplers, {} batch(es) "
                "(CPU memory released)",
                m_images.size(), m_samplerRefs.size(), m_lastUploadBatchCount);
}

i32 TextureManager::AppendTexture(const Texture& texture) {
    if (!texture.IsValid()) {
        QL_LOG_WARN("AppendTexture: invalid texture '{}'", texture.name);
        return -1;
    }

    auto gpuImage = UploadTexture(texture);
    if (!gpuImage) {
        return -1;
    }
    std::unique_ptr<gpu::Sampler> sampler = CreateSampler(texture.sampler);

    const i32 index = static_cast<i32>(m_images.size());
    m_textureRefs.push_back(gpuImage->Ref());
    m_samplerRefs.push_back(sampler->Ref());
    m_samplerObjects.push_back(std::move(sampler));
    m_images.push_back(std::move(gpuImage));

    QL_LOG_INFO("  Appended texture '{}' at index {} ({} total)", texture.name, index,
                m_images.size());
    return index;
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

std::unique_ptr<gpu::Texture> TextureManager::UploadTexture(const Texture& texture) const {
    auto pending = PrepareTexture(texture);
    m_device.ExecuteImmediate([&](gpu::CommandList& list) {
        RecordTextureUpload(list, *pending);
    });
    return std::move(pending->image);
}

std::unique_ptr<TextureManager::PendingUpload> TextureManager::PrepareTexture(
    const Texture& texture) const {
    // Validate texture data
    if (texture.pixels.empty()) {
        QL_LOG_ERROR("Texture '{}' has no pixel data", texture.name);
        throw std::runtime_error("Cannot upload empty texture");
    }

    if (texture.channels != 4) {
        QL_LOG_ERROR("Texture '{}' has {} channels (expected 4 for RGBA8)",
                     texture.name, texture.channels);
        throw std::runtime_error("Only RGBA8 textures are supported");
    }

    u64 uncompressedSize = static_cast<u64>(texture.width) * texture.height * 4;

    if (texture.pixels.size() != uncompressedSize) {
        QL_LOG_ERROR("Texture '{}' pixel data size mismatch: expected {} bytes, got {}",
                     texture.name, uncompressedSize, texture.pixels.size());
        throw std::runtime_error("Texture pixel data size mismatch");
    }

    // ========================================================================
    // Try BC7 Compression (4:1 VRAM savings)
    // ========================================================================

    // Check for pre-compressed BC7 data first (from parallel compression during loading)
    if (texture.HasBC7Data()) {
        // Use pre-compressed data directly
        BC7CompressedData compressed;
        compressed.width = texture.width;
        compressed.height = texture.height;
        compressed.blockCountX = texture.bc7Data->blockCountX;
        compressed.blockCountY = texture.bc7Data->blockCountY;
        compressed.isSRGB = texture.isSRGB;
        compressed.data = texture.bc7Data->data;  // Copy to avoid modifying original

        QL_LOG_DEBUG("  Using pre-compressed BC7 data for '{}'", texture.name);
        return PrepareBC7Texture(texture, compressed);
    }

    // Fall back to runtime compression if no pre-compressed data
    if (TextureCompressor::IsAvailable() && TextureCompressor::CanCompress(texture)) {
        auto compressed = TextureCompressor::CompressBC7(texture, false /* fast mode */);
        if (compressed.has_value()) {
            return PrepareBC7Texture(texture, compressed.value());
        }
        // Fall through to uncompressed upload if compression failed
        QL_LOG_WARN("BC7 compression failed for '{}', using uncompressed", texture.name);
    }

    // ========================================================================
    // Uncompressed Upload Path (original code)
    // ========================================================================
    gpu::Format format = texture.isSRGB ? gpu::Format::RGBA8Srgb : gpu::Format::RGBA8Unorm;
    const char* formatName = texture.isSRGB ? "RGBA8_SRGB" : "RGBA8_UNORM";

    QL_LOG_INFO("  Uploading texture '{}': {}x{} {} ({} bytes)",
                texture.name, texture.width, texture.height, formatName, uncompressedSize);

    // Step 1: Create staging buffer (CPU-accessible)
    auto pending = std::make_unique<PendingUpload>();
    pending->staging = CreateBufferOrThrow(
        m_device,
        gpu::BufferDesc{uncompressedSize, gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly, 0});

    // Step 2: Upload CPU pixel data to staging buffer
    pending->staging->Upload(texture.pixels.data(), uncompressedSize);

    // Step 3: Create device-local GPU image with correct format and full mipmap chain
    u32 mipLevels = static_cast<u32>(std::floor(std::log2(std::max(texture.width, texture.height)))) + 1;
    gpu::TextureDesc imageDesc;
    imageDesc.width = texture.width;
    imageDesc.height = texture.height;
    imageDesc.mipLevels = mipLevels;  // Full mipmap chain
    imageDesc.format = format;  // sRGB or UNORM based on texture usage
    imageDesc.usage = gpu::TextureUsage::TransferSrc | gpu::TextureUsage::TransferDst |
                      gpu::TextureUsage::Sampled;
    imageDesc.memory = gpu::MemoryClass::DeviceLocal;
    pending->image = CreateTextureOrThrow(m_device, imageDesc);

    pending->format = format;
    pending->width = texture.width;
    pending->height = texture.height;
    pending->mipLevels = mipLevels;
    return pending;
}

std::unique_ptr<gpu::Sampler> TextureManager::CreateSampler(const TextureSampler& samplerInfo) const {
    // Field for field the sampler create info this filled by hand; every field it left
    // alone is the desc's zero default.
    gpu::SamplerDesc desc;

    // Filter modes
    desc.magFilter = ToFilter(samplerInfo.magFilter);
    desc.minFilter = ToFilter(samplerInfo.minFilter);

    // Wrap modes
    desc.addressModeU = ToAddressMode(samplerInfo.wrapS);
    desc.addressModeV = ToAddressMode(samplerInfo.wrapT);
    desc.addressModeW = gpu::AddressMode::Repeat;  // Default for W

    // Anisotropy (disabled for M1 - can be enabled in M2+ if needed)
    desc.anisotropyEnable = false;
    desc.maxAnisotropy = 1.0f;

    // Border color (for clamp-to-border mode, not used in glTF)
    desc.borderColor = gpu::BorderColor::IntOpaqueBlack;

    // Coordinate system (normalized [0, 1] for glTF)
    desc.unnormalizedCoordinates = false;

    // Comparison (for shadow mapping, not used here)
    desc.compareEnable = false;
    desc.compareOp = gpu::CompareOp::Always;

    // Mipmapping (trilinear filtering)
    desc.mipmapMode = gpu::Filter::Linear;
    desc.mipLodBias = 0.0f;
    desc.minLod = 0.0f;
    desc.maxLod = gpu::kLodClampNone;  // Use all mip levels

    auto sampler = m_device.CreateSampler(desc);
    if (!sampler.has_value()) {
        // The backend's Err is the result code as a decimal
        QL_LOG_ERROR("Failed to create texture sampler: error code {}", sampler.error());
        throw std::runtime_error("Texture sampler creation failed");
    }

    return std::move(sampler.value());
}

gpu::Filter TextureManager::ToFilter(TextureSampler::Filter filter) {
    switch (filter) {
        case TextureSampler::Filter::Nearest:
            return gpu::Filter::Nearest;
        case TextureSampler::Filter::Linear:
            return gpu::Filter::Linear;
        default:
            QL_LOG_WARN("Unknown TextureSampler::Filter, defaulting to LINEAR");
            return gpu::Filter::Linear;
    }
}

gpu::AddressMode TextureManager::ToAddressMode(TextureSampler::WrapMode wrapMode) {
    switch (wrapMode) {
        case TextureSampler::WrapMode::Repeat:
            return gpu::AddressMode::Repeat;
        case TextureSampler::WrapMode::ClampToEdge:
            return gpu::AddressMode::ClampToEdge;
        case TextureSampler::WrapMode::MirroredRepeat:
            return gpu::AddressMode::MirroredRepeat;
        default:
            QL_LOG_WARN("Unknown TextureSampler::WrapMode, defaulting to REPEAT");
            return gpu::AddressMode::Repeat;
    }
}

Texture TextureManager::CreateDummyTexture() {
    Texture dummy;
    dummy.name = "DummyWhiteTexture";
    dummy.sourceUri = "";
    dummy.width = 1;
    dummy.height = 1;
    dummy.channels = 4;

    // 1x1 white pixel (RGBA8: 255, 255, 255, 255)
    dummy.pixels = {255, 255, 255, 255};

    // Default sampler: linear filtering, repeat wrapping
    dummy.sampler.minFilter = TextureSampler::Filter::Linear;
    dummy.sampler.magFilter = TextureSampler::Filter::Linear;
    dummy.sampler.wrapS = TextureSampler::WrapMode::Repeat;
    dummy.sampler.wrapT = TextureSampler::WrapMode::Repeat;

    return dummy;
}

// ============================================================================
// BC7 Compressed Texture Upload
// ============================================================================

std::unique_ptr<TextureManager::PendingUpload> TextureManager::PrepareBC7Texture(
    const Texture& texture,
    const BC7CompressedData& compressed) const {

    // Choose BC7 format based on color space
    gpu::Format format = compressed.isSRGB ? gpu::Format::BC7Srgb : gpu::Format::BC7Unorm;
    const char* formatName = compressed.isSRGB ? "BC7_SRGB" : "BC7_UNORM";

    QL_LOG_INFO("  Uploading BC7 texture '{}': {}x{} {} ({} bytes, ratio: {:.1f}x)",
                texture.name, texture.width, texture.height, formatName,
                compressed.GetCompressedSize(), compressed.GetCompressionRatio());

    // Get aligned dimensions for BC7 (must be multiple of 4)
    u32 alignedWidth = compressed.blockCountX * 4;
    u32 alignedHeight = compressed.blockCountY * 4;

    // Calculate mip levels for BC7 (limited because each level must be at least 4x4)
    // For BC7, we can't generate mipmaps via blit because it's a block-compressed format
    // Option 1: Single mip level (simpler, used here)
    // Option 2: Pre-compress all mip levels on CPU (more complex, future enhancement)
    u32 mipLevels = 1;

    // Step 1: Create staging buffer for compressed data
    u64 compressedSize = static_cast<u64>(compressed.GetCompressedSize());
    auto pending = std::make_unique<PendingUpload>();
    pending->staging = CreateBufferOrThrow(
        m_device,
        gpu::BufferDesc{compressedSize, gpu::BufferUsage::TransferSrc, gpu::MemoryClass::HostOnly, 0});

    // Step 2: Copy compressed data to staging buffer
    pending->staging->Upload(compressed.data.data(), compressedSize);

    // Step 3: Create device-local GPU image with BC7 format
    // Note: For BC7, image dimensions should be the original texture dimensions,
    // not the aligned dimensions. The driver handles the block alignment internally.
    gpu::TextureDesc imageDesc;
    imageDesc.width = texture.width;    // Original width (the driver handles alignment)
    imageDesc.height = texture.height;  // Original height
    imageDesc.mipLevels = mipLevels;
    imageDesc.format = format;
    imageDesc.usage = gpu::TextureUsage::TransferDst | gpu::TextureUsage::Sampled;
    imageDesc.memory = gpu::MemoryClass::DeviceLocal;
    pending->image = CreateTextureOrThrow(m_device, imageDesc);

    pending->format = format;
    pending->width = texture.width;
    pending->height = texture.height;
    pending->mipLevels = mipLevels;
    pending->bufferRowLength = alignedWidth;
    pending->bufferImageHeight = alignedHeight;
    pending->compressed = true;
    return pending;
}

void TextureManager::RecordTextureUpload(gpu::CommandList& list,
                                         const PendingUpload& upload) const {
    // Sampled textures have per-state layouts: transfer destination, then shader read.
    const gpu::TextureRef image =
        upload.image->Ref().WithLayout(gpu::TextureLayoutPolicy::Optimal);

    rendercore::LegacyTransition(list, image, rendercore::Layout::Undefined,
                                 rendercore::Layout::TransferDst,
                                 upload.compressed ? upload.mipLevels : 1);

    gpu::BufferTextureCopy region;
    region.bufferRowLength = upload.bufferRowLength;
    region.bufferImageHeight = upload.bufferImageHeight;
    region.width = upload.width;
    region.height = upload.height;
    list.CopyBufferToTexture(upload.staging->Ref(), image, region);

    if (!upload.compressed) {
        GenerateMipmaps(list, *upload.image, upload.format,
                        upload.width, upload.height, upload.mipLevels);
        return;
    }

    gpu::TextureBarrier ready;
    ready.texture = image;
    ready.from = gpu::Access::TransferWrite;
    ready.to = kSampleAccess;
    ready.mipCount = upload.mipLevels;
    list.Barrier(ready);
}

// ============================================================================
// Mipmap Generation Helper
// ============================================================================

void TextureManager::GenerateMipmaps(gpu::CommandList& list, const gpu::Texture& gpuImage,
                                      gpu::Format format, u32 texWidth, u32 texHeight,
                                      u32 mipLevels) const {
    // Generate mipmaps by blitting each level from the one above it. Every barrier
    // covers one mip level of layer 0.
    const gpu::TextureRef image =
        gpuImage.Ref().WithLayout(gpu::TextureLayoutPolicy::Optimal);

    u32 mipWidth = texWidth;
    u32 mipHeight = texHeight;

    for (u32 i = 1; i < mipLevels; ++i) {
        // Transition previous level to TRANSFER_SRC_OPTIMAL
        gpu::TextureBarrier toSource;
        toSource.texture = image;
        toSource.from = gpu::Access::TransferWrite;
        toSource.to = gpu::Access::TransferRead;
        toSource.baseMip = i - 1;
        list.Barrier(toSource);

        // Blit from level i-1 to level i
        gpu::BlitRegion blit;
        blit.srcMip = i - 1;
        blit.srcX1 = static_cast<i32>(mipWidth);
        blit.srcY1 = static_cast<i32>(mipHeight);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;

        blit.dstMip = i;
        blit.dstX1 = static_cast<i32>(mipWidth);
        blit.dstY1 = static_cast<i32>(mipHeight);

        // Transition current level to TRANSFER_DST before blit. Nothing is waited on, but
        // at the transfer stage: no access alone would make the source top of pipe.
        gpu::TextureBarrier toDestination;
        toDestination.texture = image;
        toDestination.to = gpu::Access::TransferWrite;
        toDestination.discard = true;
        toDestination.baseMip = i;
        gpu::BarrierBatch toDestinationBatch;
        toDestinationBatch.textures = std::span<const gpu::TextureBarrier>(&toDestination, 1);
        toDestinationBatch.srcStages = gpu::PipelineStage::Transfer;
        list.Barrier(toDestinationBatch);

        list.BlitTexture(image, image, blit, gpu::Filter::Linear);

        // Transition previous level to SHADER_READ_ONLY
        gpu::TextureBarrier toSampled;
        toSampled.texture = image;
        toSampled.from = gpu::Access::TransferRead;
        toSampled.to = kSampleAccess;
        toSampled.baseMip = i - 1;
        list.Barrier(toSampled);
    }

    // Transition last mip level to SHADER_READ_ONLY
    gpu::TextureBarrier last;
    last.texture = image;
    last.from = gpu::Access::TransferWrite;
    last.to = kSampleAccess;
    last.baseMip = mipLevels - 1;
    list.Barrier(last);
}

} // namespace quantiloom
