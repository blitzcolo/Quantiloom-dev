#include "io/ImageIO.hpp"

// stb_image for PNG/JPEG/BMP/TGA/HDR reading
// Note: STB_IMAGE_IMPLEMENTATION is defined in GltfLoader.cpp via tinygltf
// We only need to include the header here
#include <stb_image.h>

// stb_image_write for PNG output (header-only library)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfStringAttribute.h>

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <limits>

namespace quantiloom {
namespace {
// Insert the small provenance summary after IHDR. The compressed pixel payload
// is untouched; stb still owns PNG encoding and its existing colour conversion.
struct PngSummaryWriter {
    std::ofstream output;
    String chunks;
    usize offset = 0;
    static void BigEndian(String& out, u32 value) {
        for (int shift = 24; shift >= 0; shift -= 8)
            out.push_back(static_cast<char>((value >> shift) & 255));
    }
    void AddText(const String& key, const String& value) {
        String chunk = "iTXt" + key + String(5, '\0') + value;
        u32 crc = 0xffffffffu;
        for (unsigned char byte : chunk) {
            crc ^= byte;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0u);
        }
        BigEndian(chunks, static_cast<u32>(chunk.size() - 4));
        chunks += chunk;
        BigEndian(chunks, crc ^ 0xffffffffu);
    }
    static void Write(void* context, void* bytes, int count) {
        auto& writer = *static_cast<PngSummaryWriter*>(context);
        const auto* data = static_cast<const char*>(bytes);
        const usize size = static_cast<usize>(count);
        if (writer.offset < 33 && writer.offset + size >= 33) {
            const auto prefix = 33 - writer.offset;
            writer.output.write(data, static_cast<std::streamsize>(prefix));
            writer.output.write(writer.chunks.data(), static_cast<std::streamsize>(writer.chunks.size()));
            writer.output.write(data + prefix, static_cast<std::streamsize>(size - prefix));
        } else writer.output.write(data, count);
        writer.offset += size;
    }
};
} // namespace

// ============================================================================
// Helper: Convert Image to EXR FrameBuffer
// ============================================================================

static void SetupFrameBufferForWrite(
    Imf::FrameBuffer& fb,
    const Image& img,
    std::vector<std::vector<float>>& buffers)
{
    // EXR expects separate buffers for each channel
    // We need to de-interleave our channel-last format

    buffers.resize(img.channels);
    for (u32 c = 0; c < img.channels; ++c) {
        buffers[c].resize(img.width * img.height);

        // De-interleave: extract channel c from image data
        for (u32 y = 0; y < img.height; ++y) {
            for (u32 x = 0; x < img.width; ++x) {
                buffers[c][y * img.width + x] = img(x, y, c);
            }
        }

        // Insert channel into FrameBuffer
        const char* channelName = img.channelNames[c].c_str();
        fb.insert(
            channelName,
            Imf::Slice(
                Imf::FLOAT,                                  // type
                reinterpret_cast<char *>(buffers[c].data()), // base
                sizeof(float),                               // xStride
                sizeof(float) * img.width                    // yStride
            )
        );
    }
}

// ============================================================================
// Helper: Convert EXR FrameBuffer to Image
// ============================================================================

static void ReadFrameBufferToImage(
    Imf::InputFile& file,
    Image& img,
    const std::vector<std::string>& channelNames)
{
    img.channels = static_cast<u32>(channelNames.size());
    img.channelNames = channelNames;
    img.data.resize(img.width * img.height * img.channels, 0.0f);

    // Allocate temporary per-channel buffers
    std::vector<std::vector<float>> buffers(img.channels);
    for (u32 c = 0; c < img.channels; ++c) {
        buffers[c].resize(img.width * img.height);
    }

    // Setup FrameBuffer for reading
    Imf::FrameBuffer fb;
    for (u32 c = 0; c < img.channels; ++c) {
        fb.insert(
            channelNames[c].c_str(),
            Imf::Slice(
                Imf::FLOAT,
                reinterpret_cast<char *>(buffers[c].data()),
                sizeof(float),
                sizeof(float) * img.width
            )
        );
    }

    file.setFrameBuffer(fb);
    file.readPixels(0, static_cast<int>(img.height) - 1);

    // Interleave channels into image data (channel-last format)
    for (u32 y = 0; y < img.height; ++y) {
        for (u32 x = 0; x < img.width; ++x) {
            for (u32 c = 0; c < img.channels; ++c) {
                img(x, y, c) = buffers[c][y * img.width + x];
            }
        }
    }
}

// ============================================================================
// Public API: WriteEXR
// ============================================================================

Result<void, String> ImageIO::WriteUIntEXR(const String& filepath, const UIntImage& image) {
    if (image.width == 0 || image.height == 0 ||
        image.width > static_cast<u32>(std::numeric_limits<int>::max()) ||
        image.height > static_cast<u32>(std::numeric_limits<int>::max()) ||
        static_cast<u64>(image.width) * image.height != image.pixels.size())
        return Result<void, String>::Err("invalid integer image dimensions or pixel count");
    try {
        Imf::Header header(static_cast<int>(image.width), static_cast<int>(image.height));
        header.compression() = Imf::ZIP_COMPRESSION;
        header.channels().insert("instance_id", Imf::Channel(Imf::UINT));
        for (const auto& [key, value] : image.metadata) {
            if (key.empty() || key.find('\0') != String::npos || header.find(key.c_str()) != header.end())
                return Result<void, String>::Err("reserved or invalid integer EXR metadata key: " + key);
            header.insert(key.c_str(), Imf::StringAttribute(value));
        }
        Imf::FrameBuffer buffer;
        buffer.insert("instance_id", Imf::Slice(Imf::UINT,
            reinterpret_cast<char*>(const_cast<u32*>(image.pixels.data())),
            sizeof(u32), sizeof(u32) * image.width));
        Imf::OutputFile file(filepath.c_str(), header);
        file.setFrameBuffer(buffer);
        file.writePixels(static_cast<int>(image.height));
        return Result<void, String>::Ok();
    } catch (const std::exception& e) {
        return Result<void, String>::Err(e.what());
    }
}

Result<UIntImage, String> ImageIO::ReadUIntEXR(const String& filepath) {
    try {
        Imf::InputFile file(filepath.c_str());
        const auto& header = file.header();
        const auto window = header.dataWindow();
        if (window.min.x != 0 || window.min.y != 0 || window.max.x < 0 || window.max.y < 0 ||
            window != header.displayWindow())
            return Result<UIntImage, String>::Err("integer EXR requires a complete grid at origin (0,0)");
        const auto& channels = header.channels();
        auto channel = channels.begin();
        if (channel == channels.end() || String(channel.name()) != "instance_id" ||
            channel.channel().type != Imf::UINT || channel.channel().xSampling != 1 ||
            channel.channel().ySampling != 1 || ++channel != channels.end())
            return Result<UIntImage, String>::Err("integer EXR requires one full-resolution UINT instance_id channel");
        // Reject codecs that do not promise lossless storage as a format contract.
        switch (header.compression()) {
        case Imf::NO_COMPRESSION: case Imf::RLE_COMPRESSION:
        case Imf::ZIPS_COMPRESSION: case Imf::ZIP_COMPRESSION: case Imf::PIZ_COMPRESSION: break;
        default: return Result<UIntImage, String>::Err("integer EXR requires lossless compression");
        }
        UIntImage image;
        image.width = static_cast<u32>(window.max.x) + 1;
        image.height = static_cast<u32>(window.max.y) + 1;
        const u64 count = static_cast<u64>(image.width) * image.height;
        if (count > image.pixels.max_size())
            return Result<UIntImage, String>::Err("integer EXR grid is too large");
        image.pixels.resize(static_cast<usize>(count));
        for (auto it = header.begin(); it != header.end(); ++it)
            if (const auto* value = dynamic_cast<const Imf::StringAttribute*>(&it.attribute()))
                image.metadata.emplace(it.name(), value->value());
        Imf::FrameBuffer buffer;
        buffer.insert("instance_id", Imf::Slice(Imf::UINT,
            reinterpret_cast<char*>(image.pixels.data()), sizeof(u32), sizeof(u32) * image.width));
        file.setFrameBuffer(buffer);
        file.readPixels(0, window.max.y);
        return image;
    } catch (const std::exception& e) {
        return Result<UIntImage, String>::Err(e.what());
    }
}

bool ImageIO::WriteEXR(const std::string& filepath, const Image& image) {
    if (!image.IsValid()) {
        QL_LOG_ERROR("ImageIO::WriteEXR: Invalid image");
        return false;
    }

    try {
        // Create EXR header
        Imf::Header header(static_cast<int>(image.width), static_cast<int>(image.height));

        // Add channels to header
        for (u32 c = 0; c < image.channels; ++c) {
            header.channels().insert(
                image.channelNames[c].c_str(),
                Imf::Channel(Imf::FLOAT)
            );
        }

        // Add metadata as string attributes
        for (const auto& [key, value] : image.metadata) {
            header.insert(key.c_str(), Imf::StringAttribute(value));
        }

        // Create output file
        Imf::OutputFile file(filepath.c_str(), header);

        // Setup FrameBuffer
        Imf::FrameBuffer fb;
        std::vector<std::vector<float>> buffers;
        SetupFrameBufferForWrite(fb, image, buffers);

        file.setFrameBuffer(fb);
        file.writePixels(static_cast<int>(image.height));

        QL_LOG_INFO("ImageIO::WriteEXR: Wrote {}x{} image with {} channels to {}",
                    image.width, image.height, image.channels, filepath);
        return true;

    } catch (const std::exception& e) {
        QL_LOG_ERROR("ImageIO::WriteEXR: Failed to write {}: {}", filepath, e.what());
        return false;
    }
}

// ============================================================================
// Public API: WritePNG
// ============================================================================

// Helper: Apply sRGB gamma encoding (IEC 61966-2-1)
// Input: linear value [0, 1]
// Output: sRGB encoded value [0, 1]
static float LinearToSRGB(float linear) {
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    }
}

bool ImageIO::WritePNG(const std::string& filepath, const Image& image) {
    if (!image.IsValid()) {
        QL_LOG_ERROR("ImageIO::WritePNG: Invalid image");
        return false;
    }

    // PNG supports 1 (grayscale), 3 (RGB), or 4 (RGBA) channels
    if (image.channels != 1 && image.channels != 3 && image.channels != 4) {
        QL_LOG_ERROR("ImageIO::WritePNG: Unsupported channel count {} (need 1, 3, or 4)",
                     image.channels);
        return false;
    }

    try {
        // Camera display products are already encoded sRGB. Ordinary render
        // images still take the legacy linear-to-sRGB path.
        const auto cameraKind = image.metadata.find("camera_signal_kind");
        const bool alreadySrgb = cameraKind != image.metadata.end() &&
            (cameraKind->second == "display_srgb" ||
             cameraKind->second == "device_preview_srgb");
        // Convert float [0,1] to uint8 [0,255] with clamping and sRGB gamma
        std::vector<uint8_t> pixels(image.width * image.height * image.channels);

        for (u32 y = 0; y < image.height; ++y) {
            for (u32 x = 0; x < image.width; ++x) {
                for (u32 c = 0; c < image.channels; ++c) {
                    float value = image(x, y, c);

                    // Clamp to [0, 1] range (HDR values may exceed 1.0)
                    value = std::clamp(value, 0.0f, 1.0f);

                    // Apply sRGB gamma encoding (shader outputs linear RGB)
                    // Alpha channel (c == 3) should NOT be gamma encoded
                    if (c < 3 && !alreadySrgb) {
                        value = LinearToSRGB(value);
                    }

                    // Convert to 8-bit
                    pixels[(y * image.width + x) * image.channels + c] =
                        static_cast<uint8_t>(value * 255.0f + 0.5f);
                }
            }
        }

        int result = 0;
        if (image.metadata.contains("quantiloom_record_id")) {
            PngSummaryWriter writer;
            writer.output.open(std::filesystem::path(filepath), std::ios::binary | std::ios::trunc);
            if (!writer.output) return false;
            for (const char* key : {"quantiloom_record_id", "quantiloom_product_id", "quantiloom_sidecar"}) {
                const auto found = image.metadata.find(key);
                if (found != image.metadata.end()) writer.AddText(key, found->second);
            }
            result = stbi_write_png_to_func(PngSummaryWriter::Write, &writer,
                static_cast<int>(image.width), static_cast<int>(image.height),
                static_cast<int>(image.channels), pixels.data(),
                static_cast<int>(image.width * image.channels));
            writer.output.close();
            if (!writer.output || writer.offset < 33) result = 0;
        } else {
            result = stbi_write_png(filepath.c_str(), static_cast<int>(image.width),
                static_cast<int>(image.height), static_cast<int>(image.channels), pixels.data(),
                static_cast<int>(image.width * image.channels));
        }

        if (result == 0) {
            QL_LOG_ERROR("ImageIO::WritePNG: stbi_write_png failed for {}", filepath);
            return false;
        }

        QL_LOG_INFO("ImageIO::WritePNG: Wrote {}x{} image with {} channels to {} (sRGB encoded)",
                    image.width, image.height, image.channels, filepath);
        return true;

    } catch (const std::exception& e) {
        QL_LOG_ERROR("ImageIO::WritePNG: Failed to write {}: {}", filepath, e.what());
        return false;
    }
}

// ============================================================================
// Public API: ReadEXR
// ============================================================================

std::optional<Image> ImageIO::ReadEXR(const std::string& filepath) {
    if (!FileExists(filepath)) {
        QL_LOG_ERROR("ImageIO::ReadEXR: File not found: {}", filepath);
        return std::nullopt;
    }

    try {
        Imf::InputFile file(filepath.c_str());
        const Imf::Header& header = file.header();

        // Get dimensions
        const Imath::Box2i dw = header.dataWindow();
        u32 width = dw.max.x - dw.min.x + 1;
        u32 height = dw.max.y - dw.min.y + 1;

        // Get channel names
        std::vector<std::string> channelNames;
        const Imf::ChannelList& channels = header.channels();
        for (auto it = channels.begin(); it != channels.end(); ++it) {
            channelNames.emplace_back(it.name());
        }

        if (channelNames.empty()) {
            QL_LOG_ERROR("ImageIO::ReadEXR: No channels found in {}", filepath);
            return std::nullopt;
        }

        // Create image
        Image img;
        img.width = width;
        img.height = height;

        // Read pixel data
        ReadFrameBufferToImage(file, img, channelNames);

        // Read metadata (string attributes)
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (const auto* attr =
                header.findTypedAttribute<Imf::StringAttribute>(it.name())) {
                img.metadata[it.name()] = attr->value();
            }
        }

        QL_LOG_INFO("ImageIO::ReadEXR: Read {}x{} image with {} channels from {}",
                    width, height, channelNames.size(), filepath);
        return img;

    } catch (const std::exception& e) {
        QL_LOG_ERROR("ImageIO::ReadEXR: Failed to read {}: {}", filepath, e.what());
        return std::nullopt;
    }
}

// ============================================================================
// Public API: ReadImage (supports EXR, PNG, JPEG, BMP, TGA, HDR)
// ============================================================================

std::optional<Image> ImageIO::ReadImage(const std::string& filepath) {
    if (!FileExists(filepath)) {
        QL_LOG_ERROR("ImageIO::ReadImage: File not found: {}", filepath);
        return std::nullopt;
    }

    // Determine format by extension
    std::string ext = std::filesystem::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // EXR files use OpenEXR library
    if (ext == ".exr") {
        return ReadEXR(filepath);
    }

    // All other formats (PNG, JPEG, BMP, TGA, HDR, PSD, GIF, PIC, PNM) use stb_image
    // stb_image supports: JPEG, PNG, BMP, PSD, TGA, GIF, HDR, PIC, PNM

    int width = 0, height = 0, channels = 0;

    // Check if it's an HDR file (Radiance .hdr format)
    bool isHDR = (ext == ".hdr");
    float* floatData = nullptr;
    unsigned char* byteData = nullptr;

    if (isHDR) {
        // Load as float for HDR files
        floatData = stbi_loadf(filepath.c_str(), &width, &height, &channels, 0);
        if (!floatData) {
            QL_LOG_ERROR("ImageIO::ReadImage: stbi_loadf failed for {}: {}",
                         filepath, stbi_failure_reason());
            return std::nullopt;
        }
    } else {
        // Load as 8-bit for LDR files
        byteData = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
        if (!byteData) {
            QL_LOG_ERROR("ImageIO::ReadImage: stbi_load failed for {}: {}",
                         filepath, stbi_failure_reason());
            return std::nullopt;
        }
    }

    // Create Image and fill with data
    Image img;
    img.width = static_cast<u32>(width);
    img.height = static_cast<u32>(height);
    img.channels = static_cast<u32>(channels);
    img.data.resize(static_cast<size_t>(width) * height * channels);

    // Set default channel names based on channel count
    if (channels == 1) {
        img.channelNames = {"Gray"};
    } else if (channels == 2) {
        img.channelNames = {"Gray", "Alpha"};
    } else if (channels == 3) {
        img.channelNames = {"R", "G", "B"};
    } else if (channels == 4) {
        img.channelNames = {"R", "G", "B", "A"};
    } else {
        for (u32 c = 0; c < img.channels; ++c) {
            img.channelNames.push_back("Channel" + std::to_string(c));
        }
    }

    if (isHDR) {
        // Direct copy for HDR (already float)
        std::memcpy(img.data.data(), floatData,
                    static_cast<size_t>(width) * height * channels * sizeof(float));
        stbi_image_free(floatData);
    } else {
        // Convert 8-bit to float [0, 1]
        for (size_t i = 0; i < img.data.size(); ++i) {
            img.data[i] = static_cast<float>(byteData[i]) / 255.0f;
        }
        stbi_image_free(byteData);
    }

    QL_LOG_INFO("ImageIO::ReadImage: Read {}x{} image with {} channels from {}",
                width, height, channels, filepath);
    return img;
}

// ============================================================================
// Public API: FileExists
// ============================================================================

bool ImageIO::FileExists(const std::string& filepath) {
    return std::filesystem::exists(filepath) &&
           std::filesystem::is_regular_file(filepath);
}

// ============================================================================
// Public API: GetDimensions
// ============================================================================

std::optional<std::tuple<u32, u32, u32>> ImageIO::GetDimensions(const std::string& filepath) {
    if (!FileExists(filepath)) {
        return std::nullopt;
    }

    try {
        const Imf::InputFile file(filepath.c_str());
        const Imf::Header& header = file.header();

        const Imath::Box2i& dw = header.dataWindow();
        u32 width = dw.max.x - dw.min.x + 1;
        u32 height = dw.max.y - dw.min.y + 1;

        u32 channels = 0;
        const Imf::ChannelList& channelList = header.channels();
        for (auto it = channelList.begin(); it != channelList.end(); ++it) {
            ++channels;
        }

        return std::make_tuple(width, height, channels);

    } catch (const std::exception& e) {
        QL_LOG_ERROR("ImageIO::GetDimensions: Failed: {}", e.what());
        return std::nullopt;
    }
}

} // namespace quantiloom
