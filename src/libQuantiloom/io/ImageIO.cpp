#include "ImageIO.hpp"

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

namespace quantiloom {

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
                (char*)buffers[c].data(),                    // base
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
                (char*)buffers[c].data(),
                sizeof(float),
                sizeof(float) * img.width
            )
        );
    }

    file.setFrameBuffer(fb);
    file.readPixels(0, img.height - 1);

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

bool ImageIO::WriteEXR(const std::string& filepath, const Image& image) {
    if (!image.IsValid()) {
        QL_LOG_ERROR("ImageIO::WriteEXR: Invalid image");
        return false;
    }

    try {
        // Create EXR header
        Imf::Header header(image.width, image.height);

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
        file.writePixels(image.height);

        QL_LOG_INFO("ImageIO::WriteEXR: Wrote {}x{} image with {} channels to {}",
                    image.width, image.height, image.channels, filepath);
        return true;

    } catch (const std::exception& e) {
        QL_LOG_ERROR("ImageIO::WriteEXR: Failed to write {}: {}", filepath, e.what());
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
        Imath::Box2i dw = header.dataWindow();
        u32 width = dw.max.x - dw.min.x + 1;
        u32 height = dw.max.y - dw.min.y + 1;

        // Get channel names
        std::vector<std::string> channelNames;
        const Imf::ChannelList& channels = header.channels();
        for (auto it = channels.begin(); it != channels.end(); ++it) {
            channelNames.push_back(it.name());
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
            if (const Imf::StringAttribute* attr =
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
        Imf::InputFile file(filepath.c_str());
        const Imf::Header& header = file.header();

        Imath::Box2i dw = header.dataWindow();
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
