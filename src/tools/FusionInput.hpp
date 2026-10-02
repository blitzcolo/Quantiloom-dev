#pragma once

#include "core/Image.hpp"
#include "core/Types.hpp"
#include <array>

namespace quantiloom {

// Validate the joint extent before extracting any band. Each source keeps its
// own named radiance channel (EXR channel zero can be alpha).
inline Result<std::array<Image, 3>, String> PrepareFusionInputs(
    const Image& vis, const Image& swir, const Image& mwir) {
    const std::array<const Image*, 3> sources{&vis, &swir, &mwir};
    for (const auto* source : sources) {
        if (!source->IsValid())
            return Result<std::array<Image, 3>, String>::Err("Invalid fusion input image");
        if (source->width != vis.width || source->height != vis.height)
            return Result<std::array<Image, 3>, String>::Err("Fusion image dimensions must match");
        if (source->LuminanceChannelIndex() >= source->channels)
            return Result<std::array<Image, 3>, String>::Err("Invalid fusion radiance channel");
    }
    std::array<Image, 3> bands;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto& source = *sources[i];
        const u32 channel = source.LuminanceChannelIndex();
        bands[i] = Image(source.width, source.height, 1);
        for (u32 y = 0; y < source.height; ++y)
            for (u32 x = 0; x < source.width; ++x)
                bands[i](x, y, 0) = source(x, y, channel);
    }
    return Result<std::array<Image, 3>, String>(std::move(bands));
}

} // namespace quantiloom
