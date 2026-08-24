/**
 * @file SpectralUnmixer.cpp
 * @brief Base-colour textures to endmember weight maps
 */

#include "renderer/SpectralUnmixer.hpp"

#include "core/Log.hpp"
#include "core/NNLS.hpp"
#include "io/ImageIO.hpp"
#include "renderer/ConfigResolve.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <numeric>

namespace quantiloom::rendercore {

namespace {

// Weights are stored divided by this, so the [0, 1] a UNORM8 texture holds
// covers [0, kWeightScale]. One endmember is brightness modulation, and a texel
// brighter than the endmember's own colour needs w > 1 -- an unscaled encoding
// would clip every highlight to the curve's own brightness and flatten exactly
// what this is meant to recover.
//
// 6, not the 2 this started at. These weight distributions are heavily
// right-skewed -- a base colour texture is mostly dark body with a thin tail of
// highlights and markings -- so with the mean anchored to 1 the median sits
// around 0.4 to 0.6 and the 99th percentile at 5 to 9. A ceiling of 2 therefore
// truncated 18 to 31 % of the texels of the four KV-2 tank materials. Measured
// over those four textures:
//
//     ceiling   clipped (hull / track / turret / wheels)   quantum vs median
//        2.0      18.5 %  30.8 %  23.3 %  22.0 %                  1.3 %
//        4.0       4.8 %   6.0 %   1.9 %   9.9 %                  2.6 %
//        6.0       0.1 %   0.8 %   1.3 %   4.0 %                  3.8 %
//        8.0       0.0 %   0.1 %   1.1 %   0.9 %                  5.1 %
//
// The trade is truncation of the bright tail against quantisation everywhere,
// and 6 is where the first has essentially stopped while the second is still
// under 4 % of a typical texel. It does not change the mean -- the anchoring
// below is exact at any ceiling -- only how much of the variation survives.
//
// MUST MATCH the decode in SampleEndmemberWeights (src/shaders/closesthit.rchit),
// which multiplies the sampled UNORM by this number and whose fallback texel is
// its reciprocal. Nothing checks the pair at build time. The value itself lives
// in SpectralUnmixer.hpp so the tests share it.

// sRGB EOTF, per byte value. Built once: the alternative is a pow() per
// channel per texel, which on a 4 megapixel base colour is 12 million of them.
const std::array<f32, 256>& SrgbToLinearTable() {
    static const std::array<f32, 256> table = [] {
        std::array<f32, 256> t{};
        for (i32 i = 0; i < 256; ++i) {
            const f32 s = static_cast<f32>(i) / 255.0f;
            t[static_cast<usize>(i)] =
                s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
        }
        return t;
    }();
    return table;
}

u8 EncodeWeight(f32 w, bool& clipped) {
    const f32 scaled = w / kWeightScale;
    if (scaled > 1.0f) {
        clipped = true;
    }
    return static_cast<u8>(std::clamp(scaled, 0.0f, 1.0f) * 255.0f + 0.5f);
}

/// Load an authored weight map and re-encode it into the halved convention.
/// Authors paint weights in [0, 1] because that is what a paint program and a
/// remote-sensing abundance map both mean by a weight; the halving is an
/// internal storage detail and should not leak into an asset.
std::optional<Texture> LoadAuthoredWeightTexture(const String& path, const String& materialName) {
    auto imageResult = ImageIO::ReadImage(path);
    if (!imageResult.has_value()) {
        QL_LOG_WARN("  Material '{}': cannot read weight texture '{}'", materialName, path);
        return std::nullopt;
    }

    const Image& img = imageResult.value();
    const usize texelCount = static_cast<usize>(img.width) * img.height;
    if (texelCount == 0 || img.channels < 1) {
        QL_LOG_WARN("  Material '{}': weight texture '{}' is empty", materialName, path);
        return std::nullopt;
    }

    Texture tex;
    tex.name = "__unmix_" + materialName;
    tex.width = img.width;
    tex.height = img.height;
    tex.channels = 4;
    tex.sourceUri = path;
    tex.isSRGB = false;              // weights are numbers, not colour
    tex.skipBlockCompression = true; // and BC7 would smear them across blocks
    tex.pixels.resize(texelCount * 4);

    // Endmember k lives in the k-th colour channel, which is not the k-th
    // stored channel for an .exr: it comes back name-sorted, so a four-
    // endmember weight map authored RGBA arrives ABGR and every endmember
    // would be fed its neighbour's weights. See Image::ChannelIndex.
    const u32 channelOf[4] = {img.ChannelIndex("R", 0), img.ChannelIndex("G", 1),
                              img.ChannelIndex("B", 2), img.ChannelIndex("A", 3)};

    bool clipped = false;
    for (usize i = 0; i < texelCount; ++i) {
        for (i32 c = 0; c < 4; ++c) {
            const f32 v = c < static_cast<i32>(img.channels)
                              ? img.data[i * img.channels + channelOf[c]]
                              : 0.0f;
            tex.pixels[i * 4 + static_cast<usize>(c)] = EncodeWeight(v, clipped);
        }
    }
    if (clipped) {
        QL_LOG_WARN("  Material '{}': weight texture has values above {:.0f}, clamped",
                    materialName, kWeightScale);
    }
    return tex;
}

}  // namespace

u64 UnmixTexels(const u8* rgba, const u64 texelCount, const bool srgb,
                const glm::vec3& baseColorFactor, const glm::vec3* colors, const i32 k,
                u8* rgbaOut, f32* meanWeightsOut) {
    if (meanWeightsOut != nullptr) {
        for (i32 c = 0; c < Material::MAX_ENDMEMBERS; ++c) {
            meanWeightsOut[c] = 0.0f;
        }
    }
    if (rgba == nullptr || rgbaOut == nullptr || texelCount == 0) {
        return 0;
    }

    const auto& toLinear = SrgbToLinearTable();
    const usize count = static_cast<usize>(texelCount);

    // Indices rather than pointers so the parallel policy has something to
    // partition; the work per texel is a handful of flops, so this only pays
    // on textures of any size, which is exactly where it is needed.
    Vector<u64> rows(count);
    std::iota(rows.begin(), rows.end(), u64{0});

    // Solved weights are kept as floats for a second pass. Transient, and
    // freed on the way out -- the alternative is solving every texel twice.
    Vector<f32> weights(count * Material::MAX_ENDMEMBERS, 0.0f);

    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](const u64 i) {
        const usize base = static_cast<usize>(i) * 4;

        glm::vec3 texel;
        if (srgb) {
            texel = glm::vec3(toLinear[rgba[base + 0]], toLinear[rgba[base + 1]],
                              toLinear[rgba[base + 2]]);
        } else {
            texel = glm::vec3(static_cast<f32>(rgba[base + 0]) / 255.0f,
                              static_cast<f32>(rgba[base + 1]) / 255.0f,
                              static_cast<f32>(rgba[base + 2]) / 255.0f);
        }
        texel *= baseColorFactor;

        SolveNNLS(colors, k, texel, &weights[base]);
    });

    // ------------------------------------------------------------------
    // Anchor the average to the measured curves.
    // ------------------------------------------------------------------
    // Raw weights make the TEXTURE decide the surface's absolute reflectance:
    // a base colour painted darker than the measured material darkens it, and
    // binding a curve stops meaning that the surface has that curve's
    // reflectance. Scaling every weight by one number so the mean total weight
    // is 1 restores that -- the mixture's area average is the measured
    // material, and the texture contributes only the variation around it,
    // which is the part it actually knows about.
    //
    // One scalar over the whole texture, not a per-texel normalisation: making
    // each texel's weights sum to 1 would flatten the brightness variation
    // this exists to recover.
    f64 perEndmember[Material::MAX_ENDMEMBERS] = {};
    for (usize i = 0; i < count; ++i) {
        for (i32 c = 0; c < k; ++c) {
            perEndmember[c] +=
                static_cast<f64>(weights[i * Material::MAX_ENDMEMBERS + static_cast<usize>(c)]);
        }
    }
    f64 weightSum = 0.0;
    for (i32 c = 0; c < k; ++c) {
        weightSum += perEndmember[c];
    }
    const f64 meanTotal = weightSum / static_cast<f64>(count);
    f32 normalise = meanTotal > 1e-6 ? static_cast<f32>(1.0 / meanTotal) : 1.0f;

    // Re-anchor against the clamp, not just against the raw weights.
    //
    // EncodeWeight stores w/kWeightScale in a UNORM8, so anything above
    // kWeightScale is truncated. Solving for mean(w) = 1 BEFORE that truncation
    // therefore did not achieve mean(w) = 1 after it: every texel above the
    // ceiling lost its excess, and the surface rendered darker than the curve it
    // was bound to. On the KV-2 tank textures -- which are dark and heavily
    // skewed, so 13-18 % of texels sat above the ceiling -- the four materials
    // came out at 0.68 to 0.80 of their measured band reflectance while the
    // ground, having no base-colour texture and so no weight map at all,
    // rendered at exactly 1.00. That is an asymmetry between the two objects
    // being compared, which is the worst place for one to hide.
    //
    // Fixed point rather than closed form: raising the scale pushes more texels
    // into the clamp, so the correction feeds back. It converges quickly because
    // the clipped population only shrinks -- a handful of iterations, and the
    // loop exits as soon as the encoded mean is within a UNORM8 quantum of 1.
    const auto encodedMean = [&](const f32 scale) {
        f64 sum = 0.0;
        for (usize i = 0; i < count; ++i) {
            for (i32 c = 0; c < k; ++c) {
                const f32 w = weights[i * Material::MAX_ENDMEMBERS + static_cast<usize>(c)] * scale;
                sum += std::min(static_cast<f64>(w), static_cast<f64>(kWeightScale));
            }
        }
        return sum / static_cast<f64>(count);
    };

    constexpr i32 kMaxAnchorIters = 8;
    constexpr f64 kAnchorTol = 1.0 / 255.0 / 2.0;  // half a UNORM8 step
    for (i32 iter = 0; iter < kMaxAnchorIters; ++iter) {
        const f64 achieved = encodedMean(normalise);
        if (achieved < 1e-9 || std::abs(achieved - 1.0) <= kAnchorTol) {
            break;
        }
        normalise = static_cast<f32>(static_cast<f64>(normalise) / achieved);
    }

    if (meanWeightsOut != nullptr) {
        // Reported post-clamp, so the "endmember split" log describes the
        // mixture that was actually encoded rather than the one solved for.
        for (i32 c = 0; c < k; ++c) {
            f64 sum = 0.0;
            for (usize i = 0; i < count; ++i) {
                const f32 w = weights[i * Material::MAX_ENDMEMBERS + static_cast<usize>(c)] * normalise;
                sum += std::min(static_cast<f64>(w), static_cast<f64>(kWeightScale));
            }
            meanWeightsOut[c] = static_cast<f32>(sum / static_cast<f64>(count));
        }
    }

    std::atomic<u64> clippedCount{0};
    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](const u64 i) {
        const usize base = static_cast<usize>(i) * 4;
        bool clipped = false;
        for (i32 c = 0; c < 4; ++c) {
            rgbaOut[base + static_cast<usize>(c)] =
                c < k ? EncodeWeight(weights[base + static_cast<usize>(c)] * normalise, clipped)
                      : u8{0};
        }
        if (clipped) {
            clippedCount.fetch_add(1, std::memory_order_relaxed);
        }
    });

    return clippedCount.load(std::memory_order_relaxed);
}

void BuildUnmixWeightTextures(Scene& scene, ResolvedMaterialSpectra& spectra,
                              const String& baseDir, ConfigApplyReport& report) {
    if (spectra.materialNameToEndmembers.empty()) {
        return;
    }

    u32 built = 0;
    for (auto& material : scene.materials) {
        auto slotsIt = spectra.materialNameToEndmembers.find(material.name);
        if (slotsIt == spectra.materialNameToEndmembers.end()) {
            continue;
        }
        EndmemberSlots& slots = slotsIt->second;
        if (slots.unmix == Material::SpectralUnmixMode::Off || slots.count < 1) {
            continue;
        }

        std::optional<Texture> weightTexture;

        if (slots.unmix == Material::SpectralUnmixMode::Texture) {
            weightTexture = LoadAuthoredWeightTexture(
                ResolveConfigPath(slots.weightTexturePath, baseDir), material.name);
        } else if (material.baseColorTextureIndex >= 0 &&
                   material.baseColorTextureIndex < static_cast<i32>(scene.textures.size())) {
            const Texture& source = scene.textures[static_cast<usize>(material.baseColorTextureIndex)];
            if (source.pixels.empty()) {
                // Reached only if something uploaded before this ran. Worth a
                // warning rather than a silent flat surface, because the fix
                // is an ordering fix and there is nothing in the image to hint
                // at it.
                QL_LOG_WARN("  Material '{}': base colour pixels already released, "
                            "cannot unmix (this runs before texture upload)",
                            material.name);
            } else if (source.channels == 4) {
                const u64 texelCount = static_cast<u64>(source.width) * source.height;

                Texture tex;
                tex.name = "__unmix_" + material.name;
                tex.width = source.width;
                tex.height = source.height;
                tex.channels = 4;
                tex.isSRGB = false;
                tex.skipBlockCompression = true;
                tex.pixels.resize(static_cast<usize>(texelCount) * 4);

                f32 meanWeights[Material::MAX_ENDMEMBERS] = {};
                const u64 clipped =
                    UnmixTexels(source.pixels.data(), texelCount, source.isSRGB,
                                glm::vec3(material.baseColorFactor), slots.colorsLinear,
                                slots.count, tex.pixels.data(), meanWeights);

                // How the surface divided. An endmember at a few percent was
                // not really in the texture, and the only way to find that out
                // is to be told -- the render just looks like the others.
                if (slots.count > 1) {
                    String split;
                    for (i32 c = 0; c < slots.count; ++c) {
                        if (c > 0) split += " / ";
                        split += std::to_string(static_cast<i32>(meanWeights[c] * 100.0f + 0.5f)) + "%";
                    }
                    QL_LOG_INFO("  Material '{}': endmember split {}", material.name, split);
                }

                // One percent is the line between "a few specular highlights
                // are brighter than the measured curve", which is normal, and
                // "the endmembers are too dark for this texture", which shows
                // up as flattened bright areas.
                if (clipped * 100 > texelCount) {
                    QL_LOG_WARN("  Material '{}': {:.1f}% of texels want more than {:.0f}x "
                                "an endmember and were clamped -- the mixture may be too "
                                "dark for this texture",
                                material.name,
                                100.0 * static_cast<f64>(clipped) / static_cast<f64>(texelCount),
                                kWeightScale);
                }
                weightTexture = std::move(tex);
            }
        }

        if (!weightTexture.has_value()) {
            // No base colour to unmix, or the authored map would not load.
            // Both are fine: the shader reads a missing weight texture as
            // w = (1, 0, 0, 0) and renders the first curve flat, which is what
            // this material did before endmembers existed.
            continue;
        }

        slots.weightTextureIndex = static_cast<i32>(scene.textures.size());
        scene.textures.push_back(std::move(weightTexture.value()));
        ++built;

        QL_LOG_INFO("  Material '{}': {} endmember(s), weight texture index {} ({}x{})",
                    material.name, slots.count, slots.weightTextureIndex,
                    scene.textures.back().width, scene.textures.back().height);
    }

    if (built > 0) {
        report.messages.push_back(
            {ConfigApplyMessage::Severity::Info, "spectral.unmix",
             "  Unmixed " + std::to_string(built) + " base colour(s) into endmember weights"});
    }
}

}  // namespace quantiloom::rendercore
