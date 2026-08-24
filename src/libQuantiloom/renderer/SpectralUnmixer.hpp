/**
 * @file SpectralUnmixer.hpp
 * @brief Base-colour textures to endmember weight maps
 *
 * A material bound to a measured spectral curve renders as one flat
 * reflectance: the curve replaces the base-colour texture, and everything the
 * texture said about where the surface varies is lost. This turns that texture
 * back into information the spectral path can use, by asking of every texel
 * how much of each endmember would produce that colour:
 *
 *   argmin || sum_i w_i c_i - texel ||,  w >= 0
 *
 * where c_i is what endmember i looks like under D65. The weights go into an
 * RGBA texture the shader samples per hit, so the reflectance stays measured
 * and only the proportions vary across the surface.
 *
 * With one endmember this reduces to a projection -- the weight is just how
 * bright the texel is relative to the curve's own colour -- which is why it is
 * on by default for materials that already name a single reference.
 *
 * TIMING. This must run after ResolveMaterialSpectra, which is what supplies
 * the endmember colours, and before TextureManager::UploadTextures, which
 * releases the very base-colour pixels it reads. Both front ends have that
 * window; see the call sites in OfflineRenderer and ExternalRenderContext.
 */

#pragma once

#include "core/Types.hpp"
#include "renderer/ConfigApply.hpp"
#include "renderer/ConfigResolve.hpp"
#include "scene/Scene.hpp"

#include <glm/glm.hpp>

namespace quantiloom::rendercore {

/// Encoded weights are stored divided by this, so the [0, 1] a UNORM8 texture
/// holds covers [0, kWeightScale]. Declared here rather than in the .cpp so the
/// tests decode with the same number the writer encodes with -- they had their
/// own copy, and it silently decoded every weight a third of its true value the
/// moment this changed.
///
/// The shader has the only remaining mirror, in SampleEndmemberWeights
/// (src/shaders/closesthit.rchit); nothing checks that pair at build time.
/// SpectralUnmixer.cpp carries the measurements behind the value.
inline constexpr f32 kWeightScale = 6.0f;

/**
 * @brief Unmix one texel buffer against a set of endmember colours
 *
 * @param rgba        source pixels, 4 bytes per texel
 * @param texelCount  number of texels
 * @param srgb        true when the source is gamma-encoded and must be
 *                    linearised first -- getting this wrong biases every
 *                    weight, since sRGB 0.5 is linear 0.21
 * @param baseColorFactor  glTF factor the texture is modulated by
 * @param colors      endmember colours in linear sRGB
 * @param k           endmember count, 1..Material::MAX_ENDMEMBERS
 * @param rgbaOut     receives 4 bytes per texel: channel i is
 *                    w_i / kWeightScale
 * @param meanWeightsOut  optional, receives Material::MAX_ENDMEMBERS mean
 *                    weights after normalisation. They sum to 1 and say how
 *                    the surface divided between the endmembers, which is the
 *                    only feedback on whether a chosen endmember was present
 *                    in the texture at all.
 * @return number of texels whose weights were clipped by the encoding
 *
 * The weights are scaled by one scalar so their mean total is 1, which keeps
 * the surface's average reflectance equal to the measured mixture: without it
 * a base colour painted darker than the material would darken the material,
 * and binding a measured curve would stop meaning the surface has that curve's
 * reflectance. The texture supplies the variation, the curve supplies the
 * level. An authored weight map is NOT rescaled -- it already says what it
 * means.
 *
 * Separated from the scene walk so it can be tested on known colours, and so
 * the interactive path can reuse it for a single material.
 */
u64 UnmixTexels(const u8* rgba, u64 texelCount, bool srgb,
                const glm::vec3& baseColorFactor, const glm::vec3* colors, i32 k,
                u8* rgbaOut, f32* meanWeightsOut = nullptr);

/**
 * @brief Build a weight texture for every material with an endmember mixture
 *
 * Appends the weight textures to @p scene.textures and records their indices
 * in @p spectra. Materials whose unmix mode is Off, and materials whose
 * unmixing fails, are left with no weight texture -- which the shader reads as
 * w = (1, 0, 0, 0), the single flat curve, so a failure here costs detail and
 * never correctness.
 *
 * @param baseDir  directory a spectral_weight_texture path is relative to
 */
void BuildUnmixWeightTextures(Scene& scene, ResolvedMaterialSpectra& spectra,
                              const String& baseDir, ConfigApplyReport& report);

}  // namespace quantiloom::rendercore
