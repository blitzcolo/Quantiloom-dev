/**
 * @file McpImageUtil.hpp
 * @brief Turns a rendered frame into something an agent can afford to look at
 *
 * A capture off the render target is f32 linear radiance at viewport
 * resolution. Handed to a model as-is it is both unreadable and ruinously
 * expensive: an image block costs roughly width x height / 750 tokens, so a
 * 1920x1080 frame is about 2,800 and the base64 text of the same PNG is an
 * order of magnitude worse. Downsampling to 768 on the long edge brings that
 * under a thousand while leaving every lighting artefact anyone would ask about
 * plainly visible.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Image.hpp"
#include "core/Types.hpp"

namespace quantiloom::mcp {

/**
 * @brief Box-filter an image down so its long edge is at most `maxDimension`
 *
 * Box rather than anything fancier: this is for looking at, the source is
 * already noisy, and a separable filter would need a temporary the size of the
 * frame. Returns the input unchanged when it is already small enough.
 */
[[nodiscard]] Image Downsample(const Image& image, u32 maxDimension);

/**
 * @brief Encode to PNG in memory and base64 it
 *
 * The image is sRGB-encoded on the way out, matching ImageIO::WritePNG, so what
 * the agent sees is what Studio would have written to disk. Values are clamped;
 * a linear HDR frame with no exposure applied will clip, which is itself worth
 * seeing.
 *
 * @param image  1, 3 or 4 channel. More channels than that use the first three.
 * @return Base64 PNG, or an empty string if encoding failed.
 */
[[nodiscard]] String EncodePngBase64(const Image& image);

}  // namespace quantiloom::mcp
