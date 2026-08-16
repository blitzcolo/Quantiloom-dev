/**
 * @file DisplayControl.hpp
 * @brief What happens to the render between the accumulation and the screen
 *
 * The viewport has no tone mapping other than this. The blit to the target is
 * a bare format conversion, so an image whose radiance sits at 1e-2 -- which
 * every LWIR render does -- is black until something maps it into [0,1].
 * "Display enhancement" is therefore not a beautifier that can be left off; it
 * is the only reason an infrared scene is visible at all.
 *
 * Two independent stages, which is the whole point of this header:
 *
 *   1. a TONE operator, taking a scalar to [0,1] -- the contrast stretch;
 *   2. a PALETTE, taking that scalar to a colour -- which changes no contrast.
 *
 * They are separate fields rather than one list of modes because they compose:
 * an ironbow ramp over a linear stretch and an ironbow ramp over an equalised
 * one are both things a user asks for, and enumerating the product would be a
 * dozen names for two decisions.
 *
 * Pure POD, no QL_API, alongside ThermalControl.hpp: the facade methods that
 * pass them are exported, and the host reads the fields directly.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom {

/**
 * @brief How the scalar reaches [0,1]
 *
 * The difference that matters for a thermogram is not sharpness, it is
 * whether the mapping is the same everywhere in the image. Only the first two
 * are, and "brighter is hotter" is a claim about the whole picture.
 */
enum class DisplayToneMode : u8 {
    /// Stretch the percentile window linearly. Globally monotone, and the
    /// only operator that leaves equal temperatures looking equal. What a
    /// thermal camera calls linear AGC.
    Linear = 0,
    /// Histogram equalisation over the whole image, clipped at clipLimit --
    /// thermal cameras call it plateau AGC. Still globally monotone, and
    /// spends the display range where the pixels actually are.
    Equalize,
    /// Per-tile equalisation, bilinearly blended: CLAHE. The best local
    /// detail and the worst radiometry, since two pixels at one temperature
    /// in different tiles come out as different greys. For finding an edge,
    /// not for reading a temperature off the screen.
    Clahe
};

/**
 * @brief How the scalar becomes a colour
 *
 * Grey is the identity and the only one that leaves a colour render alone;
 * every other palette replaces the image's colour with the scalar's.
 */
enum class DisplayPalette : u8 {
    Grey = 0,      ///< White-hot, and what the display did before palettes existed
    GreyInverted,  ///< Black-hot
    Ironbow,       ///< The classic thermal ramp: black, violet, red, white
    Rainbow,       ///< False colour. High apparent contrast, no perceptual order
    Viridis        ///< Perceptually uniform -- the one to publish
};

struct DisplayEnhancementParams {
    bool enabled = false;

    DisplayToneMode toneMode = DisplayToneMode::Linear;
    DisplayPalette palette = DisplayPalette::Grey;

    /// Plateau limit for Equalize and Clahe, as a multiple of the mean bin
    /// count. 1.0 is no equalisation at all; 2 to 4 is the usual range.
    /// Ignored by Linear.
    f32 clipLimit = 2.0f;
    /// Tiles per axis for Clahe. Ignored by the other two.
    i32 tileSize = 8;
    /// Map the luminance and scale the channels by the result, rather than
    /// mapping each channel on its own. Only meaningful under the Grey
    /// palette; the others take their colour from the scalar.
    bool luminanceOnly = true;

    /// The window the tone operator works in, as percentiles of the image's
    /// own luminance. Widening it toward 0 and 100 hands the range back to the
    /// outliers, which on an infrared render is one hot pixel deciding what
    /// everything else looks like.
    f32 percentileLow = 1.0f;
    f32 percentileHigh = 99.0f;
};

}  // namespace quantiloom
