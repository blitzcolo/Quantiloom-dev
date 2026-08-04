/**
 * @file D65Illuminant.hpp
 * @brief CIE standard illuminant D65, relative spectral power distribution
 *
 * Internal, like CIE_CMF_Data.hpp beside it: the table is an implementation
 * detail of the colour conversions, not part of the public surface.
 *
 * D65 is the white point sRGB is defined against, which is what makes it the
 * right illuminant for turning a measured reflectance into the colour a
 * texture author would have painted. It is NOT the right illuminant for
 * rendering -- that is the scene's own solar spectrum (assets/luts/astmg173.csv),
 * and the two must not be confused. Sun colour goes through
 * SpectralIrradianceToLinearSrgb; reflectance colour goes through
 * ReflectanceToLinearSrgbD65.
 *
 * Values are CIE 15:2004 as published, 5 nm from 380 to 780 nm, normalised to
 * 100 at 560 nm. Transcribed from assets/luts/CIE_std_illum_D65.csv, which is
 * the CIE's own 1 nm dataset; 5 nm is the resolution the standard tabulates
 * the S components at, and the curve is smooth enough between knots that
 * linear interpolation costs nothing measurable.
 *
 * Compiled in rather than read from assets/ because the unmixer runs inside
 * scene loading, where a missing asset file would be a silent quality
 * regression rather than a loud failure.
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom {

inline constexpr f32 D65_LAMBDA_MIN = 380.0f;
inline constexpr f32 D65_LAMBDA_MAX = 780.0f;
inline constexpr f32 D65_LAMBDA_STEP = 5.0f;
inline constexpr u32 D65_LUT_SIZE = 81;

inline constexpr f32 CIE_D65[D65_LUT_SIZE] = {
    49.9755f, 52.3118f, 54.6482f, 68.7015f, 82.7549f, 87.1204f, 91.4860f, 92.4589f,
    93.4318f, 90.0570f, 86.6823f, 95.7736f, 104.8650f, 110.9360f, 117.0080f, 117.4100f,
    117.8120f, 116.3360f, 114.8610f, 115.3920f, 115.9230f, 112.3670f, 108.8110f, 109.0820f,
    109.3540f, 108.5780f, 107.8020f, 106.2960f, 104.7900f, 106.2390f, 107.6890f, 106.0470f,
    104.4050f, 104.2250f, 104.0460f, 102.0230f, 100.0000f, 98.1671f, 96.3342f, 96.0611f,
    95.7880f, 92.2368f, 88.6856f, 89.3459f, 90.0062f, 89.8026f, 89.5991f, 88.6489f,
    87.6987f, 85.4936f, 83.2886f, 83.4939f, 83.6992f, 81.8630f, 80.0268f, 80.1207f,
    80.2146f, 81.2462f, 82.2778f, 80.2810f, 78.2842f, 74.0027f, 69.7213f, 70.6652f,
    71.6091f, 72.9790f, 74.3490f, 67.9765f, 61.6040f, 65.7448f, 69.8856f, 72.4863f,
    75.0870f, 69.3398f, 63.5927f, 55.0054f, 46.4182f, 56.6118f, 66.8054f, 65.0941f,
    63.3828f,
};

// Relative spectral power at an arbitrary wavelength, linearly interpolated.
// Zero outside 380-780 nm, which is also where the observer's response ends,
// so the two supports agree and nothing is silently truncated.
[[nodiscard]] inline f32 D65Relative(f32 lambda_nm) {
    if (lambda_nm < D65_LAMBDA_MIN || lambda_nm > D65_LAMBDA_MAX) {
        return 0.0f;
    }
    const f32 pos = (lambda_nm - D65_LAMBDA_MIN) / D65_LAMBDA_STEP;
    const u32 i = static_cast<u32>(pos);
    if (i >= D65_LUT_SIZE - 1) {
        return CIE_D65[D65_LUT_SIZE - 1];
    }
    const f32 t = pos - static_cast<f32>(i);
    return CIE_D65[i] + t * (CIE_D65[i + 1] - CIE_D65[i]);
}

}  // namespace quantiloom
