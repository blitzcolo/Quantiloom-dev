#pragma once

/**
 * @file CameraDemosaic.hpp
 * @brief Malvar-He-Cutler 2004 demosaic coefficients (internal)
 *
 * The 5x5 gradient-corrected linear interpolation kernels from
 *   Malvar, He, Cutler, "High-quality linear interpolation for demosaicing
 *   of Bayer-patterned color images", ICASSP 2004 (also US 7,502,505).
 * Figure 2 lists the coefficients scaled by 8; the tables below carry the
 * final floating-point weights (the paper's integers divided by 8).
 *
 * Every kernel sums to 1, so a constant CFA reproduces that constant exactly.
 *
 * These values are duplicated, by design, in src/shaders/camera_isp.hlsli for
 * the GPU passes; tests/test_postprocess/test_isp_coefficients.cpp parses the
 * HLSL text at runtime and holds the two copies equal. Change one and the
 * test fails until the other matches.
 */

#include "core/Types.hpp"

#include <array>

namespace quantiloom::camera {

namespace mhc {

// Row-major 5x5, offsets dy=-2..2, dx=-2..2.

// Green at a red or blue site. Bilinear cardinal average plus a same-color
// correction along both axes.
inline constexpr std::array<std::array<f64, 5>, 5> kGreenAtRedBlue = {{
    {{0.0, 0.0, -0.125, 0.0, 0.0}},
    {{0.0, 0.0, 0.25, 0.0, 0.0}},
    {{-0.125, 0.25, 0.5, 0.25, -0.125}},
    {{0.0, 0.0, 0.25, 0.0, 0.0}},
    {{0.0, 0.0, -0.125, 0.0, 0.0}},
}};

// Red at a green site in a red row (horizontal red neighbors), and blue at a
// green site in a blue row. Horizontal bilinear interpolation of the color,
// vertical gradient correction from green.
inline constexpr std::array<std::array<f64, 5>, 5> kColorAtGreenHorizontal = {{
    {{0.0, 0.0, 0.0625, 0.0, 0.0}},
    {{0.0, -0.125, 0.0, -0.125, 0.0}},
    {{-0.125, 0.5, 0.625, 0.5, -0.125}},
    {{0.0, -0.125, 0.0, -0.125, 0.0}},
    {{0.0, 0.0, 0.0625, 0.0, 0.0}},
}};

// Red at a green site in a blue row / blue at a green site in a red row: the
// transpose of kColorAtGreenHorizontal.
inline constexpr std::array<std::array<f64, 5>, 5> kColorAtGreenVertical = {{
    {{0.0, 0.0, -0.125, 0.0, 0.0}},
    {{0.0, -0.125, 0.0, -0.125, 0.0}},
    {{0.0625, 0.5, 0.625, 0.5, 0.0625}},
    {{0.0, -0.125, 0.0, -0.125, 0.0}},
    {{0.0, 0.0, -0.125, 0.0, 0.0}},
}};

// Red at a blue site / blue at a red site: diagonal bilinear interpolation of
// the color with a same-color correction.
inline constexpr std::array<std::array<f64, 5>, 5> kColorAtOpposite = {{
    {{0.0, 0.0, -0.1875, 0.0, 0.0}},
    {{0.0, 0.25, 0.0, 0.25, 0.0}},
    {{-0.1875, 0.0, 0.75, 0.0, -0.1875}},
    {{0.0, 0.25, 0.0, 0.25, 0.0}},
    {{0.0, 0.0, -0.1875, 0.0, 0.0}},
}};

} // namespace mhc

} // namespace quantiloom::camera
