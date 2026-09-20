/**
 * @file camera_isp.hlsli
 * @brief Shared ISP constants for the camera display passes
 *
 * MHC demosaic coefficients. These values must stay equal to the constexpr
 * tables in src/libQuantiloom/postprocess/CameraDemosaic.hpp; that equality is
 * enforced by tests/test_postprocess/test_isp_coefficients.cpp, which parses
 * this file at runtime and compares number by number. Row-major 5x5,
 * offsets dy=-2..2, dx=-2..2.
 *
 * Malvar, He, Cutler, "High-quality linear interpolation for demosaicing of
 * Bayer-patterned color images", ICASSP 2004. Figure 2 coefficients / 8.
 *
 * This header is included by the camera display passes; no pass consumes it
 * yet at the stage it is introduced, but the coefficient contract is tested
 * from the start so the GPU and CPU chains can never drift apart silently.
 */

#ifndef CAMERA_ISP_HLSLI
#define CAMERA_ISP_HLSLI

// Green at a red or blue site.
static const float kMhcGreenAtRedBlue[25] = {
    0.0, 0.0, -0.125, 0.0, 0.0,
    0.0, 0.0, 0.25, 0.0, 0.0,
    -0.125, 0.25, 0.5, 0.25, -0.125,
    0.0, 0.0, 0.25, 0.0, 0.0,
    0.0, 0.0, -0.125, 0.0, 0.0,
};

// Red at a green site in a red row / blue at a green site in a blue row.
static const float kMhcColorAtGreenHorizontal[25] = {
    0.0, 0.0, 0.0625, 0.0, 0.0,
    0.0, -0.125, 0.0, -0.125, 0.0,
    -0.125, 0.5, 0.625, 0.5, -0.125,
    0.0, -0.125, 0.0, -0.125, 0.0,
    0.0, 0.0, 0.0625, 0.0, 0.0,
};

// Transpose of kMhcColorAtGreenHorizontal.
static const float kMhcColorAtGreenVertical[25] = {
    0.0, 0.0, -0.125, 0.0, 0.0,
    0.0, -0.125, 0.0, -0.125, 0.0,
    0.0625, 0.5, 0.625, 0.5, 0.0625,
    0.0, -0.125, 0.0, -0.125, 0.0,
    0.0, 0.0, -0.125, 0.0, 0.0,
};

// Red at a blue site / blue at a red site.
static const float kMhcColorAtOpposite[25] = {
    0.0, 0.0, -0.1875, 0.0, 0.0,
    0.0, 0.25, 0.0, 0.25, 0.0,
    -0.1875, 0.0, 0.75, 0.0, -0.1875,
    0.0, 0.25, 0.0, 0.25, 0.0,
    0.0, 0.0, -0.1875, 0.0, 0.0,
};

#endif // CAMERA_ISP_HLSLI
