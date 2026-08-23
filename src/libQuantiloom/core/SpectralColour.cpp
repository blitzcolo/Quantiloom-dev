/**
 * @file SpectralColour.cpp
 * @brief Illuminant spectrum to linear sRGB, via the CIE 1931 observer
 *
 * Lives in its own translation unit because the colour matching table is
 * internal (core/CIE_CMF_Data.hpp) while the functions are public: an inline
 * implementation in the header would have to export the table with them.
 */

#include "core/SpectralData.hpp"
#include "core/CIE_CMF_Data.hpp"
#include "core/D65Illuminant.hpp"

#include <algorithm>

namespace quantiloom {

namespace {

// CIE_1931_2DEG is tabulated at 1 nm from 380 to 780, which is the whole
// support of the observer -- outside it the response is zero and a solar
// spectrum's infrared tail contributes nothing to colour, however much energy
// it carries.
constexpr f32 kCieLambdaMin = 380.0f;

// Trapezoid, for the same reason the shaders use it: N samples span N-1
// intervals, and summing N full-width rectangles overstates the integral by
// one interval's worth.
struct Tristimulus {
    f64 x = 0.0, y = 0.0, z = 0.0;
};

Tristimulus IntegrateAgainstObserver(const SpectralCurve& curve) {
    Tristimulus sum;
    f64 prevX = 0.0, prevY = 0.0, prevZ = 0.0;
    bool havePrev = false;

    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f32 lambda = kCieLambdaMin + static_cast<f32>(i);
        const f64 e = static_cast<f64>(curve.Evaluate(lambda));
        const f64 cx = e * CIE_1931_2DEG[i][0];
        const f64 cy = e * CIE_1931_2DEG[i][1];
        const f64 cz = e * CIE_1931_2DEG[i][2];

        if (havePrev) {
            sum.x += 0.5 * (cx + prevX);  // 1 nm spacing, so dλ = 1
            sum.y += 0.5 * (cy + prevY);
            sum.z += 0.5 * (cz + prevZ);
        }
        prevX = cx;
        prevY = cy;
        prevZ = cz;
        havePrev = true;
    }
    return sum;
}

}  // namespace

glm::vec3 SpectralIrradianceToLinearSrgb(const SpectralCurve& curve) {
    if (curve.samples.empty()) {
        return glm::vec3(0.0f);
    }

    const Tristimulus xyz = IntegrateAgainstObserver(curve);

    // XYZ -> linear sRGB, D65 primaries (IEC 61966-2-1). No clamping: see the
    // note on the declaration.
    return glm::vec3(
        static_cast<f32>( 3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z),
        static_cast<f32>(-0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z),
        static_cast<f32>( 0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z));
}

glm::vec3 EmissionSpectrumToRenderedLinearSrgb(const SpectralCurve& curve) {
    // One constant apart from the function above, shared with the D65 factor the
    // CIE buffer carries so the host and the shader cannot drift.
    const f64 yIntegral = CieLuminanceIntegral();
    if (yIntegral <= 0.0) {
        return glm::vec3(0.0f);
    }
    return SpectralIrradianceToLinearSrgb(curve) / static_cast<f32>(yIntegral);
}

glm::vec3 ReflectanceToLinearSrgbD65(const SpectralCurve& reflectance) {
    if (reflectance.samples.empty()) {
        return glm::vec3(0.0f);
    }

    // Same trapezoid over the same 1 nm observer grid as above, but weighted by
    // D65 and divided by the illuminant's own luminance, so a perfect diffuser
    // (rho == 1) comes out at (1, 1, 1) instead of at D65's absolute magnitude.
    // That is the difference from SpectralIrradianceToLinearSrgb, which reports
    // the curve's own magnitude because for a sun the magnitude IS the answer.
    Tristimulus xyz;
    f64 yNorm = 0.0;
    f64 prevX = 0.0, prevY = 0.0, prevZ = 0.0, prevN = 0.0;
    bool havePrev = false;

    for (u32 i = 0; i < CIE_CMF_LUT_SIZE; ++i) {
        const f32 lambda = kCieLambdaMin + static_cast<f32>(i);
        const f64 e = static_cast<f64>(D65Relative(lambda));
        const f64 r = static_cast<f64>(reflectance.Evaluate(lambda));
        const f64 cx = r * e * CIE_1931_2DEG[i][0];
        const f64 cy = r * e * CIE_1931_2DEG[i][1];
        const f64 cz = r * e * CIE_1931_2DEG[i][2];
        const f64 cn = e * CIE_1931_2DEG[i][1];

        if (havePrev) {
            xyz.x += 0.5 * (cx + prevX);
            xyz.y += 0.5 * (cy + prevY);
            xyz.z += 0.5 * (cz + prevZ);
            yNorm += 0.5 * (cn + prevN);
        }
        prevX = cx;
        prevY = cy;
        prevZ = cz;
        prevN = cn;
        havePrev = true;
    }

    if (yNorm <= 0.0) {
        return glm::vec3(0.0f);
    }
    xyz.x /= yNorm;
    xyz.y /= yNorm;
    xyz.z /= yNorm;

    const glm::vec3 rgb(
        static_cast<f32>( 3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z),
        static_cast<f32>(-0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z),
        static_cast<f32>( 0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z));

    // Clamped, unlike the illuminant conversion. This colour is compared
    // against texels, which are themselves in [0, 1]; a spectrum outside the
    // sRGB gamut that came back with a negative channel would pull the unmix
    // toward weights that no texture can express.
    return glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(1.0f));
}

SpectralCurve MakeEqualEnergyIlluminant(f32 lambdaMin_nm, f32 lambdaMax_nm) {
    SpectralCurve curve;
    if (!(lambdaMax_nm > lambdaMin_nm)) {
        return curve;
    }

    // The luminance of a flat spectrum of height h is h times the observer's
    // own y-bar integral, so the height that puts Y at 1 is its reciprocal.
    // Computed rather than hardcoded as 1/106.857: the table is the authority
    // on its own integral, and it has been resampled once already.
    f64 yBarIntegral = 0.0;
    for (u32 i = 1; i < CIE_CMF_LUT_SIZE; ++i) {
        yBarIntegral += 0.5 * (CIE_1931_2DEG[i][1] + CIE_1931_2DEG[i - 1][1]);
    }
    const f32 height = yBarIntegral > 0.0
                           ? static_cast<f32>(1.0 / yBarIntegral)
                           : 0.0f;

    // Two samples are enough for a constant: Evaluate interpolates linearly
    // between them and clamps outside, so the spectrum is flat everywhere and
    // extends across whatever band the caller asked for.
    curve.samples.emplace_back(lambdaMin_nm, height);
    curve.samples.emplace_back(lambdaMax_nm, height);
    return curve;
}

}  // namespace quantiloom
